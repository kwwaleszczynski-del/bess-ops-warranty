import numpy as np
import pandas as pd
import pytest

from bess_ops.auxiliaries import fit_aux_temperature
from bess_ops.capacity import evaluate_tests, find_test_windows, fit_fade, forecast_soh, prior_rates, \
    relative_uncertainty
from bess_ops.config import ltsa_from_dict, warranty_from_dict

from conftest import hourly_frame


def test_uncertainty_by_hand():
    t = {"meter_class_pct": 0.2, "ct_pct": 0.2, "vt_pct": 0.2, "temp_coeff_per_k": 0.003, "temp_u_k": 1.0,
         "temp_coeff_u_per_k": 0.001, "t_ref_c": 25.0}
    u = relative_uncertainty(t, 35.0)
    # three rectangular 0.2 % limits: 0.2 / sqrt(3) % each; temperature: hypot(0.003 x 1, 0.001 x 10) = 1.044 %
    expected = np.sqrt(3 * (0.002 / np.sqrt(3)) ** 2 + 0.003 ** 2 + 0.01 ** 2)
    assert u["combined"] == pytest.approx(expected)


def test_capacity_test_evaluation(site_cfg):
    # 15-minute data: full at 100 %, rest, 2 h at 10 MW (20 MWh) at 35 degC, a ramp-down quarter of 1 MW
    n = 24
    p = np.zeros(n)
    p[8:16] = 10.0
    p[16] = 1.0
    soc = np.full(n, 1.0)
    soc[9:17] = np.linspace(0.875, 0.05, 8)
    soc[17:] = 0.0
    idx = pd.date_range(pd.Timestamp("2026-06-01 06:00").tz_localize("Europe/Warsaw"), periods=n, freq="15min")
    from bess_ops.schema import site_frame
    site = site_frame({"p_meter": p, "soc": soc, "t_cell_avg": np.full(n, 35.0)}, idx, "Europe/Warsaw")
    w = warranty_from_dict({"retention": {"years": [0, 10], "soh": [1.0, 0.8], "equivalent_age": False}}, site_cfg)
    lt = ltsa_from_dict({"capacity": {"test": {"power_mw": 10, "temp_coeff_per_k": 0.003,
                                               "decision_rule": "guard_band"}}}, site_cfg)
    windows = find_test_windows(site, lt)
    assert len(windows) == 1
    t = evaluate_tests(site, windows, lt, w, site_cfg).iloc[0]
    assert t["e_meas_mwh"] == pytest.approx(20.25)
    assert t["e_corr_mwh"] == pytest.approx(20.25 / 1.03)
    assert t["soc_start"] == 1.0 and t["soc_end"] == 0.0
    elapsed = t["elapsed_years"]
    tz = "Europe/Warsaw"
    assert elapsed == pytest.approx((pd.Timestamp("2026-06-01 08:00", tz=tz) - pd.Timestamp("2025-01-01", tz=tz))
                                    / pd.Timedelta(days=365.25))
    assert t["e_guaranteed_mwh"] == pytest.approx(20 * (1 - 0.02 * elapsed))
    assert t["verdict"] in ("pass", "inconclusive")
    assert (t["e_corr_mwh"] - t["U_mwh"] >= t["e_guaranteed_mwh"]) == (t["verdict"] == "pass")


def test_fade_fit_recovers_rates_and_forecast_crossing(site_cfg):
    a, b, s0 = 0.02, 0.025, 1.03
    years = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    efc = years * 300 + np.array([0, 30, -20, 40, 0, -30])        # cycling varies a little year to year
    tests = pd.DataFrame({"elapsed_years": years, "efc_total": efc,
                          "soh": s0 - a * np.sqrt(years) - b * efc / 1000, "U_mwh": 0.2, "e_corr_mwh": 100.0})
    prior = (0.03, 0.02)
    fit = fit_fade(tests, prior, prior_rel_sd=10.0)                 # nearly flat prior: the data decide
    assert fit["a"] == pytest.approx(a, abs=2e-3) and fit["b"] == pytest.approx(b, abs=2e-3)
    w = warranty_from_dict({"reference_energy_mwh": 100}, site_cfg)
    lt = ltsa_from_dict({"capacity": {"forecast": {"required_mwh": 85, "prior_rel_sd": 10.0}}}, site_cfg)
    fc = forecast_soh(tests, lt, w, site_cfg, efc_per_year=300, cell_temp_c=25.0)
    yrs = fc["curve"]["years"].to_numpy()
    true = s0 - a * np.sqrt(yrs) - b * (efc[-1] + 300 * np.maximum(yrs - 5, 0)) / 1000
    cross = yrs[(yrs >= 5) & (true < 0.85)][0]
    assert fc["requirement_years"]["p50"] == pytest.approx(cross, abs=0.25)
    assert fc["requirement_years"]["p10"] <= fc["requirement_years"]["p50"] <= fc["requirement_years"]["p90"]


def test_prior_is_scaled_by_temperature():
    fc = {"prior_k_cal_per_sqrt_day": 1.64e-3, "prior_ea_cal": 30_500.0, "prior_k_cyc_per_efc": 2.1e-5,
          "prior_ea_cyc": 31_800.0}
    a25, b25 = prior_rates(fc, 25.0)
    a35, b35 = prior_rates(fc, 35.0)
    assert a25 == pytest.approx(1.64e-3 * np.sqrt(365.25)) and b25 == pytest.approx(0.021)
    assert 1.3 < a35 / a25 < 1.6 and 1.3 < b35 / b25 < 1.6


def test_aux_model_recovers_parameters(site_cfg):
    rng = np.random.default_rng(3)
    n = 24 * 120
    t = 10 + 12 * np.sin(np.arange(n) / 24 / 120 * 2 * np.pi) + rng.normal(0, 3, n)
    pcs = np.where(np.arange(n) % 24 < 4, -8.0, np.where(np.arange(n) % 24 > 18, 6.0, 0.0))
    aux = 0.05 + 0.01 * np.maximum(8 - t, 0) + 0.02 * np.maximum(t - 20, 0) + 0.004 * np.abs(pcs)
    site = hourly_frame({"p_meter": pcs - aux, "p_aux": aux, "t_amb": t}, "2025-01-01")
    p = fit_aux_temperature(site, site_cfg, grid_step=0.5)["params"]
    assert p["b0_mw"] == pytest.approx(0.05, abs=2e-3)
    assert p["heat_mw_per_k"] == pytest.approx(0.01, abs=1e-3) and p["t_heat_c"] == pytest.approx(8, abs=0.6)
    assert p["cool_mw_per_k"] == pytest.approx(0.02, abs=1e-3) and p["t_cool_c"] == pytest.approx(20, abs=0.6)
    assert p["mw_per_mw"] == pytest.approx(0.004, abs=5e-4)
