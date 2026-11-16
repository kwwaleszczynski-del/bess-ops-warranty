import numpy as np
import pandas as pd
import pytest

from bess_ops.energy import (corrected_rte, energy_balance, inverted_days, state_jumps, window_keys, window_rte)
from bess_ops.config import ltsa_from_dict, site_from_dict
from bess_ops.schema import site_frame


def test_corrected_rte_by_hand():
    # 80 out, 100 in, 10 MWh more in store at the end, one-way efficiency 0.9: (80 + 9) / 100
    assert corrected_rte(80, 100, 10, 0.9) == pytest.approx(0.89)
    # 10 MWh less in store: refilling would need 10 / 0.9 = 11.11 MWh more input
    assert corrected_rte(80, 100, -10, 0.9) == pytest.approx(80 / (100 + 10 / 0.9))
    assert corrected_rte(80, 100, 0, 0.9) == pytest.approx(0.8)


def toy_battery(days=3, eta=0.95, start_soc=0.3, end_extra=0.2, tz="Europe/Warsaw", start="2025-06-02"):
    """Hourly toy battery: one cycle a day, efficiency eta each way, no auxiliaries; the last day ends fuller."""
    idx = pd.date_range(pd.Timestamp(start).tz_localize(tz), periods=24 * days, freq="1h")
    p = np.zeros(len(idx))
    hour = idx.hour.to_numpy()
    p[(hour >= 10) & (hour < 14)] = -5.0
    p[(hour >= 18) & (hour < 21)] = 5.0 * 4 * eta * eta / 3      # discharge what was stored
    last = idx.normalize() == idx.normalize()[-1]
    p[last & (hour >= 18) & (hour < 21)] *= (1 - end_extra)
    e = np.empty(len(idx))
    e[0] = start_soc * 40
    for i in range(1, len(idx)):
        q = p[i - 1]
        e[i] = e[i - 1] + (-q * eta if q < 0 else -q / eta)
    return site_frame({"p_meter": p, "e_stored": e}, idx, tz), {"aux_metering": "none", "soc_energy_mwh": 40.0,
                                                                "timezone": tz, "cod": None}


def test_window_rte_recovers_efficiency_despite_state_change():
    site, cfg = toy_battery()
    daily = window_rte(site, cfg, "D", eta=0.95)
    assert daily["rte"].iloc[:-1].to_numpy() == pytest.approx(0.95 ** 2, abs=1e-9)
    # the last day leaves energy in store: raw ratio too low, corrected one right
    assert daily["rte_raw"].iloc[-1] < 0.85
    assert daily["d_stored"].iloc[-1] > 0
    auto = window_rte(site, cfg, "M")
    assert auto["eta"].iloc[0] == pytest.approx(0.95, abs=1e-6)


def test_local_days_have_23_and_25_hours():
    idx = pd.date_range(pd.Timestamp("2025-10-25").tz_localize("Europe/Warsaw"), periods=72, freq="1h")
    keys = window_keys(idx, "Europe/Warsaw", "D")
    assert keys.value_counts().loc["2025-10-26"] == 25
    idx = pd.date_range(pd.Timestamp("2026-03-28").tz_localize("Europe/Warsaw"), periods=72, freq="1h")
    assert window_keys(idx, "Europe/Warsaw", "D").value_counts().loc["2026-03-29"] == 23


def test_balance_splits_auxiliaries_and_losses():
    idx = pd.date_range("2025-06-01", periods=4, freq="1h", tz="UTC")
    site = site_frame({"p_meter": [-10.2, -0.2, 8.8, -0.2], "p_aux": [0.2, 0.2, 0.2, 0.2],
                       "p_dc": [-9.5, 0, 9.4, 0], "e_stored": [10, 19.3, 19.3, 9.7]}, idx, "UTC")
    cfg = {"aux_metering": "behind_meter", "soc_energy_mwh": 20.0, "timezone": "UTC", "cod": None}
    b = energy_balance(site, cfg, "M").iloc[0]
    # the last interval has no state after it and is left out: in 10.4, out 8.8, aux 0.6, stored -0.3
    assert b["e_in"] == pytest.approx(10.4)
    assert b["e_aux"] == pytest.approx(0.6)
    assert b["losses_total"] == pytest.approx(10.4 - 8.8 + 0.3)
    assert b["losses_pcs"] == pytest.approx((10.0 - 9.5) + (9.4 - 9.0))
    assert b["losses_battery"] == pytest.approx(9.5 - 9.4 + 0.3)


def test_state_jumps_removes_persistent_jump_only():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-06-01", periods=12 * 24 * 20, freq="5min", tz="UTC")
    p = np.where((idx.hour >= 10) & (idx.hour < 14), -20.0, np.where((idx.hour >= 18) & (idx.hour < 21), 25.0, 0.0))
    dt = 5 / 60
    e_in, e_out = (-p).clip(0) * dt, p.clip(0) * dt
    e = 50 + np.cumsum(np.r_[0, (0.95 * e_in - e_out / 0.95)[:-1]]) + rng.normal(0, 0.05, len(p))
    jump_at = 12 * 24 * 7 + 12 * 3                     # 03:00 on day 8, idle
    e[jump_at:] += 15.0                                # BMS recalibration
    lag = 12 * 24 * 12 + 12 * 18                       # a transient at an hour start: one reading off, then back
    e[lag] += 8.0
    adj, jumps, fit = state_jumps(pd.Series(e, index=idx), pd.Series(e_in, index=idx), pd.Series(e_out, index=idx),
                                  100.0)
    assert len(jumps) == 1 and jumps["jump_mwh"].iloc[0] == pytest.approx(15.0, abs=0.5)
    assert fit["a"] == pytest.approx(0.95, abs=0.01) and fit["b"] == pytest.approx(1 / 0.95, abs=0.01)
    assert (adj.iloc[-1] - adj.iloc[0]) == pytest.approx(e[-1] - 15.0 - e[0], abs=0.5)


def test_inverted_days():
    idx = pd.date_range("2025-06-01", periods=24 * 4, freq="1h", tz="UTC")
    p = np.tile(np.r_[np.full(6, -10.0), np.full(6, 10.0), np.zeros(12)], 4)
    e_in, e_out = pd.Series((-p).clip(0), index=idx), pd.Series(p.clip(0), index=idx)
    e = np.r_[0, np.cumsum(e_in - e_out)[:-1]] + 50
    e[:48] = 100 - e[:48]                               # first two days report the headroom
    c = inverted_days(pd.Series(e, index=idx), e_in, e_out, "UTC")
    assert (c.iloc[:2] < -0.9).all() and (c.iloc[2:3] > 0.9).all()


# ---------------------------------------------------------------------------------------------------------------------
# Missing data: a gap is missing energy, not zero energy
# ---------------------------------------------------------------------------------------------------------------------

def month_of_cycles(eta=0.95):
    """30 days, hourly, one cycle a day (charge 4 h, discharge 4 h at the round trip eta**2): the month RTE is exact."""
    tz = "Europe/Warsaw"
    idx = pd.date_range(pd.Timestamp("2025-06-01").tz_localize(tz), periods=24 * 30, freq="1h")
    p = np.zeros(len(idx))
    hour = idx.hour.to_numpy()
    p[(hour >= 10) & (hour < 14)] = -5.0
    p[(hour >= 18) & (hour < 22)] = 5.0 * eta * eta
    e = np.empty(len(idx))
    e[0] = 8.0
    for i in range(1, len(idx)):
        q = p[i - 1]
        e[i] = e[i - 1] + (-q * eta if q < 0 else -q / eta)
    cfg = {"aux_metering": "none", "soc_energy_mwh": 20.0, "timezone": tz, "cod": None}
    return idx, p, e, cfg, tz


GAP = slice(9 * 24 + 18, 9 * 24 + 22)               # the four discharge hours of day 10


def test_complete_month_is_exact():
    idx, p, e, cfg, tz = month_of_cycles()
    r = window_rte(site_frame({"p_meter": p, "e_stored": e}, idx, tz), cfg, "M", eta=0.95).iloc[0]
    assert r["rte"] == pytest.approx(0.95 ** 2) and r["coverage"] == 1.0 and r["valid"]


def test_nan_meter_reading_has_no_rte():
    # old code: coverage 0.994 >= 0.95, valid, RTE 0.872 (the missing discharge counted as zero energy)
    idx, p, e, cfg, tz = month_of_cycles()
    p[GAP] = np.nan
    r = window_rte(site_frame({"p_meter": p, "e_stored": e}, idx, tz), cfg, "M", eta=0.95).iloc[0]
    assert r["coverage"] == pytest.approx(1 - 4 / (24 * 30 - 1))
    assert np.isnan(r["rte"]) and np.isnan(r["rte_raw"]) and not r["valid"]
    assert r["e_in"] > 0                                    # the summed energies stay visible


def test_rows_absent_from_the_index_count_as_missing():
    # old code: coverage 1.0 (computed from the rows present), valid, RTE 0.872
    idx, p, e, cfg, tz = month_of_cycles()
    keep = np.ones(len(idx), bool)
    keep[GAP] = False
    r = window_rte(site_frame({"p_meter": p[keep], "e_stored": e[keep]}, idx[keep], tz), cfg, "M",
                   eta=0.95).iloc[0]
    assert r["coverage"] < 1.0
    assert np.isnan(r["rte"]) and not r["valid"]


def test_lower_min_coverage_accepts_the_bias_knowingly():
    idx, p, e, cfg, tz = month_of_cycles()
    p[GAP] = np.nan
    r = window_rte(site_frame({"p_meter": p, "e_stored": e}, idx, tz), cfg, "M", eta=0.95,
                   min_coverage=0.99).iloc[0]
    assert r["valid"] and r["rte"] < 0.95 ** 2 - 0.02       # valid, and visibly too low: the caller chose it
    with pytest.raises(ValueError):
        window_rte(site_frame({"p_meter": p, "e_stored": e}, idx, tz), cfg, "M", min_coverage=0.0)


def test_only_the_window_with_the_gap_is_dropped_and_eta_comes_from_the_others():
    idx, p, e, cfg, tz = month_of_cycles()
    p[GAP] = np.nan
    r = window_rte(site_frame({"p_meter": p, "e_stored": e}, idx, tz), cfg, "D")
    assert r["valid"].sum() == 29 and not r.loc["2025-06-10", "valid"]
    assert r["eta"].iloc[0] == pytest.approx(0.95, abs=1e-6)        # old code: 0.9, the gap dragged it down
    assert r.loc[r["valid"], "rte"].to_numpy() == pytest.approx(0.95 ** 2)


def test_energy_balance_has_no_losses_where_it_does_not_close():
    idx, p, e, cfg, tz = month_of_cycles()
    full = energy_balance(site_frame({"p_meter": p, "e_stored": e}, idx, tz), cfg, "M").iloc[0]
    assert full["losses_total"] == pytest.approx(full["e_in"] - full["e_out"] - full["d_stored"])
    p[GAP] = np.nan
    b = energy_balance(site_frame({"p_meter": p, "e_stored": e}, idx, tz), cfg, "M").iloc[0]
    assert np.isnan(b["losses_total"]) and b["coverage"] < 1.0


def test_hourly_sums_for_the_state_checks_need_every_interval():
    idx = pd.date_range("2025-06-01", periods=4 * 12, freq="5min", tz="UTC")
    x = pd.Series(1.0, index=idx)
    x.iloc[5] = np.nan                                      # one missing interval in the first hour
    from bess_ops.energy import _hour_sum
    s = _hour_sum(x, idx.floor("1h"))
    assert np.isnan(s.iloc[0]) and s.iloc[1] == 12.0


# ---------------------------------------------------------------------------------------------------------------------
# An undetermined one-way efficiency (NaN) is no data, never a number
# ---------------------------------------------------------------------------------------------------------------------

def month_ending_fuller(state_gap=False):
    """The month of cycles, but the last day discharges less, so the month ends fuller: the correction needs eta."""
    idx, p, e, cfg, tz = month_of_cycles()
    last = (idx.normalize() == idx.normalize()[-1]) & (idx.hour >= 18) & (idx.hour < 22)
    p = np.where(last, p * 0.7, p)
    e = np.empty(len(idx))
    e[0] = 8.0
    for i in range(1, len(idx)):
        q = p[i - 1]
        e[i] = e[i - 1] + (-q * 0.95 if q < 0 else -q / 0.95)
    if state_gap:                       # the stored energy is unknown for 7 hours around every day boundary
        for d in range(1, 30):
            e[d * 24 - 3: d * 24 + 4] = np.nan
    return site_frame({"p_meter": p, "e_stored": e}, idx, tz), cfg


def test_a_window_without_a_determined_efficiency_is_not_valid():
    # old code: the complete month had valid=True with an RTE of NaN, and a damages routine read that as 0
    site, cfg = month_ending_fuller()
    r = window_rte(site, cfg, "M", float("nan")).iloc[0]
    assert np.isnan(r["rte"]) and not r["valid"]
    assert r["status"] == "no data: one-way efficiency not determined"
    assert window_rte(site, cfg, "M", 0.95).iloc[0]["valid"]


def test_status_names_the_reason_for_every_window():
    site, cfg = month_ending_fuller()
    d = window_rte(site, cfg, "D", 0.95)
    assert set(d["status"]) <= {"ok", "too few cycles"} and d.loc["2025-06-30", "status"] == "ok"
    idx = site.index
    gap = site.copy()
    gap.iloc[5 * 24 + 18: 5 * 24 + 22, gap.columns.get_loc("p_meter")] = np.nan
    gap.iloc[9 * 24 - 3: 9 * 24 + 4, gap.columns.get_loc("e_stored")] = np.nan
    s = window_rte(gap, cfg, "D", 0.95)
    assert s.loc["2025-06-06", "status"] == "no data: gap in the meter readings"
    assert s.loc["2025-06-10", "status"] == "no data: stored energy unknown at the window ends"
    assert s.loc["2025-06-09", "status"] == "no data: stored energy unknown at the window ends"
    idle = site.copy()
    idle.iloc[:24, idle.columns.get_loc("p_meter")] = 0.0
    assert window_rte(idle, cfg, "D", 0.95).loc["2025-06-01", "status"] == "too few cycles"
    assert idx is not None


def test_carried_efficiency_falls_back_to_auto_so_a_complete_month_keeps_its_rte():
    # old code: the daily NaN was handed to the month, which lost an RTE it could compute (and a missing import)
    from bess_ops.energy import carried_eta
    site, cfg = month_ending_fuller(state_gap=True)
    d = window_rte(site, cfg, "D")
    assert not d["valid"].any() and np.isnan(d["eta"].iloc[0])
    assert carried_eta(d) == "auto"
    m = window_rte(site, cfg, "M", carried_eta(d)).iloc[0]
    assert m["valid"] and m["rte"] == pytest.approx(0.95 ** 2, abs=2e-3) and m["eta"] == pytest.approx(0.95, abs=1e-6)
    full, _ = month_ending_fuller()
    assert carried_eta(window_rte(full, cfg, "D")) == pytest.approx(0.95, abs=1e-6)


def test_windows_without_data_are_listed_not_the_idle_ones():
    from bess_ops.energy import no_data
    site, cfg = month_ending_fuller(state_gap=True)
    nd = no_data({"day": window_rte(site, cfg, "D"), "month": window_rte(site, cfg, "M", float("nan"))})
    assert set(nd["freq"]) == {"day", "month"} and len(nd[nd["freq"] == "day"]) == 30
    assert nd.loc[nd["freq"] == "month", "reason"].iloc[0] == "no data: one-way efficiency not determined"
    assert nd["reason"].str.startswith("no data").all()


def test_damages_count_windows_without_data_instead_of_reading_them_as_zero():
    # old code: no such column, and the window silently counted as "no damages"
    from bess_ops.pipeline import ld_by_year
    site, cfg = month_ending_fuller()
    lt = ltsa_from_dict({"ld": {"rte": {"method": "energy_cost", "price_per_mwh": 100.0}},
                         "rte": {"guaranteed": 0.99, "min_efc": 1.0}},
                        site_from_dict({"power_mw": 10, "energy_mwh": 20}))
    w = window_rte(site, cfg, "M", float("nan"))
    out = ld_by_year(None, w, None, lt, {"timezone": "Europe/Warsaw", "cod": None})
    assert out["rte_windows_no_data"].iloc[0] == 1 and out["rte"].iloc[0] == 0.0
    ok = ld_by_year(None, window_rte(site, cfg, "M", 0.95), None, lt, {"timezone": "Europe/Warsaw", "cod": None})
    assert ok["rte_windows_no_data"].iloc[0] == 0 and ok["rte"].iloc[0] > 0
