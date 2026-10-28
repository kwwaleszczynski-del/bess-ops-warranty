"""Capacity tests: evaluation, temperature correction, measurement uncertainty, SoH trajectory and fade forecast.

Test procedure (ILLUSTRATIVE, in the spirit of IEC 62933-2-1 and common LTSA practice; set in ``ltsa.yaml``):
charge to full, rest, discharge at the test power until the EMS stops at its lower limit. The discharged energy is
integrated at the meter (or at the DC terminals), optionally with the auxiliary energy added back, and corrected to
the reference temperature with a linear coefficient::

    E_corr = E_meas / (1 + alpha * (T_test - T_ref))

Uncertainty (GUM, relative standard uncertainties added in quadrature; expanded with coverage factor k):

    u_meter = class / sqrt(3),  u_CT = ct / sqrt(3),  u_VT = vt / sqrt(3)        (rectangular error limits)
    u_temp  = alpha * u_T  (+)  u_alpha * |T_test - T_ref|                        (temperature correction)

Decision rule: ``simple`` passes when E_corr >= E_guaranteed; ``guard_band`` passes only when E_corr - U >=
E_guaranteed, fails when E_corr + U < E_guaranteed, and is ``inconclusive`` in between.

SoH forecast. The test results (SoH = E_corr / reference energy) are fitted with the two-term fade law of the
companion project ``bess-capacity-augmentation`` (calendar ageing ~ sqrt(time), cycle ageing ~ equivalent cycles)::

    SoH(t) = s0 - a * sqrt(t) - b * N(t) / 1000          t in years since COD, N cumulative EFC

with a Gaussian prior on ``a`` and ``b`` from its LFP cell fit (Arrhenius-scaled to the site's cell temperature), so
that two or three tests, in which time and cycles grow together, still give a usable split. The forecast extends
N(t) at the recent cycling rate and returns when SoH crosses the energy requirement and the guaranteed curve. What
the cheapest response is (augmentation, overbuild, operating limits) is the subject of that project; this one hands
over the measured starting point and fade rates (``handoff``).

Usage:
    tests = evaluate_tests(site, windows, ltsa, warranty, site_cfg)
    fc = forecast_soh(tests, ltsa, warranty, site_cfg, efc_per_year=330, cell_temp_c=26)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from bess_ops.config import retention_at
from bess_ops.cycles import throughput_energy
from bess_ops.warranty import guaranteed_soh

log = logging.getLogger(__name__)
R_GAS = 8.314
T_REF_K = 298.15


def _steady_runs(p: np.ndarray, power: float, tol: float) -> list[tuple[int, int]]:
    """(first, last) positions of runs at the test power (within the tolerance), longest first."""
    m = np.abs(p - power) <= tol * power
    d = np.diff(np.r_[0, m.astype(int), 0])
    runs = list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))
    return sorted(runs, key=lambda r: r[0] - r[1])


def find_test_windows(site: pd.DataFrame, ltsa: dict, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Test windows: the ``capacity_test`` events of the log, else full discharges at the test power in the data
    that start at or above ``start_soc_min`` and end at or below ``end_soc_max``."""
    if events is not None and not events.empty and (events["category"] == "capacity_test").any():
        ev = events[events["category"] == "capacity_test"]
        return ev[["start", "end"]].reset_index(drop=True)
    t = ltsa["capacity"]["test"]
    p = np.nan_to_num(site["p_meter"].to_numpy(float))
    dt = site.attrs["dt_h"]
    soc = site["soc"].to_numpy(float) if "soc" in site else None
    rows = []
    for s, e in _steady_runs(p, t["power_mw"], t["steady_tolerance"]):
        if (e - s + 1) * dt < t["min_duration_h"]:
            continue
        lo, hi = _discharge_run(p, s, e, 0.02 * t["power_mw"])
        if soc is not None and hi + 1 < len(soc) and not (soc[lo] >= t["start_soc_min"]
                                                          and soc[hi + 1] <= t["end_soc_max"]):
            continue
        rows.append({"start": site.index[lo], "end": site.index[hi] + pd.Timedelta(hours=dt)})
    return pd.DataFrame(rows, columns=["start", "end"]).sort_values("start").reset_index(drop=True)


def _discharge_run(p: np.ndarray, lo: int, hi: int, eps: float) -> tuple[int, int]:
    """Widen [lo, hi] to the whole contiguous run of discharge (p > eps) around the steady part."""
    while lo > 0 and p[lo - 1] > eps:
        lo -= 1
    while hi < len(p) - 1 and p[hi + 1] > eps:
        hi += 1
    return lo, hi


def relative_uncertainty(t: dict, temp_c: float) -> dict:
    """Relative standard uncertainties of one test and their combination."""
    u = {"meter": t["meter_class_pct"] / 100 / np.sqrt(3), "ct": t["ct_pct"] / 100 / np.sqrt(3),
         "vt": t["vt_pct"] / 100 / np.sqrt(3),
         "temperature": np.hypot(t["temp_coeff_per_k"] * t["temp_u_k"],
                                 t["temp_coeff_u_per_k"] * abs(temp_c - t["t_ref_c"])) if np.isfinite(temp_c)
         else 0.0}
    u["combined"] = float(np.sqrt(sum(v ** 2 for v in u.values())))
    return u


def guaranteed_energy(ltsa: dict, warranty: dict, elapsed_years, efc_total) -> np.ndarray:
    """Guaranteed energy (MWh) at the test: warranty retention curve, a fixed share, or an own {years, soh} table."""
    g = ltsa["capacity"]["guaranteed"]
    e_ref = warranty["reference_energy_mwh"]
    if g == "warranty":
        soh = guaranteed_soh(warranty["retention"], elapsed_years, efc_total,
                             warranty["throughput"]["efc_per_year"])
    elif isinstance(g, dict):
        soh = retention_at({"years": g["years"], "soh": g["soh"]}, elapsed_years)
    else:
        soh = np.full(np.shape(elapsed_years), float(g))
    return np.asarray(soh, float) * e_ref


def evaluate_tests(site: pd.DataFrame, windows: pd.DataFrame, ltsa: dict, warranty: dict,
                   site_cfg: dict) -> pd.DataFrame:
    """One row per test window: measured and corrected energy, uncertainty, guarantee and verdict."""
    t = ltsa["capacity"]["test"]
    dt = site.attrs["dt_h"]
    p = site["p_meter"].to_numpy(float)
    e_ref = warranty["reference_energy_mwh"]
    cum_efc = (throughput_energy(site, warranty["throughput"]["basis"]).fillna(0).cumsum() / e_ref).to_numpy()
    tz = site_cfg["timezone"]
    cod = site_cfg.get("cod")
    loc0 = site.index[0].tz_convert(tz)
    cod_ts = pd.Timestamp(cod).tz_localize(tz) if cod is not None else loc0.normalize()
    rows = []
    for w in windows.itertuples():
        sel = np.flatnonzero((site.index >= w.start) & (site.index < w.end))
        runs = _steady_runs(np.nan_to_num(p[sel]), t["power_mw"], t["steady_tolerance"]) if len(sel) else []
        if not runs:
            log.warning("no discharge at the test power in the test window %s - %s", w.start, w.end)
            continue
        lo, hi = _discharge_run(np.nan_to_num(p), sel[0] + runs[0][0], sel[0] + runs[0][1],
                                0.02 * t["power_mw"])
        seg = slice(lo, hi + 1)
        if t["measure"] == "dc" and "p_dc" in site:
            e_meas = float(np.nansum(site["p_dc"].to_numpy()[seg].clip(min=0)) * dt)
        else:
            e_meas = float(np.nansum(p[seg].clip(min=0)) * dt)
        if t["aux_correction"] == "add_aux" and "p_aux" in site:
            e_meas += float(np.nansum(site["p_aux"].to_numpy()[seg]) * dt)
        temp_col = t["temp_signal"] if t["temp_signal"] in site else next(
            (c for c in ("t_cell_avg", "t_cell_max", "t_amb") if c in site), None)
        temp = float(np.nanmean(site[temp_col].to_numpy()[seg])) if temp_col else np.nan
        alpha = t["temp_coeff_per_k"]
        factor = 1 + alpha * (temp - t["t_ref_c"]) if np.isfinite(temp) else 1.0
        e_corr = e_meas / factor
        u = relative_uncertainty(t, temp)
        big_u = t["coverage_k"] * u["combined"] * e_corr
        soc = site["soc"].to_numpy(float) if "soc" in site else None
        soc_start = float(soc[lo]) if soc is not None else np.nan
        soc_end = float(soc[hi + 1]) if soc is not None and hi + 1 < len(soc) else np.nan
        start_local = site.index[lo].tz_convert(tz)
        elapsed = (start_local - cod_ts) / pd.Timedelta(days=365.25)
        efc = float(cum_efc[lo - 1]) if lo > 0 else 0.0
        e_g = float(guaranteed_energy(ltsa, warranty, elapsed, efc))
        if t["decision_rule"] == "guard_band":
            verdict = "pass" if e_corr - big_u >= e_g else ("fail" if e_corr + big_u < e_g else "inconclusive")
        else:
            verdict = "pass" if e_corr >= e_g else "fail"
        checks = []
        if np.isfinite(soc_start) and soc_start < t["start_soc_min"]:
            checks.append(f"start SoC {soc_start:.1%} < {t['start_soc_min']:.0%}")
        if np.isfinite(soc_end) and soc_end > t["end_soc_max"]:
            checks.append(f"end SoC {soc_end:.1%} > {t['end_soc_max']:.0%}")
        rows.append({"start": site.index[lo], "end": site.index[hi] + pd.Timedelta(hours=dt),
                     "duration_h": (hi - lo + 1) * dt, "mean_power_mw": e_meas / ((hi - lo + 1) * dt),
                     "temp_c": temp, "e_meas_mwh": e_meas, "temp_factor": factor, "e_corr_mwh": e_corr,
                     "u_rel_combined": u["combined"], "U_mwh": big_u, "soc_start": soc_start, "soc_end": soc_end,
                     "elapsed_years": elapsed, "efc_total": efc, "soh": e_corr / e_ref,
                     "e_guaranteed_mwh": e_g, "soh_guaranteed": e_g / e_ref, "margin_mwh": e_corr - e_g,
                     "verdict": verdict, "procedure_checks": "; ".join(checks)})
    return pd.DataFrame(rows)


def prior_rates(fc: dict, cell_temp_c: float) -> tuple[float, float]:
    """Prior a (per sqrt(year)) and b (per 1000 EFC) from the cell-fit rates at the site's cell temperature."""
    tk = cell_temp_c + 273.15
    arr_cal = np.exp(-fc["prior_ea_cal"] / R_GAS * (1 / tk - 1 / T_REF_K))
    arr_cyc = np.exp(-fc["prior_ea_cyc"] / R_GAS * (1 / tk - 1 / T_REF_K))
    a = fc["prior_k_cal_per_sqrt_day"] * np.sqrt(365.25) * arr_cal
    b = fc["prior_k_cyc_per_efc"] * 1000 * arr_cyc
    return float(a), float(b)


def fit_fade(tests: pd.DataFrame, prior: tuple[float, float], prior_rel_sd: float,
             coverage_k: float = 2.0) -> dict:
    """Posterior of (s0, a, b) by weighted least squares with a Gaussian prior on a and b (flat on s0)."""
    t = tests["elapsed_years"].clip(lower=0).to_numpy(float)
    n = tests["efc_total"].to_numpy(float) / 1000
    y = tests["soh"].to_numpy(float)
    sigma = (tests["U_mwh"] / coverage_k / tests["e_corr_mwh"] * tests["soh"]).to_numpy(float)
    sigma = np.where(sigma > 0, sigma, 0.002)
    x = np.column_stack([np.ones_like(t), -np.sqrt(t), -n])
    w = 1 / sigma ** 2
    prec = np.diag([0.0, 1 / (prior_rel_sd * prior[0]) ** 2, 1 / (prior_rel_sd * prior[1]) ** 2])
    mu = np.array([0.0, prior[0], prior[1]])
    a_mat = x.T @ (w[:, None] * x) + prec
    rhs = x.T @ (w * y) + prec @ mu
    cov = np.linalg.inv(a_mat)
    theta = cov @ rhs
    resid = y - x @ theta
    return {"s0": theta[0], "a": theta[1], "b": theta[2], "cov": cov, "resid": resid,
            "prior_a": prior[0], "prior_b": prior[1], "n_tests": len(y)}


def forecast_soh(tests: pd.DataFrame, ltsa: dict, warranty: dict, site_cfg: dict, efc_per_year: float,
                 cell_temp_c: float, now_years: float | None = None, now_efc: float | None = None) -> dict:
    """Fade fit and SoH projection with a P10-P90 band; crossing years of the requirement and the guarantee."""
    fc = ltsa["capacity"]["forecast"]
    prior = prior_rates(fc, cell_temp_c)
    fit = fit_fade(tests, prior, fc["prior_rel_sd"], ltsa["capacity"]["test"]["coverage_k"])
    now_years = float(tests["elapsed_years"].max()) if now_years is None else now_years
    now_efc = float(tests["efc_total"].max()) if now_efc is None else now_efc
    years = np.round(np.arange(0, fc["horizon_years"] + 1e-9, 1 / 12), 6)
    n = np.where(years <= now_years, now_efc * years / max(now_years, 1e-9),
                 now_efc + efc_per_year * (years - now_years)) / 1000
    x = np.column_stack([np.ones_like(years), -np.sqrt(years), -n])
    mean = x @ np.array([fit["s0"], fit["a"], fit["b"]])
    sd = np.sqrt(np.einsum("ij,jk,ik->i", x, fit["cov"], x))
    z = 1.2816
    e_ref = warranty["reference_energy_mwh"]
    curve = pd.DataFrame({"years": years, "efc_total": n * 1000, "soh_p50": mean, "soh_p10": mean - z * sd,
                          "soh_p90": mean + z * sd})
    curve["soh_guaranteed"] = guaranteed_energy(ltsa, warranty, years, n * 1000) / e_ref
    out = {"fit": fit, "curve": curve, "efc_per_year": efc_per_year, "cell_temp_c": cell_temp_c,
           "now_years": now_years}
    req = fc["required_mwh"]
    future = curve["years"] >= now_years
    for name, thr in (("requirement", (req / e_ref) if req else None), ("guarantee", curve["soh_guaranteed"])):
        if thr is None:
            out[f"{name}_years"] = {"p10": np.nan, "p50": np.nan, "p90": np.nan}
            continue
        res = {}
        for q, col in (("p10", "soh_p10"), ("p50", "soh_p50"), ("p90", "soh_p90")):
            below = future & (curve[col] < thr)
            res[q] = float(curve.loc[below, "years"].iloc[0]) if below.any() else np.nan
        out[f"{name}_years"] = res
    return out


def handoff(fc: dict, site_cfg: dict, warranty: dict, ltsa: dict) -> dict:
    """Measured fade in the parameters of bess-capacity-augmentation's AgingModel (25 degC reference)."""
    f = fc["fit"]
    p = ltsa["capacity"]["forecast"]
    tk = fc["cell_temp_c"] + 273.15
    arr_cal = np.exp(-p["prior_ea_cal"] / R_GAS * (1 / tk - 1 / T_REF_K))
    arr_cyc = np.exp(-p["prior_ea_cyc"] / R_GAS * (1 / tk - 1 / T_REF_K))
    cod = site_cfg.get("cod")
    to_date = (lambda y: (pd.Timestamp(cod) + pd.Timedelta(days=365.25 * y)).date().isoformat()
               if cod is not None and np.isfinite(y) else None)
    return {
        "site": site_cfg["name"],
        "reference_energy_mwh": warranty["reference_energy_mwh"],
        "soh_at_last_test": float(fc["curve"].loc[fc["curve"]["years"] >= fc["now_years"], "soh_p50"].iloc[0]),
        "s0": float(f["s0"]), "fade_calendar_per_sqrt_year": float(f["a"]), "fade_per_1000_efc": float(f["b"]),
        "cell_temp_c": fc["cell_temp_c"], "efc_per_year": fc["efc_per_year"],
        "aging_model_k_cal_per_sqrt_day_25c": float(f["a"] / np.sqrt(365.25) / arr_cal),
        "aging_model_k_cyc_per_efc_25c": float(f["b"] / 1000 / arr_cyc),
        "below_requirement": _dates(fc["requirement_years"], to_date),
        "below_guarantee": _dates(fc["guarantee_years"], to_date),
        "note": "k values plug into bess_augment.aging.AgingModel(k_cal=..., k_cyc=...) for the augmentation plan",
    }


def _dates(years: dict, to_date) -> dict:
    """Crossing dates: early (P10 of SoH), central (P50) and late (P90); None = not within the horizon."""
    return {"early": to_date(years["p10"]), "central": to_date(years["p50"]), "late": to_date(years["p90"])}
