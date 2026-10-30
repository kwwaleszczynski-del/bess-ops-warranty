"""Operating conditions of a battery warranty and their violations, found in the operating data.

Typical conditions of utility-scale LFP warranties (all limits ILLUSTRATIVE and set in ``warranty.yaml``):

==================  ===========================================================================================
rule                violation
==================  ===========================================================================================
efc_per_year        equivalent full cycles in a warranty year above the limit (start: the moment it is passed)
efc_per_day         equivalent full cycles in a local calendar day above the limit
throughput_total    cumulative throughput above the allowance
soc_low / soc_high  SoC below ``min`` / above ``max`` for at least ``sustained_minutes``
mean_soc            rolling mean SoC over ``mean_window_days`` above ``max_mean`` (resting at high SoC)
c_rate              |P| / reference energy above ``max`` for at least ``sustained_minutes`` (P = p_dc or p_meter)
cell_temp_high/low  highest cell temperature above / mean cell temperature below the limit, sustained
cell_temp_annual    mean cell temperature over a warranty year above the limit
ambient_high/low    ambient temperature outside the range, sustained
==================  ===========================================================================================

The cycle count uses the throughput basis of the warranty (see ``cycles.throughput_energy``) and the reference
energy (usually the nameplate energy at the start of the warranty). The state of the warranty at the end of the
record (cycles used, equivalent age, guaranteed state of health) is in ``warranty_status``.

Usage:
    events = detect_violations(site, warranty, site_cfg)
    status = warranty_status(site, warranty, site_cfg)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from res_asset_kpi import contracts, timeseries

from bess_ops.config import retention_at
from bess_ops.cycles import throughput_energy

COLUMNS = ["rule", "start", "end", "duration_h", "value", "limit", "unit", "detail"]


def runs(mask: pd.Series, min_hours: float, dt_h: float) -> list[tuple[int, int]]:
    """(first, last) positions of consecutive True runs lasting at least ``min_hours``."""
    m = mask.fillna(False).to_numpy(bool)
    if not m.any():
        return []
    d = np.diff(np.r_[0, m.astype(int), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    need = max(int(np.ceil(min_hours / dt_h - 1e-9)), 1)
    return [(s, e) for s, e in zip(starts, ends) if e - s + 1 >= need]


def _run_events(site, mask, values, rule, limit, unit, sustained_min, peak="max", detail="") -> list[dict]:
    dt = site.attrs["dt_h"]
    step = pd.Timedelta(hours=dt)
    out = []
    for s, e in runs(mask, sustained_min / 60.0, dt):
        seg = values.iloc[s:e + 1]
        out.append({"rule": rule, "start": site.index[s], "end": site.index[e] + step,
                    "duration_h": (e - s + 1) * dt, "value": float(getattr(seg, peak)()), "limit": limit,
                    "unit": unit, "detail": detail})
    return out


def warranty_years(site: pd.DataFrame, site_cfg: dict) -> np.ndarray:
    cod = site_cfg.get("cod")
    loc = site.index.tz_convert(site_cfg["timezone"])
    if cod is None:
        cod = loc[0].normalize().tz_localize(None)
    return timeseries.contract_years(site.index, cod, site_cfg["timezone"])


def detect_violations(site: pd.DataFrame, warranty: dict, site_cfg: dict) -> pd.DataFrame:
    ev: list[dict] = []
    e_ref = warranty["reference_energy_mwh"]
    tp = warranty["throughput"]
    tz = site_cfg["timezone"]
    energy = throughput_energy(site, tp["basis"])
    efc = energy / e_ref
    if tp["efc_per_year"]:
        years = warranty_years(site, site_cfg)
        for y in np.unique(years[years > 0]):
            sel = years == y
            cum = efc[sel].fillna(0).cumsum()
            over = cum > tp["efc_per_year"]
            if over.any():
                first = over.idxmax()
                ev.append({"rule": "efc_per_year", "start": first, "end": site.index[sel][-1],
                           "duration_h": np.nan, "value": float(cum.iloc[-1]), "limit": tp["efc_per_year"],
                           "unit": "EFC", "detail": f"warranty year {y}"})
    if tp["efc_per_day"]:
        days = site.index.tz_convert(tz).strftime("%Y-%m-%d")
        daily = efc.groupby(np.asarray(days)).sum()
        for day, val in daily[daily > tp["efc_per_day"]].items():
            sel = np.asarray(days) == day
            ev.append({"rule": "efc_per_day", "start": site.index[sel][0], "end": site.index[sel][-1],
                       "duration_h": 24.0, "value": float(val), "limit": tp["efc_per_day"], "unit": "EFC",
                       "detail": day})
    if tp["total_mwh"]:
        cum = energy.fillna(0).cumsum()
        if (cum > tp["total_mwh"]).any():
            first = (cum > tp["total_mwh"]).idxmax()
            ev.append({"rule": "throughput_total", "start": first, "end": site.index[-1], "duration_h": np.nan,
                       "value": float(cum.iloc[-1]), "limit": tp["total_mwh"], "unit": "MWh", "detail": ""})
    soc = warranty["soc"]
    if "soc" in site:
        if soc["min"] is not None:
            ev += _run_events(site, site["soc"] < soc["min"], site["soc"], "soc_low", soc["min"], "fraction",
                              soc["sustained_minutes"], peak="min")
        if soc["max"] is not None:
            ev += _run_events(site, site["soc"] > soc["max"], site["soc"], "soc_high", soc["max"], "fraction",
                              soc["sustained_minutes"])
        if soc["max_mean"] is not None:
            n = max(int(round(soc["mean_window_days"] * 24 / site.attrs["dt_h"])), 1)
            roll = site["soc"].rolling(n, min_periods=int(0.8 * n)).mean()
            ev += _run_events(site, roll > soc["max_mean"], roll, "mean_soc", soc["max_mean"], "fraction", 0,
                              detail=f"rolling {soc['mean_window_days']} days")
    cr = warranty["c_rate"]
    if cr["max"] is not None:
        col = cr["basis"] if cr["basis"] in site else "p_meter"
        rate = site[col].abs() / e_ref
        ev += _run_events(site, rate > cr["max"], rate, "c_rate", cr["max"], "1/h", cr["sustained_minutes"],
                          detail=col)
    t = warranty["temperature"]
    if "t_cell_max" in site and t["cell_max_c"] is not None:
        ev += _run_events(site, site["t_cell_max"] > t["cell_max_c"], site["t_cell_max"], "cell_temp_high",
                          t["cell_max_c"], "degC", t["sustained_minutes"])
    cell_mean = site["t_cell_avg"] if "t_cell_avg" in site else site.get("t_cell_max")
    if cell_mean is not None and t["cell_min_c"] is not None:
        ev += _run_events(site, cell_mean < t["cell_min_c"], cell_mean, "cell_temp_low", t["cell_min_c"], "degC",
                          t["sustained_minutes"], peak="min")
    if cell_mean is not None and t["annual_mean_cell_max_c"] is not None:
        years = warranty_years(site, site_cfg)
        for y in np.unique(years[years > 0]):
            sel = years == y
            m = float(cell_mean[sel].mean())
            if m > t["annual_mean_cell_max_c"]:
                ev.append({"rule": "cell_temp_annual", "start": site.index[sel][0], "end": site.index[sel][-1],
                           "duration_h": np.nan, "value": m, "limit": t["annual_mean_cell_max_c"], "unit": "degC",
                           "detail": f"warranty year {y}"})
    if "t_amb" in site:
        if t["ambient_max_c"] is not None:
            ev += _run_events(site, site["t_amb"] > t["ambient_max_c"], site["t_amb"], "ambient_high",
                              t["ambient_max_c"], "degC", t["sustained_minutes"])
        if t["ambient_min_c"] is not None:
            ev += _run_events(site, site["t_amb"] < t["ambient_min_c"], site["t_amb"], "ambient_low",
                              t["ambient_min_c"], "degC", t["sustained_minutes"], peak="min")
    out = pd.DataFrame(ev, columns=COLUMNS)
    return out.sort_values(["start", "rule"]).reset_index(drop=True)


def warranty_status(site: pd.DataFrame, warranty: dict, site_cfg: dict) -> dict:
    """Cycles and throughput used, equivalent age and guaranteed SoH at the end of the record."""
    e_ref = warranty["reference_energy_mwh"]
    tp = warranty["throughput"]
    energy = throughput_energy(site, tp["basis"]).fillna(0)
    loc = site.index.tz_convert(site_cfg["timezone"])
    cod = site_cfg.get("cod")
    cod = pd.Timestamp(cod).tz_localize(loc.tz) if cod is not None else loc[0].normalize()
    end = loc[-1] + pd.Timedelta(hours=site.attrs["dt_h"])
    elapsed = (end - cod) / pd.Timedelta(days=365.25)
    years = warranty_years(site, site_cfg)
    current = int(years[-1])
    in_year = years == current
    efc_year = float(energy[in_year].sum() / e_ref)
    year_start = cod + pd.DateOffset(years=current - 1)
    days_in = (end - year_start) / pd.Timedelta(days=1)
    efc_total = float(energy.sum() / e_ref)
    r = warranty["retention"]
    eq_age = max(elapsed, efc_total / tp["efc_per_year"]) if r["equivalent_age"] and tp["efc_per_year"] \
        else elapsed
    return {"elapsed_years": float(elapsed), "warranty_year": current, "efc_this_year": efc_year,
            "efc_this_year_projected": efc_year * 365.25 / days_in if days_in > 0 else np.nan,
            "efc_per_year_limit": tp["efc_per_year"], "efc_total": efc_total,
            "throughput_mwh": float(energy.sum()), "throughput_allowance_mwh": tp["total_mwh"],
            "equivalent_age_years": float(eq_age),
            "guaranteed_soh_now": float(retention_at(r, eq_age)),
            "guaranteed_soh_by_time": float(retention_at(r, elapsed))}


def guaranteed_soh(retention: dict, elapsed_years, efc_total, efc_per_year) -> np.ndarray:
    """Guaranteed SoH at a test: read at the equivalent age max(years, EFC / allowance) if the warranty says so."""
    efc = efc_total if retention["equivalent_age"] else None
    return contracts.guaranteed_retention(retention["years"], retention["soh"], elapsed_years, efc, efc_per_year)
