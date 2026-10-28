"""Contractual availability of a battery as typically defined in an LTSA, with the formula set in YAML.

Per interval ``i`` the measured availability ``a_i`` follows from the available power the EMS reports in each
direction (discharge, charge) against the contracted power, and for the energy formula also the available energy:

    time:      a_i = 1 if P_avail,d / P_contract,d >= threshold for every direction d, else 0
    capacity:  a_i = min_d min(1, P_avail,d / P_contract,d)          (or the mean over directions)
    energy:    a_i = capacity term x min(1, E_avail / E_contract)

Events from the operator's log (start, end, category) then decide how an interval counts:

    deemed_available   a_i = 1 (e.g. an instruction of the grid operator, a grid outage, an owner request)
    excluded           removed from numerator and denominator (e.g. force majeure, the capacity test itself)
    allowance          excluded up to the planned-maintenance allowance of the period, counted afterwards
    counted            a_i as measured (forced outages)

and the availability of a period (month or LTSA year) is ``A = sum(a_i dt_i) / sum(dt_i)`` over the intervals that
are not excluded. Intervals without data and without an event are excluded, counted as unavailable or as available
(``missing_data``). Which categories exist and how each is treated is contract-specific: the defaults are an
illustration, not a reading of any particular LTSA.

Usage:
    events = read_events(Path("events.csv"), "Europe/Warsaw")
    per_interval = interval_availability(site, ltsa, events)
    periods = period_availability(per_interval, ltsa, site_cfg)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bess_ops.energy import window_keys

# when events overlap, the treatment listed first wins
PRIORITY = ("excluded", "deemed_available", "allowance", "counted")


def read_events(path: Path | None, timezone: str) -> pd.DataFrame:
    """Event log (CSV): start, end, category [, note]. Naive times are local times in ``timezone``."""
    if path is None:
        return pd.DataFrame(columns=["start", "end", "category", "note"])
    ev = pd.read_csv(path)
    return normalise_events(ev, timezone)


def normalise_events(ev: pd.DataFrame, timezone: str) -> pd.DataFrame:
    ev = ev.copy()
    for c in ("start", "end"):
        ts = pd.to_datetime(ev[c], format="ISO8601") if not pd.api.types.is_datetime64_any_dtype(ev[c]) else ev[c]
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="shift_forward")
        ev[c] = ts.dt.tz_convert("UTC")
    if (ev["end"] <= ev["start"]).any():
        raise ValueError("events: end must be after start")
    if "note" not in ev:
        ev["note"] = ""
    return ev[["start", "end", "category", "note"]].reset_index(drop=True)


def measured_availability(site: pd.DataFrame, ltsa: dict) -> pd.Series:
    a = ltsa["availability"]
    c = ltsa["contract"]
    ratios = []
    for d in a["directions"]:
        col, ref = ("p_avail_dis", c["power_mw"]) if d == "discharge" else ("p_avail_ch", c["charge_power_mw"])
        if col not in site:
            raise ValueError(f"availability in direction {d} needs the signal {col}")
        ratios.append((site[col].abs() / ref).clip(upper=1.0))
    r = pd.concat(ratios, axis=1)
    nan = r.isna().any(axis=1)
    if a["formula"] == "time":
        out = (r >= a["threshold"] - 1e-12).all(axis=1).astype(float)
    else:
        out = r.min(axis=1) if a["combine"] == "min" else r.mean(axis=1)
        if a["formula"] == "energy":
            if "e_avail" not in site:
                raise ValueError("the energy formula needs the signal e_avail")
            e = (site["e_avail"] / c["energy_mwh"]).clip(upper=1.0)
            nan |= e.isna()
            out = out * e
    return out.where(~nan)


def event_categories(index: pd.DatetimeIndex, events: pd.DataFrame, treatments: dict) -> pd.Series:
    """Category of each interval (by its start time); overlapping events resolved by PRIORITY of their treatment."""
    cat = pd.Series(pd.NA, index=index, dtype="object")
    if events is None or events.empty:
        return cat
    unknown = set(events["category"]) - set(treatments)
    if unknown:
        raise ValueError(f"events with categories not in the LTSA configuration: {sorted(unknown)}")
    rank = {t: i for i, t in enumerate(PRIORITY)}
    order = events.assign(r=events["category"].map(treatments).map(rank)).sort_values("r", ascending=False)
    for ev in order.itertuples():
        cat[(index >= ev.start) & (index < ev.end)] = ev.category
    return cat


def interval_availability(site: pd.DataFrame, ltsa: dict, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per interval: measured availability, event category and its treatment, interval length (h)."""
    treatments = ltsa["availability"]["categories"]
    out = pd.DataFrame(index=site.index)
    out["measured"] = measured_availability(site, ltsa)
    out["category"] = event_categories(site.index, events, treatments)
    out["treatment"] = out["category"].map(treatments).fillna("counted")
    out["dt_h"] = site.attrs["dt_h"]
    return out


def period_availability(iv: pd.DataFrame, ltsa: dict, site_cfg: dict, period: str | None = None) -> pd.DataFrame:
    """Availability per period (``month`` or ``year`` = LTSA year from COD; default from the LTSA) and hour counts."""
    a = ltsa["availability"]
    period = period or a["period"]
    freq = {"month": "M", "year": "LY" if site_cfg.get("cod") is not None else "Y", "all": None}[period]
    keys = window_keys(iv.index, site_cfg["timezone"], freq, site_cfg.get("cod")) if freq \
        else pd.Series("all", index=iv.index)
    rows = []
    for key, g in iv.groupby(keys.to_numpy(), sort=True):
        rows.append(_one_period(key, g, a))
    return pd.DataFrame(rows).set_index("period")


def _one_period(key, g: pd.DataFrame, a: dict) -> dict:
    dt = g["dt_h"].to_numpy(float)
    hours = dt.sum()
    treat = g["treatment"].to_numpy(object)
    meas = g["measured"].to_numpy(float)
    allowance = a["planned_allowance_hours"] * hours / 8760.0
    is_allow = treat == "allowance"
    used = np.cumsum(np.where(is_allow, dt, 0.0))
    allow_excluded = is_allow & (used <= allowance + 1e-9)
    allow_counted = is_allow & ~allow_excluded
    excluded = (treat == "excluded") | allow_excluded
    deemed = treat == "deemed_available"
    counted = ~excluded & ~deemed
    missing = counted & np.isnan(meas)
    value = np.where(deemed, 1.0, meas)
    md = a["missing_data"]
    if md == "excluded":
        excluded = excluded | missing
    else:
        value = np.where(missing, 1.0 if md == "available" else 0.0, value)
    base = ~excluded
    denom = dt[base].sum()
    num = (np.nan_to_num(value[base]) * dt[base]).sum()
    return {"period": key, "hours": hours, "hours_excluded": dt[excluded].sum(),
            "hours_deemed_available": dt[deemed].sum(), "hours_planned_in_allowance": dt[allow_excluded].sum(),
            "hours_planned_counted": dt[allow_counted].sum(), "hours_missing": dt[missing].sum(),
            "equivalent_unavailable_hours": denom - num,
            "availability": num / denom if denom > 0 else np.nan, "guaranteed": a["guaranteed"]}
