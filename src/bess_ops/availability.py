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
illustration, not a reading of any particular LTSA. The formulas, the event treatments and the period availability
are computed by the res-asset-kpi library (``res_asset_kpi.bess``).

Usage:
    events = read_events(Path("events.csv"), "Europe/Warsaw")
    per_interval = interval_availability(site, ltsa, events)
    periods = period_availability(per_interval, ltsa, site_cfg, events=events)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from res_asset_kpi import bess
from res_asset_kpi.bess import PRIORITY  # noqa: F401  (when events overlap, the treatment listed first wins)

from bess_ops.energy import window_keys


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
    return bess.measured_availability(site, power=c["power_mw"], charge_power=c["charge_power_mw"],
                                      formula=a["formula"], threshold=a["threshold"], directions=a["directions"],
                                      combine=a["combine"], energy=c["energy_mwh"])


def interval_availability(site: pd.DataFrame, ltsa: dict, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per interval: measured availability, event category and its treatment, interval length (h)."""
    treatments = ltsa["availability"]["categories"]
    out = pd.DataFrame(index=site.index)
    out["measured"] = measured_availability(site, ltsa)
    tr = bess.interval_treatment(site.index, events, treatments)
    out["category"], out["treatment"] = tr["category"], tr["treatment"]
    out["dt_h"] = site.attrs["dt_h"]
    return out


def period_availability(iv: pd.DataFrame, ltsa: dict, site_cfg: dict, period: str | None = None,
                        events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Availability per period (``month`` or ``year`` = LTSA year from COD; default from the LTSA) and hour counts.

    ``iv`` is the output of ``interval_availability`` (possibly cut at a report date) and ``events`` the event log it
    was built with.
    """
    a = ltsa["availability"]
    period = period or a["period"]
    freq = {"month": "M", "year": "LY" if site_cfg.get("cod") is not None else "Y", "all": None}[period]
    keys = window_keys(iv.index, site_cfg["timezone"], freq, site_cfg.get("cod")) if freq \
        else pd.Series("all", index=iv.index)
    return bess.ltsa_availability(iv["measured"], events=events, treatments=a["categories"],
                                  planned_allowance_h=a["planned_allowance_hours"], missing_data=a["missing_data"],
                                  periods=keys.to_numpy(), guaranteed=a["guaranteed"])
