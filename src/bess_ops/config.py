"""Site, warranty and LTSA terms from YAML files, merged over documented defaults.

Every contractual value here is ILLUSTRATIVE: the defaults and the example files in ``config/`` are generalised from
public sources (product data sheets, DNV and NREL publications), not from any real contract. Check every term
against the actual warranty and LTSA before using a result.

Unknown keys are an error, so a typo does not silently fall back to a default.

Usage:
    site = load_site(Path("config/site.example.yaml"))
    warranty = load_warranty(Path("config/warranty.example.yaml"), site)
    ltsa = load_ltsa(Path("config/ltsa.example.yaml"), site)
    ltsa["availability"]["guaranteed"]        # nested dicts with every key present
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SITE_DEFAULTS = {
    "name": "BESS",
    "power_mw": None,            # contracted discharge power at the meter
    "charge_power_mw": None,     # default: power_mw
    "energy_mwh": None,          # nameplate (contracted) energy at the meter, beginning of life
    "soc_energy_mwh": None,      # energy that SoC 0..1 stands for; default energy_mwh
    "timezone": "Europe/Warsaw",
    "aux_metering": "behind_meter",  # behind_meter | separate | none
    "cod": None,                 # commercial operation date: start of warranty and LTSA years (local date)
}

WARRANTY_DEFAULTS = {
    "term_years": 15,
    "reference_energy_mwh": None,    # energy the cycles are counted on; default site energy_mwh
    "throughput": {
        "basis": "discharge_meter",  # discharge_meter | discharge_dc | half_throughput_meter
        "efc_per_year": 365.0,       # per warranty year
        "efc_per_day": None,
        "total_mwh": None,           # cumulative throughput allowance on the same basis
    },
    "soc": {"min": None, "max": None, "sustained_minutes": 15, "max_mean": None, "mean_window_days": 30},
    "c_rate": {"max": None, "basis": "p_dc", "sustained_minutes": 15},
    "temperature": {"cell_max_c": None, "cell_min_c": None, "sustained_minutes": 30,
                    "annual_mean_cell_max_c": None, "ambient_min_c": None, "ambient_max_c": None},
    "retention": {"years": [0, 15], "soh": [1.0, 0.7], "equivalent_age": True},
}

LTSA_DEFAULTS = {
    "contract": {"power_mw": None, "charge_power_mw": None, "energy_mwh": None},
    "availability": {
        "formula": "time",           # time | capacity | energy
        "period": "year",            # month | year (LTSA years from COD)
        "threshold": 0.95,           # time formula: available if P_avail >= threshold x contracted
        "directions": ["discharge", "charge"],
        "combine": "min",            # capacity/energy formula: min | mean over directions
        "missing_data": "excluded",  # excluded | unavailable | available
        "guaranteed": 0.97,
        "planned_allowance_hours": 0.0,  # per LTSA year, pro rata for shorter periods
        "categories": {
            "tso_instruction": "deemed_available",
            "grid_outage": "deemed_available",
            "owner_request": "deemed_available",
            "force_majeure": "excluded",
            "capacity_test": "excluded",
            "planned_maintenance": "allowance",
            "forced_outage": "counted",
        },
    },
    "rte": {"definition": "incl_aux", "window": "month", "guaranteed": 0.85, "min_efc": 5.0,
            "one_way_efficiency": "auto"},
    "capacity": {
        "guaranteed": "warranty",    # warranty (its retention curve) | a share of energy_mwh | {years:, soh:}
        "test": {
            "power_mw": None, "start_soc_min": 0.97, "end_soc_max": 0.03, "steady_tolerance": 0.05,
            "min_duration_h": 1.0, "measure": "meter", "aux_correction": "none",  # none | add_aux
            "t_ref_c": 25.0, "temp_signal": "t_cell_avg", "temp_coeff_per_k": 0.0, "temp_coeff_u_per_k": 0.0,
            "meter_class_pct": 0.2, "ct_pct": 0.2, "vt_pct": 0.2, "temp_u_k": 1.0, "coverage_k": 2.0,
            "decision_rule": "simple",   # simple | guard_band
        },
        # SoH forecast: prior of the fade rates from the LFP cell fit of the companion project
        # bess-capacity-augmentation (AgingModel defaults, Yagci et al. 2025 cells, 25 degC reference)
        "forecast": {
            "required_mwh": None, "horizon_years": 20, "efc_per_year": "auto", "cell_temp_c": "auto",
            "prior_k_cal_per_sqrt_day": 1.64e-3, "prior_ea_cal": 30_500.0,
            "prior_k_cyc_per_efc": 2.1e-5, "prior_ea_cyc": 31_800.0, "prior_rel_sd": 0.5,
        },
    },
    "ld": {
        "currency": "EUR",
        "annual_fee": None,
        "cap_share_of_fee": None,
        "availability": {"rate_per_pp": 0.0, "per_mw": False},
        "rte": {"method": "energy_cost", "price_per_mwh": 0.0, "rate_per_pp": 0.0},
        "capacity": {"rate_per_mwh": 0.0},
    },
}

AVAILABILITY_TREATMENTS = ("deemed_available", "excluded", "counted", "allowance")


def deep_merge(defaults: dict, override: dict | None, path: str = "") -> dict:
    out = copy.deepcopy(defaults)
    for k, v in (override or {}).items():
        if k not in defaults:
            raise ValueError(f"unknown key {path + k!r}; expected one of {sorted(defaults)}")
        if isinstance(defaults[k], dict) and k != "categories":
            if not isinstance(v, dict):
                raise ValueError(f"{path + k} must be a mapping")
            out[k] = deep_merge(defaults[k], v, path + k + ".")
        else:
            out[k] = v
    return out


def _read(path: Path | None) -> dict:
    if path is None:
        return {}
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def site_from_dict(raw: dict) -> dict:
    s = deep_merge(SITE_DEFAULTS, raw)
    for key in ("power_mw", "energy_mwh"):
        if s[key] is None:
            raise ValueError(f"site: {key} is required")
    s["charge_power_mw"] = s["charge_power_mw"] or s["power_mw"]
    s["soc_energy_mwh"] = s["soc_energy_mwh"] or s["energy_mwh"]
    if s["aux_metering"] not in ("behind_meter", "separate", "none"):
        raise ValueError("site: aux_metering must be behind_meter, separate or none")
    s["cod"] = pd.Timestamp(s["cod"]) if s["cod"] is not None else None
    return s


def load_site(path: Path) -> dict:
    return site_from_dict(_read(path))


def warranty_from_dict(raw: dict, site: dict) -> dict:
    w = deep_merge(WARRANTY_DEFAULTS, raw)
    w["reference_energy_mwh"] = w["reference_energy_mwh"] or site["energy_mwh"]
    r = w["retention"]
    if len(r["years"]) != len(r["soh"]) or sorted(r["years"]) != list(r["years"]):
        raise ValueError("warranty.retention: years must be increasing and as many as soh values")
    return w


def load_warranty(path: Path | None, site: dict) -> dict:
    return warranty_from_dict(_read(path), site)


def ltsa_from_dict(raw: dict, site: dict) -> dict:
    lt = deep_merge(LTSA_DEFAULTS, raw)
    c = lt["contract"]
    c["power_mw"] = c["power_mw"] or site["power_mw"]
    c["charge_power_mw"] = c["charge_power_mw"] or site["charge_power_mw"]
    c["energy_mwh"] = c["energy_mwh"] or site["energy_mwh"]
    cats = dict(LTSA_DEFAULTS["availability"]["categories"])
    cats.update((raw.get("availability") or {}).get("categories") or {})
    bad = {k: v for k, v in cats.items() if v not in AVAILABILITY_TREATMENTS}
    if bad:
        raise ValueError(f"availability.categories: unknown treatment {bad}; use {AVAILABILITY_TREATMENTS}")
    lt["availability"]["categories"] = cats
    if lt["availability"]["formula"] not in ("time", "capacity", "energy"):
        raise ValueError("availability.formula must be time, capacity or energy")
    lt["capacity"]["test"]["power_mw"] = lt["capacity"]["test"]["power_mw"] or c["power_mw"]
    return lt


def load_ltsa(path: Path | None, site: dict) -> dict:
    return ltsa_from_dict(_read(path), site)


def retention_at(retention: dict, years) -> np.ndarray | float:
    """Guaranteed state of health (share of the reference energy) after ``years``, linear between table points."""
    return np.interp(years, retention["years"], retention["soh"])
