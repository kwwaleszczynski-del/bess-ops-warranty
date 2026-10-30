"""Illustrative liquidated damages (LD) for missed availability, RTE and capacity guarantees.

All rates, caps and formulas are ILLUSTRATIVE placeholders to show the mechanics; LD clauses differ between
contracts and must be read from the actual LTSA or warranty.

    availability  LD = rate_per_pp x max(0, A_guaranteed - A) x 100      (x contracted MW if per_mw)
    RTE, per_pp   LD = rate_per_pp x max(0, RTE_guaranteed - RTE) x 100
    RTE, energy_cost
                  LD = E_out x (1 / RTE - 1 / RTE_guaranteed) x price     (extra energy bought to deliver E_out)
    capacity      LD = rate_per_mwh x max(0, E_guaranteed - E_corrected)  (per failed test)

The total per LTSA year is limited to ``cap_share_of_fee`` x ``annual_fee`` when both are set. The amounts are
computed by the res-asset-kpi library (``res_asset_kpi.contracts.damages_amount``).

Usage:
    ld = liquidated_damages(availability_periods, rte_windows, tests, ltsa)
    ld["items"], ld["total"], ld["capped_total"]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from res_asset_kpi.contracts import Damages, damages_amount


def availability_ld(periods: pd.DataFrame, ltsa: dict) -> pd.DataFrame:
    cfg = ltsa["ld"]["availability"]
    g = ltsa["availability"]["guaranteed"]
    short = ((g - periods["availability"]).clip(lower=0) * 100).fillna(0)
    d = Damages(method="per_point", rate_per_pp=cfg["rate_per_pp"],
                multiplier=ltsa["contract"]["power_mw"] if cfg["per_mw"] else 1.0)
    amount = pd.Series(np.nan_to_num(damages_amount(d, periods["availability"], g)), index=periods.index)
    return pd.DataFrame({"kind": "availability", "period": periods.index.astype(str), "measured":
                         periods["availability"].to_numpy(), "guaranteed": g, "shortfall": short.to_numpy() / 100,
                         "amount": amount.to_numpy()})


def rte_ld(windows: pd.DataFrame, ltsa: dict) -> pd.DataFrame:
    cfg = ltsa["ld"]["rte"]
    r = ltsa["rte"]
    col = "rte" if r["definition"] == "incl_aux" else "rte_excl_aux"
    g = r["guaranteed"]
    assessed = windows[windows["valid"]] if "valid" in windows else windows[windows["efc"] >= r["min_efc"]]
    rte = assessed[col]
    short = (g - rte).clip(lower=0).fillna(0)
    if cfg["method"] == "energy_cost":
        d = Damages(method="rte_energy_cost", price_per_mwh=cfg["price_per_mwh"])
    else:
        d = Damages(method="per_point", rate_per_pp=cfg["rate_per_pp"])
    amount = pd.Series(np.nan_to_num(damages_amount(d, rte, g, e_out_mwh=assessed["e_out"])), index=rte.index)
    return pd.DataFrame({"kind": "rte", "period": assessed.index.astype(str), "measured": rte.to_numpy(),
                         "guaranteed": g, "shortfall": short.to_numpy(), "amount": amount.to_numpy()})


def capacity_ld(tests: pd.DataFrame, ltsa: dict) -> pd.DataFrame:
    cfg = ltsa["ld"]["capacity"]
    if tests is None or tests.empty:
        return pd.DataFrame(columns=["kind", "period", "measured", "guaranteed", "shortfall", "amount"])
    failed = tests["verdict"] == "fail"
    short = (tests["e_guaranteed_mwh"] - tests["e_corr_mwh"]).clip(lower=0).where(failed, 0.0)
    amount = damages_amount(Damages(method="per_mwh", rate_per_mwh=cfg["rate_per_mwh"]), tests["e_corr_mwh"],
                            tests["e_guaranteed_mwh"])
    return pd.DataFrame({"kind": "capacity", "period": tests["start"].astype(str),
                         "measured": tests["e_corr_mwh"].to_numpy(), "guaranteed": tests["e_guaranteed_mwh"].to_numpy(),
                         "shortfall": short.to_numpy(), "amount": np.where(failed, amount, 0.0)})


def liquidated_damages(periods: pd.DataFrame | None, windows: pd.DataFrame | None, tests: pd.DataFrame | None,
                       ltsa: dict) -> dict:
    parts = []
    if periods is not None and len(periods):
        parts.append(availability_ld(periods, ltsa))
    if windows is not None and len(windows):
        parts.append(rte_ld(windows, ltsa))
    if tests is not None and len(tests):
        parts.append(capacity_ld(tests, ltsa))
    items = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["kind", "period", "measured", "guaranteed", "shortfall", "amount"])
    total = float(items["amount"].sum()) if len(items) else 0.0
    cfg = ltsa["ld"]
    cap = cfg["annual_fee"] * cfg["cap_share_of_fee"] if cfg["annual_fee"] and cfg["cap_share_of_fee"] else None
    return {"items": items, "total": total, "cap": cap,
            "capped_total": min(total, cap) if cap is not None else total, "currency": cfg["currency"]}
