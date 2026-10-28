"""Auxiliary consumption (HVAC, transformers, control) against ambient temperature.

Hourly mean auxiliary power is modelled as a base load plus heating below and cooling above a balance temperature,
plus a term proportional to the PCS power (heat the HVAC has to remove when the battery works)::

    P_aux = b0 + h * max(T_h - T, 0) + c * max(T - T_c, 0) + k * |P_pcs|

The break points ``T_h <= T_c`` are found by grid search, the coefficients by non-negative least squares. Without a
metered auxiliary signal, the import at the meter in hours when the battery is idle (``|P_dc|`` or the
non-auxiliary flow near zero) is used as a proxy; that only sees auxiliaries behind the revenue meter and also
contains transformer no-load losses.

Usage:
    fit = fit_aux_temperature(site, site_cfg)
    fit["params"]          # b0, heat_mw_per_k, t_heat, cool_mw_per_k, t_cool, mw_per_mw, r2
    fit["bins"]            # mean auxiliary power by 2 degC bin of ambient temperature
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import nnls

log = logging.getLogger(__name__)


def hourly_aux(site: pd.DataFrame, site_cfg: dict, idle_share: float = 0.01) -> pd.DataFrame | None:
    """Hourly frame: aux (MW), t_amb, p_abs (mean |PCS power|, MW), source ('metered' or 'idle proxy')."""
    if "t_amb" not in site:
        return None
    hours = site.index.floor("1h")
    if "p_aux" in site:
        aux, source = site["p_aux"], "metered"
        pcs = site["p_meter"] + site["p_aux"] if site_cfg["aux_metering"] == "behind_meter" else site["p_meter"]
    elif site_cfg["aux_metering"] == "behind_meter":
        ref = site["p_dc"] if "p_dc" in site else site["p_meter"]
        idle = ref.abs() <= idle_share * site_cfg["power_mw"]
        if "p_dc" not in site:
            idle &= site["p_meter"] <= 0
        aux = (-site["p_meter"]).where(idle)
        pcs, source = (site["p_dc"] if "p_dc" in site else site["p_meter"] * 0.0), "idle proxy"
    else:
        return None
    h = pd.DataFrame({"aux": aux.groupby(hours).mean(), "t_amb": site["t_amb"].groupby(hours).mean(),
                      "p_abs": pcs.abs().groupby(hours).mean()})
    h["source"] = source
    return h.dropna(subset=["aux", "t_amb", "p_abs"])


def _design(t, p_abs, t_h, t_c):
    return np.column_stack([np.ones_like(t), np.maximum(t_h - t, 0), np.maximum(t - t_c, 0), p_abs])


def fit_aux_temperature(site: pd.DataFrame, site_cfg: dict, grid_step: float = 1.0) -> dict | None:
    h = hourly_aux(site, site_cfg)
    if h is None or len(h) < 48:
        log.info("auxiliary model skipped: needs ambient temperature and auxiliary power (or idle hours)")
        return None
    t, y, pa = h["t_amb"].to_numpy(float), h["aux"].to_numpy(float), h["p_abs"].to_numpy(float)
    lo, hi = np.nanpercentile(t, [2, 98])
    cands = np.arange(np.floor(lo), np.ceil(hi) + grid_step, grid_step)
    best = None
    for i, th in enumerate(cands):
        for tc in cands[i:]:
            x = _design(t, pa, th, tc)
            coef, rnorm = nnls(x, y)
            if best is None or rnorm < best[0]:
                best = (rnorm, th, tc, coef)
    _, th, tc, coef = best
    pred = _design(t, pa, th, tc) @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    params = {"b0_mw": coef[0], "heat_mw_per_k": coef[1], "t_heat_c": th, "cool_mw_per_k": coef[2],
              "t_cool_c": tc, "mw_per_mw": coef[3], "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
              "rmse_mw": np.sqrt(ss_res / len(y)), "hours": len(y), "source": h["source"].iloc[0]}
    edges = np.arange(np.floor(lo / 2) * 2, np.ceil(hi / 2) * 2 + 2, 2.0)
    cut = pd.cut(h["t_amb"], edges)
    bins = h.groupby(cut, observed=True).agg(t_amb=("t_amb", "mean"), aux_mw=("aux", "mean"),
                                             hours=("aux", "size"))
    bins["model_mw"] = _design(bins["t_amb"].to_numpy(), np.full(len(bins), pa.mean()), th, tc) @ coef
    bins.index = [f"{iv.left:.0f}..{iv.right:.0f}" for iv in bins.index]
    return {"params": params, "bins": bins, "hourly": h.assign(model=pred)}


def predict_aux(params: dict, t_amb, p_abs=0.0) -> np.ndarray:
    t = np.asarray(t_amb, float)
    return (params["b0_mw"] + params["heat_mw_per_k"] * np.maximum(params["t_heat_c"] - t, 0)
            + params["cool_mw_per_k"] * np.maximum(t - params["t_cool_c"], 0) + params["mw_per_mw"] * p_abs)
