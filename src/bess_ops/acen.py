"""Validation of the market-data benchmark against an operator's own data: ACEN's New England BESS (NSW).

ACEN Australia publishes 5-minute operating data of the New England BESS (2 x 100 MW / 200 MWh registered, AEMO
units NESBESS1 and NESBESS2): power and state of charge of both sections. The format (as documented in the
companion project ``bess-soc-soh-degradation``, which downloads the files): one CSV per month, the time stamp is
the START of the interval in NEM time, power is the interval average in kW although the header says W, the last
four columns are power and SoC (%) of section 1 and section 2. No licence is stated, so the files stay local
(``--acen-dir`` or ``BESS_ACEN_DIR``) and only derived statistics are published.

The validation computes the same metrics twice, from the operator's data (its power and SoC) and from the market
data (AEMO SCADA and the stored energy reported to AEMO), and compares them:

* the consistency check of the reported energy (it should flag the months in which NESBESS1 reported its headroom
  instead of its energy, as found in the companion project);
* daily and monthly RTE with the SoC correction; the operator's SoC is converted to energy with the usable
  energy ``E = 1 / sqrt(a b)`` from the hourly regression of SoC changes on charged and discharged energy
  (symmetric split of the efficiency, as in the companion project);
* cycles: discharged energy per registered MWh and rainflow cycles of the SoC; time near full and empty.

Usage:
    frame = load(Path("../bess-soc-soh-degradation/data/acen"))
    out = validate(cache, Path("../bess-soc-soh-degradation/data/acen"))
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bess_ops.cycles import rainflow, rainflow_efc
from bess_ops.energy import interval_flows, inverted_days, state_jumps, window_keys, window_rte
from bess_ops.nem import NEM_TZ, Cache, unit_benchmark, unit_site
from bess_ops.schema import site_frame

log = logging.getLogger(__name__)
SECTIONS = {1: "NESBESS1", 2: "NESBESS2"}
REGISTERED_MWH = 200.0
# documented in the companion project: NESBESS1 reported its headroom until 5 August 2026; the validation detects
# the first day of a correct report itself (day after the last day with a correlation below -0.5) and compares
FIXED_REPORTING = {"NESBESS1": "2026-08-06"}


def parse(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=0, dtype=str)
    stamp = raw.iloc[:, 0].str.strip()
    fmt = "%d/%m/%Y %H:%M" if stamp.str.contains("/").any() else "%Y-%m-%d %H:%M:%S.%f"
    idx = pd.DatetimeIndex(pd.to_datetime(stamp, format=fmt), name="interval_start")
    num = raw.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").set_axis(idx)
    return pd.DataFrame({"mw_1": num.iloc[:, -4].to_numpy() / 1000, "soc_1": num.iloc[:, -3].to_numpy() / 100,
                         "mw_2": num.iloc[:, -2].to_numpy() / 1000, "soc_2": num.iloc[:, -1].to_numpy() / 100},
                        index=idx)


def load(acen_dir: Path) -> pd.DataFrame:
    files = sorted(Path(acen_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no ACEN CSV files in {acen_dir}")
    df = pd.concat([parse(f) for f in files]).sort_index()
    return df[~df.index.duplicated(keep="first")]


def operator_site(acen: pd.DataFrame, k: int) -> tuple[pd.DataFrame, dict]:
    """Site frame of one section from the operator's data; the SoC is turned into energy with the fitted scale."""
    grid = pd.date_range(acen.index.min(), acen.index.max(), freq="5min")
    a = acen.reindex(grid)
    site = site_frame({"p_meter": a[f"mw_{k}"].to_numpy(), "soc": a[f"soc_{k}"].to_numpy()},
                      grid.tz_localize(NEM_TZ), NEM_TZ)
    dt = site.attrs["dt_h"]
    e_out = site["p_meter"].clip(lower=0) * dt
    e_in = (-site["p_meter"]).clip(lower=0) * dt
    soc_adj, jumps, fit = state_jumps(site["soc"], e_in, e_out, 1.0)
    scale = 1 / np.sqrt(fit["a"] * fit["b"]) if fit["a"] > 0 and fit["b"] > 0 else np.nan
    site["e_stored"] = soc_adj * scale
    return site.drop(columns="soc"), {"scale_mwh": scale, "rte_regression": fit["a"] / fit["b"],
                                      "standby_mw": fit["c"] * scale, "soc_jumps": len(jumps), "r2": fit["r2"]}


def validate(cache: Cache, acen_dir: Path) -> dict:
    """Operator vs market metrics for both sections.

    Daily RTE in three variants separates the metering point from the state source:
    A = operator power + operator SoC, B = AEMO SCADA + operator SoC, C = AEMO SCADA + energy reported to AEMO.
    """
    acen = load(acen_dir)
    rows, daily_rows, month_rows = [], [], []
    for k, duid in SECTIONS.items():
        op, cal = operator_site(acen, k)
        mk = unit_site(cache, duid)
        common = op.index.intersection(mk.index)
        op, mk = op.loc[common], mk.loc[common]
        cfg = {"aux_metering": "none", "soc_energy_mwh": cal["scale_mwh"], "timezone": NEM_TZ, "cod": None}
        d_a = window_rte(op, cfg, "D")
        m_a = window_rte(op, cfg, "M", float(d_a["eta"].iloc[0]), min_efc=1.0)
        # B: SCADA flows with the operator's SoC (scale fitted on the SCADA flows)
        soc = pd.Series(acen[f"soc_{k}"].reindex(common.tz_convert(NEM_TZ).tz_localize(None)).to_numpy(),
                        index=common)
        soc_adj, _, fit_b = state_jumps(soc, mk["e_in_exact"], mk["e_out_exact"], 1.0)
        scale_b = 1 / np.sqrt(fit_b["a"] * fit_b["b"])
        b_site = mk[["p_meter"]].copy()
        b_site.attrs.update(mk.attrs)
        b_site["e_stored"] = soc_adj * scale_b
        d_b = window_rte(b_site, dict(cfg, soc_energy_mwh=scale_b), "D", "auto", _flows(b_site, mk))
        monthly, summary, d_c = unit_benchmark(cache, duid)
        corr = inverted_days(mk["e_stored"], mk["e_in_exact"], mk["e_out_exact"], NEM_TZ)
        bad = corr.index[corr < -0.5]
        detected = (pd.Timestamp(bad[-1]) + pd.Timedelta(days=1)).date().isoformat() if len(bad) else corr.index[0]
        fixed = detected
        days = d_a.index[d_a["valid"]].intersection(d_b.index[d_b["valid"]]).intersection(d_c.index)
        if fixed:
            days = days[days >= fixed]
        both = pd.DataFrame({"duid": duid, "rte_operator": d_a.loc[days, "rte"],
                             "rte_scada_operator_soc": d_b.loc[days, "rte"], "rte_market": d_c.loc[days]})
        daily_rows.append(both)
        mm = monthly.set_index(monthly.index.astype(str))
        dt = mk.attrs["dt_h"]
        mkeys = window_keys(common, NEM_TZ, "M").to_numpy()
        for m in m_a.index:
            sel = mkeys == m
            if m not in mm.index or sel.sum() < 288 * 10:
                continue
            op_out = float((op["p_meter"][sel].clip(lower=0) * dt).sum())
            op_in = float(((-op["p_meter"][sel]).clip(lower=0) * dt).sum())
            sc_out, sc_in = float(mk["e_out_exact"][sel].sum()), float(mk["e_in_exact"][sel].sum())
            month_rows.append({"duid": duid, "month": m, "e_out_operator_mwh": op_out, "e_out_scada_mwh": sc_out,
                               "e_in_operator_mwh": op_in, "e_in_scada_mwh": sc_in,
                               "rte_raw_operator": op_out / op_in if op_in else np.nan,
                               "rte_raw_scada": sc_out / sc_in if sc_in else np.nan,
                               "rte_operator": m_a.loc[m, "rte"], "rte_market": mm.loc[m, "rte"],
                               "extra_import_mw": ((sc_in - op_in) - (op_out - sc_out)) / (sel.sum() * dt),
                               "market_stored_consistent": bool(mm.loc[m, "stored_consistent"]),
                               "market_stored_r2": mm.loc[m, "stored_r2"]})
        start = pd.Timestamp(fixed).tz_localize(NEM_TZ) if fixed else common[0]
        after = common >= start
        soc_raw = soc[after]
        rep_raw = (mk["e_stored"] / REGISTERED_MWH)[after]
        rep_adj = _market_soc(mk[after])
        rf_op = rainflow_efc(rainflow(soc_adj[after].to_numpy(), 0.01))
        rf_mkt = rainflow_efc(rainflow(rep_adj.to_numpy(), 0.01))
        has = len(both) > 0
        rows.append({
            "duid": duid, "operator_scale_mwh": cal["scale_mwh"], "operator_rte_regression": cal["rte_regression"],
            "operator_standby_mw": cal["standby_mw"], "operator_soc_jumps": cal["soc_jumps"],
            "scada_scale_mwh": scale_b, "reported_correctly_from": start.date().isoformat(),
            "documented_fix": FIXED_REPORTING.get(duid, ""), "inverted_days": int(len(bad)),
            "days": float(after.sum() * dt / 24),
            "efc_discharge_scada": float(mk["e_out_exact"][after].sum() / REGISTERED_MWH),
            "rainflow_efc_operator_soc": rf_op,
            "rainflow_efc_operator_mwh_basis": rf_op * cal["scale_mwh"] / REGISTERED_MWH,
            "rainflow_efc_market": rf_mkt,
            "soc_above_95_operator": float((soc_raw >= 0.95).mean()),
            "soc_above_95_market": float((rep_raw >= 0.95).mean()),
            "soc_below_5_operator": float((soc_raw <= 0.05).mean()),
            "soc_below_5_market": float((rep_raw <= 0.05).mean()),
            "daily_days": len(both),
            "daily_rte_median_A_operator": float(both["rte_operator"].median()) if has else np.nan,
            "daily_rte_median_B_scada_operator_soc": float(both["rte_scada_operator_soc"].median()) if has else np.nan,
            "daily_rte_median_C_market": float(both["rte_market"].median()) if has else np.nan,
            "daily_mean_abs_diff_B_C": float((both["rte_market"] - both["rte_scada_operator_soc"]).abs().mean())
            if has else np.nan,
            "daily_mean_abs_diff_A_B": float((both["rte_scada_operator_soc"] - both["rte_operator"]).abs().mean())
            if has else np.nan,
            "inconsistent_months_flagged": ", ".join(mm.index[~mm["stored_consistent"].astype(bool)
                                                             & mm["operational"].astype(bool)]),
        })
        log.info("ACEN %s: operator scale %.0f MWh, %d common days", duid, cal["scale_mwh"], len(both))
    return {"summary": pd.DataFrame(rows).set_index("duid"), "daily": pd.concat(daily_rows),
            "monthly": pd.DataFrame(month_rows)}


def _flows(site: pd.DataFrame, mk: pd.DataFrame) -> pd.DataFrame:
    f = interval_flows(site, {"aux_metering": "none", "soc_energy_mwh": 1.0})
    f["e_out"], f["e_in"] = mk["e_out_exact"], mk["e_in_exact"]
    return f


def _market_soc(mk: pd.DataFrame) -> pd.Series:
    """Reported stored energy / registered MWh with BMS jumps removed."""
    adj, _, _ = state_jumps(mk["e_stored"], mk["e_in_exact"], mk["e_out_exact"], REGISTERED_MWH)
    return adj / REGISTERED_MWH
