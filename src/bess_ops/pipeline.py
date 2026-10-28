"""Every estimator on one site frame: the analysis behind the monthly report.

Usage:
    res = analyse(site, site_cfg, warranty, ltsa, events)
    res["monthly"]          # KPI table, one row per local calendar month
    res["violations"]       # warranty violations with the event category they overlap
    res["ld"]               # illustrative liquidated damages per LTSA year
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from bess_ops import availability as av
from bess_ops.auxiliaries import fit_aux_temperature
from bess_ops.capacity import evaluate_tests, find_test_windows, forecast_soh, handoff
from bess_ops.cycles import depth_histogram, rainflow, rainflow_efc, throughput_energy
from bess_ops.energy import energy_balance, interval_flows, window_keys, window_rte
from bess_ops.ld import liquidated_damages
from bess_ops.schema import data_quality
from bess_ops.warranty import detect_violations, warranty_status

log = logging.getLogger(__name__)
AGGREGATE_RULES = ("efc_per_year", "throughput_total", "cell_temp_annual")


def soc_series(site: pd.DataFrame, site_cfg: dict) -> pd.Series | None:
    if "soc" in site:
        return site["soc"]
    if "e_stored" in site:
        return site["e_stored"] / site_cfg["soc_energy_mwh"]
    return None


def violation_context(violations: pd.DataFrame, events: pd.DataFrame | None) -> pd.Series:
    """Category of the logged events a violation overlaps (e.g. a capacity test), empty if none."""
    if events is None or events.empty or violations.empty:
        return pd.Series("", index=violations.index)
    out = []
    for v in violations.itertuples():
        if v.rule in AGGREGATE_RULES:
            out.append("")
            continue
        hit = events[(events["start"] < v.end) & (events["end"] > v.start)]
        out.append(", ".join(sorted(set(hit["category"]))))
    return pd.Series(out, index=violations.index)


def monthly_kpis(site, site_cfg, warranty, ltsa, monthly_rte, avail_m, violations) -> pd.DataFrame:
    keys = window_keys(site.index, site_cfg["timezone"], "M")
    dt = site.attrs["dt_h"]
    e_ref = warranty["reference_energy_mwh"]
    k = pd.DataFrame(index=monthly_rte.index)
    k["e_in_mwh"] = monthly_rte["e_in"]
    k["e_out_mwh"] = monthly_rte["e_out"]
    k["e_aux_mwh"] = monthly_rte["e_aux"]
    k["rte_raw"] = monthly_rte["rte_raw"]
    k["rte"] = monthly_rte["rte"]
    k["rte_excl_aux"] = monthly_rte["rte_excl_aux"]
    k["rte_guaranteed"] = ltsa["rte"]["guaranteed"]
    if avail_m is not None:
        k["availability"] = avail_m["availability"].reindex(k.index)
        k["availability_guaranteed"] = ltsa["availability"]["guaranteed"]
    tp = throughput_energy(site, warranty["throughput"]["basis"])
    k["efc"] = (tp.groupby(keys.to_numpy()).sum() / e_ref).reindex(k.index)
    soc = soc_series(site, site_cfg)
    if soc is not None:
        g = soc.groupby(keys.to_numpy())
        k["rainflow_efc"] = g.apply(lambda s: rainflow_efc(rainflow(s.to_numpy(), 0.01))).reindex(k.index)
        k["mean_soc"] = g.mean().reindex(k.index)
        w = warranty["soc"]
        if w["max"] is not None:
            k["hours_soc_above_max"] = ((soc > w["max"]).groupby(keys.to_numpy()).sum() * dt).reindex(k.index)
        if w["min"] is not None:
            k["hours_soc_below_min"] = ((soc < w["min"]).groupby(keys.to_numpy()).sum() * dt).reindex(k.index)
    basis = warranty["c_rate"]["basis"] if warranty["c_rate"]["basis"] in site else "p_meter"
    k["max_c_rate"] = (site[basis].abs() / e_ref).groupby(keys.to_numpy()).max().reindex(k.index)
    for col in ("t_cell_max", "t_cell_avg", "t_amb"):
        if col in site:
            agg = "max" if col == "t_cell_max" else "mean"
            k[f"{col}_{agg}"] = getattr(site[col].groupby(keys.to_numpy()), agg)().reindex(k.index)
    if not violations.empty:
        vk = window_keys(pd.DatetimeIndex(violations["start"]), site_cfg["timezone"], "M")
        k["violations"] = vk.value_counts().reindex(k.index).fillna(0).astype(int)
    else:
        k["violations"] = 0
    k["coverage"] = monthly_rte["coverage"]
    k.index.name = "month"
    return k


def analyse(site: pd.DataFrame, site_cfg: dict, warranty: dict, ltsa: dict,
            events: pd.DataFrame | None = None) -> dict:
    res: dict = {"site_cfg": site_cfg, "warranty": warranty, "ltsa": ltsa}
    res["quality"] = data_quality(site, site_cfg["soc_energy_mwh"])
    eta = ltsa["rte"]["one_way_efficiency"]
    flows = interval_flows(site, site_cfg)
    res["daily"] = window_rte(site, site_cfg, "D", eta, flows)
    eta_used = float(res["daily"]["eta"].iloc[0])
    res["monthly_rte"] = window_rte(site, site_cfg, "M", eta_used, flows, ltsa["rte"]["min_efc"])
    year_freq = "LY" if site_cfg.get("cod") is not None else "Y"
    res["yearly_rte"] = window_rte(site, site_cfg, year_freq, eta_used, flows)
    res["balance"] = energy_balance(site, site_cfg, "M", flows)
    res["aux"] = fit_aux_temperature(site, site_cfg)

    if {"p_avail_dis", "p_avail_ch"} & set(site.columns):
        iv = av.interval_availability(site, ltsa, events)
        res["availability_interval"] = iv
        res["availability_month"] = av.period_availability(iv, ltsa, site_cfg, "month")
        res["availability_year"] = av.period_availability(iv, ltsa, site_cfg, "year")
    else:
        log.info("availability skipped: no available-power signals")
        res["availability_month"] = res["availability_year"] = None

    v = detect_violations(site, warranty, site_cfg)
    v["context"] = violation_context(v, events)
    res["violations"] = v

    soc = soc_series(site, site_cfg)
    if soc is not None:
        cyc = rainflow(soc.to_numpy(), 0.01)
        years = len(site) * site.attrs["dt_h"] / 8760
        res["rainflow"] = cyc
        res["depth_histogram"] = depth_histogram(cyc, years)

    windows = find_test_windows(site, ltsa, events)
    tests = evaluate_tests(site, windows, ltsa, warranty, site_cfg) if len(windows) else pd.DataFrame()
    res["tests"] = tests
    res["site"] = site
    res.update(state_at(res, None))
    res["monthly"] = monthly_kpis(site, site_cfg, warranty, ltsa, res["monthly_rte"], res["availability_month"],
                                  v)
    return res


def state_at(res: dict, end: pd.Timestamp | None) -> dict:
    """Warranty status, LTSA-year-to-date availability, SoH forecast and LD as they stood at instant ``end``
    (UTC; None = the end of the record), so that a monthly report shows only what was known at its month end."""
    site, site_cfg, warranty, ltsa = res["site"], res["site_cfg"], res["warranty"], res["ltsa"]
    upto = site if end is None else site[site.index < end]
    out = {"warranty_status": warranty_status(upto, warranty, site_cfg)}
    iv = res.get("availability_interval")
    out["availability_year"] = None if iv is None else av.period_availability(
        iv if end is None else iv[iv.index < end], ltsa, site_cfg, "year")
    tests = res["tests"]
    if end is not None and len(tests):
        tests = tests[pd.DatetimeIndex(tests["end"]) <= end]
    out["forecast"] = None
    out["handoff"] = None
    if len(tests) >= 2:
        fc = ltsa["capacity"]["forecast"]
        tp = throughput_energy(upto, warranty["throughput"]["basis"]).fillna(0)
        last_year = tp[tp.index >= tp.index[-1] - pd.Timedelta(days=365)]
        span = (last_year.index[-1] - last_year.index[0]) / pd.Timedelta(days=365.25)
        efc_rate = float(last_year.sum() / warranty["reference_energy_mwh"] / max(span, 1e-9)) \
            if fc["efc_per_year"] == "auto" else float(fc["efc_per_year"])
        if fc["cell_temp_c"] == "auto":
            col = next((c for c in ("t_cell_avg", "t_cell_max") if c in upto), None)
            temp = float(upto[col].mean()) if col else 25.0
        else:
            temp = float(fc["cell_temp_c"])
        st = out["warranty_status"]
        out["forecast"] = forecast_soh(tests, ltsa, warranty, site_cfg, efc_rate, temp,
                                       now_years=st["elapsed_years"], now_efc=st["efc_total"])
        out["handoff"] = handoff(out["forecast"], site_cfg, warranty, ltsa)
    windows = res["monthly_rte"] if ltsa["rte"]["window"] == "month" else res["yearly_rte"]
    if end is not None and ltsa["rte"]["window"] == "month":
        month_end = end.tz_convert(site_cfg["timezone"])
        windows = windows[windows.index < month_end.strftime("%Y-%m")]
    out["ld"] = ld_by_year(out["availability_year"], windows, tests, ltsa, site_cfg)
    return out


def as_of(res: dict, month: str) -> dict:
    """``state_at`` the end of a local calendar month (YYYY-MM)."""
    tz = res["site_cfg"]["timezone"]
    end = (pd.Timestamp(month + "-01") + pd.offsets.MonthBegin(1)).tz_localize(tz).tz_convert("UTC")
    return state_at(res, end)


def ld_by_year(avail_year, rte_windows, tests, ltsa, site_cfg) -> pd.DataFrame:
    """Illustrative LD per LTSA year (cap applied per year)."""
    cod = site_cfg.get("cod")
    tz = site_cfg["timezone"]

    def year_of(ts_index) -> np.ndarray:
        idx = pd.DatetimeIndex(ts_index)
        if cod is None:
            return idx.tz_convert(tz).strftime("%Y").to_numpy()
        return window_keys(idx, tz, "LY", cod).to_numpy()

    rows, items = [], []
    years = list(avail_year.index) if avail_year is not None else []
    if rte_windows is not None and len(rte_windows):
        labels = rte_windows.index.astype(str)
        rte_years = labels.to_numpy() if labels.str.fullmatch(r"\d{1,4}").all() \
            else year_of(_window_starts(rte_windows, tz))
    else:
        rte_years = np.array([])
    test_years = year_of(tests["start"]) if tests is not None and len(tests) else np.array([])
    years = sorted(set(map(str, years)) | set(map(str, rte_years)) | set(map(str, test_years)))
    for y in years:
        ay = avail_year.loc[[y]] if avail_year is not None and y in avail_year.index.astype(str) else None
        rw = rte_windows[rte_years == y] if len(rte_years) else None
        ts = tests[test_years == y] if len(test_years) else None
        ld = liquidated_damages(ay, rw, ts, ltsa)
        it = ld["items"].assign(ltsa_year=y)
        items.append(it)
        by = it.groupby("kind")["amount"].sum()
        rows.append({"ltsa_year": y, "availability": by.get("availability", 0.0), "rte": by.get("rte", 0.0),
                     "capacity": by.get("capacity", 0.0), "total": ld["total"], "cap": ld["cap"],
                     "capped_total": ld["capped_total"], "currency": ld["currency"]})
    out = pd.DataFrame(rows)
    out.attrs["items"] = pd.concat(items, ignore_index=True) if items else pd.DataFrame()
    return out


def _window_starts(windows: pd.DataFrame, tz: str) -> pd.DatetimeIndex:
    """First local instant of each day or month window label (YYYY-MM-DD or YYYY-MM)."""
    labels = windows.index.astype(str)
    return pd.DatetimeIndex(pd.to_datetime(labels, format="ISO8601")).tz_localize(tz)
