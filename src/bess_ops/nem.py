"""Fleet benchmark on AEMO NEM open data: RTE, offered availability, cycles and usable energy of every battery.

The NEM publishes, for every grid-scale battery (a bidirectional unit since the 2024 integrated-storage rules),
5-minute data that map onto this project's canonical signals:

=======================================  =================  ===============================================
AEMO table.column                        canonical signal   note
=======================================  =================  ===============================================
DISPATCH_UNIT_SCADA.SCADAVALUE           p_meter            instantaneous MW at the interval start, + export
DISPATCHLOAD.INITIAL_ENERGY_STORAGE      e_stored           stored energy (MWh) the participant reports
DISPATCHLOAD.AVAILABILITY                p_avail_dis        discharge availability offered (bid MAXAVAIL)
DISPATCHLOAD.MIN_AVAILABILITY            p_avail_ch         charge availability offered (fetch-nem)
BIDPEROFFER_D.PASAAVAILABILITY           technical          physical availability per direction (fetch-nem)
DUDETAIL MAXCAPACITY, MINCAPACITY,       contract values    registered discharge MW, charge MW and MWh,
MAXSTORAGECAPACITY                                          versioned by date
=======================================  =================  ===============================================

The cache is the one built by the companion project ``nem-battery-behaviour`` (monthly Parquet files of these
tables, AEMO MMSDM archive); its path is passed with ``--nem-data-dir`` or ``BESS_NEM_DATA_DIR``. All timestamps are
NEM market time (AEST, UTC+10, no DST), labelled by interval END; here intervals are indexed by their start.

What market data do NOT give, and the benchmark therefore cannot: the auxiliary consumption behind the connection
point (SCADA is the unit's net output; HVAC supplied from elsewhere is invisible), capacity tests, temperatures and
the warranty terms. ``AVAILABILITY`` is what the participant offers, so it mixes outages with energy limits (a
near-empty battery offers less), commissioning and capacity held back under other contracts (system-integrity
schemes): an "offered availability". ``PASAAVAILABILITY``, the physical availability the participant declares with
its bids (``fetch-nem``, optional), is the closest public proxy of the technical availability an LTSA measures;
it is still a declaration, not a measurement.

Usage:
    cache = load_cache(Path("../nem-battery-behaviour/data"))
    site = unit_site(cache, "TARBESS1")
    monthly, summary, daily = unit_benchmark(cache, "TARBESS1")
    res = fleet_benchmark(cache)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from bess_ops.cycles import rainflow
from bess_ops.energy import interval_flows, state_jumps, window_keys, window_rte
from bess_ops.schema import site_frame

log = logging.getLogger(__name__)
NEM_TZ = "Australia/Brisbane"          # AEST, the NEM's market time zone (no daylight saving)
STEP = pd.Timedelta("5min")
DT_H = 5 / 60


@dataclass
class Cache:
    registry: pd.DataFrame
    stations: pd.DataFrame | None
    scada: pd.DataFrame
    dispatch: pd.DataFrame
    extra: dict | None = None      # fetch-nem tables: {"dispatchload": ..., "bids": ...}
    split: dict = field(default_factory=dict, repr=False)


def load_cache(data_dir: Path, months: list[str] | None = None) -> Cache:
    data_dir = Path(data_dir)
    if not (data_dir / "registry.parquet").exists():
        raise FileNotFoundError(f"no NEM cache in {data_dir} (registry.parquet missing); build it with the "
                                "companion project nem-battery-behaviour: python -m nem_battery fetch")

    def read(kind, columns):
        files = sorted((data_dir / kind).glob("*.parquet"))
        if months:
            files = [f for f in files if f.stem in months]
        if not files:
            raise FileNotFoundError(f"no {kind} files in {data_dir / kind}")
        return pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)

    scada = read("scada", ["SETTLEMENTDATE", "DUID", "SCADAVALUE"])
    dispatch = read("dispatch", ["SETTLEMENTDATE", "DUID", "INTERVENTION", "AVAILABILITY",
                                 "INITIAL_ENERGY_STORAGE"])
    dispatch = dispatch.sort_values("INTERVENTION").drop_duplicates(["SETTLEMENTDATE", "DUID"], keep="first")
    for df in (scada, dispatch):
        df["DUID"] = df["DUID"].astype(str)
    stations = pd.read_parquet(data_dir / "stations.parquet") if (data_dir / "stations.parquet").exists() else None
    return Cache(pd.read_parquet(data_dir / "registry.parquet"), stations, scada, dispatch)


def load_extra(extra_dir: Path, months: list[str] | None = None) -> dict:
    """Charge-side and physical availability written by ``fetch-nem`` (see ``bess_ops.mms``)."""
    out = {}
    for kind in ("dispatchload", "bids"):
        files = sorted((Path(extra_dir) / kind).glob("*.parquet"))
        if months:
            files = [f for f in files if f.stem in months]
        if not files:
            raise FileNotFoundError(f"no {kind} files in {extra_dir}; run: python -m bess_ops fetch-nem")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        for c in ("DUID", "DIRECTION"):
            if c in df:
                df[c] = df[c].astype(str)
        out[kind] = df
    return out


def extra_rows(cache: Cache, duid: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "extra_dispatchload" not in cache.split:
        for kind in ("dispatchload", "bids"):
            cache.split[f"extra_{kind}"] = {k: g for k, g in cache.extra[kind].groupby("DUID", sort=False)}
    dl = cache.split["extra_dispatchload"].get(duid, cache.extra["dispatchload"].iloc[:0])
    bids = cache.split["extra_bids"].get(duid, cache.extra["bids"].iloc[:0])
    return dl, bids


def technical_frame(cache: Cache, duid: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Offered charge availability and declared physical availability per direction on the unit's UTC grid."""
    dl, bids = extra_rows(cache, duid)

    def to_grid(times: pd.Series, values: pd.Series) -> np.ndarray:
        start = (pd.DatetimeIndex(times) - STEP).tz_localize(NEM_TZ).tz_convert("UTC")
        s = pd.Series(values.to_numpy(float), index=start)
        return s[~s.index.duplicated()].reindex(index).to_numpy()

    out = pd.DataFrame(index=index)
    out["p_avail_ch"] = to_grid(dl["SETTLEMENTDATE"], dl["MIN_AVAILABILITY"])
    for d, name in (("GEN", "dis"), ("LOAD", "ch")):
        b = bids[bids["DIRECTION"] == d]
        out[f"pasa_{name}"] = to_grid(b["INTERVAL_DATETIME"], b["PASAAVAILABILITY"])
        out[f"maxavail_{name}"] = to_grid(b["INTERVAL_DATETIME"], b["MAXAVAIL"])
    return out


def unit_rows(cache: Cache, duid: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SCADA and dispatch rows of one unit; the tables are split by unit once per cache."""
    if not cache.split:
        for name in ("scada", "dispatch"):
            cache.split[name] = {k: g for k, g in getattr(cache, name).groupby("DUID", sort=False)}
    return (cache.split["scada"].get(duid, cache.scada.iloc[:0]),
            cache.split["dispatch"].get(duid, cache.dispatch.iloc[:0]))


def battery_table(cache: Cache) -> pd.DataFrame:
    """Latest registration of every battery: name, region, MW, MWh, registered efficiencies."""
    reg = cache.registry.sort_values(["DUID", "EFFECTIVEDATE", "VERSIONNO"])
    last = reg.groupby("DUID").tail(1).set_index("DUID")
    out = pd.DataFrame({"region": last["REGIONID"].str.replace("1", "", regex=False),
                        "mw": last["MAXCAPACITY"], "mwh": last["MAXSTORAGECAPACITY"],
                        "eta_import": last["STORAGEIMPORTEFFICIENCYFACTOR"],
                        "eta_export": last["STORAGEEXPORTEFFICIENCYFACTOR"]})
    out["name"] = out.index
    if cache.stations is not None:
        names = cache.stations.set_index("STATIONID")["STATIONNAME"].str.strip()
        out["name"] = last["STATIONID"].map(names).fillna(pd.Series(out.index, index=out.index))
    return out


def registration_daily(cache: Cache, duid: str, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Registered MW and MWh valid on each NEM-time day (the latest version effective on or before it)."""
    reg = cache.registry[cache.registry["DUID"] == duid].sort_values(["EFFECTIVEDATE", "VERSIONNO"])
    cols = ["MAXCAPACITY", "MAXSTORAGECAPACITY"] + (["MINCAPACITY"] if "MINCAPACITY" in reg else [])
    reg = reg.groupby("EFFECTIVEDATE").tail(1).set_index("EFFECTIVEDATE")[cols]
    out = reg.reindex(reg.index.union(days)).sort_index().ffill().reindex(days)
    out = out.rename(columns={"MAXCAPACITY": "mw", "MAXSTORAGECAPACITY": "mwh", "MINCAPACITY": "mw_ch"}).bfill()
    if "mw_ch" not in out:
        out["mw_ch"] = out["mw"]
    out["mw_ch"] = out["mw_ch"].where(out["mw_ch"] > 0, out["mw"])      # registered charge capacity
    return out


def exact_split(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Discharge and charge energy (MWh) of each interval between instantaneous readings x[k] and x[k+1], with the
    straight line between them split exactly where it crosses zero."""
    a, b = x[:-1], x[1:]
    pos = np.where((a >= 0) & (b >= 0), (a + b) / 2, 0.0)
    neg = np.where((a <= 0) & (b <= 0), -(a + b) / 2, 0.0)
    cross = (a * b) < 0
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(cross, a / (a - b), 0.0)            # share of the interval before the zero crossing
    top = np.where(a > 0, a, b)
    bottom = np.where(a > 0, b, a)
    share_pos = np.where(a > 0, f, 1 - f)
    pos = np.where(cross, top * share_pos / 2, pos)
    neg = np.where(cross, -bottom * (1 - share_pos) / 2, neg)
    return np.r_[pos, np.nan] * DT_H, np.r_[neg, np.nan] * DT_H


def unit_site(cache: Cache, duid: str) -> pd.DataFrame:
    """Site frame of one battery on its 5-minute grid (UTC, interval start)."""
    sc, di = unit_rows(cache, duid)
    start = (min(sc["SETTLEMENTDATE"].min(), di["SETTLEMENTDATE"].min()) - STEP)
    end = (max(sc["SETTLEMENTDATE"].max(), di["SETTLEMENTDATE"].max()) - STEP)
    grid = pd.date_range(start, end, freq=STEP)
    reading = sc.set_index(sc["SETTLEMENTDATE"] - STEP)["SCADAVALUE"].astype(float)
    reading = reading[~reading.index.duplicated()].reindex(grid)
    d = di.set_index(di["SETTLEMENTDATE"] - STEP)
    d = d[~d.index.duplicated()].reindex(grid)
    x = reading.to_numpy()
    e_out, e_in = exact_split(x)
    p_mean = (x + np.r_[x[1:], np.nan]) / 2                # trapezoid: interval mean of the line
    index = grid.tz_localize(NEM_TZ)
    site = site_frame({"p_meter": p_mean, "e_stored": d["INITIAL_ENERGY_STORAGE"].astype(float).to_numpy(),
                       "p_avail_dis": d["AVAILABILITY"].astype(float).to_numpy()}, index, NEM_TZ)
    site["e_out_exact"] = e_out
    site["e_in_exact"] = e_in
    return site


def unit_benchmark(cache: Cache, duid: str, min_month_efc: float = 1.0) -> tuple[pd.DataFrame, dict, pd.Series]:
    """Monthly metrics, a summary over the operational months and the valid daily RTE of one battery."""
    site = unit_site(cache, duid)
    local = site.index.tz_convert(NEM_TZ)
    days = local.normalize().tz_localize(None)
    reg = registration_daily(cache, duid, pd.DatetimeIndex(days.unique()))
    mw = pd.Series(reg["mw"].reindex(days).to_numpy(), index=site.index)
    mwh = pd.Series(reg["mwh"].reindex(days).to_numpy(), index=site.index)
    cfg = {"aux_metering": "none", "soc_energy_mwh": float(np.nanmedian(mwh)), "timezone": NEM_TZ, "cod": None}
    flows = interval_flows(site, cfg)
    flows["e_out"] = site["e_out_exact"]
    flows["e_in"] = site["e_in_exact"]
    months = window_keys(site.index, NEM_TZ, "M")
    mk = months.to_numpy()

    # consistency of the reported stored energy with the metered energy, per month: hourly regression, BMS
    # recalibrations (jumps) removed, inverted or incoherent months left out of every state-based metric
    adjusted = pd.Series(np.nan, index=site.index)
    checks = {}
    for m in np.unique(mk):
        sel = mk == m
        adj, jumps, fit = state_jumps(site["e_stored"][sel], flows["e_in"][sel], flows["e_out"][sel],
                                      float(np.nanmedian(mwh[sel])))
        adjusted[sel] = adj.to_numpy()
        checks[m] = fit
    checks = pd.DataFrame(checks).T
    consistent = (checks["a"] > 0.5) & (checks["b"] > 0.5) & (checks["r2"] >= 0.95)
    stored_ok = pd.Series(consistent.reindex(mk).to_numpy(), index=site.index).fillna(False).astype(bool)
    flows_c = flows.copy()
    flows_c["stored"] = adjusted.where(stored_ok)
    soc = adjusted / mwh

    mrte = window_rte(site, cfg, "M", "auto", flows_c, min_efc=min_month_efc)
    eta = float(mrte["eta"].iloc[0])
    drte = window_rte(site, cfg, "D", eta, flows_c)

    g = pd.DataFrame({"m": mk, "mw": mw.to_numpy(), "mwh": mwh.to_numpy(), "avail": site["p_avail_dis"].to_numpy(),
                      "soc": (site["e_stored"] / mwh).to_numpy(), "e": site["e_stored"].to_numpy()})
    g["ratio"] = (g["avail"] / g["mw"]).clip(upper=1.0)
    g["full"] = g["ratio"] >= 0.95 - 1e-9
    g["half"] = g["ratio"] >= 0.5
    g["soc_ok"] = g["soc"] > 0.10
    g["zero_soc_ok"] = (g["avail"] <= 0.01 * g["mw"]) & g["soc_ok"]
    tech = cache.extra is not None
    if tech:
        t = technical_frame(cache, duid, site.index)
        mw_ch = reg["mw_ch"].reindex(days).to_numpy()
        g["ratio_ch"] = (t["p_avail_ch"].to_numpy() / mw_ch).clip(max=1.0)
        g["pasa_dis"] = (t["pasa_dis"].to_numpy() / g["mw"]).clip(upper=1.0)
        g["pasa_ch"] = (t["pasa_ch"].to_numpy() / mw_ch).clip(max=1.0)
        g["pasa_both"] = g[["pasa_dis", "pasa_ch"]].min(axis=1, skipna=False)
        # an empty battery cannot discharge and a full one cannot charge: an LTSA does not count that as an
        # outage, so near the SoC limits only the other direction counts
        soc_rep = g["soc"].to_numpy()
        g["pasa_adj"] = np.where(soc_rep <= 0.10, g["pasa_ch"],
                                 np.where(soc_rep >= 0.90, g["pasa_dis"], g["pasa_both"]))
        g.loc[np.isnan(soc_rep), "pasa_adj"] = g["pasa_both"]
        g["pasa_full"] = (g["pasa_adj"] >= 0.95 - 1e-9).where(g["pasa_adj"].notna())
        g["pasa_out"] = (g["pasa_adj"] <= 0.05).where(g["pasa_adj"].notna())
        g["withheld"] = (g["pasa_dis"] - g["ratio"]).clip(lower=0)
    by = g.groupby("m")
    monthly = pd.DataFrame({
        "mw": by["mw"].median(), "mwh": by["mwh"].median(),
        "e_out_mwh": mrte["e_out"], "e_in_mwh": mrte["e_in"], "d_stored_mwh": mrte["d_stored"],
        "rte_raw": mrte["rte_raw"], "rte": mrte["rte"].where(consistent.reindex(mrte.index).fillna(False)),
        "stored_r2": checks["r2"], "stored_jumps": checks["jumps"], "stored_consistent": consistent,
        "availability_capacity": by["ratio"].mean(), "availability_time": by["full"].mean(),
        "offered_half_share": by["half"].mean(),
        "availability_capacity_soc_ok": g[g["soc_ok"]].groupby("m")["ratio"].mean(),
        "zero_offer_hours_soc_ok": by["zero_soc_ok"].sum() * DT_H,
        "e_p995_mwh": by["e"].quantile(0.995), "e_p005_mwh": by["e"].quantile(0.005),
        "soc_above_95": (g["soc"] >= 0.95).groupby(g["m"]).mean(),
        "soc_below_5": (g["soc"] <= 0.05).groupby(g["m"]).mean(),
        "mean_soc": by["soc"].mean(),
    })
    if tech:
        monthly["availability_charge_offered"] = by["ratio_ch"].mean()
        monthly["availability_technical_dis"] = by["pasa_dis"].mean()
        monthly["availability_technical_ch"] = by["pasa_ch"].mean()
        monthly["availability_technical_both"] = by["pasa_both"].mean()
        monthly["availability_technical"] = by["pasa_adj"].mean()
        monthly["availability_technical_time"] = by["pasa_full"].mean()
        monthly["technical_outage_hours"] = by["pasa_out"].sum() * DT_H
        monthly["withheld_share"] = by["withheld"].mean()
        monthly["technical_coverage"] = by["pasa_both"].count() / by.size()
    monthly["efc"] = monthly["e_out_mwh"] / monthly["mwh"]
    monthly["e_window_share"] = (monthly["e_p995_mwh"] - monthly["e_p005_mwh"]) / monthly["mwh"]
    rf = []
    for m in monthly.index:
        sel = (mk == m) & stored_ok.to_numpy()
        if sel.sum() < 288 * 5:
            rf.append((np.nan, np.nan))
            continue
        c = rainflow(soc.to_numpy()[sel], 0.01)
        efc = float((c["depth"] * c["count"]).sum())
        deep = float((c["depth"] * c["count"])[c["depth"] >= 0.5].sum())
        rf.append((efc, deep / efc if efc > 0 else np.nan))
    monthly["rainflow_efc"], monthly["deep_cycle_share"] = zip(*rf)
    hours = by.size() * DT_H
    monthly["days"] = hours / 24
    monthly["operational"] = (monthly["efc"] >= min_month_efc) & (monthly["days"] >= 20)
    monthly.insert(0, "duid", duid)
    op = monthly[monthly["operational"]]
    plausible = drte["rte"].between(0.3, 1.0)        # a physical round trip loses energy and is not absurdly lossy
    vd = drte[drte["valid"] & plausible
              & pd.Series(drte.index.str[:7]).isin(op.index[op["stored_consistent"]]).to_numpy()]
    summary = {"duid": duid, "months": len(monthly), "operational_months": len(op)}
    if len(op):
        days_op = op["days"].sum()
        summary.update({
            "mw": float(op["mw"].iloc[-1]), "mwh": float(op["mwh"].iloc[-1]),
            "first_month": op.index[0], "last_month": op.index[-1],
            "efc_per_year": float(op["efc"].sum() / days_op * 365.25),
            "rainflow_efc_per_year": float(op["rainflow_efc"].sum() / op.loc[op["rainflow_efc"].notna(), "days"].sum()
                                           * 365.25) if op["rainflow_efc"].notna().any() else np.nan,
            "deep_cycle_share": float((op["deep_cycle_share"] * op["rainflow_efc"]).sum() / op["rainflow_efc"].sum())
            if op["rainflow_efc"].sum() > 0 else np.nan,
            "rte_raw": float(op["e_out_mwh"].sum() / op["e_in_mwh"].sum()),
            "rte_corrected": _corrected_total(op, eta),
            "rte_daily_median": float(vd["rte"].median()) if len(vd) else np.nan,
            "rte_daily_p25": float(vd["rte"].quantile(0.25)) if len(vd) else np.nan,
            "rte_daily_p75": float(vd["rte"].quantile(0.75)) if len(vd) else np.nan,
            "valid_days": int(len(vd)), "consistent_months": int(op["stored_consistent"].sum()),
            "jumps_per_month": float(op["stored_jumps"].mean()),
            "availability_capacity": _weighted(op, "availability_capacity"),
            "availability_time": _weighted(op, "availability_time"),
            "offered_half_share": _weighted(op, "offered_half_share"),
            "availability_capacity_soc_ok": _weighted(op, "availability_capacity_soc_ok"),
            "zero_offer_hours_per_year": float(op["zero_offer_hours_soc_ok"].sum() / days_op * 365.25),
            "soc_above_95": _weighted(op, "soc_above_95"), "soc_below_5": _weighted(op, "soc_below_5"),
            "mean_soc": _weighted(op, "mean_soc"),
            "e_max_share": float(op["e_p995_mwh"].max() / op["mwh"].iloc[-1]),
        })
        if tech:
            # in service: from the first operational month with at least 80 % declared (commissioning before)
            served = op["availability_technical"] >= 0.8
            ts = op.loc[served.idxmax():] if served.any() else op
            summary.update({c: _weighted(ts, c) for c in (
                "availability_charge_offered", "availability_technical_dis", "availability_technical_ch",
                "availability_technical_both", "availability_technical", "availability_technical_time",
                "withheld_share")})
            summary["technical_outage_hours_per_year"] = float(ts["technical_outage_hours"].sum()
                                                               / ts["days"].sum() * 365.25)
            summary["in_service_from"] = ts.index[0] if served.any() else None
            summary["in_service_months"] = int(len(ts)) if served.any() else 0
            summary["availability_offered_in_service"] = _weighted(ts, "availability_capacity")
            summary["availability_offered_soc_ok_in_service"] = _weighted(ts, "availability_capacity_soc_ok")
        summary.update(window_trend(op))
    return monthly, summary, vd["rte"].rename(duid)


def _weighted(op: pd.DataFrame, col: str) -> float:
    ok = op[col].notna()
    return float((op.loc[ok, col] * op.loc[ok, "days"]).sum() / op.loc[ok, "days"].sum()) if ok.any() else np.nan


def _corrected_total(op: pd.DataFrame, eta: float) -> float:
    from bess_ops.energy import corrected_rte
    ok = op["stored_consistent"]
    if not ok.any():
        return np.nan
    return float(corrected_rte(op.loc[ok, "e_out_mwh"].sum(), op.loc[ok, "e_in_mwh"].sum(),
                               op.loc[ok, "d_stored_mwh"].sum(), eta))


def window_trend(op: pd.DataFrame, min_months: int = 6) -> dict:
    """Linear trend of the monthly upper envelope of reported energy (P99.5, share of the latest registered MWh),
    in percent of the registered energy per year, over months with consistent data and the same registration,
    from the first month the envelope reaches 95 % of its maximum (after commissioning)."""
    sel = op[op["stored_consistent"] & (op["mwh"] == op["mwh"].iloc[-1])]
    if len(sel):
        full = sel["e_p995_mwh"] >= 0.95 * sel["e_p995_mwh"].max()    # from the first month at the full range on
        sel = sel.loc[full.idxmax():]
    if len(sel) < min_months:
        return {"e_max_trend_pct_per_yr": np.nan, "trend_months": len(sel)}
    t = (pd.PeriodIndex(sel.index, freq="M").asi8 - pd.PeriodIndex(sel.index, freq="M").asi8[0]) / 12.0
    y = sel["e_p995_mwh"].to_numpy() / sel["mwh"].iloc[-1] * 100
    slope = np.polyfit(t, y, 1)[0]
    return {"e_max_trend_pct_per_yr": float(slope), "trend_months": len(sel)}


def fleet_benchmark(cache: Cache, min_mw: float = 5.0) -> dict:
    """Every battery of at least ``min_mw``: monthly table, one summary row each, the valid daily RTE values."""
    table = battery_table(cache)
    present = set(cache.scada["DUID"].unique()) & set(cache.dispatch["DUID"].unique())
    duids = [d for d in table.index if table.loc[d, "mw"] >= min_mw and d in present]
    monthly, summary, daily = [], [], []
    for i, duid in enumerate(sorted(duids, key=lambda d: -table.loc[d, "mw"])):
        try:
            m, s, d = unit_benchmark(cache, duid)
        except Exception as e:      # a unit with too little data should not stop the fleet run
            log.warning("%s skipped: %s", duid, e)
            continue
        monthly.append(m)
        summary.append(s)
        daily.append(d)
        log.info("%2d/%d %s: %d operational months", i + 1, len(duids), duid, s["operational_months"])
    summ = pd.DataFrame(summary).set_index("duid").join(table[["name", "region", "eta_import", "eta_export"]])
    summ["rte_registered"] = summ["eta_import"] * summ["eta_export"]
    return {"summary": summ, "monthly": pd.concat(monthly), "daily_rte": daily}
