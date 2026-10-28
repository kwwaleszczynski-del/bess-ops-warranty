"""The runs behind the CLI: your own data, the synthetic demo, the verification against ground truth, the NEM fleet.

Usage:
    run_user(Path("ems.parquet"), mapping, site, warranty, ltsa, events, "2026-08", Path("out"))
    run_demo(Path("out/demo"), "2026-08")
    verify(Path("results"))
    run_nem(Path("../nem-battery-behaviour/data"), Path("../bess-soc-soh-degradation/data/acen"), None,
            Path("results"))
"""
from __future__ import annotations

import json
import logging
import shutil
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from bess_ops.availability import read_events
from bess_ops.config import load_ltsa, load_site, load_warranty
from bess_ops.pipeline import analyse
from bess_ops.report import write_excel, write_html
from bess_ops.schema import load_mapping, read_table, to_site_frame

log = logging.getLogger(__name__)
EXAMPLES = ("mapping.example.yaml", "site.example.yaml", "warranty.example.yaml", "ltsa.example.yaml")


def example_path(name: str) -> Path:
    return Path(str(resources.files("bess_ops.examples").joinpath(name)))


def _json(obj):
    if isinstance(obj, dict):
        return {k: _json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return _json(obj.tolist())
    return obj


def default_month(res: dict) -> str:
    """The last month with at least 27 days of data."""
    k = res["monthly"]
    full = k.index[res["monthly_rte"]["hours"].reindex(k.index) >= 27 * 24]
    return full[-1] if len(full) else k.index[-1]


def write_outputs(res: dict, month: str | None, out: Path, prefix: str = "") -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    month = month or default_month(res)
    paths = {"excel": write_excel(res, month, out / f"{prefix}report_{month}.xlsx"),
             "html": write_html(res, month, out / f"{prefix}report_{month}.html")}
    res["monthly"].to_csv(out / "kpi_monthly.csv")
    res["daily"].to_csv(out / "rte_daily.csv")
    v = res["violations"].copy()
    tz = res["site_cfg"]["timezone"]
    for c in ("start", "end"):
        v[c] = pd.DatetimeIndex(v[c]).tz_convert(tz).strftime("%Y-%m-%d %H:%M") if len(v) else v[c]
    v.to_csv(out / "warranty_violations.csv", index=False)
    res["ld"].to_csv(out / "ld_by_ltsa_year.csv", index=False)
    if len(res["tests"]):
        res["tests"].to_csv(out / "capacity_tests.csv", index=False)
    if res.get("handoff"):
        (out / "augmentation_handoff.json").write_text(json.dumps(_json(res["handoff"]), indent=2),
                                                       encoding="utf-8")
    log.info("report for %s in %s", month, out)
    return paths


def run_user(input_path: Path, mapping: Path, site: Path, warranty: Path | None, ltsa: Path | None,
             events: Path | None, month: str | None, out: Path) -> dict:
    m = load_mapping(mapping)
    sc = load_site(site)
    w = load_warranty(warranty, sc)
    lt = load_ltsa(ltsa, sc)
    frame = to_site_frame(read_table(input_path), m)
    ev = read_events(events, sc["timezone"])
    log.info("%d intervals of %s, %s .. %s", len(frame), m.step, frame.index[0], frame.index[-1])
    res = analyse(frame, sc, w, lt, ev)
    write_outputs(res, month, out)
    return res


def demo_inputs(out: Path, seed: int = 1):
    """Simulate the synthetic battery and write it like a real export: long table, event log, YAML files."""
    from bess_ops.synthetic import simulate
    out = Path(out)
    (out / "config").mkdir(parents=True, exist_ok=True)
    sim = simulate(seed=seed)
    sim.long.to_parquet(out / "ems_long.parquet", index=False)
    sim.events.to_csv(out / "events.csv", index=False)
    for name in EXAMPLES:
        shutil.copy(example_path(name), out / "config" / name)
    return sim


def analyse_demo(out: Path, seed: int = 1):
    sim = demo_inputs(out, seed)
    cfg = Path(out) / "config"
    sc = load_site(cfg / "site.example.yaml")
    w = load_warranty(cfg / "warranty.example.yaml", sc)
    lt = load_ltsa(cfg / "ltsa.example.yaml", sc)
    frame = to_site_frame(pd.read_parquet(Path(out) / "ems_long.parquet"), load_mapping(cfg / "mapping.example.yaml"))
    ev = read_events(Path(out) / "events.csv", sc["timezone"])
    return sim, frame, analyse(frame, sc, w, lt, ev)


def run_demo(out: Path, month: str = "2026-08", seed: int = 1) -> dict:
    sim, frame, res = analyse_demo(out, seed)
    write_outputs(res, month, out)
    return res


# ---------------------------------------------------------------------------------------------------------------------
# Verification against the synthetic ground truth
# ---------------------------------------------------------------------------------------------------------------------

def true_availability(truth: dict, ltsa: dict, year: int) -> float:
    """Time-based availability of a calendar year computed from the outage list (independent of the estimator)."""
    tz = "Europe/Warsaw"
    a = ltsa["availability"]
    start, end = pd.Timestamp(f"{year}-01-01").tz_localize(tz), pd.Timestamp(f"{year + 1}-01-01").tz_localize(tz)
    hours = (end - start) / pd.Timedelta("1h")
    from bess_ops.synthetic import TESTS
    excluded = 24.0 * sum(1 for d, _ in TESTS if d.startswith(str(year)))
    unavailable, planned = 0.0, []
    for s, e, share, cat in truth["outages"]:
        s, e = pd.Timestamp(s).tz_localize(tz), pd.Timestamp(e).tz_localize(tz)
        if s.year != year:
            continue
        h = (e - s) / pd.Timedelta("1h")
        treat = a["categories"].get(cat, "counted") if cat else "counted"
        if treat == "excluded":
            excluded += h
        elif treat == "allowance":
            planned.append(h)
        elif treat == "counted" and share < a["threshold"]:
            unavailable += h
    allowance = a["planned_allowance_hours"]
    in_allow = min(sum(planned), allowance)
    unavailable += sum(planned) - in_allow
    return 1 - unavailable / (hours - excluded - in_allow)


def true_losses(frame: pd.DataFrame, plant, tz: str) -> pd.DataFrame:
    dt = 0.25
    ac, dc = frame["p_ac"], frame["p_dc"]
    pcs = np.where(ac > 0, dc - ac, -ac + dc) * dt                     # charge: |ac| - |dc|, discharge: dc - ac
    batt = np.where(dc > 0, dc / plant.eta_batt - dc, -dc * (1 - plant.eta_batt)) * dt
    d_stored = frame["stored_next"] - frame["stored"]
    out = pd.DataFrame({"losses_pcs": pcs, "losses_battery": batt, "e_aux": frame["p_aux"] * dt,
                        "e_in": (-frame["p_meter"]).clip(lower=0) * dt, "e_out": frame["p_meter"].clip(lower=0) * dt,
                        "d_stored": d_stored}, index=frame.index)
    return out.groupby(out.index.tz_convert(tz).strftime("%Y-%m")).sum()


def verify(results: Path, seed: int = 1) -> dict:
    from bess_ops.synthetic import Plant
    import tempfile
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        sim, frame, res = analyse_demo(Path(tmp), seed)
    truth, plant = sim.truth, Plant()
    tz = res["site_cfg"]["timezone"]
    d = res["daily"]
    vd = d[d["valid"]]
    out: dict = {"rte_excl_aux": {
        "truth": truth["rte_excl_aux"], "yearly": res["yearly_rte"]["rte_excl_aux"].tolist(),
        "monthly_max_abs_error": float((res["monthly_rte"]["rte_excl_aux"] - truth["rte_excl_aux"]).abs().max()),
        "daily_median": float(vd["rte_excl_aux"].median()),
        "daily_p05_p95": [float(vd["rte_excl_aux"].quantile(0.05)), float(vd["rte_excl_aux"].quantile(0.95))],
        "daily_raw_p05_p95_excl_aux": [float((vd["e_out_pcs"] / vd["e_in_pcs"]).quantile(0.05)),
                                       float((vd["e_out_pcs"] / vd["e_in_pcs"]).quantile(0.95))],
        "eta_auto": float(d["eta"].iloc[0]), "eta_truth": truth["one_way"], "valid_days": int(len(vd)),
    }}
    out["rte_incl_aux"] = {"yearly": res["yearly_rte"]["rte"].tolist(),
                           "monthly_min": float(res["monthly_rte"]["rte"].min()),
                           "monthly_max": float(res["monthly_rte"]["rte"].max())}
    p = res["aux"]["params"]
    out["aux"] = {"estimate": {k: float(p[k]) for k in truth["aux"]}, "truth": truth["aux"], "r2": float(p["r2"])}
    ay = res["availability_year"]
    out["availability"] = {str(y): {"estimate": float(ay.loc[str(i + 1), "availability"]),
                                    "truth": true_availability(truth, res["ltsa"], y)}
                           for i, y in enumerate((2025, 2026))}
    tests = res["tests"]
    out["capacity_tests"] = [{"date": t["date"], "truth_25c_mwh": t["e_meter_25c_mwh"],
                              "estimate_mwh": float(tests.iloc[i]["e_corr_mwh"]),
                              "U_mwh": float(tests.iloc[i]["U_mwh"]), "verdict": tests.iloc[i]["verdict"]}
                             for i, t in enumerate(truth["tests"])]
    v = res["violations"]
    out["violations"] = v.groupby("rule").size().to_dict()
    out["violations_injected"] = {"efc_per_day": "14 days of the high-cycling fortnight",
                                  "soc_high": "EMS misconfiguration 1-4 May 2026 (+ 4 capacity tests)",
                                  "c_rate": "PCS overload test 11 March 2026",
                                  "cell_temp_high": "HVAC failure 21-22 July 2026",
                                  "efc_per_year": "second daily cycle from September 2026"}
    bal = res["balance"]
    tl = true_losses(sim.frame, plant, tz)
    both = bal.join(tl.add_suffix("_true"))
    out["energy_balance"] = {
        "truth_closure_max_mwh": float((tl["e_in"] - tl["e_out"] - tl["e_aux"] - tl["losses_pcs"]
                                        - tl["losses_battery"] - tl["d_stored"]).abs().max()),
        "pcs_losses_max_rel_error": float(((both["losses_pcs"] - both["losses_pcs_true"]).abs()
                                           / both["losses_pcs_true"]).max()),
        "battery_losses_max_rel_error": float(((both["losses_battery"] - both["losses_battery_true"]).abs()
                                               / both["losses_battery_true"]).max()),
        "battery_losses_total_rel_error": float((both["losses_battery"].sum() - both["losses_battery_true"].sum())
                                                / both["losses_battery_true"].sum()),
    }
    both[["e_in", "e_out", "d_stored", "d_stored_true", "e_aux", "losses_pcs", "losses_pcs_true",
          "losses_battery", "losses_battery_true"]].round(3).to_csv(results / "synthetic_energy_balance.csv")
    fc = res["forecast"]
    curve = fc["curve"]
    efc_rate = fc["efc_per_year"]
    yrs = curve["years"].to_numpy()
    last = truth["tests"][-1]
    n_true = np.where(yrs <= last["years"], last["efc_total"] * yrs / last["years"],
                      last["efc_total"] + efc_rate * (yrs - last["years"]))
    e_use = plant.e0_stored_mwh * (1 - plant.fade_a * np.sqrt(yrs) - plant.fade_b * n_true / 1000)
    aux_test = last["e_use_mwh"] * plant.eta_pcs * plant.eta_batt - last["e_meter_mwh"]
    soh_true = (e_use * plant.eta_pcs * plant.eta_batt - aux_test) / plant.energy_mwh
    req = res["ltsa"]["capacity"]["forecast"]["required_mwh"] / plant.energy_mwh
    cross = yrs[soh_true < req]
    out["forecast"] = {"requirement_soh": req, "true_crossing_years": float(cross[0]) if len(cross) else None,
                       "forecast_crossing_years": fc["requirement_years"],
                       "soh_5y_truth": float(np.interp(5, yrs, soh_true)),
                       "soh_5y_forecast": [float(np.interp(5, yrs, curve[c])) for c in ("soh_p10", "soh_p50",
                                                                                        "soh_p90")],
                       "fit": {"s0": float(fc["fit"]["s0"]), "a": float(fc["fit"]["a"]), "b": float(fc["fit"]["b"])},
                       "truth_scaled": {"a": plant.fade_a * plant.e0_stored_mwh * plant.eta_pcs * plant.eta_batt
                                        / plant.energy_mwh,
                                        "b": plant.fade_b * plant.e0_stored_mwh * plant.eta_pcs * plant.eta_batt
                                        / plant.energy_mwh}}
    curve.assign(soh_true=soh_true).to_csv(results / "synthetic_soh_forecast.csv", index=False)
    ld = res["ld"]
    out["ld"] = ld.drop(columns="currency").to_dict(orient="records")
    (results / "verification.json").write_text(json.dumps(_json(out), indent=2), encoding="utf-8")
    res["monthly"].round(4).to_csv(results / "synthetic_kpi_monthly.csv")
    log.info("verification written to %s", results / "verification.json")
    return out


# ---------------------------------------------------------------------------------------------------------------------
# NEM fleet and ACEN validation
# ---------------------------------------------------------------------------------------------------------------------

def run_nem(nem_dir: Path, acen_dir: Path | None, regression_csv: Path | None, results: Path,
            extra_dir: Path | None = None) -> dict:
    from bess_ops.nem import fleet_benchmark, load_cache, load_extra
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    cache = load_cache(nem_dir)
    if extra_dir is not None and Path(extra_dir).exists():
        cache.extra = load_extra(extra_dir)
        log.info("charge-side and physical availability from %s", extra_dir)
    else:
        log.info("no fetch-nem directory: offered discharge availability only")
    fleet = fleet_benchmark(cache)
    summ = fleet["summary"]
    if regression_csv is not None and Path(regression_csv).exists():
        reg = pd.read_csv(regression_csv).set_index("duid")
        summ = summ.join(reg[["rte", "standby_pct_of_mw", "scale_trend_pct_per_yr"]].rename(columns={
            "rte": "rte_regression_companion", "standby_pct_of_mw": "standby_pct_companion",
            "scale_trend_pct_per_yr": "scale_trend_companion"}))
    summ.round(5).to_csv(results / "nem_fleet_summary.csv")
    fleet["monthly"].round(5).to_csv(results / "nem_fleet_monthly.csv")
    stats = pd.DataFrame({d.name: d.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]) for d in fleet["daily_rte"]
                          if len(d)}).T
    stats.round(5).to_csv(results / "nem_daily_rte_distribution.csv")
    out = {"fleet": fleet}
    if acen_dir is not None and Path(acen_dir).exists():
        from bess_ops.acen import validate
        val = validate(cache, acen_dir)
        val["summary"].round(5).to_csv(results / "acen_validation_summary.csv")
        val["monthly"].round(5).to_csv(results / "acen_validation_monthly.csv", index=False)
        val["daily"].round(5).to_csv(results / "acen_validation_daily.csv")
        out["acen"] = val
    else:
        log.info("no ACEN directory: validation skipped")
    log.info("NEM results in %s", results)
    return out
