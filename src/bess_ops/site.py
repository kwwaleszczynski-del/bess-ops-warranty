"""README charts, the dashboard data (docs/data.js) and the example monthly report in docs/report/.

Reads ``results/`` (written by ``nem`` and ``verify``) and re-runs the synthetic demo for the example report.

Usage:
    python -m bess_ops site --results results --docs docs
"""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bess_ops.study import _json, analyse_demo, write_outputs  # noqa: E402

log = logging.getLogger(__name__)
REGION_COLORS = {"NSW": "#2a78d6", "VIC": "#1baf7a", "QLD": "#e3a33b", "SA": "#c8423b", "TAS": "#8e7cc3"}
EXAMPLE_MONTH = "2026-08"


def _records(df: pd.DataFrame) -> list[dict]:
    return _json(df.replace({np.nan: None}).to_dict(orient="records"))


def fleet_data(results: Path) -> dict:
    summ = pd.read_csv(results / "nem_fleet_summary.csv", index_col=0)
    summ = summ[summ["operational_months"] >= 3].sort_values("mw", ascending=False)
    monthly = pd.read_csv(results / "nem_fleet_monthly.csv", index_col=0)
    by_unit = {}
    for duid, g in monthly.groupby("duid"):
        if duid not in summ.index:
            continue
        g = g[g["operational"]]
        by_unit[duid] = {"month": list(g.index), "e_max_share": _json(list(g["e_p995_mwh"] / g["mwh"])),
                         "rte": _json(list(g["rte"])), "rte_raw": _json(list(g["rte_raw"])),
                         "availability": _json(list(g["availability_capacity"])),
                         "availability_soc_ok": _json(list(g["availability_capacity_soc_ok"])),
                         "technical": _json(list(g["availability_technical"])) if "availability_technical" in g
                         else None,
                         "efc": _json(list(g["efc"])), "consistent": [bool(x) for x in g["stored_consistent"]]}
    return {"summary": _records(summ.reset_index()), "monthly": by_unit}


def acen_data(results: Path) -> dict | None:
    if not (results / "acen_validation_summary.csv").exists():
        return None
    s = pd.read_csv(results / "acen_validation_summary.csv", index_col=0)
    d = pd.read_csv(results / "acen_validation_daily.csv", index_col=0)
    m = pd.read_csv(results / "acen_validation_monthly.csv")
    daily = {u: {"day": list(g.index), "A": _json(list(g["rte_operator"])),
                 "B": _json(list(g["rte_scada_operator_soc"])), "C": _json(list(g["rte_market"]))}
             for u, g in d.groupby("duid")}
    return {"summary": _records(s.reset_index()), "daily": daily, "monthly": _records(m)}


def synthetic_data(results: Path, res: dict) -> dict:
    ver = json.loads((results / "verification.json").read_text(encoding="utf-8"))
    fc = pd.read_csv(results / "synthetic_soh_forecast.csv")
    d = res["daily"]
    k = res["monthly"]
    t = res["tests"]
    return {"verification": ver,
            "forecast": {c: _json(list(fc[c])) for c in ("years", "soh_p10", "soh_p50", "soh_p90", "soh_guaranteed",
                                                            "soh_true")},
            "tests": {"years": _json(list(t["elapsed_years"])), "soh": _json(list(t["soh"])),
                      "u": _json(list(t["U_mwh"] / res["warranty"]["reference_energy_mwh"]))},
            "daily": {"day": list(d.index), "valid": [bool(x) for x in d["valid"]],
                      "corrected": _json(list(d["rte_excl_aux"])),
                      "raw": _json(list(d["e_out_pcs"] / d["e_in_pcs"].where(d["e_in_pcs"] > 0))),
                      "incl_aux": _json(list(d["rte"]))},
            "monthly": {"month": list(k.index), "rte": _json(list(k["rte"])),
                        "rte_excl_aux": _json(list(k["rte_excl_aux"])), "availability": _json(list(k["availability"])),
                        "efc": _json(list(k["efc"]))},
            "aux_bins": _records(res["aux"]["bins"].reset_index().rename(columns={"index": "bin"})),
            "required_soh": res["ltsa"]["capacity"]["forecast"]["required_mwh"]
            / res["warranty"]["reference_energy_mwh"],
            "rte_guaranteed": res["ltsa"]["rte"]["guaranteed"],
            "availability_guaranteed": res["ltsa"]["availability"]["guaranteed"]}


# ---------------------------------------------------------------------------------------------------------------------
# README charts
# ---------------------------------------------------------------------------------------------------------------------

def _save(fig, path: Path):
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("chart %s", path)


def _labels(s: pd.DataFrame) -> list[str]:
    return [f"{n[:24]} {d} ({r}, {mw:.0f} MW / {mwh:.0f} MWh)"
            for n, d, r, mw, mwh in zip(s["name"], s.index, s["region"], s["mw"], s["mwh"])]


def chart_fleet_rte(summary: pd.DataFrame, path: Path):
    s = summary[summary["valid_days"] >= 30].dropna(subset=["rte_daily_median"]).sort_values("rte_daily_median")
    fig, ax = plt.subplots(figsize=(9.5, max(4, 0.26 * len(s))))
    y = np.arange(len(s))
    col = [REGION_COLORS.get(r, "#888") for r in s["region"]]
    ax.hlines(y, s["rte_daily_p25"], s["rte_daily_p75"], color=col, lw=4, alpha=0.5)
    ax.scatter(s["rte_daily_median"], y, color=col, zorder=3, s=18, label="daily RTE, median and IQR")
    ax.scatter(s["rte_corrected"], y, marker="|", color="k", s=80, zorder=4, label="RTE over the year")
    if "rte_regression_companion" in s:
        ax.scatter(s["rte_regression_companion"], y, marker="x", color="#666", s=18, zorder=4,
                   label="conversion RTE, regression (bess-soc-soh-degradation)")
    reg = s["rte_registered"].where(s["rte_registered"] < 0.999)
    ax.scatter(reg, y, marker="o", facecolors="none", edgecolors="#999", s=18,
               label="registered efficiency factors (where below 1)")
    ax.set_yticks(y, _labels(s), fontsize=8.5)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("round-trip efficiency at the SCADA point")
    ax.legend(fontsize=8.5, loc="upper left")  # the best batteries sit right, the space to their left is empty
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("NEM batteries: round-trip efficiency at AEMO's SCADA point (Oct 2025 - Sep 2026)", fontsize=10)
    _save(fig, path)


def chart_fleet_availability(summary: pd.DataFrame, path: Path):
    tech = "availability_technical" in summary
    if tech:
        summary = summary[summary["in_service_months"] >= 6]
        off, ok = "availability_offered_in_service", "availability_offered_soc_ok_in_service"
        key = "availability_technical"
    else:
        off, ok, key = "availability_capacity", "availability_capacity_soc_ok", "availability_capacity_soc_ok"
    s = summary.dropna(subset=[off]).sort_values(key)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.22 * len(s))))
    y = np.arange(len(s))
    right = s[[off, ok] + ([key] if tech else [])].max(axis=1)
    ax.hlines(y, s[off], right, color="#bbb", lw=2)
    ax.scatter(s[off], y, color="#eb6834", s=16, zorder=3, label="offered, all intervals")
    ax.scatter(s[ok], y, color="#2a78d6", s=16, zorder=3,
               label="offered, intervals with more than 10 % of the registered energy stored")
    if tech:
        ax.scatter(s["availability_technical"], y, color="#1baf7a", marker="D", s=16, zorder=4,
                   label="technical: declared physical availability (PASA), SoC-aware, both directions")
    ax.set_yticks(y, _labels(s), fontsize=7)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("capacity-weighted availability (MW available / registered MW)")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("NEM batteries in service for 6+ months: offered and technical availability", fontsize=10)
    _save(fig, path)


def chart_acen(results: Path, path: Path):
    d = pd.read_csv(results / "acen_validation_daily.csv", index_col=0)
    d = d[d["duid"] == "NESBESS1"]
    x = pd.to_datetime(d.index)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(x, d["rte_operator"], "o-", ms=3, color="#1baf7a", label="A: operator power + operator SoC")
    ax.plot(x, d["rte_scada_operator_soc"], "s-", ms=3, color="#2a78d6", label="B: AEMO SCADA + operator SoC")
    ax.plot(x, d["rte_market"], "^-", ms=3, color="#eb6834", label="C: AEMO SCADA + energy reported to AEMO")
    low = d[["rte_operator", "rte_scada_operator_soc", "rte_market"]].min().min()
    ax.set_ylim(min(0.6, np.floor((low - 0.02) * 20) / 20), 1.0)  # every point inside the axes
    ax.set_ylabel("daily RTE")
    ax.legend(fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.32), frameon=False)
    ax.grid(alpha=0.3)
    ax.set_title("New England BESS, section 1: daily RTE from the operator's and from market data", fontsize=10)
    fig.autofmt_xdate()
    _save(fig, path)


def chart_synthetic(res: dict, results: Path, path: Path):
    d = res["daily"][res["daily"]["valid"]]
    fc = pd.read_csv(results / "synthetic_soh_forecast.csv")
    t = res["tests"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ax = axes[0]
    raw = d["e_out_pcs"] / d["e_in_pcs"]
    bins = np.linspace(0.88, 0.94, 61)
    ax.hist(raw, bins=bins, color="#eb6834", alpha=0.6, label="out / in (uncorrected)")
    ax.hist(d["rte_excl_aux"], bins=bins, color="#2a78d6", alpha=0.7, label="SoC-corrected")
    ax.axvline(0.975 ** 2 * 0.98 ** 2, color="k", ls="--", lw=1, label="truth 0.913")
    ax.set_xlabel("daily RTE without auxiliaries")
    ax.set_ylabel("days")
    ax.legend(fontsize=8.5)
    ax.set_title("Synthetic battery: daily RTE", fontsize=10)
    ax = axes[1]
    ax.fill_between(fc["years"], fc["soh_p10"], fc["soh_p90"], color="#2a78d6", alpha=0.2, label="forecast P10-P90")
    ax.plot(fc["years"], fc["soh_p50"], color="#2a78d6", label="forecast")
    ax.plot(fc["years"], fc["soh_true"], color="k", ls=":", label="truth")
    ax.plot(fc["years"], fc["soh_guaranteed"], color="#c8423b", ls="--", label="guaranteed (illustrative)")
    req = res["ltsa"]["capacity"]["forecast"]["required_mwh"] / res["warranty"]["reference_energy_mwh"]
    ax.axhline(req, color="#e3a33b", ls="-.", lw=1, label="requirement 90 MWh")
    ax.errorbar(t["elapsed_years"], t["soh"], yerr=t["U_mwh"] / 100, fmt="ko", ms=3, label="capacity tests")
    ax.set_xlim(0, 12)
    ax.set_ylim(0.8, 1.03)
    ax.set_xlabel("years since COD")
    ax.set_ylabel("tested energy / 100 MWh")
    ax.legend(fontsize=8, loc="lower left")  # the curves leave the lower left empty
    ax.set_title("Capacity tests and SoH forecast", fontsize=10)
    _save(fig, path)


def build(results: Path, docs: Path) -> None:
    results, docs = Path(results), Path(docs)
    (docs / "img").mkdir(parents=True, exist_ok=True)
    (docs / "report").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        sim, frame, res = analyse_demo(Path(tmp))
        paths = write_outputs(res, EXAMPLE_MONTH, Path(tmp) / "out", prefix="example_")
        shutil.copy(paths["html"], docs / "report" / f"example-{EXAMPLE_MONTH}.html")
        shutil.copy(paths["excel"], docs / "report" / f"example-{EXAMPLE_MONTH}.xlsx")
    data = {"example_month": EXAMPLE_MONTH, "synthetic": synthetic_data(results, res)}
    chart_synthetic(res, results, docs / "img" / "synthetic.png")
    if (results / "nem_fleet_summary.csv").exists():
        data["fleet"] = fleet_data(results)
        summ = pd.read_csv(results / "nem_fleet_summary.csv", index_col=0)
        summ = summ[summ["operational_months"] >= 3]
        chart_fleet_rte(summ, docs / "img" / "fleet_rte.png")
        chart_fleet_availability(summ, docs / "img" / "fleet_availability.png")
    acen = acen_data(results)
    if acen:
        data["acen"] = acen
        chart_acen(results, docs / "img" / "acen_rte.png")
    (docs / "data.js").write_text("window.DATA = " + json.dumps(_json(data), separators=(",", ":")) + ";\n",
                                  encoding="utf-8")
    log.info("docs data written to %s", docs / "data.js")
