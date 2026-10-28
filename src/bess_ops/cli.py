"""Command line interface.

Usage:
    python -m bess_ops demo --out out/demo                     # synthetic battery: data, analysis, monthly report
    python -m bess_ops run ems.parquet --mapping config/mapping.example.yaml --site config/site.example.yaml \\
        --warranty config/warranty.example.yaml --ltsa config/ltsa.example.yaml --events events.csv \\
        --month 2026-08 --out out/
    python -m bess_ops verify --results results                # estimators against the synthetic ground truth
    python -m bess_ops nem --nem-data-dir ../nem-battery-behaviour/data --acen-dir ../bess-soc-soh-degradation/data/acen
    python -m bess_ops site                                    # charts, docs/data.js and the example report

Paths come from arguments or the environment variables BESS_NEM_DATA_DIR and BESS_ACEN_DIR; nothing is hard-coded.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("bess_ops")


def _env_path(name: str) -> Path | None:
    v = os.environ.get(name)
    return Path(v) if v else None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="bess_ops", description=__doc__.splitlines()[0])
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="analyse your own long-format EMS/SCADA export and write the monthly report")
    r.add_argument("input", type=Path, help="Parquet or CSV: timestamp, device_id, signal, value (see README)")
    r.add_argument("--mapping", type=Path, required=True, help="YAML: columns, time zone, signal mapping")
    r.add_argument("--site", type=Path, required=True, help="YAML: ratings, time zone, auxiliary metering, COD")
    r.add_argument("--warranty", type=Path, help="YAML: warranty terms (defaults if omitted)")
    r.add_argument("--ltsa", type=Path, help="YAML: availability, RTE, capacity guarantees and LD")
    r.add_argument("--events", type=Path, help="CSV: start, end, category [, note] (local times)")
    r.add_argument("--month", help="YYYY-MM of the report (default: the last complete month)")
    r.add_argument("--out", type=Path, default=Path("out"))

    d = sub.add_parser("demo", help="simulate the synthetic battery, analyse it and write its monthly report")
    d.add_argument("--out", type=Path, default=Path("out/demo"))
    d.add_argument("--month", default="2026-08")
    d.add_argument("--seed", type=int, default=1)

    v = sub.add_parser("verify", help="compare every estimator with the synthetic ground truth")
    v.add_argument("--results", type=Path, default=Path("results"))
    v.add_argument("--seed", type=int, default=1)

    n = sub.add_parser("nem", help="fleet benchmark on the AEMO cache, validated with ACEN's operator data")
    n.add_argument("--nem-data-dir", type=Path, default=_env_path("BESS_NEM_DATA_DIR"),
                   help="cache of nem-battery-behaviour (env BESS_NEM_DATA_DIR)")
    n.add_argument("--acen-dir", type=Path, default=_env_path("BESS_ACEN_DIR"),
                   help="ACEN New England BESS CSV files (env BESS_ACEN_DIR); skipped if not given")
    n.add_argument("--regression-results", type=Path,
                   help="fleet_reported_soc_summary.csv of bess-soc-soh-degradation, for the comparison")
    n.add_argument("--results", type=Path, default=Path("results"))

    s = sub.add_parser("site", help="README charts, docs/data.js and the example monthly report for docs/")
    s.add_argument("--results", type=Path, default=Path("results"))
    s.add_argument("--docs", type=Path, default=Path("docs"))

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    if a.cmd == "run":
        from bess_ops.study import run_user
        run_user(a.input, a.mapping, a.site, a.warranty, a.ltsa, a.events, a.month, a.out)
    elif a.cmd == "demo":
        from bess_ops.study import run_demo
        run_demo(a.out, a.month, a.seed)
    elif a.cmd == "verify":
        from bess_ops.study import verify
        verify(a.results, a.seed)
    elif a.cmd == "nem":
        if a.nem_data_dir is None:
            p.error("--nem-data-dir (or BESS_NEM_DATA_DIR) is required")
        from bess_ops.study import run_nem
        run_nem(a.nem_data_dir, a.acen_dir, a.regression_results, a.results)
    elif a.cmd == "site":
        from bess_ops.site import build
        build(a.results, a.docs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
