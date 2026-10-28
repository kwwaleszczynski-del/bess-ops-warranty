"""Download of two more AEMO tables for the batteries: charge-side and physical availability.

The cache of the companion project nem-battery-behaviour has the discharge availability a battery offers
(``DISPATCHLOAD.AVAILABILITY``). Two more fields complete the picture of a bidirectional unit:

====================================  ==========================================================================
AEMO table.column                     meaning
====================================  ==========================================================================
DISPATCHLOAD.MIN_AVAILABILITY         charge-side (load) availability offered, MW, per 5-minute interval
BIDPEROFFER_D.MAXAVAIL                availability offered in the final energy bid, per direction (GEN, LOAD)
BIDPEROFFER_D.PASAAVAILABILITY        physical availability declared to PASA: what the unit could deliver with
                                      24 hours' notice, per direction, regardless of price or commercial reserve
====================================  ==========================================================================

The monthly MMSDM archive files (one zip per table and month, about 120 MB for DISPATCHLOAD and 170 MB for
BIDPEROFFER_D) are downloaded, filtered on the fly to the batteries (and to energy bids) and stored as small
Parquet files; the zips are deleted afterwards.

    <out>/dispatchload/2025-10.parquet   SETTLEMENTDATE, DUID, AVAILABILITY, MIN_AVAILABILITY
    <out>/bids/2025-10.parquet           INTERVAL_DATETIME, DUID, DIRECTION, MAXAVAIL, PASAAVAILABILITY

Times are AEMO's: NEM market time, interval END.

Usage:
    python -m bess_ops fetch-nem --registry ../nem-battery-behaviour/data/registry.parquet --out data/nem_extra \\
        --start 2025-10 --end 2026-09 --jobs 4
"""
from __future__ import annotations

import csv
import io
import logging
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)
BASE_URL = "https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM"
USER_AGENT = "bess-ops-warranty (open-source research; github)"

TABLES = {
    "dispatchload": {"table": "DISPATCHLOAD",
                     "columns": ["SETTLEMENTDATE", "DUID", "INTERVENTION", "AVAILABILITY", "MIN_AVAILABILITY"],
                     "numeric": ["AVAILABILITY", "MIN_AVAILABILITY"], "where": {}},
    "bids": {"table": "BIDPEROFFER_D",
             "columns": ["INTERVAL_DATETIME", "DUID", "BIDTYPE", "DIRECTION", "OFFERDATE", "VERSIONNO", "MAXAVAIL",
                         "PASAAVAILABILITY"],
             "numeric": ["MAXAVAIL", "PASAAVAILABILITY"], "where": {"BIDTYPE": {"ENERGY"}}},
}


def mmsdm_url(table: str, year: int, month: int, part: int = 1) -> str:
    folder = f"{BASE_URL}/{year}/MMSDM_{year}_{month:02d}/MMSDM_Historical_Data_SQLLoader/DATA"
    return f"{folder}/PUBLIC_ARCHIVE%23{table}%23FILE{part:02d}%23{year}{month:02d}010000.zip"


def download(url: str, dest: Path, retries: int = 4) -> bool:
    """Stream ``url`` to ``dest``; False on HTTP 404."""
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    for attempt in range(retries):
        try:
            with session.get(url, stream=True, timeout=180) as r:
                if r.status_code == 404:
                    return False
                r.raise_for_status()
                tmp = dest.with_suffix(".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                tmp.replace(dest)
                return True
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            log.warning("download failed (%s), retrying", e)
            time.sleep(5 * 2 ** attempt)
    return False


def read_mms_zip(path: Path, columns: list[str], where: dict[str, set[str]],
                 chunksize: int = 1_000_000) -> pd.DataFrame:
    """Data rows of the (single) table in an MMS CSV zip, only ``columns``, only rows matching ``where``.

    Every line of an MMS file starts with a record type: C (comment, header, footer), I (column names of the table
    that follows) or D (data). The I line gives the names; rows are read in chunks with pandas and filtered.
    """
    frames = []
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not name.upper().endswith(".CSV"):
                continue
            with zf.open(name) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
                text.readline()                                      # C: file header
                header = next(csv.reader([text.readline()]))         # I: record type, package, table, version, cols
                names = header[:4] + header[4:]
                names[0] = "_record"
                use = ["_record"] + [c for c in columns if c in names]
                for chunk in pd.read_csv(text, header=None, names=names, usecols=use, dtype=str, engine="c",
                                         chunksize=chunksize, on_bad_lines="skip"):
                    keep = chunk["_record"] == "D"
                    for col, values in where.items():
                        keep &= chunk[col].isin(values)
                    frames.append(chunk.loc[keep, use[1:]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def fetch_month(kind: str, year: int, month: int, out: Path, duids: list[str], refresh: bool = False) -> Path:
    spec = TABLES[kind]
    dest = out / kind / f"{year}-{month:02d}.parquet"
    if dest.exists() and not refresh:
        return dest
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames, zips = [], []
    for part in range(1, 10):
        z = raw_dir / f"{spec['table']}_{year}{month:02d}_{part:02d}.zip"
        if not z.exists() and not download(mmsdm_url(spec["table"], year, month, part), z):
            break
        zips.append(z)
        where = dict(spec["where"], DUID=set(duids))
        frames.append(read_mms_zip(z, spec["columns"], where))
    if not zips:
        raise FileNotFoundError(f"{spec['table']} {year}-{month:02d} not found on NEMWeb")
    df = pd.concat(frames, ignore_index=True)
    for c in spec["numeric"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    if kind == "dispatchload":
        df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S")
        df["INTERVENTION"] = pd.to_numeric(df["INTERVENTION"]).astype("int8")
        df = df.sort_values("INTERVENTION").drop_duplicates(["SETTLEMENTDATE", "DUID"], keep="first")
    else:
        df["INTERVAL_DATETIME"] = pd.to_datetime(df["INTERVAL_DATETIME"], format="%Y/%m/%d %H:%M:%S")
        df["VERSIONNO"] = pd.to_numeric(df["VERSIONNO"], errors="coerce")
        # BIDPEROFFER_D holds the bids in force; if a period appears more than once keep the latest offer
        df = (df.sort_values(["OFFERDATE", "VERSIONNO"])
              .drop_duplicates(["INTERVAL_DATETIME", "DUID", "DIRECTION"], keep="last")
              .drop(columns=["BIDTYPE", "OFFERDATE", "VERSIONNO"]))
    for c in ("DUID", "DIRECTION"):
        if c in df:
            df[c] = df[c].astype("category")
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    for z in zips:
        z.unlink(missing_ok=True)
    log.info("%s %d-%02d: %d rows -> %s", kind, year, month, len(df), dest)
    return dest


def fetch(registry: Path, out: Path, start: str, end: str, jobs: int = 4, refresh: bool = False) -> None:
    """Both tables for every month from ``start`` to ``end`` (YYYY-MM), at most ``jobs`` processes."""
    duids = sorted(pd.read_parquet(registry)["DUID"].astype(str).unique())
    months = [(p.year, p.month) for p in pd.period_range(start, end, freq="M")]
    tasks = [(kind, y, m) for y, m in months for kind in TABLES]
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=max(1, min(jobs, 4))) as pool:
        futures = {pool.submit(fetch_month, k, y, m, out, duids, refresh): (k, y, m) for k, y, m in tasks}
        for f in as_completed(futures):
            k, y, m = futures[f]
            f.result()
            log.info("done %s %d-%02d", k, y, m)
