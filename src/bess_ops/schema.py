"""Generic input: EMS/SCADA signals of a battery in a long table, mapped to canonical signals by a YAML file.

Input table (Parquet or CSV), one row per timestamp, device and signal::

    timestamp            device_id   signal       value
    2026-03-29 01:45:00  METER-01    P_POI_kW     -24810.0
    2026-03-29 01:45:00  PCS-01      P_DC_kW      -12620.5
    2026-03-29 01:45:00  BMS-01      SOC_PCT          41.3

The mapping file (``config/mapping.example.yaml``) names the four columns, the time zone and interval convention of
the timestamps and maps your signal names to the canonical signals below, with a scale factor (kW -> MW, % ->
fraction), an optional sign flip and the rule that combines several devices into one site value.

=============  ====================================================================  ==============  =========
signal         meaning                                                               unit            combine
=============  ====================================================================  ==============  =========
p_meter        active power at the revenue meter / point of connection, + export     MW              sum
p_dc           DC power at the battery terminals, + discharge                        MW              sum
p_aux          auxiliary consumption (HVAC, transformers, control), + consumption    MW              sum
soc            state of charge                                                       fraction 0..1   mean
e_stored       stored energy (alternative to ``soc``)                                MWh             sum
t_amb          ambient temperature                                                   degC            mean
t_cell_max     highest cell temperature                                              degC            max
t_cell_avg     mean cell temperature                                                 degC            mean
p_avail_dis    available discharge power (what the EMS reports it can deliver)       MW              sum
p_avail_ch     available charge power (magnitude)                                    MW              sum
e_avail        available (dischargeable) energy                                      MWh             sum
=============  ====================================================================  ==============  =========

Only ``p_meter`` is required; every estimator states which signals it needs and is skipped without them.

Time:

* Timestamps without an offset are read in the mapping's ``timezone`` (default ``Europe/Warsaw``). In autumn the
  hour 02:00-03:00 occurs twice; the first occurrence of each timestamp per device and signal is taken as summer
  time (CEST) and the second as winter time (CET), which is what an export in chronological order contains. Spring
  timestamps that do not exist (02:00-03:00 on the last Sunday of March) are an error. Timestamps with an offset
  (ISO 8601, e.g. ``2026-10-25T02:15:00+01:00``) or ``timezone: UTC`` are unambiguous.
* Internally every series is on a regular UTC grid indexed by the START of each interval. Mean values labelled by
  the END of their interval (``timestamp_label: end``) are moved back by one step. State signals (``soc``,
  ``e_stored``) are instants: the value in row ``t`` is the state at instant ``t``.
* Power given as instantaneous readings (``kind: sample``) is turned into interval means with the trapezoidal rule.

Usage:
    mapping = load_mapping(Path("config/mapping.example.yaml"))
    long = read_table(Path("ems_export.parquet"))
    site = to_site_frame(long, mapping)        # one column per canonical signal, regular UTC grid
    quality = data_quality(site)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)

SIGNALS = {
    "p_meter": ("MW", "sum"), "p_dc": ("MW", "sum"), "p_aux": ("MW", "sum"),
    "soc": ("fraction", "mean"), "e_stored": ("MWh", "sum"),
    "t_amb": ("degC", "mean"), "t_cell_max": ("degC", "max"), "t_cell_avg": ("degC", "mean"),
    "p_avail_dis": ("MW", "sum"), "p_avail_ch": ("MW", "sum"), "e_avail": ("MWh", "sum"),
}
STATE_SIGNALS = ("soc", "e_stored")
POWER_SIGNALS = ("p_meter", "p_dc", "p_aux")


@dataclass
class SignalMap:
    name: str                       # signal name in your table
    scale: float = 1.0              # multiply the raw value by this (e.g. 0.001 for kW -> MW)
    offset: float = 0.0             # then add this
    sign: int = 1                   # -1 flips the sign (e.g. a meter that counts import as positive)
    devices: list[str] | None = None  # only these devices (default: every device reporting the signal)
    combine: str | None = None      # sum | mean | max | min (default from SIGNALS)
    kind: str = "mean"              # mean (interval average) | sample (instantaneous reading)


@dataclass
class Mapping:
    columns: dict = field(default_factory=lambda: {"timestamp": "timestamp", "device_id": "device_id",
                                                   "signal": "signal", "value": "value"})
    timezone: str = "Europe/Warsaw"
    timestamp_label: str = "start"  # start | end of the interval a mean value belongs to
    step: str = "15min"
    nonexistent: str = "raise"      # spring-forward timestamps: raise | shift_forward | NaT
    timestamp_format: str | None = None  # strftime pattern if not ISO 8601, e.g. "%d.%m.%Y %H:%M"
    signals: dict[str, SignalMap] = field(default_factory=dict)

    @property
    def step_td(self) -> pd.Timedelta:
        return pd.Timedelta(self.step)


def load_mapping(path: Path) -> Mapping:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return mapping_from_dict(raw)


def mapping_from_dict(raw: dict) -> Mapping:
    m = Mapping()
    m.columns.update(raw.get("columns") or {})
    for key in ("timezone", "timestamp_label", "step", "nonexistent", "timestamp_format"):
        if key in raw:
            setattr(m, key, raw[key])
    if m.timestamp_label not in ("start", "end"):
        raise ValueError("timestamp_label must be 'start' or 'end'")
    for canon, spec in (raw.get("signals") or {}).items():
        if canon not in SIGNALS:
            raise ValueError(f"unknown canonical signal {canon!r}; known: {list(SIGNALS)}")
        spec = {"name": spec} if isinstance(spec, str) else dict(spec)
        m.signals[canon] = SignalMap(**spec)
    if "p_meter" not in m.signals:
        raise ValueError("the mapping must define p_meter (active power at the meter)")
    return m


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def to_utc(ts: pd.Series, tz: str, keys: pd.DataFrame | None = None, nonexistent: str = "raise",
           timestamp_format: str | None = None) -> pd.Series:
    """Parse timestamps and convert them to UTC.

    Naive local times are localised in ``tz``. For the repeated autumn hour the n-th occurrence of a timestamp within
    each group of ``keys`` (device and signal) decides: first = summer time, later = winter time.
    """
    if pd.api.types.is_datetime64_any_dtype(ts):
        if getattr(ts.dt, "tz", None) is not None:
            return ts.dt.tz_convert("UTC")
        parsed = ts
    else:
        text = ts.astype(str).str.strip()
        if text.str.contains(r"(?:[+-]\d{2}:?\d{2}|Z)$", regex=True).all():
            return pd.to_datetime(text, utc=True, format=timestamp_format or "ISO8601")
        parsed = pd.to_datetime(text, format=timestamp_format or "ISO8601")
    if tz.upper() == "UTC":
        return parsed.dt.tz_localize("UTC")
    frame = pd.DataFrame({"ts": parsed})
    if keys is not None:
        frame = pd.concat([frame, keys.reset_index(drop=True).set_axis(frame.index)], axis=1)
    occurrence = frame.groupby(list(frame.columns), sort=False).cumcount().to_numpy()
    first_is_dst = occurrence == 0
    try:
        local = parsed.dt.tz_localize(tz, ambiguous=first_is_dst, nonexistent=nonexistent)
    except Exception as e:  # pytz/zoneinfo raise different classes for non-existent times
        raise ValueError(f"timestamp does not exist in {tz} (spring DST gap?): {e}") from e
    return local.dt.tz_convert("UTC")


def to_site_frame(long: pd.DataFrame, mapping: Mapping) -> pd.DataFrame:
    """Long table -> regular UTC grid (interval start) with one column per mapped canonical signal."""
    c = mapping.columns
    missing = [v for v in c.values() if v not in long.columns]
    if missing:
        raise ValueError(f"input lacks columns {missing}; the mapping expects {list(c.values())}")
    step = mapping.step_td
    df = pd.DataFrame({"device": long[c["device_id"]].astype(str).to_numpy(),
                       "signal": long[c["signal"]].astype(str).to_numpy(),
                       "value": pd.to_numeric(long[c["value"]], errors="coerce").to_numpy()})
    df["ts"] = to_utc(long[c["timestamp"]].reset_index(drop=True), mapping.timezone,
                      df[["device", "signal"]], mapping.nonexistent, mapping.timestamp_format).to_numpy()
    if df["ts"].isna().any():
        log.warning("%d timestamps could not be placed in %s and are dropped", df["ts"].isna().sum(),
                    mapping.timezone)
        df = df.dropna(subset=["ts"])
    off_grid = (df["ts"] - df["ts"].dt.floor(step)) != pd.Timedelta(0)
    if off_grid.any():
        log.warning("%d rows not on the %s grid are floored to it", int(off_grid.sum()), mapping.step)
        df["ts"] = df["ts"].dt.floor(step)
    start, end = df["ts"].min(), df["ts"].max()
    grid = pd.date_range(start, end, freq=step, tz="UTC", name="interval_start")
    out = pd.DataFrame(index=grid)
    for canon, sm in mapping.signals.items():
        sel = df[df["signal"] == sm.name]
        if sm.devices:
            sel = sel[sel["device"].isin(sm.devices)]
        if sel.empty:
            log.warning("signal %s (%s) not found in the input", canon, sm.name)
            continue
        val = (sel["value"] * sm.scale + sm.offset) * sm.sign
        wide = pd.DataFrame({"ts": sel["ts"], "device": sel["device"], "v": val})
        dup = wide.duplicated(["ts", "device"])
        if dup.any():
            log.warning("%s: %d duplicated timestamps per device, mean taken", canon, int(dup.sum()))
        wide = wide.groupby(["ts", "device"])["v"].mean().unstack("device").reindex(grid)
        if sm.kind == "sample" and canon not in STATE_SIGNALS:
            wide = (wide + wide.shift(-1)) / 2           # trapezoid between this and the next reading
        elif mapping.timestamp_label == "end" and canon not in STATE_SIGNALS:
            wide = wide.shift(-1)                        # value labelled t belongs to the interval starting t-step
        how = sm.combine or SIGNALS[canon][1]
        series = getattr(wide, how)(axis=1, skipna=True)
        series[wide.isna().all(axis=1)] = np.nan
        out[canon] = series.astype(float)
    out.attrs["dt_h"] = step / pd.Timedelta("1h")
    out.attrs["timezone"] = mapping.timezone if mapping.timezone.upper() != "UTC" else "UTC"
    return out


def site_frame(data: dict[str, np.ndarray] | pd.DataFrame, index: pd.DatetimeIndex, timezone: str) -> pd.DataFrame:
    """Build a site frame directly from arrays (used by the NEM adapter and the tests)."""
    out = pd.DataFrame(data, index=index)
    if out.index.tz is None:
        raise ValueError("index must be tz-aware")
    out.index = out.index.tz_convert("UTC")
    out.index.name = "interval_start"
    step = out.index[1] - out.index[0]
    out.attrs["dt_h"] = step / pd.Timedelta("1h")
    out.attrs["timezone"] = timezone
    return out


def local_index(site: pd.DataFrame) -> pd.DatetimeIndex:
    return site.index.tz_convert(site.attrs.get("timezone", "UTC"))


def stored_energy(site: pd.DataFrame, soc_energy_mwh: float | None) -> pd.Series | None:
    """Stored energy (MWh) at every instant: ``e_stored`` if given, else ``soc`` x the energy SoC 1.0 stands for."""
    if "e_stored" in site:
        return site["e_stored"]
    if "soc" in site and soc_energy_mwh:
        return site["soc"] * soc_energy_mwh
    return None


def data_quality(site: pd.DataFrame, soc_energy_mwh: float | None = None) -> pd.DataFrame:
    """Coverage and plausibility per signal, and a sign check of the state signal against the metered power."""
    rows = []
    n = len(site)
    for col in site.columns:
        s = site[col]
        rows.append({"signal": col, "coverage": float(s.notna().mean()) if n else 0.0,
                     "min": float(s.min()) if s.notna().any() else np.nan,
                     "max": float(s.max()) if s.notna().any() else np.nan, "check": ""})
    q = pd.DataFrame(rows).set_index("signal") if rows else pd.DataFrame(columns=["coverage", "min", "max", "check"])
    if "soc" in q.index and (q.loc["soc", "max"] > 1.05 or q.loc["soc", "min"] < -0.05):
        q.loc["soc", "check"] = "outside 0..1: is the scale right (percent -> fraction)?"
    e = stored_energy(site, soc_energy_mwh or 1.0)
    if e is not None and "p_meter" in site:
        corr = state_power_correlation(e, site["p_meter"])
        key = "e_stored" if "e_stored" in site else "soc"
        if np.isfinite(corr) and corr > 0:
            q.loc[key, "check"] = (f"stored energy rises while discharging (corr {corr:+.2f}): inverted or "
                                   "reported as headroom?")
        elif np.isfinite(corr):
            q.loc[key, "check"] = f"consistent with metered power (corr {corr:+.2f})"
    return q


def state_power_correlation(e: pd.Series, p_meter: pd.Series) -> float:
    """Correlation of the change of stored energy over an interval with the metered power (expected negative)."""
    de = e.shift(-1) - e
    ok = de.notna() & p_meter.notna() & (p_meter.abs() > 0)
    if ok.sum() < 10:
        return float("nan")
    return float(np.corrcoef(de[ok], p_meter[ok])[0, 1])
