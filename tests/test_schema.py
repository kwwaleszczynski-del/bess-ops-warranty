import numpy as np
import pandas as pd
import pytest

from bess_ops.config import deep_merge, site_from_dict, warranty_from_dict
from bess_ops.schema import data_quality, mapping_from_dict, to_site_frame, to_utc

TZ = "Europe/Warsaw"


def local_export(start: str, periods: int, step="15min") -> pd.Series:
    """Naive local wall-clock stamps of a chronological export (the autumn hour appears twice)."""
    utc = pd.date_range(pd.Timestamp(start).tz_localize(TZ).tz_convert("UTC"), periods=periods, freq=step)
    return pd.Series(utc.tz_convert(TZ).tz_localize(None))


def test_autumn_repeated_hour_is_resolved_in_order():
    ts = local_export("2025-10-26 00:00", 100)                 # the 25-hour day
    assert ts.duplicated().sum() == 4                         # 02:00-02:45 twice
    utc = to_utc(ts, TZ, pd.DataFrame({"d": ["m"] * len(ts)}))
    assert utc.is_monotonic_increasing and utc.is_unique
    assert (utc.diff().dropna() == pd.Timedelta("15min")).all()
    first = ts[ts == pd.Timestamp("2025-10-26 02:00")].index
    assert utc[first[0]] == pd.Timestamp("2025-10-26 00:00", tz="UTC")      # 02:00 CEST
    assert utc[first[1]] == pd.Timestamp("2025-10-26 01:00", tz="UTC")      # 02:00 CET


def test_spring_gap_is_an_error_and_offsets_are_accepted():
    with pytest.raises(ValueError):
        to_utc(pd.Series(pd.to_datetime(["2026-03-29 02:15"])), TZ)
    iso = pd.Series(["2025-10-26T02:15:00+02:00", "2025-10-26T02:15:00+01:00"])
    utc = to_utc(iso, TZ)
    assert (utc.diff().iloc[1]) == pd.Timedelta("1h")


def long_table(ts: pd.Series, values: dict) -> pd.DataFrame:
    rows = []
    for (dev, sig), v in values.items():
        rows.append(pd.DataFrame({"ts": ts, "dev": dev, "sig": sig, "val": v}))
    df = pd.concat(rows, ignore_index=True)
    order = np.argsort(np.tile(np.arange(len(ts)), len(values)), kind="stable")
    return df.iloc[order].reset_index(drop=True)


def test_site_frame_scales_combines_and_shifts():
    ts = local_export("2025-10-25 22:00", 16)
    n = len(ts)
    long = long_table(ts, {("M", "P_kW"): np.arange(n) * 1000.0, ("B1", "SOC"): np.full(n, 40.0),
                           ("B2", "SOC"): np.full(n, 60.0), ("P1", "TC"): np.full(n, 20.0),
                           ("P2", "TC"): np.full(n, 30.0)})
    m = mapping_from_dict({"columns": {"timestamp": "ts", "device_id": "dev", "signal": "sig", "value": "val"},
                           "timestamp_label": "end",
                           "signals": {"p_meter": {"name": "P_kW", "scale": 0.001}, "soc": {"name": "SOC",
                                                                                           "scale": 0.01},
                                       "t_cell_max": "TC"}})
    site = to_site_frame(long, m)
    assert len(site) == n
    assert site.attrs["dt_h"] == 0.25
    # mean values labelled by interval end move back one step; states do not
    assert site["p_meter"].iloc[0] == pytest.approx(1.0)
    assert np.isnan(site["p_meter"].iloc[-1])
    assert site["soc"].iloc[0] == pytest.approx(0.5)            # mean of the two BMS
    assert site["t_cell_max"].iloc[0] == 30.0                   # max over devices


def test_sample_power_becomes_interval_mean():
    ts = pd.Series(pd.date_range("2025-06-01", periods=4, freq="5min"))
    long = pd.DataFrame({"timestamp": ts, "device_id": "U", "signal": "P", "value": [0.0, 10.0, 10.0, -10.0]})
    m = mapping_from_dict({"timezone": "UTC", "step": "5min", "signals": {"p_meter": {"name": "P",
                                                                                      "kind": "sample"}}})
    site = to_site_frame(long, m)
    assert site["p_meter"].tolist()[:3] == [5.0, 10.0, 0.0]


def test_quality_flags_inverted_state():
    idx = pd.date_range("2025-06-01", periods=200, freq="15min", tz="UTC")
    p = np.where(np.arange(200) % 40 < 20, 10.0, -10.0)
    e = 50 + np.cumsum(p) * 0.25                               # rises while discharging: inverted
    site = pd.DataFrame({"p_meter": p, "e_stored": np.r_[50, e[:-1]]}, index=idx)
    site.attrs.update(dt_h=0.25, timezone="UTC")
    q = data_quality(site)
    assert "inverted" in q.loc["e_stored", "check"]


def test_config_rejects_unknown_keys_and_fills_defaults():
    with pytest.raises(ValueError):
        deep_merge({"a": 1}, {"b": 2})
    site = site_from_dict({"power_mw": 5, "energy_mwh": 10})
    assert site["soc_energy_mwh"] == 10 and site["charge_power_mw"] == 5
    w = warranty_from_dict({"throughput": {"efc_per_year": 300}}, site)
    assert w["throughput"]["efc_per_year"] == 300 and w["reference_energy_mwh"] == 10
    with pytest.raises(ValueError):
        warranty_from_dict({"throughput": {"efc_per_yer": 300}}, site)
