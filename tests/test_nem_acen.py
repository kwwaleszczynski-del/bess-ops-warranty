"""The NEM adapter on a small fake AEMO cache and the ACEN parser on both published time-stamp formats."""
import numpy as np
import pandas as pd
import pytest

from bess_ops.acen import parse
from bess_ops.nem import Cache, exact_split, unit_benchmark


def test_exact_split_by_hand():
    out, inn = exact_split(np.array([10.0, -10.0, -10.0, 0.0, 20.0]))
    dt = 5 / 60
    # 10 -> -10 crosses zero halfway: 2.5 MW-intervals each way; -10 -> -10: 10 in; -10 -> 0: 5 in; 0 -> 20: 10 out
    assert out[:4] == pytest.approx(np.array([2.5, 0, 0, 10]) * dt)
    assert inn[:4] == pytest.approx(np.array([2.5, 10, 5, 0]) * dt)
    assert np.isnan(out[-1])


def fake_cache(days=40, eta=0.92, inverted_days=0, mw=100.0, mwh=200.0) -> Cache:
    """A battery in the AEMO format: one cycle a day, SCADA at the interval start, stored energy reported."""
    start = pd.Timestamp("2026-01-01 00:05")
    settle = pd.date_range(start, periods=288 * days, freq="5min")          # interval END (SETTLEMENTDATE)
    begin = settle - pd.Timedelta("5min")
    hour = begin.hour.to_numpy() + begin.minute.to_numpy() / 60
    # 48 readings of -40 MW (160 MWh) charge, 40 readings discharge what that stored at the round trip eta
    p = np.where((hour >= 10) & (hour < 14), -40.0, np.where((hour >= 18) & (hour < 21.3), 160 * eta / (40 / 12), 0.0))
    one_way = np.sqrt(eta)
    dt = 5 / 60
    e = np.empty(len(p))
    e[0] = 20.0
    for i in range(1, len(p)):
        q = p[i - 1]
        e[i] = e[i - 1] - (q / one_way if q > 0 else q * one_way) * dt
    e_rep = e.copy()
    n_inv = 288 * inverted_days
    e_rep[:n_inv] = mwh - e[:n_inv]
    avail = np.full(len(p), mw)
    avail[(begin >= "2026-01-20 00:00") & (begin < "2026-01-21 00:00")] = 0.0            # one day offline
    scada = pd.DataFrame({"SETTLEMENTDATE": settle, "DUID": "TEST1", "SCADAVALUE": p})
    dispatch = pd.DataFrame({"SETTLEMENTDATE": settle, "DUID": "TEST1", "INTERVENTION": 0, "AVAILABILITY": avail,
                             "INITIAL_ENERGY_STORAGE": e_rep})
    reg = pd.DataFrame({"DUID": ["TEST1"], "EFFECTIVEDATE": [pd.Timestamp("2025-01-01")], "VERSIONNO": [1],
                        "MAXCAPACITY": [mw], "MAXSTORAGECAPACITY": [mwh], "REGIONID": ["NSW1"],
                        "STATIONID": ["TEST"], "STORAGEIMPORTEFFICIENCYFACTOR": [0.95],
                        "STORAGEEXPORTEFFICIENCYFACTOR": [0.95]})
    return Cache(reg, None, scada, dispatch)


def test_unit_benchmark_recovers_rte_and_availability():
    monthly, summary, daily = unit_benchmark(fake_cache(), "TEST1")
    assert summary["rte_corrected"] == pytest.approx(0.92, abs=0.005)
    assert daily.median() == pytest.approx(0.92, abs=0.01)
    jan = monthly.loc["2026-01"]
    assert jan["availability_capacity"] == pytest.approx(30 / 31, abs=0.002)
    assert jan["stored_consistent"]


def test_inverted_report_is_flagged():
    monthly, summary, daily = unit_benchmark(fake_cache(inverted_days=31), "TEST1")
    assert not monthly.loc["2026-01", "stored_consistent"]
    assert monthly.loc["2026-02", "stored_consistent"]


def test_acen_parser_both_formats(tmp_path):
    header = '"t_stamp",' + ",".join(f'"c{i}"' for i in range(24)) + "\n"
    row = lambda ts: ts + "," + ",".join(["0"] * 20) + ",50000,41.5,-25000,60\n"  # noqa: E731
    a = tmp_path / "a.csv"
    a.write_text(header + row("1/05/2026 0:00") + row("1/05/2026 0:05"))
    b = tmp_path / "b.csv"
    b.write_text(header + row("2026-09-01 00:00:00.000"))
    fa, fb = parse(a), parse(b)
    assert fa.index[1] == pd.Timestamp("2026-05-01 00:05")
    assert fb.index[0] == pd.Timestamp("2026-09-01 00:00")
    assert fa["mw_1"].iloc[0] == 50.0 and fa["soc_1"].iloc[0] == pytest.approx(0.415)
    assert fa["mw_2"].iloc[0] == -25.0 and fa["soc_2"].iloc[0] == pytest.approx(0.60)
