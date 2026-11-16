"""The NEM adapter on a small fake AEMO cache and the ACEN parser on both published time-stamp formats."""
import numpy as np
import pandas as pd
import pytest

from bess_ops.acen import parse
from bess_ops.mms import read_mms_zip
from bess_ops.nem import Cache, exact_split, unit_benchmark


def test_exact_split_by_hand():
    out, inn = exact_split(np.array([10.0, -10.0, -10.0, 0.0, 20.0]))
    dt = 5 / 60
    # 10 -> -10 crosses zero halfway: 2.5 MW-intervals each way; -10 -> -10: 10 in; -10 -> 0: 5 in; 0 -> 20: 10 out
    assert out[:4] == pytest.approx(np.array([2.5, 0, 0, 10]) * dt)
    assert inn[:4] == pytest.approx(np.array([2.5, 10, 5, 0]) * dt)
    assert np.isnan(out[-1])


def fake_cache(days=40, eta=0.92, inverted_days=0, mw=100.0, mwh=200.0, e0=20.0) -> Cache:
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
    e[0] = e0
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


def test_inverted_days_inside_a_passing_month_are_left_out():
    # five inverted days pass the monthly regression check of January but are caught day by day
    monthly, summary, daily = unit_benchmark(fake_cache(inverted_days=5), "TEST1")
    assert monthly.loc["2026-01", "inverted_days"] == 5
    assert not any(d < "2026-01-06" for d in daily.index)
    assert np.isnan(monthly.loc["2026-01", "rte"])          # the month starts on an inverted day: no state there


def test_withheld_only_counts_intervals_with_energy_stored():
    cache = with_extra(fake_cache())
    disp = cache.dispatch
    low = disp["INITIAL_ENERGY_STORAGE"] <= 0.10 * 200 + 1e-6      # near empty: offers nothing
    disp.loc[low, "AVAILABILITY"] = 0.0
    cache.extra["dispatchload"]["AVAILABILITY"] = disp["AVAILABILITY"].to_numpy()
    monthly, summary, daily = unit_benchmark(cache, "TEST1")
    jan = monthly.loc["2026-01"]
    assert jan["withheld_share_all_intervals"] > jan["withheld_share"] + 0.05
    assert jan["withheld_share"] == pytest.approx(1 / 31, abs=0.01)     # the offline day only


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


def with_extra(cache: Cache) -> Cache:
    """Charge-side and physical availability: full, except the discharge side at half on 10 January."""
    settle = cache.dispatch["SETTLEMENTDATE"]
    dl = pd.DataFrame({"SETTLEMENTDATE": settle, "DUID": "TEST1", "AVAILABILITY": cache.dispatch["AVAILABILITY"],
                       "MIN_AVAILABILITY": 100.0})
    pasa = np.where((settle - pd.Timedelta("5min") >= "2026-01-10") & (settle - pd.Timedelta("5min") < "2026-01-11"),
                    50.0, 100.0)
    bids = pd.concat([pd.DataFrame({"INTERVAL_DATETIME": settle, "DUID": "TEST1", "DIRECTION": "GEN",
                                    "MAXAVAIL": cache.dispatch["AVAILABILITY"], "PASAAVAILABILITY": pasa}),
                      pd.DataFrame({"INTERVAL_DATETIME": settle, "DUID": "TEST1", "DIRECTION": "LOAD",
                                    "MAXAVAIL": 100.0, "PASAAVAILABILITY": 100.0})], ignore_index=True)
    cache.extra = {"dispatchload": dl, "bids": bids}
    return cache


def test_technical_availability_from_pasa():
    # stored energy between 16 % and 77 % of 250 MWh: both directions count all the time
    monthly, summary, daily = unit_benchmark(with_extra(fake_cache(mwh=250.0, e0=40.0)), "TEST1")
    jan = monthly.loc["2026-01"]
    assert jan["availability_technical_both"] == pytest.approx(30.5 / 31, abs=1e-3)
    assert jan["availability_technical"] == pytest.approx(30.5 / 31, abs=1e-3)       # half on one day
    assert jan["availability_technical_time"] == pytest.approx(30 / 31, abs=1e-3)
    assert jan["availability_charge_offered"] == pytest.approx(1.0)
    # the offline day (nothing offered) is physically available: a whole day withheld; the half-PASA day is not
    assert jan["withheld_share"] == pytest.approx(1 / 31, abs=1e-3)
    assert jan["technical_outage_hours"] == 0
    # at 10 % stored energy or less only the charge side counts: the half-PASA discharge hours then do not
    low = unit_benchmark(with_extra(fake_cache()), "TEST1")[0].loc["2026-01"]
    assert low["availability_technical"] > low["availability_technical_both"]


def test_mms_zip_reader(tmp_path):
    import zipfile
    text = chr(10).join([
        'C,SETP.WORLD,DVD_DISPATCHLOAD,AEMO,PUBLIC,2026/07/08',
        'I,DISPATCH,UNIT_SOLUTION,6,SETTLEMENTDATE,DUID,AVAILABILITY,MIN_AVAILABILITY',
        'D,DISPATCH,UNIT_SOLUTION,6,"2026/06/01 00:05:00",BAT1,100,90',
        'D,DISPATCH,UNIT_SOLUTION,6,"2026/06/01 00:05:00",GEN1,500,',
        'D,DISPATCH,UNIT_SOLUTION,6,"2026/06/01 00:10:00",BAT1,80,70',
        'C,END OF REPORT,5',
    ]) + chr(10)
    z = tmp_path / "t.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("PUBLIC_ARCHIVE#DISPATCHLOAD#FILE01#202606010000.CSV", text)
    df = read_mms_zip(z, ["SETTLEMENTDATE", "DUID", "MIN_AVAILABILITY", "NOT_IN_FILE"], {"DUID": {"BAT1"}})
    assert list(df.columns) == ["SETTLEMENTDATE", "DUID", "MIN_AVAILABILITY"]
    assert df["MIN_AVAILABILITY"].tolist() == ["90", "70"]
    assert df["SETTLEMENTDATE"].iloc[1] == "2026/06/01 00:10:00"


def test_exact_split_keeps_a_missing_reading_missing():
    # old code: zero energy for the two intervals around the missing reading
    out, inn = exact_split(np.array([10.0, 10.0, np.nan, -10.0, -10.0]))
    dt = 5 / 60
    assert out[0] == pytest.approx(10.0 * dt) and inn[3] == pytest.approx(10.0 * dt)
    assert np.isnan(out[1]) and np.isnan(inn[1]) and np.isnan(out[2]) and np.isnan(inn[2])


def test_missing_scada_readings_invalidate_the_month_not_its_rte():
    cache = fake_cache()
    s = cache.scada
    gap = (s["SETTLEMENTDATE"] >= "2026-01-12 18:05") & (s["SETTLEMENTDATE"] < "2026-01-12 19:05")
    cache.scada = s[~gap].reset_index(drop=True)              # an hour of discharge missing from the file
    monthly, summary, daily = unit_benchmark(cache, "TEST1")
    assert np.isnan(monthly.loc["2026-01", "rte"]) and monthly.loc["2026-02", "rte"] == pytest.approx(0.92, abs=0.01)
    assert "2026-01-12" not in daily.index and "2026-01-13" in daily.index


def test_corrected_total_only_counts_months_with_a_known_change_of_stored_energy():
    # old code: the second month's energy counted, its unknown change of stored energy counted as zero
    from bess_ops.nem import _corrected_total
    from bess_ops.energy import corrected_rte
    op = pd.DataFrame({"stored_consistent": [True, True, True], "e_out_mwh": [80.0, 90.0, 70.0],
                       "e_in_mwh": [100.0, 100.0, 100.0], "d_stored_mwh": [10.0, np.nan, 0.0]},
                      index=["2026-01", "2026-02", "2026-03"])
    assert _corrected_total(op, 0.9) == pytest.approx(float(corrected_rte(150.0, 200.0, 10.0, 0.9)))
    assert np.isnan(_corrected_total(op, float("nan")))            # an undetermined efficiency is no RTE


def test_a_battery_without_a_consistent_report_has_no_rte_and_says_why():
    # old code: no status at all, the NaN simply fell out of every median
    monthly, summary, daily = unit_benchmark(fake_cache(days=31, inverted_days=31), "TEST1")
    assert np.isnan(summary["rte_corrected"]) and len(daily) == 0
    assert summary["rte_status"] == "no data: no month with a consistent stored-energy report"
    ok = unit_benchmark(fake_cache(), "TEST1")[1]
    assert ok["rte_status"] == "ok" and np.isfinite(ok["eta"]) and ok["eta"] == pytest.approx(0.92 ** 0.5, abs=0.01)
