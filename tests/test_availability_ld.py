"""Availability formulas and liquidated damages on examples small enough to check by hand."""
import numpy as np
import pandas as pd
import pytest

from bess_ops.availability import interval_availability, normalise_events, period_availability
from bess_ops.config import ltsa_from_dict, site_from_dict
from bess_ops.ld import availability_ld, capacity_ld, liquidated_damages, rte_ld

from conftest import hourly_frame

SITE = site_from_dict({"power_mw": 10, "energy_mwh": 20, "timezone": "UTC", "cod": None})


def ltsa(**availability):
    return ltsa_from_dict({"availability": {"period": "month", "directions": ["discharge"], **availability}}, SITE)


def ten_hours(avail, e_avail=None):
    data = {"p_meter": np.zeros(10), "p_avail_dis": avail}
    if e_avail is not None:
        data["e_avail"] = e_avail
    return hourly_frame(data, start="2025-06-02 00:00", tz="UTC")


def events(rows):
    return normalise_events(pd.DataFrame(rows, columns=["start", "end", "category"]), "UTC")


AVAIL = [10, 10, 0, 0, 10, 5, 10, 10, 10, 10]
EV = [("2025-06-02 02:00", "2025-06-02 03:00", "tso_instruction"),     # hour 2: deemed available
      ("2025-06-02 03:00", "2025-06-02 04:00", "forced_outage")]       # hour 3: counted


def availability(site, lt, ev=None):
    iv = interval_availability(site, lt, ev)
    return period_availability(iv, lt, SITE, events=ev).iloc[0]


def test_time_formula():
    # available hours: 0, 1, 2 (deemed), 4, 6, 7, 8, 9 -> 8 of 10; hour 5 at 50 % is below the 95 % threshold
    assert availability(ten_hours(AVAIL), ltsa(formula="time"), events(EV))["availability"] == pytest.approx(0.8)


def test_capacity_formula():
    # 1 + 1 + 1 (deemed) + 0 + 1 + 0.5 + 4 x 1 = 8.5 of 10
    assert availability(ten_hours(AVAIL), ltsa(formula="capacity"), events(EV))["availability"] == \
        pytest.approx(0.85)


def test_energy_formula():
    e = [20, 20, 20, 20, 10, 20, 20, 20, 20, 20]       # hour 4: half the energy available
    a = availability(ten_hours(AVAIL, e), ltsa(formula="energy"), events(EV))
    assert a["availability"] == pytest.approx(0.8)    # 8.5 - 0.5


def test_force_majeure_excluded_and_missing_data():
    ev = EV + [("2025-06-02 05:00", "2025-06-02 06:00", "force_majeure")]
    a = availability(ten_hours(AVAIL), ltsa(formula="time"), events(ev))
    assert a["availability"] == pytest.approx(8 / 9)
    assert a["hours_excluded"] == 1
    gaps = list(AVAIL)
    gaps[9] = np.nan
    for treatment, expected in (("excluded", 7 / 9), ("unavailable", 7 / 10), ("available", 8 / 10)):
        a = availability(ten_hours(gaps), ltsa(formula="time", missing_data=treatment), events(EV))
        assert a["availability"] == pytest.approx(expected), treatment


def test_planned_maintenance_allowance():
    # planned maintenance in hours 0-3; an allowance of 1752 h a year is 1752 x 10 / 8760 = 2 h in 10 hours
    avail = [0, 0, 0, 0, 10, 10, 10, 10, 10, 10]
    ev = [("2025-06-02 00:00", "2025-06-02 04:00", "planned_maintenance")]
    lt = ltsa(formula="time", planned_allowance_hours=1752.0)
    a = availability(ten_hours(avail), lt, events(ev))
    assert a["hours_planned_in_allowance"] == 2 and a["hours_planned_counted"] == 2
    assert a["availability"] == pytest.approx(6 / 8)


def test_overlapping_events_priority():
    ev = [("2025-06-02 02:00", "2025-06-02 04:00", "forced_outage"),
          ("2025-06-02 03:00", "2025-06-02 04:00", "force_majeure")]      # excluded wins over counted
    a = availability(ten_hours(AVAIL), ltsa(formula="time"), events(ev))
    assert a["hours_excluded"] == 1
    assert a["availability"] == pytest.approx(7 / 9)


def test_unknown_event_category_is_an_error():
    with pytest.raises(ValueError):
        availability(ten_hours(AVAIL), ltsa(), events([("2025-06-02 02:00", "2025-06-02 03:00", "lunch")]))


LD = {"ld": {"currency": "EUR", "annual_fee": 100_000, "cap_share_of_fee": 0.2,
             "availability": {"rate_per_pp": 5000, "per_mw": True},
             "rte": {"method": "energy_cost", "price_per_mwh": 100.0},
             "capacity": {"rate_per_mwh": 1000}},
      "availability": {"guaranteed": 0.97}, "rte": {"guaranteed": 0.85, "min_efc": 1.0}}


def test_availability_ld():
    lt = ltsa_from_dict(LD, SITE)
    periods = pd.DataFrame({"availability": [0.955, 0.99]}, index=["1", "2"])
    ld = availability_ld(periods, lt)
    # 1.5 pp short x 5000 per pp per MW x 10 MW = 75 000; the second year meets the guarantee
    assert ld["amount"].tolist() == pytest.approx([75_000, 0])


def test_rte_ld_energy_cost_and_per_pp():
    lt = ltsa_from_dict(LD, SITE)
    w = pd.DataFrame({"e_out": [1000.0, 1000.0], "rte": [0.80, 0.90], "rte_excl_aux": [0.85, 0.95],
                      "efc": [50, 50], "valid": [True, True]}, index=["2025-01", "2025-02"])
    ld = rte_ld(w, lt)
    assert ld["amount"].iloc[0] == pytest.approx(1000 * (1 / 0.80 - 1 / 0.85) * 100)   # 7 352.94
    assert ld["amount"].iloc[1] == 0
    lt["ld"]["rte"]["method"] = "per_pp"
    lt["ld"]["rte"]["rate_per_pp"] = 2000
    assert rte_ld(w, lt)["amount"].iloc[0] == pytest.approx(10_000)


def test_capacity_ld_and_cap():
    lt = ltsa_from_dict(LD, SITE)
    tests = pd.DataFrame({"start": pd.to_datetime(["2025-06-01"]).tz_localize("UTC"), "e_corr_mwh": [17.5],
                          "e_guaranteed_mwh": [19.0], "verdict": ["fail"]})
    assert capacity_ld(tests, lt)["amount"].iloc[0] == pytest.approx(1500)
    periods = pd.DataFrame({"availability": [0.955]}, index=["1"])
    out = liquidated_damages(periods, None, tests, lt)
    assert out["total"] == pytest.approx(76_500)
    assert out["capped_total"] == pytest.approx(20_000)            # 20 % of the 100 000 fee
