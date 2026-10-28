import numpy as np
import pytest

from bess_ops.config import warranty_from_dict
from bess_ops.cycles import depth_histogram, rainflow, rainflow_efc, reversals
from bess_ops.warranty import detect_violations, guaranteed_soh, warranty_status

from conftest import hourly_frame


def test_rainflow_astm_example():
    # ASTM E1049-85, Fig. 6: ranges 3 (0.5), 4 (1.5), 6 (0.5), 8 (1.0), 9 (0.5)
    c = rainflow([-2, 1, -3, 5, -1, 3, -4, 4, -2])
    got = c.groupby("depth")["count"].sum().to_dict()
    assert got == {3.0: 0.5, 4.0: 1.5, 6.0: 0.5, 8.0: 1.0, 9.0: 0.5}


def test_full_cycles_and_hysteresis():
    soc = np.tile([0.0, 0.5, 1.0, 0.5], 10)
    # rainflow EFC = half the total variation: 39 steps of 0.5 -> 9.75
    assert rainflow_efc(rainflow(soc)) == pytest.approx(9.75)
    h = depth_histogram(rainflow(soc), years=1.0)
    assert h.loc["80%-100%", "efc"] == pytest.approx(9.5)           # 19 legs of full depth
    # ten smooth full cycles with measurement noise: the hysteresis filter removes the spurious turning points
    smooth = 0.5 - 0.5 * np.cos(np.linspace(0, 20 * np.pi, 2001))
    assert rainflow_efc(rainflow(smooth)) == pytest.approx(10.0, abs=1e-3)
    noisy = smooth + np.random.default_rng(0).normal(0, 0.003, len(smooth))
    assert len(reversals(noisy)[0]) > 100
    assert 21 <= len(reversals(noisy, 0.02)[0]) <= 22          # 20 extremes, start and possibly the end point
    assert rainflow_efc(rainflow(noisy, 0.02)) == pytest.approx(10.0, abs=0.1)     # extremes ride on noise


def test_violations_by_hand(site_cfg):
    n = 72
    soc = np.full(n, 0.5)
    soc[10:12] = 0.99               # 2 h above 97 %: sustained (60 min)
    soc[20] = 0.99                  # 1 h: exactly the tolerance, counts
    soc[30] = 0.01                  # below 2 %
    p = np.zeros(n)
    p[40:42] = 12.0                 # 12 MW / 20 MWh = 0.6 C for 2 h
    for d in range(3):
        p[d * 24 + 1: d * 24 + 3] = -10                 # one cycle a day ...
        p[d * 24 + 4: d * 24 + 6] = 10
    p[48 + 7: 48 + 11] = 10                             # ... and on day 3 two more discharge hours each way
    t = np.full(n, 25.0)
    t[50:52] = 45.0
    site = hourly_frame({"p_meter": p, "p_dc": p, "soc": soc, "t_cell_max": t, "t_cell_avg": t - 2}, "2025-06-02")
    w = warranty_from_dict({"throughput": {"efc_per_year": 2.0, "efc_per_day": 1.5},
                            "soc": {"min": 0.02, "max": 0.97, "sustained_minutes": 60},
                            "c_rate": {"max": 0.55, "basis": "p_dc", "sustained_minutes": 60},
                            "temperature": {"cell_max_c": 40, "sustained_minutes": 60}}, site_cfg)
    v = detect_violations(site, w, site_cfg)
    counts = v["rule"].value_counts().to_dict()
    assert counts == {"soc_high": 2, "soc_low": 1, "c_rate": 1, "cell_temp_high": 1, "efc_per_day": 2,
                      "efc_per_year": 1}
    day = v[v["rule"] == "efc_per_day"]
    assert day["value"].tolist() == pytest.approx([(20 + 24) / 20, 60 / 20])
    year = v[v["rule"] == "efc_per_year"].iloc[0]
    assert year["value"] == pytest.approx((20 + 44 + 60) / 20)


def test_equivalent_age_and_status(site_cfg):
    r = {"years": [0, 10], "soh": [1.0, 0.8], "equivalent_age": True}
    # 2 years old but 1000 cycles at an allowance of 250 a year: read at 4 years
    assert guaranteed_soh(r, 2.0, 1000, 250) == pytest.approx(0.92)
    assert guaranteed_soh(dict(r, equivalent_age=False), 2.0, 1000, 250) == pytest.approx(0.96)
    idx = 24 * 30
    p = np.where(np.arange(idx) % 24 < 2, 10.0, 0.0)             # 1 EFC of 20 MWh a day
    site = hourly_frame({"p_meter": p}, "2025-01-01")
    w = warranty_from_dict({"throughput": {"efc_per_year": 365}}, site_cfg)
    st = warranty_status(site, w, site_cfg)
    assert st["efc_total"] == pytest.approx(30)
    assert st["efc_this_year_projected"] == pytest.approx(365.25)
