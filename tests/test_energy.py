import numpy as np
import pandas as pd
import pytest

from bess_ops.energy import (corrected_rte, energy_balance, inverted_days, state_jumps, window_keys, window_rte)
from bess_ops.schema import site_frame


def test_corrected_rte_by_hand():
    # 80 out, 100 in, 10 MWh more in store at the end, one-way efficiency 0.9: (80 + 9) / 100
    assert corrected_rte(80, 100, 10, 0.9) == pytest.approx(0.89)
    # 10 MWh less in store: refilling would need 10 / 0.9 = 11.11 MWh more input
    assert corrected_rte(80, 100, -10, 0.9) == pytest.approx(80 / (100 + 10 / 0.9))
    assert corrected_rte(80, 100, 0, 0.9) == pytest.approx(0.8)


def toy_battery(days=3, eta=0.95, start_soc=0.3, end_extra=0.2, tz="Europe/Warsaw", start="2025-06-02"):
    """Hourly toy battery: one cycle a day, efficiency eta each way, no auxiliaries; the last day ends fuller."""
    idx = pd.date_range(pd.Timestamp(start).tz_localize(tz), periods=24 * days, freq="1h")
    p = np.zeros(len(idx))
    hour = idx.hour.to_numpy()
    p[(hour >= 10) & (hour < 14)] = -5.0
    p[(hour >= 18) & (hour < 21)] = 5.0 * 4 * eta * eta / 3      # discharge what was stored
    last = idx.normalize() == idx.normalize()[-1]
    p[last & (hour >= 18) & (hour < 21)] *= (1 - end_extra)
    e = np.empty(len(idx))
    e[0] = start_soc * 40
    for i in range(1, len(idx)):
        q = p[i - 1]
        e[i] = e[i - 1] + (-q * eta if q < 0 else -q / eta)
    return site_frame({"p_meter": p, "e_stored": e}, idx, tz), {"aux_metering": "none", "soc_energy_mwh": 40.0,
                                                                "timezone": tz, "cod": None}


def test_window_rte_recovers_efficiency_despite_state_change():
    site, cfg = toy_battery()
    daily = window_rte(site, cfg, "D", eta=0.95)
    assert daily["rte"].iloc[:-1].to_numpy() == pytest.approx(0.95 ** 2, abs=1e-9)
    # the last day leaves energy in store: raw ratio too low, corrected one right
    assert daily["rte_raw"].iloc[-1] < 0.85
    assert daily["d_stored"].iloc[-1] > 0
    auto = window_rte(site, cfg, "M")
    assert auto["eta"].iloc[0] == pytest.approx(0.95, abs=1e-6)


def test_local_days_have_23_and_25_hours():
    idx = pd.date_range(pd.Timestamp("2025-10-25").tz_localize("Europe/Warsaw"), periods=72, freq="1h")
    keys = window_keys(idx, "Europe/Warsaw", "D")
    assert keys.value_counts().loc["2025-10-26"] == 25
    idx = pd.date_range(pd.Timestamp("2026-03-28").tz_localize("Europe/Warsaw"), periods=72, freq="1h")
    assert window_keys(idx, "Europe/Warsaw", "D").value_counts().loc["2026-03-29"] == 23


def test_balance_splits_auxiliaries_and_losses():
    idx = pd.date_range("2025-06-01", periods=4, freq="1h", tz="UTC")
    site = site_frame({"p_meter": [-10.2, -0.2, 8.8, -0.2], "p_aux": [0.2, 0.2, 0.2, 0.2],
                       "p_dc": [-9.5, 0, 9.4, 0], "e_stored": [10, 19.3, 19.3, 9.7]}, idx, "UTC")
    cfg = {"aux_metering": "behind_meter", "soc_energy_mwh": 20.0, "timezone": "UTC", "cod": None}
    b = energy_balance(site, cfg, "M").iloc[0]
    # the last interval has no state after it and is left out: in 10.4, out 8.8, aux 0.6, stored -0.3
    assert b["e_in"] == pytest.approx(10.4)
    assert b["e_aux"] == pytest.approx(0.6)
    assert b["losses_total"] == pytest.approx(10.4 - 8.8 + 0.3)
    assert b["losses_pcs"] == pytest.approx((10.0 - 9.5) + (9.4 - 9.0))
    assert b["losses_battery"] == pytest.approx(9.5 - 9.4 + 0.3)


def test_state_jumps_removes_persistent_jump_only():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-06-01", periods=12 * 24 * 20, freq="5min", tz="UTC")
    p = np.where((idx.hour >= 10) & (idx.hour < 14), -20.0, np.where((idx.hour >= 18) & (idx.hour < 21), 25.0, 0.0))
    dt = 5 / 60
    e_in, e_out = (-p).clip(0) * dt, p.clip(0) * dt
    e = 50 + np.cumsum(np.r_[0, (0.95 * e_in - e_out / 0.95)[:-1]]) + rng.normal(0, 0.05, len(p))
    jump_at = 12 * 24 * 7 + 12 * 3                     # 03:00 on day 8, idle
    e[jump_at:] += 15.0                                # BMS recalibration
    lag = 12 * 24 * 12 + 12 * 18                       # a transient at an hour start: one reading off, then back
    e[lag] += 8.0
    adj, jumps, fit = state_jumps(pd.Series(e, index=idx), pd.Series(e_in, index=idx), pd.Series(e_out, index=idx),
                                  100.0)
    assert len(jumps) == 1 and jumps["jump_mwh"].iloc[0] == pytest.approx(15.0, abs=0.5)
    assert fit["a"] == pytest.approx(0.95, abs=0.01) and fit["b"] == pytest.approx(1 / 0.95, abs=0.01)
    assert (adj.iloc[-1] - adj.iloc[0]) == pytest.approx(e[-1] - 15.0 - e[0], abs=0.5)


def test_inverted_days():
    idx = pd.date_range("2025-06-01", periods=24 * 4, freq="1h", tz="UTC")
    p = np.tile(np.r_[np.full(6, -10.0), np.full(6, 10.0), np.zeros(12)], 4)
    e_in, e_out = pd.Series((-p).clip(0), index=idx), pd.Series(p.clip(0), index=idx)
    e = np.r_[0, np.cumsum(e_in - e_out)[:-1]] + 50
    e[:48] = 100 - e[:48]                               # first two days report the headroom
    c = inverted_days(pd.Series(e, index=idx), e_in, e_out, "UTC")
    assert (c.iloc[:2] < -0.9).all() and (c.iloc[2:3] > 0.9).all()
