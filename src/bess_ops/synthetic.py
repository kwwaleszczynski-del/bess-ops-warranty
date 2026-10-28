"""A synthetic 50 MW / 100 MWh battery with known efficiencies, auxiliaries, outages and capacity fade.

Two years of 15-minute EMS/SCADA data are simulated from a physical energy balance, exported in the generic long
format with naive Europe/Warsaw timestamps (so the DST changes are in the data), and returned with the ground
truth every estimator is checked against:

* PCS (AC/DC, incl. transformer copper losses) one-way efficiency 0.975, battery (DC/chemical) 0.98:
  round-trip efficiency without auxiliaries ``(0.975 * 0.98) ** 2 = 0.913``;
* auxiliaries behind the meter: base load + heating below 12 degC + cooling above 18 degC + 0.6 % of PCS power;
* arbitrage-like dispatch (morning and evening discharge, midday charge) with noise, EMS SoC window 5-95 %;
* forced outages (full and one of two PCS blocks), planned maintenance, grid-operator instructions and a force-
  majeure day, logged in an event list (one forced outage is deliberately missing from the log);
* capacity fade ``E(t) = E0 * (1 - a sqrt(t) - b N / 1000)`` and four capacity tests at different cell
  temperatures, the discharged energy scaled by ``1 + alpha (T - 25)``;
* injected warranty violations: a high-cycling fortnight (> 2 EFC/day), a misconfigured EMS resting at 100 % SoC,
  a PCS overload test (C-rate) and an HVAC failure (cell temperature).

Usage:
    sim = simulate(seed=1)
    sim.long.to_parquet("demo/ems_long.parquet")       # input in the generic format
    sim.events.to_csv("demo/events.csv", index=False)  # operator's log (local times)
    sim.truth                                          # dict of known values
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TZ = "Europe/Warsaw"
STEP_H = 0.25


@dataclass
class Plant:
    power_mw: float = 50.0
    pcs_rating_mw: float = 60.0
    energy_mwh: float = 100.0           # nameplate at the meter (contract)
    e0_stored_mwh: float = 107.0        # usable chemical energy at beginning of life
    eta_pcs: float = 0.975
    eta_batt: float = 0.98
    aux_base_mw: float = 0.10
    aux_heat_mw_per_k: float = 0.018
    aux_t_heat: float = 12.0
    aux_cool_mw_per_k: float = 0.030
    aux_t_cool: float = 18.0
    aux_per_mw: float = 0.006
    fade_a: float = 0.020               # per sqrt(year)
    fade_b: float = 0.020               # per 1000 EFC (meter discharge / nameplate)
    alpha: float = 0.003                # dischargeable energy per K of cell temperature
    soc_lo: float = 0.05
    soc_hi: float = 0.95


@dataclass
class Simulation:
    long: pd.DataFrame
    events: pd.DataFrame
    truth: dict
    frame: pd.DataFrame = field(repr=False)   # true per-interval values (UTC), for tests


# (start, end, available share, category or None if not logged) in local time
OUTAGES = [
    ("2025-03-03 08:00", "2025-03-03 20:00", 0.0, "forced_outage"),
    ("2025-04-15 07:00", "2025-04-15 17:00", 0.0, "planned_maintenance"),
    ("2025-05-20 02:00", "2025-05-21 14:00", 0.5, "forced_outage"),
    ("2025-06-12 10:00", "2025-06-12 16:00", 0.0, "tso_instruction"),
    ("2025-08-10 13:00", "2025-08-10 19:00", 0.0, None),
    ("2025-09-16 07:00", "2025-09-16 15:00", 0.0, "planned_maintenance"),
    ("2025-10-26 01:00", "2025-10-26 05:00", 0.0, "forced_outage"),       # spans the repeated autumn hour
    ("2025-12-03 00:00", "2025-12-04 00:00", 0.0, "force_majeure"),
    ("2026-02-09 06:00", "2026-02-10 18:00", 0.5, "forced_outage"),
    ("2026-03-29 01:00", "2026-03-29 05:00", 0.0, "forced_outage"),       # spans the spring gap
    ("2026-04-21 07:00", "2026-04-21 19:00", 0.0, "planned_maintenance"),
    ("2026-06-03 11:00", "2026-06-03 15:00", 0.0, "tso_instruction"),
    ("2026-08-18 09:00", "2026-08-28 09:00", 0.0, "forced_outage"),       # main transformer failure
    ("2026-10-07 07:00", "2026-10-07 13:00", 0.0, "planned_maintenance"),
]
# capacity tests (local date, cell temperature held by HVAC during the test)
TESTS = [("2025-01-08", 22.0), ("2026-01-12", 21.0), ("2026-07-14", 31.0), ("2026-12-09", 24.0)]
HIGH_CYCLING = ("2025-11-03", "2025-11-16")       # extra night cycle
SOC_MISCONFIG = ("2026-05-01 12:00", "2026-05-04 12:00")
OVERLOAD = ("2026-03-11 17:00", "2026-03-11 18:30", 57.0)
TWO_CYCLES_FROM = "2026-09-01"                    # a second daily cycle from here on (annual limit)
HEAT_WAVE = ("2026-07-18", "2026-07-24", 7.0)
HVAC_FAILURE = ("2026-07-21 11:00", "2026-07-22 15:00")


def _local(s: str) -> pd.Timestamp:
    return pd.Timestamp(s).tz_localize(TZ)


def ambient_temperature(local: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    doy = local.day_of_year.to_numpy()
    hour = local.hour.to_numpy() + local.minute.to_numpy() / 60
    season = -np.cos(2 * np.pi * (doy - 20) / 365.25)
    daily = 3.5 + 1.5 * season
    t = 8.5 + 10.5 * season + daily * np.cos(2 * np.pi * (hour - 15) / 24)
    # day-to-day weather: AR(1) on daily anomalies, interpolated
    days = (local.normalize() - local[0].normalize()).days.to_numpy()
    n_days = days.max() + 2
    anom = np.zeros(n_days)
    for d in range(1, n_days):
        anom[d] = 0.8 * anom[d - 1] + rng.normal(0, 1.8)
    frac = hour / 24
    t += anom[days] * (1 - frac) + anom[days + 1] * frac
    hw = (local >= _local(HEAT_WAVE[0])) & (local < _local(HEAT_WAVE[1]))
    t[hw] += HEAT_WAVE[2]
    return t + rng.normal(0, 0.3, len(t))


def schedule(local: pd.DatetimeIndex, p: Plant, rng: np.random.Generator) -> np.ndarray:
    """Desired AC power (MW, + discharge) before availability and SoC limits."""
    hour = local.hour.to_numpy() + local.minute.to_numpy() / 60
    dates = local.normalize()
    day_id, uniq = pd.factorize(dates)
    n_days = len(uniq)
    amp_c = rng.uniform(0.45, 0.65, n_days) * p.power_mw
    amp_e = rng.uniform(0.7, 1.0, n_days) * p.power_mw
    amp_m = rng.uniform(0.2, 0.5, n_days) * p.power_mw
    weekend = np.asarray(uniq.day_of_week >= 5)
    amp_e[weekend] *= 0.75
    cmd = np.zeros(len(local))
    cmd[(hour >= 6) & (hour < 8)] = amp_m[day_id][(hour >= 6) & (hour < 8)]
    charge = (hour >= 10) & (hour < 15)
    cmd[charge] = -amp_c[day_id][charge]
    ev = (hour >= 17) & (hour < 21.5)
    cmd[ev] = amp_e[day_id][ev]
    night = (hour >= 0) & (hour < 5)
    hc = (dates >= _local(HIGH_CYCLING[0])) & (dates <= _local(HIGH_CYCLING[1]))
    hc_night = np.asarray(hc) & night
    cmd[hc_night & (hour < 2.5)] = -p.power_mw
    cmd[hc_night & (hour >= 2.5)] = p.power_mw
    cmd[np.asarray(hc) & (hour >= 5) & (hour < 6)] = -p.power_mw      # top up for the morning peak
    hc_mid = np.asarray(hc) & (hour >= 15) & (hour < 17)
    cmd[hc_mid] = -p.power_mw           # refill for the evening
    two = np.asarray(dates >= _local(TWO_CYCLES_FROM))
    cmd[two & (hour >= 1) & (hour < 4)] = -0.6 * p.power_mw
    cmd[two & (hour >= 6) & (hour < 9)] = 0.6 * p.power_mw
    noise = rng.normal(0, 1.5, len(cmd))  # frequency-response-like activity
    return cmd + noise * (np.abs(cmd) < 1)


def simulate(seed: int = 1, start: str = "2025-01-01", end: str = "2027-01-01",
             plant: Plant | None = None) -> Simulation:
    p = plant or Plant()
    rng = np.random.default_rng(seed)
    utc = pd.date_range(_local(start).tz_convert("UTC"), _local(end).tz_convert("UTC"), freq="15min",
                        inclusive="left", tz="UTC")
    local = utc.tz_convert(TZ)
    n = len(utc)
    dt = STEP_H
    t_amb = ambient_temperature(local, rng)
    cmd = schedule(local, p, rng)

    avail = np.ones(n)
    for s, e, share, _ in OUTAGES:
        avail[(local >= _local(s)) & (local < _local(e))] = share
    soc_hi = np.full(n, p.soc_hi)
    soc_lo = np.full(n, p.soc_lo)
    mis = (local >= _local(SOC_MISCONFIG[0])) & (local < _local(SOC_MISCONFIG[1]))
    soc_hi[mis] = 1.0
    cmd[mis & (cmd > 0)] = 0.0                                   # nothing discharges: it rests full
    cmd[mis & (cmd <= 0)] = -0.3 * p.power_mw
    ov = (local >= _local(OVERLOAD[0])) & (local < _local(OVERLOAD[1]))
    cmd[ov] = OVERLOAD[2]

    # capacity tests: discharge to 0, charge to 100 %, rest 1 h, discharge at 50 MW (meter) to 0, recharge to 50 %
    # phases: 0 normal, 1 pre-discharge, 2 charge, 3 rest, 4 test, 5 recharge
    test_temp = np.full(n, np.nan)
    test_days = []
    for d, temp in TESTS:
        day = (local.normalize() == _local(d))
        test_days.append((d, temp, np.flatnonzero(day)))
        test_temp[day] = temp

    fade_t = ((local - local[0]) / pd.Timedelta(days=365.25)).to_numpy()
    stored = np.zeros(n + 1)
    stored[0] = 0.5 * p.e0_stored_mwh
    p_ac = np.zeros(n)
    p_aux = np.zeros(n)
    p_dc = np.zeros(n)
    scale = np.zeros(n)
    efc_cum = 0.0
    tcell = np.zeros(n)
    tc = 24.0
    hvac_fail = (local >= _local(HVAC_FAILURE[0])) & (local < _local(HVAC_FAILURE[1]))
    test_phase = {}
    for d, temp, idx in test_days:
        for i in idx:
            test_phase[i] = d
    phase = 0
    test_energy = {}
    rest_left = 0
    for i in range(n):
        e_use = p.e0_stored_mwh * (1 - p.fade_a * np.sqrt(fade_t[i]) - p.fade_b * efc_cum / 1000)
        in_test = i in test_phase
        if in_test and phase == 0:
            phase = 1 if stored[i] > 1e-9 else 2
        elif not in_test:
            phase = 0
        # the test scale (temperature effect) applies between the two empty points of the test
        sc = e_use * (1 + p.alpha * (test_temp[i] - 25.0)) if phase in (2, 3, 4) else e_use
        scale[i] = sc
        aux_t = (p.aux_base_mw + p.aux_heat_mw_per_k * max(p.aux_t_heat - t_amb[i], 0)
                 + p.aux_cool_mw_per_k * max(t_amb[i] - p.aux_t_cool, 0))
        if phase in (1,):
            c = 0.5 * p.power_mw
        elif phase in (2, 5):
            c = -0.5 * p.power_mw
        elif phase == 3:
            c = 0.0
        elif phase == 4:
            c = (p.power_mw + aux_t) / (1 - p.aux_per_mw)       # 50 MW at the meter
        else:
            c = cmd[i]
            a = avail[i]
            upper = a * (OVERLOAD[2] if ov[i] else p.power_mw)
            c = float(np.clip(c, -a * p.power_mw, upper))
        lo = 0.0 if phase in (1, 2, 3, 4) else soc_lo[i]
        hi = 1.0 if phase in (1, 2, 3, 4) else (0.5 if phase == 5 else soc_hi[i])
        if c > 0:
            room = max(stored[i] - lo * sc, 0.0)
            c = min(c, room * p.eta_batt * p.eta_pcs / dt)
            dc = c / p.eta_pcs
            stored[i + 1] = stored[i] - dc / p.eta_batt * dt
        elif c < 0:
            room = max(hi * sc - stored[i], 0.0)
            c = max(c, -room / p.eta_batt / p.eta_pcs / dt)
            dc = c * p.eta_pcs
            stored[i + 1] = stored[i] - dc * p.eta_batt * dt
        else:
            dc = 0.0
            stored[i + 1] = stored[i]
        p_ac[i], p_dc[i] = c, dc
        # thermal: HVAC holds about 24 degC, the battery warms with load; fails on HVAC_FAILURE
        target = 23.0 + 0.12 * (t_amb[i] - 15) + 3.0 * abs(c) / p.power_mw
        if hvac_fail[i]:
            target = t_amb[i] + 12.0
        if in_test:
            target = test_temp[i]
        tc += (target - tc) * (1 - np.exp(-dt / (0.5 if in_test else 2.0)))
        tcell[i] = tc
        p_aux[i] = aux_t + p.aux_per_mw * abs(c)
        meter = c - p_aux[i]
        if meter > 0:
            efc_cum += meter * dt / p.energy_mwh
        if not in_test:
            continue
        # test state machine (transitions at the end of the interval)
        d = test_phase[i]
        if phase == 1 and stored[i + 1] <= 1e-9:
            phase = 2
        elif phase == 2 and stored[i + 1] >= sc * (1 - 1e-9):
            phase, rest_left = 3, 4
        elif phase == 3:
            rest_left -= 1
            if rest_left == 0:
                phase = 4
                test_energy[d] = {"start": i + 1, "energy": 0.0, "temp": test_temp[i]}
        elif phase == 4:
            if meter > 0:
                test_energy[d]["energy"] += meter * dt
                test_energy[d]["end"] = i + 1
            if stored[i + 1] <= 1e-9:
                phase = 5
                test_energy[d].update(e_use=e_use, efc=efc_cum, years=fade_t[i])

    p_meter = p_ac - p_aux
    soc_true = stored[:-1] / scale
    t_cell_avg = tcell + rng.normal(0, 0.15, n)
    t_cell_max = tcell + 2.0 + 1.5 * np.abs(p_ac) / p.power_mw + rng.normal(0, 0.15, n)
    frame = pd.DataFrame({
        "p_meter": p_meter, "p_ac": p_ac, "p_dc": p_dc, "p_aux": p_aux, "stored": stored[:-1],
        "stored_next": stored[1:], "scale": scale, "soc": soc_true, "t_amb": t_amb, "t_cell_avg": t_cell_avg,
        "t_cell_max": t_cell_max, "avail": avail,
    }, index=utc)
    long = to_long(frame, p, rng)
    events = event_log(p)
    truth = {
        "eta_pcs": p.eta_pcs, "eta_batt": p.eta_batt, "rte_excl_aux": (p.eta_pcs * p.eta_batt) ** 2,
        "one_way": p.eta_pcs * p.eta_batt,
        "aux": {"b0_mw": p.aux_base_mw, "heat_mw_per_k": p.aux_heat_mw_per_k, "t_heat_c": p.aux_t_heat,
                "cool_mw_per_k": p.aux_cool_mw_per_k, "t_cool_c": p.aux_t_cool, "mw_per_mw": p.aux_per_mw},
        "fade_a": p.fade_a, "fade_b": p.fade_b, "e0_stored_mwh": p.e0_stored_mwh, "alpha": p.alpha,
        "tests": [{"date": d, "temp_c": v["temp"], "e_meter_mwh": v["energy"],
                   "e_meter_25c_mwh": v["energy"] / (1 + p.alpha * (v["temp"] - 25.0)),
                   "e_use_mwh": v["e_use"], "efc_total": v["efc"], "years": v["years"]}
                  for d, v in test_energy.items()],
        "outages": OUTAGES, "high_cycling": HIGH_CYCLING, "soc_misconfig": SOC_MISCONFIG,
        "overload": OVERLOAD, "hvac_failure": HVAC_FAILURE,
    }
    return Simulation(long=long, events=events, truth=truth, frame=frame)


def to_long(frame: pd.DataFrame, p: Plant, rng: np.random.Generator) -> pd.DataFrame:
    """Export like an EMS would: kW and %, two PCS/BMS blocks, naive local timestamps in chronological order."""
    n = len(frame)
    ts = frame.index.tz_convert(TZ).tz_localize(None)
    soc_meas = np.round(np.clip(frame["soc"].to_numpy() + rng.normal(0, 0.001, n), 0, 1) * 100, 1)
    half = frame["avail"].to_numpy()
    blk1 = np.where(half >= 1.0, 1.0, np.where(half >= 0.5, 1.0, 0.0))
    blk2 = np.where(half >= 1.0, 1.0, 0.0)
    e_avail = (frame["scale"] * p.eta_pcs * p.eta_batt * frame["avail"]).to_numpy()
    signals = {
        ("METER-01", "P_POI_kW"): frame["p_meter"].to_numpy() * 1000 * (1 + rng.normal(0, 0.0005, n)),
        ("PCS-01", "P_DC_kW"): frame["p_dc"].to_numpy() * 500,
        ("PCS-02", "P_DC_kW"): frame["p_dc"].to_numpy() * 500,
        ("PCS-01", "P_AVAIL_DIS_kW"): blk1 * p.power_mw * 500, ("PCS-02", "P_AVAIL_DIS_kW"): blk2 * p.power_mw * 500,
        ("PCS-01", "P_AVAIL_CH_kW"): blk1 * p.power_mw * 500, ("PCS-02", "P_AVAIL_CH_kW"): blk2 * p.power_mw * 500,
        ("BMS-01", "SOC_PCT"): soc_meas, ("BMS-02", "SOC_PCT"): soc_meas,
        ("BMS-01", "T_CELL_MAX_C"): frame["t_cell_max"].to_numpy(),
        ("BMS-02", "T_CELL_MAX_C"): frame["t_cell_max"].to_numpy() - 0.4,
        ("BMS-01", "T_CELL_AVG_C"): frame["t_cell_avg"].to_numpy(),
        ("BMS-02", "T_CELL_AVG_C"): frame["t_cell_avg"].to_numpy(),
        ("AUX-01", "P_AUX_kW"): frame["p_aux"].to_numpy() * 1000,
        ("MET-01", "T_AMB_C"): frame["t_amb"].to_numpy(),
        ("EMS-01", "E_AVAIL_kWh"): e_avail * 1000,
    }
    parts = [pd.DataFrame({"timestamp": ts, "device_id": dev, "signal": sig, "value": np.round(val, 3)})
             for (dev, sig), val in signals.items()]
    long = pd.concat(parts, ignore_index=True)
    order = np.argsort(np.tile(np.arange(n), len(signals)), kind="stable")
    return long.iloc[order].reset_index(drop=True)


def event_log(p: Plant) -> pd.DataFrame:
    rows = [{"start": s, "end": e, "category": c, "note": f"available {share:.0%}"}
            for s, e, share, c in OUTAGES if c is not None]
    for d, _ in TESTS:
        rows.append({"start": f"{d} 00:00", "end": (pd.Timestamp(d) + pd.Timedelta(days=1)).strftime("%Y-%m-%d %H:%M"),
                     "category": "capacity_test", "note": "annual capacity test"})
    return pd.DataFrame(rows).sort_values("start").reset_index(drop=True)
