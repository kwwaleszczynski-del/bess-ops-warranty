"""Equivalent full cycles, energy throughput and rainflow cycle counting of the state of charge.

* Equivalent full cycles (EFC) on the basis a warranty names: discharged energy at the meter or at the DC terminals,
  or half the metered throughput (charge + discharge) / 2, divided by the reference energy.
* Rainflow counting (ASTM E1049-85, three-point method, residue counted as half cycles) of the SoC, with an optional
  hysteresis filter against measurement noise. ``sum(depth * count)`` is the rainflow EFC.

Usage:
    efc = throughput(site, warranty, freq_keys)            # MWh and EFC per window
    cycles = rainflow(soc.to_numpy(), min_range=0.01)      # depth, mean, count, start, end
    hist = depth_histogram(cycles, years=1.0)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def throughput_energy(site: pd.DataFrame, basis: str) -> pd.Series:
    """Energy (MWh) per interval that the warranty counts as throughput."""
    dt = site.attrs["dt_h"]
    if basis == "discharge_meter":
        return site["p_meter"].clip(lower=0) * dt
    if basis == "discharge_dc":
        if "p_dc" not in site:
            raise ValueError("throughput basis discharge_dc needs the signal p_dc")
        return site["p_dc"].clip(lower=0) * dt
    if basis == "half_throughput_meter":
        return site["p_meter"].abs() * dt / 2
    raise ValueError(f"unknown throughput basis {basis!r}")


def reversals(x: np.ndarray, min_range: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Positions and values of the turning points (first and last point included, NaNs and plateaus dropped).

    With ``min_range`` > 0 a turning point only counts once the signal has moved back by more than that much.
    """
    x = np.asarray(x, float)
    pos = np.flatnonzero(~np.isnan(x))
    v = x[pos]
    if len(v) == 0:
        return pos, v
    keep = np.r_[True, np.diff(v) != 0]
    pos, v = pos[keep], v[keep]
    if len(v) < 3:
        return pos, v
    if min_range <= 0:
        d = np.sign(np.diff(v))
        idx = np.r_[0, np.flatnonzero(d[1:] != d[:-1]) + 1, len(v) - 1]
        return pos[idx], v[idx]
    out = [0]
    direction, cand, hi, lo = 0, 0, 0, 0
    for i in range(1, len(v)):
        if direction == 0:
            hi = i if v[i] > v[hi] else hi
            lo = i if v[i] < v[lo] else lo
            if v[hi] - v[0] > min_range:
                direction, cand = 1, hi
            elif v[0] - v[lo] > min_range:
                direction, cand = -1, lo
        elif direction > 0:
            if v[i] >= v[cand]:
                cand = i
            elif v[cand] - v[i] > min_range:
                out.append(cand)
                direction, cand = -1, i
        else:
            if v[i] <= v[cand]:
                cand = i
            elif v[i] - v[cand] > min_range:
                out.append(cand)
                direction, cand = 1, i
    if direction != 0:
        out.append(cand)
    if out[-1] != len(v) - 1:
        out.append(len(v) - 1)
    idx = np.asarray(out)
    return pos[idx], v[idx]


def rainflow(x, min_range: float = 0.0) -> pd.DataFrame:
    """Rainflow cycles of a signal: depth (range), mean, count (1 or 0.5), start and end position."""
    pos, v = reversals(x, min_range)
    stack_v: list[float] = []
    stack_p: list[int] = []
    rows: list[tuple] = []
    for p, val in zip(pos, v):
        stack_v.append(val)
        stack_p.append(p)
        while len(stack_v) >= 3:
            x_rng = abs(stack_v[-1] - stack_v[-2])
            y_rng = abs(stack_v[-2] - stack_v[-3])
            if x_rng < y_rng:
                break
            if len(stack_v) == 3:
                rows.append((y_rng, (stack_v[0] + stack_v[1]) / 2, 0.5, stack_p[0], stack_p[1]))
                del stack_v[0], stack_p[0]
            else:
                rows.append((y_rng, (stack_v[-2] + stack_v[-3]) / 2, 1.0, stack_p[-3], stack_p[-2]))
                del stack_v[-3:-1], stack_p[-3:-1]
    for i in range(len(stack_v) - 1):
        rows.append((abs(stack_v[i + 1] - stack_v[i]), (stack_v[i + 1] + stack_v[i]) / 2, 0.5, stack_p[i],
                     stack_p[i + 1]))
    return pd.DataFrame(rows, columns=["depth", "mean", "count", "start", "end"])


DEPTH_EDGES = (0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0001)


def depth_histogram(cycles: pd.DataFrame, years: float = 1.0) -> pd.DataFrame:
    """Cycles and their EFC (depth x count) per depth bin, per year."""
    edges = np.asarray(DEPTH_EDGES)
    cut = pd.cut(cycles["depth"], edges, right=False, include_lowest=True)
    h = pd.DataFrame({"cycles": cycles["count"], "efc": cycles["count"] * cycles["depth"]}) \
        .groupby(cut, observed=False).sum() / max(years, 1e-9)
    h.index = [f"{a:.0%}-{min(b, 1):.0%}" for a, b in zip(edges[:-1], edges[1:])]
    return h


def rainflow_efc(cycles: pd.DataFrame) -> float:
    return float((cycles["depth"] * cycles["count"]).sum())
