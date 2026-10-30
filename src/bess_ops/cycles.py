"""Equivalent full cycles, energy throughput and rainflow cycle counting of the state of charge.

* Equivalent full cycles (EFC) on the basis a warranty names: discharged energy at the meter or at the DC terminals,
  or half the metered throughput (charge + discharge) / 2, divided by the reference energy.
* Rainflow counting (ASTM E1049-85, three-point method, residue counted as half cycles) of the SoC, with an optional
  hysteresis filter against measurement noise. ``sum(depth * count)`` is the rainflow EFC.

Usage:
    efc = throughput(site, warranty, freq_keys)            # MWh and EFC per window
    cycles = rainflow(soc.to_numpy(), min_range=0.01)      # depth, mean, count, start, end
    hist = depth_histogram(cycles, years=1.0)

All of it is computed by the res-asset-kpi library (``res_asset_kpi.bess``).
"""
from __future__ import annotations

import pandas as pd
from res_asset_kpi import bess
from res_asset_kpi.bess import DEPTH_EDGES, depth_histogram, rainflow, rainflow_efc, reversals  # noqa: F401


def throughput_energy(site: pd.DataFrame, basis: str) -> pd.Series:
    """Energy (MWh) per interval that the warranty counts as throughput."""
    return bess.throughput(site, site.attrs["dt_h"], basis)
