"""Energy balance and round-trip efficiency (RTE) at the meter in daily, monthly and yearly windows.

Per interval the metered power is split into energy out (export, discharge) and energy in (import, charge). Where
the auxiliaries (HVAC, transformers, control) sit behind the revenue meter, the meter sees the battery's net flow;
with an auxiliary power signal the flow of the power conversion system (PCS) is ``p_meter + p_aux``. Where the
auxiliaries have their own supply (``aux_metering: separate``) their energy is added to the input for the RTE that
includes them.

RTE over a window ``w`` that does not end at the state of charge it started with is corrected for the change of
stored energy ``dE`` (MWh, from SoC x usable energy or a reported stored-energy signal), with a one-way efficiency
``eta`` between the meter and the stored energy::

    dE >= 0:  RTE = (E_out + eta * dE) / E_in             the energy left in store is worth eta * dE at the meter
    dE <  0:  RTE = E_out / (E_in + |dE| / eta)           the energy taken from store would need |dE| / eta to refill

``eta`` is either given or ``auto``: ``sqrt(RTE)`` over the whole record (excluding auxiliaries where possible),
found by fixed-point iteration. With ``dE = 0`` both lines reduce to ``E_out / E_in``.

Windows are calendar days, months and years in the site's local time zone (a local day in Europe/Warsaw has 23 or 25
hours at the DST changes) or LTSA years counted from the commercial operation date.

Missing data. The record is first put on its complete regular grid, so an interval absent from the index counts as
missing like one with a NaN. A window with a missing meter reading has an energy balance that does not close: the change
of stored energy covers the whole window, the metered energy does not. Its ratios are therefore NaN and ``valid`` is
False unless the caller accepts a lower ``min_coverage`` (default 1.0: every expected interval; the interval after
which no stored energy is known, at the end of the record, is not expected). ``coverage`` is the share of the expected
intervals with a complete reading, and the automatic one-way efficiency is found from the complete windows only.

Usage:
    flows = interval_flows(site, site_cfg)
    daily = window_rte(site, site_cfg, "D")
    monthly = window_rte(site, site_cfg, "M")
    balance = energy_balance(site, site_cfg, "M")

The interval flows, the window labels, the corrected RTE and the automatic one-way efficiency are those of the
res-asset-kpi library (``res_asset_kpi.bess``); the windows are summed here so that the NEM and ACEN studies can pass
their own flows (exact metered energy, a stored energy cleaned of BMS jumps).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from res_asset_kpi import bess
from res_asset_kpi.bess import auto_eta, corrected_rte  # noqa: F401

log = logging.getLogger(__name__)
_FREQ = {"LY": "CY"}  # LTSA years are the library's contract years


def window_keys(index: pd.DatetimeIndex, timezone: str, freq: str, cod: pd.Timestamp | None = None) -> pd.Series:
    """Label of the window every interval belongs to: D (local day), M (month), Y (year), LY (LTSA year from COD)."""
    if freq == "LY" and cod is None:
        raise ValueError("LTSA years need the commercial operation date (site.cod)")
    return bess.window_keys(index, _FREQ.get(freq, freq), timezone, cod)


def interval_flows(site: pd.DataFrame, site_cfg: dict) -> pd.DataFrame:
    """Energy per interval (MWh): meter out/in, PCS out/in (auxiliaries removed), auxiliaries, DC; stored energy."""
    return bess.interval_flows(site, site_cfg["aux_metering"], site_cfg["soc_energy_mwh"])


def _state_at(e: pd.Series, pos: np.ndarray, tolerance: int = 2) -> np.ndarray:
    """Stored energy at row positions (instants); gaps of up to ``tolerance`` samples are interpolated."""
    filled = e.interpolate(limit=tolerance, limit_area="inside").to_numpy(float)
    vals = np.full(len(pos), np.nan)
    ok = (pos >= 0) & (pos < len(filled))
    vals[ok] = filled[pos[ok]]
    return vals


def full_grid(frame: pd.DataFrame) -> pd.DataFrame:
    """The frame on its complete regular grid: intervals missing from the index become rows of NaN."""
    idx = frame.index
    if len(idx) < 2:
        return frame
    step = pd.Series(idx[1:] - idx[:-1]).mode().iloc[0]
    if step <= pd.Timedelta(0) or not idx.is_unique or not idx.is_monotonic_increasing:
        return frame
    grid = pd.date_range(idx[0], idx[-1], freq=step, name=idx.name)
    if len(grid) == len(idx):
        return frame
    out = frame.reindex(grid)
    out.attrs = dict(frame.attrs)
    return out


def _complete_rows(flows: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    """Rows in which every flow the ratios need is present (columns that do not exist at all are not needed)."""
    ok = np.ones(len(flows), dtype=bool)
    for c in columns:
        if c in flows and flows[c].notna().any():
            ok &= flows[c].notna().to_numpy()
    return ok


def window_sums(flows: pd.DataFrame, keys: pd.Series) -> pd.DataFrame:
    """Sum of the flows and the change of stored energy over each window.

    ``flows`` must be on its complete regular grid (see ``full_grid``). The change of stored energy runs from the
    instant the window starts to the instant it ends (the start of the next window). If no state is known after the
    last interval of the record, that interval is left out of the window and is not expected: ``coverage`` is the
    share of the remaining intervals that have a complete meter reading (``coverage_dc`` the same for the DC signal).
    """
    n = len(flows)
    pos = pd.Series(np.arange(n), index=flows.index)
    g = pos.groupby(keys.to_numpy(), sort=True)
    first, last = g.min(), g.max()
    end = (last + 1).to_numpy()
    stored = flows["stored"]
    e_end = _state_at(stored, end)
    shorten = np.isnan(e_end) & (end == n)
    end = np.where(shorten, end - 1, end)
    e_end = np.where(shorten, _state_at(stored, end), e_end)
    e_start = _state_at(stored, first.to_numpy())
    cum = flows.drop(columns="stored").fillna(0.0).cumsum()
    cum = pd.concat([pd.DataFrame(0.0, index=[0], columns=cum.columns), cum.reset_index(drop=True)],
                    ignore_index=True)
    sums = pd.DataFrame(cum.to_numpy()[end] - cum.to_numpy()[first.to_numpy()], index=first.index,
                        columns=cum.columns)
    starts = first.to_numpy()
    for name, columns in (("coverage", ("e_out", "e_in", "e_out_pcs", "e_in_pcs", "e_aux")),
                          ("coverage_dc", ("e_dc_dis", "e_dc_ch"))):
        done = np.r_[0, np.cumsum(_complete_rows(flows, columns))]
        expected = end - starts
        with np.errstate(divide="ignore", invalid="ignore"):
            sums[name] = np.where(expected > 0, (done[end] - done[starts]) / expected, np.nan)
    sums["hours"] = (end - first.to_numpy()) * (flows.index[1] - flows.index[0]) / pd.Timedelta("1h") \
        if n > 1 else 0.0
    sums["stored_start"] = e_start
    sums["stored_end"] = e_end
    sums["d_stored"] = e_end - e_start
    for c in ("e_aux", "e_out_pcs", "e_in_pcs", "e_dc_dis", "e_dc_ch"):
        if c in flows and flows[c].isna().all():
            sums[c] = np.nan
    sums.index.name = "window"
    return sums


def window_rte(site: pd.DataFrame, site_cfg: dict, freq: str, eta: float | str = "auto",
               flows: pd.DataFrame | None = None, min_efc: float | None = None,
               min_coverage: float = 1.0) -> pd.DataFrame:
    """Energy in/out and RTE per window (``rte`` includes auxiliaries, ``rte_excl_aux`` does not).

    Columns: e_out, e_in, e_aux, d_stored, coverage, efc (input energy of the PCS / SoC energy), rte_raw
    (E_out / E_in at the meter, uncorrected), rte, rte_excl_aux, eta (the one-way efficiency used) and valid: the
    window has at least ``min_efc`` cycles (default 0.3 a day, 5 a month), a known change of stored energy and at
    least ``min_coverage`` of its expected intervals (default 1.0, see the module docstring). On days with little
    throughput the auxiliaries and the SoC resolution dominate and the ratio means little. Below ``min_coverage``
    the ratios are NaN; ``rte_raw`` needs only the coverage.
    """
    if not 0.0 < min_coverage <= 1.0:
        raise ValueError("min_coverage must be in (0, 1]")
    flows = interval_flows(full_grid(site), site_cfg) if flows is None else full_grid(flows)
    keys = window_keys(flows.index, site_cfg["timezone"], freq, site_cfg.get("cod"))
    s = window_sums(flows, keys)
    have_pcs = s["e_in_pcs"].notna().any()
    enough = (s["coverage"] >= min_coverage - 1e-12).to_numpy()
    complete = enough & s["d_stored"].notna().to_numpy()
    if eta == "auto":
        ok = s[complete]
        if len(ok) < len(s):
            log.warning("one-way efficiency from %d of %d windows (the others lack data)", len(ok), len(s))
        cols = ("e_out_pcs", "e_in_pcs") if have_pcs else ("e_out", "e_in")
        eta = auto_eta(float(ok[cols[0]].sum()), float(ok[cols[1]].sum()), float(ok["d_stored"].sum()))
    eta = float(eta)
    mode = site_cfg["aux_metering"]
    if mode == "separate" and s["e_aux"].notna().any():
        e_in_incl = s["e_in"] + s["e_aux"]
    else:
        e_in_incl = s["e_in"]
    s["rte_raw"] = (s["e_out"] / s["e_in"].where(s["e_in"] > 0)).where(enough)
    s["rte"] = np.where(complete, corrected_rte(s["e_out"], e_in_incl, s["d_stored"], eta), np.nan)
    if have_pcs:
        s["rte_excl_aux"] = np.where(complete, corrected_rte(s["e_out_pcs"], s["e_in_pcs"], s["d_stored"], eta),
                                     np.nan)
    else:
        s["rte_excl_aux"] = np.nan
    basis = s["e_in_pcs"] if have_pcs else s["e_in"]
    s["efc"] = basis / site_cfg["soc_energy_mwh"]
    s["eta"] = eta
    need = bess.MIN_EFC.get(_FREQ.get(freq, freq), 0.0) if min_efc is None else min_efc
    s["valid"] = (s["efc"] >= need) & complete
    return s


def _hour_sum(x: pd.Series, hour: pd.DatetimeIndex) -> pd.Series:
    """Energy per hour; NaN if any interval of the hour is missing (a partial sum is not the energy of the hour)."""
    g = x.groupby(hour)
    return g.sum(min_count=1).where(g.count() == g.size())


def state_jumps(stored: pd.Series, e_in: pd.Series, e_out: pd.Series, capacity_mwh: float,
                threshold_share: float = 0.05, iterations: int = 3) -> tuple[pd.Series, pd.DataFrame, dict]:
    """Find and remove changes of the stored energy that the metered energy cannot explain.

    Battery management systems recalibrate their SoC (a jump of several percent within minutes) and reports can be
    inverted or frozen. Per hour ``h`` the change of the state is regressed on the energy charged and discharged,
    ``dE_h = a * E_in,h - b * E_out,h - c`` (efficiencies and standby drain). An hour whose residual exceeds
    ``threshold_share`` of the capacity (or six robust standard deviations) and stays exceeded over the next two
    hours is a jump; a residual that reverses within two hours is a timing offset between the power and the
    state readings and is left alone. The adjusted state has every jump subtracted from the instants after it.
    Returns the adjusted state, the jumps and the fit
    (``a``, ``b``, ``c``, ``r2`` of the hours without jumps; ``a`` and ``b`` negative = inverted report).
    """
    hour = stored.index.floor("1h")
    first = stored.groupby(hour).first()
    d_e = first.shift(-1) - first
    x_in = _hour_sum(e_in, hour)
    x_out = _hour_sum(e_out, hour)
    ok = (d_e.notna() & x_in.notna() & x_out.notna()).to_numpy()
    fit = {"a": np.nan, "b": np.nan, "c": np.nan, "r2": np.nan, "hours": int(ok.sum()), "jumps": 0}
    if ok.sum() < 24:
        return stored, pd.DataFrame(columns=["hour", "jump_mwh"]), fit
    y = d_e.to_numpy(float)
    x = np.column_stack([x_in.to_numpy(float), -x_out.to_numpy(float), -np.ones(len(y))])
    use = ok.copy()
    resid = np.zeros(len(y))
    coef = np.zeros(3)
    for _ in range(iterations):
        coef, *_ = np.linalg.lstsq(x[use], y[use], rcond=None)
        resid = np.where(ok, y - np.nan_to_num(x) @ coef, 0.0)
        mad = np.median(np.abs(resid[use] - np.median(resid[use]))) * 1.4826
        thr = max(threshold_share * capacity_mwh, 6 * mad)
        use = ok & (np.abs(resid) <= thr)
    big = ok & ~use
    r1 = np.r_[resid[1:], 0.0]
    r2 = np.r_[resid[2:], 0.0, 0.0]
    persistent = (np.abs(resid + r1) > thr) & (np.abs(resid + r1 + r2) > thr) \
        & (np.sign(resid + r1 + r2) == np.sign(resid))
    r0 = np.r_[0.0, resid[:-1]]
    reverting = (np.abs(r0) > thr) & (np.abs(r0 + resid) <= thr)    # the second half of a transient
    jumps = big & persistent & ~reverting
    ss_res = float((resid[use] ** 2).sum())
    ss_tot = float(((y[use] - y[use].mean()) ** 2).sum())
    fit.update(a=float(coef[0]), b=float(coef[1]), c=float(coef[2]),
               r2=1 - ss_res / ss_tot if ss_tot > 0 else np.nan, jumps=int(jumps.sum()))
    table = pd.DataFrame({"hour": first.index[jumps], "jump_mwh": resid[jumps]})
    step = pd.Series(np.where(jumps, resid, 0.0), index=first.index).cumsum().shift(1, fill_value=0.0)
    adjusted = stored - step.reindex(hour).to_numpy()
    return adjusted, table, fit


def inverted_days(stored: pd.Series, e_in: pd.Series, e_out: pd.Series, timezone: str,
                  min_hours: int = 6) -> pd.Series:
    """Correlation per local day between the hourly change of the stored energy and the hourly net charge.

    A healthy report correlates strongly positively; a report of the headroom (or a sign error) negatively.
    """
    hour = stored.index.floor("1h")
    first = stored.groupby(hour).first()
    d_e = first.shift(-1) - first
    net = _hour_sum(e_in, hour) - _hour_sum(e_out, hour)
    frame = pd.DataFrame({"d_e": d_e, "net": net}).dropna()
    frame = frame[frame["net"].abs() > 1e-6]
    day = frame.index.tz_convert(timezone).strftime("%Y-%m-%d")
    out = {}
    for d, g in frame.groupby(day):
        out[d] = float(np.corrcoef(g["d_e"], g["net"])[0, 1]) if len(g) >= min_hours and g["d_e"].std() > 0 \
            else np.nan
    return pd.Series(out, name="corr")


def energy_balance(site: pd.DataFrame, site_cfg: dict, freq: str = "M",
                   flows: pd.DataFrame | None = None, min_coverage: float = 1.0) -> pd.DataFrame:
    """Where the imported energy went, per window (MWh).

    ``e_in = e_out + d_stored + losses``; with an auxiliary signal behind the meter the losses split into
    ``e_aux`` and conversion losses; with a DC signal the conversion losses split further into the PCS
    (``e_in_pcs - e_dc_ch + e_dc_dis - e_out_pcs``) and the battery (``e_dc_ch - e_dc_dis - d_stored``). The
    split is an identity on the measured signals; what tests it is the battery term, which must be small and
    positive: a negative one means the SoC scale or the DC measurement does not fit the metered energy. A window
    with less than ``min_coverage`` of its intervals (the DC terms: of its DC readings) has no closing balance: its
    loss terms are NaN, the summed energies and ``coverage`` stay.
    """
    flows = interval_flows(full_grid(site), site_cfg) if flows is None else full_grid(flows)
    s = window_sums(flows, window_keys(flows.index, site_cfg["timezone"], freq, site_cfg.get("cod")))
    out = pd.DataFrame(index=s.index)
    out["e_in"] = s["e_in"]
    out["e_out"] = s["e_out"]
    out["d_stored"] = s["d_stored"]
    out["losses_total"] = s["e_in"] - s["e_out"] - s["d_stored"]
    behind = site_cfg["aux_metering"] == "behind_meter" and s["e_aux"].notna().any()
    out["e_aux"] = s["e_aux"]
    if behind:
        out["losses_conversion"] = out["losses_total"] - s["e_aux"].fillna(0)
    else:
        out["losses_conversion"] = out["losses_total"]
    if "e_dc_ch" in s and s["e_dc_ch"].notna().any() and s["e_in_pcs"].notna().any():
        out["losses_pcs"] = s["e_in_pcs"] - s["e_dc_ch"] + s["e_dc_dis"] - s["e_out_pcs"]
        out["losses_battery"] = s["e_dc_ch"] - s["e_dc_dis"] - s["d_stored"]
        out["battery_loss_share"] = out["losses_battery"] / s["e_dc_ch"].where(s["e_dc_ch"] > 0)
    gap = (s["coverage"] < min_coverage - 1e-12).to_numpy()
    loss_cols = [c for c in ("losses_total", "losses_conversion", "losses_pcs", "losses_battery",
                             "battery_loss_share") if c in out]
    out.loc[gap, loss_cols] = np.nan
    if "losses_pcs" in out:
        out.loc[(s["coverage_dc"] < min_coverage - 1e-12).to_numpy(),
                [c for c in ("losses_pcs", "losses_battery", "battery_loss_share") if c in out]] = np.nan
    out["coverage"] = s["coverage"]
    return out
