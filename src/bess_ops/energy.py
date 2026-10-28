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

Usage:
    flows = interval_flows(site, site_cfg)
    daily = window_rte(site, site_cfg, "D")
    monthly = window_rte(site, site_cfg, "M")
    balance = energy_balance(site, site_cfg, "M")
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bess_ops.schema import stored_energy


def window_keys(index: pd.DatetimeIndex, timezone: str, freq: str, cod: pd.Timestamp | None = None) -> pd.Series:
    """Label of the window every interval belongs to: D (local day), M (month), Y (year), LY (LTSA year from COD)."""
    loc = index.tz_convert(timezone)
    if freq in ("D", "M", "Y"):
        y, m, d = loc.year.to_numpy(), loc.month.to_numpy(), loc.day.to_numpy()
        code = {"D": y * 10000 + m * 100 + d, "M": y * 100 + m, "Y": y}[freq]
        uniq, inv = np.unique(code, return_inverse=True)
        fmt = {"D": lambda c: f"{c // 10000:04d}-{c // 100 % 100:02d}-{c % 100:02d}",
               "M": lambda c: f"{c // 100:04d}-{c % 100:02d}", "Y": lambda c: f"{c:04d}"}[freq]
        keys = np.array([fmt(int(c)) for c in uniq], dtype=object)[inv]
    elif freq == "LY":
        if cod is None:
            raise ValueError("LTSA years need the commercial operation date (site.cod)")
        keys = pd.Index(contract_year(loc, cod)).astype(str)
    else:
        raise ValueError(f"unknown window {freq!r}")
    return pd.Series(np.asarray(keys), index=index)


def contract_year(local: pd.DatetimeIndex, cod: pd.Timestamp) -> np.ndarray:
    """1-based contract year of each local timestamp (year 1 starts at COD, local midnight)."""
    cod = pd.Timestamp(cod)
    if cod.tz is None:
        cod = cod.tz_localize(local.tz)
    years = np.zeros(len(local), dtype=int)
    k = 0
    while True:
        lo = cod + pd.DateOffset(years=k)
        if lo > local.max():
            break
        hi = cod + pd.DateOffset(years=k + 1)
        years[(local >= lo) & (local < hi)] = k + 1
        k += 1
    return years


def interval_flows(site: pd.DataFrame, site_cfg: dict) -> pd.DataFrame:
    """Energy per interval (MWh): meter out/in, PCS out/in (auxiliaries removed), auxiliaries, DC; stored energy."""
    dt = site.attrs["dt_h"]
    p = site["p_meter"]
    out = pd.DataFrame(index=site.index)
    out["e_out"] = p.clip(lower=0) * dt
    out["e_in"] = (-p).clip(lower=0) * dt
    aux = site.get("p_aux")
    mode = site_cfg["aux_metering"]
    out["e_aux"] = aux * dt if aux is not None else np.nan
    if aux is not None and mode == "behind_meter":
        pcs = p + aux
    elif mode == "separate":
        pcs = p
    else:
        pcs = None
    out["e_out_pcs"] = pcs.clip(lower=0) * dt if pcs is not None else np.nan
    out["e_in_pcs"] = (-pcs).clip(lower=0) * dt if pcs is not None else np.nan
    if "p_dc" in site:
        out["e_dc_dis"] = site["p_dc"].clip(lower=0) * dt
        out["e_dc_ch"] = (-site["p_dc"]).clip(lower=0) * dt
    e = stored_energy(site, site_cfg["soc_energy_mwh"])
    out["stored"] = e if e is not None else np.nan
    return out


def _state_at(e: pd.Series, pos: np.ndarray, tolerance: int = 2) -> np.ndarray:
    """Stored energy at row positions (instants); gaps of up to ``tolerance`` samples are interpolated."""
    filled = e.interpolate(limit=tolerance, limit_area="inside").to_numpy(float)
    vals = np.full(len(pos), np.nan)
    ok = (pos >= 0) & (pos < len(filled))
    vals[ok] = filled[pos[ok]]
    return vals


def window_sums(flows: pd.DataFrame, keys: pd.Series) -> pd.DataFrame:
    """Sum of the flows and the change of stored energy over each window.

    The change of stored energy runs from the instant the window starts to the instant it ends (the start of the
    next window). If no state is known after the last interval of the record, that interval is left out.
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
    valid_p = flows["e_out"].notna().astype(float).groupby(keys.to_numpy(), sort=True).mean()
    sums["coverage"] = valid_p.to_numpy()
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


def corrected_rte(e_out, e_in, d_e, eta) -> np.ndarray:
    """RTE corrected for the change of stored energy ``d_e`` (see module docstring)."""
    e_out, e_in, d_e = (np.asarray(x, float) for x in (e_out, e_in, d_e))
    d = np.nan_to_num(d_e)
    gain = d > 0
    num = e_out + np.where(gain, eta * d, 0.0)
    den = e_in + np.where(gain, 0.0, -d / eta)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den > 0, num / den, np.nan)
    return np.where(np.isnan(d_e), np.nan, r)


def auto_eta(e_out: float, e_in: float, d_e: float, iterations: int = 50) -> float:
    """One-way efficiency ``sqrt(RTE)`` consistent with the corrected RTE of the whole record."""
    eta = 0.95
    for _ in range(iterations):
        r = float(corrected_rte(e_out, e_in, d_e, eta))
        if not np.isfinite(r) or r <= 0:
            return 0.95
        eta = min(np.sqrt(r), 1.0)
    return eta


MIN_EFC = {"D": 0.3, "M": 5.0, "Y": 50.0, "LY": 50.0}


def window_rte(site: pd.DataFrame, site_cfg: dict, freq: str, eta: float | str = "auto",
               flows: pd.DataFrame | None = None, min_efc: float | None = None,
               min_coverage: float = 0.95) -> pd.DataFrame:
    """Energy in/out and RTE per window (``rte`` includes auxiliaries, ``rte_excl_aux`` does not).

    Columns: e_out, e_in, e_aux, d_stored, coverage, efc (input energy of the PCS / SoC energy), rte_raw
    (E_out / E_in at the meter, uncorrected), rte, rte_excl_aux, eta (the one-way efficiency used) and valid: the
    window has at least ``min_efc`` cycles (default 0.3 a day, 5 a month) and ``min_coverage`` of the data. On
    days with little throughput the auxiliaries and the SoC resolution dominate and the ratio means little.
    """
    flows = interval_flows(site, site_cfg) if flows is None else flows
    keys = window_keys(site.index, site_cfg["timezone"], freq, site_cfg.get("cod"))
    s = window_sums(flows, keys)
    have_pcs = s["e_in_pcs"].notna().any()
    if eta == "auto":
        total = window_sums(flows, pd.Series("all", index=flows.index)).iloc[0]
        if have_pcs:
            eta = auto_eta(total["e_out_pcs"], total["e_in_pcs"], total["d_stored"])
        else:
            eta = auto_eta(total["e_out"], total["e_in"], total["d_stored"])
    eta = float(eta)
    mode = site_cfg["aux_metering"]
    if mode == "separate" and s["e_aux"].notna().any():
        e_in_incl = s["e_in"] + s["e_aux"]
    else:
        e_in_incl = s["e_in"]
    s["rte_raw"] = s["e_out"] / s["e_in"].where(s["e_in"] > 0)
    s["rte"] = corrected_rte(s["e_out"], e_in_incl, s["d_stored"], eta)
    if have_pcs:
        s["rte_excl_aux"] = corrected_rte(s["e_out_pcs"], s["e_in_pcs"], s["d_stored"], eta)
    else:
        s["rte_excl_aux"] = np.nan
    basis = s["e_in_pcs"] if have_pcs else s["e_in"]
    s["efc"] = basis / site_cfg["soc_energy_mwh"]
    s["eta"] = eta
    need = MIN_EFC.get(freq, 0.0) if min_efc is None else min_efc
    s["valid"] = (s["efc"] >= need) & (s["coverage"] >= min_coverage) & s["d_stored"].notna()
    return s


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
    x_in = e_in.groupby(hour).sum(min_count=1)
    x_out = e_out.groupby(hour).sum(min_count=1)
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
    net = e_in.groupby(hour).sum(min_count=1) - e_out.groupby(hour).sum(min_count=1)
    frame = pd.DataFrame({"d_e": d_e, "net": net}).dropna()
    frame = frame[frame["net"].abs() > 1e-6]
    day = frame.index.tz_convert(timezone).strftime("%Y-%m-%d")
    out = {}
    for d, g in frame.groupby(day):
        out[d] = float(np.corrcoef(g["d_e"], g["net"])[0, 1]) if len(g) >= min_hours and g["d_e"].std() > 0 \
            else np.nan
    return pd.Series(out, name="corr")


def energy_balance(site: pd.DataFrame, site_cfg: dict, freq: str = "M",
                   flows: pd.DataFrame | None = None) -> pd.DataFrame:
    """Where the imported energy went, per window (MWh).

    ``e_in = e_out + d_stored + losses``; with an auxiliary signal behind the meter the losses split into
    ``e_aux`` and conversion losses; with a DC signal the conversion losses split further into the PCS
    (``e_in_pcs - e_dc_ch + e_dc_dis - e_out_pcs``) and the battery (``e_dc_ch - e_dc_dis - d_stored``). The
    split is an identity on the measured signals; what tests it is the battery term, which must be small and
    positive: a negative one means the SoC scale or the DC measurement does not fit the metered energy.
    """
    flows = interval_flows(site, site_cfg) if flows is None else flows
    s = window_sums(flows, window_keys(site.index, site_cfg["timezone"], freq, site_cfg.get("cod")))
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
    out["coverage"] = s["coverage"]
    return out
