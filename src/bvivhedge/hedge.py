"""Hedge ratios, hedging rules and a bar-by-bar P&L engine.

The book is long ``qty`` BTC (spot, or a perpetual that additionally pays
funding).  The hedge is a long position of ``H`` BVIV-perpetual contracts, each
paying one dollar per index point.  A position decided at the close of bar *t*
is held over bar *t+1*; trades pay fees and slippage on traded notional.

Four rule families are compared:

* **Always-on** -- the rolling minimum-variance hedge (Johnson 1960), scaled.
* **VWAP-switch** -- the downside hedge switched on below the VWAP band and off
  above VWAP (hysteresis).  The naive use of a VWAP signal.
* **VWAP-ratchet** -- a core hedge plus a crash overlay that jumps to the
  downside hedge on a VWAP breakdown and decays slowly afterwards: fast to
  protect, slow to un-hedge, so whipsaw around VWAP costs nothing.
* **Oracle** -- the downside hedge only in the latent stress regime
  (simulation only; an infeasible upper bound for any timing signal).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from .constants import BARS_PER_DAY, DT
from .forecast import iv_to_daily_variance
from .vwap import gate_state, intraday_vol_forecast, rolling_vwap, session_elapsed_days, session_vwap, vwap_zscore


@dataclass(frozen=True)
class HedgeConfig:
    """Settings shared by all rules: book, estimators, signal, costs."""

    qty: float = 1.0                    # BTC held
    instrument: str = "spot"            # "spot" or "perp"
    beta_halflife_days: float = 14.0    # EWMA memory of the spot-vol regression
    downside_prior_days: float = 2.0    # shrinkage of the downside ratio to the full-sample one
    anchor: str = "rolling"             # "rolling" 24h VWAP or "session" (UTC day)
    rebalance_band: float = 0.25        # trade only if the target moves > 25% of the position
    fee_bps: float = 6.0                # per side, on traded BVIV-perp notional
    slippage_bps: float = 10.0
    max_notional_frac: float = 0.5      # cap: hedge notional <= 50% of BTC notional
    warmup_days: int = 30               # no hedging before estimators have data
    cheapness_lookback_days: int = 180

    def with_(self, **changes) -> "HedgeConfig":
        return replace(self, **changes)


@dataclass(frozen=True)
class Rule:
    """One hedging rule.  ``kind`` in {unhedged, always, switch, ratchet, oracle}."""

    name: str
    kind: str
    scale: float = 1.0          # multiplies the hedge size
    z_enter: float = -1.0       # breakdown threshold (VWAP z-score)
    z_exit: float = 0.0         # switch only: off above this
    min_hold: int = 4           # switch only: bars
    floor: float = 0.0          # ratchet: core hedge as a fraction of the always-on hedge
    halflife_days: float = 1.0  # ratchet: decay of the crash overlay
    use_forecast: bool = False  # scale the overlay by forecast cheapness
    size_gamma: float = 1.0
    size_bounds: tuple[float, float] = (0.5, 1.5)
    placebo_shift_days: float = 0.0  # ratchet only: circularly shift the trigger series (timing placebo)


DEFAULT_RULES = (
    Rule("Unhedged", "unhedged"),
    Rule("Always-on", "always"),
    Rule("VWAP-switch", "switch"),
    Rule("VWAP-ratchet", "ratchet", floor=0.5, halflife_days=2.0, z_enter=-1.5, use_forecast=True),
    Rule("Oracle", "oracle"),
)


# --------------------------------------------------------------------------- estimators
def ewma_beta(x: np.ndarray, y: np.ndarray, halflife_bars: float, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """EWMA regression slope of x on y through the origin, optionally on masked bars only.

    Returns the slope and the effective number of observations behind it.
    On the full sample, 15-minute means are negligible relative to their
    dispersion, so this is the usual covariance ratio.  On a masked subsample
    (falling bars) the moments are deliberately *not* demeaned: the slope
    E[x y | mask] / E[y^2 | mask] is the hedge ratio that minimises the
    downside second moment, i.e. it also offsets the mean loss on falling bars.
    """
    lam = 0.5 ** (1.0 / halflife_bars)
    m = np.ones_like(x) if mask is None else mask.astype(float)

    def smooth(u: np.ndarray) -> np.ndarray:
        return lfilter([1 - lam], [1, -lam], u)

    sxy, syy, w = smooth(m * x * y), smooth(m * y * y), smooth(m)
    beta = np.divide(sxy, syy, out=np.zeros_like(sxy), where=syy > 0)
    return beta, w / (1 - lam)


def forecast_cheapness(bars: pd.DataFrame, daily_rv_forecast: pd.Series, lookback_days: int = 180) -> pd.Series:
    """log(forecast RV / implied variance), demeaned by its trailing median, per bar.

    Positive values mean protection is cheap relative to the forecast; the
    forecast made at the close of day d drives the bars of day d+1.
    """
    iv_daily = bars["bviv"].resample("1D").last()
    ratio = np.log(daily_rv_forecast.reindex(iv_daily.index) / iv_to_daily_variance(iv_daily))
    centred = ratio - ratio.rolling(lookback_days, min_periods=30).median()
    return centred.shift(1).reindex(bars.index.floor("1D")).set_axis(bars.index).fillna(0.0)


def ratchet(trigger: np.ndarray, halflife_bars: float) -> np.ndarray:
    """Overlay intensity: 1 on a trigger bar, geometric decay otherwise."""
    decay = 0.5 ** (1.0 / halflife_bars)
    out = np.empty(len(trigger))
    level = 0.0
    for t, hit in enumerate(trigger):
        level = 1.0 if hit else level * decay
        out[t] = level
    return out


# --------------------------------------------------------------------------- signals
def build_signals(bars: pd.DataFrame, cfg: HedgeConfig, daily_rv_forecast: pd.Series | None = None) -> pd.DataFrame:
    """Everything the rules need, computed causally at each bar close."""
    r = np.log(bars["close"]).diff().fillna(0.0).to_numpy()
    d_mark = bars["bviv_mark"].diff().fillna(0.0).to_numpy()
    hl = cfg.beta_halflife_days * BARS_PER_DAY

    vol = intraday_vol_forecast(bars)
    if cfg.anchor == "session":
        vwap, window = session_vwap(bars), session_elapsed_days(bars.index)
    else:
        vwap, window = rolling_vwap(bars), 1.0
    z = vwap_zscore(bars, vwap, vol["sigma_day"], window)

    beta_all, _ = ewma_beta(r, d_mark, hl)
    beta_dn, n_dn = ewma_beta(r, d_mark, hl, mask=r < 0)   # downside second-moment hedge ratio: the crash response
    w = n_dn / (n_dn + cfg.downside_prior_days * BARS_PER_DAY)
    beta_dn = w * beta_dn + (1 - w) * beta_all

    usd_per_logret = cfg.qty * bars["close"].to_numpy()
    out = pd.DataFrame(
        {
            "vwap": vwap, "z": z, "sigma_day": vol["sigma_day"],
            "beta_all": beta_all, "beta_down": beta_dn,
            "mv_all": np.maximum(-beta_all, 0.0) * usd_per_logret,     # contracts, long-vol only
            "mv_down": np.maximum(-beta_dn, 0.0) * usd_per_logret,
            "cap": cfg.max_notional_frac * usd_per_logret / bars["bviv_mark"].to_numpy(),
        },
        index=bars.index,
    )
    out["cheapness"] = (forecast_cheapness(bars, daily_rv_forecast, cfg.cheapness_lookback_days)
                        if daily_rv_forecast is not None else 0.0)
    return out


def target_hedge(bars: pd.DataFrame, sig: pd.DataFrame, rule: Rule, cfg: HedgeConfig) -> tuple[np.ndarray, np.ndarray]:
    """Target contracts per bar and the bars where a discrete switch forces a trade."""
    n = len(bars)
    mv_all, mv_down, z = sig["mv_all"].to_numpy(), sig["mv_down"].to_numpy(), sig["z"].to_numpy()
    switch = np.zeros(n, dtype=np.int8)
    if rule.kind == "unhedged":
        h = np.zeros(n)
    elif rule.kind == "always":
        h = mv_all
    elif rule.kind == "switch":
        switch = gate_state(z, rule.z_enter, rule.z_exit, rule.min_hold)
        h = switch * mv_down
    elif rule.kind == "ratchet":
        size = 1.0
        if rule.use_forecast:
            lo, hi = rule.size_bounds
            size = np.clip(np.exp(rule.size_gamma * sig["cheapness"].to_numpy()), lo, hi)
        trigger = np.nan_to_num(z, nan=0.0) < rule.z_enter
        if rule.placebo_shift_days:
            # same number and clustering of triggers, timing scrambled: isolates the VWAP information
            trigger = np.roll(trigger, int(rule.placebo_shift_days * BARS_PER_DAY))
        overlay = ratchet(trigger, rule.halflife_days * BARS_PER_DAY)
        core = rule.floor * mv_all
        h = core + overlay * np.maximum(mv_down * size - core, 0.0)
    elif rule.kind == "oracle":
        if "regime" not in bars:
            raise ValueError("the oracle rule needs the simulated regime column")
        switch = (bars["regime"].to_numpy() == 1).astype(np.int8)
        h = switch * mv_down
    else:
        raise ValueError(f"unknown rule kind {rule.kind!r}")
    h = np.minimum(rule.scale * h, sig["cap"].to_numpy())
    h[: cfg.warmup_days * BARS_PER_DAY] = 0.0
    return h, switch


def apply_rebalance_band(target: np.ndarray, switch: np.ndarray, band: float) -> np.ndarray:
    """Hold the position until the target drifts by more than ``band`` or a switch flips."""
    pos = np.empty_like(target)
    cur, prev = 0.0, 0
    for t, tgt in enumerate(target):
        if switch[t] != prev or abs(tgt - cur) > band * max(abs(tgt), abs(cur)) or (tgt == 0.0 and cur != 0.0):
            cur = tgt
        pos[t] = cur
        prev = switch[t]
    return pos


# --------------------------------------------------------------------------- P&L engine
def backtest(bars: pd.DataFrame, position: np.ndarray, cfg: HedgeConfig) -> pd.DataFrame:
    """Bar P&L in USD of the BTC leg, the hedge leg and trading costs.

    ``position[t]`` contracts are decided at the close of bar t; they earn the
    mark change and pay the funding of bar t+1.
    """
    s = bars["close"].to_numpy()
    mark = bars["bviv_mark"].to_numpy()
    held = np.r_[0.0, position[:-1]]

    btc = cfg.qty * np.r_[0.0, np.diff(s)]
    if cfg.instrument == "perp":
        btc -= cfg.qty * np.r_[0.0, s[:-1] * bars["btc_funding"].to_numpy()[:-1] * DT]
    funding = held * np.r_[0.0, bars["bviv_funding"].to_numpy()[:-1]]
    carry = held * np.r_[0.0, bars["bviv_carry"].to_numpy()[:-1]]
    hedge = held * np.r_[0.0, np.diff(mark)] - funding
    traded = np.abs(np.diff(np.r_[0.0, position]))
    cost = traded * mark * (cfg.fee_bps + cfg.slippage_bps) / 1e4
    return pd.DataFrame(
        {"btc": btc, "hedge": hedge, "funding": funding, "carry": carry, "cost": cost, "position": position,
         "notional": position * mark, "traded_notional": traded * mark, "btc_notional": cfg.qty * s},
        index=bars.index,
    )


def run_rules(bars: pd.DataFrame, cfg: HedgeConfig, rules=DEFAULT_RULES,
              daily_rv_forecast: pd.Series | None = None, signals: pd.DataFrame | None = None
              ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Backtest every rule on one sample; the oracle is skipped on live data."""
    sig = signals if signals is not None else build_signals(bars, cfg, daily_rv_forecast)
    results = {}
    for rule in rules:
        if rule.kind == "oracle" and "regime" not in bars:
            continue
        target, switch = target_hedge(bars, sig, rule, cfg)
        results[rule.name] = backtest(bars, apply_rebalance_band(target, switch, cfg.rebalance_band), cfg)
    return results, sig
