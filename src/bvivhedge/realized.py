"""Realised-variance measures built from 15-minute bars (24/7 calendar, UTC days)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import BARS_PER_DAY, DAYS_PER_YEAR

_MU1 = np.sqrt(2.0 / np.pi)  # E|Z| for a standard normal


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff().dropna()


def daily_realized(bars: pd.DataFrame) -> pd.DataFrame:
    """Daily realised measures in *daily* variance units.

    ``rv``  realised variance (sum of squared 15m log returns)
    ``bv``  bipower variation (jump-robust; Barndorff-Nielsen & Shephard)
    ``rs_neg``/``rs_pos``  realised semivariances
    ``jv``  jump variation max(rv - bv, 0)
    ``ret`` close-to-close daily log return
    """
    r = log_returns(bars["close"])
    day = r.index.floor("1D")
    absr = r.abs()
    bp = (absr * absr.shift(1)).where(day == pd.Series(day, index=r.index).shift(1))
    out = pd.DataFrame(
        {
            "rv": (r**2).groupby(day).sum(),
            "bv": bp.groupby(day).sum() / _MU1**2,
            "rs_neg": (r.clip(upper=0) ** 2).groupby(day).sum(),
            "rs_pos": (r.clip(lower=0) ** 2).groupby(day).sum(),
            "ret": r.groupby(day).sum(),
            "n": r.groupby(day).size(),
        }
    )
    out = out[out["n"] >= BARS_PER_DAY * 0.9]  # drop incomplete first/last days
    out["jv"] = (out["rv"] - out["bv"]).clip(lower=0)
    out.index.name = "date"
    return out


def annualised_vol(daily_var):
    """Daily variance -> annualised volatility in vol points (x100)."""
    return 100.0 * np.sqrt(daily_var * DAYS_PER_YEAR)


def hour_of_week(index: pd.DatetimeIndex) -> np.ndarray:
    return index.dayofweek.to_numpy() * 24 + index.hour.to_numpy()


def seasonal_variance_profile(returns: pd.Series, shrink: float = 0.5) -> pd.Series:
    """Relative 15m variance by hour-of-week (mean 1), shrunk towards the hour-of-day profile.

    Returns are winsorised at four robust standard deviations so that a few
    jumps cannot dominate a cell, and the 168-cell estimate is shrunk towards
    the hour-of-day profile; the result multiplies a de-seasonalised forecast.
    """
    how = hour_of_week(returns.index)
    scale = 1.4826 * (returns - returns.median()).abs().median()
    sq = returns.clip(-4 * scale, 4 * scale) ** 2
    hw = sq.groupby(how).mean()
    hd = sq.groupby(returns.index.hour).mean()
    hd_on_week = pd.Series(hd.to_numpy()[np.arange(168) % 24], index=np.arange(168))
    prof = (1 - shrink) * hw.reindex(range(168)).fillna(hd_on_week) + shrink * hd_on_week
    return prof / prof.mean()
