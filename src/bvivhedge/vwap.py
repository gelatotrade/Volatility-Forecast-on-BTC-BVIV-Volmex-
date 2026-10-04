"""VWAP anchors, volatility-scaled VWAP bands and the hedge-gating state machine.

All quantities at bar *t* use bars up to and including *t* (known at its close);
a position decided at the close of *t* is held over bar *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import BARS_PER_DAY
from .realized import hour_of_week, seasonal_variance_profile


def bar_vwap(bars: pd.DataFrame) -> pd.Series:
    """Exact bar VWAP from exchange klines: quote volume / base volume."""
    return bars["quote_volume"] / bars["volume"]


def rolling_vwap(bars: pd.DataFrame, window: int = BARS_PER_DAY) -> pd.Series:
    """Trailing VWAP over ``window`` bars (default 24h) -- the natural anchor for a 24/7 market."""
    qv = bars["quote_volume"].rolling(window, min_periods=1).sum()
    v = bars["volume"].rolling(window, min_periods=1).sum()
    return qv / v


def session_vwap(bars: pd.DataFrame, anchor: str = "1D") -> pd.Series:
    """Classic anchored VWAP that resets at every UTC ``anchor`` boundary."""
    key = bars.index.floor(anchor)
    return bars["quote_volume"].groupby(key).cumsum() / bars["volume"].groupby(key).cumsum()


def intraday_vol_forecast(bars: pd.DataFrame, halflife_bars: int = BARS_PER_DAY, profile_days: int = 60) -> pd.DataFrame:
    """Causal one-day-ahead volatility from 15m returns.

    De-seasonalise squared returns with a trailing hour-of-week profile, smooth
    with an EWMA and re-scale to a 24-hour horizon.  Returns ``sigma_day`` (daily
    log-return s.d.) and ``season`` (the bar's relative variance) per bar.
    """
    r = np.log(bars["close"]).diff().fillna(0.0)
    how = hour_of_week(bars.index)
    season = np.ones(len(bars))
    step = BARS_PER_DAY * 7  # re-estimate weekly from the trailing window; seasonality is slow-moving
    for i in range(step, len(bars), step):
        lo = max(0, i - profile_days * BARS_PER_DAY)
        prof = seasonal_variance_profile(r.iloc[lo:i])
        season[i : i + step] = prof.to_numpy()[how[i : i + step]]
    desea = r**2 / season
    desea.iloc[0] = np.nan                       # no return before the first bar; start the EWMA causally
    ewma = desea.ewm(halflife=halflife_bars, adjust=True, ignore_na=True).mean()
    sigma_day = np.sqrt(ewma * BARS_PER_DAY)
    return pd.DataFrame({"sigma_day": sigma_day, "season": season}, index=bars.index)


def vwap_zscore(bars: pd.DataFrame, vwap: pd.Series, sigma_day: pd.Series, window_days: float | pd.Series = 1.0) -> pd.Series:
    """Distance of the close from VWAP in units of its own standard deviation.

    For an arithmetic Brownian motion observed over a window of length w, the
    gap between the last price and the window's time average has standard
    deviation sigma * sqrt(w / 3); we use that as the band scale.
    """
    w = np.maximum(window_days, 1.0 / 24.0)
    scale = (sigma_day * np.sqrt(w / 3.0)).where(sigma_day > 0)
    return np.log(bars["close"] / vwap) / scale


def session_elapsed_days(index: pd.DatetimeIndex, anchor: str = "1D") -> pd.Series:
    elapsed = (index - index.floor(anchor)) / pd.Timedelta("1D") + 1.0 / BARS_PER_DAY
    return pd.Series(np.asarray(elapsed, float), index=index)


def gate_state(z: pd.Series | np.ndarray, enter: float = -1.0, exit: float = 0.0, min_hold: int = 4) -> np.ndarray:
    """Hysteresis gate: switch ON when z < enter, OFF when z > exit (after ``min_hold`` bars).

    The band between ``enter`` and ``exit`` suppresses whipsaw re-hedging around VWAP.
    """
    zv = np.asarray(z, float)
    state = np.zeros(len(zv), dtype=np.int8)
    on, held = 0, 0
    for t, val in enumerate(zv):
        if np.isnan(val):
            state[t] = on
            continue
        held += 1
        if not on and val < enter:
            on, held = 1, 0
        elif on and val > exit and held >= min_hold:
            on, held = 0, 0
        state[t] = on
    return state
