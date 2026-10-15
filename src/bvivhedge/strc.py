"""The STRC study: Strategy's Stretch preferred stock (Nasdaq STRC) and its Hyperliquid perpetual as a short BTC hedge.

The panel extends live.py's conventions (same 15-minute BTC grid, same columns, so the hedge engine is shared):

* ``bviv`` is the official Volmex index: it drives the forecast-cheapness sizing of the frozen ratchet, as in the
  BVIV study, so the rules decide *when* to want protection exactly as before and only the instrument changes;
* ``bviv_mark`` holds the STRC price: the Nasdaq last trade at each hourly close in regular hours (venue ``cash``),
  or from the perpetual's listing on its mark every 15 minutes (venue ``perp``), carried forward in between;
* ``hedge_update`` marks the bars at which the price is observed and ``tradable`` those at which the hedge trades;
* ``bviv_funding[t]`` is what a long share or contract pays over bar t+1 (``backtest`` charges ``held_t x
  bviv_funding[t-1]``): on the cash market minus the dividend, on the bar whose interval crosses the ex-date open,
  plus a short's borrow fee written as negative long funding (the hedge is always short); on the perpetual its
  hourly funding, rate x oracle price, paid by the position held at settlement;
* ``session`` labels every bar with the US trading day whose close ends its interval: books are compared over
  trading days, so a weekend BTC move and the Monday gap of STRC fall on the same day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data
from .constants import DT

SYMBOL = "STRC"
PERP = "xyz:STRC"
ET = "America/New_York"


def hourly_closes(hourly: pd.DataFrame) -> pd.Series:
    """Yahoo hourly bars (open time) -> close, indexed by the 15-minute bar that ends where the hourly bar ends.

    Regular sessions open at 9:30 ET, so every hourly bar ends at :30 except the last, which ends at the close
    (16:00 ET, or 13:00 ET on a half day).
    """
    opens = hourly.index.tz_convert(ET)
    date = opens.normalize()
    last_open = pd.Series(opens, index=opens).groupby(date).transform("max")
    close = pd.Series(np.where(last_open.dt.hour >= 15, 16, 13), index=opens)
    session_close = date + pd.to_timedelta(close.to_numpy(), unit="h")
    hour_end = opens + pd.Timedelta("1h")
    end = hour_end.where(hour_end <= session_close, session_close).tz_convert("UTC")
    s = pd.Series(hourly["close"].to_numpy(), index=end - pd.Timedelta("15min"))
    return s[~s.index.duplicated(keep="last")].sort_index()


def session_labels(index: pd.DatetimeIndex, closes: pd.DatetimeIndex) -> pd.Series:
    """For each bar (open time), the trading day whose close is the first at or after the bar's close."""
    bar_close = index + pd.Timedelta("15min")
    pos = np.searchsorted(closes.asi8, bar_close.asi8, side="left")
    pos = np.minimum(pos, len(closes) - 1)
    label = closes[pos].tz_convert(ET).normalize().tz_localize(None)
    return pd.Series(label, index=index).where(bar_close <= closes[-1])


def dividend_per_bar(divs: pd.DataFrame, index: pd.DatetimeIndex, update: np.ndarray) -> pd.Series:
    """Minus each dividend, on the row before the first observed bar of its ex-date (the bar carrying the drop)."""
    out = pd.Series(0.0, index=index)
    obs = index[update]
    for ex_date, amount in divs["amount"].items():
        first = obs[obs >= pd.Timestamp(ex_date).tz_localize(ET).tz_convert("UTC") + pd.Timedelta("9h30min")]
        if len(first):
            row = index.get_loc(first[0]) - 1
            out.iloc[row] -= amount
    return out


def perp_funding_per_bar(rates: pd.Series, oracle: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Hourly settlement at E (rate x oracle) paid by the position held over the bar closing at E: row E - 30m."""
    pay = rates * oracle.reindex(rates.index.floor("15min"), method="ffill").to_numpy()
    pay.index = rates.index.floor("15min") - pd.Timedelta("30min")
    return pay.groupby(level=0).sum().reindex(index).fillna(0.0)


def build_panel(start: str = "2023-01-01", end: str = "2026-10-04", venue: str = "cash",
                borrow_annual: float = 0.0) -> dict:
    """All inputs of the STRC study on one 15-minute grid (``end`` exclusive)."""
    klines = data.fetch_binance_klines(start[:7], end[:7], until=end)
    index_bviv = data.fetch_volmex_public(start, end)
    hourly = data.fetch_yahoo_chart(SYMBOL, "60m", "2025-07-29", end)
    daily = data.fetch_yahoo_chart(SYMBOL, "1d", "2025-07-01", end)
    divs = data.fetch_nasdaq_dividends(SYMBOL, end)

    bars = klines.loc[pd.Timestamp(start, tz="UTC"): pd.Timestamp(end, tz="UTC") - pd.Timedelta("15min")].copy()
    bars = bars.loc[: min(bars.index[-1], index_bviv.index[-1])]
    bars["bviv"] = index_bviv.reindex(bars.index).ffill()
    cash = hourly_closes(hourly).reindex(bars.index)
    ipo = cash.first_valid_index()
    update = cash.notna().to_numpy()
    mark = cash.copy()
    tradable = update.copy()
    out = {"ipo": ipo, "dividends": divs, "daily": daily, "hourly": hourly}

    if venue == "perp":
        prices = data.parse_oxarchive_series(
            data.oxarchive_rows("prices/{symbol}", PERP, "2026-06-22", end, {"interval": "15m"}),
            ("mark_price", "oracle_price", "mid_price"))
        candles = data.parse_oxarchive_series(
            data.oxarchive_rows("candles/{symbol}", PERP, "2026-06-22", end, {"interval": "15m"}),
            ("open", "high", "low", "close", "volume", "quote_volume", "trade_count"))
        oi = data.parse_oxarchive_series(
            data.oxarchive_rows("openinterest/{symbol}", PERP, "2026-06-22", end, {"interval": "1h"}), ("open_interest",))
        rates = data.parse_hyperliquid_funding(data.fetch_hyperliquid("funding", PERP, "2026-06-22", end))
        listing = candles.index[candles["trade_count"] > 0][0]
        # a price sampled at T is the bar's value at its close: bar open T - 15m
        at_close = prices.set_axis(prices.index - pd.Timedelta("15min")).reindex(bars.index)
        live = (bars.index >= listing) & at_close["mark_price"].notna().to_numpy()
        mark = mark.where(~live, at_close["mark_price"])
        update = update & ~(bars.index >= listing) | live
        tradable = live.copy()
        bars["perp_mid"] = at_close["mid_price"].where(live)
        bars["perp_oracle"] = at_close["oracle_price"].where(live)
        bars["open_interest"] = (oi["open_interest"].set_axis(oi.index - pd.Timedelta("15min"))
                                 .reindex(bars.index, method="ffill", limit=4).where(live))
        bars["bviv_funding"] = perp_funding_per_bar(rates, prices["oracle_price"], bars.index).where(live, 0.0)
        out.update({"listing": listing, "prices": prices, "candles": candles, "oi": oi, "rates": rates})

    bars["bviv_mark"] = mark.ffill().bfill()
    bars["hedge_update"] = update
    bars["tradable"] = tradable
    if venue == "cash":
        bars["bviv_funding"] = (dividend_per_bar(divs, bars.index, update)
                                - borrow_annual * bars["bviv_mark"] * DT * (bars.index >= ipo))
    bars["bviv_carry"] = bars["bviv_funding"]
    bars["btc_funding"] = 0.0

    closes = hourly_closes(hourly).index + pd.Timedelta("15min")
    et_dates = closes.tz_convert(ET).normalize()
    session_close = pd.Series(closes, index=closes).groupby(et_dates).max()
    bars["session"] = session_labels(bars.index, pd.DatetimeIndex(session_close.to_numpy()).tz_convert("UTC"))
    out["bars"] = bars.dropna(subset=["close", "bviv"])
    return out
