"""The STRC study: Strategy's Stretch preferred stock (Nasdaq STRC) and its Hyperliquid perpetual as a short BTC hedge.

The panel extends live.py's conventions (same 15-minute BTC grid, same columns, so the hedge engine is shared):

* ``bviv`` is the official Volmex index: it drives the forecast-cheapness sizing of the frozen ratchet, as in the
  BVIV study, so the rules decide *when* to want protection exactly as before and only the instrument changes;
* ``bviv_mark`` holds the STRC price, carried forward between observations.  On the cash market (venue ``cash``)
  STRC is observed at the official open (9:30 ET), at each hourly close and at the official close of every
  session; the session calendar comes from the daily bars and the exchange's early closes, never from the hourly
  bars a source happens to return, and the official prints fill any hole in the hourly history.  From the
  perpetual's listing on (venue ``perp``) the price is its mark every 15 minutes: 0xArchive stamps each value at
  the start of the 15 minutes it averages, so it is the value of the bar that opens at that stamp;
* ``hedge_update`` marks the bars at which the price is observed and ``tradable`` those at which the hedge trades;
* ``bviv_funding[t]`` is what a long share or contract pays over bar t+1 (``backtest`` charges ``held_t x
  bviv_funding[t-1]``): on the cash market minus the dividend, on the row before the first observation of the
  ex-date, plus a short's borrow fee written as negative long funding (the hedge is always short); on the
  perpetual its hourly funding, rate x oracle price, paid by the position held at settlement;
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
#: Nasdaq early closes (13:00 ET) from STRC's listing to the end of 2026.
EARLY_CLOSES = ("2025-11-28", "2025-12-24", "2026-11-27", "2026-12-24")
BAR = pd.Timedelta("15min")


def sessions(daily: pd.DataFrame) -> pd.DataFrame:
    """One row per trading day of the daily bars: open and close times (UTC) and the official open and close."""
    date = daily.index.tz_convert(ET).normalize().tz_localize(None)
    early = date.isin(pd.DatetimeIndex(EARLY_CLOSES))
    local = pd.Series(date, index=date)
    open_t = (local + pd.Timedelta("9h30min")).dt.tz_localize(ET).dt.tz_convert("UTC")
    close_t = (local + pd.to_timedelta(np.where(early, 13, 16), unit="h")).dt.tz_localize(ET).dt.tz_convert("UTC")
    return pd.DataFrame({"open_time": open_t, "close_time": close_t,
                         "open": pd.Series(daily["open"].to_numpy(), index=date),
                         "close": pd.Series(daily["close"].to_numpy(), index=date)})


def cash_observations(hourly: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
    """STRC prices by observation time (UTC): official opens and closes, and the hourly closes in between.

    An hourly bar opening at t ends at t + 1h or at its session's close, whichever is first; bars outside the
    sessions of the daily data are dropped, and the official close replaces an hourly print at the close.
    """
    s = sessions(daily)
    opens = hourly.index.tz_convert(ET).normalize().tz_localize(None)
    inside = opens.isin(s.index)
    h = hourly[inside]
    hour_end = h.index + pd.Timedelta("1h")
    close_t = pd.DatetimeIndex(s["close_time"].reindex(opens[inside]))
    hourly_obs = pd.Series(h["close"].to_numpy(), index=hour_end.where(hour_end <= close_t, close_t))
    official = pd.concat([pd.Series(s["open"].to_numpy(), index=pd.DatetimeIndex(s["open_time"])),
                          pd.Series(s["close"].to_numpy(), index=pd.DatetimeIndex(s["close_time"]))])
    obs = pd.concat([hourly_obs[~hourly_obs.index.isin(official.index)], official]).sort_index()
    return obs[~obs.index.duplicated(keep="last")]


def hourly_gaps(hourly: pd.DataFrame, daily: pd.DataFrame) -> list[str]:
    """Sessions whose hourly history is incomplete (fewer bars than the session has hours)."""
    s = sessions(daily)
    count = pd.Series(1, index=hourly.index.tz_convert(ET).normalize().tz_localize(None)).groupby(level=0).sum()
    expected = np.ceil((s["close_time"] - s["open_time"]) / pd.Timedelta("1h")).astype(int)
    have = count.reindex(s.index).fillna(0).astype(int)
    return [str(d.date()) for d in s.index[have < expected]]


def session_labels(index: pd.DatetimeIndex, closes: pd.DatetimeIndex) -> pd.Series:
    """For each bar (open time), the trading day whose close is the first at or after the bar's close."""
    bar_close = index + BAR
    pos = np.searchsorted(closes.as_unit("ns").asi8, bar_close.as_unit("ns").asi8, side="left")
    pos = np.minimum(pos, len(closes) - 1)
    label = closes[pos].tz_convert(ET).normalize().tz_localize(None)
    return pd.Series(label, index=index).where(bar_close <= closes[-1])


def dividend_per_bar(divs: pd.DataFrame, index: pd.DatetimeIndex, update: np.ndarray) -> pd.Series:
    """Minus each dividend, on the row before the first observed bar of its ex-date (the bar carrying the drop)."""
    out = pd.Series(0.0, index=index)
    obs = index[update]
    for ex_date, amount in divs["amount"].items():
        first = obs[obs + BAR >= pd.Timestamp(ex_date).tz_localize(ET).tz_convert("UTC") + pd.Timedelta("9h30min")]
        if len(first):
            row = index.get_loc(first[0]) - 1
            out.iloc[row] -= amount
    return out


def perp_funding_per_bar(rates: pd.Series, oracle: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Hourly settlement at E (rate x oracle) paid by the position held over the bar closing at E: row E - 30m.

    ``oracle`` is stamped at the start of each 15-minute average, so the value just before E is the row E - 15m.
    """
    at = rates.index.floor("15min")
    pay = rates * oracle.reindex(at - BAR, method="ffill").to_numpy()
    pay.index = at - 2 * BAR
    return pay.groupby(level=0).sum().reindex(index).fillna(0.0)


def build_panel(start: str = "2023-01-01", end: str = "2026-10-04", venue: str = "cash",
                borrow_annual: float = 0.0) -> dict:
    """All inputs of the STRC study on one 15-minute grid (``end`` exclusive)."""
    klines = data.fetch_binance_klines(start[:7], end[:7], until=end)
    index_bviv = data.fetch_volmex_public(start, end)
    hourly = data.fetch_yahoo_chart(SYMBOL, "60m", "2025-07-29", end)
    daily = data.fetch_yahoo_chart(SYMBOL, "1d", "2025-07-01", end)
    divs = data.fetch_nasdaq_dividends(SYMBOL, end)

    bars = klines.loc[pd.Timestamp(start, tz="UTC"): pd.Timestamp(end, tz="UTC") - BAR].copy()
    bars = bars.loc[: min(bars.index[-1], index_bviv.index[-1])]
    bars["bviv"] = index_bviv.reindex(bars.index).ffill()
    obs = cash_observations(hourly, daily)
    cash = pd.Series(obs.to_numpy(), index=obs.index - BAR).reindex(bars.index)   # the bar that closes at the print
    ipo = cash.first_valid_index()
    update = cash.notna().to_numpy()
    mark = cash.copy()
    tradable = update.copy()
    cal = sessions(daily)
    out = {"ipo": ipo, "dividends": divs, "daily": daily, "hourly": hourly, "sessions": cal,
           "observations": obs, "hourly_gaps": hourly_gaps(hourly, daily)}

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
        in_bar = prices.reindex(bars.index)                       # the average over [T, T + 15m): bar T's mark
        after = bars.index >= listing
        live = after & in_bar["mark_price"].notna().to_numpy()
        # from the listing on the hedge is priced by the perpetual alone (its last mark across a missing sample)
        mark = mark.where(~after, in_bar["mark_price"].where(after).ffill())
        update = update & ~after | live
        tradable = live.copy()
        bars["perp_mid"] = in_bar["mid_price"].where(live)
        bars["perp_oracle"] = in_bar["oracle_price"].where(live)
        # an hourly average over [T, T + 1h) is known at T + 1h: the bar opening at T + 45m
        bars["open_interest"] = (oi["open_interest"].set_axis(oi.index + pd.Timedelta("45min"))
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
    bars["session"] = session_labels(bars.index, pd.DatetimeIndex(cal["close_time"]))
    out["bars"] = bars.dropna(subset=["close", "bviv"])
    return out
