"""The study on live data: one panel from Binance, Volmex, Bitfinex and Deribit, and single-path inference.

Panel conventions (same schema as the simulator, so every module is shared):

* bars are indexed by their open time; ``close`` is the BTC price at open + 15 minutes;
* ``bviv`` is the official Volmex index (signals and forecasts);
* ``bviv_mark`` is the Bitfinex mark price of the BVIV perpetual (index x USD/USDt) once it is live
  (first snapshot with open interest), the Volmex index before and across status gaps of more than an hour;
  hedges are marked and, by default, filled at it;
* ``bviv_mid`` is the perpetual's quoted order-book mid (Bitfinex DERIV_PRICE).  It is not executable as such:
  dust orders often sit at the top of the book and 96% of bars see no trade.  Copy it to ``bviv_fill`` to
  price fills at it (a sensitivity);
* ``bviv_funding[t]`` is the funding a long contract pays over bar t+1 -- the Bitfinex settlement
  at 00/08/16 UTC, rate x mark, booked on the bar whose holding period ends at the settlement;
* ``bviv_carry`` equals the funding: on live data all funding is a cost of the hedge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data

BVIV_PERP = "tBVIVF0:USTF0"
BTC_PERP = "tBTCF0:USTF0"


def status_to_bars(status: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Last Bitfinex status snapshot inside each 15m bar (bar open time -> value at bar close)."""
    s = status[["mark", "deriv_price", "open_interest"]].copy()
    s.index = s.index.floor("15min")                       # snapshot in [open, open + 15m) -> that bar
    last = s.groupby(level=0).last()
    return last.reindex(index)


def funding_per_bar(events: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Funding paid per long contract, booked so that ``backtest`` charges the position held at the event.

    A settlement at time E is paid by the position held over the bar that closes at E (open E - 15m).
    ``backtest`` charges ``held_t * bviv_funding[t-1]`` for bar t, so the payment sits on the row of
    bar E - 30m.
    """
    pay = events["rate"] * events["mark"]
    pay.index = pay.index - pd.Timedelta("30min")
    return pay.groupby(level=0).sum().reindex(index).fillna(0.0)


def btc_perp_funding(events: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Annualised BTC perpetual funding rate per bar from Bitfinex settlements (8h rates).

    A settlement at E accrues over (E - 8h, E]; row t carries the rate of bar t+1, because ``backtest``
    charges ``btc_funding[t-1]`` to bar t (the convention of ``bviv_funding`` and of the simulator).
    """
    rate = events["rate"].copy()
    rate.index = rate.index - pd.Timedelta("8h") - pd.Timedelta("15min")
    return (rate.reindex(index, method="ffill").fillna(0.0) * 3 * 365)


def build_panel(start: str = "2023-01-01", end: str = "2026-10-04", perp_start: str | None = None) -> dict:
    """All live inputs on one 15m grid, plus the raw funding events for reporting."""
    klines = data.fetch_binance_klines(start[:7], end[:7], until=end)
    index = data.fetch_volmex_public(start, end)
    status = data.fetch_bitfinex_status(BVIV_PERP, "2024-04-01", end)
    events = data.bitfinex_funding_events(status)
    # the market is live from its first snapshot with open interest; earlier snapshots are placeholders
    first_live = status.index[status["open_interest"] > 0][0].floor("15min")
    if perp_start is not None:
        first_live = max(first_live, pd.Timestamp(perp_start, tz="UTC"))

    bars = klines.loc[pd.Timestamp(start, tz="UTC"): pd.Timestamp(end, tz="UTC") - pd.Timedelta("15min")].copy()  # end exclusive
    bars = bars.loc[: min(bars.index[-1], index.index[-1])]
    bars["bviv"] = index.reindex(bars.index).ffill()
    st = status_to_bars(status, bars.index).ffill(limit=4)       # never carry a snapshot for more than an hour
    live = (bars.index >= first_live) & st["mark"].notna().to_numpy()
    bars["bviv_mark"] = np.where(live, st["mark"], bars["bviv"])
    bars["bviv_mid"] = np.where(live, st["deriv_price"], bars["bviv_mark"])
    bars["open_interest"] = st["open_interest"].where(live)
    bars["bviv_funding"] = funding_per_bar(events, bars.index).where(bars.index >= first_live, 0.0)
    bars["bviv_carry"] = bars["bviv_funding"]
    try:
        btc_events = data.bitfinex_funding_events(data.fetch_bitfinex_status(BTC_PERP, "2024-04-01", end))
        bars["btc_funding"], btc_source = btc_perp_funding(btc_events, bars.index), "Bitfinex tBTCF0:USTF0"
    except Exception as err:                                     # recorded in the result, never silent
        print(f"warning: Bitfinex BTC-perp status unavailable ({err}); using Deribit funding instead")
        hourly = data.fetch_deribit_funding(start, end)
        bars["btc_funding"] = hourly.reindex(bars.index, method="ffill").fillna(0.0) * 24 * 365
        btc_source = "Deribit BTC-PERPETUAL"
    bars = bars.dropna(subset=["close", "bviv", "bviv_mark"])
    return {"bars": bars, "events": events.loc[first_live:], "status": status.loc[first_live:],
            "first_live": first_live, "perp_start": first_live.floor("1D"), "btc_funding_source": btc_source}


def funding_summary(events: pd.DataFrame, start=None, end=None) -> dict[str, float]:
    """What a long-volatility position paid: rates per 8h, share of settlements at the clamp, annualised cost."""
    e = events.loc[start:end]
    rate = e["rate"]
    cap = rate.abs().max()
    per_year = rate.mean() * 3 * 365
    return {
        "n": len(e), "mean_8h": float(rate.mean()), "median_8h": float(rate.median()),
        "share_positive": float((rate > 0).mean()), "share_at_cap": float((rate.abs() >= cap - 1e-12).mean()),
        "cap_8h": float(cap), "annual_pct_notional": float(100 * per_year),
        "annual_vol_points": float((rate * e["mark"]).mean() * 3 * 365),
    }


def stationary_bootstrap_index(n: int, block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap: geometric blocks with mean length ``block``."""
    idx = np.empty(n, dtype=int)
    t = rng.integers(n)
    for i in range(n):
        idx[i] = t
        t = rng.integers(n) if rng.random() < 1.0 / block else (t + 1) % n
    return idx


def paired_block_bootstrap(books: dict[str, pd.DataFrame], base: str, metric_fn, a: str, b: str,
                           n_boot: int = 2000, block: float = 10.0, seed: int = 0) -> dict[str, float]:
    """CI for metric(a) - metric(b) on one path, resampling the same days for every book."""
    rng = np.random.default_rng(seed)
    u, ra, rb = (books[k]["total"].to_numpy() for k in (base, a, b))
    point = metric_fn(ra, u) - metric_fn(rb, u)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        j = stationary_bootstrap_index(len(u), block, rng)
        draws[i] = metric_fn(ra[j], u[j]) - metric_fn(rb[j], u[j])
    return {"diff": float(point), "lo": float(np.quantile(draws, 0.025)), "hi": float(np.quantile(draws, 0.975)),
            "p_le0": float((draws <= 0).mean())}

