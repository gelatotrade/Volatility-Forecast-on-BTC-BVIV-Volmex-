"""Daily risk and cost metrics for hedged BTC books."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import DAYS_PER_YEAR


def daily_book(pnl: pd.DataFrame, warmup_days: int = 0) -> pd.DataFrame:
    """Aggregate bar P&L to UTC days, expressed as returns on the BTC notional at the prior close."""
    day = pnl.index.floor("1D")
    d = pnl[["btc", "hedge", "funding", "carry", "cost", "traded_notional"]].groupby(day).sum()
    d["notional"] = pnl["notional"].groupby(day).mean()
    d["time_on"] = (pnl["notional"] > 1e-3 * pnl["btc_notional"]).groupby(day).mean()
    prior = pnl["btc_notional"].shift(1).fillna(pnl["btc_notional"].iloc[0])
    capital = prior.groupby(day).first()  # notional at the prior close
    d = d.div(capital, axis=0).assign(time_on=d["time_on"])
    d["total"] = d["btc"] + d["hedge"] - d["cost"]
    return d.iloc[warmup_days:]


def expected_shortfall(x: np.ndarray, level: float = 0.975) -> float:
    q = np.quantile(x, 1 - level)
    return float(-x[x <= q].mean())


def max_drawdown(r: np.ndarray) -> float:
    eq = np.cumprod(1 + r)
    return float(-(eq / np.maximum.accumulate(eq) - 1).min())


def book_metrics(book: pd.DataFrame, base: pd.DataFrame, tail_q: float = 0.05) -> dict[str, float]:
    """Metrics of one hedged book relative to the unhedged ``base`` (same days)."""
    r, u = book["total"].to_numpy(), base["total"].to_numpy()
    worst = u <= np.quantile(u, tail_q)
    hedge_net = (book["hedge"] - book["cost"]).to_numpy()
    tail_loss = book["btc"].to_numpy()[worst].sum()
    return {
        "vol": 100 * r.std(ddof=1) * np.sqrt(DAYS_PER_YEAR),
        "var_red": 100 * (1 - r.var(ddof=1) / u.var(ddof=1)),
        "es": 100 * expected_shortfall(r),
        "es_red": 100 * (1 - expected_shortfall(r) / expected_shortfall(u)),
        "mdd": 100 * max_drawdown(r),
        "worst": -100 * r.min(),
        "ret": 100 * r.mean() * DAYS_PER_YEAR,
        "hedge_pnl": 100 * book["hedge"].mean() * DAYS_PER_YEAR,
        "funding": 100 * book["funding"].mean() * DAYS_PER_YEAR,
        "cost": 100 * book["cost"].mean() * DAYS_PER_YEAR,
        "carry": 100 * book["carry"].mean() * DAYS_PER_YEAR,
        "hedge_cost": 100 * (book["carry"] + book["cost"]).mean() * DAYS_PER_YEAR,
        "turnover": book["traded_notional"].mean() * DAYS_PER_YEAR,
        "notional": 100 * book["notional"].mean(),
        "time_on": 100 * book["time_on"].mean(),
        "tail_offset": 100 * (-hedge_net[worst].sum() / tail_loss) if tail_loss < 0 else np.nan,
    }


METRIC_LABELS = {
    "vol": "Volatility (% p.a.)",
    "var_red": "Variance reduction (%)",
    "es": "ES$_{97.5}$ daily (%)",
    "es_red": "ES reduction (%)",
    "mdd": "Max drawdown (%)",
    "worst": "Worst day (%)",
    "ret": "Return (% p.a.)",
    "hedge_pnl": "Hedge-leg P\\&L (% p.a.)",
    "funding": "Funding paid (% p.a.)",
    "carry": "Carry premium (% p.a.)",
    "cost": "Trading cost (% p.a.)",
    "hedge_cost": "Hedge cost: carry + trading (% p.a.)",
    "turnover": "Turnover (x p.a.)",
    "notional": "Avg. hedge notional (%)",
    "time_on": "Time hedged (%)",
    "tail_offset": "Tail-loss offset (%)",
}
