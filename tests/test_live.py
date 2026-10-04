import json

import numpy as np
import pandas as pd

from bvivhedge.data import bitfinex_funding_events, parse_bitfinex_status_raw
from bvivhedge.hedge import HedgeConfig, backtest
from bvivhedge.live import funding_per_bar, paired_block_bootstrap, stationary_bootstrap_index, status_to_bars


def _status_rows(start_ms: int, n: int, event_ms: int, rate_before: float, rate_after: float):
    rows = []
    for i in range(n):
        t = start_ms + i * 60_000
        nxt = event_ms if t < event_ms else event_ms + 8 * 3600_000
        cur = rate_before if t < event_ms else rate_after
        rows.append([t, None, 50.2, 50.0, None, 1e6, None, nxt, 0.001, i, None, cur, None, None, 50.0, None, None, 10.0])
    return json.dumps(rows).encode()


def test_funding_event_reconstruction():
    ev = pd.Timestamp("2025-01-01 08:00", tz="UTC")
    st = parse_bitfinex_status_raw(_status_rows(int(ev.timestamp() * 1000) - 30 * 60_000, 60, int(ev.timestamp() * 1000), 0.0005, 0.0025))
    events = bitfinex_funding_events(st)
    assert list(events.index) == [ev]
    assert events["rate"].iloc[0] == 0.0025 and events["mark"].iloc[0] == 50.0


def test_funding_is_charged_to_the_position_held_at_settlement():
    idx = pd.date_range("2025-01-01 07:00", "2025-01-01 09:00", freq="15min", tz="UTC")
    events = pd.DataFrame({"rate": [0.001], "mark": [50.0]}, index=[pd.Timestamp("2025-01-01 08:00", tz="UTC")])
    bars = pd.DataFrame({"close": 1.0, "bviv_mark": 50.0}, index=idx)
    bars["bviv_funding"] = funding_per_bar(events, idx)
    bars["bviv_carry"] = bars["bviv_funding"]
    held_over_settlement = np.zeros(len(idx))
    held_over_settlement[idx.get_loc(pd.Timestamp("2025-01-01 07:30", tz="UTC"))] = 10.0   # held 07:45 -> 08:00
    pnl = backtest(bars, held_over_settlement, HedgeConfig(fee_bps=0, slippage_bps=0))
    assert np.isclose(pnl["funding"].sum(), 10 * 0.001 * 50.0)
    after = np.zeros(len(idx))
    after[idx.get_loc(pd.Timestamp("2025-01-01 07:45", tz="UTC"))] = 10.0                  # opened at the settlement
    assert np.isclose(backtest(bars, after, HedgeConfig(fee_bps=0, slippage_bps=0))["funding"].sum(), 0.0)


def test_basis_cost_on_fills():
    idx = pd.date_range("2025-01-01", periods=4, freq="15min", tz="UTC")
    bars = pd.DataFrame({"close": 1.0, "bviv_mark": 50.0, "bviv_fill": 50.5, "bviv_funding": 0.0,
                         "bviv_carry": 0.0}, index=idx)
    pnl = backtest(bars, np.array([0.0, 2.0, 2.0, 0.0]), HedgeConfig(fee_bps=0, slippage_bps=0))
    assert np.isclose(pnl["basis"].sum(), 2 * 0.5 - 2 * 0.5)          # buy at a premium, sell at the same premium
    assert np.isclose(pnl["basis"].iloc[1], 1.0)


def test_status_to_bars_takes_last_snapshot_in_bar():
    t = pd.to_datetime(["2025-01-01 00:01", "2025-01-01 00:14", "2025-01-01 00:16"], utc=True)
    st = pd.DataFrame({"mark": [1.0, 2.0, 3.0], "deriv_price": 0.0, "open_interest": 0.0}, index=t)
    out = status_to_bars(st, pd.date_range("2025-01-01", periods=2, freq="15min", tz="UTC"))
    assert out["mark"].tolist() == [2.0, 3.0]


def test_block_bootstrap():
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_index(500, 10, rng)
    assert len(idx) == 500 and idx.min() >= 0 and idx.max() < 500
    days = pd.date_range("2025-01-01", periods=300, freq="D")
    u = pd.DataFrame({"total": rng.standard_normal(300) * 0.03}, index=days)
    books = {"U": u, "A": u * 0.9, "B": u * 0.9}
    res = paired_block_bootstrap(books, "U", lambda r, base: r.std() / base.std(), "A", "B", n_boot=200)
    assert res["diff"] == 0.0 and res["lo"] == 0.0 and res["hi"] == 0.0
