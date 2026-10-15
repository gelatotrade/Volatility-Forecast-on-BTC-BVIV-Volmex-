import json

import numpy as np
import pandas as pd

from bvivhedge import data, strc
from bvivhedge.hedge import HedgeConfig, Rule, backtest, build_signals, hold_between, run_rules, synchronous_moves


def test_parse_yahoo_chart_drops_bars_without_close():
    payload = json.dumps({"chart": {"result": [{"meta": {}, "timestamp": [1753882200, 1753885800, 1753889400],
                                                "indicators": {"quote": [{"open": [94.0, None, 94.6], "high": [95.0, None, 94.9],
                                                                          "low": [93.5, None, 94.1], "close": [94.5, None, 94.7],
                                                                          "volume": [1e5, None, 8e4]}]}}]}}).encode()
    bars = data.parse_yahoo_chart(payload)
    assert len(bars) == 2 and bars["close"].tolist() == [94.5, 94.7]
    assert bars.index[0] == pd.Timestamp("2025-07-30 13:30", tz="UTC")


def test_parse_nasdaq_dividends():
    payload = json.dumps({"data": {"dividends": {"rows": [
        {"exOrEffDate": "06/15/2026", "type": "Cash", "amount": "$0.95833333", "declarationDate": "05/30/2026",
         "recordDate": "06/15/2026", "paymentDate": "06/30/2026", "currency": "USD"},
        {"exOrEffDate": "03/13/2026", "type": "Cash", "amount": "$0.95833333", "declarationDate": "N/A",
         "recordDate": "03/15/2026", "paymentDate": "03/31/2026", "currency": "USD"}]}}}).encode()
    d = data.parse_nasdaq_dividends(payload)
    assert list(d.index) == [pd.Timestamp("2026-03-13"), pd.Timestamp("2026-06-15")]
    assert np.isclose(d["amount"].iloc[0], 0.95833333) and pd.isna(d["declared"].iloc[0])


def _hourly(day: str, opens_et: list[str], closes: list[float]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(f"{day} {h}", tz=strc.ET) for h in opens_et]).tz_convert("UTC")
    return pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1.0}, index=idx)


def test_hourly_closes_end_on_the_half_hour_and_at_the_close():
    full = _hourly("2026-03-10", ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"], list(range(7)))
    half = _hourly("2025-11-28", ["09:30", "10:30", "11:30", "12:30", "13:00"], list(range(5)))
    s = strc.hourly_closes(pd.concat([half, full]))
    ends = (s.index + pd.Timedelta("15min")).tz_convert(strc.ET).strftime("%Y-%m-%d %H:%M").tolist()
    assert ends[:4] == ["2025-11-28 10:30", "2025-11-28 11:30", "2025-11-28 12:30", "2025-11-28 13:00"]
    assert s.iloc[3] == 4.0                                    # the 13:00 print replaces the 12:30 bar's close
    assert ends[-2:] == ["2026-03-10 15:30", "2026-03-10 16:00"]


def test_session_labels_put_the_weekend_on_monday():
    closes = pd.DatetimeIndex([pd.Timestamp("2026-03-06 16:00", tz=strc.ET), pd.Timestamp("2026-03-09 16:00", tz=strc.ET)]).tz_convert("UTC")
    bars = pd.DatetimeIndex([pd.Timestamp("2026-03-06 15:45", tz=strc.ET), pd.Timestamp("2026-03-07 12:00", tz=strc.ET),
                             pd.Timestamp("2026-03-09 15:45", tz=strc.ET), pd.Timestamp("2026-03-09 16:00", tz=strc.ET)]).tz_convert("UTC")
    lab = strc.session_labels(bars, closes)
    assert lab.iloc[0] == pd.Timestamp("2026-03-06") and lab.iloc[1] == lab.iloc[2] == pd.Timestamp("2026-03-09")
    assert pd.isna(lab.iloc[3])                               # after the last close: no session yet


def _two_asset_bars(n: int = 4000, seed: int = 1, link: float = 0.3) -> pd.DataFrame:
    """BTC and a hedge instrument that moves with it (an STRC-like claim)."""
    rng = np.random.default_rng(seed)
    r = 0.004 * rng.standard_normal(n)
    btc = 60_000 * np.exp(np.cumsum(r))
    inst = 100 * np.exp(np.cumsum(link * r + 0.0005 * rng.standard_normal(n)))
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"close": btc, "volume": 1.0, "quote_volume": btc, "bviv": 50.0, "bviv_mark": inst,
                         "bviv_funding": 0.0, "bviv_carry": 0.0, "btc_funding": 0.0}, index=idx)


def test_short_hedge_offsets_btc_losses():
    bars = _two_asset_bars()
    cfg = HedgeConfig(direction=-1, max_notional_frac=10.0, fee_bps=0.0, slippage_bps=0.0, warmup_days=5)
    res, sig = run_rules(bars, cfg, [Rule("Unhedged", "unhedged"), Rule("always", "always")])
    pos = res["always"]["position"].to_numpy()
    assert (pos[5 * 96:] < 0).all() and (sig["mv_all"].iloc[5 * 96:] > 0).all()
    on = slice(5 * 96, None)
    hedged = (res["always"]["btc"] + res["always"]["hedge"]).iloc[on]
    assert hedged.var() < 0.3 * res["Unhedged"]["btc"].iloc[on].var()
    assert (res["always"]["notional"] >= 0).all()             # gross notional, also for a short


def test_long_hedge_ignores_a_positively_linked_instrument():
    bars = _two_asset_bars()
    sig = build_signals(bars, HedgeConfig(direction=1))
    assert (sig["mv_all"].iloc[5 * 96:] == 0).all()


def test_dividend_is_paid_by_a_short_and_received_by_a_long():
    bars = _two_asset_bars(n=10)
    bars["bviv_mark"] = 100.0
    bars.iloc[6:, bars.columns.get_loc("bviv_mark")] = 99.0     # the ex-date drop, on bar 6
    bars["bviv_funding"] = 0.0
    bars.iloc[5, bars.columns.get_loc("bviv_funding")] = -1.0   # minus the dividend, on the row before
    cfg = HedgeConfig(fee_bps=0.0, slippage_bps=0.0)
    long_ = backtest(bars, np.full(10, 10.0), cfg)
    short = backtest(bars, np.full(10, -10.0), cfg)
    assert np.isclose(long_["hedge"].sum(), 0.0)               # price drop and dividend cancel for a holder
    assert np.isclose(short["hedge"].sum(), 0.0) and np.isclose(short["funding"].sum(), 10.0)


def test_positions_change_only_on_tradable_bars():
    target = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=float)
    tradable = np.array([False, False, True, False, False, True, False, False])
    assert hold_between(target, tradable).tolist() == [0, 0, 2, 2, 2, 5, 5, 5]


def test_synchronous_moves_accumulate_btc_between_observations():
    bars = _two_asset_bars(n=8)
    upd = np.array([True, False, False, True, False, True, False, True])
    bars["hedge_update"] = upd
    bars.loc[~upd, "bviv_mark"] = np.nan
    bars["bviv_mark"] = bars["bviv_mark"].ffill()
    r, d, ends = synchronous_moves(bars)
    lp = np.log(bars["close"].to_numpy())
    assert np.isclose(r[3], lp[3] - lp[0]) and np.isclose(r[5], lp[5] - lp[3]) and r[4] == 0
    assert ends.tolist() == [False, False, False, True, False, True, False, True]
    assert np.isclose(d[3], bars["bviv_mark"].iloc[3] - bars["bviv_mark"].iloc[0])


def test_hyperliquid_paging_stops_when_the_reply_repeats(monkeypatch):
    hour = 3_600_000
    t0 = int(pd.Timestamp("2026-06-22", tz="UTC").timestamp() * 1000)

    def fake_post(url, body):
        start = body["req"]["startTime"]
        first = max(t0, start - (start - t0) % hour)          # the reply starts with the candle containing the cursor
        rows = [{"t": t, "T": t + hour - 1, "s": "x", "i": "1h", "o": "1", "c": "1", "h": "1", "l": "1", "v": "0", "n": 0}
                for t in range(first, min(first + 3 * hour, t0 + 5 * hour), hour)]
        return json.dumps(rows).encode()

    monkeypatch.setattr(data, "_post_json", fake_post)
    monkeypatch.setattr(data, "_window_ended", lambda end: False)
    rows = json.loads(data.fetch_hyperliquid("candles", "xyz:TEST", "2026-06-22", "2026-06-23", "1h"))
    assert [r["t"] for r in rows] == [t0 + k * hour for k in range(5)]
