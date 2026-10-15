import json

import numpy as np
import pandas as pd
import pytest

from bvivhedge import data, strc
from bvivhedge.hedge import HedgeConfig, Rule, backtest, build_signals, hold_between, run_rules, synchronous_moves
from bvivhedge.metrics import daily_book


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


def _bars_et(day: str, opens_et: list[str], closes: list[float]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(f"{day} {h}", tz=strc.ET) for h in opens_et]).tz_convert("UTC")
    return pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1.0}, index=idx)


def _daily(days: list[str], opens: list[float], closes: list[float]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(f"{d} 09:30", tz=strc.ET) for d in days]).tz_convert("UTC")
    return pd.DataFrame({"open": opens, "high": closes, "low": closes, "close": closes, "volume": 1.0}, index=idx)


def test_cash_observations_follow_the_exchange_calendar():
    # 28 Nov 2025 closes at 13:00 ET; on 30 Jan 2026 the hourly source stops at 11:30 ET
    hourly = pd.concat([_bars_et("2025-11-28", ["09:30", "10:30", "11:30", "12:30", "13:00"], [1, 2, 3, 4, 5]),
                        _bars_et("2026-01-30", ["09:30", "10:30"], [10, 11])])
    daily = _daily(["2025-11-28", "2026-01-30"], [0.5, 9.5], [5.5, 12.0])
    obs = strc.cash_observations(hourly, daily)
    et = [(t.tz_convert(strc.ET).strftime("%Y-%m-%d %H:%M"), v) for t, v in obs.items()]
    assert et[:5] == [("2025-11-28 09:30", 0.5), ("2025-11-28 10:30", 1), ("2025-11-28 11:30", 2),
                      ("2025-11-28 12:30", 3), ("2025-11-28 13:00", 5.5)]          # the official close at the early close
    assert et[5:] == [("2026-01-30 09:30", 9.5), ("2026-01-30 10:30", 10), ("2026-01-30 11:30", 11),
                      ("2026-01-30 16:00", 12.0)]                                   # not a half day: the close is filled
    assert strc.hourly_gaps(hourly, daily) == ["2026-01-30"]


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


def test_dividend_is_booked_before_the_ex_date_open():
    idx = pd.date_range("2026-07-14 19:00", "2026-07-15 15:00", freq="15min", tz="UTC")
    update = np.zeros(len(idx), dtype=bool)
    opening = pd.Timestamp("2026-07-15 13:15", tz="UTC")          # bar closing at 9:30 ET (EDT)
    update[[idx.get_loc(pd.Timestamp("2026-07-14 19:45", tz="UTC")), idx.get_loc(opening)]] = True
    divs = pd.DataFrame({"amount": [0.5]}, index=pd.DatetimeIndex(["2026-07-15"]))
    f = strc.dividend_per_bar(divs, idx, update)
    assert f.sum() == -0.5 and f.idxmin() == opening - pd.Timedelta("15min")


def test_perp_funding_is_paid_by_the_position_held_at_settlement():
    idx = pd.date_range("2026-08-01", periods=12, freq="15min", tz="UTC")
    rates = pd.Series([0.001], index=pd.DatetimeIndex([pd.Timestamp("2026-08-01 02:00:00.002", tz="UTC")]))
    oracle = pd.Series(np.arange(12, dtype=float) + 90.0, index=idx)   # stamped at the start of each 15m average
    f = strc.perp_funding_per_bar(rates, oracle, idx)
    row = pd.Timestamp("2026-08-01 01:30", tz="UTC")                   # E - 30m: backtest charges it on bar E - 15m
    assert np.isclose(f[row], 0.001 * oracle[pd.Timestamp("2026-08-01 01:45", tz="UTC")]) and np.isclose(f.sum(), f[row])


def test_daily_book_over_trading_day_labels():
    bars = _two_asset_bars(n=8)
    pnl = backtest(bars, np.full(8, -5.0), HedgeConfig(fee_bps=0.0, slippage_bps=0.0))
    day = pd.Series([pd.Timestamp("2026-01-02")] * 4 + [pd.Timestamp("2026-01-05")] * 3 + [pd.NaT], index=bars.index)
    book = daily_book(pnl, day=day)
    assert list(book.index) == [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-05")]
    assert np.isclose(book["btc"].iloc[0] * pnl["btc_notional"].iloc[0], pnl["btc"].iloc[:4].sum())


def test_run_rules_fills_only_on_tradable_bars():
    bars = _two_asset_bars(n=600)
    bars["tradable"] = np.arange(600) % 7 == 0
    res, _ = run_rules(bars, HedgeConfig(direction=-1, max_notional_frac=10.0, warmup_days=1, exec_delay=1),
                       [Rule("always", "always")])
    changes = np.flatnonzero(np.diff(res["always"]["position"].to_numpy()) != 0) + 1
    assert len(changes) and bars["tradable"].to_numpy()[changes].all()


def test_oxarchive_rows_follow_the_cursor_and_stop(monkeypatch, tmp_path):
    pages = {None: {"data": [{"timestamp": "2026-07-01T00:00:00Z", "x": 1}], "meta": {"has_more": True, "next_cursor": "c1"}},
             "c1": {"data": [{"timestamp": "2026-07-01T00:15:00Z", "x": 2}], "meta": {"has_more": False}}}
    seen = []

    class Reply:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(self.body).encode()

    def fake_urlopen(req, timeout=60):
        cursor = req.full_url.split("cursor=")[1] if "cursor=" in req.full_url else None
        seen.append(cursor)
        return Reply(pages[cursor])

    monkeypatch.setattr(data, "urlopen", fake_urlopen)
    monkeypatch.setattr(data, "RAW", tmp_path)
    rows = json.loads(data.oxarchive_rows("prices/{symbol}", "xyz:TEST", "2026-07-01", "2026-07-02", {"interval": "15m"}, api_key="k"))
    assert [r["x"] for r in rows] == [1, 2] and seen == [None, "c1"]
    seen.clear()
    one = json.loads(data.oxarchive_rows("prices/{symbol}", "xyz:TEST", pd.Timestamp("2026-07-03 20:00", tz="UTC"),
                                         pd.Timestamp("2026-07-03 20:02", tz="UTC"), api_key="k", max_pages=1))
    assert len(one) == 1 and seen == [None]
    assert all(":" not in f.name for f in tmp_path.iterdir())          # cache names are portable
    monkeypatch.delenv("OXARCHIVE_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        data.oxarchive_rows("prices/{symbol}", "xyz:TEST", "2026-07-05", "2026-07-06")


def test_yahoo_chart_is_cut_to_the_window(monkeypatch):
    ts = [int(pd.Timestamp(x, tz="UTC").timestamp()) for x in ("2026-10-02 19:30", "2026-10-05 13:30")]
    payload = json.dumps({"chart": {"result": [{"meta": {}, "timestamp": ts, "indicators": {"quote": [
        {"open": [1.0, 2.0], "high": [1.0, 2.0], "low": [1.0, 2.0], "close": [1.0, 2.0], "volume": [1.0, 1.0]}]}}]}}).encode()
    monkeypatch.setattr(data, "_get", lambda url, retries=4: payload)
    monkeypatch.setattr(data, "_window_ended", lambda end: False)
    bars = data.fetch_yahoo_chart("STRC", "60m", "2026-10-01", "2026-10-04")
    assert len(bars) == 1 and bars.index[0] == pd.Timestamp("2026-10-02 19:30", tz="UTC")


def _fake_inputs(monkeypatch):
    """Monday 6 to Wednesday 8 July 2026 (Friday 3 July is a holiday), hourly STRC, a perpetual listed on Tuesday."""
    idx = pd.date_range("2026-07-04", "2026-07-09", freq="15min", tz="UTC", inclusive="left")
    btc = pd.DataFrame({"open": 60_000.0, "high": 60_000.0, "low": 60_000.0, "close": 60_000.0 + np.arange(len(idx)),
                        "volume": 1.0, "quote_volume": 60_000.0}, index=idx)
    days = ["2026-07-06", "2026-07-07", "2026-07-08"]
    hours = ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"]
    hourly = pd.concat([_bars_et(d, hours, [100.0 + i + 0.1 * k for k in range(7)]) for i, d in enumerate(days)])
    daily = _daily(days, [99.9, 100.9, 101.9], [100.7, 101.7, 102.7])
    divs = pd.DataFrame({"amount": [0.5], "record_date": pd.NaT, "pay_date": pd.NaT, "declared": pd.NaT},
                        index=pd.DatetimeIndex(["2026-07-06"]))                # ex-date the Monday after the holiday
    listing = pd.Timestamp("2026-07-07 18:00", tz="UTC")
    perp_idx = idx[idx >= listing]
    marks = pd.Series(200.0 + np.arange(len(perp_idx)), index=perp_idx)
    marks = marks.drop(perp_idx[5])                                            # one missing 15-minute sample
    oi_idx = pd.date_range(listing, idx[-1], freq="1h")
    payloads = {
        "prices": [{"timestamp": t.isoformat(), "mark_price": m, "oracle_price": m, "mid_price": m} for t, m in marks.items()],
        "candles": [{"timestamp": t.isoformat(), "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1, "quote_volume": 1,
                     "trade_count": 3} for t in perp_idx],
        "openinterest": [{"timestamp": t.isoformat(), "open_interest": 1000.0 + k} for k, t in enumerate(oi_idx)],
    }
    funding = [{"coin": "xyz:STRC", "fundingRate": "-0.0001", "premium": "0", "time": int((t + pd.Timedelta("1ms")).timestamp() * 1000)}
               for t in oi_idx[1:]]
    monkeypatch.setattr(data, "fetch_binance_klines", lambda *a, **k: btc)
    monkeypatch.setattr(data, "fetch_volmex_public", lambda *a, **k: pd.Series(50.0, index=idx))
    monkeypatch.setattr(data, "fetch_yahoo_chart", lambda symbol, interval, start, end: hourly if interval == "60m" else daily)
    monkeypatch.setattr(data, "fetch_nasdaq_dividends", lambda symbol, until: divs)
    monkeypatch.setattr(data, "oxarchive_rows", lambda route, *a, **k: json.dumps(payloads[route.split("/")[0]]).encode())
    monkeypatch.setattr(data, "fetch_hyperliquid", lambda kind, *a, **k: json.dumps(funding).encode())
    return listing, perp_idx[5]


def test_build_panel_cash(monkeypatch):
    _fake_inputs(monkeypatch)
    p = strc.build_panel("2026-07-04", "2026-07-09", "cash", borrow_annual=0.01)
    b = p["bars"]
    obs = b.index[b["hedge_update"]]
    assert ((obs + strc.BAR).tz_convert(strc.ET).strftime("%H:%M")[:9] == ["09:30", "10:30", "11:30", "12:30", "13:30",
                                                                          "14:30", "15:30", "16:00", "09:30"]).all()
    assert (b["tradable"] == b["hedge_update"]).all()
    first = obs[0]                                                     # the Monday open carries the ex-dividend drop
    assert np.isclose(b["bviv_funding"].loc[first - strc.BAR], -0.5)   # no borrow fee before the first price
    assert np.isclose(b["bviv_funding"].loc[first], -0.01 * b["bviv_mark"].loc[first] * strc.DT)
    assert b["session"].loc[pd.Timestamp("2026-07-05 12:00", tz="UTC")] == pd.Timestamp("2026-07-06")   # Sunday -> Monday
    assert b.loc[first, "bviv_mark"] == 99.9 and b["bviv_mark"].loc[obs[7]] == 100.7   # official open and close


def test_build_panel_perp(monkeypatch):
    listing, missing = _fake_inputs(monkeypatch)
    p = strc.build_panel("2026-07-04", "2026-07-09", "perp")
    b = p["bars"]
    before, after = b.index < listing, b.index >= listing
    assert not b["tradable"][before].any() and b["tradable"][after].sum() == after.sum() - 1
    assert not b.loc[missing, "tradable"] and b.loc[missing, "bviv_mark"] == b.loc[missing - strc.BAR, "bviv_mark"]
    assert b.loc[listing, "bviv_mark"] == 200.0                        # the bar opening at T carries the average stamped T
    oi = b["open_interest"].dropna()
    assert oi.index[0] == listing + pd.Timedelta("45min") and oi.iloc[0] == 1000.0     # known when its hour has ended
    assert (b["bviv_funding"][before] == 0).all() and (b["bviv_funding"][after] < 0).any()


def test_parse_oxarchive_series_and_dividend_cut():
    rows = json.dumps([{"timestamp": "2026-07-01T00:15:00Z", "mark_price": "2"}, {"timestamp": "2026-07-01T00:00:00Z", "mark_price": "1"},
                       {"timestamp": "2026-07-01T00:00:00Z", "mark_price": "1"}]).encode()
    s = data.parse_oxarchive_series(rows, ("mark_price",))
    assert s["mark_price"].tolist() == [1.0, 2.0] and str(s.index.tz) == "UTC"
    assert data.parse_oxarchive_series(b"[]", ("mark_price",)).empty
