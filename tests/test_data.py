import json

import numpy as np
import pandas as pd

from bvivhedge.data import (
    assemble_live_bars, regularise_klines, parse_binance_funding, parse_binance_klines, parse_bitfinex_candles,
    parse_bitfinex_status, parse_deribit_dvol, parse_volmex_history,
)


def test_binance_klines_ms_and_us_timestamps():
    ms = b"1704067200000,42000,42100,41900,42050,10.5,1704068099999,441000,100,5,210000,0\n"
    us = b"1735689600000000,94000,94100,93900,94050,8.0,1735690499999999,752000,90,4,376000,0\n"
    a, b = parse_binance_klines(ms), parse_binance_klines(us)
    assert a.index[0] == pd.Timestamp("2024-01-01", tz="UTC")
    assert b.index[0] == pd.Timestamp("2025-01-01", tz="UTC")
    assert np.isclose(a["quote_volume"].iloc[0] / a["volume"].iloc[0], 42000)


def test_binance_funding():
    s = parse_binance_funding(json.dumps([{"fundingTime": 1704067200000, "fundingRate": "0.0001"}]).encode())
    assert s.iloc[0] == 1e-4


def test_bitfinex_parsers():
    candles = parse_bitfinex_candles(json.dumps([[1712102400000, 55.0, 56.0, 56.5, 54.5, 1200.0]]).encode())
    assert candles["close"].iloc[0] == 56.0 and candles["high"].iloc[0] == 56.5
    row = [1712102400000, None, 55.2, 55.0, None, 1e6, None, 1712131200000, 0.0, 0, None, 0.0001, None, None, 55.1]
    status = parse_bitfinex_status(json.dumps([row]).encode())
    assert status["index"].iloc[0] == 55.0 and status["mark"].iloc[0] == 55.1
    assert status["funding_8h"].iloc[0] == 1e-4


def test_deribit_dvol_parser():
    payload = {"result": {"data": [[1704067200000, 50, 51, 49, 50.5]], "continuation": 1704060000000}}
    s, cont = parse_deribit_dvol(json.dumps(payload).encode())
    assert s.iloc[0] == 50.5 and cont == 1704060000000


def test_assemble_live_bars_schema():
    idx = pd.date_range("2024-01-01", periods=96, freq="15min", tz="UTC")
    k = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0, "quote_volume": 1.0}, index=idx)
    iv = pd.Series(50.0, index=idx)
    perp = pd.DataFrame({"perp_close": 50.5, "mark": 50.4, "funding_8h": 3.2e-4, "index": 50.0}, index=idx)
    bars = assemble_live_bars(k, iv, perp)
    assert {"bviv", "bviv_mark", "bviv_funding", "bviv_carry", "btc_funding"} <= set(bars.columns)
    assert np.isclose(bars["bviv_funding"].iloc[0], 3.2e-4 * 50.4 / 32)


def test_volmex_parser_accepts_udf_and_records():
    udf = parse_volmex_history(json.dumps({"s": "ok", "t": [1704067200], "c": [52.5]}).encode())
    rec = parse_volmex_history(json.dumps([{"time": 1704067200000, "close": 52.5}]).encode())
    assert udf.index[0] == rec.index[0] == pd.Timestamp("2024-01-01", tz="UTC")
    assert udf.iloc[0] == rec.iloc[0] == 52.5


def test_kline_gaps_carry_no_volume():
    idx = pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC")
    k = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": [1.0, 1.5, 1.2, 1.3],
                      "volume": 3.0, "quote_volume": 4.0}, index=idx).drop(idx[2])
    out = regularise_klines(k)
    assert len(out) == 4 and out["volume"].iloc[2] == 0.0 and out["close"].iloc[2] == 1.5
    assert out["high"].iloc[2] == 1.5


def test_live_bars_trim_to_index_and_accrue_funding_forward():
    idx = pd.date_range("2024-01-01", periods=96 * 2, freq="15min", tz="UTC")
    k = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0, "quote_volume": 1.0}, index=idx)
    iv = pd.Series(50.0, index=idx[10:150])
    funding = pd.Series([1e-4, 3e-4], index=pd.to_datetime(["2024-01-01 08:00", "2024-01-01 16:00"], utc=True))
    bars = assemble_live_bars(k, iv, btc_funding_8h=funding)
    assert bars.index[0] == idx[10] and bars.index[-1] == idx[149]
    # the rate settled at 16:00 accrues from 08:00 onwards
    assert np.isclose(bars.loc[pd.Timestamp("2024-01-01 09:00", tz="UTC"), "btc_funding"], 3e-4 * 3 * 365)
