"""Live market data: BTC 15m klines, BTC perp funding, the BVIV index and its perpetual.

Sources (all public, no key required):

* Binance bulk archive ``data.binance.vision`` -- BTCUSDT 15m klines (spot or USD-M);
  ``quote_volume / volume`` is each bar's exact VWAP.
* Binance USD-M REST -- BTCUSDT funding-rate history.
* Bitfinex -- the BVIV perpetual ``tBVIVF0:USTF0`` (listed April 2024): trade candles and
  the derivatives-status history (index price, mark price, current funding).
* Volmex REST API ``rest-v1.volmex.finance/v2/history`` -- BVIV bars at 1-60 minute and
  daily resolution (historical ranges need an API key, read from ``VOLMEX_API_KEY``).
* Deribit -- the DVOL index, a close substitute for BVIV before the perpetual existed.
* Any CSV export of BVIV (e.g. from a charting platform) via :func:`load_index_csv`.

Parsers are separated from fetchers so they can be tested offline.  Every
fetch is cached under ``data/raw`` so a rerun is free.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .constants import DT, MINUTES_PER_BAR

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
BAR = f"{MINUTES_PER_BAR}min"
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
              "trades", "taker_base", "taker_quote", "ignore"]


# --------------------------------------------------------------------------- http + cache
def _get(url: str, retries: int = 4) -> bytes:
    for attempt in range(retries):
        try:
            with urlopen(Request(url, headers={"User-Agent": "bvivhedge/1.0"}), timeout=60) as resp:
                return resp.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("unreachable")


def _cached(name: str, fetch) -> bytes:
    path = RAW / name
    if path.exists():
        return path.read_bytes()
    blob = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return blob


def _to_utc(ms_or_us: pd.Series) -> pd.DatetimeIndex:
    """Binance switched spot archives to microseconds in 2025; accept both."""
    v = ms_or_us.astype("int64")
    unit = np.where(v > 10**14, "us", "ms")
    if (unit == "us").all() or (unit == "ms").all():
        return pd.to_datetime(v, unit=unit[0], utc=True)
    return pd.DatetimeIndex([pd.Timestamp(x, unit=u, tz="UTC") for x, u in zip(v, unit)])


# --------------------------------------------------------------------------- Binance
def parse_binance_klines(csv_bytes: bytes) -> pd.DataFrame:
    raw = pd.read_csv(io.BytesIO(csv_bytes), header=None)
    if not str(raw.iloc[0, 0]).isdigit():          # newer archives carry a header row
        raw = raw.iloc[1:]
    raw.columns = KLINE_COLS[: raw.shape[1]]
    out = raw[["open", "high", "low", "close", "volume", "quote_volume"]].astype(float)
    out.index = _to_utc(raw["open_time"])
    return out


def fetch_binance_klines(start: str, end: str, symbol: str = "BTCUSDT", market: str = "spot") -> pd.DataFrame:
    """15m klines from the monthly bulk archive, ``start``/``end`` as 'YYYY-MM' (inclusive)."""
    base = {"spot": "spot", "perp": "futures/um"}[market]
    frames = []
    for month in pd.period_range(start, end, freq="M"):
        fname = f"{symbol}-{BAR.replace('min', 'm')}-{month.year}-{month.month:02d}.zip"
        url = f"https://data.binance.vision/data/{base}/monthly/klines/{symbol}/15m/{fname}"
        blob = _cached(f"binance_{market}_{fname}", lambda u=url: _get(u))
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            frames.append(parse_binance_klines(zf.read(zf.namelist()[0])))
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated()].asfreq(BAR).ffill()


def parse_binance_funding(payload: bytes) -> pd.Series:
    rows = json.loads(payload)
    s = pd.Series([float(r["fundingRate"]) for r in rows],
                  index=pd.to_datetime([int(r["fundingTime"]) for r in rows], unit="ms", utc=True))
    return s.sort_index()


def fetch_binance_funding(start: str, end: str, symbol: str = "BTCUSDT") -> pd.Series:
    """Funding rate per 8h interval (fraction), USD-M perpetual."""
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    t1 = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    parts = []
    while t0 < t1:
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&startTime={t0}&endTime={t1}&limit=1000"
        s = parse_binance_funding(_cached(f"binance_funding_{symbol}_{t0}.json", lambda u=url: _get(u)))
        if s.empty:
            break
        parts.append(s)
        t0 = int(s.index[-1].timestamp() * 1000) + 1
    return pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)


# --------------------------------------------------------------------------- Bitfinex BVIV perpetual
BITFINEX_SYMBOL = "tBVIVF0:USTF0"
# /v2/status/deriv/{key}/hist rows: [MTS, _, DERIV_PRICE, SPOT_PRICE, _, INSURANCE_FUND, _,
#   NEXT_FUNDING_TS, NEXT_FUNDING_ACCRUED, NEXT_FUNDING_STEP, _, CURRENT_FUNDING, _, _, MARK_PRICE, ...]
_STATUS_IDX = {"deriv_price": 2, "index": 3, "funding_8h": 11, "mark": 14}


def parse_bitfinex_candles(payload: bytes) -> pd.DataFrame:
    rows = json.loads(payload)
    df = pd.DataFrame(rows, columns=["mts", "open", "close", "high", "low", "volume"])
    df.index = pd.to_datetime(df.pop("mts"), unit="ms", utc=True)
    return df.sort_index().astype(float)


def parse_bitfinex_status(payload: bytes) -> pd.DataFrame:
    rows = json.loads(payload)
    data = {k: [r[i] if len(r) > i else None for r in rows] for k, i in _STATUS_IDX.items()}
    df = pd.DataFrame(data, index=pd.to_datetime([r[0] for r in rows], unit="ms", utc=True))
    return df.sort_index().astype(float)


def _bitfinex_paged(path: str, start: str, end: str, parser, limit: int) -> pd.DataFrame:
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    t1 = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    parts = []
    while t0 < t1:
        url = f"https://api-pub.bitfinex.com/v2/{path}?start={t0}&end={t1}&limit={limit}&sort=1"
        df = parser(_cached(f"bitfinex_{path.replace('/', '_')}_{t0}.json", lambda u=url: _get(u)))
        if df.empty:
            break
        parts.append(df)
        t0 = int(df.index[-1].timestamp() * 1000) + 1
        time.sleep(1.0)  # public rate limit
    return pd.concat(parts).sort_index() if parts else pd.DataFrame()


def fetch_bitfinex_bviv(start: str, end: str) -> pd.DataFrame:
    """15m BVIV perpetual: trade close, index, mark and funding (8h rate, fraction)."""
    candles = _bitfinex_paged(f"candles/trade:15m:{BITFINEX_SYMBOL}/hist", start, end, parse_bitfinex_candles, 10_000)
    status = _bitfinex_paged(f"status/deriv/{BITFINEX_SYMBOL}/hist", start, end, parse_bitfinex_status, 5_000)
    status15 = status.resample(BAR, label="left", closed="left").last()
    out = pd.DataFrame({"perp_close": candles["close"]}).join(status15, how="outer")
    return out[~out.index.duplicated()].sort_index()


# --------------------------------------------------------------------------- Volmex BVIV
def parse_volmex_history(payload: bytes) -> pd.Series:
    """Accept a TradingView-UDF reply ({"t": [...], "c": [...]}) or a list of {time, close} records."""
    obj = json.loads(payload)
    if isinstance(obj, dict) and "t" in obj:
        t, c = obj["t"], obj["c"]
    else:
        rows = obj["data"] if isinstance(obj, dict) else obj
        t, c = [r["time"] for r in rows], [r["close"] for r in rows]
    t = np.asarray(t, dtype="int64")
    unit = "ms" if t.size and t.max() > 10**11 else "s"
    return pd.Series(np.asarray(c, float), index=pd.to_datetime(t, unit=unit, utc=True)).sort_index()


def fetch_volmex_bviv(start: str, end: str, symbol: str = "BVIV", api_key: str | None = None) -> pd.Series:
    """BVIV closes on 15-minute bars from the Volmex REST API (paged by month)."""
    import os

    key = api_key or os.environ.get("VOLMEX_API_KEY", "")
    parts = []
    for month in pd.period_range(start, end, freq="M"):
        t0 = int(month.start_time.tz_localize("UTC").timestamp())
        t1 = int(month.end_time.tz_localize("UTC").timestamp())
        url = (f"https://rest-v1.volmex.finance/v2/history?symbol={symbol}&resolution=15&from={t0}&to={t1}"
               + (f"&apiKey={key}" if key else ""))
        parts.append(parse_volmex_history(_cached(f"volmex_{symbol}_{month}.json", lambda u=url: _get(u))))
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated()].resample(BAR, label="left", closed="left").last()


# --------------------------------------------------------------------------- Deribit DVOL
def parse_deribit_dvol(payload: bytes) -> tuple[pd.Series, int | None]:
    res = json.loads(payload)["result"]
    data = res.get("data", [])
    s = pd.Series([row[4] for row in data], index=pd.to_datetime([row[0] for row in data], unit="ms", utc=True))
    return s.sort_index(), res.get("continuation")


def fetch_deribit_dvol(start: str, end: str, currency: str = "BTC") -> pd.Series:
    """DVOL closes at 1-minute resolution, resampled to 15m (last)."""
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    t1 = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    parts, cursor = [], t1
    while cursor and cursor > t0:
        url = ("https://www.deribit.com/api/v2/public/get_volatility_index_data?"
               f"currency={currency}&start_timestamp={t0}&end_timestamp={cursor}&resolution=60")
        s, cursor = parse_deribit_dvol(_cached(f"deribit_dvol_{currency}_{cursor}.json", lambda u=url: _get(u)))
        if s.empty:
            break
        parts.append(s)
    dvol = pd.concat(parts).sort_index()
    return dvol[~dvol.index.duplicated()].resample(BAR, label="left", closed="left").last()


def load_index_csv(path: str | Path, time_col: str = "time", value_col: str = "close") -> pd.Series:
    """BVIV (or any index) from a CSV with unix-seconds or ISO timestamps, resampled to 15m."""
    df = pd.read_csv(path)
    t = df[time_col]
    idx = pd.to_datetime(t, unit="s", utc=True) if np.issubdtype(t.dtype, np.number) else pd.to_datetime(t, utc=True)
    return pd.Series(df[value_col].astype(float).to_numpy(), index=idx).sort_index().resample(BAR).last()


# --------------------------------------------------------------------------- assembly
def assemble_live_bars(klines: pd.DataFrame, index_15m: pd.Series, perp: pd.DataFrame | None = None,
                       btc_funding_8h: pd.Series | None = None, hedge_carry: float = 6.0) -> pd.DataFrame:
    """Bring live series into the simulator's schema so every downstream module is shared.

    Without perpetual data the mark is the index and funding is a constant carry
    of ``hedge_carry`` vol points a year (the simulator's baseline).
    """
    bars = klines.copy()
    bars["bviv"] = index_15m.reindex(bars.index).ffill()
    if perp is not None and not perp.empty:
        p = perp.reindex(bars.index).ffill()
        bars["bviv_mark"] = p["mark"].fillna(p["perp_close"]).fillna(bars["bviv"])
        # an 8h funding rate on notional -> vol points per contract per 15m bar
        bars["bviv_funding"] = (p["funding_8h"].fillna(0.0) * bars["bviv_mark"]) / 32.0
        # the carry premium is the funding a long pays on average (drift compensation averages out)
        bars["bviv_carry"] = bars["bviv_funding"].mean()
    else:
        bars["bviv_mark"] = bars["bviv"]
        bars["bviv_funding"] = hedge_carry * DT
        bars["bviv_carry"] = hedge_carry * DT
    if btc_funding_8h is not None and not btc_funding_8h.empty:
        rate = btc_funding_8h.reindex(bars.index, method="ffill").fillna(0.0)
        bars["btc_funding"] = rate * 3 * 365          # annualised
    else:
        bars["btc_funding"] = 0.0
    return bars.dropna(subset=["close", "bviv"])
