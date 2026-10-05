"""Live market data: BTC 15m klines, BTC perp funding, the BVIV index and its perpetual.

Sources (all public; the study needs no key):

* Binance bulk archive ``data.binance.vision`` -- BTCUSDT 15m klines (spot or USD-M);
  ``quote_volume / volume`` is each bar's exact VWAP.
* Binance USD-M REST -- BTCUSDT funding-rate history.
* Bitfinex -- the BVIV perpetual ``tBVIVF0:USTF0`` (listed April 2024): trade candles and
  the derivatives-status history (index price, mark price, current funding).
* Hyperliquid ``api.hyperliquid.xyz/info`` -- the HIP-3 BVIV perpetual ``mkts:BVIV``
  (listed September 2026): 15m candles and hourly funding.  The public API keeps
  only the latest 5000 candles (about 52 days of 15m bars).
* Volmex public history ``rest-v1.volmex.finance/public/iv/history`` -- the official BVIV index
  at 15 minutes (used by the study, no key).  The keyed ``/v2/history`` loader
  (``VOLMEX_API_KEY``) is optional.
* Deribit -- the DVOL index, a close substitute for BVIV before the perpetual existed.
* Any CSV export of BVIV (e.g. from a charting platform) via :func:`load_index_csv`.

Parsers are separated from fetchers so they can be tested offline.  Every
fetch of a window that has already ended is cached under ``data/raw`` (a partial month
under a name that carries its end), so a rerun with the same dates is free and offline.
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


def _post_json(url: str, body: dict, retries: int = 4) -> bytes:
    data = json.dumps(body).encode()
    for attempt in range(retries):
        try:
            req = Request(url, data=data, headers={"Content-Type": "application/json", "User-Agent": "bvivhedge/1.0"})
            with urlopen(req, timeout=60) as resp:
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


def fetch_binance_klines(start: str, end: str, symbol: str = "BTCUSDT", market: str = "spot",
                         until: str | None = None) -> pd.DataFrame:
    """15m klines from the monthly bulk archive, ``start``/``end`` as 'YYYY-MM' (inclusive).

    ``until`` (a date, exclusive) stops the daily fallback at the end of the sample, so a month whose
    daily files are all cached is read without touching the network.
    """
    base = {"spot": "spot", "perp": "futures/um"}[market]
    frames = []
    stop = pd.Timestamp.now(tz="UTC").floor("1D")
    if until is not None:
        stop = min(stop, pd.Timestamp(until, tz="UTC"))
    for month in pd.period_range(start, end, freq="M"):
        fname = f"{symbol}-15m-{month.year}-{month.month:02d}.zip"
        url = f"https://data.binance.vision/data/{base}/monthly/klines/{symbol}/15m/{fname}"
        days = [d for d in pd.date_range(month.start_time, month.end_time.normalize(), freq="D")
                if d.tz_localize("UTC") < stop]
        if not days:                                            # the month lies entirely after the sample
            continue
        daily_names = [f"binance_{market}_{symbol}-15m-{d:%Y-%m-%d}.zip" for d in days]
        cached_daily = not (RAW / f"binance_{market}_{fname}").exists() and all((RAW / n).exists() for n in daily_names)
        try:
            if cached_daily:
                raise FileNotFoundError("use the cached daily files")
            blobs = [_cached(f"binance_{market}_{fname}", lambda u=url: _get(u, retries=1))]
        except Exception:
            # the monthly file appears a few days after month end: fall back to daily files
            blobs = []
            for day in days:
                dname = f"{symbol}-15m-{day:%Y-%m-%d}.zip"
                durl = f"https://data.binance.vision/data/{base}/daily/klines/{symbol}/15m/{dname}"
                blobs.append(_cached(f"binance_{market}_{dname}", lambda u=durl: _get(u)))
        for blob in blobs:
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                frames.append(parse_binance_klines(zf.read(zf.namelist()[0])))
    return regularise_klines(pd.concat(frames))


def regularise_klines(klines: pd.DataFrame) -> pd.DataFrame:
    """Sorted, de-duplicated, gap-free 15m klines: a missing bar is flat at the last close with zero volume."""
    out = klines.sort_index()
    out = out[~out.index.duplicated()].asfreq(BAR)
    out["close"] = out["close"].ffill()
    for col in ("open", "high", "low"):
        out[col] = out[col].fillna(out["close"])
    out[["volume", "quote_volume"]] = out[["volume", "quote_volume"]].fillna(0.0)
    return out


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
    now = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)     # only a window that has ended is cached
    parts = []
    while t0 < t1:
        url = f"https://api-pub.bitfinex.com/v2/{path}?start={t0}&end={t1}&limit={limit}&sort=1"
        name = f"bitfinex_{path.replace('/', '_').replace(':', '_')}_{t0}_{t1}.json"
        df = parser(_cached(name, lambda u=url: _get(u)) if t1 <= now else _get(url))
        if df.empty:
            break
        parts.append(df)
        t0 = int(df.index[-1].timestamp() * 1000) + 1
        time.sleep(1.0)  # public rate limit
    return pd.concat(parts).sort_index() if parts else pd.DataFrame()


def fetch_bitfinex_candles(symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    """Trade candles (only intervals with at least one trade), ``end`` exclusive."""
    end_ms = pd.Timestamp(end, tz="UTC") - pd.Timedelta("1ms")
    return _bitfinex_paged(f"candles/trade:{timeframe}:{symbol}/hist", start, str(end_ms), parse_bitfinex_candles, 10_000)


def fetch_bitfinex_bviv(start: str, end: str) -> pd.DataFrame:
    """15m BVIV perpetual: trade close, index, mark and funding (8h rate, fraction)."""
    candles = _bitfinex_paged(f"candles/trade:15m:{BITFINEX_SYMBOL}/hist", start, end, parse_bitfinex_candles, 10_000)
    status = _bitfinex_paged(f"status/deriv/{BITFINEX_SYMBOL}/hist", start, end, parse_bitfinex_status, 5_000)
    status15 = status.resample(BAR, label="left", closed="left").last()
    out = pd.DataFrame({"perp_close": candles["close"]}).join(status15, how="outer")
    return out[~out.index.duplicated()].sort_index()


STATUS_COLS = {"deriv_price": 2, "index": 3, "next_funding_mts": 7, "accrued": 8, "step": 9,
               "current_funding": 11, "mark": 14, "open_interest": 17}


def parse_bitfinex_status_raw(payload: bytes) -> pd.DataFrame:
    """Every field of /v2/status/deriv/{key}/hist that the study uses, one row per snapshot."""
    rows = json.loads(payload)
    data = {k: [r[i] if len(r) > i else None for r in rows] for k, i in STATUS_COLS.items()}
    df = pd.DataFrame(data, index=pd.to_datetime([r[0] for r in rows], unit="ms", utc=True)).astype(float)
    return df.sort_index()


def fetch_bitfinex_status(symbol: str, start: str, end: str, pause: float = 0.7) -> pd.DataFrame:
    """Full derivatives-status history (about one snapshot a minute), cached per month as csv.gz."""
    parts = []
    for month in pd.period_range(pd.Timestamp(start).to_period("M"), pd.Timestamp(end).to_period("M"), freq="M"):
        m0 = max(month.start_time.tz_localize("UTC"), pd.Timestamp(start, tz="UTC"))
        m1 = min((month + 1).start_time.tz_localize("UTC"), pd.Timestamp(end, tz="UTC"))
        head = "" if m0 == month.start_time.tz_localize("UTC") else f"_from_{m0:%Y-%m-%d}"
        partial = "" if m1 == (month + 1).start_time.tz_localize("UTC") else f"_to_{m1:%Y-%m-%d}"
        path = RAW / f"bitfinex_status_{symbol.replace(':', '_')}_{month}{head}{partial}.csv.gz"
        complete = m1 <= pd.Timestamp.now(tz="UTC")             # the window has ended: its history is final
        if path.exists():
            parts.append(pd.read_csv(path, index_col=0, parse_dates=True))
            continue
        t0, t1, chunks = int(m0.timestamp() * 1000), int(m1.timestamp() * 1000), []
        while t0 < t1:
            url = f"https://api-pub.bitfinex.com/v2/status/deriv/{symbol}/hist?start={t0}&end={t1 - 1}&limit=5000&sort=1"
            df = parse_bitfinex_status_raw(_get(url))
            time.sleep(pause)                                   # public rate limit
            if df.empty:
                break
            chunks.append(df)
            t0 = int(df.index[-1].timestamp() * 1000) + 1
        if not chunks:
            continue
        month_df = pd.concat(chunks)
        if complete:
            path.parent.mkdir(parents=True, exist_ok=True)
            month_df.to_csv(path, compression="gzip")
        parts.append(month_df)
    out = pd.concat(parts).sort_index()
    out.index = pd.to_datetime(out.index, utc=True)
    return out


def bitfinex_funding_events(status: pd.DataFrame) -> pd.DataFrame:
    """Funding settlements: event time, the rate applied (CURRENT_FUNDING just after the event),
    the premium accrued just before it, and the mark price at settlement.

    Bitfinex applies sign(a) * min(max(|a| - 0.05%, 0), 0.25%) per 8h to the accrued premium a.
    """
    nxt = pd.to_datetime(status["next_funding_mts"], unit="ms", utc=True)
    snap = status.assign(_event=nxt.to_numpy())
    snap = snap[snap.index < snap["_event"]]
    before = snap.groupby("_event")["accrued"].last()               # last snapshot of each funding period
    events = before.index
    pos = status.index.searchsorted(events)                         # first snapshot at or after the event
    ok = pos < len(status)
    events, pos, accrued = events[ok], pos[ok], before.to_numpy()[ok]
    gap = status.index[pos] - events
    keep = np.asarray(gap <= pd.Timedelta("30min"))
    out = pd.DataFrame({"rate": status["current_funding"].to_numpy()[pos][keep],
                        "accrued": accrued[keep], "mark": status["mark"].to_numpy()[pos][keep]},
                       index=pd.DatetimeIndex(events[keep], name="time"))
    return out


def fetch_volmex_public(start: str, end: str, symbol: str = "BVIV") -> pd.Series:
    """Official Volmex index at 15m from the public history endpoint (no key), cached per month."""
    parts = []
    now = pd.Timestamp.now(tz="UTC")
    for month in pd.period_range(pd.Timestamp(start).to_period("M"), pd.Timestamp(end).to_period("M"), freq="M"):
        m1 = min((month + 1).start_time.tz_localize("UTC"), pd.Timestamp(end, tz="UTC"))
        t0, t1 = int(month.start_time.tz_localize("UTC").timestamp()), int(min(m1, now).timestamp())
        url = f"https://rest-v1.volmex.finance/public/iv/history?symbol={symbol}&resolution=15&from={t0}&to={t1}"
        partial = "" if m1 == (month + 1).start_time.tz_localize("UTC") else f"_to_{m1:%Y-%m-%d}"
        name = f"volmex_public_{symbol}_15_{month}{partial}.json"
        blob = _cached(name, lambda u=url: _get(u)) if m1 <= now else _get(url)
        parts.append(parse_volmex_history(blob))
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated()]


def parse_deribit_funding(payload: bytes) -> pd.Series:
    rows = json.loads(payload)["result"]
    return pd.Series([float(r["interest_1h"]) for r in rows],
                     index=pd.to_datetime([int(r["timestamp"]) for r in rows], unit="ms", utc=True), dtype=float).sort_index()


def fetch_deribit_funding(start: str, end: str, instrument: str = "BTC-PERPETUAL") -> pd.Series:
    """Hourly realised funding of the Deribit BTC perpetual (fraction of notional per hour), ``end`` exclusive."""
    parts = []
    stop = pd.Timestamp(end)
    for month in pd.period_range(pd.Timestamp(start).to_period("M"), stop.to_period("M"), freq="M"):
        for half in (0, 1):                                    # the endpoint caps the number of rows
            a = month.start_time + pd.Timedelta(days=15 * half)
            full = month.start_time + pd.Timedelta(days=15) if half == 0 else (month + 1).start_time
            if a >= stop:                                       # the half-month lies after the sample
                continue
            b = min(full, stop)
            t0, t1 = int(a.tz_localize("UTC").timestamp() * 1000), int(b.tz_localize("UTC").timestamp() * 1000)
            url = (f"https://www.deribit.com/api/v2/public/get_funding_rate_history?instrument_name={instrument}"
                   f"&start_timestamp={t0}&end_timestamp={t1}")
            name = f"deribit_funding_{instrument}_{t0}" + ("" if b == full else f"_to_{t1}") + ".json"
            complete = b.tz_localize("UTC") <= pd.Timestamp.now(tz="UTC")
            blob = _cached(name, lambda u=url: _get(u)) if complete else _get(url)
            parts.append(parse_deribit_funding(blob))
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated()]


# --------------------------------------------------------------------------- Hyperliquid BVIV perpetual
HYPERLIQUID_INFO = "https://api.hyperliquid.xyz/info"
HYPERLIQUID_COIN = "mkts:BVIV"   # HIP-3 markets are addressed as "<dex>:<coin>"


def parse_hyperliquid_candles(payload: bytes) -> pd.DataFrame:
    """candleSnapshot rows {t, T, s, i, o, c, h, l, v, n} -> OHLCV indexed by open time."""
    rows = json.loads(payload)
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"], dtype=float)
    df = pd.DataFrame(rows)
    out = df[["o", "h", "l", "c", "v"]].astype(float).set_axis(["open", "high", "low", "close", "volume"], axis=1)
    out.index = pd.to_datetime(df["t"].astype("int64"), unit="ms", utc=True)
    return out.sort_index()


def parse_hyperliquid_funding(payload: bytes) -> pd.Series:
    """fundingHistory rows {coin, fundingRate, premium, time}: hourly rate on notional."""
    rows = json.loads(payload)
    return pd.Series([float(r["fundingRate"]) for r in rows],
                     index=pd.to_datetime([int(r["time"]) for r in rows], unit="ms", utc=True), dtype=float).sort_index()


def fetch_hyperliquid_bviv(start: str, end: str, coin: str = HYPERLIQUID_COIN) -> pd.DataFrame:
    """15m BVIV perpetual on Hyperliquid in the same schema as :func:`fetch_bitfinex_bviv`.

    The API has no historical oracle series, so ``index`` and ``mark`` are the
    perpetual's own close (funding keeps it within a small premium of the index).
    Hourly funding is expressed as an 8h-equivalent rate (x8) for the shared assembly.
    """
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    t1 = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    candles, cursor = [], t0
    while cursor < t1:
        body = {"type": "candleSnapshot", "req": {"coin": coin, "interval": "15m", "startTime": cursor, "endTime": t1}}
        df = parse_hyperliquid_candles(_post_json(HYPERLIQUID_INFO, body))
        if df.empty:
            break
        candles.append(df)
        cursor = int(df.index[-1].timestamp() * 1000) + 1
    funding, cursor = [], t0
    while cursor < t1:
        body = {"type": "fundingHistory", "coin": coin, "startTime": cursor, "endTime": t1}
        s = parse_hyperliquid_funding(_post_json(HYPERLIQUID_INFO, body))
        if s.empty:
            break
        funding.append(s)
        cursor = int(s.index[-1].timestamp() * 1000) + 1
    if not candles:
        raise RuntimeError(f"no Hyperliquid candles for {coin} between {start} and {end}")
    c = pd.concat(candles)
    c = c[~c.index.duplicated()].sort_index()
    out = pd.DataFrame({"perp_close": c["close"], "mark": c["close"], "index": c["close"]})
    if funding:
        f = pd.concat(funding)
        f = f[~f.index.duplicated()].sort_index()
        # an hourly rate settled at T accrues over (T - 1h, T]
        f.index = f.index.floor(BAR) - pd.Timedelta("1h")
        out["funding_8h"] = 8.0 * f.reindex(out.index, method="ffill")
    else:
        out["funding_8h"] = np.nan
    return out


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
    current = pd.Timestamp.now(tz="UTC").tz_localize(None).to_period("M")
    for month in pd.period_range(start, end, freq="M"):
        t0 = int(month.start_time.tz_localize("UTC").timestamp())
        t1 = int(month.end_time.tz_localize("UTC").timestamp())
        url = (f"https://rest-v1.volmex.finance/v2/history?symbol={symbol}&resolution=15&from={t0}&to={t1}"
               + (f"&apiKey={key}" if key else ""))
        name = f"volmex_{symbol}_15_{month}_{'key' if key else 'nokey'}.json"
        # never cache a month that is still in progress
        blob = _get(url) if month >= current else _cached(name, lambda u=url: _get(u))
        parts.append(parse_volmex_history(blob))
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
        s, cursor = parse_deribit_dvol(_cached(f"deribit_dvol_{currency}_{t0}_{cursor}.json", lambda u=url: _get(u)))
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
    lo, hi = index_15m.first_valid_index(), index_15m.last_valid_index()
    bars = klines.loc[lo:hi].copy()                  # never extrapolate the index beyond its coverage
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
        # a rate settled at T accrues over (T - 8h, T]: stamp it at the start of its interval
        accrual = btc_funding_8h.copy()
        accrual.index = accrual.index.floor(BAR) - pd.Timedelta("8h")
        rate = accrual.sort_index().reindex(bars.index, method="ffill").fillna(0.0)
        bars["btc_funding"] = rate * 3 * 365          # annualised
    else:
        bars["btc_funding"] = 0.0
    return bars.dropna(subset=["close", "bviv"])
