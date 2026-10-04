"""Download and cache every live input of the study (idempotent; reruns only fetch what is missing).

    python scripts/fetch_data.py --start 2023-01-01 --end 2026-10-04
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bvivhedge import data  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--perp-start", default="2024-04-01")
    ap.add_argument("--end", default="2026-10-04")
    ap.add_argument("--only", nargs="*", default=["binance", "volmex", "bitfinex", "btcperp", "candles", "deribit"])
    a = ap.parse_args()
    t = time.time()
    if "volmex" in a.only:
        s = data.fetch_volmex_public(a.start, a.end)
        print(f"volmex BVIV: {len(s)} bars {s.index[0]} .. {s.index[-1]}  ({time.time() - t:.0f}s)", flush=True)
    if "binance" in a.only:
        k = data.fetch_binance_klines(a.start[:7], a.end[:7])
        print(f"binance BTCUSDT 15m: {len(k)} bars {k.index[0]} .. {k.index[-1]}  ({time.time() - t:.0f}s)", flush=True)
    if "deribit" in a.only:
        f = data.fetch_deribit_funding(a.start, a.end)
        print(f"deribit BTC-PERPETUAL funding: {len(f)} hours  ({time.time() - t:.0f}s)", flush=True)
    if "bitfinex" in a.only:
        st = data.fetch_bitfinex_status("tBVIVF0:USTF0", a.perp_start, a.end)
        ev = data.bitfinex_funding_events(st)
        print(f"bitfinex BVIV status: {len(st)} snapshots, {len(ev)} funding events  ({time.time() - t:.0f}s)", flush=True)
    if "candles" in a.only:
        for tf in ("1D", "15m"):
            c = data.fetch_bitfinex_candles("tBVIVF0:USTF0", tf, a.perp_start, a.end)
            print(f"bitfinex BVIV trade candles {tf}: {len(c)} intervals with trades  ({time.time() - t:.0f}s)", flush=True)
    if "btcperp" in a.only:
        st = data.fetch_bitfinex_status("tBTCF0:USTF0", a.perp_start, a.end)
        print(f"bitfinex BTC perp status: {len(st)} snapshots  ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
