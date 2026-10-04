"""The study on live data: Binance BTC, the official Volmex BVIV index and the Bitfinex BVIV perpetual.

    python scripts/fetch_data.py                 # download and cache every input (data/raw, not tracked)
    python scripts/run_live.py                   # forecasts, hedging, placebos, bootstrap -> results/live/

Hedging rules are frozen: they were selected on simulated training paths (results/selection.json),
so the live sample is an out-of-sample evaluation of the rules.

Execution baseline: a decision at the close of a 15-minute bar is filled at the close of the next bar,
at the Bitfinex mark plus fees and slippage.  Same-bar fills would collect the index's predictable
catch-up after BTC moves (the index lags), and the quoted mid is not executable as such (dust at the
top of the book, no trade on 96% of bars); both are reported as sensitivities.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from bvivhedge import data, live  # noqa: E402
from bvivhedge.experiments import Protocol, analyse, rule_from_key, rule_grid  # noqa: E402
from bvivhedge.hedge import HedgeConfig, Rule, run_rules  # noqa: E402
from bvivhedge.metrics import book_metrics, daily_book, expected_shortfall  # noqa: E402
from bvivhedge.realized import daily_realized  # noqa: E402

OUT = ROOT / "results" / "live"
START, END = "2023-01-01", "2026-10-04"     # END is exclusive: the sample ends with 3 October 2026
PLACEBO_STEP = 7                            # days between the selected ratchet's placebo shifts
GRID_STEP = 14                              # days between placebo shifts for the 72-rule grid
MID_BAND = 0.05                             # mid-fill sensitivity: quotes beyond +-5% of the mark are not credible
BOOK_COST_BPS = 50.0                        # per side, before the exchange fee: the dated book's price impact for one BTC's hedge
BLOCKS = (1, 2, 5, 10, 20, 30, 40)          # mean block lengths (days) for the bootstrap sensitivity


def es_red(r: np.ndarray, u: np.ndarray) -> float:
    return 100 * (1 - expected_shortfall(r) / expected_shortfall(u))


def var_red(r: np.ndarray, u: np.ndarray) -> float:
    return 100 * (1 - r.var(ddof=1) / u.var(ddof=1))


def placebo_rule(base: Rule, days: int, name: str | None = None) -> Rule:
    return Rule(name or f"placebo|{days}", "ratchet", z_enter=base.z_enter, halflife_days=base.halflife_days,
                floor=base.floor, use_forecast=base.use_forecast, placebo_shift_days=float(days))


def bar_close(t: pd.Timestamp) -> str:
    return f"{t + pd.Timedelta('15min'):%H:%M}"


# --------------------------------------------------------------------------- descriptive facts
def stylised_facts(bars: pd.DataFrame, daily: pd.DataFrame, eval_start: pd.Timestamp) -> dict:
    iv_d = bars["bviv"].resample("1D").last().reindex(daily.index)
    rv30 = daily["rv"][::-1].rolling(30).mean()[::-1].shift(-1)
    gap = iv_d - np.sqrt(rv30 * 365) * 100
    r = np.log(bars["close"]).diff()
    d_iv = bars["bviv"].diff()
    out = {
        "days": len(daily), "start": str(daily.index[0].date()), "end": str(daily.index[-1].date()),
        "rv": float(np.sqrt(daily["rv"].mean() * 365) * 100), "iv": float(iv_d.mean()),
        "iv_min": float(iv_d.min()), "iv_max": float(iv_d.max()), "iv_max_15m": float(bars["bviv"].max()),
        "vrp": float(gap.mean()), "iv_above_rv": float(100 * (gap.dropna() > 0).mean()),
        "corr_15m": float(r.corr(d_iv)), "corr_down": float(r[r < 0].corr(d_iv[r < 0])),
        "corr_up": float(r[r > 0].corr(d_iv[r > 0])), "corr_daily": float(daily["ret"].corr(iv_d.diff())),
        "kurt_daily": float(daily["ret"].kurt()), "worst_day": float(-100 * daily["ret"].min()),
        "ac1_15m_iv": float(d_iv.autocorr(1)),
    }
    # the index lags BTC: how much of its response to a 15m return arrives in the three bars after it
    rr, dd = r.loc[eval_start:], bars["bviv_mark"].diff().loc[eval_start:]
    slope = [float(np.polyfit(rr.iloc[1:-3].to_numpy(), dd.shift(-k).iloc[1:-3].to_numpy(), 1)[0]) for k in range(4)]
    out["lag_corr"] = [float(rr.corr(dd.shift(-k))) for k in range(4)]
    out["lag_share_after"] = float(100 * sum(slope[1:]) / sum(slope))
    by_year = {}
    for y in sorted(set(daily.index.year)):
        m = daily.index.year == y
        by_year[str(y)] = {"rv": float(np.sqrt(daily["rv"][m].mean() * 365) * 100), "iv": float(iv_d[m].mean()),
                           "corr_daily": float(daily["ret"][m].corr(iv_d.diff()[m])), "days": int(m.sum())}
    out["by_year"] = by_year
    return out


def perp_facts(panel: dict, candles_1d: pd.DataFrame, candles_15m: pd.DataFrame) -> dict:
    bars, status, start = panel["bars"], panel["status"], panel["perp_start"]
    live_bars = bars.loc[panel["first_live"]:]
    premium = (live_bars["bviv_mid"] / live_bars["bviv_mark"] - 1).dropna()
    days = pd.date_range(start, live_bars.index[-1].floor("1D"), freq="D")
    vol = candles_1d["volume"].reindex(days).fillna(0.0)
    notional = vol * candles_1d["close"].reindex(days).ffill()
    oi = status["open_interest"].resample("1D").last().reindex(days)
    index_gap = (live_bars["bviv_mark"] / live_bars["bviv"] - 1).dropna()
    return {
        "first_day": str(start.date()), "first_live": str(panel["first_live"]), "days": len(days),
        "days_traded": int((vol > 0).sum()), "median_daily_contracts": float(vol.median()),
        "median_daily_usd": float(notional.median()), "mean_daily_usd": float(notional.mean()),
        "max_oi_contracts": float(oi.max()),
        "bars": len(live_bars), "bars_traded": int(candles_15m.index.isin(live_bars.index).sum()),
        "median_premium_pct": float(100 * premium.median()), "mean_premium_pct": float(100 * premium.mean()),
        "share_mid_beyond_5pct": float(100 * (premium.abs() > MID_BAND).mean()),
        "max_mid_premium_pct": float(100 * premium.max()),
        "mark_vs_volmex_pct_median": float(100 * index_gap.median()),
        "mark_vs_volmex_pct_p95": float(100 * index_gap.abs().quantile(0.95)),
    }


def funding_fit(events: pd.DataFrame) -> dict:
    """How well sign(a) min(max(|a| - 0.05%, 0), 0.25%) of the accrued premium a reproduces the settled rates."""
    a = events["accrued"]
    rec = np.sign(a) * np.minimum(np.maximum(a.abs() - 0.0005, 0.0), 0.0025)
    err = (rec - events["rate"]).abs()
    return {"corr": float(np.corrcoef(rec, events["rate"])[0, 1]), "max_err_bp": float(1e4 * err.max()),
            "share_exact": float(100 * (err < 1e-7).mean()), "n": len(events)}


def book_facts(contracts: float) -> dict:
    """Depth of the perpetual's order book from one dated snapshot (cached: the book has no public history).

    The mid is taken from the first levels with at least one contract on each side (the top of the book
    often holds dust), and one BTC's hedge is bought and sold by walking the book.
    """
    path = OUT / "book_snapshot.json"
    if not path.exists():
        snap = {"time": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
                "book": json.loads(data._get(f"https://api-pub.bitfinex.com/v2/book/{live.BVIV_PERP}/P0?len=25"))}
        path.write_text(json.dumps(snap))
    snap = json.loads(path.read_text())
    book = np.array(snap["book"], dtype=float)                     # [price, count, amount]; asks have amount < 0
    bids = book[book[:, 2] > 0][:, [0, 2]]
    asks = np.abs(book[book[:, 2] < 0][:, [0, 2]])
    bids, asks = bids[np.argsort(-bids[:, 0])], asks[np.argsort(asks[:, 0])]
    mid = (bids[bids[:, 1] >= 1, 0].max() + asks[asks[:, 1] >= 1, 0].min()) / 2

    def walk(side: np.ndarray) -> float:
        if side[:, 1].sum() < contracts:                           # the snapshot's levels cannot fill the order
            return float("nan")
        filled = np.minimum(np.cumsum(side[:, 1]), contracts)
        return float(np.diff(np.r_[0.0, filled]) @ side[:, 0] / contracts)

    ask_depth = asks[asks[:, 0] <= mid * 1.025, 1].sum()
    bid_depth = bids[bids[:, 0] >= mid * 0.975, 1].sum()
    return {"time": snap["time"][:10], "mid": float(mid),
            "top_spread_pct": float(100 * (asks[0, 0] - bids[0, 0]) / mid),
            "size_spread_pct": float(100 * (asks[asks[:, 1] >= 1, 0].min() - bids[bids[:, 1] >= 1, 0].max()) / mid),
            "buy_cost_pct": float(100 * (walk(asks) / mid - 1)), "sell_cost_pct": float(100 * (1 - walk(bids) / mid)),
            "ask_depth_2_5pct": float(ask_depth), "bid_depth_2_5pct": float(bid_depth),
            "btc_capacity_2_5pct": float(min(ask_depth, bid_depth) / contracts)}


# --------------------------------------------------------------------------- hedging
def daily_books(results: dict, start: pd.Timestamp) -> dict:
    return {k: daily_book(v).loc[start:] for k, v in results.items()}


def run_books(bars: pd.DataFrame, cfg: HedgeConfig, rules: list[Rule], sig: pd.DataFrame, start: pd.Timestamp,
              chunk: int = 16) -> dict:
    """Daily books of many rules, run in chunks so the 15-minute P&L of only a few is in memory at once."""
    books = {}
    for i in range(0, len(rules), chunk):
        res, _ = run_rules(bars, cfg, [Rule("Unhedged", "unhedged"), *rules[i:i + chunk]], signals=sig)
        books.update(daily_books(res, start))
    return books


def _grid_job(job):
    bars, cfg, sig, rules, start = job
    books = run_books(bars, cfg, rules, sig, start)
    return {k: float(es_red(b["total"].to_numpy(), books["Unhedged"]["total"].to_numpy()))
            for k, b in books.items() if k != "Unhedged"}


def inference(books: dict, rt: str, sw: str, placebo_keys: list[str], block: float = 10.0) -> dict:
    """Placebo rank of the ratchet and paired block-bootstrap CIs on one live path."""
    u = books["Unhedged"]["total"].to_numpy()
    real = es_red(books[rt]["total"].to_numpy(), u)
    fake = np.array([es_red(books[k]["total"].to_numpy(), u) for k in placebo_keys])
    out = {"placebo_beaten": float(100 * (real > fake).mean()), "placebo_mean": float(fake.mean())}
    for a, b in ((rt, "always|1"), (sw, "always|1")):
        out[f"{a} - {b}"] = live.paired_block_bootstrap(books, "Unhedged", es_red, a, b, block=block)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fee", type=float, default=6.0)
    ap.add_argument("--slippage", type=float, default=10.0)
    ap.add_argument("--no-grid", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    selection = json.loads((ROOT / "results" / "selection.json").read_text())
    sw, rt = selection["switch"], selection["ratchet"]

    panel = live.build_panel(START, END)
    bars, perp_start = panel["bars"], panel["perp_start"]
    days_to_perp = int((perp_start - bars.index[0]).days)
    cfg = HedgeConfig(fee_bps=args.fee, slippage_bps=args.slippage, warmup_days=days_to_perp, exec_delay=1,
                      warmup_bars=int(bars.index.get_loc(panel["first_live"])))   # no position before the listing
    proto = Protocol(hedge_eval_start=days_to_perp, forecast_eval_start=365, full_forecasts=True)
    candles_1d = data.fetch_bitfinex_candles(live.BVIV_PERP, "1D", "2024-04-01", END)
    candles_15m = data.fetch_bitfinex_candles(live.BVIV_PERP, "15m", "2024-04-01", END)
    print(f"panel {bars.index[0]} .. {bars.index[-1]}, live from {panel['first_live']}, "
          f"{len(panel['events'])} funding events, BTC funding: {panel['btc_funding_source']}", flush=True)

    # ---- facts
    daily = daily_realized(bars)
    eval_days = len(daily) - days_to_perp
    events = panel["events"]
    facts = {"stylised": stylised_facts(bars, daily, perp_start), "perp": perp_facts(panel, candles_1d, candles_15m),
             "funding": live.funding_summary(events), "funding_fit": funding_fit(events),
             "funding_by_year": {str(y): live.funding_summary(g) for y, g in events.groupby(events.index.year)},
             "btc_funding_source": panel["btc_funding_source"],
             "baseline": {"exec_delay_bars": cfg.exec_delay, "fills": "mark", "fee_bps": args.fee,
                          "slippage_bps": args.slippage}}

    # ---- forecasts + frozen rules + placebos on the live path (baseline execution)
    base_rt = rule_from_key(rt)
    shifts = list(range(PLACEBO_STEP, eval_days - 7, PLACEBO_STEP))
    placebos = [placebo_rule(base_rt, d) for d in shifts]
    rules = [Rule("Unhedged", "unhedged"), rule_from_key("always|1"), rule_from_key("always|1.5"),
             rule_from_key(sw), base_rt]
    out = analyse(bars, cfg, rules, proto)
    start, sig = out["hedge_start"], out["signals"]
    metrics = out["metrics"]
    metrics.to_csv(OUT / "metrics.csv")
    placebo_books = run_books(bars, cfg, placebos, sig, start)

    # ---- untimed benchmarks for the size term: the MV hedge scaled to the placebos' average notional,
    # and the ratchet's overlay held permanently (o_t = 1, i.e. a trigger on every bar)
    pl_notional = np.mean([book_metrics(placebo_books[p.name], placebo_books["Unhedged"])["notional"] for p in placebos])
    scale = float(pl_notional / metrics.loc["always|1", "notional"])
    bench = [Rule("always|matched", "always", scale=scale),
             Rule("overlay|held", "ratchet", z_enter=np.inf, halflife_days=base_rt.halflife_days, floor=base_rt.floor,
                  use_forecast=base_rt.use_forecast)]
    bench_books = run_books(bars, cfg, bench, sig, start)
    facts["benchmarks"] = {"matched_scale": scale, "placebo_notional": float(pl_notional),
                           **{r.name: book_metrics(bench_books[r.name], bench_books["Unhedged"]) for r in bench}}
    evals = pd.concat([ev.assign(h=h).rename_axis("model").reset_index() for h, ev in out["evals"].items()])
    evals.to_csv(OUT / "forecast_evals.csv", index=False)
    facts["forecast_n"] = {str(h): int(ev.attrs.get("n", 0)) for h, ev in out["evals"].items()}
    print(metrics.loc[[k for k in metrics.index if not k.startswith("placebo|")],
                      ["es_red", "var_red", "mdd", "hedge_pnl", "funding", "cost", "turnover", "notional"]].round(2), flush=True)

    # ---- single-path inference: paired stationary block bootstrap over days
    books = {k: v.loc[start:] for k, v in out["books"].items()}
    boot = {}
    for a, b in ((rt, "always|1"), (rt, "always|1.5"), (sw, "always|1"), ("always|1", "Unhedged"), ("always|1.5", "always|1")):
        for name, fn in (("es", es_red), ("var", var_red)):
            boot[f"{a} - {b} ({name})"] = live.paired_block_bootstrap(books, "Unhedged", fn, a, b)
    for block in BLOCKS:
        boot[f"always|1 - Unhedged (es, block {block:g})"] = live.paired_block_bootstrap(
            books, "Unhedged", es_red, "always|1", "Unhedged", block=float(block))
        boot[f"{rt} - always|1 (es, block {block:g})"] = live.paired_block_bootstrap(
            books, "Unhedged", es_red, rt, "always|1", block=float(block))
    (OUT / "bootstrap.json").write_text(json.dumps(boot, indent=2))

    # ---- placebo distribution for the selected ratchet
    pl = pd.DataFrame({k: book_metrics(placebo_books[k], placebo_books["Unhedged"]) for k in (p.name for p in placebos)}).T
    pl = pl[["es_red", "var_red", "hedge_cost"]]
    pl["shift_days"] = [int(k.split("|")[1]) for k in pl.index]
    pl.to_csv(OUT / "placebo.csv")
    real = metrics.loc[rt, "es_red"]
    facts["placebo"] = {"n": len(pl), "real": float(real), "mean": float(pl["es_red"].mean()),
                        "share_beaten": float(100 * (real > pl["es_red"]).mean()),
                        "p05": float(pl["es_red"].quantile(0.05)), "p95": float(pl["es_red"].quantile(0.95)),
                        "min": float(pl["es_red"].min()), "max": float(pl["es_red"].max()),
                        "step_days": PLACEBO_STEP}

    # ---- the same inference under the two execution assumptions the baseline avoids
    mark = bars["bviv_mark"]
    mid_fill = bars.assign(bviv_fill=bars["bviv_mid"].clip(mark * (1 - MID_BAND), mark * (1 + MID_BAND)))
    variants = {"baseline": inference({**books, **placebo_books}, rt, sw, list(pl.index))}
    for label, b, c in (("same-bar execution", bars, cfg.with_(exec_delay=0)),
                        ("fills at the quoted mid", mid_fill, cfg)):
        bk = run_books(b, c, [*rules[1:], *placebos], sig, start)
        variants[label] = {**inference(bk, rt, sw, list(pl.index)),
                           **{k: book_metrics(bk[k], bk["Unhedged"]) for k in ("always|1", "always|1.5", sw, rt)}}
    (OUT / "variants.json").write_text(json.dumps(variants, indent=2))

    # ---- the whole ratchet grid vs its placebos on the live path
    if not args.no_grid:
        grid = [r for r in rule_grid() if r.kind == "ratchet"]
        grid_shifts = list(range(GRID_STEP, eval_days - 7, GRID_STEP))
        jobs = [(bars, cfg, sig, [r] + [placebo_rule(r, d, f"{r.name}|shift{d}") for d in grid_shifts], start)
                for r in grid]
        with ProcessPoolExecutor(os.cpu_count() or 2) as ex:
            res = list(ex.map(_grid_job, jobs))
        rows = []
        for r, es in zip(grid, res):
            fake = np.array([es[f"{r.name}|shift{d}"] for d in grid_shifts])
            rows.append({"rule": r.name, "floor": r.floor, "z_enter": r.z_enter, "halflife": r.halflife_days,
                         "forecast": r.use_forecast, "es_red": es[r.name], "placebo_mean": fake.mean(),
                         "timing": es[r.name] - fake.mean(), "rank_pct": 100 * (es[r.name] > fake).mean()})
        pd.DataFrame(rows).to_csv(OUT / "grid.csv", index=False)
        facts["grid"] = {"shifts": len(grid_shifts), "step_days": GRID_STEP}

    # ---- sensitivities (frozen rules, one assumption at a time)
    sens = {}
    base_rules = rules
    for label, b, c in (("same-bar execution", bars, cfg.with_(exec_delay=0)),
                        ("execution 1 hour later", bars, cfg.with_(exec_delay=4)),
                        ("execution 1 day later", bars, cfg.with_(exec_delay=96)),
                        ("fills at the quoted mid", mid_fill, cfg),
                        ("book costs", bars, cfg.with_(slippage_bps=BOOK_COST_BPS)),
                        ("no trading costs", bars, cfg.with_(fee_bps=0, slippage_bps=0)),
                        ("funding ignored", bars.assign(bviv_funding=0.0, bviv_carry=0.0), cfg),
                        ("perpetual BTC book", bars, cfg.with_(instrument="perp"))):
        res, _ = run_rules(b, c, base_rules, signals=sig)
        bk = daily_books(res, start)
        sens[label] = {k: book_metrics(v, bk["Unhedged"]) for k, v in bk.items() if k != "Unhedged"}
    (OUT / "sensitivity.json").write_text(json.dumps(sens, indent=2))

    # ---- capacity: hedge contracts for a 1-BTC book against open interest and traded volume
    res_ev = {k: v.loc[start:] for k, v in out["results"].items()}
    pos, oi = res_ev["always|1"]["position"], bars["open_interest"].loc[start:]
    held, seen = pos > 0, oi.notna()
    day_vol = candles_1d["volume"].reindex(pd.date_range(start, bars.index[-1].floor("1D"), freq="D")).fillna(0.0)
    facts["capacity"] = {
        "share_held": float(100 * held.mean()), "median_contracts_per_btc": float(pos[held].median()),
        "median_oi": float(oi[seen].median()), "median_oi_when_held": float(oi[held & seen].median()),
        "share_above_oi_when_held": float(100 * (pos > oi)[held & seen].mean()),
        "share_above_oi_all": float(100 * (held & seen & (pos > oi)).sum() / seen.sum()),
        "max_notional_pct": float(100 * (res_ev["always|1"]["notional"] / res_ev["always|1"]["btc_notional"]).max()),
        "max_notional_pct_onehalf": float(100 * (res_ev["always|1.5"]["notional"] / res_ev["always|1.5"]["btc_notional"]).max()),
    }
    for k in ("always|1", rt, sw):
        traded = res_ev[k]["position"].diff().abs()
        days_traded = traded[traded > 0].index.floor("1D")
        facts["capacity"][f"{k}|trades_on_dead_days"] = float(100 * (day_vol.reindex(days_traded).to_numpy() == 0).mean())
    facts["book"] = book_facts(facts["capacity"]["median_contracts_per_btc"])

    # ---- where the always-on hedge earned its keep
    hedge, u = books["always|1"]["hedge"], books["Unhedged"]["total"]
    worst = u.nsmallest(10).index
    best5 = hedge.nlargest(5).index
    tail = u[u <= u.quantile(0.025)].index
    day = hedge.idxmax()

    def es_without(k: str, drop) -> float:
        return es_red(books[k]["total"].drop(drop).to_numpy(), u.drop(drop).to_numpy())

    def on(d: pd.Timestamp, s: pd.Series) -> pd.Series:
        return s.loc[d: d + pd.Timedelta("1D") - pd.Timedelta("15min")]

    facts["concentration"] = {
        "hedge_total_pct": float(100 * hedge.sum()), "top10_pct": float(100 * hedge.nlargest(10).sum()),
        "ex_top10_pct": float(100 * (hedge.sum() - hedge.nlargest(10).sum())),
        "worst10_btc_pct": float(100 * books["always|1"]["btc"].loc[worst].sum()),
        "worst10_hedge_pct": float(100 * hedge.loc[worst].sum()),
        "best_day": str(day.date()), "best_day_pct": float(100 * hedge.max()),
        "es_ex_best_day": es_without("always|1", [day]), "es_ex_best5": es_without("always|1", list(best5)),
        "es_ratchet_ex_best_day": es_without(rt, [day]),
        "tail_days": len(tail), "tail_days_flat": int((books["always|1"]["notional"].loc[tail] == 0).sum()),
        "best5": [{"day": str(d.date()), "max_contracts": float(on(d, pos).max()), "median_oi": float(on(d, oi).median())}
                  for d in best5],
        "best_day_traded_contracts": float(on(day, pos).diff().abs().sum()),
        "best_day_market_contracts": float(day_vol.get(day, 0.0)),
    }
    facts["es_by_year"] = {str(y): {k: es_red(books[k]["total"][u.index.year == y].to_numpy(), u[u.index.year == y].to_numpy())
                                    for k in ("always|1", rt)} for y in sorted(set(u.index.year))}
    facts["es_horizon"] = {str(h): {k: es_red(books[k]["total"].rolling(h).sum().dropna().to_numpy(),
                                              u.rolling(h).sum().dropna().to_numpy()) for k in ("always|1", rt, sw)}
                           for h in (1, 7, 14)}

    # ---- the best hedge day at 15-minute resolution (episode figure and text)
    win = slice(day - pd.Timedelta(days=2), day + pd.Timedelta(days=3) - pd.Timedelta(minutes=15))
    s = sig.loc[win]
    episode = pd.DataFrame({
        "close": bars["close"].loc[win], "vwap": s["vwap"],
        "band": s["vwap"] * np.exp(base_rt.z_enter * s["sigma_day"] / np.sqrt(3.0)), "z": s["z"],
        "bviv": bars["bviv"].loc[win], "bviv_mark": bars["bviv_mark"].loc[win], "bviv_mid": bars["bviv_mid"].loc[win],
    })
    for k in ("always|1", sw, rt):
        episode[f"{k}|position"] = out["results"][k]["position"].loc[win]
    episode.to_csv(OUT / "episode.csv")
    iv, z = on(day, bars["bviv"]), on(day, s["z"])
    trig = z.index[z < base_rt.z_enter][0]
    after = iv.loc[trig: trig + pd.Timedelta("45min")]
    pos_day = {k: on(day, out["results"][k]["position"]) for k in ("always|1", sw, rt)}
    later = pos_day["always|1"].index >= trig + pd.Timedelta("15min") * cfg.exec_delay
    ratio = (pos_day[rt] / pos_day["always|1"])[later]
    facts["episode"] = {
        "trigger_close": bar_close(trig), "iv_midnight": float(bars["bviv"].loc[:day].iloc[-2]),   # value at 00:00
        "iv_hour_before": float(iv.loc[trig - pd.Timedelta("1h")]), "iv_trigger": float(iv.loc[trig]),
        "iv_after_max": float(after.max()), "iv_after_time": bar_close(after.idxmax()),
        "iv_peak": float(iv.max()), "iv_peak_time": bar_close(iv.idxmax()),
        "btc_day_pct": float(100 * (on(day, bars["close"]).iloc[-1] / bars["close"].loc[:day].iloc[-2] - 1)),
        "contracts_open_min": float(min(p.iloc[0] for p in pos_day.values())),
        "contracts_open_max": float(max(p.iloc[0] for p in pos_day.values())),
        "contracts_close_always": float(pos_day["always|1"].iloc[-1]),
        "ratchet_over_always_min": float(100 * (ratio.min() - 1)), "ratchet_over_always_max": float(100 * (ratio.max() - 1)),
    }

    # ---- daily series for figures
    daily_out = pd.DataFrame({
        "btc": bars["close"].resample("1D").last(), "bviv": bars["bviv"].resample("1D").last(),
        "bviv_mark": bars["bviv_mark"].resample("1D").last(),
    })
    daily_out["funding_8h_mean"] = events["rate"].resample("1D").mean()
    for k in ("always|1", rt, "Unhedged"):
        b = books[k]
        daily_out[f"{k}|hedge_mtm"] = (b["hedge"] + b["funding"]).reindex(daily_out.index)
        daily_out[f"{k}|funding"] = b["funding"].reindex(daily_out.index)
        daily_out[f"{k}|cost"] = b["cost"].reindex(daily_out.index)
        daily_out[f"{k}|total"] = b["total"].reindex(daily_out.index)
    daily_out.to_csv(OUT / "daily.csv")
    facts["eval_days"] = int(len(books["Unhedged"]))
    facts["eval_start"], facts["eval_end"] = str(start.date()), str(books["Unhedged"].index[-1].date())
    facts["selection"] = selection
    (OUT / "facts.json").write_text(json.dumps(facts, indent=2, default=float))
    print(json.dumps({k: facts[k] for k in ("placebo", "capacity", "episode")}, indent=1, default=float))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, dict)} for k, v in variants.items()}, indent=1))


if __name__ == "__main__":
    main()
