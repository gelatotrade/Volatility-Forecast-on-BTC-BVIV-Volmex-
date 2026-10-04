"""The study on live data: Binance BTC, the official Volmex BVIV index and the Bitfinex BVIV perpetual.

    python scripts/fetch_data.py                 # download and cache every input (data/raw, not tracked)
    python scripts/run_live.py                   # forecasts, hedging, placebos, bootstrap -> results/live/

Hedging rules are frozen: they were selected on simulated training paths (results/selection.json),
so the live sample is a pure out-of-sample evaluation.
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
from bvivhedge.hedge import HedgeConfig, Rule, build_signals, run_rules  # noqa: E402
from bvivhedge.metrics import book_metrics, daily_book, expected_shortfall  # noqa: E402
from bvivhedge.realized import daily_realized  # noqa: E402

OUT = ROOT / "results" / "live"
START, END = "2023-01-01", "2026-10-04"
GRID_SHIFT_STEP = 35        # days between placebo shifts in the grid (multiples of a week)


def es_red(r: np.ndarray, u: np.ndarray) -> float:
    return 100 * (1 - expected_shortfall(r) / expected_shortfall(u))


def var_red(r: np.ndarray, u: np.ndarray) -> float:
    return 100 * (1 - r.var(ddof=1) / u.var(ddof=1))


def placebo_rule(base: Rule, days: int, name: str | None = None) -> Rule:
    return Rule(name or f"placebo|{days}", "ratchet", z_enter=base.z_enter, halflife_days=base.halflife_days,
                floor=base.floor, use_forecast=base.use_forecast, placebo_shift_days=float(days))


# --------------------------------------------------------------------------- descriptive facts
def stylised_facts(bars: pd.DataFrame, daily: pd.DataFrame) -> dict:
    iv_d = bars["bviv"].resample("1D").last().reindex(daily.index)
    rv30 = daily["rv"][::-1].rolling(30).mean()[::-1].shift(-1)
    gap = iv_d - np.sqrt(rv30 * 365) * 100
    r = np.log(bars["close"]).diff()
    d_iv = bars["bviv"].diff()
    out = {
        "days": len(daily), "start": str(daily.index[0].date()), "end": str(daily.index[-1].date()),
        "rv": float(np.sqrt(daily["rv"].mean() * 365) * 100), "iv": float(iv_d.mean()),
        "iv_min": float(iv_d.min()), "iv_max": float(iv_d.max()), "iv_p05": float(iv_d.quantile(0.05)),
        "iv_p95": float(iv_d.quantile(0.95)), "vrp": float(gap.mean()), "iv_above_rv": float(100 * (gap.dropna() > 0).mean()),
        "vrp_var": float(((iv_d / 100) ** 2 - rv30 * 365).mean()),
        "corr_15m": float(r.corr(d_iv)), "corr_down": float(r[r < 0].corr(d_iv[r < 0])),
        "corr_up": float(r[r > 0].corr(d_iv[r > 0])), "corr_daily": float(daily["ret"].corr(iv_d.diff())),
        "kurt_daily": float(daily["ret"].kurt()), "worst_day": float(-100 * daily["ret"].min()),
        "ac1_15m_iv": float(d_iv.autocorr(1)),
    }
    by_year = {}
    for y in sorted(set(daily.index.year)):
        m = daily.index.year == y
        by_year[str(y)] = {"rv": float(np.sqrt(daily["rv"][m].mean() * 365) * 100), "iv": float(iv_d[m].mean()),
                           "corr_daily": float(daily["ret"][m].corr(iv_d.diff()[m])), "days": int(m.sum())}
    out["by_year"] = by_year
    return out


def perp_facts(panel: dict) -> dict:
    bars, status, start = panel["bars"], panel["status"], panel["perp_start"]
    live_bars = bars.loc[start:]
    premium = (live_bars["bviv_mid"] / live_bars["bviv_mark"] - 1).dropna()
    candles = data.parse_bitfinex_candles(data._get(
        f"https://api-pub.bitfinex.com/v2/candles/trade:1D:{live.BVIV_PERP}/hist?limit=10000&sort=1"))
    days = pd.date_range(start.floor("1D"), live_bars.index[-1].floor("1D"), freq="D")
    vol = candles["volume"].reindex(days).fillna(0.0)
    notional = vol * candles["close"].reindex(days).ffill()
    oi = status["open_interest"].resample("1D").last().reindex(days)
    index_gap = (live_bars["bviv_mark"] / live_bars["bviv"] - 1).dropna()
    return {
        "first_day": str(start.date()), "days": len(days), "days_traded": int((vol > 0).sum()),
        "median_daily_contracts": float(vol.median()), "median_daily_usd": float(notional.median()),
        "mean_daily_usd": float(notional.mean()), "median_oi_contracts": float(oi.median()),
        "max_oi_contracts": float(oi.max()), "median_premium_pct": float(100 * premium.median()),
        "mean_premium_pct": float(100 * premium.mean()), "share_premium_positive": float(100 * (premium > 0).mean()),
        "mark_vs_volmex_pct_median": float(100 * index_gap.median()), "mark_vs_volmex_pct_p95": float(100 * index_gap.abs().quantile(0.95)),
    }


def book_facts(contracts_per_btc: float) -> dict:
    """Depth of the perpetual's order book from one dated snapshot (cached: the book has no public history)."""
    path = OUT / "book_snapshot.json"
    if not path.exists():
        snap = {"time": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
                "book": json.loads(data._get(f"https://api-pub.bitfinex.com/v2/book/{live.BVIV_PERP}/P0?len=25"))}
        path.write_text(json.dumps(snap))
    snap = json.loads(path.read_text())
    book = np.array(snap["book"], dtype=float)                     # [price, count, amount]; asks have amount < 0
    bids, asks = book[book[:, 2] > 0], book[book[:, 2] < 0]
    asks = asks[np.argsort(asks[:, 0])]
    bid, ask = bids[:, 0].max(), asks[:, 0].min()
    mid = (bid + ask) / 2
    near = asks[asks[:, 0] <= mid * 1.025]
    walk = np.minimum(np.cumsum(-asks[:, 2]), contracts_per_btc)   # buy one BTC's hedge by walking the asks
    fill = np.diff(np.r_[0.0, walk]) @ asks[:, 0] / contracts_per_btc
    return {"time": snap["time"][:10], "mid": float(mid), "spread_pct": float(100 * (ask - bid) / mid),
            "best_ask_contracts": float(-asks[0, 2]), "best_ask_pct": float(100 * (ask / mid - 1)),
            "ask_contracts_2_5pct": float(-near[:, 2].sum()), "btc_capacity_2_5pct": float(-near[:, 2].sum() / contracts_per_btc),
            "fill_cost_one_btc_pct": float(100 * (fill / mid - 1))}


# --------------------------------------------------------------------------- hedging
def _grid_job(job):
    bars, cfg, sig, rules, start = job
    results, _ = run_rules(bars, cfg, rules, signals=sig)
    books = {k: daily_book(v).loc[start:] for k, v in results.items()}
    return {k: float(es_red(b["total"].to_numpy(), books["Unhedged"]["total"].to_numpy()))
            for k, b in books.items() if k != "Unhedged"}


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
    cfg = HedgeConfig(fee_bps=args.fee, slippage_bps=args.slippage, warmup_days=days_to_perp)
    proto = Protocol(hedge_eval_start=days_to_perp, forecast_eval_start=365, full_forecasts=True)
    print(f"panel {bars.index[0]} .. {bars.index[-1]}, perp from {perp_start}, {len(panel['events'])} funding events", flush=True)

    # ---- facts
    daily = daily_realized(bars)
    facts = {"stylised": stylised_facts(bars, daily), "perp": perp_facts(panel),
             "funding": live.funding_summary(panel["events"], perp_start)}
    funding_by_year = {str(y): live.funding_summary(g) for y, g in panel["events"].loc[perp_start:].groupby(
        panel["events"].loc[perp_start:].index.year)}
    facts["funding_by_year"] = funding_by_year

    # ---- forecasts + frozen rules + placebos on the live path
    base_rt = rule_from_key(rt)
    shifts = list(range(7, len(daily) - 7, 7))
    placebos = [placebo_rule(base_rt, d) for d in shifts]
    rules = [Rule("Unhedged", "unhedged"), rule_from_key("always|1"), rule_from_key("always|1.5"),
             rule_from_key(sw), base_rt, *placebos]
    out = analyse(bars, cfg, rules, proto)
    start = out["hedge_start"]
    metrics = out["metrics"]
    metrics.to_csv(OUT / "metrics.csv")
    evals = pd.concat([ev.assign(h=h).rename_axis("model").reset_index() for h, ev in out["evals"].items()])
    evals.to_csv(OUT / "forecast_evals.csv", index=False)
    for h, ev in out["evals"].items():
        facts.setdefault("forecast_n", {})[str(h)] = int(ev.attrs.get("n", 0))
    print(metrics.loc[[k for k in metrics.index if not k.startswith("placebo|")],
                      ["es_red", "var_red", "mdd", "hedge_pnl", "funding", "cost", "turnover", "notional"]].round(2), flush=True)

    # ---- single-path inference: paired stationary block bootstrap over days
    books = {k: v.loc[start:] for k, v in out["books"].items()}
    boot = {}
    for a, b in ((rt, "always|1"), (rt, "always|1.5"), (sw, "always|1"), ("always|1", "Unhedged"), ("always|1.5", "always|1")):
        for name, fn in (("es", es_red), ("var", var_red)):
            boot[f"{a} - {b} ({name})"] = live.paired_block_bootstrap(books, "Unhedged", fn, a, b)
    (OUT / "bootstrap.json").write_text(json.dumps(boot, indent=2))

    # ---- placebo distribution for the selected ratchet
    pl = metrics.loc[metrics.index.str.startswith("placebo|"), ["es_red", "var_red", "hedge_cost"]].copy()
    pl["shift_days"] = [int(k.split("|")[1]) for k in pl.index]
    pl.to_csv(OUT / "placebo.csv")
    real = metrics.loc[rt, "es_red"]
    facts["placebo"] = {"n": len(pl), "real": float(real), "mean": float(pl["es_red"].mean()),
                        "share_beaten": float(100 * (real > pl["es_red"]).mean()),
                        "p05": float(pl["es_red"].quantile(0.05)), "p95": float(pl["es_red"].quantile(0.95))}

    # ---- the whole ratchet grid vs its placebos on the live path
    if not args.no_grid:
        sig = build_signals(bars, cfg, out["forecasts"][30]["HAR-IV"])
        grid = [r for r in rule_grid() if r.kind == "ratchet"]
        grid_shifts = list(range(GRID_SHIFT_STEP, len(daily) - 7, GRID_SHIFT_STEP))
        jobs = []
        for r in grid:
            rules_r = [Rule("Unhedged", "unhedged"), r] + [placebo_rule(r, d, f"{r.name}|shift{d}") for d in grid_shifts]
            jobs.append((bars, cfg, sig, rules_r, start))
        with ProcessPoolExecutor(os.cpu_count() or 2) as ex:
            res = list(ex.map(_grid_job, jobs))
        rows = []
        for r, es in zip(grid, res):
            fake = np.array([es[f"{r.name}|shift{d}"] for d in grid_shifts])
            rows.append({"rule": r.name, "floor": r.floor, "z_enter": r.z_enter, "halflife": r.halflife_days,
                         "forecast": r.use_forecast, "es_red": es[r.name], "placebo_mean": fake.mean(),
                         "timing": es[r.name] - fake.mean(), "rank_pct": 100 * (es[r.name] > fake).mean()})
        pd.DataFrame(rows).to_csv(OUT / "grid.csv", index=False)

    # ---- sensitivities (frozen rules)
    sens = {}
    base_rules = rules[:5]
    for label, c, b in (("execution 1 bar later", cfg.with_(exec_delay=1), bars),
                        ("execution 1 hour later", cfg.with_(exec_delay=4), bars),
                        ("execution 1 day later", cfg.with_(exec_delay=96), bars),
                        ("no trading costs", cfg.with_(fee_bps=0, slippage_bps=0), bars),
                        ("trading costs x2", cfg.with_(fee_bps=2 * args.fee, slippage_bps=2 * args.slippage), bars),
                        ("funding ignored", cfg, bars.assign(bviv_funding=0.0, bviv_carry=0.0)),
                        ("fills at mark (no premium)", cfg, bars.drop(columns="bviv_mid")),
                        ("perpetual BTC book", cfg.with_(instrument="perp"), bars)):
        res, _ = run_rules(b, c, base_rules, signals=build_signals(b, c, out["forecasts"][30]["HAR-IV"]))
        bk = {k: daily_book(v).loc[start:] for k, v in res.items()}
        sens[label] = {k: book_metrics(v, bk["Unhedged"]) for k, v in bk.items() if k != "Unhedged"}
    (OUT / "sensitivity.json").write_text(json.dumps(sens, indent=2))

    # ---- capacity: hedge contracts for a 1-BTC book against the perpetual's open interest
    pos = out["results"]["always|1"]["position"].loc[start:]
    oi = bars["open_interest"].loc[start:]
    held = pos > 0
    facts["capacity"] = {"median_contracts_per_btc": float(pos[held].median()),
                         "median_oi": float(oi.median()),
                         "share_bars_position_above_oi": float(100 * (pos[held] > oi[held]).mean()),
                         "max_notional_pct": float(100 * (out["results"]["always|1"]["notional"] /
                                                          out["results"]["always|1"]["btc_notional"]).loc[start:].max())}
    facts["book"] = book_facts(facts["capacity"]["median_contracts_per_btc"])
    hedge = books["always|1"]["hedge"]
    worst = books["Unhedged"]["total"].nsmallest(10).index
    facts["concentration"] = {"hedge_total_pct": float(100 * hedge.sum()), "top10_pct": float(100 * hedge.nlargest(10).sum()),
                              "ex_top10_pct": float(100 * (hedge.sum() - hedge.nlargest(10).sum())),
                              "worst10_btc_pct": float(100 * books["always|1"]["btc"].loc[worst].sum()),
                              "worst10_hedge_pct": float(100 * hedge.loc[worst].sum()),
                              "best_day": str(hedge.idxmax().date()), "best_day_pct": float(100 * hedge.max())}

    # ---- the best hedge day at 15-minute resolution (episode figure)
    day = pd.Timestamp(facts["concentration"]["best_day"], tz="UTC")
    win = slice(day - pd.Timedelta(days=2), day + pd.Timedelta(days=3) - pd.Timedelta(minutes=15))
    sig = out["signals"].loc[win]
    episode = pd.DataFrame({
        "close": bars["close"].loc[win], "vwap": sig["vwap"],
        "band": sig["vwap"] * np.exp(base_rt.z_enter * sig["sigma_day"] / np.sqrt(3.0)), "z": sig["z"],
        "bviv": bars["bviv"].loc[win], "bviv_mark": bars["bviv_mark"].loc[win], "bviv_mid": bars["bviv_mid"].loc[win],
    })
    for k in ("always|1", sw, rt):
        episode[f"{k}|position"] = out["results"][k]["position"].loc[win]
    episode.to_csv(OUT / "episode.csv")

    # ---- daily series for figures
    daily_out = pd.DataFrame({
        "btc": bars["close"].resample("1D").last(), "bviv": bars["bviv"].resample("1D").last(),
        "bviv_mark": bars["bviv_mark"].resample("1D").last(),
        "premium_pct": (100 * (bars["bviv_mid"] / bars["bviv_mark"] - 1)).resample("1D").mean(),
    })
    ev = panel["events"].loc[perp_start:]
    daily_out["funding_8h_mean"] = ev["rate"].resample("1D").mean()
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
    facts["costs"] = {"fee_bps": args.fee, "slippage_bps": args.slippage}
    (OUT / "facts.json").write_text(json.dumps(facts, indent=2, default=float))
    print(json.dumps({k: facts[k] for k in ("perp", "funding", "placebo")}, indent=1, default=float))


if __name__ == "__main__":
    main()
