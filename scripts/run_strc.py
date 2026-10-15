"""The STRC study: a short position in Strategy's Stretch preferred stock as a BTC crash hedge.

    python scripts/fetch_data.py                 # BTC and BVIV inputs (data/raw)
    python scripts/run_strc.py                   # STRC cash and perpetual study -> results/strc/

STRC (Nasdaq, since 30 July 2025) is a variable-rate perpetual preferred of Strategy Inc., whose assets are
mostly bitcoin; its perpetual future ``xyz:STRC`` trades on Hyperliquid since 22 June 2026.  The hedging rules
keep the parameters frozen in the BVIV study (results/selection.json), including the ratchet's sizing by BVIV
forecast cheapness; the instrument, cap, costs and execution change, and the hedge is short.  The notional cap
carries the BVIV study's risk budget over to STRC: half the BTC notional in BVIV, scaled by how much more a
dollar of BVIV moved than a dollar of STRC (total return) over STRC's first 30 calendar days, before any hedge.

Books are compared over US trading days (close to close), so that a weekend BTC move and the Monday gap of STRC
fall on the same day.  On Nasdaq a decision is filled at the next observed print: the official open, an hourly
close (the finest STRC history available since the listing) or the official close; on the perpetual at the next
15-minute mark, around the clock, with slippage measured on its order book at the session closes.  A Nasdaq short
pays every dividend and a borrow fee (no interest is credited on the short-sale proceeds); a perpetual short pays
the perpetual's funding.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from bvivhedge import data, live, strc  # noqa: E402
from bvivhedge.experiments import rule_from_key  # noqa: E402
from bvivhedge.forecast import make_forecasts  # noqa: E402
from bvivhedge.hedge import HedgeConfig, Rule, build_signals, run_rules  # noqa: E402
from bvivhedge.metrics import book_metrics, daily_book  # noqa: E402
from bvivhedge.realized import daily_realized  # noqa: E402
from run_live import es_red, placebo_mean_bootstrap, placebo_rule, var_red  # noqa: E402

OUT = ROOT / "results" / "strc"
START, END = "2023-01-01", "2026-10-04"     # END exclusive; the last STRC session is Friday 2 October 2026
WARMUP_DAYS = 30                            # calendar days after the first STRC price before any hedge
PLACEBO_STEP = 7
CASH_FEE_BPS, CASH_SLIPPAGE_BPS = 1.0, 2.0  # commission and half the one-cent spread on a $100 share
PERP_FEE_BPS = 9.0                          # taker fee of HIP-3 markets: twice Hyperliquid's base 0.045%
BORROW = 0.01                               # annual fee to borrow STRC (IBKR: 0.3-1.3% from July to October 2026)
TRADING_DAYS = 252                          # rows per year of a book over US trading days


# --------------------------------------------------------------------------- helpers
def session_books(results: dict, day: pd.Series, start) -> dict:
    return {k: daily_book(v, day=day).loc[start:] for k, v in results.items()}


def run_books(bars: pd.DataFrame, cfg: HedgeConfig, rules: list[Rule], sig: pd.DataFrame, start, chunk: int = 12) -> dict:
    """Trading-day books of many rules, run in chunks so the 15-minute P&L of only a few is in memory at once."""
    books = {}
    for i in range(0, len(rules), chunk):
        res, _ = run_rules(bars, cfg, [Rule("Unhedged", "unhedged"), *rules[i:i + chunk]], signals=sig)
        books.update(session_books(res, bars["session"], start))
    return books


def metrics_of(books: dict) -> pd.DataFrame:
    return pd.DataFrame({k: book_metrics(b, books["Unhedged"], per_year=TRADING_DAYS) for k, b in books.items()}).T


def boot(books: dict, a: str, b: str, fn=es_red, block: float = 10.0) -> dict:
    return live.paired_block_bootstrap(books, "Unhedged", fn, a, b, block=block)


def combine(a: pd.DataFrame, b: pd.DataFrame, w: float = 0.5) -> pd.DataFrame:
    """Hold a fraction w of hedge a and 1 - w of hedge b on the same BTC book (same days)."""
    out = a.copy()
    for c in ("hedge", "funding", "carry", "cost", "notional", "traded_notional"):
        out[c] = w * a[c] + (1 - w) * b[c]
    out["total"] = out["btc"] + out["hedge"] - out["cost"]
    return out


def utc(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def session_series(s: pd.Series, day: pd.Series) -> pd.Series:
    return s.groupby(day.reindex(s.index)).last()


# --------------------------------------------------------------------------- descriptive facts
def official_closes(panel: dict) -> pd.Series:
    """STRC's official Nasdaq close by session date (descriptive facts; the hedge trades at its observations)."""
    return panel["sessions"]["close"]


def stylised(bars: pd.DataFrame, panel: dict, start, end) -> dict:
    """Daily co-movement of STRC (official closes, total return of a long share) and BTC (at each session close)."""
    day = bars["session"]
    px = official_closes(panel).loc[start:end]
    btc = session_series(bars["close"], day).reindex(px.index)
    divs = panel["dividends"]
    div = divs["amount"].groupby(divs.index).sum().reindex(px.index).fillna(0.0)
    r_btc = btc.pct_change().dropna()
    r_strc = ((px + div) / px.shift(1) - 1).dropna()
    r_btc, r_strc = r_btc.align(r_strc, join="inner")
    worst = r_btc <= r_btc.quantile(0.05)

    def beta(lo, hi):
        x, y = r_btc.loc[lo:hi], r_strc.loc[lo:hi]
        return float(np.polyfit(x, y, 1)[0])

    dd = px / px.cummax() - 1
    return {
        "days": int(len(r_btc)), "start": str(r_btc.index[0].date()), "end": str(r_btc.index[-1].date()),
        "vol_strc": float(r_strc.std() * np.sqrt(252) * 100), "vol_btc": float(r_btc.std() * np.sqrt(252) * 100),
        "corr": float(r_btc.corr(r_strc)), "beta": float(np.polyfit(r_btc, r_strc, 1)[0]),
        "beta_before_june": beta(None, "2026-05-29"), "beta_june": beta("2026-06-01", "2026-06-30"),
        "beta_after_june": beta("2026-07-01", None),
        "tail_days": int(worst.sum()), "tail_btc_mean": float(100 * r_btc[worst].mean()),
        "tail_strc_mean": float(100 * r_strc[worst].mean()),
        "share_near_par": float(100 * ((px - 100).abs() <= 1.0).mean()), "share_below_95": float(100 * (px < 95).mean()),
        "min_close": float(px.min()), "min_day": str(px.idxmin().date()), "max_drawdown": float(-100 * dd.min()),
        "first_close": float(px.iloc[0]), "last_close": float(px.iloc[-1]), "dividends_total": float(div.sum()),
        "returns": {str(d.date()): {"strc": float(100 * r_strc[d]), "btc": float(100 * r_btc[d])}
                    for d in pd.DatetimeIndex(["2026-02-05", "2026-02-06", "2026-06-24", "2026-06-25", "2026-06-29"])
                    if d in r_strc.index},
    }


def episode(books: dict, keys: list[str], lo: str, hi: str) -> dict:
    """Cumulative P&L of the BTC leg and of each hedge, net of carry and costs (% of BTC notional), over [lo, hi]."""
    u = books["Unhedged"].loc[lo:hi]
    out = {"days": int(len(u)), "btc_pct": float(100 * u["btc"].sum())}
    for k in keys:
        b = books[k].loc[lo:hi]
        out[k] = {"hedge_pct": float(100 * (b["hedge"] - b["cost"]).sum())}
    return out


def carry_per_notional(book: pd.DataFrame) -> float:
    """Carry paid per unit of hedge notional, % a year (dividends and borrow, or funding)."""
    return float(100 * book["carry"].mean() / book["notional"].mean() * TRADING_DAYS) if book["notional"].mean() > 0 else float("nan")


def perp_market(panel: dict, start, end) -> dict:
    """Funding, basis, off-hours prices and size of the perpetual over its first months.

    0xArchive stamps each mark at the start of the 15 minutes it averages, so the mark just before time t is the
    row stamped t - 15m.  On weeknights the oracle follows the extended-hours STRC market (post-market, the Blue
    Ocean overnight session, pre-market); from Friday 20:00 to Sunday 20:05 ET it is internal, set by the
    perpetual's own book within the discovery bounds.
    """
    rates, prices, candles, oi, cal = (panel[k] for k in ("rates", "prices", "candles", "oi", "sessions"))
    w = slice(utc(start), utc(end))
    r = rates.loc[w]
    daily_usd = candles["quote_volume"].loc[w].resample("1D").sum()
    mark = prices["mark_price"]

    def before(t):
        return float(mark.asof(t - pd.Timedelta("15min")))

    cal = cal[(cal["open_time"] >= utc(start)) & (cal["close_time"] <= utc(end))]
    rows = []
    for (d0, s0), (d1, s1) in zip(cal.iloc[:-1].iterrows(), cal.iloc[1:].iterrows()):
        gap = float(s1["open"] / s0["close"] - 1)
        row = {"perp_night": before(s1["open_time"]) / before(s0["close_time"]) - 1, "cash_gap": gap,
               "weekend": bool(d0.dayofweek == 4 and d1.dayofweek == 0), "monday": d1}
        if row["weekend"]:          # the internal window only: Friday 20:00 to Sunday 20:05 ET
            fri = (d0 + pd.Timedelta("20h")).tz_localize(strc.ET).tz_convert("UTC")
            sun = (d1 - pd.Timedelta("1D") + pd.Timedelta("20h")).tz_localize(strc.ET).tz_convert("UTC")
            row["perp_internal"] = before(sun) / before(fri) - 1
        rows.append(row)
    nights = pd.DataFrame(rows)
    wk = nights[nights["weekend"]]
    basis = pd.Series([before(c) for c in cal["close_time"]], index=cal.index) / cal["close"] - 1
    return {
        "listing": str(panel["listing"]), "hours": int(len(r)), "funding_mean_hourly": float(r.mean()),
        "funding_annual_pct": float(100 * r.mean() * 24 * 365), "funding_share_negative": float(100 * (r < 0).mean()),
        "funding_by_month_annual_pct": {str(k.date())[:7]: float(100 * v * 24 * 365) for k, v in r.resample("ME").mean().items()},
        "median_daily_usd": float(daily_usd.median()), "mean_daily_usd": float(daily_usd.mean()),
        "median_oi": float(oi["open_interest"].loc[w].median()), "max_oi": float(oi["open_interest"].loc[w].max()),
        "median_oi_usd": float((oi["open_interest"] * mark.reindex(oi.index, method="ffill")).loc[w].median()),
        "basis_median_pct": float(100 * basis.median()), "basis_abs_p95_pct": float(100 * basis.abs().quantile(0.95)),
        "nights": int(len(nights)), "night_corr": float(nights["perp_night"].corr(nights["cash_gap"])),
        "night_slope": float(np.polyfit(nights["perp_night"], nights["cash_gap"], 1)[0]),
        "weekends": int(len(wk)), "weekend_corr": float(wk["perp_night"].corr(wk["cash_gap"])),
        "weekend_internal_corr": float(wk["perp_internal"].corr(wk["cash_gap"])),
        "weekend_internal_max_pct": float(100 * wk["perp_internal"].abs().max()),
        "weekend_gap_max_pct": float(100 * wk["cash_gap"].abs().max()),
        "weekend_big_gap_day": str(wk["monday"].loc[wk["cash_gap"].abs().idxmax()].date()),
        "weekend_big_gap_internal_pct": float(100 * wk["perp_internal"].loc[wk["cash_gap"].abs().idxmax()]),
        "weekend_big_gap_pct": float(100 * wk["cash_gap"].loc[wk["cash_gap"].abs().idxmax()]),
        "weekend_internal_corr_ex_big": float(wk.drop(wk["cash_gap"].abs().idxmax())["perp_internal"]
                                              .corr(wk.drop(wk["cash_gap"].abs().idxmax())["cash_gap"])),
    }


def book_depth(symbol: str, times: list[pd.Timestamp], notional_usd: float) -> dict:
    """Order-book depth of the perpetual at given times (0xArchive snapshots): cost of selling ``notional_usd``."""
    rows = []
    for t in times:
        snap = json.loads(data.oxarchive_rows("orderbook/{symbol}/l2/history", symbol, t, t + pd.Timedelta("2min"),
                                              limit=1, max_pages=1))
        if not snap:
            continue
        s = snap[0]
        bids = np.array([[float(x["px"]), float(x["sz"])] for x in s["bids"]])
        asks = np.array([[float(x["px"]), float(x["sz"])] for x in s["asks"]])
        mid = (bids[0, 0] + asks[0, 0]) / 2
        qty = notional_usd / mid
        filled = np.minimum(np.cumsum(bids[:, 1]), qty)
        got = np.diff(np.r_[0.0, filled])
        cost = 1 - (got @ bids[:, 0]) / (got.sum() * mid) if got.sum() > 0 else np.nan
        rows.append({"time": str(t), "spread_bps": float(1e4 * (asks[0, 0] - bids[0, 0]) / mid),
                     "bid_depth_usd_1pct": float((bids[bids[:, 0] >= mid * 0.99, 1] * bids[bids[:, 0] >= mid * 0.99, 0]).sum()),
                     "fill_share": float(min(1.0, bids[:, 1].sum() / qty)), "sell_cost_bps": float(1e4 * cost)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "book_snapshots.csv", index=False)
    return {"snapshots": int(len(df)), "notional_usd": float(notional_usd),
            "median_spread_bps": float(df["spread_bps"].median()), "median_depth_1pct_usd": float(df["bid_depth_usd_1pct"].median()),
            "median_sell_cost_bps": float(df["sell_cost_bps"].median()), "p90_sell_cost_bps": float(df["sell_cost_bps"].quantile(0.9)),
            "share_full_fill": float(100 * (df["fill_share"] >= 1).mean())}


# --------------------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    selection = json.loads((ROOT / "results" / "selection.json").read_text())
    sw, rt = selection["switch"], selection["ratchet"]

    cashp = strc.build_panel(START, END, "cash", borrow_annual=BORROW)
    bars, ipo, divs, cal = cashp["bars"], cashp["ipo"], cashp["dividends"], cashp["sessions"]
    day = bars["session"]
    ev_bar = bars.index[(bars.index >= ipo + pd.Timedelta(days=WARMUP_DAYS)) & bars["tradable"].to_numpy()][0]
    ev_day = day.loc[ev_bar]
    days_to_ev = int((ev_bar.floor("1D") - bars.index[0]).days)

    # ---- the BVIV study's panel: risk budget for the cap, and the BVIV hedge on the same trading days
    bviv_panel = live.build_panel(START, END)
    bb = bviv_panel["bars"]
    warm = cal.loc[cal.index < ev_day]                                          # STRC's sessions before the first hedge
    strc_total = (warm["close"] + divs["amount"].groupby(divs.index).sum().reindex(warm.index).fillna(0.0)) / warm["close"].shift(1)
    bviv_close = session_series(bb["bviv_mark"], day.reindex(bb.index)).reindex(warm.index)
    vol_ratio = float(np.log(bviv_close).diff().std() / np.log(strc_total).std())
    cap = HedgeConfig().max_notional_frac * vol_ratio

    cfg = HedgeConfig(direction=-1, max_notional_frac=cap, fee_bps=CASH_FEE_BPS, slippage_bps=CASH_SLIPPAGE_BPS,
                      exec_delay=1, warmup_days=days_to_ev, warmup_bars=int(bars.index.get_loc(ev_bar)),
                      placebo_start_days=days_to_ev)
    daily = daily_realized(bars)
    fc30 = make_forecasts(daily, bars["bviv"].resample("1D").last(), 30, garch=False)["HAR-IV"]
    sig = build_signals(bars, cfg, fc30)
    print(f"STRC from {ipo}, hedging from {ev_bar} (session {ev_day.date()}), cap {cap:.2f}x BTC notional "
          f"(BVIV/STRC daily vol {vol_ratio:.2f} over {len(warm) - 1} warm-up returns); hourly gaps: {cashp['hourly_gaps']}", flush=True)

    facts = {"ipo": str(ipo), "eval_start": str(ev_day.date()), "warmup_days": WARMUP_DAYS, "warmup_returns": len(warm) - 1,
             "cap": cap, "vol_ratio_warmup": vol_ratio, "selection": selection, "hourly_gaps": cashp["hourly_gaps"],
             "costs": {"cash_fee_bps": CASH_FEE_BPS, "cash_slippage_bps": CASH_SLIPPAGE_BPS, "borrow_annual_pct": 100 * BORROW,
                       "perp_fee_bps": PERP_FEE_BPS},
             "dividend_rate_last_pct": float(100 * divs["amount"].iloc[-1] * 24 / 100),
             "environment": {"python": platform.python_version(),
                             **{m: importlib.metadata.version(m) for m in ("numpy", "pandas", "scipy", "arch", "statsmodels", "matplotlib")}}}
    facts["stylised"] = stylised(bars, cashp, ev_day, None)
    facts["stylised_full"] = stylised(bars, cashp, None, None)
    usd = (cashp["daily"]["close"] * cashp["daily"]["volume"]).loc[ipo.floor("1D"):]
    facts["nasdaq_volume"] = {"median_daily_usd": float(usd.median()), "mean_daily_usd": float(usd.mean()), "days": int(len(usd))}

    # ---- frozen rules, placebos, untimed benchmarks
    base_rt = rule_from_key(rt)
    rules = [rule_from_key("always|1"), rule_from_key("always|1.5"), rule_from_key(sw), base_rt]
    eval_days = int((bars.index[-1] - ev_bar).days)
    placebos = [placebo_rule(base_rt, d) for d in range(PLACEBO_STEP, eval_days - 7, PLACEBO_STEP)]
    books = run_books(bars, cfg, rules, sig, ev_day)
    metrics = metrics_of(books)
    placebo_books = run_books(bars, cfg, placebos, sig, ev_day)
    pl_notional = np.mean([book_metrics(placebo_books[p.name], placebo_books["Unhedged"], per_year=TRADING_DAYS)["notional"] for p in placebos])
    scale = float(pl_notional / metrics.loc["always|1", "notional"])
    bench = [Rule("always|matched", "always", scale=scale),
             Rule("overlay|held", "ratchet", z_enter=np.inf, halflife_days=base_rt.halflife_days, floor=base_rt.floor,
                  use_forecast=base_rt.use_forecast)]
    bench_books = run_books(bars, cfg, bench, sig, ev_day)
    allb = {**books, **{k: v for k, v in bench_books.items() if k != "Unhedged"}}
    metrics = metrics_of(allb)
    metrics.to_csv(OUT / "metrics.csv")
    print(metrics[["es_red", "var_red", "mdd", "hedge_pnl", "funding", "cost", "notional", "turnover"]].round(2), flush=True)
    res_always, _ = run_rules(bars, cfg, [rule_from_key("always|1")], signals=sig)
    capped = (sig["mv_all"] >= sig["cap"]).loc[ev_bar:]
    facts["cap_binding_pct"] = float(100 * capped.mean())

    u = books["Unhedged"]["total"].to_numpy()
    fake = np.array([es_red(placebo_books[p.name]["total"].to_numpy(), u) for p in placebos])
    real = es_red(books[rt]["total"].to_numpy(), u)
    facts["placebo"] = {"n": len(placebos), "real": float(real), "mean": float(fake.mean()),
                        "share_beaten": float(100 * (real > fake).mean()), "p05": float(np.quantile(fake, 0.05)),
                        "p95": float(np.quantile(fake, 0.95)), "step_days": PLACEBO_STEP}
    facts["benchmarks"] = {"matched_scale": scale, **{r.name: book_metrics(bench_books[r.name], bench_books["Unhedged"], per_year=TRADING_DAYS) for r in bench}}

    bs = {}
    for a, b in ((rt, "always|1"), (sw, "always|1"), ("always|1", "Unhedged"), ("always|1.5", "always|1")):
        for name, fn in (("es", es_red), ("var", var_red)):
            bs[f"{a} - {b} ({name})"] = boot(allb, a, b, fn)
    bs["overlay|held - " + rt + " (es)"] = boot(allb, "overlay|held", rt)
    bs["placebo mean - always|matched (es)"] = placebo_mean_bootstrap(
        [placebo_books[p.name]["total"] for p in placebos], bench_books["always|matched"]["total"], books["Unhedged"]["total"])

    # ---- where the hedge earned its keep: crash days, the June squeeze, horizons
    hedge_net = books["always|1"]["hedge"] - books["always|1"]["cost"]
    best5 = list(hedge_net.nlargest(5).index)
    june5 = [d for d in best5 if d.month == 6 and d.year == 2026]
    ub = books["Unhedged"]["total"]

    def es_without(bk: dict, k: str, drop) -> float:
        drop = [d for d in drop if d in bk[k].index]
        return es_red(bk[k]["total"].drop(drop).to_numpy(), bk["Unhedged"]["total"].drop(drop).to_numpy())

    tail = ub[ub <= ub.quantile(0.025)].index
    facts["concentration"] = {
        "best5": [{"day": str(d.date()), "hedge_pct": float(100 * hedge_net[d]), "btc_pct": float(100 * books["Unhedged"]["btc"][d])} for d in best5],
        "june_in_best5": len(june5), "june_in_tail": int(sum(d.month == 6 and d.year == 2026 for d in tail)),
        "tail_days": [str(d.date()) for d in tail],
        "es_ex_best5": es_without(books, "always|1", best5), "es_ex_feb5": es_without(books, "always|1", [pd.Timestamp("2026-02-05")]),
        "es_ex_june_best": es_without(books, "always|1", june5), "hedge_total_pct": float(100 * hedge_net.sum()),
    }
    facts["es_by_period"] = {lab: {k: es_red(books[k]["total"].loc[lo:hi].to_numpy(), ub.loc[lo:hi].to_numpy())
                                   for k in ("always|1", rt, sw)}
                             for lab, (lo, hi) in {"before June 2026": ("2025-08-29", "2026-05-29"),
                                                   "from June 2026": ("2026-06-01", "2026-10-02")}.items()}
    facts["episodes"] = {"feb_week": episode(books, ["always|1", rt], "2026-02-02", "2026-02-06"),
                         "feb5": episode(books, ["always|1"], "2026-02-05", "2026-02-05"),
                         "feb6": episode(books, ["always|1"], "2026-02-06", "2026-02-06"),
                         "june_squeeze": episode(books, ["always|1", rt], "2026-06-15", "2026-06-26"),
                         "june29": episode(books, ["always|1"], "2026-06-29", "2026-06-29"),
                         "june_to_end": episode(books, ["always|1"], "2026-06-15", "2026-10-02"),
                         "nov_drawdown": episode(books, ["always|1", rt], "2025-11-03", "2025-11-21")}
    pos = res_always["always|1"]["position"]
    notional_ratio = (-pos * bars["bviv_mark"] / bars["close"]).groupby(day.reindex(pos.index)).last()
    facts["hedge_ratio_feb"] = {str(d.date()): float(notional_ratio.get(d, np.nan)) for d in pd.DatetimeIndex(["2026-02-02", "2026-02-04", "2026-02-05"])}

    # ---- sensitivities (one assumption at a time)
    sens = {}

    def with_borrow(rate: float) -> pd.DataFrame:
        f = bars["bviv_funding"] - (rate - BORROW) * bars["bviv_mark"] * strc.DT * (bars.index >= ipo)
        return bars.assign(bviv_funding=f, bviv_carry=f)

    for label, b, c in (("same-bar execution", bars, cfg.with_(exec_delay=0)),
                        ("cap 1x BTC notional", bars, cfg.with_(max_notional_frac=1.0)),
                        ("no cap", bars, cfg.with_(max_notional_frac=1e6)),
                        ("no borrow fee", with_borrow(0.0), cfg),
                        ("borrow 2% a year", with_borrow(0.02), cfg),
                        ("ten times the trading costs", bars, cfg.with_(fee_bps=10 * CASH_FEE_BPS, slippage_bps=10 * CASH_SLIPPAGE_BPS))):
        res, _ = run_rules(b, c, [Rule("Unhedged", "unhedged"), *rules], signals=build_signals(b, c, fc30) if c.max_notional_frac != cap else sig)
        bk = session_books(res, day, ev_day)
        sens[label] = {k: book_metrics(v, bk["Unhedged"], per_year=TRADING_DAYS) for k, v in bk.items() if k != "Unhedged"}
    res, _ = run_rules(bars, cfg, [Rule("Unhedged", "unhedged"), *rules], signals=sig)
    utc_books = {k: daily_book(v).loc[ev_bar.floor("1D"):] for k, v in res.items()}
    sens["UTC calendar days"] = {k: book_metrics(v, utc_books["Unhedged"]) for k, v in utc_books.items() if k != "Unhedged"}
    (OUT / "sensitivity.json").write_text(json.dumps(sens, indent=2))

    # ---- the same hedge ratio on the perpetual, from its listing, against the Nasdaq hedge over the same days
    perpp = strc.build_panel(START, END, "perp")
    pbars = perpp["bars"]
    listing = perpp["listing"]
    lst_bar = pbars.index[pbars.index >= listing][0]
    lst_day = day.loc[pbars.index[(pbars.index >= listing) & pbars["tradable"].to_numpy()][0]]
    first_full = books["Unhedged"].index[books["Unhedged"].index > lst_day][0]       # the first whole session after listing
    # slippage from the perpetual's book at the session closes, for the Nasdaq hedge's median size over the window
    cpos = res_always["always|1"]["position"].loc[lst_bar:]
    hedge_usd = float((-cpos * bars["bviv_mark"].loc[lst_bar:])[cpos != 0].median())
    closes = [t for t in cal["close_time"] if lst_bar < t < utc(END)]
    facts["perp_book"] = book_depth(strc.PERP, closes, hedge_usd)
    perp_slip = facts["perp_book"]["median_sell_cost_bps"]
    pcfg = cfg.with_(fee_bps=PERP_FEE_BPS, slippage_bps=perp_slip, warmup_bars=int(pbars.index.get_loc(lst_bar)))
    pres, _ = run_rules(pbars, pcfg, [Rule("Unhedged", "unhedged"), *rules], signals=sig)   # the same hedge ratio
    pbooks = session_books(pres, pbars["session"], first_full)
    cbooks = {k: v.loc[first_full:] for k, v in allb.items()}
    window = {"start": str(first_full.date()), "end": str(cbooks["Unhedged"].index[-1].date()), "days": int(len(cbooks["Unhedged"])),
              "perp_slippage_bps": perp_slip,
              "cash": {k: book_metrics(cbooks[k], cbooks["Unhedged"], per_year=TRADING_DAYS) for k in ("always|1", rt, sw)},
              "perp": {k: book_metrics(pbooks[k], pbooks["Unhedged"], per_year=TRADING_DAYS) for k in ("always|1", rt, sw)},
              "carry_per_notional_pct": {"cash": carry_per_notional(cbooks["always|1"]), "perp": carry_per_notional(pbooks["always|1"])}}
    both = {"Unhedged": cbooks["Unhedged"], "perp": pbooks["always|1"], "cash": cbooks["always|1"]}
    window["perp - cash (es)"] = live.paired_block_bootstrap(both, "Unhedged", es_red, "perp", "cash")
    window["perp - cash (var)"] = live.paired_block_bootstrap(both, "Unhedged", var_red, "perp", "cash")
    window["weekend_trades"] = int(((pres["always|1"]["position"].diff().abs() > 0)
                                    & (pres["always|1"].index.tz_convert(strc.ET).dayofweek >= 5)).sum())
    facts["perp_window"] = window
    facts["perp_market"] = perp_market(perpp, lst_bar, END)
    ppos = pres["always|1"]["position"].loc[lst_bar:]
    oi = pbars["open_interest"].loc[lst_bar:]
    held = (ppos != 0) & oi.notna()
    facts["perp_capacity"] = {"median_contracts_per_btc": float(-ppos[held].median()), "median_oi": float(oi[held].median()),
                              "median_share_of_oi_pct": float(100 * (-ppos[held] / oi[held]).median()),
                              "median_hedge_usd": float((-ppos[held] * pbars["bviv_mark"].loc[lst_bar:][held]).median())}

    # ---- against the BVIV hedge on the same trading days, and both at half size
    bcfg = HedgeConfig(fee_bps=6.0, slippage_bps=10.0, exec_delay=1, warmup_days=int((bviv_panel["perp_start"] - bb.index[0]).days),
                       warmup_bars=int(bb.index.get_loc(bviv_panel["first_live"])))
    bsig = build_signals(bb, bcfg, make_forecasts(daily_realized(bb), bb["bviv"].resample("1D").last(), 30, garch=False)["HAR-IV"])
    bres, _ = run_rules(bb, bcfg, [Rule("Unhedged", "unhedged"), rule_from_key("always|1"), base_rt], signals=bsig)
    bday = day.reindex(bb.index)
    bbooks = session_books(bres, bday, ev_day)
    cmp = {"Unhedged": books["Unhedged"], "STRC": books["always|1"], "BVIV": bbooks["always|1"],
           "half each": combine(bbooks["always|1"], books["always|1"]),
           "STRC ratchet": books[rt], "BVIV ratchet": bbooks[rt], "half each, ratchets": combine(bbooks[rt], books[rt])}
    assert np.allclose(cmp["BVIV"]["btc"], cmp["STRC"]["btc"])
    facts["vs_bviv"] = {k: book_metrics(v, cmp["Unhedged"], per_year=TRADING_DAYS) for k, v in cmp.items() if k != "Unhedged"}
    net = {k: cmp[k]["hedge"] - cmp[k]["cost"] for k in ("STRC", "BVIV")}
    feb56 = pd.DatetimeIndex(["2026-02-05", "2026-02-06"])
    facts["vs_bviv_extra"] = {
        "hedge_corr": float(net["STRC"].corr(net["BVIV"])),
        "hedge_corr_ex_feb": float(net["STRC"].drop(feb56).corr(net["BVIV"].drop(feb56))),
        "bviv_es_ex_feb5": es_without(cmp, "BVIV", [pd.Timestamp("2026-02-05")]),
        "feb_week": episode(cmp, ["STRC", "BVIV"], "2026-02-02", "2026-02-06"),
        "june_squeeze": episode(cmp, ["STRC", "BVIV"], "2026-06-15", "2026-06-26"),
        "es_horizon": {str(h): {k: es_red(cmp[k]["total"].rolling(h).sum().dropna().to_numpy(),
                                          cmp["Unhedged"]["total"].rolling(h).sum().dropna().to_numpy())
                                for k in ("STRC", "BVIV", "half each", "STRC ratchet")} for h in (1, 5, 10)},
    }
    for a, b in (("half each", "BVIV"), ("half each", "STRC"), ("STRC", "BVIV")):
        bs[f"{a} - {b} (es)"] = live.paired_block_bootstrap(cmp, "Unhedged", es_red, a, b)
        bs[f"{a} - {b} (var)"] = live.paired_block_bootstrap(cmp, "Unhedged", var_red, a, b)
    (OUT / "bootstrap.json").write_text(json.dumps(bs, indent=2))

    # ---- daily series for the figure (official closes for STRC; the perpetual's mark before 16:00 ET)
    daily_out = pd.DataFrame({"btc": session_series(bars["close"], day), "strc": official_closes(cashp),
                              "perp": session_series(pbars["bviv_mark"].where(pbars["tradable"]), pbars["session"]),
                              "bviv": session_series(bb["bviv_mark"], bday)})
    for k, b in (("strc_always", books["always|1"]), ("strc_ratchet", books[rt]), ("bviv_always", bbooks["always|1"]),
                 ("half", cmp["half each"]), ("unhedged", books["Unhedged"])):
        daily_out[f"{k}|total"] = b["total"].reindex(daily_out.index)
        daily_out[f"{k}|hedge"] = (b["hedge"] - b["cost"]).reindex(daily_out.index)
    daily_out.to_csv(OUT / "daily.csv")
    facts["eval_days"] = int(len(books["Unhedged"]))
    facts["eval_end"] = str(books["Unhedged"].index[-1].date())
    (OUT / "facts.json").write_text(json.dumps(facts, indent=2, default=float))
    print(json.dumps({k: facts[k] for k in ("concentration", "episodes", "hedge_ratio_feb", "vs_bviv_extra", "perp_market",
                                            "perp_capacity", "perp_book")}, indent=1, default=float))
    print(json.dumps({k: {kk: round(vv, 2) for kk, vv in v.items() if kk in ("es_red", "var_red", "hedge_cost", "funding")}
                      for k, v in facts["vs_bviv"].items() if isinstance(v, dict)}, indent=1))


if __name__ == "__main__":
    main()
