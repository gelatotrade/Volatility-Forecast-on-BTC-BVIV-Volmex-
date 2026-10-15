"""The STRC study: a short position in Strategy's Stretch preferred stock as a BTC crash hedge.

    python scripts/fetch_data.py                 # BTC and BVIV inputs (data/raw)
    python scripts/run_strc.py                   # STRC cash and perpetual study -> results/strc/

STRC (Nasdaq, since 30 July 2025) is a variable-rate perpetual preferred of Strategy Inc., whose assets are
mostly bitcoin; its perpetual future ``xyz:STRC`` trades on Hyperliquid since 22 June 2026.  The hedging rules
are the frozen ones of the BVIV study (results/selection.json): only the instrument changes, and the hedge is
short.  The rules' notional cap carries the BVIV study's risk budget over to STRC: half the BTC notional in
BVIV, scaled by how much more a dollar of BVIV moves than a dollar of STRC over STRC's first 30 trading days
(before any hedge is placed).

Books are compared over US trading days (16:00 ET to 16:00 ET), so that a weekend BTC move and the Monday gap
of STRC fall on the same day.  On the cash market a decision is filled at the next hourly close in regular hours
(the finest STRC history available since the listing); on the perpetual at the next 15-minute mark, around the
clock.  A cash short pays every dividend and a borrow fee; a perpetual short pays the perpetual's funding.
"""

from __future__ import annotations

import argparse
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
def stylised(bars: pd.DataFrame, divs: pd.DataFrame, start, end) -> dict:
    day = bars["session"]
    btc = session_series(bars["close"], day).loc[start:end]
    px = session_series(bars["bviv_mark"].where(bars["hedge_update"]), day).loc[start:end]
    div = divs["amount"].groupby(divs.index).sum().reindex(px.index).fillna(0.0)
    r_btc = np.log(btc).diff().dropna()
    r_strc = ((px + div) / px.shift(1) - 1).dropna()          # total return of a long share
    r_btc, r_strc = r_btc.align(r_strc, join="inner")
    worst = r_btc <= r_btc.quantile(0.05)
    beta = np.polyfit(r_btc, r_strc, 1)[0]
    by_month = pd.DataFrame({"btc": btc.resample("ME").last(), "strc": px.resample("ME").last()}).pct_change().dropna() * 100
    dd = px / px.cummax() - 1
    worst_strc = r_strc.nsmallest(6)
    return {
        "days": int(len(r_btc)), "start": str(r_btc.index[0].date()), "end": str(r_btc.index[-1].date()),
        "vol_strc": float(r_strc.std() * np.sqrt(252) * 100), "vol_btc": float(r_btc.std() * np.sqrt(252) * 100),
        "corr": float(r_btc.corr(r_strc)), "beta": float(beta),
        "corr_down": float(r_btc[r_btc < 0].corr(r_strc[r_btc < 0])), "corr_up": float(r_btc[r_btc > 0].corr(r_strc[r_btc > 0])),
        "tail_days": int(worst.sum()), "tail_btc_mean": float(100 * r_btc[worst].mean()),
        "tail_strc_mean": float(100 * r_strc[worst].mean()), "tail_beta": float(np.polyfit(r_btc[worst], r_strc[worst], 1)[0]),
        "share_near_par": float(100 * ((px - 100).abs() <= 1.0).mean()), "share_below_95": float(100 * (px < 95).mean()),
        "min_close": float(px.min()), "min_day": str(px.idxmin().date()), "max_drawdown": float(-100 * dd.min()),
        "first_close": float(px.iloc[0]), "last_close": float(px.iloc[-1]),
        "dividends_total": float(div.sum()), "dividend_yield_last": float(100 * divs["amount"].iloc[-1] * 24 / 100),
        "worst_strc_days": [{"day": str(d.date()), "strc": float(100 * v), "btc": float(100 * r_btc[d])} for d, v in worst_strc.items()],
        "worst_btc_days": [{"day": str(d.date()), "btc": float(100 * v), "strc": float(100 * r_strc[d])} for d, v in r_btc.nsmallest(6).items()],
        "by_month": {str(k.date())[:7]: {"btc": float(v["btc"]), "strc": float(v["strc"])} for k, v in by_month.iterrows()},
    }


def episode(books: dict, keys: list[str], lo: str, hi: str) -> dict:
    """Cumulative P&L of the BTC leg and of each hedge (% of BTC notional) over [lo, hi]."""
    u = books["Unhedged"].loc[lo:hi]
    out = {"days": int(len(u)), "btc_pct": float(100 * u["btc"].sum())}
    for k in keys:
        b = books[k].loc[lo:hi]
        out[k] = {"hedge_pct": float(100 * (b["hedge"] - b["cost"]).sum()), "offset_pct": float(100 * -(b["hedge"] - b["cost"]).sum() / u["btc"].sum())}
    return out


def perp_market(panel: dict, start, end) -> dict:
    """Funding, basis, off-hours price discovery and size of the perpetual over its first months."""
    bars, rates, prices, candles, oi, divs = (panel[k] for k in ("bars", "rates", "prices", "candles", "oi", "dividends"))
    w = slice(utc(start), utc(end))
    r = rates.loc[w]
    daily_usd = candles["quote_volume"].loc[w].resample("1D").sum()
    hourly = panel["hourly"]
    cash = strc.hourly_closes(hourly)
    et = hourly.index.tz_convert(strc.ET)
    opens = hourly["open"].groupby(et.normalize()).first()                      # the opening print of each session
    closes = cash.groupby(cash.index.tz_convert(strc.ET).normalize()).tail(1)    # last hourly close: the session close
    mark = prices["mark_price"]
    rows = []
    for t0, (d1, open1) in zip(closes.index[:-1], list(opens.items())[1:]):
        c0 = t0 + pd.Timedelta("15min")
        o1 = (d1 + pd.Timedelta("9h30min")).tz_convert("UTC")
        if c0 < utc(start) or o1 > utc(end) or o1 <= c0:
            continue
        rows.append({"perp_night": float(mark.asof(o1) / mark.asof(c0) - 1), "cash_gap": float(open1 / closes.loc[t0] - 1),
                     "weekend": bool((o1 - c0) > pd.Timedelta("1D"))})
    nights = pd.DataFrame(rows)
    slope = float(np.polyfit(nights["perp_night"], nights["cash_gap"], 1)[0])
    basis = (mark.reindex(closes.index + pd.Timedelta("15min")).to_numpy() / closes.to_numpy() - 1)
    basis = pd.Series(basis, index=closes.index).loc[w].dropna()
    return {
        "listing": str(panel["listing"]), "hours": int(len(r)), "funding_mean_hourly": float(r.mean()),
        "funding_annual_pct": float(100 * r.mean() * 24 * 365), "funding_share_negative": float(100 * (r < 0).mean()),
        "funding_by_month_annual_pct": {str(k.date())[:7]: float(100 * v * 24 * 365) for k, v in r.resample("ME").mean().items()},
        "median_daily_usd": float(daily_usd.median()), "mean_daily_usd": float(daily_usd.mean()),
        "median_oi": float(oi["open_interest"].loc[w].median()), "max_oi": float(oi["open_interest"].loc[w].max()),
        "median_oi_usd": float((oi["open_interest"] * mark.reindex(oi.index, method="ffill")).loc[w].median()),
        "basis_median_pct": float(100 * basis.median()), "basis_abs_p95_pct": float(100 * basis.abs().quantile(0.95)),
        "nights": int(len(nights)), "night_corr": float(nights["perp_night"].corr(nights["cash_gap"])), "night_slope": slope,
        "weekend_corr": float(nights.loc[nights["weekend"], "perp_night"].corr(nights.loc[nights["weekend"], "cash_gap"])),
        "weekends": int(nights["weekend"].sum()),
    }


def book_depth(symbol: str, times: list[pd.Timestamp], notional_usd: float) -> dict:
    """Order-book depth of the perpetual at given times (0xArchive snapshots): cost of selling ``notional_usd``."""
    rows = []
    for t in times:
        t0 = t.strftime("%Y-%m-%dT%H:%M:%S")
        t1 = (t + pd.Timedelta("2min")).strftime("%Y-%m-%dT%H:%M:%S")
        snap = json.loads(data.oxarchive_rows("orderbook/{symbol}/l2/history", symbol, t0, t1, limit=1, max_pages=1))
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
    return {"snapshots": int(len(df)), "notional_usd": float(notional_usd),
            "median_spread_bps": float(df["spread_bps"].median()), "median_depth_1pct_usd": float(df["bid_depth_usd_1pct"].median()),
            "median_sell_cost_bps": float(df["sell_cost_bps"].median()), "p90_sell_cost_bps": float(df["sell_cost_bps"].quantile(0.9)),
            "share_full_fill": float(100 * (df["fill_share"] >= 1).mean())}


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-book", action="store_true", help="skip the order-book snapshots (0xArchive)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    selection = json.loads((ROOT / "results" / "selection.json").read_text())
    sw, rt = selection["switch"], selection["ratchet"]

    cashp = strc.build_panel(START, END, "cash", borrow_annual=BORROW)
    bars, ipo, divs = cashp["bars"], cashp["ipo"], cashp["dividends"]
    day = bars["session"]
    ev_bar = bars.index[(bars.index >= ipo + pd.Timedelta(days=WARMUP_DAYS)) & bars["tradable"].to_numpy()][0]
    ev_day = day.loc[ev_bar]
    days_to_ev = int((ev_bar.floor("1D") - bars.index[0]).days)

    # ---- the BVIV study's panel: risk budget for the cap, and the BVIV hedge on the same trading days
    bviv_panel = live.build_panel(START, END)
    bb = bviv_panel["bars"]
    warm = slice(ipo, ev_bar)
    vol = lambda s: float(np.log(session_series(s.loc[warm], day)).diff().std())  # noqa: E731
    vol_ratio = vol(bb["bviv_mark"]) / vol(bars["bviv_mark"].where(bars["hedge_update"]).ffill())
    cap = HedgeConfig().max_notional_frac * vol_ratio

    cfg = HedgeConfig(direction=-1, max_notional_frac=cap, fee_bps=CASH_FEE_BPS, slippage_bps=CASH_SLIPPAGE_BPS,
                      exec_delay=1, warmup_days=days_to_ev, warmup_bars=int(bars.index.get_loc(ev_bar)),
                      placebo_start_days=days_to_ev)
    daily = daily_realized(bars)
    fc30 = make_forecasts(daily, bars["bviv"].resample("1D").last(), 30, garch=False)["HAR-IV"]
    sig = build_signals(bars, cfg, fc30)
    print(f"STRC from {ipo}, hedging from {ev_bar} (session {ev_day.date()}), cap {cap:.2f}x BTC notional "
          f"(BVIV/STRC daily vol {vol_ratio:.2f} over the warm-up)", flush=True)

    facts = {"ipo": str(ipo), "eval_start": str(ev_day.date()), "warmup_days": WARMUP_DAYS,
             "cap": cap, "vol_ratio_warmup": vol_ratio, "selection": selection,
             "costs": {"cash_fee_bps": CASH_FEE_BPS, "cash_slippage_bps": CASH_SLIPPAGE_BPS, "borrow_annual_pct": 100 * BORROW,
                       "perp_fee_bps": PERP_FEE_BPS},
             "environment": {"python": platform.python_version(),
                             **{m: importlib.metadata.version(m) for m in ("numpy", "pandas", "scipy", "arch", "statsmodels", "matplotlib")}}}
    facts["stylised"] = stylised(bars, divs, ev_day, None)
    usd = (cashp["daily"]["close"] * cashp["daily"]["volume"]).loc[ipo.floor("1D"):]
    facts["nasdaq_volume"] = {"median_daily_usd": float(usd.median()), "mean_daily_usd": float(usd.mean()), "days": int(len(usd))}
    facts["stylised_full"] = stylised(bars, divs, None, None)

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

    # ---- where the hedge earned its keep: crashes, concentration, halves of the sample
    hedge_net = books["always|1"]["hedge"] - books["always|1"]["cost"]
    best = hedge_net.idxmax()
    best5 = list(hedge_net.nlargest(5).index)
    ub = books["Unhedged"]["total"]

    def es_without(k: str, drop) -> float:
        return es_red(books[k]["total"].drop(drop).to_numpy(), ub.drop(drop).to_numpy())

    facts["concentration"] = {
        "best_day": str(best.date()), "best_day_pct": float(100 * hedge_net.max()),
        "best5": [{"day": str(d.date()), "hedge_pct": float(100 * hedge_net[d]), "btc_pct": float(100 * books["Unhedged"]["btc"][d])} for d in best5],
        "es_ex_best_day": es_without("always|1", [best]), "es_ex_best5": es_without("always|1", best5),
        "hedge_total_pct": float(100 * hedge_net.sum()),
    }
    halves = {"2025-08-29..2026-05-29": ("2025-08-29", "2026-05-29"), "2026-06-01..2026-10-02": ("2026-06-01", "2026-10-02")}
    facts["es_by_period"] = {lab: {k: es_red(books[k]["total"].loc[lo:hi].to_numpy(), ub.loc[lo:hi].to_numpy())
                                   for k in ("always|1", rt, sw)} for lab, (lo, hi) in halves.items()}
    facts["episodes"] = {"btc_crash_2026_02": episode(books, ["always|1", rt], "2026-02-02", "2026-02-06"),
                         "strc_crash_2026_06": episode(books, ["always|1", rt], "2026-06-15", "2026-06-26"),
                         "btc_drawdown_2025_11": episode(books, ["always|1", rt], "2025-11-03", "2025-11-21")}

    # ---- sensitivities (one assumption at a time)
    sens = {}
    for label, b, c in (("same-bar execution", bars, cfg.with_(exec_delay=0)),
                        ("cap 1x BTC notional", bars, cfg.with_(max_notional_frac=1.0)),
                        ("no cap", bars, cfg.with_(max_notional_frac=1e6)),
                        ("no borrow fee", bars.assign(bviv_funding=bars["bviv_funding"] + BORROW * bars["bviv_mark"] * strc.DT * (bars.index >= ipo)), cfg),
                        ("borrow 2% a year", bars.assign(bviv_funding=bars["bviv_funding"] - (0.02 - BORROW) * bars["bviv_mark"] * strc.DT * (bars.index >= ipo)), cfg),
                        ("ten times the trading costs", bars, cfg.with_(fee_bps=10 * CASH_FEE_BPS, slippage_bps=10 * CASH_SLIPPAGE_BPS))):
        res, _ = run_rules(b, c, [Rule("Unhedged", "unhedged"), *rules], signals=build_signals(b, c, fc30) if c.max_notional_frac != cap else sig)
        bk = session_books(res, day, ev_day)
        sens[label] = {k: book_metrics(v, bk["Unhedged"], per_year=TRADING_DAYS) for k, v in bk.items() if k != "Unhedged"}
    # UTC calendar days instead of trading days: the weekend mismatch of an exchange-hours hedge
    res, _ = run_rules(bars, cfg, [Rule("Unhedged", "unhedged"), *rules], signals=sig)
    utc = {k: daily_book(v).loc[ev_bar.floor("1D"):] for k, v in res.items()}
    sens["UTC calendar days"] = {k: book_metrics(v, utc["Unhedged"]) for k, v in utc.items() if k != "Unhedged"}
    (OUT / "sensitivity.json").write_text(json.dumps(sens, indent=2))

    # ---- the same hedge on the perpetual, from its listing, against the cash hedge over the same days
    perpp = strc.build_panel(START, END, "perp")
    pbars = perpp["bars"]
    listing = perpp["listing"]
    lst_bar = pbars.index[pbars.index >= listing][0]
    lst_day = day.loc[pbars.index[(pbars.index >= listing) & pbars["tradable"].to_numpy()][0]]
    first_full = books["Unhedged"].index[books["Unhedged"].index > lst_day][0]       # the first whole session after listing
    psig = sig                                                    # the same hedge ratio: only the venue changes
    pcfg = cfg.with_(fee_bps=PERP_FEE_BPS, slippage_bps=0.0, warmup_bars=int(pbars.index.get_loc(lst_bar)))
    pres, _ = run_rules(pbars, pcfg, [Rule("Unhedged", "unhedged"), *rules], signals=psig)
    pbooks = session_books(pres, pbars["session"], first_full)
    cbooks = {k: v.loc[first_full:] for k, v in allb.items()}
    window = {"start": str(first_full.date()), "end": str(cbooks["Unhedged"].index[-1].date()), "days": int(len(cbooks["Unhedged"])),
              "cash": {k: book_metrics(cbooks[k], cbooks["Unhedged"], per_year=TRADING_DAYS) for k in ("always|1", rt, sw)},
              "perp": {k: book_metrics(pbooks[k], pbooks["Unhedged"], per_year=TRADING_DAYS) for k in ("always|1", rt, sw)}}
    both = {"Unhedged": cbooks["Unhedged"], "perp": pbooks["always|1"], "cash": cbooks["always|1"]}
    window["perp - cash (es)"] = live.paired_block_bootstrap(both, "Unhedged", es_red, "perp", "cash")
    window["perp - cash (var)"] = live.paired_block_bootstrap(both, "Unhedged", var_red, "perp", "cash")
    facts["perp_window"] = window
    facts["perp_market"] = perp_market(perpp, lst_bar, END)
    pos = pres["always|1"]["position"].loc[lst_bar:]
    oi = pbars["open_interest"].loc[lst_bar:]
    held = (pos != 0) & oi.notna()
    facts["perp_capacity"] = {"median_contracts_per_btc": float(-pos[held].median()), "median_oi": float(oi[held].median()),
                              "median_share_of_oi_pct": float(100 * (-pos[held] / oi[held]).median()),
                              "median_hedge_usd": float((-pos[held] * pbars["bviv_mark"].loc[lst_bar:][held]).median())}
    if not args.no_book:
        closes = [t + pd.Timedelta("15min") for t in pbars.index[(pbars.index >= lst_bar) & pbars["hedge_update"].to_numpy()]
                  if (t + pd.Timedelta("15min")).tz_convert(strc.ET).strftime("%H:%M") == "16:00"]
        facts["perp_book"] = book_depth(strc.PERP, closes, facts["perp_capacity"]["median_hedge_usd"])

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
    facts["vs_bviv"]["hedge_corr"] = float((cmp["STRC"]["hedge"] - cmp["STRC"]["cost"]).corr(cmp["BVIV"]["hedge"] - cmp["BVIV"]["cost"]))
    for a, b in (("half each", "BVIV"), ("half each", "STRC"), ("STRC", "BVIV")):
        bs[f"{a} - {b} (es)"] = live.paired_block_bootstrap(cmp, "Unhedged", es_red, a, b)
    (OUT / "bootstrap.json").write_text(json.dumps(bs, indent=2))

    # ---- daily series for the figure
    px = session_series(bars["bviv_mark"].where(bars["hedge_update"]), day)
    daily_out = pd.DataFrame({"btc": session_series(bars["close"], day), "strc": px,
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
    print(json.dumps({k: facts[k] for k in ("placebo", "concentration", "episodes", "perp_market", "perp_capacity")}, indent=1, default=float))
    print(json.dumps({k: {kk: round(vv, 2) for kk, vv in v.items() if kk in ("es_red", "var_red", "hedge_cost", "funding")}
                      for k, v in facts["vs_bviv"].items() if isinstance(v, dict)}, indent=1))


if __name__ == "__main__":
    main()
