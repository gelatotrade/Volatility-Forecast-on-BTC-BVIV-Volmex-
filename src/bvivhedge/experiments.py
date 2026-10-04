"""Experiment orchestration: one sample end-to-end, Monte Carlo, rule grids.

Protocol: rule parameters are *selected* on training seeds and *reported* on
disjoint test seeds, so no reported number is in-sample to a design choice.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .forecast import MODELS, evaluate, make_forecasts
from .hedge import DEFAULT_RULES, HedgeConfig, Rule, build_signals, run_rules
from .metrics import book_metrics, daily_book
from .realized import daily_realized
from .simulate import MarketParams, simulate_market

HORIZONS = (1, 7, 30)
TRAIN_SEED0 = 10_000


@dataclass(frozen=True)
class Protocol:
    """Sample splits in days from the first bar."""

    hedge_eval_start: int = 270       # after the 30-day HAR-IV forecast and its cheapness signal exist
    forecast_eval_start: int = 365    # out-of-sample forecast evaluation from here
    horizons: tuple[int, ...] = HORIZONS
    full_forecasts: bool = True       # False: only the 30-day HAR-IV forecast used for sizing


def analyse(bars: pd.DataFrame, cfg: HedgeConfig, rules=DEFAULT_RULES, proto: Protocol = Protocol()) -> dict:
    """Forecasts, forecast evaluation, backtests and metrics for one sample."""
    daily = daily_realized(bars)
    iv_daily = bars["bviv"].resample("1D").last()
    horizons = proto.horizons if proto.full_forecasts else (30,)
    forecasts = {h: make_forecasts(daily, iv_daily, h, garch=proto.full_forecasts) for h in horizons}
    f_start = daily.index[min(proto.forecast_eval_start, len(daily) - 1)]
    evals = {h: evaluate(forecasts[h], h=h, start=f_start) for h in horizons} if proto.full_forecasts else {}

    signals = build_signals(bars, cfg, forecasts[30]["HAR-IV"])
    results, _ = run_rules(bars, cfg, rules, signals=signals)
    books = {k: daily_book(v) for k, v in results.items()}
    h_start = books["Unhedged"].index[proto.hedge_eval_start]
    base = books["Unhedged"].loc[h_start:]
    metrics = pd.DataFrame({k: book_metrics(b.loc[h_start:], base) for k, b in books.items()}).T
    return {"daily": daily, "forecasts": forecasts, "evals": evals, "results": results,
            "signals": signals, "books": books, "metrics": metrics, "hedge_start": h_start}


def _calibration_row(seed: int, bars: pd.DataFrame, daily: pd.DataFrame, signals: pd.DataFrame, jumps: int) -> dict:
    r = np.log(bars["close"]).diff()
    d_iv = bars["bviv"].diff()
    rv30 = daily["rv"][::-1].rolling(30).mean()[::-1].shift(-1)
    iv_d = bars["bviv"].resample("1D").last().reindex(daily.index)
    day_r = daily["ret"]
    gap = iv_d - np.sqrt(rv30 * 365) * 100
    return {
        "seed": seed,
        "rv": float(np.sqrt(daily["rv"].mean() * 365) * 100),
        "iv": float(bars["bviv"].mean()),
        "iv_p05": float(bars["bviv"].quantile(0.05)),
        "iv_p95": float(bars["bviv"].quantile(0.95)),
        "vrp": float(gap.mean()),
        "iv_above_rv": float((gap.dropna() > 0).mean() * 100),
        "corr_15m": float(r.corr(d_iv)),
        "corr_down": float(r[r < 0].corr(d_iv[r < 0])),
        "corr_up": float(r[r > 0].corr(d_iv[r > 0])),
        "corr_daily": float(day_r.corr(iv_d.diff())),
        "kurt_daily": float(day_r.kurt()),
        "worst_day": float(-day_r.min() * 100),
        "jumps": jumps,
        "z_below_m1": float((signals["z"] < -1).mean() * 100),
    }


def _path_summary(job: tuple) -> dict:
    seed, params, cfg, rules, proto = job
    market = simulate_market(params, seed)
    out = analyse(market.bars, cfg, rules, proto)
    m = out["metrics"].assign(seed=seed).rename_axis("strategy").reset_index()
    e = [ev.assign(seed=seed, h=h).rename_axis("model").reset_index() for h, ev in out["evals"].items()]
    calib = _calibration_row(seed, market.bars, out["daily"], out["signals"], market.meta["jump_count"])
    return {"metrics": m, "evals": pd.concat(e) if e else pd.DataFrame(), "calib": calib}


def monte_carlo(seeds, params: MarketParams = MarketParams(), cfg: HedgeConfig = HedgeConfig(),
                rules=DEFAULT_RULES, proto: Protocol = Protocol(), workers: int | None = None) -> dict[str, pd.DataFrame]:
    """Run independent simulated paths; returns long-format tables (metrics, forecast evals, calibration)."""
    jobs = [(int(s), params, cfg, tuple(rules), proto) for s in seeds]
    workers = workers or os.cpu_count() or 1
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            parts = list(ex.map(_path_summary, jobs, chunksize=1))
    else:
        parts = [_path_summary(j) for j in jobs]
    evals = [p["evals"] for p in parts if len(p["evals"])]
    return {
        "metrics": pd.concat([p["metrics"] for p in parts], ignore_index=True),
        "evals": pd.concat(evals, ignore_index=True) if evals else pd.DataFrame(),
        "calib": pd.DataFrame([p["calib"] for p in parts]),
    }


# --------------------------------------------------------------------------- rule grid
def rule_grid() -> list[Rule]:
    """Candidate rules for the training-seed search, plus scaled always-on and oracle ladders."""
    rules = [Rule("Unhedged", "unhedged")]
    rules += [Rule(f"always|{s:g}", "always", scale=s) for s in (0.5, 1.0, 1.5, 2.0, 3.0)]
    rules += [Rule(f"oracle|{s:g}", "oracle", scale=s) for s in (0.5, 1.0, 2.0, 3.0)]
    rules += [Rule(f"switch|{ze:g}|{zx:g}", "switch", z_enter=ze, z_exit=zx, min_hold=16)
              for ze in (-1.0, -1.5, -2.0) for zx in (0.0, 0.5)]
    rules += [Rule(f"ratchet|{ze:g}|{hl:g}|{fl:g}|{int(uf)}", "ratchet", z_enter=ze, halflife_days=hl, floor=fl, use_forecast=uf)
              for ze in (-1.0, -1.5, -2.0) for hl in (0.5, 1.0, 2.0, 4.0) for fl in (0.0, 0.5, 1.0) for uf in (False, True)]
    return rules


def select_rules(train_metrics: pd.DataFrame, families=("switch", "ratchet")) -> dict[str, str]:
    """Per rule family, the training-path winner on median ES reduction.

    ES is measured on daily returns net of funding and trading costs, so the
    criterion already charges every rule for what it pays; ties go to the
    cheaper rule.
    """
    med = summarise_metrics(train_metrics)
    chosen = {}
    for family in families:
        cand = med[med.index.str.startswith(family + "|")]
        best = cand.sort_values(["es_red", "hedge_cost"], ascending=[False, True])
        chosen[family] = best.index[0]
    return chosen


def rule_from_key(key: str, name: str | None = None) -> Rule:
    parts = key.split("|")
    kind = parts[0]
    if kind == "always":
        return Rule(name or key, "always", scale=float(parts[1]))
    if kind == "oracle":
        return Rule(name or key, "oracle", scale=float(parts[1]))
    if kind == "switch":
        return Rule(name or key, "switch", z_enter=float(parts[1]), z_exit=float(parts[2]), min_hold=16)
    if kind == "ratchet":
        return Rule(name or key, "ratchet", z_enter=float(parts[1]), halflife_days=float(parts[2]),
                    floor=float(parts[3]), use_forecast=bool(int(parts[4])))
    raise ValueError(key)


# --------------------------------------------------------------------------- summaries
def summarise_metrics(long: pd.DataFrame, cols=None) -> pd.DataFrame:
    """Median across paths (strategy x metric)."""
    cols = cols or [c for c in long.columns if c not in ("strategy", "seed")]
    return long.groupby("strategy", sort=False)[cols].median()


def paired(long: pd.DataFrame, metric: str, a: str, b: str, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Paired comparison of ``a`` minus ``b`` across paths: mean, 95% bootstrap CI, share positive."""
    w = long.pivot(index="seed", columns="strategy", values=metric)
    d = (w[a] - w[b]).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, (n_boot, len(d))).mean(axis=1)
    return {"mean": float(d.mean()), "lo": float(np.quantile(boots, 0.025)), "hi": float(np.quantile(boots, 0.975)),
            "share_pos": float((d > 0).mean())}


def summarise_forecasts(evals: pd.DataFrame) -> pd.DataFrame:
    """Per horizon x model: mean QLIKE, median loss ratios vs HAR, MZ R^2, share of DM wins/losses."""
    rows = []
    for (h, model), g in evals.groupby(["h", "model"], sort=False):
        har = evals[(evals.h == h) & (evals.model == "HAR")].set_index("seed")
        g = g.set_index("seed")
        rows.append({
            "h": h, "model": model,
            "qlike": g["qlike"].mean(),
            "qlike_ratio": (g["qlike"] / har["qlike"]).median(),
            "mse_ratio": (g["mse"] / har["mse"]).median(),
            "mz_r2": g["mz_r2"].mean(),
            "dm_win": (g["dm_vs_HAR"] < -1.96).mean() * 100,
            "dm_loss": (g["dm_vs_HAR"] > 1.96).mean() * 100,
        })
    out = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(MODELS)}
    out["_o"] = out["model"].map(order)
    return out.sort_values(["h", "_o"]).drop(columns="_o").reset_index(drop=True)
