"""Reproduce every number, table and figure of the paper -- and optionally compile it.

    python scripts/run_paper.py                       # full Monte Carlo study (simulated market)
    python scripts/run_paper.py --quick               # small smoke run
    python scripts/run_paper.py --stage paper         # rebuild figures/tables from cached results

The live-data study (Volmex index, Bitfinex BVIV perpetual, Binance BTC) is scripts/run_live.py;
the paper stage reads its outputs from results/live/.

Stages: train (rule selection on training seeds) -> test (Monte Carlo on disjoint
test seeds) -> robust (scenario grid) -> placebo (timing placebo for the selected
ratchet) -> placebo-grid (every ratchet configuration vs. its placebo) -> paper.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from bvivhedge import plots, report  # noqa: E402
from bvivhedge.experiments import (  # noqa: E402
    TRAIN_SEED0, Protocol, monte_carlo, paired, rule_from_key, rule_grid, select_rules,
    summarise_forecasts, summarise_metrics,
)
from bvivhedge.hedge import HedgeConfig, Rule  # noqa: E402
from bvivhedge.simulate import MarketParams  # noqa: E402

RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
SIM_DAYS = 900
ROBUST_SEED0 = 20_000   # robustness scenarios run on their own, disjoint seed block

def signed(x: float, nd: int = 2) -> str:
    """'+0.61' / '-1.67', but a value that rounds to zero prints as plain '0.00'."""
    text = f"{x:+.{nd}f}"
    return text[1:] if float(text) == 0 else text


def final_rules(selection: dict[str, str]) -> list[Rule]:
    return [Rule("Unhedged", "unhedged"), rule_from_key("always|1"), rule_from_key(selection["switch"]),
            rule_from_key(selection["ratchet"]), rule_from_key("oracle|1")]


# --------------------------------------------------------------------------- stages
def stage_train(n: int, params: MarketParams, cfg: HedgeConfig) -> dict[str, str]:
    seeds = range(TRAIN_SEED0, TRAIN_SEED0 + n)
    mc = monte_carlo(seeds, params, cfg, rule_grid(), Protocol(full_forecasts=False))
    mc["metrics"].to_csv(RESULTS / "train_grid_metrics.csv", index=False)
    selection = select_rules(mc["metrics"])
    (RESULTS / "selection.json").write_text(json.dumps({**selection, "train_seeds": [seeds.start, seeds.stop]}, indent=2))
    print("selected on training seeds:", selection)
    return selection


def stage_test(n: int, params: MarketParams, cfg: HedgeConfig):
    mc = monte_carlo(range(n), params, cfg, rule_grid(), Protocol(full_forecasts=True))
    mc["metrics"].to_csv(RESULTS / "test_metrics.csv", index=False)
    mc["evals"].to_csv(RESULTS / "test_forecast_evals.csv", index=False)
    mc["calib"].to_csv(RESULTS / "test_calibration.csv", index=False)


SCENARIOS = [
    ("Baseline", {}, {}),
    ("Trading costs x2", {}, {"fee_bps": 12.0, "slippage_bps": 20.0}),
    ("Trading costs x0.5", {}, {"fee_bps": 3.0, "slippage_bps": 5.0}),
    ("Funding carry 0", {"hedge_carry": 0.0}, {}),
    ("Funding carry 12", {"hedge_carry": 12.0}, {}),
    ("Funding carry 24", {"hedge_carry": 24.0}, {}),
    ("VIX-like spot-vol", {"regime_theta": (-0.50, -0.75, -0.10), "sentiment_theta": (-0.45, -0.60, -0.10)}, {}),
    ("Inverse leverage", {"regime_theta": (0.00, -0.40, 0.50), "sentiment_theta": (0.00, -0.30, 0.40)}, {}),
    ("Crashes cluster in stress", {"jump_rate": (3.0, 70.0, 6.0)}, {}),
    ("Session-anchored VWAP", {}, {"anchor": "session"}),
    ("Perpetual BTC book", {}, {"instrument": "perp"}),
]


def stage_robust(n: int, params: MarketParams, cfg: HedgeConfig, selection: dict[str, str]):
    rules = final_rules(selection)[:4]
    rows = []
    for name, p_over, c_over in SCENARIOS:
        seeds = range(ROBUST_SEED0, ROBUST_SEED0 + n)   # common random numbers across scenarios
        mc = monte_carlo(seeds, params.with_(**p_over), cfg.with_(**c_over), rules, Protocol(full_forecasts=False))
        m = mc["metrics"]
        med = summarise_metrics(m)
        d = paired(m, "es_red", selection["ratchet"], "always|1")
        rows.append({"scenario": name, "paths": n, "es_always": med.loc["always|1", "es_red"], "es_ratchet": med.loc[selection["ratchet"], "es_red"],
                     "es_switch": med.loc[selection["switch"], "es_red"], "d_mean": d["mean"], "d_lo": d["lo"], "d_hi": d["hi"],
                     "share": 100 * d["share_pos"], "cost_always": med.loc["always|1", "hedge_cost"],
                     "cost_ratchet": med.loc[selection["ratchet"], "hedge_cost"]})
        print(f"  {name:28s} dES={d['mean']:+.2f} [{d['lo']:+.2f},{d['hi']:+.2f}] share={d['share_pos']:.2f}")
    pd.DataFrame(rows).to_csv(RESULTS / "robustness.csv", index=False)


PLACEBO_SHIFTS = (35, 91, 147, 203, 259)  # days; multiples of 7 keep the triggers' hour-of-week profile


def stage_placebo(n: int, params: MarketParams, cfg: HedgeConfig, selection: dict[str, str]):
    """Ratchet vs. the same ratchet with its trigger series circularly shifted (timing placebo)."""
    base = rule_from_key(selection["ratchet"])
    placebos = [Rule(f"placebo|{d}", "ratchet", z_enter=base.z_enter, halflife_days=base.halflife_days, floor=base.floor,
                     use_forecast=base.use_forecast, placebo_shift_days=float(d)) for d in PLACEBO_SHIFTS]
    rules = [Rule("Unhedged", "unhedged"), rule_from_key("always|1"), base, *placebos]
    mc = monte_carlo(range(n), params, cfg, rules, Protocol(full_forecasts=False))
    mc["metrics"].to_csv(RESULTS / "placebo.csv", index=False)


GRID_SHIFTS = (91, 147, 203)
GRID_PATHS = 64


def stage_placebo_grid(params: MarketParams, cfg: HedgeConfig):
    """Every ratchet configuration of the grid against its own timing placebos."""
    ratchets = [r for r in rule_grid() if r.kind == "ratchet"]
    placebos = [Rule(f"{r.name}|shift{d}", "ratchet", z_enter=r.z_enter, halflife_days=r.halflife_days, floor=r.floor,
                     use_forecast=r.use_forecast, placebo_shift_days=float(d)) for r in ratchets for d in GRID_SHIFTS]
    rules = [Rule("Unhedged", "unhedged"), rule_from_key("always|1"), *ratchets, *placebos]
    mc = monte_carlo(range(GRID_PATHS), params, cfg, rules, Protocol(full_forecasts=False))
    mc["metrics"][["strategy", "seed", "es_red", "var_red", "hedge_cost"]].to_csv(RESULTS / "placebo_grid.csv", index=False)


def placebo_grid_summary(grid: pd.DataFrame) -> dict[str, float]:
    """Timing value (real minus mean placebo ES reduction) per configuration, with paired bootstrap CIs."""
    w = grid.pivot(index="seed", columns="strategy", values="es_red")
    real = [c for c in w.columns if c.startswith("ratchet|") and "|shift" not in c]
    rows = []
    for c in real:
        fake = w[[f"{c}|shift{d}" for d in GRID_SHIFTS]].mean(axis=1)
        d = pd.DataFrame({"seed": w.index, "strategy": "diff", "es_red": (w[c] - fake).to_numpy()})
        st = paired(pd.concat([d, d.assign(strategy="zero", es_red=0.0)]), "es_red", "diff", "zero")
        rows.append(st)
    t = pd.DataFrame(rows, index=real)
    t["floor"] = [float(c.split("|")[3]) for c in real]
    t["es"] = w[real].median()
    t["cost"] = grid.pivot(index="seed", columns="strategy", values="hedge_cost")[real].median()
    best = t["mean"].idxmax()
    by_floor = t.groupby("floor")["mean"].mean()
    return {"n": len(t), "mean": float(t["mean"].mean()), "min": float(t["mean"].min()), "max": float(t["mean"].max()),
            "sig_pos": int((t["lo"] > 0).sum()), "sig_neg": int((t["hi"] < 0).sum()),
            "floor_zero": float(by_floor[0.0]), "floor_half": float(by_floor[0.5]), "floor_one": float(by_floor[1.0]),
            "es_floor_zero": float(t.loc[t["floor"] == 0.0, "es"].mean()), "es_floor_one": float(t.loc[t["floor"] == 1.0, "es"].mean()),
            "best_es": float(t.loc[best, "es"]), "best_cost": float(t.loc[best, "cost"]),
            "corr": float(np.corrcoef(t["mean"], t["es"])[0, 1]),
            "sig_pos_zero": int(((t["lo"] > 0) & (t["floor"] == 0.0)).sum()), "n_zero": int((t["floor"] == 0.0).sum()),
            "neg_point": int((t["mean"] < 0).sum()),
            "always_es": float(w["always|1"].median())}


def placebo_summary(placebo: pd.DataFrame, ratchet_key: str) -> dict[str, float]:
    w = placebo.pivot(index="seed", columns="strategy", values="es_red")
    fake = w[[c for c in w.columns if c.startswith("placebo|")]].mean(axis=1)
    d = pd.DataFrame({"seed": w.index, "strategy": "diff", "es_red": (w[ratchet_key] - fake).to_numpy()})
    zero = d.assign(strategy="zero", es_red=0.0)
    stats = paired(pd.concat([d, zero]), "es_red", "diff", "zero")
    return {"es_placebo": float(fake.median()), "es_ratchet": float(w[ratchet_key].median()), **stats}


def stage_paper(params: MarketParams, cfg: HedgeConfig, selection: dict[str, str], compile_pdf: bool):
    PAPER.mkdir(parents=True, exist_ok=True)
    fig_dir, tab_dir = PAPER / "figures", PAPER / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(RESULTS / "test_metrics.csv")
    evals = pd.read_csv(RESULTS / "test_forecast_evals.csv")
    calib = pd.read_csv(RESULTS / "test_calibration.csv")
    robust = pd.read_csv(RESULTS / "robustness.csv")
    placebo = placebo_summary(pd.read_csv(RESULTS / "placebo.csv"), selection["ratchet"])
    pgrid = placebo_grid_summary(pd.read_csv(RESULTS / "placebo_grid.csv"))
    sw, rt = selection["switch"], selection["ratchet"]

    # ---- simulation tables (the controlled experiment)
    med_all = summarise_metrics(test)
    pl = pd.read_csv(RESULTS / "placebo.csv")
    pl = pl[pl["strategy"].str.startswith("placebo|")].groupby("seed").mean(numeric_only=True)
    med_tab = pd.concat([med_all, pl.median().rename("placebo").to_frame().T])
    order = [("Unhedged", "Unhedged"), ("always|1", "Always-on (MV)"), ("always|1.5", "Always-on, 1.5$\\times$"),
             (sw, "VWAP-switch"), (rt, "VWAP-ratchet"), ("placebo", "Ratchet, placebo timing"),
             ("oracle|1", "Oracle (infeasible)")]
    report.table_hedging(med_tab, order, [
        ("es_red", "ES red.", 1), ("var_red", "Var red.", 1), ("mdd", "Max DD", 1), ("tail_offset", "Tail offset", 1),
        ("carry", "Carry", 2), ("cost", "Trading", 2), ("turnover", "Turnover", 1), ("time_on", "Time on", 0),
    ], tab_dir / "hedging.tex")
    report.table_robustness(robust.to_dict("records"), tab_dir / "robustness.tex")

    # ---- number macros for the text
    d_rt = paired(test, "es_red", rt, "always|1")
    d_sw = paired(test, "es_red", sw, "always|1")
    rr, ss = rule_from_key(rt), rule_from_key(sw)
    f = summarise_forecasts(evals).set_index(["h", "model"])
    costs = robust[robust.scenario.str.contains("costs|carry")]["d_mean"]
    nums = {
        "n_test": f"{test['seed'].nunique()}",
        "n_train": f"{json.loads((RESULTS / 'selection.json').read_text())['train_seeds'][1] - TRAIN_SEED0}",
        "n_robust": f"{int(robust['paths'].iloc[0])}", "n_placebo": f"{len(PLACEBO_SHIFTS)}", "sim_days": f"{params.days}",
        "hedge_carry": f"{params.hedge_carry:g}", "fee": f"{cfg.fee_bps:g}", "slip": f"{cfg.slippage_bps:g}",
        "ratchet_z": f"{rr.z_enter:g}", "ratchet_hl": f"{rr.halflife_days:g}", "ratchet_floor": f"{rr.floor:g}",
        "switch_z": f"{ss.z_enter:g}", "switch_exit": f"{ss.z_exit:g}",
        "es_red_always": med_all.loc["always|1", "es_red"], "es_red_always_onehalf": med_all.loc["always|1.5", "es_red"],
        "d_es_ratchet": signed(d_rt["mean"]), "d_es_ratchet_lo": signed(d_rt["lo"]), "d_es_ratchet_hi": signed(d_rt["hi"]),
        "d_es_switch_abs": f"{abs(d_sw['mean']):.2f}", "d_es_placebo": signed(placebo["mean"]),
        "pgrid_floor_zero": signed(pgrid["floor_zero"]), "pgrid_floor_one": signed(pgrid["floor_one"]),
        "ql_har_iv_one": f"{f.loc[(1, 'HAR-IV'), 'qlike_ratio']:.2f}", "ql_har_iv_seven": f"{f.loc[(7, 'HAR-IV'), 'qlike_ratio']:.2f}",
        "ql_iv_thirty": f"{f.loc[(30, 'IV'), 'qlike_ratio']:.2f}",
        "corr_daily": f"{calib['corr_daily'].median():.2f}",
        "rob_cost_min": signed(costs.min()), "rob_cost_max": signed(costs.max()),
    }
    for r in robust.to_dict("records"):
        key = "rob." + r["scenario"].lower().replace("x0.5", "half").replace("x2", "double").replace(" 0", " zero").replace(" 12", " twelve").replace(" 24", " twentyfour")
        nums[key] = signed(r['d_mean'])
        nums[key.replace("rob.", "rob lo.")] = signed(r['d_lo'])
        nums[key.replace("rob.", "rob hi.")] = signed(r['d_hi'])
    live = live_paper(selection, fig_dir, tab_dir, calib)
    nums.update(live)
    report.write_numbers(nums, PAPER / "numbers.tex")
    if PAPER == ROOT / "paper" and live:
        write_readme_results(pd.read_csv(LIVE / "metrics.csv", index_col=0),
                             [("Unhedged", "Unhedged"), ("always|1", "Always-on minimum-variance hedge"),
                              ("always|1.5", "Always-on, scaled 1.5×"), (sw, "VWAP-switch"),
                              (rt, "VWAP-ratchet (MV core + breakdown overlay)")], nums)

    if compile_pdf:
        compile_paper()


LIVE = ROOT / "results" / "live"


def live_paper(selection: dict[str, str], fig_dir: Path, tab_dir: Path, calib_sim: pd.DataFrame) -> dict:
    """Tables, figures and number macros for the live-data sections (results/live from run_live.py)."""
    if not (LIVE / "facts.json").exists():
        return {}
    facts = json.loads((LIVE / "facts.json").read_text())
    metrics = pd.read_csv(LIVE / "metrics.csv", index_col=0)
    boot = json.loads((LIVE / "bootstrap.json").read_text())
    sens = json.loads((LIVE / "sensitivity.json").read_text())
    evals = pd.read_csv(LIVE / "forecast_evals.csv")
    placebo = pd.read_csv(LIVE / "placebo.csv", index_col=0)
    grid = pd.read_csv(LIVE / "grid.csv")
    daily = pd.read_csv(LIVE / "daily.csv", index_col=0, parse_dates=True)
    sw, rt = selection["switch"], selection["ratchet"]
    perp_start = pd.Timestamp(facts["perp"]["first_day"], tz="UTC")
    if daily.index.tz is None:
        daily.index = daily.index.tz_localize("UTC")

    plots.fig_live_market(daily, perp_start, fig_dir / "live_market.pdf")
    plots.fig_live_hedge(daily.loc[perp_start:], "always|1", fig_dir / "live_hedge.pdf")
    plots.fig_live_timing(placebo, facts["placebo"]["real"], grid, fig_dir / "live_timing.pdf")
    episode = pd.read_csv(LIVE / "episode.csv", index_col=0, parse_dates=True)
    plots.fig_live_episode(episode, rule_from_key(rt).z_enter, {"always|1": "Always-on", sw: "VWAP-switch", rt: "VWAP-ratchet"},
                           fig_dir / "live_episode.pdf")

    st = facts["stylised"]
    report.table_live_vs_sim(st, calib_sim, [
        ("rv", "Realised vol, annualised (\\%)", "rv", 1),
        ("iv", "Mean BVIV (vol pts)", "iv", 1),
        ("vrp", "IV $-$ subsequent 30d RV (vol pts)", "", 1),
        ("iv_above_rv", "Share of days IV $>$ RV (\\%)", "", 0),
        ("corr_daily", "corr(daily return, $\\Delta$IV)", "corr_daily", 2),
        ("corr_down", "corr on falling 15m bars", "", 2),
        ("corr_up", "corr on rising 15m bars", "", 2),
        ("kurt_daily", "Excess kurtosis, daily returns", "", 1),
        ("worst_day", "Worst daily loss (\\%)", "", 1),
    ], tab_dir / "live_facts.tex")
    report.table_live_forecasts(evals, tab_dir / "live_forecasts.tex")
    order = [("Unhedged", "Unhedged"), ("always|1", "Always-on (MV)"), ("always|1.5", "Always-on, 1.5$\\times$"),
             (sw, "VWAP-switch"), (rt, "VWAP-ratchet")]
    report.table_hedging(metrics, order, [
        ("es_red", "ES red.", 1), ("var_red", "Var red.", 1), ("mdd", "Max DD", 1), ("tail_offset", "Tail offset", 1),
        ("hedge_pnl", "Hedge P\\&L", 2), ("funding", "Funding", 2), ("cost", "Trading", 2), ("turnover", "Turnover", 1),
        ("notional", "Notional", 1),
    ], tab_dir / "live_hedging.tex")
    sens_rows = {k: v for k, v in sens.items()}
    report.table_live_sensitivity(sens_rows, [("always|1", "Always-on"), ("always|1.5", "1.5$\\times$"),
                                              (sw, "Switch"), (rt, "Ratchet")], tab_dir / "live_sensitivity.tex")

    ev = evals.set_index(["h", "model"])
    def ql(h, m):
        return f"{ev.loc[(h, m), 'qlike'] / ev.loc[(h, 'HAR'), 'qlike']:.2f}"
    def dm(h, m):
        return signed(ev.loc[(h, m), "dm_vs_HAR"])
    b = {k: v for k, v in boot.items()}
    def bt(a, c, kind="es"):
        return b[f"{a} - {c} ({kind})"]
    fund, perp, cap, conc, book = facts["funding"], facts["perp"], facts["capacity"], facts["concentration"], facts["book"]
    fy = facts["funding_by_year"]
    by = st["by_year"]
    g = grid.groupby("floor")["timing"].mean()
    paid = daily.loc[perp_start:, "always|1|funding"]
    paid = 100 * 365 * paid.groupby(paid.index.year).mean()          # funding of the MV hedge, % of BTC a year
    def date(text):
        t = pd.Timestamp(text)
        return f"{t.day} {t:%B %Y}"
    nums = {
        "live start": date(st["start"]), "live end": date(st["end"]), "live days": f"{st['days']}",
        "live eval start": date(facts["eval_start"]), "live eval end": date(facts["eval_end"]),
        "live eval days": f"{facts['eval_days']}",
        "live rv": st["rv"], "live iv": st["iv"], "live iv min": st["iv_min"], "live iv max": st["iv_max"],
        "live vrp": st["vrp"], "live iv above": f"{st['iv_above_rv']:.0f}", "live ac": f"{st['ac1_15m_iv']:.2f}",
        "live corr daily": signed(st["corr_daily"]), "live corr down": signed(st["corr_down"]), "live corr up": signed(st["corr_up"]),
        "live corr a": signed(by["2023"]["corr_daily"]), "live corr b": signed(by["2024"]["corr_daily"]),
        "live corr c": signed(by["2025"]["corr_daily"]), "live corr d": signed(by["2026"]["corr_daily"]),
        "live worst": st["worst_day"],
        "live fund n": f"{fund['n']:,}".replace(",", "{,}"), "live fund cap share": f"{100 * fund['share_at_cap']:.0f}",
        "live fund pos share": f"{100 * fund['share_positive']:.0f}",
        "live fund annual pct": f"{fund['annual_pct_notional']:.0f}", "live fund annual pts": f"{fund['annual_vol_points']:.0f}",
        "live fund pts a": f"{fy['2024']['annual_vol_points']:.0f}", "live fund pts b": f"{fy['2025']['annual_vol_points']:.0f}",
        "live fund pts c": signed(fy['2026']['annual_vol_points'], 0),
        "live fund cap a": f"{100 * fy['2024']['share_at_cap']:.0f}", "live fund cap c": f"{100 * fy['2026']['share_at_cap']:.0f}",
        "live days traded": f"{perp['days_traded']}", "live perp days": f"{perp['days']}",
        "live median usd": f"{perp['median_daily_usd']:,.0f}".replace(",", "{,}"),
        "live mean usd": f"{perp['mean_daily_usd']:,.0f}".replace(",", "{,}"),
        "live median oi": f"{perp['median_oi_contracts']:.0f}", "live max oi": f"{perp['max_oi_contracts']:,.0f}".replace(",", "{,}"),
        "live premium median": f"{perp['median_premium_pct']:.2f}", "live premium mean": f"{perp['mean_premium_pct']:.2f}",
        "live premium pos": f"{perp['share_premium_positive']:.0f}",
        "live mark gap": f"{perp['mark_vs_volmex_pct_p95']:.2f}",
        "live contracts": f"{cap['median_contracts_per_btc']:.0f}", "live above oi": f"{cap['share_bars_position_above_oi']:.0f}",
        "live book date": date(book["time"]),
        "live spread": f"{book['spread_pct']:.2f}", "live best ask": f"{book['best_ask_contracts']:.0f}",
        "live best ask pct": f"{book['best_ask_pct']:.2f}", "live depth": f"{book['ask_contracts_2_5pct']:,.0f}".replace(",", "{,}"),
        "live book btc": f"{book['btc_capacity_2_5pct']:.0f}",
        "live top ten": f"{conc['top10_pct']:.0f}", "live ex top ten": signed(conc["ex_top10_pct"], 0),
        "live hedge total": signed(conc["hedge_total_pct"], 1), "live best day": date(conc["best_day"]),
        "live best day pct": f"{conc['best_day_pct']:.0f}", "live worst btc": f"{-conc['worst10_btc_pct']:.0f}",
        "live worst hedge": f"{conc['worst10_hedge_pct']:.0f}",
        "live es always": metrics.loc["always|1", "es_red"], "live var always": metrics.loc["always|1", "var_red"],
        "live es onehalf": metrics.loc["always|1.5", "es_red"], "live es switch": metrics.loc[sw, "es_red"],
        "live es ratchet": metrics.loc[rt, "es_red"], "live var ratchet": metrics.loc[rt, "var_red"],
        "live mdd unhedged": metrics.loc["Unhedged", "mdd"], "live mdd always": metrics.loc["always|1", "mdd"],
        "live cost always": f"{metrics.loc['always|1', 'hedge_cost']:.2f}", "live cost ratchet": f"{metrics.loc[rt, 'hedge_cost']:.2f}",
        "live cost switch": f"{metrics.loc[sw, 'hedge_cost']:.2f}", "live turn switch": metrics.loc[sw, "turnover"],
        "live fund always": metrics.loc["always|1", "funding"],
        "live fund hedge a": f"{paid[2024]:.1f}", "live fund hedge b": f"{paid[2025]:.1f}", "live fund hedge c": f"{-paid[2026]:.1f}",
        "live es switch mark fill": sens["fills at mark (no premium)"][sw]["es_red"],
        "live pnl always": signed(metrics.loc["always|1", "hedge_pnl"]), "live pnl switch": signed(metrics.loc[sw, "hedge_pnl"]),
        "live notional always": metrics.loc["always|1", "notional"], "live time on always": f"{metrics.loc['always|1', 'time_on']:.0f}",
        "live es always lo": bt("always|1", "Unhedged")["lo"], "live es always hi": bt("always|1", "Unhedged")["hi"],
        "live var always lo": bt("always|1", "Unhedged", "var")["lo"], "live var always hi": bt("always|1", "Unhedged", "var")["hi"],
        "live d ratchet": signed(bt(rt, "always|1")["diff"]), "live d ratchet lo": signed(bt(rt, "always|1")["lo"]),
        "live d ratchet hi": signed(bt(rt, "always|1")["hi"]),
        "live d ratchet half": signed(bt(rt, "always|1.5")["diff"]), "live d ratchet half lo": signed(bt(rt, "always|1.5")["lo"]),
        "live d ratchet half hi": signed(bt(rt, "always|1.5")["hi"]),
        "live d switch": signed(bt(sw, "always|1")["diff"]), "live d switch lo": signed(bt(sw, "always|1")["lo"]),
        "live d switch hi": signed(bt(sw, "always|1")["hi"]),
        "live placebo n": f"{facts['placebo']['n']}", "live placebo mean": facts["placebo"]["mean"],
        "live placebo beaten": f"{facts['placebo']['share_beaten']:.0f}",
        "live placebo lo": facts["placebo"]["p05"], "live placebo hi": facts["placebo"]["p95"],
        "live grid zero": signed(g[0.0]), "live grid half": signed(g[0.5]), "live grid one": signed(g[1.0]),
        "live grid rank zero": f"{grid.loc[grid.floor == 0, 'rank_pct'].mean():.0f}",
        "live grid rank one": f"{grid.loc[grid.floor == 1, 'rank_pct'].mean():.0f}",
        "live delay one": sens["execution 1 bar later"]["always|1"]["es_red"],
        "live delay hour": sens["execution 1 hour later"]["always|1"]["es_red"],
        "live delay day": sens["execution 1 day later"]["always|1"]["es_red"],
        "live delay switch pnl": signed(sens["execution 1 hour later"][sw]["hedge_pnl"]),
        "live delay day pnl": signed(sens["execution 1 day later"]["always|1"]["hedge_pnl"]),
        "live ql one": ql(1, "HAR-IV"), "live dm one": dm(1, "HAR-IV"), "live ql iv one": ql(1, "IV"),
        "live ql seven": ql(7, "HAR-IV"), "live dm seven": dm(7, "HAR-IV"), "live ql iv seven": ql(7, "IV"),
        "live ql thirty": ql(30, "IV"), "live dm thirty": dm(30, "IV"), "live ql har thirty": ql(30, "HAR-IV"),
        "live ql garch one": ql(1, "GARCH"), "live ql raw thirty": ql(30, "IV-raw"),
        "live forecast n": f"{facts['forecast_n']['1']:,}".replace(",", "{,}"),
    }
    return nums


def write_readme_results(live: pd.DataFrame, rows: list[tuple[str, str]], nums: dict, path: Path = ROOT / "README.md"):
    """Regenerate the README's key-results block from the same numbers as the paper."""
    def plain(v):
        text = str(v).replace("\\ensuremath{-}", "−").replace("{,}", ",") if isinstance(v, str) else f"{v:.1f}"
        return re.sub(r"(?<![\w.])-(?=\d)", "−", text)          # typographic minus for negative numbers
    table = ["| Rule | ES reduction | Variance reduction | Max drawdown | Hedge P&L | Funding | Trading costs | Turnover |",
             "|---|---|---|---|---|---|---|---|"]
    for key, label in rows:
        r = live.loc[key]
        table.append(f"| {label} | {r['es_red']:.1f}% | {r['var_red']:.1f}% | {r['mdd']:.1f}% | {plain(signed(r['hedge_pnl']))} "
                     f"| {plain(signed(r['funding']).lstrip('+'))} | {r['cost']:.2f} | {r['turnover']:.1f}× |")
    n = {k.replace(" ", "_"): plain(v) for k, v in nums.items()}
    block = "\n".join([
        "<!-- RESULTS:START (generated by scripts/run_paper.py) -->",
        f"Live data, {n['live_eval_start']} to {n['live_eval_end']} ({n['live_eval_days']} days): a 1-BTC spot book hedged "
        "with the Bitfinex BVIV perpetual, rules frozen on simulated training paths. ES = expected shortfall (97.5%) of daily "
        "returns. Hedge P&L (index P&L minus funding), funding and trading costs (fees, slippage, the perpetual's premium) "
        "in % of BTC notional a year; turnover = perpetual notional traded per year over BTC notional.",
        "", *table, "",
        f"1. **Forecasting.** HAR with implied variance is best at 1 day (QLIKE {n['live_ql_one']} × HAR, Diebold–Mariano "
        f"t = {n['live_dm_one']}) and 7 days ({n['live_ql_seven']} × HAR); the bias-corrected index is best at 30 days "
        f"({n['live_ql_thirty']} × HAR). The simulation has the same winners.",
        f"2. **The hedge works, in crashes.** The minimum-variance hedge cut ES by {n['live_es_always']}% (95% block-bootstrap CI "
        f"{n['live_es_always_lo']}–{n['live_es_always_hi']}%) and the maximum drawdown from {n['live_mdd_unhedged']}% to "
        f"{n['live_mdd_always']}%. Net of funding, its ten best days earned {n['live_top_ten']}% of BTC notional and all other "
        f"days {n['live_ex_top_ten']}%.",
        f"3. **Funding is the price.** Bitfinex funding settled at its ±0.25% cap in {n['live_fund_cap_share']}% of "
        f"{n['live_fund_n']} eight-hour periods; a permanently long contract paid {n['live_fund_annual_pts']} vol points a year "
        f"({n['live_fund_pts_a']} in 2024, {n['live_fund_pts_b']} in 2025, {n['live_fund_pts_c']} in 2026).",
        f"4. **Capacity is the limit.** The perpetual traded a median ${n['live_median_usd']} a day; hedging one BTC needs a "
        f"median {n['live_contracts']} contracts against a median open interest of {n['live_median_oi']}. On "
        f"{n['live_book_date']} the book offered {n['live_depth']} contracts within 2.5% of the mid, the hedge of about "
        f"{n['live_book_btc']} BTC.",
        f"5. **Timing adds size, not information.** The ratchet beat the static hedge by {n['live_d_ratchet']} pp of ES "
        f"reduction (CI [{n['live_d_ratchet_lo']}, {n['live_d_ratchet_hi']}]) but beat only {n['live_placebo_beaten']}% of "
        f"{n['live_placebo_n']} placebos with the same triggers at shifted dates; a 1.5× static hedge did as well "
        f"({n['live_es_onehalf']}%). In the simulation ({n['n_test']} test paths) the ratchet beats the static hedge by "
        f"{n['d_es_ratchet']} pp but its placebo by only {n['d_es_placebo']} pp.",
        f"6. **Speed matters more than the signal.** Executing one hour late cut the ES reduction to {n['live_delay_hour']}%, "
        f"one day late to {n['live_delay_day']}%.",
        "<!-- RESULTS:END -->",
    ])
    text = path.read_text()
    start, end = text.index("<!-- RESULTS:START"), text.index("<!-- RESULTS:END -->") + len("<!-- RESULTS:END -->")
    path.write_text(text[:start] + block + text[end:])


def compile_paper():
    latexmk = shutil.which("latexmk")
    if not latexmk:
        print("latexmk not found -- skipping PDF build")
        return
    subprocess.run([latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=PAPER, check=True,
                   stdout=subprocess.DEVNULL)
    print("built", PAPER / "main.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["all", "train", "test", "robust", "placebo", "placebo-grid", "paper"], default="all")
    ap.add_argument("--quick", action="store_true", help="few paths, for a smoke test")
    ap.add_argument("--train-paths", type=int, default=48)
    ap.add_argument("--test-paths", type=int, default=200)
    ap.add_argument("--robust-paths", type=int, default=96)
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()
    if args.quick:
        # smoke run: tiny samples, written to results/quick/ so the study's outputs stay intact
        global RESULTS, PAPER, GRID_PATHS
        args.train_paths, args.test_paths, args.robust_paths = 8, 8, 8
        RESULTS, PAPER, GRID_PATHS = ROOT / "results" / "quick", ROOT / "results" / "quick" / "paper", 4
        args.no_pdf = True

    RESULTS.mkdir(parents=True, exist_ok=True)
    params, cfg = MarketParams(days=SIM_DAYS), HedgeConfig()
    sel_path = RESULTS / "selection.json"

    if args.stage in ("all", "train"):
        selection = stage_train(args.train_paths, params, cfg)
    else:
        selection = json.loads(sel_path.read_text())
    if args.stage in ("all", "test"):
        stage_test(args.test_paths, params, cfg)
    if args.stage in ("all", "robust"):
        stage_robust(args.robust_paths, params, cfg, selection)
    if args.stage in ("all", "placebo"):
        stage_placebo(args.test_paths, params, cfg, selection)
    if args.stage in ("all", "placebo-grid"):
        stage_placebo_grid(params, cfg)
    if args.stage in ("all", "paper"):
        stage_paper(params, cfg, selection, compile_pdf=not args.no_pdf)


if __name__ == "__main__":
    main()
