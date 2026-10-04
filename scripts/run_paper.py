"""Reproduce every number, table and figure of the paper -- and optionally compile it.

    python scripts/run_paper.py                       # full Monte Carlo study (simulated market)
    python scripts/run_paper.py --quick               # small smoke run
    python scripts/run_paper.py --stage paper         # rebuild figures/tables from cached results
    python scripts/run_paper.py --live --start 2024-04 --end 2026-09
                                                      # same pipeline on Binance + Bitfinex BVIV data

Stages: train (rule selection on training seeds) -> test (Monte Carlo on disjoint
test seeds) -> robust (scenario grid) -> placebo (timing placebo for the ratchet)
-> paper (figures, tables, macros, PDF).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from bvivhedge import plots, report  # noqa: E402
from bvivhedge.experiments import (  # noqa: E402
    TRAIN_SEED0, Protocol, analyse, monte_carlo, paired, rule_from_key, rule_grid, select_rules,
    summarise_forecasts, summarise_metrics,
)
from bvivhedge.hedge import HedgeConfig, Rule  # noqa: E402
from bvivhedge.simulate import MarketParams, simulate_market  # noqa: E402

RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
SIM_DAYS = 900

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
        mc = monte_carlo(range(n), params.with_(**p_over), cfg.with_(**c_over), rules, Protocol(full_forecasts=False))
        m = mc["metrics"]
        med = summarise_metrics(m)
        d = paired(m, "es_red", selection["ratchet"], "always|1")
        rows.append({"scenario": name, "es_always": med.loc["always|1", "es_red"], "es_ratchet": med.loc[selection["ratchet"], "es_red"],
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


def placebo_summary(placebo: pd.DataFrame, ratchet_key: str) -> dict[str, float]:
    w = placebo.pivot(index="seed", columns="strategy", values="es_red")
    fake = w[[c for c in w.columns if c.startswith("placebo|")]].mean(axis=1)
    d = pd.DataFrame({"seed": w.index, "strategy": "diff", "es_red": (w[ratchet_key] - fake).to_numpy()})
    zero = d.assign(strategy="zero", es_red=0.0)
    stats = paired(pd.concat([d, zero]), "es_red", "diff", "zero")
    return {"es_placebo": float(fake.median()), "es_ratchet": float(w[ratchet_key].median()), **stats}


def crash_cluster_diagnostic(params: MarketParams, cfg: HedgeConfig, selection: dict[str, str], n: int = 24) -> dict[str, float]:
    """Hedge-leg P&L on the hedged book's own worst 2.5% days, baseline vs. crashes clustered in stress."""
    rules = [Rule("Unhedged", "unhedged"), rule_from_key("always|1"), rule_from_key(selection["ratchet"])]
    out = {}
    for scen, p in (("base", params), ("cluster", params.with_(**{k: v for k, v, _ in SCENARIOS}["Crashes cluster in stress"]))):
        acc = {"always|1": [], selection["ratchet"]: []}
        for seed in range(n):
            res = analyse(simulate_market(p, seed).bars, cfg, rules, Protocol(full_forecasts=False))
            for key in acc:
                book = res["books"][key].loc[res["hedge_start"]:]
                worst = book["total"] <= book["total"].quantile(0.025)
                acc[key].append(100 * book["hedge"][worst].mean())
        out[scen] = {k: float(sum(v) / len(v)) for k, v in acc.items()}
    return out


def stage_paper(params: MarketParams, cfg: HedgeConfig, selection: dict[str, str], compile_pdf: bool):
    fig_dir, tab_dir = PAPER / "figures", PAPER / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(RESULTS / "test_metrics.csv")
    evals = pd.read_csv(RESULTS / "test_forecast_evals.csv")
    calib = pd.read_csv(RESULTS / "test_calibration.csv")
    robust = pd.read_csv(RESULTS / "robustness.csv")
    placebo = placebo_summary(pd.read_csv(RESULTS / "placebo.csv"), selection["ratchet"])
    sw, rt = selection["switch"], selection["ratchet"]

    # ---- figures on a representative path (test seed 0)
    market = simulate_market(params, 0)
    out = analyse(market.bars, cfg, final_rules(selection), Protocol(full_forecasts=False))
    plots.fig_market(market.bars, fig_dir / "market.pdf")
    episode = plots.pick_episode(out["signals"], out["hedge_start"])
    z_enter = rule_from_key(rt).z_enter
    plots.fig_mechanics(market.bars, out["signals"], out["results"], episode, z_enter,
                        {"always|1": "Always-on", sw: "VWAP-switch", rt: "VWAP-ratchet"}, fig_dir / "mechanics.pdf")
    med_all = summarise_metrics(test)
    iqr = test.groupby("strategy")["es_red"].quantile([0.25, 0.75]).unstack().rename(columns={0.25: "q25", 0.75: "q75"})
    plots.fig_frontier(med_all.drop(index=["Unhedged"]).loc[lambda d: ~d.index.str.startswith("oracle")], iqr,
                       {"switch": sw, "ratchet": rt}, fig_dir / "frontier.pdf")
    fsum = summarise_forecasts(evals)
    plots.fig_forecasts(fsum, fig_dir / "forecasts.pdf")

    # ---- tables
    report.table_calibration(calib, [
        ("rv", "Realised vol, annualised (\\%)", "calibration target $\\approx$ 50", 1),
        ("iv", "Mean BVIV (vol pts)", "35.5 (Aug 2026) to $>$96 (Feb 2026)", 1),
        ("vrp", "IV $-$ subsequent 30d RV (vol pts)", "VRP $\\approx$ 0.14 p.a.\\ in variance", 1),
        ("iv_above_rv", "Share of days IV $>$ RV (\\%)", "VRP positive on average", 0),
        ("corr_daily", "corr(daily return, $\\Delta$IV)", "sign varies by regime", 2),
        ("corr_down", "corr on falling 15m bars", "IV spikes in sell-offs", 2),
        ("corr_up", "corr on rising 15m bars", "IV up in 2023, down in 2025 rallies", 2),
        ("kurt_daily", "Excess kurtosis, daily returns", "fat tails", 1),
        ("worst_day", "Worst daily loss (\\%)", "fat left tail", 1),
    ], tab_dir / "calibration.tex")
    report.table_forecasts(fsum, tab_dir / "forecasts.tex")
    order = [("Unhedged", "Unhedged"), ("always|1", "Always-on (MV)"), (sw, "VWAP-switch"), (rt, "VWAP-ratchet"),
             ("oracle|1", "Oracle (infeasible)")]
    report.table_hedging(med_all, order, [
        ("es_red", "ES red.", 1), ("var_red", "Var red.", 1), ("mdd", "Max DD", 1), ("tail_offset", "Tail offset", 1),
        ("carry", "Carry", 2), ("cost", "Trading", 2), ("turnover", "Turnover", 1), ("time_on", "Time on", 0),
    ], tab_dir / "hedging.tex")
    report.table_robustness(robust.to_dict("records"), tab_dir / "robustness.tex")

    # ---- number macros for the text
    d_rt = paired(test, "es_red", rt, "always|1")
    d_sw = paired(test, "es_red", sw, "always|1")
    d_mdd = paired(test, "mdd", rt, "always|1")
    d_var = paired(test, "var_red", rt, "always|1")
    d_tail = paired(test, "tail_offset", rt, "always|1")
    fc_twin = rt[:-1] + ("1" if rt.endswith("0") else "0")
    d_fc = paired(test, "es_red", fc_twin, rt)
    ladder = med_all[med_all.index.str.startswith("always|")]
    grid = med_all[med_all.index.str.startswith("ratchet|")]
    best_static = grid["hedge_cost"].map(lambda c: ladder.loc[ladder["hedge_cost"] <= c, "es_red"].max())
    above = float((grid["es_red"] > best_static.fillna(-1e9)).mean())
    diag = crash_cluster_diagnostic(params, cfg, selection)
    rr, ss = rule_from_key(rt), rule_from_key(sw)
    f = fsum.set_index(["h", "model"])
    nums = {
        "n_test": f"{test['seed'].nunique()}", "n_train": f"{json.loads((RESULTS / 'selection.json').read_text())['train_seeds'][1] - TRAIN_SEED0}",
        "n_rules": f"{len(rule_grid()) - 1}", "sim_days": f"{params.days}",
        "es_red_always": med_all.loc["always|1", "es_red"], "es_red_ratchet": med_all.loc[rt, "es_red"],
        "es_red_switch": med_all.loc[sw, "es_red"], "es_red_oracle": med_all.loc["oracle|1", "es_red"],
        "var_red_always": med_all.loc["always|1", "var_red"], "var_red_ratchet": med_all.loc[rt, "var_red"],
        "tail_always": med_all.loc["always|1", "tail_offset"], "tail_ratchet": med_all.loc[rt, "tail_offset"],
        "cost_always": f"{med_all.loc['always|1', 'hedge_cost']:.2f}", "cost_ratchet": f"{med_all.loc[rt, 'hedge_cost']:.2f}",
        "cost_switch": f"{med_all.loc[sw, 'hedge_cost']:.2f}", "turn_switch": med_all.loc[sw, "turnover"],
        "turn_always": med_all.loc["always|1", "turnover"], "turn_ratchet": med_all.loc[rt, "turnover"],
        "notional_always": med_all.loc["always|1", "notional"], "notional_ratchet": med_all.loc[rt, "notional"],
        "d_es_ratchet": f"{d_rt['mean']:+.2f}", "d_es_ratchet_lo": f"{d_rt['lo']:+.2f}", "d_es_ratchet_hi": f"{d_rt['hi']:+.2f}",
        "share_ratchet": f"{100 * d_rt['share_pos']:.0f}", "d_es_switch": f"{d_sw['mean']:+.2f}",
        "d_mdd_ratchet": f"{d_mdd['mean']:+.2f}",
        "es_red_always_two": med_all.loc["always|2", "es_red"], "es_red_always_three": med_all.loc["always|3", "es_red"],
        "es_red_always_onehalf": med_all.loc["always|1.5", "es_red"], "cost_always_onehalf": f"{med_all.loc['always|1.5', 'hedge_cost']:.2f}",
        "var_red_always_onehalf": med_all.loc["always|1.5", "var_red"], "var_red_switch": med_all.loc[sw, "var_red"],
        "d_var_ratchet": f"{d_var['mean']:+.2f}", "d_tail_ratchet": f"{d_tail['mean']:+.1f}",
        "share_tail_ratchet": f"{100 * d_tail['share_pos']:.0f}", "d_es_switch_lo": f"{d_sw['lo']:+.2f}", "d_es_switch_hi": f"{d_sw['hi']:+.2f}",
        "d_es_fc": f"{d_fc['mean']:+.2f}", "d_es_fc_lo": f"{d_fc['lo']:+.2f}", "d_es_fc_hi": f"{d_fc['hi']:+.2f}",
        "share_above_ladder": f"{100 * above:.0f}", "n_ratchet_grid": f"{len(grid)}",
        "diag_base_always": f"{diag['base']['always|1']:+.2f}", "diag_base_ratchet": f"{diag['base'][rt]:+.2f}",
        "diag_cluster_always": f"{diag['cluster']['always|1']:+.2f}", "diag_cluster_ratchet": f"{diag['cluster'][rt]:+.2f}",
        "ratchet_z": f"{rr.z_enter:g}", "ratchet_hl": f"{rr.halflife_days:g}", "ratchet_floor": f"{rr.floor:g}",
        "ratchet_fc": "with" if rr.use_forecast else "without",
        "switch_z": f"{ss.z_enter:g}", "switch_exit": f"{ss.z_exit:g}",
        "ql_har_iv_one": f"{f.loc[(1, 'HAR-IV'), 'qlike_ratio']:.2f}", "ql_iv_one": f"{f.loc[(1, 'IV'), 'qlike_ratio']:.2f}",
        "ql_iv_thirty": f"{f.loc[(30, 'IV'), 'qlike_ratio']:.2f}", "ql_har_iv_thirty": f"{f.loc[(30, 'HAR-IV'), 'qlike_ratio']:.2f}",
        "ql_garch_one": f"{f.loc[(1, 'GARCH'), 'qlike_ratio']:.2f}", "ql_ivraw_thirty": f"{f.loc[(30, 'IV-raw'), 'qlike_ratio']:.2f}",
        "ql_har_iv_seven": f"{f.loc[(7, 'HAR-IV'), 'qlike_ratio']:.2f}", "ql_iv_seven": f"{f.loc[(7, 'IV'), 'qlike_ratio']:.2f}",
        "corr_daily": f"{calib['corr_daily'].median():.2f}", "corr_down": f"{calib['corr_down'].median():.2f}",
        "corr_up": f"{calib['corr_up'].median():.2f}", "vrp": calib["vrp"].median(),
        "hedge_carry": f"{params.hedge_carry:g}", "fee": f"{cfg.fee_bps:g}", "slip": f"{cfg.slippage_bps:g}",
        "es_red_placebo": placebo["es_placebo"], "d_es_placebo": f"{placebo['mean']:+.2f}",
        "d_es_placebo_lo": f"{placebo['lo']:+.2f}", "d_es_placebo_hi": f"{placebo['hi']:+.2f}",
        "share_placebo": f"{100 * placebo['share_pos']:.0f}", "n_placebo": f"{len(PLACEBO_SHIFTS)}",
    }
    for r in robust.to_dict("records"):
        key = "rob." + r["scenario"].lower().replace("x0.5", "half").replace("x2", "double").replace(" 0", " zero").replace(" 12", " twelve")
        nums[key] = f"{r['d_mean']:+.2f}"
        nums[key.replace("rob.", "rob lo.")] = f"{r['d_lo']:+.2f}"
        nums[key.replace("rob.", "rob hi.")] = f"{r['d_hi']:+.2f}"
    report.write_numbers(nums, PAPER / "numbers.tex")

    if compile_pdf:
        compile_paper()


def compile_paper():
    latexmk = shutil.which("latexmk")
    if not latexmk:
        print("latexmk not found -- skipping PDF build")
        return
    subprocess.run([latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=PAPER, check=True,
                   stdout=subprocess.DEVNULL)
    print("built", PAPER / "main.pdf")


# --------------------------------------------------------------------------- live data
def run_live(args, cfg: HedgeConfig, selection: dict[str, str]):
    from bvivhedge import data

    klines = data.fetch_binance_klines(args.start, args.end, market="perp" if args.book == "perp" else "spot")
    funding = data.fetch_binance_funding(f"{args.start}-01", f"{args.end}-28")
    perp = data.fetch_bitfinex_bviv(f"{args.start}-01", f"{args.end}-28") if args.iv == "bitfinex" else None
    if args.iv == "bitfinex":
        index = perp["index"]
    elif args.iv == "volmex":
        index = data.fetch_volmex_bviv(args.start, args.end)
    elif args.iv == "dvol":
        index = data.fetch_deribit_dvol(f"{args.start}-01", f"{args.end}-28")
    else:
        index = data.load_index_csv(args.iv)
    bars = data.assemble_live_bars(klines, index, perp, funding)
    out = analyse(bars, cfg.with_(instrument=args.book), final_rules(selection)[:4])
    RESULTS.mkdir(exist_ok=True)
    out["metrics"].to_csv(RESULTS / "live_metrics.csv")
    for h, ev in out["evals"].items():
        ev.to_csv(RESULTS / f"live_forecast_eval_h{h}.csv")
    print(out["metrics"].round(2).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["all", "train", "test", "robust", "placebo", "paper"], default="all")
    ap.add_argument("--quick", action="store_true", help="few paths, for a smoke test")
    ap.add_argument("--train-paths", type=int, default=48)
    ap.add_argument("--test-paths", type=int, default=200)
    ap.add_argument("--robust-paths", type=int, default=96)
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--start", default="2024-04")
    ap.add_argument("--end", default="2026-09")
    ap.add_argument("--iv", default="bitfinex", help="bitfinex | volmex | dvol | path/to/bviv.csv")
    ap.add_argument("--book", default="spot", choices=["spot", "perp"])
    args = ap.parse_args()
    if args.quick:
        args.train_paths, args.test_paths, args.robust_paths = 8, 8, 8

    RESULTS.mkdir(exist_ok=True)
    params, cfg = MarketParams(days=SIM_DAYS), HedgeConfig()
    sel_path = RESULTS / "selection.json"

    if args.live:
        selection = json.loads(sel_path.read_text()) if sel_path.exists() else stage_train(args.train_paths, params, cfg)
        return run_live(args, cfg, selection)
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
    if args.stage in ("all", "paper"):
        stage_paper(params, cfg, selection, compile_pdf=not args.no_pdf)


if __name__ == "__main__":
    main()
