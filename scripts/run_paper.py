"""Reproduce every number, table and figure of the paper -- and optionally compile it.

    python scripts/run_paper.py                       # full Monte Carlo study (simulated market)
    python scripts/run_paper.py --quick               # small smoke run
    python scripts/run_paper.py --stage paper         # rebuild figures/tables from cached results
    python scripts/run_paper.py --live --start 2024-04 --end 2026-09
                                                      # same pipeline on Binance + Bitfinex BVIV data

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
    TRAIN_SEED0, Protocol, analyse, monte_carlo, paired, rule_from_key, rule_grid, select_rules,
    summarise_forecasts, summarise_metrics,
)
from bvivhedge.hedge import HedgeConfig, Rule  # noqa: E402
from bvivhedge.vwap import gate_state  # noqa: E402
from bvivhedge.simulate import MarketParams, simulate_market  # noqa: E402

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


def crash_cluster_diagnostic(params: MarketParams, cfg: HedgeConfig, selection: dict[str, str], n: int = 24) -> dict:
    """Diagnostics on the first ``n`` test paths.

    * hedge-leg P&L on the hedged book's own worst 2.5% of days, baseline vs. crashes clustered in stress;
    * share of the unhedged book's worst 2.5% of days that contain no stress-regime bar;
    * share of VWAP-switch openings not followed by one of the worst 5% of days within 24 hours.
    """
    rules = [Rule("Unhedged", "unhedged"), rule_from_key("always|1"), rule_from_key(selection["switch"]),
             rule_from_key(selection["ratchet"])]
    sw = rule_from_key(selection["switch"])
    out, outside, false_alarm = {}, [], []
    for scen, p in (("base", params), ("cluster", params.with_(**{k: v for k, v, _ in SCENARIOS}["Crashes cluster in stress"]))):
        acc = {"always|1": [], selection["ratchet"]: []}
        for seed in range(n):
            bars = simulate_market(p, seed).bars
            res = analyse(bars, cfg, rules, Protocol(full_forecasts=False))
            start = res["hedge_start"]
            for key in acc:
                book = res["books"][key].loc[start:]
                worst = book["total"] <= book["total"].quantile(0.025)
                acc[key].append(100 * book["hedge"][worst].mean())
            if scen == "base":
                unhedged = res["books"]["Unhedged"].loc[start:, "total"]
                stress_day = (bars["regime"] == 1).groupby(bars.index.floor("1D")).any().reindex(unhedged.index)
                tail = unhedged <= unhedged.quantile(0.025)
                outside.append(float((~stress_day[tail]).mean()))
                gate = pd.Series(gate_state(res["signals"]["z"].to_numpy(), sw.z_enter, sw.z_exit, sw.min_hold),
                                 index=bars.index).loc[start:]
                opens = gate.index[(gate.diff() == 1).to_numpy()]
                bad = set(unhedged.index[unhedged <= unhedged.quantile(0.05)])
                hits = [any(d in bad for d in (t.floor("1D"), (t + pd.Timedelta("1D")).floor("1D"))) for t in opens]
                false_alarm.append(1.0 - float(np.mean(hits)) if hits else np.nan)
        out[scen] = {k: float(sum(v) / len(v)) for k, v in acc.items()}
    out["tail_outside_stress"] = float(np.mean(outside))
    out["switch_false_alarm"] = float(np.nanmean(false_alarm))
    out["n"] = n
    return out


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

    # ---- figures on a representative path (test seed 0)
    market = simulate_market(params, 0)
    out = analyse(market.bars, cfg, final_rules(selection), Protocol(full_forecasts=False))
    plots.fig_market(market.bars, fig_dir / "market.pdf")
    z_enter = rule_from_key(rt).z_enter
    episode = plots.pick_episode(out["signals"], out["hedge_start"], z_enter)
    plots.fig_mechanics(market.bars, out["signals"], out["results"], episode, z_enter,
                        {"always|1": "Always-on", sw: "VWAP-switch", rt: "VWAP-ratchet"}, fig_dir / "mechanics.pdf")
    med_all = summarise_metrics(test)
    iqr = test.groupby("strategy")["es_red"].quantile([0.25, 0.75]).unstack().rename(columns={0.25: "q25", 0.75: "q75"})
    plots.fig_frontier(med_all.drop(index=["Unhedged"]).loc[lambda d: ~d.index.str.startswith("oracle")], iqr,
                       {"switch": sw, "ratchet": rt}, fig_dir / "frontier.pdf")
    fsum = summarise_forecasts(evals)
    plots.fig_forecasts(fsum, fig_dir / "forecasts.pdf")

    # ---- tables
    calib = calib.assign(vrp_var=(calib["iv"] / 100) ** 2 - (calib["rv"] / 100) ** 2)
    report.table_calibration(calib, [
        ("rv", "Realised vol, annualised (\\%)", "calibration target $\\approx$ 50", 1),
        ("iv", "Mean BVIV (vol pts)", "35.5 (Aug 2026) to $>$96 (Feb 2026)", 1),
        ("vrp", "IV $-$ subsequent 30d RV (vol pts)", "positive on average", 1),
        ("vrp_var", "IV$^2-$RV$^2$ (variance units)", "$\\approx$ 0.14 in 2017--22 (higher vol)", 2),
        ("iv_above_rv", "Share of days IV $>$ RV (\\%)", "VRP positive on average", 0),
        ("corr_daily", "corr(daily return, $\\Delta$IV)", "sign varies by regime", 2),
        ("corr_down", "corr on falling 15m bars", "IV spikes in sell-offs", 2),
        ("corr_up", "corr on rising 15m bars", "IV up in 2023, down in 2025 rallies", 2),
        ("kurt_daily", "Excess kurtosis, daily returns", "fat tails", 1),
        ("worst_day", "Worst daily loss (\\%)", "fat left tail", 1),
    ], tab_dir / "calibration.tex")
    report.table_forecasts(fsum, tab_dir / "forecasts.tex")
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
    d_mdd = paired(test, "mdd", rt, "always|1")
    d_var = paired(test, "var_red", rt, "always|1")
    d_tail = paired(test, "tail_offset", rt, "always|1")
    fc_twin = rt[:-1] + ("1" if rt.endswith("0") else "0")
    d_fc = paired(test, "es_red", fc_twin, rt)
    ladder = med_all[med_all.index.str.startswith("always|")].sort_values("hedge_cost")
    grid = med_all[med_all.index.str.startswith("ratchet|")]
    # the blue line of the frontier figure: unhedged origin, then the scaled static hedges
    line_x = np.r_[0.0, ladder["hedge_cost"].to_numpy()]
    line_y = np.maximum.accumulate(np.r_[0.0, ladder["es_red"].to_numpy()])   # efficient envelope of static scalings
    above_mask = grid["es_red"].to_numpy() > np.interp(grid["hedge_cost"].to_numpy(), line_x, line_y)
    above = float(above_mask.mean())
    above_full_core = int((above_mask & grid.index.str.contains(r"\|1\|[01]$", regex=True)).sum())
    diag = crash_cluster_diagnostic(params, cfg, selection)
    rr, ss = rule_from_key(rt), rule_from_key(sw)
    f = fsum.set_index(["h", "model"])
    loss = evals.pivot_table(index=["h", "seed"], columns="model", values="qlike")
    blowups = int((loss["HAR-IV"] > 2 * loss["HAR"]).sum())          # paths x horizons with a collapse
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
        "d_es_ratchet": signed(d_rt['mean']), "d_es_ratchet_lo": signed(d_rt['lo']), "d_es_ratchet_hi": signed(d_rt['hi']),
        "share_ratchet": f"{100 * d_rt['share_pos']:.0f}", "d_es_switch": signed(d_sw['mean']),
        "d_mdd_ratchet": signed(d_mdd['mean']),
        "es_red_always_two": med_all.loc["always|2", "es_red"], "es_red_always_three": med_all.loc["always|3", "es_red"],
        "es_red_always_onehalf": med_all.loc["always|1.5", "es_red"], "cost_always_onehalf": f"{med_all.loc['always|1.5', 'hedge_cost']:.2f}",
        "var_red_always_onehalf": med_all.loc["always|1.5", "var_red"], "var_red_switch": med_all.loc[sw, "var_red"],
        "d_var_ratchet": signed(d_var['mean']), "d_tail_ratchet": signed(d_tail['mean'], 1),
        "share_tail_ratchet": f"{100 * d_tail['share_pos']:.0f}", "d_es_switch_lo": signed(d_sw['lo']), "d_es_switch_hi": signed(d_sw['hi']),
        "d_es_fc": signed(d_fc['mean']), "d_es_fc_lo": signed(d_fc['lo']), "d_es_fc_hi": signed(d_fc['hi']),
        "share_above_ladder": f"{100 * above:.0f}", "n_ratchet_grid": f"{len(grid)}",
        "n_above_ladder": f"{int(round(above * len(grid)))}", "n_above_full_core": f"{above_full_core}",
        "ratchet_twin": "unsized" if rr.use_forecast else "forecast-sized",
        "tail_outside_stress": f"{100 * diag['tail_outside_stress']:.0f}", "switch_false_alarm": f"{100 * diag['switch_false_alarm']:.0f}",
        "diag_paths": f"{diag['n']}",
        "pgrid_floor_zero_abs": f"{abs(pgrid['floor_zero']):.2f}", "pgrid_sig_pos_zero": f"{pgrid['sig_pos_zero']}",
        "pgrid_n_zero": f"{pgrid['n_zero']}", "pgrid_neg_point": f"{pgrid['neg_point']}",
        "cost_placebo": f"{med_tab.loc['placebo', 'hedge_cost']:.2f}",
        "es_vix_always": robust.set_index("scenario").loc["VIX-like spot-vol", "es_always"],
        "es_inverse_always": robust.set_index("scenario").loc["Inverse leverage", "es_always"],
        "es_cluster_always": robust.set_index("scenario").loc["Crashes cluster in stress", "es_always"],
        "es_cluster_ratchet": robust.set_index("scenario").loc["Crashes cluster in stress", "es_ratchet"],
        "rob_cost_min": signed(robust[robust.scenario.str.contains("costs|carry")]["d_mean"].min()),
        "rob_cost_max": signed(robust[robust.scenario.str.contains("costs|carry")]["d_mean"].max()),
        "vrp_var": f"{((calib['iv'] / 100) ** 2 - (calib['rv'] / 100) ** 2).median():.2f}",
        "d_es_switch_abs": f"{abs(d_sw['mean']):.2f}", "d_var_ratchet_abs": f"{abs(d_var['mean']):.2f}",
        "diag_base_always": signed(diag['base']['always|1']), "diag_base_ratchet": signed(diag['base'][rt]),
        "diag_cluster_always": signed(diag['cluster']['always|1']), "diag_cluster_ratchet": signed(diag['cluster'][rt]),
        "ratchet_z": f"{rr.z_enter:g}", "ratchet_hl": f"{rr.halflife_days:g}", "ratchet_floor": f"{rr.floor:g}",
        "ratchet_fc": "with" if rr.use_forecast else "without",
        "switch_z": f"{ss.z_enter:g}", "switch_exit": f"{ss.z_exit:g}",
        "ql_har_iv_one": f"{f.loc[(1, 'HAR-IV'), 'qlike_ratio']:.2f}", "ql_iv_one": f"{f.loc[(1, 'IV'), 'qlike_ratio']:.2f}",
        "ql_iv_thirty": f"{f.loc[(30, 'IV'), 'qlike_ratio']:.2f}", "ql_har_iv_thirty": f"{f.loc[(30, 'HAR-IV'), 'qlike_ratio']:.2f}",
        "ql_garch_one": f"{f.loc[(1, 'GARCH'), 'qlike_ratio']:.2f}", "ql_ivraw_thirty": f"{f.loc[(30, 'IV-raw'), 'qlike_ratio']:.2f}",
        "ql_har_iv_seven": f"{f.loc[(7, 'HAR-IV'), 'qlike_ratio']:.2f}", "ql_iv_seven": f"{f.loc[(7, 'IV'), 'qlike_ratio']:.2f}",
        "dm_win_har_iv_one": f"{f.loc[(1, 'HAR-IV'), 'dm_win']:.0f}", "dm_loss_har_iv_one": f"{f.loc[(1, 'HAR-IV'), 'dm_loss']:.0f}",
        "dm_win_iv_thirty": f"{f.loc[(30, 'IV'), 'dm_win']:.0f}", "dm_loss_iv_thirty": f"{f.loc[(30, 'IV'), 'dm_loss']:.0f}",
        "dm_win_har_iv_seven": f"{f.loc[(7, 'HAR-IV'), 'dm_win']:.0f}", "dm_loss_har_iv_seven": f"{f.loc[(7, 'HAR-IV'), 'dm_loss']:.0f}",
        "ql_har_iv_one_mean": f"{f.loc[(1, 'HAR-IV'), 'qlike']:.3f}", "ql_har_one_mean": f"{f.loc[(1, 'HAR'), 'qlike']:.3f}",
        "har_iv_blowups": f"{blowups}",
        "corr_daily": f"{calib['corr_daily'].median():.2f}", "corr_down": f"{calib['corr_down'].median():.2f}",
        "corr_up": f"{calib['corr_up'].median():.2f}", "vrp": calib["vrp"].median(),
        "hedge_carry": f"{params.hedge_carry:g}", "fee": f"{cfg.fee_bps:g}", "slip": f"{cfg.slippage_bps:g}",
        "es_red_placebo": placebo["es_placebo"], "d_es_placebo": signed(placebo['mean']),
        "d_es_placebo_lo": signed(placebo['lo']), "d_es_placebo_hi": signed(placebo['hi']),
        "share_placebo": f"{100 * placebo['share_pos']:.0f}", "n_placebo": f"{len(PLACEBO_SHIFTS)}",
        "pgrid_n": f"{pgrid['n']}", "pgrid_mean": signed(pgrid['mean']), "pgrid_min": signed(pgrid['min']),
        "pgrid_max": signed(pgrid['max']),
        "pgrid_max_abs": f"{abs(pgrid['max']):.2f}", "pgrid_sig_pos": f"{pgrid['sig_pos']}", "pgrid_sig_neg": f"{pgrid['sig_neg']}",
        "pgrid_paths": f"{GRID_PATHS}", "pgrid_shifts": f"{len(GRID_SHIFTS)}",
        "pgrid_floor_zero": signed(pgrid['floor_zero']), "pgrid_floor_half": signed(pgrid['floor_half']),
        "pgrid_floor_one": signed(pgrid['floor_one']), "pgrid_es_floor_zero": pgrid["es_floor_zero"],
        "pgrid_es_floor_one": pgrid["es_floor_one"], "pgrid_best_es": pgrid["best_es"],
        "pgrid_best_cost": f"{pgrid['best_cost']:.1f}", "pgrid_corr": f"{pgrid['corr']:.2f}",
        "pgrid_always_es": pgrid["always_es"],
    }
    for r in robust.to_dict("records"):
        key = "rob." + r["scenario"].lower().replace("x0.5", "half").replace("x2", "double").replace(" 0", " zero").replace(" 12", " twelve")
        nums[key] = signed(r['d_mean'])
        nums[key.replace("rob.", "rob lo.")] = signed(r['d_lo'])
        nums[key.replace("rob.", "rob hi.")] = signed(r['d_hi'])
    report.write_numbers(nums, PAPER / "numbers.tex")
    if PAPER == ROOT / "paper":
        write_readme_results(med_tab, [("always|1", "Always-on minimum-variance hedge"), ("always|1.5", "Always-on, scaled 1.5×"),
                                       (sw, "VWAP-switch"), (rt, "**VWAP-ratchet** (MV core + breakdown overlay)"),
                                       ("placebo", "Ratchet with placebo timing")], nums)

    if compile_pdf:
        compile_paper()


def write_readme_results(med: pd.DataFrame, rows: list[tuple[str, str]], nums: dict, path: Path = ROOT / "README.md"):
    """Regenerate the README's key-results block from the same numbers as the paper."""
    def plain(v):
        text = str(v).replace("\\ensuremath{-}", "−") if isinstance(v, str) else f"{v:.1f}"
        return re.sub(r"(?<![\w.])-(?=\d)", "−", text)          # typographic minus for negative numbers
    table = ["| Rule | ES reduction | Variance reduction | Hedge cost (% p.a.) | Turnover |", "|---|---|---|---|---|"]
    for key, label in rows:
        r = med.loc[key]
        table.append(f"| {label} | {r['es_red']:.1f}% | {r['var_red']:.1f}% | {r['hedge_cost']:.2f} | {r['turnover']:.1f}× |")
    n = {k: plain(v) for k, v in nums.items()}
    block = "\n".join([
        "<!-- RESULTS:START (generated by scripts/run_paper.py) -->",
        f"Out-of-sample results on {n['n_test']} simulated test paths; rules were selected on {n['n_train']} separate "
        "training paths. ES = expected shortfall (97.5%) of daily returns, net of carry and trading costs. "
        "Turnover = BVIV-perp notional traded per year as a multiple of BTC notional.",
        "", *table, "",
        f"1. **Forecasting.** HAR-IV (in logs) is best at 1 day (QLIKE {n['ql_har_iv_one']} × HAR) and 7 days "
        f"({n['ql_har_iv_seven']} × HAR); the bias-corrected implied index is best at 30 days ({n['ql_iv_thirty']} × HAR).",
        f"2. **BVIV hedges are partial.** The spot-vol correlation is weak ({n['corr_daily']} daily) and changes sign by regime; "
        f"the minimum-variance hedge cuts ES by {n['es_red_always']}% for {n['cost_always']}% of notional a year.",
        f"3. **Naive VWAP switching fails.** Its ES reduction trails the always-on hedge by {n['d_es_switch_abs']} pp "
        f"at a turnover of {n['turn_switch']}×.",
        f"4. **The ratchet's gain over the MV hedge ({n['d_es_ratchet']} pp ES, 95% CI [{n['d_es_ratchet_lo']}, "
        f"{n['d_es_ratchet_hi']}]) is a size effect:** a placebo with the same trigger statistics but scrambled timing "
        f"does as well ({n['d_es_placebo']} pp, CI [{n['d_es_placebo_lo']}, {n['d_es_placebo_hi']}]), and a 1.5× static "
        f"hedge reaches the same ES reduction ({n['es_red_always_onehalf']}%).",
        f"5. **VWAP timing information is real but secondary.** Across {n['pgrid_n']} ratchet configurations (first "
        f"{n['pgrid_paths']} test paths), {n['pgrid_sig_pos']} beat their placebo at the 5% level and none loses significantly. "
        f"Timing adds {n['pgrid_floor_zero']} pp without a core hedge and {n['pgrid_floor_one']} pp with a full core.",
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


# --------------------------------------------------------------------------- live data
def run_live(args, cfg: HedgeConfig, selection: dict[str, str]):
    from bvivhedge import data

    first = f"{args.start}-01"
    last = (pd.Period(args.end, "M") + 1).start_time.strftime("%Y-%m-%d")   # exclusive end: whole last month
    klines = data.fetch_binance_klines(args.start, args.end, market="perp" if args.book == "perp" else "spot")
    funding = data.fetch_binance_funding(first, last)
    perp = {"bitfinex": data.fetch_bitfinex_bviv, "hyperliquid": data.fetch_hyperliquid_bviv}.get(args.iv)
    perp = perp(first, last) if perp else None
    if args.iv in ("bitfinex", "hyperliquid"):
        index = perp["index"]
    elif args.iv == "volmex":
        index = data.fetch_volmex_bviv(args.start, args.end)
    elif args.iv == "dvol":
        index = data.fetch_deribit_dvol(first, last)
    else:
        index = data.load_index_csv(args.iv)
    bars = data.assemble_live_bars(klines, index, perp, funding)
    days = (bars.index[-1] - bars.index[0]).days
    need = Protocol().forecast_eval_start + 90
    if days < need:
        raise SystemExit(f"live window has {days} days; at least {need} are needed (forecast evaluation starts on day "
                         f"{Protocol().forecast_eval_start}, hedging on day {Protocol().hedge_eval_start})")
    base = rule_from_key(selection["ratchet"])
    placebos = [Rule(f"placebo|{d}", "ratchet", z_enter=base.z_enter, halflife_days=base.halflife_days, floor=base.floor,
                     use_forecast=base.use_forecast, placebo_shift_days=float(d))
                for d in range(7, days - 7, 7)]                       # every weekly shift: the placebo distribution
    out = analyse(bars, cfg.with_(instrument=args.book), [*final_rules(selection)[:4], *placebos])
    RESULTS.mkdir(exist_ok=True)
    out["metrics"].to_csv(RESULTS / "live_metrics.csv")
    for h, ev in out["evals"].items():
        ev.to_csv(RESULTS / f"live_forecast_eval_h{h}.csv")
    m = out["metrics"]
    fake = m.loc[m.index.str.startswith("placebo|"), "es_red"]
    real = m.loc[selection["ratchet"], "es_red"]
    print(m.loc[~m.index.str.startswith("placebo|")].round(2).to_string())
    print(f"timing placebo: ratchet ES reduction {real:.2f}% vs. {len(fake)} weekly shifts: mean {fake.mean():.2f}%, "
          f"share of shifts beaten {100 * (real > fake).mean():.0f}% (one path: judge against this distribution)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["all", "train", "test", "robust", "placebo", "placebo-grid", "paper"], default="all")
    ap.add_argument("--quick", action="store_true", help="few paths, for a smoke test")
    ap.add_argument("--train-paths", type=int, default=48)
    ap.add_argument("--test-paths", type=int, default=200)
    ap.add_argument("--robust-paths", type=int, default=96)
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--start", default="2024-04")
    ap.add_argument("--end", default="2026-09")
    ap.add_argument("--iv", default="bitfinex", help="bitfinex | hyperliquid | volmex | dvol | path/to/bviv.csv")
    ap.add_argument("--book", default="spot", choices=["spot", "perp"])
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
    if args.stage in ("all", "placebo-grid"):
        stage_placebo_grid(params, cfg)
    if args.stage in ("all", "paper"):
        stage_paper(params, cfg, selection, compile_pdf=not args.no_pdf)


if __name__ == "__main__":
    main()
