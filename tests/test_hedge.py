import numpy as np

from bvivhedge.constants import BARS_PER_DAY
from bvivhedge.hedge import (
    HedgeConfig, Rule, apply_rebalance_band, backtest, build_signals, ewma_beta, ratchet, run_rules,
)
from bvivhedge.metrics import book_metrics, daily_book


def test_pnl_accounting_identity(bars):
    cfg = HedgeConfig(fee_bps=0.0, slippage_bps=0.0)
    pos = np.full(len(bars), 100.0)
    pnl = backtest(bars, pos, cfg)
    s, f = bars["close"].to_numpy(), bars["bviv_mark"].to_numpy()
    assert np.isclose(pnl["btc"].sum(), s[-1] - s[0])
    # decided at the close of bar 0, the position earns from bar 1 onwards and pays funding set at bars 0..T-2
    expected = 100.0 * (f[-1] - f[0]) - 100.0 * bars["bviv_funding"].to_numpy()[:-1].sum()
    assert np.isclose(pnl["hedge"].sum(), expected)


def test_costs_charged_on_traded_notional(bars):
    cfg = HedgeConfig(fee_bps=5.0, slippage_bps=5.0)
    pos = np.zeros(len(bars))
    pos[10:20] = 50.0
    pnl = backtest(bars, pos, cfg)
    mark = bars["bviv_mark"].to_numpy()
    assert np.isclose(pnl["cost"].sum(), 50 * (mark[10] + mark[20]) * 10 / 1e4)


def test_ewma_beta_recovers_slope():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(200_000)
    x = -0.4 * y + rng.standard_normal(y.size)
    beta, n_eff = ewma_beta(x, y, halflife_bars=20_000)
    assert abs(beta[-1] + 0.4) < 0.02 and n_eff[-1] > 10_000


def test_ratchet_jumps_and_decays():
    trig = np.zeros(20, dtype=bool)
    trig[2] = True
    o = ratchet(trig, halflife_bars=4)
    assert o[1] == 0 and o[2] == 1 and np.isclose(o[6], 0.5)


def test_rebalance_band_cuts_trading():
    target = 100 + np.sin(np.arange(1000) / 5.0) * 10
    pos = apply_rebalance_band(target, np.zeros(1000, dtype=np.int8), band=0.25)
    assert np.count_nonzero(np.diff(pos)) < np.count_nonzero(np.diff(target)) / 10


def test_signals_use_no_future_information(bars):
    cfg = HedgeConfig()
    cut = 60 * BARS_PER_DAY
    full = build_signals(bars, cfg)
    part = build_signals(bars.iloc[:cut], cfg)
    for col in ("z", "beta_all", "beta_down", "mv_down"):
        assert np.allclose(full[col].iloc[:cut], part[col], equal_nan=True), col


def test_rules_run_and_unhedged_is_zero(bars):
    rules = (Rule("Unhedged", "unhedged"), Rule("Always-on", "always"),
             Rule("Ratchet", "ratchet", floor=0.5, z_enter=-1.5, halflife_days=1.0))
    results, _ = run_rules(bars, HedgeConfig(), rules)
    assert (results["Unhedged"]["position"] == 0).all()
    base = daily_book(results["Unhedged"])
    m = book_metrics(daily_book(results["Always-on"]), base)
    assert m["notional"] > 0 and np.isfinite(m["es_red"])
    assert (results["Ratchet"]["position"] >= 0).all()


def test_placebo_preserves_trigger_count(bars):
    from bvivhedge.hedge import target_hedge
    cfg = HedgeConfig()
    sig = build_signals(bars, cfg)
    real = Rule("r", "ratchet", z_enter=-1.0, halflife_days=1.0)
    fake = Rule("p", "ratchet", z_enter=-1.0, halflife_days=1.0, placebo_shift_days=17.0)
    h_real, _ = target_hedge(bars, sig, real, cfg)
    h_fake, _ = target_hedge(bars, sig, fake, cfg)
    assert not np.allclose(h_real, h_fake)
    z = np.nan_to_num(sig["z"].to_numpy(), nan=0.0) < -1.0
    assert np.roll(z, 17 * 96).sum() == z.sum()
