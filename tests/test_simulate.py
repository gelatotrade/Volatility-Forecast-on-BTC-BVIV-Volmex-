import numpy as np
import pandas as pd

from bvivhedge.constants import BARS_PER_DAY, DT
from bvivhedge.simulate import (
    MarketParams, egarch_shock, expected_average_variance, expected_next_index,
    simulate_market, stationary_regime_probs,
)


def test_schema_and_kline_consistency(bars):
    assert len(bars) == 120 * BARS_PER_DAY
    assert bars.index.tz is not None and bars.index.freq is not None
    vwap = bars["quote_volume"] / bars["volume"]
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all()
    assert ((vwap >= bars["low"] * (1 - 1e-9)) & (vwap <= bars["high"] * (1 + 1e-9))).all()
    assert (bars["volume"] > 0).all() and (bars["bviv"] > 0).all()


def test_reproducible():
    a = simulate_market(MarketParams(days=5), seed=3).bars
    b = simulate_market(MarketParams(days=5), seed=3).bars
    pd.testing.assert_frame_equal(a, b)


def test_stationary_regime_probs_sum_to_one():
    p = stationary_regime_probs(MarketParams())
    assert np.isclose(p.sum(), 1.0) and (p > 0).all()


def test_egarch_shock_moments():
    rng = np.random.default_rng(0)
    z, e = rng.standard_normal((2, 400_000))
    s = egarch_shock(z, np.full(z.size, -0.5), 0.35, e)
    assert abs(s.mean()) < 0.01 and abs(s.std() - 1) < 0.01
    assert abs(np.corrcoef(z, s)[0, 1] + 0.5) < 0.01
    # the size effect: variance shocks are larger after big moves of either sign
    assert s[np.abs(z) > 2].mean() > 0.2


def test_perp_funding_is_an_unbiased_drift_transfer(market):
    """E_t[I_{t+1}] from the closed form equals a brute-force Monte Carlo of the next bar."""
    p, lat, reg = market.params, market.meta["latent"], market.bars["regime"].to_numpy()
    rng = np.random.default_rng(1)
    phi_s, phi_f = np.exp(-p.kappa_slow * DT), np.exp(-p.kappa_fast * DT)
    sd_s = p.xi_slow * np.sqrt((1 - phi_s**2) / (2 * p.kappa_slow))
    sd_f = p.xi_fast * np.sqrt((1 - phi_f**2) / (2 * p.kappa_fast))
    phi_e = 0.5 ** (1 / (p.sentiment_halflife_days * BARS_PER_DAY))
    sd_e = p.sentiment_sd * np.sqrt(1 - phi_e**2)
    scale = 100 * np.sqrt(np.exp(p.iv_log_premium))
    leave = 1 / (np.asarray(p.regime_days) * BARS_PER_DAY)
    exits = np.cumsum(np.asarray(p.regime_exits), axis=1)
    k = 200_000
    for t in (1000, 6000, 11000):
        i = reg[t]
        nxt = np.where(rng.random(k) < leave[i], np.searchsorted(exits[i], rng.random(k), side="right"), i)
        z, e1, e2, e3 = rng.standard_normal((4, k))
        jumps = np.where(rng.random(k) < np.asarray(p.jump_rate)[nxt] * DT,
                         np.asarray(p.jump_mean)[nxt] + p.jump_std * rng.standard_normal(k), 0.0)
        xs = phi_s * lat["x_slow"][t] + sd_s * e2
        xf = (phi_f * lat["x_fast"][t] + (1 - phi_f) * 2 * np.asarray(p.regime_log_vol)[nxt]
              + sd_f * egarch_shock(z, np.asarray(p.regime_theta)[nxt], p.size_effect, e1)
              + p.jump_var_kick * np.abs(jumps) / p.jump_std)
        se = phi_e * lat["sentiment"][t] + sd_e * egarch_shock(z, np.asarray(p.sentiment_theta)[nxt], p.sentiment_size, e3)
        draws = scale * np.sqrt(expected_average_variance(xs, xf, nxt, p)) * np.exp(se)
        closed = expected_next_index(lat["x_slow"][t:t + 1], lat["x_fast"][t:t + 1],
                                     lat["sentiment"][t:t + 1], reg[t:t + 1], p)[0]
        assert abs(closed - draws.mean()) < 4 * draws.std() / np.sqrt(k)


def test_index_is_the_model_expectation_of_average_variance():
    """Closed form (Feynman-Kac over regime paths) vs brute-force simulation of the next 30 days."""
    p = MarketParams()
    rng = np.random.default_rng(3)
    phi_s, phi_f = np.exp(-p.kappa_slow * DT), np.exp(-p.kappa_fast * DT)
    sd_s = p.xi_slow * np.sqrt((1 - phi_s**2) / (2 * p.kappa_slow))
    sd_f = p.xi_fast * np.sqrt((1 - phi_f**2) / (2 * p.kappa_fast))
    leave = 1 / (np.asarray(p.regime_days) * BARS_PER_DAY)
    exits = np.cumsum(np.asarray(p.regime_exits), axis=1)
    m, lam, jm = 2 * np.asarray(p.regime_log_vol), np.asarray(p.jump_rate), np.asarray(p.jump_mean)
    k, horizon = 6000, 30 * BARS_PER_DAY
    xs, xf, reg = np.zeros(k), np.full(k, m[0]), np.full(k, 1)          # stress regime, calm variance level
    acc = np.zeros(k)
    for _ in range(horizon):
        move = rng.random(k) < leave[reg]
        if move.any():
            reg = reg.copy()
            reg[move] = [np.searchsorted(exits[r], u, side="right") for r, u in zip(reg[move], rng.random(move.sum()))]
        z, e1, e2 = rng.standard_normal((3, k))
        jumps = np.where(rng.random(k) < lam[reg] * DT, jm[reg] + p.jump_std * rng.standard_normal(k), 0.0)
        xs = phi_s * xs + sd_s * e2
        xf = (phi_f * xf + (1 - phi_f) * m[reg] + sd_f * egarch_shock(z, np.asarray(p.regime_theta)[reg], p.size_effect, e1)
              + p.jump_var_kick * np.abs(jumps) / p.jump_std)
        acc += np.exp(2 * np.log(p.vol_level) + xs + xf) + lam[reg] * (jm[reg] ** 2 + p.jump_std**2)
    draws = acc / horizon
    closed = expected_average_variance(np.zeros(1), np.full(1, m[0]), np.ones(1, dtype=int), p)[0]
    assert abs(closed - draws.mean()) < 4 * draws.std() / np.sqrt(k) + 0.005 * draws.mean()
