"""Calibrated Monte Carlo market: BTC spot/perp, the BVIV index and a BVIV perpetual.

The generator reproduces the stylised facts the hedging problem hinges on, and
little else:

* two-factor log-variance (slow + fast), so volatility clusters at two speeds;
* a hidden three-state regime (calm / stress / euphoria) that shifts drift,
  volatility level and -- crucially -- the sign of the spot-vol correlation;
* EGARCH-type variance innovations (Nelson 1991): log-variance loads on the
  signed return shock (leverage, regime-dependent sign) *and* on its size, so
  implied volatility jumps on crashes but barely moves -- or rises -- in rallies;
* compound-Poisson jumps, clustered in stress, that kick variance up (the gap
  risk a gated hedge can miss);
* intraday and weekend seasonality in volatility and volume;
* an implied-volatility index equal to the model's own 30-day variance forecast,
  inflated by a variance risk premium and perturbed by persistent sentiment;
* a BVIV perpetual whose mark prices in the index's mean reversion and whose
  funding charges long-volatility holders a calibrated carry.

Every 15-minute bar is assembled from fifteen one-minute steps, so OHLC and the
bar's volume-weighted price behave like exchange klines (where
``quote_volume / volume`` is the bar VWAP).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.signal import lfilter
from scipy.stats import norm

from .constants import BARS_PER_DAY, DAYS_PER_YEAR, DT, IV_HORIZON_DAYS, MINUTES_PER_BAR

REGIMES = ("calm", "stress", "euphoria")
CALM, STRESS, EUPHORIA = range(3)

#: Relative volatility by UTC hour: quiet Asian early morning, peak around the
#: US equity open (13:30-16:00 UTC).  Normalised inside :func:`seasonal_factor`.
HOURLY_VOL = (
    1.05, 0.95, 0.90, 0.85, 0.80, 0.78, 0.80, 0.88, 0.98, 0.95, 0.92, 0.92,
    1.00, 1.15, 1.35, 1.30, 1.20, 1.10, 1.05, 1.00, 1.05, 1.00, 0.95, 1.00,
)


@dataclass(frozen=True)
class MarketParams:
    """Parameters of the simulated market (annualised unless stated)."""

    days: int = 730
    start: str = "2024-01-01"
    s0: float = 60_000.0

    # Two-factor log-variance: v = exp(theta0 + x_slow + x_fast)
    vol_level: float = 0.42
    kappa_slow: float = 6.0          # half-life ~ 42 days
    xi_slow: float = 1.4
    kappa_fast: float = 90.0         # half-life ~ 2.8 days
    xi_fast: float = 6.5

    # Hidden regimes: calm, stress, euphoria
    regime_log_vol: tuple[float, float, float] = (-0.12, 0.40, 0.18)
    regime_drift: tuple[float, float, float] = (0.20, -2.00, 1.50)
    regime_theta: tuple[float, float, float] = (-0.30, -0.60, 0.25)  # corr(return shock, variance shock)
    size_effect: float = 0.35        # loading of the variance shock on |z| - E|z|
    regime_days: tuple[float, float, float] = (30.0, 5.0, 10.0)
    regime_exits: tuple[tuple[float, float, float], ...] = (
        (0.00, 0.55, 0.45),
        (0.75, 0.00, 0.25),
        (0.65, 0.35, 0.00),
    )

    # Jumps in log price (rate and mean by regime); each jump kicks the fast variance factor up
    jump_rate: tuple[float, float, float] = (8.0, 40.0, 10.0)
    jump_mean: tuple[float, float, float] = (-0.005, -0.03, 0.01)
    jump_std: float = 0.05
    jump_var_kick: float = 0.6

    # Seasonality
    weekend_factor: float = 0.75
    hourly_vol: tuple[float, ...] = HOURLY_VOL

    # Implied volatility index (BVIV)
    iv_log_premium: float = 0.22     # Q/P variance ratio exp(0.22) ~ 1.25, i.e. IV ~ 1.12 x expected RV
    sentiment_halflife_days: float = 4.0
    sentiment_sd: float = 0.06       # stationary s.d. of log-IV sentiment
    sentiment_theta: tuple[float, float, float] = (-0.30, -0.50, 0.20)
    sentiment_size: float = 0.20
    iv_noise: float = 0.15           # vol points, i.i.d. fixing noise

    # BVIV perpetual
    hedge_carry: float = 6.0         # vol points per year paid by longs (~11% APR interest leg at BVIV 55)

    # BTC perpetual funding (annualised rate paid by longs)
    btc_funding_base: float = 0.06
    btc_funding_regime: tuple[float, float, float] = (0.00, -0.08, 0.20)
    btc_funding_sd: float = 0.05
    btc_funding_halflife_days: float = 1.0

    # Volume (BTC per 15-minute bar)
    volume_btc: float = 900.0
    volume_vol_elasticity: float = 1.0
    volume_noise: float = 0.35

    def with_(self, **changes) -> "MarketParams":
        return replace(self, **changes)


def seasonal_factor(index: pd.DatetimeIndex, params: MarketParams) -> np.ndarray:
    """Volatility multiplier per bar with E[s^2] = 1 over a full week."""
    hour = index.hour.to_numpy()
    weekend = index.dayofweek.to_numpy() >= 5
    s = np.asarray(params.hourly_vol)[hour] * np.where(weekend, params.weekend_factor, 1.0)
    week = pd.date_range("2024-01-01", periods=7 * BARS_PER_DAY, freq=f"{MINUTES_PER_BAR}min", tz="UTC")
    ws = np.asarray(params.hourly_vol)[week.hour] * np.where(week.dayofweek >= 5, params.weekend_factor, 1.0)
    return s / np.sqrt(np.mean(ws**2))


def _simulate_regimes(n: int, params: MarketParams, rng: np.random.Generator) -> np.ndarray:
    leave = 1.0 / (np.asarray(params.regime_days) * BARS_PER_DAY)
    exits = np.cumsum(np.asarray(params.regime_exits), axis=1)
    u_leave, u_dest = rng.random(n), rng.random(n)
    reg = np.empty(n, dtype=np.int8)
    state = CALM
    for t in range(n):
        if u_leave[t] < leave[state]:
            state = int(np.searchsorted(exits[state], u_dest[t], side="right"))
        reg[t] = state
    return reg


def stationary_regime_probs(params: MarketParams) -> np.ndarray:
    """Long-run share of time in each regime (continuous-time Markov chain)."""
    q = generator_matrix(params)
    a = np.vstack([q.T, np.ones(3)])
    b = np.r_[np.zeros(3), 1.0]
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _ar1(innov: np.ndarray, phi: float, x0: float = 0.0) -> np.ndarray:
    """x_t = phi x_{t-1} + innov_t, vectorised."""
    out, _ = lfilter([1.0], [1.0, -phi], innov, zi=[phi * x0])
    return out


_ABS_MEAN = np.sqrt(2.0 / np.pi)


def egarch_shock(z: np.ndarray, sign_load: np.ndarray, size_load: float, noise: np.ndarray) -> np.ndarray:
    """Unit-variance shock loading on z (signed) and |z| - E|z| (size), as in EGARCH.

    corr(z, shock) = sign_load; the size term makes large moves of either sign
    raise variance, which produces the asymmetric spot-vol response.
    """
    size_var = size_load**2 * (1.0 - _ABS_MEAN**2)
    rest = np.sqrt(np.clip(1.0 - sign_load**2 - size_var, 0.0, None))
    return sign_load * z + size_load * (np.abs(z) - _ABS_MEAN) + rest * noise


def generator_matrix(params: MarketParams) -> np.ndarray:
    """Continuous-time generator Q of the regime chain (per year)."""
    rates = DAYS_PER_YEAR / np.asarray(params.regime_days)
    q = np.asarray(params.regime_exits) * rates[:, None]
    q[np.diag_indices(3)] = -rates
    return q


def effective_fast_level(params: MarketParams) -> tuple[np.ndarray, np.ndarray]:
    """Regime levels of the fast log-variance factor including the drift that jump kicks add,
    and the kicks' variance contribution rate (per year)."""
    p = params
    mu, sd = np.asarray(p.jump_mean), p.jump_std
    lam = np.asarray(p.jump_rate)
    abs_j = sd * np.sqrt(2 / np.pi) * np.exp(-(mu**2) / (2 * sd**2)) + mu * (1 - 2 * norm.cdf(-mu / sd))
    kick_mean = p.jump_var_kick * abs_j / sd
    kick_sq = p.jump_var_kick**2 * (mu**2 + sd**2) / sd**2
    level = 2.0 * np.asarray(p.regime_log_vol) + lam * kick_mean / p.kappa_fast
    return level, lam * kick_sq


def _variance_nodes(
    x_slow: np.ndarray, x_fast: np.ndarray, regime: np.ndarray, params: MarketParams,
    offset: float = 0.0, nodes: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Quadrature terms of E_t[average variance] over a 30-day window.

    Returns per-node contributions ``ev`` (n x nodes, already divided by
    ``nodes``), the factor loadings ``e_slow``/``e_fast`` of each node (for
    derivatives) and the jump-variance term.  The slow factor is a Gaussian OU;
    the fast factor tracks the regime level, so E[exp(x_fast)] combines its
    decaying gap to the current level with the exact regime transition
    probabilities P(s) = expm(Q s).
    """
    p = params
    theta0 = 2.0 * np.log(p.vol_level)
    level, kick_var_rate = effective_fast_level(p)
    s = offset + (np.arange(nodes) + 0.5) / nodes * (IV_HORIZON_DAYS / DAYS_PER_YEAR)
    a, ks = p.kappa_fast, p.kappa_slow
    e_fast, e_slow = np.exp(-a * s), np.exp(-ks * s)
    trans = np.stack([expm(generator_matrix(p) * si) for si in s])          # (nodes, 3, 3)
    level_mix = (trans @ np.exp(level))[:, regime].T                         # (n, nodes)
    var_fast = (p.xi_fast**2 + kick_var_rate[regime][:, None]) / (2 * a) * (1 - e_fast**2)[None, :]
    var_slow = (p.xi_slow**2 / (2 * ks) * (1 - e_slow**2))[None, :]
    log_ev = (theta0 + x_slow[:, None] * e_slow[None, :] + 0.5 * var_slow
              + (x_fast - level[regime])[:, None] * e_fast[None, :] + 0.5 * var_fast)
    ev = np.exp(log_ev) * level_mix / nodes
    jump_var = (np.asarray(p.jump_rate) * (np.asarray(p.jump_mean) ** 2 + p.jump_std**2))[regime]
    return ev, e_slow, e_fast, jump_var


def expected_average_variance(
    x_slow: np.ndarray, x_fast: np.ndarray, regime: np.ndarray, params: MarketParams, offset: float = 0.0,
) -> np.ndarray:
    """Physical-measure expectation of average (annualised) variance over the index window."""
    ev, _, _, jump_var = _variance_nodes(x_slow, x_fast, regime, params, offset)
    return ev.sum(axis=1) + jump_var


def expected_next_index(
    x_slow: np.ndarray, x_fast: np.ndarray, sentiment: np.ndarray, regime: np.ndarray, params: MarketParams,
) -> np.ndarray:
    """E_t[I_{t+1}]: the index expected one bar ahead, for the simulator's own index function.

    Exact over regime transitions (expm(Q dt)), exact over jump kicks
    (Gauss-Hermite in the jump size), second order in the Gaussian factor and
    sentiment shocks -- whose one-bar s.d. is ~0.02, so the expansion error is
    negligible.  This makes the perpetual's funding an unbiased transfer of the
    index's drift.
    """
    p = params
    phi_s, phi_f = np.exp(-p.kappa_slow * DT), np.exp(-p.kappa_fast * DT)
    sd_s = p.xi_slow * np.sqrt((1 - phi_s**2) / (2 * p.kappa_slow))
    sd_f = p.xi_fast * np.sqrt((1 - phi_f**2) / (2 * p.kappa_fast))
    phi_e = 0.5 ** (1.0 / (p.sentiment_halflife_days * BARS_PER_DAY))
    sd_e = p.sentiment_sd * np.sqrt(1 - phi_e**2)
    step = expm(generator_matrix(p) * DT)
    gh_x, gh_w = np.polynomial.hermite.hermgauss(9)
    gh_w = gh_w / np.sqrt(np.pi)
    size_cov = 1.0 - _ABS_MEAN**2

    xs_next = phi_s * x_slow
    out = np.zeros(len(x_slow))
    for j in range(3):
        reg_j = np.full(len(x_slow), j)
        xf_next = phi_f * x_fast + (1 - phi_f) * 2.0 * p.regime_log_vol[j]
        ev, e_s, e_f, jv = _variance_nodes(xs_next, xf_next, reg_j, p)
        a = ev.sum(axis=1) + jv
        root = np.sqrt(a)
        a_s, a_ss, a_f, a_ff = ev @ e_s, ev @ e_s**2, ev @ e_f, ev @ e_f**2
        c_f = a_f / (2 * root)
        c_ss = a_ss / (2 * root) - a_s**2 / (4 * a * root)
        c_ff = a_ff / (2 * root) - a_f**2 / (4 * a * root)
        cov_fe = sd_f * sd_e * (p.regime_theta[j] * p.sentiment_theta[j] + p.size_effect * p.sentiment_size * size_cov)
        diffuse = root + 0.5 * c_ss * sd_s**2 + 0.5 * c_ff * sd_f**2 + c_f * cov_fe
        jumps = p.jump_mean[j] + p.jump_std * np.sqrt(2.0) * gh_x
        kicks = p.jump_var_kick * np.abs(jumps) / p.jump_std
        kicked = sum(w * np.sqrt(ev @ np.exp(k * e_f) + jv) for w, k in zip(gh_w, kicks))
        p_jump = p.jump_rate[j] * DT
        out += step[regime, j] * ((1 - p_jump) * diffuse + p_jump * kicked)
    scale = 100.0 * np.sqrt(np.exp(p.iv_log_premium))
    return scale * out * np.exp(phi_e * sentiment + 0.5 * sd_e**2)


@dataclass
class SimulatedMarket:
    """Container returned by :func:`simulate_market`."""

    bars: pd.DataFrame
    params: MarketParams
    seed: int
    meta: dict = field(default_factory=dict)


def simulate_market(params: MarketParams | None = None, seed: int = 0) -> SimulatedMarket:
    """Simulate ``params.days`` of 15-minute bars.

    Columns: ``open high low close volume quote_volume`` (exchange-kline style),
    ``bviv`` (index), ``bviv_mark`` (perpetual mark), ``bviv_funding`` (vol points
    paid per long contract over the *next* bar), ``bviv_carry`` (its premium part,
    i.e. what longs pay beyond compensation for the index's expected drift),
    ``btc_funding`` (annualised perp funding rate), plus latent ``regime`` and
    ``true_var`` for diagnostics.
    """
    p = params or MarketParams()
    rng = np.random.default_rng(seed)
    n = p.days * BARS_PER_DAY
    index = pd.date_range(p.start, periods=n, freq=f"{MINUTES_PER_BAR}min", tz="UTC")

    regime = _simulate_regimes(n, p, rng)
    season = seasonal_factor(index, p)

    # --- variance factors -------------------------------------------------
    theta0 = 2.0 * np.log(p.vol_level)
    phi_s, phi_f = np.exp(-p.kappa_slow * DT), np.exp(-p.kappa_fast * DT)
    sd_s = p.xi_slow * np.sqrt((1 - phi_s**2) / (2 * p.kappa_slow))
    sd_f = p.xi_fast * np.sqrt((1 - phi_f**2) / (2 * p.kappa_fast))
    z, eps_s, eps_f = rng.standard_normal((3, n))         # z drives returns
    shock_f = egarch_shock(z, np.asarray(p.regime_theta)[regime], p.size_effect, eps_f)

    lam = np.asarray(p.jump_rate)[regime]
    jmu = np.asarray(p.jump_mean)[regime]
    jump_hit = rng.random(n) < lam * DT
    jump = np.where(jump_hit, jmu + p.jump_std * rng.standard_normal(n), 0.0)
    kick = p.jump_var_kick * np.abs(jump) / p.jump_std

    m_reg = 2.0 * np.asarray(p.regime_log_vol)[regime]
    x_slow = _ar1(sd_s * eps_s, phi_s)
    x_fast = _ar1((1 - phi_f) * m_reg + sd_f * shock_f + kick, phi_f, x0=m_reg[0])
    var = np.exp(theta0 + x_slow + x_fast)               # annualised variance after bar t
    var_prev = np.r_[var[0], var[:-1]]                   # predictable variance for bar t

    # --- returns ----------------------------------------------------------
    mu = np.asarray(p.regime_drift)[regime]
    jump_comp = lam * (np.exp(jmu + 0.5 * p.jump_std**2) - 1.0)
    sigma_bar = np.sqrt(var_prev * DT) * season
    diffusive = (mu - jump_comp - 0.5 * var_prev * season**2) * DT + sigma_bar * z
    log_ret = diffusive + jump
    log_close = np.log(p.s0) + np.cumsum(log_ret)
    close = np.exp(log_close)
    open_ = np.r_[p.s0, close[:-1]]

    # --- intra-bar path: Brownian bridge on 1-minute steps, jump at a random minute
    k = MINUTES_PER_BAR
    steps = rng.standard_normal((n, k)) * (sigma_bar / np.sqrt(k))[:, None]
    cum = np.cumsum(steps, axis=1)
    frac = (np.arange(1, k + 1) / k)[None, :]
    path = cum - frac * (cum[:, -1] - diffusive)[:, None]
    jump_minute = rng.integers(0, k, n)
    path += jump[:, None] * (np.arange(k)[None, :] >= jump_minute[:, None])
    high = open_ * np.exp(np.maximum(path.max(axis=1), 0.0))
    low = open_ * np.exp(np.minimum(path.min(axis=1), 0.0))

    # --- volume: seasonal, rises with variance and with the size of the move
    vol_profile = season**1.5 / np.mean(season**1.5)
    activity = (var_prev / np.exp(theta0)) ** (p.volume_vol_elasticity / 2)
    noise = np.exp(p.volume_noise * rng.standard_normal(n) - 0.5 * p.volume_noise**2)
    surprise = 1.0 + 0.5 * (np.abs(log_ret) / np.maximum(sigma_bar, 1e-12) - 0.8)
    volume = p.volume_btc * vol_profile * activity * noise * np.clip(surprise, 0.3, None)
    minute_w = rng.gamma(2.0, 1.0, (n, k))
    minute_w /= minute_w.sum(axis=1, keepdims=True)
    mid_path = 0.5 * (np.c_[np.zeros(n), path[:, :-1]] + path)
    bar_vwap = open_ * (minute_w * np.exp(mid_path)).sum(axis=1)
    quote_volume = volume * bar_vwap

    # --- implied volatility index -----------------------------------------
    phi_e = 0.5 ** (1.0 / (p.sentiment_halflife_days * BARS_PER_DAY))
    eta = egarch_shock(z, np.asarray(p.sentiment_theta)[regime], p.sentiment_size, rng.standard_normal(n))
    sentiment = _ar1(p.sentiment_sd * np.sqrt(1 - phi_e**2) * eta, phi_e)
    iv_scale = 100.0 * np.sqrt(np.exp(p.iv_log_premium))
    iv_core = iv_scale * np.sqrt(expected_average_variance(x_slow, x_fast, regime, p))
    bviv = iv_core * np.exp(sentiment) + p.iv_noise * rng.standard_normal(n)

    # --- BVIV perpetual ---------------------------------------------------
    # A perpetual with funding intensity k trades at E_t[ int k e^{-ks} I_{t+s} ds ]
    # (He et al. 2022).  We use the fast-funding limit: the mark equals the index
    # and funding transfers the index's expected one-bar drift, plus a
    # hedging-demand premium, so long holders pay ``hedge_carry`` vol points a
    # year in every state of the world -- no free lunch from mean reversion.
    mark = bviv
    drift = expected_next_index(x_slow, x_fast, sentiment, regime, p) - bviv
    bviv_funding = drift + p.hedge_carry * DT

    # --- BTC perpetual funding -------------------------------------------
    phi_b = 0.5 ** (1.0 / (p.btc_funding_halflife_days * BARS_PER_DAY))
    f_noise = _ar1(p.btc_funding_sd * np.sqrt(1 - phi_b**2) * rng.standard_normal(n), phi_b)
    btc_funding = p.btc_funding_base + np.asarray(p.btc_funding_regime)[regime] + f_noise

    bars = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": quote_volume,
            "bviv": bviv,
            "bviv_mark": mark,
            "bviv_funding": bviv_funding,
            "bviv_carry": np.full(n, p.hedge_carry * DT),
            "btc_funding": btc_funding,
            "regime": regime,
            "true_var": var,
        },
        index=index,
    )
    meta = {"jump_count": int(jump_hit.sum()), "latent": {"x_slow": x_slow, "x_fast": x_fast, "sentiment": sentiment}}
    return SimulatedMarket(bars=bars, params=p, seed=seed, meta=meta)
