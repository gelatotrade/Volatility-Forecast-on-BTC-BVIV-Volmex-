import numpy as np
import pandas as pd

from bvivhedge.forecast import (
    diebold_mariano, expanding_ols_forecast, forward_mean, har_features, make_forecasts, qlike,
)
from bvivhedge.realized import daily_realized


def _har_process(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    rv = np.full(n, 1e-3)
    for t in range(30, n):
        mean = 1e-4 + 0.35 * rv[t - 1] + 0.3 * rv[t - 7 : t].mean() + 0.2 * rv[t - 30 : t].mean()
        rv[t] = mean * rng.lognormal(-0.125, 0.5)
    return pd.Series(rv, index=pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC"))


def test_expanding_ols_has_no_lookahead():
    rv = _har_process()
    h = 7
    x, y = har_features(rv), forward_mean(rv, h)
    full = expanding_ols_forecast(x, y, h, min_obs=200)
    cut = 900
    part = expanding_ols_forecast(x.iloc[:cut], forward_mean(rv.iloc[:cut], h), h, min_obs=200)
    assert np.allclose(full.iloc[:cut], part, equal_nan=True)


def test_log_forecasts_are_positive_and_causal():
    rv = _har_process()
    h = 7
    x, y = har_features(rv), forward_mean(rv, h)
    full = expanding_ols_forecast(x, y, h, min_obs=200, log=True)
    part = expanding_ols_forecast(x.iloc[:900], forward_mean(rv.iloc[:900], h), h, min_obs=200, log=True)
    assert (full.dropna() > 0).all()
    assert np.allclose(full.iloc[:900], part, equal_nan=True)


def test_har_recovers_persistence():
    rv = _har_process(4000)
    x, y = har_features(rv), forward_mean(rv, 1)
    ok = x.notna().all(axis=1) & y.notna()
    beta = np.linalg.lstsq(x[ok].to_numpy(), y[ok].to_numpy(), rcond=None)[0]
    assert abs(beta[1:].sum() - 0.85) < 0.1


def test_qlike_and_dm():
    r = np.array([1.0, 2.0, 3.0])
    assert np.allclose(qlike(r, r), 0.0)
    assert (qlike(r, 0.5 * r) > qlike(r, 2.0 * r)).all()   # under-prediction is penalised more
    rng = np.random.default_rng(0)
    good, bad = rng.random(500), rng.random(500) + 0.2
    assert diebold_mariano(good, bad, h=1) < -1.96


def test_make_forecasts_alignment(bars):
    daily = daily_realized(bars)
    fc = make_forecasts(daily, bars["bviv"].resample("1D").last(), h=7, min_obs=30, garch=False)
    assert {"HAR", "HAR-IV", "IV", "IV-raw", "EWMA", "RW"} <= set(fc.columns)
    # the target at t is the mean of the next 7 days' realised variance
    t = 50
    assert np.isclose(fc["target"].iloc[t], daily["rv"].iloc[t + 1 : t + 8].mean())
