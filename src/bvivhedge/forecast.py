"""Daily BTC volatility forecasts and their out-of-sample evaluation.

Every forecast made at the close of day *t* targets the average daily realised
variance over days t+1..t+h and uses only information available at that close.
Regression models are re-estimated on an expanding window whose targets are
already fully observed (no overlap leakage).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .constants import DAYS_PER_YEAR

MODELS = ("RW", "EWMA", "GARCH", "HAR", "IV-raw", "IV", "HAR-IV")


# --------------------------------------------------------------------------- targets & features
def forward_mean(x: pd.Series, h: int) -> pd.Series:
    """y_t = mean(x_{t+1}, ..., x_{t+h})."""
    return x[::-1].rolling(h).mean()[::-1].shift(-1)


def iv_to_daily_variance(iv: pd.Series) -> pd.Series:
    """Implied vol in vol points -> implied daily variance."""
    return (iv / 100.0) ** 2 / DAYS_PER_YEAR


def har_features(rv: pd.Series, iv_var: pd.Series | None = None) -> pd.DataFrame:
    """HAR regressors (Corsi 2009) on a 7-day crypto week; optional implied variance."""
    x = pd.DataFrame({"const": 1.0, "rv_d": rv, "rv_w": rv.rolling(7).mean(), "rv_m": rv.rolling(30).mean()})
    if iv_var is not None:
        x["iv"] = iv_var.reindex(rv.index)
    return x


# --------------------------------------------------------------------------- estimators
def expanding_ols_forecast(x: pd.DataFrame, y: pd.Series, h: int, min_obs: int = 180) -> pd.Series:
    """Out-of-sample OLS forecast at each t using rows whose h-day target ended by t.

    Implemented with cumulative cross-products, so it is O(n k^2) rather than n regressions.
    """
    ok = x.notna().all(axis=1) & y.notna()
    xv = x.to_numpy(float)
    yv = y.to_numpy(float)
    n, k = xv.shape
    xz = np.where(ok.to_numpy()[:, None], xv, 0.0)
    yz = np.where(ok.to_numpy(), yv, 0.0)
    xtx = np.cumsum(xz[:, :, None] * xz[:, None, :], axis=0)
    xty = np.cumsum(xz * yz[:, None], axis=0)
    cnt = np.cumsum(ok.to_numpy())
    out = np.full(n, np.nan)
    ridge = 1e-12 * np.eye(k)
    for t in range(n):
        s = t - h  # last usable training row: its target window ends at t
        if s < 0 or cnt[s] < min_obs or not np.isfinite(xv[t]).all():
            continue
        beta = np.linalg.solve(xtx[s] + ridge * np.trace(xtx[s]), xty[s])
        out[t] = xv[t] @ beta
    return pd.Series(out, index=x.index)


def ewma_variance(ret: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics variance forecast for day t+1 from squared daily returns up to t."""
    r2 = ret**2
    out = r2.ewm(alpha=1 - lam, adjust=False).mean()
    return out


def garch_forecast(ret: pd.Series, h: int, min_obs: int = 365, refit_every: int = 30) -> pd.Series:
    """GARCH(1,1) with Student-t innovations; parameters re-estimated every ``refit_every`` days.

    Between re-estimations the conditional variance is filtered forward with
    fixed parameters, so each forecast uses returns up to its own date only.
    """
    from arch import arch_model

    scaled = 100.0 * ret.dropna()
    out = pd.Series(np.nan, index=ret.index)
    model = arch_model(scaled, mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False)
    starts = range(min_obs, len(scaled), refit_every)
    for start in starts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit(last_obs=start, disp="off", show_warning=False)
            end = min(start + refit_every, len(scaled))
            fc = res.forecast(horizon=h, start=start - 1, reindex=False)
        var = fc.variance.iloc[: end - start + 1].mean(axis=1) / 1e4
        out.loc[var.index] = var.to_numpy()
    return out


# --------------------------------------------------------------------------- orchestration
def make_forecasts(daily: pd.DataFrame, iv_close: pd.Series, h: int, min_obs: int = 180, garch: bool = True) -> pd.DataFrame:
    """All model forecasts of the h-day average daily variance, aligned on forecast date.

    ``garch=False`` skips the (comparatively slow) GARCH re-estimations.
    """
    rv, ret = daily["rv"], daily["ret"]
    iv_var = iv_to_daily_variance(iv_close.reindex(daily.index))
    y = forward_mean(rv, h)
    har_x = har_features(rv)
    hariv_x = har_features(rv, iv_var)
    out = pd.DataFrame(index=daily.index)
    out["RW"] = rv.rolling(h).mean()
    out["EWMA"] = ewma_variance(ret)
    if garch:
        out["GARCH"] = garch_forecast(ret, h, min_obs=max(min_obs, 365))
    out["HAR"] = expanding_ols_forecast(har_x, y, h, min_obs)
    out["IV-raw"] = iv_var
    out["IV"] = expanding_ols_forecast(pd.DataFrame({"const": 1.0, "iv": iv_var}), y, h, min_obs)
    out["HAR-IV"] = expanding_ols_forecast(hariv_x, y, h, min_obs)
    floor = 0.05 * rv.rolling(30, min_periods=1).mean()  # keep OLS forecasts strictly positive
    out = out.clip(lower=floor, axis=0)
    out["target"] = y
    return out


# --------------------------------------------------------------------------- evaluation
def qlike(realised: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """Patton (2011) robust loss: r/f - ln(r/f) - 1."""
    ratio = realised / forecast
    return ratio - np.log(ratio) - 1.0


def newey_west_var(d: np.ndarray, lags: int) -> float:
    d = d - d.mean()
    n = len(d)
    v = d @ d / n
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1)
        v += 2 * w * (d[l:] @ d[:-l]) / n
    return v


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int) -> float:
    """DM statistic for E[loss_a - loss_b] = 0 with a Newey-West long-run variance."""
    d = loss_a - loss_b
    n = len(d)
    lags = max(h, int(np.floor(4 * (n / 100) ** (2 / 9))))
    return float(d.mean() / np.sqrt(newey_west_var(d, lags) / n))


def evaluate(fc: pd.DataFrame, models=MODELS, benchmark: str = "HAR", h: int = 1, start=None) -> pd.DataFrame:
    """QLIKE, MSE (x1e8), Mincer-Zarnowitz R^2 and DM vs benchmark on the common sample."""
    cols = [m for m in models if m in fc]
    sample = fc[cols + ["target"]].dropna()
    if start is not None:
        sample = sample.loc[start:]
    y = sample["target"].to_numpy()
    base = qlike(y, sample[benchmark].to_numpy())
    rows = {}
    for m in cols:
        f = sample[m].to_numpy()
        lq = qlike(y, f)
        b = np.polyfit(f, y, 1)
        r2 = np.corrcoef(f, y)[0, 1] ** 2
        rows[m] = {
            "qlike": lq.mean(),
            "mse": np.mean((y - f) ** 2) * 1e8,
            "mz_alpha": b[1] * 1e4,
            "mz_beta": b[0],
            "mz_r2": r2,
            "dm_vs_" + benchmark: np.nan if m == benchmark else diebold_mariano(lq, base, h),
        }
    out = pd.DataFrame(rows).T
    out.attrs["n"] = len(sample)
    return out
