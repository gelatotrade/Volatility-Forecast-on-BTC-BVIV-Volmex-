import numpy as np
import pandas as pd

from bvivhedge.constants import BARS_PER_DAY
from bvivhedge.vwap import gate_state, intraday_vol_forecast, rolling_vwap, session_vwap, vwap_zscore


def test_rolling_vwap_matches_definition(bars):
    v = rolling_vwap(bars, window=8)
    t = 500
    window = bars.iloc[t - 7 : t + 1]
    assert np.isclose(v.iloc[t], window["quote_volume"].sum() / window["volume"].sum())


def test_session_vwap_resets_at_utc_midnight(bars):
    v = session_vwap(bars)
    first_bars = bars.index.hour.to_numpy() == 0
    first_bars &= bars.index.minute.to_numpy() == 0
    bar_vwap = bars["quote_volume"] / bars["volume"]
    assert np.allclose(v[first_bars], bar_vwap[first_bars])


def test_signals_use_no_future_information(bars):
    """Truncating the sample must not change any value already computed -- also inside the first day."""
    short = intraday_vol_forecast(bars.iloc[:20])
    assert np.allclose(intraday_vol_forecast(bars)["sigma_day"].iloc[:20], short["sigma_day"], equal_nan=True)
    cut = 40 * BARS_PER_DAY
    full = intraday_vol_forecast(bars)
    part = intraday_vol_forecast(bars.iloc[:cut])
    assert np.allclose(full["sigma_day"].iloc[:cut], part["sigma_day"], equal_nan=True)
    z_full = vwap_zscore(bars, rolling_vwap(bars), full["sigma_day"])
    z_part = vwap_zscore(bars.iloc[:cut], rolling_vwap(bars.iloc[:cut]), part["sigma_day"])
    assert np.allclose(z_full.iloc[:cut], z_part, equal_nan=True)


def test_gate_hysteresis():
    z = np.array([0.0, -1.2, -0.5, -0.2, 0.3, 0.2, -0.8, 0.5])
    s = gate_state(z, enter=-1.0, exit=0.0, min_hold=2)
    # on at the breakdown, held through the dead band, off only once above VWAP after min_hold
    assert s.tolist() == [0, 1, 1, 1, 0, 0, 0, 0]


def test_zscore_is_scale_free(bars):
    sigma = pd.Series(0.03, index=bars.index)
    z1 = vwap_zscore(bars, rolling_vwap(bars), sigma)
    z2 = vwap_zscore(bars, rolling_vwap(bars), 2 * sigma)
    assert np.allclose(z1, 2 * z2, equal_nan=True)
