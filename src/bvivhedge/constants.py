"""Calendar constants for a 24/7 market sampled on 15-minute bars."""

MINUTES_PER_BAR = 15
BARS_PER_HOUR = 60 // MINUTES_PER_BAR
BARS_PER_DAY = 24 * BARS_PER_HOUR
DAYS_PER_YEAR = 365
BARS_PER_YEAR = DAYS_PER_YEAR * BARS_PER_DAY
DT = 1.0 / BARS_PER_YEAR  # year fraction of one bar

#: Horizon of the BVIV index (30-day constant-maturity implied volatility).
IV_HORIZON_DAYS = 30
