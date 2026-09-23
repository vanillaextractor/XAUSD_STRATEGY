"""
constants.py - Shared configuration constants for the XAUUSD PDV strategy pipeline.

Centralizes magic numbers that previously appeared in 8+ files, ensuring
a single source of truth for parameters that must stay synchronized.
"""

# --------------------------------------------------------------------------
# Sigma-hat floor
# --------------------------------------------------------------------------
# The PDV walk-forward linear regression can occasionally produce forecasts
# near zero or negative.  This floor prevents delta_p (and therefore stops
# and targets) from collapsing inside the bid-ask spread.
#
# Value chosen so that k1_stop * sigma_hat_floor * price ≈ 2× typical spread:
#   1.5 * 5e-4 * 2300 ≈ 1.73 USD  >>  0.20 USD flat spread
#
# As of 2019-2025 XAUUSD data, the actual minimum OOS sigma_hat is 0.001084
# (~10.8 bps), so this floor never binds under normal conditions.
SIGMA_HAT_FLOOR = 5e-4
