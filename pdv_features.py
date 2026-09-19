"""
pdv_features.py - Target Variable & TSPL Feature Construction for PDV Model

Implements:
1. Forward realized volatility target: RV_fwd_t = sqrt( sum_{j=1}^h GK_bar_{t+j}^2 )
2. Trailing realized volatility (for Naive baseline & AR(1))
3. Normalized Time-Shifted Power Law (TSPL) kernel:
   k(i; alpha, delta) = (i + delta)^(-alpha) / sum_{j=1}^N (j + delta)^(-alpha)
4. Path-dependent features R1_t and R2_t using strictly causal filtering (lags i=1..N).
"""

import numpy as np
import pandas as pd
import scipy.signal


def compute_tspl_weights(alpha: float, delta: float, N: int) -> np.ndarray:
    """
    Computes normalized TSPL kernel weights for lags i = 1, ..., N:
    k(i; alpha, delta) = (i + delta)^(-alpha) / sum_{j=1}^N (j + delta)^(-alpha)
    
    Guarantees sum(weights) == 1.0.
    """
    i_arr = np.arange(1, N + 1, dtype=np.float64)
    weights = (i_arr + delta) ** (-alpha)
    sum_w = np.sum(weights)
    if sum_w <= 0.0 or not np.isfinite(sum_w):
        # Fallback to uniform if degenerate
        return np.full(N, 1.0 / N, dtype=np.float64)
    return weights / sum_w


def compute_r1_r2(
    returns: np.ndarray,
    alpha1: float,
    delta1: float,
    alpha2: float,
    delta2: float,
    N: int = 500
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes R1_t = sum_{i=1}^N k1(i) * r_{t-i}
    and R2_t = sum_{i=1}^N k2(i) * r_{t-i}^2.
    
    Uses strictly causal filtering with zero weight at lag 0 (b[0] = 0).
    Bar t is NEVER included in R1_t or R2_t.
    """
    k1 = compute_tspl_weights(alpha1, delta1, N)
    k2 = compute_tspl_weights(alpha2, delta2, N)
    
    # Filter vector with b[0] = 0 ensures strictly lags 1..N
    b1 = np.concatenate(([0.0], k1))
    b2 = np.concatenate(([0.0], k2))
    
    r1 = scipy.signal.lfilter(b1, [1.0], returns)
    r2 = scipy.signal.lfilter(b2, [1.0], returns ** 2)
    
    # Clip r2 to non-negative due to float precision
    r2 = np.clip(r2, 0.0, None)
    return r1, r2


def compute_forward_rv(gk_var: pd.Series, h: int = 24) -> pd.Series:
    """
    Computes forward realized volatility over next h bars:
    RV_fwd_t = sqrt( sum_{j=1}^h GK_bar_{t+j}^2 )
    
    Note: At index t, this strictly uses GK_var from t+1 to t+h.
    The last h bars in the series will be NaN.
    """
    # Reverse rolling sum:
    # At position t, we want sum(gk_var[t+1 : t+h+1])
    rev_roll = gk_var.iloc[::-1].rolling(h).sum().iloc[::-1].shift(-1)
    return np.sqrt(np.clip(rev_roll, 0.0, None))


def compute_trailing_rv(gk_var: pd.Series, h: int = 24) -> pd.Series:
    """
    Computes backward-looking trailing realized volatility over current and past h-1 bars:
    RV_trailing_t = sqrt( sum_{j=0}^{h-1} GK_bar_{t-j}^2 )
    
    Strictly uses bars up to and including t (no lookahead).
    Used as the Naive Persistence baseline and input to AR(1).
    """
    roll_sum = gk_var.rolling(h).sum()
    return np.sqrt(np.clip(roll_sum, 0.0, None))
