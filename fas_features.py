"""
fas_features.py - Fractal-Adaptive Sampling (FAS) Feature Generator for XAUUSD

Adapter for new_engine/01_sampling/fractal_adaptive_sampler.py.
Computes:
1. Rolling log-returns and volatility (sigma_t).
2. Reference volatility (sigma_ref).
3. Energy scaling factor V_t = sigma_t / sigma_ref.
4. Local Hurst exponent H_t via K-over-N Ratio of Variations:
   H_t = (1 / ln(2)) * ln(m2 / m1)
5. Fractal penalty F_t = exp(lambda * (0.5 - H_t)).
6. Dynamic threshold theta_t = theta_base * V_t * F_t.
7. Directional Change (DC) detection.

All metrics are shifted by 1 bar to strictly enforce causality (zero lookahead).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add new_engine/01_sampling to sys.path
SAMPLING_DIR = Path(__file__).parent / "new_engine" / "01_sampling"
if str(SAMPLING_DIR) not in sys.path:
    sys.path.insert(0, str(SAMPLING_DIR))

from fractal_adaptive_sampler import FractalAdaptiveSampler, GapConfig


def compute_fas_features(
    df_bars: pd.DataFrame,
    theta_base: float = 0.002,
    window: int = 30,
    sigma_ref_window: int = 288,
    lambda_aggression: float = 2.0
) -> pd.DataFrame:
    """
    Computes causal FAS features for XAUUSD price series.
    
    Parameters
    ----------
    df_bars : pd.DataFrame
        DataFrame with DatetimeIndex and 'Close' column.
    theta_base : float
        Base directional change threshold (e.g. 0.002 = 20 bps = $4-5 on Gold).
    window : int
        Lookback window for local estimators (30 bars).
    sigma_ref_window : int
        Reference volatility lookback window (288 bars = 24 hours on 5m).
    lambda_aggression : float
        Sensitivity to Hurst exponent deviations from Brownian (0.5).
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['hurst', 'energy_v', 'fractal_f', 'theta_dynamic', 'sigma_local']
    """
    prices = df_bars["Close"].copy()
    
    # Gap detection
    time_diffs = pd.Series(df_bars.index).diff().dt.total_seconds() / 60.0
    gap_mask = (time_diffs > 120.0).values
    
    sampler = FractalAdaptiveSampler(
        theta_base=theta_base,
        window=window,
        sigma_ref_window=sigma_ref_window,
        lambda_aggression=lambda_aggression,
        gap_config=GapConfig(enabled=True, min_gap_minutes=120, mode="exclude")
    )
    
    metrics = sampler._compute_rolling_metrics(prices, gap_mask)
    
    out_df = pd.DataFrame({
        "hurst": metrics["H_t"],
        "energy_v": metrics["V_t"],
        "fractal_f": metrics["F_t"],
        "theta_dynamic": metrics["theta_t"],
        "sigma_local": metrics["sigma_t"]
    }, index=df_bars.index)
    
    return out_df
