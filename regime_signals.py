"""
regime_signals.py - Regime Signal & Strategy Feature Generation for PDV Model

Implements:
1. Regime Gate:
   - Rolling 60-day median or bottom tercile of sigma_hat
   - "ranging" if sigma_hat <= threshold, else "expansion"
2. Mean-Reversion Signals:
   - Z-score: z = (Close - MA) / Std (window 20..50 bars)
   - Entry triggers when regime == "ranging"
3. Adaptive PDV Stop & Target Sizing:
   - Delta_P = sigma_hat * Close
   - Long: stop = Close - k1 * Delta_P, target = Close + k2 * Delta_P
   - Short: stop = Close + k1 * Delta_P, target = Close - k2 * Delta_P
   - Time stop: current_bar + h
4. Directional Tilt:
   - Evaluates beta1_sign * R1_t to skip entries against vol-expanding momentum.
"""

import numpy as np
import pandas as pd


def compute_regime_gate(
    sigma_hat: pd.Series,
    window_days: int = 60,
    method: str = "median"
) -> tuple[pd.Series, pd.Series]:
    """
    Computes forward-looking regime gate from sigma_hat.
    
    BUG 9 FIX: Uses calendar-based rolling window ('60D') instead of
    bar-count-based rolling, so the window correctly spans 60 calendar
    days regardless of bars_per_day assumptions.
    
    threshold:
    - 'median': 50th percentile of rolling window
    - 'tercile': 33.33rd percentile (bottom tercile)
    
    Returns:
    - regime: pd.Series of strings ('ranging' or 'expansion')
    - threshold_series: pd.Series of the rolling threshold values
    """
    roll_window = f"{window_days}D"
    min_p = 500  # Require ~2 trading days of data minimum
    
    if method == "tercile":
        threshold = sigma_hat.rolling(roll_window, min_periods=min_p).quantile(0.3333)
    else:
        threshold = sigma_hat.rolling(roll_window, min_periods=min_p).median()
        
    is_ranging = sigma_hat <= threshold
    regime = pd.Series(np.where(is_ranging, "ranging", "expansion"), index=sigma_hat.index)
    return regime, threshold


def compute_zscore(price: pd.Series, window: int = 30) -> pd.Series:
    """
    Computes rolling z-score: (price - MA) / Std
    """
    ma = price.rolling(window).mean()
    std = price.rolling(window).std(ddof=0)
    std = std.replace(0.0, np.nan)
    return (price - ma) / std


def generate_feature_dataframe(
    df_bars: pd.DataFrame,
    sigma_hat: pd.Series,
    beta1_sign: pd.Series,
    r1: pd.Series = None,
    h: int = 24,
    z_window: int = 30,
    z_threshold: float = 2.0,
    k1_stop: float = 1.0,
    k2_target: float = 1.0,
    regime_method: str = "median",
    apply_beta1_tilt: bool = False
) -> pd.DataFrame:
    """
    Generates standardized strategy interface DataFrame matching Requirement 8:
    Columns:
    - sigma_hat: float, forecasted realized vol
    - regime: str ('ranging' or 'expansion')
    - beta1_sign: int (+1 or -1)
    - z_score: float
    - stop_price: float (adaptive stop based on sigma_hat)
    - target_price: float (adaptive target based on sigma_hat)
    - time_stop_bar: int or timestamp offset (h bars)
    - signal: int (+1 long, -1 short, 0 flat)
    """
    df_out = pd.DataFrame(index=df_bars.index)
    
    df_out["Close"] = df_bars["Close"]
    df_out["sigma_hat"] = sigma_hat
    df_out["beta1_sign"] = beta1_sign
    
    # 1. Regime gate
    regime, threshold = compute_regime_gate(sigma_hat, method=regime_method)
    df_out["regime"] = regime
    df_out["regime_threshold"] = threshold
    
    # 2. Z-score
    z_score = compute_zscore(df_bars["Close"], window=z_window)
    df_out["z_score"] = z_score
    
    # 3. Price volatility scale: Delta_P = sigma_hat * Close
    # Note: sigma_hat is already an h-bar cumulative volatility forecast, so no sqrt(h)
    sigma_hat_clean = np.maximum(sigma_hat, 5e-4)
    delta_p = sigma_hat_clean * df_bars["Close"]
    # Floor delta_p so stop distance cannot sit inside typical spread (~0.20 USD)
    min_delta_p = 0.40 / max(k1_stop, 1e-6)
    delta_p = np.maximum(delta_p, min_delta_p)
    df_out["delta_p"] = delta_p
    
    # 4. Signal generation
    # Long entry: z <= -z_threshold and regime == 'ranging'
    # Short entry: z >= +z_threshold and regime == 'ranging'
    is_ranging = (df_out["regime"] == "ranging")
    
    long_condition = is_ranging & (z_score <= -z_threshold)
    short_condition = is_ranging & (z_score >= z_threshold)
    
    # Optional directional tilt if beta1 is stable
    # e.g., if beta1 > 0 and r1 > 0, recent up-move forecasts vol expansion -> skip long
    if apply_beta1_tilt and r1 is not None:
        tilt = beta1_sign * r1
        # Skip long when recent move forecasts high vol expansion
        long_condition = long_condition & (tilt <= 0)
        short_condition = short_condition & (tilt >= 0)
        
    signal = np.zeros(len(df_out), dtype=int)
    signal[long_condition] = 1
    signal[short_condition] = -1
    df_out["signal"] = signal
    
    # 5. Stop and Target prices
    stop_price = np.full(len(df_out), np.nan)
    target_price = np.full(len(df_out), np.nan)
    
    # For Longs:
    stop_price[long_condition] = df_bars["Close"][long_condition] - k1_stop * delta_p[long_condition]
    target_price[long_condition] = df_bars["Close"][long_condition] + k2_target * delta_p[long_condition]
    
    # For Shorts:
    stop_price[short_condition] = df_bars["Close"][short_condition] + k1_stop * delta_p[short_condition]
    target_price[short_condition] = df_bars["Close"][short_condition] - k2_target * delta_p[short_condition]
    
    df_out["stop_price"] = stop_price
    df_out["target_price"] = target_price
    
    # Time stop in terms of bar index offset
    bar_indices = np.arange(len(df_out))
    time_stop_bar = np.full(len(df_out), -1, dtype=int)
    time_stop_bar[signal != 0] = bar_indices[signal != 0] + h
    df_out["time_stop_bar"] = time_stop_bar
    
    return df_out
