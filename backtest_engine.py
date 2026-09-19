"""
backtest_engine.py - Intraday Execution Simulator & EDGE Spread Cost Engine

Implements:
1. EDGE time-varying bid-ask spread estimation using `bidask.edge_rolling`.
2. High-fidelity backtest simulation:
   - 5-minute signals executed against 1-minute resolution bars.
   - Entry filled at next 1-minute open + half effective spread.
   - Dynamic stop loss (k1 * sigma_hat * Close * sqrt(h)).
   - Dynamic take profit (k2 * sigma_hat * Close * sqrt(h)).
   - Zero-crossing exit (z-score crosses 0).
   - Time stop (h bars horizon).
3. Performance metrics: Sharpe, Drawdown, Win Rate, Profit Factor, Trade Count.
4. Comparison between time-varying EDGE spread vs flat spread assumption.
"""

import numpy as np
import pandas as pd
import bidask


def compute_edge_spread(
    df_m5: pd.DataFrame,
    rolling_window: int = 288,
    min_spread_bps: float = 0.5,
    max_spread_bps: float = 10.0
) -> pd.Series:
    """
    Computes time-varying effective spread using the EDGE estimator (Ardia et al., 2024).
    rolling_window=288 corresponds to 24 hours on 5-minute bars.
    Returns effective spread in relative decimal (e.g., 0.0001 = 1 bps).
    """
    print("Estimating time-varying bid-ask spread via EDGE estimator...")
    # Prepare OHLC DataFrame for bidask
    ohlc = df_m5[["Open", "High", "Low", "Close"]].copy()
    
    # edge_rolling returns relative spread fraction
    edge_series = bidask.edge_rolling(ohlc, window=rolling_window)
    
    # Fill initial NaNs with expanding median or global median
    median_val = edge_series.dropna().median()
    if pd.isna(median_val) or median_val <= 0:
        median_val = 1.5e-5  # default fallback ~1.5 bps
        
    edge_clean = edge_series.fillna(median_val)
    # Clip spread to reasonable bounds for gold (e.g., 0.5 bps to 10 bps)
    lower_bound = min_spread_bps * 1e-4
    upper_bound = max_spread_bps * 1e-4
    edge_clean = edge_clean.clip(lower=lower_bound, upper=upper_bound)
    
    print(f"EDGE Spread Mean: {edge_clean.mean()*10000:.2f} bps "
          f"({edge_clean.mean() * df_m5['Close'].mean():.3f} USD/oz), "
          f"Median: {edge_clean.median()*10000:.2f} bps")
    return edge_clean


def run_intraday_backtest(
    df_features: pd.DataFrame,
    df_m1: pd.DataFrame,
    edge_spread: pd.Series,
    flat_spread_usd: float = 0.20,  # 20 cents flat spread benchmark
    use_edge_cost: bool = True
) -> dict:
    """
    Simulates strategy execution using 5-minute signals and 1-minute execution bars.
    
    Returns:
    - trades_df: Detailed list of all completed trades
    - equity_curve: Daily/bar-level cumulative PnL
    - metrics: Dictionary of performance statistics
    """
    print(f"Running Intraday Backtest ({'EDGE Time-Varying Spread' if use_edge_cost else f'Flat Spread {flat_spread_usd}$'})...")
    
    # Ensure index is strictly unique
    df_features = df_features[~df_features.index.duplicated(keep="first")].copy()
    
    signals = df_features[df_features["signal"] != 0].copy()
    if len(signals) == 0:
        print("No signals generated.")
        return {"trades": pd.DataFrame(), "metrics": {}}

    trades = []
    
    # Reindex m1 datetime index for fast lookup
    m1_index = df_m1.index
    m1_opens = df_m1["Open"].values
    m1_highs = df_m1["High"].values
    m1_lows = df_m1["Low"].values
    m1_closes = df_m1["Close"].values
    
    # Map m1 timestamps to index
    m1_time_map = {t: idx for idx, t in enumerate(m1_index)}
    
    in_position = False
    pos_side = 0
    pos_entry_price = 0.0
    pos_stop_price = 0.0
    pos_target_price = 0.0
    pos_entry_time = None
    pos_time_stop_time = None
    pos_entry_m1_idx = 0
    pos_signal_z = 0.0
    
    signal_times = signals.index
    
    # Precompute z_score map for instant O(1) lookups
    z_map = df_features["z_score"].to_dict()
    
    for sig_t in signal_times:
        row = df_features.loc[sig_t]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        sig = int(row["signal"])
        
        # If already in a position, we only check if that position has exited before sig_t
        if in_position:
            continue
            
        # Entry occurs at next 1-minute bar
        entry_time_m1 = sig_t + pd.Timedelta(minutes=5)
        if entry_time_m1 not in m1_time_map:
            continue
            
        entry_idx = m1_time_map[entry_time_m1]
        raw_entry_open = m1_opens[entry_idx]
        
        # Determine spread cost at entry
        if use_edge_cost:
            spread_fraction = edge_spread.get(sig_t, 1.5e-5)
            spread_usd = spread_fraction * raw_entry_open
        else:
            spread_usd = flat_spread_usd
            
        half_spread = 0.5 * spread_usd
        
        if sig == 1:  # Long
            entry_price = raw_entry_open + half_spread
            stop_price = row["stop_price"]
            target_price = row["target_price"]
        else:  # Short
            entry_price = raw_entry_open - half_spread
            stop_price = row["stop_price"]
            target_price = row["target_price"]
            
        time_stop_bars = int(row["time_stop_bar"] - df_features.index.get_loc(sig_t))
        time_stop_time = sig_t + pd.Timedelta(minutes=time_stop_bars * 5)
        
        # Simulate forward in 1-minute resolution until exit
        max_m1_bars = int(time_stop_bars * 5) + 5
        curr_m1_idx = entry_idx
        exit_found = False
        
        while curr_m1_idx < len(df_m1) and (curr_m1_idx - entry_idx) <= max_m1_bars:
            m1_bar_time = m1_index[curr_m1_idx]
            bar_high = m1_highs[curr_m1_idx]
            bar_low = m1_lows[curr_m1_idx]
            bar_close = m1_closes[curr_m1_idx]
            
            # Spread at exit
            if use_edge_cost:
                exit_spread_usd = edge_spread.asof(m1_bar_time) * bar_close
                if pd.isna(exit_spread_usd):
                    exit_spread_usd = 1.5e-5 * bar_close
            else:
                exit_spread_usd = flat_spread_usd
            exit_half_spread = 0.5 * exit_spread_usd
            
            # 1. Stop Loss Check
            if sig == 1 and bar_low <= stop_price:
                exit_price = min(bar_low, stop_price) - exit_half_spread
                exit_reason = "stop_loss"
                exit_found = True
            elif sig == -1 and bar_high >= stop_price:
                exit_price = max(bar_high, stop_price) + exit_half_spread
                exit_reason = "stop_loss"
                exit_found = True
                
            # 2. Target Take Profit Check
            elif sig == 1 and bar_high >= target_price:
                exit_price = target_price - exit_half_spread
                exit_reason = "take_profit"
                exit_found = True
            elif sig == -1 and bar_low <= target_price:
                exit_price = target_price + exit_half_spread
                exit_reason = "take_profit"
                exit_found = True
                
            # 3. Time Stop Check
            elif m1_bar_time >= time_stop_time:
                exit_price = bar_close - (exit_half_spread if sig == 1 else -exit_half_spread)
                exit_reason = "time_stop"
                exit_found = True
                
            # 4. Zero Crossing Check (at 5-minute boundaries)
            elif curr_m1_idx > entry_idx and m1_bar_time in z_map:
                curr_z = z_map[m1_bar_time]
                if (sig == 1 and curr_z >= 0.0) or (sig == -1 and curr_z <= 0.0):
                    exit_price = bar_close - (exit_half_spread if sig == 1 else -exit_half_spread)
                    exit_reason = "z_zero_cross"
                    exit_found = True
                    
            if exit_found:
                pnl_usd = (exit_price - entry_price) if sig == 1 else (entry_price - exit_price)
                ret_pct = pnl_usd / entry_price
                duration_min = (m1_bar_time - entry_time_m1).total_seconds() / 60.0
                
                trades.append({
                    "signal_time": sig_t,
                    "entry_time": entry_time_m1,
                    "exit_time": m1_bar_time,
                    "side": "LONG" if sig == 1 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_usd": pnl_usd,
                    "return_pct": ret_pct,
                    "exit_reason": exit_reason,
                    "duration_min": duration_min,
                    "sigma_hat": row["sigma_hat"],
                    "z_score": row["z_score"],
                    "regime": row["regime"]
                })
                break
                
            curr_m1_idx += 1

    trades_df = pd.DataFrame(trades)
    
    if len(trades_df) == 0:
        return {"trades": pd.DataFrame(), "metrics": {}}

    # Compute aggregate metrics
    pnl = trades_df["pnl_usd"].values
    rets = trades_df["return_pct"].values
    win_trades = pnl > 0
    loss_trades = pnl < 0
    
    win_rate = np.mean(win_trades) if len(trades_df) > 0 else 0.0
    total_pnl = np.sum(pnl)
    profit_factor = np.sum(pnl[win_trades]) / np.abs(np.sum(pnl[loss_trades])) if np.sum(loss_trades) != 0 else np.nan
    
    # Cumulative equity & Drawdown
    cum_pnl = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum_pnl)
    drawdowns = cum_pnl - peak
    max_dd = np.min(drawdowns)
    
    # Annualized Sharpe (assuming ~252 trading days)
    # Group PnL by day for standard daily Sharpe
    trades_df["exit_date"] = pd.to_datetime(trades_df["exit_time"]).dt.date
    daily_pnl = trades_df.groupby("exit_date")["pnl_usd"].sum()
    daily_sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252) if daily_pnl.std() > 0 else 0.0

    metrics = {
        "total_trades": len(trades_df),
        "win_rate": win_rate,
        "total_pnl_usd": total_pnl,
        "profit_factor": profit_factor,
        "max_drawdown_usd": max_dd,
        "daily_sharpe": daily_sharpe,
        "mean_duration_min": trades_df["duration_min"].mean(),
        "exit_reasons": trades_df["exit_reason"].value_counts().to_dict()
    }
    
    return {
        "trades": trades_df,
        "metrics": metrics,
        "daily_pnl": daily_pnl
    }
