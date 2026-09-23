"""
backtest_engine.py - Intraday Execution Simulator & EDGE Spread Cost Engine

FIXED BUGS:
- BUG 1: Replaced z=0 hard exit with trailing z-score threshold (z=-0.5 for longs, z=+0.5 for shorts)
         so PDV-based targets actually have a chance to be reached.
- BUG 2: Added cooldown_bars parameter to suppress signal clustering.
- BUG 3: Fixed in_position state machine — properly set to True on entry, False on exit.
- BUG 4: Stop/target recomputed from actual fill price, not signal bar close.
- BUG 7: Stop fill uses stop_price (resting order model), not min(bar_low, stop_price).
- BUG 8: EDGE NaN fill uses expanding median (no future lookahead).
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

    BUG 8 FIX: NaN values filled with expanding median (no future lookahead).
    """
    print("Estimating time-varying bid-ask spread via EDGE estimator...")
    ohlc = df_m5[["Open", "High", "Low", "Close"]].copy()

    edge_series = bidask.edge_rolling(ohlc, window=rolling_window)

    # BUG 8 FIX: Use expanding median instead of global median for NaN fill
    expanding_median = edge_series.expanding().median()
    edge_clean = edge_series.fillna(expanding_median)
    # Any remaining NaNs at the very start get a conservative default
    edge_clean = edge_clean.fillna(2.0e-5)  # ~2 bps default

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
    flat_spread_usd: float = 0.20,
    use_edge_cost: bool = True,
    cooldown_bars: int = 24,
    z_exit_threshold: float = -0.5,
    k1_stop: float = 1.5,
    k2_target: float = 1.0,
    h: int = 24
) -> dict:
    """
    Simulates strategy execution using 5-minute signals and 1-minute execution bars.

    BUG FIXES APPLIED:
    - BUG 3: in_position state machine properly tracks open/close.
    - BUG 2: cooldown_bars suppresses signal clustering (no new entry within N bars of last entry).
    - BUG 1: z_exit_threshold replaces hard z=0 exit (default: -0.5 for longs, +0.5 for shorts).
    - BUG 4: stop/target recomputed from actual fill price.
    - BUG 7: stop fill uses stop_price (resting order), not bar extreme.
    """
    print(f"Running Intraday Backtest ({'EDGE' if use_edge_cost else f'Flat ${flat_spread_usd}'} spread, "
          f"cooldown={cooldown_bars} bars, z_exit={z_exit_threshold})...")

    # Ensure index is strictly unique
    df_features = df_features[~df_features.index.duplicated(keep="first")].copy()

    signals = df_features[df_features["signal"] != 0].copy()
    if len(signals) == 0:
        print("No signals generated.")
        return {"trades": pd.DataFrame(), "metrics": {}}

    trades = []

    # Precompute numpy arrays for M1 data (fast indexed access)
    m1_index = df_m1.index
    m1_opens = df_m1["Open"].values
    m1_highs = df_m1["High"].values
    m1_lows = df_m1["Low"].values
    m1_closes = df_m1["Close"].values

    # Map m1 timestamps to integer index for O(1) lookup
    m1_time_map = {t: idx for idx, t in enumerate(m1_index)}

    # Precompute z_score and delta_p maps for O(1) lookups
    z_map = df_features["z_score"].to_dict()
    delta_p_map = df_features["delta_p"].to_dict()

    # State tracking: enforce single active position and cooldown
    last_exit_time = None
    last_entry_bar_idx = -cooldown_bars - 1  # Allow first signal immediately

    # Build positional index map for cooldown checking
    feat_idx_map = {ts: i for i, ts in enumerate(df_features.index)}

    signal_times = signals.index

    for sig_t in signal_times:
        # Entry occurs at next 1-minute bar after the 5-minute signal bar closes
        entry_time_m1 = sig_t + pd.Timedelta(minutes=5)
        if entry_time_m1 not in m1_time_map:
            continue

        # Position tracking: Cannot open new position if previous trade has not exited
        in_position = (last_exit_time is not None and (sig_t < last_exit_time or entry_time_m1 <= last_exit_time))
        if in_position:
            continue

        # Cooldown: Skip if too close to last entry
        sig_bar_idx = feat_idx_map.get(sig_t, -1)
        if sig_bar_idx < 0:
            continue
        if (sig_bar_idx - last_entry_bar_idx) < cooldown_bars:
            continue

        row = df_features.loc[sig_t]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        sig = int(row["signal"])

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
        else:  # Short
            entry_price = raw_entry_open - half_spread

        # Compute stop/target from ACTUAL FILL PRICE
        # Note: sigma_hat is already an h-bar cumulative volatility forecast, so no sqrt(h)
        sigma_hat_val = max(float(row["sigma_hat"]), 5e-4)
        delta_p_entry = sigma_hat_val * entry_price

        # Enforce minimum stop distance: at least 2x the spread cost so stop never sits inside spread
        min_stop_distance = 2.0 * spread_usd
        if k1_stop * delta_p_entry < min_stop_distance:
            delta_p_entry = min_stop_distance / max(k1_stop, 1e-6)

        if sig == 1:
            stop_price = entry_price - k1_stop * delta_p_entry
            target_price = entry_price + k2_target * delta_p_entry
        else:
            stop_price = entry_price + k1_stop * delta_p_entry
            target_price = entry_price - k2_target * delta_p_entry

        # Time stop
        time_stop_time = sig_t + pd.Timedelta(minutes=h * 5)

        last_entry_bar_idx = sig_bar_idx

        # Simulate forward in 1-minute resolution until exit
        max_m1_bars = h * 5 + 10  # h 5-min bars = h*5 1-min bars + buffer
        curr_m1_idx = entry_idx + 1  # Start checking from the bar AFTER entry
        exit_found = False

        while curr_m1_idx < len(df_m1) and (curr_m1_idx - entry_idx) <= max_m1_bars:
            m1_bar_time = m1_index[curr_m1_idx]
            bar_high = m1_highs[curr_m1_idx]
            bar_low = m1_lows[curr_m1_idx]
            bar_close = m1_closes[curr_m1_idx]

            # Spread at exit
            if use_edge_cost:
                exit_spread_frac = edge_spread.asof(m1_bar_time)
                if pd.isna(exit_spread_frac):
                    exit_spread_frac = 1.5e-5
                exit_spread_usd = exit_spread_frac * bar_close
            else:
                exit_spread_usd = flat_spread_usd
            exit_half_spread = 0.5 * exit_spread_usd

            # --- Exit Priority Order ---
            # 1. Stop Loss Check
            if sig == 1 and bar_low <= stop_price:
                exit_price = stop_price - exit_half_spread
                exit_reason = "stop_loss"
                exit_found = True
            elif sig == -1 and bar_high >= stop_price:
                exit_price = stop_price + exit_half_spread
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
                if sig == 1:
                    exit_price = bar_close - exit_half_spread
                else:
                    exit_price = bar_close + exit_half_spread
                exit_reason = "time_stop"
                exit_found = True

            # 4. Trailing z-score exit (optional, can be disabled with None)
            elif z_exit_threshold is not None and m1_bar_time in z_map:
                curr_z = z_map[m1_bar_time]
                if not np.isnan(curr_z):
                    if sig == 1 and curr_z >= z_exit_threshold:
                        exit_price = bar_close - exit_half_spread
                        exit_reason = "z_revert"
                        exit_found = True
                    elif sig == -1 and curr_z <= -z_exit_threshold:
                        exit_price = bar_close + exit_half_spread
                        exit_reason = "z_revert"
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
                    "stop_price": stop_price,
                    "target_price": target_price,
                    "pnl_usd": pnl_usd,
                    "return_pct": ret_pct,
                    "exit_reason": exit_reason,
                    "duration_min": duration_min,
                    "sigma_hat": sigma_hat_val,
                    "z_score_entry": row["z_score"],
                    "regime": row["regime"],
                    "spread_usd": spread_usd
                })

                last_exit_time = m1_bar_time
                break

            curr_m1_idx += 1

        # If we exhausted bars without finding an exit (e.g. data ends), force close
        if not exit_found and curr_m1_idx > entry_idx:
            exit_bar_idx = min(curr_m1_idx - 1, len(df_m1) - 1)
            exit_m1_time = m1_index[exit_bar_idx]
            exit_close = m1_closes[exit_bar_idx]
            exit_half_spread = 0.5 * flat_spread_usd
            exit_price = (exit_close - exit_half_spread) if sig == 1 else (exit_close + exit_half_spread)
            pnl_usd = (exit_price - entry_price) if sig == 1 else (entry_price - exit_price)
            ret_pct = pnl_usd / entry_price
            duration_min = (exit_m1_time - entry_time_m1).total_seconds() / 60.0

            trades.append({
                "signal_time": sig_t,
                "entry_time": entry_time_m1,
                "exit_time": exit_m1_time,
                "side": "LONG" if sig == 1 else "SHORT",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "pnl_usd": pnl_usd,
                "return_pct": ret_pct,
                "exit_reason": "end_of_data",
                "duration_min": duration_min,
                "sigma_hat": sigma_hat_val,
                "z_score_entry": row["z_score"],
                "regime": row["regime"],
                "spread_usd": spread_usd
            })
            last_exit_time = exit_m1_time

    trades_df = pd.DataFrame(trades)

    if len(trades_df) == 0:
        return {"trades": pd.DataFrame(), "metrics": {}}

    # Compute aggregate metrics
    pnl = trades_df["pnl_usd"].values
    win_trades = pnl > 0
    loss_trades = pnl < 0

    win_rate = np.mean(win_trades)
    total_pnl = np.sum(pnl)
    gross_profit = np.sum(pnl[win_trades]) if win_trades.any() else 0.0
    gross_loss = np.abs(np.sum(pnl[loss_trades])) if loss_trades.any() else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

    # Cumulative equity & Drawdown
    cum_pnl = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum_pnl)
    drawdowns = cum_pnl - peak
    max_dd = np.min(drawdowns)

    # Annualized Sharpe from daily PnL
    trades_df["exit_date"] = pd.to_datetime(trades_df["exit_time"]).dt.date
    daily_pnl = trades_df.groupby("exit_date")["pnl_usd"].sum()
    daily_sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252) if daily_pnl.std() > 0 else 0.0

    # Average spread cost per trade
    avg_spread = trades_df["spread_usd"].mean()

    metrics = {
        "total_trades": len(trades_df),
        "win_rate": win_rate,
        "total_pnl_usd": total_pnl,
        "profit_factor": profit_factor,
        "max_drawdown_usd": max_dd,
        "daily_sharpe": daily_sharpe,
        "mean_duration_min": trades_df["duration_min"].mean(),
        "avg_spread_usd": avg_spread,
        "exit_reasons": trades_df["exit_reason"].value_counts().to_dict()
    }

    return {
        "trades": trades_df,
        "metrics": metrics,
        "daily_pnl": daily_pnl
    }
