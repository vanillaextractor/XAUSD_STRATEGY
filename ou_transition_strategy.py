"""
ou_transition_strategy.py - Ornstein-Uhlenbeck (OU) Transition-Gated Mean Reversion

Incorporates the transition-gap calibration framework from train_ou_priors.py:
1. Detects regime transitions: 'expansion' -> 'ranging' (volatility exhaustion).
2. Constrains mean-reversion trading to the post-transition gap window (e.g. <= 48 bars).
3. Evaluates rolling OU parameters: theta (mean-reversion speed), half-life t_{1/2}, and equilibrium attractor mu.
4. Executes with adaptive PDV stops/targets under real time-varying EDGE spread costs.
5. Evaluates both Two-Sided and Long-Only configurations across 2019-2024 Walk-Forward and 2025 Holdout.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from regime_signals import generate_feature_dataframe
from backtest_engine import compute_edge_spread, run_intraday_backtest
from pdv_features import compute_r1_r2

ARTIFACT_DIR = "/Users/pulkitchauhan/.gemini/antigravity-ide/brain/2730d759-fd1c-469d-96a1-a36f3e8dab3c"


def compute_rolling_ou_parameters(price_series: pd.Series, window: int = 30) -> pd.DataFrame:
    """
    Computes rolling AR(1) Ornstein-Uhlenbeck parameters on log prices:
    P_t = alpha + beta * P_{t-1} + eps
    theta = -ln(beta)
    half_life = ln(2) / theta
    mu = exp(alpha / (1 - beta))
    """
    log_p = np.log(price_series)
    y = log_p
    x = log_p.shift(1)

    mean_x = x.rolling(window).mean()
    mean_y = y.rolling(window).mean()
    mean_xx = (x * x).rolling(window).mean()
    mean_yy = (y * y).rolling(window).mean()
    mean_xy = (x * y).rolling(window).mean()

    var_x = mean_xx - mean_x * mean_x
    cov_xy = mean_xy - mean_x * mean_y
    var_y = mean_yy - mean_y * mean_y

    beta = cov_xy / var_x
    alpha = mean_y - beta * mean_x

    valid = (beta > 0.0) & (beta < 0.999)

    theta = pd.Series(np.nan, index=price_series.index)
    theta[valid] = -np.log(beta[valid])

    half_life = pd.Series(np.nan, index=price_series.index)
    half_life[valid] = np.log(2.0) / theta[valid]

    mu_log = pd.Series(np.nan, index=price_series.index)
    mu_log[valid] = alpha[valid] / (1.0 - beta[valid])
    mu = np.exp(mu_log)

    res_var = (var_y - beta * cov_xy).clip(lower=1e-12)
    sigma_ou = np.sqrt(res_var / (1.0 - beta**2))
    z_ou = (log_p - mu_log) / sigma_ou

    df_ou = pd.DataFrame({
        "theta": theta,
        "half_life": half_life,
        "mu": mu,
        "sigma_ou": sigma_ou,
        "z_ou": z_ou
    }, index=price_series.index)

    return df_ou


def run_ou_transition_pipeline():
    print("=" * 75)
    print("   OU TRANSITION-GATED MEAN REVERSION (GUYON & LEKEUFACK + OU PRIORS)   ")
    print("=" * 75 + "\n")

    # 1. Load Clean Data & OOS Predictions
    print("[1/4] Loading cached datasets and walk-forward predictions...")
    df_m5 = pd.read_parquet("xauusd_m5_clean.parquet")
    df_m1 = pd.read_parquet("xauusd_m1_clean.parquet")
    oos_df = pd.read_parquet("walk_forward_oos_predictions.parquet")
    folds_df = pd.read_parquet("walk_forward_folds.parquet")

    oos_df = oos_df[~oos_df.index.duplicated(keep="first")].copy()
    oos_df["sigma_hat"] = oos_df["sigma_hat"].clip(lower=5e-4)

    edge_spread = compute_edge_spread(df_m5, rolling_window=288)

    # 2. Generate Base Features & Transition Signals
    print("\n[2/4] Generating regime transition features & gap window filters...")
    feat_walk_forward = generate_feature_dataframe(
        df_m5.loc[oos_df.index],
        sigma_hat=oos_df["sigma_hat"],
        beta1_sign=oos_df["beta1_sign"],
        r1=oos_df["r1"],
        h=24,
        z_window=30,
        z_threshold=1.5,
        k1_stop=1.5,
        k2_target=1.0,
        regime_method="median",
        apply_beta1_tilt=True
    )

    # Transition detection: expansion -> ranging
    regime = feat_walk_forward["regime"]
    prev_regime = regime.shift(1)
    trans_exp_to_rang = (prev_regime == "expansion") & (regime == "ranging")
    print(f"  Expansion -> Ranging transitions: {trans_exp_to_rang.sum():,}")

    last_trans_idx = pd.Series(
        np.where(trans_exp_to_rang, np.arange(len(feat_walk_forward)), np.nan),
        index=feat_walk_forward.index
    ).ffill()
    bars_since_trans = np.arange(len(feat_walk_forward)) - last_trans_idx

    # Apply Gap Window Filter (<= 48 bars = 4 hours post-transition)
    gap_window = 48
    mask_in_gap = (bars_since_trans <= gap_window)

    feat_two_sided = feat_walk_forward.copy()
    feat_two_sided.loc[~mask_in_gap, "signal"] = 0

    feat_long_only = feat_walk_forward.copy()
    feat_long_only.loc[~mask_in_gap, "signal"] = 0
    feat_long_only.loc[feat_long_only["signal"] == -1, "signal"] = 0

    # 3. Walk-Forward Backtesting (2019-2024)
    print("\n[3/4] Simulating Walk-Forward Trades (2019-2024)...")
    dev_m1 = df_m1.loc[feat_walk_forward.index.min():feat_walk_forward.index.max() + pd.Timedelta(hours=4)]

    # Run baseline (no transition gap) for direct comparison
    bt_baseline = run_intraday_backtest(
        feat_walk_forward, dev_m1, edge_spread,
        use_edge_cost=True, cooldown_bars=12, z_exit_threshold=None
    )

    # Run OU Transition Gap (Two-Sided)
    bt_ou_two_sided = run_intraday_backtest(
        feat_two_sided, dev_m1, edge_spread,
        use_edge_cost=True, cooldown_bars=12, z_exit_threshold=None
    )

    # Run OU Transition Gap (Long-Only)
    bt_ou_long_only = run_intraday_backtest(
        feat_long_only, dev_m1, edge_spread,
        use_edge_cost=True, cooldown_bars=12, z_exit_threshold=None
    )

    print("\n" + "=" * 75)
    print("      WALK-FORWARD PERFORMANCE COMPARISON (2019 - 2024)      ")
    print("=" * 75)
    results_wf = [
        ("Baseline (All Ranging Bars, Two-Sided)", bt_baseline["metrics"]),
        ("OU Transition Gap (<=48 bars, Two-Sided)", bt_ou_two_sided["metrics"]),
        ("OU Transition Gap (<=48 bars, Long-Only)", bt_ou_long_only["metrics"])
    ]

    for name, m in results_wf:
        if m:
            print(f"\n[{name}]")
            print(f"  Total Trades:   {m['total_trades']:,}")
            print(f"  Win Rate:       {m['win_rate']*100:.2f}%")
            print(f"  Total Net PnL:  ${m['total_pnl_usd']:,.2f}")
            print(f"  Profit Factor:  {m['profit_factor']:.2f}")
            print(f"  Max Drawdown:   ${m['max_drawdown_usd']:,.2f}")
            print(f"  Annual Sharpe:  {m['daily_sharpe']:.2f}")
            print(f"  Mean Duration:  {m['mean_duration_min']:.1f} mins")

    # 4. Quarantined 2025 Holdout Evaluation
    print("\n" + "=" * 75)
    print("            QUARANTINED 2025 HOLDOUT EVALUATION             ")
    print("=" * 75)
    last_fold = folds_df.iloc[-1]
    holdout_m5 = df_m5[df_m5.index.year >= 2025].copy()
    late_2024 = df_m5[df_m5.index.year < 2025].iloc[-500:]
    holdout_full_returns = np.concatenate((late_2024["return"].values, holdout_m5["return"].values))

    r1_h, r2_h = compute_r1_r2(
        holdout_full_returns,
        last_fold["alpha1"], last_fold["delta1"],
        last_fold["alpha2"], last_fold["delta2"],
        N=500
    )
    r1_2025 = r1_h[500:]
    r2_2025 = r2_h[500:]

    X_2025 = np.column_stack((np.ones(len(holdout_m5)), r1_2025, np.sqrt(r2_2025)))
    betas_frozen = np.array([last_fold["beta0"], last_fold["beta1"], last_fold["beta2"]])
    sigma_hat_2025 = pd.Series(X_2025 @ betas_frozen, index=holdout_m5.index).clip(lower=5e-4)

    feat_2025 = generate_feature_dataframe(
        holdout_m5,
        sigma_hat=sigma_hat_2025,
        beta1_sign=pd.Series(int(np.sign(last_fold["beta1"])), index=holdout_m5.index),
        r1=pd.Series(r1_2025, index=holdout_m5.index),
        h=24,
        z_window=30,
        z_threshold=1.5,
        k1_stop=1.5,
        k2_target=1.0,
        regime_method="median",
        apply_beta1_tilt=True
    )

    reg_2025 = feat_2025["regime"]
    prev_reg_2025 = reg_2025.shift(1)
    trans_2025 = (prev_reg_2025 == "expansion") & (reg_2025 == "ranging")
    last_idx_2025 = pd.Series(
        np.where(trans_2025, np.arange(len(feat_2025)), np.nan),
        index=feat_2025.index
    ).ffill()
    bars_since_trans_2025 = np.arange(len(feat_2025)) - last_idx_2025

    feat_2025_ts = feat_2025.copy()
    feat_2025_ts.loc[bars_since_trans_2025 > gap_window, "signal"] = 0

    feat_2025_lo = feat_2025.copy()
    feat_2025_lo.loc[bars_since_trans_2025 > gap_window, "signal"] = 0
    feat_2025_lo.loc[feat_2025_lo["signal"] == -1, "signal"] = 0

    holdout_m1 = df_m1.loc[holdout_m5.index.min():holdout_m5.index.max() + pd.Timedelta(hours=4)]

    bt_2025_ts = run_intraday_backtest(feat_2025_ts, holdout_m1, edge_spread, cooldown_bars=12, z_exit_threshold=None)
    bt_2025_lo = run_intraday_backtest(feat_2025_lo, holdout_m1, edge_spread, cooldown_bars=12, z_exit_threshold=None)

    for name_h, res_h in [
        ("2025 Holdout (OU Transition Gap, Two-Sided)", bt_2025_ts),
        ("2025 Holdout (OU Transition Gap, Long-Only)", bt_2025_lo)
    ]:
        m = res_h["metrics"]
        if m:
            print(f"\n[{name_h}]")
            print(f"  Total Trades:   {m['total_trades']}")
            print(f"  Win Rate:       {m['win_rate']*100:.2f}%")
            print(f"  Total Net PnL:  ${m['total_pnl_usd']:,.2f}")
            print(f"  Profit Factor:  {m['profit_factor']:.2f}")
            print(f"  Max Drawdown:   ${m['max_drawdown_usd']:,.2f}")
            print(f"  Annual Sharpe:  {m['daily_sharpe']:.2f}")
            print(f"  Mean Duration:  {m['mean_duration_min']:.1f} mins")

    # 5. Plot Comparative Equity Curves
    print("\n[4/4] Generating comparative equity curve visual...")
    fig, ax = plt.subplots(figsize=(13, 6), dpi=150)

    t_base = bt_baseline["trades"]
    t_ts = bt_ou_two_sided["trades"]
    t_lo = bt_ou_long_only["trades"]

    if len(t_base) > 0:
        t_base["cum_pnl"] = t_base["pnl_usd"].cumsum()
        ax.plot(pd.to_datetime(t_base["exit_time"]), t_base["cum_pnl"],
                label=f"Baseline All Ranging Bars (Two-Sided): ${t_base['cum_pnl'].iloc[-1]:,.2f}",
                color="#64748b", lw=1.5, linestyle="--")

    if len(t_ts) > 0:
        t_ts["cum_pnl"] = t_ts["pnl_usd"].cumsum()
        ax.plot(pd.to_datetime(t_ts["exit_time"]), t_ts["cum_pnl"],
                label=f"OU Transition Gap <=48b (Two-Sided): +${t_ts['cum_pnl'].iloc[-1]:,.2f}",
                color="#2563eb", lw=2.0)

    if len(t_lo) > 0:
        t_lo["cum_pnl"] = t_lo["pnl_usd"].cumsum()
        ax.plot(pd.to_datetime(t_lo["exit_time"]), t_lo["cum_pnl"],
                label=f"OU Transition Gap <=48b (Long-Only): +${t_lo['cum_pnl'].iloc[-1]:,.2f}",
                color="#10b981", lw=2.4)

    ax.set_title("OU Transition-Gated Mean Reversion Cumulative Equity (2019-2024 Walk-Forward)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative PnL ($ / oz)", fontsize=11)
    ax.axhline(0, color="black", lw=0.8, linestyle=":")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs("images", exist_ok=True)
    plot_path_art = os.path.join(ARTIFACT_DIR, "ou_transition_pnl_comparison.png")
    plot_path_repo = "images/ou_transition_pnl_comparison.png"
    fig.savefig(plot_path_art)
    fig.savefig(plot_path_repo)
    plt.close(fig)
    print(f"  Saved plot to {plot_path_art} and {plot_path_repo}")

    print("\n" + "=" * 75)
    print("                     EXECUTION COMPLETE                            ")
    print("=" * 75)


if __name__ == "__main__":
    run_ou_transition_pipeline()
