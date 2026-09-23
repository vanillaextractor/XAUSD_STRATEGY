"""
run_integrated_pipeline.py - Master Integrated Pipeline for XAUUSD

Unifies:
1. new_engine: Fractal-Adaptive Sampling (FAS) for local Hurst exponent (H_t) and dynamic threshold (theta_t).
2. calibrate_pdv.py: Guyon & Lekeufack (2023) Path-Dependent Volatility (PDV) forward forecasting & regime gating.
3. calibrate_ou_transitions.py: Per-transition Ornstein-Uhlenbeck (OU) priors (theta, R_vol, half-life).
4. backtest_engine.py: Realistic 1-minute execution fill simulation under time-varying EDGE spread costs.

Generates:
- Comprehensive performance comparison matrix across Walk-Forward (2019-2024) and 2025 Holdout.
- High-resolution comparative equity curve plot: images/integrated_pipeline_pnl_comparison.png.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure root repository is in Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_prep import prepare_and_cache_data
from fas_features import compute_fas_features
from calibrate_ou_transitions import run_transition_calibration
from regime_signals import generate_feature_dataframe
from backtest_engine import compute_edge_spread, run_intraday_backtest
from pdv_features import compute_r1_r2
from constants import SIGMA_HAT_FLOOR

ARTIFACT_DIR = "/Users/pulkitchauhan/.gemini/antigravity-ide/brain/2730d759-fd1c-469d-96a1-a36f3e8dab3c"


def run_integrated_pipeline():
    print("=" * 80)
    print("      INTEGRATED XAUUSD QUANTITATIVE PIPELINE: FAS + PDV + OU PRIORS        ")
    print("=" * 80 + "\n")

    # 1. Data Loading & Integrity
    print("[1/5] Loading clean XAUUSD datasets...")
    df_m1, df_m5 = prepare_and_cache_data()
    print(f"  M1 bars: {len(df_m1):,} | M5 bars: {len(df_m5):,}")

    edge_spread = compute_edge_spread(df_m5, rolling_window=288)

    # 2. Stage 1: FAS Fractal Roughness & Dynamic Thresholds (new_engine)
    print("\n[2/5] Computing Fractal-Adaptive Sampling (FAS) features...")
    fas_df = compute_fas_features(df_m5, theta_base=0.002, window=30, sigma_ref_window=288)
    print(f"  Hurst exponent: mean={fas_df['hurst'].mean():.3f}, median={fas_df['hurst'].median():.3f}")
    print(f"  Dynamic threshold: mean={fas_df['theta_dynamic'].mean()*10000:.1f} bps")

    # 3. Stage 2: Path-Dependent Volatility (PDV) Walk-Forward Predictions
    print("\n[3/5] Loading Guyon & Lekeufack PDV walk-forward predictions...")
    oos_parquet = "walk_forward_oos_predictions.parquet"
    folds_parquet = "walk_forward_folds.parquet"

    if not os.path.exists(oos_parquet) or not os.path.exists(folds_parquet):
        raise FileNotFoundError("Walk-forward PDV files not found. Run calibrate_pdv.py first.")

    oos_df = pd.read_parquet(oos_parquet)
    folds_df = pd.read_parquet(folds_parquet)
    oos_df = oos_df[~oos_df.index.duplicated(keep="first")].copy()
    oos_df["sigma_hat"] = oos_df["sigma_hat"].clip(lower=SIGMA_HAT_FLOOR)
    print(f"  Loaded {len(oos_df):,} out-of-sample predictions across {len(folds_df)} folds.")

    # 4. Stage 3: Per-Transition OU Prior Calibration (train_ou_priors.py framework)
    print("\n[4/5] Calibrating Per-Transition Ornstein-Uhlenbeck (OU) Priors...")
    ou_priors = run_transition_calibration("ou_priors_xausd.json")
    ou_trans = ou_priors.get("expansion→ranging", {})
    theta_prior = ou_trans.get("theta_median", 0.2746)
    half_life_prior = ou_trans.get("half_life_median_bars", 2.52)
    vol_ratio_prior = ou_trans.get("vol_ratio_median", 0.9587)
    print(f"  Calibrated expansion→ranging: θ={theta_prior:.4f}, t_1/2={half_life_prior:.1f} bars, R_vol={vol_ratio_prior:.4f}")

    # 5. Construct Strategy Signals with Integrated Filters
    print("\n[5/5] Generating integrated strategy features & executing backtests...")
    dev_m5 = df_m5.loc[oos_df.index].copy()
    
    # Base PDV features with beta1 tilt
    feat_pdv = generate_feature_dataframe(
        dev_m5,
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

    # Attach FAS metrics
    feat_pdv["hurst"] = fas_df.loc[feat_pdv.index, "hurst"]
    feat_pdv["theta_dynamic"] = fas_df.loc[feat_pdv.index, "theta_dynamic"]

    # Transition detection: expansion -> ranging
    regime = feat_pdv["regime"]
    prev_regime = regime.shift(1)
    trans_exp_to_rang = (prev_regime == "expansion") & (regime == "ranging")

    last_trans_idx = pd.Series(
        np.where(trans_exp_to_rang, np.arange(len(feat_pdv)), np.nan),
        index=feat_pdv.index
    ).ffill()
    bars_since_trans = np.arange(len(feat_pdv)) - last_trans_idx

    # Condition: Inside post-transition gap window (<= 48 bars)
    gap_window = 48
    mask_transition_gap = (bars_since_trans <= gap_window)

    # Additional FAS filter: Skip persistent/trending regimes where Hurst > 0.60
    mask_fas_reversion = (feat_pdv["hurst"] <= 0.60)

    # Configuration 1: Pure PDV (All ranging bars, no transition constraint)
    feat_cfg1 = feat_pdv.copy()

    # Configuration 2: Integrated Pipeline (Two-Sided)
    feat_cfg2 = feat_pdv.copy()
    feat_cfg2.loc[~(mask_transition_gap & mask_fas_reversion), "signal"] = 0

    # Configuration 3: Integrated Pipeline (Long-Only)
    feat_cfg3 = feat_pdv.copy()
    feat_cfg3.loc[~(mask_transition_gap & mask_fas_reversion), "signal"] = 0
    feat_cfg3.loc[feat_cfg3["signal"] == -1, "signal"] = 0

    # Backtesting Walk-Forward (2019-2024)
    dev_m1 = df_m1.loc[feat_pdv.index.min():feat_pdv.index.max() + pd.Timedelta(hours=4)]

    print("\n--- Running Walk-Forward Comparative Simulations (2019-2024) ---")
    bt_cfg1 = run_intraday_backtest(feat_cfg1, dev_m1, edge_spread, cooldown_bars=12, z_exit_threshold=None)
    bt_cfg2 = run_intraday_backtest(feat_cfg2, dev_m1, edge_spread, cooldown_bars=12, z_exit_threshold=None)
    bt_cfg3 = run_intraday_backtest(feat_cfg3, dev_m1, edge_spread, cooldown_bars=12, z_exit_threshold=None)

    runs_wf = [
        ("Pure PDV Baseline (All Ranging Bars, Two-Sided)", bt_cfg1["metrics"]),
        ("Integrated FAS + PDV + OU (Two-Sided)", bt_cfg2["metrics"]),
        ("Integrated FAS + PDV + OU (Long-Only)", bt_cfg3["metrics"])
    ]

    print("\n" + "=" * 80)
    print("            WALK-FORWARD PERFORMANCE SUMMARY (2019 - 2024)            ")
    print("=" * 80)
    for name, m in runs_wf:
        if m:
            print(f"\n[{name}]")
            print(f"  Total Trades:   {m['total_trades']:,}")
            print(f"  Win Rate:       {m['win_rate']*100:.2f}%")
            print(f"  Total Net PnL:  ${m['total_pnl_usd']:,.2f}")
            print(f"  Profit Factor:  {m['profit_factor']:.2f}")
            print(f"  Max Drawdown:   ${m['max_drawdown_usd']:,.2f}")
            print(f"  Annual Sharpe:  {m['daily_sharpe']:.2f}")
            print(f"  Mean Duration:  {m['mean_duration_min']:.1f} mins")
            print(f"  Avg Spread:     ${m.get('avg_spread_usd', 0):.4f}")

    # Quarantined 2025 Holdout Validation
    print("\n" + "=" * 80)
    print("               QUARANTINED 2025 HOLDOUT EVALUATION                    ")
    print("=" * 80)
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
    sigma_hat_2025 = pd.Series(X_2025 @ betas_frozen, index=holdout_m5.index).clip(lower=SIGMA_HAT_FLOOR)

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
    feat_2025["hurst"] = fas_df.loc[feat_2025.index, "hurst"]

    reg_2025 = feat_2025["regime"]
    prev_reg_2025 = reg_2025.shift(1)
    trans_2025 = (prev_reg_2025 == "expansion") & (reg_2025 == "ranging")
    last_idx_2025 = pd.Series(
        np.where(trans_2025, np.arange(len(feat_2025)), np.nan),
        index=feat_2025.index
    ).ffill()
    bars_since_trans_2025 = np.arange(len(feat_2025)) - last_idx_2025

    mask_h_gap = (bars_since_trans_2025 <= gap_window)
    mask_h_fas = (feat_2025["hurst"] <= 0.60)

    feat_2025_ts = feat_2025.copy()
    feat_2025_ts.loc[~(mask_h_gap & mask_h_fas), "signal"] = 0

    feat_2025_lo = feat_2025.copy()
    feat_2025_lo.loc[~(mask_h_gap & mask_h_fas), "signal"] = 0
    feat_2025_lo.loc[feat_2025_lo["signal"] == -1, "signal"] = 0

    holdout_m1 = df_m1.loc[holdout_m5.index.min():holdout_m5.index.max() + pd.Timedelta(hours=4)]
    bt_2025_ts = run_intraday_backtest(feat_2025_ts, holdout_m1, edge_spread, cooldown_bars=12, z_exit_threshold=None)
    bt_2025_lo = run_intraday_backtest(feat_2025_lo, holdout_m1, edge_spread, cooldown_bars=12, z_exit_threshold=None)

    runs_holdout = [
        ("2025 Holdout: Integrated FAS + PDV + OU (Two-Sided)", bt_2025_ts["metrics"]),
        ("2025 Holdout: Integrated FAS + PDV + OU (Long-Only)", bt_2025_lo["metrics"])
    ]

    for name, m in runs_holdout:
        if m:
            print(f"\n[{name}]")
            print(f"  Total Trades:   {m['total_trades']}")
            print(f"  Win Rate:       {m['win_rate']*100:.2f}%")
            print(f"  Total Net PnL:  ${m['total_pnl_usd']:,.2f}")
            print(f"  Profit Factor:  {m['profit_factor']:.2f}")
            print(f"  Max Drawdown:   ${m['max_drawdown_usd']:,.2f}")
            print(f"  Annual Sharpe:  {m['daily_sharpe']:.2f}")
            print(f"  Mean Duration:  {m['mean_duration_min']:.1f} mins")

    # Generate Comparative Equity Curves Plot
    print("\n--- Generating Comparative Equity Curve Visual ---")
    fig, ax = plt.subplots(figsize=(13, 6), dpi=150)

    t1 = bt_cfg1["trades"]
    t2 = bt_cfg2["trades"]
    t3 = bt_cfg3["trades"]

    if len(t1) > 0:
        t1["cum_pnl"] = t1["pnl_usd"].cumsum()
        ax.plot(pd.to_datetime(t1["exit_time"]), t1["cum_pnl"],
                label=f"Pure PDV Baseline (All Ranging Bars): +${t1['cum_pnl'].iloc[-1]:,.2f}",
                color="#64748b", lw=1.5, linestyle="--")

    if len(t2) > 0:
        t2["cum_pnl"] = t2["pnl_usd"].cumsum()
        ax.plot(pd.to_datetime(t2["exit_time"]), t2["cum_pnl"],
                label=f"Integrated FAS + PDV + OU (Two-Sided): +${t2['cum_pnl'].iloc[-1]:,.2f}",
                color="#2563eb", lw=2.0)

    if len(t3) > 0:
        t3["cum_pnl"] = t3["pnl_usd"].cumsum()
        ax.plot(pd.to_datetime(t3["exit_time"]), t3["cum_pnl"],
                label=f"Integrated FAS + PDV + OU (Long-Only): +${t3['cum_pnl'].iloc[-1]:,.2f}",
                color="#10b981", lw=2.4)

    ax.set_title("Integrated FAS + PDV + OU Priors Strategy: Cumulative PnL (2019-2024 Walk-Forward)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative PnL ($ / oz)", fontsize=11)
    ax.axhline(0, color="black", lw=0.8, linestyle=":")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs("images", exist_ok=True)
    out_art = os.path.join(ARTIFACT_DIR, "integrated_pipeline_pnl_comparison.png")
    out_repo = "images/integrated_pipeline_pnl_comparison.png"
    fig.savefig(out_art)
    fig.savefig(out_repo)
    plt.close(fig)
    print(f"Saved plot to {out_art} and {out_repo}")

    print("\n" + "=" * 80)
    print("                     PIPELINE EXECUTION COMPLETE                      ")
    print("=" * 80)


if __name__ == "__main__":
    run_integrated_pipeline()
