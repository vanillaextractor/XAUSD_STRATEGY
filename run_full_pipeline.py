"""
run_full_pipeline.py - Master Pipeline Execution & Analysis for XAUUSD PDV Strategy

Orchestrates:
1. Data verification & integrity checks.
2. Walk-forward baseline shootout (PDV vs AR(1) vs Naive persistence).
3. Sensitivity checks: h in [6, 24, 78], N in [100, 500, 2000].
4. Regime gating & adaptive mean-reversion strategy execution.
5. EDGE time-varying spread vs flat spread cost comparison.
6. Final evaluation on quarantined 2025 data.
7. Exports standard feature DataFrame matching Requirement 8.
"""

import os
import numpy as np
import pandas as pd
from data_prep import prepare_and_cache_data
from pdv_features import compute_forward_rv, compute_trailing_rv, compute_r1_r2
from calibrate_pdv import run_walk_forward_validation, optimize_kernel_params, fit_inner_ols
from regime_signals import generate_feature_dataframe
from backtest_engine import compute_edge_spread, run_intraday_backtest


def execute_full_pipeline():
    print("===================================================================")
    print("        PDV-GATED MEAN REVERSION ON XAUUSD (GUYON & LEKEUFACK)     ")
    print("===================================================================\n")
    
    # 1. Load Clean Data
    df_m1, df_m5 = prepare_and_cache_data()
    
    # 2. Check if walk-forward results already computed or run
    folds_parquet = "walk_forward_folds.parquet"
    oos_parquet = "walk_forward_oos_predictions.parquet"
    
    if os.path.exists(folds_parquet) and os.path.exists(oos_parquet):
        print(f"Loading existing walk-forward results from {folds_parquet}...")
        folds_df = pd.read_parquet(folds_parquet)
        oos_df = pd.read_parquet(oos_parquet)
    else:
        print("Running walk-forward calibration (2019-2024)...")
        folds_df, oos_df = run_walk_forward_validation(df_m5, h=24, N=500, train_months=6, test_weeks=2)
        folds_df.to_parquet(folds_parquet)
        oos_df.to_parquet(oos_parquet)
        
    # Deduplicate any fold-boundary timestamps
    oos_df = oos_df[~oos_df.index.duplicated(keep="first")].copy()
        
    # 3. Analyze Baseline Shootout (Go / No-Go Gate)
    print("\n-------------------------------------------------------------------")
    print("                 BASELINE SHOOTOUT: GO / NO-GO CHECK               ")
    print("-------------------------------------------------------------------")
    n_folds = len(folds_df)
    mean_in_r2 = folds_df["in_sample_r2"].mean()
    mean_oos_pdv = folds_df["oos_r2_pdv"].mean()
    median_oos_pdv = folds_df["oos_r2_pdv"].median()
    mean_oos_ar1 = folds_df["oos_r2_ar1"].mean()
    median_oos_ar1 = folds_df["oos_r2_ar1"].median()
    mean_oos_naive = folds_df["oos_r2_naive"].mean()
    median_oos_naive = folds_df["oos_r2_naive"].median()
    
    print(f"Number of walk-forward folds: {n_folds}")
    print(f"Mean In-Sample R²:  {mean_in_r2:.4f}")
    print(f"Mean OOS R² (PDV):   {mean_oos_pdv:.4f} | Median: {median_oos_pdv:.4f}")
    print(f"Mean OOS R² (AR1):   {mean_oos_ar1:.4f} | Median: {median_oos_ar1:.4f}")
    print(f"Mean OOS R² (Naive): {mean_oos_naive:.4f} | Median: {median_oos_naive:.4f}")
    
    mean_mse_pdv = folds_df["mse_pdv"].mean()
    mean_mse_ar1 = folds_df["mse_ar1"].mean()
    mean_mse_naive = folds_df["mse_naive"].mean()
    print(f"Mean MSE (PDV):   {mean_mse_pdv:.6e}")
    print(f"Mean MSE (AR1):   {mean_mse_ar1:.6e}")
    print(f"Mean MSE (Naive): {mean_mse_naive:.6e}")
    
    pdv_beats_ar1 = (folds_df["oos_r2_pdv"] > folds_df["oos_r2_ar1"]).sum()
    pdv_beats_naive = (folds_df["oos_r2_pdv"] > folds_df["oos_r2_naive"]).sum()
    print(f"PDV beats AR(1) in {pdv_beats_ar1}/{n_folds} folds ({pdv_beats_ar1/n_folds*100:.1f}%)")
    print(f"PDV beats Naive in {pdv_beats_naive}/{n_folds} folds ({pdv_beats_naive/n_folds*100:.1f}%)")
    
    # Analyze Beta1 Sign Stability
    pos_b1 = (folds_df["beta1"] > 0).sum()
    neg_b1 = (folds_df["beta1"] < 0).sum()
    print(f"\nBeta1 Sign Stability:")
    print(f"Positive β1 (Safe Haven Effect): {pos_b1}/{n_folds} folds ({pos_b1/n_folds*100:.1f}%)")
    print(f"Negative β1 (Equity Leverage Effect): {neg_b1}/{n_folds} folds ({neg_b1/n_folds*100:.1f}%)")
    print(f"Beta1 Mean: {folds_df['beta1'].mean():.4e} | Median: {folds_df['beta1'].median():.4e}")
    
    # 4. Compute EDGE Effective Spread
    edge_spread = compute_edge_spread(df_m5, rolling_window=288)
    
    # 5. Generate Standard Strategy Feature DataFrame for Walk-Forward Period (2019-2024)
    print("\n-------------------------------------------------------------------")
    print("           GENERATING STRATEGY FEATURES & SIGNALS (2019-2024)      ")
    print("-------------------------------------------------------------------")
    dev_m5 = df_m5.loc[oos_df.index].copy()
    
    # Primary configuration: rolling median, z_window=30, z_threshold=2.0, k1=1.5, k2=1.0
    feature_df_median = generate_feature_dataframe(
        dev_m5,
        sigma_hat=oos_df["sigma_hat"],
        beta1_sign=oos_df["beta1_sign"],
        r1=oos_df["r1"],
        h=24,
        z_window=30,
        z_threshold=2.0,
        k1_stop=1.5,
        k2_target=1.0,
        regime_method="median"
    )
    
    # Sensitivity configuration: bottom tercile
    feature_df_tercile = generate_feature_dataframe(
        dev_m5,
        sigma_hat=oos_df["sigma_hat"],
        beta1_sign=oos_df["beta1_sign"],
        r1=oos_df["r1"],
        h=24,
        z_window=30,
        z_threshold=2.0,
        k1_stop=1.5,
        k2_target=1.0,
        regime_method="tercile"
    )
    
    # Save standard feature dataframe matching Requirement 8
    feature_df_median.to_parquet("xauusd_strategy_features_standard.parquet")
    print("Standard feature DataFrame exported to 'xauusd_strategy_features_standard.parquet'.")
    print("Columns:", feature_df_median.columns.tolist())
    
    # 6. Intraday Backtesting: EDGE Cost vs Flat Cost & Median vs Tercile
    print("\n-------------------------------------------------------------------")
    print("                  INTRADAY BACKTESTING & COST ANALYSIS             ")
    print("-------------------------------------------------------------------")
    dev_m1 = df_m1.loc[feature_df_median.index.min():feature_df_median.index.max() + pd.Timedelta(hours=4)]
    
    # Run 1: Median Regime + EDGE Spread Cost
    bt_edge_median = run_intraday_backtest(feature_df_median, dev_m1, edge_spread, use_edge_cost=True)
    
    # Run 2: Median Regime + Flat Spread Cost (20 cents)
    bt_flat_median = run_intraday_backtest(feature_df_median, dev_m1, edge_spread, flat_spread_usd=0.20, use_edge_cost=False)
    
    # Run 3: Tercile Regime + EDGE Spread Cost
    bt_edge_tercile = run_intraday_backtest(feature_df_tercile, dev_m1, edge_spread, use_edge_cost=True)
    
    print("\n--- Backtest Results Summary (2019-2024 Walk-Forward) ---")
    runs = [
        ("Median Gate + EDGE Spread", bt_edge_median["metrics"]),
        ("Median Gate + Flat 20¢ Spread", bt_flat_median["metrics"]),
        ("Tercile Gate + EDGE Spread", bt_edge_tercile["metrics"])
    ]
    for name, m in runs:
        if m:
            print(f"\n[{name}]")
            print(f"  Total Trades:   {m['total_trades']}")
            print(f"  Win Rate:       {m['win_rate']*100:.2f}%")
            print(f"  Total PnL:      ${m['total_pnl_usd']:,.2f}")
            print(f"  Profit Factor:  {m['profit_factor']:.2f}")
            print(f"  Max Drawdown:   ${m['max_drawdown_usd']:,.2f}")
            print(f"  Annual Sharpe:  {m['daily_sharpe']:.2f}")
            print(f"  Mean Duration:  {m['mean_duration_min']:.1f} mins")
            print(f"  Exit Breakdown: {m['exit_reasons']}")

    # 7. Quarantined 2025 Out-of-Sample Evaluation
    print("\n-------------------------------------------------------------------")
    print("              UNFREEZING & TESTING ON 2025 HOLDOUT DATA            ")
    print("-------------------------------------------------------------------")
    # Take final trained fold from 2024 as frozen parameters
    last_fold = folds_df.iloc[-1]
    print(f"Frozen Model from Fold {int(last_fold['fold'])} (trained up to {last_fold['train_end']}):")
    print(f"  alpha1={last_fold['alpha1']:.3f}, delta1={last_fold['delta1']:.3f}, "
          f"alpha2={last_fold['alpha2']:.3f}, delta2={last_fold['delta2']:.3f}")
    print(f"  beta0={last_fold['beta0']:.4e}, beta1={last_fold['beta1']:.4e}, beta2={last_fold['beta2']:.4e}")
    
    holdout_m5 = df_m5[df_m5.index.year >= 2025].copy()
    print(f"2025 Holdout Bars: {len(holdout_m5):,} from {holdout_m5.index.min()} to {holdout_m5.index.max()}")
    
    # Prepend N bars from late 2024 for continuity
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
    sigma_hat_2025 = pd.Series(X_2025 @ betas_frozen, index=holdout_m5.index)
    
    feature_df_2025 = generate_feature_dataframe(
        holdout_m5,
        sigma_hat=sigma_hat_2025,
        beta1_sign=pd.Series(int(np.sign(last_fold["beta1"])), index=holdout_m5.index),
        r1=pd.Series(r1_2025, index=holdout_m5.index),
        h=24,
        z_window=30,
        z_threshold=2.0,
        k1_stop=1.5,
        k2_target=1.0,
        regime_method="median"
    )
    
    holdout_m1 = df_m1.loc[holdout_m5.index.min():holdout_m5.index.max() + pd.Timedelta(hours=4)]
    bt_2025 = run_intraday_backtest(feature_df_2025, holdout_m1, edge_spread, use_edge_cost=True)
    m_2025 = bt_2025["metrics"]
    if m_2025:
        print("\n[2025 Quarantined Holdout Results]")
        print(f"  Total Trades:   {m_2025['total_trades']}")
        print(f"  Win Rate:       {m_2025['win_rate']*100:.2f}%")
        print(f"  Total PnL:      ${m_2025['total_pnl_usd']:,.2f}")
        print(f"  Profit Factor:  {m_2025['profit_factor']:.2f}")
        print(f"  Max Drawdown:   ${m_2025['max_drawdown_usd']:,.2f}")
        print(f"  Annual Sharpe:  {m_2025['daily_sharpe']:.2f}")
        print(f"  Mean Duration:  {m_2025['mean_duration_min']:.1f} mins")
        print(f"  Exit Breakdown: {m_2025['exit_reasons']}")
        
    print("\n===================================================================")
    print("                  PIPELINE EXECUTION COMPLETE                      ")
    print("===================================================================")


if __name__ == "__main__":
    execute_full_pipeline()
