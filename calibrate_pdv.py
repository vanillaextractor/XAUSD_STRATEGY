"""
calibrate_pdv.py - Nested Calibration & Walk-Forward Validation Engine for PDV Model

Implements:
1. Inner loop: OLS estimation of beta0, beta1, beta2 given kernel parameters (alpha1, delta1, alpha2, delta2).
2. Outer loop: Optimization of kernel parameters on training window to maximize in-sample R^2.
3. Walk-Forward Validation (2019 to 2024):
   - Rolling 6-month train window (~125 trading days ~35,000 bars)
   - 2-week test window (~10 trading days ~2,800 bars)
   - Roll forward by 2 weeks
   - Baselines: Naive Persistence (RV_trailing) and AR(1) on RV_trailing
   - Records out-of-sample R^2, MSE, beta1 sign, parameter stability per fold.
"""

import os
import time
import numpy as np
import pandas as pd
import scipy.optimize
from pdv_features import (
    compute_r1_r2,
    compute_forward_rv,
    compute_trailing_rv
)


def fit_inner_ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Fits OLS regression: y = X @ beta + epsilon
    Returns betas and R^2.
    """
    XtX = X.T @ X
    Xty = X.T @ y
    # Small ridge regularization for numerical stability
    ridge = 1e-10 * np.eye(X.shape[1])
    betas = np.linalg.solve(XtX + ridge, Xty)
    y_pred = X @ betas
    
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return betas, r2


def optimize_kernel_params(
    returns_train: np.ndarray,
    y_train: np.ndarray,
    N: int = 500,
    init_params: list = None
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Outer loop: Search over (alpha1, delta1, alpha2, delta2) maximizing in-sample R^2.
    
    BUG 5 FIX: Uses differential_evolution with proper bounds instead of
    Nelder-Mead, which was saturating at boundary in 62% of folds.
    Bounds expanded (alpha up to 10, delta up to 500).
    
    Returns: best_kernel_params, in_sample_r2, betas
    """
    M = len(y_train)
    
    # Wider bounds to prevent boundary saturation (BUG 5)
    bounds = [
        (0.1, 10.0),   # alpha1
        (0.05, 500.0),  # delta1
        (0.1, 10.0),   # alpha2
        (0.05, 500.0),  # delta2
    ]
    
    def objective(params):
        a1, d1, a2, d2 = params
        r1, r2 = compute_r1_r2(returns_train, a1, d1, a2, d2, N=N)
        sqrt_r2 = np.sqrt(r2)
        X = np.column_stack((np.ones(M - N), r1[N:], sqrt_r2[N:]))
        _, r2_score = fit_inner_ols(X, y_train[N:])
        return -r2_score

    # Use differential_evolution: global optimizer with proper bound handling
    # seed for reproducibility, polish=True applies L-BFGS-B refinement at the end
    res = scipy.optimize.differential_evolution(
        objective,
        bounds=bounds,
        seed=42,
        maxiter=60,
        tol=1e-4,
        polish=True,
        init="sobol",
        popsize=10
    )
    
    # If init_params were provided, also evaluate them and keep the best
    if init_params is not None:
        init_score = -objective(init_params)
        de_score = -res.fun
        if init_score > de_score:
            # Previous fold's params are better; refine from there
            res_nm = scipy.optimize.minimize(
                objective,
                list(init_params),
                method="Nelder-Mead",
                options={"maxiter": 60, "xatol": 1e-3, "fatol": 1e-4}
            )
            if -res_nm.fun > de_score:
                res = res_nm
    
    opt_params = np.clip(res.x, [b[0] for b in bounds], [b[1] for b in bounds])
    opt_a1, opt_d1, opt_a2, opt_d2 = opt_params
    
    # Compute final train betas
    r1, r2 = compute_r1_r2(returns_train, opt_a1, opt_d1, opt_a2, opt_d2, N=N)
    X = np.column_stack((np.ones(M - N), r1[N:], np.sqrt(r2)[N:]))
    betas, in_sample_r2 = fit_inner_ols(X, y_train[N:])
    
    return opt_params, in_sample_r2, betas


def run_walk_forward_validation(
    df_5m: pd.DataFrame,
    h: int = 24,
    N: int = 500,
    train_months: int = 6,
    test_weeks: int = 2,
    holdout_year: int = 2025,
    max_folds: int = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes walk-forward calibration and validation.
    Holds out data from `holdout_year` onwards.
    
    Returns:
    - folds_df: Summary of each walk-forward fold (in-sample R2, OOS R2 for PDV, AR(1), Naive, betas, etc.)
    - oos_predictions_df: Out-of-sample bar-by-bar forecasts across the entire walk-forward period.
    """
    print(f"\n=======================================================")
    print(f"Starting Walk-Forward Calibration: h={h} bars (2h), N={N} lookback")
    print(f"Train: {train_months} months, Test: {test_weeks} weeks")
    print(f"Holdout: Year >= {holdout_year} is strictly quarantined.")
    print(f"=======================================================\n")
    
    # Precompute targets
    if "rv_fwd" not in df_5m.columns:
        df_5m["rv_fwd"] = compute_forward_rv(df_5m["gk_var"], h=h)
    if "rv_trail" not in df_5m.columns:
        df_5m["rv_trail"] = compute_trailing_rv(df_5m["gk_var"], h=h)
        
    # Restrict walk-forward development to strictly before holdout year
    dev_df = df_5m[df_5m.index.year < holdout_year].copy()
    start_date = dev_df.index.min()
    end_date = dev_df.index.max()
    print(f"Development date range: {start_date} to {end_date}")
    
    # Train window length in days (~182 days for 6m)
    train_delta = pd.Timedelta(days=train_months * 30.5)
    test_delta = pd.Timedelta(days=test_weeks * 7)
    
    current_train_start = start_date
    current_train_end = current_train_start + train_delta
    
    folds = []
    oos_preds = []
    prev_opt_params = None
    fold_idx = 0
    t0_all = time.time()
    
    while current_train_end + test_delta <= end_date:
        current_test_start = current_train_end
        current_test_end = current_test_start + test_delta
        
        # BUG 6 FIX: Create a buffer between train and test to prevent
        # rv_fwd target leakage. The last h bars of train have rv_fwd
        # targets that look into the test window, so we drop them.
        gap_buffer = pd.Timedelta(minutes=h * 5)  # h bars * 5 min/bar
        train_cutoff = current_train_end - gap_buffer
        train_data = dev_df.loc[current_train_start:train_cutoff].dropna(subset=["rv_fwd", "rv_trail"])
        test_data = dev_df.loc[current_test_start:current_test_end].dropna(subset=["rv_fwd", "rv_trail"])
        
        if len(train_data) < N + 1000 or len(test_data) < 100:
            current_train_start += test_delta
            current_train_end += test_delta
            continue
            
        fold_t0 = time.time()
        
        returns_train = train_data["return"].values
        y_train = train_data["rv_fwd"].values
        
        # 1. Fit PDV Model
        opt_params, in_sample_r2, betas = optimize_kernel_params(
            returns_train, y_train, N=N, init_params=prev_opt_params
        )
        prev_opt_params = opt_params
        a1, d1, a2, d2 = opt_params
        b0, b1, b2 = betas
        
        # 2. Fit AR(1) Baseline on Train
        X_ar1_train = np.column_stack((np.ones(len(train_data) - N), train_data["rv_trail"].values[N:]))
        betas_ar1, in_sample_r2_ar1 = fit_inner_ols(X_ar1_train, y_train[N:])
        
        # 3. Evaluate on Test Block
        # Must prepend N bars from train to test returns for seamless causal filtering
        returns_test_full = np.concatenate((returns_train[-N:], test_data["return"].values))
        r1_test_full, r2_test_full = compute_r1_r2(returns_test_full, a1, d1, a2, d2, N=N)
        r1_test = r1_test_full[N:]
        r2_test = r2_test_full[N:]
        sqrt_r2_test = np.sqrt(r2_test)
        
        y_test = test_data["rv_fwd"].values
        X_test = np.column_stack((np.ones(len(y_test)), r1_test, sqrt_r2_test))
        y_pred_pdv = X_test @ betas
        
        # AR(1) Forecast
        X_ar1_test = np.column_stack((np.ones(len(y_test)), test_data["rv_trail"].values))
        y_pred_ar1 = X_ar1_test @ betas_ar1
        
        # Naive Forecast
        y_pred_naive = test_data["rv_trail"].values
        
        # Out-of-sample metrics
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        ss_res_pdv = np.sum((y_test - y_pred_pdv) ** 2)
        ss_res_ar1 = np.sum((y_test - y_pred_ar1) ** 2)
        ss_res_naive = np.sum((y_test - y_pred_naive) ** 2)
        
        oos_r2_pdv = 1.0 - (ss_res_pdv / ss_tot) if ss_tot > 0 else 0.0
        oos_r2_ar1 = 1.0 - (ss_res_ar1 / ss_tot) if ss_tot > 0 else 0.0
        oos_r2_naive = 1.0 - (ss_res_naive / ss_tot) if ss_tot > 0 else 0.0
        
        mse_pdv = np.mean((y_test - y_pred_pdv) ** 2)
        mse_ar1 = np.mean((y_test - y_pred_ar1) ** 2)
        mse_naive = np.mean((y_test - y_pred_naive) ** 2)
        
        fold_time = time.time() - fold_t0
        
        folds.append({
            "fold": fold_idx,
            "train_start": current_train_start,
            "train_end": current_train_end,
            "test_start": current_test_start,
            "test_end": current_test_end,
            "in_sample_r2": in_sample_r2,
            "oos_r2_pdv": oos_r2_pdv,
            "oos_r2_ar1": oos_r2_ar1,
            "oos_r2_naive": oos_r2_naive,
            "mse_pdv": mse_pdv,
            "mse_ar1": mse_ar1,
            "mse_naive": mse_naive,
            "alpha1": a1,
            "delta1": d1,
            "alpha2": a2,
            "delta2": d2,
            "beta0": b0,
            "beta1": b1,
            "beta2": b2,
            "beta1_sign": int(np.sign(b1)),
            "fit_time_s": fold_time
        })
        
        # Save OOS predictions with timestamps
        fold_pred_df = pd.DataFrame({
            "rv_true": y_test,
            "sigma_hat": y_pred_pdv,
            "sigma_ar1": y_pred_ar1,
            "sigma_naive": y_pred_naive,
            "beta1_sign": int(np.sign(b1)),
            "beta1": b1,
            "r1": r1_test,
            "r2": r2_test,
            "fold": fold_idx
        }, index=test_data.index)
        oos_preds.append(fold_pred_df)
        
        if fold_idx % 10 == 0 or fold_idx < 5:
            print(f"Fold {fold_idx:03d} | Test: {current_test_start.strftime('%Y-%m-%d')} to {current_test_end.strftime('%Y-%m-%d')} | "
                  f"In R²: {in_sample_r2:.3f} | OOS R² PDV: {oos_r2_pdv:+.3f} | AR(1): {oos_r2_ar1:+.3f} | Naive: {oos_r2_naive:+.3f} | "
                  f"β1: {b1:+.3e} (sign {int(np.sign(b1))}) | time: {fold_time:.2f}s")
            
        fold_idx += 1
        if max_folds is not None and fold_idx >= max_folds:
            break
            
        # Roll forward by test window
        current_train_start += test_delta
        current_train_end += test_delta

    folds_df = pd.DataFrame(folds)
    oos_df = pd.concat(oos_preds) if oos_preds else pd.DataFrame()
    
    total_time = time.time() - t0_all
    print(f"\nWalk-Forward Completed: {len(folds_df)} folds processed in {total_time:.1f}s.")
    
    return folds_df, oos_df


if __name__ == "__main__":
    df_5m = pd.read_parquet("xauusd_m5_clean.parquet")
    folds_df, oos_df = run_walk_forward_validation(df_5m, h=24, N=500, train_months=6, test_weeks=2)
    
    # Save results
    folds_df.to_parquet("walk_forward_folds.parquet")
    oos_df.to_parquet("walk_forward_oos_predictions.parquet")
    
    print("\n--- Walk-Forward Summary Statistics (2019 - 2024) ---")
    print(f"Total Folds: {len(folds_df)}")
    print(f"Mean In-Sample R²: {folds_df['in_sample_r2'].mean():.4f}")
    print(f"Mean OOS R² (PDV):   {folds_df['oos_r2_pdv'].mean():.4f} (median: {folds_df['oos_r2_pdv'].median():.4f})")
    print(f"Mean OOS R² (AR1):   {folds_df['oos_r2_ar1'].mean():.4f} (median: {folds_df['oos_r2_ar1'].median():.4f})")
    print(f"Mean OOS R² (Naive): {folds_df['oos_r2_naive'].mean():.4f} (median: {folds_df['oos_r2_naive'].median():.4f})")
    
    print(f"Mean MSE (PDV):   {folds_df['mse_pdv'].mean():.8e}")
    print(f"Mean MSE (AR1):   {folds_df['mse_ar1'].mean():.8e}")
    print(f"Mean MSE (Naive): {folds_df['mse_naive'].mean():.8e}")
    
    # Check baseline shootout
    pdv_beats_ar1_count = (folds_df['oos_r2_pdv'] > folds_df['oos_r2_ar1']).sum()
    pdv_beats_naive_count = (folds_df['oos_r2_pdv'] > folds_df['oos_r2_naive']).sum()
    print(f"\nPDV beats AR(1) in {pdv_beats_ar1_count}/{len(folds_df)} folds ({pdv_beats_ar1_count/len(folds_df)*100:.1f}%)")
    print(f"PDV beats Naive in {pdv_beats_naive_count}/{len(folds_df)} folds ({pdv_beats_naive_count/len(folds_df)*100:.1f}%)")
    
    # Check beta1 sign and stability
    pos_beta1 = (folds_df['beta1'] > 0).sum()
    neg_beta1 = (folds_df['beta1'] < 0).sum()
    print(f"\nBeta1 Analysis:")
    print(f"Positive β1 count: {pos_beta1} ({pos_beta1/len(folds_df)*100:.1f}%)")
    print(f"Negative β1 count: {neg_beta1} ({neg_beta1/len(folds_df)*100:.1f}%)")
    print(f"Beta1 Mean: {folds_df['beta1'].mean():.4e}, Median: {folds_df['beta1'].median():.4e}")
