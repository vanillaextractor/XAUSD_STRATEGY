"""
test_sensitivity.py - Sensitivity Analysis for Horizon h and Lookback N

Tests:
- Horizon h in [6 (30m), 24 (2h), 78 (6.5h)]
- Lookback N in [100 (8h), 500 (4d), 2000 (16d)]
Evaluates where the model's explanatory power (in-sample and out-of-sample R^2) peaks.
"""

import numpy as np
import pandas as pd
from pdv_features import compute_forward_rv, compute_trailing_rv, compute_r1_r2
from calibrate_pdv import optimize_kernel_params, fit_inner_ols


def run_sensitivity_analysis():
    print("Loading 5-minute data...")
    df_5m = pd.read_parquet("xauusd_m5_clean.parquet")
    
    # Test on representative 2023-2024 period (e.g., 2023 train, early 2024 test)
    train_slice = df_5m.loc["2023-01-01":"2023-07-01"].copy()
    test_slice = df_5m.loc["2023-07-02":"2023-08-01"].copy()
    
    returns_train = train_slice["return"].values
    returns_test_raw = test_slice["return"].values
    
    results = []
    
    # 1. Horizon Sensitivity (fixed N=500)
    print("\n--- Horizon Sensitivity (h = 6, 24, 78 bars; N = 500) ---")
    for h in [6, 24, 78]:
        # Compute forward targets
        y_train = compute_forward_rv(train_slice["gk_var"], h=h).values
        y_test = compute_forward_rv(test_slice["gk_var"], h=h).values
        y_trail_test = compute_trailing_rv(test_slice["gk_var"], h=h).values
        
        # Valid mask
        valid_train = ~np.isnan(y_train)
        valid_test = ~np.isnan(y_test) & ~np.isnan(y_trail_test)
        
        opt_params, in_r2, betas = optimize_kernel_params(
            returns_train[valid_train], y_train[valid_train], N=500
        )
        
        # OOS evaluation
        a1, d1, a2, d2 = opt_params
        returns_full_test = np.concatenate((returns_train[-500:], returns_test_raw))
        r1_full, r2_full = compute_r1_r2(returns_full_test, a1, d1, a2, d2, N=500)
        r1_test = r1_full[500:][valid_test]
        r2_test = r2_full[500:][valid_test]
        y_test_clean = y_test[valid_test]
        
        X_test = np.column_stack((np.ones(len(y_test_clean)), r1_test, np.sqrt(r2_test)))
        y_pred = X_test @ betas
        
        ss_tot = np.sum((y_test_clean - np.mean(y_test_clean)) ** 2)
        ss_res = np.sum((y_test_clean - y_pred) ** 2)
        oos_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        # AR(1) OOS
        y_trail_train = compute_trailing_rv(train_slice["gk_var"], h=h).values[valid_train]
        X_ar1_train = np.column_stack((np.ones(len(y_trail_train) - 500), y_trail_train[500:]))
        betas_ar1, _ = fit_inner_ols(X_ar1_train, y_train[valid_train][500:])
        
        y_trail_test_clean = y_trail_test[valid_test]
        X_ar1_test = np.column_stack((np.ones(len(y_trail_test_clean)), y_trail_test_clean))
        y_pred_ar1 = X_ar1_test @ betas_ar1
        oos_r2_ar1 = 1.0 - np.sum((y_test_clean - y_pred_ar1) ** 2) / ss_tot
        
        print(f"Horizon h={h:02d} ({h*5:3d} mins) | In-Sample R²: {in_r2:.4f} | OOS R² (PDV): {oos_r2:.4f} | OOS R² (AR1): {oos_r2_ar1:.4f} | β1: {betas[1]:+.3e}")
        results.append({"type": "horizon", "h": h, "N": 500, "in_r2": in_r2, "oos_r2": oos_r2, "oos_r2_ar1": oos_r2_ar1})

    # 2. Lookback Sensitivity (fixed h=24)
    print("\n--- Lookback Sensitivity (N = 100, 500, 2000 bars; h = 24) ---")
    y_train_24 = compute_forward_rv(train_slice["gk_var"], h=24).values
    y_test_24 = compute_forward_rv(test_slice["gk_var"], h=24).values
    valid_train = ~np.isnan(y_train_24)
    valid_test = ~np.isnan(y_test_24)
    
    for N in [100, 500, 2000]:
        opt_params, in_r2, betas = optimize_kernel_params(
            returns_train[valid_train], y_train_24[valid_train], N=N
        )
        
        a1, d1, a2, d2 = opt_params
        returns_full_test = np.concatenate((returns_train[-N:], returns_test_raw))
        r1_full, r2_full = compute_r1_r2(returns_full_test, a1, d1, a2, d2, N=N)
        r1_test = r1_full[N:][valid_test]
        r2_test = r2_full[N:][valid_test]
        y_test_clean = y_test_24[valid_test]
        
        X_test = np.column_stack((np.ones(len(y_test_clean)), r1_test, np.sqrt(r2_test)))
        y_pred = X_test @ betas
        
        ss_tot = np.sum((y_test_clean - np.mean(y_test_clean)) ** 2)
        ss_res = np.sum((y_test_clean - y_pred) ** 2)
        oos_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        print(f"Lookback N={N:4d} bars | In-Sample R²: {in_r2:.4f} | OOS R² (PDV): {oos_r2:.4f} | β1: {betas[1]:+.3e}")
        results.append({"type": "lookback", "h": 24, "N": N, "in_r2": in_r2, "oos_r2": oos_r2})

    df_res = pd.DataFrame(results)
    df_res.to_parquet("sensitivity_results.parquet")
    print("\nSensitivity analysis completed and saved to 'sensitivity_results.parquet'.")


if __name__ == "__main__":
    run_sensitivity_analysis()
