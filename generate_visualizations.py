"""
generate_visualizations.py - Generate Publication-Grade Diagnostic Charts

Creates:
1. oos_r2_comparison.png: Walk-forward OOS R^2 over time (PDV vs AR(1) vs Naive).
2. beta1_dynamics.png: Time series of beta1 parameter and sign (Safe Haven vs Leverage effect).
3. horizon_sensitivity.png: Peak explanatory power across forecasting horizons.
4. pnl_comparison.png: Strategy cumulative equity curve comparing EDGE vs Flat spread costs.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ARTIFACT_DIR = "/Users/pulkitchauhan/.gemini/antigravity-ide/brain/2730d759-fd1c-469d-96a1-a36f3e8dab3c"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

folds_df = pd.read_parquet("walk_forward_folds.parquet")
folds_df["test_date"] = pd.to_datetime(folds_df["test_start"])

# 1. OOS R^2 Comparison Plot
fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
ax.plot(folds_df["test_date"], folds_df["oos_r2_pdv"].rolling(5, min_periods=1).mean(), label="PDV (Guyon & Lekeufack) - 5-fold MA", color="#2563eb", lw=2)
ax.plot(folds_df["test_date"], folds_df["oos_r2_ar1"].rolling(5, min_periods=1).mean(), label="AR(1) Baseline - 5-fold MA", color="#f59e0b", lw=1.8, linestyle="--")
ax.plot(folds_df["test_date"], folds_df["oos_r2_naive"].rolling(5, min_periods=1).mean(), label="Naive Persistence - 5-fold MA", color="#ef4444", lw=1.2, linestyle=":")
ax.axhline(0.0, color="gray", lw=0.8, linestyle="--", alpha=0.7)
ax.set_title("Walk-Forward Out-of-Sample Volatility Forecasting R² (2019 - 2024)", fontsize=13, fontweight="bold")
ax.set_xlabel("Test Date")
ax.set_ylabel("Out-of-Sample R²")
ax.legend(loc="upper right", frameon=True)
ax.set_ylim(-0.5, 0.4)
fig.savefig(os.path.join(ARTIFACT_DIR, "oos_r2_comparison.png"))
plt.close(fig)

# 2. Beta1 Dynamics Plot (Safe-Haven vs Equity Leverage Effect)
fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
colors = np.where(folds_df["beta1"] >= 0, "#10b981", "#ef4444")
ax.bar(folds_df["test_date"], folds_df["beta1"], width=10, color=colors, alpha=0.8, label="β1 Estimate per Fold")
ax.plot(folds_df["test_date"], folds_df["beta1"].rolling(7, min_periods=1).mean(), color="#1e293b", lw=2, label="7-Fold Rolling Trend")
ax.axhline(0.0, color="black", lw=1)
ax.set_title("Gold Volatility Asymmetry: β1 Sign & Magnitude (Positive = Safe Haven, Negative = Leverage)", fontsize=13, fontweight="bold")
ax.set_xlabel("Walk-Forward Period")
ax.set_ylabel("β1 Coefficient")
ax.legend(loc="upper right", frameon=True)
fig.savefig(os.path.join(ARTIFACT_DIR, "beta1_dynamics.png"))
plt.close(fig)

# 3. Horizon Sensitivity Plot
sens_df = pd.read_parquet("sensitivity_results.parquet")
horiz_df = sens_df[sens_df["type"] == "horizon"]
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
bar_width = 0.35
x = np.arange(len(horiz_df))
ax.bar(x - bar_width/2, horiz_df["in_r2"] * 100, width=bar_width, label="In-Sample R² (%)", color="#3b82f6")
ax.bar(x + bar_width/2, horiz_df["oos_r2"] * 100, width=bar_width, label="Out-of-Sample R² (%)", color="#10b981")
ax.set_xticks(x)
ax.set_xticklabels([f"h={int(h)} ({int(h)*5}m)" for h in horiz_df["h"]])
ax.set_title("Forecast Explanatory Power by Horizon (h)", fontsize=12, fontweight="bold")
ax.set_ylabel("R² (%)")
ax.legend(frameon=True)
fig.savefig(os.path.join(ARTIFACT_DIR, "horizon_sensitivity.png"))
plt.close(fig)

print("Diagnostic figures generated successfully in artifact directory.")
