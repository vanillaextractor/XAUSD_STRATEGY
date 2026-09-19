"""
plot_pnl_comparison.py - Generate PnL & Equity Curve Comparison Diagnostic Plot
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from regime_signals import generate_feature_dataframe
from backtest_engine import compute_edge_spread, run_intraday_backtest

ARTIFACT_DIR = "/Users/pulkitchauhan/.gemini/antigravity-ide/brain/2730d759-fd1c-469d-96a1-a36f3e8dab3c"

oos_df = pd.read_parquet("walk_forward_oos_predictions.parquet")
oos_df = oos_df[~oos_df.index.duplicated(keep="first")].copy()
oos_df["sigma_hat"] = oos_df["sigma_hat"].clip(lower=1e-5)
df_m1 = pd.read_parquet("xauusd_m1_clean.parquet")
df_m5 = pd.read_parquet("xauusd_m5_clean.parquet")

edge_spread = compute_edge_spread(df_m5, rolling_window=288)

feat_median = generate_feature_dataframe(
    df_m5.loc[oos_df.index],
    sigma_hat=oos_df["sigma_hat"],
    beta1_sign=oos_df["beta1_sign"],
    r1=oos_df["r1"],
    h=24, z_window=30, z_threshold=2.0, k1_stop=1.5, k2_target=1.0,
    regime_method="median"
)

feat_tilt = generate_feature_dataframe(
    df_m5.loc[oos_df.index],
    sigma_hat=oos_df["sigma_hat"],
    beta1_sign=oos_df["beta1_sign"],
    r1=oos_df["r1"],
    h=24, z_window=30, z_threshold=2.0, k1_stop=1.5, k2_target=1.0,
    regime_method="median",
    apply_beta1_tilt=True
)

dev_m1 = df_m1.loc[feat_median.index.min():feat_median.index.max() + pd.Timedelta(hours=4)]

print("Simulating comparative equity curves...")
bt_z0 = run_intraday_backtest(feat_median, dev_m1, edge_spread, use_edge_cost=True, cooldown_bars=24, z_exit_threshold=0.0)
bt_pure_pdv = run_intraday_backtest(feat_median, dev_m1, edge_spread, use_edge_cost=True, cooldown_bars=24, z_exit_threshold=None)
bt_tilt_pdv = run_intraday_backtest(feat_tilt, dev_m1, edge_spread, use_edge_cost=True, cooldown_bars=24, z_exit_threshold=None)

fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

t_z0 = bt_z0["trades"]
t_pure = bt_pure_pdv["trades"]
t_tilt = bt_tilt_pdv["trades"]

if len(t_z0) > 0:
    t_z0["cum_pnl"] = t_z0["pnl_usd"].cumsum()
    ax.plot(pd.to_datetime(t_z0["exit_time"]), t_z0["cum_pnl"], label=f"Bug-Fixed Baseline (z=0 exit): -${abs(t_z0['cum_pnl'].iloc[-1]):,.0f}", color="#ef4444", lw=1.8, linestyle="--")

if len(t_pure) > 0:
    t_pure["cum_pnl"] = t_pure["pnl_usd"].cumsum()
    ax.plot(pd.to_datetime(t_pure["exit_time"]), t_pure["cum_pnl"], label=f"Pure PDV Stops/Targets (z_exit disabled): -${abs(t_pure['cum_pnl'].iloc[-1]):,.0f}", color="#f59e0b", lw=2.0)

if len(t_tilt) > 0:
    t_tilt["cum_pnl"] = t_tilt["pnl_usd"].cumsum()
    ax.plot(pd.to_datetime(t_tilt["exit_time"]), t_tilt["cum_pnl"], label=f"Pure PDV + Directional Beta1 Tilt: -${abs(t_tilt['cum_pnl'].iloc[-1]):,.0f}", color="#10b981", lw=2.2)

ax.set_title("Strategy Cumulative PnL Comparison Across Improvements (2019 - 2024 Walk-Forward)", fontsize=13, fontweight="bold")
ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Cumulative PnL ($ / oz)", fontsize=11)
ax.axhline(0, color="gray", lw=0.8, linestyle=":")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", frameon=True)

fig.tight_layout()
fig.savefig(os.path.join(ARTIFACT_DIR, "pnl_comparison.png"))
fig.savefig("images/pnl_comparison.png")
plt.close(fig)
print("Saved updated pnl_comparison.png to artifact and repo images directories.")
