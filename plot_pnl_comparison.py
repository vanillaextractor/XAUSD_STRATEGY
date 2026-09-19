"""
plot_pnl_comparison.py - Generate PnL & Equity Curve Diagnostic Plot
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from backtest_engine import compute_edge_spread, run_intraday_backtest

ARTIFACT_DIR = "/Users/pulkitchauhan/.gemini/antigravity-ide/brain/2730d759-fd1c-469d-96a1-a36f3e8dab3c"

df_features = pd.read_parquet("xauusd_strategy_features_standard.parquet")
df_m1 = pd.read_parquet("xauusd_m1_clean.parquet")
df_m5 = pd.read_parquet("xauusd_m5_clean.parquet")

edge_spread = compute_edge_spread(df_m5, rolling_window=288)

dev_m1 = df_m1.loc[df_features.index.min():df_features.index.max() + pd.Timedelta(hours=4)]

print("Simulating trades for equity curve...")
bt_edge = run_intraday_backtest(df_features, dev_m1, edge_spread, use_edge_cost=True)
bt_flat = run_intraday_backtest(df_features, dev_m1, edge_spread, flat_spread_usd=0.20, use_edge_cost=False)

trades_edge = bt_edge["trades"]
trades_flat = bt_flat["trades"]

fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
if len(trades_edge) > 0 and len(trades_flat) > 0:
    trades_edge["cum_pnl"] = trades_edge["pnl_usd"].cumsum()
    trades_flat["cum_pnl"] = trades_flat["pnl_usd"].cumsum()
    
    ax.plot(pd.to_datetime(trades_edge["exit_time"]), trades_edge["cum_pnl"], label="EDGE Time-Varying Spread Cost", color="#2563eb", lw=1.8)
    ax.plot(pd.to_datetime(trades_flat["exit_time"]), trades_flat["cum_pnl"], label="Flat Spread Cost ($0.20/oz)", color="#ef4444", lw=1.8, linestyle="--")
    
    ax.set_title("Strategy Cumulative PnL: Time-Varying EDGE Spread vs. Flat Spread (2019 - 2024)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative PnL ($ / oz)")
    ax.legend(loc="lower left", frameon=True)
    
fig.savefig(os.path.join(ARTIFACT_DIR, "pnl_comparison.png"))
plt.close(fig)
print("Saved pnl_comparison.png to artifact directory.")
