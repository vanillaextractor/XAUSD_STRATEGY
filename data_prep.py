"""
data_prep.py - Data Preparation & Session Grid for XAUUSD PDV Model

Resamples 1-minute OHLC to 5-minute bars.
Handles session boundaries and weekend/holiday gaps so lag windows don't
silently span a 48-hour gap.
Computes session-aware log returns and per-bar Garman-Klass variance.
"""

import os
import numpy as np
import pandas as pd

RAW_CSV_PATH = "XAUUSD_M1_2019_2025_FULL.csv"
CLEAN_M1_PARQUET = "xauusd_m1_clean.parquet"
CLEAN_M5_PARQUET = "xauusd_m5_clean.parquet"

# Constant for Garman-Klass estimator: 2 * ln(2) - 1
GK_CONST = 2.0 * np.log(2.0) - 1.0  # ~0.38629436


def load_and_clean_m1(csv_path: str = RAW_CSV_PATH) -> pd.DataFrame:
    """
    Loads raw 1-minute CSV, deduplicates timestamps, and ensures strictly monotonic datetime.
    """
    print(f"Loading raw M1 data from {csv_path}...")
    df = pd.read_csv(csv_path, parse_dates=["Datetime"])
    
    # Drop any duplicate timestamps, keeping first occurrence
    initial_len = len(df)
    df = df.drop_duplicates(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)
    dedup_len = len(df)
    print(f"Loaded {dedup_len:,} rows (dropped {initial_len - dedup_len} duplicate timestamps).")
    
    df.set_index("Datetime", inplace=True)
    # Ensure float types
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].astype(np.float64)
        
    return df[["Open", "High", "Low", "Close"]]


def compute_garman_klass_var(df: pd.DataFrame) -> pd.Series:
    """
    Computes Garman-Klass per-bar variance:
    GK_var = 0.5 * (ln(High / Low))^2 - (2*ln(2) - 1) * (ln(Close / Open))^2
    """
    log_hl = np.log(df["High"] / df["Low"])
    log_co = np.log(df["Close"] / df["Open"])
    gk_var = 0.5 * (log_hl ** 2) - GK_CONST * (log_co ** 2)
    # Ensure non-negative due to numerical precision
    return gk_var.clip(lower=0.0)


def resample_to_5min(df_m1: pd.DataFrame) -> pd.DataFrame:
    """
    Resamples 1-minute OHLC to 5-minute bars.
    Computes session-aware log returns and Garman-Klass variance.
    Session boundaries and weekend gaps are explicitly marked, and returns
    across gaps are computed using intrabar log(Close/Open) rather than close-to-close,
    preventing 48-hour gap returns from corrupting kernel-weighted features.
    """
    print("Resampling 1-minute bars to 5-minute bars...")
    agg_dict = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last"
    }
    df_5m = df_m1.resample("5min").agg(agg_dict).dropna()
    print(f"Resampled to {len(df_5m):,} 5-minute bars.")
    
    # Calculate time delta between consecutive bars
    time_diff = df_5m.index.to_series().diff()
    
    # Mark gap bars: any bar where gap > 5 minutes (daily breaks, weekends, holidays)
    is_gap = (time_diff > pd.Timedelta(minutes=5)) | (time_diff.isna())
    gap_minutes = (time_diff.dt.total_seconds() / 60.0).fillna(0.0)
    
    # Compute returns:
    # 1. Standard close-to-close return: log(Close_t / Close_{t-1})
    log_ret_c2c = np.log(df_5m["Close"] / df_5m["Close"].shift(1))
    
    # 2. Intrabar return: log(Close_t / Open_t)
    log_ret_intra = np.log(df_5m["Close"] / df_5m["Open"])
    
    # For gap bars, use intrabar return to prevent weekend jump contamination
    ret = log_ret_c2c.copy()
    ret[is_gap] = log_ret_intra[is_gap]
    # First bar in entire series
    if len(ret) > 0 and pd.isna(ret.iloc[0]):
        ret.iloc[0] = log_ret_intra.iloc[0]
        
    df_5m["return"] = ret
    df_5m["return_c2c_raw"] = log_ret_c2c  # kept for diagnostic comparison
    df_5m["is_gap"] = is_gap
    df_5m["gap_minutes"] = gap_minutes
    
    # Garman-Klass variance per bar
    df_5m["gk_var"] = compute_garman_klass_var(df_5m)
    df_5m["gk_bar"] = np.sqrt(df_5m["gk_var"])
    
    return df_5m


def prepare_and_cache_data(csv_path: str = RAW_CSV_PATH, force_reload: bool = False):
    """
    Main data preparation pipeline with parquet caching.
    """
    if not force_reload and os.path.exists(CLEAN_M5_PARQUET) and os.path.exists(CLEAN_M1_PARQUET):
        print(f"Loading cached datasets from {CLEAN_M5_PARQUET} and {CLEAN_M1_PARQUET}...")
        df_5m = pd.read_parquet(CLEAN_M5_PARQUET)
        df_1m = pd.read_parquet(CLEAN_M1_PARQUET)
        return df_1m, df_5m

    df_1m = load_and_clean_m1(csv_path)
    df_5m = resample_to_5min(df_1m)
    
    print(f"Caching clean M1 data to {CLEAN_M1_PARQUET}...")
    df_1m.to_parquet(CLEAN_M1_PARQUET)
    print(f"Caching clean M5 data to {CLEAN_M5_PARQUET}...")
    df_5m.to_parquet(CLEAN_M5_PARQUET)
    
    return df_1m, df_5m


if __name__ == "__main__":
    df_1m, df_5m = prepare_and_cache_data()
    print("\n--- Data Preparation Summary ---")
    print(f"1-Minute Bars: {len(df_1m):,} from {df_1m.index.min()} to {df_1m.index.max()}")
    print(f"5-Minute Bars: {len(df_5m):,} from {df_5m.index.min()} to {df_5m.index.max()}")
    print(f"Gap Bars Detected: {df_5m['is_gap'].sum():,} ({df_5m['is_gap'].mean()*100:.2f}%)")
    
    # Check max return comparison: raw c2c vs cleaned return
    max_raw = df_5m['return_c2c_raw'].abs().max()
    max_clean = df_5m['return'].abs().max()
    print(f"Max raw C2C absolute return: {max_raw:.5f} ({max_raw*100:.2f}%)")
    print(f"Max cleaned return (gap-aware): {max_clean:.5f} ({max_clean*100:.2f}%)")
    print(f"GK Variance mean: {df_5m['gk_var'].mean():.8e}, std: {df_5m['gk_var'].std():.8e}")
    print("Data prep completed successfully.")
