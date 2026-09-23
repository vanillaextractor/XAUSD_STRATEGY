"""
calibrate_ou_transitions.py - Per-Transition OU Prior Calibration for XAUUSD

Implements the transition-specific calibration logic from train_ou_priors.py:
1. Detects transitions between PDV regimes:
   - 'expansion' -> 'ranging' (volatility exhaustion, key mean-reversion setup)
   - 'ranging' -> 'expansion' (volatility breakout)
2. Calibrates:
   - theta (mean reversion speed via AR(1) on post-transition gap window)
   - t_{1/2} (theoretical half-life of mean-reversion)
   - R_vol (volatility ratio: sigma_dest / sigma_source)
3. Outputs structured JSON matching train_ou_priors.py output schema:
   ou_priors_xausd.json
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import linregress


def calculate_ar1_theta(prices: np.ndarray) -> Optional[float]:
    """
    Calculate theta from AR(1) regression: P_t = alpha + beta * P_{t-1} + eps
    theta = -ln(beta)
    """
    if len(prices) < 5:
        return None
    y = prices[1:]
    x = prices[:-1]
    try:
        slope, _, _, _, _ = linregress(x, y)
        if slope <= 0.0 or slope >= 1.0:
            return None
        theta = -np.log(slope)
        if theta < 0.001 or theta > 5.0:
            return None
        return float(theta)
    except Exception:
        return None


def calculate_volatility(prices: np.ndarray) -> float:
    """Calculate realized return standard deviation."""
    if len(prices) < 3:
        return np.nan
    ret = np.diff(np.log(prices))
    return float(np.std(ret, ddof=1))


class XAUUSDTransitionOUCalibrator:
    """
    Calibrates empirical OU priors for XAUUSD regime transitions.
    """

    def __init__(
        self,
        df_bars: pd.DataFrame,
        regimes: pd.Series,
        lookback_bars: int = 30,
        gap_window: int = 48
    ):
        self.df_bars = df_bars
        self.regimes = regimes
        self.lookback_bars = lookback_bars
        self.gap_window = gap_window
        self.priors: Dict[str, Dict] = {}

    def calibrate(self) -> Dict[str, Dict]:
        regime_arr = self.regimes.values
        prices = self.df_bars["Close"].values
        n = len(prices)

        transitions = [
            ("expansion", "ranging"),
            ("ranging", "expansion")
        ]

        for source, dest in transitions:
            key = f"{source}→{dest}"
            trans_indices = []

            for i in range(1, n):
                if regime_arr[i - 1] == source and regime_arr[i] == dest:
                    trans_indices.append(i)

            print(f"Calibrating {key} (found {len(trans_indices)} transition points)...")

            if len(trans_indices) == 0:
                self.priors[key] = {
                    "theta_median": 0.20,
                    "half_life_median_bars": 3.5,
                    "vol_ratio_median": 0.85,
                    "n_transitions": 0,
                    "fallback_used": True
                }
                continue

            # 1. Pool gap returns for robust AR(1) theta calculation
            pooled_gap_returns = []
            vol_ratios = []
            segment_thetas = []

            for idx in trans_indices:
                # Find destination segment end
                dest_end = idx
                while dest_end < n - 1 and regime_arr[dest_end + 1] == dest:
                    dest_end += 1

                gap_end = min(idx + self.gap_window, dest_end + 1)
                if gap_end - idx >= 4:
                    seg_prices = prices[idx:gap_end]
                    seg_theta = calculate_ar1_theta(seg_prices)
                    if seg_theta is not None:
                        segment_thetas.append(seg_theta)

                    gap_ret = np.diff(np.log(seg_prices))
                    pooled_gap_returns.extend(gap_ret.tolist())

                # Volatility ratio: sigma_dest(first N bars) / sigma_source(last N bars)
                src_start = max(0, idx - self.lookback_bars)
                src_prices = prices[src_start:idx]
                dest_prices = prices[idx:min(n, idx + self.lookback_bars)]

                v_src = calculate_volatility(src_prices)
                v_dest = calculate_volatility(dest_prices)

                if v_src > 1e-8 and v_dest > 1e-8:
                    vr = v_dest / v_src
                    if 0.1 <= vr <= 5.0:
                        vol_ratios.append(vr)

            # Calculate pooled theta
            pooled_theta = None
            if len(pooled_gap_returns) >= 10:
                # AR(1) on pooled returns
                y = np.array(pooled_gap_returns[1:])
                x = np.array(pooled_gap_returns[:-1])
                try:
                    slope, _, _, _, _ = linregress(x, y)
                    if 0.0 < slope < 1.0:
                        pooled_theta = float(-np.log(slope))
                except Exception:
                    pass

            # Fallbacks if fitting failed
            theta_median = float(np.median(segment_thetas)) if segment_thetas else (pooled_theta or 0.20)
            vol_ratio_median = float(np.median(vol_ratios)) if vol_ratios else 0.85
            half_life_median = float(np.log(2.0) / theta_median) if theta_median > 0 else 3.5

            self.priors[key] = {
                "theta_median": round(theta_median, 4),
                "half_life_median_bars": round(half_life_median, 2),
                "vol_ratio_median": round(vol_ratio_median, 4),
                "n_transitions": len(trans_indices),
                "n_valid_thetas": len(segment_thetas),
                "n_valid_vol_ratios": len(vol_ratios),
                "fallback_used": len(segment_thetas) == 0
            }

            print(f"  ✓ {key}: θ_median={theta_median:.4f}, t_1/2={half_life_median:.1f} bars, R_vol={vol_ratio_median:.4f}")

        return self.priors


def run_transition_calibration(output_path: str = "ou_priors_xausd.json") -> Dict:
    """Load data, run calibration, and save JSON."""
    oos_df = pd.read_parquet("walk_forward_oos_predictions.parquet")
    oos_df = oos_df[~oos_df.index.duplicated(keep="first")].copy()
    oos_df["sigma_hat"] = oos_df["sigma_hat"].clip(lower=1e-5)
    df_m5 = pd.read_parquet("xauusd_m5_clean.parquet")

    from regime_signals import generate_feature_dataframe
    feat = generate_feature_dataframe(
        df_m5.loc[oos_df.index],
        sigma_hat=oos_df["sigma_hat"],
        beta1_sign=oos_df["beta1_sign"],
        r1=oos_df["r1"],
        regime_method="median"
    )

    calibrator = XAUUSDTransitionOUCalibrator(
        df_bars=df_m5.loc[oos_df.index],
        regimes=feat["regime"],
        lookback_bars=30,
        gap_window=48
    )

    priors = calibrator.calibrate()

    with open(output_path, "w") as f:
        json.dump(priors, f, indent=2)

    print(f"\nSaved calibrated OU priors to {output_path}")
    return priors


if __name__ == "__main__":
    run_transition_calibration()
