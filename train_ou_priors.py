#!/usr/bin/env python3
"""
Train OU Priors Calibration Script - PER-TRANSITION VERSION (NO LOOKAHEAD)
===========================================================================

Empirically calibrates Ornstein-Uhlenbeck (OU) process parameters from training data
for EACH (model, transition) combination.

CRITICAL: NO RUNTIME LOOKAHEAD BIAS
- Theta: Calculated from DESTINATION state history AFTER transition (train-only lookahead)
- Vol Ratio: Historical volatility change ratio (descriptive calibration)
- NO future regime scanning during backtesting
- NO segment length filtering

For each transition type (0→1, 0→2, 1→0, 1→2, 2→0, 2→1):
- Theta (θ): Mean reversion speed from AR(1) regression on DESTINATION state AFTER transition
- Vol Ratio (R_vol): σ(destination first N bars) / σ(source last N bars)

Output structure:
{
    "model_name": {
        "0→1": {"theta_median": 0.35, "vol_ratio_median": 0.85, ...},
        "0→2": {"theta_median": 0.52, "vol_ratio_median": 1.25, ...},
        ...
    }
}

Author: OU Grid Calibration System (No Lookahead)
Date: 2026-01-22
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')


# All possible transitions for 3-state HMM
ALL_TRANSITIONS = ['0→1', '0→2', '1→0', '1→2', '2→0', '2→1']

# Global fallback values
THETA_GLOBAL_FALLBACK = 0.35
VOL_RATIO_GLOBAL_FALLBACK = 0.70


class TransitionOUCalibrator:
    """
    Calibrates OU process parameters for EACH transition type.

    For transition source→dest:
    - θ: Calculated from AR(1) regression on destination state segments
    - R_vol: σ(destination) / σ(source) around the transition point
    """

    def __init__(
        self,
        model: str,
        regime_probs_path: Path,
        lookback_bars: int = 30,
        strategies_csv_path: Optional[Path] = None
    ):
        """
        Initialize per-transition OU calibrator.

        Args:
            model: Model name (e.g., 'dkF_vw27_nr3')
            regime_probs_path: Path to regime probabilities parquet (train split)
            lookback_bars: Bars to look back for source state volatility
            strategies_csv_path: Path to CSV with wait/hold values (optional, defaults to marginal params CSV)
        """
        self.model = model
        self.regime_probs_path = regime_probs_path
        self.lookback_bars = lookback_bars
        self.strategies_csv_path = strategies_csv_path or Path(
            'results/parameter_optimization/marginal_optimal_parameters_all_models.csv'
        )

        # Data
        self.train_data = None
        self.n_regimes = None

        # Results per transition
        self.transition_results: Dict[str, Dict] = {}

        # Load wait values for gap window calibration
        self.transition_wait_map = self._load_wait_values()

    def _load_wait_values(self) -> Dict[str, int]:
        """
        Load maximum wait values for each transition from CSV.

        For each transition, uses the maximum wait across LONG/SHORT positions
        since both share the same calibrated theta.

        Returns:
            Dict mapping transition (e.g., '0→1') to maximum wait value
        """
        try:
            df = pd.read_csv(self.strategies_csv_path)
            model_df = df[df['Model'] == self.model]

            if len(model_df) == 0:
                print(f"  ⚠️  Model {self.model} not found in CSV, using default gap window of 5")
                return {}

            # Get max wait for each transition (across LONG/SHORT)
            wait_map = {}
            for transition in model_df['Transition'].unique():
                max_wait = model_df[model_df['Transition'] == transition]['Wait'].max()
                wait_map[transition] = int(max_wait)

            return wait_map

        except FileNotFoundError:
            print(f"  ⚠️  CSV file not found: {self.strategies_csv_path}, using default gap window of 5")
            return {}
        except Exception as e:
            print(f"  ⚠️  Error loading CSV: {e}, using default gap window of 5")
            return {}

    def load_data(self):
        """Load regime probabilities with price data."""
        print(f"\n{'='*80}")
        print(f"PER-TRANSITION OU CALIBRATION: {self.model}")
        print(f"{'='*80}\n")

        print("[1/4] Loading data...")

        if not Path(self.regime_probs_path).exists():
            raise FileNotFoundError(f"Regime probs not found: {self.regime_probs_path}")

        self.train_data = pd.read_parquet(self.regime_probs_path)
        print(f"  ✓ Loaded: {len(self.train_data)} rows")

        # Validate required columns
        if 'close' not in self.train_data.columns:
            raise ValueError("'close' column not found")

        # Calculate log prices
        self.train_data['log_close'] = np.log(self.train_data['close'])

        # Detect regime (argmax of probabilities)
        prob_cols = [c for c in self.train_data.columns
                     if c.startswith('regime_') and c.endswith('_prob')]

        if not prob_cols:
            raise ValueError("No regime probability columns found")

        self.train_data['regime'] = self.train_data[prob_cols].values.argmax(axis=1)
        self.n_regimes = self.train_data['regime'].nunique()

        # Find all transition points
        self.train_data['prev_regime'] = self.train_data['regime'].shift(1)
        self.train_data['is_transition'] = (
            self.train_data['regime'] != self.train_data['prev_regime']
        )

        n_transitions = self.train_data['is_transition'].sum()
        print(f"  ✓ Regimes: {self.n_regimes}")
        print(f"  ✓ Transition points: {n_transitions}")

    def find_transition_segments(self, source: int, dest: int) -> List[Tuple[int,]]:
        """
        Find all transition points where source→dest occurred.

        CRITICAL: NO LOOKAHEAD - Only records transition index.
        Does NOT scan future regimes to find segment boundaries.
        Does NOT filter by segment length.

        Returns:
            List of (transition_idx,) single-element tuples
            - transition_idx: Bar where source→dest transition happened
        """
        df = self.train_data
        regimes = df['regime'].values

        segments = []

        # Find all transition points
        for i in range(1, len(df)):
            if regimes[i-1] == source and regimes[i] == dest:
                # Record transition point ONLY
                # Do NOT scan forward to find seg_end
                # Do NOT filter by segment length
                segments.append((i,))  # Single-element tuple

        return segments

    def calculate_theta(self, log_prices: np.ndarray) -> Optional[float]:
        """
        Calculate theta from AR(1) regression on log prices.

        AR(1): P_t = α + β*P_{t-1} + ε
        Theta: θ = -ln(β)

        Returns:
            Theta value, or None if invalid
        """
        if len(log_prices) < 5:
            return None

        y = log_prices[1:]   # P_t
        x = log_prices[:-1]  # P_{t-1}

        try:
            slope, _, _, _, _ = linregress(x, y)

            # Beta must be in (0, 1) for mean reversion
            if slope <= 0 or slope >= 1:
                return None

            theta = -np.log(slope)

            # Sanity check
            if theta < 0.01 or theta > 2.0:
                return None

            return theta

        except Exception:
            return None

    def calculate_volatility(self, log_prices: np.ndarray) -> float:
        """Calculate realized volatility from log returns."""
        if len(log_prices) < 2:
            return np.nan

        log_returns = np.diff(log_prices)
        return np.std(log_returns, ddof=1)

    def calibrate_transition(self, source: int, dest: int) -> Dict:
        """
        Calibrate OU parameters for a specific transition.

        UPDATED APPROACH: Forward-looking for theta (train-only lookahead).
        - Theta: Calculated from DESTINATION state AFTER transition (where gap trading occurs)
        - Vol Ratio: σ(destination first N bars) / σ(source last N bars)

        Args:
            source: Source regime (0, 1, or 2)
            dest: Destination regime (0, 1, or 2)

        Returns:
            Dict with calibration results for this transition
        """
        transition_key = f"{source}→{dest}"

        # Find all source→dest transition points
        segments = self.find_transition_segments(source, dest)

        if len(segments) == 0:
            return {
                'transition': transition_key,
                'theta_median': THETA_GLOBAL_FALLBACK,
                'vol_ratio_median': VOL_RATIO_GLOBAL_FALLBACK,
                'n_segments': 0,
                'fallback_used': True,
                'error': f'No valid {transition_key} segments found'
            }

        log_close = self.train_data['log_close'].values
        regimes = self.train_data['regime'].values

        theta_values = []
        vol_ratio_values = []

        # === POOLED GAP-PERIOD THETA CALCULATION ===
        #
        # CRITICAL DESIGN DECISION:
        # 1. Gap trading occurs at bars [trans_idx, trans_idx + gap_hold]
        # 2. Theta must characterize mean-reversion during the GAP PERIOD, not entire regime
        # 3. Each transition uses its specific gap window from CSV (wait parameter)
        # 4. Pool gap periods from ALL instances to get sufficient data for AR(1)
        #
        # TRAIN-ONLY LOOKAHEAD:
        # This uses future data relative to trans_idx, but ONLY during calibration.
        # Backtesting never sees this - it uses pre-calibrated priors from JSON.

        # Get gap window for this transition (from CSV wait values)
        gap_window = self.transition_wait_map.get(transition_key, 5)  # Default to 5 if not found
        print(f"  Transition {transition_key}: Using gap window = {gap_window} bars (from CSV)")

        # === TRUE POOLING with Log Returns ===
        # Pool gap periods from ALL transition instances
        # SOLUTION: Pool LOG RETURNS (not prices) - naturally stationary and poolable
        # AR(1) on returns estimates the same mean-reversion speed
        pooled_gap_returns = []
        valid_segments = 0
        skipped_short = 0

        for (trans_idx,) in segments:
            # Find destination state end (for boundary check)
            dest_end = trans_idx
            while dest_end < len(regimes) - 1 and regimes[dest_end + 1] == dest:
                dest_end += 1

            # Extract ONLY gap period (first gap_window bars after transition)
            gap_end = min(trans_idx + gap_window, dest_end + 1)
            gap_length = gap_end - trans_idx

            # Skip if destination state too short for this gap window
            # Need at least 2 bars to compute 1 return
            if gap_length < 2:
                skipped_short += 1
                continue

            # Extract gap window prices
            gap_prices = log_close[trans_idx:gap_end]

            # Convert to log returns (first differences)
            # Returns are stationary and can be pooled across segments
            gap_returns = np.diff(gap_prices)

            # Add to pooled returns
            pooled_gap_returns.extend(gap_returns.tolist())
            valid_segments += 1

        print(f"    Valid segments: {valid_segments}/{len(segments)}, Skipped (too short): {skipped_short}")
        print(f"    Pooled returns: {len(pooled_gap_returns)}")

        # Calculate ONE theta from pooled returns (single calculation)
        # For returns, AR(1) is: r_t = α + β*r_{t-1} + ε
        # Mean-reversion speed is the same: θ = -ln(β)
        if len(pooled_gap_returns) >= 5:
            theta = self.calculate_theta(np.array(pooled_gap_returns))
            if theta is not None:
                theta_values = [theta]  # Single value from pooled data
                print(f"    ✓ Pooled theta: {theta:.4f}")
                print(f"    ✓ Avg returns per segment: {len(pooled_gap_returns) / valid_segments:.1f}")
            else:
                print(f"    ✗ Theta calculation failed (invalid AR(1) result)")
        else:
            print(f"    ✗ Insufficient pooled returns: {len(pooled_gap_returns)} < 5")

        # Re-iterate through segments for vol_ratio calculation
        for (trans_idx,) in segments:

            # === VOL RATIO: σ(dest first N bars) / σ(source last N bars) ===

            # Find source state segment for volatility calculation
            # Scan BACKWARD from trans_idx-1 to find source segment boundaries
            source_end = trans_idx - 1  # Last bar of source state
            source_start = source_end

            # Scan backward to find where source state started
            while source_start > 0 and regimes[source_start - 1] == source:
                source_start -= 1

            # Source volatility: Last lookback_bars of source state
            source_vol_start = max(source_start, source_end - self.lookback_bars + 1)
            source_vol_prices = log_close[source_vol_start:source_end + 1]
            source_vol = self.calculate_volatility(source_vol_prices)

            # Destination volatility: First lookback_bars AFTER transition
            # CAREFUL: This uses post-transition data but it's for vol_ratio calibration
            # Vol_ratio is a descriptive stat, not predictive. We need it to understand
            # typical volatility change magnitude for this transition type.
            dest_vol_end = min(len(log_close), trans_idx + self.lookback_bars)
            dest_vol_prices = log_close[trans_idx:dest_vol_end]
            dest_vol = self.calculate_volatility(dest_vol_prices)

            if source_vol > 1e-10 and dest_vol > 1e-10:
                vol_ratio = dest_vol / source_vol

                # Sanity check
                if 0.1 < vol_ratio < 5.0:
                    vol_ratio_values.append(vol_ratio)

        # Aggregate results
        if len(theta_values) == 0:
            theta_median = THETA_GLOBAL_FALLBACK
            theta_fallback = True
        else:
            theta_median = float(np.median(theta_values))
            theta_fallback = False

        if len(vol_ratio_values) == 0:
            vol_ratio_median = VOL_RATIO_GLOBAL_FALLBACK
            vol_ratio_fallback = True
        else:
            vol_ratio_median = float(np.median(vol_ratio_values))
            vol_ratio_fallback = False

        return {
            'transition': transition_key,
            'theta_median': theta_median,
            'theta_mean': float(np.mean(theta_values)) if theta_values else THETA_GLOBAL_FALLBACK,
            'theta_std': float(np.std(theta_values)) if len(theta_values) > 1 else 0.0,
            'vol_ratio_median': vol_ratio_median,
            'vol_ratio_mean': float(np.mean(vol_ratio_values)) if vol_ratio_values else VOL_RATIO_GLOBAL_FALLBACK,
            'vol_ratio_std': float(np.std(vol_ratio_values)) if len(vol_ratio_values) > 1 else 0.0,
            'n_segments': len(segments),
            'n_theta_valid': len(theta_values),
            'n_vol_ratio_valid': len(vol_ratio_values),
            'theta_fallback_used': theta_fallback,
            'vol_ratio_fallback_used': vol_ratio_fallback,
            'fallback_used': theta_fallback or vol_ratio_fallback
        }

    def calibrate_all_transitions(self):
        """Calibrate OU parameters for all transition types."""
        print(f"\n[2/4] Calibrating all transitions...")

        # Determine which transitions are possible (based on n_regimes)
        if self.n_regimes == 2:
            transitions = [(0, 1), (1, 0)]
        else:  # 3 regimes
            transitions = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]

        for source, dest in transitions:
            trans_key = f"{source}→{dest}"
            result = self.calibrate_transition(source, dest)
            self.transition_results[trans_key] = result

            status = "✓" if not result['fallback_used'] else "⚠"
            print(f"  {status} {trans_key}: θ={result['theta_median']:.4f}, "
                  f"R_vol={result['vol_ratio_median']:.4f} "
                  f"(n={result['n_segments']} segments)")

    def get_results(self) -> Dict[str, Dict]:
        """Return calibration results for all transitions."""
        return {
            'model': self.model,
            'transitions': self.transition_results
        }

    def print_summary(self):
        """Print calibration summary."""
        print(f"\n[3/4] Calibration Summary for {self.model}")
        print("-" * 70)
        print(f"{'Transition':<12} {'θ':>8} {'R_vol':>8} {'Segments':>10} {'Status':>10}")
        print("-" * 70)

        for trans_key, result in self.transition_results.items():
            status = "OK" if not result['fallback_used'] else "FALLBACK"
            print(f"{trans_key:<12} {result['theta_median']:>8.4f} "
                  f"{result['vol_ratio_median']:>8.4f} "
                  f"{result['n_segments']:>10} {status:>10}")

        print("-" * 70)


def batch_calibrate_all_models(
    data_dir: Path = Path('/Users/snowlerr/Desktop/Orion_Model_Backtest/data/regimes_sweep'),
    output_dir: Path = Path('production_backtest/ou_calibration'),
    lookback_bars: int = 30
) -> Dict[str, Dict]:
    """
    Calibrate per-transition OU priors for all models.

    Args:
        data_dir: Directory containing regime_probs parquet files
        output_dir: Directory to save calibration results
        lookback_bars: Bars to look back for volatility calculation

    Returns:
        Master dictionary: {model: {transition: calibration_results}}
    """
    print(f"\n{'='*80}")
    print("BATCH PER-TRANSITION OU CALIBRATION")
    print(f"{'='*80}\n")

    # Discover all training files
    train_files = list(data_dir.glob('regime_probs_max_dc_train_*.parquet'))

    if not train_files:
        raise FileNotFoundError(f"No training files found in {data_dir}")

    # Extract model names
    models = []
    for f in train_files:
        model = f.stem.replace('regime_probs_max_dc_train_', '')
        models.append(model)

    models = sorted(models)

    print(f"Found {len(models)} models to calibrate")
    print()

    # Calibrate each model
    all_priors = {}

    for i, model in enumerate(models, 1):
        print(f"\n{'─'*80}")
        print(f"[{i}/{len(models)}] {model}")
        print(f"{'─'*80}")

        regime_probs_path = data_dir / f'regime_probs_max_dc_train_{model}.parquet'

        try:
            calibrator = TransitionOUCalibrator(
                model=model,
                regime_probs_path=regime_probs_path,
                lookback_bars=lookback_bars
            )

            calibrator.load_data()
            calibrator.calibrate_all_transitions()
            calibrator.print_summary()

            # Store results
            all_priors[model] = calibrator.transition_results

        except Exception as e:
            print(f"  ✗ FAILED: {str(e)}")
            # Create fallback entries for all transitions
            all_priors[model] = {}
            for trans in ALL_TRANSITIONS:
                all_priors[model][trans] = {
                    'transition': trans,
                    'theta_median': THETA_GLOBAL_FALLBACK,
                    'vol_ratio_median': VOL_RATIO_GLOBAL_FALLBACK,
                    'n_segments': 0,
                    'fallback_used': True,
                    'error': str(e)
                }

    # Save master file
    print(f"\n{'='*80}")
    print("[4/4] SAVING MASTER CALIBRATION FILE")
    print(f"{'='*80}\n")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    master_file = output_dir / 'ou_priors_per_transition.json'
    with open(master_file, 'w') as f:
        json.dump(all_priors, f, indent=2)

    print(f"  ✓ Saved: {master_file}")

    # Summary statistics
    print(f"\n{'='*80}")
    print("CALIBRATION COMPLETE")
    print(f"{'='*80}\n")

    total_transitions = 0
    fallback_transitions = 0

    for model, trans_dict in all_priors.items():
        for trans_key, result in trans_dict.items():
            total_transitions += 1
            if result.get('fallback_used', False):
                fallback_transitions += 1

    print(f"Total models: {len(models)}")
    print(f"Total transitions calibrated: {total_transitions}")
    print(f"Transitions using fallbacks: {fallback_transitions} ({100*fallback_transitions/total_transitions:.1f}%)")

    # Show worst models (most fallbacks)
    model_fallbacks = {}
    for model, trans_dict in all_priors.items():
        n_fallback = sum(1 for r in trans_dict.values() if r.get('fallback_used', False))
        if n_fallback > 0:
            model_fallbacks[model] = n_fallback

    if model_fallbacks:
        print(f"\nModels with fallbacks:")
        for model, count in sorted(model_fallbacks.items(), key=lambda x: -x[1]):
            print(f"  {model}: {count} transitions")

    print(f"\n{'='*80}\n")

    return all_priors


def main():
    """Main calibration pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Calibrate per-transition OU parameters from training data'
    )
    parser.add_argument('--batch', action='store_true',
                       help='Calibrate all models (default)')
    parser.add_argument('--model', type=str, default=None,
                       help='Single model to calibrate')
    parser.add_argument('--data-dir', type=str,
                       default='/Users/snowlerr/Desktop/Orion_Model_Backtest/data/regimes_sweep',
                       help='Directory containing regime_probs files')
    parser.add_argument('--output', type=str,
                       default='production_backtest/ou_calibration',
                       help='Output directory for calibration results')
    parser.add_argument('--lookback', type=int, default=30,
                       help='Lookback bars for volatility (default: 30)')

    args = parser.parse_args()

    if args.model:
        # Single model calibration
        regime_probs_path = Path(args.data_dir) / f'regime_probs_max_dc_train_{args.model}.parquet'

        calibrator = TransitionOUCalibrator(
            model=args.model,
            regime_probs_path=regime_probs_path,
            lookback_bars=args.lookback
        )

        calibrator.load_data()
        calibrator.calibrate_all_transitions()
        calibrator.print_summary()

        # Save individual model results
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f'ou_priors_{args.model}.json'
        with open(output_file, 'w') as f:
            json.dump(calibrator.get_results(), f, indent=2)

        print(f"\n  ✓ Saved: {output_file}")

    else:
        # Batch calibration (default)
        batch_calibrate_all_models(
            data_dir=Path(args.data_dir),
            output_dir=Path(args.output),
            lookback_bars=args.lookback
        )


if __name__ == '__main__':
    main()
