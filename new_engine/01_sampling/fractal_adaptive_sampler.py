"""
Fractal-Adaptive Sampling (FAS) Engine

A dynamic, regime-aware sampler for generating Directional Change (DC) events
from raw 1-minute Close price data. Uses local Hurst exponent and volatility
estimates to dynamically adjust sampling thresholds.

Mathematical Foundation:
    θ_t = θ_base × V_t × F_t

Where:
    - θ_base: Base threshold
    - V_t = σ_t / σ_ref (Energy Scaling Factor)
    - F_t = exp(λ × (0.5 - H_t)) (Fractal Penalty Factor)
    - H_t: Local Hurst exponent via K-over-N Ratio of Variations
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config


@dataclass
class GapConfig:
    """Configuration for gap/weekend handling."""
    enabled: bool = True
    min_gap_minutes: int = 120
    mode: str = "bridge"  # "reset", "bridge", or "exclude"
    max_gap_return: float = 0.10
    gap_theta_multiplier: float = 1.5


@dataclass
class DCEvent:
    """Represents a single Directional Change event."""
    timestamp: pd.Timestamp
    price: float
    event_type: str  # 'upturn' or 'downturn'
    theta_dynamic: float
    hurst_snapshot: float
    is_gap_event: bool = False


class FractalAdaptiveSampler:
    """
    Fractal-Adaptive Sampler for Directional Change event detection.

    Uses dynamic thresholds based on local volatility and roughness (Hurst exponent)
    to normalize the geometric roughness of the sampled path.

    Parameters
    ----------
    theta_base : float, default=0.002
        Base threshold for directional change (0.2% = 0.002).
        Optimized: tighter threshold lets FAS expand automatically in high vol.
    window : int, default=30
        Lookback window for rolling estimators (σ_t, H_t).
        Optimized: faster reflexes for detecting regime changes.
    sigma_ref_window : int, default=1440
        Lookback window for reference volatility (σ_ref). Default is 1440
        (1 trading day of 1-minute bars) to represent "climate" not "weather".
    lambda_aggression : float, default=2.0
        Aggression factor for fractal penalty. Higher values increase
        sensitivity to roughness deviations from Brownian (H=0.5).
    hurst_min : float, default=0.05
        Minimum Hurst exponent bound for numerical stability.
    hurst_max : float, default=0.95
        Maximum Hurst exponent bound for numerical stability.
    gap_config : GapConfig, optional
        Configuration for weekend/gap handling. If None, uses defaults.
    """

    def __init__(
        self,
        theta_base: float = 0.002,
        window: int = 30,
        sigma_ref_window: int = 1440,
        lambda_aggression: float = 2.0,
        hurst_min: float = 0.05,
        hurst_max: float = 0.95,
        gap_config: Optional[GapConfig] = None
    ):
        self.theta_base = theta_base
        self.window = window
        self.sigma_ref_window = sigma_ref_window
        self.lambda_aggression = lambda_aggression
        self.hurst_min = hurst_min
        self.hurst_max = hurst_max
        self.gap_config = gap_config or GapConfig()

        # Store computed metrics for debugging/visualization
        self._metrics: Optional[pd.DataFrame] = None
        self._gap_indices: Optional[np.ndarray] = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'FractalAdaptiveSampler':
        """
        Create a FractalAdaptiveSampler from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary (typically loaded from YAML).

        Returns
        -------
        FractalAdaptiveSampler
            Configured sampler instance.
        """
        sampler_cfg = config.get('sampler', {})
        gap_cfg = config.get('gap_handling', {})

        gap_config = GapConfig(
            enabled=gap_cfg.get('enabled', True),
            min_gap_minutes=gap_cfg.get('min_gap_minutes', 120),
            mode=gap_cfg.get('mode', 'bridge'),
            max_gap_return=gap_cfg.get('max_gap_return', 0.10),
            gap_theta_multiplier=gap_cfg.get('gap_theta_multiplier', 1.5)
        )

        return cls(
            theta_base=sampler_cfg.get('theta_base', 0.005),
            window=sampler_cfg.get('window', 50),
            sigma_ref_window=sampler_cfg.get('sigma_ref_window', 1440),
            lambda_aggression=sampler_cfg.get('lambda_aggression', 2.0),
            hurst_min=sampler_cfg.get('hurst_min', 0.05),
            hurst_max=sampler_cfg.get('hurst_max', 0.95),
            gap_config=gap_config
        )

    def _detect_gaps(self, timestamps: pd.Series) -> np.ndarray:
        """
        Detect gaps (weekends, market closures) in the timestamp series.

        Parameters
        ----------
        timestamps : pd.Series
            Timestamp series.

        Returns
        -------
        np.ndarray
            Boolean array where True indicates a gap at that index.
        """
        if not self.gap_config.enabled:
            return np.zeros(len(timestamps), dtype=bool)

        # Calculate time differences in minutes
        time_diffs = timestamps.diff().dt.total_seconds() / 60.0

        # Mark gaps where time difference exceeds threshold
        gap_mask = time_diffs > self.gap_config.min_gap_minutes

        return gap_mask.values

    def _compute_rolling_metrics(
        self,
        prices: pd.Series,
        gap_mask: np.ndarray
    ) -> pd.DataFrame:
        """
        Pre-compute all rolling metrics using vectorized operations.

        For gap handling in "exclude" mode, gap returns are replaced with NaN
        before computing rolling statistics.

        Parameters
        ----------
        prices : pd.Series
            Close price series.
        gap_mask : np.ndarray
            Boolean array indicating gap locations.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: sigma_t, sigma_ref, V_t, H_t, F_t, theta_t
        """
        w = self.window

        # 1. Log-returns
        returns = np.log(prices / prices.shift(1))

        # For "exclude" mode, mask out gap returns from volatility calculation
        if self.gap_config.enabled and self.gap_config.mode == "exclude":
            returns_for_vol = returns.copy()
            returns_for_vol[gap_mask] = np.nan
        else:
            returns_for_vol = returns

        # 2. Rolling volatility (σ_t) - standard deviation of log-returns
        sigma_t = returns_for_vol.rolling(window=w, min_periods=w).std()

        # 3. Reference volatility (σ_ref) - fixed lookback mean of σ_t
        sigma_ref = sigma_t.rolling(window=self.sigma_ref_window, min_periods=1).mean()

        # 4. Energy Scaling Factor (V_t)
        V_t = np.where(sigma_ref > 0, sigma_t / sigma_ref, 1.0)
        V_t = pd.Series(V_t, index=prices.index)
        V_t = V_t.fillna(1.0)

        # 5. Hurst exponent (H_t) via K-over-N Ratio of Variations
        # For "exclude" mode, mask gap price differences
        if self.gap_config.enabled and self.gap_config.mode == "exclude":
            prices_for_hurst = prices.copy()
            # Set gap prices to NaN to exclude from Hurst calculation
            prices_for_hurst[gap_mask] = np.nan
            abs_diff_1 = np.abs(prices_for_hurst.diff(1))
            abs_diff_2 = np.abs(prices_for_hurst.diff(2))
        else:
            abs_diff_1 = np.abs(prices.diff(1))
            abs_diff_2 = np.abs(prices.diff(2))

        m1 = abs_diff_1.rolling(window=w, min_periods=w).sum()
        m2 = abs_diff_2.rolling(window=w, min_periods=w).sum()

        # Compute Hurst: H_t = (1/ln(2)) * ln(m2/m1)
        eps = 1e-10
        ratio = m2 / (m1 + eps)
        ratio = np.maximum(ratio, eps)
        H_t = (1.0 / np.log(2.0)) * np.log(ratio)

        # Clamp H_t for numerical stability
        H_t = np.clip(H_t, self.hurst_min, self.hurst_max)
        H_t = pd.Series(H_t, index=prices.index)
        H_t = H_t.fillna(0.5)

        # 6. Fractal Penalty (F_t)
        F_t = np.exp(self.lambda_aggression * (0.5 - H_t))

        # 7. Dynamic Threshold (θ_t)
        theta_t = self.theta_base * V_t * F_t

        metrics = pd.DataFrame({
            'sigma_t': sigma_t,
            'sigma_ref': sigma_ref,
            'V_t': V_t,
            'H_t': H_t,
            'F_t': F_t,
            'theta_t': theta_t,
            'is_gap': gap_mask
        }, index=prices.index)

        # CRITICAL: Enforce causality by lagging all metrics by 1.
        # The threshold for evaluating the move from t-1 to t must be
        # computed using only information available at t-1.
        metrics_to_shift = ['sigma_t', 'sigma_ref', 'V_t', 'H_t', 'F_t', 'theta_t']
        metrics[metrics_to_shift] = metrics[metrics_to_shift].shift(1)

        # Fill NaN values introduced by the shift
        metrics['V_t'] = metrics['V_t'].fillna(1.0)
        metrics['H_t'] = metrics['H_t'].fillna(0.5)
        metrics['F_t'] = metrics['F_t'].fillna(1.0)
        metrics['theta_t'] = metrics['theta_t'].fillna(self.theta_base)

        return metrics

    def _detect_dc_events(
        self,
        prices: np.ndarray,
        timestamps: np.ndarray,
        theta_t: np.ndarray,
        H_t: np.ndarray,
        gap_mask: np.ndarray
    ) -> list:
        """
        Detect Directional Change events using threshold-crossing logic.

        Records a new event whenever price crosses the threshold from the last
        confirmed event, regardless of direction. This allows consecutive events
        in the same direction (e.g., multiple upturns if price repeatedly crosses
        the upper threshold).

        Handles gaps according to the configured mode:
        - "reset": Reset sampler state after gap
        - "bridge": Evaluate gap with pre-gap threshold (with optional multiplier)
        - "exclude": Skip gap from volatility but still detect DC events

        Parameters
        ----------
        prices : np.ndarray
            Price array.
        timestamps : np.ndarray
            Timestamp array.
        theta_t : np.ndarray
            Pre-computed dynamic threshold array.
        H_t : np.ndarray
            Pre-computed Hurst exponent array.
        gap_mask : np.ndarray
            Boolean array indicating gap locations.

        Returns
        -------
        list
            List of DCEvent objects.
        """
        n = len(prices)
        events = []
        gap_cfg = self.gap_config

        # Start after we have enough data for metrics
        start_idx = self.window + self.sigma_ref_window
        if start_idx >= n:
            return events

        # Initialize: track last confirmed event price and extremes
        P_last_event = prices[start_idx]  # Last confirmed event price
        t_last_event = start_idx
        
        # Track running high/low since last event for initialization
        running_high = prices[start_idx]
        running_low = prices[start_idx]
        high_idx = start_idx
        low_idx = start_idx
        
        initialized = False  # Whether we've recorded the first event

        for i in range(start_idx + 1, n):
            P_new = prices[i]
            theta = theta_t[i]
            is_gap = gap_mask[i] if gap_cfg.enabled else False

            # Skip if threshold is NaN or invalid
            if np.isnan(theta) or theta <= 0:
                theta = self.theta_base

            # Handle gap according to mode
            if is_gap and gap_cfg.enabled:
                # Calculate gap return
                P_prev = prices[i - 1]
                gap_return = abs(P_new - P_prev) / P_prev if P_prev > 0 else 0

                if gap_cfg.mode == "reset":
                    # Reset state after gap - start fresh
                    P_last_event = P_new
                    t_last_event = i
                    running_high = P_new
                    running_low = P_new
                    high_idx = i
                    low_idx = i
                    initialized = False
                    continue

                elif gap_cfg.mode == "bridge":
                    # Check if gap exceeds maximum allowed (likely data error)
                    if gap_return > gap_cfg.max_gap_return:
                        # Treat as reset - gap too large
                        P_last_event = P_new
                        t_last_event = i
                        running_high = P_new
                        running_low = P_new
                        high_idx = i
                        low_idx = i
                        initialized = False
                        continue

                    # Apply gap theta multiplier for bridge mode
                    theta = theta * gap_cfg.gap_theta_multiplier

            # Update running extremes
            if P_new > running_high:
                running_high = P_new
                high_idx = i
            if P_new < running_low:
                running_low = P_new
                low_idx = i

            if not initialized:
                # Initialization phase: wait for first threshold crossing
                up_move = (running_high - running_low) / running_low if running_low > 0 else 0

                if up_move >= theta:
                    # Determine which extreme was hit first
                    if high_idx > low_idx:
                        # Hit high first - record upturn
                        events.append(DCEvent(
                            timestamp=timestamps[high_idx],
                            price=running_high,
                            event_type='upturn',
                            theta_dynamic=theta,
                            hurst_snapshot=H_t[high_idx],
                            is_gap_event=is_gap
                        ))
                        P_last_event = running_high
                        t_last_event = high_idx
                    else:
                        # Hit low first - record downturn
                        events.append(DCEvent(
                            timestamp=timestamps[low_idx],
                            price=running_low,
                            event_type='downturn',
                            theta_dynamic=theta,
                            hurst_snapshot=H_t[low_idx],
                            is_gap_event=is_gap
                        ))
                        P_last_event = running_low
                        t_last_event = low_idx
                    
                    # Reset running extremes from last event
                    running_high = P_new
                    running_low = P_new
                    high_idx = i
                    low_idx = i
                    initialized = True

            else:
                # Threshold-crossing logic: check for crossings from last event
                # Calculate thresholds from last confirmed event
                upper_threshold = P_last_event * (1.0 + theta)
                lower_threshold = P_last_event * (1.0 - theta)

                # Check if we've crossed either threshold
                crossed_upper = P_new >= upper_threshold
                crossed_lower = P_new <= lower_threshold

                if crossed_upper or crossed_lower:
                    # Determine which threshold was crossed first by checking running extremes
                    if crossed_upper and not crossed_lower:
                        # Crossed upper threshold - record upturn
                        events.append(DCEvent(
                            timestamp=timestamps[i],
                            price=P_new,
                            event_type='upturn',
                            theta_dynamic=theta,
                            hurst_snapshot=H_t[i],
                            is_gap_event=is_gap
                        ))
                        P_last_event = P_new
                        t_last_event = i
                        running_high = P_new
                        running_low = P_new
                        high_idx = i
                        low_idx = i
                        
                    elif crossed_lower and not crossed_upper:
                        # Crossed lower threshold - record downturn
                        events.append(DCEvent(
                            timestamp=timestamps[i],
                            price=P_new,
                            event_type='downturn',
                            theta_dynamic=theta,
                            hurst_snapshot=H_t[i],
                            is_gap_event=is_gap
                        ))
                        P_last_event = P_new
                        t_last_event = i
                        running_high = P_new
                        running_low = P_new
                        high_idx = i
                        low_idx = i
                        
                    else:
                        # Both crossed (large move) - determine which was hit first
                        # Check which extreme (high or low) was reached first since last event
                        if high_idx > low_idx:
                            # Hit upper threshold first
                            events.append(DCEvent(
                                timestamp=timestamps[i],
                                price=P_new,
                                event_type='upturn',
                                theta_dynamic=theta,
                                hurst_snapshot=H_t[i],
                                is_gap_event=is_gap
                            ))
                        else:
                            # Hit lower threshold first
                            events.append(DCEvent(
                                timestamp=timestamps[i],
                                price=P_new,
                                event_type='downturn',
                                theta_dynamic=theta,
                                hurst_snapshot=H_t[i],
                                is_gap_event=is_gap
                            ))
                        
                        P_last_event = P_new
                        t_last_event = i
                        running_high = P_new
                        running_low = P_new
                        high_idx = i
                        low_idx = i

        return events

    def process(
        self,
        prices: pd.Series,
        timestamps: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Process price series and extract DC events.

        Parameters
        ----------
        prices : pd.Series
            Close price series.
        timestamps : pd.Series, optional
            Timestamp series. If None, uses prices.index.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: timestamp, price, type, theta_dynamic,
            hurst_snapshot, is_gap_event
        """
        if timestamps is None:
            timestamps = prices.index.to_series()

        # Ensure aligned indices
        prices = prices.reset_index(drop=True)
        timestamps = timestamps.reset_index(drop=True)

        # Detect gaps
        gap_mask = self._detect_gaps(timestamps)
        self._gap_indices = np.where(gap_mask)[0]

        # Pre-compute rolling metrics (vectorized)
        self._metrics = self._compute_rolling_metrics(prices, gap_mask)

        # Convert to numpy for fast iteration
        prices_arr = prices.values.astype(np.float64)
        timestamps_arr = timestamps.values
        theta_arr = self._metrics['theta_t'].values.astype(np.float64)
        H_arr = self._metrics['H_t'].values.astype(np.float64)
        gap_arr = self._metrics['is_gap'].values

        # Detect DC events
        events = self._detect_dc_events(
            prices_arr, timestamps_arr, theta_arr, H_arr, gap_arr
        )

        # Convert to DataFrame
        if not events:
            return pd.DataFrame(columns=[
                'timestamp', 'price', 'type', 'theta_dynamic',
                'hurst_snapshot', 'is_gap_event'
            ])

        df = pd.DataFrame([
            {
                'timestamp': e.timestamp,
                'price': e.price,
                'type': e.event_type,
                'theta_dynamic': e.theta_dynamic,
                'hurst_snapshot': e.hurst_snapshot,
                'is_gap_event': e.is_gap_event
            }
            for e in events
        ])

        return df

    def process_dataframe(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        time_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Process DataFrame and extract DC events.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with price and timestamp columns.
        price_col : str, default='close'
            Name of the close price column.
        time_col : str, default='timestamp'
            Name of the timestamp column.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: timestamp, price, type, theta_dynamic,
            hurst_snapshot, is_gap_event
        """
        prices = df[price_col]
        timestamps = df[time_col]
        return self.process(prices, timestamps)

    def get_metrics(self) -> Optional[pd.DataFrame]:
        """
        Get the computed rolling metrics from the last process() call.

        Returns
        -------
        pd.DataFrame or None
            DataFrame with columns: sigma_t, sigma_ref, V_t, H_t, F_t, theta_t, is_gap
            Returns None if process() hasn't been called.
        """
        return self._metrics

    def get_gap_indices(self) -> Optional[np.ndarray]:
        """
        Get the indices where gaps were detected.

        Returns
        -------
        np.ndarray or None
            Array of gap indices, or None if process() hasn't been called.
        """
        return self._gap_indices


def main():
    """Simple test with synthetic data."""
    # Generate synthetic price data (geometric Brownian motion)
    np.random.seed(42)
    n = 5000
    dt = 1.0 / (252 * 390)
    mu = 0.0
    sigma = 0.20

    returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), n)
    prices = 100 * np.exp(np.cumsum(returns))

    timestamps = pd.date_range('2024-01-01', periods=n, freq='1min')
    prices_series = pd.Series(prices, index=timestamps)

    # Try loading from config if available
    try:
        config = load_config("config.yaml")
        sampler = FractalAdaptiveSampler.from_config(config)
        print("Loaded configuration from config.yaml")
    except FileNotFoundError:
        sampler = FractalAdaptiveSampler()  # Uses optimized defaults
        print("Using default configuration (optimized)")

    events_df = sampler.process(prices_series)
    metrics = sampler.get_metrics()

    print(f"\nTotal events: {len(events_df)}")
    print(f"\nEvent distribution:")
    print(events_df['type'].value_counts())

    if len(events_df) > 0:
        gap_events = events_df['is_gap_event'].sum()
        print(f"\nGap events: {gap_events}")
        print(f"\nSample events:")
        print(events_df.head(10))
        print(f"\nThreshold statistics:")
        print(f"  Mean θ_t: {events_df['theta_dynamic'].mean():.6f}")
        print(f"  Std θ_t: {events_df['theta_dynamic'].std():.6f}")
        print(f"\nHurst statistics:")
        print(f"  Mean H_t: {events_df['hurst_snapshot'].mean():.4f}")
        print(f"  Std H_t: {events_df['hurst_snapshot'].std():.4f}")


if __name__ == '__main__':
    main()
