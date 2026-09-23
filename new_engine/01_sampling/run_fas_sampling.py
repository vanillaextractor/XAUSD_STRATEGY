#!/usr/bin/env python3
"""
FAS Sampling - Stage 1 of FAS Sweep Pipeline

Generates multi-scale DC events using the Fractal-Adaptive Sampler (FAS)
with causality enforcement.

This script runs FAS at 3 scales (small_dc, mid_dc, max_dc) and saves
DC event files for downstream signature generation.

Output:
    data/dc_events/dc_events_small_dc.parquet (~50k events)
    data/dc_events/dc_events_mid_dc.parquet (~25k events)
    data/dc_events/dc_events_max_dc.parquet (~12k events)

Usage:
    python src/run_fas_sampling.py

Author: FAS Sweep Pipeline
Date: 2026-01-14
"""

import sys
import yaml
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from fractal_adaptive_sampler import FractalAdaptiveSampler, GapConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data(config: dict, asset: str = 'EURUSD') -> pd.DataFrame:
    """
    Load and prepare raw price data from parquet file.

    Args:
        config: Configuration dictionary
        asset: Asset symbol (e.g., EURUSD, GBPUSD)

    Returns:
        DataFrame with columns: timestamp, close
    """
    data_cfg = config.get('data', {})
    filepath_template = data_cfg.get('path', 'data_{asset}/raw/{asset}2017_2025_cleaned.parquet')
    filepath = filepath_template.format(asset=asset)

    price_col = data_cfg.get('price_col', 'CLOSE')
    date_col = data_cfg.get('date_col', 'DATE')
    time_col = data_cfg.get('time_col', 'TIME')
    date_format = data_cfg.get('date_format', '%Y.%m.%d %H:%M:%S')

    start_date = data_cfg.get('start_date', None)
    end_date = data_cfg.get('end_date', None)

    logger.info(f"Loading price data from: {filepath}")
    df = pd.read_parquet(filepath)

    # Create timestamp
    df['timestamp'] = pd.to_datetime(
        df[date_col] + ' ' + df[time_col],
        format=date_format
    )
    df['close'] = df[price_col]

    # Apply date filters
    if start_date is not None:
        start_dt = pd.to_datetime(start_date)
        df = df[df['timestamp'] >= start_dt]
        logger.info(f"  Filtered from: {start_date}")

    if end_date is not None:
        end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1)
        df = df[df['timestamp'] < end_dt]
        logger.info(f"  Filtered to:   {end_date}")

    df = df.sort_values('timestamp').reset_index(drop=True)
    logger.info(f"  Loaded {len(df)} bars")

    return df[['timestamp', 'close']]


def run_multi_scale_sampler(
    data: pd.DataFrame,
    config: dict
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Run FAS at multiple theta scales with per-scale optimized parameters.

    Returns:
        events_dict: Dictionary mapping scale name to events DataFrame
        metrics: Metrics from the last scale
    """
    sampler_cfg = config.get('sampler', {})
    gap_cfg = config.get('gap_handling', {})

    # Get scales configuration
    scales = sampler_cfg.get('scales', {})

    # Common parameters
    sigma_ref_window = sampler_cfg.get('sigma_ref_window', 1440)
    hurst_min = sampler_cfg.get('hurst_min', 0.05)
    hurst_max = sampler_cfg.get('hurst_max', 0.95)

    # Gap configuration
    gap_config = GapConfig(
        enabled=gap_cfg.get('enabled', True),
        min_gap_minutes=gap_cfg.get('min_gap_minutes', 120),
        mode=gap_cfg.get('mode', 'bridge'),
        max_gap_return=gap_cfg.get('max_gap_return', 0.10),
        gap_theta_multiplier=gap_cfg.get('gap_theta_multiplier', 1.5)
    )

    logger.info(f"Running FAS at {len(scales)} scales")
    logger.info(f"  Sigma reference window: {sigma_ref_window}")
    logger.info(f"  Hurst bounds: [{hurst_min}, {hurst_max}]")
    logger.info(f"  Gap handling: mode={gap_config.mode}, enabled={gap_config.enabled}")

    events_dict = {}
    metrics = None

    for scale_name, scale_params in scales.items():
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing scale: {scale_name}")
        logger.info(f"{'='*70}")
        logger.info(f"  Theta base: {scale_params['theta']}")
        logger.info(f"  Window: {scale_params['window']}")
        logger.info(f"  Lambda: {scale_params['lambda']}")

        # Initialize sampler for this scale
        sampler = FractalAdaptiveSampler(
            theta_base=scale_params['theta'],
            window=scale_params['window'],
            sigma_ref_window=sigma_ref_window,
            lambda_aggression=scale_params['lambda'],
            hurst_min=hurst_min,
            hurst_max=hurst_max,
            gap_config=gap_config
        )

        # Run FAS
        events = sampler.process_dataframe(data)
        metrics = sampler.get_metrics()

        # Add scale metadata
        events['scale'] = scale_name
        events['theta_base'] = scale_params['theta']
        events['window'] = scale_params['window']
        events['lambda'] = scale_params['lambda']

        events_dict[scale_name] = events

        logger.info(f"  Detected {len(events)} DC events")
        logger.info(f"  Upturn events: {(events['type'] == 'upturn').sum()}")
        logger.info(f"  Downturn events: {(events['type'] == 'downturn').sum()}")
        logger.info(f"  Gap events: {events['is_gap_event'].sum()}")

    return events_dict, metrics


def save_events(events_dict: Dict[str, pd.DataFrame], output_dir: Path, asset: str):
    """Save DC events for each scale to parquet files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*70}")
    logger.info(f"Saving DC event files for {asset}")
    logger.info(f"{'='*70}")

    for scale_name, events in events_dict.items():
        output_path = output_dir / f"dc_events_{scale_name}.parquet"
        events.to_parquet(output_path, index=False)
        logger.info(f"  ✓ {scale_name}: {output_path} ({len(events)} events)")


def run_fas_sampling(asset: str = 'EURUSD'):
    """
    Main entry point for FAS sampling stage.

    Args:
        asset: Asset symbol (e.g., EURUSD, GBPUSD)
    """

    logger.info("="*80)
    logger.info(f"STAGE 1: FAS SAMPLING - STARTED ({asset})")
    logger.info("="*80)

    # Paths
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "config" / "config.yaml"
    output_dir = project_root / f"data_{asset}" / "dc_events"

    # Load config
    logger.info(f"Loading configuration from: {config_path}")
    config = load_config(str(config_path))

    # Load price data
    data = load_data(config, asset)

    # Run multi-scale FAS
    events_dict, metrics = run_multi_scale_sampler(data, config)

    # Save events
    save_events(events_dict, output_dir, asset)

    # Summary
    logger.info("\n" + "="*80)
    logger.info(f"STAGE 1: FAS SAMPLING - COMPLETED ({asset})")
    logger.info("="*80)
    logger.info(f"Output directory: {output_dir}")

    total_events = sum(len(events) for events in events_dict.values())
    logger.info(f"Total DC events generated: {total_events:,}")

    for scale_name, events in events_dict.items():
        logger.info(f"  {scale_name}: {len(events):,} events")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run FAS sampling for a specific asset')
    parser.add_argument('--asset', type=str, default='EURUSD',
                        help='Asset symbol (e.g., EURUSD, GBPUSD)')
    args = parser.parse_args()

    try:
        run_fas_sampling(asset=args.asset)
    except Exception as e:
        logger.error(f"FAS sampling failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
