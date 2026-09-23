#!/usr/bin/env python3
"""
Signature Generation Sweep - Stage 2 of FAS Sweep Pipeline

Generates log-signatures for all 6 velocity windows using max_dc scale DC events.

This script iterates through velocity windows [10, 15, 17, 20, 27, 50] and
generates depth-4 log-signatures with lead-lag transformation for each.

Output:
    data/signatures/signatures_max_dc_vw10.parquet
    data/signatures/signatures_max_dc_vw15.parquet
    data/signatures/signatures_max_dc_vw17.parquet
    data/signatures/signatures_max_dc_vw20.parquet
    data/signatures/signatures_max_dc_vw27.parquet
    data/signatures/signatures_max_dc_vw50.parquet

Usage:
    python src/run_signature_sweep.py

Author: FAS Sweep Pipeline
Date: 2026-01-14
"""

import sys
import yaml
import subprocess
import logging
from pathlib import Path

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


def save_config(config: dict, config_path: str):
    """Save configuration to YAML file."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def update_config_window(config_path: Path, window: int):
    """Update max_dc window in config to match velocity_window."""
    logger.info(f"  → Updating config: max_dc.window = {window}")

    config = load_config(str(config_path))
    config['sampler']['scales']['max_dc']['window'] = window
    save_config(config, str(config_path))

    logger.info(f"  ✓ Config updated")


def split_signatures_by_date(signatures_path: Path, config: dict, window: int, project_root: Path, asset: str = "EURUSD"):
    """
    Split signatures into train+test and OOS files based on config dates.

    Args:
        signatures_path: Path to full signatures file
        config: Configuration dictionary
        window: Velocity window size
        project_root: Project root directory
        asset: Asset symbol
    """
    import pandas as pd

    logger.info(f"\n  Splitting signatures by date...")

    # Load full signatures
    df = pd.read_parquet(signatures_path)
    logger.info(f"  Loaded {len(df)} signature windows")

    # Get split configuration
    splits_cfg = config.get('signature_splits', {})
    train_test_cfg = splits_cfg.get('train_test', {})
    oos_cfg = splits_cfg.get('oos', {})

    # Parse dates
    train_test_start = pd.to_datetime(train_test_cfg.get('start_date'))
    train_test_end = pd.to_datetime(train_test_cfg.get('end_date'))
    oos_start = pd.to_datetime(oos_cfg.get('start_date'))

    # Determine timestamp column
    timestamp_col = None
    for col in ['end_timestamp', 'start_timestamp', 'timestamp']:
        if col in df.columns:
            timestamp_col = col
            break

    if timestamp_col is None:
        logger.error("  ✗ No timestamp column found in signatures!")
        return False

    # Convert to datetime
    df['date'] = pd.to_datetime(df[timestamp_col])

    # Split data
    train_test_df = df[(df['date'] >= train_test_start) & (df['date'] <= train_test_end)].copy()
    oos_df = df[df['date'] >= oos_start].copy()

    # Remove temporary date column
    train_test_df = train_test_df.drop(columns=['date'])
    oos_df = oos_df.drop(columns=['date'])

    logger.info(f"  Train+Test: {len(train_test_df)} windows ({train_test_start.date()} to {train_test_end.date()})")
    logger.info(f"  OOS: {len(oos_df)} windows (from {oos_start.date()})")

    # Save split files - asset-specific directory
    signatures_dir = project_root / f"data_{asset}" / "signatures"

    train_test_path = signatures_dir / f"signatures_max_dc_vw{window}_train_test.parquet"
    oos_path = signatures_dir / f"signatures_max_dc_vw{window}_oos.parquet"

    train_test_df.to_parquet(train_test_path, index=False)
    oos_df.to_parquet(oos_path, index=False)

    logger.info(f"  ✓ Train+Test saved: {train_test_path}")
    logger.info(f"  ✓ OOS saved: {oos_path}")

    return True


def run_signature_generation(window: int, project_root: Path, asset: str = "EURUSD") -> bool:
    """
    Run signature generation for a specific velocity window and asset.

    Args:
        window: Velocity window size
        project_root: Project root directory
        asset: Asset symbol

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"GENERATING SIGNATURES FOR {asset} - VELOCITY_WINDOW={window}")
    logger.info(f"{'='*70}")

    try:
        # Update config
        config_path = project_root / "config" / "config.yaml"
        update_config_window(config_path, window)

        # Reload config to get split settings
        config = load_config(str(config_path))

        # Run signature generation (full data)
        script_path = project_root / "src" / "02_signatures" / "generate_signatures.py"
        suffix = f"vw{window}"

        cmd = [
            sys.executable,
            str(script_path),
            "--output_suffix", suffix,
            "--asset", asset
        ]

        logger.info(f"\n  Running: {' '.join(cmd)}")
        logger.info("")

        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            check=True,
            capture_output=False
        )

        # Verify output exists - asset-specific path
        output_file = project_root / f"data_{asset}" / "signatures" / f"signatures_max_dc_{suffix}.parquet"
        if not output_file.exists():
            logger.error(f"\n  ✗ Output file not found: {output_file}")
            return False

        logger.info(f"\n  ✓ Full signatures saved: {output_file}")

        # Split signatures by date
        success = split_signatures_by_date(output_file, config, window, project_root, asset)

        if not success:
            return False

        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"\n  ✗ Signature generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        logger.error(f"\n  ✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_signature_sweep(asset: str = "EURUSD"):
    """
    Main entry point for signature sweep stage.

    Args:
        asset: Asset symbol (e.g., EURUSD, GBPUSD)
    """

    logger.info("="*80)
    logger.info(f"STAGE 2: SIGNATURE GENERATION SWEEP - STARTED ({asset})")
    logger.info("="*80)

    # Paths
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "config" / "config.yaml"
    signatures_dir = project_root / f"data_{asset}" / "signatures"

    # Load config
    logger.info(f"Loading configuration from: {config_path}")
    config = load_config(str(config_path))

    # Get velocity windows from sweep config
    velocity_windows = config.get('sweep', {}).get('velocity_windows', [10, 15, 17, 20, 27, 50])
    logger.info(f"Velocity windows to process: {velocity_windows}")

    # Check DC events exist
    dc_events_path = project_root / f"data_{asset}" / "dc_events" / "dc_events_max_dc.parquet"
    if not dc_events_path.exists():
        logger.error(f"DC events not found: {dc_events_path}")
        logger.error("Please run Stage 1 (FAS sampling) first")
        sys.exit(1)

    logger.info(f"Using DC events: {dc_events_path}")

    # Run signature generation for each window
    success_count = 0
    failed_windows = []

    for window in velocity_windows:
        success = run_signature_generation(window, project_root, asset)

        if success:
            success_count += 1
        else:
            failed_windows.append(window)

    # Summary
    logger.info("\n" + "="*80)
    logger.info(f"STAGE 2: SIGNATURE GENERATION SWEEP - COMPLETED ({asset})")
    logger.info("="*80)
    logger.info(f"Output directory: {signatures_dir}")
    logger.info(f"Successful: {success_count}/{len(velocity_windows)}")

    if failed_windows:
        logger.warning(f"Failed windows: {failed_windows}")
        sys.exit(1)

    logger.info("\nGenerated signature files:")
    for window in velocity_windows:
        train_test_file = signatures_dir / f"signatures_max_dc_vw{window}_train_test.parquet"
        oos_file = signatures_dir / f"signatures_max_dc_vw{window}_oos.parquet"
        if train_test_file.exists() and oos_file.exists():
            logger.info(f"  ✓ vw{window}: {train_test_file.name} + {oos_file.name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run signature generation sweep for a specific asset')
    parser.add_argument('--asset', type=str, default='EURUSD',
                        help='Asset symbol (e.g., EURUSD, GBPUSD)')
    args = parser.parse_args()

    try:
        run_signature_sweep(asset=args.asset)
    except Exception as e:
        logger.error(f"Signature sweep failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
