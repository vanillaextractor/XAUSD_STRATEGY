#!/usr/bin/env python3
"""
Multi-Asset Signature Generation Sweep

Runs signature generation for all assets × all velocity windows.
This creates 6 assets × 6 velocity windows = 36 signature files per split (train_test + oos).

Input:
    data_{ASSET}/dc_events/dc_events_max_dc.parquet (for each asset)

Output:
    data_{ASSET}/signatures/signatures_max_dc_vw{window}_train_test.parquet
    data_{ASSET}/signatures/signatures_max_dc_vw{window}_oos.parquet

    For 6 assets and 6 velocity windows: 72 total files (36 train_test + 36 oos)

Usage:
    python src/02_signatures/run_all_assets_signature_sweep.py
    python src/02_signatures/run_all_assets_signature_sweep.py --parallel  # Run in parallel

Author: Multi-Asset Pipeline
Date: 2026-02-01
"""

import sys
import yaml
import logging
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

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


def run_signature_sweep_for_asset(asset: str, script_path: Path) -> dict:
    """
    Run signature sweep for a single asset (all 6 velocity windows).

    Args:
        asset: Asset symbol
        script_path: Path to run_signature_sweep.py

    Returns:
        Dictionary with asset name and status
    """
    logger.info(f"Starting signature sweep for {asset}...")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path), '--asset', asset],
            capture_output=True,
            text=True,
            check=True
        )

        logger.info(f"✅ {asset}: Completed successfully")
        return {'asset': asset, 'status': 'success', 'message': 'Completed'}

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {asset}: Failed with error")
        logger.error(f"   stdout: {e.stdout}")
        logger.error(f"   stderr: {e.stderr}")
        return {'asset': asset, 'status': 'failed', 'message': str(e)}

    except Exception as e:
        logger.error(f"❌ {asset}: Unexpected error: {e}")
        return {'asset': asset, 'status': 'failed', 'message': str(e)}


def run_all_assets_signature_sweep(parallel: bool = False):
    """
    Main entry point for multi-asset signature generation sweep.

    Args:
        parallel: If True, run assets in parallel. If False, run sequentially.
    """
    logger.info("=" * 100)
    logger.info("MULTI-ASSET SIGNATURE GENERATION SWEEP - STARTED")
    logger.info("=" * 100)

    # Paths
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "config" / "config.yaml"
    script_path = Path(__file__).parent / "run_signature_sweep.py"

    # Load config and get asset list
    logger.info(f"Loading configuration from: {config_path}")
    config = load_config(str(config_path))

    assets = config.get('assets', ['EURUSD'])
    velocity_windows = config.get('sweep', {}).get('velocity_windows', [10, 15, 17, 20, 27, 50])

    total_combinations = len(assets) * len(velocity_windows)

    logger.info(f"Assets to process: {assets} ({len(assets)} assets)")
    logger.info(f"Velocity windows: {velocity_windows} ({len(velocity_windows)} windows)")
    logger.info(f"Total combinations: {total_combinations} (= {len(assets)} assets × {len(velocity_windows)} windows)")
    logger.info(f"Execution mode: {'PARALLEL' if parallel else 'SEQUENTIAL'}")
    logger.info("")

    # Check DC events exist for all assets
    logger.info("Verifying DC events for all assets...")
    missing_assets = []
    for asset in assets:
        dc_path = project_root / f"data_{asset}" / "dc_events" / "dc_events_max_dc.parquet"
        if not dc_path.exists():
            missing_assets.append(asset)
            logger.warning(f"  ⚠️  {asset}: DC events not found at {dc_path}")
        else:
            logger.info(f"  ✓ {asset}: DC events found")

    if missing_assets:
        logger.error(f"\n❌ Missing DC events for assets: {missing_assets}")
        logger.error("Please run DC sampling (run_all_assets_sampling.py) first")
        sys.exit(1)

    logger.info("\nAll DC events verified!\n")

    results = []

    if parallel:
        # Parallel execution
        logger.info("Running signature sweeps in PARALLEL mode...")
        logger.info("-" * 100)

        with ProcessPoolExecutor(max_workers=min(len(assets), 3)) as executor:
            # Submit all tasks
            future_to_asset = {
                executor.submit(run_signature_sweep_for_asset, asset, script_path): asset
                for asset in assets
            }

            # Collect results as they complete
            for future in as_completed(future_to_asset):
                asset = future_to_asset[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"❌ {asset}: Exception during execution: {e}")
                    results.append({'asset': asset, 'status': 'failed', 'message': str(e)})

    else:
        # Sequential execution
        logger.info("Running signature sweeps in SEQUENTIAL mode...")
        logger.info("-" * 100)

        for idx, asset in enumerate(assets, 1):
            logger.info(f"\n[{idx}/{len(assets)}] Processing {asset}...")
            logger.info("-" * 80)

            result = run_signature_sweep_for_asset(asset, script_path)
            results.append(result)

            logger.info("")

    # Summary
    logger.info("")
    logger.info("=" * 100)
    logger.info("MULTI-ASSET SIGNATURE GENERATION SWEEP - COMPLETED")
    logger.info("=" * 100)

    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = sum(1 for r in results if r['status'] == 'failed')

    logger.info(f"Total assets: {len(assets)}")
    logger.info(f"Total velocity windows per asset: {len(velocity_windows)}")
    logger.info(f"Total combinations: {total_combinations}")
    logger.info(f"Successful assets: {success_count}")
    logger.info(f"Failed assets: {failed_count}")
    logger.info("")

    if failed_count > 0:
        logger.info("Failed assets:")
        for result in results:
            if result['status'] == 'failed':
                logger.info(f"  - {result['asset']}: {result['message']}")
        logger.info("")

    logger.info("Output structure:")
    for asset in assets:
        logger.info(f"  {asset}:")
        signatures_dir = project_root / f"data_{asset}" / "signatures"
        for window in velocity_windows:
            train_test_file = signatures_dir / f"signatures_max_dc_vw{window}_train_test.parquet"
            oos_file = signatures_dir / f"signatures_max_dc_vw{window}_oos.parquet"
            if train_test_file.exists() and oos_file.exists():
                logger.info(f"    ✓ vw{window}: train_test + oos")
            else:
                logger.info(f"    ✗ vw{window}: MISSING")

    logger.info("=" * 100)

    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run signature generation sweep for all configured assets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run sequentially (default) - safer for large computations
  python src/02_signatures/run_all_assets_signature_sweep.py

  # Run in parallel - faster but uses more resources
  python src/02_signatures/run_all_assets_signature_sweep.py --parallel

Output (for 6 assets × 6 velocity windows):
  data_EURUSD/signatures/signatures_max_dc_vw10_train_test.parquet
  data_EURUSD/signatures/signatures_max_dc_vw10_oos.parquet
  ...
  data_GBPUSD/signatures/signatures_max_dc_vw10_train_test.parquet
  data_GBPUSD/signatures/signatures_max_dc_vw10_oos.parquet
  ...
  (72 total files: 6 assets × 6 windows × 2 splits)
        """
    )

    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run assets in parallel (default: sequential)'
    )

    args = parser.parse_args()

    try:
        run_all_assets_signature_sweep(parallel=args.parallel)
    except Exception as e:
        logger.error(f"Multi-asset signature sweep failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
