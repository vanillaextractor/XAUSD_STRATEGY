#!/usr/bin/env python3
"""
Multi-Asset FAS Sampling Sweep

Runs FAS DC event sampling for all configured assets in parallel.

This script:
1. Reads asset list from config/config.yaml
2. Runs FAS sampling for each asset
3. Generates DC events in data_{ASSET}/dc_events/ folders

Output structure:
    data_EURUSD/dc_events/dc_events_max_dc.parquet
    data_GBPUSD/dc_events/dc_events_max_dc.parquet
    data_USDJPY/dc_events/dc_events_max_dc.parquet
    ... (for all configured assets)

Usage:
    python src/01_sampling/run_all_assets_sampling.py
    python src/01_sampling/run_all_assets_sampling.py --parallel  # Run in parallel

Author: Multi-Asset Pipeline
Date: 2026-02-01
"""

import sys
import yaml
import logging
import subprocess
from pathlib import Path
from typing import List
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


def run_sampling_for_asset(asset: str, script_path: Path) -> dict:
    """
    Run FAS sampling for a single asset.

    Args:
        asset: Asset symbol
        script_path: Path to run_fas_sampling.py

    Returns:
        Dictionary with asset name and status
    """
    logger.info(f"Starting FAS sampling for {asset}...")

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


def run_all_assets_sampling(parallel: bool = False):
    """
    Main entry point for multi-asset FAS sampling sweep.

    Args:
        parallel: If True, run assets in parallel. If False, run sequentially.
    """
    logger.info("=" * 100)
    logger.info("MULTI-ASSET FAS SAMPLING SWEEP - STARTED")
    logger.info("=" * 100)

    # Paths
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "config" / "config.yaml"
    script_path = Path(__file__).parent / "run_fas_sampling.py"

    # Load config and get asset list
    logger.info(f"Loading configuration from: {config_path}")
    config = load_config(str(config_path))

    assets = config.get('assets', ['EURUSD'])
    logger.info(f"Found {len(assets)} assets to process: {assets}")
    logger.info(f"Execution mode: {'PARALLEL' if parallel else 'SEQUENTIAL'}")
    logger.info("")

    results = []

    if parallel:
        # Parallel execution
        logger.info("Running FAS sampling in PARALLEL mode...")
        logger.info("-" * 100)

        with ProcessPoolExecutor(max_workers=min(len(assets), 4)) as executor:
            # Submit all tasks
            future_to_asset = {
                executor.submit(run_sampling_for_asset, asset, script_path): asset
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
        logger.info("Running FAS sampling in SEQUENTIAL mode...")
        logger.info("-" * 100)

        for idx, asset in enumerate(assets, 1):
            logger.info(f"\n[{idx}/{len(assets)}] Processing {asset}...")
            logger.info("-" * 80)

            result = run_sampling_for_asset(asset, script_path)
            results.append(result)

            logger.info("")

    # Summary
    logger.info("")
    logger.info("=" * 100)
    logger.info("MULTI-ASSET FAS SAMPLING SWEEP - COMPLETED")
    logger.info("=" * 100)

    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = sum(1 for r in results if r['status'] == 'failed')

    logger.info(f"Total assets: {len(assets)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info("")

    if failed_count > 0:
        logger.info("Failed assets:")
        for result in results:
            if result['status'] == 'failed':
                logger.info(f"  - {result['asset']}: {result['message']}")
        logger.info("")

    logger.info("Output structure:")
    for asset in assets:
        output_dir = project_root / f"data_{asset}" / "dc_events"
        logger.info(f"  {output_dir}/")

    logger.info("=" * 100)

    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run FAS sampling for all configured assets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run sequentially (default)
  python src/01_sampling/run_all_assets_sampling.py

  # Run in parallel
  python src/01_sampling/run_all_assets_sampling.py --parallel

Output:
  data_EURUSD/dc_events/dc_events_max_dc.parquet
  data_GBPUSD/dc_events/dc_events_max_dc.parquet
  data_USDJPY/dc_events/dc_events_max_dc.parquet
  ... (for all configured assets)
        """
    )

    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run assets in parallel (default: sequential)'
    )

    args = parser.parse_args()

    try:
        run_all_assets_sampling(parallel=args.parallel)
    except Exception as e:
        logger.error(f"Multi-asset sampling sweep failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
