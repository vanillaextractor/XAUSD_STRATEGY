"""
State-Space Lead-Lag Log-Signature Generation Script

This script transforms DC event time series into deep log-signature features
using the State-Space Lead-Lag pipeline with GPU acceleration.

Input:  dc_events_{scale}.parquet (from FAS DC detection)
Output: signatures_{scale}.parquet (log-signature features)

Pipeline:
---------
1. Load DC events with required columns
2. Prepare 4-channel state features (P, σ, H, U)
3. Create sliding windows
4. Apply Min-Max whitening
5. Apply Lead-Lag transform (4D → 8D)
6. Compute depth-5 log-signatures on GPU
7. Save to Parquet with metadata

Requirements:
-------------
- CUDA-capable GPU (falls back to CPU if unavailable)
- PyTorch with CUDA support
- signatory library (pip install signatory==1.2.6.1.9.0)
- 8GB+ GPU RAM recommended

Usage:
------
python generate_signatures.py [config.yaml]
"""

import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings

from signature_core import (
    compute_state_space_signatures,
    save_signatures_chunked,
    validate_signature_output
)

# ============================================================
# Project Paths (Auto-detect from script location)
# ============================================================
def get_project_root():
    """Get the project root directory (auto-detect from script location)."""
    # This script is in: Model_backtest/src/sampling/generate_signatures.py
    # Project root is: Model_backtest/
    script_dir = Path(__file__).parent  # src/sampling/
    return script_dir.parent.parent  # Model_backtest/


# Default config path (can be overridden via command line)
CONFIG_PATH = None  # Will be set to project_root/config.yaml if not provided


def load_configuration(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    print(f"\n{'='*60}")
    print("State-Space Lead-Lag Log-Signature Generation")
    print(f"{'='*60}\n")

    print(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("[OK] Configuration loaded\n")

    return config


def check_hardware():
    """Check hardware configuration and return device."""
    print(f"{'='*60}")
    print("Hardware Check")
    print(f"{'='*60}")

    device = 'cpu'

    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[OK] CUDA available: {device_name}")
            print(f"     VRAM: {vram:.1f} GB")
            device = 'cuda'
        else:
            print("[Info] CUDA not available")
    except ImportError:
        print("[Warning] PyTorch not installed")

    # Check for signature libraries
    sig_lib = None
    try:
        import signatory
        print(f"[OK] signatory version: {signatory.__version__}")
        sig_lib = 'signatory'
    except ImportError:
        pass

    if sig_lib is None:
        try:
            import iisignature
            print(f"[OK] iisignature version: {iisignature.version()}")
            sig_lib = 'iisignature'
            device = 'cpu'  # iisignature is CPU-only
        except ImportError:
            pass

    if sig_lib is None:
        raise ImportError(
            "No signature library available. Install one with:\n"
            "  pip install iisignature  (CPU, easier to install)\n"
            "  pip install signatory    (GPU, requires CUDA)"
        )

    print(f"[OK] Using {sig_lib} on {device.upper()}")
    print(f"{'='*60}\n")
    return device


def load_dc_events(parquet_path: str) -> pd.DataFrame:
    """
    Load DC events from Parquet with required columns for state-space.

    Required columns:
    - timestamp: Event timestamp
    - price: Event price
    - theta_dynamic: Dynamic threshold at event
    - theta_base: Base threshold for scale
    - hurst_snapshot: Local Hurst exponent
    - lambda: Lambda (aggression) parameter

    Parameters:
    -----------
    parquet_path : str
        Path to DC events parquet file

    Returns:
    --------
    pd.DataFrame
        DC events with required columns
    """
    print(f"Loading DC events from {parquet_path}...")

    df = pd.read_parquet(parquet_path)

    required_cols = [
        'timestamp', 'price', 'theta_dynamic',
        'hurst_snapshot', 'theta_base', 'lambda'
    ]

    # Check for required columns
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"[OK] Loaded {len(df):,} events")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    if 'scale' in df.columns:
        print(f"  Scale: {df['scale'].iloc[0]}")

    if 'type' in df.columns:
        print(f"  Event types: {df['type'].value_counts().to_dict()}")

    print()
    return df


def process_single_scale(
    scale_name: str,
    config: dict,
    device: str,
    input_dir: str = ".",
    output_dir: str = ".",
    output_suffix: str = "",
    asset: str = "EURUSD"
) -> pd.DataFrame:
    """
    Process a single DC scale: generate state-space signatures.

    Parameters:
    -----------
    scale_name : str
        Name of scale (e.g., "small_dc", "mid_dc", "max_dc")
    config : dict
        Configuration dictionary
    device : str
        'cuda' or 'cpu'
    input_dir : str
        Directory containing DC events parquet files
    output_dir : str
        Directory for output signature files

    Returns:
    --------
    pd.DataFrame
        Features DataFrame with signatures and metadata
    """
    print(f"\n{'='*60}")
    print(f"Processing {scale_name}")
    print(f"{'='*60}\n")

    start_time = datetime.now()

    # Load events
    input_path = Path(input_dir) / f"dc_events_{scale_name}.parquet"

    if not input_path.exists():
        raise FileNotFoundError(f"DC events file not found: {input_path}")

    events_df = load_dc_events(input_path)

    # Get scale-specific window size from config
    sampler_cfg = config.get('sampler', {})
    scales_cfg = sampler_cfg.get('scales', {})
    scale_cfg = scales_cfg.get(scale_name, {})

    window_size = scale_cfg.get('window', 80)

    # Signature parameters
    depth = 4  # Depth 4 for faster computation (~500 features vs ~2800 at depth 5)
    batch_size = 5000

    print(f"Configuration:")
    print(f"  Window size: {window_size}")
    print(f"  Signature depth: {depth}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device}")
    print()

    # Check if enough events
    if len(events_df) < window_size:
        warnings.warn(
            f"Insufficient events for {scale_name}: "
            f"{len(events_df)} events, need >= {window_size}. Skipping."
        )
        return None

    # Compute state-space signatures
    signatures, metadata = compute_state_space_signatures(
        events_df,
        window_size=window_size,
        stride=1,
        depth=depth,
        batch_size=batch_size,
        device=device
    )

    # Validate output
    print("Validating output...")
    validate_signature_output(signatures, expected_channels=8, expected_depth=depth)

    # Build output DataFrame
    print("\nBuilding features DataFrame...")
    n_windows = signatures.shape[0]
    sig_dim = signatures.shape[1]

    # For large datasets, use chunked saving
    # Include suffix if provided
    if output_suffix:
        output_path = Path(output_dir) / f"signatures_{scale_name}_{output_suffix}.parquet"
    else:
        output_path = Path(output_dir) / f"signatures_{scale_name}.parquet"

    if n_windows > 100000:
        print(f"Large dataset ({n_windows:,} windows), using chunked save...")
        save_signatures_chunked(signatures, metadata, str(output_path))
    else:
        # Standard save for smaller datasets
        sig_cols = [f'sig_{i}' for i in range(sig_dim)]
        features_df = pd.DataFrame(signatures, columns=sig_cols)

        # Add metadata columns
        for key, values in metadata.items():
            features_df[key] = values

        features_df.to_parquet(output_path, compression='snappy', index=False)
        print(f"[OK] Saved to: {output_path}")

    # Load back to verify and get stats
    features_df = pd.read_parquet(output_path)
    file_size = output_path.stat().st_size / 1e6

    # Print processing time
    elapsed = (datetime.now() - start_time).total_seconds()
    throughput = n_windows / elapsed

    print(f"\n{'='*60}")
    print(f"Processing Summary for {scale_name}")
    print(f"{'='*60}")
    print(f"  Events processed: {len(events_df):,}")
    print(f"  Windows generated: {n_windows:,}")
    print(f"  Signature dimension: {sig_dim}")
    print(f"  Processing time: {elapsed:.1f}s")
    print(f"  Throughput: {throughput:,.0f} windows/second")
    print(f"  Output shape: {features_df.shape}")
    print(f"  File size: {file_size:.1f} MB")
    print(f"{'='*60}\n")

    return features_df


def main(config_path: str = None, output_suffix: str = "", asset: str = "EURUSD"):
    """
    Main processing pipeline.

    Args:
        config_path: Path to config YAML file
        output_suffix: Suffix to add to output filenames (e.g., 'vw15')
        asset: Asset symbol (e.g., EURUSD, GBPUSD)
    """
    overall_start = datetime.now()

    # Use project root config if not provided
    if config_path is None:
        root = get_project_root()
        config_path = root / "config" / "config.yaml"

    # Load configuration
    config = load_configuration(str(config_path))

    # Check hardware
    device = check_hardware()

    # Get scales from config
    sampler_cfg = config.get('sampler', {})
    scales_cfg = sampler_cfg.get('scales', {})

    if not scales_cfg:
        print("[Error] No scales found in config")
        return

    # ONLY PROCESS MAX_DC SCALE
    scale_names = ['max_dc'] if 'max_dc' in scales_cfg else []

    if not scale_names:
        print("[Error] max_dc scale not found in config")
        return

    print(f"{'='*60}")
    print(f"Processing {len(scale_names)} DC scale for {asset} (max_dc only)")
    print(f"{'='*60}")
    print(f"Asset: {asset}")
    print(f"Scales: {', '.join(scale_names)}\n")

    # Get directories - asset-specific paths
    root = get_project_root()
    input_dir = root / f'data_{asset}' / 'dc_events'
    output_dir = root / f'data_{asset}' / 'signatures'

    # Fallback to current dir if new structure doesn't exist
    if not input_dir.exists():
        input_dir = Path('.')
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}\n")

    # Process each scale
    results = {}

    for scale_name in scale_names:
        try:
            features_df = process_single_scale(
                scale_name, config, device,
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                output_suffix=output_suffix,
                asset=asset
            )

            if features_df is not None:
                sig_cols = [c for c in features_df.columns if c.startswith('sig_')]
                results[scale_name] = {
                    'n_windows': len(features_df),
                    'n_features': len(sig_cols),
                    'total_cols': features_df.shape[1]
                }
        except FileNotFoundError as e:
            print(f"\n[Warning] Skipping {scale_name}: {e}")
            continue
        except Exception as e:
            print(f"\n[ERROR] Failed to process {scale_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final summary
    overall_elapsed = (datetime.now() - overall_start).total_seconds()

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total processing time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} min)")
    print(f"\nResults:")

    for scale_name, stats in results.items():
        print(f"  {scale_name}:")
        print(f"    Windows: {stats['n_windows']:,}")
        print(f"    Signature features: {stats['n_features']}")
        print(f"    Total columns: {stats['total_cols']}")

    print(f"\n{'='*60}")
    print("[OK] All processing complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate path signatures from DC events")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file (default: use hardcoded path)")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Suffix to add to output filenames (e.g., 'vw15')")
    parser.add_argument("--asset", type=str, default="EURUSD",
                        help="Asset symbol (e.g., EURUSD, GBPUSD)")

    args = parser.parse_args()

    main(config_path=args.config, output_suffix=args.output_suffix, asset=args.asset)
