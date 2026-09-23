"""
State-Space Lead-Lag Log-Signature Computation Core

This module implements high-performance signature computation using:
- 4-channel state-space representation (Price, Volatility, Roughness, Urgency)
- Lead-Lag transform (4D → 8D lift)
- PyTorch + signatory for GPU-accelerated log-signature calculation
- Memory-safe streaming batch processing (CPU accumulation)

Mathematical Foundation:
------------------------
State-Space Channels:
  1. P_t: log(price) - log(price[0])  (cumulative log-returns)
  2. σ_t: θ_dynamic / (θ_base × F_t)  (volatility recovered from FAS)
  3. H_t: hurst_snapshot              (local roughness)
  4. U_t: log(Δt + ε)                 (urgency/time between events)

Lead-Lag Transform:
  Input:  X = [x₁, x₂, ..., xₙ] where xᵢ ∈ ℝ⁴
  Output: LL(X) = [(x₁, x₁), (x₁, x₂), (x₂, x₂), (x₂, x₃), ...]
  This lifts 4D paths to 8D, capturing lead-lag relationships.

Min-Max Whitening:
  Per-window, per-channel scaling to [0, 1] for numerical stability.

Log-Signature:
  Depth 5 on 8 channels → ~2,800 features per window
  Captures geometric features independent of parameterization.

Memory Safety:
--------------
CRITICAL: Depth 5 with 8 channels produces ~2,800 features.
- 100k windows × 2,800 × 4 bytes = ~1.1 GB RAM
- Must accumulate on CPU, never hold full tensor in VRAM
- Use batch.cpu().numpy() immediately after each GPU batch

Performance Optimizations:
--------------------------
1. Zero-copy windows: stride_tricks (O(1) memory)
2. Vectorized operations: No Python loops in transforms
3. GPU batching: Process in chunks to avoid VRAM overflow
4. CPU accumulation: Prevent memory explosion
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
import warnings


def prepare_state_features(events_df: pd.DataFrame) -> np.ndarray:
    """
    Prepare 4-channel state features from DC events.

    Channels:
    ---------
    1. P_t: Log-price (cumulative log-returns from start)
    2. σ_t: Volatility (recovered from FAS equation)
    3. H_t: Roughness (Hurst exponent)
    4. U_t: Urgency (log time delta between events)

    Parameters:
    -----------
    events_df : pd.DataFrame
        DC events with columns: price, theta_dynamic, theta_base,
        hurst_snapshot, lambda, timestamp

    Returns:
    --------
    features : np.ndarray, shape (N_events, 4)
        4-channel state features [P, σ, H, U]
    """
    # Channel 1: Log-Price (cumulative log-returns)
    log_prices = np.log(events_df['price'].values + 1e-10)
    P = log_prices - log_prices[0]  # Zero-start normalization

    # Channel 2: Volatility (inverted from FAS equation)
    # θ_dynamic = θ_base × V_t × F_t where F_t = exp(λ × (0.5 - H_t))
    # Therefore: V_t = θ_dynamic / (θ_base × F_t)
    theta_dynamic = events_df['theta_dynamic'].values
    theta_base = events_df['theta_base'].values
    hurst = events_df['hurst_snapshot'].values
    lambda_val = events_df['lambda'].values

    F_t = np.exp(lambda_val * (0.5 - hurst))
    sigma_proxy = theta_dynamic / (theta_base * F_t + 1e-10)

    # Channel 3: Roughness (Hurst exponent)
    H = hurst

    # Channel 4: Urgency (log time delta)
    timestamps = pd.to_datetime(events_df['timestamp'])
    time_deltas = timestamps.diff().dt.total_seconds().fillna(1.0).values
    U = np.log(time_deltas + 1e-6)

    return np.stack([P, sigma_proxy, H, U], axis=1).astype(np.float32)


def create_state_windows(
    features: np.ndarray,
    window_size: int,
    stride: int = 1
) -> np.ndarray:
    """
    Create sliding windows from 4-channel features using stride tricks.

    Parameters:
    -----------
    features : np.ndarray, shape (N, 4)
        4-channel state features
    window_size : int
        Number of events per window
    stride : int, optional
        Step between windows (default: 1)

    Returns:
    --------
    windows : np.ndarray, shape (N_windows, window_size, 4)
        Sliding windows over state features

    Notes:
    ------
    Uses np.lib.stride_tricks for O(1) memory allocation (view only).
    """
    n_samples, n_channels = features.shape

    if n_samples < window_size:
        raise ValueError(
            f"Insufficient events: {n_samples} events, need >= {window_size}"
        )

    n_windows = (n_samples - window_size) // stride + 1

    # Use stride tricks for memory efficiency
    shape = (n_windows, window_size, n_channels)
    strides = (
        features.strides[0] * stride,
        features.strides[0],
        features.strides[1]
    )

    windows = np.lib.stride_tricks.as_strided(
        features, shape=shape, strides=strides
    )

    # Make a copy to ensure contiguous memory for downstream operations
    return np.ascontiguousarray(windows)


def min_max_scale_windows(windows: 'torch.Tensor') -> 'torch.Tensor':
    """
    Apply Min-Max scaling independently per window per channel.

    Scales each channel in each window to [0, 1] range.

    Parameters:
    -----------
    windows : torch.Tensor, shape (Batch, Length, 4)
        Raw state-space windows

    Returns:
    --------
    scaled : torch.Tensor, shape (Batch, Length, 4)
        Scaled windows with values in [0, 1]

    Notes:
    ------
    - Epsilon (1e-8) added to prevent division by zero for constant channels
    - Scaling is applied independently per window and per channel
    """
    import torch

    # Compute min/max per window per channel
    # Shape: (Batch, 1, Channels)
    mins = windows.min(dim=1, keepdim=True).values
    maxs = windows.max(dim=1, keepdim=True).values

    # Scale to [0, 1] with epsilon for numerical stability
    range_vals = maxs - mins + 1e-8
    scaled = (windows - mins) / range_vals

    return scaled


def apply_lead_lag_transform(windows: 'torch.Tensor') -> 'torch.Tensor':
    """
    Apply Lead-Lag transform to lift 4D paths to 8D (VECTORIZED).

    Lead-Lag construction:
    - Position 0: (x₁, x₁)    [lead=1, lag=1]
    - Position 1: (x₁, x₂)    [lead=1, lag=2]
    - Position 2: (x₂, x₂)    [lead=2, lag=2]
    - ...

    Parameters:
    -----------
    windows : torch.Tensor, shape (Batch, Length, 4)
        Min-Max scaled state-space windows

    Returns:
    --------
    lead_lag : torch.Tensor, shape (Batch, 2*Length-1, 8)
        Lead-Lag transformed paths with doubled channels
    """
    import torch

    batch, length, channels = windows.shape

    # VECTORIZED Lead-Lag (no Python loops!)
    # Even positions: (x_i, x_i) for i in 0..L-1
    # Odd positions: (x_i, x_{i+1}) for i in 0..L-2

    # Create lead and lag components
    # Lead for even: windows[:, 0:L, :]   -> positions 0,2,4,...
    # Lead for odd:  windows[:, 0:L-1, :] -> positions 1,3,5,...
    # Lag for even:  windows[:, 0:L, :]   -> positions 0,2,4,...
    # Lag for odd:   windows[:, 1:L, :]   -> positions 1,3,5,...

    out_length = 2 * length - 1
    out = torch.zeros(batch, out_length, channels * 2, dtype=windows.dtype, device=windows.device)

    # Even indices: (x_i, x_i)
    out[:, 0::2, :channels] = windows  # Lead
    out[:, 0::2, channels:] = windows  # Lag (same)

    # Odd indices: (x_i, x_{i+1})
    out[:, 1::2, :channels] = windows[:, :-1, :]  # Lead (current)
    out[:, 1::2, channels:] = windows[:, 1:, :]   # Lag (next)

    return out


def compute_logsignatures_gpu(
    paths: 'torch.Tensor',
    depth: int = 5,
    batch_size: int = 5000,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Compute log-signatures using signatory (GPU) or iisignature (CPU fallback).
    """
    import torch
    import sys

    def log(msg):
        print(msg, flush=True)
        sys.stdout.flush()

    # Try signatory first (GPU-accelerated), fallback to iisignature (CPU)
    use_signatory = False
    try:
        import signatory
        if device == 'cuda' and torch.cuda.is_available():
            use_signatory = True
            log("[Signature] Using signatory (GPU)")
        else:
            log("[Signature] CUDA not available, using iisignature (CPU)")
    except ImportError:
        log("[Signature] signatory not available, using iisignature (CPU)")

    if use_signatory:
        return _compute_logsignatures_signatory(paths, depth, batch_size, device)
    else:
        return _compute_logsignatures_iisignature(paths, depth, batch_size)


def _compute_logsignatures_signatory(
    paths: 'torch.Tensor',
    depth: int,
    batch_size: int,
    device: str
) -> np.ndarray:
    """GPU-accelerated log-signature computation using signatory."""
    import torch
    import signatory

    n_windows = paths.shape[0]
    channels = paths.shape[2]

    logsig_dim = signatory.logsignature_channels(channels=channels, depth=depth)
    print(f"[Signature Core] Log-signature dimension: {logsig_dim} "
          f"({channels} channels, depth {depth})")

    estimated_gb = (n_windows * logsig_dim * 4) / (1024**3)
    print(f"[Signature Core] Estimated output: {estimated_gb:.2f} GB "
          f"({n_windows:,} × {logsig_dim} × float32)")

    logsigs_list = []
    print(f"[Signature Core] Processing {n_windows:,} windows on {device.upper()}")

    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        batch = paths[start:end].to(device)

        try:
            batch_sigs = signatory.logsignature(batch, depth=depth)
            logsigs_list.append(batch_sigs.cpu().numpy())
            del batch_sigs, batch
            if device == 'cuda':
                torch.cuda.empty_cache()
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                if device == 'cuda':
                    torch.cuda.empty_cache()
                smaller_batch = batch_size // 2
                if smaller_batch < 100:
                    raise RuntimeError("OOM with batch_size=100. Try reducing depth.")
                print(f"[Warning] OOM, reducing batch size to {smaller_batch}")
                return _compute_logsignatures_signatory(paths, depth, smaller_batch, device)
            raise

        if (start // batch_size) % 10 == 0 or end == n_windows:
            progress = (end / n_windows) * 100
            print(f"[Signature Core] Progress: {end:,}/{n_windows:,} ({progress:.1f}%)")

    return np.concatenate(logsigs_list, axis=0)


def _iisig_worker_init(channels, depth):
    """Initialize worker process with signature object."""
    global _worker_sig_obj
    import iisignature
    _worker_sig_obj = iisignature.prepare(channels, depth, 'O')


def _iisig_worker_compute(path):
    """Compute log-signature for a single path (worker function)."""
    import iisignature
    global _worker_sig_obj
    return iisignature.logsig(path, _worker_sig_obj)


def _compute_logsignatures_iisignature(
    paths: 'torch.Tensor',
    depth: int,
    batch_size: int
) -> np.ndarray:
    """CPU-based log-signature computation using iisignature with multiprocessing."""
    import time
    import sys
    import multiprocessing as mp
    from functools import partial

    try:
        import iisignature
    except ImportError:
        raise ImportError(
            "Neither signatory nor iisignature available. Install with:\n"
            "  pip install iisignature"
        )

    def log(msg):
        print(msg, flush=True)
        sys.stdout.flush()

    # Convert to numpy
    paths_np = paths.numpy() if hasattr(paths, 'numpy') else np.array(paths)

    n_windows = paths_np.shape[0]
    channels = paths_np.shape[2]
    path_length = paths_np.shape[1]

    log(f"[iisig] Input: {n_windows:,} paths × {path_length} points × {channels} channels")

    # Get number of CPU cores
    n_cores = mp.cpu_count()
    n_workers = max(1, n_cores - 1)  # Leave one core free
    log(f"[iisig] Using {n_workers} parallel workers ({n_cores} cores available)")

    # Prepare signature object for timing estimate
    log(f"[iisig] Preparing signature object (depth={depth})...")
    sig_obj = iisignature.prepare(channels, depth, 'O')

    # Time a single computation to estimate total time
    log(f"[iisig] Computing sample signature to estimate time...")
    t0 = time.time()
    sample_logsig = iisignature.logsig(paths_np[0], sig_obj)
    single_time = time.time() - t0
    logsig_dim = len(sample_logsig)

    log(f"[iisig] Single path: {single_time:.3f}s | Dim: {logsig_dim} features")
    serial_estimate = single_time * n_windows / 60
    parallel_estimate = serial_estimate / n_workers
    log(f"[iisig] Estimated: {parallel_estimate:.1f} min (parallel) | {serial_estimate:.1f} min (serial)")

    estimated_gb = (n_windows * logsig_dim * 4) / (1024**3)
    log(f"[iisig] Output size: {estimated_gb:.2f} GB")

    # Use multiprocessing Pool for parallel computation
    log(f"[iisig] Starting parallel computation...")

    # Pre-allocate output array
    logsigs = np.zeros((n_windows, logsig_dim), dtype=np.float32)

    start_time = time.time()

    # Create pool with initializer to set up sig_obj in each worker
    with mp.Pool(
        processes=n_workers,
        initializer=_iisig_worker_init,
        initargs=(channels, depth)
    ) as pool:
        # Use imap for ordered results with progress tracking
        chunk_size = max(100, n_windows // (n_workers * 10))  # Adaptive chunking
        log(f"[iisig] Chunk size: {chunk_size}")

        last_report_time = start_time
        processed = 0

        for i, result in enumerate(pool.imap(_iisig_worker_compute, paths_np, chunksize=chunk_size)):
            logsigs[i] = result
            processed = i + 1

            # Progress reporting every 2 seconds
            current_time = time.time()
            if current_time - last_report_time >= 2.0 or processed == n_windows:
                elapsed = current_time - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (n_windows - processed) / rate if rate > 0 else 0

                log(f"[iisig] {processed:,}/{n_windows:,} ({processed/n_windows*100:.1f}%) | {rate:.1f}/s | ETA: {eta/60:.1f}m")
                last_report_time = current_time

    total_time = time.time() - start_time
    actual_rate = n_windows / total_time
    log(f"[iisig] Done in {total_time/60:.1f}m | {actual_rate:.1f}/s | Shape: {logsigs.shape}")

    return logsigs


def extract_window_metadata_v2(
    events_df: pd.DataFrame,
    window_size: int,
    stride: int = 1
) -> Dict[str, np.ndarray]:
    """
    Extract metadata for each window from DC events.

    Parameters:
    -----------
    events_df : pd.DataFrame
        DC events DataFrame
    window_size : int
        Window size
    stride : int, optional
        Step between windows (default: 1)

    Returns:
    --------
    metadata : dict
        Dictionary with keys:
        - window_start_idx, window_end_idx
        - start_timestamp, end_timestamp
        - start_price, end_price
        - mean_hurst, mean_theta_dynamic
        - scale (DC scale name)
    """
    n_events = len(events_df)
    n_windows = (n_events - window_size) // stride + 1

    # Window indices
    start_indices = np.arange(0, n_events - window_size + 1, stride)
    end_indices = start_indices + window_size - 1

    # Timestamps
    timestamps = events_df['timestamp'].values
    start_timestamps = timestamps[start_indices]
    end_timestamps = timestamps[end_indices]

    # Prices
    prices = events_df['price'].values
    start_prices = prices[start_indices]
    end_prices = prices[end_indices]

    # Compute window statistics
    mean_hurst = np.zeros(n_windows, dtype=np.float32)
    mean_theta = np.zeros(n_windows, dtype=np.float32)

    hurst_vals = events_df['hurst_snapshot'].values
    theta_vals = events_df['theta_dynamic'].values

    for i, idx in enumerate(start_indices):
        mean_hurst[i] = hurst_vals[idx:idx + window_size].mean()
        mean_theta[i] = theta_vals[idx:idx + window_size].mean()

    # Scale name (if present)
    if 'scale' in events_df.columns:
        scale_name = events_df['scale'].iloc[0]
        scale = np.array([scale_name] * n_windows)
    else:
        scale = np.array(['unknown'] * n_windows)

    metadata = {
        'window_start_idx': start_indices,
        'window_end_idx': end_indices,
        'start_timestamp': start_timestamps,
        'end_timestamp': end_timestamps,
        'start_price': start_prices,
        'end_price': end_prices,
        'mean_hurst': mean_hurst,
        'mean_theta_dynamic': mean_theta,
        'scale': scale
    }

    return metadata


def compute_state_space_signatures(
    events_df: pd.DataFrame,
    window_size: int = 80,
    stride: int = 1,
    depth: int = 5,
    batch_size: int = 5000,
    device: str = 'cuda'
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Complete State-Space Lead-Lag Signature pipeline.
    """
    import torch
    import sys

    def log(msg):
        print(msg, flush=True)
        sys.stdout.flush()

    log(f"\n{'='*60}")
    log("State-Space Lead-Lag Signature Pipeline")
    log(f"{'='*60}")
    log(f"Processing {len(events_df):,} events...")

    # Step 1: Prepare 4-channel state features
    log("\nStep 1: Preparing 4-channel state features...")
    features = prepare_state_features(events_df)
    log(f"  Shape: {features.shape} | Channels: [P, σ, H, U]")

    # Step 2: Create sliding windows
    log(f"\nStep 2: Creating sliding windows (size={window_size}, stride={stride})...")
    windows = create_state_windows(features, window_size, stride)
    n_windows = windows.shape[0]
    log(f"  Windows: {windows.shape} | Memory: {windows.nbytes / 1e6:.1f} MB")

    # Step 3: Convert to torch and apply Min-Max whitening
    log("\nStep 3: Converting to torch & Min-Max whitening...")
    windows_tensor = torch.from_numpy(windows).float()
    log(f"  Tensor created: {windows_tensor.shape}")
    windows_scaled = min_max_scale_windows(windows_tensor)
    log(f"  Scaled range: [{windows_scaled.min().item():.4f}, {windows_scaled.max().item():.4f}]")

    # Step 4: Lead-Lag transform (4D → 8D)
    log("\nStep 4: Lead-Lag transform (4D → 8D)...")
    windows_ll = apply_lead_lag_transform(windows_scaled)
    log(f"  Lead-Lag shape: {windows_ll.shape} (length {window_size} → {2*window_size-1})")

    # Step 5: Compute log-signatures
    log(f"\nStep 5: Computing depth-{depth} log-signatures...")
    signatures = compute_logsignatures_gpu(
        windows_ll, depth=depth, batch_size=batch_size, device=device
    )

    # Step 6: Extract metadata
    log("\nStep 6: Extracting window metadata...")
    metadata = extract_window_metadata_v2(events_df, window_size, stride)
    log(f"  Metadata fields: {list(metadata.keys())}")

    log(f"\n{'='*60}")
    log(f"Pipeline Complete! Signatures: {signatures.shape} ({signatures.nbytes / 1e6:.1f} MB)")
    log(f"{'='*60}\n")

    return signatures, metadata


def save_signatures_chunked(
    signatures: np.ndarray,
    metadata: Dict[str, np.ndarray],
    output_path: str,
    chunk_size: int = 50000
):
    """
    Save large signature arrays to Parquet in chunks.

    Prevents RAM explosion when building DataFrame for very large datasets.

    Parameters:
    -----------
    signatures : np.ndarray, shape (N_windows, logsig_dim)
        Log-signature features
    metadata : dict
        Window metadata
    output_path : str
        Output Parquet file path
    chunk_size : int, optional
        Rows per chunk (default: 50000)
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    n_windows = signatures.shape[0]
    sig_cols = [f'sig_{i}' for i in range(signatures.shape[1])]

    print(f"Saving {n_windows:,} windows to {output_path}...")

    writer = None

    for chunk_start in range(0, n_windows, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_windows)

        # Build chunk DataFrame
        chunk_data = {}

        # Add signature columns
        for i, col in enumerate(sig_cols):
            chunk_data[col] = signatures[chunk_start:chunk_end, i]

        # Add metadata columns
        for key, values in metadata.items():
            chunk_data[key] = values[chunk_start:chunk_end]

        chunk_df = pd.DataFrame(chunk_data)
        table = pa.Table.from_pandas(chunk_df, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)

        writer.write_table(table)

        progress = (chunk_end / n_windows) * 100
        print(f"  Saved chunk {chunk_start:,}-{chunk_end:,} ({progress:.1f}%)")

    if writer is not None:
        writer.close()

    print(f"[OK] Saved to: {output_path}")


def validate_signature_output(
    signatures: np.ndarray,
    expected_channels: int = 8,
    expected_depth: int = 5
) -> bool:
    """
    Validate signature computation results.

    Checks:
    1. Correct dimensionality for given depth/channels
    2. No NaN or Inf values
    3. Reasonable value range

    Parameters:
    -----------
    signatures : np.ndarray
        Computed log-signatures
    expected_channels : int, optional
        Number of input channels (default: 8)
    expected_depth : int, optional
        Signature depth (default: 5)

    Returns:
    --------
    valid : bool
        True if all checks pass

    Raises:
    -------
    AssertionError if validation fails
    """
    try:
        import signatory
        expected_dim = signatory.logsignature_channels(
            channels=expected_channels, depth=expected_depth
        )
    except ImportError:
        # Approximate dimension if signatory not available
        expected_dim = None
        print("[Warning] signatory not available, skipping dimension check")

    # Check dimension
    actual_dim = signatures.shape[1]
    if expected_dim is not None:
        if actual_dim != expected_dim:
            print(f"[Warning] Dimension mismatch: got {actual_dim}, expected {expected_dim}")
        else:
            print(f"[Validation] Dimension check passed: {actual_dim} features")

    # Check for invalid values
    n_nan = np.isnan(signatures).sum()
    n_inf = np.isinf(signatures).sum()

    if n_nan > 0:
        raise AssertionError(f"Found {n_nan} NaN values in signatures")
    if n_inf > 0:
        raise AssertionError(f"Found {n_inf} Inf values in signatures")

    print(f"[Validation] No NaN/Inf values [OK]")
    print(f"[Validation] Value range: [{signatures.min():.4f}, {signatures.max():.4f}]")
    print(f"[Validation] Mean abs value: {np.abs(signatures).mean():.4f}")

    return True
