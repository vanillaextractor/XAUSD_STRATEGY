#!/usr/bin/env python3
"""
Convert CSV forex data files to Parquet format with standardized naming and schema.

Expected CSV format (MT5 standard):
- Columns: DATE, TIME, OPEN, HIGH, LOW, CLOSE, TICKVOL, VOL, SPREAD
- DATE format: YYYY.MM.DD
- TIME format: HH:MM:SS
- All price columns: float
- All volume/spread columns: int

Output naming convention: {SYMBOL}2017_2025_cleaned.parquet
Example: GBPUSD2017_2025_cleaned.parquet, USDJPY2017_2025_cleaned.parquet
"""

import pandas as pd
import argparse
from pathlib import Path
import sys


def validate_dataframe(df: pd.DataFrame, symbol: str) -> tuple[bool, str]:
    """
    Validate that DataFrame matches expected schema.

    Returns:
        (is_valid, error_message)
    """
    # Check required columns
    required_cols = ['DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL', 'VOL', 'SPREAD']
    missing_cols = set(required_cols) - set(df.columns)

    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"

    # Check data types
    price_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
    for col in price_cols:
        if df[col].dtype not in ['float64', 'float32']:
            return False, f"Column {col} must be float, got {df[col].dtype}"

    int_cols = ['TICKVOL', 'VOL', 'SPREAD']
    for col in int_cols:
        if df[col].dtype not in ['int64', 'int32']:
            return False, f"Column {col} must be int, got {df[col].dtype}"

    # Check DATE format
    if df['DATE'].dtype != 'object':
        return False, f"DATE column must be string/object, got {df['DATE'].dtype}"

    # Validate DATE format (YYYY.MM.DD)
    sample_date = df['DATE'].iloc[0]
    if not isinstance(sample_date, str) or sample_date.count('.') != 2:
        return False, f"DATE format must be YYYY.MM.DD, got: {sample_date}"

    # Check TIME format
    if df['TIME'].dtype != 'object':
        return False, f"TIME column must be string/object, got {df['TIME'].dtype}"

    # Validate TIME format (HH:MM:SS)
    sample_time = df['TIME'].iloc[0]
    if not isinstance(sample_time, str) or sample_time.count(':') != 2:
        return False, f"TIME format must be HH:MM:SS, got: {sample_time}"

    # Check for NaN values
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        return False, f"Found NaN values in columns: {nan_cols}"

    # OHLC consistency check (sample 1000 rows)
    sample = df.sample(min(1000, len(df)))
    invalid_ohlc = sample[
        (sample['HIGH'] < sample['LOW']) |
        (sample['HIGH'] < sample['OPEN']) |
        (sample['HIGH'] < sample['CLOSE']) |
        (sample['LOW'] > sample['OPEN']) |
        (sample['LOW'] > sample['CLOSE'])
    ]

    if len(invalid_ohlc) > 0:
        return False, f"Found {len(invalid_ohlc)} rows with invalid OHLC relationships"

    return True, "Valid"


def convert_csv_to_parquet(
    csv_path: str,
    symbol: str = None,
    year_start: int = 2017,
    year_end: int = 2025,
    dry_run: bool = False
) -> str:
    """
    Convert CSV file to Parquet with standardized naming.

    Creates parallel data_{SYMBOL} folders to keep existing pipeline code unchanged.

    Args:
        csv_path: Path to input CSV file
        symbol: Symbol name (e.g., GBPUSD, USDJPY). If None, extracted from filename
        year_start: Start year for naming convention
        year_end: End year for naming convention
        dry_run: If True, only validate without writing

    Returns:
        Path to output parquet file
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Extract symbol from filename if not provided
    if symbol is None:
        # Assume filename like GBPUSD_data.csv or GBPUSD2017_2025.csv
        filename = csv_path.stem  # Without extension
        # Extract first uppercase sequence (symbol)
        import re
        match = re.match(r'^([A-Z]{6})', filename)
        if match:
            symbol = match.group(1)
        else:
            raise ValueError(
                f"Could not extract symbol from filename: {filename}. "
                "Please provide --symbol argument."
            )

    symbol = symbol.upper()
    print(f"\n{'='*80}")
    print(f"Converting: {csv_path.name}")
    print(f"Symbol: {symbol}")
    print(f"{'='*80}")

    # Read CSV - try different separators and handle different formats
    print("\n1. Reading CSV file...")
    try:
        # First try reading with default comma separator
        df = pd.read_csv(csv_path)

        # If we got only one column, it might be tab-separated
        if len(df.columns) == 1:
            df = pd.read_csv(csv_path, sep='\t')

        # Clean column names - remove angle brackets and convert to uppercase
        df.columns = df.columns.str.replace('<', '').str.replace('>', '').str.upper()

        # Handle XAUUSD format (Date/Time instead of DATE/TIME, missing volume columns)
        if 'DATE' in df.columns and 'TICKVOL' not in df.columns:
            print("   ⚠️  Detected non-MT5 format - adding missing volume/spread columns with defaults")
            df['TICKVOL'] = 1
            df['VOL'] = 1000000
            df['SPREAD'] = 0

        # Remove any extra columns that aren't in the standard schema
        standard_columns = ['DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL', 'VOL', 'SPREAD']
        extra_cols = [col for col in df.columns if col not in standard_columns]
        if extra_cols:
            print(f"   ⚠️  Removing extra columns: {extra_cols}")
            df = df[standard_columns]

    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}")

    print(f"   Loaded {len(df):,} rows")
    print(f"   Columns: {df.columns.tolist()}")

    # Validate schema
    print("\n2. Validating data schema...")
    is_valid, message = validate_dataframe(df, symbol)

    if not is_valid:
        raise ValueError(f"Schema validation failed: {message}")

    print(f"   ✅ {message}")
    print(f"   Date range: {df['DATE'].iloc[0]} to {df['DATE'].iloc[-1]}")
    print(f"   Price range: {df['CLOSE'].min():.5f} - {df['CLOSE'].max():.5f}")

    # Data statistics
    print("\n3. Data statistics:")
    print(f"   Total bars: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"   OHLC dtypes: {df[['OPEN', 'HIGH', 'LOW', 'CLOSE']].dtypes.unique()}")
    print(f"   Volume dtypes: {df[['TICKVOL', 'VOL']].dtypes.unique()}")

    if dry_run:
        print("\n✅ DRY RUN: Validation passed. No file written.")
        return None

    # Create parallel data folder for this asset (e.g., data_GBPUSD)
    # This keeps existing code unchanged - each asset has its own data folder
    data_folder = f"data_{symbol}"
    raw_dir = Path(data_folder) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename (same naming as EURUSD)
    output_filename = f"{symbol}{year_start}_{year_end}_cleaned.parquet"
    output_path = raw_dir / output_filename

    print(f"\n4. Writing Parquet file...")
    print(f"   Output: {output_path}")

    # Write to parquet with compression
    df.to_parquet(
        output_path,
        engine='pyarrow',
        compression='snappy',
        index=False
    )

    # Verify output
    output_size_mb = output_path.stat().st_size / 1024**2
    compression_ratio = (df.memory_usage(deep=True).sum() / 1024**2) / output_size_mb

    print(f"   ✅ Saved: {output_size_mb:.2f} MB")
    print(f"   Compression ratio: {compression_ratio:.1f}x")

    # Verify by reading back
    print("\n5. Verifying output...")
    df_verify = pd.read_parquet(output_path)

    if len(df_verify) != len(df):
        raise ValueError(f"Row count mismatch after write: {len(df_verify)} vs {len(df)}")

    if not df_verify.columns.equals(df.columns):
        raise ValueError(f"Column mismatch after write")

    print(f"   ✅ Verified: {len(df_verify):,} rows")

    print(f"\n{'='*80}")
    print(f"✅ SUCCESS: {output_filename}")
    print(f"{'='*80}")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert forex CSV files to Parquet with standardized naming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file (auto-detect symbol from filename)
  python convert_csv_to_parquet.py GBPUSD_data.csv

  # Specify symbol explicitly
  python convert_csv_to_parquet.py data.csv --symbol USDJPY

  # Dry run (validate only, don't write)
  python convert_csv_to_parquet.py NZDUSD.csv --dry-run

  # Batch convert multiple files
  python convert_csv_to_parquet.py GBPUSD.csv USDJPY.csv AUDUSD.csv

Expected CSV format:
  Columns: DATE, TIME, OPEN, HIGH, LOW, CLOSE, TICKVOL, VOL, SPREAD
  DATE format: YYYY.MM.DD (e.g., 2017.01.02)
  TIME format: HH:MM:SS (e.g., 14:30:00)

Output (parallel data folders):
  - data_GBPUSD/raw/GBPUSD2017_2025_cleaned.parquet
  - data_USDJPY/raw/USDJPY2017_2025_cleaned.parquet
  - data_AUDUSD/raw/AUDUSD2017_2025_cleaned.parquet

  This keeps existing pipeline code unchanged - just point it to data_GBPUSD instead of data
        """
    )

    parser.add_argument(
        'csv_files',
        nargs='+',
        help='CSV file(s) to convert'
    )

    parser.add_argument(
        '--symbol',
        type=str,
        help='Symbol name (e.g., GBPUSD, USDJPY). If not provided, extracted from filename'
    )


    parser.add_argument(
        '--year-start',
        type=int,
        default=2017,
        help='Start year for naming (default: 2017)'
    )

    parser.add_argument(
        '--year-end',
        type=int,
        default=2025,
        help='End year for naming (default: 2025)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate only, do not write output file'
    )

    args = parser.parse_args()

    # Track results
    success_count = 0
    failed_files = []

    # Process each file
    for csv_file in args.csv_files:
        try:
            output_path = convert_csv_to_parquet(
                csv_path=csv_file,
                symbol=args.symbol if len(args.csv_files) == 1 else None,  # Only use explicit symbol for single file
                year_start=args.year_start,
                year_end=args.year_end,
                dry_run=args.dry_run
            )
            success_count += 1
        except Exception as e:
            print(f"\n❌ FAILED: {csv_file}")
            print(f"   Error: {e}")
            failed_files.append((csv_file, str(e)))

    # Summary
    print(f"\n{'='*80}")
    print("CONVERSION SUMMARY")
    print(f"{'='*80}")
    print(f"Total files: {len(args.csv_files)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_files)}")

    if failed_files:
        print(f"\nFailed files:")
        for file, error in failed_files:
            print(f"  - {file}: {error}")
        sys.exit(1)
    else:
        print(f"\n✅ All files converted successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()
