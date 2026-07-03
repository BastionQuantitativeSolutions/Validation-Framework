#!/usr/bin/env python3
# Author: JG
"""
Generate missing timeframe data files from M1 data
Resamples M1 data to M5, M15, M30, H1, H4
"""

import os
import pandas as pd

DATA_DIR = "DATA_MODELS/data_parquet"
SYMBOLS = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD"]
TIMEFRAMES = ["M5", "M15", "M30", "H1", "H4"]


def resample_to_timeframe(df_m1, target_tf):
    """
    Resample M1 data to target timeframe
    """
    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "tick_volume": "sum",
    }

    if target_tf == "M5":
        freq = "5T"
    elif target_tf == "M15":
        freq = "15T"
    elif target_tf == "M30":
        freq = "30T"
    elif target_tf == "H1":
        freq = "1H"
    elif target_tf == "H4":
        freq = "4H"
    else:
        raise ValueError(f"Unknown timeframe: {target_tf}")

    df_resampled = df_m1.resample(freq).agg(agg_dict).dropna()
    df_resampled["time"] = df_resampled.index
    return df_resampled.reset_index(drop=True)


def generate_missing_timeframes():
    print("=" * 70)
    print("GENERATING MISSING TIMEFRAME DATA FILES")
    print("=" * 70)

    generated_count = 0
    skipped_count = 0

    for symbol in SYMBOLS:
        m1_file = os.path.join(DATA_DIR, f"{symbol}_M1.parquet")

        if not os.path.exists(m1_file):
            print(f"\nSKIP: {symbol}_M1.parquet not found")
            skipped_count += 6
            continue

        print(f"\nProcessing {symbol}...")
        df_m1 = pd.read_parquet(m1_file)

        if "time" in df_m1.columns:
            df_m1 = df_m1.set_index("time")
        df_m1 = df_m1.sort_index()

        print(f"  M1 loaded: {len(df_m1):,} rows")

        for tf in TIMEFRAMES:
            target_file = os.path.join(DATA_DIR, f"{symbol}_{tf}.parquet")

            if os.path.exists(target_file):
                print(f"  SKIP: {symbol}_{tf}.parquet already exists")
                skipped_count += 1
                continue

            print(f"  Generating {tf}...")
            df_resampled = resample_to_timeframe(df_m1, tf)

            df_resampled.to_parquet(target_file, index=False)
            print(f"    Saved: {len(df_resampled):,} rows to {target_file}")
            generated_count += 1

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Generated: {generated_count} files")
    print(f"Skipped: {skipped_count} files")
    print("\nAll timeframes ready for training.")


if __name__ == "__main__":
    generate_missing_timeframes()
