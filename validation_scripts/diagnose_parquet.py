"""
diagnose_parquet.py
===================
Quick sanity-check for one parquet file.
Prints: path, shape, columns, dtypes, NaN counts, and runs a mini feature check.

Usage:
    python CORE_MODULES/validation/diagnose_parquet.py
    python CORE_MODULES/validation/diagnose_parquet.py --pair GBPUSD --tf M15
"""
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
import argparse
from pathlib import Path

import pandas as pd

# ── locate root ───────────────────────────────────────────────────────────────
def _find_root() -> Path:
    for start in [Path(__file__).resolve().parent, Path.cwd()]:
        for p in [start] + list(start.parents):
            if (p / "DATA_MODELS" / "data_parquet").exists() and (p / "CORE_MODULES").exists():
                return p
    return Path(__file__).resolve().parents[2]   # <ROOT>/CORE_MODULES/validation/

parser = argparse.ArgumentParser()
parser.add_argument("--pair", default="EURUSD")
parser.add_argument("--tf",   default="M5")
args = parser.parse_args()

ROOT = _find_root()
parquet_path = ROOT / "DATA_MODELS" / "data_parquet" / f"{args.pair}_{args.tf}.parquet"

print("=" * 60)
print(f"  diagnose_parquet.py — {args.pair}_{args.tf}")
print("=" * 60)
print(f"  ROOT      : {ROOT}")
print(f"  DATA path : {parquet_path}")
print(f"  Exists    : {parquet_path.exists()}")
print()

if not parquet_path.exists():
    # Try to list what IS in data_parquet to help diagnose
    dp = ROOT / "DATA_MODELS" / "data_parquet"
    if dp.exists():
        files = sorted(dp.glob(f"{args.pair}*.parquet"))
        print(f"  Available {args.pair}* files in data_parquet:")
        for f in files:
            print(f"    {f.name}")
    else:
        print(f"  data_parquet directory does not exist at: {dp}")
    sys.exit(1)

df = pd.read_parquet(parquet_path)

print(f"  Shape     : {df.shape}")
print(f"  Index type: {type(df.index).__name__}")
print(f"  Columns   : {list(df.columns)}")
print()
print("  Dtypes:")
for col, dt in df.dtypes.items():
    print(f"    {col:30s} {dt}")
print()
print("  NaN counts per column:")
nan_counts = df.isna().sum()
for col, n in nan_counts.items():
    pct = 100 * n / len(df) if len(df) > 0 else 0
    flag = " ← ALL NaN!" if n == len(df) else (" ← HIGH" if pct > 50 else "")
    print(f"    {col:30s} {n:8,}  ({pct:.1f}%){flag}")
print()
print("  First 3 rows (close, high, low, open):")
show_cols = [c for c in ["open", "high", "low", "close", "tick_volume"] if c in df.columns]
print(df[show_cols].head(3).to_string())
print()
print("  Last 3 rows:")
print(df[show_cols].tail(3).to_string())
print()

# Quick feature check
df.columns = [c.lower() for c in df.columns]
if not isinstance(df.index, pd.DatetimeIndex):
    for col in ["time", "date", "datetime", "timestamp"]:
        if col in df.columns:
            df.index = pd.to_datetime(df[col])
            print(f"  Datetime index set from '{col}' column")
            break
    else:
        print("  WARNING: no datetime column found, keeping integer index")

df = df.sort_index().dropna(subset=["close"])
print(f"  After sort/dropna: {len(df)} rows")

# Attempt a mini compute_features
try:
    c = df["close"]
    h = df["high"]
    low_price = df["low"]
    sma20 = c.rolling(20).mean()
    sma200 = c.rolling(200).mean()
    rsi_delta = c.diff()
    gain = rsi_delta.clip(lower=0).rolling(14).mean()
    loss = (-rsi_delta.clip(upper=0)).rolling(14).mean()
    rsi14 = 100 - 100 / (1 + gain / (loss + 1e-10))

    valid_rows = sma200.notna() & rsi14.notna()
    print("  Mini feature check:")
    print(f"    sma_20  NaN rows : {sma20.isna().sum():,}")
    print(f"    sma_200 NaN rows : {sma200.isna().sum():,}")
    print(f"    rsi_14  NaN rows : {rsi14.isna().sum():,}")
    print(f"    Valid rows (sma_200 & rsi_14 both non-NaN): {valid_rows.sum():,}")
    if valid_rows.sum() > 0:
        print(f"\n  ✓ Feature computation is working — {valid_rows.sum():,} valid rows available")
    else:
        print("\n  ✗ Feature computation produces 0 valid rows — investigate column values")
except Exception as e:
    print(f"  Feature check error: {e}")

print("\n" + "=" * 60)
