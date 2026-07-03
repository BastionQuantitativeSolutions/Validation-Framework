#!/usr/bin/env python3
"""
generate_d1_parquets.py
=======================
Task 1 Engineering Fix — Generate D1 parquet files for all 14 live pairs.

Source: {SYMBOL}_M1.parquet (already present, DatetimeIndex, 5-year+ history)
Output: {SYMBOL}_D1.parquet — standard OHLCV + DatetimeIndex

Checklist requirement:
  REQUIRED_TFS includes "D1"; all 14 live pairs were failing this check
  because no D1 files existed in DATA_MODELS/data_parquet/.

Run:
    python CORE_MODULES/validation/generate_d1_parquets.py
    python CORE_MODULES/validation/generate_d1_parquets.py --overwrite   # force rebuild
    python CORE_MODULES/validation/generate_d1_parquets.py --pair EURUSD  # single pair
"""

import sys
import os
import argparse
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Root detection
# ---------------------------------------------------------------------------
def _find_root() -> Path:
    env_root = os.getenv("CAVALIER_ROOT") or os.getenv("PROJECT_ROOT")
    if env_root:
        candidate = Path(env_root).resolve()
        if (candidate / "DATA_MODELS" / "data_parquet").exists():
            return candidate
    for start in [Path(__file__).resolve().parent, Path.cwd()]:
        for parent in [start] + list(start.parents):
            if (
                (parent / "DATA_MODELS" / "data_parquet").exists()
                and (parent / "CORE_MODULES").exists()
            ):
                return parent
    return Path(__file__).resolve().parents[2]

ROOT     = _find_root()
DATA_DIR = ROOT / "DATA_MODELS" / "data_parquet"

# ---------------------------------------------------------------------------
# Canonical live pairs (matches pre_monday_checklist.py)
# ---------------------------------------------------------------------------
ALL_PAIRS = [
    "EURUSD", "GBPUSD", "XAUUSD", "USDJPY", "AUDUSD",
    "USDCAD", "USDCHF", "GBPCHF", "XAGUSD", "NZDUSD",
    "USOIL",  "UKOIL",  "US100",  "JP225",  "HK50",
    "UK100",  "BTCUSD", "ETHUSD",
]
QUARANTINED = {"USDJPY", "GBPCHF", "NZDUSD", "UKOIL"}
LIVE_PAIRS  = [p for p in ALL_PAIRS if p not in QUARANTINED]

# D1 resample rule — "calendar day" aligned, label = left edge
D1_FREQ = "D"

OHLCV_AGG = {
    "open":        "first",
    "high":        "max",
    "low":         "min",
    "close":       "last",
    "tick_volume": "sum",
}


def resample_m1_to_d1(pair: str, data_dir: Path) -> pd.DataFrame | None:
    """
    Load M1 parquet for `pair` and resample to D1.
    Returns a DataFrame with DatetimeIndex and columns [open, high, low, close, tick_volume].
    Returns None on failure.
    """
    m1_path = data_dir / f"{pair}_M1.parquet"
    if not m1_path.exists():
        print(f"  ✗  {pair}_M1.parquet not found — cannot generate D1")
        return None

    try:
        df = pd.read_parquet(m1_path)
    except Exception as e:
        print(f"  ✗  {pair}_M1.parquet read error: {e}")
        return None

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ["time", "date", "datetime", "timestamp"]:
            if col in df.columns:
                df = df.set_index(col)
                df.index = pd.to_datetime(df.index)
                break
        else:
            print(f"  ✗  {pair}_M1: no datetime index and no datetime column found")
            return None

    df = df.sort_index()

    # Normalise column names
    df.columns = [c.lower() for c in df.columns]

    # Only keep OHLCV columns; handle missing tick_volume gracefully
    available_agg = {k: v for k, v in OHLCV_AGG.items() if k in df.columns}
    if "close" not in available_agg:
        print(f"  ✗  {pair}_M1: no 'close' column — cannot resample")
        return None
    if "tick_volume" not in available_agg:
        df["tick_volume"] = 0
        available_agg["tick_volume"] = "sum"

    try:
        d1 = df.resample(D1_FREQ).agg(available_agg)
    except Exception as e:
        print(f"  ✗  {pair} resample failed: {e}")
        return None

    # Drop empty days (weekends / public holiday gaps)
    d1 = d1.dropna(subset=["close"])
    d1 = d1[d1["close"] > 0]

    # Ensure column order matches all other parquet files
    final_cols = ["open", "high", "low", "close", "tick_volume"]
    for col in final_cols:
        if col not in d1.columns:
            d1[col] = 0
    d1 = d1[final_cols]

    return d1


def main():
    parser = argparse.ArgumentParser(description="Generate D1 parquet files from M1 data")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing D1 files (default: skip if present)")
    parser.add_argument("--pair", type=str, default=None,
                        help="Process a single pair only (e.g. EURUSD)")
    args = parser.parse_args()

    pairs = [args.pair] if args.pair else LIVE_PAIRS

    print("=" * 66)
    print("  CAVALIER BASTION — D1 PARQUET GENERATOR")
    print(f"  ROOT      : {ROOT}")
    print(f"  DATA_DIR  : {DATA_DIR}")
    print(f"  Pairs     : {len(pairs)}")
    print(f"  Overwrite : {args.overwrite}")
    print("=" * 66)

    generated, skipped, failed = [], [], []
    t0 = time.time()

    for pair in pairs:
        d1_path = DATA_DIR / f"{pair}_D1.parquet"

        if d1_path.exists() and not args.overwrite:
            print(f"\n  SKIP  {pair}_D1.parquet already exists ({d1_path.stat().st_size // 1024} KB)")
            skipped.append(pair)
            continue

        print(f"\n  Building {pair}_D1 from M1...")
        d1 = resample_m1_to_d1(pair, DATA_DIR)
        if d1 is None:
            failed.append(pair)
            continue

        try:
            d1.to_parquet(d1_path, index=True)
            size_kb = d1_path.stat().st_size // 1024
            print(f"  ✓  {pair}_D1.parquet → {len(d1):,} daily bars | {size_kb} KB")
            print(f"     Range: {d1.index[0].date()} → {d1.index[-1].date()}")
            generated.append(pair)
        except Exception as e:
            print(f"  ✗  {pair}_D1 write failed: {e}")
            failed.append(pair)

    elapsed = time.time() - t0
    print(f"\n{'=' * 66}")
    print(f"  SUMMARY  ({elapsed:.1f}s)")
    print(f"{'=' * 66}")
    print(f"  Generated : {len(generated)} → {generated}")
    print(f"  Skipped   : {len(skipped)} (already exist)")
    print(f"  Failed    : {len(failed)} → {failed}")

    if not failed:
        print("\n  ✓  All D1 files ready — re-run pre_monday_checklist.py to verify")
    else:
        print(f"\n  ⚠  {len(failed)} pair(s) failed — check M1 source files")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
