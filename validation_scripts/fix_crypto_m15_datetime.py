#!/usr/bin/env python3
"""
fix_crypto_m15_datetime.py
==========================
Task 2 Engineering Fix — Repair BTCUSD_M15 and ETHUSD_M15 datetime index.

ROOT CAUSE:
  The augmentation pipeline that produced BTCUSD_M15.parquet and
  ETHUSD_M15.parquet stripped the DatetimeIndex and reset to a RangeIndex.
  The _original.parquet variants retain the correct DatetimeIndex.
  The augmented files contain 60 pre-computed feature columns (returns, RSI, etc.)
  that are incompatible with validate_v10_5_0.py, which recomputes features
  internally from raw OHLCV columns. This is why integer folding occurred:
  the walk-forward engine could not detect a temporal boundary.

FIX STRATEGY:
  1. Read the _original.parquet (DatetimeIndex, clean OHLCV, 5 columns)
  2. Extend via BTCUSD_M1 / ETHUSD_M1 resampling to cover any gap
  3. Write back as {SYMBOL}_M15.parquet with DatetimeIndex preserved

The _original files cover only ~6,720 rows (2026-01-12 to ~2026-04-03).
BTCUSD_M1 and ETHUSD_M1 also only cover Jan–Apr 2026 (100k M1 bars each).
This is thin but correct — the walk-forward engine will use what's available.

Run:
    python CORE_MODULES/validation/fix_crypto_m15_datetime.py
    python CORE_MODULES/validation/fix_crypto_m15_datetime.py --dry-run
"""

import sys
import os
import argparse
import shutil
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

CRYPTO_PAIRS = ["BTCUSD", "ETHUSD"]

OHLCV_AGG = {
    "open":        "first",
    "high":        "max",
    "low":         "min",
    "close":       "last",
    "tick_volume": "sum",
}

OHLCV_COLS = ["open", "high", "low", "close", "tick_volume"]


def load_and_fix(pair: str, data_dir: Path, dry_run: bool) -> bool:
    """Rebuild SYMBOL_M15.parquet with proper DatetimeIndex from original + M1."""
    orig_path   = data_dir / f"{pair}_M15_original.parquet"
    m15_path    = data_dir / f"{pair}_M15.parquet"
    m1_path     = data_dir / f"{pair}_M1.parquet"
    backup_path = data_dir / f"{pair}_M15_augmented_backup.parquet"

    print(f"\n  {'='*58}")
    print(f"  {pair} — M15 DatetimeIndex Repair")
    print(f"  {'='*58}")

    # ── Step 1: Load the original clean file ──────────────────────────────────
    if not orig_path.exists():
        print(f"  ✗  {pair}_M15_original.parquet not found — cannot repair")
        return False

    df_orig = pd.read_parquet(orig_path)
    print(f"  Original  : {orig_path.name} → {len(df_orig):,} rows | "
          f"index={type(df_orig.index).__name__}")

    # Ensure the original has a DatetimeIndex
    if not isinstance(df_orig.index, pd.DatetimeIndex):
        # Try to recover from 'time' column inside the original
        for col in ["time", "date", "datetime", "timestamp"]:
            if col in df_orig.columns:
                df_orig.index = pd.to_datetime(df_orig[col])
                df_orig = df_orig.drop(columns=[col])
                print(f"  ⚑  Recovered datetime from '{col}' column in original")
                break
        else:
            print(f"  ✗  {pair}_M15_original has no datetime index — repair aborted")
            return False

    df_orig = df_orig.sort_index()
    # Keep only OHLCV
    for col in OHLCV_COLS:
        if col not in df_orig.columns:
            df_orig[col] = 0
    df_orig = df_orig[OHLCV_COLS]

    # ── Step 2: Extend via M1 resampling (fills any gap after original end) ───
    df_combined = df_orig.copy()

    if m1_path.exists():
        print(f"  Loading   : {pair}_M1.parquet for extension...")
        df_m1 = pd.read_parquet(m1_path)
        df_m1.columns = [c.lower() for c in df_m1.columns]

        if not isinstance(df_m1.index, pd.DatetimeIndex):
            for col in ["time", "date", "datetime", "timestamp"]:
                if col in df_m1.columns:
                    df_m1.index = pd.to_datetime(df_m1[col])
                    df_m1 = df_m1.drop(columns=[col], errors="ignore")
                    break

        df_m1 = df_m1.sort_index()

        # Only resample M1 rows that are AFTER the original data ends
        orig_end = df_orig.index[-1]
        df_m1_new = df_m1[df_m1.index > orig_end]

        if len(df_m1_new) > 0:
            available_agg = {k: v for k, v in OHLCV_AGG.items() if k in df_m1_new.columns}
            if "close" in available_agg:
                if "tick_volume" not in available_agg:
                    df_m1_new["tick_volume"] = 0
                    available_agg["tick_volume"] = "sum"

                df_m1_resampled = df_m1_new.resample("15min").agg(available_agg).dropna(subset=["close"])
                df_m1_resampled = df_m1_resampled[df_m1_resampled["close"] > 0]

                for col in OHLCV_COLS:
                    if col not in df_m1_resampled.columns:
                        df_m1_resampled[col] = 0
                df_m1_resampled = df_m1_resampled[OHLCV_COLS]

                df_combined = pd.concat([df_orig, df_m1_resampled]).sort_index()
                df_combined = df_combined[~df_combined.index.duplicated(keep="last")]
                print(f"  Extended  : +{len(df_m1_resampled):,} bars from M1 "
                      f"({df_m1_resampled.index[0].date()} → {df_m1_resampled.index[-1].date()})")
            else:
                print("  ⚠  M1 data has no 'close' column — using original only")
        else:
            print(f"  ✓  No M1 data beyond original end ({orig_end.date()}) — original is current")
    else:
        print(f"  ⚠  {pair}_M1.parquet not found — using original data only")

    print(f"  Combined  : {len(df_combined):,} bars | "
          f"{df_combined.index[0].date()} → {df_combined.index[-1].date()}")
    print(f"  Index     : {type(df_combined.index).__name__} ✓")

    # ── Step 3: Verify ────────────────────────────────────────────────────────
    assert isinstance(df_combined.index, pd.DatetimeIndex), "DatetimeIndex check failed"
    assert "close" in df_combined.columns, "close column missing"

    if dry_run:
        print(f"\n  [DRY RUN] Would write {pair}_M15.parquet ({len(df_combined):,} rows) — skipping")
        return True

    # ── Step 4: Backup the augmented file (in case needed later) ─────────────
    if m15_path.exists() and not backup_path.exists():
        shutil.copy2(m15_path, backup_path)
        print(f"  Backup    : {backup_path.name} ({backup_path.stat().st_size // 1024} KB)")

    # ── Step 5: Write repaired file ───────────────────────────────────────────
    df_combined.to_parquet(m15_path, index=True)
    size_kb = m15_path.stat().st_size // 1024
    print(f"  ✓  Wrote   : {m15_path.name} | {len(df_combined):,} rows | {size_kb} KB")

    # ── Step 6: Verification read-back ───────────────────────────────────────
    verify = pd.read_parquet(m15_path)
    assert isinstance(verify.index, pd.DatetimeIndex), f"VERIFY FAILED: index is {type(verify.index).__name__}"
    print("  ✓  Verify  : DatetimeIndex confirmed on read-back")
    return True


def main():
    parser = argparse.ArgumentParser(description="Fix BTCUSD/ETHUSD M15 datetime index")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without writing files")
    args = parser.parse_args()

    print("=" * 66)
    print("  CAVALIER BASTION — CRYPTO M15 DATETIME INDEX REPAIR")
    print(f"  ROOT     : {ROOT}")
    print(f"  DATA_DIR : {DATA_DIR}")
    print(f"  Dry run  : {args.dry_run}")
    print("=" * 66)

    results = {}
    for pair in CRYPTO_PAIRS:
        results[pair] = load_and_fix(pair, DATA_DIR, args.dry_run)

    print(f"\n{'=' * 66}")
    print("  SUMMARY")
    print(f"{'=' * 66}")
    for pair, ok in results.items():
        status = "✓ REPAIRED" if ok else "✗ FAILED"
        print(f"  {pair}_M15 : {status}")

    all_ok = all(results.values())
    if all_ok:
        print("\n  ✓  Both crypto M15 files repaired.")
        print("     Re-run validate_v10_5_0.py --pair BTCUSD --tf M15 to confirm.")
    else:
        print("\n  ✗  One or more repairs failed — check output above.")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
