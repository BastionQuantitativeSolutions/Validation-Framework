"""
Deprecation Script: Move Old Backtest Files
===========================================

This script moves old backtest files to the deprecated folder
and creates a README explaining the new unified framework.
"""

import shutil
from pathlib import Path
from datetime import datetime

DEPRECATED_DIR = Path(__file__).parent / "deprecated"
ROOT = Path(__file__).parents[2]
OLD_FILES = [
    ROOT / "simulate_system_m30.py",
    ROOT / "simulate_system_m30_fixed.py",
    ROOT / "training_data/mt5_m30_comparison/verified_backtest.py",
    ROOT / "CORE_MODULES/training/live_equivalent_backtest.py",
    ROOT / "CORE_MODULES/validation/rag_1y_backtester.py",
    ROOT / "CORE_MODULES/validation/corrected_backtest.py",
    ROOT / "tmp/detailed_backtest_2w.py",
    ROOT / "tmp/fetch_backtest_data.py",
    Path(__file__).parent / "walkforward_montecarlo_test.py",
]

README_CONTENT = """# Deprecated Backtest Files
===========================

These files have been deprecated in favor of the unified backtest framework.

## New Unified Framework

**Location:** `CORE_MODULES/validation/unified_backtest_framework.py`

### Usage:
```python
from unified_backtest_framework import run_backtest

trades, stats = run_backtest(
    data_dir="training_data/mt5_m30_comparison",
    pairs=["EURUSD", "GBPUSD"],
    start_date="2025-01-01",
    end_date="2025-12-31",
)
```

### Key Features:
- Single source of truth for all parameters (from constants.py)
- 10 governance gates (mirrors live trading exactly)
- Unified SL/TP calculation
- Signal fusion matching live system

## Files Moved Here

1. `simulate_system_m30.py` - Replaced by unified_backtest_framework.py
2. `simulate_system_m30_fixed.py` - Replaced by unified_backtest_framework.py
3. `verified_backtest.py` - Replaced by unified_backtest_framework.py
4. `live_equivalent_backtest.py` - Replaced by unified_backtest_framework.py
5. `rag_1y_backtester.py` - Replaced by unified_backtest_framework.py
6. `corrected_backtest.py` - Replaced by unified_backtest_framework.py
7. `detailed_backtest_2w.py` - Replaced by unified_backtest_framework.py
8. `fetch_backtest_data.py` - Integrated into unified_backtest_framework.py
9. `walkforward_montecarlo_test.py` - Replaced by walkforward tests in unified framework

## Migration Guide

### Old Way:
```python
# simulate_system_m30.py
from some_module import run_backtest_old
results = run_backtest_old(...)
```

### New Way:
```python
# unified_backtest_framework.py
from CORE_MODULES.validation.unified_backtest_framework import run_backtest
results = run_backtest(...)
```

## Validation

Run validation to ensure unified system works:
```bash
python CORE_MODULES/validation/validate_unified_system.py
```

---

Deprecated: {date}
"""


def main():
    print("=" * 60)
    print("DEPRECATING OLD BACKTEST FILES")
    print("=" * 60)

    DEPRECATED_DIR.mkdir(exist_ok=True)

    moved_count = 0
    skipped_count = 0

    for src in OLD_FILES:
        if src.exists():
            dst = DEPRECATED_DIR / src.name
            shutil.move(str(src), str(dst))
            print(f"  Moved: {src.name}")
            moved_count += 1
        else:
            print(f"  Skipped (not found): {src}")
            skipped_count += 1

    readme_path = DEPRECATED_DIR / "README.md"
    readme_path.write_text(README_CONTENT.format(date=datetime.now().strftime("%Y-%m-%d")))

    print("-" * 60)
    print(f"Moved: {moved_count} files")
    print(f"Skipped: {skipped_count} files")
    print(f"Created: {readme_path}")
    print("=" * 60)
    print("\nOld backtest files have been deprecated.")
    print("Use unified_backtest_framework.py for all future backtesting.")


if __name__ == "__main__":
    main()
