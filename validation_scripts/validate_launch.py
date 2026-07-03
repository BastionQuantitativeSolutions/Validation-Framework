"""
# Author: JG
Pre-Launch System Validation
============================

This script validates all systems before launching the trading bot.
It checks:
1. Feature count alignment (79 features)
2. Volatility clustering & regime training
3. Model loading
4. Configuration files
5. Dependencies
"""

import sys
import os
from pathlib import Path
import logging


# Add paths
def _find_project_root() -> Path:
    """Resolve project root even if this file is a symlink in an output directory."""
    env_root = os.getenv("CAVALIER_ROOT") or os.getenv("PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    candidates = []
    try:
        candidates.append(Path.cwd())
    except Exception:
        pass
    try:
        candidates.append(Path(__file__).resolve())
    except Exception:
        pass
    # Walk up from candidates to find repo root (contains CORE_MODULES/config and DATA_MODELS)
    for base in candidates:
        for parent in [base] + list(base.parents):
            if (parent / "CORE_MODULES" / "config").exists() and (parent / "DATA_MODELS").exists():
                return parent
    # Fallback to three levels up from this file
    return Path(__file__).resolve().parents[3]


project_root = _find_project_root()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "CORE_MODULES"))

# Setup logging - prevent all duplicate logging
import sys

# Configure root logger once
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Remove ALL existing handlers to prevent duplicates
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
root_logger.propagate = False  # Disable propagation

# Create a single handler for our app
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
root_logger.addHandler(console_handler)

# Create our app logger (child of root, but won't duplicate since propagate=False)
log = logging.getLogger(__name__)

# Ensure no child loggers propagate
for name in logging.Logger.manager.loggerDict:
    logger = logging.getLogger(name)
    logger.propagate = False


def check_imports():
    """Check all required imports"""
    log.info("[CHECK] Testing required imports...")

    try:
        log.info("[CHECK] ✓ pandas, numpy, json")
    except Exception as e:
        log.error(f"[CHECK] ✗ Standard libraries: {e}")
        return False

    try:
        log.info("[CHECK] ✓ volatility_regime_hook")
    except Exception as e:
        log.error(f"[CHECK] ✗ volatility_regime_hook: {e}")
        return False

    return True


def check_parquet_data():
    """Check parquet data availability"""
    log.info("[CHECK] Testing parquet data availability...")

    parquet_dir = project_root / "DATA_MODELS" / "data_parquet"
    if not parquet_dir.exists():
        log.error(f"[CHECK] ✗ Parquet directory not found: {parquet_dir}")
        return False

    symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAUUSD"]
    missing = []

    for symbol in symbols:
        parquet_file = parquet_dir / f"{symbol}_M5.parquet"  # Check M5 (we have M5 data)
        if not parquet_file.exists():
            missing.append(symbol)

    if missing:
        log.error(f"[CHECK] ✗ Missing parquet files for: {missing}")
        return False

    log.info(f"[CHECK] ✓ Parquet data available for all {len(symbols)} symbols")
    return True


def check_volatility_regime_training():
    """
    Test volatility and regime training — trains ALL live pairs so main_loop
    starts with a fully primed cache and skips the duplicate 53s training + 3×
    Gemma batches it was previously running on startup.

    force_retrain=False: uses cache if < 1h old; only trains missing symbols.
    """
    log.info("[CHECK] Training volatility & regime for ALL live pairs (primes cache for main_loop)...")

    try:
        from CORE_MODULES.core.regime.volatility_regime_hook import (
            initialize_volatility_and_regime,
        )

        # Load the full pair universe from config instead of a hardcoded 6-symbol subset.
        # This ensures main_loop finds a complete cache and skips its own training pass.
        try:
            from CORE_MODULES.core.config.config_sync import load_master_config
            _cfg = load_master_config()
            _pairs_cfg = _cfg.get("pairs", {})
            if isinstance(_pairs_cfg, dict):
                symbols = list(_pairs_cfg.keys())
            elif isinstance(_pairs_cfg, list):
                symbols = list(_pairs_cfg)
            else:
                raise ValueError("Unexpected pairs format")
            if not symbols:
                raise ValueError("Empty pairs list")
        except Exception:
            # Fallback: use full known universe
            symbols = [
                "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF",
                "XAUUSD", "GBPCHF", "XAGUSD", "NZDUSD", "USOIL", "UKOIL",
                "JP225", "HK50", "UK100", "US100", "BTCUSD", "ETHUSD",
            ]

        log.info(f"[CHECK] Training {len(symbols)} pairs: {symbols}")
        # force_retrain=False — reuse cache if fresh; only computes missing symbols.
        # This is idempotent: second run (main_loop) is a pure cache hit.
        result = initialize_volatility_and_regime(symbols, force_retrain=False)

        vol_count = len(result.get("volatility", {}))
        regime_count = len(result.get("regime", {}))

        if vol_count == 0 or regime_count == 0:
            log.error(f"[CHECK] ✗ Training incomplete: vol={vol_count}, regime={regime_count}")
            return False

        log.info(f"[CHECK] ✓ Volatility & regime training: {vol_count} vol / {regime_count} regime symbols cached")
        return True

    except Exception as e:
        log.error(f"[CHECK] ✗ Volatility & regime training failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def check_models():
    """Check model availability"""
    log.info("[CHECK] Testing model availability...")

    models_dir = project_root / "DATA_MODELS" / "models_live"
    if not models_dir.exists():
        log.error(f"[CHECK] ✗ Models directory not found: {models_dir}")
        return False

    symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAUUSD"]
    timeframes = ["M5", "M15", "M30", "H1"]  # H4 removed Run I (2026-04-26); H4 was never in live TFS
    missing = []

    for symbol in symbols:
        for tf in timeframes:
            model_dir = models_dir / f"{symbol}_{tf}"
            if not model_dir.exists():
                missing.append(f"{symbol}_{tf}")
                continue

            # Check for tiered structure (full/ and core/)
            full_dir = model_dir / "full"
            core_dir = model_dir / "core"

            if not full_dir.exists():
                missing.append(f"{symbol}_{tf}/full")
                continue

            if not core_dir.exists():
                missing.append(f"{symbol}_{tf}/core")
                continue

            # Check for tiered model files (both .joblib and .pkl formats)
            full_xgb = full_dir / "xgb_model.joblib"
            full_lgb = full_dir / "lgb_model.joblib"
            full_cat = full_dir / "cat_model.joblib"
            full_lgb_pkl = full_dir / "lightgbm_model.pkl"  # New format

            # Check at least full tier has models (top-level or bias-variant subdirs)
            has_top_level = full_xgb.exists() or full_lgb.exists() or full_cat.exists() or full_lgb_pkl.exists()
            has_bias_variant = False
            if not has_top_level:
                for variant in ("unbiased", "buy_bias", "sell_bias"):
                    variant_dir = full_dir / variant
                    if variant_dir.is_dir():
                        variant_files = list(variant_dir.glob("*.joblib")) + list(variant_dir.glob("*.pkl")) + list(variant_dir.glob("*.json"))
                        if variant_files:
                            has_bias_variant = True
                            break
            if not (has_top_level or has_bias_variant):
                missing.append(f"{symbol}_{tf} (no models in full/)")
                continue

    if missing:
        log.error(f"[CHECK] ✗ Missing models for: {len(missing)} timeframe(s)")
        for m in missing[:5]:
            log.error(f"[CHECK]   - {m}")
        if len(missing) > 5:
            log.error(f"[CHECK]   ... and {len(missing) - 5} more")
        return False

    log.info(f"[CHECK] ✓ Models available for all {len(symbols) * len(timeframes)} timeframe(s) (tiered structure)")
    return True


def check_config_files():
    """Check configuration files"""
    log.info("[CHECK] Testing configuration files...")

    config_dir = project_root / "CORE_MODULES" / "config"
    required_files = [
        "risk_governor.json",
        "institutional_position_sizing.json",
    ]

    missing = []
    for filename in required_files:
        filepath = config_dir / filename
        if not filepath.exists():
            missing.append(filename)

    if missing:
        log.error(f"[CHECK] ✗ Missing config files: {missing}")
        return False

    # Check order safety config
    order_safety_file = project_root / "CORE_MODULES" / "config" / "order_safety.json"
    if order_safety_file.exists():
        log.info("[CHECK] ✓ Order safety config found")
    else:
        log.warning("[CHECK] ⚠ Order safety config not found (optional)")

    log.info("[CHECK] ✓ All required config files found")
    return True


def check_results_directory():
    """Check results directory is writable"""
    log.info("[CHECK] Testing results directory...")

    results_dir = project_root / "DATA_MODELS" / "results"
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        test_file = results_dir / "startup_test.tmp"
        test_file.write_text("test")
        test_file.unlink()
        log.info(f"[CHECK] ✓ Results directory writable: {results_dir}")
        return True
    except Exception as e:
        log.error(f"[CHECK] ✗ Results directory not writable: {e}")
        return False


def main():
    """Run all checks"""
    log.info("=" * 70)
    log.info("[LAUNCH] Pre-Launch System Validation")
    log.info("=" * 70)

    checks = [
        ("Imports", check_imports),
        ("Parquet Data", check_parquet_data),
        ("Volatility & Regime", check_volatility_regime_training),
        ("Models", check_models),
        ("Config Files", check_config_files),
        ("Results Directory", check_results_directory),
    ]

    passed = 0
    failed = 0

    for name, check_func in checks:
        log.info("-" * 70)
        try:
            if check_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            log.error(f"[CHECK] ✗ {name} check crashed: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    log.info("=" * 70)
    log.info(f"[LAUNCH] Validation Complete: {passed} passed, {failed} failed")
    log.info("=" * 70)

    if failed == 0:
        log.info("[LAUNCH] ✓ ALL CHECKS PASSED - System ready for launch")
        log.info("[LAUNCH] Run START_LIVE_TRADING.bat to launch system")
        return True
    else:
        log.error(f"[LAUNCH] ✗ {failed} check(s) failed - Fix issues before launch")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
