"""
# Author: JG
Test volatility_regime_hook module

This script tests the volatility clustering and regime training
on the 22M parquet dataset before full system launch.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Also add CORE_MODULES/core to path for direct imports
core_modules = project_root / "CORE_MODULES" / "core"
if str(core_modules) not in sys.path:
    sys.path.insert(0, str(core_modules))

import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

log = logging.getLogger(__name__)


def main():
    """Test volatility and regime training"""

    log.info("=" * 60)
    log.info("[TEST] Testing Volatility & Regime Training Hook")
    log.info("=" * 60)

    try:
        from CORE_MODULES.core.regime.volatility_regime_hook import initialize_volatility_and_regime

        # Test symbols
        test_symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]

        # Initialize
        log.info("[TEST] Initializing with test symbols...")
        result = initialize_volatility_and_regime(symbols=test_symbols, force_retrain=True)

        # Verify results
        vol_clusters = result.get("volatility", {})
        regime_states = result.get("regime", {})

        log.info("=" * 60)
        log.info("[TEST] RESULTS")
        log.info("=" * 60)
        log.info(f"[TEST] Volatility clusters: {len(vol_clusters)} symbols")
        log.info(f"[TEST] Regime states: {len(regime_states)} symbols")

        # Show sample results
        if vol_clusters:
            sample_symbol = list(vol_clusters.keys())[0]
            sample_cluster = vol_clusters[sample_symbol]
            log.info(f"[TEST] Sample volatility for {sample_symbol}:")
            log.info(f"[TEST]   - Current regime: {sample_cluster.get('current_regime')}")
            log.info(f"[TEST]   - Volatility: {sample_cluster.get('current_short_vol', 0):.4f}%")
            log.info(f"[TEST]   - Percentile: {sample_cluster.get('volatility_percentile', 0):.1%}")

        if regime_states:
            sample_symbol = list(regime_states.keys())[0]
            sample_regime = regime_states[sample_symbol]
            log.info(f"[TEST] Sample regime for {sample_symbol}:")
            log.info(f"[TEST]   - Regime: {sample_regime.get('regime')}")
            log.info(f"[TEST]   - ADX: {sample_regime.get('adx', 0):.1f}")
            log.info(f"[TEST]   - Trend strength: {sample_regime.get('trend_strength', 0):.2f}")

        log.info("=" * 60)
        log.info("[TEST] SUCCESS: Volatility & Regime hook working")
        log.info("=" * 60)

        return True

    except Exception as e:
        log.error(f"[TEST] FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
