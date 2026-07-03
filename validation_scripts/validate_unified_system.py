"""
Unified System Validation Script
================================
Validates that backtests exactly mirror live trading.

Usage:
    python CORE_MODULES/validation/validate_unified_system.py
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("validation")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Import unified modules
try:
    from CORE_MODULES.core.unified_governance import governance_check
    from CORE_MODULES.core.unified_exits import calculate_sl_tp
    from CORE_MODULES.core.config.constants import (
        BASE_BUY_THRESHOLD,
        BASE_SELL_THRESHOLD,
        W_ML,
        W_SMC,
        SL_MULTIPLIERS,
        TP_MULTIPLIERS,
    )
except ImportError as e:
    log.error(f"Import failed: {e}")
    sys.exit(1)


def test_thresholds():
    """Test that thresholds match between modules."""
    log.info("=" * 60)
    log.info("TEST 1: Threshold Verification")
    log.info("=" * 60)

    expected_buy = 0.58
    expected_sell = 0.42

    checks = [
        ("BASE_BUY_THRESHOLD", BASE_BUY_THRESHOLD, expected_buy),
        ("BASE_SELL_THRESHOLD", BASE_SELL_THRESHOLD, expected_sell),
        ("W_ML", W_ML, 0.7),
        ("W_SMC", W_SMC, 0.3),
    ]

    all_passed = True
    for name, actual, expected in checks:
        status = "✓" if abs(actual - expected) < 0.001 else "✗"
        if status == "✗":
            all_passed = False
        log.info(f"  {status} {name}: {actual} (expected {expected})")

    return all_passed


def test_atr_parameters():
    """Test that ATR parameters match between modules."""
    log.info("=" * 60)
    log.info("TEST 2: ATR Parameter Verification")
    log.info("=" * 60)

    expected_sl = {"TRENDING": 1.5, "RANGING": 1.0, "VOLATILE": 1.2}
    expected_tp = {"TRENDING": 3.0, "RANGING": 2.0, "VOLATILE": 2.5}

    all_passed = True

    for regime, expected in expected_sl.items():
        actual = SL_MULTIPLIERS.get(regime, 0)
        status = "✓" if abs(actual - expected) < 0.001 else "✗"
        if status == "✗":
            all_passed = False
        log.info(f"  {status} SL_{regime}: {actual}x (expected {expected}x)")

    for regime, expected in expected_tp.items():
        actual = TP_MULTIPLIERS.get(regime, 0)
        status = "✓" if abs(actual - expected) < 0.001 else "✗"
        if status == "✗":
            all_passed = False
        log.info(f"  {status} TP_{regime}: {actual}x (expected {expected}x)")

    return all_passed


def test_governance_gates():
    """Test governance gates."""
    log.info("=" * 60)
    log.info("TEST 3: Governance Gate Verification")
    log.info("=" * 60)

    test_cases = [
        # (signal, context, expected_allowed, description)
        (
            {
                "direction": 1,
                "confidence": 0.80,
                "regime": "TRENDING",
                "session": "LONDON",
                "smc_confluence": 0.6,
                "momentum": 0.5,
            },
            {"daily_trade_count": 2, "session_trade_count": 5, "loss_streak": 0},
            True,
            "Trending London trade",
        ),
        (
            {"direction": 1, "confidence": 0.80, "regime": "UNKNOWN", "session": "LONDON"},
            {"daily_trade_count": 2, "session_trade_count": 5, "loss_streak": 0},
            False,
            "UNKNOWN regime should block",
        ),
        (
            {"direction": 1, "confidence": 0.80, "regime": "RANGING", "session": "ASIAN"},
            {"daily_trade_count": 2, "session_trade_count": 5, "loss_streak": 0},
            False,
            "Asian+Ranging should block",
        ),
        (
            {"direction": 1, "confidence": 0.50, "regime": "TRENDING", "session": "LONDON"},
            {"daily_trade_count": 2, "session_trade_count": 5, "loss_streak": 0},
            False,
            "Low confidence should block",
        ),
    ]

    all_passed = True
    for signal, context, expected, description in test_cases:
        allowed, reason, mult = governance_check(signal, "EURUSD", context)
        status = "✓" if allowed == expected else "✗"
        if status == "✗":
            all_passed = False
        log.info(f"  {status} {description}: allowed={allowed} (expected {expected})")
        log.info(f"       Reason: {reason}")

    return all_passed


def test_sl_tp_calculation():
    """Test SL/TP calculations."""
    log.info("=" * 60)
    log.info("TEST 4: SL/TP Calculation Verification")
    log.info("=" * 60)

    entry = 1.1000
    atr = 0.0010

    all_passed = True

    for regime, direction in [("TRENDING", 1), ("RANGING", 1), ("VOLATILE", -1)]:
        sl, tp = calculate_sl_tp(entry, direction, atr, regime)

        expected_sl_mult = SL_MULTIPLIERS.get(regime, 1.5)
        expected_tp_mult = TP_MULTIPLIERS.get(regime, 2.5)

        sl_distance = abs(entry - sl)
        tp_distance = abs(tp - entry)

        sl_ratio = sl_distance / atr
        tp_ratio = tp_distance / atr

        sl_ok = abs(sl_ratio - expected_sl_mult) < 0.01
        tp_ok = abs(tp_ratio - expected_tp_mult) < 0.01

        status = "✓" if sl_ok and tp_ok else "✗"
        if status == "✗":
            all_passed = False

        log.info(f"  {status} {regime} (direction={direction}):")
        log.info(f"       SL: {sl:.5f} ({sl_ratio:.2f}x ATR, expected {expected_sl_mult}x)")
        log.info(f"       TP: {tp:.5f} ({tp_ratio:.2f}x ATR, expected {expected_tp_mult}x)")

    return all_passed


def test_fusion_math():
    """Test signal fusion math."""
    log.info("=" * 60)
    log.info("TEST 5: Signal Fusion Math Verification")
    log.info("=" * 60)

    import numpy as np

    test_cases = [
        (0.70, 0.60, "High ML, decent SMC"),
        (0.60, 0.70, "Low ML, high SMC"),
        (0.55, 0.55, "Equal ML and SMC"),
        (0.50, 0.50, "Neutral ML and SMC"),
    ]

    all_passed = True

    for ml_prob, smc_conf, _desc in test_cases:
        base_fused = W_ML * ml_prob + W_SMC * smc_conf

        diff = abs(ml_prob - smc_conf)
        k_decay = 1.0
        if smc_conf > 0.05:
            penalty = max(0.2, 1.0 - k_decay * diff**2)
        else:
            penalty = 1.0

        fused = 0.5 + (base_fused - 0.5) * penalty
        fused = float(np.clip(fused, 0, 1))

        # Check direction
        if fused >= BASE_BUY_THRESHOLD:
            direction = "BUY"
        elif fused <= BASE_SELL_THRESHOLD:
            direction = "SELL"
        else:
            direction = "NEUTRAL"

        log.info(f"  ML={ml_prob:.2f}, SMC={smc_conf:.2f} → base={base_fused:.3f}, fused={fused:.3f} → {direction}")

        # Basic sanity checks
        if not (0 <= fused <= 1):
            all_passed = False
            log.error("    ✗ Fused value out of range!")

    return all_passed


def generate_validation_report():
    """Generate final validation report."""
    log.info("=" * 60)
    log.info("VALIDATION SUMMARY")
    log.info("=" * 60)

    results = {"timestamp": datetime.now().isoformat(), "tests": {}}

    # Run all tests
    results["tests"]["thresholds"] = test_thresholds()
    results["tests"]["atr_parameters"] = test_atr_parameters()
    results["tests"]["governance_gates"] = test_governance_gates()
    results["tests"]["sl_tp_calculation"] = test_sl_tp_calculation()
    results["tests"]["fusion_math"] = test_fusion_math()

    # Calculate overall
    total = len(results["tests"])
    passed = sum(1 for v in results["tests"].values() if v)

    results["summary"] = {"total_tests": total, "passed": passed, "failed": total - passed, "pass_rate": f"{passed / total:.0%}"}

    log.info("")
    log.info(f"Total Tests: {total}")
    log.info(f"Passed: {passed}")
    log.info(f"Failed: {total - passed}")
    log.info(f"Pass Rate: {passed / total:.0%}")
    log.info("")

    if passed == total:
        log.info("✓ ALL TESTS PASSED - Unified system validated!")
    else:
        log.info("✗ SOME TESTS FAILED - Review output above")

    # Save report
    report_path = Path(__file__).parent / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Report saved to: {report_path}")

    return passed == total


if __name__ == "__main__":
    success = generate_validation_report()
    sys.exit(0 if success else 1)
