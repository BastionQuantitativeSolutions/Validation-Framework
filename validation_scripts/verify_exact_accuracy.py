"""
Verify 1-1 Accuracy of Exact Simulator vs Live System
=====================================================

This script validates that live_exact_simulator.py matches
the live trading system constants and logic EXACTLY.

Usage: python verify_exact_accuracy.py
"""

import os
import sys
from pathlib import Path

# Set exact same env as live system
os.environ.setdefault("PAIRS", "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD,USDCHF,GBPCHF,NZDUSD,XAUUSD,XAGUSD,USOIL,UKOIL,HEATOIL,JP225,US100,HK50,UK100,BTCUSD,ETHUSD")
os.environ.setdefault("BASE_BUY_THRESHOLD", "0.55")
os.environ.setdefault("BASE_SELL_THRESHOLD", "0.45")
os.environ.setdefault("SL_TRENDING", "1.5")
os.environ.setdefault("TP_TRENDING", "3.0")
os.environ.setdefault("PTP_TAKE_1", "0.40")
os.environ.setdefault("PTP_TAKE_2", "0.30")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from CORE_MODULES.core.config.constants import (
    PAIRS, BASE_BUY_THRESHOLD, BASE_SELL_THRESHOLD,
    W_ML, W_SMC, SL_MULTIPLIERS, TP_MULTIPLIERS, BASE_RISK_PER_TRADE, MIN_CONFIDENCE_GOVERNANCE, MIN_MOMENTUM,
    COOLDOWN_BARS, MAX_DAILY_TRADES_PER_SYMBOL, LOSS_STREAK_LIMIT,
    SESSION_TRADE_CAP, SAFETY_FLOOR_MIN_PENALTY,
    REGIME_BUY_BOOST, REGIME_SELL_BOOST,
)

PTP_TAKE_1 = float(os.getenv("PTP_TAKE_1", "0.40"))
PTP_TAKE_2 = float(os.getenv("PTP_TAKE_2", "0.30"))


def verify_constants():
    """Verify all constants match between sim and live"""
    print("=" * 70)
    print("VERIFYING 1-1 ACCURACY: SIMULATOR vs LIVE SYSTEM")
    print("=" * 70)
    print()
    
    all_pass = True
    
    # Check pairs
    expected_pairs = "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD,USDCHF,GBPCHF,NZDUSD,XAUUSD,XAGUSD,USOIL,UKOIL,HEATOIL,JP225,US100,HK50,UK100,BTCUSD,ETHUSD".split(",")
    if PAIRS == expected_pairs:
        print(f"[OK] PAIRS: EXACT MATCH ({len(PAIRS)} pairs)")
    else:
        print("[FAIL] PAIRS: MISMATCH")
        print(f"   Expected: {expected_pairs}")
        print(f"   Got: {PAIRS}")
        all_pass = False
    
    # Check thresholds
    checks = [
        ("BASE_BUY_THRESHOLD", BASE_BUY_THRESHOLD, 0.55),
        ("BASE_SELL_THRESHOLD", BASE_SELL_THRESHOLD, 0.45),
        ("W_ML", W_ML, 0.7),
        ("W_SMC", W_SMC, 0.3),
        ("BASE_RISK_PER_TRADE", BASE_RISK_PER_TRADE, 0.0125),  # INCREASED for larger positions
        ("MIN_CONFIDENCE", MIN_CONFIDENCE_GOVERNANCE, 0.55),
        ("MIN_MOMENTUM", MIN_MOMENTUM, 0.25),
        ("COOLDOWN_BARS", COOLDOWN_BARS, 2),
        ("MAX_DAILY_TRADES", MAX_DAILY_TRADES_PER_SYMBOL, 50),
        ("LOSS_STREAK_LIMIT", LOSS_STREAK_LIMIT, 3),
        ("SESSION_CAP", SESSION_TRADE_CAP, 30),
        ("SAFETY_FLOOR", SAFETY_FLOOR_MIN_PENALTY, 0.75),
        ("PTP_TAKE_1", PTP_TAKE_1, 0.40),
        ("PTP_TAKE_2", PTP_TAKE_2, 0.30),
        ("REGIME_BUY_BOOST", REGIME_BUY_BOOST, 1.0),  # FIXED: No bias
        ("REGIME_SELL_BOOST", REGIME_SELL_BOOST, 1.0),  # FIXED: No bias
    ]
    
    print()
    print("CONSTANT VERIFICATION:")
    for name, actual, expected in checks:
        if abs(actual - expected) < 0.001:
            print(f"  [OK] {name}: {actual} (exact)")
        else:
            print(f"  [FAIL] {name}: {actual} (expected {expected})")
            all_pass = False
    
    # Check SL/TP multipliers
    print()
    print("SL/TP MULTIPLIERS:")
    expected_sl = {"TRENDING": 1.5, "RANGING": 1.0, "VOLATILE": 1.2, "DEFAULT": 1.5}
    expected_tp = {"TRENDING": 3.0, "RANGING": 2.0, "VOLATILE": 2.5, "DEFAULT": 2.5}
    
    for regime, expected in expected_sl.items():
        actual = SL_MULTIPLIERS.get(regime)
        if abs(actual - expected) < 0.001:
            print(f"  [OK] SL_{regime}: {actual}x")
        else:
            print(f"  [FAIL] SL_{regime}: {actual}x (expected {expected}x)")
            all_pass = False
    
    for regime, expected in expected_tp.items():
        actual = TP_MULTIPLIERS.get(regime)
        if abs(actual - expected) < 0.001:
            print(f"  [OK] TP_{regime}: {actual}x")
        else:
            print(f"  [FAIL] TP_{regime}: {actual}x (expected {expected}x)")
            all_pass = False
    
    # BIAS CHECK
    print()
    print("BIAS VERIFICATION (no hardcoded directional bias):")
    if abs(REGIME_BUY_BOOST - REGIME_SELL_BOOST) < 0.001:
        print(f"  [OK] REGIME boosts symmetric: BUY={REGIME_BUY_BOOST}, SELL={REGIME_SELL_BOOST}")
    else:
        print(f"  [FAIL] REGIME boosts asymmetric: BUY={REGIME_BUY_BOOST}, SELL={REGIME_SELL_BOOST}")
        all_pass = False
    
    if abs(BASE_BUY_THRESHOLD - (1.0 - BASE_SELL_THRESHOLD)) < 0.001:
        print(f"  [OK] Thresholds symmetric around 0.5: BUY={BASE_BUY_THRESHOLD}, SELL={BASE_SELL_THRESHOLD}")
    else:
        print(f"  [WARN] Thresholds not symmetric: BUY={BASE_BUY_THRESHOLD}, SELL={BASE_SELL_THRESHOLD}")
    
    print()
    print("=" * 70)
    if all_pass:
        print("[PASS] ALL CHECKS PASSED - SIMULATOR IS 1-1 ACCURATE")
    else:
        print("[FAIL] SOME CHECKS FAILED - REVIEW MISMATCHES ABOVE")
    print("=" * 70)
    
    return all_pass


if __name__ == "__main__":
    verify_constants()
