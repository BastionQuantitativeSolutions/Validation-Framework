import sys
import os
import logging
from datetime import datetime, timedelta

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Import the module to test
from core.features.mtf_confluence import MTFConfluenceTracker

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def test_weighted_confluence():
    print("\n=== Testing MTF Weighted Confluence ===")
    tracker = MTFConfluenceTracker()
    pair = "EURUSD"

    # Use the expected weights: H1: 40, M15: 25, M5: 20, M1: 15
    tracker.record(pair, "M1", 1, 0.8)
    tracker.record(pair, "M5", 1, 0.8)
    score = tracker.get_weighted_confluence_score(pair, 1)
    print(f"Scenario 1 (M1+M5): Score = {score:.4f} (Expected ~0.35)")

    tracker.record(pair, "H1", 1, 0.8)
    tracker.record(pair, "M15", 1, 0.8)
    score = tracker.get_weighted_confluence_score(pair, 1)
    print(f"Scenario 2 (H1+M15+M5+M1): Score = {score:.4f} (Expected 1.0)")


def test_volatility_logic():
    print("\n=== Testing Volatility Block Logic (Simulated) ===")

    def mock_vol_check(current_atr, avg_atr, multiplier):
        if avg_atr > 0 and current_atr > (avg_atr * multiplier):
            return "BLOCKED"
        return "ALLOWED"

    multiplier = 2.2
    print(f"Scenario 1 (Normal): {mock_vol_check(0.0015, 0.0010, multiplier)} (Expected ALLOWED)")
    print(f"Scenario 2 (Spike): {mock_vol_check(0.0035, 0.0010, multiplier)} (Expected BLOCKED)")


def test_adaptive_overrides():
    print("\n=== Testing Adaptive Overrides (Simulated) ===")
    # Simulate the logic from ensemble_all_in_one.py
    PAIR_PROFILES = {"USDCAD": {"min_confidence": 0.78, "vol_multiplier_override": 1.5}}

    def get_eff_threshold(pair, base):
        overrides = PAIR_PROFILES.get(pair, {})
        return overrides.get("min_confidence", base)

    def get_eff_vol_multiplier(pair, base):
        overrides = PAIR_PROFILES.get(pair, {})
        return overrides.get("vol_multiplier_override", base)

    print(f"USDCAD Threshold: {get_eff_threshold('USDCAD', 0.65)} (Expected 0.78)")
    print(f"EURUSD Threshold: {get_eff_threshold('EURUSD', 0.65)} (Expected 0.65)")
    print(f"USDCAD Vol Mult: {get_eff_vol_multiplier('USDCAD', 2.0)} (Expected 1.5)")


def test_cooldown_logic():
    print("\n=== Testing Adaptive Cooldown Logic (Simulated) ===")
    COOLDOWN_TRACKER = {}

    def update_cooldown(pair):
        COOLDOWN_TRACKER[pair] = datetime.now()

    def is_on_cooldown(pair, COOLDOWN_TRACKER, duration_hrs):
        last_loss = COOLDOWN_TRACKER.get(pair)
        if last_loss:
            expiry = last_loss + timedelta(hours=duration_hrs)
            if datetime.now() < expiry:
                return True
        return False

    pair = "USDCAD"
    print(f"Initial: Cooldown active? {is_on_cooldown(pair, COOLDOWN_TRACKER, 4)} (Expected False)")

    update_cooldown(pair)
    print(f"After Update: Cooldown active? {is_on_cooldown(pair, COOLDOWN_TRACKER, 4)} (Expected True)")

    # Simulate time passing (fast-forward expiry)
    COOLDOWN_TRACKER[pair] = datetime.now() - timedelta(hours=5)
    print(f"After 5h pass: Cooldown active? {is_on_cooldown(pair, COOLDOWN_TRACKER, 4)} (Expected False)")


if __name__ == "__main__":
    try:
        test_weighted_confluence()
        test_volatility_logic()
        test_adaptive_overrides()
        test_cooldown_logic()
        print("\nVerification Complete.")
    except Exception as e:
        print(f"Error during verification: {e}")
        sys.exit(1)
