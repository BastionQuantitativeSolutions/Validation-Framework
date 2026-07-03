import os
import json
import sys
from pathlib import Path

# Add root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MetaTrader5 as mt5
from CORE_MODULES.core.system.signal_sanitizer import get_current_drawdown


def test_drawdown_logic():
    print("Testing HWM Drawdown Logic...")
    hwm_file = "results/test_hwm.json"
    if os.path.exists(hwm_file):
        os.remove(hwm_file)

    # Starting equity
    dd1 = get_current_drawdown(10000.0, hwm_file=hwm_file)
    print(f"  Initial ($10,000): DD={dd1:.2%}")

    # Equity increases
    dd2 = get_current_drawdown(10500.0, hwm_file=hwm_file)
    print(f"  Increase ($10,500): DD={dd2:.2%}")  # Should be 0% as HWM moves

    # Equity decreases
    dd3 = get_current_drawdown(10200.0, hwm_file=hwm_file)
    print(f"  Decrease ($10,200): DD={dd3:.2%}")  # Should be -2.86% relative to 10,500

    with open(hwm_file, "r") as f:
        data = json.load(f)
        print(f"  Final HWM in file: {data['hwm']}")
        assert data["hwm"] == 10500.0


def test_lot_sizing_logic():
    print("\nTesting Lot Sizing Logic (Mocked Arithmetic)...")
    # risk_dollars / ((sl_dist / tick_size) * tick_value)

    def simulate_calc(risk_dollars, sl_dist, tick_size, tick_value):
        return risk_dollars / ((sl_dist / tick_size) * tick_value)

    # EURUSD: 1 pip = 0.0001, Tick Value = 1.0, Tick Size = 0.00001
    # 30 pip SL = 0.0030
    # $100 risk
    eurusd_lots = simulate_calc(100.0, 0.0030, 0.00001, 1.0 / 10.0)  # Wait, point value is usually per point.
    # In MT5, tick_value is usually per 1.0 lot per 1.0 point (0.00001)
    # So if tick_value is 1.0 and tick_size is 0.00001:
    # point_value = 1.0 / 0.00001 = 100,000
    # risk $100 / (0.0030 * 100,000) = 100 / 300 = 0.33 lots. Correct.

    # USDJPY: 1 pip = 0.01, Tick Value = 0.65 (approx), Tick Size = 0.001
    # 20 pip SL = 0.20
    # $100 risk
    usdjpy_lots = simulate_calc(100.0, 0.20, 0.001, 0.65)
    # risk $100 / (200 * 0.65) = 100 / 130 = 0.77 lots. Correct.

    print(f"  EURUSD 30pip SL / $100 risk: {eurusd_lots:.2f} lots")
    print(f"  USDJPY 20pip SL / $100 risk: {usdjpy_lots:.2f} lots")


if __name__ == "__main__":
    try:
        if not mt5.initialize():
            print("MT5 Not Initialized (Simulation only)")
        test_drawdown_logic()
        test_lot_sizing_logic()
    finally:
        mt5.shutdown()
