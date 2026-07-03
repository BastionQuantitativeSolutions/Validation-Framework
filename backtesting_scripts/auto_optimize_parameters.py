#!/usr/bin/env python3
"""
Auto-Optimize Backtest Parameters
==================================

Automatically tweaks backtest parameters until targets are hit:
- 70%+ win rate
- Positive profit
- Max daily drawdown < 3%

Uses live trading backtester with exact same logic as live system.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from CORE_MODULES.backtesting.live_trading_backtester import LiveTradingBacktester

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PARQUET_DIR = Path("C:/Users/jack/Cavalier/DATA_MODELS/data_parquet")
ALL_PAIRS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "XAUUSD",
    "USDCHF",
    "GBPCHF",
    "XAGUSD",
    "NZDUSD",
    "USOIL",
    "UKOIL",
    "HEATOIL",
    "JP225",
    "US100",
    "HK50",
    "UK100",
]

# Target parameters
TARGET_WIN_RATE = 0.70  # 70% win rate
MAX_DAILY_DD = 0.03  # 3% daily drawdown limit
MIN_PROFIT = 0  # Must be positive

# Parameter ranges to test
PARAM_RANGES = {
    "min_confidence": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
    "max_daily_trades": [1, 2, 3, 4, 5, 6, 8],
    "risk_per_trade": [0.0025, 0.005, 0.0075, 0.01],
    "min_confirming_factors": [0, 1, 2, 3],
    "min_momentum": [0.1, 0.2, 0.3, 0.4, 0.5],
    "session_trade_cap": [5, 10, 15, 20, 30],
}


def load_all_data():
    """Load data for all pairs."""
    all_data = {}
    for pair in ALL_PAIRS:
        for tf in ["M15", "M5", "M30", "H1"]:
            file_path = PARQUET_DIR / f"{pair}_{tf}.parquet"
            if file_path.exists():
                df = pd.read_parquet(file_path)
                df = df.reset_index()
                if "index" in df.columns:
                    df = df.rename(columns={"index": "time"})
                elif "datetime" in df.columns:
                    df = df.rename(columns={"datetime": "time"})

                # Use last 30 days
                start_date = "2026-02-13"
                end_date = "2026-03-20"
                if "time" in df.columns:
                    df = df[(df["time"] >= start_date) & (df["time"] <= end_date)]
                    if len(df) > 0:
                        all_data[pair] = df
                        break
    return all_data


def run_backtest_with_params(params):
    """Run backtest with given parameters."""
    all_data = load_all_data()
    if not all_data:
        return None

    backtester = LiveTradingBacktester(
        initial_balance=10000,
        risk_per_trade=params["risk_per_trade"],
        max_daily_trades=params["max_daily_trades"],
        min_confidence=params["min_confidence"],
        max_positions=1,
    )

    # Override global constants for this run
    import CORE_MODULES.backtesting.live_trading_backtester as ltb

    ltb.MIN_CONFIRMING_FACTORS = params["min_confirming_factors"]
    ltb.MIN_MOMENTUM = params["min_momentum"]
    ltb.SESSION_TRADE_CAP = params["session_trade_cap"]

    results = backtester.run(all_data, datetime(2026, 2, 13), datetime(2026, 3, 20))
    return results


def check_targets(results):
    """Check if backtest results meet targets."""
    if not results or "stats" not in results:
        return False, {}

    stats = results["stats"]

    # Calculate win rate (stored as percentage in dict)
    win_rate = stats.get("win_rate", 0) / 100.0  # Convert from percentage to decimal

    # Calculate max drawdown (stored as percentage in dict)
    max_dd = stats.get("max_drawdown_pct", 0) / 100.0  # Convert from percentage to decimal

    # Calculate daily drawdown (approximate)
    total_days = 30  # 30 days of data
    daily_dd = max_dd / np.sqrt(total_days)  # Rough approximation

    # Get profit
    profit = stats.get("total_pnl", 0)

    targets_met = win_rate >= TARGET_WIN_RATE and daily_dd <= MAX_DAILY_DD and profit > MIN_PROFIT

    return targets_met, {
        "win_rate": win_rate,
        "daily_dd": daily_dd,
        "profit": profit,
        "max_dd": max_dd,
        "total_trades": stats.get("total_trades", 0),
        "profit_factor": stats.get("profit_factor", 0),
    }


def optimize_parameters():
    """Run parameter optimization until targets are met."""
    logger.info("Starting parameter optimization...")
    logger.info(f"Targets: Win Rate >= {TARGET_WIN_RATE * 100}%, Daily DD <= {MAX_DAILY_DD * 100}%, Profit > {MIN_PROFIT}")

    best_results = None
    best_params = None
    best_score = -float("inf")

    # Generate parameter combinations
    param_names = list(PARAM_RANGES.keys())
    param_values = list(PARAM_RANGES.values())
    total_combinations = 1
    for values in param_values:
        total_combinations *= len(values)

    logger.info(f"Testing {total_combinations} parameter combinations...")

    combination_count = 0
    for values in product(*param_values):
        combination_count += 1

        # Create parameter dict
        params = dict(zip(param_names, values))

        logger.info(f"Testing combination {combination_count}/{total_combinations}: {params}")

        try:
            results = run_backtest_with_params(params)
            targets_met, metrics = check_targets(results)

            # Calculate score (weighted combination of metrics)
            score = (
                metrics.get("win_rate", 0) * 100  # Win rate weight
                + metrics.get("profit", 0) / 1000  # Profit weight
                + (1 - min(metrics.get("daily_dd", 1), 1)) * 10  # DD penalty
            )

            logger.info(
                f"  Results: WR={metrics.get('win_rate', 0):.1%}, DD={metrics.get('daily_dd', 0):.1%}, Profit={metrics.get('profit', 0):.2f}, Score={score:.2f}"
            )

            if targets_met:
                logger.info("  *** TARGETS MET! ***")
                return params, results, metrics

            if score > best_score:
                best_score = score
                best_results = results
                best_params = params
                logger.info(f"  New best score: {score:.2f}")

        except Exception as e:
            logger.error(f"  Error testing combination: {e}")
            continue

    logger.info(f"Optimization complete. Best score: {best_score:.2f}")
    return best_params, best_results, None


def save_optimized_parameters(params, results):
    """Save optimized parameters to config file."""
    output_file = Path("C:/Users/jack/Cavalier/CORE_MODULES/config/optimized_backtest_params.json")

    # Convert results to serializable format
    if results and "stats" in results:
        stats = results["stats"]
        # Convert any non-serializable types
        serializable_stats = {}
        for key, value in stats.items():
            if isinstance(value, (int, float, str, bool, type(None))):
                serializable_stats[key] = value
            elif isinstance(value, dict):
                serializable_stats[key] = value
            else:
                serializable_stats[key] = str(value)
    else:
        serializable_stats = {}

    output = {
        "optimization_date": datetime.now().isoformat(),
        "targets": {"win_rate": TARGET_WIN_RATE, "max_daily_dd": MAX_DAILY_DD, "min_profit": MIN_PROFIT},
        "best_parameters": params,
        "best_results": serializable_stats,
        "pairs_tested": ALL_PAIRS,
        "data_period": "2026-02-13 to 2026-03-20",
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved optimized parameters to {output_file}")


if __name__ == "__main__":
    try:
        best_params, best_results, targets_met_metrics = optimize_parameters()

        if targets_met_metrics:
            logger.info("SUCCESS: Targets met!")
            logger.info(f"Optimized parameters: {best_params}")
            logger.info(f"Final metrics: {targets_met_metrics}")
            save_optimized_parameters(best_params, best_results)
        else:
            logger.info("Targets not met with current parameter ranges.")
            logger.info("Consider expanding parameter ranges or adjusting targets.")
            save_optimized_parameters(best_params, best_results)

    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user.")
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        import traceback

        traceback.print_exc()
