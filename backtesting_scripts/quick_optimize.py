#!/usr/bin/env python3
"""
Quick Parameter Optimization
=============================

Focused parameter optimization to hit 70% win rate.
Tests a smaller set of parameter combinations based on analysis of live system.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json

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

# Focused parameter combinations based on analysis
PARAM_COMBINATIONS = [
    # Live system defaults (from constants.py)
    {
        "min_confidence": 0.70,
        "max_daily_trades": 8,
        "risk_per_trade": 0.0075,
        "min_confirming_factors": 1,
        "min_momentum": 0.25,
        "session_trade_cap": 30,
    },
    # More restrictive (higher win rate focus)
    {
        "min_confidence": 0.75,
        "max_daily_trades": 5,
        "risk_per_trade": 0.0075,
        "min_confirming_factors": 2,
        "min_momentum": 0.30,
        "session_trade_cap": 20,
    },
    {
        "min_confidence": 0.80,
        "max_daily_trades": 4,
        "risk_per_trade": 0.0075,
        "min_confirming_factors": 2,
        "min_momentum": 0.35,
        "session_trade_cap": 15,
    },
    {
        "min_confidence": 0.85,
        "max_daily_trades": 3,
        "risk_per_trade": 0.0075,
        "min_confirming_factors": 3,
        "min_momentum": 0.40,
        "session_trade_cap": 10,
    },
    # Lower risk (reduce drawdown)
    {
        "min_confidence": 0.70,
        "max_daily_trades": 8,
        "risk_per_trade": 0.005,
        "min_confirming_factors": 1,
        "min_momentum": 0.25,
        "session_trade_cap": 30,
    },
    {
        "min_confidence": 0.75,
        "max_daily_trades": 5,
        "risk_per_trade": 0.005,
        "min_confirming_factors": 2,
        "min_momentum": 0.30,
        "session_trade_cap": 20,
    },
    # Very restrictive (high win rate, few trades)
    {
        "min_confidence": 0.90,
        "max_daily_trades": 2,
        "risk_per_trade": 0.01,
        "min_confirming_factors": 4,
        "min_momentum": 0.50,
        "session_trade_cap": 5,
    },
]


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

    # Override global constants for this run
    import CORE_MODULES.backtesting.live_trading_backtester as ltb

    ltb.MIN_CONFIRMING_FACTORS = params["min_confirming_factors"]
    ltb.MIN_MOMENTUM = params["min_momentum"]
    ltb.SESSION_TRADE_CAP = params["session_trade_cap"]
    ltb.MAX_DAILY_TRADES_PER_SYMBOL = params["max_daily_trades"]
    ltb.MIN_CONFIDENCE = params["min_confidence"]

    backtester = LiveTradingBacktester(
        initial_balance=10000,
        risk_per_trade=params["risk_per_trade"],
        max_daily_trades=params["max_daily_trades"],
        min_confidence=params["min_confidence"],
        max_positions=1,
    )

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
    logger.info("Starting quick parameter optimization...")
    logger.info(f"Targets: Win Rate >= {TARGET_WIN_RATE * 100}%, Daily DD <= {MAX_DAILY_DD * 100}%, Profit > {MIN_PROFIT}")

    best_results = None
    best_params = None
    best_score = -float("inf")
    best_metrics = None

    for i, params in enumerate(PARAM_COMBINATIONS):
        logger.info(f"Testing combination {i + 1}/{len(PARAM_COMBINATIONS)}: {params}")

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
                best_metrics = metrics
                logger.info(f"  New best score: {score:.2f}")

        except Exception as e:
            logger.error(f"  Error testing combination: {e}")
            import traceback

            traceback.print_exc()
            continue

    logger.info(f"Optimization complete. Best score: {best_score:.2f}")
    if best_metrics:
        logger.info(
            f"Best results: WR={best_metrics.get('win_rate', 0):.1%}, DD={best_metrics.get('daily_dd', 0):.1%}, Profit={best_metrics.get('profit', 0):.2f}"
        )
    return best_params, best_results, best_metrics


def save_optimized_parameters(params, results, metrics):
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
        "best_metrics": metrics or {},
        "pairs_tested": ALL_PAIRS,
        "data_period": "2026-02-13 to 2026-03-20",
        "note": "Parameters generated by quick_optimize.py. Update live system with these values if targets are met.",
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved optimized parameters to {output_file}")


if __name__ == "__main__":
    try:
        best_params, best_results, best_metrics = optimize_parameters()

        if best_metrics:
            # Check if targets were met with best parameters
            targets_met = (
                best_metrics.get("win_rate", 0) >= TARGET_WIN_RATE
                and best_metrics.get("daily_dd", 1) <= MAX_DAILY_DD
                and best_metrics.get("profit", 0) > MIN_PROFIT
            )

            if targets_met:
                logger.info("SUCCESS: Targets met!")
                logger.info(f"Optimized parameters: {best_params}")
                logger.info(f"Final metrics: {best_metrics}")
                save_optimized_parameters(best_params, best_results, best_metrics)

                # Update constants.py with optimized parameters
                logger.info("To update live system, modify CORE_MODULES/core/config/constants.py with these values:")
                logger.info(f"  MIN_CONFIDENCE_GOVERNANCE = {best_params['min_confidence']}")
                logger.info(f"  MAX_DAILY_TRADES_PER_SYMBOL = {best_params['max_daily_trades']}")
                logger.info(f"  BASE_RISK_PER_TRADE = {best_params['risk_per_trade']}")
                logger.info(f"  MIN_CONFIRMING_FACTORS = {best_params['min_confirming_factors']}")
                logger.info(f"  MIN_MOMENTUM = {best_params['min_momentum']}")
                logger.info(f"  SESSION_TRADE_CAP = {best_params['session_trade_cap']}")
            else:
                logger.info("Targets not met with current parameter combinations.")
                logger.info("Best results achieved:")
                logger.info(f"  Win rate: {best_metrics.get('win_rate', 0):.1%} (target: {TARGET_WIN_RATE:.1%})")
                logger.info(f"  Daily DD: {best_metrics.get('daily_dd', 0):.1%} (target: <{MAX_DAILY_DD:.1%})")
                logger.info(f"  Profit: {best_metrics.get('profit', 0):.2f} (target: >{MIN_PROFIT})")
                save_optimized_parameters(best_params, best_results, best_metrics)
        else:
            logger.error("No valid results obtained.")

    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user.")
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        import traceback

        traceback.print_exc()
