#!/usr/bin/env python3
"""
Fast Parameter Optimization
============================

Optimized version that runs faster to hit 70% win rate target.
Uses vectorized operations and pre-computed indicators.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PARQUET_DIR = Path("./sample_project/DATA_MODELS/data_parquet")
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

TARGET_WIN_RATE = 0.70
MAX_DAILY_DD = 0.03
MIN_PROFIT = 0

INSTRUMENT_CONFIGS = {
    "EURUSD": {"pip_size": 0.0001, "pip_value_per_lot": 10.0},
    "GBPUSD": {"pip_size": 0.0001, "pip_value_per_lot": 10.0},
    "USDJPY": {"pip_size": 0.01, "pip_value_per_lot": 1000.0},
    "AUDUSD": {"pip_size": 0.0001, "pip_value_per_lot": 10.0},
    "USDCAD": {"pip_size": 0.0001, "pip_value_per_lot": 10.0},
    "USDCHF": {"pip_size": 0.0001, "pip_value_per_lot": 10.0},
    "NZDUSD": {"pip_size": 0.0001, "pip_value_per_lot": 10.0},
    "GBPCHF": {"pip_size": 0.0001, "pip_value_per_lot": 10.0},
    "XAUUSD": {"pip_size": 0.01, "pip_value_per_lot": 1.0},
    "XAGUSD": {"pip_size": 0.001, "pip_value_per_lot": 5.0},
    "USOIL": {"pip_size": 0.01, "pip_value_per_lot": 10.0},
    "UKOIL": {"pip_size": 0.01, "pip_value_per_lot": 10.0},
    "HEATOIL": {"pip_size": 0.0001, "pip_value_per_lot": 4.2},
    "JP225": {"pip_size": 1.0, "pip_value_per_lot": 0.00685},
    "US100": {"pip_size": 0.1, "pip_value_per_lot": 20.0},
    "HK50": {"pip_size": 1.0, "pip_value_per_lot": 1.0},
    "UK100": {"pip_size": 0.1, "pip_value_per_lot": 10.0},
}


def get_instrument_config(symbol):
    return INSTRUMENT_CONFIGS.get(symbol, {"pip_size": 0.0001, "pip_value_per_lot": 10.0})


def calculate_position_size(symbol, balance, risk_percent, atr, regime):
    config = get_instrument_config(symbol)
    pip_size = config["pip_size"]
    pip_value_per_lot = config["pip_value_per_lot"]

    sl_multiplier = 1.5 if regime == "TRENDING" else (1.0 if regime == "RANGING" else 1.2)
    sl_distance = atr * sl_multiplier
    sl_pips = sl_distance / pip_size

    if sl_pips == 0:
        return 0.01

    risk_amount = balance * (risk_percent / 100)
    position_size = risk_amount / (sl_pips * pip_value_per_lot)
    return max(0.01, min(position_size, 10.0))


def calculate_pnl(symbol, direction, entry_price, exit_price, volume):
    config = get_instrument_config(symbol)
    pip_size = config["pip_size"]
    pip_value_per_lot = config["pip_value_per_lot"]

    price_diff = (exit_price - entry_price) if direction == 1 else (entry_price - exit_price)
    pips = price_diff / pip_size
    return pips * pip_value_per_lot * volume


def precompute_indicators(df):
    """Pre-compute all indicators for faster processing."""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # EMAs
    df["ema_fast"] = close.ewm(span=8, adjust=False).mean()
    df["ema_mid"] = close.ewm(span=21, adjust=False).mean()
    df["ema_slow"] = close.ewm(span=50, adjust=False).mean()

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # Momentum
    df["momentum"] = close.pct_change(10)

    # ATR
    high_low = high - low
    high_close = abs(high - close.shift(1))
    low_close = abs(low - close.shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(window=14).mean()

    # Price change for regime detection
    df["price_change_20"] = close.pct_change(20)

    # Volatility ratio
    avg_range = (high - low).rolling(window=10).mean()
    df["volatility_ratio"] = (high - low) / avg_range

    return df.dropna()


def detect_regime_fast(price_change, volatility_ratio):
    if abs(price_change) > 0.005:
        return "TRENDING"
    elif volatility_ratio > 1.5:
        return "VOLATILE"
    else:
        return "RANGING"


def run_fast_backtest(all_data, min_confidence=0.85, min_momentum=0.015, risk_per_trade=0.75):
    total_trades = 0
    winning_trades = 0
    total_pnl = 0.0
    max_drawdown = 0.0
    peak_balance = 10000
    current_balance = 10000

    for symbol, df in all_data.items():
        df = df.sort_values("time").reset_index(drop=True)
        df = precompute_indicators(df)

        for i in range(50, len(df)):
            row = df.iloc[i]

            # Quick filters
            regime = detect_regime_fast(row["price_change_20"], row["volatility_ratio"])
            if regime != "TRENDING":
                continue

            momentum = row["momentum"]
            if abs(momentum) < min_momentum:
                continue

            rsi = row["rsi"]
            if rsi < 30 or rsi > 70:
                continue

            ema_fast = row["ema_fast"]
            ema_mid = row["ema_mid"]
            ema_slow = row["ema_slow"]

            if momentum > 0:
                if ema_fast <= ema_mid or ema_mid <= ema_slow:
                    continue
                direction = 1
            else:
                if ema_fast >= ema_mid or ema_mid >= ema_slow:
                    continue
                direction = -1

            # Calculate confidence
            confidence = 0.7
            if abs(momentum) > 0.02:
                confidence += 0.1
            if 40 < rsi < 60:
                confidence += 0.1
            if abs(ema_fast - ema_slow) / ema_slow > 0.005:
                confidence += 0.1

            confidence = min(confidence, 0.95)

            if confidence < min_confidence:
                continue

            # Simulate trade
            entry_price = row["close"]
            atr = row["atr"]
            volume = calculate_position_size(symbol, current_balance, risk_per_trade, atr, regime)

            sl_multiplier = 1.5
            tp_multiplier = 3.0

            if direction == 1:
                sl_price = entry_price - (atr * sl_multiplier)
                tp_price = entry_price + (atr * tp_multiplier)
            else:
                sl_price = entry_price + (atr * sl_multiplier)
                tp_price = entry_price - (atr * tp_multiplier)

            # Look for exit
            for j in range(i + 1, len(df)):
                bar = df.iloc[j]
                bar_high = bar["high"]
                bar_low = bar["low"]

                if direction == 1:
                    if bar_low <= sl_price:
                        pnl = calculate_pnl(symbol, direction, entry_price, sl_price, volume)
                        break
                    if bar_high >= tp_price:
                        pnl = calculate_pnl(symbol, direction, entry_price, tp_price, volume)
                        break
                else:
                    if bar_high >= sl_price:
                        pnl = calculate_pnl(symbol, direction, entry_price, sl_price, volume)
                        break
                    if bar_low <= tp_price:
                        pnl = calculate_pnl(symbol, direction, entry_price, tp_price, volume)
                        break
            else:
                last_price = df.iloc[-1]["close"]
                pnl = calculate_pnl(symbol, direction, entry_price, last_price, volume)

            total_trades += 1
            total_pnl += pnl
            current_balance += pnl

            if pnl > 0:
                winning_trades += 1

            if current_balance > peak_balance:
                peak_balance = current_balance
            drawdown = (peak_balance - current_balance) / peak_balance
            if drawdown > max_drawdown:
                max_drawdown = drawdown

    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "max_drawdown": max_drawdown,
        "final_balance": current_balance,
    }


def load_all_data():
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

                start_date = "2026-02-13"
                end_date = "2026-03-20"
                if "time" in df.columns:
                    df = df[(df["time"] >= start_date) & (df["time"] <= end_date)]
                    if len(df) > 0:
                        all_data[pair] = df
                        break
    return all_data


if __name__ == "__main__":
    logger.info("Loading data...")
    all_data = load_all_data()
    logger.info(f"Loaded data for {len(all_data)} pairs")

    parameter_combinations = [
        {"min_confidence": 0.85, "min_momentum": 0.015, "risk_per_trade": 0.75},
        {"min_confidence": 0.90, "min_momentum": 0.020, "risk_per_trade": 0.50},
        {"min_confidence": 0.95, "min_momentum": 0.025, "risk_per_trade": 0.25},
        {"min_confidence": 0.98, "min_momentum": 0.030, "risk_per_trade": 0.10},
        {"min_confidence": 0.99, "min_momentum": 0.035, "risk_per_trade": 0.05},
    ]

    for i, params in enumerate(parameter_combinations):
        logger.info(f"\nTrying parameter combination {i + 1}: {params}")

        results = run_fast_backtest(all_data, **params)

        logger.info(f"Results: WR={results['win_rate']:.1%}, PnL=${results['total_pnl']:.2f}, DD={results['max_drawdown']:.1%}")

        targets_met = results["win_rate"] >= TARGET_WIN_RATE and results["max_drawdown"] <= MAX_DAILY_DD and results["total_pnl"] > MIN_PROFIT

        if targets_met:
            logger.info("\n*** TARGETS MET! ***")
            logger.info(f"Win rate: {results['win_rate']:.1%} >= {TARGET_WIN_RATE:.1%}")
            logger.info(f"Max drawdown: {results['max_drawdown']:.1%} <= {MAX_DAILY_DD:.1%}")
            logger.info(f"Total PnL: ${results['total_pnl']:.2f} > ${MIN_PROFIT:.2f}")

            output_file = Path("./sample_project/CORE_MODULES/config/optimized_params.json")
            with open(output_file, "w") as f:
                json.dump({"timestamp": datetime.now().isoformat(), "parameters": params, "results": results, "targets_met": True}, f, indent=2)
            logger.info(f"Saved optimized parameters to {output_file}")
            break
    else:
        logger.info("\nCould not meet targets with any parameter combination.")
        logger.info("The live system's ML models are needed for better signal generation.")
