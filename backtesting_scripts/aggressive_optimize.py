#!/usr/bin/env python3
"""
Aggressive Parameter Optimization
==================================

Loops through parameter combinations until 70% win rate is achieved.
Uses simplified signal generation with strict filters.
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

# Target parameters
TARGET_WIN_RATE = 0.70  # 70% win rate
MAX_DAILY_DD = 0.03  # 3% daily drawdown limit
MIN_PROFIT = 0  # Must be positive

# Instrument-specific configurations
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


def get_instrument_config(symbol: str):
    return INSTRUMENT_CONFIGS.get(symbol, {"pip_size": 0.0001, "pip_value_per_lot": 10.0})


def calculate_position_size(symbol: str, balance: float, risk_percent: float, atr: float, regime: str):
    config = get_instrument_config(symbol)
    pip_size = config["pip_size"]
    pip_value_per_lot = config["pip_value_per_lot"]

    sl_multiplier = 1.5
    if regime == "RANGING":
        sl_multiplier = 1.0
    elif regime == "VOLATILE":
        sl_multiplier = 1.2

    sl_distance = atr * sl_multiplier
    sl_pips = sl_distance / pip_size

    if sl_pips == 0:
        return 0.01

    risk_amount = balance * (risk_percent / 100)
    position_size = risk_amount / (sl_pips * pip_value_per_lot)
    return max(0.01, min(position_size, 10.0))


def calculate_pnl(symbol: str, direction: int, entry_price: float, exit_price: float, volume: float):
    config = get_instrument_config(symbol)
    pip_size = config["pip_size"]
    pip_value_per_lot = config["pip_value_per_lot"]

    if direction == 1:
        price_diff = exit_price - entry_price
    else:
        price_diff = entry_price - exit_price

    pips = price_diff / pip_size
    return pips * pip_value_per_lot * volume


def detect_regime(df: pd.DataFrame, idx: int):
    if idx < 20:
        return "RANGING"

    lookback = min(20, idx)
    start_price = df.iloc[idx - lookback]["close"]
    end_price = df.iloc[idx]["close"]
    price_change = (end_price - start_price) / start_price

    high_low_range = df.iloc[max(0, idx - 10) : idx + 1]["high"] - df.iloc[max(0, idx - 10) : idx + 1]["low"]
    avg_range = high_low_range.mean()
    current_range = df.iloc[idx]["high"] - df.iloc[idx]["low"]
    volatility_ratio = current_range / avg_range if avg_range > 0 else 1.0

    if abs(price_change) > 0.005:
        return "TRENDING"
    elif volatility_ratio > 1.5:
        return "VOLATILE"
    else:
        return "RANGING"


def generate_signal(df: pd.DataFrame, idx: int, symbol: str, min_confidence: float = 0.85, min_momentum: float = 0.015):
    if idx < 50:
        return None

    close = df.iloc[idx]["close"]

    # Calculate indicators
    ema_fast = df.iloc[max(0, idx - 8) : idx + 1]["close"].ewm(span=8, adjust=False).mean().iloc[-1]
    ema_mid = df.iloc[max(0, idx - 21) : idx + 1]["close"].ewm(span=21, adjust=False).mean().iloc[-1]
    ema_slow = df.iloc[max(0, idx - 50) : idx + 1]["close"].ewm(span=50, adjust=False).mean().iloc[-1]

    delta = df.iloc[max(0, idx - 14) : idx + 1]["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean().iloc[-1]
    avg_loss = loss.rolling(window=14).mean().iloc[-1]
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    momentum = (close - df.iloc[idx - 10]["close"]) / df.iloc[idx - 10]["close"]

    regime = detect_regime(df, idx)

    # Filters
    if regime != "TRENDING":
        return None

    if abs(momentum) < min_momentum:
        return None

    if rsi < 30 or rsi > 70:
        return None

    if momentum > 0:
        if ema_fast <= ema_mid or ema_mid <= ema_slow:
            return None
        direction = 1
    else:
        if ema_fast >= ema_mid or ema_mid >= ema_slow:
            return None
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
        return None

    high_low = df.iloc[max(0, idx - 14) : idx + 1]["high"] - df.iloc[max(0, idx - 14) : idx + 1]["low"]
    high_close = abs(df.iloc[max(0, idx - 14) : idx + 1]["high"] - df.iloc[max(0, idx - 14) : idx + 1]["close"].shift(1))
    low_close = abs(df.iloc[max(0, idx - 14) : idx + 1]["low"] - df.iloc[max(0, idx - 14) : idx + 1]["close"].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=14).mean().iloc[-1]

    return {"direction": direction, "confidence": confidence, "momentum": momentum, "rsi": rsi, "atr": atr, "regime": regime}


def simulate_trade(
    symbol: str, direction: int, entry_price: float, sl_price: float, tp_price: float, df: pd.DataFrame, entry_idx: int, volume: float
):
    for i in range(entry_idx + 1, len(df)):
        bar = df.iloc[i]
        bar_high = bar["high"]
        bar_low = bar["low"]

        if direction == 1:
            if bar_low <= sl_price:
                return {"exit_price": sl_price, "pnl": calculate_pnl(symbol, direction, entry_price, sl_price, volume), "exit_reason": "SL"}
            if bar_high >= tp_price:
                return {"exit_price": tp_price, "pnl": calculate_pnl(symbol, direction, entry_price, tp_price, volume), "exit_reason": "TP"}
        else:
            if bar_high >= sl_price:
                return {"exit_price": sl_price, "pnl": calculate_pnl(symbol, direction, entry_price, sl_price, volume), "exit_reason": "SL"}
            if bar_low <= tp_price:
                return {"exit_price": tp_price, "pnl": calculate_pnl(symbol, direction, entry_price, tp_price, volume), "exit_reason": "TP"}

    last_price = df.iloc[-1]["close"]
    return {"exit_price": last_price, "pnl": calculate_pnl(symbol, direction, entry_price, last_price, volume), "exit_reason": "END_OF_DATA"}


def run_backtest(all_data: dict, balance: float = 10000, risk_per_trade: float = 0.75, min_confidence: float = 0.85, min_momentum: float = 0.015):
    total_trades = 0
    winning_trades = 0
    total_pnl = 0.0
    max_drawdown = 0.0
    peak_balance = balance
    current_balance = balance

    for symbol, df in all_data.items():
        df = df.sort_values("time").reset_index(drop=True)

        for i in range(50, len(df)):
            signal = generate_signal(df, i, symbol, min_confidence, min_momentum)
            if signal is None:
                continue

            volume = calculate_position_size(symbol, current_balance, risk_per_trade, signal["atr"], signal["regime"])
            entry_price = df.iloc[i]["close"]
            atr = signal["atr"]

            sl_multiplier = 1.5
            tp_multiplier = 3.0
            if signal["regime"] == "RANGING":
                sl_multiplier = 1.0
                tp_multiplier = 2.0
            elif signal["regime"] == "VOLATILE":
                sl_multiplier = 1.2
                tp_multiplier = 2.5

            if signal["direction"] == 1:
                sl_price = entry_price - (atr * sl_multiplier)
                tp_price = entry_price + (atr * tp_multiplier)
            else:
                sl_price = entry_price + (atr * sl_multiplier)
                tp_price = entry_price - (atr * tp_multiplier)

            trade_result = simulate_trade(symbol, signal["direction"], entry_price, sl_price, tp_price, df, i, volume)

            total_trades += 1
            total_pnl += trade_result["pnl"]
            current_balance += trade_result["pnl"]

            if trade_result["pnl"] > 0:
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

    # Try different parameter combinations
    parameter_combinations = [
        {"min_confidence": 0.85, "min_momentum": 0.015, "risk_per_trade": 0.75},
        {"min_confidence": 0.90, "min_momentum": 0.020, "risk_per_trade": 0.50},
        {"min_confidence": 0.95, "min_momentum": 0.025, "risk_per_trade": 0.25},
        {"min_confidence": 0.98, "min_momentum": 0.030, "risk_per_trade": 0.10},
    ]

    for i, params in enumerate(parameter_combinations):
        logger.info(f"\nTrying parameter combination {i + 1}: {params}")

        results = run_backtest(
            all_data=all_data,
            balance=10000,
            risk_per_trade=params["risk_per_trade"],
            min_confidence=params["min_confidence"],
            min_momentum=params["min_momentum"],
        )

        logger.info(f"Results: WR={results['win_rate']:.1%}, PnL=${results['total_pnl']:.2f}, DD={results['max_drawdown']:.1%}")

        # Check if targets met
        targets_met = results["win_rate"] >= TARGET_WIN_RATE and results["max_drawdown"] <= MAX_DAILY_DD and results["total_pnl"] > MIN_PROFIT

        if targets_met:
            logger.info("\n*** TARGETS MET! ***")
            logger.info(f"Win rate: {results['win_rate']:.1%} >= {TARGET_WIN_RATE:.1%}")
            logger.info(f"Max drawdown: {results['max_drawdown']:.1%} <= {MAX_DAILY_DD:.1%}")
            logger.info(f"Total PnL: ${results['total_pnl']:.2f} > ${MIN_PROFIT:.2f}")

            # Save results
            output_file = Path("./sample_project/CORE_MODULES/config/optimized_params.json")
            with open(output_file, "w") as f:
                json.dump({"timestamp": datetime.now().isoformat(), "parameters": params, "results": results, "targets_met": True}, f, indent=2)
            logger.info(f"Saved optimized parameters to {output_file}")
            break
    else:
        logger.info("\nCould not meet targets with any parameter combination.")
        logger.info("Consider using live system's ML models for better signal generation.")
