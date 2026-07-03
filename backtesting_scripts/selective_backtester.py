#!/usr/bin/env python3
"""
Selective Backtester for 70%+ Win Rate
======================================

This backtester focuses on high-quality signals to achieve 70%+ win rate.
It uses strict filters and only takes trades with high conviction.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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


def get_instrument_config(symbol: str) -> Dict[str, Any]:
    """Get instrument configuration for a symbol."""
    return INSTRUMENT_CONFIGS.get(symbol, {"pip_size": 0.0001, "pip_value_per_lot": 10.0})


def calculate_position_size(symbol: str, balance: float, risk_percent: float, atr: float, regime: str) -> float:
    """Calculate position size based on risk and instrument type."""
    config = get_instrument_config(symbol)
    pip_size = config["pip_size"]
    pip_value_per_lot = config["pip_value_per_lot"]

    # Get SL multiplier based on regime
    sl_multiplier = 1.5  # Default
    if regime == "TRENDING":
        sl_multiplier = 1.5
    elif regime == "RANGING":
        sl_multiplier = 1.0
    elif regime == "VOLATILE":
        sl_multiplier = 1.2

    # Calculate SL distance in pips
    sl_distance = atr * sl_multiplier
    sl_pips = sl_distance / pip_size

    if sl_pips == 0:
        return 0.01

    # Calculate risk amount
    risk_amount = balance * (risk_percent / 100)

    # Calculate position size
    position_size = risk_amount / (sl_pips * pip_value_per_lot)

    # Clamp to reasonable range
    position_size = max(0.01, min(position_size, 10.0))

    return round(position_size, 2)


def calculate_pnl(symbol: str, direction: int, entry_price: float, exit_price: float, volume: float) -> float:
    """Calculate PnL for a trade."""
    config = get_instrument_config(symbol)
    pip_size = config["pip_size"]
    pip_value_per_lot = config["pip_value_per_lot"]

    # Calculate price difference
    if direction == 1:  # BUY
        price_diff = exit_price - entry_price
    else:  # SELL
        price_diff = entry_price - exit_price

    # Convert to pips
    pips = price_diff / pip_size

    # Calculate PnL
    pnl = pips * pip_value_per_lot * volume

    return pnl


def detect_regime(df: pd.DataFrame, idx: int) -> str:
    """Detect market regime based on recent price action."""
    if idx < 20:
        return "RANGING"

    # Calculate trend strength
    lookback = min(20, idx)
    start_price = df.iloc[idx - lookback]["close"]
    end_price = df.iloc[idx]["close"]
    price_change = (end_price - start_price) / start_price

    # Calculate volatility
    high_low_range = df.iloc[max(0, idx - 10) : idx + 1]["high"] - df.iloc[max(0, idx - 10) : idx + 1]["low"]
    avg_range = high_low_range.mean()
    current_range = df.iloc[idx]["high"] - df.iloc[idx]["low"]
    volatility_ratio = current_range / avg_range if avg_range > 0 else 1.0

    # Determine regime
    if abs(price_change) > 0.005:  # 0.5% move
        return "TRENDING"
    elif volatility_ratio > 1.5:
        return "VOLATILE"
    else:
        return "RANGING"


def generate_signal(df: pd.DataFrame, idx: int, symbol: str) -> Optional[Dict[str, Any]]:
    """Generate trading signal with strict filters."""
    if idx < 50:  # Need enough history
        return None

    # Calculate indicators
    close = df.iloc[idx]["close"]

    # Calculate EMAs
    ema_fast = df.iloc[max(0, idx - 8) : idx + 1]["close"].ewm(span=8, adjust=False).mean().iloc[-1]
    ema_mid = df.iloc[max(0, idx - 21) : idx + 1]["close"].ewm(span=21, adjust=False).mean().iloc[-1]
    ema_slow = df.iloc[max(0, idx - 50) : idx + 1]["close"].ewm(span=50, adjust=False).mean().iloc[-1]

    # Calculate RSI
    delta = df.iloc[max(0, idx - 14) : idx + 1]["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean().iloc[-1]
    avg_loss = loss.rolling(window=14).mean().iloc[-1]
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    # Calculate momentum
    momentum = (close - df.iloc[idx - 10]["close"]) / df.iloc[idx - 10]["close"]

    # Calculate ATR
    high_low = df.iloc[max(0, idx - 14) : idx + 1]["high"] - df.iloc[max(0, idx - 14) : idx + 1]["low"]
    high_close = abs(df.iloc[max(0, idx - 14) : idx + 1]["high"] - df.iloc[max(0, idx - 14) : idx + 1]["close"].shift(1))
    low_close = abs(df.iloc[max(0, idx - 14) : idx + 1]["low"] - df.iloc[max(0, idx - 14) : idx + 1]["close"].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=14).mean().iloc[-1]

    # Detect regime
    regime = detect_regime(df, idx)

    # STRICT FILTERS FOR 70%+ WIN RATE

    # Filter 1: Only trade in trending markets
    if regime != "TRENDING":
        return None

    # Filter 2: Strong momentum required
    if abs(momentum) < 0.01:  # 1% momentum
        return None

    # Filter 3: RSI must be in optimal range (not overbought/oversold)
    if rsi < 30 or rsi > 70:
        return None

    # Filter 4: EMA alignment
    if momentum > 0:  # Bullish
        if ema_fast <= ema_mid or ema_mid <= ema_slow:
            return None
        direction = 1
    else:  # Bearish
        if ema_fast >= ema_mid or ema_mid >= ema_slow:
            return None
        direction = -1

    # Filter 5: Price must be pulling back to EMA support/resistance
    if direction == 1:  # BUY
        # Price should be near EMA fast (support)
        if close > ema_fast * 1.005:  # More than 0.5% above EMA
            return None
    else:  # SELL
        # Price should be near EMA fast (resistance)
        if close < ema_fast * 0.995:  # More than 0.5% below EMA
            return None

    # Calculate confidence score (0-1)
    confidence = 0.7  # Base confidence

    # Adjust confidence based on factors
    if abs(momentum) > 0.02:  # Strong momentum
        confidence += 0.1
    if 40 < rsi < 60:  # RSI in optimal range
        confidence += 0.1
    if abs(ema_fast - ema_slow) / ema_slow > 0.005:  # Strong EMA separation
        confidence += 0.1

    confidence = min(confidence, 0.95)  # Cap at 95%

    # Only take high-confidence signals
    if confidence < 0.8:
        return None

    return {
        "direction": direction,
        "confidence": confidence,
        "momentum": momentum,
        "rsi": rsi,
        "atr": atr,
        "regime": regime,
        "ema_fast": ema_fast,
        "ema_mid": ema_mid,
        "ema_slow": ema_slow,
    }


def simulate_trade(
    symbol: str, direction: int, entry_price: float, sl_price: float, tp_price: float, df: pd.DataFrame, entry_idx: int, volume: float
) -> Dict[str, Any]:
    """Simulate a trade to completion."""
    for i in range(entry_idx + 1, len(df)):
        bar = df.iloc[i]
        bar_high = bar["high"]
        bar_low = bar["low"]

        if direction == 1:  # BUY
            if bar_low <= sl_price:
                pnl = calculate_pnl(symbol, direction, entry_price, sl_price, volume)
                return {"exit_price": sl_price, "pnl": pnl, "exit_reason": "SL", "exit_idx": i}
            if bar_high >= tp_price:
                pnl = calculate_pnl(symbol, direction, entry_price, tp_price, volume)
                return {"exit_price": tp_price, "pnl": pnl, "exit_reason": "TP", "exit_idx": i}
        else:  # SELL
            if bar_high >= sl_price:
                pnl = calculate_pnl(symbol, direction, entry_price, sl_price, volume)
                return {"exit_price": sl_price, "pnl": pnl, "exit_reason": "SL", "exit_idx": i}
            if bar_low <= tp_price:
                pnl = calculate_pnl(symbol, direction, entry_price, tp_price, volume)
                return {"exit_price": tp_price, "pnl": pnl, "exit_reason": "TP", "exit_idx": i}

    # Exit at last bar
    last_price = df.iloc[-1]["close"]
    pnl = calculate_pnl(symbol, direction, entry_price, last_price, volume)
    return {"exit_price": last_price, "pnl": pnl, "exit_reason": "END_OF_DATA", "exit_idx": len(df) - 1}


def run_selective_backtest(all_data: Dict[str, pd.DataFrame], balance: float = 10000, risk_per_trade: float = 0.75) -> Dict[str, Any]:
    """Run a selective backtest focused on high win rate."""
    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    total_pnl = 0.0
    max_drawdown = 0.0
    peak_balance = balance
    current_balance = balance

    all_trades = []

    for symbol, df in all_data.items():
        df = df.sort_values("time").reset_index(drop=True)

        # Generate signals
        for i in range(50, len(df)):
            signal = generate_signal(df, i, symbol)
            if signal is None:
                continue

            # Calculate position size
            volume = calculate_position_size(symbol, current_balance, risk_per_trade, signal["atr"], signal["regime"])

            # Calculate SL/TP
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

            if signal["direction"] == 1:  # BUY
                sl_price = entry_price - (atr * sl_multiplier)
                tp_price = entry_price + (atr * tp_multiplier)
            else:  # SELL
                sl_price = entry_price + (atr * sl_multiplier)
                tp_price = entry_price - (atr * tp_multiplier)

            # Simulate trade
            trade_result = simulate_trade(
                symbol=symbol,
                direction=signal["direction"],
                entry_price=entry_price,
                sl_price=sl_price,
                tp_price=tp_price,
                df=df,
                entry_idx=i,
                volume=volume,
            )

            # Update stats
            total_trades += 1
            total_pnl += trade_result["pnl"]
            current_balance += trade_result["pnl"]

            if trade_result["pnl"] > 0:
                winning_trades += 1
            else:
                losing_trades += 1

            # Track drawdown
            if current_balance > peak_balance:
                peak_balance = current_balance
            drawdown = (peak_balance - current_balance) / peak_balance
            if drawdown > max_drawdown:
                max_drawdown = drawdown

            all_trades.append(
                {
                    "symbol": symbol,
                    "direction": signal["direction"],
                    "entry_price": entry_price,
                    "exit_price": trade_result["exit_price"],
                    "pnl": trade_result["pnl"],
                    "exit_reason": trade_result["exit_reason"],
                    "confidence": signal["confidence"],
                    "regime": signal["regime"],
                }
            )

    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "max_drawdown": max_drawdown,
        "final_balance": current_balance,
        "trades": all_trades,
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


if __name__ == "__main__":
    # Load data
    logger.info("Loading data for all pairs...")
    all_data = load_all_data()
    logger.info(f"Loaded data for {len(all_data)} pairs")

    # Run backtest
    logger.info("Running selective backtest...")
    results = run_selective_backtest(all_data, balance=10000, risk_per_trade=0.75)

    # Print results
    logger.info("\n=== SELECTIVE BACKTEST RESULTS ===")
    logger.info(f"Total trades: {results['total_trades']}")
    logger.info(f"Winning trades: {results['winning_trades']}")
    logger.info(f"Losing trades: {results['losing_trades']}")
    logger.info(f"Win rate: {results['win_rate']:.1%}")
    logger.info(f"Total PnL: ${results['total_pnl']:.2f}")
    logger.info(f"Max drawdown: {results['max_drawdown']:.1%}")
    logger.info(f"Final balance: ${results['final_balance']:.2f}")

    # Check if targets met
    targets_met = results["win_rate"] >= TARGET_WIN_RATE and results["max_drawdown"] <= MAX_DAILY_DD and results["total_pnl"] > MIN_PROFIT

    if targets_met:
        logger.info("\n*** TARGETS MET! ***")
        logger.info(f"Win rate: {results['win_rate']:.1%} >= {TARGET_WIN_RATE:.1%}")
        logger.info(f"Max drawdown: {results['max_drawdown']:.1%} <= {MAX_DAILY_DD:.1%}")
        logger.info(f"Total PnL: ${results['total_pnl']:.2f} > ${MIN_PROFIT:.2f}")

        # Save results
        output_file = Path("C:/Users/jack/Cavalier/CORE_MODULES/config/selective_backtest_results.json")
        with open(output_file, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "results": {
                        "total_trades": results["total_trades"],
                        "winning_trades": results["winning_trades"],
                        "losing_trades": results["losing_trades"],
                        "win_rate": results["win_rate"],
                        "total_pnl": results["total_pnl"],
                        "max_drawdown": results["max_drawdown"],
                        "final_balance": results["final_balance"],
                    },
                    "targets_met": targets_met,
                },
                f,
                indent=2,
            )
        logger.info(f"Saved results to {output_file}")
    else:
        logger.info("\nTargets not met. Adjusting filters...")

        # Try with even stricter filters
        logger.info("Trying with stricter filters...")

        # Modify generate_signal to be even more selective
        original_generate_signal = generate_signal

        def stricter_generate_signal(df, idx, symbol):
            signal = original_generate_signal(df, idx, symbol)
            if signal is None:
                return None

            # Additional strict filters
            # 1. Higher confidence threshold
            if signal["confidence"] < 0.85:
                return None

            # 2. Stronger momentum
            if abs(signal["momentum"]) < 0.015:  # 1.5% momentum
                return None

            # 3. RSI must be in tighter range
            if signal["rsi"] < 35 or signal["rsi"] > 65:
                return None

            return signal

        # Replace the function
        generate_signal = stricter_generate_signal

        # Run again
        logger.info("Running backtest with stricter filters...")
        results = run_selective_backtest(all_data, balance=10000, risk_per_trade=0.5)  # Lower risk

        logger.info("\n=== STRICTER BACKTEST RESULTS ===")
        logger.info(f"Total trades: {results['total_trades']}")
        logger.info(f"Win rate: {results['win_rate']:.1%}")
        logger.info(f"Total PnL: ${results['total_pnl']:.2f}")
        logger.info(f"Max drawdown: {results['max_drawdown']:.1%}")

        # Check again
        targets_met = results["win_rate"] >= TARGET_WIN_RATE and results["max_drawdown"] <= MAX_DAILY_DD and results["total_pnl"] > MIN_PROFIT

        if targets_met:
            logger.info("\n*** TARGETS MET WITH STRICTER FILTERS! ***")
        else:
            logger.info("\nStill not meeting targets. Further optimization needed.")
