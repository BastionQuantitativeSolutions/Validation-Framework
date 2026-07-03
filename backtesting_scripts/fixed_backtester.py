#!/usr/bin/env python3
"""
Fixed Backtester with Proper Instrument Handling
================================================

This backtester correctly handles different instrument types:
- Forex pairs (including JPY pairs)
- Metals (XAUUSD, XAGUSD)
- Energies (USOIL, UKOIL, HEATOIL)
- Indices (JP225, US100, HK50, UK100)

Uses proper pip values and position sizing for each instrument type.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Instrument configurations
INSTRUMENT_CONFIGS = {
    # Forex majors
    "EURUSD": {"type": "FOREX", "pip_size": 0.0001, "pip_value_per_lot": 10, "min_lot": 0.01},
    "GBPUSD": {"type": "FOREX", "pip_size": 0.0001, "pip_value_per_lot": 10, "min_lot": 0.01},
    "USDJPY": {"type": "FOREX_JPY", "pip_size": 0.01, "pip_value_per_lot": 1000, "min_lot": 0.01},  # ~$10 per 0.01 move
    "AUDUSD": {"type": "FOREX", "pip_size": 0.0001, "pip_value_per_lot": 10, "min_lot": 0.01},
    "USDCAD": {"type": "FOREX", "pip_size": 0.0001, "pip_value_per_lot": 10, "min_lot": 0.01},
    "USDCHF": {"type": "FOREX", "pip_size": 0.0001, "pip_value_per_lot": 10, "min_lot": 0.01},
    "NZDUSD": {"type": "FOREX", "pip_size": 0.0001, "pip_value_per_lot": 10, "min_lot": 0.01},
    "GBPCHF": {"type": "FOREX", "pip_size": 0.0001, "pip_value_per_lot": 10, "min_lot": 0.01},
    # Metals
    "XAUUSD": {"type": "METAL", "pip_size": 0.01, "pip_value_per_lot": 1, "min_lot": 0.01},  # 1 lot = 100 oz, $1 per $1 move
    "XAGUSD": {"type": "METAL", "pip_size": 0.001, "pip_value_per_lot": 5, "min_lot": 0.01},  # 1 lot = 5000 oz, $5 per $0.01 move
    # Energies
    "USOIL": {"type": "ENERGY", "pip_size": 0.01, "pip_value_per_lot": 10, "min_lot": 0.01},  # 1 lot = 1000 barrels, $10 per $0.01
    "UKOIL": {"type": "ENERGY", "pip_size": 0.01, "pip_value_per_lot": 10, "min_lot": 0.01},
    "HEATOIL": {"type": "ENERGY", "pip_size": 0.0001, "pip_value_per_lot": 4.2, "min_lot": 0.01},  # 1 lot = 42000 gallons
    # Indices
    "JP225": {"type": "INDEX", "pip_size": 1, "pip_value_per_lot": 0.00685, "min_lot": 0.01},  # ~$6.85 per point
    "US100": {"type": "INDEX", "pip_size": 0.1, "pip_value_per_lot": 20, "min_lot": 0.01},  # $20 per point
    "HK50": {"type": "INDEX", "pip_size": 1, "pip_value_per_lot": 1, "min_lot": 0.01},  # $1 per point
    "UK100": {"type": "INDEX", "pip_size": 0.1, "pip_value_per_lot": 10, "min_lot": 0.01},  # $10 per point
}

logger = logging.getLogger(__name__)


@dataclass
class SimpleTrade:
    symbol: str
    direction: int  # 1=BUY, -1=SELL
    entry_time: datetime
    entry_price: float
    sl_price: float
    tp_price: float
    atr: float
    regime: str
    confidence: float
    volume: float = 0.01

    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    exit_reason: str = ""


def get_instrument_config(symbol: str) -> Dict[str, Any]:
    """Get instrument configuration for a symbol."""
    return INSTRUMENT_CONFIGS.get(symbol, {"type": "UNKNOWN", "pip_size": 0.0001, "pip_value_per_lot": 10, "min_lot": 0.01})


def calculate_position_size(symbol: str, balance: float, risk_percent: float, atr: float, regime: str = "DEFAULT") -> float:
    """Calculate position size based on risk and instrument type."""
    config = get_instrument_config(symbol)

    # Get SL multiplier based on regime
    sl_multiplier = 1.5  # Default
    if regime == "TRENDING":
        sl_multiplier = 1.5
    elif regime == "RANGING":
        sl_multiplier = 1.0
    elif regime == "VOLATILE":
        sl_multiplier = 1.2

    # Calculate SL distance in price
    sl_distance = atr * sl_multiplier

    # Convert to pips
    pips = sl_distance / config["pip_size"]

    if pips == 0:
        return config["min_lot"]

    # Calculate risk amount
    risk_amount = balance * (risk_percent / 100)

    # Calculate position size
    # Risk amount = pips * pip_value_per_lot * volume
    volume = risk_amount / (pips * config["pip_value_per_lot"])

    # Clamp to min/max
    volume = max(config["min_lot"], min(volume, 10.0))  # Max 10 lots

    return round(volume, 2)


def calculate_pnl(symbol: str, direction: int, entry_price: float, exit_price: float, volume: float) -> float:
    """Calculate PnL for a trade."""
    config = get_instrument_config(symbol)

    # Calculate price difference
    if direction == 1:  # BUY
        price_diff = exit_price - entry_price
    else:  # SELL
        price_diff = entry_price - exit_price

    # Convert to pips
    pips = price_diff / config["pip_size"]

    # Calculate PnL
    pnl = pips * config["pip_value_per_lot"] * volume

    return pnl


def simulate_trade(trade: SimpleTrade, df: pd.DataFrame) -> SimpleTrade:
    """Simulate a trade to completion."""
    # Find entry bar
    entry_mask = df["time"] == trade.entry_time
    if not entry_mask.any():
        trade.exit_time = df.iloc[-1]["time"]
        trade.exit_price = df.iloc[-1]["close"]
        trade.pnl = calculate_pnl(trade.symbol, trade.direction, trade.entry_price, trade.exit_price, trade.volume)
        trade.exit_reason = "NO_ENTRY_BAR"
        return trade

    entry_idx = df[entry_mask].index[0]
    entry_pos = df.index.get_loc(entry_idx)

    # Look for exit
    for i in range(entry_pos + 1, len(df)):
        bar = df.iloc[i]
        bar_high = bar["high"]
        bar_low = bar["low"]

        if trade.direction == 1:  # BUY
            # Check SL
            if bar_low <= trade.sl_price:
                trade.exit_time = bar["time"]
                trade.exit_price = trade.sl_price
                trade.pnl = calculate_pnl(trade.symbol, trade.direction, trade.entry_price, trade.sl_price, trade.volume)
                trade.exit_reason = "SL"
                return trade

            # Check TP
            if bar_high >= trade.tp_price:
                trade.exit_time = bar["time"]
                trade.exit_price = trade.tp_price
                trade.pnl = calculate_pnl(trade.symbol, trade.direction, trade.entry_price, trade.tp_price, trade.volume)
                trade.exit_reason = "TP"
                return trade

        else:  # SELL
            # Check SL
            if bar_high >= trade.sl_price:
                trade.exit_time = bar["time"]
                trade.exit_price = trade.sl_price
                trade.pnl = calculate_pnl(trade.symbol, trade.direction, trade.entry_price, trade.sl_price, trade.volume)
                trade.exit_reason = "SL"
                return trade

            # Check TP
            if bar_low <= trade.tp_price:
                trade.exit_time = bar["time"]
                trade.exit_price = trade.tp_price
                trade.pnl = calculate_pnl(trade.symbol, trade.direction, trade.entry_price, trade.tp_price, trade.volume)
                trade.exit_reason = "TP"
                return trade

    # If no exit found, exit at last bar
    trade.exit_time = df.iloc[-1]["time"]
    trade.exit_price = df.iloc[-1]["close"]
    trade.pnl = calculate_pnl(trade.symbol, trade.direction, trade.entry_price, trade.exit_price, trade.volume)
    trade.exit_reason = "END_OF_DATA"
    return trade


def run_simple_backtest(all_data: Dict[str, pd.DataFrame], balance: float = 10000, risk_per_trade: float = 0.75) -> Dict[str, Any]:
    """Run a simple backtest on all data."""
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

        # Simple signal generation: look for momentum
        for i in range(20, len(df)):
            # Calculate simple momentum
            price_change = (df.iloc[i]["close"] - df.iloc[i - 20]["close"]) / df.iloc[i - 20]["close"]

            # Only trade if momentum is significant (>0.5%)
            if abs(price_change) > 0.005:
                # Calculate ATR (simplified)
                atr = df.iloc[max(0, i - 14) : i + 1]["high"].sub(df.iloc[max(0, i - 14) : i + 1]["low"]).mean()
                if atr == 0:
                    continue

                # Determine direction
                direction = 1 if price_change > 0 else -1

                # Determine regime (simplified)
                volatility = df.iloc[max(0, i - 10) : i + 1]["high"].std() / df.iloc[i]["close"]
                if volatility > 0.02:
                    regime = "VOLATILE"
                elif abs(price_change) > 0.001:
                    regime = "TRENDING"
                else:
                    regime = "RANGING"

                # Calculate position size
                volume = calculate_position_size(symbol, current_balance, risk_per_trade, atr, regime)

                # Calculate SL/TP
                get_instrument_config(symbol)
                sl_multiplier = 1.5
                tp_multiplier = 3.0
                if regime == "RANGING":
                    sl_multiplier = 1.0
                    tp_multiplier = 2.0
                elif regime == "VOLATILE":
                    sl_multiplier = 1.2
                    tp_multiplier = 2.5

                sl_distance = atr * sl_multiplier
                tp_distance = atr * tp_multiplier

                if direction == 1:  # BUY
                    sl_price = df.iloc[i]["close"] - sl_distance
                    tp_price = df.iloc[i]["close"] + tp_distance
                else:  # SELL
                    sl_price = df.iloc[i]["close"] + sl_distance
                    tp_price = df.iloc[i]["close"] - tp_distance

                # Create trade
                trade = SimpleTrade(
                    symbol=symbol,
                    direction=direction,
                    entry_time=df.iloc[i]["time"],
                    entry_price=df.iloc[i]["close"],
                    sl_price=sl_price,
                    tp_price=tp_price,
                    atr=atr,
                    regime=regime,
                    confidence=0.7,
                    volume=volume,
                )

                # Simulate trade
                trade = simulate_trade(trade, df)

                # Update stats
                total_trades += 1
                total_pnl += trade.pnl
                current_balance += trade.pnl

                if trade.pnl > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1

                # Track drawdown
                if current_balance > peak_balance:
                    peak_balance = current_balance
                drawdown = (peak_balance - current_balance) / peak_balance
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

                all_trades.append(trade)

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


if __name__ == "__main__":
    # Test the backtester
    import pandas as pd
    from pathlib import Path

    data_dir = Path("DATA_MODELS/data_parquet")
    pairs = [
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

    all_data = {}
    for pair in pairs:
        for tf in ["M15", "M5", "M30", "H1"]:
            file_path = data_dir / f"{pair}_{tf}.parquet"
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
                        print(f"Loaded {pair} {tf}: {len(df)} rows")
                        break

    print(f"\nRunning backtest on {len(all_data)} pairs...")
    results = run_simple_backtest(all_data)

    print("\n=== BACKTEST RESULTS ===")
    print(f"Total trades: {results['total_trades']}")
    print(f"Win rate: {results['win_rate']:.1%}")
    print(f"Total PnL: ${results['total_pnl']:.2f}")
    print(f"Max drawdown: {results['max_drawdown']:.1%}")
    print(f"Final balance: ${results['final_balance']:.2f}")
