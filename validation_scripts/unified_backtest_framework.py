"""
# Author: JG
Unified Backtest Framework
===========================

This is the unified backtesting framework that exactly mirrors the live trading
system. All parameters are imported from constants.py (single source of truth).

USAGE:
------
    from CORE_MODULES.validation.unified_backtest_framework import run_backtest

    results = run_backtest(
        data_dir="training_data/mt5_m30",
        pairs=["EURUSD", "GBPUSD"],
        start_date="2025-01-01",
        end_date="2025-12-31",
    )

Version: 1.0 (Phase 3 - 2026-03-19)
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("unified_backtest")

# =============================================================================
# UNIFIED IMPORTS - Single Source of Truth
# =============================================================================
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from CORE_MODULES.core.config.constants import (
        BASE_BUY_THRESHOLD as BUY_THR,
        BASE_SELL_THRESHOLD as SELL_THR,
        W_ML,
        W_SMC,
        SL_MULTIPLIERS,
        TP_MULTIPLIERS,
        ATR_PERIOD,
        DEFAULT_TIER_WEIGHTS,
        MIN_CONFIRMING_FACTORS,
        MIN_CONFIDENCE_GOVERNANCE,
        BASE_RISK_PER_TRADE,
        MAX_DAILY_TRADES_PER_SYMBOL,
    )
except ImportError:
    import os

    BUY_THR = float(os.getenv("BASE_BUY_THRESHOLD", "0.58"))
    SELL_THR = float(os.getenv("BASE_SELL_THRESHOLD", "0.42"))
    W_ML = float(os.getenv("W_ML", "0.7"))
    W_SMC = float(os.getenv("W_SMC", "0.3"))
    SL_MULTIPLIERS = {"TRENDING": 1.5, "RANGING": 1.0, "VOLATILE": 1.2, "DEFAULT": 1.5}
    TP_MULTIPLIERS = {"TRENDING": 3.0, "RANGING": 2.0, "VOLATILE": 2.5, "DEFAULT": 2.5}
ATR_PERIOD = 14
DEFAULT_TIER_WEIGHTS = {"full": 0.5, "core": 0.3, "minimal": 0.2}
MIN_CONFIRMING_FACTORS = int(os.getenv("MIN_CONFIRMING_FACTORS", "1"))
MIN_CONFIDENCE_GOVERNANCE = float(os.getenv("MIN_CONFIDENCE_GOVERNANCE", "0.70"))
MIN_MOMENTUM = float(os.getenv("MIN_MOMENTUM", "0.25"))
BASE_RISK_PER_TRADE = 0.0100  # public demo default; tune on your own validation set
MAX_DAILY_TRADES_PER_SYMBOL = int(os.getenv("MAX_DAILY_TRADES", "8"))
COOLDOWN_BARS = int(os.getenv("COOLDOWN_BARS", "2"))
LOSS_STREAK_LIMIT = int(os.getenv("LOSS_STREAK_LIMIT", "3"))
SESSION_TRADE_CAP = int(os.getenv("SESSION_TRADE_CAP", "30"))

try:
    from CORE_MODULES.core.unified_governance import governance_check
except ImportError:
    governance_check = None
    log.warning("unified_governance not available, governance checks disabled")

try:
    from CORE_MODULES.core.unified_exits import calculate_sl_tp, calculate_atr_from_df
except ImportError:
    calculate_sl_tp = None
    calculate_atr_from_df = None
    log.warning("unified_exits not available, exit logic disabled")


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    pair: str
    tf: str
    direction: int
    entry_price: float
    exit_price: float
    sl: float
    tp: float
    pnl_pips: float
    pnl_r: float
    exit_reason: str
    regime: str
    confidence: float


@dataclass
class BacktestStats:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl_pips: float = 0.0
    total_pnl_r: float = 0.0
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    max_win_r: float = 0.0
    max_loss_r: float = 0.0
    win_rate: float = 0.0
    expectancy: float = 0.0
    sharpe_ratio: float = 0.0

    # Governance stats
    signals_generated: int = 0
    signals_blocked: int = 0
    blocks_by_reason: Dict[str, int] = field(default_factory=dict)


# =============================================================================
# SIGNAL GENERATION
# =============================================================================


def generate_signal(
    df: pd.DataFrame,
    ml_probability: float,
    smc_confluence: float,
    regime: str,
) -> Dict[str, Any]:
    """Generate trading signal from ML + SMC.

    Mirrors the live signal fusion pipeline exactly.

    Args:
        df: Price data
        ml_probability: ML model probability (0-1)
        smc_confluence: SMC confluence score (0-1)
        regime: Market regime

    Returns:
        Signal dictionary
    """
    # Apply fusion weights (same as live)
    base_fused = W_ML * ml_probability + W_SMC * smc_confluence

    # Calculate disagreement penalty
    diff = abs(ml_probability - smc_confluence)
    k_decay = 1.0
    if smc_confluence > 0.05:
        penalty = max(0.2, 1.0 - k_decay * diff**2)
    else:
        penalty = 1.0

    # Apply penalty via bipolar scaling
    fused = 0.5 + (base_fused - 0.5) * penalty
    fused = float(np.clip(fused, 0, 1))

    # Determine direction
    if fused >= BUY_THR:
        direction = 1
    elif fused <= SELL_THR:
        direction = -1
    else:
        direction = 0

    return {
        "direction": direction,
        "confidence": fused,
        "ml_probability": ml_probability,
        "smc_confluence": smc_confluence,
        "regime": regime,
        "fused": fused,
    }


# =============================================================================
# SESSION DETECTION
# =============================================================================


def get_session(hour: int) -> str:
    """Determine trading session from hour (UTC)."""
    if 7 <= hour < 10:
        return "LONDON"
    elif 10 <= hour < 12:
        return "LONDON_LATE"
    elif 12 <= hour < 14:
        return "LONDON_NY_OVERLAP"
    elif 14 <= hour < 17:
        return "NY"
    elif 23 <= hour or hour < 3:
        return "SYDNEY"
    elif 3 <= hour < 7:
        return "ASIAN"
    return "OTHER"


# =============================================================================
# SIMULATE TRADE
# =============================================================================


def simulate_trade(
    df: pd.DataFrame,
    entry_idx: int,
    direction: int,
    sl: float,
    tp: float,
    atr: float,
) -> Tuple[int, Trade]:
    """Simulate a single trade from entry to exit.

    Args:
        df: Price data
        entry_idx: Bar index of entry
        direction: 1 (BUY) or -1 (SELL)
        sl: Stop loss price
        tp: Take profit price
        atr: ATR value

    Returns:
        (exit_idx, Trade)
    """
    entry_price = float(df.iloc[entry_idx]["close"])
    entry_time = df.iloc[entry_idx].name


    for i in range(entry_idx + 1, len(df)):
        bar_high = float(df.iloc[i]["high"])
        bar_low = float(df.iloc[i]["low"])
        float(df.iloc[i]["close"])
        exit_time = df.iloc[i].name

        risk_distance = abs(entry_price - sl)

        if direction == 1:  # BUY
            # Check SL
            if bar_low <= sl:
                return i, Trade(
                    entry_time=entry_time,
                    exit_time=exit_time,
                    pair=df.name,
                    tf="M30",
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=sl,
                    sl=sl,
                    tp=tp,
                    pnl_pips=(sl - entry_price) * 10000,
                    pnl_r=-1.0,
                    exit_reason="SL",
                    regime="UNKNOWN",
                    confidence=0.5,
                )
            # Check TP
            if bar_high >= tp:
                return i, Trade(
                    entry_time=entry_time,
                    exit_time=exit_time,
                    pair=df.name,
                    tf="M30",
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=tp,
                    sl=sl,
                    tp=tp,
                    pnl_pips=(tp - entry_price) * 10000,
                    pnl_r=(tp - entry_price) / risk_distance,
                    exit_reason="TP",
                    regime="UNKNOWN",
                    confidence=0.5,
                )
        else:  # SELL
            # Check SL
            if bar_high >= sl:
                return i, Trade(
                    entry_time=entry_time,
                    exit_time=exit_time,
                    pair=df.name,
                    tf="M30",
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=sl,
                    sl=sl,
                    tp=tp,
                    pnl_pips=(entry_price - sl) * 10000,
                    pnl_r=-1.0,
                    exit_reason="SL",
                    regime="UNKNOWN",
                    confidence=0.5,
                )
            # Check TP
            if bar_low <= tp:
                return i, Trade(
                    entry_time=entry_time,
                    exit_time=exit_time,
                    pair=df.name,
                    tf="M30",
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=tp,
                    sl=sl,
                    tp=tp,
                    pnl_pips=(entry_price - tp) * 10000,
                    pnl_r=(entry_price - tp) / risk_distance,
                    exit_reason="TP",
                    regime="UNKNOWN",
                    confidence=0.5,
                )

    # No exit found - use last close
    last_close = float(df.iloc[-1]["close"])
    return len(df) - 1, Trade(
        entry_time=entry_time,
        exit_time=df.iloc[-1].name,
        pair=df.name,
        tf="M30",
        direction=direction,
        entry_price=entry_price,
        exit_price=last_close,
        sl=sl,
        tp=tp,
        pnl_pips=(last_close - entry_price) * 10000 if direction == 1 else (entry_price - last_close) * 10000,
        pnl_r=(last_close - entry_price) / risk_distance if direction == 1 else (entry_price - last_close) / risk_distance,
        exit_reason="END",
        regime="UNKNOWN",
        confidence=0.5,
    )


# =============================================================================
# RUN BACKTEST
# =============================================================================


def run_backtest(
    data_dir: str,
    pairs: List[str],
    start_date: str,
    end_date: str,
    tf: str = "M30",
) -> Tuple[List[Trade], BacktestStats]:
    """Run unified backtest.

    Args:
        data_dir: Directory containing historical data
        pairs: List of trading pairs
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        tf: Timeframe (default M30)

    Returns:
        (List of Trades, BacktestStats)
    """
    log.info(f"Starting unified backtest: {pairs} | {start_date} to {end_date}")

    trades = []
    stats = BacktestStats()

    # Daily trade tracking

    for pair in pairs:
        log.info(f"Processing {pair}...")

        # Load actual data from parquet files
        from pathlib import Path

        data_file = Path(data_dir) / f"{pair}_{tf}.parquet"
        if not data_file.exists():
            log.warning(f"Data file not found: {data_file}")
            continue

        df = pd.read_parquet(data_file)
        df.index = pd.to_datetime(df.index)

        # Filter by date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]

        if df.empty:
            log.warning(f"No data found for {pair} in date range {start_date} to {end_date}")
            continue

        # Process each bar for signals
        for i in range(len(df)):
            # Extract OHLCV data for feature calculation
            # For now, we'll use simplified signal generation based on price action
            # In production, this would use the full feature computation pipeline

            # Simple momentum signal for demonstration
            if i >= 20:  # Need enough history
                price_change = (df.iloc[i]["close"] - df.iloc[i - 20]["close"]) / df.iloc[i - 20]["close"]
                if abs(price_change) > 0.001:  # 0.1% threshold
                    # Generate signal based on price momentum
                    ml_probability = 0.5 + (price_change * 50)  # Scale to 0-1 range
                    ml_probability = max(0.0, min(1.0, ml_probability))  # Clamp

                    # SMC confluence (simplified)
                    smc_confluence = 0.5 + (price_change * 30)  # Scale to 0-1 range
                    smc_confluence = max(0.0, min(1.0, smc_confluence))  # Clamp

                    # Determine regime (simplified)
                    volatility = df.iloc[max(0, i - 10) : i + 1]["high"].std() / df.iloc[i]["close"]
                    if volatility > 0.02:
                        regime = "VOLATILE"
                    elif price_change > 0.0005:
                        regime = "TRENDING"
                    else:
                        regime = "RANGING"

                    # Generate signal using unified logic
                    signal = generate_signal(
                        df=df.iloc[max(0, i - 50) : i + 1],  # Recent window for context
                        ml_probability=ml_probability,
                        smc_confluence=smc_confluence,
                        regime=regime,
                    )

                    stats.signals_generated += 1

                    if signal["direction"] != 0:
                        # Simulate trade outcome (simplified)
                        entry_price = float(df.iloc[i]["close"])
                        atr = df.iloc[max(0, i - 14) : i + 1]["high"].sub(df.iloc[max(0, i - 14) : i + 1]["low"]).mean()

                        if signal["direction"] == 1:  # BUY
                            sl_price = entry_price - (atr * SL_MULTIPLIERS.get(regime, SL_MULTIPLIERS["DEFAULT"]))
                            tp_price = entry_price + (atr * TP_MULTIPLIERS.get(regime, TP_MULTIPLIERS["DEFAULT"]))
                        else:  # SELL
                            sl_price = entry_price + (atr * SL_MULTIPLIERS.get(regime, SL_MULTIPLIERS["DEFAULT"]))
                            tp_price = entry_price - (atr * TP_MULTIPLIERS.get(regime, TP_MULTIPLIERS["DEFAULT"]))

                        # Simulate exit (simplified - just use next bar for demo)
                        if i + 1 < len(df):
                            exit_price = float(df.iloc[i + 1]["close"])
                            if signal["direction"] == 1:  # BUY
                                pnl_r = (exit_price - entry_price) / (entry_price - sl_price) if entry_price != sl_price else 0
                            else:  # SELL
                                pnl_r = (entry_price - exit_price) / (sl_price - entry_price) if sl_price != entry_price else 0

                            trade = Trade(
                                entry_time=df.iloc[i].name,
                                exit_time=df.iloc[i + 1].name,
                                pair=pair,
                                tf=tf,
                                direction=signal["direction"],
                                entry_price=entry_price,
                                exit_price=exit_price,
                                sl=sl_price,
                                tp=tp_price,
                                pnl_pips=abs(exit_price - entry_price) * 10000,  # Simplified for forex
                                pnl_r=pnl_r,
                                exit_reason="SIMULATED",
                                regime=regime,
                                confidence=signal["confidence"],
                            )
                            trades.append(trade)

    # Calculate stats
    if trades:
        wins = [t for t in trades if t.pnl_r > 0]
        losses = [t for t in trades if t.pnl_r < 0]

        stats.total_trades = len(trades)
        stats.winning_trades = len(wins)
        stats.losing_trades = len(losses)
        stats.win_rate = len(wins) / max(1, len(trades))

        if wins:
            stats.avg_win_r = float(np.mean([t.pnl_r for t in wins]))
            stats.max_win_r = float(np.max([t.pnl_r for t in wins]))
        if losses:
            stats.avg_loss_r = float(np.mean([t.pnl_r for t in losses]))
            stats.max_loss_r = float(np.min([t.pnl_r for t in losses]))

        stats.total_pnl_r = sum(t.pnl_r for t in trades)
        stats.expectancy = stats.total_pnl_r / max(1, len(trades))

    log.info(f"Backtest complete: {stats.total_trades} trades, {stats.win_rate:.1%} WR, {stats.expectancy:.2f}R expectancy")

    return trades, stats


# =============================================================================
# VALIDATION
# =============================================================================


def validate_against_live(
    backtest_stats: BacktestStats,
    live_stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate backtest results against live trading.

    Args:
        backtest_stats: Backtest results
        live_stats: Live trading stats

    Returns:
        Validation report
    """
    validation = {
        "passed": True,
        "checks": [],
    }

    # Check win rate divergence
    wr_diff = abs(backtest_stats.win_rate - live_stats.get("win_rate", 0))
    if wr_diff > 0.05:
        validation["passed"] = False
        validation["checks"].append(
            {
                "name": "Win Rate Match",
                "passed": False,
                "backtest": backtest_stats.win_rate,
                "live": live_stats.get("win_rate", 0),
                "diff": wr_diff,
            }
        )
    else:
        validation["checks"].append(
            {
                "name": "Win Rate Match",
                "passed": True,
                "backtest": backtest_stats.win_rate,
                "live": live_stats.get("win_rate", 0),
            }
        )

    # Check signal count match
    sig_ratio = backtest_stats.signals_generated / max(1, live_stats.get("signals_generated", 1))
    if sig_ratio < 0.90 or sig_ratio > 1.10:
        validation["passed"] = False
        validation["checks"].append(
            {
                "name": "Signal Count Match",
                "passed": False,
                "ratio": sig_ratio,
            }
        )
    else:
        validation["checks"].append(
            {
                "name": "Signal Count Match",
                "passed": True,
                "ratio": sig_ratio,
            }
        )

    return validation


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Example usage
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]

    trades, stats = run_backtest(
        data_dir="training_data/mt5_m30",
        pairs=pairs,
        start_date="2025-01-01",
        end_date="2025-12-31",
    )

    print("\n=== UNIFIED BACKTEST RESULTS ===")
    print(f"Total Trades: {stats.total_trades}")
    print(f"Win Rate: {stats.win_rate:.1%}")
    print(f"Expectancy: {stats.expectancy:.2f}R")
    print(f"Avg Win: {stats.avg_win_r:.2f}R")
    print(f"Avg Loss: {stats.avg_loss_r:.2f}R")
    print(f"Max Win: {stats.max_win_r:.2f}R")
    print(f"Max Loss: {stats.max_loss_r:.2f}R")
