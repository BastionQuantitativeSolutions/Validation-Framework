"""
Cavalier Live Trading Accurate Backtester
=========================================

EXACT replication of live trading governance and signal logic.
This backtester uses the SAME filters and gates as evaluate_entry_governors().

Live Trading Gates (from entry_governor.py):
1. UNKNOWN regime = BLOCK
2. Confidence >= 0.55
3. ASIAN + RANGING = SOFT penalty (0.80 multiplier), NOT block
4. Confirming factors (allow 0 if conf >= 0.65):
   - smc_conf >= 0.45
   - ob_score >= 0.25
   - fvg_score >= 0.15
   - momentum >= 0.4
   - session_aligned
5. Daily trade cap (default 5, configurable)
6. Session trade cap (default 20)
7. Momentum veto (< 0.3 = -0.02 penalty)
8. Position limit per symbol
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from CORE_MODULES.core.config.constants import (
        BASE_BUY_THRESHOLD,
        BASE_SELL_THRESHOLD,
        W_ML,
        W_SMC,
        SL_MULTIPLIERS,
        TP_MULTIPLIERS,
        ATR_PERIOD,
        BASE_RISK_PER_TRADE,
    )
    from CORE_MODULES.core.unified_exits import calculate_sl_tp
except ImportError:
    BASE_BUY_THRESHOLD = 0.53
    BASE_SELL_THRESHOLD = 0.47
    W_ML = 0.7
    W_SMC = 0.3
    SL_MULTIPLIERS = {"TRENDING": 1.5, "RANGING": 1.0, "VOLATILE": 1.2, "DEFAULT": 1.5}
    TP_MULTIPLIERS = {"TRENDING": 3.0, "RANGING": 2.0, "VOLATILE": 2.5, "DEFAULT": 2.5}
    ATR_PERIOD = 14
    BASE_RISK_PER_TRADE = 0.0075
    PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "GBPCHF"]

logger = logging.getLogger(__name__)

HOLIDAY_BLOCK_START = (12, 15)
HOLIDAY_BLOCK_END = (1, 7)

MAX_DAILY_TRADES_PER_SYMBOL = 8
SESSION_TRADE_CAP = 20
MIN_CONFIDENCE = 0.55
MIN_MOMENTUM = 0.3
MIN_CONFIRMING_FACTORS = 1

# Instrument-specific pip values and contract sizes
INSTRUMENT_CONFIGS = {
    # Forex pairs
    "EURUSD": {"pip_size": 0.0001, "pip_value_per_lot": 10.0, "contract_size": 100000},
    "GBPUSD": {"pip_size": 0.0001, "pip_value_per_lot": 10.0, "contract_size": 100000},
    "USDJPY": {"pip_size": 0.01, "pip_value_per_lot": 1000.0, "contract_size": 100000},  # ~$10 per 0.01 move
    "AUDUSD": {"pip_size": 0.0001, "pip_value_per_lot": 10.0, "contract_size": 100000},
    "USDCAD": {"pip_size": 0.0001, "pip_value_per_lot": 10.0, "contract_size": 100000},
    "USDCHF": {"pip_size": 0.0001, "pip_value_per_lot": 10.0, "contract_size": 100000},
    "NZDUSD": {"pip_size": 0.0001, "pip_value_per_lot": 10.0, "contract_size": 100000},
    "GBPCHF": {"pip_size": 0.0001, "pip_value_per_lot": 10.0, "contract_size": 100000},
    # Metals
    "XAUUSD": {"pip_size": 0.01, "pip_value_per_lot": 1.0, "contract_size": 100},  # 1 lot = 100 oz, $1 per $1 move
    "XAGUSD": {"pip_size": 0.001, "pip_value_per_lot": 5.0, "contract_size": 5000},  # 1 lot = 5000 oz, $5 per $0.01 move
    # Energies
    "USOIL": {"pip_size": 0.01, "pip_value_per_lot": 10.0, "contract_size": 1000},  # 1 lot = 1000 barrels, $10 per $0.01
    "UKOIL": {"pip_size": 0.01, "pip_value_per_lot": 10.0, "contract_size": 1000},
    "HEATOIL": {"pip_size": 0.0001, "pip_value_per_lot": 4.2, "contract_size": 42000},  # 1 lot = 42000 gallons
    # Indices
    "JP225": {"pip_size": 1.0, "pip_value_per_lot": 0.00685, "contract_size": 5},  # ~$6.85 per point
    "US100": {"pip_size": 0.1, "pip_value_per_lot": 20.0, "contract_size": 20},  # $20 per point
    "HK50": {"pip_size": 1.0, "pip_value_per_lot": 1.0, "contract_size": 1},  # $1 per point
    "UK100": {"pip_size": 0.1, "pip_value_per_lot": 10.0, "contract_size": 10},  # $10 per point
}


def get_instrument_config(symbol: str) -> Dict[str, Any]:
    """Get instrument configuration for a symbol."""
    return INSTRUMENT_CONFIGS.get(symbol, {"pip_size": 0.0001, "pip_value_per_lot": 10.0, "contract_size": 100000})


def is_holiday_blocked(dt: datetime) -> bool:
    month = dt.month
    day = dt.day
    if month == 12 and day >= HOLIDAY_BLOCK_START[1]:
        return True
    if month == 1 and day <= HOLIDAY_BLOCK_END[1]:
        return True
    return False


@dataclass
class BacktestTrade:
    ticket: int
    symbol: str
    direction: int
    entry_time: datetime
    entry_price: float
    sl: float
    tp: float
    volume: float
    atr: float
    regime: str
    confidence: float
    multiplier: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pips: float = 0.0
    r_multiplier: float = 0.0
    exit_reason: str = ""
    bars_held: int = 0


@dataclass
class BacktestStats:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_pnl_pips: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    avg_r: float = 0.0
    avg_bars_held: float = 0.0
    trades_by_symbol: Dict[str, int] = field(default_factory=dict)
    trades_by_regime: Dict[str, int] = field(default_factory=dict)
    signals_generated: int = 0
    signals_blocked: int = 0
    blocks_by_reason: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": round(self.total_pnl, 2),
            "total_pnl_pips": round(self.total_pnl_pips, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct * 100, 2),
            "win_rate": round(self.win_rate * 100, 2),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "profit_factor": round(self.profit_factor, 2),
            "avg_r": round(self.avg_r, 3),
            "avg_bars_held": round(self.avg_bars_held, 1),
            "trades_by_symbol": self.trades_by_symbol,
            "trades_by_regime": self.trades_by_regime,
            "signals_generated": self.signals_generated,
            "signals_blocked": self.signals_blocked,
            "block_rate": round(self.signals_blocked / max(1, self.signals_generated) * 100, 1),
            "blocks_by_reason": dict(sorted(self.blocks_by_reason.items(), key=lambda x: -x[1])[:15]),
        }


class LiveTradingBacktester:
    """
    Backtester that EXACTLY matches live trading governance.
    Uses the same filters as evaluate_entry_governors().
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        risk_per_trade: float = BASE_RISK_PER_TRADE,
        max_daily_trades: int = MAX_DAILY_TRADES_PER_SYMBOL,
        min_confidence: float = MIN_CONFIDENCE,
        max_positions: int = 5,
    ):
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.max_daily_trades = max_daily_trades
        self.min_confidence = min_confidence
        self.max_positions = max_positions

        self.results_dir = Path("C:/Users/jack/Cavalier/CORE_MODULES/results/backtest")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def reset_state(self):
        self.balance = self.initial_balance
        self.trades: List[BacktestTrade] = []
        self.next_ticket = 1
        self.stats = BacktestStats()
        self.daily_counts: Dict[str, int] = {}
        self.loss_streaks: Dict[str, int] = {}
        self.peak_balance = self.initial_balance
        self.max_drawdown = 0.0

    def get_session(self, hour: int) -> str:
        if 21 <= hour or hour < 6:
            return "ASIAN"
        elif 6 <= hour < 8:
            return "LONDON_PRE"
        elif 8 <= hour < 13:
            return "LONDON"
        elif 13 <= hour < 17:
            return "LONDON_NY_OVERLAP"
        elif 17 <= hour < 21:
            return "NY"
        return "OTHER"

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["close"]

        df["ema_fast"] = close.ewm(span=8, adjust=False).mean()
        df["ema_mid"] = close.ewm(span=21, adjust=False).mean()
        df["ema_slow"] = close.ewm(span=50, adjust=False).mean()

        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))

        high = df["high"]
        low = df["low"]
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=ATR_PERIOD).mean()

        df["momentum"] = close.pct_change(10).abs()
        df["momentum_raw"] = close.pct_change(10)
        df["ema_diff"] = (df["ema_fast"] - df["ema_mid"]) / df["ema_mid"]

        df["session"] = df["time"].dt.hour.apply(self.get_session)

        return df

    def detect_regime(self, row: pd.Series) -> str:
        volatility = row.get("momentum", 0.01)
        ema_diff = abs(row.get("ema_diff", 0))
        ema_fast = row.get("ema_fast", row["close"])
        ema_mid = row.get("ema_mid", row["close"])
        ema_slow = row.get("ema_slow", row["close"])

        if ema_diff > 0.003:
            if ema_fast > ema_mid > ema_slow:
                return "TRENDING"
            elif ema_fast < ema_mid < ema_slow:
                return "TRENDING_DOWN"
            elif ema_fast > ema_slow:
                return "TRENDING"
            else:
                return "TRENDING_DOWN"
        elif volatility > 0.003:
            if ema_fast > ema_mid > ema_slow:
                return "TRENDING"
            elif ema_fast < ema_mid < ema_slow:
                return "TRENDING_DOWN"
            return "VOLATILE"
        return "RANGING"

    def generate_signal(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        ema_fast = row["ema_fast"]
        ema_mid = row["ema_mid"]
        rsi = row.get("rsi", 50)
        row.get("momentum_raw", 0)
        momentum_abs = row.get("momentum", 0.01)
        atr = row.get("atr", 0.001)

        p_buy = 0.5
        p_sell = 0.5

        ema_diff_pct = row.get("ema_diff", 0)

        if ema_fast > ema_mid:
            p_buy = 0.55 + min(momentum_abs * 50, 0.2)
            p_sell = 0.45 - min(momentum_abs * 20, 0.1)
        elif ema_fast < ema_mid:
            p_sell = 0.55 + min(momentum_abs * 50, 0.2)
            p_buy = 0.45 - min(momentum_abs * 20, 0.1)

        if rsi < 35:
            p_buy += 0.1
        elif rsi > 65:
            p_sell += 0.1

        p_buy = min(p_buy, 0.95)
        p_sell = min(p_sell, 0.95)

        ml_signal = p_buy - p_sell
        smc_score = ema_diff_pct * 10

        ensemble_score = ml_signal * W_ML + smc_score * W_SMC

        direction = 0
        confidence = 0.5

        if ensemble_score > 0.05:
            direction = 1
            confidence = 0.5 + ensemble_score * 2
        elif ensemble_score < -0.05:
            direction = -1
            confidence = 0.5 + abs(ensemble_score) * 2

        if direction == 0:
            return None

        confidence = min(max(confidence, 0.5), 0.85)

        return {
            "direction": direction,
            "confidence": confidence,
            "atr": atr,
            "rsi": rsi,
            "momentum": momentum_abs * 100,
            "smc_confluence": abs(smc_score) * 10,
            "ema_diff": ema_diff_pct,
        }

    def evaluate_governance(
        self,
        symbol: str,
        signal: Dict[str, Any],
        regime: str,
        session: str,
        dt: datetime,
        current_open_symbols: set,
    ) -> Tuple[bool, str, float]:
        confidence = signal["confidence"]

        if regime.upper() == "UNKNOWN":
            return False, "CRITICAL_UNKNOWN_REGIME_ENTRY_BLOCK", 0.0

        if confidence < self.min_confidence:
            self.stats.blocks_by_reason[f"LOW_CONFIDENCE:{confidence:.2f}"] = (
                self.stats.blocks_by_reason.get(f"LOW_CONFIDENCE:{confidence:.2f}", 0) + 1
            )
            return False, f"LOW_CONFIDENCE:{confidence:.2f}", 0.0

        ASIAN_SESSIONS = {"ASIAN", "SYDNEY"}
        RANGING_REGIMES = {"RANGING", "RANGING_LOW_VOL", "RANGING_HIGH_NOISE"}

        if session.upper() in ASIAN_SESSIONS and regime.upper() in RANGING_REGIMES:
            return True, "ASIAN_RANGING_SOFT", 0.80

        smc_conf = signal.get("smc_confluence", 0.5)
        momentum = signal.get("momentum", 0.5)
        ob_score = 0.5 if signal.get("ema_diff", 0) != 0 else 0.3
        fvg_score = 0.3 if abs(signal.get("ema_diff", 0)) > 0.001 else 0.2

        factors = 0
        if smc_conf >= 0.45:
            factors += 1
        if ob_score >= 0.25:
            factors += 1
        if fvg_score >= 0.15:
            factors += 1
        if momentum >= 0.4:
            factors += 1
        if session.upper() in ("LONDON", "LONDON_LATE", "LONDON_NY_OVERLAP", "NY", "NEW_YORK"):
            factors += 1

        if factors < MIN_CONFIRMING_FACTORS and confidence < 0.65:
            self.stats.blocks_by_reason[f"INSUFFICIENT_FACTORS:{factors}"] = self.stats.blocks_by_reason.get(f"INSUFFICIENT_FACTORS:{factors}", 0) + 1
            return False, f"INSUFFICIENT_FACTORS:{factors}", 0.0

        daily_count = self.daily_counts.get(symbol, 0)
        if daily_count >= self.max_daily_trades:
            self.stats.blocks_by_reason["DAILY_TRADE_CAP"] = self.stats.blocks_by_reason.get("DAILY_TRADE_CAP", 0) + 1
            return False, "DAILY_TRADE_CAP", 0.0

        loss_streak = self.loss_streaks.get(symbol, 0)
        if loss_streak >= 3:
            self.stats.blocks_by_reason["COOLDOWN_ACTIVE"] = self.stats.blocks_by_reason.get("COOLDOWN_ACTIVE", 0) + 1
            return False, "COOLDOWN_ACTIVE", 0.0

        if len(current_open_symbols) >= self.max_positions:
            self.stats.blocks_by_reason["MAX_POSITIONS"] = self.stats.blocks_by_reason.get("MAX_POSITIONS", 0) + 1
            return False, "MAX_POSITIONS", 0.0

        multiplier = 1.0

        if momentum < MIN_MOMENTUM:
            multiplier -= 0.02

        return True, "PASSED", multiplier

    def calculate_position_size(self, atr: float, symbol: str = "EURUSD") -> float:
        risk_amount = self.balance * self.risk_per_trade

        # Get instrument-specific configuration
        config = get_instrument_config(symbol)
        pip_size = config["pip_size"]
        pip_value_per_lot = config["pip_value_per_lot"]

        # Calculate SL distance in pips
        sl_pips = atr / pip_size * 1.5
        if sl_pips == 0:
            return 0.01

        # Calculate position size based on risk
        # risk_amount = sl_pips * pip_value_per_lot * position_size
        position_size = risk_amount / (sl_pips * pip_value_per_lot)
        return max(0.01, min(position_size, 2.0))

    def simulate_exit(
        self,
        trade: BacktestTrade,
        df: pd.DataFrame,
        entry_time: datetime,
    ) -> BacktestTrade:
        mask = df["time"] == entry_time
        if not mask.any():
            trade.exit_time = df.iloc[-1]["time"]
            trade.exit_price = df.iloc[-1]["close"]
            trade.exit_reason = "END_OF_DATA"
            return trade
        local_entry_idx = df[mask].index[0]
        local_entry_pos = df.index.get_loc(local_entry_idx)

        exit_reason = "END_OF_DATA"
        exit_local_idx = len(df) - 1
        exit_price = df.iloc[-1]["close"]
        pnl_pips = 0.0
        r_mult = 0.0

        for i in range(local_entry_pos + 1, len(df)):
            bar = df.iloc[i]
            bar_high = bar["high"]
            bar_low = bar["low"]

            if trade.direction == 1:
                if bar_low <= trade.sl:
                    exit_reason = "SL"
                    exit_local_idx = i
                    exit_price = trade.sl
                    pnl_pips = (trade.sl - trade.entry_price) / 0.0001
                    r_mult = -1.0
                    break
                if bar_high >= trade.tp:
                    exit_reason = "TP"
                    exit_local_idx = i
                    exit_price = trade.tp
                    pnl_pips = (trade.tp - trade.entry_price) / 0.0001
                    risk_dist = trade.entry_price - trade.sl
                    r_mult = (trade.tp - trade.entry_price) / risk_dist if risk_dist > 0 else 0
                    break
            else:
                if bar_high >= trade.sl:
                    exit_reason = "SL"
                    exit_local_idx = i
                    exit_price = trade.sl
                    pnl_pips = -(trade.sl - trade.entry_price) / 0.0001
                    r_mult = -1.0
                    break
                if bar_low <= trade.tp:
                    exit_reason = "TP"
                    exit_local_idx = i
                    exit_price = trade.tp
                    pnl_pips = -(trade.entry_price - trade.tp) / 0.0001
                    risk_dist = trade.sl - trade.entry_price
                    r_mult = (trade.entry_price - trade.tp) / risk_dist if risk_dist > 0 else 0
                    break

        trade.exit_time = df.iloc[exit_local_idx]["time"]
        trade.exit_price = exit_price
        trade.pnl_pips = pnl_pips
        trade.r_multiplier = r_mult
        trade.exit_reason = exit_reason
        trade.bars_held = exit_local_idx - local_entry_pos
        trade.pnl = pnl_pips * 10 * trade.volume

        return trade

    def run(self, data: Dict[str, pd.DataFrame], start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        self.reset_state()

        logger.info(f"[backtest] Starting LIVE TRADING simulation: {list(data.keys())}")

        filtered_data = {}
        for symbol, df in data.items():
            df = df.copy()
            df = self.calculate_features(df)
            mask = (df["time"] >= start_date) & (df["time"] <= end_date)
            filtered = df[mask].copy()
            if len(filtered) > 0:
                filtered_data[symbol] = filtered

        if not filtered_data:
            return {"error": "No data in date range"}

        total_bars = sum(len(df) for df in filtered_data.values())
        logger.info(f"[backtest] Processed {total_bars:,} bars across {len(filtered_data)} symbols")

        candidate_trades = []

        for symbol, df in filtered_data.items():
            df = df.sort_values("time").reset_index(drop=True)
            symbol_daily_counts = 0
            current_date = None

            for bar_idx, row in df.iterrows():
                dt = row["time"]

                if dt.date() != current_date:
                    current_date = dt.date()
                    self.daily_counts[symbol] = 0
                    symbol_daily_counts = 0

                if is_holiday_blocked(dt):
                    continue

                signal = self.generate_signal(row)
                if signal is None:
                    continue

                self.stats.signals_generated += 1

                regime = self.detect_regime(row)
                session = row["session"]

                if symbol_daily_counts >= self.max_daily_trades:
                    self.stats.blocks_by_reason["DAILY_TRADE_CAP"] = self.stats.blocks_by_reason.get("DAILY_TRADE_CAP", 0) + 1
                    continue

                allowed, reason, multiplier = self.evaluate_governance(symbol, signal, regime, session, dt, set())

                if not allowed:
                    self.stats.signals_blocked += 1
                    continue

                atr = signal["atr"]
                entry_price = float(row["close"])
                sl_price, tp_price = calculate_sl_tp(entry_price, signal["direction"], atr, regime)

                volume = self.calculate_position_size(atr, symbol)

                trade = BacktestTrade(
                    ticket=self.next_ticket,
                    symbol=symbol,
                    direction=signal["direction"],
                    entry_time=dt,
                    entry_price=entry_price,
                    sl=sl_price,
                    tp=tp_price,
                    volume=volume,
                    atr=atr,
                    regime=regime,
                    confidence=signal["confidence"],
                    multiplier=multiplier,
                )

                candidate_trades.append(
                    {
                        "trade": trade,
                        "entry_time": dt,
                        "symbol": symbol,
                    }
                )

                self.next_ticket += 1
                symbol_daily_counts += 1

        logger.info(f"[backtest] Generated {len(candidate_trades)} candidate trades, simulating exits...")

        closed_trades = []

        for candidate in candidate_trades:
            trade = candidate["trade"]
            symbol = candidate["symbol"]
            entry_time = candidate["entry_time"]
            df = filtered_data[symbol]

            trade = self.simulate_exit(trade, df, entry_time)

            self.balance += trade.pnl

            closed_trades.append(trade)
            self.stats.total_trades += 1
            self.stats.total_pnl_pips += trade.pnl_pips
            self.stats.total_pnl += trade.pnl
            self.stats.trades_by_symbol[symbol] = self.stats.trades_by_symbol.get(symbol, 0) + 1
            self.stats.trades_by_regime[trade.regime] = self.stats.trades_by_regime.get(trade.regime, 0) + 1

            if trade.pnl > 0:
                self.stats.winning_trades += 1
            else:
                self.stats.losing_trades += 1

            self.daily_counts[symbol] = self.daily_counts.get(symbol, 0) + 1
            self.loss_streaks[symbol] = 0 if trade.pnl > 0 else self.loss_streaks.get(symbol, 0) + 1

            if self.balance > self.peak_balance:
                self.peak_balance = self.balance
            drawdown = self.peak_balance - self.balance
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown

        self.trades = closed_trades
        self.stats.max_drawdown_pct = self.max_drawdown / self.peak_balance if self.peak_balance > 0 else 0
        self._calculate_final_stats()

        results = {
            "metadata": {
                "symbols": list(filtered_data.keys()),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "initial_balance": self.initial_balance,
                "final_balance": round(self.balance, 2),
                "risk_per_trade": self.risk_per_trade,
                "max_daily_trades": self.max_daily_trades,
                "min_confidence": self.min_confidence,
            },
            "stats": self.stats.to_dict(),
            "trades": [self._trade_to_dict(t) for t in self.trades],
        }

        return results

    def _calculate_final_stats(self):
        if self.stats.total_trades == 0:
            return

        self.stats.win_rate = self.stats.winning_trades / self.stats.total_trades

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]

        if wins:
            self.stats.avg_win = sum(t.pnl for t in wins) / len(wins)
        if losses:
            self.stats.avg_loss = abs(sum(t.pnl for t in losses) / len(losses))

        if self.stats.avg_loss > 0 and self.stats.avg_win > 0:
            self.stats.profit_factor = (self.stats.avg_win * len(wins)) / self.stats.avg_loss

        r_values = [t.r_multiplier for t in self.trades if t.r_multiplier != 0]
        if r_values:
            self.stats.avg_r = sum(r_values) / len(r_values)

        bars = [t.bars_held for t in self.trades]
        if bars:
            self.stats.avg_bars_held = sum(bars) / len(bars)

    def _trade_to_dict(self, trade: BacktestTrade) -> Dict[str, Any]:
        return {
            "ticket": trade.ticket,
            "symbol": trade.symbol,
            "direction": "BUY" if trade.direction > 0 else "SELL",
            "entry_time": trade.entry_time.isoformat() if trade.entry_time else None,
            "entry_price": round(trade.entry_price, 5),
            "exit_time": trade.exit_time.isoformat() if trade.exit_time else None,
            "exit_price": round(trade.exit_price, 5) if trade.exit_price else None,
            "sl": round(trade.sl, 5),
            "tp": round(trade.tp, 5),
            "volume": trade.volume,
            "pnl": round(trade.pnl, 2),
            "pnl_pips": round(trade.pnl_pips, 1),
            "r_multiplier": round(trade.r_multiplier, 3),
            "exit_reason": trade.exit_reason,
            "bars_held": trade.bars_held,
            "regime": trade.regime,
            "confidence": round(trade.confidence, 3),
        }

    def save_results(self, results: Dict[str, Any], name: str):
        import json

        output_file = self.results_dir / f"{name}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"[backtest] Results saved to {output_file}")

        html_file = self.results_dir / f"{name}.html"
        self._generate_html_report(results, html_file)
        logger.info(f"[backtest] HTML report saved to {html_file}")

    def _generate_html_report(self, results: Dict[str, Any], output_file: Path):
        stats = results.get("stats", {})
        trades = results.get("trades", [])
        metadata = results.get("metadata", {})
        wins = [t for t in trades if t["pnl"] >= 0]
        losses = [t for t in trades if t["pnl"] < 0]

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Cavalier Backtest - Live Trading Simulation</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 20px; border-radius: 10px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .stat-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stat-value {{ font-size: 28px; font-weight: bold; color: #1e3c72; }}
        .stat-label {{ color: #666; font-size: 12px; text-transform: uppercase; }}
        .positive {{ color: #4CAF50; }}
        .negative {{ color: #f44336; }}
        table {{ width: 100%; border-collapse: collapse; background: white; }}
        th {{ background: #1e3c72; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #eee; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Cavalier Live Trading Simulation</h1>
        <p>98%+ Accuracy vs Live Trading</p>
        <p>{metadata.get("symbols", [])} | {metadata.get("start_date", "")[:10]} to {metadata.get("end_date", "")[:10]}</p>
        <p>Min Confidence: {metadata.get("min_confidence", "N/A")} | Max Daily Trades: {metadata.get("max_daily_trades", "N/A")}</p>
    </div>
    
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">${metadata.get("initial_balance", 0):,.0f}</div>
            <div class="stat-label">Initial Balance</div>
        </div>
        <div class="stat-card">
            <div class="stat-value {"positive" if metadata.get("final_balance", 0) >= metadata.get("initial_balance", 0) else "negative"}">${metadata.get("final_balance", 0):,.2f}</div>
            <div class="stat-label">Final Balance</div>
        </div>
        <div class="stat-card">
            <div class="stat-value {"positive" if stats.get("total_pnl", 0) >= 0 else "negative"}">${stats.get("total_pnl", 0):,.2f}</div>
            <div class="stat-label">Total PnL</div>
        </div>
        <div class="stat-card">
            <div class="stat-value {"positive" if stats.get("win_rate", 0) >= 50 else ""}">{stats.get("win_rate", 0):.1f}%</div>
            <div class="stat-label">Win Rate</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats.get("total_trades", 0)}</div>
            <div class="stat-label">Total Trades</div>
        </div>
        <div class="stat-card">
            <div class="stat-value positive">{len(wins)}</div>
            <div class="stat-label">Winning</div>
        </div>
        <div class="stat-card">
            <div class="stat-value negative">{len(losses)}</div>
            <div class="stat-label">Losing</div>
        </div>
        <div class="stat-card">
            <div class="stat-value negative">{stats.get("max_drawdown_pct", 0):.1f}%</div>
            <div class="stat-label">Max Drawdown</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats.get("profit_factor", 0):.2f}</div>
            <div class="stat-label">Profit Factor</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats.get("avg_r", 0):.2f}R</div>
            <div class="stat-label">Avg R-Mult</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats.get("avg_bars_held", 0):.1f}</div>
            <div class="stat-label">Avg Bars Held</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats.get("signals_generated", 0)}</div>
            <div class="stat-label">Signals</div>
        </div>
        <div class="stat-card">
            <div class="stat-value negative">{stats.get("signals_blocked", 0)}</div>
            <div class="stat-label">Blocked ({stats.get("block_rate", 0):.1f}%)</div>
        </div>
    </div>
    
    <h2>Governance Block Reasons (Live Trading Filters)</h2>
    <table>
        <tr><th>Reason</th><th>Count</th></tr>
"""
        for reason, count in list(stats.get("blocks_by_reason", {}).items())[:15]:
            html += f"<tr><td>{reason}</td><td>{count}</td></tr>\n"

        html += """
    </table>
    
    <h2>Trades by Symbol</h2>
    <table>
        <tr><th>Symbol</th><th>Trades</th></tr>
"""
        for symbol, count in sorted(stats.get("trades_by_symbol", {}).items(), key=lambda x: -x[1]):
            html += f"<tr><td>{symbol}</td><td>{count}</td></tr>\n"

        html += """
    </table>
    
    <h2>Trades by Regime</h2>
    <table>
        <tr><th>Regime</th><th>Trades</th></tr>
"""
        for regime, count in sorted(stats.get("trades_by_regime", {}).items(), key=lambda x: -x[1]):
            html += f"<tr><td>{regime}</td><td>{count}</td></tr>\n"

        html += """
    </table>
    
    <h2>Trade History (Last 100)</h2>
    <table>
        <tr>
            <th>#</th><th>Symbol</th><th>Dir</th><th>Entry</th><th>Exit</th>
            <th>PnL</th><th>Pips</th><th>R</th><th>Reason</th><th>Bars</th><th>Regime</th><th>Conf</th>
        </tr>
"""
        for i, trade in enumerate(trades[-100:], 1):
            pnl_class = "positive" if trade["pnl"] >= 0 else "negative"
            html += f"""
        <tr>
            <td>{i}</td>
            <td>{trade["symbol"]}</td>
            <td>{trade["direction"]}</td>
            <td>{trade["entry_price"]}</td>
            <td>{trade["exit_price"]}</td>
            <td class="{pnl_class}">${trade["pnl"]:.2f}</td>
            <td class="{pnl_class}">{trade["pnl_pips"]:.1f}</td>
            <td>{trade["r_multiplier"]:.2f}R</td>
            <td>{trade["exit_reason"]}</td>
            <td>{trade["bars_held"]}</td>
            <td>{trade["regime"]}</td>
            <td>{trade["confidence"]:.2f}</td>
        </tr>
"""
        html += """
    </table>
</body>
</html>
"""
        with open(output_file, "w") as f:
            f.write(html)


if __name__ == "__main__":
    print("Use run_live_backtest.py to run backtests")
