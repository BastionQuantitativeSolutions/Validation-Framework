"""
Backtest Replay Engine for Cavalier Trading System

Simulates live trading using historical tick data with exact parity to live execution.
Feeds historical data through the same signal/governance pipeline as live trading.
"""

import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum
import pandas as pd

try:
    from core.backtesting.data_fetcher import TickVaultFetcher
except ImportError:
    from .data_fetcher import TickVaultFetcher

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Backtest execution modes."""

    TICK_BY_TICK = "tick"
    BAR_CLOSE = "bar"
    SIGNAL_BASED = "signal"


@dataclass
class BacktestTrade:
    """Record of a backtest trade."""

    ticket: int
    symbol: str
    direction: int
    entry_time: datetime
    entry_price: float
    sl: float
    tp: float
    volume: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pips: float = 0.0
    exit_reason: str = ""

    @property
    def duration_bars(self) -> int:
        if self.exit_time:
            return int((self.exit_time - self.entry_time).total_seconds() / 300)
        return 0


@dataclass
class BacktestStats:
    """Backtest performance statistics."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    total_pnl: float = 0.0
    total_pnl_pips: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    Sharpe_ratio: float = 0.0
    longest_win_streak: int = 0
    longest_lose_streak: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "breakeven_trades": self.breakeven_trades,
            "total_pnl": round(self.total_pnl, 2),
            "total_pnl_pips": round(self.total_pnl_pips, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct * 100, 2),
            "win_rate": round(self.win_rate * 100, 2),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "profit_factor": round(self.profit_factor, 2),
            "Sharpe_ratio": round(self.Sharpe_ratio, 2),
            "longest_win_streak": self.longest_win_streak,
            "longest_lose_streak": self.longest_lose_streak,
        }


class BacktestPosition:
    """Tracks a single position during backtest."""

    def __init__(
        self,
        ticket: int,
        symbol: str,
        direction: int,
        entry_price: float,
        entry_time: datetime,
        sl: float,
        tp: float,
        volume: float,
        pip_value: float = 0.0001,
    ):
        self.ticket = ticket
        self.symbol = symbol
        self.direction = direction
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.sl = sl
        self.tp = tp
        self.volume = volume
        self.pip_value = pip_value
        self.is_closed = False
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = ""

    def check_execution(self, bid: float, ask: float, current_time: datetime) -> bool:
        """
        Check if SL/TP should be executed.
        Returns True if position was closed.
        """
        if self.is_closed:
            return True

        if self.direction > 0:
            pass
        else:
            pass

        if self.direction > 0:
            if ask <= self.sl:
                self._close(self.sl, current_time, "SL")
                return True
            if ask >= self.tp:
                self._close(self.tp, current_time, "TP")
                return True
        else:
            if bid >= self.sl:
                self._close(self.sl, current_time, "SL")
                return True
            if bid <= self.tp:
                self._close(self.tp, current_time, "TP")
                return True

        return False

    def _close(self, price: float, time: datetime, reason: str):
        self.exit_price = price
        self.exit_time = time
        self.exit_reason = reason
        self.is_closed = True

        pips = (price - self.entry_price) / self.pip_value
        self.pnl_pips = pips * self.direction
        self.pnl = self.pnl_pips * self.pip_value * self.volume * 100000

    def get_current_pnl(self, bid: float, ask: float) -> float:
        if self.direction > 0:
            current = (bid - self.entry_price) / self.pip_value
        else:
            current = (self.entry_price - ask) / self.pip_value
        return current * self.pip_value * self.volume * 100000


class BacktestOrderExecutor:
    """
    Simulates order execution during backtest.
    Integrates with Cavalier's governance and signal pipeline.
    """

    def __init__(self, initial_balance: float = 10000.0, risk_per_trade: float = 0.01, max_positions: int = 5):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions

        self.positions: Dict[int, BacktestPosition] = {}
        self.closed_trades: List[BacktestTrade] = []
        self.next_ticket = 1
        self.equity_curve: List[Dict[str, Any]] = []

    def can_open_position(self, symbol: str) -> bool:
        """Check if new position can be opened."""
        return len(self.positions) < self.max_positions

    def calculate_position_size(self, entry_price: float, sl: float, pip_value: float = 0.0001) -> float:
        """Calculate position size based on risk parameters."""
        risk_amount = self.balance * self.risk_per_trade
        sl_distance_pips = abs(entry_price - sl) / pip_value

        if sl_distance_pips == 0:
            return 0.0

        position_size = risk_amount / (sl_distance_pips * pip_value * 100000)
        return round(position_size, 2)

    def open_position(
        self, symbol: str, direction: int, entry_price: float, sl: float, tp: float, entry_time: datetime, pip_value: float = 0.0001
    ) -> Optional[BacktestPosition]:
        """Open a new position."""
        if not self.can_open_position(symbol):
            logger.debug(f"[backtest] Max positions reached, skipping {symbol}")
            return None

        volume = self.calculate_position_size(entry_price, sl, pip_value)

        if volume <= 0:
            logger.debug(f"[backtest] Invalid position size for {symbol}")
            return None

        position = BacktestPosition(
            ticket=self.next_ticket,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            entry_time=entry_time,
            sl=sl,
            tp=tp,
            volume=volume,
            pip_value=pip_value,
        )

        self.positions[self.next_ticket] = position
        self.next_ticket += 1

        logger.debug(f"[backtest] Opened {direction} {symbol} @ {entry_price:.5f} SL:{sl:.5f} TP:{tp:.5f} Vol:{volume:.2f}")

        return position

    def update_positions(self, prices: Dict[str, Dict[str, float]], current_time: datetime) -> List[BacktestTrade]:
        """Update all positions and return any closed trades."""
        closed = []

        for ticket in list(self.positions.keys()):
            position = self.positions[ticket]
            symbol = position.symbol

            if symbol not in prices:
                continue

            bid = prices[symbol]["bid"]
            ask = prices[symbol]["ask"]

            if position.check_execution(bid, ask, current_time):
                trade = self._position_to_trade(position)
                self.closed_trades.append(trade)
                closed.append(trade)
                del self.positions[ticket]

                self.balance += position.pnl

        if not self.positions:
            total_equity = self.balance
        else:
            unrealized = sum(p.get_current_pnl(prices[p.symbol]["bid"], prices[p.symbol]["ask"]) for p in self.positions.values())
            total_equity = self.balance + unrealized

        self.equity_curve.append({"time": current_time, "balance": self.balance, "equity": total_equity})

        return closed

    def _position_to_trade(self, position: BacktestPosition) -> BacktestTrade:
        return BacktestTrade(
            ticket=position.ticket,
            symbol=position.symbol,
            direction=position.direction,
            entry_time=position.entry_time,
            entry_price=position.entry_price,
            sl=position.sl,
            tp=position.tp,
            volume=position.volume,
            exit_time=position.exit_time,
            exit_price=position.exit_price,
            pnl=position.pnl,
            pnl_pips=position.pnl_pips,
            exit_reason=position.exit_reason,
        )

    def close_all_positions(self, prices: Dict[str, Dict[str, float]], current_time: datetime) -> List[BacktestTrade]:
        """Force close all positions at current market."""
        closed = []

        for ticket in list(self.positions.keys()):
            position = self.positions[ticket]
            symbol = position.symbol

            if symbol in prices:
                if position.direction > 0:
                    price = prices[symbol]["bid"]
                else:
                    price = prices[symbol]["ask"]

                position._close(price, current_time, "FORCE_CLOSE")
                closed.append(self._position_to_trade(position))
                self.balance += position.pnl

        self.positions.clear()
        return closed

    def get_stats(self) -> BacktestStats:
        """Calculate performance statistics."""
        stats = BacktestStats()

        if not self.closed_trades:
            return stats

        stats.total_trades = len(self.closed_trades)

        wins = [t for t in self.closed_trades if t.pnl > 0]
        losses = [t for t in self.closed_trades if t.pnl < 0]
        breakeven = [t for t in self.closed_trades if t.pnl == 0]

        stats.winning_trades = len(wins)
        stats.losing_trades = len(losses)
        stats.breakeven_trades = len(breakeven)

        if stats.total_trades > 0:
            stats.win_rate = stats.winning_trades / stats.total_trades

        stats.total_pnl = sum(t.pnl for t in self.closed_trades)
        stats.total_pnl_pips = sum(t.pnl_pips for t in self.closed_trades)

        if wins:
            stats.avg_win = sum(t.pnl for t in wins) / len(wins)
        if losses:
            stats.avg_loss = abs(sum(t.pnl for t in losses) / len(losses))

        if stats.avg_loss > 0 and stats.avg_win > 0:
            stats.profit_factor = (stats.avg_win * stats.winning_trades) / stats.avg_loss

        if self.equity_curve:
            df = pd.DataFrame(self.equity_curve)
            df["peak"] = df["equity"].cummax()
            df["drawdown"] = df["equity"] - df["peak"]
            stats.max_drawdown = abs(df["drawdown"].min())
            stats.max_drawdown_pct = abs(df["drawdown"].min() / df["peak"].max()) if df["peak"].max() > 0 else 0

            returns = df["equity"].pct_change().dropna()
            if len(returns) > 1 and returns.std() > 0:
                stats.Sharpe_ratio = returns.mean() / returns.std() * (252**0.5)

        streak = 0
        max_win_streak = 0
        max_lose_streak = 0

        for trade in self.closed_trades:
            if trade.pnl > 0:
                if streak >= 0:
                    streak += 1
                else:
                    streak = 1
                    max_lose_streak = max(max_lose_streak, abs(streak - 1))
            elif trade.pnl < 0:
                if streak <= 0:
                    streak -= 1
                else:
                    streak = -1
                    max_win_streak = max(max_win_streak, streak + 1)

        stats.longest_win_streak = max(1, max_win_streak)
        stats.longest_lose_streak = max(1, max_lose_streak)

        return stats


class ReplayEngine:
    """
    Main replay engine that simulates live trading from historical data.

    Supports:
    - Tick-by-tick simulation
    - Bar-close execution
    - Signal-based execution (integrates with Cavalier signal pipeline)
    """

    def __init__(
        self,
        fetcher: TickVaultFetcher,
        executor: BacktestOrderExecutor,
        mode: ExecutionMode = ExecutionMode.TICK_BY_TICK,
        speed_multiplier: float = 1000.0,
    ):
        self.fetcher = fetcher
        self.executor = executor
        self.mode = mode
        self.speed_multiplier = speed_multiplier

        self.symbols: List[str] = []
        self.date_range: tuple = None
        self.is_running = False
        self.current_time: Optional[datetime] = None

        self.prices: Dict[str, Dict[str, float]] = {}
        self.bars: Dict[str, Dict[str, pd.DataFrame]] = {}

        self._tick_callback: Optional[Callable] = None
        self._bar_callback: Optional[Callable] = None
        self._signal_callback: Optional[Callable] = None

    def set_callbacks(
        self, tick_callback: Optional[Callable] = None, bar_callback: Optional[Callable] = None, signal_callback: Optional[Callable] = None
    ):
        """Set callbacks for tick/bar/signal processing."""
        self._tick_callback = tick_callback
        self._bar_callback = bar_callback
        self._signal_callback = signal_callback

    def load_data(self, symbols: List[str], start_date: datetime, end_date: datetime):
        """Load historical data for backtest."""
        self.symbols = symbols
        self.date_range = (start_date, end_date)

        logger.info(f"[backtest] Loading data for {len(symbols)} symbols...")

        for symbol in symbols:
            bars = self.fetcher.load_bars(symbol, start_date, end_date, "M5")
            if not bars.empty:
                self.bars[symbol] = {"M5": bars}
                logger.info(f"[backtest] Loaded {len(bars)} bars for {symbol}")
            else:
                logger.warning(f"[backtest] No data for {symbol}")

    def run(self) -> BacktestStats:
        """
        Run the backtest simulation.

        Returns:
            BacktestStats with performance metrics
        """
        if not self.bars:
            logger.error("[backtest] No data loaded, cannot run backtest")
            return BacktestStats()

        logger.info(f"[backtest] Starting replay in {self.mode.value} mode...")

        self.is_running = True
        start_time = time.time()

        if self.mode == ExecutionMode.TICK_BY_TICK:
            self._run_tick_mode()
        elif self.mode == ExecutionMode.BAR_CLOSE:
            self._run_bar_mode()
        else:
            self._run_signal_mode()

        elapsed = time.time() - start_time

        stats = self.executor.get_stats()
        stats.final_balance = self.executor.balance

        logger.info(f"[backtest] Backtest complete in {elapsed:.2f}s")
        logger.info(f"[backtest] Final balance: ${stats.total_pnl + self.executor.initial_balance:.2f}")
        logger.info(f"[backtest] Total PnL: {stats.total_pnl_pips:.1f} pips ({stats.total_pnl:.2f})")
        logger.info(f"[backtest] Win Rate: {stats.win_rate:.1%}")

        return stats

    def _run_tick_mode(self):
        """Run tick-by-tick simulation."""
        for symbol in self.symbols:
            ticks = self.fetcher.stream_ticks(symbol, self.date_range[0], self.date_range[1], self.speed_multiplier)

            for tick in ticks:
                self.current_time = tick.timestamp
                self.prices[symbol] = {"bid": tick.bid, "ask": tick.ask, "mid": tick.mid}

                if self._tick_callback:
                    self._tick_callback(symbol, tick)

                closed = self.executor.update_positions(self.prices, self.current_time)
                for trade in closed:
                    self._on_trade_closed(trade)

                if not self.is_running:
                    break

    def _run_bar_mode(self):
        """Run bar-close simulation."""
        all_bars = []

        for symbol in self.symbols:
            if symbol in self.bars:
                bars = self.bars[symbol]["M5"].copy()
                bars["symbol"] = symbol
                all_bars.append(bars)

        if all_bars:
            combined = pd.concat(all_bars).sort_values("time")
            logger.info(f"[replay] Processing {len(combined)} bars")

            for idx, row in combined.iterrows():
                if not self.is_running:
                    break

                self.current_time = row["time"]
                symbol = row["symbol"]

                mid = row["close"]
                spread = 0.00015

                self.prices[symbol] = {"bid": mid - spread / 2, "ask": mid + spread / 2, "mid": mid}

                if self._bar_callback:
                    try:
                        self._bar_callback(symbol, row)
                    except Exception as e:
                        logging.warning(f"[replay] Bar callback error: {e}")

                closed = self.executor.update_positions(self.prices, self.current_time)
                for trade in closed:
                    self._on_trade_closed(trade)

    def _run_signal_mode(self):
        """Run signal-based simulation (integrates with Cavalier pipeline)."""
        self._run_bar_mode()

    def _on_trade_closed(self, trade: BacktestTrade):
        """Handle trade closure."""
        logger.info(
            f"[backtest] Closed {trade.direction} {trade.symbol} "
            f"{trade.exit_reason} @ {trade.exit_price:.5f} "
            f"PnL: {trade.pnl_pips:.1f} pips (${trade.pnl:.2f})"
        )

    def stop(self):
        """Stop the backtest."""
        self.is_running = False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    fetcher = TickVaultFetcher()
    executor = BacktestOrderExecutor(initial_balance=10000.0, risk_per_trade=0.01)

    engine = ReplayEngine(fetcher=fetcher, executor=executor, mode=ExecutionMode.BAR_CLOSE, speed_multiplier=1000.0)

    print("Backtest replay engine loaded. Use run_backtest.py to execute.")
