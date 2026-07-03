"""
Cavalier Backtesting Module

Provides historical data fetching and backtesting capabilities using TickVault
for absolute parity with live trading.
"""

try:
    from core.backtesting.data_fetcher import TickVaultFetcher, TickData, BarData, download_symbols_async
except ImportError:
    from .data_fetcher import TickVaultFetcher, TickData, BarData, download_symbols_async

try:
    from core.backtesting.replay_engine import ReplayEngine, BacktestOrderExecutor, BacktestPosition, BacktestTrade, BacktestStats, ExecutionMode
except ImportError:
    from .replay_engine import ReplayEngine, BacktestOrderExecutor, BacktestPosition, BacktestTrade, BacktestStats, ExecutionMode

try:
    from core.backtesting.cavalier_integration import CavalierSignalGenerator, CavalierBacktester, download_and_backtest
except ImportError:
    from .cavalier_integration import CavalierSignalGenerator, CavalierBacktester, download_and_backtest

__all__ = [
    "TickVaultFetcher",
    "TickData",
    "BarData",
    "download_symbols_async",
    "ReplayEngine",
    "BacktestOrderExecutor",
    "BacktestPosition",
    "BacktestTrade",
    "BacktestStats",
    "ExecutionMode",
    "CavalierSignalGenerator",
    "CavalierBacktester",
    "download_and_backtest",
]
