#!/usr/bin/env python3
"""
Cavalier Backtest Runner

Usage:
    python run_backtest.py download --symbols EURUSD GBPUSD USDJPY --start 2026-01-01 --end 2026-01-31
    python run_backtest.py backtest --symbols EURUSD GBPUSD USDJPY --start 2026-01-01 --end 2026-01-31
    python run_backtest.py full --symbols EURUSD GBPUSD USDJPY --start 2026-01-01 --end 2026-01-31
    python run_backtest.py info --symbol EURUSD
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from core.backtesting.data_fetcher import TickVaultFetcher, download_symbols_async
    from core.backtesting.replay_engine import ExecutionMode
    from core.backtesting.cavalier_integration import CavalierBacktester
except ImportError:
    from CORE_MODULES.backtesting.data_fetcher import TickVaultFetcher, download_symbols_async
    from CORE_MODULES.backtesting.replay_engine import ExecutionMode
    from CORE_MODULES.backtesting.cavalier_integration import CavalierBacktester

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_date(date_str: str) -> datetime:
    """Parse date string in various formats."""
    formats = ["%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {date_str}")


def cmd_download(args):
    """Download historical data."""
    symbols = args.symbols.upper().split(",") if isinstance(args.symbols, str) else args.symbols
    start_date = parse_date(args.start)
    end_date = parse_date(args.end)

    logger.info(f"Downloading {symbols} from {start_date.date()} to {end_date.date()}")

    results = asyncio.run(download_symbols_async(symbols, start_date, end_date))

    for symbol, result in results.items():
        if "error" in result:
            logger.error(f"{symbol}: {result['error']}")
        else:
            downloaded = len(result.get("dates_downloaded", []))
            cached = len(result.get("dates_cached", []))
            failed = len(result.get("dates_failed", []))
            logger.info(f"{symbol}: {downloaded} downloaded, {cached} cached, {failed} failed")


def cmd_backtest(args):
    """Run backtest with existing data."""
    symbols = args.symbols.upper().split(",") if isinstance(args.symbols, str) else args.symbols
    start_date = parse_date(args.start)
    end_date = parse_date(args.end)

    TickVaultFetcher()

    parquet_dir = Path("C:/Users/jack/Cavalier/DATA_MODELS/data_parquet")
    for symbol in symbols:
        parquet_files = list(parquet_dir.glob(f"{symbol.upper()}_*.parquet"))
        if len(parquet_files) == 0:
            logger.error(f"No parquet data for {symbol}. Please provide symbols with existing data.")
            return

    backtester = CavalierBacktester(initial_balance=args.balance, risk_per_trade=args.risk / 100, max_positions=args.max_positions)

    mode = ExecutionMode.TICK_BY_TICK if getattr(args, "tick_mode", False) else ExecutionMode.BAR_CLOSE

    logger.info(f"Running backtest: {symbols}")
    logger.info(f"Period: {start_date.date()} to {end_date.date()}")
    logger.info(f"Mode: {mode.value}")
    logger.info(f"Initial Balance: ${args.balance}")
    logger.info(f"Risk per Trade: {args.risk}%")

    results = backtester.run(symbols=symbols, start_date=start_date, end_date=end_date, mode=mode, output_name=args.output)

    if "error" in results:
        logger.error(f"Backtest failed: {results['error']}")
        return

    stats = results["stats"]

    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print("-" * 50)
    print(f"Total Trades:        {stats['total_trades']}")
    print(f"Winning Trades:      {stats['winning_trades']}")
    print(f"Losing Trades:        {stats['losing_trades']}")
    print(f"Win Rate:            {stats['win_rate']:.1f}%")
    print("-" * 50)
    print(f"Total PnL:           ${stats['total_pnl']:.2f} ({stats['total_pnl_pips']:.1f} pips)")
    print(f"Max Drawdown:         {stats['max_drawdown_pct']:.1f}%")
    print(f"Profit Factor:       {stats['profit_factor']:.2f}")
    print(f"Sharpe Ratio:        {stats['Sharpe_ratio']:.2f}")
    print("-" * 50)
    print(f"Final Balance:        ${args.balance + stats['total_pnl']:.2f}")
    print("=" * 50)

    html_path = backtester.results_dir / f"{args.output or 'backtest'}.html"
    print(f"\nHTML Report: {html_path}")


def cmd_full(args):
    """Download and run backtest in one command."""
    symbols = args.symbols.upper().split(",") if isinstance(args.symbols, str) else args.symbols
    start_date = parse_date(args.start)
    end_date = parse_date(args.end)

    logger.info("Step 1: Downloading data...")
    results = asyncio.run(download_symbols_async(symbols, start_date, end_date))

    for symbol, result in results.items():
        if "error" not in result:
            downloaded = len(result.get("dates_downloaded", []))
            logger.info(f"  {symbol}: {downloaded} days downloaded")

    logger.info("\nStep 2: Running backtest...")
    cmd_backtest(args)


def cmd_info(args):
    """Show cached data info."""
    symbol = args.symbol.upper()
    fetcher = TickVaultFetcher()
    info = fetcher.get_symbol_info(symbol)

    if info.get("cached_days", 0) == 0:
        logger.info(f"No cached data for {symbol}")
        return

    print(f"\n{symbol} Cached Data:")
    print(f"  Days Cached: {info['cached_days']}")
    if info.get("date_range"):
        print(f"  Date Range: {info['date_range']['start'][:10]} to {info['date_range']['end'][:10]}")
    print(f"  Cache Dir: {info['cache_directory']}")


def cmd_list(args):
    """List all cached symbols."""
    fetcher = TickVaultFetcher()
    base_dir = fetcher.base_directory

    if not base_dir.exists():
        logger.info("No cached data found")
        return

    symbols = [d.name for d in base_dir.iterdir() if d.is_dir()]

    if not symbols:
        logger.info("No cached data found")
        return

    print("\nCached Symbols:")
    for symbol in sorted(symbols):
        info = fetcher.get_symbol_info(symbol)
        days = info.get("cached_days", 0)
        if info.get("date_range"):
            date_range = f"({info['date_range']['start'][:10]} to {info['date_range']['end'][:10]})"
        else:
            date_range = ""
        print(f"  {symbol}: {days} days {date_range}")


def main():
    parser = argparse.ArgumentParser(description="Cavalier Backtest Runner - Historical data and backtesting")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    download_parser = subparsers.add_parser("download", help="Download historical data")
    download_parser.add_argument("--symbols", "-s", required=True, help="Comma-separated symbols")
    download_parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    download_parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    download_parser.set_defaults(func=cmd_download)

    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument("--symbols", "-s", required=True, help="Comma-separated symbols")
    backtest_parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    backtest_parser.add_argument("--balance", "-b", type=float, default=10000, help="Initial balance")
    backtest_parser.add_argument("--risk", "-r", type=float, default=1.0, help="Risk per trade %%")
    backtest_parser.add_argument("--max-positions", "-m", type=int, default=5, help="Max concurrent positions")
    backtest_parser.add_argument("--output", "-o", default=None, help="Output name prefix")
    backtest_parser.add_argument("--tick", action="store_true", help="Use tick-by-tick mode")
    backtest_parser.set_defaults(func=cmd_backtest)

    full_parser = subparsers.add_parser("full", help="Download and backtest")
    full_parser.add_argument("--symbols", "-s", required=True, help="Comma-separated symbols")
    full_parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    full_parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    full_parser.add_argument("--balance", "-b", type=float, default=10000, help="Initial balance")
    full_parser.add_argument("--risk", "-r", type=float, default=1.0, help="Risk per trade %%")
    full_parser.add_argument("--max-positions", "-m", type=int, default=5, help="Max concurrent positions")
    full_parser.add_argument("--output", "-o", default=None, help="Output name prefix")
    full_parser.add_argument("--tick", action="store_true", help="Use tick-by-tick mode")
    full_parser.set_defaults(func=cmd_full)

    info_parser = subparsers.add_parser("info", help="Show cached data info")
    info_parser.add_argument("--symbol", "-s", required=True, help="Symbol to check")
    info_parser.set_defaults(func=cmd_info)

    list_parser = subparsers.add_parser("list", help="List all cached symbols")
    list_parser.set_defaults(func=cmd_list)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
