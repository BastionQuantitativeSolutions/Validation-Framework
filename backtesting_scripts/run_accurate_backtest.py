#!/usr/bin/env python3
"""
Accurate Backtest Runner
========================

Runs accurate backtests using the EXACT same logic as live trading.
Uses: unified_governance, unified_exits, regime detection.

Usage:
    python run_accurate_backtest.py --symbols EURUSD,GBPUSD,USDJPY --start 2024-01-01 --end 2024-12-31
    python run_accurate_backtest.py --symbols ALL --start 2024-01-01 --end 2024-12-31
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from CORE_MODULES.backtesting.optimized_backtester import OptimizedBacktester

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PARQUET_DIR = Path("./sample_project/DATA_MODELS/data_parquet")

FOREX_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "GBPCHF"]

ALL_PAIRS = FOREX_PAIRS + ["XAUUSD", "XAGUSD", "USOIL", "UKOIL", "HEATOIL", "US100", "JP225", "HK50", "UK100"]


def load_parquet_data(symbols: list, start_date: datetime, end_date: datetime, timeframe: str = "M5") -> dict:
    """Load parquet data for symbols."""
    data = {}

    for symbol in symbols:
        parquet_file = PARQUET_DIR / f"{symbol}_{timeframe}.parquet"

        if not parquet_file.exists():
            for tf in ["M15", "M30", "H1"]:
                alt_file = PARQUET_DIR / f"{symbol}_{tf}.parquet"
                if alt_file.exists():
                    parquet_file = alt_file
                    break

        if not parquet_file.exists():
            logger.warning(f"No data file found for {symbol}")
            continue

        try:
            df = pd.read_parquet(parquet_file)

            if df.index.name == "time" or df.index.name == "timestamp" or df.index.name == "datetime":
                df = df.reset_index()

            if "time" not in df.columns and "timestamp" in df.columns:
                df = df.rename(columns={"timestamp": "time"})
            elif "time" not in df.columns and "datetime" in df.columns:
                df = df.rename(columns={"datetime": "time"})

            df["time"] = pd.to_datetime(df["time"])

            required_cols = ["time", "open", "high", "low", "close"]
            if all(col in df.columns for col in required_cols):
                mask = (df["time"] >= start_date) & (df["time"] <= end_date)
                filtered = df[mask].copy()

                if len(filtered) > 0:
                    data[symbol] = filtered
                    logger.info(f"Loaded {len(filtered)} bars for {symbol} ({start_date.date()} to {end_date.date()})")
                else:
                    logger.warning(f"No data for {symbol} in date range")
            else:
                logger.warning(f"Missing required columns in {symbol}: {df.columns.tolist()}")

        except Exception as e:
            logger.error(f"Error loading {symbol}: {e}")

    return data


def cmd_backtest(args):
    """Run backtest."""
    start_date = datetime.strptime(args.start, "%Y-%m-%d")
    end_date = datetime.strptime(args.end, "%Y-%m-%d")

    if args.symbols.upper() == "ALL":
        symbols = ALL_PAIRS
    elif args.symbols.upper() == "FOREX":
        symbols = FOREX_PAIRS
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]

    logger.info(f"Starting backtest: {symbols}")
    logger.info(f"Period: {start_date.date()} to {end_date.date()}")

    data = load_parquet_data(symbols, start_date, end_date)

    if not data:
        logger.error("No data loaded. Exiting.")
        return

    logger.info(f"Loaded {len(data)} symbols with {sum(len(df) for df in data.values())} total bars")

    backtester = OptimizedBacktester(
        initial_balance=args.balance,
        risk_per_trade=args.risk / 100,
        max_positions=args.max_positions,
    )

    results = backtester.run(
        data=data,
        start_date=start_date,
        end_date=end_date,
        min_confidence=0.55,
        min_momentum=0.10,
        max_daily_trades=12,
        min_confirming_factors=1,
    )

    if "error" in results:
        logger.error(f"Backtest error: {results['error']}")
        return

    output_name = args.output or f"accurate_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    backtester.save_results(results, output_name)

    print_results(results, args)


def print_results(results: dict, args):
    """Print formatted results."""
    stats = results.get("stats", {})
    metadata = results.get("metadata", {})

    print("\n" + "=" * 70)
    print("ACCURATE BACKTEST RESULTS (Live Trading Simulation)")
    print("=" * 70)
    print(f"Symbols: {', '.join(metadata.get('symbols', []))}")
    print(f"Period: {metadata.get('start_date', '')[:10]} to {metadata.get('end_date', '')[:10]}")
    print("-" * 70)
    print(f"{'Initial Balance:':<20} ${metadata.get('initial_balance', 0):,.2f}")
    print(
        f"{'Final Balance:':<20} ${metadata.get('final_balance', 0):,.2f} ({'+' if metadata.get('final_balance', 0) >= metadata.get('initial_balance', 0) else ''}${metadata.get('final_balance', 0) - metadata.get('initial_balance', 0):,.2f})"
    )
    print("-" * 70)
    print(f"{'Total Trades:':<20} {stats.get('total_trades', 0)}")
    print(f"{'Winning Trades:':<20} {stats.get('winning_trades', 0)} ({stats.get('win_rate', 0):.1f}%)")
    print(f"{'Losing Trades:':<20} {stats.get('losing_trades', 0)}")
    print("-" * 70)
    print(f"{'Total PnL:':<20} ${stats.get('total_pnl', 0):,.2f} ({stats.get('total_pnl_pips', 0):,.1f} pips)")
    print(f"{'Avg Win:':<20} ${stats.get('avg_win', 0):.2f}")
    print(f"{'Avg Loss:':<20} ${stats.get('avg_loss', 0):.2f}")
    print(f"{'Profit Factor:':<20} {stats.get('profit_factor', 0):.2f}")
    print("-" * 70)
    print(f"{'Max Drawdown:':<20} {stats.get('max_drawdown_pct', 0):.1f}%")
    print(f"{'Avg R-Multiple:':<20} {stats.get('avg_r', 0):.2f}R")
    print(f"{'Avg Bars Held:':<20} {stats.get('avg_bars_held', 0):.1f}")
    print("-" * 70)
    print(f"{'Signals Generated:':<20} {stats.get('signals_generated', 0)}")
    print(f"{'Signals Blocked:':<20} {stats.get('signals_blocked', 0)} ({stats.get('block_rate', 0):.1f}%)")
    print("-" * 70)
    print("\nTop Block Reasons:")
    for reason, count in list(stats.get("blocks_by_reason", {}).items())[:5]:
        print(f"  {reason}: {count}")
    print("-" * 70)
    print("\nTrades by Symbol:")
    for symbol, count in sorted(stats.get("trades_by_symbol", {}).items(), key=lambda x: -x[1]):
        print(f"  {symbol}: {count}")
    print("=" * 70)

    html_path = Path("./sample_project/CORE_MODULES/results/backtest") / f"{args.output or 'accurate_backtest'}.html"
    print(f"\nHTML Report: {html_path}")


def main():
    parser = argparse.ArgumentParser(description="Accurate Backtest Runner - Live Trading Simulation")

    parser.add_argument("--symbols", "-s", required=True, help="Comma-separated symbols, 'FOREX', or 'ALL'")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--balance", "-b", type=float, default=10000, help="Initial balance")
    parser.add_argument("--risk", "-r", type=float, default=0.75, help="Risk per trade %%")
    parser.add_argument("--max-positions", "-m", type=int, default=5, help="Max concurrent positions")
    parser.add_argument("--output", "-o", default=None, help="Output name prefix")

    args = parser.parse_args()
    cmd_backtest(args)


if __name__ == "__main__":
    main()
