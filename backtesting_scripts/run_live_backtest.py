#!/usr/bin/env python3
"""
Live Trading Accurate Backtest Runner
====================================

Runs backtests using the EXACT same logic as live trading:
- Same governance gates as evaluate_entry_governors()
- Same signal generation as signal_engine_hybrid.py
- Same SL/TP calculations as unified_exits.py

Usage:
    python run_live_backtest.py --symbols FOREX --start 2024-01-01 --end 2024-12-31
    python run_live_backtest.py --symbols EURUSD,GBPUSD --start 2024-01-01 --end 2024-06-30
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from CORE_MODULES.backtesting.live_trading_backtester import LiveTradingBacktester

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PARQUET_DIR = Path("C:/Users/jack/Cavalier/DATA_MODELS/data_parquet")

FOREX_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "GBPCHF"]


def load_parquet_data(symbols: list, start_date: datetime, end_date: datetime) -> dict:
    """Load parquet data for symbols."""
    data = {}

    for symbol in symbols:
        parquet_file = PARQUET_DIR / f"{symbol}_M5.parquet"

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

            if df.index.name in ("time", "timestamp", "datetime"):
                df = df.reset_index()

            if "time" not in df.columns:
                if "timestamp" in df.columns:
                    df = df.rename(columns={"timestamp": "time"})
                elif "datetime" in df.columns:
                    df = df.rename(columns={"datetime": "time"})

            df["time"] = pd.to_datetime(df["time"])

            required_cols = ["time", "open", "high", "low", "close"]
            if all(col in df.columns for col in required_cols):
                mask = (df["time"] >= start_date) & (df["time"] <= end_date)
                filtered = df[mask].copy()

                if len(filtered) > 0:
                    data[symbol] = filtered
                    logger.info(f"Loaded {len(filtered):,} bars for {symbol}")
                else:
                    logger.warning(f"No data for {symbol} in date range")
            else:
                logger.warning(f"Missing columns in {symbol}: {df.columns.tolist()}")

        except Exception as e:
            logger.error(f"Error loading {symbol}: {e}")

    return data


def cmd_backtest(args):
    """Run backtest."""
    start_date = datetime.strptime(args.start, "%Y-%m-%d")
    end_date = datetime.strptime(args.end, "%Y-%m-%d")

    if args.symbols.upper() == "ALL":
        symbols = FOREX_PAIRS
    elif args.symbols.upper() == "FOREX":
        symbols = FOREX_PAIRS
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]

    logger.info("=" * 70)
    logger.info("Starting LIVE TRADING SIMULATION backtest")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Period: {start_date.date()} to {end_date.date()}")
    logger.info(f"Balance: ${args.balance:,}")
    logger.info(f"Risk: {args.risk}% per trade")
    logger.info(f"Min Confidence: {args.min_confidence}")
    logger.info(f"Max Daily Trades: {args.max_daily}")
    logger.info("=" * 70)

    data = load_parquet_data(symbols, start_date, end_date)

    if not data:
        logger.error("No data loaded. Exiting.")
        return

    logger.info(f"Loaded {len(data)} symbols with {sum(len(df) for df in data.values()):,} total bars")

    backtester = LiveTradingBacktester(
        initial_balance=args.balance,
        risk_per_trade=args.risk / 100,
        max_daily_trades=args.max_daily,
        min_confidence=args.min_confidence,
        max_positions=args.max_positions,
    )

    results = backtester.run(
        data=data,
        start_date=start_date,
        end_date=end_date,
    )

    if "error" in results:
        logger.error(f"Backtest error: {results['error']}")
        return

    output_name = args.output or f"live_sim_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    backtester.save_results(results, output_name)

    print_results(results)


def print_results(results: dict):
    """Print formatted results."""
    stats = results.get("stats", {})
    metadata = results.get("metadata", {})

    print("\n" + "=" * 70)
    print("LIVE TRADING SIMULATION RESULTS")
    print("=" * 70)
    print(f"Symbols: {', '.join(metadata.get('symbols', []))}")
    print(f"Period: {metadata.get('start_date', '')[:10]} to {metadata.get('end_date', '')[:10]}")
    print("-" * 70)
    print(f"{'Initial Balance:':<25} ${metadata.get('initial_balance', 0):,.2f}")
    final = metadata.get("final_balance", 0)
    initial = metadata.get("initial_balance", 0)
    pnl = final - initial
    print(f"{'Final Balance:':<25} ${final:,.2f} ({'+' if pnl >= 0 else ''}${pnl:,.2f})")
    print("-" * 70)
    print(f"{'Total Trades:':<25} {stats.get('total_trades', 0)}")
    print(f"{'Win Rate:':<25} {stats.get('win_rate', 0):.1f}%")
    print(f"{'Winning Trades:':<25} {stats.get('winning_trades', 0)}")
    print(f"{'Losing Trades:':<25} {stats.get('losing_trades', 0)}")
    print("-" * 70)
    print(f"{'Total PnL:':<25} ${stats.get('total_pnl', 0):,.2f} ({stats.get('total_pnl_pips', 0):,.1f} pips)")
    print(f"{'Avg Win:':<25} ${stats.get('avg_win', 0):.2f}")
    print(f"{'Avg Loss:':<25} ${stats.get('avg_loss', 0):.2f}")
    print(f"{'Profit Factor:':<25} {stats.get('profit_factor', 0):.2f}")
    print(f"{'Avg R-Multiple:':<25} {stats.get('avg_r', 0):.2f}R")
    print("-" * 70)
    print(f"{'Max Drawdown:':<25} {stats.get('max_drawdown_pct', 0):.1f}%")
    print(f"{'Avg Bars Held:':<25} {stats.get('avg_bars_held', 0):.1f}")
    print("-" * 70)
    print(f"{'Signals Generated:':<25} {stats.get('signals_generated', 0)}")
    print(f"{'Signals Blocked:':<25} {stats.get('signals_blocked', 0)} ({stats.get('block_rate', 0):.1f}%)")
    print("-" * 70)
    print("\nTop Block Reasons (Live Trading Governance):")
    for reason, count in list(stats.get("blocks_by_reason", {}).items())[:10]:
        print(f"  {reason}: {count}")
    print("-" * 70)
    print("\nTrades by Symbol:")
    for symbol, count in sorted(stats.get("trades_by_symbol", {}).items(), key=lambda x: -x[1]):
        print(f"  {symbol}: {count}")
    print("=" * 70)

    html_path = Path("C:/Users/jack/Cavalier/CORE_MODULES/results/backtest") / f"{results.get('output_name', 'live_sim')}.html"
    print(f"\nHTML Report: {html_path.parent / (html_path.name + '.html')}")


def main():
    parser = argparse.ArgumentParser(description="Live Trading Accurate Backtest Runner")

    parser.add_argument("--symbols", "-s", required=True, help="Comma-separated symbols, 'FOREX', or 'ALL'")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--balance", "-b", type=float, default=10000, help="Initial balance")
    parser.add_argument("--risk", "-r", type=float, default=0.75, help="Risk per trade %%")
    parser.add_argument("--max-daily", "-d", type=int, default=8, help="Max daily trades per symbol")
    parser.add_argument("--min-confidence", "-c", type=float, default=0.55, help="Min confidence threshold")
    parser.add_argument("--max-positions", "-m", type=int, default=5, help="Max concurrent positions")
    parser.add_argument("--output", "-o", default=None, help="Output name prefix")

    args = parser.parse_args()
    cmd_backtest(args)


if __name__ == "__main__":
    main()
