"""
Unified Backtest Runner
=======================
Runs full backtest using the unified framework.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("backtest_runner")

import os

sys.path.insert(0, os.getcwd())

from unified_backtest_framework import run_backtest
from CORE_MODULES.core.config.constants import BASE_BUY_THRESHOLD as BUY_THR, BASE_SELL_THRESHOLD


def main():
    log.info("=" * 60)
    log.info("UNIFIED BACKTEST RUNNER - Full System Validation")
    log.info("=" * 60)

    pairs = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"]

    log.info("Trading Parameters:")
    log.info(f"  BUY_THRESHOLD: {BUY_THR}")
    log.info(f"  SELL_THRESHOLD: {BASE_SELL_THRESHOLD}")
    log.info(f"  Pairs: {pairs}")

    start = "2025-01-01"
    end = "2025-12-31"

    log.info(f"Period: {start} to {end}")
    log.info("-" * 60)

    trades, stats = run_backtest(
        data_dir="DATA_MODELS/data_parquet",
        pairs=pairs,
        start_date=start,
        end_date=end,
    )

    results = {
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "buy_threshold": BUY_THR,
            "sell_threshold": BASE_SELL_THRESHOLD,
            "pairs": pairs,
            "start_date": start,
            "end_date": end,
        },
        "results": {
            "total_trades": stats.total_trades,
            "winning_trades": stats.winning_trades,
            "losing_trades": stats.losing_trades,
            "win_rate": stats.win_rate,
            "total_pnl_r": stats.total_pnl_r,
            "expectancy": stats.expectancy,
            "avg_win_r": stats.avg_win_r,
            "avg_loss_r": stats.avg_loss_r,
            "max_win_r": stats.max_win_r,
            "max_loss_r": stats.max_loss_r,
        },
    }

    output_path = Path(__file__).parent / "backtest_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info("=" * 60)
    log.info("BACKTEST COMPLETE")
    log.info("=" * 60)
    log.info(f"Total Trades:    {stats.total_trades}")
    log.info(f"Win Rate:        {stats.win_rate:.1%}")
    log.info(f"Expectancy:      {stats.expectancy:.3f}R")
    log.info(f"Total P&L:       {stats.total_pnl_r:.2f}R")
    log.info(f"Avg Win:         {stats.avg_win_r:.2f}R")
    log.info(f"Avg Loss:        {stats.avg_loss_r:.2f}R")
    log.info(f"Max Win:         {stats.max_win_r:.2f}R")
    log.info(f"Max Loss:        {stats.max_loss_r:.2f}R")
    log.info("-" * 60)
    log.info(f"Results saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
