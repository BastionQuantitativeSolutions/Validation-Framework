#!/usr/bin/env python3
"""
Walk-Forward Validation for ML Label-Aligned Mode
=================================================
Runs the live-accurate backtester on hold-out M1 data with the execution
layer forced to match the ML training labels: 1R stop-loss / 1.3R take-profit,
no partials, breakeven, trailing or time-based exits.

This is the P0 diagnostic gate:
    - If profit factor >= 1.2, win rate >= 52%, payoff ratio >= 0.8 and
      at least 30 trades, the models have a real edge and execution was the
      problem.
    - If not, the signal layer must be retrained or rebuilt.

Usage:
    python run_ml_label_aligned_wfv.py --symbols EURUSD,GBPUSD --start 2025-11-01 --end 2026-06-23
    python run_ml_label_aligned_wfv.py --symbols FOREX --start 2025-10-24 --end 2026-06-23 --fold-months 1
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_RAW_DIR = PROJECT_ROOT / "DATA_MODELS" / "data_raw"
DATA_PARQUET_DIR = PROJECT_ROOT / "DATA_MODELS" / "data_parquet"
CONFIG_PATH = PROJECT_ROOT / "CORE_MODULES" / "config" / "cavalier_unified_config.json"
RESULTS_DIR = PROJECT_ROOT / "CORE_MODULES" / "results" / "backtest" / "ml_label_aligned_wfv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

FOREX_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "GBPCHF"
]

GATES = {
    "profit_factor": 1.2,
    "win_rate": 0.52,
    "payoff_ratio": 0.8,
    "min_trades": 30,
}


def activate_ml_label_aligned_config() -> str:
    """Create a temporary config with ml_label_aligned_mode=true and point env to it."""
    with open(CONFIG_PATH, "r", encoding="utf-8-sig") as fh:
        cfg = json.load(fh)

    cfg.setdefault("execution", {})["ml_label_aligned_mode"] = True

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(cfg, tmp, indent=2)
    tmp.close()

    os.environ["CAVALIER_UNIFIED_CONFIG"] = tmp.name
    logger.info(f"[wfv] Activated ml_label_aligned_mode via temp config: {tmp.name}")
    return tmp.name


def _read_raw_csv(path: Path) -> pd.DataFrame:
    """Read a Dukascopy-style raw M1 CSV file.

    Supports two formats:
        - Monthly: time,open,high,low,close,tick_volume
        - Yearly:  YYYY.MM.DD,HH:MM,open,high,low,close,tick_volume
    """
    with open(path, "r", encoding="utf-8") as fh:
        header = fh.readline().strip()

    if header.startswith("time,"):
        df = pd.read_csv(path, parse_dates=["time"])
        df = df[["time", "open", "high", "low", "close", "tick_volume"]]
    else:
        # Yearly format has no header: date, time, o, h, l, c, v
        df = pd.read_csv(
            path,
            header=None,
            names=["date", "time_str", "open", "high", "low", "close", "tick_volume"],
        )
        df["time"] = pd.to_datetime(df["date"] + " " + df["time_str"], format="%Y.%m.%d %H:%M")
        df = df[["time", "open", "high", "low", "close", "tick_volume"]]

    for col in ["open", "high", "low", "close", "tick_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _file_covers_range(path: Path, start: datetime, end: datetime) -> bool:
    """Heuristic: does a raw M1 file likely contain bars inside [start, end]?"""
    stem = path.stem
    suffix = stem.split("_")[-1]

    if len(suffix) == 6:  # YYYYMM
        year, month = int(suffix[:4]), int(suffix[4:])
        file_start = datetime(year, month, 1)
        file_end = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)
    elif len(suffix) == 4:  # YYYY
        year = int(suffix)
        file_start = datetime(year, 1, 1)
        file_end = datetime(year + 1, 1, 1)
    else:
        return False

    return file_end > start and file_start <= end


def load_raw_m1(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Load M1 bars from raw CSV files for a symbol and date range."""
    symbol_lower = symbol.lower()
    files = [
        f for f in sorted(DATA_RAW_DIR.glob(f"dat_mt_{symbol_lower}_m1_*"))
        if _file_covers_range(f, start, end)
    ]

    if not files:
        logger.warning(f"[wfv] No raw M1 files found for {symbol} in range")
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            df = _read_raw_csv(f)
            if not df.empty:
                frames.append(df)
                logger.debug(f"[wfv] Read {len(df):,} rows from {f.name}")
        except Exception as exc:
            logger.warning(f"[wfv] Failed to read {f.name}: {exc}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    mask = (df["time"] >= start) & (df["time"] <= end)
    return df.loc[mask].copy()


def load_parquet_m1(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Load M1 bars from parquet if available."""
    pq = DATA_PARQUET_DIR / f"{symbol}_M1.parquet"
    if not pq.exists():
        return pd.DataFrame()

    df = pd.read_parquet(pq)
    if df.index.name in ("time", "timestamp", "datetime"):
        df = df.reset_index()
    if "time" not in df.columns:
        if "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "time"})
        elif "datetime" in df.columns:
            df = df.rename(columns={"datetime": "time"})

    df["time"] = pd.to_datetime(df["time"])
    df = df[["time", "open", "high", "low", "close", "tick_volume"]]
    mask = (df["time"] >= start) & (df["time"] <= end)
    return df.loc[mask].copy()


def load_m1_data(symbols: List[str], start: datetime, end: datetime) -> Dict[str, pd.DataFrame]:
    """Load M1 data for all requested symbols, preferring parquet then raw CSV."""
    data: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        df = load_parquet_m1(symbol, start, end)
        if df.empty:
            df = load_raw_m1(symbol, start, end)
        if not df.empty:
            data[symbol] = df
            logger.info(f"[wfv] Loaded {len(df):,} M1 bars for {symbol}")
        else:
            logger.warning(f"[wfv] No M1 data for {symbol}")
    return data


def generate_folds(start: datetime, end: datetime, fold_months: int) -> List[Tuple[datetime, datetime]]:
    """Generate walk-forward folds of `fold_months` length."""
    folds = []
    cur = datetime(start.year, start.month, 1)
    while cur < end:
        fold_start = max(cur, start)

        # Advance by fold_months
        years = (cur.month - 1 + fold_months) // 12
        month = ((cur.month - 1 + fold_months) % 12) + 1
        nxt = datetime(cur.year + years, month, 1)

        fold_end = min(nxt - timedelta(seconds=1), end)
        folds.append((fold_start, fold_end))
        cur = nxt
    return folds


def build_label_aligned_backtester(args):
    """Import the live backtester and create a label-aligned subclass instance."""
    # Imports must happen AFTER env var is set so config_sync loads the temp config.
    from CORE_MODULES.backtesting.live_trading_backtester import (
        LiveTradingBacktester, get_instrument_config
    )

    class LabelAlignedBacktester(LiveTradingBacktester):
        def __init__(self, *args, sl_atr_mult: float = 1.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.sl_atr_mult = sl_atr_mult

        def calculate_position_size(self, atr: float, symbol: str = "EURUSD") -> float:
            config = get_instrument_config(symbol)
            pip_size = config["pip_size"]
            pip_value_per_lot = config["pip_value_per_lot"]
            sl_pips = atr / pip_size * self.sl_atr_mult
            if sl_pips == 0:
                return 0.01
            risk_amount = self.balance * self.risk_per_trade
            position_size = risk_amount / (sl_pips * pip_value_per_lot)
            return max(0.01, min(position_size, 2.0))

    return LabelAlignedBacktester(
        initial_balance=args.balance,
        risk_per_trade=args.risk / 100,
        max_daily_trades=args.max_daily,
        min_confidence=args.min_confidence,
        max_positions=args.max_positions,
        sl_atr_mult=1.0,
    )


def evaluate_fold(backtester, symbol_data: Dict[str, pd.DataFrame], fold_start: datetime, fold_end: datetime) -> Dict:
    """Run the backtester on one fold."""
    logger.info(f"[wfv] Running fold {fold_start.date()} -> {fold_end.date()}")
    results = backtester.run(symbol_data, fold_start, fold_end)
    if "error" in results:
        logger.error(f"[wfv] Fold error: {results['error']}")
    return results


def aggregate_results(fold_results: List[Dict]) -> Dict:
    """Aggregate stats across all folds and check pass/fail gates."""
    total_trades = sum(r["stats"].get("total_trades", 0) for r in fold_results)
    winning_trades = sum(r["stats"].get("winning_trades", 0) for r in fold_results)
    losing_trades = sum(r["stats"].get("losing_trades", 0) for r in fold_results)
    total_pnl = sum(r["stats"].get("total_pnl", 0) for r in fold_results)
    signals_generated = sum(r["stats"].get("signals_generated", 0) for r in fold_results)
    signals_blocked = sum(r["stats"].get("signals_blocked", 0) for r in fold_results)

    total_won = sum(
        r["stats"].get("avg_win", 0) * r["stats"].get("winning_trades", 0)
        for r in fold_results
    )
    total_lost = sum(
        r["stats"].get("avg_loss", 0) * r["stats"].get("losing_trades", 0)
        for r in fold_results
    )

    win_rate = winning_trades / total_trades if total_trades else 0.0
    profit_factor = total_won / total_lost if total_lost > 0 else 0.0
    payoff_ratio = (
        (total_won / winning_trades) / (total_lost / losing_trades)
        if winning_trades and losing_trades and total_lost > 0
        else 0.0
    )

    avg_r_weighted = sum(
        r["stats"].get("avg_r", 0) * r["stats"].get("total_trades", 0)
        for r in fold_results
    ) / max(1, total_trades)

    max_dd = max((r["stats"].get("max_drawdown_pct", 0) or 0) for r in fold_results)

    gates_passed = {
        "profit_factor": profit_factor >= GATES["profit_factor"],
        "win_rate": win_rate >= GATES["win_rate"],
        "payoff_ratio": payoff_ratio >= GATES["payoff_ratio"],
        "min_trades": total_trades >= GATES["min_trades"],
    }
    overall_pass = all(gates_passed.values())

    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 3),
        "payoff_ratio": round(payoff_ratio, 3),
        "total_pnl": round(total_pnl, 2),
        "avg_r": round(avg_r_weighted, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "signals_generated": signals_generated,
        "signals_blocked": signals_blocked,
        "gates": gates_passed,
        "overall_pass": overall_pass,
        "thresholds": GATES,
    }


def save_report(report: Dict, output_name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{output_name}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    logger.info(f"[wfv] Report saved: {out_path}")
    return out_path


def print_report(report: Dict):
    print("\n" + "=" * 70)
    print("ML LABEL-ALIGNED WALK-FORWARD VALIDATION")
    print("=" * 70)
    print(f"{'Total Trades:':<25} {report['total_trades']}")
    print(f"{'Win Rate:':<25} {report['win_rate']*100:.2f}%")
    print(f"{'Profit Factor:':<25} {report['profit_factor']:.2f}")
    print(f"{'Payoff Ratio:':<25} {report['payoff_ratio']:.2f}")
    print(f"{'Avg R-Multiple:':<25} {report['avg_r']:.3f}R")
    print(f"{'Total PnL:':<25} ${report['total_pnl']:,.2f}")
    print(f"{'Max Drawdown:':<25} {report['max_drawdown_pct']:.2f}%")
    print(f"{'Signals Generated:':<25} {report['signals_generated']}")
    print(f"{'Signals Blocked:':<25} {report['signals_blocked']}")
    print("-" * 70)
    print("Gates:")
    for gate, passed in report["gates"].items():
        status = "PASS" if passed else "FAIL"
        threshold = report["thresholds"][gate]
        value = report.get(gate, report["total_trades"] if gate == "min_trades" else "-")
        print(f"  {gate:<20} {status:<5} (value={value}, threshold={threshold})")
    print("-" * 70)
    verdict = "PASS - Edge is real; execution was the problem" if report["overall_pass"] else "FAIL - Models lack edge; retrain required"
    print(f"Verdict: {verdict}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="ML label-aligned walk-forward validation")
    parser.add_argument("--symbols", "-s", required=True, help="Comma-separated symbols or 'FOREX'/'ALL'")
    parser.add_argument("--start", required=True, help="Hold-out start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="Hold-out end date (YYYY-MM-DD)")
    parser.add_argument("--fold-months", type=int, default=1, help="Months per walk-forward fold")
    parser.add_argument("--balance", "-b", type=float, default=10000, help="Initial balance")
    parser.add_argument("--risk", "-r", type=float, default=0.5, help="Risk per trade %%")
    parser.add_argument("--max-daily", "-d", type=int, default=8, help="Max daily trades per symbol")
    parser.add_argument("--min-confidence", "-c", type=float, default=0.55, help="Min confidence threshold")
    parser.add_argument("--max-positions", "-m", type=int, default=5, help="Max concurrent positions")
    parser.add_argument("--output", "-o", default=None, help="Output report name")
    args = parser.parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d")
    end_date = datetime.strptime(args.end, "%Y-%m-%d")

    if args.symbols.upper() == "ALL" or args.symbols.upper() == "FOREX":
        symbols = FOREX_PAIRS
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]

    # Activate label-aligned mode before any config-sensitive imports.
    activate_ml_label_aligned_config()

    logger.info("=" * 70)
    logger.info("Starting ML label-aligned walk-forward validation")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Hold-out period: {start_date.date()} to {end_date.date()}")
    logger.info(f"Fold size: {args.fold_months} month(s)")
    logger.info(f"Balance: ${args.balance:,.0f} | Risk: {args.risk}% | Max daily: {args.max_daily}")
    logger.info("=" * 70)

    # Load hold-out M1 data once.
    symbol_data = load_m1_data(symbols, start_date, end_date)
    if not symbol_data:
        logger.error("[wfv] No M1 data loaded. Exiting.")
        sys.exit(1)

    folds = generate_folds(start_date, end_date, args.fold_months)
    if not folds:
        logger.error("[wfv] No folds generated. Exiting.")
        sys.exit(1)

    logger.info(f"[wfv] {len(folds)} fold(s) to evaluate")

    backtester = build_label_aligned_backtester(args)

    fold_results = []
    fold_summaries = []
    for fold_start, fold_end in folds:
        results = evaluate_fold(backtester, symbol_data, fold_start, fold_end)
        fold_results.append(results)
        fold_summaries.append({
            "start": fold_start.isoformat(),
            "end": fold_end.isoformat(),
            "stats": results.get("stats", {}),
        })

    report = aggregate_results(fold_results)
    report["metadata"] = {
        "symbols": symbols,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "fold_months": args.fold_months,
        "folds": fold_summaries,
        "initial_balance": args.balance,
        "risk_per_trade": args.risk / 100,
        "max_daily_trades": args.max_daily,
        "min_confidence": args.min_confidence,
        "max_positions": args.max_positions,
    }

    output_name = args.output or f"ml_aligned_wfv_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    save_report(report, output_name)
    print_report(report)

    sys.exit(0 if report["overall_pass"] else 1)


if __name__ == "__main__":
    main()
