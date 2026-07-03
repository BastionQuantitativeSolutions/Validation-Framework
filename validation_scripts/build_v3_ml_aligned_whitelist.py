#!/usr/bin/env python3
"""
Build a V3 ML Label-Aligned Pair Whitelist
===========================================
Runs the V3 model-driven WFV script pair-by-pair and emits a whitelist of
symbols that pass the edge gate on the given date range.

Pass criteria (hard-coded to match run_v3_ml_label_aligned_wfv.py):
    profit_factor >= 1.2
    win_rate >= 52%
    payoff_ratio >= 0.8
    total_trades >= 30

Usage:
    python build_v3_ml_aligned_whitelist.py --symbols FOREX --start 2025-10-01 --end 2025-10-31
    python build_v3_ml_aligned_whitelist.py --symbols EURUSD,GBPUSD,AUDUSD --start 2025-10-01 --end 2025-10-31 --bar-step 5
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WFV_SCRIPT = PROJECT_ROOT / "CORE_MODULES" / "validation" / "run_v3_ml_label_aligned_wfv.py"
RESULTS_DIR = PROJECT_ROOT / "CORE_MODULES" / "results" / "backtest" / "v3_ml_label_aligned_wfv"

FOREX_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "GBPCHF"
]


def run_for_symbol(symbol: str, args) -> Dict:
    """Run the WFV script for a single symbol and return its report dict."""
    output_name = f"whitelist_{symbol}_{args.start.replace('-', '')}_{args.end.replace('-', '')}"
    cmd = [
        sys.executable,
        str(WFV_SCRIPT),
        "--symbols", symbol,
        "--start", args.start,
        "--end", args.end,
        "--bar-step", str(args.bar_step),
        "--lookback-days", str(args.lookback_days),
        "--balance", str(args.balance),
        "--risk", str(args.risk),
        "--max-daily", str(args.max_daily),
        "--max-lot", str(args.max_lot),
        "--output", output_name,
    ]

    logger.info(f"[whitelist] Running WFV for {symbol}")
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)

    report_path = RESULTS_DIR / f"{output_name}.json"
    if not report_path.exists():
        logger.error(f"[whitelist] No report generated for {symbol}")
        logger.error(proc.stderr[-500:] if proc.stderr else "")
        return {"symbol": symbol, "error": "no report", "stdout": proc.stdout[-500:], "stderr": proc.stderr[-500:]}

    try:
        with open(report_path, "r", encoding="utf-8") as fh:
            report = json.load(fh)
    except Exception as exc:
        logger.error(f"[whitelist] Failed to parse report for {symbol}: {exc}")
        return {"symbol": symbol, "error": str(exc)}

    report["symbol"] = symbol
    return report


def build_whitelist(reports: List[Dict]) -> Dict:
    """Aggregate individual reports into a whitelist summary."""
    rows = []
    whitelist = []
    for r in reports:
        if "error" in r:
            rows.append({
                "symbol": r["symbol"],
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "payoff_ratio": 0.0,
                "avg_r": 0.0,
                "total_pnl": 0.0,
                "passed": False,
                "error": r["error"],
            })
            continue

        gates = r.get("gates", {})
        passed = r.get("overall_pass", False)
        row = {
            "symbol": r["symbol"],
            "total_trades": r.get("total_trades", 0),
            "win_rate": r.get("win_rate", 0.0),
            "profit_factor": r.get("profit_factor", 0.0),
            "payoff_ratio": r.get("payoff_ratio", 0.0),
            "avg_r": r.get("avg_r", 0.0),
            "total_pnl": r.get("total_pnl", 0.0),
            "passed": passed,
            "gate_details": gates,
        }
        rows.append(row)
        if passed:
            whitelist.append(r["symbol"])

    return {
        "whitelisted_symbols": whitelist,
        "symbol_results": rows,
        "summary": {
            "total_symbols_tested": len(rows),
            "symbols_passed": len(whitelist),
            "symbols_failed": len(rows) - len(whitelist),
        },
    }


def print_whitelist(summary: Dict):
    rows = summary["symbol_results"]
    print("\n" + "=" * 90)
    print("V3 ML LABEL-ALIGNED PAIR WHITELIST")
    print("=" * 90)
    print(f"{'Symbol':<10} {'Trades':>7} {'WR%':>7} {'PF':>6} {'Payoff':>7} {'AvgR':>7} {'PnL $':>10} {'Status':>8}")
    print("-" * 90)
    for row in sorted(rows, key=lambda x: (-x["passed"], -x["profit_factor"])):
        status = "PASS" if row["passed"] else "FAIL"
        if "error" in row:
            print(f"{row['symbol']:<10} {'ERROR':>7} {row.get('error', ''):<60}")
            continue
        print(
            f"{row['symbol']:<10} {row['total_trades']:>7} {row['win_rate']*100:>7.2f} "
            f"{row['profit_factor']:>6.2f} {row['payoff_ratio']:>7.2f} {row['avg_r']:>7.3f} "
            f"{row['total_pnl']:>10.2f} {status:>8}"
        )
    print("-" * 90)
    print(f"Whitelisted ({summary['summary']['symbols_passed']}): {', '.join(summary['whitelisted_symbols']) or 'NONE'}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Build V3 ML label-aligned pair whitelist")
    parser.add_argument("--symbols", "-s", required=True, help="Comma-separated symbols or 'FOREX'/'ALL'")
    parser.add_argument("--start", required=True, help="Hold-out start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="Hold-out end date (YYYY-MM-DD)")
    parser.add_argument("--bar-step", type=int, default=5, help="Evaluate every Nth M1 bar")
    parser.add_argument("--lookback-days", type=int, default=5, help="Extra history for feature windows")
    parser.add_argument("--balance", "-b", type=float, default=10000, help="Initial balance")
    parser.add_argument("--risk", "-r", type=float, default=0.5, help="Risk per trade %%")
    parser.add_argument("--max-daily", "-d", type=int, default=8, help="Max daily trades per symbol")
    parser.add_argument("--max-lot", type=float, default=2.0, help="Max lot size per trade")
    parser.add_argument("--output", "-o", default=None, help="Output whitelist JSON name")
    args = parser.parse_args()

    if args.symbols.upper() in ("ALL", "FOREX"):
        symbols = FOREX_PAIRS
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]

    logger.info("=" * 70)
    logger.info("Building V3 ML label-aligned pair whitelist")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Period: {args.start} to {args.end}")
    logger.info("=" * 70)

    reports = [run_for_symbol(sym, args) for sym in symbols]
    summary = build_whitelist(reports)

    output_name = args.output or f"v3_ml_aligned_whitelist_{args.start.replace('-', '')}_{args.end.replace('-', '')}"
    out_path = RESULTS_DIR / f"{output_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(f"[whitelist] Whitelist saved: {out_path}")

    print_whitelist(summary)

    if summary["whitelisted_symbols"]:
        print(f"\nRecommended live universe: {', '.join(summary['whitelisted_symbols'])}")
        sys.exit(0)
    else:
        print("\nNo symbols passed the gate. Do not go live with this period/parameter set.")
        sys.exit(1)


if __name__ == "__main__":
    main()
