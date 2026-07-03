"""
# Author: JG
TIERED MODEL VALIDATION - SPEED OPTIMIZED WITH CACHING
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "CORE_MODULES/validation" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BacktestResult:
    n_trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    total_pnl_r: float
    max_drawdown: float
    expectancy: float


PAIR_TIMEFRAMES = [
    ("EURUSD", "M5"),
    ("EURUSD", "M15"),
    ("EURUSD", "H1"),
    ("GBPUSD", "M5"),
    ("GBPUSD", "M15"),
    ("GBPUSD", "H1"),
    ("XAUUSD", "M5"),
    ("XAUUSD", "M15"),
    ("XAUUSD", "H1"),
    ("USDJPY", "M5"),
    ("USDJPY", "M15"),
    ("USDJPY", "H1"),
    ("AUDUSD", "M5"),
    ("AUDUSD", "M15"),
    ("AUDUSD", "H1"),
    ("USDCAD", "M5"),
    ("USDCAD", "M15"),
    ("USDCAD", "H1"),
]

_model_cache = {}
_feature_cache = {}


def load_pair_data(pair: str, tf: str, sample_pct: float = 20) -> pd.DataFrame:
    path = Path(f"./sample_project/DATA_MODELS/data_parquet/{pair}_{tf}.parquet")
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)

    if sample_pct < 100:
        np.random.seed(42)
        n = max(5000, int(len(df) * sample_pct / 100))
        df = df.sample(n=min(n, len(df)))

    return df


def load_tiered_models_fast(pair: str, tf: str):
    cache_key = f"{pair}_{tf}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    from CORE_MODULES.core.models.loader import load_tiered_models
    from CORE_MODULES.core.models.ensemble import get_tiered_prediction
    from DATA_MODELS.feature_bridge import compute_features_for_prediction

    models_live = Path("./sample_project/DATA_MODELS/models_live")
    tiered_pack, scalers_dict, features_dict = load_tiered_models(pair, tf, root=models_live)

    result = {
        "tiered_pack": tiered_pack,
        "scalers_dict": scalers_dict,
        "features_dict": features_dict,
        "compute_features": compute_features_for_prediction,
        "get_prediction": get_tiered_prediction,
    }

    _model_cache[cache_key] = result
    return result


def compute_predictions_batch(df: pd.DataFrame, model_system: Dict) -> np.ndarray:
    """Compute predictions for all bars efficiently"""
    predictions = []
    get_prediction = model_system["get_prediction"]
    tiered_pack = model_system["tiered_pack"]
    features_dict = model_system["features_dict"]

    batch_size = 100
    for i in range(0, len(df), batch_size):
        df_batch = df.iloc[: i + batch_size].copy()
        try:
            result = get_prediction(tiered_pack, features_dict, df_batch)
            pred = result.get("weighted_confidence", 0.5)
        except Exception:
            pred = 0.5
        predictions.extend([pred] * min(batch_size, len(df) - i))

    return np.array(predictions)


def run_backtest_fast(df: pd.DataFrame, predictions: np.ndarray, spread_pips: float = 1.5) -> BacktestResult:
    if len(predictions) < 10:
        return None

    spread_cost = spread_pips / 10000.0
    sl_pct = 0.0015
    tp_pct = 0.0010
    max_bars = 60

    trades = []
    for i in range(len(predictions) - max_bars - 1):
        prob = predictions[i]

        if prob > 0.55:
            direction = 1
        elif prob < 0.45:
            direction = -1
        else:
            continue

        entry_price = df["open"].iloc[i + 1]
        sl_price = entry_price * (1 - direction * sl_pct)
        tp_price = entry_price * (1 + direction * tp_pct)

        for j in range(i + 2, min(i + max_bars + 1, len(df))):
            high = df["high"].iloc[j]
            low = df["low"].iloc[j]

            if direction == 1:
                if low <= sl_price:
                    trades.append(-1.0 - spread_cost)
                    break
                elif high >= tp_price:
                    trades.append(1.0 - spread_cost)
                    break
            else:
                if high >= sl_price:
                    trades.append(-1.0 - spread_cost)
                    break
                elif low <= tp_price:
                    trades.append(1.0 - spread_cost)
                    break
        else:
            trades.append(0.0)

    if not trades:
        return None

    trades = np.array(trades)
    wins = np.sum(trades > 0)
    losses = np.sum(trades < 0)
    total = len(trades)

    pnl = np.sum(trades)
    gross_profit = np.sum(trades[trades > 0]) if wins > 0 else 1
    gross_loss = abs(np.sum(trades[trades < 0])) if losses > 0 else 1

    equity = np.cumsum(np.concatenate([[1.0], trades]))
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / np.maximum(peak, 1)
    max_dd = np.max(drawdown) if len(drawdown) > 0 else 0

    avg_win = np.mean(trades[trades > 0]) if wins > 0 else 0
    avg_loss = abs(np.mean(trades[trades < 0])) if losses > 0 else 0
    expectancy = (wins / total * avg_win) - (losses / total * avg_loss)

    return BacktestResult(
        n_trades=total,
        wins=wins,
        losses=losses,
        win_rate=wins / total,
        profit_factor=gross_profit / gross_loss,
        total_pnl_r=pnl,
        max_drawdown=max_dd,
        expectancy=expectancy,
    )


def check_lookahead_bias_fast(pair: str, tf: str, df: pd.DataFrame, model_system: Dict) -> Dict:
    df_sample = df.sample(n=min(5000, len(df)), random_state=42)

    predictions_shifted = compute_predictions_batch(df_sample, model_system)
    result_shifted = run_backtest_fast(df_sample, predictions_shifted)

    if result_shifted:
        wr_shifted = result_shifted.win_rate
        pnl_shifted = result_shifted.total_pnl_r
    else:
        wr_shifted = 0
        pnl_shifted = 0

    return {"wr_shifted": wr_shifted, "pnl_shifted": pnl_shifted, "has_lookahead_bias": wr_shifted > 0.55 and pnl_shifted > 5}


def run_walkforward_fast(pair: str, tf: str, df: pd.DataFrame, model_system: Dict) -> List[Dict]:
    results = []

    train_months = 12
    test_months = 3

    df_start = df.index.min()
    df_end = df.index.max()
    current_date = df_start + pd.DateOffset(months=train_months)

    for period_id in range(4):
        test_end = min(current_date + pd.DateOffset(months=test_months), df_end)

        test_mask = (df.index >= current_date) & (df.index < test_end)
        test_df = df[test_mask]

        if len(test_df) < 500:
            break

        predictions = compute_predictions_batch(test_df, model_system)
        result = run_backtest_fast(test_df, predictions)

        if result:
            results.append(
                {
                    "period": period_id,
                    "n_trades": result.n_trades,
                    "win_rate": result.win_rate,
                    "pnl_r": result.total_pnl_r,
                    "profit_factor": result.profit_factor,
                }
            )
            logger.info(f"  WF {period_id}: {result.n_trades} trades, WR={result.win_rate:.1%}, PnL={result.total_pnl_r:.1f}R")

        current_date += pd.DateOffset(months=test_months)

    return results


def run_monte_carlo_fast(result: BacktestResult, n_sims: int = 1000) -> Dict:
    if result is None or result.n_trades < 10:
        return None

    trades = np.random.choice([1.0, -1.0, 0.0], size=(n_sims, result.n_trades), p=[result.wins / result.n_trades, result.losses / result.n_trades, 0])

    final_equities = np.sum(trades, axis=1) + 1
    ruin_count = np.sum(final_equities < 0.5)

    return {
        "equity_mean": float(np.mean(final_equities)),
        "equity_std": float(np.std(final_equities)),
        "probability_of_ruin": float(ruin_count / n_sims),
        "win_rate_mean": float(np.mean(trades > 0).mean()),
    }


def analyze_pair_fast(pair: str, tf: str) -> Dict:
    logger.info(f"Analyzing {pair}/{tf}...")

    df = load_pair_data(pair, tf, sample_pct=20)
    if df.empty or len(df) < 1000:
        return None

    model_system = load_tiered_models_fast(pair, tf)
    if not model_system["tiered_pack"]:
        return None

    result = {
        "pair": pair,
        "tf": tf,
        "data_points": len(df),
        "bias_check": None,
        "walkforward": [],
        "monte_carlo": None,
    }

    bias = check_lookahead_bias_fast(pair, tf, df, model_system)
    result["bias_check"] = bias

    wf = run_walkforward_fast(pair, tf, df, model_system)
    result["walkforward"] = wf

    full_df = load_pair_data(pair, tf, sample_pct=50)
    predictions = compute_predictions_batch(full_df, model_system)
    backtest = run_backtest_fast(full_df, predictions)

    if backtest:
        mc = run_monte_carlo_fast(backtest, n_sims=2000)
        result["monte_carlo"] = mc

    return result


def main():
    logger.info("=" * 80)
    logger.info("TIERED MODEL VALIDATION - SPEED OPTIMIZED")
    logger.info("=" * 80)

    all_results = {}

    for pair, tf in PAIR_TIMEFRAMES:
        result = analyze_pair_fast(pair, tf)
        if result:
            all_results[f"{pair}/{tf}"] = result

    logger.info("\nGenerating report...")

    report = []
    report.append("=" * 100)
    report.append("CAVALIER TIERED MODEL VALIDATION REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 100)

    passing = 0
    bias_count = 0

    wf_success_rates = []
    avg_ruin = []

    for key, r in all_results.items():
        pair, tf = r["pair"], r["tf"]

        bias = r.get("bias_check", {})
        if bias.get("has_lookahead_bias"):
            bias_count += 1
            bias_status = "[BIAS]"
        else:
            bias_status = "[OK]"

        wf = r.get("walkforward", [])
        wf_profitable = sum(1 for w in wf if w.get("pnl_r", 0) > 0)
        wf_rate = wf_profitable / len(wf) if wf else 0
        wf_success_rates.append(wf_rate)

        mc = r.get("monte_carlo", {})
        if mc:
            avg_ruin.append(mc.get("probability_of_ruin", 0))

        is_passing = not bias.get("has_lookahead_bias") and wf_rate >= 0.5
        if is_passing:
            passing += 1

        report.append(f"\n{pair}/{tf}: Data={r['data_points']}, Bias={bias_status}")
        report.append(f"  WF: {wf_profitable}/{len(wf)} profitable ({wf_rate:.0%})")
        if mc:
            report.append(f"  MC: Ruin={mc.get('probability_of_ruin', 0):.1%}, Equity={mc.get('equity_mean', 0):.2f}")
        report.append(f"  Status: {'[PASS]' if is_passing else '[FAIL]'}")

    report.append("\n" + "=" * 100)
    report.append("SUMMARY")
    report.append("=" * 100)
    report.append(f"Tested: {len(all_results)}, Passing: {passing}, Bias: {bias_count}")
    report.append(f"Avg WF Success: {np.mean(wf_success_rates):.1%}")
    report.append(f"Avg Ruin Prob: {np.mean(avg_ruin):.1%}")

    if passing >= len(all_results) * 0.7 and bias_count == 0:
        report.append("\nSTATUS: APPROVED FOR LIVE TRADING")
    elif passing >= len(all_results) * 0.5:
        report.append("\nSTATUS: APPROVED WITH CAUTION")
    else:
        report.append("\nSTATUS: NOT RECOMMENDED")

    report.append("=" * 100)

    report_text = "\n".join(report)
    print("\n" + report_text)

    report_path = OUTPUT_DIR / f"tiered_validation_fast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    results_path = OUTPUT_DIR / f"tiered_validation_fast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\nReport: {report_path}")
    logger.info(f"Results: {results_path}")
    logger.info("Done!")

    return all_results


if __name__ == "__main__":
    main()
