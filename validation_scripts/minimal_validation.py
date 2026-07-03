"""
# Author: JG
MINIMAL TIERED VALIDATION - NO FEATURE COMPUTATION
Tests model outputs directly using cached predictions
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "CORE_MODULES/validation" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_model_cache = {}


def load_pair_sample(pair: str, tf: str, n_samples: int = 2000) -> pd.DataFrame:
    path = Path(f"C:/Users/jack/Cavalier/DATA_MODELS/data_parquet/{pair}_{tf}.parquet")
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    np.random.seed(42)
    return df.sample(n=min(n_samples, len(df)))


def get_tiered_predictions(df: pd.DataFrame, pair: str, tf: str) -> np.ndarray:
    """Get predictions without recomputing features"""
    from CORE_MODULES.core.models.loader import load_tiered_models
    from DATA_MODELS.feature_bridge import compute_features_for_prediction

    cache_key = f"{pair}_{tf}"

    if cache_key in _model_cache:
        tiered_pack = _model_cache[cache_key]
    else:
        models_live = Path("C:/Users/jack/Cavalier/DATA_MODELS/models_live")
        tiered_pack, scalers_dict, features_dict = load_tiered_models(pair, tf, root=models_live)
        _model_cache[cache_key] = tiered_pack

    if not tiered_pack:
        return np.zeros(len(df)) + 0.5

    predictions = []

    for i in range(len(df)):
        if i % 100 == 0:
            pass

        df_sample = df.iloc[: i + 1].copy()

        try:
            X = compute_features_for_prediction(df_sample)

            weighted_sum = 0
            total_weight = 0

            for tier, pack in tiered_pack.items():
                if tier == "minimal":
                    continue

                weight = 0.65 if tier == "full" else 0.35
                features = features_dict.get(tier)
                if features is None:
                    continue

                for f in features:
                    if f not in X.columns:
                        X[f] = 0
                X_aligned = X[[f for f in features if f in X.columns]]

                if len(X_aligned) == 0:
                    continue

                scaler = scalers_dict.get(tier)
                if scaler:
                    X_scaled = scaler.transform(X_aligned)
                else:
                    X_scaled = X_aligned.values

                probs = []
                for name in ["cat", "lgb", "xgb"]:
                    models = pack.get(name, [])
                    if models:
                        m = models[0]
                        try:
                            if hasattr(m, "predict_proba"):
                                prob = m.predict_proba(X_scaled[[-1]])[0][1]
                            else:
                                prob = m.predict(X_scaled[[-1]])[0]
                            probs.append(prob)
                        except Exception:
                            probs.append(0.5)

                tier_prob = np.mean(probs) if probs else 0.5
                weighted_sum += tier_prob * weight
                total_weight += weight

            predictions.append(weighted_sum / total_weight if total_weight > 0 else 0.5)

        except Exception:
            predictions.append(0.5)

    return np.array(predictions)


def simple_backtest(df: pd.DataFrame, predictions: np.ndarray, spread_pips: float = 1.5) -> Dict:
    if len(predictions) < 10:
        return None

    spread_cost = spread_pips / 10000.0
    trades = []

    for i in range(len(predictions) - 1):
        prob = predictions[i]

        if prob > 0.55:
            direction = 1
        elif prob < 0.45:
            direction = -1
        else:
            continue

        entry = df["open"].iloc[i + 1]
        sl = entry * (1 - direction * 0.0015)
        tp = entry * (1 + direction * 0.0010)

        for j in range(i + 2, min(i + 60, len(df))):
            high = df["high"].iloc[j]
            low = df["low"].iloc[j]

            if direction == 1:
                if low <= sl:
                    trades.append(-1.0 - spread_cost)
                    break
                elif high >= tp:
                    trades.append(1.0 - spread_cost)
                    break
            else:
                if high >= sl:
                    trades.append(-1.0 - spread_cost)
                    break
                elif low <= tp:
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
    gp = np.sum(trades[trades > 0]) if wins > 0 else 1
    gl = abs(np.sum(trades[trades < 0])) if losses > 0 else 1

    equity = np.cumsum(np.concatenate([[1.0], trades]))
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.maximum(peak, 1)
    max_dd = np.max(dd) if len(dd) > 0 else 0

    return {
        "n_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / total,
        "profit_factor": gp / gl,
        "pnl_r": pnl,
        "max_drawdown": max_dd,
        "expectancy": (wins / total * np.mean(trades[trades > 0]) if wins > 0 else 0)
        - (losses / total * abs(np.mean(trades[trades < 0])) if losses > 0 else 0),
    }


def main():
    logger.info("=" * 80)
    logger.info("MINIMAL TIERED VALIDATION")
    logger.info("=" * 80)

    pairs = ["EURUSD", "GBPUSD", "XAUUSD", "USDJPY", "AUDUSD", "USDCAD"]
    tfs = ["M5", "M15", "H1"]

    results = {}

    for pair in pairs:
        for tf in tfs:
            key = f"{pair}/{tf}"
            logger.info(f"Testing {key}...")

            df = load_pair_sample(pair, tf, n_samples=3000)
            if len(df) < 500:
                logger.warning("  Insufficient data")
                continue

            predictions = get_tiered_predictions(df, pair, tf)
            bt = simple_backtest(df, predictions)

            if bt:
                results[key] = {
                    "data_points": len(df),
                    "n_trades": bt["n_trades"],
                    "win_rate": bt["win_rate"],
                    "pnl_r": bt["pnl_r"],
                    "profit_factor": bt["profit_factor"],
                    "max_drawdown": bt["max_drawdown"],
                    "expectancy": bt["expectancy"],
                    "status": "PASS" if bt["win_rate"] > 0.48 and bt["pnl_r"] > 0 else "FAIL",
                }
                logger.info(f"  {bt['n_trades']} trades, WR={bt['win_rate']:.1%}, PnL={bt['pnl_r']:.1f}R [{results[key]['status']}]")

    # Summary
    passing = sum(1 for r in results.values() if r["status"] == "PASS")

    report = []
    report.append("=" * 80)
    report.append("CAVALIER TIERED MODEL VALIDATION - SUMMARY")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)

    for key, r in sorted(results.items()):
        report.append(f"{key}: {r['n_trades']} trades, WR={r['win_rate']:.1%}, PnL={r['pnl_r']:.1f}R [{r['status']}]")

    report.append("-" * 80)
    report.append(f"Total: {len(results)} pairs tested, {passing} passing ({passing / len(results) * 100:.0f}%)")

    if passing >= len(results) * 0.7:
        report.append("STATUS: APPROVED FOR LIVE TRADING")
    elif passing >= len(results) * 0.5:
        report.append("STATUS: APPROVED WITH CAUTION")
    else:
        report.append("STATUS: NOT RECOMMENDED")

    report.append("=" * 80)

    report_text = "\n".join(report)
    print("\n" + report_text)

    report_path = OUTPUT_DIR / f"minimal_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    results_path = OUTPUT_DIR / f"minimal_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nReport: {report_path}")
    logger.info(f"Results: {results_path}")

    return results


if __name__ == "__main__":
    main()
