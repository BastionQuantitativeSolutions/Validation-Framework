"""
# Author: JG
OPTIMIZED VALIDATION - EURUSD/M5 ONLY
Caching + smaller sample for quick performance assessment
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
import json
import pickle
import hashlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CORE_MODULES"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "DATA_MODELS"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "CORE_MODULES/validation" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class FeatureCache:
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache = {}
        self._load_cache()

    def _load_cache(self):
        cache_file = self.cache_dir / "feature_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    self.cache = pickle.load(f)
                logger.info(f"[CACHE] Loaded {len(self.cache)} cached feature sets")
            except Exception:
                self.cache = {}

    def _save_cache(self):
        cache_file = self.cache_dir / "feature_cache.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(self.cache, f)
        logger.info(f"[CACHE] Saved {len(self.cache)} feature sets")

    def get_key(self, df):
        if len(df) < 2:
            return None
        sig = df.index[-1].strftime("%Y%m%d%H%M%S") + str(len(df))
        return hashlib.md5(sig.encode()).hexdigest()

    def get(self, df):
        key = self.get_key(df)
        if key and key in self.cache:
            return self.cache[key]
        return None

    def set(self, df, features):
        key = self.get_key(df)
        if key:
            self.cache[key] = features


def load_pair_data(pair: str, tf: str, n_samples: int = 200) -> pd.DataFrame:
    path = Path(f"C:/Users/jack/Cavalier/DATA_MODELS/data_parquet/{pair}_{tf}.parquet")
    if not path.exists():
        logger.error(f"[DATA] File not found: {path}")
        return pd.DataFrame()

    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.dropna()
    logger.info(f"[DATA] Loaded {len(df)} bars for {pair}/{tf}")
    return df.tail(n_samples * 2)


def get_tiered_predictions_batch(df: pd.DataFrame, pair: str, tf: str) -> np.ndarray:
    from CORE_MODULES.core.models.loader import load_tiered_models
    from DATA_MODELS.feature_bridge import compute_features_for_prediction

    models_live = Path("C:/Users/jack/Cavalier/DATA_MODELS/models_live")
    tiered_pack, scalers_dict, features_dict = load_tiered_models(pair, tf, root=models_live)

    if not tiered_pack:
        logger.error(f"[MODEL] Failed to load models for {pair}/{tf}")
        return np.zeros(len(df)) + 0.5

    logger.info(f"[MODEL] Loaded tiered models for {pair}/{tf}")
    predictions = []
    feature_cache = FeatureCache(CACHE_DIR)

    for i in range(len(df)):
        if i % 50 == 0:
            logger.info(f"[PRED] Progress: {i + 1}/{len(df)}")

        df_sample = df.iloc[: i + 1].copy()

        cached = feature_cache.get(df_sample)
        if cached is not None:
            X = cached
        else:
            X = compute_features_for_prediction(df_sample)
            feature_cache.set(df_sample, X)

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

            tier_probs = []
            for name in ["cat", "lgb", "xgb"]:
                models = pack.get(name, [])
                if models:
                    m = models[0]
                    try:
                        if hasattr(m, "predict_proba"):
                            prob = m.predict_proba(X_scaled[-1:])[0, 1]
                        else:
                            prob = m.predict(X_scaled[-1:])
                            if isinstance(prob, (list, np.ndarray)):
                                prob = prob[0]
                    except Exception:
                        prob = 0.5
                    tier_probs.append(prob)

            if tier_probs:
                tier_avg = np.mean(tier_probs)
                weighted_sum += tier_avg * weight
                total_weight += weight

        if total_weight > 0:
            final_prob = weighted_sum / total_weight
        else:
            final_prob = 0.5

        predictions.append(final_prob)

    feature_cache._save_cache()
    return np.array(predictions)


def compute_metrics(predictions: np.ndarray, df: pd.DataFrame) -> dict:
    if "returns" not in df.columns:
        returns = df["close"].pct_change().shift(-1).fillna(0)
    else:
        returns = df["returns"].fillna(0)

    direction = (predictions - 0.5) * 2
    strat_returns = direction * returns

    wins = (strat_returns > 0).sum()
    losses = (strat_returns < 0).sum()
    total = wins + losses
    win_rate = wins / total if total > 0 else 0

    avg_win = strat_returns[strat_returns > 0].mean() if wins > 0 else 0
    avg_loss = abs(strat_returns[strat_returns < 0].mean()) if losses > 0 else 0

    profit_factor = (avg_win * wins) / (avg_loss * losses) if (avg_loss * losses) > 0 else float("inf")

    cumulative = (1 + strat_returns).cumprod()
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / peak
    max_dd = drawdown.max()

    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    return {
        "n_trades": int(total),
        "win_rate": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "profit_factor": float(profit_factor),
        "max_drawdown": float(max_dd),
        "expectancy": float(expectancy),
        "cumulative_return": float(cumulative[-1]) if len(cumulative) > 0 else 1.0,
    }


def main():
    pair = "EURUSD"
    tf = "M5"
    n_samples = 50

    logger.info(f"{'=' * 60}")
    logger.info(f"OPTIMIZED VALIDATION: {pair}/{tf}")
    logger.info(f"{'=' * 60}")

    df = load_pair_data(pair, tf, n_samples)
    if df.empty:
        logger.error("[FAIL] No data loaded")
        return

    predictions = get_tiered_predictions_batch(df, pair, tf)
    if len(predictions) == 0:
        logger.error("[FAIL] No predictions generated")
        return

    metrics = compute_metrics(predictions, df)

    results = {
        "timestamp": datetime.now().isoformat(),
        "pair": pair,
        "tf": tf,
        "n_samples": n_samples,
        "predictions_mean": float(predictions.mean()),
        "predictions_std": float(predictions.std()),
        "metrics": metrics,
    }

    out_file = OUTPUT_DIR / f"optimized_validation_{pair}_{tf}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"{'=' * 60}")
    logger.info(f"RESULTS: {pair}/{tf}")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Trades: {metrics['n_trades']}")
    logger.info(f"  Win Rate: {metrics['win_rate'] * 100:.1f}%")
    logger.info(f"  Avg Win: {metrics['avg_win'] * 100:.2f}%")
    logger.info(f"  Avg Loss: {metrics['avg_loss'] * 100:.2f}%")
    logger.info(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    logger.info(f"  Max Drawdown: {metrics['max_drawdown'] * 100:.1f}%")
    logger.info(f"  Expectancy: {metrics['expectancy'] * 100:.3f}%")
    logger.info(f"  Cumulative Return: {(metrics['cumulative_return'] - 1) * 100:.2f}%")
    logger.info(f"{'=' * 60}")

    if metrics["win_rate"] > 0.50 and metrics["expectancy"] > 0:
        logger.info("[PASS] Strategy shows positive expectancy")
    else:
        logger.info("[FAIL] Strategy does not show positive expectancy")

    logger.info(f"[SAVE] Results saved to: {out_file}")


if __name__ == "__main__":
    main()
