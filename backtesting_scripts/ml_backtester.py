#!/usr/bin/env python3
"""
ML-Based Backtester Using Live System Models
=============================================

Uses the actual ML models from the live system to generate signals.
This should match live trading performance more closely.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PARQUET_DIR = Path("./sample_project/DATA_MODELS/data_parquet")
MODELS_DIR = Path("./sample_project/DATA_MODELS/models_live")
ALL_PAIRS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "XAUUSD",
    "USDCHF",
    "GBPCHF",
    "XAGUSD",
    "NZDUSD",
    "USOIL",
    "UKOIL",
    "HEATOIL",
    "JP225",
    "US100",
    "HK50",
    "UK100",
]

TARGET_WIN_RATE = 0.70
MAX_DAILY_DD = 0.03
MIN_PROFIT = 0

INSTRUMENT_CONFIGS = {
    "EURUSD": {"pip_size": 0.0001, "pip_value_per_lot": 10.0},
    "GBPUSD": {"pip_size": 0.0001, "pip_value_per_lot": 10.0},
    "USDJPY": {"pip_size": 0.01, "pip_value_per_lot": 1000.0},
    "AUDUSD": {"pip_size": 0.0001, "pip_value_per_lot": 10.0},
    "USDCAD": {"pip_size": 0.0001, "pip_value_per_lot": 10.0},
    "USDCHF": {"pip_size": 0.0001, "pip_value_per_lot": 10.0},
    "NZDUSD": {"pip_size": 0.0001, "pip_value_per_lot": 10.0},
    "GBPCHF": {"pip_size": 0.0001, "pip_value_per_lot": 10.0},
    "XAUUSD": {"pip_size": 0.01, "pip_value_per_lot": 1.0},
    "XAGUSD": {"pip_size": 0.001, "pip_value_per_lot": 5.0},
    "USOIL": {"pip_size": 0.01, "pip_value_per_lot": 10.0},
    "UKOIL": {"pip_size": 0.01, "pip_value_per_lot": 10.0},
    "HEATOIL": {"pip_size": 0.0001, "pip_value_per_lot": 4.2},
    "JP225": {"pip_size": 1.0, "pip_value_per_lot": 0.00685},
    "US100": {"pip_size": 0.1, "pip_value_per_lot": 20.0},
    "HK50": {"pip_size": 1.0, "pip_value_per_lot": 1.0},
    "UK100": {"pip_size": 0.1, "pip_value_per_lot": 10.0},
}


def get_instrument_config(symbol):
    return INSTRUMENT_CONFIGS.get(symbol, {"pip_size": 0.0001, "pip_value_per_lot": 10.0})


def calculate_position_size(symbol, balance, risk_percent, atr, regime):
    config = get_instrument_config(symbol)
    pip_size = config["pip_size"]
    pip_value_per_lot = config["pip_value_per_lot"]

    sl_multiplier = 1.5 if regime == "TRENDING" else (1.0 if regime == "RANGING" else 1.2)
    sl_distance = atr * sl_multiplier
    sl_pips = sl_distance / pip_size

    if sl_pips == 0:
        return 0.01

    risk_amount = balance * (risk_percent / 100)
    position_size = risk_amount / (sl_pips * pip_value_per_lot)
    return max(0.01, min(position_size, 10.0))


def calculate_pnl(symbol, direction, entry_price, exit_price, volume):
    config = get_instrument_config(symbol)
    pip_size = config["pip_size"]
    pip_value_per_lot = config["pip_value_per_lot"]

    price_diff = (exit_price - entry_price) if direction == 1 else (entry_price - exit_price)
    pips = price_diff / pip_size
    return pips * pip_value_per_lot * volume


def load_models_for_pair_tf(pair, tf):
    """Load ML models for a specific pair and timeframe."""
    model_dir = MODELS_DIR / f"{pair}_{tf}"
    if not model_dir.exists():
        return None

    models = {}
    try:
        # Try to load different model types
        for model_file in model_dir.glob("*.pkl"):
            with open(model_file, "rb") as f:
                models[model_file.stem] = pickle.load(f)

        # Also check for .joblib files
        for model_file in model_dir.glob("*.joblib"):
            import joblib

            models[model_file.stem] = joblib.load(model_file)

        return models if models else None
    except Exception as e:
        logger.warning(f"Error loading models for {pair}_{tf}: {e}")
        return None


def compute_features(df):
    """Compute features for ML models."""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    features = pd.DataFrame(index=df.index)

    # Price-based features
    features["returns_1"] = close.pct_change(1)
    features["returns_5"] = close.pct_change(5)
    features["returns_10"] = close.pct_change(10)
    features["returns_20"] = close.pct_change(20)

    # Volatility features
    features["atr_14"] = (high - low).rolling(window=14).mean()
    features["volatility_ratio"] = (high - low) / features["atr_14"]

    # Trend features
    features["ema_8"] = close.ewm(span=8, adjust=False).mean()
    features["ema_21"] = close.ewm(span=21, adjust=False).mean()
    features["ema_50"] = close.ewm(span=50, adjust=False).mean()
    features["ema_diff_8_21"] = (features["ema_8"] - features["ema_21"]) / features["ema_21"]
    features["ema_diff_21_50"] = (features["ema_21"] - features["ema_50"]) / features["ema_50"]

    # Momentum features
    features["rsi_14"] = 100 - (
        100 / (1 + (close.diff().where(lambda x: x > 0, 0).rolling(14).mean() / (-close.diff().where(lambda x: x < 0, 0)).rolling(14).mean()))
    )

    # Volume features (if available)
    if "tick_volume" in df.columns:
        features["volume_ratio"] = df["tick_volume"] / df["tick_volume"].rolling(20).mean()

    # Time features
    features["hour"] = df.index.hour
    features["day_of_week"] = df.index.dayofweek

    return features.dropna()


def generate_ml_signal(df, idx, pair, tf, models):
    """Generate signal using ML models."""
    if idx < 50 or models is None:
        return None

    # Compute features for this bar
    features_df = compute_features(df.iloc[max(0, idx - 100) : idx + 1])
    if len(features_df) == 0:
        return None

    # Get features for current bar
    current_features = features_df.iloc[-1:]

    # Get predictions from all models
    predictions = []
    probabilities = []

    for model_name, model in models.items():
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(current_features)[0]
                if len(proba) == 2:  # Binary classification
                    probabilities.append(proba[1])  # Probability of class 1 (BUY)
                else:
                    probabilities.append(proba)
            elif hasattr(model, "predict"):
                pred = model.predict(current_features)[0]
                predictions.append(pred)
        except Exception as e:
            logger.debug(f"Error predicting with {model_name}: {e}")
            continue

    if not probabilities and not predictions:
        return None

    # Average probabilities if available
    if probabilities:
        avg_proba = np.mean(probabilities)
        confidence = float(avg_proba)
    else:
        # Use predictions
        avg_pred = np.mean(predictions)
        confidence = float(avg_pred)

    # Determine direction based on confidence
    if confidence > 0.55:
        direction = 1  # BUY
    elif confidence < 0.45:
        direction = -1  # SELL
    else:
        return None  # No clear signal

    # Calculate ATR
    high_low = df.iloc[max(0, idx - 14) : idx + 1]["high"] - df.iloc[max(0, idx - 14) : idx + 1]["low"]
    high_close = abs(df.iloc[max(0, idx - 14) : idx + 1]["high"] - df.iloc[max(0, idx - 14) : idx + 1]["close"].shift(1))
    low_close = abs(df.iloc[max(0, idx - 14) : idx + 1]["low"] - df.iloc[max(0, idx - 14) : idx + 1]["close"].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=14).mean().iloc[-1]

    # Detect regime
    price_change = (df.iloc[idx]["close"] - df.iloc[idx - 20]["close"]) / df.iloc[idx - 20]["close"]
    volatility_ratio = (df.iloc[idx]["high"] - df.iloc[idx]["low"]) / atr if atr > 0 else 1.0

    if abs(price_change) > 0.005:
        regime = "TRENDING"
    elif volatility_ratio > 1.5:
        regime = "VOLATILE"
    else:
        regime = "RANGING"

    # Apply filters based on confidence
    if confidence < 0.65:  # Require higher confidence
        return None

    return {"direction": direction, "confidence": confidence, "atr": atr, "regime": regime, "ml_score": confidence}


def simulate_trade(symbol, direction, entry_price, sl_price, tp_price, df, entry_idx, volume):
    for i in range(entry_idx + 1, len(df)):
        bar = df.iloc[i]
        bar_high = bar["high"]
        bar_low = bar["low"]

        if direction == 1:
            if bar_low <= sl_price:
                return {"exit_price": sl_price, "pnl": calculate_pnl(symbol, direction, entry_price, sl_price, volume), "exit_reason": "SL"}
            if bar_high >= tp_price:
                return {"exit_price": tp_price, "pnl": calculate_pnl(symbol, direction, entry_price, tp_price, volume), "exit_reason": "TP"}
        else:
            if bar_high >= sl_price:
                return {"exit_price": sl_price, "pnl": calculate_pnl(symbol, direction, entry_price, sl_price, volume), "exit_reason": "SL"}
            if bar_low <= tp_price:
                return {"exit_price": tp_price, "pnl": calculate_pnl(symbol, direction, entry_price, tp_price, volume), "exit_reason": "TP"}

    last_price = df.iloc[-1]["close"]
    return {"exit_price": last_price, "pnl": calculate_pnl(symbol, direction, entry_price, last_price, volume), "exit_reason": "END_OF_DATA"}


def run_ml_backtest(all_data, models_dict, balance=10000, risk_per_trade=0.75, min_confidence=0.65):
    total_trades = 0
    winning_trades = 0
    total_pnl = 0.0
    max_drawdown = 0.0
    peak_balance = balance
    current_balance = balance

    for symbol, df in all_data.items():
        df = df.sort_values("time").reset_index(drop=True)

        # Get models for this symbol (try M15 first)
        models = models_dict.get(symbol, {}).get("M15")
        if models is None:
            # Try other timeframes
            for tf in ["M30", "H1", "M5"]:
                models = models_dict.get(symbol, {}).get(tf)
                if models:
                    break

        if models is None:
            logger.debug(f"No models found for {symbol}, skipping")
            continue

        logger.info(f"Using ML models for {symbol}")

        for i in range(50, len(df)):
            signal = generate_ml_signal(df, i, symbol, "M15", models)
            if signal is None:
                continue

            if signal["confidence"] < min_confidence:
                continue

            volume = calculate_position_size(symbol, current_balance, risk_per_trade, signal["atr"], signal["regime"])
            entry_price = df.iloc[i]["close"]
            atr = signal["atr"]

            sl_multiplier = 1.5
            tp_multiplier = 3.0

            if signal["direction"] == 1:
                sl_price = entry_price - (atr * sl_multiplier)
                tp_price = entry_price + (atr * tp_multiplier)
            else:
                sl_price = entry_price + (atr * sl_multiplier)
                tp_price = entry_price - (atr * tp_multiplier)

            trade_result = simulate_trade(symbol, signal["direction"], entry_price, sl_price, tp_price, df, i, volume)

            total_trades += 1
            total_pnl += trade_result["pnl"]
            current_balance += trade_result["pnl"]

            if trade_result["pnl"] > 0:
                winning_trades += 1

            if current_balance > peak_balance:
                peak_balance = current_balance
            drawdown = (peak_balance - current_balance) / peak_balance
            if drawdown > max_drawdown:
                max_drawdown = drawdown

    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "max_drawdown": max_drawdown,
        "final_balance": current_balance,
    }


def load_all_data():
    all_data = {}
    for pair in ALL_PAIRS:
        for tf in ["M15", "M5", "M30", "H1"]:
            file_path = PARQUET_DIR / f"{pair}_{tf}.parquet"
            if file_path.exists():
                df = pd.read_parquet(file_path)
                df = df.reset_index()
                if "index" in df.columns:
                    df = df.rename(columns={"index": "time"})
                elif "datetime" in df.columns:
                    df = df.rename(columns={"datetime": "time"})

                start_date = "2026-02-13"
                end_date = "2026-03-20"
                if "time" in df.columns:
                    df = df[(df["time"] >= start_date) & (df["time"] <= end_date)]
                    if len(df) > 0:
                        all_data[pair] = df
                        break
    return all_data


def load_all_models():
    """Load all available ML models."""
    models_dict = {}

    for pair in ALL_PAIRS:
        models_dict[pair] = {}
        for tf in ["M15", "M5", "M30", "H1"]:
            models = load_models_for_pair_tf(pair, tf)
            if models:
                models_dict[pair][tf] = models
                logger.info(f"Loaded {len(models)} models for {pair}_{tf}")

    return models_dict


if __name__ == "__main__":
    logger.info("Loading data...")
    all_data = load_all_data()
    logger.info(f"Loaded data for {len(all_data)} pairs")

    logger.info("Loading ML models...")
    models_dict = load_all_models()

    # Check how many pairs have models
    pairs_with_models = sum(1 for pair in models_dict if any(models_dict[pair].values()))
    logger.info(f"Found models for {pairs_with_models} pairs")

    if pairs_with_models == 0:
        logger.warning("No ML models found! Using simple signal generation instead.")
        # Fall back to simple signal generation
        from fast_optimize import run_fast_backtest

        results = run_fast_backtest(all_data, min_confidence=0.85, min_momentum=0.015, risk_per_trade=0.75)
    else:
        # Use ML models
        logger.info("Running ML-based backtest...")
        results = run_ml_backtest(all_data, models_dict, balance=10000, risk_per_trade=0.75, min_confidence=0.65)

    logger.info("\n=== ML BACKTEST RESULTS ===")
    logger.info(f"Total trades: {results['total_trades']}")
    logger.info(f"Winning trades: {results['winning_trades']}")
    logger.info(f"Win rate: {results['win_rate']:.1%}")
    logger.info(f"Total PnL: ${results['total_pnl']:.2f}")
    logger.info(f"Max drawdown: {results['max_drawdown']:.1%}")
    logger.info(f"Final balance: ${results['final_balance']:.2f}")

    # Check if targets met
    targets_met = results["win_rate"] >= TARGET_WIN_RATE and results["max_drawdown"] <= MAX_DAILY_DD and results["total_pnl"] > MIN_PROFIT

    if targets_met:
        logger.info("\n*** TARGETS MET! ***")
        output_file = Path("./sample_project/CORE_MODULES/config/ml_backtest_results.json")
        with open(output_file, "w") as f:
            json.dump(
                {"timestamp": datetime.now().isoformat(), "results": results, "targets_met": True, "models_used": pairs_with_models}, f, indent=2
            )
        logger.info(f"Saved results to {output_file}")
    else:
        logger.info(f"\nTargets not met. Win rate: {results['win_rate']:.1%} (target: {TARGET_WIN_RATE:.1%})")
