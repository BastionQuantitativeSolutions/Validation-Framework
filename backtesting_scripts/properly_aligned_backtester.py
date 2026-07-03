#!/usr/bin/env python3
"""
Properly Aligned ML Backtester
==============================

Uses align_df to ensure features match what models expect.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PARQUET_DIR = Path("C:/Users/jack/Cavalier/DATA_MODELS/data_parquet")
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
SIGNAL_INTERVAL = 4  # Check every 4 bars (1 hour on M15)

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

try:
    from CORE_MODULES.core.features.compute import build_features
    from CORE_MODULES.core.models.loader import load_tiered_models
    from CORE_MODULES.core.models.ensemble import align_df
    from CORE_MODULES.core.config.constants import (
        BASE_BUY_THRESHOLD,
        BASE_SELL_THRESHOLD,
        W_ML,
        W_SMC,
        SL_MULTIPLIERS,
        TP_MULTIPLIERS,
        MIN_CONFIDENCE_GOVERNANCE,
    )

    LIVE_MODULES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import live system modules: {e}")
    LIVE_MODULES_AVAILABLE = False
    BASE_BUY_THRESHOLD = 0.58
    BASE_SELL_THRESHOLD = 0.42
    W_ML = 0.7
    W_SMC = 0.3
    SL_MULTIPLIERS = {"TRENDING": 1.5, "RANGING": 1.0, "VOLATILE": 1.2, "DEFAULT": 1.5}
    TP_MULTIPLIERS = {"TRENDING": 3.0, "RANGING": 2.0, "VOLATILE": 2.5, "DEFAULT": 2.5}
    MIN_CONFIDENCE_GOVERNANCE = 0.70

MODELS_CACHE = {}
FEATURES_CACHE = {}


def get_instrument_config(symbol):
    return INSTRUMENT_CONFIGS.get(symbol, {"pip_size": 0.0001, "pip_value_per_lot": 10.0})


def calculate_position_size(symbol, balance, risk_percent, atr, regime):
    config = get_instrument_config(symbol)
    pip_size = config["pip_size"]
    pip_value_per_lot = config["pip_value_per_lot"]

    sl_multiplier = SL_MULTIPLIERS.get(regime, SL_MULTIPLIERS["DEFAULT"])
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


def calculate_atr(df, idx, period=14):
    if idx < period:
        return None

    high = df.iloc[max(0, idx - period) : idx + 1]["high"].values
    low = df.iloc[max(0, idx - period) : idx + 1]["low"].values
    close = df.iloc[max(0, idx - period) : idx + 1]["close"].values

    tr = np.zeros(len(high))
    for i in range(len(high)):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    return np.mean(tr)


def detect_regime(df, idx, atr):
    if idx < 20 or atr is None or atr == 0:
        return "RANGING"

    start_price = df.iloc[idx - 20]["close"]
    end_price = df.iloc[idx]["close"]
    price_change = (end_price - start_price) / start_price

    current_range = df.iloc[idx]["high"] - df.iloc[idx]["low"]
    volatility_ratio = current_range / atr

    if abs(price_change) > 0.005:
        return "TRENDING"
    elif volatility_ratio > 1.5:
        return "VOLATILE"
    else:
        return "RANGING"


def get_models_cached(pair, tf="M15"):
    cache_key = f"{pair}_{tf}"

    if cache_key not in MODELS_CACHE:
        if not LIVE_MODULES_AVAILABLE:
            MODELS_CACHE[cache_key] = ([], [])
        else:
            try:
                models_tuple = load_tiered_models(pair, tf)
                if models_tuple and len(models_tuple) == 3:
                    all_models = []
                    feature_cols = []

                    full_models, core_models, minimal_models = models_tuple

                    # Extract models from full tier
                    if full_models and "full" in full_models:
                        tier_data = full_models["full"]
                        for model_key in ["cat", "lgb", "xgb"]:
                            if model_key in tier_data:
                                model_list = tier_data[model_key]
                                if isinstance(model_list, list) and len(model_list) > 0:
                                    all_models.append(model_list[0])

                    # Get feature columns from features_dict if available
                    # The features are stored in the model directory
                    import pickle

                    model_dir = Path(f"DATA_MODELS/models_live/{pair}_{tf}/full")
                    features_file = model_dir / "features.pkl"
                    if features_file.exists():
                        with open(features_file, "rb") as f:
                            feature_cols = pickle.load(f)

                    MODELS_CACHE[cache_key] = (all_models, feature_cols)
                    FEATURES_CACHE[cache_key] = feature_cols
                    logger.info(f"Cached {len(all_models)} models and {len(feature_cols)} features for {pair}_{tf}")
                else:
                    MODELS_CACHE[cache_key] = ([], [])
            except Exception as e:
                logger.debug(f"Error loading models for {pair}_{tf}: {e}")
                MODELS_CACHE[cache_key] = ([], [])

    return MODELS_CACHE[cache_key]


def generate_signal(df, idx, pair):
    if idx < 100:
        return None

    models, feature_cols = get_models_cached(pair, "M15")

    if not models:
        return generate_simple_signal(df, idx)

    try:
        window_start = max(0, idx - 100)
        window_df = df.iloc[window_start : idx + 1].copy()
        features_df = build_features(window_df)

        if features_df is None or len(features_df) == 0:
            return None

        last_features = features_df.iloc[-1:]

        # Remove non-numeric columns
        numeric_cols = last_features.select_dtypes(include=[np.number]).columns
        last_features_numeric = last_features[numeric_cols]

        # Align features to match model expectations
        if feature_cols:
            aligned_features = align_df(last_features_numeric, feature_cols)
        else:
            aligned_features = last_features_numeric

        # Replace inf/nan values
        aligned_features = aligned_features.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        predictions = []
        for model in models:
            try:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(aligned_features)
                    if len(proba) > 0 and len(proba[0]) == 2:
                        predictions.append(proba[0][1])
                elif hasattr(model, "predict"):
                    pred = model.predict(aligned_features)
                    if len(pred) > 0:
                        predictions.append(float(pred[0]))
            except Exception:
                continue

        if not predictions:
            return None

        ml_prob = np.mean(predictions)
        fused = ml_prob

        if fused >= BASE_BUY_THRESHOLD:
            direction = 1
        elif fused <= BASE_SELL_THRESHOLD:
            direction = -1
        else:
            return None

        atr = calculate_atr(df, idx)
        if atr is None or atr == 0:
            return None

        regime = detect_regime(df, idx, atr)
        confidence = abs(fused - 0.5) * 2

        if confidence < 0.2:
            return None

        return {"direction": direction, "confidence": confidence, "ml_prob": ml_prob, "fused": fused, "atr": atr, "regime": regime}

    except Exception:
        return generate_simple_signal(df, idx)


def generate_simple_signal(df, idx):
    if idx < 50:
        return None

    ema_fast = df.iloc[max(0, idx - 8) : idx + 1]["close"].ewm(span=8, adjust=False).mean().iloc[-1]
    ema_slow = df.iloc[max(0, idx - 21) : idx + 1]["close"].ewm(span=21, adjust=False).mean().iloc[-1]

    if ema_fast > ema_slow:
        direction = 1
    elif ema_fast < ema_slow:
        direction = -1
    else:
        return None

    atr = calculate_atr(df, idx)
    if atr is None:
        return None

    regime = detect_regime(df, idx, atr)

    return {
        "direction": direction,
        "confidence": 0.7,
        "ml_prob": 0.7 if direction == 1 else 0.3,
        "fused": 0.7 if direction == 1 else 0.3,
        "atr": atr,
        "regime": regime,
    }


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


def run_backtest(all_data, balance=10000, risk_per_trade=0.75, signal_interval=SIGNAL_INTERVAL, max_pairs=3):
    total_trades = 0
    winning_trades = 0
    total_pnl = 0.0
    max_drawdown = 0.0
    peak_balance = balance
    current_balance = balance

    pairs_processed = 0
    for symbol, df in all_data.items():
        if pairs_processed >= max_pairs:
            break

        df = df.sort_values("time").reset_index(drop=True)
        get_models_cached(symbol, "M15")

        logger.info(f"Processing {symbol} ({len(df)} bars)...")

        for i in range(100, len(df), signal_interval):
            signal = generate_signal(df, i, symbol)
            if signal is None:
                continue

            volume = calculate_position_size(symbol, current_balance, risk_per_trade, signal["atr"], signal["regime"])
            entry_price = df.iloc[i]["close"]
            atr = signal["atr"]

            tp_multiplier = TP_MULTIPLIERS.get(signal["regime"], TP_MULTIPLIERS["DEFAULT"])
            sl_multiplier = SL_MULTIPLIERS.get(signal["regime"], SL_MULTIPLIERS["DEFAULT"])

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

        pairs_processed += 1

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


if __name__ == "__main__":
    logger.info("Loading data...")
    all_data = load_all_data()
    logger.info(f"Loaded data for {len(all_data)} pairs")

    logger.info("Running properly aligned ML backtest on first 3 pairs...")
    results = run_backtest(all_data, balance=10000, risk_per_trade=0.75, signal_interval=SIGNAL_INTERVAL, max_pairs=3)

    logger.info("\n=== BACKTEST RESULTS ===")
    logger.info(f"Total trades: {results['total_trades']}")
    logger.info(f"Winning trades: {results['winning_trades']}")
    logger.info(f"Win rate: {results['win_rate']:.1%}")
    logger.info(f"Total PnL: ${results['total_pnl']:.2f}")
    logger.info(f"Max drawdown: {results['max_drawdown']:.1%}")
    logger.info(f"Final balance: ${results['final_balance']:.2f}")

    targets_met = results["win_rate"] >= TARGET_WIN_RATE and results["max_drawdown"] <= MAX_DAILY_DD and results["total_pnl"] > MIN_PROFIT

    if targets_met:
        logger.info("\n*** TARGETS MET! ***")
        output_file = Path("C:/Users/jack/Cavalier/CORE_MODULES/config/backtest_results.json")
        with open(output_file, "w") as f:
            json.dump({"timestamp": datetime.now().isoformat(), "results": results, "targets_met": True}, f, indent=2)
        logger.info(f"Saved results to {output_file}")
    else:
        logger.info(f"\nTargets not met. Win rate: {results['win_rate']:.1%} (target: {TARGET_WIN_RATE:.1%})")
