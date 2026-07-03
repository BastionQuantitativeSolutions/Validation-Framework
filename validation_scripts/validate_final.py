"""
# Author: JG
Validate all 6 pairs with fresh model loading
"""

import sys

sys.path.insert(0, "C:/Users/jack/Cavalier")
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

OUTPUT_DIR = Path("C:/Users/jack/Cavalier/DATA_MODELS/models_live")
DATA_DIR = Path("C:/Users/jack/Cavalier/DATA_MODELS/data_parquet")
PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAUUSD"]
TP_PIPS = {
    "EURUSD": 5,
    "GBPUSD": 10,
    "USDJPY": 10,
    "AUDUSD": 10,
    "USDCAD": 5,
    "XAUUSD": 10,
}
SL_PIPS = {
    "EURUSD": 8,
    "GBPUSD": 15,
    "USDJPY": 15,
    "AUDUSD": 15,
    "USDCAD": 8,
    "XAUUSD": 15,
}
SPREAD_PIPS = 1.5


def compute_features(df):
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
    df["close_open_range"] = (df["close"] - df["open"]) / df["open"]
    for period in [5, 10, 20, 50, 100]:
        df[f"roc_{period}"] = df["close"].pct_change(period)
        df[f"momentum_{period}"] = df["close"] - df["close"].shift(period)
    for period in [5, 10, 20, 50, 100, 200]:
        df[f"sma_{period}"] = df["close"].rolling(period).mean()
        df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
    df["sma_20_50_cross"] = (df["sma_20"] - df["sma_50"]) / df["close"]
    df["ema_20_50_cross"] = (df["ema_20"] - df["ema_50"]) / df["close"]
    df["price_sma20_dist"] = (df["close"] - df["sma_20"]) / df["close"]
    df["price_ema20_dist"] = (df["close"] - df["ema_20"]) / df["close"]
    for period in [5, 10, 20, 50]:
        df[f"volatility_{period}"] = df["returns"].rolling(period).std()
        # True Range calculation: max(high-low, abs(high-previous_close), abs(low-previous_close))
        high_low = df["high"] - df["low"]
        high_pc = abs(df["high"] - df["close"].shift(1))
        low_pc = abs(df["low"] - df["close"].shift(1))
        true_range = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
        df[f"atr_{period}"] = true_range.rolling(period).mean()
    for period in [20, 50]:
        sma = df["close"].rolling(period).mean()
        std = df["close"].rolling(period).std()
        df[f"bb_upper_{period}"] = sma + 2 * std
        df[f"bb_lower_{period}"] = sma - 2 * std
        df[f"bb_width_{period}"] = (df[f"bb_upper_{period}"] - df[f"bb_lower_{period}"]) / sma
        df[f"bb_position_{period}"] = (df["close"] - df[f"bb_lower_{period}"]) / (df[f"bb_upper_{period}"] - df[f"bb_lower_{period}"])
    for period in [7, 14, 21]:
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]
    for period in [14, 21]:
        low_min = df["low"].rolling(period).min()
        high_max = df["high"].rolling(period).max()
        df[f"stoch_{period}"] = 100 * (df["close"] - low_min) / (high_max - low_min)
        df[f"stoch_signal_{period}"] = df[f"stoch_{period}"].rolling(3).mean()
    if "tick_volume" in df.columns:
        df["volume_sma"] = df["tick_volume"].rolling(20).mean()
        df["volume_ratio"] = df["tick_volume"] / df["volume_sma"]
    for lag in [1, 2, 3, 5, 10]:
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
        df[f"returns_lag_{lag}"] = df["returns"].shift(lag)
        df[f"rsi_14_lag_{lag}"] = df["rsi_14"].shift(lag)
    df["trend_strength"] = abs(df["ema_20"] - df["ema_50"]) / df["close"]
    for period in [20, 50, 100]:
        period_high = df["high"].rolling(period).max()
        period_low = df["low"].rolling(period).min()
        df[f"price_position_{period}"] = (df["close"] - period_low) / (period_high - period_low)
    for period in [7, 28]:
        high_max = df["high"].rolling(period).max()
        low_min = df["low"].rolling(period).min()
        df[f"williams_r_{period}"] = -100 * (high_max - df["close"]) / (high_max - low_min)
    df["cci"] = (df["close"] - df["close"].rolling(20).mean()) / (0.015 * df["close"].rolling(20).std())
    if "tick_volume" in df.columns:
        df["volume_std"] = df["tick_volume"].rolling(20).std()
        df["volume_change"] = df["tick_volume"].pct_change()
    df["momentum_accel"] = df["momentum_5"] - df["momentum_5"].shift(1)
    df["roc_accel"] = df["roc_5"] - df["roc_5"].shift(1)
    df["price_velocity"] = df["close"] - df["close"].shift(1)
    df["price_acceleration"] = df["price_velocity"] - df["price_velocity"].shift(1)
    return df.bfill().ffill().fillna(0.0)


print("=" * 70)
print("VALIDATION: All 6 Pairs (Fresh Models)")
print("Trade when: (1) model signal AND (2) bar hits TP/SL")
print("Spread: 1.5 pips")
print("=" * 70)

results = []
for pair in PAIRS:
    model_dir = OUTPUT_DIR / f"{pair}_M1"
    with open(model_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(model_dir / "features.pkl", "rb") as f:
        features = pickle.load(f)
    lgb_model = lgb.Booster(model_file=str(model_dir / "lightgbm_model.txt"))
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(model_dir / "xgboost_model.json"))
    cat_model = CatBoostClassifier()
    cat_model.load_model(str(model_dir / "catboost_model.cbm"))

    df = pd.read_parquet(DATA_DIR / f"{pair}_M1.parquet")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().tail(50000)

    feat = compute_features(df)
    tp = TP_PIPS[pair] / 10000.0
    sl = -SL_PIPS[pair] / 10000.0

    X = feat[features].iloc[:-1].copy()
    X = X.replace([np.inf, -np.inf], 0).fillna(0)
    Xs = scaler.transform(X)

    lgb_preds = lgb_model.predict(Xs)
    xgb_preds = xgb_model.predict(Xs)
    cat_preds = cat_model.predict(Xs)
    ensemble = (lgb_preds + xgb_preds + cat_preds) / 3
    signals = np.sign(ensemble)

    open_prices = df["open"].iloc[1:].values
    close_next = df["close"].iloc[1:].values
    spread = SPREAD_PIPS / 10000.0

    trades = 0
    wins = 0
    pnl_r = 0
    for i in range(len(signals)):
        direction = int(signals[i])
        if direction == 0:
            continue

        ret = (close_next[i] - open_prices[i]) / open_prices[i] * direction
        ret_adjusted = ret - spread if direction > 0 else -ret - spread

        hit_tp = ret_adjusted >= tp
        hit_sl = ret_adjusted <= sl

        if not hit_tp and not hit_sl:
            continue

        trades += 1
        if hit_tp:
            wins += 1
            pnl_r += 1
        else:
            pnl_r -= 1

    wr = wins / trades * 100 if trades > 0 else 0
    pf = pnl_r / (trades - wins) if (trades - wins) > 0 else float("inf") if pnl_r > 0 else 0
    results.append({"pair": pair, "trades": trades, "wr": wr, "pnl": pnl_r, "pf": pf})
    print(f"{pair:8} | {trades:5} trades | {wr:5.1f}% WR | {pnl_r:+7.1f}R | PF: {pf:.2f}")

total_trades = sum(r["trades"] for r in results)
total_pnl = sum(r["pnl"] for r in results)
avg_wr = sum(r["wr"] * r["trades"] for r in results) / total_trades if total_trades > 0 else 0
sep = "=" * 70
print(sep)
print("TOTAL     | " + str(total_trades).rjust(5) + " trades | " + f"{avg_wr:5.1f}% WR | " + f"{total_pnl:+7.1f}R")
print(sep)
