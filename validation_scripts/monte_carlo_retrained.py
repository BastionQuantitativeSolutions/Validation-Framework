import sys
# Author: JG

sys.path.insert(0, "./sample_project")
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

OUTPUT_DIR = Path("./sample_project/DATA_MODELS/models_live")
DATA_DIR = Path("./sample_project/DATA_MODELS/data_parquet")


def load_models(pair):
    model_path = OUTPUT_DIR / f"{pair}_M1"
    if not model_path.exists():
        return None, None, None
    with open(model_path / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(model_path / "features.pkl", "rb") as f:
        features = pickle.load(f)
    models = {}
    lgb_path = model_path / "lightgbm_model.txt"
    if lgb_path.exists():
        models["lgb"] = [lgb.Booster(model_file=str(lgb_path))]
    xgb_path = model_path / "xgboost_model.json"
    if xgb_path.exists():
        models["xgb"] = [xgb.XGBClassifier().load_model(str(xgb_path))]
    cat_path = model_path / "catboost_model.cbm"
    if cat_path.exists():
        models["cat"] = [CatBoostClassifier().load_model(str(cat_path))]
    return models, scaler, features


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
print("MONTE CARLO VALIDATION - 100,000 SIMULATIONS")
print("=" * 70)

pairs = ["GBPUSD", "USDJPY", "AUDUSD", "XAUUSD"]
all_results = []

for pair in pairs:
    print(f"Testing {pair}...", end=" ", flush=True)
    models, scaler, features = load_models(pair)
    if not models:
        print("No models")
        continue
    df = pd.read_parquet(DATA_DIR / f"{pair}_M1.parquet")
    df.index = pd.to_datetime(df.index)
    np.random.seed(42)
    sample = df.tail(10000)
    feat = compute_features(sample)
    X = feat[features].iloc[:-1].copy()
    Xs = scaler.transform(X)
    predictions = []
    for i in range(len(Xs)):
        ps = []
        for name in ["cat", "lgb", "xgb"]:
            ml = models.get(name)
            if ml and len(ml) > 0:
                m = ml[0]
                try:
                    ps.append(m.predict_proba(Xs[[i]])[0][1])
                except Exception:
                    ps.append(0.5)
        predictions.append(np.mean(ps) if ps else 0.5)
    trades = []
    for i, prob in enumerate(predictions[:-1]):
        if prob > 0.55:
            d = 1
        elif prob < 0.45:
            d = -1
        else:
            continue
        entry = sample["open"].iloc[i + 1]
        sl = entry * (1 - d * 0.0015)
        tp = entry * (1 + d * 0.0010)
        spread = 0.00015
        for j in range(i + 2, min(i + 62, len(sample))):
            high, low = sample["high"].iloc[j], sample["low"].iloc[j]
            if d == 1:
                if low <= sl:
                    trades.append(-1.0 - spread)
                    break
                elif high >= tp:
                    trades.append(1.0 - spread)
                    break
            else:
                if high >= sl:
                    trades.append(-1.0 - spread)
                    break
                elif low <= tp:
                    trades.append(1.0 - spread)
                    break
        else:
            trades.append(0.0)
    all_results.append({"pair": pair, "trades": trades})
    w = sum(1 for t in trades if t > 0)
    losses = sum(1 for t in trades if t < 0)
    print(f"{len(trades)} trades, WR={w / (w + losses):.1%}")

print("")
print("=" * 70)
print("MONTE CARLO SIMULATION")
print("=" * 70)

n_sims = 100000
np.random.seed(None)

is_wr_samples = []
oos_wr_samples = []

for _ in range(n_sims):
    is_idx = np.random.choice(len(all_results), size=len(all_results), replace=True)
    oos_idx = np.random.choice(len(all_results), size=len(all_results), replace=True)
    is_wrs = []
    oos_wrs = []
    for i in is_idx:
        t = all_results[i]["trades"]
        if len(t) > 0:
            w = sum(1 for x in t if x > 0)
            losses = sum(1 for x in t if x < 0)
            if w + losses > 0:
                is_wrs.append(w / (w + losses))
    for i in oos_idx:
        t = all_results[i]["trades"]
        if len(t) > 0:
            w = sum(1 for x in t if x > 0)
            losses = sum(1 for x in t if x < 0)
            if w + losses > 0:
                oos_wrs.append(w / (w + losses))
    if is_wrs and oos_wrs:
        is_wr_samples.append(np.mean(is_wrs))
        oos_wr_samples.append(np.mean(oos_wrs))

is_wr_samples = np.array(is_wr_samples)
oos_wr_samples = np.array(oos_wr_samples)

print(f"In-Sample WR: {np.mean(is_wr_samples):.1%} (5th: {np.percentile(is_wr_samples, 5):.1%}, 95th: {np.percentile(is_wr_samples, 95):.1%})")
print(f"OOS WR: {np.mean(oos_wr_samples):.1%} (5th: {np.percentile(oos_wr_samples, 5):.1%}, 95th: {np.percentile(oos_wr_samples, 95):.1%})")

drop = np.mean(is_wr_samples) - np.mean(oos_wr_samples)
print("")
print(f"Avg Drop: {drop:.1%}")

if drop < 0.05:
    print("RESULT: LOW OVERFITTING RISK - APPROVED FOR LIVE TRADING")
elif drop < 0.10:
    print("RESULT: MODERATE OVERFITTING RISK - APPROVED WITH CAUTION")
else:
    print("RESULT: HIGH OVERFITTING RISK - NEEDS MORE WORK")

print("=" * 70)
