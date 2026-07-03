import sys

sys.path.insert(0, "CORE_MODULES")
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from core.features.compute import build_features
from core.models.loader import load_tiered_models
from core.models.ensemble import align_df

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
}


def test_pair(pair, df_full, start_date, end_date):
    df = df_full[(df_full["time"] >= start_date) & (df_full["time"] <= end_date)].copy()
    df = df.sort_values("time").reset_index(drop=True)

    if len(df) < 100:
        return []

    config = INSTRUMENT_CONFIGS.get(pair, {"pip_size": 0.0001, "pip_value_per_lot": 10.0})
    pip_size = config["pip_size"]
    config["pip_value_per_lot"]

    close = df["close"]
    high = df["high"]
    low = df["low"]
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.rolling(window=14).mean()

    models_tuple = load_tiered_models(pair, "M15")
    all_models = []
    feature_cols = []

    if models_tuple and len(models_tuple) == 3:
        full_models, core_models, minimal_models = models_tuple

        if full_models and "full" in full_models:
            tier_data = full_models["full"]
            for model_key in ["cat", "lgb", "xgb"]:
                if model_key in tier_data:
                    model_list = tier_data[model_key]
                    if isinstance(model_list, list) and len(model_list) > 0:
                        all_models.append((model_key, model_list[0]))

        model_dir = Path(f"DATA_MODELS/models_live/{pair}_M15/full")
        features_file = model_dir / "features.pkl"
        if features_file.exists():
            with open(features_file, "rb") as f:
                feature_cols = pickle.load(f)

    if not all_models:
        return []

    # Collect predictions
    predictions_data = []
    for i in range(100, len(df), 8):  # Check every 8 bars for speed
        window_start = max(0, i - 100)
        window_df = df.iloc[window_start : i + 1].copy()
        features_df = build_features(window_df)

        if features_df is None or len(features_df) == 0:
            continue

        last_features = features_df.iloc[-1:]
        numeric_cols = last_features.select_dtypes(include=[np.number]).columns
        last_features_numeric = last_features[numeric_cols]

        aligned_features = align_df(last_features_numeric, feature_cols)
        aligned_features = aligned_features.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        preds = []
        for model_name, model in all_models:
            try:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(aligned_features)
                    if len(proba) > 0 and len(proba[0]) == 2:
                        preds.append(proba[0][1])
            except Exception:
                continue

        if preds:
            predictions_data.append({"idx": i, "ml_prob": np.mean(preds), "time": df.iloc[i]["time"]})

    pred_dict = {p["idx"]: p for p in predictions_data}

    trades = []

    for i in range(100, len(df), 8):
        if i not in pred_dict:
            continue

        pred_data = pred_dict[i]
        ml_prob = pred_data["ml_prob"]
        atr = df.iloc[i]["atr"]

        if pd.isna(atr) or atr == 0:
            continue

        if ml_prob >= 0.51:
            direction = 1
        elif ml_prob <= 0.49:
            direction = -1
        else:
            continue

        entry_price = df.iloc[i]["close"]
        sl_distance = atr * 2.0
        tp_distance = atr * 0.5

        if direction == 1:
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else:
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance

        sl_pips = sl_distance / pip_size
        if sl_pips == 0:
            continue

        won = False
        exit_price = None

        for j in range(i + 1, min(i + 30, len(df))):
            bar_high = df.iloc[j]["high"]
            bar_low = df.iloc[j]["low"]

            if direction == 1:
                if bar_low <= sl_price:
                    exit_price = sl_price
                    break
                if bar_high >= tp_price:
                    exit_price = tp_price
                    won = True
                    break
            else:
                if bar_high >= sl_price:
                    exit_price = sl_price
                    break
                if bar_low <= tp_price:
                    exit_price = tp_price
                    won = True
                    break

        if exit_price is None:
            continue

        if direction == 1:
            price_diff = exit_price - entry_price
        else:
            price_diff = entry_price - exit_price

        pips = price_diff / pip_size

        trades.append({"pair": pair, "entry_time": pred_data["time"], "pips": pips, "sl_pips": sl_pips, "won": won})

    return trades


parquet_dir = Path("DATA_MODELS/data_parquet")
pairs = ["EURUSD", "GBPUSD", "USDJPY"]

# Test different risk levels to find one that keeps DD under 3%
risk_levels = [0.25, 0.50, 0.75, 1.00]

print("Testing different risk levels to find optimal DD...")
print("=" * 60)

for risk_pct in risk_levels:
    all_trades = []
    balance = 10000
    peak = balance
    max_dd = 0
    daily_balance = {}

    for pair in pairs:
        data_file = parquet_dir / f"{pair}_M15.parquet"
        if not data_file.exists():
            continue

        df_full = pd.read_parquet(data_file)
        df_full = df_full.reset_index()
        if "index" in df_full.columns:
            df_full = df_full.rename(columns={"index": "time"})

        trades = test_pair(pair, df_full, "2026-02-01", "2026-02-28")

        config = INSTRUMENT_CONFIGS.get(pair, {"pip_size": 0.0001, "pip_value_per_lot": 10.0})

        for trade in trades:
            sl_pips = trade["sl_pips"]
            risk_amount = balance * (risk_pct / 100)
            position_size = risk_amount / (sl_pips * config["pip_value_per_lot"])
            position_size = max(0.01, min(position_size, 10.0))

            pnl = trade["pips"] * config["pip_value_per_lot"] * position_size
            trade["pnl"] = pnl

            balance += pnl
            if balance > peak:
                peak = balance

            dd = (peak - balance) / peak
            if dd > max_dd:
                max_dd = dd

            trade_date = str(trade["entry_time"].date())
            daily_balance[trade_date] = balance

            all_trades.append(trade)

    winning_trades = [t for t in all_trades if t["won"]]
    total_trades = len(all_trades)
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
    total_pnl = sum(t["pnl"] for t in all_trades)

    # Calculate max daily DD
    sorted_dates = sorted(daily_balance.keys())
    daily_peak = 10000
    max_daily_dd = 0
    for date in sorted_dates:
        bal = daily_balance[date]
        if bal > daily_peak:
            daily_peak = bal
        dd = (daily_peak - bal) / daily_peak
        if dd > max_daily_dd:
            max_daily_dd = dd

    status = "PASS" if max_daily_dd <= 0.03 else "FAIL"
    print(f"Risk {risk_pct:.2f}%: PnL=${total_pnl:.0f} | WR={win_rate:.1%} | Max DD={max_daily_dd:.1%} | {status}")

print("=" * 60)

# Run with optimal risk level (0.25%)
print("\\nRunning with 0.25% risk (optimal for DD control)...")

all_trades = []
balance = 10000
peak = balance
max_dd = 0
daily_balance = {}

for pair in pairs:
    data_file = parquet_dir / f"{pair}_M15.parquet"
    if not data_file.exists():
        continue

    df_full = pd.read_parquet(data_file)
    df_full = df_full.reset_index()
    if "index" in df_full.columns:
        df_full = df_full.rename(columns={"index": "time"})

    trades = test_pair(pair, df_full, "2026-02-01", "2026-02-28")

    config = INSTRUMENT_CONFIGS.get(pair, {"pip_size": 0.0001, "pip_value_per_lot": 10.0})

    for trade in trades:
        sl_pips = trade["sl_pips"]
        risk_amount = balance * 0.0025
        position_size = risk_amount / (sl_pips * config["pip_value_per_lot"])
        position_size = max(0.01, min(position_size, 10.0))

        pnl = trade["pips"] * config["pip_value_per_lot"] * position_size
        trade["pnl"] = pnl

        balance += pnl
        if balance > peak:
            peak = balance

        dd = (peak - balance) / peak
        if dd > max_dd:
            max_dd = dd

        trade_date = str(trade["entry_time"].date())
        daily_balance[trade_date] = balance

        all_trades.append(trade)

winning_trades = [t for t in all_trades if t["won"]]
losing_trades = [t for t in all_trades if not t["won"]]
total_trades = len(all_trades)
win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
total_pnl = sum(t["pnl"] for t in all_trades)

# Calculate max daily DD
sorted_dates = sorted(daily_balance.keys())
daily_peak = 10000
max_daily_dd = 0
for date in sorted_dates:
    bal = daily_balance[date]
    if bal > daily_peak:
        daily_peak = bal
    dd = (daily_peak - bal) / daily_peak
    if dd > max_daily_dd:
        max_daily_dd = dd

print(f"\\nPairs tested: {len(pairs)}")
print(f"Total trades: {total_trades}")
print(f"Winning trades: {len(winning_trades)}")
print(f"Losing trades: {len(losing_trades)}")
print(f"Win rate: {win_rate:.1%}")
print()
print("Starting balance: $10,000.00")
print(f"Final balance: ${balance:.2f}")
print(f"Total PnL: ${total_pnl:.2f}")
print(f"Return: {total_pnl / 10000 * 100:.1f}%")
print()
print(f"Max daily drawdown: {max_daily_dd:.1%}")
if max_daily_dd <= 0.03:
    print("PASS - Max daily DD within 3% limit")
else:
    print("FAIL - Max daily DD exceeds 3% limit")
