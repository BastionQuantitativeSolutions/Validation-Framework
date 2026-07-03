import sys
# Author: JG

sys.path.insert(0, "C:/Users/jack/Cavalier")
import pandas as pd
import numpy as np
from CORE_MODULES.core.models.loader import load_models_for_pair_tf
from CORE_MODULES.core.features.compute import build_features

print("=" * 70)
print("MONTE CARLO ANALYSIS - 100,000 SIMULATIONS")
print("=" * 70)

pairs = ["EURUSD", "GBPUSD", "XAUUSD", "USDJPY", "AUDUSD", "USDCAD"]
all_wr_is = []
all_wr_oos = []

for pair in pairs:
    df = pd.read_parquet(f"C:/Users/jack/Cavalier/DATA_MODELS/data_parquet/{pair}_M1.parquet")
    df.index = pd.to_datetime(df.index)

    models, scaler, features = load_models_for_pair_tf(pair, "M1")

    train_df = df[(df.index >= "2015-01-01") & (df.index < "2018-01-01")]
    test_df = df[(df.index >= "2022-01-01") & (df.index <= "2024-12-31")]

    for name, data in [("IS", train_df), ("OOS", test_df)]:
        np.random.seed(42)
        sample = data.sample(n=min(3000, len(data)))

        try:
            feat = build_features(sample)
        except Exception:
            continue

        valid = [f for f in features if f in feat.columns]
        if len(valid) < 80:
            continue

        X = feat[valid].iloc[:-1].copy()
        Xs = scaler.transform(X)

        trades = []
        for i in range(len(Xs)):
            ps = []
            for n in ["cat", "lgb", "xgb"]:
                ml = models.get(n)
                if ml and len(ml) > 0:
                    m = ml[0]
                    try:
                        ps.append(m.predict_proba(Xs[[i]])[0][1])
                    except Exception:
                        ps.append(0.5)
            p = np.mean(ps) if ps else 0.5

            if p > 0.55:
                d = 1
            elif p < 0.45:
                d = -1
            else:
                continue

            entry = sample["close"].iloc[i]
            sl = entry * (1 - d * 0.0015)

            for j in range(i + 1, min(i + 30, len(sample))):
                high, low = sample["high"].iloc[j], sample["low"].iloc[j]
                if d == 1:
                    if low <= sl:
                        trades.append(-1)
                        break
                    elif high >= entry * 1.0010:
                        trades.append(1)
                        break
                else:
                    if high >= sl:
                        trades.append(-1)
                        break
                    elif low <= entry * 0.9990:
                        trades.append(1)
                        break
            else:
                trades.append(0)

        if len(trades) > 10:
            w = sum(1 for t in trades if t > 0)
            losses = sum(1 for t in trades if t < 0)
            wr = w / (w + losses)
            pnl = sum(trades)

            if name == "IS":
                all_wr_is.append({"pair": pair, "wr": wr, "pnl": pnl, "trades": len(trades)})
            else:
                all_wr_oos.append({"pair": pair, "wr": wr, "pnl": pnl, "trades": len(trades)})

print(f"Data loaded: {len(all_wr_is)} IS samples, {len(all_wr_oos)} OOS samples")

print("Running 100,000 bootstrap simulations...")

np.random.seed(42)
n_sims = 100000

is_wr_samples = []
oos_wr_samples = []
is_pnl_samples = []
oos_pnl_samples = []

for _ in range(n_sims):
    is_idx = np.random.choice(len(all_wr_is), size=len(all_wr_is), replace=True)
    oos_idx = np.random.choice(len(all_wr_oos), size=len(all_wr_oos), replace=True)

    is_wrs = [all_wr_is[i]["wr"] for i in is_idx]
    oos_wrs = [all_wr_oos[i]["wr"] for i in oos_idx]
    is_pnls = [all_wr_is[i]["pnl"] for i in is_idx]
    oos_pnls = [all_wr_oos[i]["pnl"] for i in oos_idx]

    is_wr_samples.append(np.mean(is_wrs))
    oos_wr_samples.append(np.mean(oos_wrs))
    is_pnl_samples.append(np.sum(is_pnls))
    oos_pnl_samples.append(np.sum(oos_pnls))

is_wr_samples = np.array(is_wr_samples)
oos_wr_samples = np.array(oos_wr_samples)
is_pnl_samples = np.array(is_pnl_samples)
oos_pnl_samples = np.array(oos_pnl_samples)

print("=" * 70)
print("MONTE CARLO RESULTS - 100,000 SIMULATIONS")
print("=" * 70)

print("\nIN-SAMPLE WIN RATE:")
print(f"  Mean: {np.mean(is_wr_samples):.4f} ({np.mean(is_wr_samples) * 100:.2f}%)")
print(f"  5th P: {np.percentile(is_wr_samples, 5):.4f} ({np.percentile(is_wr_samples, 5) * 100:.2f}%)")
print(f"  95th P: {np.percentile(is_wr_samples, 95):.4f} ({np.percentile(is_wr_samples, 95) * 100:.2f}%)")

print("\nOUT-OF-SAMPLE WIN RATE:")
print(f"  Mean: {np.mean(oos_wr_samples):.4f} ({np.mean(oos_wr_samples) * 100:.2f}%)")
print(f"  5th P: {np.percentile(oos_wr_samples, 5):.4f} ({np.percentile(oos_wr_samples, 5) * 100:.2f}%)")
print(f"  95th P: {np.percentile(oos_wr_samples, 95):.4f} ({np.percentile(oos_wr_samples, 95) * 100:.2f}%)")

print("\nOOS PnL (R):")
print(f"  Mean: {np.mean(oos_pnl_samples):.0f}")
print(f"  5th P: {np.percentile(oos_pnl_samples, 5):.0f}")
print(f"  95th P: {np.percentile(oos_pnl_samples, 95):.0f}")

print("=" * 70)
print("OVERFITTING ANALYSIS")
print("=" * 70)

wr_drops = is_wr_samples - oos_wr_samples
pnl_drops = is_pnl_samples - oos_pnl_samples

print("\nWin Rate Drop (IS - OOS):")
print(f"  Mean: {np.mean(wr_drops) * 100:.2f}%")
print(f"  5th P: {np.percentile(wr_drops, 5) * 100:.2f}%")
print(f"  95th P: {np.percentile(wr_drops, 95) * 100:.2f}%")

print("\nRisk Metrics:")
prob_below_50 = np.mean(oos_wr_samples < 0.50)
prob_loss = np.mean(oos_pnl_samples < 0)
print(f"  P(OOS WR < 50%): {prob_below_50:.4f} ({prob_below_50 * 100:.2f}%)")
print(f"  P(OOS Loss): {prob_loss:.4f} ({prob_loss * 100:.2f}%)")

print("=" * 70)
print("FINAL VERDICT")
print("=" * 70)

avg_drop = np.mean(wr_drops)
worst_drop = np.percentile(wr_drops, 95)
oos_median = np.median(oos_wr_samples)

print(f"\nAverage Drop: {avg_drop * 100:.2f}%")
print(f"Worst 95th P Drop: {worst_drop * 100:.2f}%")
print(f"OOS Median WR: {oos_median * 100:.2f}%")

if avg_drop < 0.05 and worst_drop < 0.10 and oos_median > 0.55:
    print("\nVerdict: LOW OVERFITTING RISK")
    print("Action: APPROVED FOR LIVE TRADING")
    print("Sizing: Standard (1R per trade)")
elif avg_drop < 0.08 and worst_drop < 0.15 and oos_median > 0.50:
    print("\nVerdict: MODERATE OVERFITTING RISK")
    print("Action: APPROVED WITH CAUTION")
    print("Sizing: Reduced (50-75%)")
else:
    print("\nVerdict: HIGH OVERFITTING RISK")
    print("Action: NOT RECOMMENDED FOR LIVE TRADING")

print("=" * 70)
