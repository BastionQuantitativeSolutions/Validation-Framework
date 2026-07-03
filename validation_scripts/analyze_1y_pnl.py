import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

csv_path = "C:/Users/jack/Cavalier/CORE_MODULES/results/CavalierTrades_1Y.csv"
data_dir = Path("C:/Users/jack/Cavalier/DATA_MODELS/data_1y_backtest")

print("Loading 75,000 trades from CSV...")
trades = pd.read_csv(csv_path, sep=";")
trades["Time"] = pd.to_datetime(trades["Time"], format="%Y.%m.%d %H:%M:%S")

results = []
grouped = trades.groupby("Symbol")

for symbol, sym_trades in grouped:
    print(f"Analyzing {symbol}...")
    parquet_path = data_dir / f"{symbol}_M5_1Y.parquet"
    if not parquet_path.exists():
        print(f"Skipping {symbol}, missing data")
        continue

    df = pd.read_parquet(parquet_path)
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    df = df.set_index("time").sort_index()

    times = df.index.values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    trade_times = sym_trades["Time"].values
    trade_types = sym_trades["Type"].values
    trade_prices = sym_trades["Price"].values
    trade_sls = sym_trades["SL"].values
    trade_tps = sym_trades["TP"].values

    # Check max holding time up to 12 hours (144 bars)
    MAX_HOLD = 144

    for i in tqdm(range(len(sym_trades)), desc=f"{symbol}"):
        t_time = trade_times[i]
        t_type = trade_types[i]
        t_entry = trade_prices[i]
        t_sl = trade_sls[i]
        t_tp = trade_tps[i]

        idx_arr = np.searchsorted(times, t_time)
        if idx_arr >= len(times) - 1:
            continue

        start_idx = idx_arr + 1
        end_idx = min(start_idx + MAX_HOLD, len(times))

        h_slice = highs[start_idx:end_idx]
        l_slice = lows[start_idx:end_idx]

        outcome = "OPEN"
        pnl_pips = 0.0
        pip_mult = 100.0 if "JPY" in symbol else 10000.0

        if t_type == 0:  # BUY
            sl_hits = np.where(l_slice <= t_sl)[0]
            tp_hits = np.where(h_slice >= t_tp)[0]

            first_sl = sl_hits[0] if len(sl_hits) > 0 else 9999
            first_tp = tp_hits[0] if len(tp_hits) > 0 else 9999

            if first_tp < first_sl:
                outcome = "WIN"
                pnl_pips = (t_tp - t_entry) * pip_mult
            elif first_sl <= first_tp and first_sl != 9999:
                outcome = "LOSS"
                pnl_pips = (t_sl - t_entry) * pip_mult
            else:
                exit_price = closes[end_idx - 1]
                pnl_pips = (exit_price - t_entry) * pip_mult
                outcome = "TIME_EXIT"
        else:  # SELL
            sl_hits = np.where(h_slice >= t_sl)[0]
            tp_hits = np.where(l_slice <= t_tp)[0]

            first_sl = sl_hits[0] if len(sl_hits) > 0 else 9999
            first_tp = tp_hits[0] if len(tp_hits) > 0 else 9999

            if first_tp < first_sl:
                outcome = "WIN"
                pnl_pips = (t_entry - t_tp) * pip_mult
            elif first_sl <= first_tp and first_sl != 9999:
                outcome = "LOSS"
                pnl_pips = (t_entry - t_sl) * pip_mult
            else:
                exit_price = closes[end_idx - 1]
                pnl_pips = (t_entry - exit_price) * pip_mult
                outcome = "TIME_EXIT"

        # Calculate standard 1% risk standardized Dollar value return
        # Assuming R:R from probabilities and TP/SL distance

        results.append({"Time": t_time, "Symbol": symbol, "Type": "BUY" if t_type == 0 else "SELL", "Outcome": outcome, "PnL_Pips": pnl_pips})

res_df = pd.DataFrame(results)
res_df = res_df.sort_values("Time")

wins = len(res_df[res_df["Outcome"] == "WIN"])
losses = len(res_df[res_df["Outcome"] == "LOSS"])
timeouts = len(res_df[res_df["Outcome"] == "TIME_EXIT"])
win_rate = wins / len(res_df) * 100 if len(res_df) > 0 else 0
total_pips = res_df["PnL_Pips"].sum()

print("\n" + "=" * 40)
print("       1-YEAR BACKTEST SIMULATION RESULTS")
print("=" * 40)
print(f"Total Trades Evaluated : {len(res_df):,}")
print(f"Winning Trades         : {wins:,}")
print(f"Losing Trades          : {losses:,}")
print(f"Time Exits (12h limit) : {timeouts:,}")
print(f"Global Win Rate        : {win_rate:.2f}%")
print(f"Net Profit (Pips)      : {total_pips:,.1f} pips")
print("=" * 40)

res_df["Cumulative_Pips"] = res_df["PnL_Pips"].cumsum()

plt.style.use("dark_background")
plt.figure(figsize=(14, 7))
plt.plot(res_df["Time"], res_df["Cumulative_Pips"], color="#00ff88", linewidth=1.5)
plt.title("Cavalier Ensemble ML 1-Year Backtest Equity", fontsize=16, pad=15)
plt.ylabel("Cumulative Profit (Pips)", fontsize=12)
plt.xlabel("Date", fontsize=12)
plt.grid(True, alpha=0.2, color="gray")
plt.fill_between(res_df["Time"], res_df["Cumulative_Pips"], 0, alpha=0.1, color="#00ff88")

out_path = "C:/Users/jack/Cavalier/CORE_MODULES/results/1Y_Equity_Curve.png"
plt.tight_layout()
plt.savefig(out_path, dpi=150)
print(f"-> Equity Curve Chart saved to: {out_path}")
