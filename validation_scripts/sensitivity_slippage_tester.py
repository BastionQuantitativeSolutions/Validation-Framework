"""
Slippage Sensitivity Analysis Template - Cavalier Trading System
Purpose: Calculate how much Sharpe Ratio and Expectancy drop with increased slippage.
"""

import numpy as np
import pandas as pd


def run_sensitivity_analysis(trades_pnl_pct, avg_years, slippage_pips_list, pip_value_pct=0.0001):
    """
    trades_pnl_pct: list/array of PNL % per trade
    avg_years: duration of the trade sequence in years
    slippage_pips_list: list of slippage increments (e.g. [0.0, 0.5, 1.0, 1.5])
    pip_value_pct: the value of 1 pip in % terms (default 0.01% or 0.0001)
    """
    results = []

    for slip in slippage_pips_list:
        # Subtract slippage from each trade (round trip)
        adjusted_pnl = np.array(trades_pnl_pct) - (slip * pip_value_pct)

        # Calculate Remediated Sharpe (Per-Trade)
        mean_pnl = np.mean(adjusted_pnl)
        std_pnl = np.std(adjusted_pnl)
        sharpe = (mean_pnl / std_pnl) * np.sqrt(len(adjusted_pnl) / avg_years) if std_pnl > 0 else 0

        # Calculate Expectancy (in pips)
        expectancy_pips = mean_pnl / pip_value_pct

        results.append(
            {
                "Slippage (Pips)": slip,
                "Expectancy (Pips)": round(expectancy_pips, 2),
                "Remediated Sharpe": round(sharpe, 2),
                "Efficiency (%)": round((mean_pnl / np.mean(trades_pnl_pct)) * 100, 1) if np.mean(trades_pnl_pct) != 0 else 0,
            }
        )

    return pd.DataFrame(results)


if __name__ == "__main__":
    # EXAMPLE DATA: 100 trades, 1 year, 0.2% avg win
    # Replace this with real trade data from walkforward_montecarlo_test.py
    mock_trades = np.random.normal(0.0020, 0.0150, 200)
    years = 1.0

    print("--- Cavalier Slippage Sensitivity Report ---")
    df = run_sensitivity_analysis(mock_trades, years, [0.0, 0.5, 1.0, 1.5, 2.0])
    print(df.to_string(index=False))

    print("\n[ACTION THRESHOLD]")
    kill_switch_row = df[df["Remediated Sharpe"] < 1.0].head(1)
    if not kill_switch_row.empty:
        slip_limit = kill_switch_row["Slippage (Pips)"].values[0]
        print(f"CRITICAL: Set ALPHA_DECAY_LOCKED = True if slippage > {slip_limit} pips")
    else:
        print("System robust up to 2.0 pips slippage.")
