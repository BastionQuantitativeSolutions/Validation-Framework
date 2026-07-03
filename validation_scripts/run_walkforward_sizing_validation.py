"""
Out-of-Sample Walk-Forward Validation of Position Sizing Parameters
===================================================================

This script performs a formal walk-forward validation of the sigmoid position sizing
parameters (k=10, x_0=0.55 vs k=30, x_0=0.56 vs grid search optimal parameters)
using the historical out-of-sample trades in results/trade_results_log.json.

It ensures that the position sizing curve is not overfit and remains robust out-of-sample.
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Sigmoid function and base sizing formula
def sigmoid(c: float, k: float, x0: float) -> float:
    return 1.0 / (1.0 + np.exp(-k * (c - x0)))

def calculate_multiplier(c: float, k: float, x0: float) -> float:
    # Scale from 0.10 to 2.50 to represent actual size scaling
    return 0.10 + 2.40 * sigmoid(c, k, x0)

def evaluate_performance(trades: List[Dict[str, Any]], k: float, x0: float) -> Dict[str, Any]:
    """Calculate sizing performance metrics on a list of trades."""
    if not trades:
        return {
            "total_trades": 0, "profit_factor": 1.0, "expectancy": 0.0,
            "sharpe": 0.0, "total_return_r": 0.0, "win_rate": 0.0
        }
    
    sized_returns = []
    wins = []
    losses = []
    
    for t in trades:
        confidence = t["confidence"]
        pnl_r = t["pnl_r"]
        mult = calculate_multiplier(confidence, k, x0)
        sized_r = pnl_r * mult
        sized_returns.append(sized_r)
        
        if sized_r > 0:
            wins.append(sized_r)
        elif sized_r < 0:
            losses.append(sized_r)
            
    sized_returns = np.array(sized_returns)
    total_trades = len(trades)
    win_rate = len([r for r in sized_returns if r > 0]) / total_trades if total_trades > 0 else 0.0
    
    total_return_r = float(np.sum(sized_returns))
    expectancy = float(np.mean(sized_returns)) if total_trades > 0 else 0.0
    
    # Simple Sharpe approximation (average return / std return * sqrt(252))
    std_ret = np.std(sized_returns)
    sharpe = float(np.mean(sized_returns) / (std_ret + 1e-9) * np.sqrt(252)) if std_ret > 0 else 0.0
    
    # Profit factor: sum of wins / abs(sum of losses)
    pf = float(np.sum(wins) / (abs(np.sum(losses)) + 1e-9)) if losses else float("inf")
    
    return {
        "total_trades": total_trades,
        "profit_factor": pf,
        "expectancy": expectancy,
        "sharpe": sharpe,
        "total_return_r": total_return_r,
        "win_rate": win_rate
    }

def main():
    print("==============================================================")
    print("   Out-of-Sample Walk-Forward Validation of Sizing Sigmoid    ")
    print("==============================================================")
    
    log_path = Path("C:/Users/jack/Cavalier/CORE_MODULES/results/trade_results_log.json")
    if not log_path.exists():
        print(f"[FAIL] trade_results_log.json not found at {log_path}")
        return
        
    with open(log_path, "r") as f:
        data = json.load(f)
        
    raw_trades = data.get("trades", [])
    print(f"Loaded {len(raw_trades)} trades from log.")
    
    # Extract and parse trade details
    parsed_trades = []
    for t in raw_trades:
        # Get confidence score
        sig = t.get("signal_id")
        confidence = 0.55  # default fallback
        if isinstance(sig, dict):
            confidence = sig.get("confidence") or sig.get("fused") or t.get("confidence") or 0.55
        else:
            confidence = t.get("confidence") or 0.55
            
        entry_price = t.get("entry_price") or 0.0
        exit_price = t.get("exit_price") or 0.0
        sl_price = t.get("sl_price") or 0.0
        direction = t.get("direction", "BUY")
        
        # Calculate R-multiple
        sl_dist = abs(entry_price - sl_price)
        if sl_dist <= 1e-9:
            continue
            
        is_buy = direction in ["BUY", 1, "1"]
        pnl_r = (exit_price - entry_price) / sl_dist if is_buy else (entry_price - exit_price) / sl_dist
        pnl_r = float(np.clip(pnl_r, -1.0, 3.0))
        
        entry_time_str = t.get("entry_time") or ""
        try:
            entry_time = datetime.fromisoformat(entry_time_str.replace("Z", "+00:00"))
        except Exception:
            entry_time = datetime.now()
            
        parsed_trades.append({
            "ticket_id": t.get("ticket_id"),
            "symbol": t.get("symbol"),
            "direction": direction,
            "confidence": confidence,
            "entry_time": entry_time,
            "pnl_r": pnl_r
        })
        
    # Sort chronologically
    parsed_trades = sorted(parsed_trades, key=lambda x: x["entry_time"])
    N = len(parsed_trades)
    print(f"Parsed {N} valid trades for walk-forward validation.")
    
    if N < 50:
        print("[FAIL] Insufficient trades for a reliable walk-forward validation.")
        return
        
    # Define rolling count-based folds (4 folds)
    # Fold 1: Train = [0 to 40%], Test = [40% to 55%]
    # Fold 2: Train = [15% to 55%], Test = [55% to 70%]
    # Fold 3: Train = [30% to 70%], Test = [70% to 85%]
    # Fold 4: Train = [45% to 85%], Test = [85% to 100%]
    
    fold_ranges = [
        {"train": (0.0, 0.40), "test": (0.40, 0.55)},
        {"train": (0.15, 0.55), "test": (0.55, 0.70)},
        {"train": (0.30, 0.70), "test": (0.70, 0.85)},
        {"train": (0.45, 0.85), "test": (0.85, 1.0)}
    ]
    
    # Optimization Grid
    k_grid = [5.0, 8.0, 10.0, 15.0, 20.0, 30.0]
    x0_grid = [0.45, 0.50, 0.55, 0.58, 0.60, 0.65]
    
    fold_results = []
    
    for idx, fold in enumerate(fold_ranges):
        train_start, train_end = int(N * fold["train"][0]), int(N * fold["train"][1])
        test_start, test_end = int(N * fold["test"][0]), int(N * fold["test"][1])
        
        train_trades = parsed_trades[train_start:train_end]
        test_trades = parsed_trades[test_start:test_end]
        
        print(f"\nFold {idx+1}:")
        print(f"  Train Set Size: {len(train_trades)} ({fold['train'][0]:.0%} to {fold['train'][1]:.0%})")
        print(f"  Test Set Size:  {len(test_trades)} ({fold['test'][0]:.0%} to {fold['test'][1]:.0%})")
        
        # Grid Search on Train Set
        best_pf = -1.0
        best_params = (10.0, 0.55)
        
        for k in k_grid:
            for x0 in x0_grid:
                metrics = evaluate_performance(train_trades, k, x0)
                pf = metrics["profit_factor"]
                # We want to maximize Profit Factor (as a robust risk-reward metric)
                if pf > best_pf and pf != float("inf"):
                    best_pf = pf
                    best_params = (k, x0)
                    
        print(f"  Optimal Parameters Found: k={best_params[0]:.1f}, x0={best_params[1]:.2f} (Train PF: {best_pf:.3f})")
        
        # Evaluate on Test Set (Out-of-Sample)
        baseline_equal = evaluate_performance(test_trades, 0.0, 0.5) # Flat baseline multiplier = 1.3
        default_10_55 = evaluate_performance(test_trades, 10.0, 0.55)
        default_30_56 = evaluate_performance(test_trades, 30.0, 0.56)
        opt_results = evaluate_performance(test_trades, best_params[0], best_params[1])
        
        print(f"  OOS Performance (Baseline - Equal Size): PF={baseline_equal['profit_factor']:.3f} | Expectancy={baseline_equal['expectancy']:.3f}R | Total R={baseline_equal['total_return_r']:.1f}")
        print(f"  OOS Performance (Default k=10, x0=0.55): PF={default_10_55['profit_factor']:.3f} | Expectancy={default_10_55['expectancy']:.3f}R | Total R={default_10_55['total_return_r']:.1f}")
        print(f"  OOS Performance (Default k=30, x0=0.56): PF={default_30_56['profit_factor']:.3f} | Expectancy={default_30_56['expectancy']:.3f}R | Total R={default_30_56['total_return_r']:.1f}")
        print(f"  OOS Performance (Walk-Forward Optimal):  PF={opt_results['profit_factor']:.3f} | Expectancy={opt_results['expectancy']:.3f}R | Total R={opt_results['total_return_r']:.1f}")
        
        fold_results.append({
            "fold": idx + 1,
            "train_range": fold["train"],
            "test_range": fold["test"],
            "best_train_params": {"k": best_params[0], "x0": best_params[1], "pf": best_pf},
            "oos_baseline_equal": baseline_equal,
            "oos_default_10_55": default_10_55,
            "oos_default_30_56": default_30_56,
            "oos_optimal": opt_results
        })
        
    # Aggregate results across all test folds
    total_test_trades = sum(f["oos_default_10_55"]["total_trades"] for f in fold_results)
    
    agg_baseline_pf = np.mean([f["oos_baseline_equal"]["profit_factor"] for f in fold_results if f["oos_baseline_equal"]["profit_factor"] != float("inf")])
    agg_baseline_exp = np.mean([f["oos_baseline_equal"]["expectancy"] for f in fold_results])
    agg_baseline_r = sum(f["oos_baseline_equal"]["total_return_r"] for f in fold_results)
    
    agg_default_10_55_pf = np.mean([f["oos_default_10_55"]["profit_factor"] for f in fold_results if f["oos_default_10_55"]["profit_factor"] != float("inf")])
    agg_default_10_55_exp = np.mean([f["oos_default_10_55"]["expectancy"] for f in fold_results])
    agg_default_10_55_r = sum(f["oos_default_10_55"]["total_return_r"] for f in fold_results)
    
    agg_default_30_56_pf = np.mean([f["oos_default_30_56"]["profit_factor"] for f in fold_results if f["oos_default_30_56"]["profit_factor"] != float("inf")])
    agg_default_30_56_exp = np.mean([f["oos_default_30_56"]["expectancy"] for f in fold_results])
    agg_default_30_56_r = sum(f["oos_default_30_56"]["total_return_r"] for f in fold_results)
    
    agg_optimal_pf = np.mean([f["oos_optimal"]["profit_factor"] for f in fold_results if f["oos_optimal"]["profit_factor"] != float("inf")])
    agg_optimal_exp = np.mean([f["oos_optimal"]["expectancy"] for f in fold_results])
    agg_optimal_r = sum(f["oos_optimal"]["total_return_r"] for f in fold_results)
    
    print("\n==============================================================")
    print("   Walk-Forward Aggregated Out-of-Sample Performance Summary   ")
    print("==============================================================")
    print(f"Total OOS Trades Evaluated: {total_test_trades}")
    print(f"Baseline (Equal sizing):     Avg PF = {agg_baseline_pf:.3f} | Avg Expectancy = {agg_baseline_exp:.3f}R | Total R = {agg_baseline_r:.1f}")
    print(f"Default (k=10, x0=0.55):     Avg PF = {agg_default_10_55_pf:.3f} | Avg Expectancy = {agg_default_10_55_exp:.3f}R | Total R = {agg_default_10_55_r:.1f}")
    print(f"Default (k=30, x0=0.56):     Avg PF = {agg_default_30_56_pf:.3f} | Avg Expectancy = {agg_default_30_56_exp:.3f}R | Total R = {agg_default_30_56_r:.1f}")
    print(f"Optimized (Walk-Forward):    Avg PF = {agg_optimal_pf:.3f} | Avg Expectancy = {agg_optimal_exp:.3f}R | Total R = {agg_optimal_r:.1f}")
    
    # Save output to JSON
    output = {
        "total_trades": N,
        "fold_results": fold_results,
        "aggregated_summary": {
            "total_oos_trades": total_test_trades,
            "baseline_equal": {"profit_factor": agg_baseline_pf, "expectancy": agg_baseline_exp, "total_return_r": agg_baseline_r},
            "default_10_55": {"profit_factor": agg_default_10_55_pf, "expectancy": agg_default_10_55_exp, "total_return_r": agg_default_10_55_r},
            "default_30_56": {"profit_factor": agg_default_30_56_pf, "expectancy": agg_default_30_56_exp, "total_return_r": agg_default_30_56_r},
            "optimized_wf": {"profit_factor": agg_optimal_pf, "expectancy": agg_optimal_exp, "total_return_r": agg_optimal_r}
        }
    }
    
    out_path = Path("C:/Users/jack/Cavalier/CORE_MODULES/results/walk_forward_sizing_validation.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[OK] Sizing validation complete. Results saved -> {out_path}")

if __name__ == "__main__":
    main()
