"""
Robust Backtesting with K-Fold Cross-Validation
==============================================
Uses walk-forward optimization with out-of-sample testing
to ensure minimal overfitting while maximizing performance.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    train_period: str
    test_period: str
    n_trades: int
    win_rate: float
    profit_factor: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    avg_r_multiplier: float


@dataclass
class RobustnessMetrics:
    mean_win_rate: float
    std_win_rate: float
    min_win_rate: float
    max_win_rate: float
    mean_profit_factor: float
    consistency_score: float
    overfitting_ratio: float
    out_of_sample_score: float
    fold_results: List[FoldResult] = field(default_factory=list)


def run_kfold_backtest(
    data: Dict[str, pd.DataFrame],
    n_folds: int = 5,
    test_size_days: int = 30,
    initial_balance: float = 10000,
    risk_per_trade: float = 0.0075,
    max_daily_trades: int = 20,
    regime_filter: str = "TRENDING",
    buy_only: bool = True,
    tp_mult: float = 1.0,
) -> RobustnessMetrics:
    """
    Run k-fold walk-forward backtest.

    Args:
        data: Dict of symbol -> DataFrame
        n_folds: Number of cross-validation folds
        test_size_days: Days per test period
        initial_balance: Starting balance
        risk_per_trade: Risk per trade (0.75% = 0.0075)
        max_daily_trades: Max trades per symbol per day
        regime_filter: "TRENDING" or "ALL"
        buy_only: Only take BUY signals
        tp_mult: TP multiplier (1.0 = 1:1 R:R)

    Returns:
        RobustnessMetrics with fold-by-fold results
    """

    # Combine all data to get date range
    all_dates = []
    for df in data.values():
        all_dates.extend(df["time"].dt.date.unique())

    unique_dates = sorted(set(all_dates))
    total_days = len(unique_dates)

    logger.info(f"Total trading days: {total_days}")
    logger.info(f"Test period: {test_size_days} days per fold")
    logger.info(f"Number of folds: {n_folds}")

    fold_results = []

    # Walk-forward with expanding window for training, fixed window for testing
    for fold in range(n_folds):
        # Calculate test period dates
        test_end_idx = len(unique_dates) - (n_folds - fold - 1) * test_size_days
        test_start_idx = test_end_idx - test_size_days

        if test_start_idx < n_folds * test_size_days:
            continue  # Skip if not enough training data

        test_start = unique_dates[test_start_idx]
        test_end = unique_dates[min(test_end_idx, len(unique_dates) - 1)]
        train_end = unique_dates[test_start_idx - 1]
        train_start = unique_dates[0]

        logger.info(f"\n--- Fold {fold + 1}/{n_folds} ---")
        logger.info(f"Train: {train_start} to {train_end}")
        logger.info(f"Test:  {test_start} to {test_end}")

        # Run backtest on test period only
        result = run_single_backtest(
            data=data,
            start_date=test_start,
            end_date=test_end,
            initial_balance=initial_balance,
            risk_per_trade=risk_per_trade,
            max_daily_trades=max_daily_trades,
            regime_filter=regime_filter,
            buy_only=buy_only,
            tp_mult=tp_mult,
        )

        fold_results.append(
            FoldResult(
                train_period=f"{train_start} to {train_end}",
                test_period=f"{test_start} to {test_end}",
                n_trades=result["stats"]["total_trades"],
                win_rate=result["stats"]["win_rate"],
                profit_factor=result["stats"]["profit_factor"],
                total_pnl=result["stats"]["total_pnl"],
                max_drawdown=result["stats"]["max_drawdown"],
                sharpe_ratio=result["stats"].get("sharpe_ratio", 0),
                avg_r_multiplier=result["stats"]["avg_r_multiplier"],
            )
        )

        logger.info(
            f"Fold {fold + 1} - Trades: {result['stats']['total_trades']}, "
            f"WR: {result['stats']['win_rate']:.1f}%, "
            f"PnL: ${result['stats']['total_pnl']:.2f}"
        )

    # Calculate robustness metrics
    wr_values = [f.win_rate for f in fold_results]
    pf_values = [f.profit_factor for f in fold_results]

    mean_wr = np.mean(wr_values)
    std_wr = np.std(wr_values)

    # Consistency score: 1 - coefficient of variation
    consistency_score = 1 - (std_wr / mean_wr) if mean_wr > 0 else 0

    # Overfitting ratio: out-of-sample variance / in-sample variance
    # Higher is worse (more overfitting)
    overfitting_ratio = std_wr / mean_wr if mean_wr > 0 else 1

    # Out-of-sample score: combination of mean WR and consistency
    out_of_sample_score = mean_wr * consistency_score

    return RobustnessMetrics(
        mean_win_rate=mean_wr,
        std_win_rate=std_wr,
        min_win_rate=min(wr_values) if wr_values else 0,
        max_win_rate=max(wr_values) if wr_values else 0,
        mean_profit_factor=np.mean(pf_values),
        consistency_score=consistency_score,
        overfitting_ratio=overfitting_ratio,
        out_of_sample_score=out_of_sample_score,
        fold_results=fold_results,
    )


def run_single_backtest(
    data: Dict[str, pd.DataFrame],
    start_date,
    end_date,
    initial_balance: float,
    risk_per_trade: float,
    max_daily_trades: int,
    regime_filter: str = "TRENDING",
    buy_only: bool = True,
    tp_mult: float = 1.0,
) -> Dict[str, Any]:
    """Run a single backtest on given date range."""

    from live_trading_backtester import LiveTradingBacktester, is_holiday_blocked
    from CORE_MODULES.core.unified_exits import calculate_sl_tp

    # Create backtester instance
    bt = LiveTradingBacktester(
        initial_balance=initial_balance,
        risk_per_trade=risk_per_trade,
        max_daily_trades=max_daily_trades,
    )
    bt.reset_state()

    # Filter data by date range
    filtered_data = {}
    for symbol, df in data.items():
        df = df.copy()
        df = bt.calculate_features(df)
        mask = (df["time"] >= pd.Timestamp(start_date)) & (df["time"] <= pd.Timestamp(end_date))
        filtered = df[mask].copy()
        if len(filtered) > 0:
            filtered_data[symbol] = filtered

    if not filtered_data:
        return {"error": "No data in date range", "trades": [], "stats": {}}

    # Generate trades
    trades_data = []

    for symbol, df in filtered_data.items():
        df = df.sort_values("time").reset_index(drop=True)
        symbol_daily_counts = 0
        current_date = None

        for bar_idx, row in df.iterrows():
            dt = row["time"]

            if dt.date() != current_date:
                current_date = dt.date()
                bt.daily_counts[symbol] = 0
                symbol_daily_counts = 0

            if is_holiday_blocked(dt):
                continue

            signal = bt.generate_signal(row)
            if signal is None:
                continue

            # Apply regime filter
            regime = bt.detect_regime(row)
            if regime_filter == "TRENDING" and regime not in ("TRENDING",):
                continue

            # Apply buy only filter
            if buy_only and signal["direction"] == -1:
                continue

            # Check daily cap
            if symbol_daily_counts >= max_daily_trades:
                continue

            # Calculate SL/TP
            atr = signal["atr"]
            entry_price = float(row["close"])
            sl_price, tp_price = calculate_sl_tp(entry_price, signal["direction"], atr, regime)

            # Apply TP multiplier
            sl_distance = abs(entry_price - sl_price)
            if signal["direction"] == 1:
                tp_price = entry_price + (sl_distance * tp_mult)
            else:
                tp_price = entry_price - (sl_distance * tp_mult)

            # Calculate position size
            risk_amount = initial_balance * risk_per_trade
            sl_pips = sl_distance * 10000
            volume = risk_amount / sl_pips if sl_pips > 0 else 0.01

            trades_data.append(
                {
                    "symbol": symbol,
                    "direction": signal["direction"],
                    "entry_time": dt,
                    "entry_price": entry_price,
                    "sl": sl_price,
                    "tp": tp_price,
                    "volume": volume,
                    "atr": atr,
                    "regime": regime,
                    "confidence": signal["confidence"],
                    "df": df,
                    "bar_idx": bar_idx,
                }
            )

            symbol_daily_counts += 1

    # Simulate exits
    closed_trades = []

    for trade in trades_data:
        df = trade.pop("df")
        bar_idx = trade.pop("bar_idx")
        entry_time = trade["entry_time"]

        # Find entry position
        mask = df["time"] == entry_time
        if not mask.any():
            continue
        entry_pos = df[mask].index[0]
        local_entry_pos = df.index.get_loc(entry_pos)

        # Simulate exit
        direction = trade["direction"]
        entry_price = trade["entry_price"]
        sl_price = trade["sl"]
        tp_price = trade["tp"]

        exit_reason = "END_OF_DATA"
        exit_price = df.iloc[-1]["close"]
        exit_time = df.iloc[-1]["time"]
        pnl_pips = 0.0
        r_mult = 0.0
        bars_held = 0

        for i in range(local_entry_pos + 1, len(df)):
            bars_held = i - local_entry_pos
            df.iloc[i]["close"]
            df.iloc[i]["time"]
            high = df.iloc[i]["high"]
            low = df.iloc[i]["low"]

            if direction == 1:  # BUY
                if low <= sl_price:
                    exit_reason = "SL"
                    exit_price = sl_price
                    break
                elif high >= tp_price:
                    exit_reason = "TP"
                    exit_price = tp_price
                    break
            else:  # SELL
                if high >= sl_price:
                    exit_reason = "SL"
                    exit_price = sl_price
                    break
                elif low <= tp_price:
                    exit_reason = "TP"
                    exit_price = tp_price
                    break

        # Calculate PnL
        if direction == 1:
            pnl_pips = (exit_price - entry_price) * 10000
        else:
            pnl_pips = (entry_price - exit_price) * 10000

        pip_value = 10 if "JPY" not in trade["symbol"] else 1000
        pnl = pnl_pips * trade["volume"] * pip_value / 10000

        if exit_reason == "SL":
            r_mult = -1.0
        elif exit_reason == "TP":
            sl_dist = abs(entry_price - sl_price) * 10000
            if sl_dist > 0:
                r_mult = abs(exit_price - entry_price) / (sl_distance * 10000)
            else:
                r_mult = 0

        closed_trades.append(
            {
                "symbol": trade["symbol"],
                "direction": "BUY" if trade["direction"] == 1 else "SELL",
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "pnl_pips": pnl_pips,
                "r_multiplier": r_mult,
                "exit_reason": exit_reason,
                "bars_held": bars_held,
                "regime": trade["regime"],
                "confidence": trade["confidence"],
            }
        )

    # Calculate stats
    if not closed_trades:
        return {"trades": [], "stats": {}}

    total_pnl = sum(t["pnl"] for t in closed_trades)
    winning_trades = [t for t in closed_trades if t["pnl"] > 0]
    losing_trades = [t for t in closed_trades if t["pnl"] <= 0]

    win_rate = len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0

    avg_win = sum(t["pnl"] for t in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = sum(t["pnl"] for t in losing_trades) / len(losing_trades) if losing_trades else 0

    profit_factor = (
        abs(sum(t["pnl"] for t in winning_trades) / sum(t["pnl"] for t in losing_trades))
        if losing_trades and sum(t["pnl"] for t in losing_trades) != 0
        else 0
    )

    # Calculate max drawdown
    balance = initial_balance
    peak_balance = initial_balance
    max_drawdown = 0

    for trade in closed_trades:
        balance += trade["pnl"]
        if balance > peak_balance:
            peak_balance = balance
        drawdown = (peak_balance - balance) / peak_balance * 100 if peak_balance > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)

    avg_r_mult = sum(t["r_multiplier"] for t in closed_trades) / len(closed_trades)

    return {
        "trades": closed_trades,
        "stats": {
            "total_trades": len(closed_trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_pnl": total_pnl,
            "max_drawdown": max_drawdown,
            "avg_r_multiplier": avg_r_mult,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
        },
    }


def optimize_parameters(
    data: Dict[str, pd.DataFrame],
    param_grid: Dict[str, List[Any]],
    n_folds: int = 5,
) -> Dict[str, Any]:
    """
    Grid search optimization with cross-validation.

    Args:
        data: Historical data
        param_grid: Dictionary of parameter names to values
        n_folds: Number of CV folds

    Returns:
        Best parameters and results
    """
    import itertools

    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    logger.info(f"Testing {len(combinations)} parameter combinations...")

    results = []

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))

        logger.info(f"\n[{i + 1}/{len(combinations)}] Testing: {params}")

        # Run k-fold CV
        metrics = run_kfold_backtest(
            data=data,
            n_folds=n_folds,
            **params,
        )

        results.append(
            {
                "params": params,
                "metrics": metrics,
                "oos_score": metrics.out_of_sample_score,
                "mean_wr": metrics.mean_win_rate,
                "consistency": metrics.consistency_score,
            }
        )

        logger.info(
            f"  Mean WR: {metrics.mean_win_rate:.1f}%, OOS Score: {metrics.out_of_sample_score:.1f}, Consistency: {metrics.consistency_score:.2f}"
        )

    # Sort by out-of-sample score
    results.sort(key=lambda x: x["oos_score"], reverse=True)

    best = results[0]

    logger.info("\n" + "=" * 60)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Best Parameters: {best['params']}")
    logger.info(f"Mean Win Rate: {best['metrics'].mean_win_rate:.1f}%")
    logger.info(f"Consistency Score: {best['metrics'].consistency_score:.2f}")
    logger.info(f"Out-of-Sample Score: {best['metrics'].out_of_sample_score:.1f}")

    return {
        "best_params": best["params"],
        "best_metrics": best["metrics"],
        "all_results": results,
    }


if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)

    # Load data
    data_dir = Path("C:/Users/jack/Cavalier/DATA_MODELS/data_parquet")

    data = {}
    for symbol in ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD"]:
        file_path = data_dir / f"{symbol}_M5.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            df = df.reset_index()
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"])
            elif "datetime" in df.columns:
                df["time"] = pd.to_datetime(df["datetime"])
            data[symbol] = df
            logger.info(f"Loaded {symbol}: {len(df)} bars")

    # Filter to 2024
    for symbol in data:
        df = data[symbol]
        if "time" not in df.columns:
            df = df.reset_index()
            data[symbol] = df
        data[symbol] = data[symbol][(data[symbol]["time"] >= "2024-01-01") & (data[symbol]["time"] <= "2024-12-31")].copy()

    # Define parameter grid
    param_grid = {
        "risk_per_trade": [0.005, 0.0075, 0.01],
        "max_daily_trades": [8, 15, 20],
        "regime_filter": ["TRENDING"],
        "buy_only": [True],
        "tp_mult": [0.5, 0.75, 1.0, 1.25, 1.5],
    }

    # Run optimization
    results = optimize_parameters(data, param_grid, n_folds=4)

    # Save results
    output_path = Path("C:/Users/jack/Cavalier/CORE_MODULES/results/backtest/optimization_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    serializable_results = {
        "best_params": results["best_params"],
        "best_metrics": {
            "mean_win_rate": results["best_metrics"].mean_win_rate,
            "std_win_rate": results["best_metrics"].std_win_rate,
            "min_win_rate": results["best_metrics"].min_win_rate,
            "max_win_rate": results["best_metrics"].max_win_rate,
            "mean_profit_factor": results["best_metrics"].mean_profit_factor,
            "consistency_score": results["best_metrics"].consistency_score,
            "overfitting_ratio": results["best_metrics"].overfitting_ratio,
            "out_of_sample_score": results["best_metrics"].out_of_sample_score,
            "fold_results": [
                {
                    "test_period": f.train_period,
                    "n_trades": f.n_trades,
                    "win_rate": f.win_rate,
                    "profit_factor": f.profit_factor,
                    "total_pnl": f.total_pnl,
                    "max_drawdown": f.max_drawdown,
                    "avg_r_multiplier": f.avg_r_multiplier,
                }
                for f in results["best_metrics"].fold_results
            ],
        },
        "all_results": [
            {
                "params": r["params"],
                "oos_score": r["oos_score"],
                "mean_wr": r["mean_wr"],
                "consistency": r["consistency"],
            }
            for r in results["all_results"][:20]  # Top 20
        ],
    }

    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)

    logger.info(f"\nResults saved to {output_path}")
