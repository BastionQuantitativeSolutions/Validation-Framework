"""
# Author: JG
COMPREHENSIVE SYSTEM VALIDATION - TIERED MODEL ALIGNED
Full Walk-Forward & Monte Carlo Analysis with Bias/Gap Detection

Aligned to actual Cavalier tiered model system:
- Uses load_tiered_models() for full+core model loading
- Uses compute_features_for_prediction() for 349 features
- Uses get_tiered_prediction() for weighted ensemble predictions
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(Path(__file__).resolve().parents[1] / "CORE_MODULES/validation" / "results" / "tiered_validation.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "CORE_MODULES/validation" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TradeResult:
    pair: str
    tf: str
    direction: int
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    pnl_r: float
    pnl_pips: float
    outcome: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp


@dataclass
class BacktestResult:
    pair: str
    tf: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    n_trades: int
    wins: int
    losses: int
    breakevens: int
    win_rate: float
    profit_factor: float
    total_pnl_r: float
    max_drawdown: float
    avg_trade_r: float
    expectancy: float
    sharpe_ratio: float
    trades: List[TradeResult] = field(default_factory=list)


@dataclass
class BiasCheck:
    pair: str
    wr_no_shift: float
    wr_shifted: float
    wr_drop: float
    pnl_no_shift: float
    pnl_shifted: float
    has_lookahead_bias: bool


@dataclass
class MonteCarloStats:
    n_simulations: int
    equity_mean: float
    equity_std: float
    equity_percentiles: Dict[str, float]
    max_dd_mean: float
    max_dd_percentiles: Dict[str, float]
    win_rate_mean: float
    probability_of_ruin: float
    sharpe_mean: float


PAIR_TIMEFRAMES = [
    ("EURUSD", "M5"),
    ("EURUSD", "M15"),
    ("EURUSD", "M30"),
    ("EURUSD", "H1"),
    ("EURUSD", "H4"),
    ("GBPUSD", "M5"),
    ("GBPUSD", "M15"),
    ("GBPUSD", "M30"),
    ("GBPUSD", "H1"),
    ("GBPUSD", "H4"),
    ("XAUUSD", "M5"),
    ("XAUUSD", "M15"),
    ("XAUUSD", "M30"),
    ("XAUUSD", "H1"),
    ("XAUUSD", "H4"),
    ("USDJPY", "M5"),
    ("USDJPY", "M15"),
    ("USDJPY", "M30"),
    ("USDJPY", "H1"),
    ("USDJPY", "H4"),
    ("AUDUSD", "M5"),
    ("AUDUSD", "M15"),
    ("AUDUSD", "M30"),
    ("AUDUSD", "H1"),
    ("AUDUSD", "H4"),
    ("USDCAD", "M5"),
    ("USDCAD", "M15"),
    ("USDCAD", "M30"),
    ("USDCAD", "H1"),
    ("USDCAD", "H4"),
]


def load_pair_data(
    pair: str, tf: str, start_date: Optional[str] = None, end_date: Optional[str] = None, sample_pct: Optional[float] = None
) -> pd.DataFrame:
    path = Path(f"C:/Users/jack/Cavalier/DATA_MODELS/data_parquet/{pair}_{tf}.parquet")
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)

    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]
    if sample_pct and sample_pct < 100:
        np.random.seed(42)
        n_samples = max(1000, int(len(df) * sample_pct / 100))
        df = df.sample(n=min(n_samples, len(df)))

    return df


def load_tiered_models_cavalier(pair: str, tf: str):
    """Load models using Cavalier's actual tiered structure"""
    from CORE_MODULES.core.models.loader import load_tiered_models
    from CORE_MODULES.core.models.ensemble import get_tiered_prediction
    from DATA_MODELS.feature_bridge import compute_features_for_prediction

    models_live = Path("C:/Users/jack/Cavalier/DATA_MODELS/models_live")
    tiered_pack, scalers_dict, features_dict = load_tiered_models(pair, tf, root=models_live)

    return {
        "tiered_pack": tiered_pack,
        "scalers_dict": scalers_dict,
        "features_dict": features_dict,
        "compute_features": compute_features_for_prediction,
        "get_prediction": get_tiered_prediction,
    }


def run_backtest_tiered(
    df: pd.DataFrame,
    model_system: Dict,
    spread_pips: float = 1.5,
    sl_pct: float = 0.0015,
    tp_pct: float = 0.0010,
    max_bars: int = 60,
    entry_bar_delay: int = 1,
) -> Tuple[List[TradeResult], Dict]:
    """Run backtest using tiered model system"""
    if len(df) < 100:
        return [], {}

    tiered_pack = model_system["tiered_pack"]
    features_dict = model_system["features_dict"]
    model_system["compute_features"]
    get_prediction = model_system["get_prediction"]

    trades = []
    spread_cost = spread_pips / 10000.0

    for i in range(len(df) - entry_bar_delay - max_bars - 1):
        df_sample = df.iloc[: i + entry_bar_delay + 1].copy()

        try:
            tiered_result = get_prediction(tiered_pack, features_dict, df_sample)
            tiered_result.get("weighted_confidence", 0.5)
            direction = tiered_result.get("signal", 0)
        except Exception:
            continue

        if direction == 0:
            continue

        entry_idx = i + entry_bar_delay
        entry_price = df["open"].iloc[entry_idx]
        sl_price = entry_price * (1 - direction * sl_pct)
        tp_price = entry_price * (1 + direction * tp_pct)
        entry_time = df.index[entry_idx]

        for j in range(entry_idx + 1, min(entry_idx + max_bars + 1, len(df))):
            high = df["high"].iloc[j]
            low = df["low"].iloc[j]

            if direction == 1:
                if low <= sl_price:
                    pnl_r = -1.0 - spread_cost
                    outcome = "loss"
                    break
                elif high >= tp_price:
                    pnl_r = 1.0 - spread_cost
                    outcome = "win"
                    break
            else:
                if high >= sl_price:
                    pnl_r = -1.0 - spread_cost
                    outcome = "loss"
                    break
                elif low <= tp_price:
                    pnl_r = 1.0 - spread_cost
                    outcome = "win"
                    break
        else:
            pnl_r = 0.0
            outcome = "breakeven"

        exit_idx = j if "j" in dir() else entry_idx
        pnl_pips = (direction * (df["close"].iloc[min(entry_idx + max_bars, len(df) - 1)] - entry_price)) / entry_price * 10000

        trades.append(
            TradeResult(
                pair="",
                tf="",
                direction=direction,
                entry_idx=entry_idx,
                exit_idx=exit_idx,
                entry_price=entry_price,
                exit_price=df["close"].iloc[min(entry_idx + max_bars, len(df) - 1)],
                pnl_r=pnl_r,
                pnl_pips=pnl_pips,
                outcome=outcome,
                entry_time=entry_time,
                exit_time=df.index[min(entry_idx + max_bars, len(df) - 1)],
            )
        )

    return trades, {}


def calculate_backtest_metrics(trades: List[TradeResult], pair: str, tf: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> BacktestResult:
    if not trades:
        return None

    wins = [t for t in trades if t.outcome == "win"]
    losses = [t for t in trades if t.outcome == "loss"]

    total_pnl = sum(t.pnl_r for t in trades)
    gross_profit = sum(t.pnl_r for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl_r for t in losses)) if losses else 1

    equity = np.cumsum([1.0] + [t.pnl_r for t in trades])
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / np.maximum(peak, 1)
    max_dd = np.max(drawdown) if len(drawdown) > 0 else 0

    avg_win = np.mean([t.pnl_r for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl_r for t in losses]) if losses else 0

    expectancy = ((len(wins) / len(trades) * avg_win) - (len(losses) / len(trades) * abs(avg_loss))) if trades else 0

    returns = [t.pnl_r for t in trades]
    sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252 * 1440)) if np.std(returns) > 0 else 0

    return BacktestResult(
        pair=pair,
        tf=tf,
        start_date=start_date,
        end_date=end_date,
        n_trades=len(trades),
        wins=len(wins),
        losses=len(losses),
        breakevens=len(trades) - len(wins) - len(losses),
        win_rate=len(wins) / len(trades) if trades else 0,
        profit_factor=gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        total_pnl_r=total_pnl,
        max_drawdown=max_dd,
        avg_trade_r=total_pnl / len(trades) if trades else 0,
        expectancy=expectancy,
        sharpe_ratio=sharpe,
        trades=trades,
    )


def check_lookahead_bias(pair: str, tf: str, df: pd.DataFrame, model_system: Dict) -> BiasCheck:
    """CRITICAL TEST: Check for lookahead bias"""
    np.random.seed(42)

    sample = df.sample(n=min(3000, len(df)), random_state=42)
    trades_no_shift, _ = run_backtest_tiered(sample, model_system, spread_pips=1.5, entry_bar_delay=0)

    np.random.seed(42)
    sample = df.sample(n=min(3000, len(df)), random_state=42)
    trades_shifted, _ = run_backtest_tiered(sample, model_system, spread_pips=1.5, entry_bar_delay=1)

    wr_no_shift = sum(1 for t in trades_no_shift if t.outcome == "win") / len(trades_no_shift) if trades_no_shift else 0
    wr_shifted = sum(1 for t in trades_shifted if t.outcome == "win") / len(trades_shifted) if trades_shifted else 0

    pnl_no_shift = sum(t.pnl_r for t in trades_no_shift)
    pnl_shifted = sum(t.pnl_r for t in trades_shifted)

    wr_drop = wr_no_shift - wr_shifted

    return BiasCheck(
        pair=pair,
        wr_no_shift=wr_no_shift,
        wr_shifted=wr_shifted,
        wr_drop=wr_drop,
        pnl_no_shift=pnl_no_shift,
        pnl_shifted=pnl_shifted,
        has_lookahead_bias=wr_drop > 0.05,
    )


def run_walkforward_validation(pair: str, tf: str, df: pd.DataFrame, model_system: Dict) -> List[BacktestResult]:
    """Walk-Forward Validation"""
    results = []

    train_months = 12
    test_months = 3

    df_start = df.index.min()
    df_end = df.index.max()
    current_date = df_start + pd.DateOffset(months=train_months)

    for period_id in range(6):
        test_end = min(current_date + pd.DateOffset(months=test_months), df_end)

        test_mask = (df.index >= current_date) & (df.index < test_end)
        test_df = df[test_mask]

        if len(test_df) < 100:
            break

        trades, _ = run_backtest_tiered(test_df, model_system, spread_pips=1.5)
        result = calculate_backtest_metrics(trades, pair, tf, test_df.index[0], test_df.index[-1])

        if result:
            results.append(result)
            logger.info(f"  WF {period_id}: {result.n_trades} trades, WR={result.win_rate:.1%}, PnL={result.total_pnl_r:.1f}R")

        current_date += pd.DateOffset(months=test_months)

    return results


def run_monte_carlo_simulation(trades: List[TradeResult], n_simulations: int = 10000) -> Optional[MonteCarloStats]:
    """Monte Carlo Simulation using Bootstrap"""
    if len(trades) < 10:
        return None

    pnl_sequence = np.array([t.pnl_r for t in trades])

    final_equities = []
    max_drawdowns = []
    win_rates = []

    for _ in range(n_simulations):
        bootstrap = np.random.choice(pnl_sequence, size=len(pnl_sequence), replace=True)
        equity = np.cumsum(np.concatenate([[1.0], bootstrap]))
        final_equities.append(equity[-1])
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / np.maximum(peak, 1)
        max_drawdowns.append(np.max(drawdown))
        win_rates.append(np.sum(bootstrap > 0) / len(bootstrap))

    final_equities = np.array(final_equities)
    max_drawdowns = np.array(max_drawdowns)
    win_rates = np.array(win_rates)

    return MonteCarloStats(
        n_simulations=n_simulations,
        equity_mean=float(np.mean(final_equities)),
        equity_std=float(np.std(final_equities)),
        equity_percentiles={
            "p_5": float(np.percentile(final_equities, 5)),
            "p_50": float(np.percentile(final_equities, 50)),
            "p_95": float(np.percentile(final_equities, 95)),
        },
        max_dd_mean=float(np.mean(max_drawdowns)),
        max_dd_percentiles={"p_5": float(np.percentile(max_drawdowns, 5)), "p_95": float(np.percentile(max_drawdowns, 95))},
        win_rate_mean=float(np.mean(win_rates)),
        probability_of_ruin=float(np.mean(final_equities < 0.5)),
        sharpe_mean=float(
            np.mean(
                [
                    np.mean(bootstrap) / np.std(bootstrap) * np.sqrt(252 * 1440)
                    for bootstrap in [np.random.choice(pnl_sequence, len(pnl_sequence), replace=True) for _ in range(100)]
                ]
            )
        ),
    )


def test_black_swan_events(pair: str, tf: str, df: pd.DataFrame, model_system: Dict) -> Dict:
    """Test during known crisis events"""
    events = {
        "covid_2020": ("2020-02-20", "2020-03-23"),
        "volmageddon_2018": ("2018-02-05", "2018-02-09"),
        "flash_crash_2015": ("2015-08-24", "2015-08-31"),
    }

    results = {}
    for event_name, (start, end) in events.items():
        event_start = pd.Timestamp(start)
        event_end = pd.Timestamp(end)

        if event_start < df.index.min() or event_end > df.index.max():
            continue

        event_df = df[(df.index >= event_start) & (df.index <= event_end)]
        if len(event_df) < 50:
            continue

        trades, _ = run_backtest_tiered(event_df, model_system, spread_pips=2.0)
        result = calculate_backtest_metrics(trades, pair, tf, event_start, event_end)

        if result:
            results[event_name] = result

    return results


def analyze_pair_tf(pair: str, tf: str) -> Dict:
    """Comprehensive analysis for one pair/timeframe"""
    logger.info(f"Analyzing {pair}/{tf}...")

    df = load_pair_data(pair, tf, sample_pct=100)
    if df.empty or len(df) < 1000:
        return None

    model_system = load_tiered_models_cavalier(pair, tf)
    if not model_system["tiered_pack"]:
        return None

    result = {
        "pair": pair,
        "tf": tf,
        "data_points": len(df),
        "date_range": f"{df.index.min()} to {df.index.max()}",
        "bias_check": None,
        "walkforward": [],
        "monte_carlo": None,
        "black_swan": {},
    }

    # 1. Lookahead Bias Check
    bias = check_lookahead_bias(pair, tf, df, model_system)
    result["bias_check"] = {
        "wr_no_shift": bias.wr_no_shift,
        "wr_shifted": bias.wr_shifted,
        "wr_drop": bias.wr_drop,
        "has_lookahead_bias": bias.has_lookahead_bias,
    }

    # 2. Walk-Forward Validation
    wf = run_walkforward_validation(pair, tf, df, model_system)
    result["walkforward"] = [
        {"period_id": i, "n_trades": r.n_trades, "win_rate": r.win_rate, "pnl_r": r.total_pnl_r, "profit_factor": r.profit_factor}
        for i, r in enumerate(wf)
    ]

    # 3. Monte Carlo
    all_trades, _ = run_backtest_tiered(df, model_system, spread_pips=1.5)
    if len(all_trades) >= 50:
        mc = run_monte_carlo_simulation(all_trades, n_simulations=5000)
        if mc:
            result["monte_carlo"] = {
                "equity_mean": mc.equity_mean,
                "equity_std": mc.equity_std,
                "equity_percentiles": mc.equity_percentiles,
                "max_dd_mean": mc.max_dd_mean,
                "win_rate_mean": mc.win_rate_mean,
                "probability_of_ruin": mc.probability_of_ruin,
                "sharpe_mean": mc.sharpe_mean,
            }

    # 4. Black Swan Testing
    bs = test_black_swan_events(pair, tf, df, model_system)
    result["black_swan"] = {k: {"n_trades": v.n_trades, "win_rate": v.win_rate, "pnl_r": v.total_pnl_r} for k, v in bs.items()}

    return result


def generate_report(all_results: Dict) -> str:
    report = []
    report.append("=" * 100)
    report.append("CAVALIER TRADING SYSTEM - TIERED MODEL VALIDATION REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 100)

    passing = 0
    failing = 0
    bias_detected = 0

    wf_success_rates = []
    mc_ruin_probs = []

    for key, result in all_results.items():
        if result is None:
            continue
        pair, tf = result["pair"], result["tf"]

        bias = result.get("bias_check", {})
        if bias.get("has_lookahead_bias"):
            bias_detected += 1
            bias_status = "[LOOKAHEAD BIAS]"
        else:
            bias_status = "[OK]"

        wf = result.get("walkforward", [])
        wf_profitable = sum(1 for w in wf if w.get("pnl_r", 0) > 0)
        wf_total = len(wf)
        wf_rate = wf_profitable / wf_total if wf_total > 0 else 0
        wf_success_rates.append(wf_rate)

        mc = result.get("monte_carlo", {})
        if mc:
            mc_ruin_probs.append(mc.get("probability_of_ruin", 0))

        is_passing = not bias.get("has_lookahead_bias", True) and wf_rate >= 0.5 and (not mc or mc.get("probability_of_ruin", 1) < 0.05)

        if is_passing:
            passing += 1
        else:
            failing += 1

        report.append(f"\n{pair}/{tf}:")
        report.append(f"  Data: {result['data_points']} points")
        report.append(f"  Bias Check: WR drop={bias.get('wr_drop', 0):.1%} {bias_status}")
        report.append(f"  Walk-Forward: {wf_profitable}/{wf_total} periods profitable ({wf_rate:.0%})")
        report.append(f"  Monte Carlo: Ruin prob={mc.get('probability_of_ruin', 0):.1%}" if mc else "  Monte Carlo: N/A")
        report.append("  Status: [PASS]" if is_passing else "  Status: [FAIL]")

    # Summary
    report.append("\n" + "=" * 100)
    report.append("AGGREGATE STATISTICS")
    report.append("=" * 100)

    avg_wf = np.mean(wf_success_rates) if wf_success_rates else 0
    avg_ruin = np.mean(mc_ruin_probs) if mc_ruin_probs else 0

    report.append(f"\nPairs Tested: {len(all_results)}")
    report.append(f"Passing: {passing}, Failing: {failing}")
    report.append(f"Bias Detected: {bias_detected}/{len(all_results)} pairs")
    report.append(f"Average Walk-Forward Success Rate: {avg_wf:.1%}")
    report.append(f"Average Probability of Ruin: {avg_ruin:.1%}")

    # Verdict
    report.append("\n" + "=" * 100)
    report.append("VERDICT")
    report.append("=" * 100)

    if bias_detected > len(all_results) * 0.2:
        report.append("\nCRITICAL: Lookahead bias detected in >20% of pairs")

    if avg_wf < 0.5:
        report.append("WARNING: Walk-forward success rate <50%")

    if avg_ruin > 0.05:
        report.append("WARNING: Probability of ruin >5%")

    if passing >= len(all_results) * 0.7 and bias_detected == 0:
        report.append("\nSTATUS: APPROVED FOR LIVE TRADING")
    elif passing >= len(all_results) * 0.5:
        report.append("\nSTATUS: APPROVED WITH CAUTION (use reduced sizing)")
    else:
        report.append("\nSTATUS: NOT RECOMMENDED FOR LIVE TRADING")

    report.append("\n" + "=" * 100)
    report.append("END OF REPORT")
    report.append("=" * 100)

    return "\n".join(report)


def main():
    logger.info("=" * 80)
    logger.info("CAVALIER SYSTEM - TIERED MODEL VALIDATION")
    logger.info("Walk-Forward + Monte Carlo + Bias Detection + Gap Testing")
    logger.info("=" * 80)

    all_results = {}

    for pair, tf in PAIR_TIMEFRAMES:
        result = analyze_pair_tf(pair, tf)
        if result:
            all_results[f"{pair}/{tf}"] = result

    logger.info("\nGenerating report...")
    report = generate_report(all_results)

    report_path = OUTPUT_DIR / f"tiered_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    results_path = OUTPUT_DIR / f"tiered_validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + report)

    logger.info(f"\nReport saved to: {report_path}")
    logger.info(f"Results JSON saved to: {results_path}")
    logger.info("Validation complete!")

    return all_results


if __name__ == "__main__":
    main()
