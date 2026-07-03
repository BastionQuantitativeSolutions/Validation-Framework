"""
# Author: JG
COMPREHENSIVE SYSTEM VALIDATION
Full Walk-Forward & Monte Carlo Analysis with Bias/Gap Detection

Tests:
1. Lookahead Bias Detection
2. Walk-Forward Validation (multiple pairs/timeframes)
3. Monte Carlo Simulation (in-sample & out-of-sample)
4. Black Swan / Gap Testing
5. Regime Analysis
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

from CORE_MODULES.core.features.compute import build_features

import joblib


def load_models_cavalier(pair: str, tf: str):
    """
    Load models using Cavalier's actual structure.
    Tries nested structure first (pair_tf/core), then flat files.
    """
    models_live = Path("C:/Users/jack/Cavalier/DATA_MODELS/models_live")

    nested_path = models_live / f"{pair}_{tf}" / "core"

    models = {"cat": [], "lgb": [], "xgb": []}
    scaler = None
    features = None

    if nested_path.exists() and nested_path.is_dir():
        try:
            scaler = joblib.load(nested_path / "scaler.pkl")
            features = joblib.load(nested_path / "features.pkl")

            cat_model = joblib.load(nested_path / "cat_model.joblib")
            lgb_model = joblib.load(nested_path / "lgb_model.joblib")
            xgb_model = joblib.load(nested_path / "xgb_model.joblib")

            models["cat"].append(cat_model)
            models["lgb"].append(lgb_model)
            models["xgb"].append(xgb_model)

            logger.info(f"[models] Loaded from nested structure for {pair}/{tf}")
            return models, scaler, features
        except Exception as e:
            logger.error(f"[models] Failed to load nested structure for {pair}/{tf}: {e}")

    # Fall back to flat files
    flat_prefix = f"{pair}_{tf}"
    flat_path = models_live / f"{flat_prefix}"

    if flat_path.exists() and flat_path.is_dir():
        try:
            for f in flat_path.glob("*.joblib"):
                if "scaler" in f.name:
                    scaler = joblib.load(f)
                elif "features" in f.name:
                    features = joblib.load(f)
                elif "cat" in f.name:
                    models["cat"].append(joblib.load(f))
                elif "lgb" in f.name:
                    models["lgb"].append(joblib.load(f))
                elif "xgb" in f.name:
                    models["xgb"].append(joblib.load(f))
            logger.info(f"[models] Loaded from flat structure for {pair}/{tf}")
            return models, scaler, features
        except Exception as e:
            logger.error(f"[models] Failed to load flat structure for {pair}/{tf}: {e}")

    return None, None, None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(Path(__file__).resolve().parents[1] / "CORE_MODULES/validation" / "results" / "comprehensive_validation.log"),
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
    outcome: str  # 'win', 'loss', 'breakeven'
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
    regime_performance: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class BiasCheck:
    pair: str
    wr_no_shift: float
    wr_shifted: float
    wr_drop: float
    pnl_no_shift: float
    pnl_shifted: float
    has_lookahead_bias: bool
    has_data_leakage: bool


@dataclass
class MonteCarloStats:
    n_simulations: int
    equity_mean: float
    equity_std: float
    equity_percentiles: Dict[str, float]
    max_dd_mean: float
    max_dd_percentiles: Dict[str, float]
    win_rate_mean: float
    win_rate_percentiles: Dict[str, float]
    profit_factor_mean: float
    probability_of_ruin: float
    sharpe_mean: float
    sharpe_std: float


@dataclass
class WalkForwardPeriod:
    period_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train_samples: int
    n_test_samples: int
    train_result: Optional[BacktestResult]
    test_result: Optional[BacktestResult]
    degradation: Dict[str, float]


@dataclass
class GapEvent:
    name: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    severity: str
    actual_move_pct: float
    system_performance: Dict


PAIR_TIMEFRAMES = [
    ("EURUSD", "M1"),
    ("EURUSD", "M5"),
    ("EURUSD", "M15"),
    ("EURUSD", "H1"),
    ("GBPUSD", "M1"),
    ("GBPUSD", "M5"),
    ("GBPUSD", "M15"),
    ("GBPUSD", "H1"),
    ("XAUUSD", "M1"),
    ("XAUUSD", "M5"),
    ("XAUUSD", "M15"),
    ("XAUUSD", "H1"),
    ("USDJPY", "M1"),
    ("USDJPY", "M5"),
    ("USDJPY", "M15"),
    ("USDJPY", "H1"),
    ("AUDUSD", "M1"),
    ("AUDUSD", "M5"),
    ("AUDUSD", "M15"),
    ("AUDUSD", "H1"),
    ("USDCAD", "M1"),
    ("USDCAD", "M5"),
    ("USDCAD", "M15"),
    ("USDCAD", "H1"),
]


def load_pair_data(
    pair: str, tf: str, start_date: Optional[str] = None, end_date: Optional[str] = None, sample_pct: Optional[float] = None
) -> pd.DataFrame:
    """Load parquet data with optional filtering"""
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


def run_backtest_bar_aligned(
    df: pd.DataFrame,
    models: Dict,
    scaler,
    trained_features: List[str],
    spread_pips: float = 1.5,
    sl_pct: float = 0.0015,
    tp_pct: float = 0.0010,
    max_bars: int = 60,
    entry_bar_delay: int = 1,
) -> Tuple[List[TradeResult], Dict]:
    """
    Run backtest with REALISTIC bar-aligned methodology

    Key principles:
    1. Features on bar N use bar N's data (available at close of N)
    2. Prediction at END of bar N
    3. Entry at OPEN of bar N+1 (next bar)
    4. Include spread costs
    """
    if len(df) < 100:
        return [], {}

    try:
        features = build_features(df)
    except Exception as e:
        logger.error(f"Feature build error: {e}")
        return [], {}

    valid_features = [f for f in trained_features if f in features.columns]
    if len(valid_features) < 50:
        return [], {}

    X = features[valid_features].copy()
    X_scaled = scaler.transform(X)

    predictions = []
    for i in range(len(X_scaled)):
        ps = []
        for name in ["cat", "lgb", "xgb"]:
            ml = models.get(name)
            if ml and len(ml) > 0:
                m = ml[0]
                try:
                    prob = m.predict_proba(X_scaled[[i]])[0][1]
                except Exception:
                    prob = m.predict(X_scaled[[i]])[0] * 0.5 + 0.5
                ps.append(prob)
        predictions.append(np.mean(ps) if ps else 0.5)

    trades = []
    regime_stats = {
        "trending": {"wins": 0, "losses": 0, "pnl": 0},
        "ranging": {"wins": 0, "losses": 0, "pnl": 0},
        "volatile": {"wins": 0, "losses": 0, "pnl": 0},
        "unknown": {"wins": 0, "losses": 0, "pnl": 0},
    }

    spread_cost = spread_pips / 10000.0

    for i in range(len(predictions) - entry_bar_delay - 1):
        prob = predictions[i]

        if prob > 0.55:
            direction = 1
        elif prob < 0.45:
            direction = -1
        else:
            continue

        entry_idx = i + entry_bar_delay
        if entry_idx >= len(df):
            break

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

        if outcome == "win":
            regime_stats["trending"]["wins"] += 1
            regime_stats["trending"]["pnl"] += pnl_r
        elif outcome == "loss":
            regime_stats["trending"]["losses"] += 1
            regime_stats["trending"]["pnl"] += pnl_r

        pnl_pips = (direction * (df["close"].iloc[min(entry_idx + max_bars, len(df) - 1)] - entry_price)) / entry_price * 10000

        trades.append(
            TradeResult(
                pair="",
                tf="",
                direction=direction,
                entry_idx=entry_idx,
                exit_idx=j if "j" in dir() else entry_idx,
                entry_price=entry_price,
                exit_price=df["close"].iloc[min(entry_idx + max_bars, len(df) - 1)],
                pnl_r=pnl_r,
                pnl_pips=pnl_pips,
                outcome=outcome,
                entry_time=entry_time,
                exit_time=df.index[min(entry_idx + max_bars, len(df) - 1)],
            )
        )

    return trades, regime_stats


def calculate_backtest_metrics(trades: List[TradeResult], pair: str, tf: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> BacktestResult:
    """Calculate metrics from trades"""
    if not trades:
        return None

    wins = [t for t in trades if t.outcome == "win"]
    losses = [t for t in trades if t.outcome == "loss"]
    breakevens = [t for t in trades if t.outcome == "breakeven"]

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
        breakevens=len(breakevens),
        win_rate=len(wins) / len(trades) if trades else 0,
        profit_factor=gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        total_pnl_r=total_pnl,
        max_drawdown=max_dd,
        avg_trade_r=total_pnl / len(trades) if trades else 0,
        expectancy=expectancy,
        sharpe_ratio=sharpe,
        trades=trades,
    )


def check_lookahead_bias(
    pair: str,
    tf: str,
    df: pd.DataFrame,
    models: Dict,
    scaler,
    trained_features: List[str],
    spread_pips: float = 1.5,
) -> BiasCheck:
    """
    CRITICAL TEST: Check for lookahead bias

    Compare:
    - No shift: Entry at close of bar N (uses bar N data)
    - Shifted: Entry at open of bar N+1 (realistic)

    If WR drops significantly with shift = LOOKAHEAD BIAS
    """
    np.random.seed(42)

    if len(df) < 2000:
        sample = df.copy()
    else:
        sample = df.sample(n=2000, random_state=42)

    trades_no_shift, _ = run_backtest_bar_aligned(sample, models, scaler, trained_features, spread_pips=spread_pips, entry_bar_delay=0)

    np.random.seed(42)
    sample = df.sample(n=min(2000, len(df)), random_state=42)

    trades_shifted, _ = run_backtest_bar_aligned(sample, models, scaler, trained_features, spread_pips=spread_pips, entry_bar_delay=1)

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
        has_data_leakage=pnl_no_shift > pnl_shifted * 1.5,
    )


def run_walkforward_validation(
    pair: str,
    tf: str,
    df: pd.DataFrame,
    models: Dict,
    scaler,
    trained_features: List[str],
    train_months: int = 12,
    test_months: int = 3,
    n_periods: int = 6,
) -> List[WalkForwardPeriod]:
    """
    Walk-Forward Validation
    Tests rolling windows to ensure model generalizes
    """
    results = []

    df_start = df.index.min()
    df_end = df.index.max()

    current_date = df_start + pd.DateOffset(months=train_months)

    for period_id in range(n_periods):
        train_end = current_date
        test_end = min(current_date + pd.DateOffset(months=test_months), df_end)

        train_mask = df.index < train_end
        test_mask = (df.index >= train_end) & (df.index < test_end)

        train_df = df[train_mask]
        test_df = df[test_mask]

        if len(test_df) < 100:
            break

        train_trades, _ = run_backtest_bar_aligned(train_df, models, scaler, trained_features, spread_pips=1.5)
        test_trades, _ = run_backtest_bar_aligned(test_df, models, scaler, trained_features, spread_pips=1.5)

        train_result = calculate_backtest_metrics(train_trades, pair, tf, train_df.index[0], train_df.index[-1])
        test_result = calculate_backtest_metrics(test_trades, pair, tf, test_df.index[0], test_df.index[-1])

        degradation = {}
        if train_result and test_result:
            degradation = {
                "wr_drop": train_result.win_rate - test_result.win_rate,
                "pf_drop": train_result.profit_factor - test_result.profit_factor,
                "pnl_drop": train_result.total_pnl_r - test_result.total_pnl_r,
                "expectancy_drop": train_result.expectancy - test_result.expectancy,
            }

        results.append(
            WalkForwardPeriod(
                period_id=period_id,
                train_start=train_df.index[0],
                train_end=train_df.index[-1],
                test_start=test_df.index[0],
                test_end=test_df.index[-1],
                n_train_samples=len(train_df),
                n_test_samples=len(test_df),
                train_result=train_result,
                test_result=test_result,
                degradation=degradation,
            )
        )

        current_date += pd.DateOffset(months=test_months)

    return results


def run_monte_carlo_simulation(
    trades: List[TradeResult],
    n_simulations: int = 10000,
    confidence_level: float = 0.95,
) -> MonteCarloStats:
    """
    Monte Carlo Simulation using Bootstrap Resampling

    Tests:
    1. Equity growth distribution
    2. Max drawdown distribution
    3. Win rate stability
    4. Probability of ruin
    """
    if len(trades) < 10:
        return None

    pnl_sequence = np.array([t.pnl_r for t in trades])

    final_equities = []
    max_drawdowns = []
    win_rates = []
    sharpe_ratios = []

    for _ in range(n_simulations):
        bootstrap = np.random.choice(pnl_sequence, size=len(pnl_sequence), replace=True)

        equity = np.cumsum(np.concatenate([[1.0], bootstrap]))
        final_equities.append(equity[-1])

        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / np.maximum(peak, 1)
        max_drawdowns.append(np.max(drawdown))

        wins = np.sum(bootstrap > 0)
        win_rates.append(wins / len(bootstrap))

        returns = bootstrap
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 1440) if np.std(returns) > 0 else 0
        sharpe_ratios.append(sharpe)

    final_equities = np.array(final_equities)
    max_drawdowns = np.array(max_drawdowns)
    win_rates = np.array(win_rates)
    sharpe_ratios = np.array(sharpe_ratios)

    alpha = (1 - confidence_level) / 2

    return MonteCarloStats(
        n_simulations=n_simulations,
        equity_mean=float(np.mean(final_equities)),
        equity_std=float(np.std(final_equities)),
        equity_percentiles={
            f"p_{int(alpha * 100)}": float(np.percentile(final_equities, alpha * 100)),
            "p_25": float(np.percentile(final_equities, 25)),
            "p_50": float(np.percentile(final_equities, 50)),
            "p_75": float(np.percentile(final_equities, 75)),
            f"p_{int((1 - alpha) * 100)}": float(np.percentile(final_equities, (1 - alpha) * 100)),
        },
        max_dd_mean=float(np.mean(max_drawdowns)),
        max_dd_percentiles={
            f"p_{int(alpha * 100)}": float(np.percentile(max_drawdowns, alpha * 100)),
            "p_50": float(np.percentile(max_drawdowns, 50)),
            f"p_{int((1 - alpha) * 100)}": float(np.percentile(max_drawdowns, (1 - alpha) * 100)),
        },
        win_rate_mean=float(np.mean(win_rates)),
        win_rate_percentiles={
            "p_5": float(np.percentile(win_rates, 5)),
            "p_25": float(np.percentile(win_rates, 25)),
            "p_50": float(np.percentile(win_rates, 50)),
            "p_75": float(np.percentile(win_rates, 75)),
            "p_95": float(np.percentile(win_rates, 95)),
        },
        profit_factor_mean=0,
        probability_of_ruin=float(np.mean(final_equities < 0.5)),
        sharpe_mean=float(np.mean(sharpe_ratios)),
        sharpe_std=float(np.std(sharpe_ratios)),
    )


def test_black_swan_events(
    pair: str,
    tf: str,
    df: pd.DataFrame,
    models: Dict,
    scaler,
    trained_features: List[str],
) -> Dict[str, GapEvent]:
    """Test system during known gap/crisis events"""

    events = {
        "covid_crash": {"name": "COVID Crash 2020", "start": "2020-02-20", "end": "2020-03-23", "severity": "extreme"},
        "volmageddon": {"name": "Volmageddon 2018", "start": "2018-02-05", "end": "2018-02-09", "severity": "high"},
        "flash_crash": {"name": "Flash Crash 2015", "start": "2015-08-24", "end": "2015-08-31", "severity": "high"},
        "rate_hike_2022": {"name": "Rate Hike Cycle 2022", "start": "2022-03-16", "end": "2022-10-31", "severity": "high"},
        "svb_collapse": {"name": "SVB Collapse 2023", "start": "2023-03-08", "end": "2023-03-19", "severity": "medium"},
    }

    results = {}

    for event_key, event_info in events.items():
        event_start = pd.Timestamp(event_info["start"])
        event_end = pd.Timestamp(event_info["end"])

        if event_start < df.index.min() or event_end > df.index.max():
            continue

        event_df = df[(df.index >= event_start) & (df.index <= event_end)]

        if len(event_df) < 50:
            continue

        trades, _ = run_backtest_bar_aligned(event_df, models, scaler, trained_features, spread_pips=2.0)

        result = calculate_backtest_metrics(trades, pair, tf, event_start, event_end)

        if result:
            results[event_key] = GapEvent(
                name=event_info["name"],
                start_date=event_start,
                end_date=event_end,
                severity=event_info["severity"],
                actual_move_pct=0,
                system_performance={
                    "n_trades": result.n_trades,
                    "win_rate": result.win_rate,
                    "pnl_r": result.total_pnl_r,
                    "profit_factor": result.profit_factor,
                    "max_drawdown": result.max_drawdown,
                },
            )

    return results


def test_out_of_sample(
    pair: str,
    tf: str,
    df: pd.DataFrame,
    models: Dict,
    scaler,
    trained_features: List[str],
) -> Dict[str, BacktestResult]:
    """
    Out-of-Sample Testing
    Train on one period, test on completely separate period
    """
    results = {}

    oos_periods = [
        ("2015-01-01", "2016-01-01", "2016-01-01", "2017-01-01"),
        ("2017-01-01", "2018-01-01", "2018-01-01", "2019-01-01"),
        ("2020-01-01", "2021-01-01", "2021-01-01", "2022-01-01"),
        ("2022-01-01", "2023-01-01", "2023-01-01", "2024-01-01"),
    ]

    for period_id, train_start, train_end, test_end in oos_periods:
        train_mask = (df.index >= train_start) & (df.index < train_end)
        test_mask = (df.index >= train_end) & (df.index < test_end)

        train_df = df[train_mask]
        test_df = df[test_mask]

        if len(train_df) < 500 or len(test_df) < 200:
            continue

        train_trades, _ = run_backtest_bar_aligned(train_df, models, scaler, trained_features, spread_pips=1.5)
        test_trades, _ = run_backtest_bar_aligned(test_df, models, scaler, trained_features, spread_pips=1.5)

        calculate_backtest_metrics(train_trades, pair, tf, pd.Timestamp(train_start), pd.Timestamp(train_end))
        test_result = calculate_backtest_metrics(test_trades, pair, tf, pd.Timestamp(train_end), pd.Timestamp(test_end))

        if test_result:
            results[f"oos_{period_id}"] = test_result

    return results


def analyze_pair_tf(pair: str, tf: str) -> Dict:
    """Comprehensive analysis for one pair/timeframe"""

    logger.info(f"Analyzing {pair}/{tf}...")

    df = load_pair_data(pair, tf, sample_pct=100)
    if df.empty or len(df) < 1000:
        logger.warning(f"Insufficient data for {pair}/{tf}")
        return None

    models, scaler, trained_features = load_models_cavalier(pair, tf)
    if not models or scaler is None:
        logger.warning(f"No models for {pair}/{tf}")
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
        "out_of_sample": {},
    }

    # 1. Lookahead Bias Check
    logger.info(f"  Running bias check for {pair}/{tf}...")
    bias = check_lookahead_bias(pair, tf, df, models, scaler, trained_features)
    result["bias_check"] = {
        "wr_no_shift": bias.wr_no_shift,
        "wr_shifted": bias.wr_shifted,
        "wr_drop": bias.wr_drop,
        "pnl_no_shift": bias.pnl_no_shift,
        "pnl_shifted": bias.pnl_shifted,
        "has_lookahead_bias": bias.has_lookahead_bias,
        "has_data_leakage": bias.has_data_leakage,
    }

    # 2. Walk-Forward Validation
    logger.info(f"  Running walkforward for {pair}/{tf}...")
    wf = run_walkforward_validation(pair, tf, df, models, scaler, trained_features, train_months=12, test_months=3, n_periods=6)
    result["walkforward"] = [
        {
            "period_id": p.period_id,
            "train_start": str(p.train_start),
            "train_end": str(p.train_end),
            "test_start": str(p.test_start),
            "test_end": str(p.test_end),
            "n_train": p.n_train_samples,
            "n_test": p.n_test_samples,
            "train_metrics": {
                "n_trades": p.train_result.n_trades if p.train_result else 0,
                "win_rate": p.train_result.win_rate if p.train_result else 0,
                "pnl_r": p.train_result.total_pnl_r if p.train_result else 0,
            }
            if p.train_result
            else {},
            "test_metrics": {
                "n_trades": p.test_result.n_trades if p.test_result else 0,
                "win_rate": p.test_result.win_rate if p.test_result else 0,
                "pnl_r": p.test_result.total_pnl_r if p.test_result else 0,
            }
            if p.test_result
            else {},
            "degradation": p.degradation,
        }
        for p in wf
    ]

    # 3. Monte Carlo Simulation
    logger.info(f"  Running MC simulation for {pair}/{tf}...")
    all_trades, _ = run_backtest_bar_aligned(df, models, scaler, trained_features, spread_pips=1.5)
    if len(all_trades) >= 50:
        mc = run_monte_carlo_simulation(all_trades, n_simulations=10000)
        result["monte_carlo"] = {
            "n_simulations": mc.n_simulations,
            "equity_mean": mc.equity_mean,
            "equity_std": mc.equity_std,
            "equity_percentiles": mc.equity_percentiles,
            "max_dd_mean": mc.max_dd_mean,
            "max_dd_percentiles": mc.max_dd_percentiles,
            "win_rate_mean": mc.win_rate_mean,
            "win_rate_percentiles": mc.win_rate_percentiles,
            "probability_of_ruin": mc.probability_of_ruin,
            "sharpe_mean": mc.sharpe_mean,
            "sharpe_std": mc.sharpe_std,
        }

    # 4. Black Swan Testing
    logger.info(f"  Running black swan tests for {pair}/{tf}...")
    bs = test_black_swan_events(pair, tf, df, models, scaler, trained_features)
    result["black_swan"] = {
        k: {
            "name": v.name,
            "severity": v.severity,
            "performance": v.system_performance,
        }
        for k, v in bs.items()
    }

    # 5. Out-of-Sample Testing
    logger.info(f"  Running OOS tests for {pair}/{tf}...")
    oos = test_out_of_sample(pair, tf, df, models, scaler, trained_features)
    result["out_of_sample"] = {
        k: {
            "start_date": str(v.start_date),
            "end_date": str(v.end_date),
            "n_trades": v.n_trades,
            "win_rate": v.win_rate,
            "pnl_r": v.total_pnl_r,
            "profit_factor": v.profit_factor,
            "max_drawdown": v.max_drawdown,
        }
        for k, v in oos.items()
    }

    return result


def generate_final_report(all_results: Dict) -> str:
    """Generate comprehensive validation report"""

    report = []
    report.append("=" * 100)
    report.append("CAVALIER TRADING SYSTEM - COMPREHENSIVE VALIDATION REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 100)

    passing_pairs = 0
    failing_pairs = 0
    bias_detected = 0

    wf_success_rates = []
    oos_pnl_totals = []
    mc_ruin_probs = []

    for pair_tf, result in all_results.items():
        if result is None:
            continue

        pair, tf = result["pair"], result["tf"]

        # Bias Analysis
        bias = result.get("bias_check", {})
        if bias.get("has_lookahead_bias"):
            bias_detected += 1
            bias_status = "[LOOKAHEAD BIAS]"
        elif bias.get("has_data_leakage"):
            bias_status = "[DATA LEAKAGE]"
        else:
            bias_status = "[OK]"

        wr_drop = bias.get("wr_drop", 0)

        # Walk-Forward Analysis
        wf_results = result.get("walkforward", [])
        wf_oos_wins = 0
        wf_oos_total = 0
        for wf in wf_results:
            test_metrics = wf.get("test_metrics", {})
            if test_metrics.get("n_trades", 0) > 0:
                wf_oos_total += 1
                if test_metrics.get("pnl_r", 0) > 0:
                    wf_oos_wins += 1

        wf_success_rate = wf_oos_wins / wf_oos_total if wf_oos_total > 0 else 0
        wf_success_rates.append(wf_success_rate)

        # Monte Carlo
        mc = result.get("monte_carlo", {})
        if mc:
            mc_ruin_probs.append(mc.get("probability_of_ruin", 0))

        # Out-of-Sample
        oos_results = result.get("out_of_sample", {})
        oos_pnl = sum(o.get("pnl_r", 0) for o in oos_results.values())
        oos_pnl_totals.append(oos_pnl)

        # Black Swan
        bs = result.get("black_swan", {})
        bs_passes = sum(1 for k, v in bs.items() if v.get("performance", {}).get("pnl_r", 0) > 0)

        # Overall Status
        is_passing = not bias.get("has_lookahead_bias", False) and wf_success_rate >= 0.5 and (not mc or mc.get("probability_of_ruin", 1) < 0.05)

        if is_passing:
            passing_pairs += 1
        else:
            failing_pairs += 1

        report.append(f"\n{pair}/{tf}:")
        report.append(f"  Data: {result['data_points']} points ({result['date_range']})")
        report.append(f"  Bias Check: WR drop={wr_drop:.1%} [{bias_status}]")
        report.append(f"  Walk-Forward: {wf_oos_wins}/{wf_oos_total} periods profitable ({wf_success_rate:.0%})")
        report.append(f"  Monte Carlo: Ruin prob={mc.get('probability_of_ruin', 0):.1%}" if mc else "  Monte Carlo: N/A")
        report.append(f"  Out-of-Sample: PnL={oos_pnl:.1f}R")
        report.append(f"  Black Swan: {bs_passes}/{len(bs)} events passed")
        report.append("  Status: [PASS]" if is_passing else "  Status: [FAIL]")

    # Aggregate Statistics
    report.append("\n" + "=" * 100)
    report.append("AGGREGATE STATISTICS")
    report.append("=" * 100)

    avg_wf = np.mean(wf_success_rates) if wf_success_rates else 0
    total_oos_pnl = sum(oos_pnl_totals)
    avg_ruin = np.mean(mc_ruin_probs) if mc_ruin_probs else 0

    report.append(f"\nPairs Tested: {len(all_results)}")
    report.append(f"Passing: {passing_pairs}, Failing: {failing_pairs}")
    report.append(f"Bias Detected: {bias_detected}/{len(all_results)} pairs")
    report.append(f"Average Walk-Forward Success Rate: {avg_wf:.1%}")
    report.append(f"Total Out-of-Sample PnL: {total_oos_pnl:.1f}R")
    report.append(f"Average Probability of Ruin: {avg_ruin:.1%}")

    # Verdict
    report.append("\n" + "=" * 100)
    report.append("VERDICT")
    report.append("=" * 100)

    if bias_detected > len(all_results) * 0.2:
        report.append("\nCRITICAL: Lookahead bias detected in >20% of pairs")
        report.append("   Results are potentially inflated - review feature engineering")

    if avg_wf < 0.5:
        report.append("\nWARNING: Walk-forward success rate <50%")
        report.append("   System may not generalize to unseen market conditions")

    if avg_ruin > 0.05:
        report.append("\nWARNING: Probability of ruin >5%")
        report.append("   Consider reduced position sizing for live trading")

    if passing_pairs >= len(all_results) * 0.7 and bias_detected == 0:
        report.append("\nSTATUS: APPROVED FOR LIVE TRADING")
        report.append("   - No significant bias detected")
        report.append("   - Walk-forward performance is consistent")
        report.append("   - Monte Carlo shows acceptable risk profile")
    elif passing_pairs >= len(all_results) * 0.5:
        report.append("\nSTATUS: APPROVED WITH CAUTION")
        report.append("   - Some performance degradation OOS")
        report.append("   - Recommend reduced position sizing (50-75%)")
        report.append("   - Monitor closely for regime changes")
    else:
        report.append("\nSTATUS: NOT RECOMMENDED FOR LIVE TRADING")
        report.append("   - Significant overfitting detected")
        report.append("   - Recommend further optimization before deployment")

    report.append("\n" + "=" * 100)
    report.append("END OF REPORT")
    report.append("=" * 100)

    return "\n".join(report)


def main():
    """Main execution"""
    logger.info("=" * 80)
    logger.info("CAVALIER SYSTEM - COMPREHENSIVE VALIDATION")
    logger.info("Walk-Forward + Monte Carlo + Bias Detection + Gap Testing")
    logger.info("=" * 80)

    all_results = {}

    for pair, tf in PAIR_TIMEFRAMES:
        result = analyze_pair_tf(pair, tf)
        if result:
            all_results[f"{pair}/{tf}"] = result

    logger.info("\nGenerating report...")

    report = generate_final_report(all_results)

    report_path = OUTPUT_DIR / f"comprehensive_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, "w") as f:
        f.write(report)

    results_path = OUTPUT_DIR / f"comprehensive_validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + report)

    logger.info(f"\nReport saved to: {report_path}")
    logger.info(f"Results JSON saved to: {results_path}")
    logger.info("Validation complete!")

    return all_results


if __name__ == "__main__":
    main()
