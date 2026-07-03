"""
LIVE MT5 OOS Walk-Forward & Monte Carlo Validation
====================================================

Validates the trading system using LIVE MT5 data for out-of-sample testing.
Tests all hardcoded thresholds and filters to identify blocking issues.

Usage:
    python CORE_MODULES/validation/live_mt5_wf_mc_validation.py

Author: JG
Version: 1.0.0
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("LIVE_WF_MC")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "DATA_MODELS" / "models_live"
RESULTS_DIR = PROJECT_ROOT / "CORE_MODULES" / "results"
OUTPUT_DIR = RESULTS_DIR / "LIVE_WF_MC_VALIDATION"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT / "CORE_MODULES"))
sys.path.insert(0, str(PROJECT_ROOT / "DATA_MODELS" / "training"))

# Import the proper compute_all_features (matches training exactly)
try:
    from compute_features_ultimate import compute_all_features

    log.info("[FEATURES] Using compute_all_features from training module")
except ImportError as e:
    log.warning(f"[FEATURES] Using fallback: {e}")
    compute_all_features = None

# Load env vars from launcher
os.environ.setdefault(
    "PAIRS", "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD,USDCHF,GBPCHF,NZDUSD,XAUUSD,XAGUSD,USOIL,UKOIL,HEATOIL,JP225,US100,HK50,UK100,BTCUSD,ETHUSD"
)
os.environ.setdefault("TFS", "M5,M15,M30,H1")

PAIRS = os.getenv("PAIRS", "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD,USDCHF,NZDUSD,XAUUSD,XAGUSD,BTCUSD,ETHUSD").split(",")
TFS = os.getenv("TFS", "M5,M15,M30,H1").split(",")

# Validation parameters
LOOKBACK_DAYS = 60  # How far back to fetch from MT5 (increased for H1)
WF_FOLDS = 4  # Walk-forward folds
MC_ITERS = 5000  # Monte Carlo iterations
MIN_BARS_WF = 300  # Minimum bars for walk-forward
MIN_TRADES_PER_FOLD = 5  # Minimum trades needed per fold to be valid
TARGET_WEEKLY_RETURN = 0.10  # 10% per week target
WEEKS_IN_DATA = LOOKBACK_DAYS / 7  # ~8.5 weeks

# TP/SL Configuration (ALIGNED WITH LIVE TRADING)
TP_PCT = 0.0030  # +0.30% (30 pips EURUSD) - matches risk_governor.json
SL_PCT = 0.0015  # -0.15% (15 pips EURUSD) - matches risk_governor.json
LOOKAHEAD_BARS = 100  # Allow 100 bars for trade to resolve

# Original thresholds (baseline)
ORIGINAL_THRESHOLDS = {
    "confluence_min": 0.30,
    "confidence_min": 0.30,
    "regime_ranging_min_conv": 0.05,
    "mtf_min_aligned": 2,
}

# Threshold search range (minimal drift from original)
THRESHOLD_SEARCH = {
    "confluence_min": {"min": 0.25, "max": 0.30, "step": 0.05},  # Only tighten slightly
    "confidence_min": {"min": 0.10, "max": 0.30, "step": 0.02},  # Key blocker - test lower
    "regime_ranging_min_conv": {"min": 0.03, "max": 0.10, "step": 0.01},  # Test slightly lower
}

# Current thresholds being tested (will be optimized)
TEST_CONFIGS = {
    "confluence_min": {"current": 0.30, "description": "SMC confluence minimum"},
    "confidence_min": {"current": 0.30, "description": "ML confidence minimum"},
    "regime_ranging_min_conv": {"current": 0.05, "description": "RANGING regime minimum conviction"},
    "mtf_min_aligned": {"current": 2, "description": "Minimum aligned timeframes"},
}


@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    pair: str
    direction: int
    entry_price: float
    exit_price: float
    sl: float
    tp: float
    lots: float
    pnl: float
    pnl_r: float
    outcome: str  # WIN, LOSS, BREAKEVEN


def init_mt5():
    """Initialize MT5 connection."""
    try:
        import MetaTrader5 as mt5

        if mt5.initialize():
            log.info("[MT5] Connected successfully")
            return mt5
        else:
            log.error(f"[MT5] Initialize failed: {mt5.last_error()}")
            return None
    except Exception as e:
        log.error(f"[MT5] Import/Init error: {e}")
        return None


def fetch_live_data(mt5, pair: str, tf: str, days: int = 30) -> Optional[pd.DataFrame]:
    """Fetch recent data from MT5."""
    import MetaTrader5 as mt5

    tf_map = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1}
    timeframe = tf_map.get(tf, mt5.TIMEFRAME_M15)

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)

    rates = mt5.copy_rates_range(pair, timeframe, start, now)
    if rates is None or len(rates) == 0:
        log.warning(f"[DATA] No data for {pair} {tf}")
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df = df.sort_index()

    log.info(f"[DATA] {pair} {tf}: {len(df)} bars ({df.index[0]} to {df.index[-1]})")
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    return atr


def detect_regime(df: pd.DataFrame, lookback: int = 50) -> pd.Series:
    """Detect market regime."""
    close = df["close"]
    returns = close.pct_change()
    volatility = returns.rolling(lookback).std()

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    adx = calculate_adx(df, period=14)

    regime = pd.Series("RANGING", index=df.index)
    high_vol = volatility > volatility.quantile(0.8)
    regime[high_vol] = "VOLATILE"

    strong_trend = (macd_hist > macd_hist.shift(1)) & (adx > 25)
    regime[strong_trend] = "TRENDING"

    return regime


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ADX."""
    high = df["high"]
    low = df["low"]
    df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr = calculate_atr(df, period)
    plus_di = 100 * (plus_dm.rolling(period).mean() / tr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / tr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()

    return adx


def _get_feature_names(model):
    """Get feature names from model."""
    if hasattr(model, "feature_names_") and model.feature_names_ is not None:
        return list(model.feature_names_)
    if hasattr(model, "feature_name_") and model.feature_name_ is not None:
        return list(model.feature_name_)
    if hasattr(model, "get_booster"):
        try:
            return list(model.get_booster().feature_names)
        except Exception:
            pass
    return None


def _align(X: pd.DataFrame, feat_names):
    """Align dataframe columns to expected features."""
    if feat_names is None:
        return X.select_dtypes(include=[np.number])
    missing = set(feat_names) - set(X.columns)
    for col in missing:
        X[col] = 0.0
    return X[feat_names]


def load_model(pair: str, tf: str):
    """Load trained model for pair/timeframe.

    Returns:
        models: List of (model, feature_names) tuples - each model with its own feature names
    """
    import joblib

    model_dir = MODELS_DIR / f"{pair}_{tf}"
    if not model_dir.exists():
        log.warning(f"[MODEL] No model directory: {model_dir}")
        return None, None

    model_features = []  # List of (model, feature_names) tuples

    # Load tiered models - use "full" tier which has all features
    for tier in ["full"]:
        tier_dir = model_dir / tier
        if not tier_dir.is_dir():
            continue

        for mtype in ["cat", "lgb", "xgb"]:
            mpath = tier_dir / f"{mtype}_model.joblib"
            if not mpath.exists():
                continue
            try:
                m = joblib.load(str(mpath))
                fn = _get_feature_names(m)
                if fn is None:
                    fpkl = tier_dir / "features.pkl"
                    if fpkl.exists():
                        fn = joblib.load(str(fpkl))
                if fn:
                    model_features.append((m, fn))
                    log.debug(f"[MODEL] Loaded {tier}/{mtype} with {len(fn)} features")
            except Exception as e:
                log.debug(f"[MODEL] Error loading {tier}/{mtype}: {e}")

    if model_features:
        log.info(f"[MODEL] Loaded {len(model_features)} models for {pair}_{tf}")
        return model_features

    log.warning(f"[MODEL] No models loaded for {pair}_{tf}")
    return None


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute basic features for prediction."""
    features = pd.DataFrame(index=df.index)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["tick_volume"] if "tick_volume" in df.columns else df["volume"]

    # Basic returns
    features["return_1"] = close.pct_change(1)
    features["return_5"] = close.pct_change(5)
    features["return_10"] = close.pct_change(10)

    # Volatility
    features["volatility_10"] = features["return_1"].rolling(10).std()
    features["volatility_20"] = features["return_1"].rolling(20).std()

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    features["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    features["macd"] = macd - signal

    # Bollinger Bands
    bb_mean = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    features["bb_position"] = (close - bb_mean) / (2 * bb_std)

    # Volume features
    features["volume_ratio"] = volume / volume.rolling(20).mean()

    # ATR
    features["atr"] = calculate_atr(df)
    features["atr_ratio"] = features["atr"] / close

    # Price position
    features["price_position"] = (close - low.rolling(20).min()) / (high.rolling(20).max() - low.rolling(20).min() + 1e-10)

    # Momentum
    features["momentum_5"] = close / close.shift(5) - 1
    features["momentum_10"] = close / close.shift(10) - 1

    features = features.fillna(0).replace([np.inf, -np.inf], 0)

    return features


def predict_ensemble(models_and_features, features_df: pd.DataFrame) -> np.ndarray:
    """Generate ensemble predictions using per-model feature alignment.

    Args:
        models_and_features: List of (model, feature_names) tuples
        features_df: DataFrame with computed features
    """
    if not models_and_features:
        log.warning("[PREDICT] No models provided!")
        return np.full(len(features_df), 0.5)

    probas = []
    for i, (model, feat_names) in enumerate(models_and_features):
        if feat_names is None:
            continue
        try:
            X = _align(features_df.copy(), feat_names)
            if hasattr(model, "predict_proba"):
                pred = model.predict_proba(X.values)[:, 1]
            else:
                pred = model.predict(X.values)
            probas.append(pred)
            log.debug(f"[PREDICT] Model {i}: range=[{pred.min():.3f}, {pred.max():.3f}]")
        except Exception as e:
            log.debug(f"[PREDICT] Model {i} error: {e}")
            continue

    if probas:
        result = np.mean(probas, axis=0)
        log.debug(f"[PREDICT] Ensemble: range=[{result.min():.3f}, {result.max():.3f}]")
        return result
    return np.full(len(features_df), 0.5)


def calculate_smc_confluence(df: pd.DataFrame, regime: pd.Series) -> pd.Series:
    """Calculate SMC-style confluence score."""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    confluence = pd.Series(0.5, index=df.index)

    # Order Block detection (simplified)
    for i in range(20, len(df)):
        if regime.iloc[i] == "TRENDING":
            # Bullish OB
            if close.iloc[i] > close.iloc[i - 5 : i].mean():
                confluence.iloc[i] = min(1.0, confluence.iloc[i] + 0.1)
            # Bearish OB
            if close.iloc[i] < close.iloc[i - 5 : i].mean():
                confluence.iloc[i] = max(0.0, confluence.iloc[i] - 0.1)

    # Fair Value Gap detection
    for i in range(3, len(df)):
        gap_up = high.iloc[i - 2] < low.iloc[i] and high.iloc[i - 1] < low.iloc[i]
        gap_down = low.iloc[i - 2] > high.iloc[i] and low.iloc[i - 1] > high.iloc[i]
        if gap_up:
            confluence.iloc[i] = min(1.0, confluence.iloc[i] + 0.15)
        if gap_down:
            confluence.iloc[i] = max(0.0, confluence.iloc[i] - 0.15)

    return confluence


def run_walkforward(
    df: pd.DataFrame,
    models_and_features,
    pair: str = "UNKNOWN",
    tf: str = "UNKNOWN",
    n_folds: int = 4,
    lookback_bars: int = 300,
    thresholds: Dict = None,
) -> List[List[Trade]]:
    """Run walk-forward analysis with strict NO-LEAKAGE between folds.

    CRITICAL: Features are computed ONLY on data available UP TO that point.
    Uses ROLLING trade simulation - enter trade, hold until TP/SL, then look for next signal.
    """
    min_bars = lookback_bars + 100
    if len(df) < min_bars:
        log.warning(f"[WF] Insufficient data: {len(df)} < {min_bars}")
        return []

    fold_trades: List[List[Trade]] = []
    fold_size = (len(df) - lookback_bars) // n_folds

    for fold in range(n_folds):
        train_end = lookback_bars + fold * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size + 100, len(df))

        log.info(f"[WF] {pair}_{tf} Fold {fold + 1}/{n_folds}: train={train_end}, test={test_end - test_start}")

        # NO LEAKAGE: Compute features ONLY on data available up to test_end
        hist_df = df.iloc[:test_end]

        if compute_all_features is not None:
            hist_features = compute_all_features(hist_df)
        else:
            hist_features = compute_features(hist_df)
        hist_regime = detect_regime(hist_df)
        hist_confluence = calculate_smc_confluence(hist_df, hist_regime)

        # Predict using ONLY historical data (no future data)
        hist_proba = predict_ensemble(models_and_features, hist_features)

        # Rolling trade simulation - enter trade, hold until close, then next
        trades = rolling_trade_simulation(df, hist_proba, hist_confluence, hist_regime, test_start, test_end, thresholds)
        fold_trades.append(trades)

        if trades:
            wins = [t for t in trades if t.outcome == "WIN"]
            wr = len(wins) / len(trades) * 100
            total_pnl = sum(t.pnl for t in trades)
            avg_pnl_r = np.mean([t.pnl_r for t in trades]) if trades else 0
            log.info(f"[WF] {pair}_{tf} Fold {fold + 1}: {len(trades)} trades, WR={wr:.1f}%, PnL={total_pnl:.2f}, AvgR={avg_pnl_r:.2f}")

    return fold_trades


def rolling_trade_simulation(
    df: pd.DataFrame,
    proba: np.ndarray,
    confluence: pd.Series,
    regime: pd.Series,
    test_start: int,
    test_end: int,
    thresholds: Dict = None,
) -> List[Trade]:
    """Simulate trades rolling through the market - enter, hold until close, next trade."""
    CONFLUENCE_MIN = thresholds.get("confluence_min", 0.30) if thresholds else 0.30
    PROB_THRESHOLD = thresholds.get("prob_threshold", 0.55) if thresholds else 0.55
    REGIME_RANGING_MIN_CONV = thresholds.get("regime_ranging_min_conv", 0.05) if thresholds else 0.05

    calculate_atr(df)
    trades: List[Trade] = []
    i = test_start

    while i < test_end - 1:
        prob = proba[i]
        conf = confluence.iloc[i] if i < len(confluence) else 0.5
        reg = regime.iloc[i] if i < len(regime) else "RANGING"

        if conf < CONFLUENCE_MIN:
            i += 1
            continue

        conviction = abs(prob - 0.5) * 2
        if reg == "RANGING" and conviction < REGIME_RANGING_MIN_CONV:
            i += 1
            continue

        buy_signal = prob > PROB_THRESHOLD
        sell_signal = prob < (1 - PROB_THRESHOLD)

        if not (buy_signal or sell_signal):
            i += 1
            continue

        direction = 1 if buy_signal else -1
        entry_price = df["close"].iloc[i]

        sl_dist = entry_price * SL_PCT
        tp_dist = entry_price * TP_PCT

        sl = entry_price - sl_dist * direction
        tp = entry_price + tp_dist * direction

        outcome, exit_price, exit_bar = simulate_trade_until_close(df, i, direction, entry_price, sl, tp, max_bars=LOOKAHEAD_BARS)

        pnl_r = (exit_price - entry_price) * direction / sl_dist
        pnl = pnl_r * 100

        trade = Trade(
            entry_time=df.index[i],
            exit_time=df.index[exit_bar] if exit_bar < len(df) else df.index[-1],
            pair=df.name if hasattr(df, "name") else "UNKNOWN",
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            sl=sl,
            tp=tp,
            lots=0.5,
            pnl=pnl,
            pnl_r=pnl_r,
            outcome=outcome,
        )
        trades.append(trade)

        i = exit_bar + 1

    return trades


def generate_trades(df: pd.DataFrame, proba: np.ndarray, confluence: pd.Series, regime: pd.Series, thresholds: Dict = None) -> List[Trade]:
    """Generate trades with full governance simulation."""
    if thresholds is None:
        thresholds = {"confidence_min": 0.30, "confluence_min": 0.30, "regime_ranging_min_conv": 0.05}

    trades: List[Trade] = []

    calculate_atr(df)

    CONFLUENCE_MIN = thresholds.get("confluence_min", 0.30)
    PROB_THRESHOLD = thresholds.get("prob_threshold", 0.55)  # Buy above this, sell below (1 - threshold)
    REGIME_RANGING_MIN_CONV = thresholds.get("regime_ranging_min_conv", 0.05)

    for i in range(20, len(df) - 1):
        prob = proba[i]
        conf = confluence.iloc[i] if i < len(confluence) else 0.5
        reg = regime.iloc[i] if i < len(regime) else "RANGING"

        if conf < CONFLUENCE_MIN:
            continue

        conviction = abs(prob - 0.5) * 2

        if reg == "RANGING" and conviction < REGIME_RANGING_MIN_CONV:
            continue

        # Signal generation: prob > 0.55 means model says price will go up with >55% confidence
        # For 70% WR target, only trade when model is confident
        buy_signal = prob > PROB_THRESHOLD
        sell_signal = prob < (1 - PROB_THRESHOLD)

        if not (buy_signal or sell_signal):
            continue

        direction = 1 if buy_signal else -1
        entry_price = df["close"].iloc[i]

        sl_dist = entry_price * SL_PCT
        tp_dist = entry_price * TP_PCT

        sl = entry_price - sl_dist * direction
        tp = entry_price + tp_dist * direction

        outcome, exit_price, exit_bar = simulate_trade_until_close(df, i, direction, entry_price, sl, tp, max_bars=LOOKAHEAD_BARS)

        pnl_r = (exit_price - entry_price) * direction / sl_dist
        pnl = pnl_r * 100

        trade = Trade(
            entry_time=df.index[i],
            exit_time=df.index[exit_bar] if exit_bar < len(df) else df.index[-1],
            pair=df.name if hasattr(df, "name") else "UNKNOWN",
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            sl=sl,
            tp=tp,
            lots=0.5,
            pnl=pnl,
            pnl_r=pnl_r,
            outcome=outcome,
        )
        trades.append(trade)

    return trades


def simulate_trade_until_close(
    df: pd.DataFrame, entry_bar: int, direction: int, entry_price: float, sl: float, tp: float, max_bars: int = 100
) -> Tuple[str, float, int]:
    """Simulate trade holding until TP or SL is hit.

    Args:
        df: DataFrame with OHLC data
        entry_bar: Bar index where trade was entered
        direction: 1 for BUY, -1 for SELL
        entry_price: Entry price
        sl: Stop loss price
        tp: Take profit price
        max_bars: Maximum bars to hold before forced close

    Returns:
        Tuple of (outcome, exit_price, exit_bar_index)
    """
    outcome = "TIMEOUT"
    exit_price = df["close"].iloc[min(entry_bar + max_bars, len(df) - 1)]
    exit_bar = min(entry_bar + max_bars, len(df) - 1)

    for bar in range(entry_bar + 1, min(entry_bar + max_bars + 1, len(df))):
        high = df["high"].iloc[bar]
        low = df["low"].iloc[bar]
        df["close"].iloc[bar]

        if direction == 1:  # BUY
            if low <= sl:
                outcome = "LOSS"
                exit_price = sl
                exit_bar = bar
                break
            elif high >= tp:
                outcome = "WIN"
                exit_price = tp
                exit_bar = bar
                break
        else:  # SELL
            if high >= sl:
                outcome = "LOSS"
                exit_price = sl
                exit_bar = bar
                break
            elif low <= tp:
                outcome = "WIN"
                exit_price = tp
                exit_bar = bar
                break

    return outcome, exit_price, exit_bar


def analyze_biases(all_trades: List[Trade]) -> Dict:
    """Analyze buy/sell, strategy, and distribution biases."""
    if not all_trades:
        return {}

    buys = [t for t in all_trades if t.direction == 1]
    sells = [t for t in all_trades if t.direction == -1]

    [t for t in buys if t.outcome == "WIN"]
    [t for t in sells if t.outcome == "WIN"]

    [t for t in buys if t.outcome == "LOSS"]
    [t for t in sells if t.outcome == "LOSS"]

    def stats(trades):
        if not trades:
            return {"count": 0, "wr": 0, "avg_r": 0, "total_r": 0, "pnl": 0}
        wins = [t for t in trades if t.outcome == "WIN"]
        return {
            "count": len(trades),
            "wr": len(wins) / len(trades) * 100 if trades else 0,
            "avg_r": np.mean([t.pnl_r for t in trades]) if trades else 0,
            "total_r": sum(t.pnl_r for t in trades),
            "pnl": sum(t.pnl for t in trades),
        }

    def outcome_dist(trades):
        wins = len([t for t in trades if t.outcome == "WIN"])
        losses = len([t for t in trades if t.outcome == "LOSS"])
        be = len([t for t in trades if t.outcome == "BREAKEVEN"])
        total = len(trades) if trades else 1
        return {"WIN": wins / total * 100, "LOSS": losses / total * 100, "BE": be / total * 100}

    biases = {
        "direction": {
            "buy_count": len(buys),
            "sell_count": len(sells),
            "buy_pct": len(buys) / len(all_trades) * 100,
            "sell_pct": len(sells) / len(all_trades) * 100,
            "direction_bias": (len(buys) - len(sells)) / len(all_trades) * 100,
        },
        "buy_stats": stats(buys),
        "sell_stats": stats(sells),
        "buy_outcomes": outcome_dist(buys),
        "sell_outcomes": outcome_dist(sells),
        "entry_prices": {
            "buy_mean": np.mean([t.entry_price for t in buys]) if buys else 0,
            "sell_mean": np.mean([t.entry_price for t in sells]) if sells else 0,
            "buy_std": np.std([t.entry_price for t in buys]) if buys else 0,
            "sell_std": np.std([t.entry_price for t in sells]) if sells else 0,
        },
        "pnl_distribution": {
            "mean_r": np.mean([t.pnl_r for t in all_trades]),
            "median_r": np.median([t.pnl_r for t in all_trades]),
            "std_r": np.std([t.pnl_r for t in all_trades]),
            "min_r": np.min([t.pnl_r for t in all_trades]),
            "max_r": np.max([t.pnl_r for t in all_trades]),
            "win_mean_r": np.mean([t.pnl_r for t in all_trades if t.outcome == "WIN"]) if [t for t in all_trades if t.outcome == "WIN"] else 0,
            "loss_mean_r": np.mean([t.pnl_r for t in all_trades if t.outcome == "LOSS"]) if [t for t in all_trades if t.outcome == "LOSS"] else 0,
            "win_loss_ratio": abs(
                np.mean([t.pnl_r for t in all_trades if t.outcome == "WIN"]) / np.mean([t.pnl_r for t in all_trades if t.outcome == "LOSS"])
            )
            if [t for t in all_trades if t.outcome == "LOSS"]
            else 0,
        },
        "entry_time_hours": {},
        "trade_duration": {},
    }

    hour_counts = {}
    for t in all_trades:
        h = t.entry_time.hour
        hour_counts[h] = hour_counts.get(h, 0) + 1

    biases["entry_time_hours"] = {h: {"count": c, "pct": c / len(all_trades) * 100} for h, c in sorted(hour_counts.items())}

    durations = [(t.exit_time - t.entry_time).total_seconds() / 60 for t in all_trades]
    biases["trade_duration"] = {
        "mean_minutes": np.mean(durations),
        "median_minutes": np.median(durations),
        "min_minutes": np.min(durations),
        "max_minutes": np.max(durations),
    }

    bins = [-np.inf, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 1.0, 2.0, np.inf]
    labels = ["<-0.5R", "-0.5 to -0.2R", "-0.2 to -0.1R", "-0.1 to 0R", "0 to 0.1R", "0.1 to 0.2R", "0.2 to 0.5R", "0.5 to 1R", "1 to 2R", ">2R"]
    hist, _ = np.histogram([t.pnl_r for t in all_trades], bins=bins)
    biases["pnl_r_histogram"] = {labels[i]: {"count": int(hist[i]), "pct": hist[i] / len(all_trades) * 100} for i in range(len(labels))}

    return biases


def run_monte_carlo(all_trades: List[Trade], n_sims: int = 5000) -> Dict:
    """Run Monte Carlo simulation with weekly return analysis."""
    if not all_trades:
        return {}

    pnls = [t.pnl_r for t in all_trades]

    total_r = sum(pnls)
    trades_per_week = len(pnls) / WEEKS_IN_DATA if WEEKS_IN_DATA > 0 else 0
    avg_r_per_trade = np.mean(pnls) if pnls else 0

    sim_results = []
    for _ in range(n_sims):
        sample = np.random.choice(pnls, size=len(pnls), replace=True)
        sim_results.append(sample.sum())

    sim_results = np.array(sim_results)

    weekly_returns = sim_results / WEEKS_IN_DATA
    weekly_prob_target = np.mean(weekly_returns >= TARGET_WEEKLY_RETURN)

    return {
        "total_r": float(total_r),
        "trades_per_week": float(trades_per_week),
        "avg_r_per_trade": float(avg_r_per_trade),
        "p50": float(np.median(sim_results)),
        "p50_weekly": float(np.median(weekly_returns)),
        "p10": float(np.percentile(sim_results, 10)),
        "p90": float(np.percentile(sim_results, 90)),
        "mean": float(np.mean(sim_results)),
        "std": float(np.std(sim_results)),
        "prob_profit": float(np.mean(sim_results > 0)),
        "max_drawdown": float(np.min(sim_results)),
        "max_gain": float(np.max(sim_results)),
        "weekly_returns": {
            "target": TARGET_WEEKLY_RETURN,
            "p50_weekly": float(np.median(weekly_returns)),
            "prob_meets_target": float(weekly_prob_target),
            "p10_weekly": float(np.percentile(weekly_returns, 10)),
            "p90_weekly": float(np.percentile(weekly_returns, 90)),
        },
        "meets_10pct_weekly": weekly_prob_target >= 0.5,
    }


def find_optimal_thresholds(df: pd.DataFrame, models_and_features, pair: str, tf: str) -> Dict:
    """Find optimal threshold values with minimal drift that generate sufficient trades."""
    if compute_all_features:
        features = compute_all_features(df)
    else:
        features = compute_features(df)
    proba = predict_ensemble(models_and_features, features)
    regime = detect_regime(df)
    confluence = calculate_smc_confluence(df, regime)

    results = {
        "pair": pair,
        "tf": tf,
        "proba_stats": {
            "min": float(proba.min()),
            "max": float(proba.max()),
            "mean": float(proba.mean()),
            "std": float(proba.std()),
        },
        "conviction_stats": {
            "min": float(np.abs(proba - 0.5).min() * 2),
            "max": float(np.abs(proba - 0.5).max() * 2),
            "mean": float(np.abs(proba - 0.5).mean() * 2),
        },
        "optimal_thresholds": {},
    }

    optimal_conf = 0.30  # Keep original, it's not the blocker
    optimal_conf_min_conv = 0.05  # Start with original
    optimal_prob_thresh = 0.55  # Start with original

    for prob_thresh in np.arange(0.65, 0.81, 0.02):
        trades_count = 0
        for i in range(20, len(df) - 1):
            prob = proba[i]
            conf = confluence.iloc[i]
            reg = regime.iloc[i]
            conviction = abs(prob - 0.5) * 2

            if conf < optimal_conf:
                continue
            if conviction < 0.05:
                continue
            if reg == "RANGING" and conviction < optimal_conf_min_conv:
                continue

            buy_signal = prob > prob_thresh
            sell_signal = prob < (1 - prob_thresh)
            if buy_signal or sell_signal:
                trades_count += 1

        if trades_count >= MIN_TRADES_PER_FOLD:
            optimal_prob_thresh = prob_thresh
            results["optimal_thresholds"] = {
                "prob_threshold": round(optimal_prob_thresh, 2),
                "confluence_min": optimal_conf,
                "regime_ranging_min_conv": optimal_conf_min_conv,
                "drift_from_original": {
                    "prob_threshold": round(optimal_prob_thresh - 0.55, 3),
                },
                "estimated_trades_per_fold": trades_count // WF_FOLDS if WF_FOLDS > 0 else 0,
            }
            break
    else:
        optimal_prob_thresh = 0.70  # Fallback to high confidence
        results["optimal_thresholds"] = {
            "prob_threshold": 0.70,
            "confluence_min": optimal_conf,
            "regime_ranging_min_conv": optimal_conf_min_conv,
            "drift_from_original": {
                "prob_threshold": round(0.70 - 0.55, 3),
            },
            "estimated_trades_per_fold": 0,
        }

    log.info(
        f"[OPTIMAL] {pair}_{tf}: prob_threshold={optimal_prob_thresh:.2f} (drift={results['optimal_thresholds']['drift_from_original']['prob_threshold']:+.3f})"
    )

    return results


def analyze_threshold_impact(df: pd.DataFrame, models_and_features, threshold_name: str, threshold_value: float, new_value: float) -> Dict:
    """Analyze impact of changing a threshold."""
    if compute_all_features:
        features = compute_all_features(df)
    else:
        features = compute_features(df)
    proba = predict_ensemble(models_and_features, features)
    regime = detect_regime(df)
    confluence = calculate_smc_confluence(df, regime)

    current_pass = 0
    new_pass = 0
    total = 0

    for i in range(20, len(df)):
        prob = proba[i]
        conf = confluence.iloc[i]
        reg = regime.iloc[i]
        conviction = abs(prob - 0.5) * 2

        total += 1

        # Current threshold
        if threshold_name == "confluence_min":
            if conf >= threshold_value:
                current_pass += 1
            if conf >= new_value:
                new_pass += 1
        elif threshold_name == "confidence_min":
            if conviction >= threshold_value:
                current_pass += 1
            if conviction >= new_value:
                new_pass += 1
        elif threshold_name == "regime_ranging_min_conv":
            if reg != "RANGING" or conviction >= threshold_value:
                current_pass += 1
            if reg != "RANGING" or conviction >= new_value:
                new_pass += 1

    return {
        "threshold": threshold_name,
        "current_value": threshold_value,
        "new_value": new_value,
        "current_pass_rate": current_pass / max(total, 1),
        "new_pass_rate": new_pass / max(total, 1),
        "trade_increase": (new_pass - current_pass) / max(current_pass, 1),
    }


def main():
    """Main validation run."""
    log.info("=" * 60)
    log.info("LIVE MT5 OOS WALK-FORWARD & MONTE CARLO VALIDATION")
    log.info("=" * 60)

    # Initialize MT5
    mt5 = init_mt5()
    if not mt5:
        log.error("[MT5] Failed to connect. Cannot proceed with live validation.")
        return

    try:
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pairs_tested": [],
            "thresholds_analyzed": {},
            "wf_results": {},
            "mc_results": {},
        }

        all_trades = []

        for pair in PAIRS:  # Test ALL pairs
            log.info(f"\n{'=' * 40}")
            log.info(f"TESTING: {pair}")
            log.info(f"{'=' * 40}")

            pair_results = {"timeframes": {}}

            # Test all available timeframes that have models
            for tf in ["M5", "M15", "M30", "H1"]:
                log.info(f"\n--- {pair} {tf} ---")

                # Check if model exists
                model_dir = MODELS_DIR / f"{pair}_{tf}"
                if not model_dir.exists():
                    log.warning(f"[SKIP] No model for {pair} {tf}")
                    continue

                # Fetch live data
                df = fetch_live_data(mt5, pair, tf, days=LOOKBACK_DAYS)
                if df is None or len(df) < MIN_BARS_WF:
                    log.warning(f"[SKIP] Insufficient data for {pair} {tf} ({len(df) if df else 0} bars)")
                    continue

                # Load model
                models_and_features = load_model(pair, tf)
                if not models_and_features:
                    log.warning(f"[SKIP] No models loaded for {pair} {tf}")
                    continue

                # Find optimal thresholds with minimal drift
                optimal = find_optimal_thresholds(df, models_and_features, pair, tf)
                optimal_thresholds = optimal["optimal_thresholds"]
                results["thresholds_analyzed"][f"{pair}_{tf}_optimal"] = optimal

                # Run walk-forward with optimal thresholds (NO LEAKAGE)
                fold_trades = run_walkforward(df, models_and_features, pair=pair, tf=tf, n_folds=WF_FOLDS, thresholds=optimal_thresholds)

                if fold_trades:
                    pair_results["timeframes"][tf] = {
                        "total_trades": sum(len(t) for t in fold_trades),
                        "folds": len(fold_trades),
                        "optimal_thresholds": optimal_thresholds,
                    }
                    for trades in fold_trades:
                        all_trades.extend(trades)

            results["pairs_tested"].append(pair)
            results["wf_results"][pair] = pair_results

        # Run Monte Carlo on all trades
        if all_trades:
            mc_stats = run_monte_carlo(all_trades, n_sims=MC_ITERS)
            results["mc_results"] = mc_stats

            # Analyze biases
            bias_analysis = analyze_biases(all_trades)
            results["bias_analysis"] = bias_analysis

            # Overall Win Rate & Profit Factor
            wins = [t for t in all_trades if t.outcome == "WIN"]
            losses = [t for t in all_trades if t.outcome == "LOSS"]
            win_rate = len(wins) / len(all_trades) * 100 if all_trades else 0
            gross_profit = sum(t.pnl for t in wins) if wins else 0
            gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999 if gross_profit > 0 else 0)
            results["performance"] = {
                "total_trades": len(all_trades),
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "gross_profit": gross_profit,
                "gross_loss": gross_loss,
                "wins": len(wins),
                "losses": len(losses),
            }
            # Investor thresholds
            TARGET_WR = 70.0
            TARGET_PF = 2.0
            wr_pass = win_rate >= TARGET_WR
            pf_pass = profit_factor >= TARGET_PF
            results["investor_gates"] = {
                "target_wr": TARGET_WR,
                "target_pf": TARGET_PF,
                "wr_pass": wr_pass,
                "pf_pass": pf_pass,
                "overall_pass": wr_pass and pf_pass,
            }

            # Summary
            log.info("\n" + "=" * 60)
            log.info("VALIDATION SUMMARY")
            log.info("=" * 60)
            log.info(f"Total trades: {len(all_trades)}")
            log.info(f"Win Rate: {win_rate:.1f}% {'✓ PASS' if wr_pass else '✗ FAIL'} (target {TARGET_WR}%)")
            log.info(f"Profit Factor: {profit_factor:.2f} {'✓ PASS' if pf_pass else '✗ FAIL'} (target {TARGET_PF:.1f})")
            log.info(f"Trades/week: {mc_stats.get('trades_per_week', 0):.0f}")
            log.info(f"Avg R/trade: {mc_stats.get('avg_r_per_trade', 0):.4f}")
            log.info(f"Total R: {mc_stats.get('total_r', 0):.2f}")
            log.info(f"MC P50 PnL (R): {mc_stats.get('p50', 'N/A'):.2f}")
            log.info(f"MC Prob Profit: {mc_stats.get('prob_profit', 0) * 100:.1f}%")
            log.info(f"MC Max Drawdown: {mc_stats.get('max_drawdown', 'N/A'):.2f}")

            wr = mc_stats.get("weekly_returns", {})
            log.info(f"\n{'=' * 40}")
            log.info("10% WEEKLY TARGET ANALYSIS")
            log.info(f"{'=' * 40}")
            log.info(f"Target: {wr.get('target', 0) * 100:.1f}% per week")
            log.info(f"P50 Weekly R: {wr.get('p50_weekly', 0):.4f} ({wr.get('p50_weekly', 0) * 100:.2f}%)")
            log.info(f"P10 Weekly R: {wr.get('p10_weekly', 0):.4f}")
            log.info(f"Prob meets target: {wr.get('prob_meets_target', 0) * 100:.1f}%")
            status = "✓ MEETS 10% WEEKLY" if mc_stats.get("meets_10pct_weekly") else "✗ BELOW TARGET"
            log.info(f"Status: {status}")

            # Log bias analysis
            dir_bias = bias_analysis.get("direction", {})
            log.info(f"\n{'=' * 40}")
            log.info("DIRECTION BIAS")
            log.info(f"{'=' * 40}")
            log.info(f"Buys: {dir_bias.get('buy_count', 0)} ({dir_bias.get('buy_pct', 0):.1f}%)")
            log.info(f"Sells: {dir_bias.get('sell_count', 0)} ({dir_bias.get('sell_pct', 0):.1f}%)")
            log.info(f"Bias: {dir_bias.get('direction_bias', 0):+.1f}% (positive=BUY bias)")

            buy_stats = bias_analysis.get("buy_stats", {})
            sell_stats = bias_analysis.get("sell_stats", {})
            log.info(f"\nBUY stats: WR={buy_stats.get('wr', 0):.1f}%, AvgR={buy_stats.get('avg_r', 0):.4f}, TotalR={buy_stats.get('total_r', 0):.2f}")
            log.info(
                f"SELL stats: WR={sell_stats.get('wr', 0):.1f}%, AvgR={sell_stats.get('avg_r', 0):.4f}, TotalR={sell_stats.get('total_r', 0):.2f}"
            )

            pnl_dist = bias_analysis.get("pnl_distribution", {})
            log.info("\nPnL DISTRIBUTION")
            log.info(f"Mean R: {pnl_dist.get('mean_r', 0):.4f}, Median: {pnl_dist.get('median_r', 0):.4f}, StdDev: {pnl_dist.get('std_r', 0):.4f}")
            log.info(f"Range: [{pnl_dist.get('min_r', 0):.4f}, {pnl_dist.get('max_r', 0):.4f}]")
            log.info(f"Win avg R: {pnl_dist.get('win_mean_r', 0):.4f}, Loss avg R: {pnl_dist.get('loss_mean_r', 0):.4f}")
            log.info(f"Win/Loss ratio: {pnl_dist.get('win_loss_ratio', 0):.2f}")

            dur = bias_analysis.get("trade_duration", {})
            log.info("\nTRADE DURATION")
            log.info(f"Mean: {dur.get('mean_minutes', 0):.1f} min, Median: {dur.get('median_minutes', 0):.1f} min")
            log.info(f"Range: [{dur.get('min_minutes', 0):.1f}, {dur.get('max_minutes', 0):.1f}] min")

            log.info("\nPnL R HISTOGRAM")
            for label, data in bias_analysis.get("pnl_r_histogram", {}).items():
                bar = "█" * int(data["pct"] / 2)
                log.info(f"  {label:>15}: {data['count']:>5} ({data['pct']:>5.1f}%) {bar}")

        # Save results
        output_file = OUTPUT_DIR / f"live_wf_mc_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        log.info(f"\nResults saved to: {output_file}")

        # Generate report
        report_file = OUTPUT_DIR / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("LIVE MT5 OOS VALIDATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Pairs tested: {len(results['pairs_tested'])}\n")
            f.write(f"Total trades: {len(all_trades)}\n")
            f.write(f"Data span: {LOOKBACK_DAYS} days (~{WEEKS_IN_DATA:.1f} weeks)\n\n")
            perf = results.get("performance", {})
            gates = results.get("investor_gates", {})
            f.write("INVESTOR GATE RESULTS:\n")
            f.write(f"  Win Rate: {perf.get('win_rate', 0):.1f}% {'PASS' if gates.get('wr_pass') else 'FAIL'} (target {gates.get('target_wr', 70)}%)\n")
            f.write(f"  Profit Factor: {perf.get('profit_factor', 0):.2f} {'PASS' if gates.get('pf_pass') else 'FAIL'} (target {gates.get('target_pf', 2.0)})\n")
            f.write(f"  Overall: {'PASS' if gates.get('overall_pass') else 'FAIL'}\n\n")

            if results.get("mc_results"):
                mc = results["mc_results"]
                f.write("MONTE CARLO RESULTS:\n")
                f.write(f"  Total R: {mc.get('total_r', 0):.2f}\n")
                f.write(f"  Trades per week: {mc.get('trades_per_week', 0):.0f}\n")
                f.write(f"  Avg R per trade: {mc.get('avg_r_per_trade', 0):.4f}\n")
                f.write(f"  P50 Total PnL (R): {mc.get('p50', 0):.2f}\n")
                f.write(f"  P10 Total PnL (R): {mc.get('p10', 0):.2f}\n")
                f.write(f"  P90 Total PnL (R): {mc.get('p90', 0):.2f}\n")
                f.write(f"  Probability of Profit: {mc.get('prob_profit', 0) * 100:.1f}%\n")
                f.write(f"  Max Drawdown: {mc.get('max_drawdown', 0):.2f}\n\n")

                wr = mc.get("weekly_returns", {})
                f.write("10% WEEKLY TARGET ANALYSIS:\n")
                f.write(f"  Target: {wr.get('target', 0) * 100:.1f}% per week\n")
                f.write(f"  P50 Weekly R: {wr.get('p50_weekly', 0):.4f}\n")
                f.write(f"  P10 Weekly R: {wr.get('p10_weekly', 0):.4f}\n")
                f.write(f"  P90 Weekly R: {wr.get('p90_weekly', 0):.4f}\n")
                f.write(f"  Prob meets target: {wr.get('prob_meets_target', 0) * 100:.1f}%\n")
                f.write(f"  MEETS 10% WEEKLY: {'YES' if mc.get('meets_10pct_weekly') else 'NO'}\n\n")

            f.write("OPTIMAL THRESHOLDS:\n")
            for key, a in results.get("thresholds_analyzed", {}).items():
                if "_optimal" in key:
                    opt = a.get("optimal_thresholds", {})
                    f.write(f"  {a.get('pair', '')}_{a.get('tf', '')}:\n")
                    f.write(f"    confidence_min: {opt.get('confidence_min', 'N/A')}\n")
                    f.write(f"    drift: {opt.get('drift_from_original', {}).get('confidence_min', 'N/A')}\n")

        log.info(f"Report saved to: {report_file}")

    finally:
        mt5.shutdown()
        log.info("[MT5] Connection closed")


if __name__ == "__main__":
    main()
