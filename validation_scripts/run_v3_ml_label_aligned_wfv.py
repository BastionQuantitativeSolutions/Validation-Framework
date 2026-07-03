#!/usr/bin/env python3
"""
V3 Model-Driven Walk-Forward Validation for ML Label-Aligned Mode
=================================================================
This is the real diagnostic gate. It loads the live MLS V3 model grid
(DATA_MODELS/models_live/models_all/{pair}_M1) and runs it on hold-out
M1 data with execution forced to the ML training labels:

    1R stop-loss / 1.3R take-profit, no partials/breakeven/trailing/time exits.

Pass criteria:
    - profit_factor >= 1.2
    - win_rate >= 52%
    - payoff_ratio >= 0.8
    - at least 30 trades

If the gate fails, the models are not predictive at the 1R/1.3R horizon and
must be retrained or replaced. If it passes, the execution layer was the
problem and you can go live with 0.5% risk.

Usage:
    python run_v3_ml_label_aligned_wfv.py --symbols EURUSD --start 2025-10-01 --end 2025-10-31
    python run_v3_ml_label_aligned_wfv.py --symbols FOREX --start 2025-10-24 --end 2026-06-23 --bar-step 5
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "CORE_MODULES"))
sys.path.insert(0, str(PROJECT_ROOT / "MLS_V3_GENERATOR" / "features"))

DATA_RAW_DIR = PROJECT_ROOT / "DATA_MODELS" / "data_raw"
DATA_PARQUET_DIR = PROJECT_ROOT / "DATA_MODELS" / "data_parquet"
CONFIG_PATH = PROJECT_ROOT / "CORE_MODULES" / "config" / "cavalier_unified_config.json"
RESULTS_DIR = PROJECT_ROOT / "CORE_MODULES" / "results" / "backtest" / "v3_ml_label_aligned_wfv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

FOREX_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "GBPCHF"
]

GATES = {
    "profit_factor": 1.2,
    "win_rate": 0.52,
    "payoff_ratio": 0.8,
    "min_trades": 30,
}


def activate_ml_label_aligned_config() -> str:
    """Create a temporary config with ml_label_aligned_mode=true and point env to it."""
    with open(CONFIG_PATH, "r", encoding="utf-8-sig") as fh:
        cfg = json.load(fh)
    cfg.setdefault("execution", {})["ml_label_aligned_mode"] = True
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(cfg, tmp, indent=2)
    tmp.close()
    os.environ["CAVALIER_UNIFIED_CONFIG"] = tmp.name
    logger.info(f"[v3-wfv] Activated ml_label_aligned_mode via temp config: {tmp.name}")
    return tmp.name


def _read_raw_csv(path: Path) -> pd.DataFrame:
    """Read a Dukascopy-style raw M1 CSV file."""
    with open(path, "r", encoding="utf-8") as fh:
        header = fh.readline().strip()

    if header.startswith("time,"):
        df = pd.read_csv(path, parse_dates=["time"])
        df = df[["time", "open", "high", "low", "close", "tick_volume"]]
    else:
        df = pd.read_csv(
            path,
            header=None,
            names=["date", "time_str", "open", "high", "low", "close", "tick_volume"],
        )
        df["time"] = pd.to_datetime(df["date"] + " " + df["time_str"], format="%Y.%m.%d %H:%M")
        df = df[["time", "open", "high", "low", "close", "tick_volume"]]

    for col in ["open", "high", "low", "close", "tick_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _file_covers_range(path: Path, start: datetime, end: datetime) -> bool:
    """Heuristic: does a raw M1 file likely contain bars inside [start, end]?"""
    stem = path.stem
    suffix = stem.split("_")[-1]

    if len(suffix) == 6:  # YYYYMM
        year, month = int(suffix[:4]), int(suffix[4:])
        file_start = datetime(year, month, 1)
        file_end = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)
    elif len(suffix) == 4:  # YYYY
        year = int(suffix)
        file_start = datetime(year, 1, 1)
        file_end = datetime(year + 1, 1, 1)
    else:
        return False

    return file_end > start and file_start <= end


def load_raw_m1(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Load M1 bars from raw CSV files for a symbol and date range."""
    symbol_lower = symbol.lower()
    files = [
        f for f in sorted(DATA_RAW_DIR.glob(f"dat_mt_{symbol_lower}_m1_*"))
        if _file_covers_range(f, start, end)
    ]

    if not files:
        logger.warning(f"[v3-wfv] No raw M1 files found for {symbol} in range")
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            df = _read_raw_csv(f)
            if not df.empty:
                frames.append(df)
        except Exception as exc:
            logger.warning(f"[v3-wfv] Failed to read {f.name}: {exc}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    mask = (df["time"] >= start) & (df["time"] <= end)
    return df.loc[mask].copy()


def load_parquet_m1(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Load M1 bars from parquet if available, normalising column names."""
    pq = DATA_PARQUET_DIR / f"{symbol}_M1.parquet"
    if not pq.exists():
        return pd.DataFrame()

    df = pd.read_parquet(pq)

    # Normalise index -> time column
    if df.index.name in ("time", "timestamp", "datetime") or isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    idx_col = None
    for col in ("time", "timestamp", "datetime"):
        if col in df.columns:
            idx_col = col
            break
    if idx_col is None:
        logger.warning(f"[v3-wfv] Parquet {pq.name} has no recognisable time column")
        return pd.DataFrame()
    df = df.rename(columns={idx_col: "time"})
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)

    # Normalise volume column
    if "tick_volume" not in df.columns:
        for vol_col in ("volume", "real_volume", "tick_volume"):
            if vol_col in df.columns:
                df = df.rename(columns={vol_col: "tick_volume"})
                break
    if "tick_volume" not in df.columns:
        df["tick_volume"] = 0

    required = ["time", "open", "high", "low", "close", "tick_volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(f"[v3-wfv] Parquet {pq.name} missing columns {missing}")
        return pd.DataFrame()

    df = df[required]
    mask = (df["time"] >= start) & (df["time"] <= end)
    return df.loc[mask].copy()


def load_m1_data(symbols: List[str], start: datetime, end: datetime) -> Dict[str, pd.DataFrame]:
    """Load M1 data for all requested symbols, preferring parquet then raw CSV."""
    data: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        df = load_parquet_m1(symbol, start, end)
        if df.empty:
            df = load_raw_m1(symbol, start, end)
        if not df.empty:
            data[symbol] = df
            logger.info(f"[v3-wfv] Loaded {len(df):,} M1 bars for {symbol}")
        else:
            logger.warning(f"[v3-wfv] No M1 data for {symbol}")
    return data


def compute_atr_price(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ATR in price units (not normalized)."""
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def generate_folds(start: datetime, end: datetime, fold_months: int) -> List[Tuple[datetime, datetime]]:
    """Generate walk-forward folds of `fold_months` length."""
    folds = []
    cur = datetime(start.year, start.month, 1)
    while cur < end:
        fold_start = max(cur, start)
        years = (cur.month - 1 + fold_months) // 12
        month = ((cur.month - 1 + fold_months) % 12) + 1
        nxt = datetime(cur.year + years, month, 1)
        fold_end = min(nxt - timedelta(seconds=1), end)
        folds.append((fold_start, fold_end))
        cur = nxt
    return folds


def load_models_and_features(symbols: List[str], price_data: Dict[str, pd.DataFrame]):
    """Load V3 model grid and pre-compute M1 features for each symbol."""
    from m1_features import compute_m1_features
    from core.models.mls_v3_loader import load_mls_v3_models, load_mls_v3_feature_metadata

    models: Dict[str, Dict] = {}
    meta: Dict[str, Dict] = {}
    features: Dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        try:
            models_by_target, _ = load_mls_v3_models(symbol, "M1")
            models[symbol] = models_by_target
            meta[symbol] = load_mls_v3_feature_metadata(symbol, "M1")
            logger.info(f"[v3-wfv] Loaded V3 model grid for {symbol}_M1")
        except Exception as exc:
            logger.error(f"[v3-wfv] Could not load V3 models for {symbol}: {exc}")
            continue

    for symbol, df in price_data.items():
        if symbol not in models:
            continue
        try:
            df_idx = df.set_index("time")
            feats = compute_m1_features(df_idx, symbol=symbol, timeframe="M1", data_dir=DATA_PARQUET_DIR)
            feats = feats.reset_index()
            features[symbol] = feats
            logger.info(f"[v3-wfv] Computed {feats.shape[1]} M1 features for {symbol}")
        except Exception as exc:
            logger.error(f"[v3-wfv] Feature engineering failed for {symbol}: {exc}")

    return models, meta, features


def _batch_predict_family(model, family: str, X: pd.DataFrame, feature_names: List[str], is_binary: bool, cat_features: Optional[List[str]] = None):
    """Batch predict on a feature DataFrame for one family."""
    from core.models.mls_v3_ensemble import _align_df, _catboost_cat_features
    import xgboost as xgb

    if family == "cat":
        from catboost import Pool

        cat_features = cat_features or _catboost_cat_features(model, feature_names)
        X_aligned = _align_df(X, feature_names, cat_features=cat_features)
        pool = Pool(X_aligned, cat_features=cat_features)
        if is_binary and hasattr(model, "predict_proba"):
            p = model.predict_proba(pool)
        else:
            p = model.predict(pool)
        p = np.asarray(p)
        if p.ndim > 1 and p.shape[1] >= 2:
            return p[:, 1]
        return p.ravel()

    X_aligned = _align_df(X, feature_names)
    X_values = np.asarray(X_aligned, dtype=np.float32)

    if family == "xgb":
        if is_binary and hasattr(model, "predict_proba"):
            p = model.predict_proba(X_values)
            p = np.asarray(p)
            if p.ndim > 1 and p.shape[1] >= 2:
                return p[:, 1]
            return p.ravel()
        booster = model.__dict__.get("_Booster") or getattr(model, "_Booster", None)
        if booster is not None:
            dmat = xgb.DMatrix(X_values, feature_names=feature_names)
            return np.asarray(booster.predict(dmat)).ravel()
        return np.asarray(model.predict(X_values)).ravel()

    if family == "lgb":
        return np.asarray(model.predict(X_values)).ravel()

    raise ValueError(f"Unknown family: {family}")


def build_prediction_df(
    symbol: str,
    feats: pd.DataFrame,
    models_by_target: Dict[str, Dict[str, Any]],
    meta: Dict[str, Any],
    family_weights: Dict[str, float],
) -> pd.DataFrame:
    """Compute all six V3 target predictions for every row in feats (batch)."""
    from core.models.mls_v3_ensemble import prepare_mls_v3_features

    expected_features = meta["numeric"] + meta["categorical"]
    X_full = prepare_mls_v3_features(
        feats,
        expected_features=expected_features,
        numeric_features=meta["numeric"],
        categorical_features=meta["categorical"],
        tag=f"{symbol}.M1",
    )

    targets = [
        "long_hit_1_3R", "short_hit_1_3R",
        "long_expected_r", "short_expected_r",
        "long_tail_risk", "short_tail_risk",
    ]
    pred_df = pd.DataFrame(index=feats.index)

    for target in targets:
        is_binary = "hit" in target
        family_map = models_by_target.get(target, {})
        weighted_preds = []
        active_weights = []

        for family in ("xgb", "lgb", "cat"):
            wrapper = family_map.get(family)
            if wrapper is None:
                continue
            model = wrapper.get("model")
            if model is None:
                continue
            feature_names = wrapper.get("feature_names") or expected_features
            cat_features = None
            try:
                vals = _batch_predict_family(model, family, X_full, feature_names, is_binary, cat_features)
                weighted_preds.append(vals * family_weights[family])
                active_weights.append(family_weights[family])
            except Exception as exc:
                logger.warning(f"[v3-wfv] batch predict failed for {symbol}.{target}.{family}: {exc}")
                continue

        if weighted_preds:
            total_w = sum(active_weights)
            pred_df[target] = np.sum(weighted_preds, axis=0) / total_w
        else:
            pred_df[target] = 0.5 if is_binary else 0.0

    return pred_df


# Trade dataclass
class Trade:
    def __init__(self, ticket: int, symbol: str, direction: int, entry_time: datetime,
                 entry_price: float, sl: float, tp: float, volume: float, atr: float,
                 long_score: float, short_score: float, expected_r: float):
        self.ticket = ticket
        self.symbol = symbol
        self.direction = direction
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.sl = sl
        self.tp = tp
        self.volume = volume
        self.atr = atr
        self.long_score = long_score
        self.short_score = short_score
        self.expected_r = expected_r
        self.exit_time: Optional[datetime] = None
        self.exit_price: Optional[float] = None
        self.exit_reason: str = ""
        self.pnl: float = 0.0
        self.pnl_pips: float = 0.0
        self.r_multiplier: float = 0.0
        self.bars_held: int = 0


def simulate_trades(trades: List[Trade], price_df: pd.DataFrame):
    """Vectorised first-hit exit simulation on M1 bars."""
    if not trades:
        return

    times = price_df["time"].values
    highs = price_df["high"].values
    lows = price_df["low"].values
    closes = price_df["close"].values
    time_to_idx = {t: i for i, t in enumerate(times)}

    for trade in trades:
        entry_idx = time_to_idx.get(trade.entry_time)
        if entry_idx is None or entry_idx + 1 >= len(times):
            trade.exit_time = times[-1]
            trade.exit_price = closes[-1]
            trade.exit_reason = "END_OF_DATA"
            continue

        start = entry_idx + 1
        if trade.direction == 1:
            sl_hits = np.where(lows[start:] <= trade.sl)[0]
            tp_hits = np.where(highs[start:] >= trade.tp)[0]
        else:
            sl_hits = np.where(highs[start:] >= trade.sl)[0]
            tp_hits = np.where(lows[start:] <= trade.tp)[0]

        sl_idx = sl_hits[0] + start if sl_hits.size else None
        tp_idx = tp_hits[0] + start if tp_hits.size else None

        if sl_idx is not None and tp_idx is not None:
            # Same-bar ambiguity: conservatively assume SL if both hit on the same bar.
            if sl_idx <= tp_idx:
                exit_idx, exit_price, reason = sl_idx, trade.sl, "SL"
            else:
                exit_idx, exit_price, reason = tp_idx, trade.tp, "TP"
        elif sl_idx is not None:
            exit_idx, exit_price, reason = sl_idx, trade.sl, "SL"
        elif tp_idx is not None:
            exit_idx, exit_price, reason = tp_idx, trade.tp, "TP"
        else:
            exit_idx, exit_price, reason = len(times) - 1, closes[-1], "END_OF_DATA"

        trade.exit_time = times[exit_idx]
        trade.exit_price = exit_price
        trade.exit_reason = reason
        trade.bars_held = exit_idx - entry_idx


def compute_trade_pnl(trade: Trade, instrument_config: Dict):
    pip_size = instrument_config["pip_size"]
    pip_value = instrument_config["pip_value_per_lot"]
    if trade.direction == 1:
        pnl_pips = (trade.exit_price - trade.entry_price) / pip_size
        risk_dist = trade.entry_price - trade.sl
    else:
        pnl_pips = (trade.entry_price - trade.exit_price) / pip_size
        risk_dist = trade.sl - trade.entry_price

    trade.pnl_pips = pnl_pips
    trade.pnl = pnl_pips * pip_value * trade.volume
    trade.r_multiplier = (pnl_pips * pip_size) / risk_dist if risk_dist > 0 else 0.0


def run_fold(
    symbols: List[str],
    models: Dict[str, Dict],
    meta: Dict[str, Dict],
    features: Dict[str, pd.DataFrame],
    predictions: Dict[str, pd.DataFrame],
    price_data: Dict[str, pd.DataFrame],
    fold_start: datetime,
    fold_end: datetime,
    args,
) -> Dict:
    """Run one walk-forward fold using the V3 model grid."""
    from core.models.mls_v3_ensemble import apply_mls_v3_signal_logic, load_mls_v3_thresholds
    from CORE_MODULES.backtesting.live_trading_backtester import get_instrument_config
    from CORE_MODULES.core.unified_exits import calculate_sl_tp

    thresholds = load_mls_v3_thresholds()

    balance = args.balance
    risk_amount = balance * (args.risk / 100)
    trades: List[Trade] = []
    daily_counts: Dict[str, Dict[str, int]] = {symbol: {} for symbol in symbols}
    ticket = 1

    for symbol in symbols:
        if symbol not in features or symbol not in models or symbol not in predictions:
            continue

        feats = features[symbol]
        pred_df = predictions[symbol]

        # Restrict to fold window (keep original index to align with pred_df)
        mask = (feats["time"] >= fold_start) & (feats["time"] <= fold_end)
        fold_feats = feats.loc[mask].copy()

        price_df = price_data[symbol]
        price_mask = (price_df["time"] >= fold_start) & (price_df["time"] <= fold_end)
        fold_prices = price_df.loc[price_mask].copy().reset_index(drop=True)
        if fold_feats.empty or fold_prices.empty:
            continue

        # Compute ATR on the full loaded history (including lookback) so the fold
        # start has a valid ATR value.
        full_atr = compute_atr_price(price_df)
        atr_series = full_atr.loc[price_mask].reset_index(drop=True)

        # Time -> price index map for ATR/close lookup
        price_time_to_idx = {t: i for i, t in enumerate(fold_prices["time"].values)}

        # Iterate with configurable step to keep inference cost sane.
        step = max(1, args.bar_step)
        for i in range(0, len(fold_feats), step):
            row = fold_feats.iloc[i]
            dt = row["time"]
            date_key = dt.strftime("%Y-%m-%d")

            # Daily cap per symbol
            if daily_counts[symbol].get(date_key, 0) >= args.max_daily:
                continue

            idx = fold_feats.index[i]
            preds = pred_df.loc[idx].to_dict()
            signal = apply_mls_v3_signal_logic(preds, thresholds)
            direction = signal.get("signal", 0)
            if direction == 0:
                continue

            # Get current price and ATR
            price_idx = price_time_to_idx.get(dt)
            if price_idx is None:
                continue
            entry_price = float(fold_prices.iloc[price_idx]["close"])
            atr = float(atr_series.iloc[price_idx])
            if pd.isna(atr) or atr <= 0:
                continue

            regime = str(row.get("regime", "RANGING"))
            sl_price, tp_price = calculate_sl_tp(entry_price, direction, atr, regime)

            # Position sizing: 0.5% risk over 1.0 ATR
            inst = get_instrument_config(symbol)
            sl_pips = atr / inst["pip_size"]
            volume = risk_amount / (sl_pips * inst["pip_value_per_lot"]) if sl_pips > 0 else 0.01
            volume = max(0.01, min(volume, args.max_lot))

            expected_r = signal.get("long_expected_r" if direction == 1 else "short_expected_r", 0.0)
            trade = Trade(
                ticket=ticket,
                symbol=symbol,
                direction=direction,
                entry_time=dt,
                entry_price=entry_price,
                sl=sl_price,
                tp=tp_price,
                volume=volume,
                atr=atr,
                long_score=signal.get("long_score", 0.0),
                short_score=signal.get("short_score", 0.0),
                expected_r=expected_r,
            )
            trades.append(trade)
            ticket += 1
            daily_counts[symbol][date_key] = daily_counts[symbol].get(date_key, 0) + 1

    # Simulate exits
    for symbol in symbols:
        symbol_trades = [t for t in trades if t.symbol == symbol]
        if symbol_trades:
            simulate_trades(symbol_trades, price_data[symbol])
            for t in symbol_trades:
                if t.exit_reason in ("SL", "TP"):
                    compute_trade_pnl(t, get_instrument_config(symbol))
                    balance += t.pnl

    # Only trades that actually hit SL/TP count toward the edge estimate.
    # Trades still open at fold end are censored and ignored.
    closed_trades = [t for t in trades if t.exit_reason in ("SL", "TP")]

    total_pnl = sum(t.pnl for t in closed_trades)
    wins = [t for t in closed_trades if t.pnl > 0]
    losses = [t for t in closed_trades if t.pnl <= 0]
    win_rate = len(wins) / len(closed_trades) if closed_trades else 0.0
    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(t.pnl for t in losses) / len(losses)) if losses else 0.0
    profit_factor = (avg_win * len(wins)) / (avg_loss * len(losses)) if avg_loss > 0 and losses else 0.0
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
    r_values = [t.r_multiplier for t in closed_trades]
    avg_r = sum(r_values) / len(r_values) if r_values else 0.0

    return {
        "trades": closed_trades,
        "stats": {
            "total_trades": len(closed_trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(win_rate * 100, 2),
            "profit_factor": round(profit_factor, 2),
            "payoff_ratio": round(payoff_ratio, 2),
            "avg_r": round(avg_r, 3),
            "total_pnl": round(total_pnl, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "open_trades": len(trades) - len(closed_trades),
        },
        "final_balance": round(balance, 2),
    }


def aggregate_results(fold_results: List[Dict]) -> Dict:
    """Aggregate stats across all folds and check pass/fail gates."""
    total_trades = sum(r["stats"]["total_trades"] for r in fold_results)
    winning_trades = sum(r["stats"]["winning_trades"] for r in fold_results)
    losing_trades = sum(r["stats"]["losing_trades"] for r in fold_results)
    total_pnl = sum(r["stats"]["total_pnl"] for r in fold_results)

    total_won = sum(r["stats"]["avg_win"] * r["stats"]["winning_trades"] for r in fold_results)
    total_lost = sum(r["stats"]["avg_loss"] * r["stats"]["losing_trades"] for r in fold_results)

    win_rate = winning_trades / total_trades if total_trades else 0.0
    profit_factor = total_won / total_lost if total_lost > 0 else 0.0
    payoff_ratio = (total_won / winning_trades) / (total_lost / losing_trades) if winning_trades and losing_trades and total_lost > 0 else 0.0

    avg_r = sum(r["stats"]["avg_r"] * r["stats"]["total_trades"] for r in fold_results) / max(1, total_trades)

    gates_passed = {
        "profit_factor": profit_factor >= GATES["profit_factor"],
        "win_rate": win_rate >= GATES["win_rate"],
        "payoff_ratio": payoff_ratio >= GATES["payoff_ratio"],
        "min_trades": total_trades >= GATES["min_trades"],
    }
    overall_pass = all(gates_passed.values())

    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 3),
        "payoff_ratio": round(payoff_ratio, 3),
        "avg_r": round(avg_r, 3),
        "total_pnl": round(total_pnl, 2),
        "gates": gates_passed,
        "overall_pass": overall_pass,
        "thresholds": GATES,
    }


def save_report(report: Dict, output_name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{output_name}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    logger.info(f"[v3-wfv] Report saved: {out_path}")
    return out_path


def print_report(report: Dict):
    print("\n" + "=" * 70)
    print("V3 MODEL-DRIVEN LABEL-ALIGNED WALK-FORWARD VALIDATION")
    print("=" * 70)
    print(f"{'Total Trades:':<25} {report['total_trades']}")
    print(f"{'Win Rate:':<25} {report['win_rate']*100:.2f}%")
    print(f"{'Profit Factor:':<25} {report['profit_factor']:.2f}")
    print(f"{'Payoff Ratio:':<25} {report['payoff_ratio']:.2f}")
    print(f"{'Avg R-Multiple:':<25} {report['avg_r']:.3f}R")
    print(f"{'Total PnL:':<25} ${report['total_pnl']:,.2f}")
    print("-" * 70)
    print("Gates:")
    for gate, passed in report["gates"].items():
        status = "PASS" if passed else "FAIL"
        threshold = report["thresholds"][gate]
        value = report.get(gate, report["total_trades"] if gate == "min_trades" else "-")
        print(f"  {gate:<20} {status:<5} (value={value}, threshold={threshold})")
    print("-" * 70)
    verdict = "PASS - Edge is real; execution was the problem" if report["overall_pass"] else "FAIL - Models lack edge; retrain required"
    print(f"Verdict: {verdict}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="V3 model-driven label-aligned walk-forward validation")
    parser.add_argument("--symbols", "-s", required=True, help="Comma-separated symbols or 'FOREX'/'ALL'")
    parser.add_argument("--start", required=True, help="Hold-out start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="Hold-out end date (YYYY-MM-DD)")
    parser.add_argument("--fold-months", type=int, default=1, help="Months per walk-forward fold")
    parser.add_argument("--bar-step", type=int, default=5, help="Evaluate every Nth M1 bar (default 5)")
    parser.add_argument("--lookback-days", type=int, default=5, help="Extra history to load for feature windows")
    parser.add_argument("--balance", "-b", type=float, default=10000, help="Initial balance")
    parser.add_argument("--risk", "-r", type=float, default=0.5, help="Risk per trade %%")
    parser.add_argument("--max-daily", "-d", type=int, default=8, help="Max daily trades per symbol")
    parser.add_argument("--max-lot", type=float, default=2.0, help="Max lot size per trade")
    parser.add_argument("--output", "-o", default=None, help="Output report name")
    args = parser.parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d")
    end_date = datetime.strptime(args.end, "%Y-%m-%d")

    if args.symbols.upper() in ("ALL", "FOREX"):
        symbols = FOREX_PAIRS
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]

    # Activate label-aligned mode BEFORE importing config-sensitive modules.
    activate_ml_label_aligned_config()

    logger.info("=" * 70)
    logger.info("Starting V3 model-driven label-aligned walk-forward validation")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Hold-out period: {start_date.date()} to {end_date.date()}")
    logger.info(f"Fold size: {args.fold_months} month(s) | Bar step: {args.bar_step}")
    logger.info(f"Balance: ${args.balance:,.0f} | Risk: {args.risk}% | Max daily: {args.max_daily}")
    logger.info("=" * 70)

    # Load data with lookback so feature windows are valid from the start date.
    load_start = start_date - timedelta(days=args.lookback_days)
    data = load_m1_data(symbols, load_start, end_date)
    if not data:
        logger.error("[v3-wfv] No M1 data loaded. Exiting.")
        sys.exit(1)

    # Load models and pre-compute features.
    models, meta, features = load_models_and_features(list(data.keys()), data)
    if not models:
        logger.error("[v3-wfv] No V3 models loaded. Exiting.")
        sys.exit(1)

    # Pre-compute V3 predictions once per symbol (batch inference).
    from core.models.mls_v3_ensemble import load_mls_v3_family_weights

    family_weights = load_mls_v3_family_weights()
    predictions: Dict[str, pd.DataFrame] = {}
    for symbol in list(data.keys()):
        if symbol not in models:
            continue
        try:
            pred_df = build_prediction_df(symbol, features[symbol], models[symbol], meta[symbol], family_weights)
            predictions[symbol] = pred_df
            logger.info(f"[v3-wfv] Pre-computed V3 predictions for {symbol} ({len(pred_df):,} rows)")
        except Exception as exc:
            logger.error(f"[v3-wfv] Failed to build predictions for {symbol}: {exc}")

    if not predictions:
        logger.error("[v3-wfv] No predictions generated. Exiting.")
        sys.exit(1)

    # Generate folds over the requested hold-out period only.
    folds = generate_folds(start_date, end_date, args.fold_months)
    logger.info(f"[v3-wfv] {len(folds)} fold(s) to evaluate")

    fold_results = []
    fold_summaries = []
    for fold_start, fold_end in folds:
        res = run_fold(
            symbols=list(data.keys()),
            models=models,
            meta=meta,
            features=features,
            predictions=predictions,
            price_data=data,
            fold_start=fold_start,
            fold_end=fold_end,
            args=args,
        )
        fold_results.append(res)
        fold_summaries.append({
            "start": fold_start.isoformat(),
            "end": fold_end.isoformat(),
            "stats": res["stats"],
            "final_balance": res["final_balance"],
        })

    report = aggregate_results(fold_results)
    report["metadata"] = {
        "symbols": symbols,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "fold_months": args.fold_months,
        "bar_step": args.bar_step,
        "lookback_days": args.lookback_days,
        "initial_balance": args.balance,
        "risk_per_trade": args.risk / 100,
        "max_daily_trades": args.max_daily,
        "max_lot": args.max_lot,
        "folds": fold_summaries,
    }

    output_name = args.output or f"v3_ml_aligned_wfv_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    save_report(report, output_name)
    print_report(report)

    sys.exit(0 if report["overall_pass"] else 1)


if __name__ == "__main__":
    main()
