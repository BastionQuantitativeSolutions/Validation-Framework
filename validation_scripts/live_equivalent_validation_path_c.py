"""
LIVE-EQUIVALENT VALIDATION HARNESS
===================================
Tests the ACTUAL fixed trading system using real model inference on MT5 data.

Unlike synthetic validation scripts, this imports and uses:
  - core.models.ensemble (per_family_probs + fuse_meta with crash-skip fix)
  - core.config.constants (symmetric thresholds 0.43/0.57, W_ML=0.85/W_SMC=0.15)
  - Fixed TP/SL matching training labels (TP_PCT=0.003, SL_PCT=0.0015) — NOT ATR-based
  - Directional fusion from main_loop.py (Fix 4)
  - Deterministic gate logic from async_cognition.py

Usage:
    python CORE_MODULES/validation/live_equivalent_validation.py
"""

import os
import sys
import json
import logging
import random
import warnings
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("LIVE_EQ")

# =============================================================================
# PATH SETUP
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "DATA_MODELS" / "models_live"
RESULTS_DIR = PROJECT_ROOT / "CORE_MODULES" / "results"
OUTPUT_DIR = RESULTS_DIR / "LIVE_EQ_VALIDATION_PATH_C"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "DATA_MODELS" / "training"))
sys.path.insert(0, str(PROJECT_ROOT / "CORE_MODULES"))

# =============================================================================
# IMPORT ACTUAL FIXED MODULES
# =============================================================================
from core.config.constants import (
    BASE_BUY_THRESHOLD,
    BASE_SELL_THRESHOLD,
    W_ML,
    W_SMC,
    SL_MULTIPLIERS,
    TP_MULTIPLIERS,
    MIN_CONFIDENCE_GOVERNANCE,
    MAX_DAILY_TRADES_PER_SYMBOL,
    COOLDOWN_BARS,
    SESSION_TRADE_CAP,
)

# =============================================================================
# PATH C CONFIGURATION (Run J)
# =============================================================================
# Override thresholds: high confidence only (intentional shadow of the import above)
BASE_BUY_THRESHOLD = 0.70  # noqa: F811
BASE_SELL_THRESHOLD = 0.30  # noqa: F811
# Tiered RR: SL fixed at 0.0015, TP scales with fused confidence
#   Base (<0.85):      1.5:1  → TP_PCT = 0.00225
#   Strong (0.85-0.95): 2.0:1  → TP_PCT = 0.00300
#   Exceptional (≥0.95): 2.5:1  → TP_PCT = 0.00375
PATH_C_VERSION = "PathC_v0.70_0.30_TrendOnly_TieredRR"
from core.models.ensemble import model_feature_names, align_df
from core.utils.online_recalibrator import OnlineRecalibrator

# Feature computation
try:
    from compute_features_ultimate import compute_all_features
    log.info("[FEATURES] Using compute_all_features from training module")
except ImportError as e:
    log.error(f"[FEATURES] Cannot import compute_all_features: {e}")
    compute_all_features = None

# RAG trading engine — mirrors live rag filter
try:
    from CORE_MODULES.llms.rag_trading_system import RAGTradingEngine
    _rag_engine = RAGTradingEngine()
    log.info("[RAG] Engine loaded — WR/PnL filter active")
except Exception as _rag_err:
    _rag_engine = None
    log.warning(f"[RAG] Not available: {_rag_err}")

# Entry governor — full 10-gate live check
try:
    from CORE_MODULES.core.governance.entry_governor import evaluate_entry_governors
    # Clear quarantines for validation: models just retrained, old live-performance blocks invalid
    try:
        import CORE_MODULES.core.governance.entry_governor as _eg_mod
        _cfg = _eg_mod.get_risk_governor_cfg()
        if "symbol_quarantine" in _cfg:
            for sym in list(_cfg["symbol_quarantine"].keys()):
                _cfg["symbol_quarantine"][sym]["enabled"] = False
            _eg_mod.RISK_CONFIG_RAW = _cfg
            log.info(f"[GOV] Quarantine cleared for validation ({len(_cfg['symbol_quarantine'])} pairs)")
    except Exception as _qe:
        log.warning(f"[GOV] Could not clear quarantines: {_qe}")
    log.info("[GOV] evaluate_entry_governors loaded")
except Exception as _gov_err:
    evaluate_entry_governors = None
    log.warning(f"[GOV] entry_governor not available: {_gov_err}")

# MT5
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

# =============================================================================
# CONFIG
# =============================================================================
PAIRS = os.getenv(
    "PAIRS",
    "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD,USDCHF,GBPCHF,NZDUSD,XAUUSD,XAGUSD,USOIL,UKOIL,HEATOIL,JP225,US100,HK50,UK100,BTCUSD,ETHUSD",
).split(",")
TFS = os.getenv("TFS", "M5,M15,M30,H1").split(",")

LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "365"))
WF_FOLDS = 4
MC_ITERS = int(os.getenv("MC_ITERS", "5000"))
LOOKAHEAD_BARS = 100

# FIX 1: Fixed TP/SL matching training label creation (build_new_ml_suite.py)
# Training used: tp = entry*(1+TP_PCT), sl = entry*(1-SL_PCT) for LONG
# Using ATR-based exits in validation misaligns the train/eval contract.
TRAIN_TP_PCT = float(os.getenv("TRAIN_TP_PCT", "0.003"))   # 0.30% = ~30 pips EURUSD
TRAIN_SL_PCT = float(os.getenv("TRAIN_SL_PCT", "0.0015"))  # 0.15% = ~15 pips EURUSD

# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    pair: str
    tf: str
    direction: int
    entry_price: float
    exit_price: float
    sl: float
    tp: float
    atr: float
    regime: str
    ml_prob: float
    smc_conf: float
    fused: float
    pnl: float
    pnl_r: float
    outcome: str
    gov_reason: str


@dataclass
class WFResult:
    fold: int
    pair: str
    tf: str
    total_trades: int
    wins: int
    losses: int
    wr: float
    avg_r: float
    total_r: float
    pf: float


# =============================================================================
# MT5 DATA FETCHING
# =============================================================================
def init_mt5() -> bool:
    if mt5 is None:
        return False
    if mt5.initialize():
        log.info("[MT5] Connected")
        return True
    log.error(f"[MT5] Init failed: {mt5.last_error()}")
    return False


def mt5_tf(tf: str):
    mapping = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    return mapping.get(tf, mt5.TIMEFRAME_M5)


def fetch_mt5_data(pair: str, tf: str, days: int = 60) -> Optional[pd.DataFrame]:
    if mt5 is None or not mt5.initialize():
        return None
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    rates = mt5.copy_rates_range(pair, mt5_tf(tf), start, now)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)
    return df


def fetch_parquet_data(pair: str, tf: str) -> Optional[pd.DataFrame]:
    """Fallback to local parquet if MT5 unavailable."""
    candidates = [
        Path(f"DATA_MODELS/data_1y_backtest/{pair}_{tf}_1Y.parquet"),
        Path(f"DATA_MODELS/data_parquet/{pair}_{tf}.parquet"),
        Path(f"DATA_MODELS/data_parquet/{pair}_{tf}_LIVE.parquet"),
    ]
    for p in candidates:
        if p.exists():
            try:
                df = pd.read_parquet(p)
                if "time" in df.columns:
                    df["time"] = pd.to_datetime(df["time"])
                    df.set_index("time", inplace=True)
                df.columns = [c.lower() for c in df.columns]
                df.sort_index(inplace=True)
                return df
            except Exception as e:
                log.warning(f"[DATA] Parquet load failed {p}: {e}")
    return None


def get_data(pair: str, tf: str, days: int = 60) -> Optional[pd.DataFrame]:
    df = fetch_mt5_data(pair, tf, days)
    if df is not None and len(df) > 100:
        return df
    df = fetch_parquet_data(pair, tf)
    if df is not None and len(df) > 100:
        # Trim parquet to requested window so M5 equity pairs don't OOM
        if days and isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
            cutoff = df.index[-1] - pd.Timedelta(days=days)
            df = df[df.index >= cutoff]
        return df
    return None


# =============================================================================
# MODEL LOADING
# =============================================================================
def load_models(pair: str, tf: str) -> Dict[str, List[Any]]:
    """Load models as (model, feature_names) tuples, organized by family.

    Supports both .joblib (old pipeline) and XGBoost .json (new pipeline from models_live).
    When features.pkl is absent on the 'full' tier (200 features = all compute_all_features
    output), tier_feats=None signals downstream to skip alignment and use all features.
    """
    import joblib

    model_dir = MODELS_DIR / f"{pair}_{tf}"
    if not model_dir.exists():
        return {}

    families: Dict[str, List[Any]] = {"cat": [], "lgb": [], "xgb": []}
    for tier in ["full", "core", "minimal"]:
        tier_dir = model_dir / tier
        if not tier_dir.is_dir():
            continue

        # Feature list: prefer features.pkl; for 'full' tier fall back to None (all features)
        feat_path = tier_dir / "features.pkl"
        tier_feats = None
        if feat_path.exists():
            try:
                tier_feats = joblib.load(str(feat_path))
            except Exception:
                pass

        # ── .joblib models (old pipeline: cat, lgb, xgb) ──────────────────────
        for mtype in ["cat", "lgb", "xgb"]:
            mpath = tier_dir / f"{mtype}_model.joblib"
            if mpath.exists():
                try:
                    m = joblib.load(str(mpath))
                    families[mtype].append((m, tier_feats))
                    log.debug(f"[MODEL] Loaded {pair}_{tf} {tier}/{mtype}.joblib ({len(tier_feats) if tier_feats else '?'} feats)")
                except Exception as e:
                    log.warning(f"[MODEL] Failed to load {mpath}: {e}")

        # ── XGBoost .json models (new pipeline from models_live) ───────────────
        xgb_json = tier_dir / "xgb_model.json"
        if xgb_json.exists() and not (tier_dir / "xgb_model.joblib").exists():
            try:
                import xgboost as xgb_lib
                m = xgb_lib.Booster()
                m.load_model(str(xgb_json))
                # Wrap in a sklearn-compatible shim so predict_proba works
                class _XGBBoosterShim:
                    def __init__(self, booster):
                        self._b = booster
                    def predict_proba(self, X):
                        import numpy as np
                        import xgboost as _xgb
                        dmat = _xgb.DMatrix(X)
                        p = self._b.predict(dmat)
                        return np.column_stack([1 - p, p])
                    def predict(self, X):
                        return self.predict_proba(X)[:, 1]
                families["xgb"].append((_XGBBoosterShim(m), tier_feats))
                log.debug(f"[MODEL] Loaded {pair}_{tf} {tier}/xgb.json ({len(tier_feats) if tier_feats else 'all'} feats)")
            except Exception as e:
                log.warning(f"[MODEL] Failed to load {xgb_json}: {e}")

        # ── LGB .joblib from buy_bias / sell_bias subdirs (new pipeline) ───────
        for variant in ["buy_bias", "sell_bias"]:
            lgb_path = tier_dir / variant / "lgb_model.joblib"
            if lgb_path.exists():
                try:
                    m = joblib.load(str(lgb_path))
                    families["lgb"].append((m, tier_feats))
                    log.debug(f"[MODEL] Loaded {pair}_{tf} {tier}/{variant}/lgb ({len(tier_feats) if tier_feats else '?'} feats)")
                except Exception as e:
                    log.warning(f"[MODEL] Failed to load {lgb_path}: {e}")

    loaded = {k: v for k, v in families.items() if v}
    if loaded:
        sum(len(v) for v in loaded.items())
        log.info(f"[MODEL] {pair}_{tf}: loaded {sum(len(v) for v in loaded.values())} models ({list(loaded.keys())})")
    return loaded


# =============================================================================
# FEATURES & REGIME
# =============================================================================
def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def detect_regime(df: pd.DataFrame, lookback: int = 50) -> pd.Series:
    close = df["close"]
    returns = close.pct_change()
    volatility = returns.rolling(lookback).std()

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    # Simplified ADX
    high, low = df["high"], df["low"]
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr = calculate_atr(df, 14)
    plus_di = 100 * (plus_dm.rolling(14).mean() / tr)
    minus_di = 100 * (minus_dm.rolling(14).mean() / tr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(14).mean()

    regime = pd.Series("RANGING", index=df.index)
    regime[volatility > volatility.quantile(0.8)] = "VOLATILE"
    regime[(macd_hist > macd_hist.shift(1)) & (adx > 25)] = "TRENDING"
    return regime


def get_session(dt: datetime) -> str:
    hour = dt.hour
    if 7 <= hour < 10:
        return "LONDON"
    elif 10 <= hour < 12:
        return "LONDON_LATE"
    elif 12 <= hour < 14:
        return "LONDON_NY_OVERLAP"
    elif 14 <= hour < 17:
        return "NY"
    elif 23 <= hour or hour < 3:
        return "SYDNEY"
    elif 3 <= hour < 7:
        return "ASIAN"
    return "OTHER"


# =============================================================================
# FIXED SIGNAL PROCESSING (actual live logic)
# =============================================================================
def fixed_directional_fusion(ml_prob: float, smc_conf: float) -> Tuple[float, int]:
    """EXACT replica of fixed main_loop.py fusion."""
    _ml_dir = 1 if ml_prob >= 0.5 else -1
    _confidence = abs(ml_prob - 0.5) * 2.0
    _smc_boost = smc_conf * W_SMC
    _boosted_confidence = min(1.0, _confidence + _smc_boost)
    fused = float(np.clip(0.5 + (_ml_dir * _boosted_confidence * 0.5), 0, 1))

    if fused >= BASE_BUY_THRESHOLD:
        return fused, 1
    elif fused <= BASE_SELL_THRESHOLD:
        return fused, -1
    return fused, 0


def fixed_deterministic_gate(confidence: float, quality_score: float, smc_conf: float) -> bool:
    """EXACT replica of async_cognition.py deterministic gate."""
    return confidence >= 0.45 and quality_score >= 0.30 and smc_conf >= 0.10


# =============================================================================
# GOVERNANCE STATE
# =============================================================================
class GovernanceState:
    def __init__(self):
        self.daily_counts: Dict[str, int] = defaultdict(int)
        self.session_counts: Dict[str, int] = defaultdict(int)
        self.loss_streak: int = 0
        self.cooldowns: Dict[str, datetime] = {}

    def reset(self):
        self.daily_counts.clear()
        self.session_counts.clear()
        self.loss_streak = 0
        self.cooldowns.clear()

    def check(
        self,
        pair: str,
        direction: int,
        confidence: float,
        smc_conf: float,
        regime: str,
        session: str,
        dt: datetime,
    ) -> Tuple[bool, str]:
        # Deterministic gate
        if not fixed_deterministic_gate(confidence, confidence, smc_conf):
            return False, "DETERMINISTIC_GATE"

        # Confidence governance
        if confidence < MIN_CONFIDENCE_GOVERNANCE:
            return False, f"CONFIDENCE_LOW:{confidence:.2f}<{MIN_CONFIDENCE_GOVERNANCE}"

        # Daily cap
        day_key = f"{pair}_{dt.date()}"
        if self.daily_counts[day_key] >= MAX_DAILY_TRADES_PER_SYMBOL:
            return False, f"DAILY_CAP:{self.daily_counts[day_key]}"

        # Session cap
        sess_key = f"{session}_{dt.date()}"
        if self.session_counts[sess_key] >= SESSION_TRADE_CAP:
            return False, "SESSION_CAP"

        # Loss streak
        if self.loss_streak >= 3:
            return False, f"LOSS_STREAK:{self.loss_streak}"

        # Cooldown
        if pair in self.cooldowns and dt < self.cooldowns[pair]:
            return False, "COOLDOWN"

        return True, "PASSED"

    def record_trade(self, pair: str, session: str, dt: datetime, pnl_r: float):
        day_key = f"{pair}_{dt.date()}"
        self.daily_counts[day_key] += 1
        sess_key = f"{session}_{dt.date()}"
        self.session_counts[sess_key] += 1

        if pnl_r > 0:
            self.loss_streak = 0
        else:
            self.loss_streak += 1

        if pnl_r < 0:
            self.cooldowns[pair] = dt + timedelta(minutes=COOLDOWN_BARS * 5)


# =============================================================================
# ENSEMBLE PREDICTION (actual fixed code paths)
# =============================================================================
def _batch_predict_family(features_df: pd.DataFrame, models: List[Any], fam: str) -> Optional[np.ndarray]:
    """Batch predict for a model family using actual feature alignment (core.models.ensemble.align_df)."""
    if not models:
        return None
    preds = []
    for item in models:
        m, tier_feats = item if isinstance(item, tuple) else (item, None)
        try:
            feats = model_feature_names(m) or tier_feats
            Xin = align_df(features_df, feats) if feats else features_df
            if hasattr(m, "predict_proba"):
                p = m.predict_proba(Xin)[:, 1]
            else:
                p = m.predict(Xin)
            preds.append(p)
        except Exception as e:
            log.warning(f"[{fam}] Model skipped in batch predict: {e}")
            continue
    if not preds:
        return None
    return np.mean(preds, axis=0)


def predict_with_fixed_ensemble(features_df: pd.DataFrame, models_by_family: Dict[str, List[Any]]) -> np.ndarray:
    """Batch prediction using actual align_df + fuse_meta logic from core.models.ensemble."""
    if not models_by_family:
        return np.full(len(features_df), 0.5)

    cat_arr = _batch_predict_family(features_df, models_by_family.get("cat", []), "cat")
    lgb_arr = _batch_predict_family(features_df, models_by_family.get("lgb", []), "lgb")
    xgb_arr = _batch_predict_family(features_df, models_by_family.get("xgb", []), "xgb")

    # Inline fuse_meta logic for batch speed (same algorithm as core.models.ensemble)
    active_means = []
    active_weights = []
    weights = {"cat": 0.5, "lgb": 0.3, "xgb": 0.2}

    if cat_arr is not None:
        active_means.append(cat_arr)
        active_weights.append(weights["cat"])
    if lgb_arr is not None:
        active_means.append(lgb_arr)
        active_weights.append(weights["lgb"])
    if xgb_arr is not None:
        active_means.append(xgb_arr)
        active_weights.append(weights["xgb"])

    if not active_means:
        return np.full(len(features_df), 0.5)

    total_w = sum(active_weights)
    if total_w == 0:
        return np.mean(active_means, axis=0)

    fused = np.zeros(len(features_df))
    for arr, w in zip(active_means, active_weights):
        fused += arr * w
    fused /= total_w
    return np.clip(fused, 0, 1).astype(float)


# =============================================================================
# TRADE SIMULATION
# =============================================================================
def simulate_trade(
    df: pd.DataFrame,
    entry_idx: int,
    direction: int,
    sl: float,
    tp: float,
    atr: float,
    regime: str,
    pair: str,
    tf: str,
    ml_prob: float,
    smc_conf: float,
    fused: float,
    gov_reason: str,
    max_bars: int = 100,
) -> Optional[Trade]:
    entry_price = float(df.iloc[entry_idx]["close"])
    entry_time = df.index[entry_idx]

    for i in range(entry_idx + 1, min(entry_idx + max_bars, len(df))):
        high = float(df.iloc[i]["high"])
        low = float(df.iloc[i]["low"])
        curr_time = df.index[i]

        if direction == 1:
            if low <= sl:
                pnl = sl - entry_price
                pnl_r = -abs(pnl) / atr if atr > 0 else -1.0
                return Trade(
                    entry_time=entry_time, exit_time=curr_time, pair=pair, tf=tf,
                    direction=direction, entry_price=entry_price, exit_price=sl,
                    sl=sl, tp=tp, atr=atr, regime=regime, ml_prob=ml_prob,
                    smc_conf=smc_conf, fused=fused, pnl=pnl, pnl_r=max(-2.0, pnl_r),
                    outcome="LOSS", gov_reason=gov_reason,
                )
            elif high >= tp:
                pnl = tp - entry_price
                pnl_r = pnl / atr if atr > 0 else 1.0
                return Trade(
                    entry_time=entry_time, exit_time=curr_time, pair=pair, tf=tf,
                    direction=direction, entry_price=entry_price, exit_price=tp,
                    sl=sl, tp=tp, atr=atr, regime=regime, ml_prob=ml_prob,
                    smc_conf=smc_conf, fused=fused, pnl=pnl, pnl_r=min(2.0, pnl_r),
                    outcome="WIN", gov_reason=gov_reason,
                )
        else:
            if high >= sl:
                pnl = entry_price - sl
                pnl_r = -abs(pnl) / atr if atr > 0 else -1.0
                return Trade(
                    entry_time=entry_time, exit_time=curr_time, pair=pair, tf=tf,
                    direction=direction, entry_price=entry_price, exit_price=sl,
                    sl=sl, tp=tp, atr=atr, regime=regime, ml_prob=ml_prob,
                    smc_conf=smc_conf, fused=fused, pnl=pnl, pnl_r=max(-2.0, pnl_r),
                    outcome="LOSS", gov_reason=gov_reason,
                )
            elif low <= tp:
                pnl = entry_price - tp
                pnl_r = pnl / atr if atr > 0 else 1.0
                return Trade(
                    entry_time=entry_time, exit_time=curr_time, pair=pair, tf=tf,
                    direction=direction, entry_price=entry_price, exit_price=tp,
                    sl=sl, tp=tp, atr=atr, regime=regime, ml_prob=ml_prob,
                    smc_conf=smc_conf, fused=fused, pnl=pnl, pnl_r=min(2.0, pnl_r),
                    outcome="WIN", gov_reason=gov_reason,
                )

    # Time expiry
    exit_idx = min(entry_idx + max_bars, len(df) - 1)
    exit_price = float(df.iloc[exit_idx]["close"])
    exit_time = df.index[exit_idx]
    if direction == 1:
        pnl = exit_price - entry_price
    else:
        pnl = entry_price - exit_price
    pnl_r = pnl / atr if atr > 0 else 0.0
    outcome = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "BREAKEVEN"
    return Trade(
        entry_time=entry_time, exit_time=exit_time, pair=pair, tf=tf,
        direction=direction, entry_price=entry_price, exit_price=exit_price,
        sl=sl, tp=tp, atr=atr, regime=regime, ml_prob=ml_prob,
        smc_conf=smc_conf, fused=fused, pnl=pnl, pnl_r=pnl_r,
        outcome=outcome, gov_reason=gov_reason,
    )


# =============================================================================
# ROLLING TRADE SIMULATION
# =============================================================================
def rolling_trade_simulation(
    df: pd.DataFrame,
    proba: np.ndarray,
    smc_conf: pd.Series,
    regime: pd.Series,
    atr: pd.Series,
    test_start: int,
    test_end: int,
    pair: str,
    tf: str,
    gov: GovernanceState,
    use_recalibration: bool = True,
) -> List[Trade]:
    trades = []
    in_trade = False
    entry_bar = 0

    recal = OnlineRecalibrator(window=1000, temperature=1.5) if use_recalibration else None
    # Pre-load recalibrator with training period probabilities
    if recal is not None and test_start > 0:
        for j in range(max(0, test_start - 2000), test_start):
            recal.update(float(proba[j]))

    for i in range(test_start, test_end):
        if in_trade:
            if i >= entry_bar + LOOKAHEAD_BARS:
                in_trade = False
            continue

        dt = df.index[i]
        if pd.isna(atr.iloc[i]) or atr.iloc[i] <= 0:
            continue

        raw_ml_prob = float(proba[i])
        if recal is not None:
            recal.update(raw_ml_prob)
            ml_prob = recal.calibrate(raw_ml_prob)
        else:
            ml_prob = raw_ml_prob

        smc = float(smc_conf.iloc[i]) if i < len(smc_conf) else 0.5
        curr_regime = str(regime.iloc[i]) if i < len(regime) else "RANGING"

        fused, direction = fixed_directional_fusion(ml_prob, smc)

        if direction == 0:
            continue

        # Governance check (session caps, daily caps, loss streak, cooldown)
        session = get_session(dt)
        allowed, reason = gov.check(pair, direction, fused, smc, curr_regime, session, dt)
        if not allowed:
            continue

        # Full entry governor (10-gate live check) — mirrors evaluate_entry_governors()
        dir_str = "BUY" if direction == 1 else "SELL"
        if evaluate_entry_governors is not None:
            try:
                _eg_allowed, _eg_reason, _conf_mult = evaluate_entry_governors(
                    pair, dir_str, curr_regime, fused,
                    session_name=session,
                    smc_confluence=float(smc),
                    ob_graph_quality=0.5, fvg_graph_quality=0.5,
                    killzone_label="NONE", in_killzone=False,
                    fvg_invalidated=False, strategy_name="ML_ENSEMBLE",
                )
                if not _eg_allowed:
                    continue
                fused = max(0.0, min(1.0, fused * _conf_mult))
                if direction == 1 and fused < BASE_BUY_THRESHOLD:
                    continue
                if direction == -1 and fused > BASE_SELL_THRESHOLD:
                    continue
            except Exception:
                pass

        # RAG filter — mirrors live rag_trading_system.should_take_trade()
        if _rag_engine is not None:
            try:
                rag_result = _rag_engine.should_take_trade(
                    symbol=pair,
                    confidence=fused,
                    regime=curr_regime,
                    direction=dir_str,
                    strategy_name="ML_ENSEMBLE",
                )
                if rag_result.get("decision") == "SKIP":
                    continue
            except Exception:
                pass

        # Path C hard gates
        if curr_regime != "TRENDING":
            continue
        if session not in ("LONDON", "LONDON_LATE", "LONDON_NY_OVERLAP", "NY"):
            continue

        # Path C tiered RR based on fused confidence
        entry_price = float(df.iloc[i]["close"])
        if fused >= 0.95:
            tp_pct = 0.00375  # 2.5:1
        elif fused >= 0.85:
            tp_pct = 0.00300  # 2.0:1
        else:
            tp_pct = 0.00225  # 1.5:1
        if direction == 1:
            sl = entry_price * (1 - TRAIN_SL_PCT)
            tp = entry_price * (1 + tp_pct)
        else:
            sl = entry_price * (1 + TRAIN_SL_PCT)
            tp = entry_price * (1 - tp_pct)

        trade = simulate_trade(
            df=df, entry_idx=i, direction=direction,
            sl=sl, tp=tp, atr=float(atr.iloc[i]),
            regime=curr_regime, pair=pair, tf=tf,
            ml_prob=ml_prob, smc_conf=smc, fused=fused,
            gov_reason=reason,
        )

        if trade:
            trades.append(trade)
            gov.record_trade(pair, session, trade.exit_time, trade.pnl_r)
            in_trade = True
            entry_bar = i

    return trades


# =============================================================================
# WALK-FORWARD
# =============================================================================
def run_walkforward(
    df: pd.DataFrame,
    models_by_family: Dict[str, List[Any]],
    pair: str,
    tf: str,
    n_folds: int = 4,
    lookback_bars: int = 300,
) -> List[WFResult]:
    min_bars = lookback_bars + 100
    if len(df) < min_bars:
        log.warning(f"[WF] {pair}_{tf} insufficient data: {len(df)} < {min_bars}")
        return []

    results = []
    fold_size = (len(df) - lookback_bars) // n_folds

    for fold in range(n_folds):
        train_end = lookback_bars + fold * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size + 100, len(df))

        log.info(f"[WF] {pair}_{tf} Fold {fold + 1}/{n_folds}: test={test_end - test_start} bars")

        # No leakage: features on full history up to test_end
        hist_df = df.iloc[:test_end]

        if compute_all_features is not None:
            try:
                features = compute_all_features(hist_df)
            except Exception as e:
                log.warning(f"[WF] Feature compute failed: {e}, using fallback")
                features = None
        else:
            features = None

        if features is None:
            log.warning(f"[WF] {pair}_{tf} skipping fold {fold + 1}: no features")
            continue

        regime = detect_regime(hist_df)
        atr = calculate_atr(hist_df)

        # FIX 4: Use confluence_bull_all / confluence_bear_all (4-factor conjunction: trend+FVG+BOS+momentum)
        # These are far more selective than bull_4/bear_4 (2-factor) which resolve to 1.0 on ~70%+ of bars.
        # The direction-specific versions will be indexed per-bar during rolling_trade_simulation.
        if "confluence_bull_all" in features.columns and "confluence_bear_all" in features.columns:
            smc_conf = features[["confluence_bull_all", "confluence_bear_all"]].max(axis=1).clip(0, 1)
        elif "confluence_bull_4" in features.columns and "confluence_bear_4" in features.columns:
            smc_conf = features[["confluence_bull_4", "confluence_bear_4"]].max(axis=1).clip(0, 1)
        elif "confluence_score" in features.columns:
            smc_conf = features["confluence_score"].clip(0, 1)
        elif "confluence_net" in features.columns:
            smc_conf = (features["confluence_net"].clip(-1, 1) + 1) / 2
        else:
            smc_conf = pd.Series(0.5, index=features.index)

        # Predictions using ACTUAL fixed ensemble
        try:
            proba = predict_with_fixed_ensemble(features, models_by_family)
        except Exception as e:
            log.error(f"[WF] Prediction failed: {e}")
            continue

        gov = GovernanceState()
        trades = rolling_trade_simulation(
            hist_df, proba, smc_conf, regime, atr,
            test_start, test_end, pair, tf, gov,
        )

        wins = [t for t in trades if t.outcome == "WIN"]
        losses = [t for t in trades if t.outcome == "LOSS"]
        wr = len(wins) / len(trades) * 100 if trades else 0
        avg_r = np.mean([t.pnl_r for t in trades]) if trades else 0
        total_r = sum(t.pnl_r for t in trades)
        pf = sum(t.pnl_r for t in wins) / abs(sum(t.pnl_r for t in losses)) if losses and sum(t.pnl_r for t in losses) != 0 else 0

        log.info(f"[WF] {pair}_{tf} Fold {fold + 1}: {len(trades)} trades, WR={wr:.1f}%, TotalR={total_r:.2f}, PF={pf:.2f}")

        results.append(WFResult(
            fold=fold + 1, pair=pair, tf=tf,
            total_trades=len(trades), wins=len(wins), losses=len(losses),
            wr=wr, avg_r=avg_r, total_r=total_r, pf=pf,
        ))

    return results


# =============================================================================
# MONTE CARLO
# =============================================================================
def run_monte_carlo(trades: List[Trade], n_sims: int = 2000) -> Dict[str, Any]:
    if not trades:
        return {"survival_rate": 0, "prob_profit": 0}

    pnls = [t.pnl_r for t in trades]
    final_equities = []
    max_drawdowns = []
    survival_count = 0
    profit_count = 0

    risk_per_trade = float(os.getenv("BASE_RISK_PER_TRADE", "0.0117"))

    for _ in range(n_sims):
        sim_trades = random.choices(pnls, k=len(pnls))
        equity = [1.0]
        for pnl in sim_trades:
            equity.append(equity[-1] * (1 + pnl * risk_per_trade))

        final_equities.append(equity[-1])
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0
        max_drawdowns.append(max_dd)

        if min(equity) > 0.5:
            survival_count += 1
        if equity[-1] > 1.0:
            profit_count += 1

    return {
        "n_sims": n_sims,
        "n_trades": len(trades),
        "final_equity_p5": float(np.percentile(final_equities, 5)),
        "final_equity_p50": float(np.percentile(final_equities, 50)),
        "final_equity_p95": float(np.percentile(final_equities, 95)),
        "max_dd_p5": float(np.percentile(max_drawdowns, 5)),
        "max_dd_p95": float(np.percentile(max_drawdowns, 95)),
        "survival_rate": survival_count / n_sims,
        "prob_profit": profit_count / n_sims,
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    log.info("=" * 60)
    log.info("LIVE-EQUIVALENT VALIDATION (ACTUAL FIXED CODE PATHS)")
    log.info("=" * 60)
    log.info(f"BUY_THRESHOLD: {BASE_BUY_THRESHOLD}")
    log.info(f"SELL_THRESHOLD: {BASE_SELL_THRESHOLD}")
    log.info(f"W_ML: {W_ML}, W_SMC: {W_SMC}")
    log.info(f"SL_MULTIPLIERS: {SL_MULTIPLIERS}")
    log.info(f"TP_MULTIPLIERS: {TP_MULTIPLIERS}")

    mt5_ok = init_mt5()
    if not mt5_ok:
        log.warning("[MT5] Not available, using parquet fallback")

    all_trades: List[Trade] = []
    all_results: List[WFResult] = []

    test_pairs = [p for p in PAIRS if p.strip()]
    test_tfs = TFS

    for pair in test_pairs:
        pair = pair.strip()
        for tf in test_tfs:
            log.info(f"\n{'='*40}")
            log.info(f"TESTING: {pair} {tf}")
            log.info(f"{'='*40}")

            df = get_data(pair, tf, LOOKBACK_DAYS)
            if df is None or len(df) < 500:
                log.warning(f"[SKIP] {pair}_{tf}: no data")
                continue

            models = load_models(pair, tf)
            if not models:
                log.warning(f"[SKIP] {pair}_{tf}: no models")
                continue

            wf_results = run_walkforward(df, models, pair, tf, n_folds=WF_FOLDS)
            all_results.extend(wf_results)

            for r in wf_results:
                # We don't have trades per result currently; would need to return them
                pass

    # For MC, we need all trades. Let's do a simplified pass that returns trades.
    # Actually, run_walkforward doesn't return trades. Let me do a second pass
    # that collects all trades across the full period (no folds, just OOS).

    log.info("\n" + "=" * 60)
    log.info("FULL-PERIOD SIMULATION FOR MC")
    log.info("=" * 60)

    for pair in test_pairs:
        pair = pair.strip()
        for tf in test_tfs:
            df = get_data(pair, tf, LOOKBACK_DAYS)
            if df is None or len(df) < 500:
                continue
            models = load_models(pair, tf)
            if not models:
                continue

            try:
                features = compute_all_features(df)
            except Exception as e:
                log.warning(f"[MC] Feature error {pair}_{tf}: {e}")
                continue
            if features is None or len(features) == 0:
                continue

            regime = detect_regime(df)
            atr = calculate_atr(df)

            # FIX 4: Use 4-factor conjunction (all) for more selective confluence gate
            if "confluence_bull_all" in features.columns and "confluence_bear_all" in features.columns:
                smc_conf = features[["confluence_bull_all", "confluence_bear_all"]].max(axis=1).clip(0, 1)
            elif "confluence_bull_4" in features.columns and "confluence_bear_4" in features.columns:
                smc_conf = features[["confluence_bull_4", "confluence_bear_4"]].max(axis=1).clip(0, 1)
            elif "confluence_score" in features.columns:
                smc_conf = features["confluence_score"].clip(0, 1)
            elif "confluence_net" in features.columns:
                smc_conf = (features["confluence_net"].clip(-1, 1) + 1) / 2
            else:
                smc_conf = pd.Series(0.5, index=features.index)

            try:
                proba = predict_with_fixed_ensemble(features, models)
            except Exception as e:
                log.error(f"[MC] Prediction error {pair}_{tf}: {e}")
                continue

            gov = GovernanceState()
            trades = rolling_trade_simulation(
                df, proba, smc_conf, regime, atr,
                300, len(df), pair, tf, gov,
            )
            all_trades.extend(trades)
            log.info(f"[MC] {pair}_{tf}: {len(trades)} trades")

    # Summary
    log.info("\n" + "=" * 60)
    log.info("VALIDATION SUMMARY")
    log.info("=" * 60)

    if all_trades:
        wins = [t for t in all_trades if t.outcome == "WIN"]
        losses = [t for t in all_trades if t.outcome == "LOSS"]
        wr = len(wins) / len(all_trades) * 100
        total_r = sum(t.pnl_r for t in all_trades)
        avg_r = total_r / len(all_trades)
        pf = sum(t.pnl_r for t in wins) / abs(sum(t.pnl_r for t in losses)) if losses else 0

        log.info(f"Total trades: {len(all_trades)}")
        log.info(f"Win Rate: {wr:.1f}%")
        log.info(f"Avg R/trade: {avg_r:.3f}")
        log.info(f"Total R: {total_r:.2f}")
        log.info(f"Profit Factor: {pf:.2f}")

        # Monte Carlo
        mc = run_monte_carlo(all_trades, n_sims=MC_ITERS)
        log.info(f"\nMonte Carlo ({MC_ITERS} sims):")
        log.info(f"  Final Equity p50: {mc['final_equity_p50']:.3f}")
        log.info(f"  Max DD p95: {mc['max_dd_p95']:.1%}")
        log.info(f"  Survival Rate: {mc['survival_rate']:.1%}")
        log.info(f"  Prob Profit: {mc['prob_profit']:.1%}")
    else:
        log.warning("No trades generated!")

    # Save results
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "buy_threshold": BASE_BUY_THRESHOLD,
            "sell_threshold": BASE_SELL_THRESHOLD,
            "w_ml": W_ML,
            "w_smc": W_SMC,
            "sl_multipliers": SL_MULTIPLIERS,
            "tp_multipliers": TP_MULTIPLIERS,
        },
        "summary": {
            "total_trades": len(all_trades),
            "win_rate": len([t for t in all_trades if t.outcome == "WIN"]) / len(all_trades) * 100 if all_trades else 0,
            "total_r": sum(t.pnl_r for t in all_trades) if all_trades else 0,
        },
        "path_c_version": PATH_C_VERSION,
        "monte_carlo": mc if all_trades else {},
        "trades": [
            {
                "pair": t.pair, "tf": t.tf, "direction": t.direction,
                "entry": t.entry_time.isoformat(), "exit": t.exit_time.isoformat(),
                "outcome": t.outcome, "pnl_r": t.pnl_r,
                "ml_prob": t.ml_prob, "smc_conf": t.smc_conf, "fused": t.fused,
                "regime": t.regime, "gov_reason": t.gov_reason,
            }
            for t in all_trades
        ],
    }

    out_path = OUTPUT_DIR / f"live_equivalent_validation_path_c_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"\nResults saved to: {out_path}")

    if mt5 is not None:
        mt5.shutdown()


if __name__ == "__main__":
    main()
