import json
import logging
import numpy as np
import pandas as pd
import sys
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

# Import shared thread-safety lock for gradient boosting models on Windows
# (XGBoost, LightGBM, CatBoost use OpenMP/native C++ that crashes with concurrent access)
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
from core.models.ensemble import GBM_PREDICT_LOCK as _GBM_PREDICT_LOCK

# ── ML Regime Classifier paths ────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_ML_REGIME_DIR = _PROJECT_ROOT / "DATA_MODELS" / "models_live" / "regime_classifier"
_ML_MODEL_PATH = _ML_REGIME_DIR / "cat_model.joblib"
_ML_FEATURES_PATH = _ML_REGIME_DIR / "features.pkl"
_ML_MANIFEST_PATH = _ML_REGIME_DIR / "training_manifest.json"

_LABEL_MAP = {"RANGING": 0, "TRENDING": 1, "VOLATILE": 2, "BREAKOUT": 3}
_IDX_MAP = {v: k for k, v in _LABEL_MAP.items()}

# ── Regime WR cache (loaded from training manifest) ───────────────────────────
# Maps regime name → MC p50 WR (precision when predicting that regime).
# Populated once on first call to get_regime_wr(); falls back to 0.65 if unavailable.
_REGIME_WR_CACHE: Optional[Dict[str, float]] = None


def _load_regime_wr_cache() -> Dict[str, float]:
    """Load per-regime MC p50 WR from the training manifest (once, cached)."""
    global _REGIME_WR_CACHE
    if _REGIME_WR_CACHE is not None:
        return _REGIME_WR_CACHE
    fallback = {k: 0.65 for k in _LABEL_MAP}
    try:
        if not _ML_MANIFEST_PATH.exists():
            _REGIME_WR_CACHE = fallback
            return _REGIME_WR_CACHE
        manifest = json.loads(_ML_MANIFEST_PATH.read_text(encoding="utf-8"))
        mc_intervals = manifest.get("mc_wr_intervals", {})
        if not mc_intervals:
            # v1 manifest — no WR data yet
            _REGIME_WR_CACHE = fallback
            return _REGIME_WR_CACHE
        cache = {}
        for regime, iv in mc_intervals.items():
            if isinstance(iv, dict) and "p50" in iv:
                cache[regime.upper()] = float(iv["p50"])
        _REGIME_WR_CACHE = cache if cache else fallback
        logging.info(f"[regime-wr] Loaded MC p50 WR from manifest: {_REGIME_WR_CACHE}")
    except Exception as e:
        logging.warning(f"[regime-wr] Failed to load manifest WR ({e}) — using fallback 0.65")
        _REGIME_WR_CACHE = fallback
    return _REGIME_WR_CACHE


def get_regime_wr(regime: str) -> float:
    """
    Return the MC p50 Win Rate for a given regime label.

    This is the regime classifier's precision for that class: when it predicts
    regime X, how often does the actual forward price behaviour confirm X?
    Falls back to 0.65 (neutral) if the manifest has no data for that regime.
    """
    cache = _load_regime_wr_cache()
    return cache.get(str(regime).upper(), 0.65)


class MLRegimeDetector:
    """
    ML-based regime detector using a trained CatBoost classifier.
    Predicts regime from the full 349-feature set.
    Falls back to LiveRegimeDetector if model is unavailable.
    """

    def __init__(self):
        self._model = None
        self._features = None
        self._fallback = LiveRegimeDetector()
        self._ml_loaded = False
        self._load_model()

    def _load_model(self):
        try:
            import joblib

            if not _ML_MODEL_PATH.exists() or not _ML_FEATURES_PATH.exists():
                logging.info("[MLRegimeDetector] No trained model found — using rule-based fallback.")
                return
            self._model = joblib.load(str(_ML_MODEL_PATH))
            self._features = joblib.load(str(_ML_FEATURES_PATH))
            self._ml_loaded = True
            logging.info(f"[MLRegimeDetector] Loaded ML regime model ({len(self._features)} features).")
        except Exception as e:
            logging.warning(f"[MLRegimeDetector] Load failed ({e}) — falling back to rule-based.")

    def detect_regime_vectorized(self, df: pd.DataFrame, feats_df: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (regime_labels, r_direction, strength).

        If `feats_df` is provided (pre-computed 349-feature frame) and the ML
        model is loaded, uses ML predictions.  Otherwise falls back to the
        rule-based LiveRegimeDetector.

        regime_dir (+1 / -1 / 0) is always derived from ADX DI± regardless of
        the regime label source, since the ML model doesn't output direction.
        """
        # Always compute rule-based direction (DI+/DI-)
        rule_regime, rule_dir, rule_strength = self._fallback.detect_regime_vectorized(df)

        if not self._ml_loaded or feats_df is None:
            return rule_regime, rule_dir, rule_strength

        try:
            # Align features
            X = feats_df.copy()
            for col in self._features:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[self._features]
            X = X.fillna(X.median()).replace([np.inf, -np.inf], 0.0)

            # Predict class probabilities (with thread-safety lock for GBM models)
            cls_name = str(type(self._model))
            _needs_lock = (
                "XGB" in cls_name or 
                "LGB" in cls_name or 
                "LightGBM" in cls_name or
                "Cat" in cls_name or
                "CatBoost" in cls_name
            )
            with _GBM_PREDICT_LOCK if _needs_lock else threading.Lock():
                probs = self._model.predict_proba(X)  # shape (n, 4)
            pred_idx = probs.argmax(axis=1).astype(np.int8)

            # Regime label strings
            regime = np.array([_IDX_MAP[int(i)] for i in pred_idx], dtype=object)

            # Strength = max class probability
            strength = probs.max(axis=1).astype(np.float32)

            # Direction from rule-based (ADX DI+/DI-)
            return regime, rule_dir, strength

        except Exception as e:
            logging.warning(f"[MLRegimeDetector] Prediction failed ({e}) — using rule-based.")
            return rule_regime, rule_dir, rule_strength

    @property
    def ml_available(self) -> bool:
        return self._ml_loaded


class LiveRegimeDetector:
    """
    Rule-based vectorized regime detector (ADX + Bollinger + Volatility percentile).
    Used as fallback when ML model is not trained yet.
    """

    def __init__(self):
        self.trend_breakout = 0.6
        self.trend_trending = 0.4  # Synced with regime_thresholds.json
        self.range_ranging = 0.5  # Synced with regime_thresholds.json
        self.vol_breakout = 0.6
        self.vol_volatile = 0.55  # Synced with regime_thresholds.json

    def detect_regime_vectorized(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        n = len(df)

        period = 14
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))

        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        def _ewm(data, span):
            return pd.Series(data).ewm(span=span, adjust=False).mean().values

        atr = _ewm(tr, period)
        plus_di = 100 * _ewm(plus_dm, period) / (atr + 1e-9)
        minus_di = 100 * _ewm(minus_dm, period) / (atr + 1e-9)
        denom = plus_di + minus_di
        dx = np.where(denom != 0, 100 * np.abs(plus_di - minus_di) / denom, 0.0)
        adx = _ewm(dx, period)

        slope_window = 50

        def _slope(y):
            if len(y) < slope_window:
                return 0.0
            x = np.arange(len(y))
            xm = x.mean()
            ym = y.mean()
            den = np.sum((x - xm) ** 2)
            return float(np.sum((x - xm) * (y - ym)) / den) if den > 0 else 0.0

        slope_raw = pd.Series(close).rolling(slope_window).apply(_slope, raw=True).fillna(0).values
        close_mean = pd.Series(close).rolling(slope_window).mean().fillna(1).values
        slope_str = np.abs(slope_raw) / (close_mean + 1e-8)
        trend_val = np.clip((adx / 50.0 + slope_str * 10) / 2.0, 0.0, 1.0)

        direction = np.zeros(n, dtype=np.int8)
        direction[plus_di > minus_di * 1.1] = 1
        direction[minus_di > plus_di * 1.1] = -1

        sma20 = pd.Series(close).rolling(20).mean().values
        std20 = pd.Series(close).rolling(20).std().values
        bb_width = (std20 * 2) / (sma20 + 1e-9)
        range_pct = pd.Series(bb_width).rolling(120).rank(pct=True).fillna(0.5).values
        range_str = 1.0 - range_pct
        vol_pct = pd.Series(atr).rolling(40).rank(pct=True).fillna(0.5).values

        regime = np.full(n, "RANGING", dtype=object)
        strength = np.full(n, 0.5, dtype=np.float32)
        r_dir = np.zeros(n, dtype=np.int8)

        m_bo = (trend_val > self.trend_breakout) & (vol_pct > self.vol_breakout)
        regime[m_bo] = "BREAKOUT"
        strength[m_bo] = np.clip((trend_val[m_bo] + vol_pct[m_bo]) / 2.0, 0.0, 1.0)
        r_dir[m_bo] = direction[m_bo]

        m_tr = (trend_val > self.trend_trending) & (range_str < self.range_ranging) & ~m_bo
        regime[m_tr] = "TRENDING"
        strength[m_tr] = trend_val[m_tr]
        r_dir[m_tr] = direction[m_tr]

        m_rng = (range_str > self.range_ranging) & (trend_val < (self.trend_trending - 0.1)) & ~m_bo & ~m_tr
        regime[m_rng] = "RANGING"
        strength[m_rng] = range_str[m_rng]

        m_vol = (vol_pct > self.vol_volatile) & ~m_bo & ~m_tr & ~m_rng
        regime[m_vol] = "VOLATILE"
        strength[m_vol] = vol_pct[m_vol]

        return regime, r_dir, strength
