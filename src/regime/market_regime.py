"""
# Author: JG
Canonical market regime engine.
Trend: slope + ADX
Volatility: ATR percentile
Consolidation: range compression
"""

import logging
import os
import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import hashlib

log = logging.getLogger(__name__)

# Lazy MT5 import
_mt5 = None


def _get_mt5():
    global _mt5
    if _mt5 is None:
        try:
            import MetaTrader5 as mt5

            _mt5 = mt5
        except ImportError:
            log.warning("MetaTrader5 not available")
            return None
    return _mt5


@dataclass
class RegimeState:
    """Current market regime"""

    regime: str  # "trending", "ranging", "volatile", "breakout"
    strength: float  # 0.0-1.0
    trend_direction: int  # 1 (up), -1 (down), 0 (sideways)
    volatility_percentile: float  # 0.0-1.0
    adr_ratio: float  # Average Daily Range ratio
    confidence: float  # 0.0-1.0


class MarketRegimeDetector:
    """
    Detects market regimes to adjust strategy behavior
    """

    def __init__(self, results_dir: Optional[Path] = None):
        if results_dir is None:
            results_dir = Path(__file__).resolve().parents[1] / "results"
        self.results_dir = Path(results_dir)
        self.cache_file = self.results_dir / "regime_cache.json"
        self.regime_cache: Dict[str, RegimeState] = {}
        self.trained_regime_state: Dict[str, Dict] = {}
        self.trained_volatility_state: Dict[str, Dict] = {}
        self.using_trained_instance = False
        self.live_incremental_updates = True
        self.last_live_update_applied = False
        self.last_regime_source = "live_recompute"
        self.instance_id = id(self)
        self.bound_training_hash: Optional[str] = None
        self.thresholds = self._load_thresholds()
        # FIX CRITICAL-008: Add debounce timer for cache saves
        self._last_save_time = 0.0
        self._cache_save_interval = 60.0  # Save at most once per 60 seconds

        # Load cache
        self._load_cache()

    @staticmethod
    def _env_true(name: str, default: bool = False) -> bool:
        value = os.getenv(name)
        if value is None:
            return bool(default)
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _serialize_cache(self) -> Dict[str, Dict[str, float]]:
        return {
            symbol: {
                "regime": state.regime,
                "strength": float(state.strength),
                "direction": int(state.trend_direction),
                "volatility": float(state.volatility_percentile),
                "adr_ratio": float(state.adr_ratio),
                "confidence": float(state.confidence),
            }
            for symbol, state in self.regime_cache.items()
        }

    def get_runtime_cache_hash(self) -> str:
        serialized = self._serialize_cache()
        canonical = json.dumps(serialized, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _training_payload_hash(self) -> str:
        payload = {
            "regime": self.trained_regime_state,
            "volatility": self.trained_volatility_state,
        }
        canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _regime_label_from_training(self, value: Any) -> str:
        raw = str(value or "").upper()
        if "RANG" in raw:
            return "ranging"
        if "VOLAT" in raw:
            return "volatile"
        if "BREAK" in raw:
            return "breakout"
        return "trending"

    def _state_from_training(self, symbol: str) -> Optional[RegimeState]:
        raw = self.trained_regime_state.get(symbol)
        if not isinstance(raw, dict):
            return None
        vol = self.trained_volatility_state.get(symbol, {}) if isinstance(self.trained_volatility_state, dict) else {}
        trend_strength = self._safe_float(raw.get("trend_strength", raw.get("strength", 0.5)), 0.5)
        confidence = self._safe_float(raw.get("confidence", trend_strength), trend_strength)
        direction = int(round(self._safe_float(raw.get("trend_direction", raw.get("direction", 0)), 0.0)))
        volatility_pct = self._safe_float(vol.get("volatility_percentile", 0.5), 0.5)
        adr_ratio = self._safe_float(raw.get("adr_ratio", 1.0), 1.0)
        return RegimeState(
            regime=self._regime_label_from_training(raw.get("regime")),
            strength=max(0.0, min(1.0, trend_strength)),
            trend_direction=1 if direction > 0 else -1 if direction < 0 else 0,
            volatility_percentile=max(0.0, min(1.0, volatility_pct)),
            adr_ratio=max(0.0, float(adr_ratio)),
            confidence=max(0.0, min(1.0, confidence)),
        )

    def bind_trained_state(
        self,
        trained_regime_states: Optional[Dict[str, Dict]] = None,
        trained_volatility_clusters: Optional[Dict[str, Dict]] = None,
        *,
        force_retrain: bool = False,
    ) -> None:
        """
        Bind startup-trained 22M state into this detector instance.
        This is architectural wiring only: no threshold/risk/strategy changes.
        """
        trained_regime_states = trained_regime_states or {}
        trained_volatility_clusters = trained_volatility_clusters or {}
        self.trained_regime_state = dict(trained_regime_states)
        self.trained_volatility_state = dict(trained_volatility_clusters)
        self.bound_training_hash = self._training_payload_hash()

        # FORCE_RETRAIN=1 keeps runtime in live-recompute mode by explicit operator intent.
        if force_retrain or self._env_true("FORCE_RETRAIN", False):
            self.using_trained_instance = False
            self.last_regime_source = "live_recompute"
            log.info("[regime] FORCE_RETRAIN enabled; trained cache binding skipped")
            return

        for symbol in self.trained_regime_state.keys():
            state = self._state_from_training(symbol)
            if state is not None:
                self.regime_cache[symbol] = state

        self.using_trained_instance = True
        self.last_regime_source = "trained_cache"
        self.last_live_update_applied = False
        self._save_cache()
        log.info(
            "[regime] Bound trained cache into live detector instance_id=%s symbols=%s hash=%s",
            self.instance_id,
            len(self.regime_cache),
            self.bound_training_hash,
        )

    def _compute_live_state(self, symbol: str, timeframe: str = "H1") -> Optional[RegimeState]:
        mt5 = _get_mt5()
        if mt5 is None:
            return None
        tf_map = {
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        tf_const = tf_map.get(timeframe, mt5.TIMEFRAME_H1)
        bars = mt5.copy_rates_from_pos(symbol, tf_const, 0, 200)
        if bars is None or len(bars) < 100:
            return None
        df = pd.DataFrame(bars)
        trend_strength = self._calculate_trend_strength(df)
        range_strength = self._calculate_range_strength(df)
        volatility = self._calculate_volatility_percentile(df)
        adr_ratio = self._calculate_adr_ratio(df)
        regime, strength, direction = self._classify_regime(trend_strength, range_strength, volatility)
        confidence = self._calculate_confidence(trend_strength, range_strength, volatility)
        return RegimeState(
            regime=regime,
            strength=strength,
            trend_direction=direction,
            volatility_percentile=volatility,
            adr_ratio=adr_ratio,
            confidence=confidence,
        )

    def _merge_live_update(self, base: RegimeState, live: RegimeState) -> RegimeState:
        # Incremental update keeps trained state as anchor while reflecting live evolution.
        alpha = 0.80  # Increased from 0.65 to favor current price action
        regime = base.regime
        direction = base.trend_direction
        if live.regime == base.regime:
            regime = live.regime
            direction = live.trend_direction
        elif live.confidence >= max(0.85, base.confidence + 0.10):
            regime = live.regime
            direction = live.trend_direction

        return RegimeState(
            regime=regime,
            strength=max(0.0, min(1.0, (1.0 - alpha) * base.strength + alpha * live.strength)),
            trend_direction=direction,
            volatility_percentile=max(
                0.0,
                min(1.0, (1.0 - alpha) * base.volatility_percentile + alpha * live.volatility_percentile),
            ),
            adr_ratio=max(0.0, (1.0 - alpha) * base.adr_ratio + alpha * live.adr_ratio),
            confidence=max(0.0, min(1.0, (1.0 - alpha) * base.confidence + alpha * live.confidence)),
        )

    def _detect_from_parquet(self, symbol: str) -> Optional[RegimeState]:
        """
        Detect regime from parquet historical data

        Args:
            symbol: Trading symbol

        Returns:
            RegimeState or None if parquet data not available
        """
        try:
            parquet_dir = Path(__file__).resolve().parents[3] / "DATA_MODELS" / "data_parquet"
            parquet_file = parquet_dir / f"{symbol}_M1.parquet"

            if not parquet_file.exists():
                return None

            # Load last 2000 bars for analysis
            df = pd.read_parquet(parquet_file)

            if len(df) > 2000:
                df = df.tail(2000)

            if len(df) < 1000:
                return None

            # Convert time if needed
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"])

            # Calculate regime indicators
            trend_strength = self._calculate_trend_strength(df)
            range_strength = self._calculate_range_strength(df)
            volatility = self._calculate_volatility_percentile(df)
            adr_ratio = self._calculate_adr_ratio(df)

            # Determine regime
            regime, strength, direction = self._classify_regime(trend_strength, range_strength, volatility)

            # Calculate confidence
            confidence = self._calculate_confidence(trend_strength, range_strength, volatility)

            state = RegimeState(
                regime=regime,
                strength=strength,
                trend_direction=direction,
                volatility_percentile=volatility,
                adr_ratio=adr_ratio,
                confidence=confidence,
            )

            log.info(f"[regime] {symbol} (parquet) -> {regime} (strength:{strength:.2f}, vol:{volatility:.2f}, conf:{confidence:.2f})")

            return state

        except Exception as e:
            log.debug(f"Parquet regime detection failed: {e}")
            return None

    def detect_regime(self, symbol: str, timeframe: str = "H1") -> RegimeState:
        """
        Detect current market regime for symbol

        Uses parquet data when available for better accuracy

        Args:
            symbol: Trading symbol
            timeframe: Timeframe for analysis

        Returns:
            RegimeState with current regime classification
        """
        # Explicit retrain mode bypasses trained-cache anchoring.
        force_retrain = self._env_true("FORCE_RETRAIN", False)

        # Parquet regime detection can hard-crash native libs on Windows.
        # Require explicit opt-in to use parquet; otherwise use MT5.
        if os.environ.get("ENABLE_PARQUET_REGIME", "0") == "1":
            parquet_result = self._detect_from_parquet(symbol)
            if parquet_result is not None:
                self.last_regime_source = "live_recompute"
                self.last_live_update_applied = False
                return parquet_result

        # Trained-cache anchored mode: use startup-trained instance as base and apply
        # incremental live updates in-place on the same detector object.
        if self.using_trained_instance and not force_retrain:
            base_state = self.regime_cache.get(symbol)
            if base_state is None:
                base_state = self._state_from_training(symbol)
            if base_state is None:
                base_state = self._get_default_regime()

            live_state = None
            try:
                live_state = self._compute_live_state(symbol, timeframe)
            except Exception as e:
                log.debug(f"[regime] live update computation failed: {e}")

            if live_state is not None and self.live_incremental_updates:
                state = self._merge_live_update(base_state, live_state)
                self.last_live_update_applied = True
                self.last_regime_source = "trained_cache+live_update"
            else:
                state = base_state
                self.last_live_update_applied = False
                self.last_regime_source = "trained_cache"

            self.regime_cache[symbol] = state
            self._save_cache()
            log.info(
                f"[regime] {symbol} -> {state.regime} (strength:{state.strength:.2f}, "
                f"vol:{state.volatility_percentile:.2f}, conf:{state.confidence:.2f}) source={self.last_regime_source}"
            )
            return state

        try:
            state = self._compute_live_state(symbol, timeframe)
            if state is None:
                self.last_regime_source = "live_recompute"
                self.last_live_update_applied = False
                return self._get_default_regime()

            # Cache result
            self.regime_cache[symbol] = state
            self._save_cache()
            self.last_regime_source = "live_recompute"
            self.last_live_update_applied = False

            log.info(
                f"[regime] {symbol} -> {state.regime} (strength:{state.strength:.2f}, "
                f"vol:{state.volatility_percentile:.2f}, conf:{state.confidence:.2f}) source=live_recompute"
            )

            return state

        except Exception as e:
            log.error(f"Regime detection failed: {e}")
            self.last_regime_source = "live_recompute"
            self.last_live_update_applied = False
            return self._get_default_regime()

    def _calculate_trend_strength(self, df: pd.DataFrame) -> Tuple[float, int]:
        """
        Trend strength via ADX + slope of closes.
        """
        close = df["close"]
        high = df["high"]
        low = df["low"]

        period = 14
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / (atr + 1e-9)
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / (atr + 1e-9)
        _denom = plus_di + minus_di
        dx = np.where(_denom != 0, 100 * (plus_di - minus_di).abs() / _denom, 0.0)
        dx = pd.Series(dx, index=plus_di.index)
        adx = dx.ewm(span=period, adjust=False).mean()

        # Slope strength (normalized)
        slope_window = min(50, len(close))
        if slope_window >= 10:
            # Avoid np.polyfit/np.linalg to prevent native crashes under load.
            y = close.tail(slope_window).to_numpy(dtype=float)
            x = np.arange(len(y), dtype=float)
            x_mean = x.mean()
            y_mean = y.mean()
            denom = np.sum((x - x_mean) ** 2)
            if denom > 0:
                slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
            else:
                slope = 0.0
            slope_strength = abs(slope) / (y_mean + 1e-8)
        else:
            slope_strength = 0.0

        current_adx = float(adx.iloc[-1]) if len(adx) else 0.0
        strength = min(1.0, (current_adx / 50.0 + slope_strength * 10) / 2.0)

        if plus_di.iloc[-1] > minus_di.iloc[-1] * 1.05:  # Reduced from 1.1 to 1.05 for sensitivity
            direction = 1
        elif minus_di.iloc[-1] > plus_di.iloc[-1] * 1.05:  # Reduced from 1.1 to 1.05 for sensitivity
            direction = -1
        else:
            direction = 0
        return strength, direction

    def _calculate_range_strength(self, df: pd.DataFrame) -> float:
        """
        Consolidation strength via Bollinger width percentile (narrower = higher).
        """
        close = df["close"]
        period = 20
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()
        bb_width = (std * 2) / sma
        current_width = float(bb_width.iloc[-1])
        hist = bb_width.iloc[-120:]
        if len(hist) == 0:
            return 0.5
        percentile = (hist < current_width).mean()
        return float(1.0 - percentile)

    def _calculate_volatility_percentile(self, df: pd.DataFrame) -> float:
        """
        Volatility percentile using ATR normalised to % of price.
        Normalising by close price makes the metric comparable across instruments
        of different price magnitudes (FX ~1.0, indices ~20000, energy ~80).
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(14).mean()
        # Normalise to % of price — prevents scale differences across instruments
        mid_price = close.rolling(14).mean()
        mid_price = mid_price.replace(0, float("nan"))
        atr_pct = atr / mid_price
        current = float(atr_pct.iloc[-1])
        hist = atr_pct.iloc[-40:].dropna()
        if len(hist) == 0:
            return 0.5
        percentile = (hist < current).mean()
        return float(percentile)

    def _calculate_adr_ratio(self, df: pd.DataFrame) -> float:
        """
        Calculate Average Daily Range ratio

        Returns:
            ratio: current range / average range
        """
        # Daily range
        daily_range = df["high"] - df["low"]

        # Average range (20 periods)
        avg_range = daily_range.rolling(20).mean()

        # Current vs average
        current_range = float(daily_range.iloc[-1])
        average = float(avg_range.iloc[-1])

        if average > 0:
            ratio = current_range / average
        else:
            ratio = 1.0

        return float(ratio)

    def _classify_regime(
        self,
        trend_strength: Tuple[float, int],
        range_strength: float,
        volatility: float,
    ) -> Tuple[str, float, int]:
        trend_val, direction = trend_strength
        trend_breakout = self.thresholds["trend_strength_breakout"]
        trend_trending = self.thresholds["trend_strength_trending"]
        range_ranging = self.thresholds["range_strength_ranging"]
        vol_breakout = self.thresholds["volatility_breakout"]
        vol_volatile = self.thresholds["volatility_volatile"]
        # Breakout if strong trend with elevated vol
        if trend_val > trend_breakout and volatility > vol_breakout:
            strength = min(1.0, (trend_val + volatility) / 2.0)
            return "breakout", strength, direction
        # Trend
        if trend_val > trend_trending and range_strength < range_ranging:
            return "trending", trend_val, direction
        # Volatile chop — FIX 2026-05-03: preserve direction so counter-trend
        # penalties fire correctly; was hardcoded 0 which silenced all directional gates.
        if volatility > vol_volatile:
            return "volatile", volatility, direction
        # Ranging / consolidation — FIX 2026-05-03: preserve directional bias
        # even during consolidation so regime-aware filters respect the broader trend.
        if range_strength > range_ranging and trend_val < (trend_trending - 0.1):
            return "ranging", range_strength, direction
        return "ranging", 0.5, direction

    def _calculate_confidence(
        self,
        trend_strength: Tuple[float, int],
        range_strength: float,
        volatility: float,
    ) -> float:
        trend_val, _ = trend_strength
        aligned = [
            trend_val > 0.7 and range_strength < 0.3,
            range_strength > 0.7 and trend_val < 0.3,
            volatility > 0.8,
        ]
        if any(aligned):
            return 0.85
        return float(min(1.0, max(trend_val, range_strength, volatility) * 1.25))

    def _get_default_regime(self) -> RegimeState:
        """Return default regime when detection fails"""
        return RegimeState(
            regime="ranging",
            strength=0.5,
            trend_direction=0,
            volatility_percentile=0.5,
            adr_ratio=1.0,
            confidence=0.45,  # was 0.3 — below MIN_REGIME_CONFIDENCE=0.4 which blocked pairs on MT5 data gaps
        )

    def get_regime_multiplier(self, regime: RegimeState, strategy_type: str = "hybrid") -> float:
        """
        Get position size multiplier based on regime

        Args:
            regime: Current regime
            strategy_type: "trend", "range", "hybrid"

        Returns:
            multiplier: 0.5-1.5 (adjust position size)
        """
        # Trend-following strategies
        if strategy_type == "trend":
            if regime.regime == "trending" and regime.strength > 0.6:
                return 1.3  # Increase size in strong trends
            elif regime.regime == "ranging":
                return 0.6  # Reduce size in ranges
            elif regime.regime == "volatile":
                return 0.7  # Reduce size in volatile markets
            else:
                return 1.0

        # Range-trading strategies
        elif strategy_type in ("range", "MeanRev"):
            if regime.regime == "ranging" and regime.strength > 0.5:
                # v3.2: Strategy 3 - Weight MeanRev strategies 2x in ranging markets
                return 2.0
            elif regime.regime == "trending":
                return 0.6  # Reduce size in trends
            else:
                return 1.0

        # Hybrid strategies (our system)
        else:
            if regime.regime == "breakout":
                return 1.2  # Good for both trend and range
            elif regime.regime == "trending" and regime.strength > 0.6:
                return 1.1  # Slightly favor trends
            elif regime.regime == "ranging" and regime.strength > 0.6:
                return 1.0  # Normal sizing
            elif regime.regime == "volatile":
                return 0.7  # Reduce size in chaos
            else:
                return 0.9  # Slightly conservative in mixed

    def _load_cache(self):
        """Load regime cache from disk"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        for symbol, payload in data.items():
                            if not isinstance(payload, dict):
                                continue
                            self.regime_cache[str(symbol)] = RegimeState(
                                regime=str(payload.get("regime", "ranging")),
                                strength=self._safe_float(payload.get("strength"), 0.5),
                                trend_direction=int(round(self._safe_float(payload.get("direction"), 0.0))),
                                volatility_percentile=max(0.0, min(1.0, self._safe_float(payload.get("volatility"), 0.5))),
                                adr_ratio=max(0.0, self._safe_float(payload.get("adr_ratio"), 1.0)),
                                confidence=max(0.0, min(1.0, self._safe_float(payload.get("confidence"), 0.3))),
                            )
                    log.info(f"[regime] Loaded cache with {len(self.regime_cache)} entries")
        except Exception as e:
            log.debug(f"Cache load failed: {e}")

    def _load_thresholds(self) -> Dict[str, float]:
        config_path = Path(__file__).resolve().parents[1] / "config" / "regime_thresholds.json"
        defaults = {
            "trend_strength_breakout": 0.6,
            "trend_strength_trending": 0.5,
            "range_strength_ranging": 0.6,
            "volatility_breakout": 0.6,
            "volatility_volatile": 0.7,
        }
        try:
            if config_path.exists():
                data = json.loads(config_path.read_text(encoding="utf-8"))
                return {
                    "trend_strength_breakout": float(data.get("trend_strength_breakout", defaults["trend_strength_breakout"])),
                    "trend_strength_trending": float(data.get("trend_strength_trending", defaults["trend_strength_trending"])),
                    "range_strength_ranging": float(data.get("range_strength_ranging", defaults["range_strength_ranging"])),
                    "volatility_breakout": float(data.get("volatility_breakout", defaults["volatility_breakout"])),
                    "volatility_volatile": float(data.get("volatility_volatile", defaults["volatility_volatile"])),
                }
        except Exception as e:
            log.warning(f"[regime] Failed to load thresholds: {e}")
        return defaults

    def _save_cache(self):
        """Save regime cache to disk with FIX CRITICAL-008: debounced saves."""
        import time

        # Debounce: only save once per _cache_save_interval seconds
        now = time.time()
        if now - self._last_save_time < self._cache_save_interval:
            return
        self._last_save_time = now

        try:
            self.results_dir.mkdir(parents=True, exist_ok=True)
            # Convert to serializable format
            cache_data = {
                symbol: {
                    "regime": state.regime,
                    "strength": state.strength,
                    "direction": state.trend_direction,
                    "volatility": state.volatility_percentile,
                    "adr_ratio": state.adr_ratio,
                    "confidence": state.confidence,
                }
                for symbol, state in self.regime_cache.items()
            }
            with open(self.cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            log.debug(f"Cache save failed: {e}")


# Singleton
_regime_detector: Optional[MarketRegimeDetector] = None


def get_regime_detector(results_dir: Path = None) -> MarketRegimeDetector:
    """Get or create regime detector singleton"""
    global _regime_detector
    if _regime_detector is None:
        if results_dir is None:
            results_dir = Path(__file__).resolve().parents[3] / "DATA_MODELS" / "results"
        _regime_detector = MarketRegimeDetector(results_dir)
    return _regime_detector
