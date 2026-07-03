"""
Regime-Detection Ensemble
=========================
A robustness layer that fuses multiple regime detectors into a single,
confidence-weighted decision.  The goal is to reduce flip-flops when one
sensor (live MT5 bars, trained cache, or parquet history) is noisy.

Detectors registered by default
-------------------------------
1. ``MarketRegimeDetector`` — live ADX/BB/ATR computation.
2. ``VolatilityRegimeDetector`` — pure ATR-percentile fast detector.
3. ``trained`` — startup-trained regime state, if available.

Weights are configurable via ``regime_ensemble_weights`` in
``risk_governor.json``; missing weights default to equal voting.

Usage
-----
    from CORE_MODULES.core.regime.regime_ensemble import RegimeEnsemble
    ensemble = RegimeEnsemble(results_dir=Path("CORE_MODULES/results"))
    state = ensemble.detect("EURUSD", timeframe="H1")
    print(state.regime, state.confidence)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from CORE_MODULES.core.regime.market_regime import MarketRegimeDetector, RegimeState

logger = logging.getLogger(__name__)


class VolatilityRegimeDetector:
    """Fast, pure-volatility detector used as an ensemble voter."""

    def __init__(self, atr_lookback: int = 14, percentile_lookback: int = 40) -> None:
        self.atr_lookback = atr_lookback
        self.percentile_lookback = percentile_lookback

    def detect(self, symbol: str, timeframe: str = "H1") -> Optional[RegimeState]:
        try:
            import MetaTrader5 as mt5
        except ImportError:
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
        if bars is None or len(bars) < self.percentile_lookback + 10:
            return None

        import pandas as pd

        df = pd.DataFrame(bars)
        high, low, close = df["high"], df["low"], df["close"]
        tr = pd.concat(
            [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(self.atr_lookback).mean()
        atr_pct = atr / close.rolling(self.atr_lookback).mean().replace(0, float("nan"))
        current = float(atr_pct.iloc[-1])
        hist = atr_pct.iloc[-self.percentile_lookback :].dropna()
        if len(hist) == 0:
            return None
        percentile = float((hist < current).mean())

        if percentile >= 0.75:
            regime = "volatile"
        elif percentile <= 0.25:
            regime = "ranging"
        else:
            regime = "trending"

        return RegimeState(
            regime=regime,
            strength=abs(percentile - 0.5) * 2.0,
            trend_direction=0,
            volatility_percentile=percentile,
            adr_ratio=1.0,
            confidence=abs(percentile - 0.5) * 2.0,
        )


@dataclass
class EnsembleConfig:
    weights: Dict[str, float]
    min_confidence: float = 0.35
    tie_breaker: str = "trending"

    @classmethod
    def from_governor(cls, path: Optional[Path] = None) -> "EnsembleConfig":
        path = path or Path(__file__).resolve().parents[2] / "config" / "risk_governor.json"
        default = {"market": 0.5, "volatility": 0.3, "trained": 0.2, "min_confidence": 0.35}
        if not path.exists():
            return cls(weights=default)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            cfg = data.get("regime_ensemble_weights", default)
            return cls(
                weights={k: float(v) for k, v in cfg.items() if k != "min_confidence"},
                min_confidence=float(cfg.get("min_confidence", default["min_confidence"])),
            )
        except Exception as exc:
            logger.warning(f"[regime-ensemble] Failed to load config: {exc}")
            return cls(weights=default)


class RegimeEnsemble:
    """Aggregate regime votes from multiple detectors."""

    REGIME_ORDER = ["trending", "ranging", "volatile", "breakout"]

    def __init__(
        self,
        results_dir: Optional[Path] = None,
        config: Optional[EnsembleConfig] = None,
    ) -> None:
        self.results_dir = results_dir or Path(__file__).resolve().parents[2] / "results"
        self.config = config or EnsembleConfig.from_governor()
        self._detectors: Dict[str, Callable[[str, str], Optional[RegimeState]]] = {
            "market": MarketRegimeDetector(results_dir=self.results_dir).detect_regime,
            "volatility": VolatilityRegimeDetector().detect,
        }
        self._trained_cache: Optional[Dict[str, Any]] = None

    def register_detector(
        self, name: str, detector: Callable[[str, str], Optional[RegimeState]], weight: Optional[float] = None
    ) -> None:
        self._detectors[name] = detector
        if weight is not None:
            self.config.weights[name] = weight

    def _load_trained_state(self) -> Optional[Dict[str, Any]]:
        if self._trained_cache is not None:
            return self._trained_cache
        candidate = self.results_dir / "trained_regime_state.json"
        if candidate.exists():
            try:
                self._trained_cache = json.loads(candidate.read_text(encoding="utf-8"))
                return self._trained_cache
            except Exception as exc:
                logger.debug(f"[regime-ensemble] trained state load failed: {exc}")
        return None

    def _trained_vote(self, symbol: str, timeframe: str) -> Optional[RegimeState]:
        state = self._load_trained_state()
        if not state or symbol not in state:
            return None
        raw = state[symbol]
        regime = str(raw.get("regime", "trending")).lower()
        return RegimeState(
            regime=regime,
            strength=float(raw.get("strength", 0.5)),
            trend_direction=int(raw.get("trend_direction", 0)),
            volatility_percentile=float(raw.get("volatility_percentile", 0.5)),
            adr_ratio=float(raw.get("adr_ratio", 1.0)),
            confidence=float(raw.get("confidence", 0.5)),
        )

    def detect(self, symbol: str, timeframe: str = "H1") -> RegimeState:
        """Weighted-vote regime detection."""
        votes: List[Tuple[str, RegimeState, float]] = []

        # Standard detectors.
        for name, detector in self._detectors.items():
            weight = self.config.weights.get(name, 1.0)
            try:
                state = detector(symbol, timeframe)
                if state is not None:
                    votes.append((name, state, weight))
            except Exception as exc:
                logger.debug(f"[regime-ensemble] detector {name} failed for {symbol}: {exc}")

        # Optional trained-state voter.
        trained_weight = self.config.weights.get("trained", 0.0)
        if trained_weight > 0:
            try:
                trained_state = self._trained_vote(symbol, timeframe)
                if trained_state is not None:
                    votes.append(("trained", trained_state, trained_weight))
            except Exception as exc:
                logger.debug(f"[regime-ensemble] trained vote failed for {symbol}: {exc}")

        if not votes:
            logger.warning(f"[regime-ensemble] No votes for {symbol}; returning default")
            return RegimeState("trending", 0.5, 0, 0.5, 1.0, 0.0)

        # Weighted vote by regime label.
        scores: Dict[str, float] = {}
        conf_sum = 0.0
        for _name, state, weight in votes:
            w = weight * state.confidence
            scores[state.regime] = scores.get(state.regime, 0.0) + w
            conf_sum += w

        if conf_sum <= 0:
            return RegimeState("trending", 0.5, 0, 0.5, 1.0, 0.0)

        winning_regime = max(scores, key=scores.get)
        confidence = min(1.0, scores[winning_regime] / conf_sum)

        # Average numeric fields from voters that agreed with the winner.
        agreeing = [s for _, s, _ in votes if s.regime == winning_regime]
        def avg(attr):
            return float(np.mean([getattr(s, attr) for s in agreeing])) if agreeing else 0.5

        direction_votes = [s.trend_direction for s in agreeing if s.trend_direction != 0]
        direction = int(np.sign(sum(direction_votes))) if direction_votes else 0

        result = RegimeState(
            regime=winning_regime,
            strength=avg("strength"),
            trend_direction=direction,
            volatility_percentile=avg("volatility_percentile"),
            adr_ratio=avg("adr_ratio"),
            confidence=confidence,
        )
        logger.info(
            f"[regime-ensemble] {symbol} -> {result.regime} "
            f"(conf={result.confidence:.2f}, voters={len(votes)})"
        )
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    e = RegimeEnsemble()
    s = e.detect("EURUSD", "H1")
    print(s)
