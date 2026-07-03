import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Internal Imports
from core.execution.trade_history import _append_regime_observation_event
from core.trading247.wrappers import get_current_session_info
from core.regime.regime_bridge import get_regime_wr, LiveRegimeDetector

# Singleton fallback detector — used when primary detector returns UNKNOWN
_LIVE_REGIME_FALLBACK: Optional[LiveRegimeDetector] = None


def _get_live_fallback() -> LiveRegimeDetector:
    global _LIVE_REGIME_FALLBACK
    if _LIVE_REGIME_FALLBACK is None:
        _LIVE_REGIME_FALLBACK = LiveRegimeDetector()
    return _LIVE_REGIME_FALLBACK


def _regime_from_df(df: pd.DataFrame) -> Optional[Tuple[str, int, float]]:
    """Run LiveRegimeDetector on df; return (regime, direction, strength) for last bar, or None."""
    try:
        if df is None or len(df) < 50:
            return None
        regimes, directions, strengths = _get_live_fallback().detect_regime_vectorized(df)
        last_regime = str(regimes[-1]).upper()
        last_dir = int(directions[-1])
        last_strength = float(strengths[-1])
        return last_regime, last_dir, last_strength
    except Exception as e:
        logging.debug(f"[regime-fallback] LiveRegimeDetector failed: {e}")
        return None


# Forward declarations for optional diagnostics
_regime_volatility_diag = None


@dataclass(frozen=True)
class CycleContext:
    regime: str
    volatility: str
    timestamp: float
    regime_source: str
    regime_key: str
    volatility_percentile: Optional[float] = None
    regime_confidence_raw: Optional[float] = None
    regime_confidence: float = 0.0
    direction: int = 0
    compression_warning: bool = False
    # MC p50 WR for the detected regime (from training manifest).
    # Measures how often the classifier is correct when it predicts this regime.
    regime_wr: float = 0.65
    regime_strength: float = 0.0
    vol_ratio: float = 0.0


def _normalize_confidence_01(value: Any, default: float = 0.0) -> float:
    """Normalize confidence into [0, 1] while preserving detector semantics."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(v):
        return float(default)
    if 1.0 < v <= 100.0:
        v = v / 100.0
    return max(0.0, min(1.0, float(v)))


def _extract_detector_regime_confidence(regime_state: Any) -> Optional[float]:
    """Extract detector-emitted regime confidence without recomputation."""
    candidates: List[Any] = []
    if isinstance(regime_state, dict):
        candidates.extend(
            [
                regime_state.get("confidence"),
                regime_state.get("regime_confidence"),
                (regime_state.get("probabilistic") or {}).get("regime_confidence") if isinstance(regime_state.get("probabilistic"), dict) else None,
            ]
        )
    else:
        candidates.extend([getattr(regime_state, "confidence", None), getattr(regime_state, "regime_confidence", None)])

    for candidate in candidates:
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def _resolve_cycle_context(
    pair: str,
    tf: str,
    smc: Dict[str, Any],
    regime_detector: Optional[Any],
    use_regime_filter: bool,
    df: Optional[pd.DataFrame] = None,
) -> CycleContext:
    regime = str(smc.get("regime", "UNKNOWN"))
    regime_source = "smc"
    current_volatility = "MEDIUM"
    volatility_percentile: Optional[float] = None
    regime_confidence_raw: Optional[float] = None
    regime_confidence = 0.0
    compression_warning = False
    regime_state = None  # initialised here; set by detector on success
    vol_score = smc.get("volatility")
    try:
        vol_value = float(vol_score)
    except (TypeError, ValueError):
        vol_value = None
    if vol_value is not None:
        if vol_value < 0.33:
            current_volatility = "LOW"
        elif vol_value < 0.66:
            current_volatility = "MEDIUM"
        else:
            current_volatility = "HIGH"
    if regime_detector and use_regime_filter:
        try:
            regime_state = regime_detector.detect_regime(pair, tf)
            detector_source = getattr(regime_detector, "last_regime_source", None)
            if isinstance(detector_source, str) and detector_source.strip():
                regime_source = detector_source.strip()
            else:
                regime_source = "detector"
            # Epistemic evidence capture only: this does not change execution behavior.
            _append_regime_observation_event(
                symbol=pair,
                timeframe=tf,
                session=get_current_session_info().get("session", "OTHER"),
                regime_state=regime_state,
            )
            if hasattr(regime_state, "regime"):
                regime = str(regime_state.regime).upper()
                # Preserve detector-provided source lineage for diagnostics.
                if not (isinstance(regime_source, str) and regime_source.strip()):
                    regime_source = "detector"
            vol_pct = getattr(regime_state, "volatility_percentile", None)
            if isinstance(vol_pct, (float, int)):
                volatility_percentile = float(vol_pct)
                if vol_pct < 0.33:
                    current_volatility = "LOW"
                elif vol_pct < 0.66:
                    current_volatility = "MEDIUM"
                else:
                    current_volatility = "HIGH"
            reg_conf_raw = _extract_detector_regime_confidence(regime_state)
            if reg_conf_raw is not None:
                regime_confidence_raw = float(reg_conf_raw)
                regime_confidence = _normalize_confidence_01(reg_conf_raw, default=0.0)

            if hasattr(regime_detector, "using_trained_instance"):
                cache_hash = None
                if hasattr(regime_detector, "get_runtime_cache_hash"):
                    try:
                        cache_hash = regime_detector.get_runtime_cache_hash()
                    except Exception:
                        cache_hash = None
                logging.debug(
                    "[REGIME_FIX] using_trained_instance=%s",
                    bool(getattr(regime_detector, "using_trained_instance", False)),
                )
                logging.debug("[REGIME_FIX] instance_id=%s", id(regime_detector))
                logging.debug("[REGIME_FIX] cache_hash=%s", cache_hash)
                logging.debug(
                    "[REGIME_FIX] live_update_applied=%s",
                    bool(getattr(regime_detector, "last_live_update_applied", False)),
                )
        except Exception as e:
            logging.warning(f"[regime] Pre-check failed for {pair}.{tf}: {e}")

    # --- Guaranteed fallback: LiveRegimeDetector on OHLCV df ----------------
    # If the primary detector returned UNKNOWN (missing, failed, or not wired),
    # run the rule-based LiveRegimeDetector on the bars we already have in memory.
    # This eliminates UNKNOWN from the live pipeline entirely.
    if str(regime).upper() in ("UNKNOWN", "", "NONE") and df is not None:
        _fb = _regime_from_df(df)
        if _fb is not None:
            _fb_regime, _fb_dir, _fb_strength = _fb
            logging.info(
                f"[regime-fallback] {pair}.{tf} primary=UNKNOWN → LiveRegimeDetector: {_fb_regime} (dir={_fb_dir} strength={_fb_strength:.2f})"
            )
            regime = _fb_regime
            regime_source = "live_rule_fallback"
            # Inject strength as confidence proxy (rule-based, so cap at 0.70)
            _fb_conf = min(0.70, float(_fb_strength))
            regime_confidence_raw = _fb_conf
            regime_confidence = _fb_conf

    if volatility_percentile is None and vol_value is not None:
        volatility_percentile = max(0.0, min(1.0, float(vol_value)))
    if "CONSOLID" in str(regime).upper():
        compression_warning = True
    elif "RANG" in str(regime).upper() and volatility_percentile is not None:
        compression_warning = volatility_percentile < 0.33
    try:
        if _regime_volatility_diag is not None:
            _regime_volatility_diag.note_regime_source(regime_source)
    except Exception as e:
        logging.debug(f"[REGIME_DIAG] regime-source diagnostics skipped: {e}")
    regime_key = str(regime or "UNKNOWN").strip().lower()
    resolved_regime = str(regime or "UNKNOWN").strip().upper()
    return CycleContext(
        regime=resolved_regime,
        volatility=current_volatility,
        timestamp=time.time(),
        regime_source=regime_source,
        regime_key=regime_key,
        volatility_percentile=volatility_percentile,
        regime_confidence_raw=regime_confidence_raw,
        regime_confidence=regime_confidence,
        direction=int(getattr(regime_state, "trend_direction", 0)) if regime_state is not None else 0,
        compression_warning=compression_warning,
        regime_wr=get_regime_wr(resolved_regime),
        regime_strength=float(getattr(regime_state, "strength", 0.0)) if regime_state is not None else 0.0,
        vol_ratio=float(getattr(regime_state, "volatility_percentile", 0.0)) if regime_state is not None else 0.0,
    )


def _bb_filter_debug_info(
    regime_tag: str,
    regime_key: str,
    bb_pos: float,
    bb_min: float,
    bb_max: float,
    bb_range_low: float,
    bb_range_high: float,
    dry_run: bool = False,
) -> Dict[str, Any]:
    regime_tag_norm = str(regime_tag or "UNKNOWN").strip().upper()
    regime_key_norm = str(regime_key or "UNKNOWN").strip().lower()
    is_range = "range" in regime_key_norm or regime_key_norm.startswith("rang")
    branch = "extremes" if is_range else "mid"
    used_midband = branch == "mid"
    info = {
        "regime": regime_tag_norm,
        "branch": branch,
        "bb_pos": float(bb_pos),
        "midrange": (float(bb_min), float(bb_max)),
        "extremes": (float(bb_range_low), float(bb_range_high)),
        "used_midband": used_midband,
    }
    if dry_run and regime_tag_norm in ("RANGING", "RANGE") and used_midband:
        raise RuntimeError(f"BB filter using mid-band under RANGING (dry-run guard): regime_tag={regime_tag_norm!r} regime_key={regime_key_norm!r}")
    return info


def _format_bb_filter_debug_line(info: Dict[str, Any]) -> str:
    mid_min, mid_max = info["midrange"]
    range_low, range_high = info["extremes"]
    branch_label = "ranging-extremes" if info["branch"] == "extremes" else "midrange"
    return (
        f"BB branch check regime={info['regime']} "
        f"branch={branch_label} "
        f"bb={info['bb_pos']:.2f} mid=[{mid_min:.2f},{mid_max:.2f}] "
        f"extremes=[{range_low:.2f},{range_high:.2f}]"
    )
