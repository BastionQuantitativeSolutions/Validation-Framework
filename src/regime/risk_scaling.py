import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_REGIME_PROFILE_NAMES = ("high_alignment", "neutral", "chop")

_DEFAULT_REGIME_RISK_PROFILE = {
    "high_alignment": 1.0,
    "neutral": 0.75,
    "chop": 0.1,
}
_DEFAULT_REGIME_CONFIDENCE_PROFILE = {
    "high_alignment": 1.0,
    "neutral": 0.92,
    "chop": 0.85,
}
_DEFAULT_REGIME_SIZE_PROFILE = {
    "high_alignment": 1.0,
    "neutral": 0.85,
    "chop": 0.70,
}

# Global state for active profiles (overridden by config)
REGIME_RISK_PROFILE = dict(_DEFAULT_REGIME_RISK_PROFILE)
REGIME_CONFIDENCE_PROFILE = dict(_DEFAULT_REGIME_CONFIDENCE_PROFILE)
REGIME_SIZE_PROFILE = dict(_DEFAULT_REGIME_SIZE_PROFILE)


def _normalize_confidence_01(value: Any, default: float = 0.0) -> float:
    """Normalize confidence into [0, 1]."""
    try:
        v = float(value)
        if 1.0 < v <= 100.0:
            v = v / 100.0
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return default


def _normalize_regime_profile_name(name: Any) -> str:
    """Normalize regime profile name for lookup."""
    raw = str(name or "").strip().lower()
    aliases = {
        "high": "high_alignment",
        "aligned": "high_alignment",
        "high_alignment_regime": "high_alignment",
        "neutral_regime": "neutral",
        "choppy": "chop",
        "chop_regime": "chop",
    }
    normalized = aliases.get(raw, raw)
    return normalized if normalized in _REGIME_PROFILE_NAMES else "neutral"


def _coerce_regime_profile_map(raw_map: Any, defaults: Dict[str, float]) -> Dict[str, float]:
    """Coerce raw configuration into valid profile map."""
    out = dict(defaults)
    if not isinstance(raw_map, dict):
        return out
    for key, value in raw_map.items():
        profile_name = _normalize_regime_profile_name(key)
        try:
            out[profile_name] = max(0.1, min(1.5, float(value)))
        except (TypeError, ValueError):
            continue
    return out


def load_regime_risk_profiles(raw_cfg: Dict[str, Any]) -> None:
    """Load regime-specific risk multipliers from raw configuration."""
    global REGIME_RISK_PROFILE, REGIME_CONFIDENCE_PROFILE, REGIME_SIZE_PROFILE

    base = raw_cfg.get("regime_risk_profile", {}) if isinstance(raw_cfg, dict) else {}
    risk_raw = base.get("risk_multipliers", base) if isinstance(base, dict) else {}
    confidence_raw = base.get("confidence_multipliers", {}) if isinstance(base, dict) else {}
    size_raw = base.get("size_multipliers", {}) if isinstance(base, dict) else {}

    REGIME_RISK_PROFILE = _coerce_regime_profile_map(risk_raw, _DEFAULT_REGIME_RISK_PROFILE)
    REGIME_CONFIDENCE_PROFILE = _coerce_regime_profile_map(confidence_raw, _DEFAULT_REGIME_CONFIDENCE_PROFILE)
    REGIME_SIZE_PROFILE = _coerce_regime_profile_map(size_raw, _DEFAULT_REGIME_SIZE_PROFILE)

    logger.info(f"[regime-scaling] Loaded profiles: risk={REGIME_RISK_PROFILE}, conf={REGIME_CONFIDENCE_PROFILE}")


def _classify_regime_profile(
    regime: Optional[str],
    volatility: Optional[str],
    regime_confidence: Optional[float],
) -> Tuple[str, str]:
    """Classify current market state into a regime profile."""
    regime_u = str(regime or "UNKNOWN").strip().upper()
    vol_u = str(volatility or "UNKNOWN").strip().upper()
    conf = _normalize_confidence_01(regime_confidence, default=0.0)

    trend_like = any(
        token in regime_u
        for token in (
            "TREND",
            "UPTREND",
            "DOWNTREND",
            "BREAKOUT",
            "MOMENTUM",
        )
    )
    chop_like = any(
        token in regime_u
        for token in (
            "RANG",
            "CHOP",
            "CONSOL",
            "SIDEWAYS",
            # "UNKNOWN" removed — UNKNOWN should now never reach here (LiveRegimeDetector
            # fallback in cycle_context.py guarantees a real regime). If it does slip through,
            # default to "neutral" (1.0x) rather than penalising with "chop" (0.90x).
        )
    )
    high_vol = any(token in vol_u for token in ("HIGH", "EXTREME", "CRISIS", "SHOCK"))

    if trend_like and conf >= 0.75 and not chop_like and not (high_vol and conf < 0.85):
        return "high_alignment", f"trend_like conf={conf:.2f} vol={vol_u}"
    if chop_like or conf < 0.55 or (high_vol and conf < 0.70):
        return "chop", f"chop_like conf={conf:.2f} vol={vol_u}"

    return "neutral", f"transitional conf={conf:.2f} vol={vol_u}"


def get_regime_risk_scaling(
    *,
    regime: Optional[str],
    volatility: Optional[str],
    regime_confidence: Optional[float],
    regime_wr: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculate risk scaling multipliers based on market regime.

    regime_wr: MC p50 Win Rate for this regime from the training manifest
    (precision = when the classifier predicts regime X, how often is it right?).
    Adjusts confidence_multiplier to reflect how reliable this regime call is:
      - regime_wr >= 0.75  → +5% boost  (well-calibrated, trust the call)
      - regime_wr < 0.55   → -10% penalty (poorly calibrated, reduce trust)
      - 0.55 <= wr < 0.75  → no change
    """
    profile, reason = _classify_regime_profile(regime, volatility, regime_confidence)
    base_conf_mult = float(REGIME_CONFIDENCE_PROFILE.get(profile, 1.0))

    # WR-based calibration adjustment
    wr_adj = 1.0
    wr_reason = ""
    if regime_wr is not None:
        try:
            wr = float(regime_wr)
            if wr >= 0.75:
                wr_adj = 1.05
                wr_reason = f" wr={wr:.2f}(+5%)"
            elif wr < 0.55:
                wr_adj = 0.90
                wr_reason = f" wr={wr:.2f}(-10%)"
        except (TypeError, ValueError):
            pass

    return {
        "profile": profile,
        "risk_multiplier": float(REGIME_RISK_PROFILE.get(profile, 1.0)),
        "confidence_multiplier": round(base_conf_mult * wr_adj, 4),
        "size_multiplier": float(REGIME_SIZE_PROFILE.get(profile, 1.0)),
        "reason": reason + wr_reason,
        "regime_wr": regime_wr,
    }
