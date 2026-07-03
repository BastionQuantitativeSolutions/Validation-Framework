import math
from typing import Any, Dict, Optional, Tuple

_REGIME_PROFILE_NAMES = ("high_alignment", "neutral", "chop")

_DEFAULT_REGIME_RISK_PROFILE = {
    "high_alignment": 1.0,
    "neutral": 0.75,
    "chop": 0.5,
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

REGIME_RISK_PROFILE = dict(_DEFAULT_REGIME_RISK_PROFILE)
REGIME_CONFIDENCE_PROFILE = dict(_DEFAULT_REGIME_CONFIDENCE_PROFILE)
REGIME_SIZE_PROFILE = dict(_DEFAULT_REGIME_SIZE_PROFILE)


def _normalize_regime_profile_name(name: Any) -> str:
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
    out = dict(defaults)
    if not isinstance(raw_map, dict):
        return out
    for key, value in raw_map.items():
        profile_name = _normalize_regime_profile_name(key)
        try:
            out[profile_name] = max(0.1, min(1.5, float(value)))
        except Exception:
            continue
    return out


def load_regime_risk_profiles(raw_cfg: Dict[str, Any]) -> None:
    global REGIME_RISK_PROFILE, REGIME_CONFIDENCE_PROFILE, REGIME_SIZE_PROFILE
    base = raw_cfg.get("regime_risk_profile", {}) if isinstance(raw_cfg, dict) else {}
    risk_raw = base.get("risk_multipliers", base) if isinstance(base, dict) else {}
    confidence_raw = base.get("confidence_multipliers", {}) if isinstance(base, dict) else {}
    size_raw = base.get("size_multipliers", {}) if isinstance(base, dict) else {}
    REGIME_RISK_PROFILE = _coerce_regime_profile_map(risk_raw, _DEFAULT_REGIME_RISK_PROFILE)
    REGIME_CONFIDENCE_PROFILE = _coerce_regime_profile_map(confidence_raw, _DEFAULT_REGIME_CONFIDENCE_PROFILE)
    REGIME_SIZE_PROFILE = _coerce_regime_profile_map(size_raw, _DEFAULT_REGIME_SIZE_PROFILE)


def _normalize_confidence_01(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
        if math.isnan(f):
            return default
        return max(0.0, min(1.0, f))
    except Exception:
        return default


def _classify_regime_profile(
    regime: Optional[str],
    volatility: Optional[str],
    regime_confidence: Optional[float],
) -> Tuple[str, str]:
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
    regime_confidence: Optional[float] = 0.0,
    **kwargs,  # Accept any extra args gracefully
) -> Dict[str, Any]:
    profile, reason = _classify_regime_profile(regime, volatility, regime_confidence)
    return {
        "profile": profile,
        "risk_multiplier": float(REGIME_RISK_PROFILE.get(profile, 1.0)),
        "confidence_multiplier": float(REGIME_CONFIDENCE_PROFILE.get(profile, 1.0)),
        "size_multiplier": float(REGIME_SIZE_PROFILE.get(profile, 1.0)),
        "reason": reason,
    }
