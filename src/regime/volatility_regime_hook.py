"""
# Author: JG
Volatility Clustering & Regime Training Hook
==========================================

This module provides hooks for initializing volatility clustering
and regime training before trading cycles begin.

22M Dataset: Full historical dataset (22+ million bars) used for
training regime detection and volatility clustering models.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import json
import os

try:
    from CORE_MODULES.llms.llm_manager import get_llm_manager

    LLM_MANAGER_AVAILABLE = True
    _llm_manager = get_llm_manager()
except Exception as e:
    LLM_MANAGER_AVAILABLE = False
    LLM_MANAGER_ERROR = str(e)
    _llm_manager = None

try:
    from CORE_MODULES.core.config.config_sync import is_llm_enabled

    LLM_CONFIG_AVAILABLE = True
except Exception as e:
    LLM_CONFIG_AVAILABLE = False
    LLM_CONFIG_ERROR = str(e)

    def is_llm_enabled() -> bool:
        return True


log = logging.getLogger(__name__)

_parquet_dir = None
_volatility_cache = None
_regime_cache = None


def get_parquet_dir() -> Path:
    """Get parquet data directory path"""
    global _parquet_dir
    if _parquet_dir is None:
        _parquet_dir = Path(__file__).resolve().parents[3] / "DATA_MODELS" / "data_parquet"
    return _parquet_dir


def train_volatility_clustering(symbols: list, force_retrain: bool = False, use_live_mt5: bool = False) -> Dict[str, Dict]:
    """
    Train volatility clustering on 22M dataset

    Args:
        symbols: List of symbols to train
        force_retrain: Force retrain even if cache exists
        use_live_mt5: Use live MT5 data instead of parquet files

    Returns:
        Dictionary of volatility clusters per symbol
    """
    global _volatility_cache
    parquet_dir = get_parquet_dir()
    cache_file = Path(__file__).resolve().parents[3] / "DATA_MODELS" / "results" / "volatility_clusters.json"

    # Load existing cache and only train missing symbols — never discard already-trained data
    _existing_cache = {}
    if cache_file.exists() and not force_retrain:
        try:
            with open(cache_file, "r") as f:
                _existing_cache = json.load(f)
            missing = [s for s in symbols if s not in _existing_cache]
            if not missing:
                _volatility_cache = _existing_cache
                log.info(f"[Volatility] Loaded cached volatility clusters for {len(_volatility_cache)} symbols")
                return _volatility_cache
            log.info(
                f"[Volatility] Cache has {len(_existing_cache)} symbols; training {len(missing)} missing: {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        except Exception as e:
            log.warning(f"[Volatility] Failed to load cache: {e}, retraining all...")

    # Start from existing cached data, only add missing symbols
    _volatility_cache = dict(_existing_cache)
    data_source = "LIVE MT5" if use_live_mt5 else "22M dataset"
    log.info(f"[Volatility] Training volatility clustering on {data_source}...")

    for symbol in [s for s in symbols if s not in _volatility_cache]:
        try:
            parquet_file = parquet_dir / f"{symbol}_M1.parquet"
            if not parquet_file.exists():
                log.warning(f"[Volatility] No data for {symbol}, skipping")
                continue

            df = pd.read_parquet(parquet_file)
            log.info(f"[Volatility] Loaded {len(df):,} bars for {symbol}")

            # Calculate ATR-based volatility
            df["high_low"] = df["high"] - df["low"]
            df["high_close"] = abs(df["high"] - df["close"].shift(1))
            df["low_close"] = abs(df["low"] - df["close"].shift(1))
            df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)

            # Multiple ATR periods
            atr_14 = df["tr"].rolling(14).mean()
            atr_50 = df["tr"].rolling(50).mean()
            atr_200 = df["tr"].rolling(200).mean()

            # Calculate volatility percentiles
            _close = df["close"].replace(0, np.nan)
            vol_short = atr_14 / _close * 100
            vol_medium = atr_50 / _close * 100
            vol_long = atr_200 / _close * 100

            # Cluster into regimes (low, medium, high)
            vol_clusters = {
                "low_vol_threshold": float(vol_short.quantile(0.33)),
                "medium_vol_threshold": float(vol_short.quantile(0.66)),
                "current_short_vol": float(vol_short.iloc[-1]),
                "current_medium_vol": float(vol_medium.iloc[-1]),
                "current_long_vol": float(vol_long.iloc[-1]),
                "volatility_percentile": float((vol_short < vol_short.iloc[-1]).mean()),
                "avg_volatility": float(vol_short.mean()),
                "std_volatility": float(vol_short.std()),
            }

            # Determine current cluster
            current_vol = vol_clusters["current_short_vol"]
            if current_vol < vol_clusters["low_vol_threshold"]:
                regime = "LOW_VOLATILITY"
            elif current_vol < vol_clusters["medium_vol_threshold"]:
                regime = "MEDIUM_VOLATILITY"
            else:
                regime = "HIGH_VOLATILITY"

            vol_clusters["current_regime"] = regime
            _volatility_cache[symbol] = vol_clusters

            log.info(f"[Volatility] {symbol} -> {regime} (vol={current_vol:.4f}%, pctile={vol_clusters['volatility_percentile']:.1%})")

        except Exception as e:
            log.error(f"[Volatility] Failed to train {symbol}: {e}")

    # Save cache
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        # Convert numpy types to native Python types for JSON serialization
        json_safe_cache = {}
        for k, v in _volatility_cache.items():
            json_safe_cache[k] = {kk: float(vv) if hasattr(vv, "item") else vv for kk, vv in v.items()}
        with open(cache_file, "w") as f:
            json.dump(json_safe_cache, f, indent=2)
        log.info("[Volatility] Saved volatility clusters to cache")
    except Exception as e:
        log.error(f"[Volatility] Failed to save cache: {e}")

    return _volatility_cache


def train_regime_detection(symbols: list, force_retrain: bool = False, use_live_mt5: bool = False) -> Dict[str, Dict]:
    """
    Train regime detection on 22M dataset

    Args:
        symbols: List of symbols to train
        force_retrain: Force retrain even if cache exists
        use_live_mt5: Use live MT5 data instead of parquet files

    Returns:
        Dictionary of regime states per symbol
    """
    global _regime_cache
    parquet_dir = get_parquet_dir()
    cache_file = Path(__file__).resolve().parents[3] / "DATA_MODELS" / "results" / "regime_training_cache.json"

    # Load existing cache and only train missing symbols — never discard already-trained data
    _existing_regime_cache = {}
    if cache_file.exists() and not force_retrain:
        try:
            with open(cache_file, "r") as f:
                _existing_regime_cache = json.load(f)
            missing = [s for s in symbols if s not in _existing_regime_cache]
            if not missing:
                _regime_cache = _existing_regime_cache
                log.info(f"[Regime] Loaded cached regime states for {len(_regime_cache)} symbols")
                return _regime_cache
            log.info(
                f"[Regime] Cache has {len(_existing_regime_cache)} symbols; training {len(missing)} missing: {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        except Exception as e:
            log.warning(f"[Regime] Failed to load cache: {e}, retraining all...")

    # Start from existing cached data, only add missing symbols
    _regime_cache = dict(_existing_regime_cache)
    data_source = "LIVE MT5" if use_live_mt5 else "22M dataset"
    log.info(f"[Regime] Training regime detection on {data_source}...")

    for symbol in [s for s in symbols if s not in _regime_cache]:
        try:
            parquet_file = parquet_dir / f"{symbol}_M1.parquet"
            if not parquet_file.exists():
                log.warning(f"[Volatility] No data for {symbol}, skipping")
                continue

            df = pd.read_parquet(parquet_file)
            log.info(f"[Volatility] Loaded {len(df):,} bars for {symbol}")

            # Calculate trend indicators
            df["ema_20"] = df["close"].ewm(span=20).mean()
            df["ema_50"] = df["close"].ewm(span=50).mean()
            df["ema_200"] = df["close"].ewm(span=200).mean()

            # Calculate ADX for trend strength
            df["high_low"] = df["high"] - df["low"]
            df["high_close"] = abs(df["high"] - df["close"].shift(1))
            df["low_close"] = abs(df["low"] - df["close"].shift(1))
            df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)

            plus_dm = df["high"] - df["high"].shift(1)
            minus_dm = df["low"].shift(1) - df["low"]
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

            atr = df["tr"].ewm(span=14, adjust=False).mean()
            plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr
            minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
            adx = dx.ewm(span=14, adjust=False).mean()

            # Calculate slope
            df["slope_20"] = df["close"].diff(20)
            df["slope_50"] = df["close"].diff(50)

            # Regime classification
            current_ema20 = df["ema_20"].iloc[-1]
            current_ema50 = df["ema_50"].iloc[-1]
            current_ema200 = df["ema_200"].iloc[-1]
            current_adx = adx.iloc[-1]
            current_slope20 = df["slope_20"].iloc[-1]
            current_slope50 = df["slope_50"].iloc[-1]

            # Determine regime
            if current_adx > 25:
                # Strong trend
                if current_ema20 > current_ema50 and current_ema20 > current_ema200:
                    regime = "STRONG_UPTREND"
                elif current_ema20 < current_ema50 and current_ema20 < current_ema200:
                    regime = "STRONG_DOWNTREND"
                else:
                    regime = "TRENDING"
            elif current_adx > 20:
                # Moderate trend
                if current_ema20 > current_ema50:
                    regime = "UPTREND"
                elif current_ema20 < current_ema50:
                    regime = "DOWNTREND"
                else:
                    regime = "WEAK_TREND"
            else:
                # Ranging
                regime = "RANGING"

            # Trend strength (0-1)
            trend_strength = min(1.0, current_adx / 50.0)

            regime_state = {
                "regime": regime,
                "trend_strength": float(trend_strength),
                "adx": float(current_adx),
                "ema20": float(current_ema20),
                "ema50": float(current_ema50),
                "ema200": float(current_ema200),
                "trend_direction": 1 if current_ema20 > current_ema50 else -1,
                "slope_20": float(current_slope20),
                "slope_50": float(current_slope50),
                "is_uptrend": bool(current_ema20 > current_ema50),
                "is_strong_trend": bool(current_adx > 25),
            }

            _regime_cache[symbol] = regime_state

            log.info(f"[Regime] {symbol} -> {regime} (ADX={current_adx:.1f}, strength={trend_strength:.2f})")

        except Exception as e:
            log.error(f"[Regime] Failed to train {symbol}: {e}")

    # Save cache
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        # Convert numpy types to native Python types for JSON serialization
        json_safe_cache = {}
        for k, v in _regime_cache.items():
            json_safe_cache[k] = {
                kk: float(vv) if hasattr(vv, "item") else (bool(vv) if isinstance(vv, (np.bool_, bool)) else vv) for kk, vv in v.items()
            }
        with open(cache_file, "w") as f:
            json.dump(json_safe_cache, f, indent=2)
        log.info("[Regime] Saved regime states to cache")
    except Exception as e:
        log.error(f"[Regime] Failed to save cache: {e}")

    return _regime_cache


def llm_enabled() -> bool:
    if not LLM_MANAGER_AVAILABLE or _llm_manager is None:
        return False
    if not is_llm_enabled():
        return False
    return _llm_manager.check_availability()


def _parse_llm_json_response(text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    if "{" not in text or "}" not in text:
        return None
    try:
        json_blob = text[text.find("{") : text.rfind("}") + 1]
        data = json.loads(json_blob)
        
        # STRUCTURAL REPAIR: Remap drifted keys from small models
        if "data_analysis" in data and "pair_analysis" not in data:
            data["pair_analysis"] = {}
            for item in data["data_analysis"]:
                if isinstance(item, dict) and "symbol" in item:
                    # Map "symbol": "EUR/USD" -> "EURUSD"
                    sym = item["symbol"].replace("/", "").upper()
                    data["pair_analysis"][sym] = str(item.get("assessment", "Stable"))
        
        # Ensure overall_summary exists
        if "Overall Market Sentiment" in data and "overall_summary" not in data:
            data["overall_summary"] = data["Overall Market Sentiment"]
            
        return data
    except Exception:
        return None


def llm_review_volatility_regime(vol_clusters: Dict[str, Dict], regime_states: Dict[str, Dict]) -> Optional[Dict[str, Any]]:
    if not llm_enabled():
        return None
    
    # Load upcoming high-impact news events for correlation
    _news_events = []
    try:
        _cal = Path(__file__).resolve().parents[2] / "CORE_MODULES" / "results" / "news_calendar.json"
        if _cal.exists():
            import json as _json
            _raw = _json.loads(_cal.read_text(encoding="utf-8"))
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            cutoff = now.replace(hour=23, minute=59)
            for e in _raw:
                if e.get("_meta"):
                    continue
                if e.get("impact", "").upper() != "HIGH":
                    continue
                dt_str = e.get("datetime_utc") or e.get("timestamp", "")
                if dt_str:
                    try:
                        evt_time = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                        if now <= evt_time <= cutoff:
                            _news_events.append({
                                "event": e.get("event_name", e.get("name", "Unknown")),
                                "currency": e.get("currency", "?"),
                                "time": dt_str,
                                "pairs": e.get("affected_pairs", [])
                            })
                    except Exception:
                        pass
    except Exception as _e:
        log.debug(f"Could not load news events: {_e}")
    
    blocked_symbols = {
        item.strip().upper().split(".")[0]
        for item in os.getenv("BLOCKED_SYMBOLS", "").split(",")
        if item.strip()
    }

    full_summary = {}
    for symbol, vol in vol_clusters.items():
        if symbol.upper().split(".")[0] in blocked_symbols:
            continue
        regime = regime_states.get(symbol, {})
        full_summary[symbol] = {
            "volatility": vol.get("current_regime"),
            "pctile": round(vol.get("volatility_percentile", 0), 3),
            "regime": regime.get("regime"),
            "strength": round(regime.get("trend_strength", 0), 3),
            "adx": round(regime.get("adx", 0), 1)
        }

    # BATCH & MERGE: Split 19 symbols into batches of 7 to prevent context degradation
    symbols = list(full_summary.keys())
    batch_size = 7
    batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
    
    final_pair_analysis = {}
    final_risk_flags = []
    overall_summaries = []
    
    news_context = ""
    if _news_events:
        news_context = f"""
### UPCOMING HIGH-IMPACT NEWS (next 24h):
{json.dumps(_news_events, indent=2)}

Use this news context to identify potential catalysts for regime shifts or volatility expansions."""
    
    for i, batch in enumerate(batches):
        batch_data = {s: full_summary[s] for s in batch}
        
        prompt = f"""You are CAVALIER's local Qwen live auditor, a professional trading system auditor with deep market analysis expertise.
### MISSION:
Provide a highly concise, brief analysis of the volatility and regime state for a BATCH of {len(batch_data)} symbols (Part {i+1}/{len(batches)}).
{news_context}

### DATA TO REVIEW:
{json.dumps(batch_data, indent=1)}

### MANDATORY CONSTRAINTS & PERFORMANCE TUNING FOR CPU:
1. Review EVERY symbol in this specific BATCH list (count: {len(batch_data)}).
2. DO NOT hallucinate symbols from previous context or other asset classes.
3. Your responses MUST be extremely brief to prevent execution timeouts.
4. Limit the "overall_summary" to exactly 1 short sentence (max 15 words).
5. For each symbol in "pair_analysis", limit each assessment field to a maximum of 10 words.
6. Return ONLY raw JSON matching the schema below.

### EXPECTED SCHEMA:
{{
  "overall_summary": "1 short sentence summary for this batch highlighting themes/news (max 15 words)",
  "risk_flags": ["symbol: brief risk reason"],
  "pair_analysis": {{
    "SYMBOL": {{
      "volatility_assessment": "1-sentence volatility state (max 10 words)",
      "regime_assessment": "1-sentence regime type (max 10 words)",
      "trade_implications": "1-sentence implication (max 10 words)",
      "news_correlation": "1-sentence news impact or 'none' (max 10 words)",
      "risk_notes": "1-sentence key risk (max 10 words)"
    }}
  }}
}}
"""
        response = _llm_manager.query(prompt, max_tokens=1024, json_format=True) if _llm_manager else None
        data = _parse_llm_json_response(response)
        
        if data:
            if "pair_analysis" in data:
                final_pair_analysis.update(data["pair_analysis"])
            if "risk_flags" in data and isinstance(data["risk_flags"], list):
                final_risk_flags.extend(data["risk_flags"])
            if "overall_summary" in data:
                overall_summaries.append(data["overall_summary"])

    if not final_pair_analysis:
        return None

    return {
        "overall_summary": " | ".join(overall_summaries),
        "risk_flags": list(set(final_risk_flags)),  # Dedupe flags
        "pair_analysis": final_pair_analysis,
        "pairs_reviewed_count": len(final_pair_analysis)
    }


def initialize_volatility_and_regime(symbols: list, force_retrain: bool = False, use_live_mt5: bool = False):
    """
    Initialize both volatility clustering and regime training

    This should be called before trading cycles begin.

    Args:
        symbols: List of symbols to train
        force_retrain: Force retrain even if cache exists
        use_live_mt5: Use live MT5 data instead of cached parquet files
    """
    log.info("=" * 60)
    data_source = "LIVE MT5" if use_live_mt5 else "22M dataset"
    log.info(f"[INIT] Starting volatility clustering & regime training on {data_source}")
    log.info("=" * 60)

    start_time = datetime.now()

    # Train volatility clustering
    vol_clusters = train_volatility_clustering(symbols, force_retrain, use_live_mt5)

    # Train regime detection
    regime_states = train_regime_detection(symbols, force_retrain, use_live_mt5)

    elapsed = (datetime.now() - start_time).total_seconds()

    log.info("=" * 60)
    log.info(f"[INIT] Volatility & Regime training complete in {elapsed:.1f}s")
    log.info(f"[INIT] Trained {len(vol_clusters)} volatility clusters")
    log.info(f"[INIT] Trained {len(regime_states)} regime states")
    log.info("[INIT] Ready for trading cycles")
    log.info("=" * 60)

    llm_review = None
    try:
        llm_review = llm_review_volatility_regime(vol_clusters, regime_states)
    except Exception as e:
        log.warning(f"[INIT] LLM volatility/regime review failed: {e}")
    if llm_review:
        try:
            review_file = Path(__file__).resolve().parents[3] / "DATA_MODELS" / "results" / "llm_volatility_regime_review.json"
            review_file.parent.mkdir(parents=True, exist_ok=True)
            with open(review_file, "w") as f:
                json.dump(llm_review, f, indent=2)
            log.info(f"[INIT] Saved LLM volatility/regime review to {review_file}")
        except Exception as e:
            log.warning(f"[INIT] Failed to save LLM review: {e}")

    return {
        "volatility": vol_clusters,
        "regime": regime_states,
    }


def get_volatility_cluster(symbol: str) -> Optional[Dict]:
    """Get cached volatility cluster for symbol"""
    if _volatility_cache is None:
        return None
    return _volatility_cache.get(symbol)


def get_regime_state(symbol: str) -> Optional[Dict]:
    """Get cached regime state for symbol"""
    if _regime_cache is None:
        return None
    return _regime_cache.get(symbol)
