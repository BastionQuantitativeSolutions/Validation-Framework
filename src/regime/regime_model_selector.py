"""
# Author: JG
REGIME-AWARE MODEL SELECTOR
===========================
Uses validation cache data to select best model for current regime/volatility.

Integration:
- Pulls from CORE_MODULES/validation/results/validation_cache/
- Cross-references with current market regime (from volatility_regime_hook)
- Returns: best model, confidence adjustment, position size multiplier
"""

import pickle
import os
from typing import Dict
from dataclasses import dataclass

CACHE_PATH = "CORE_MODULES/validation/results/validation_cache/"
VALIDATION_RESULTS_PATH = "CORE_MODULES/analysis/llm_powered_optimization.json"

REGIME_MODEL_PREFERENCES = {
    "STRONG_UPTREND": {"cat": 1.1, "lgb": 1.0, "xgb": 0.9},
    "STRONG_DOWNTREND": {"cat": 1.1, "lgb": 1.0, "xgb": 0.9},
    "TRENDING": {"cat": 1.05, "lgb": 1.0, "xgb": 0.95},
    "RANGING": {"lgb": 1.1, "cat": 1.0, "xgb": 0.9},
    "LOW_VOLATILITY": {"cat": 1.05, "xgb": 1.0, "lgb": 0.95},
    "HIGH_VOLATILITY": {"xgb": 1.1, "lgb": 1.0, "cat": 0.95},
}


@dataclass
class ModelSelection:
    symbol: str
    best_model: str
    best_timeframe: str
    base_accuracy: float
    regime_multiplier: float
    final_confidence: float
    size_multiplier: float
    reasoning: str


def load_validation_cache() -> Dict:
    """Load all validation cache data."""
    cache = {}
    symbols = [
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "AUDUSD",
        "USDCAD",
        "USDCHF",
        "GBPCHF",
        "XAGUSD",
        "NZDUSD",
        "XAUUSD",
        "USOIL",
        "UKOIL",
        "HEATOIL",
        "JP225",
        "US100",
        "HK50",
        "UK100",
    ]
    timeframes = ["M5", "M15", "M30", "H1"]
    models = ["cat", "lgb", "xgb"]

    for symbol in symbols:
        cache[symbol] = {}
        for tf in timeframes:
            cache[symbol][tf] = {}
            for model in models:
                f = f"{CACHE_PATH}result_{symbol}_{tf}_full_{model}.pkl"
                if os.path.exists(f):
                    with open(f, "rb") as p:
                        cache[symbol][tf][model] = pickle.load(p)
                else:
                    cache[symbol][tf][model] = None
    return cache


def get_best_model_for_conditions(
    symbol: str,
    regime: str = "TRENDING",
    volatility: str = "MEDIUM",
    current_timeframe: str = "M5",
) -> ModelSelection:
    """
    Get best model based on validation cache + regime + volatility.

    Args:
        symbol: Trading pair
        regime: Current market regime (STRONG_UPTREND, STRONG_DOWNTREND, etc.)
        volatility: Volatility level (LOW, MEDIUM, HIGH)
        current_timeframe: What TF we're trading on

    Returns:
        ModelSelection with best model, confidence, size multiplier
    """
    cache = load_validation_cache()

    if symbol not in cache:
        return ModelSelection(
            symbol=symbol,
            best_model="cat",
            best_timeframe="M5",
            base_accuracy=0.65,
            regime_multiplier=1.0,
            final_confidence=0.65,
            size_multiplier=1.0,
            reasoning="No cache data, defaulting to CatBoost M5",
        )

    best_score = 0
    best_model = "cat"
    best_tf = "M5"
    base_acc = 0.65

    for tf in ["M5", "M15", "M30", "H1"]:
        for model in ["cat", "lgb", "xgb"]:
            data = cache[symbol].get(tf, {}).get(model)
            if data is None:
                continue

            wf = data.get("walk_forward", {})
            acc = wf.get("accuracy", 0.65)

            regime_pref = REGIME_MODEL_PREFERENCES.get(regime, {})
            model_mult = regime_pref.get(model, 1.0)

            if volatility == "LOW_VOLATILITY":
                tf_mult = 1.0 if tf in ["M5", "M15"] else 0.95
            elif volatility == "HIGH_VOLATILITY":
                tf_mult = 1.0 if tf in ["M30", "H1"] else 0.95
            else:
                tf_mult = 1.0

            score = acc * model_mult * tf_mult

            if score > best_score:
                best_score = score
                best_model = model
                best_tf = tf
                base_acc = acc

    regime_mult = REGIME_MODEL_PREFERENCES.get(regime, {}).get(best_model, 1.0)
    final_conf = max(0.01, min(0.85, base_acc * regime_mult))

    size_mult = final_conf / 0.65

    reasoning = f"{symbol}: {best_model.upper()} on {best_tf} | Base acc: {base_acc:.1%} | Regime: {regime} | Final conf: {final_conf:.1%}"

    return ModelSelection(
        symbol=symbol,
        best_model=best_model,
        best_timeframe=best_tf,
        base_accuracy=base_acc,
        regime_multiplier=regime_mult,
        final_confidence=final_conf,
        size_multiplier=size_mult,
        reasoning=reasoning,
    )


def get_all_symbol_recommendations(regime: str, volatility: str) -> Dict:
    """Get model recommendations for all symbols."""
    symbols = [
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "AUDUSD",
        "USDCAD",
        "USDCHF",
        "GBPCHF",
        "XAGUSD",
        "NZDUSD",
        "XAUUSD",
        "USOIL",
        "UKOIL",
        "HEATOIL",
        "JP225",
        "US100",
        "HK50",
        "UK100",
    ]
    recommendations = {}

    for symbol in symbols:
        sel = get_best_model_for_conditions(symbol, regime, volatility)
        recommendations[symbol] = {
            "model": sel.best_model,
            "tf": sel.best_timeframe,
            "confidence": sel.final_confidence,
            "size_mult": sel.size_multiplier,
            "reasoning": sel.reasoning,
        }

    return recommendations


def create_llm_regime_prompt(regime: str, volatility: str, recommendations: Dict) -> str:
    """Create LLM prompt for regime-aware analysis."""

    rec_summary = "\n".join(
        [f"- {s}: {r['model'].upper()} @ {r['tf']} | conf={r['confidence']:.1%} | size={r['size_mult']:.2f}x" for s, r in recommendations.items()]
    )

    prompt = f"""You are an expert trading system analyst. Current market conditions:

REGIME: {regime}
VOLATILITY: {volatility}

MODEL RECOMMENDATIONS (from validation cache + regime filter):
{rec_summary}

Provide:
1. Overall market assessment (1-2 sentences)
2. Which symbols should we FOCUS ON based on model confidence?
3. Which symbols should we AVOID or reduce size on?
4. Position sizing recommendations
5. Any special rules for current conditions?

Be concise and actionable.
"""

    return prompt


if __name__ == "__main__":
    print("=" * 70)
    print("REGIME-AWARE MODEL SELECTOR")
    print("=" * 70)

    print("\n[1] LOADING VALIDATION CACHE...")
    cache = load_validation_cache()
    print(f"   Loaded cache for {len(cache)} symbols")

    print("\n[2] GETTING RECOMMENDATIONS FOR CURRENT REGIME...")
    regime = "STRONG_DOWNTREND"
    volatility = "MEDIUM"

    print(f"   Regime: {regime}")
    print(f"   Volatility: {volatility}")

    recs = get_all_symbol_recommendations(regime, volatility)

    print("\n   RECOMMENDATIONS:")
    for symbol, rec in recs.items():
        print(f"   {symbol}: {rec['model'].upper()} @ {rec['tf']} | conf={rec['confidence']:.1%} | size={rec['size_mult']:.2f}x")

    print("\n[3] CREATING LLM PROMPT...")
    prompt = create_llm_regime_prompt(regime, volatility, recs)
    print(f"   Prompt length: {len(prompt)} chars")

    print("\n" + "=" * 70)
    print("REGIME-AWARE SELECTOR READY")
    print("=" * 70)
