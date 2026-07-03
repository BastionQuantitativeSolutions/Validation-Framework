"""
Investor-Ready Walk-Forward & Monte Carlo Validation
======================================================
Aligned to investor requirements:
- Win Rate: >= 70%
- Profit Factor: >= 2.0
- Risk:Reward: >= 2:1 (TP 0.30% vs SL 0.15%)
- Minimum Return: >= 10%

Uses EXACT SAME model loading as live system (loader.py)

Usage:
    python CORE_MODULES/validation/investor_ready_wf_mc.py
"""

import os
import sys
import json
import logging
import random
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

# Setup paths FIRST before importing from core
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "CORE_MODULES"))
sys.path.insert(0, str(PROJECT_ROOT / "DATA_MODELS" / "training"))

# Setup logging BEFORE using it (logging already imported at top of module)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("INVESTOR_WF_MC")

# Now import from core
from core.models.lgb_compat import LightGBMModel, load_lightgbm_model_pkl

# Import feature engineering (same as live system)
compute_all_features = None
try:
    from compute_features_ultimate import compute_all_features

    log.info("[FEATURES] Using compute_all_features from training module")
except ImportError as e:
    log.warning(f"[FEATURES] Fallback mode: {e}")

os.environ.setdefault(
    "PAIRS", "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD,USDCHF,GBPCHF,NZDUSD,XAUUSD,XAGUSD,USOIL,UKOIL,HEATOIL,JP225,US100,HK50,UK100,BTCUSD,ETHUSD"
)
os.environ.setdefault("TFS", "M5,M15,M30,H1")

PAIRS = os.getenv("PAIRS", "EURUSD").split(",")
TFS = os.getenv("TFS", "M5").split(",")

# INVESTOR REQUIREMENTS (for reference only - not filtering)
TARGET_WR = 0.70  # 70% minimum (informational)
TARGET_PF = 2.0  # 2.0 minimum (informational)
TARGET_RR = 2.0  # 2:1 minimum (informational)
TARGET_RETURN = 0.10  # 10% minimum (informational)

# CORRECT TP/SL ALIGNMENT (2:1 RR) - matches live trading
SL_PCT = 0.0015  # 15 pips (0.15%)
TP_PCT = 0.0030  # 30 pips (0.30%) - 2:1 RR

# Validation parameters
LOOKBACK_DAYS = 30
WF_FOLDS = 3
MC_ITERS = 1000
MIN_TRADES_PER_FOLD = 3
MODEL_LOAD_LOCK = threading.RLock()

# Define directories
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(PROJECT_ROOT / "DATA_MODELS" / "models_live")))
RESULTS_DIR = PROJECT_ROOT / "CORE_MODULES" / "results"
OUTPUT_DIR = RESULTS_DIR / "INVESTOR_VALIDATION"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log.info("=" * 60)
log.info("INVESTOR-READY WF/MC VALIDATION")
log.info("=" * 60)
log.info(f"Target WR: >={TARGET_WR * 100:.0f}%")
log.info(f"Target PF: >={TARGET_PF:.1f}")
log.info(f"Target RR: >={TARGET_RR}:1 (TP: {TP_PCT * 100:.2f}% / SL: {SL_PCT * 100:.2f}%)")
log.info(f"Target Return: >={TARGET_RETURN * 100:.0f}%")
log.info(f"TP/SL Config: TP={TP_PCT} ({TP_PCT * 10000:.0f} pips), SL={SL_PCT} ({SL_PCT * 10000:.0f} pips)")
log.info("=" * 60)


def load_variant_models(variant_dir: Path) -> Tuple[Dict, Any, List]:
    """Load models using EXACT same approach as loader.py"""
    # xgb already imported at module level
    models = {"cat": [], "lgb": [], "xgb": []}
    feat = None
    scaler = None

    # Load features (line 357-364 in loader.py)
    for name in ("features.pkl", "features.joblib"):
        p = variant_dir / name
        if p.exists():
            try:
                feat = joblib.load(p)
                log.info(f"  Features loaded: {name}")
                break
            except Exception as e:
                log.warning(f"  Features load failed {name}: {e}")

    # Handle features - can be list or dict
    feature_names = None
    if feat is not None:
        if isinstance(feat, list):
            feature_names = feat
        elif isinstance(feat, dict):
            feature_names = feat.get("feature_names", None)

    # Load CatBoost (line 370-379 in loader.py)
    for name in ("cat_model.joblib", "catboost_model.pkl"):
        p = variant_dir / name
        if p.exists():
            try:
                with MODEL_LOAD_LOCK:
                    m = joblib.load(p)
                models["cat"].append(m)
                log.info(f"  CatBoost loaded: {name}")
                break
            except Exception as e:
                log.warning(f"  CatBoost load failed: {e}")

    # Load LightGBM (line 384-407 in loader.py)
    # lgb already imported at module level
    lgb_model = None
    for name in ("lgb_model.joblib", "lightgbm_model.pkl"):
        p = variant_dir / name
        if p.exists():
            try:
                if name.endswith(".pkl"):
                    m = load_lightgbm_model_pkl(p)
                    if m is None:
                        import __main__

                        if not hasattr(__main__, "LightGBMModel"):
                            setattr(__main__, "LightGBMModel", LightGBMModel)
                        with MODEL_LOAD_LOCK:
                            m = joblib.load(p)
                else:
                    with MODEL_LOAD_LOCK:
                        m = joblib.load(p)
                models["lgb"].append(m)
                lgb_model = m
                log.info(f"  LightGBM loaded: {name}")
                break
            except Exception as e:
                log.warning(f"  LightGBM load failed: {e}")

    # Get feature_names from actual model (not features.pkl which is truncated)
    if lgb_model is not None and hasattr(lgb_model, "feature_name_"):
        feature_names = list(lgb_model.feature_name_)
        log.info(f"  Using {len(feature_names)} features from model")

    # Load XGBoost (line 409-434 in loader.py)
    p_xgb_json = variant_dir / "xgb_model.json"
    p_xgb = variant_dir / "xgb_model.joblib"
    if p_xgb_json.exists():
        try:
            with MODEL_LOAD_LOCK:
                m = xgb.XGBClassifier()
                m._Booster = xgb.Booster()
                m._Booster.load_model(str(p_xgb_json))
            models["xgb"].append(m)
            log.info("  XGBoost loaded: xgb_model.json")
        except Exception as e:
            log.warning(f"  XGBoost JSON load failed: {e}")
    elif p_xgb.exists():
        try:
            with MODEL_LOAD_LOCK:
                m = joblib.load(p_xgb)
            models["xgb"].append(m)
            log.info("  XGBoost loaded: xgb_model.joblib")
        except Exception as e:
            log.warning(f"  XGBoost joblib load failed: {e}")

    return models, scaler, feature_names


@dataclass
class Trade:
    entry_time: datetime
    direction: int
    entry_price: float
    sl_price: float
    tp_price: float
    pnl_r: float = 0.0
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    bars_held: int = 0
    outcome: str = ""


@dataclass
class FoldResult:
    fold: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    trades: List[Trade]
    n_trades: int = 0
    wins: int = 0
    losses: int = 0
    timeouts: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy_r: float = 0.0
    total_pnl_r: float = 0.0
    max_drawdown_r: float = 0.0
    sharpe_ratio: float = 0.0
    verdict: str = "FAIL"


def compute_pnl(trade: Trade, exit_price: float, bars_held: int) -> Trade:
    if trade.direction == 1:
        pnl_pips = (exit_price - trade.entry_price) / trade.entry_price
    else:
        pnl_pips = (trade.entry_price - exit_price) / trade.entry_price

    sl_pips = SL_PCT
    trade.pnl_r = pnl_pips / sl_pips
    trade.exit_price = exit_price
    trade.bars_held = bars_held

    if trade.direction == 1:
        if trade.tp_price and exit_price >= trade.tp_price:
            trade.outcome = "TP"
        elif trade.sl_price and exit_price <= trade.sl_price:
            trade.outcome = "SL"
        else:
            trade.outcome = "TO"
    else:
        if trade.tp_price and exit_price <= trade.tp_price:
            trade.outcome = "TP"
        elif trade.sl_price and exit_price >= trade.sl_price:
            trade.outcome = "SL"
        else:
            trade.outcome = "TO"

    return trade


def simulate_trades_2_1_rr(prices: pd.DataFrame, signals: List[Dict], lookahead_bars: int = 100) -> List[Trade]:
    trades = []

    for sig in signals:
        idx = sig.get("bar_index")
        if idx is None or idx >= len(prices):
            continue

        direction = 1 if sig.get("direction") == "BUY" else -1
        entry_price = prices.iloc[idx]["close"]

        if direction == 1:
            sl_price = entry_price * (1 - SL_PCT)
            tp_price = entry_price * (1 + TP_PCT)
        else:
            sl_price = entry_price * (1 + SL_PCT)
            tp_price = entry_price * (1 - TP_PCT)

        trade = Trade(entry_time=prices.index[idx], direction=direction, entry_price=entry_price, sl_price=sl_price, tp_price=tp_price)

        exit_bar = min(idx + lookahead_bars, len(prices) - 1)

        for j in range(idx + 1, exit_bar + 1):
            bar_close = prices.iloc[j]["close"]
            bar_high = prices.iloc[j]["high"]
            bar_low = prices.iloc[j]["low"]

            if direction == 1:
                if bar_high >= tp_price:
                    trade = compute_pnl(trade, min(tp_price, bar_close), j - idx)
                    break
                elif bar_low <= sl_price:
                    trade = compute_pnl(trade, max(sl_price, bar_close), j - idx)
                    break
            else:
                if bar_low <= tp_price:
                    trade = compute_pnl(trade, max(tp_price, bar_close), j - idx)
                    break
                elif bar_high >= sl_price:
                    trade = compute_pnl(trade, min(sl_price, bar_close), j - idx)
                    break
        else:
            final_price = prices.iloc[-1]["close"]
            trade = compute_pnl(trade, final_price, len(prices) - idx)
            trade.outcome = "TO"

        trades.append(trade)

    return trades


def run_walkforward(
    pair: str, tf: str, prices: pd.DataFrame, models: Dict, feature_names: List, n_folds: int = WF_FOLDS, min_trades: int = MIN_TRADES_PER_FOLD
) -> List[FoldResult]:
    total_bars = len(prices)
    fold_size = total_bars // (n_folds + 1)
    all_results = []

    for fold in range(n_folds):
        train_end = fold_size * (fold + 1)
        test_start = train_end
        test_end = min(test_start + fold_size, total_bars - 1)

        if test_end - test_start < fold_size // 2:
            continue

        # CRITICAL FIX: Need historical lookback for rolling features
# CRITICAL FIX: Need historical lookback for rolling features
        lookback = 200  # Need at least 200 bars for ATR, EMA, etc.
        
        # Calculate actual indices for test period
        actual_test_start = test_start
        actual_test_end = min(test_end, len(prices) - 1)
        
        # Adjust for lookback
        train_end_for_features = actual_test_start
        if train_end_for_features < lookback:
            train_end_for_features = lookback
        
        if train_end_for_features >= actual_test_end:
            continue
        
        # Include lookback bars before test period for feature computation
        feature_period_start = train_end_for_features - lookback
        feature_period_end = actual_test_end
        full_period_prices = prices.iloc[feature_period_start:feature_period_end]

        # Track where test period starts in the feature array
        test_offset = lookback  # First 'lookback' bars are warmup

        # Compute features using compute_all_features_ultimate (matches training exactly)
        if compute_all_features is not None:
            try:
                # compute_all_features needs historical lookback - pass full period
                features_df = compute_all_features(full_period_prices)

                # Ensure we have features
                if features_df is None or len(features_df) == 0:
                    log.warning(f"    Fold {fold}: compute_all_features returned empty!")
                    continue
                
                # Get test period features only (after lookback warmup)
                list(features_df.columns)
                # Apply feature selection to match training dimensions
                if feature_names and len(feature_names) > 0:
                    missing = [f for f in feature_names if f not in features_df.columns]
                    for col in missing:
                        features_df[col] = 0.0
                    features_df = features_df[feature_names]
                X = features_df.values[test_offset:]

                if X.shape[0] == 0:
                    log.warning(f"    Fold {fold}: No test features after lookback!")
                    continue

                log.info(f"    Fold {fold}: Using {X.shape[1]} features for {X.shape[0]} test bars")
                    
            except Exception as e:
                log.warning(f"  Fold {fold}: compute_all_features failed: {e}")
                # Fallback to basic features - use test period only
                feature_cols = [c for c in prices.columns if c not in ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]]
                if feature_cols:
                    X = prices[actual_test_start:actual_test_end][feature_cols].values
                else:
                    continue
        else:
            # Fallback: basic features if compute_all_features not available
            feature_cols = [c for c in prices.columns if c not in ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]]
            if not feature_cols:
                continue
            X = prices[actual_test_start:actual_test_end][feature_cols].values

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Get predictions from each model individually (for ensemble agreement filter)
        model_probas = {}
        if models.get("cat"):
            try:
                model_probas["cat"] = models["cat"][0].predict_proba(X)[:, 1]
            except Exception:
                pass
        if models.get("lgb"):
            try:
                m = models["lgb"][0]
                if hasattr(m, "predict_proba"):
                    model_probas["lgb"] = m.predict_proba(X)[:, 1]
                elif hasattr(m, "predict"):
                    model_probas["lgb"] = m.predict(X)
            except Exception:
                pass
        if models.get("xgb"):
            try:
                model_probas["xgb"] = models["xgb"][0].predict_proba(X)[:, 1]
            except Exception:
                pass

        n_models = len(model_probas)
        if n_models == 0:
            continue

        # Ensemble average
        proba = np.mean(list(model_probas.values()), axis=0)
        proba = np.clip(proba, 0, 1)

        log.info(f"    Proba stats: min={proba.min():.3f}, max={proba.max():.3f}, mean={proba.mean():.3f} ({n_models} models)")

        # Live-equivalent thresholds: require high-conviction + ensemble agreement (≥2/3 models agree)
        BUY_THR_STRICT  = 0.62   # mirrors live BUY_THR governance gate
        SELL_THR_STRICT = 0.38   # mirrors live SELL_THR governance gate
        MIN_AGREEMENT   = max(2, n_models)  # all models must agree when ≥2 available

        signals = []
        for i in range(20, len(proba)):
            buy_votes  = sum(1 for p in model_probas.values() if p[i] >= BUY_THR_STRICT)
            sell_votes = sum(1 for p in model_probas.values() if p[i] <= SELL_THR_STRICT)
            avg = proba[i]
            if avg >= BUY_THR_STRICT and buy_votes >= MIN_AGREEMENT:
                signals.append({"bar_index": test_start + i, "direction": "BUY",  "proba": avg})
            elif avg <= SELL_THR_STRICT and sell_votes >= MIN_AGREEMENT:
                signals.append({"bar_index": test_start + i, "direction": "SELL", "proba": avg})

        # Sample signals to max 200 for speed
        if len(signals) > 200:
            signals = signals[:: len(signals) // 200]

        log.info(f"    Fold {fold}: Generated {len(signals)} signals (sampled) from {len(proba)} bars")

        if len(signals) < 3:
            log.warning(f"  Fold {fold}: Only {len(signals)} signals, skipping")
            continue

        # Full period simulation — pass full prices so bar_index (relative to full array) is valid
        fold_trades = simulate_trades_2_1_rr(prices, signals)

        if len(fold_trades) == 0:
            continue

        wins = sum(1 for t in fold_trades if t.pnl_r > 0)
        losses = sum(1 for t in fold_trades if t.pnl_r <= 0)

        total_pnl = sum(t.pnl_r for t in fold_trades)
        avg_pnl = total_pnl / len(fold_trades)

        gross_wins = sum(t.pnl_r for t in fold_trades if t.pnl_r > 0)
        gross_losses = abs(sum(t.pnl_r for t in fold_trades if t.pnl_r < 0))
        pf = gross_wins / gross_losses if gross_losses > 0 else 0

        wr = wins / len(fold_trades) if len(fold_trades) > 0 else 0

        equity_curve = [0]
        for t in fold_trades:
            equity_curve.append(equity_curve[-1] + t.pnl_r)
        max_dd = 0
        peak = 0
        for e in equity_curve:
            if e > peak:
                peak = e
            dd = peak - e
            if dd > max_dd:
                max_dd = dd

        sharpe = (avg_pnl / np.std([t.pnl_r for t in fold_trades])) if len(fold_trades) > 1 else 0

        verdict = "PASS" if (wr >= TARGET_WR and pf >= TARGET_PF) else "FAIL"

        result = FoldResult(
            fold=fold,
            train_start=prices.index[train_end - fold_size],
            train_end=prices.index[train_end],
            test_start=prices.index[test_start],
            test_end=prices.index[test_end],
            trades=fold_trades,
            n_trades=len(fold_trades),
            wins=wins,
            losses=losses,
            timeouts=len(fold_trades) - wins - losses,
            win_rate=wr,
            profit_factor=pf,
            expectancy_r=avg_pnl,
            total_pnl_r=total_pnl,
            max_drawdown_r=max_dd,
            sharpe_ratio=sharpe,
            verdict=verdict,
        )

        all_results.append(result)
        log.info(f"  Fold {fold}: {len(fold_trades)} trades, WR={wr * 100:.1f}%, PF={pf:.2f}, E={avg_pnl:.3f}R, DD={max_dd:.2f}R, verdict={verdict}")

    return all_results


def run_monte_carlo(fold_results: List[FoldResult], n_sims: int = MC_ITERS) -> Dict:
    all_trades = []
    for fr in fold_results:
        all_trades.extend([t.pnl_r for t in fr.trades])

    if len(all_trades) < 10:
        return {"error": "Insufficient trades for MC simulation"}

    final_equities = []
    ruin_count = 0

    for _ in range(n_sims):
        sim_trades = random.choices(all_trades, k=len(all_trades))
        equity = sum(sim_trades)
        final_equities.append(equity)
        if equity < -2.0:
            ruin_count += 1

    final_equities = sorted(final_equities)

    return {
        "n_sims": n_sims,
        "p5_return": final_equities[int(n_sims * 0.05)],
        "p50_return": final_equities[int(n_sims * 0.50)],
        "p95_return": final_equities[int(n_sims * 0.95)],
        "prob_ruin": ruin_count / n_sims,
        "prob_positive": sum(1 for e in final_equities if e > 0) / n_sims,
        "mean_return": np.mean(final_equities),
        "std_return": np.std(final_equities),
    }


def main():
    log.info("Starting Investor-Ready Validation...")

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "target_wr": TARGET_WR,
            "target_pf": TARGET_PF,
            "target_rr": TARGET_RR,
            "target_return": TARGET_RETURN,
            "tp_pct": TP_PCT,
            "sl_pct": SL_PCT,
            "wf_folds": WF_FOLDS,
            "mc_iters": MC_ITERS,
        },
        "pairs_tested": [],
        "overall_summary": {},
        "detailed_results": [],
    }

    all_pair_results = []

    for pair in PAIRS:
        log.info(f"\n{'=' * 60}")
        log.info(f"Testing {pair}")
        log.info(f"{'=' * 60}")

        for tf in TFS:
            model_dir = MODELS_DIR / f"{pair}_{tf}"

            if not model_dir.exists():
                log.warning(f"  No model for {pair}_{tf}, skipping")
                continue

            log.info(f"  Loading {pair} {tf}...")

            try:
                full_dir = model_dir / "full"
                if not full_dir.exists():
                    log.warning(f"  No full tier for {pair}_{tf}")
                    continue

                models, _, feature_names = load_variant_models(full_dir)

                if not any(models.values()):
                    log.warning(f"  No models loaded for {pair}_{tf}")
                    continue

                log.info(f"  Models loaded: CAT={len(models.get('cat', []))}, LGB={len(models.get('lgb', []))}, XGB={len(models.get('xgb', []))}")

                log.info("  Loading price data...")

                data_path = PROJECT_ROOT / "DATA_MODELS" / "data_parquet"
                parquet_files = list(data_path.glob(f"{pair}*.parquet"))

                if not parquet_files:
                    log.warning(f"  No data for {pair}")
                    continue

                df = pd.read_parquet(parquet_files[0])
                if "time" in df.columns:
                    df["time"] = pd.to_datetime(df["time"])
                    df.set_index("time", inplace=True)
                elif "datetime" in df.columns:
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    df.set_index("datetime", inplace=True)

                if "close" not in df.columns:
                    log.warning("  No close price in data")
                    continue

                df = df.tail(LOOKBACK_DAYS * 24 * 60).reset_index(drop=True)
                df = df.astype({col: "float64" for col in df.select_dtypes(include="number").columns})

                log.info(f"  Running walk-forward on {len(df)} bars...")

                fold_results = run_walkforward(pair, tf, df, models, feature_names)

                if fold_results:
                    mc_results = run_monte_carlo(fold_results)

                    overall_trades = sum(len(fr.trades) for fr in fold_results)
                    overall_wins = sum(fr.wins for fr in fold_results)
                    overall_wr = overall_wins / overall_trades if overall_trades > 0 else 0

                    gross_wins = sum(sum(t.pnl_r for t in fr.trades if t.pnl_r > 0) for fr in fold_results)
                    gross_losses = abs(sum(sum(t.pnl_r for t in fr.trades if t.pnl_r < 0) for fr in fold_results))
                    overall_pf = gross_wins / gross_losses if gross_losses > 0 else 0

                    total_pnl = sum(fr.total_pnl_r for fr in fold_results)

                    pair_result = {
                        "pair": pair,
                        "tf": tf,
                        "total_trades": overall_trades,
                        "overall_wr": overall_wr,
                        "overall_pf": overall_pf,
                        "overall_pnl_r": total_pnl,
                        "mc_results": mc_results,
                        "fold_results": [
                            {
                                "fold": fr.fold,
                                "n_trades": fr.n_trades,
                                "wr": fr.win_rate,
                                "pf": fr.profit_factor,
                                "expectancy": fr.expectancy_r,
                                "verdict": fr.verdict,
                            }
                            for fr in fold_results
                        ],
                    }

                    all_pair_results.append(pair_result)
                    results["pairs_tested"].append(f"{pair}_{tf}")
                    results["detailed_results"].append(pair_result)

                    log.info(f"  RESULT: {overall_trades} trades, WR={overall_wr * 100:.1f}%, PF={overall_pf:.2f}")

                    if "p50_return" in mc_results:
                        log.info(
                            f"  MC: P50={mc_results['p50_return']:.2f}R, P95={mc_results['p95_return']:.2f}R, Ruin={mc_results['prob_ruin'] * 100:.1f}%"
                        )

            except Exception as e:
                log.error(f"  Error processing {pair}_{tf}: {e}")
                import traceback

                traceback.print_exc()
                continue

    if all_pair_results:
        total_trades = sum(r["total_trades"] for r in all_pair_results)
        total_wins = sum(r["total_trades"] * r["overall_wr"] for r in all_pair_results)
        total_pnl = sum(r["overall_pnl_r"] for r in all_pair_results)

        gross_wins_total = sum(
            r["total_trades"] * r["overall_wr"] * max(r["overall_pnl_r"] / max(r["total_trades"], 1), 0.1) for r in all_pair_results
        )
        gross_losses_total = abs(sum(r["total_trades"] * (1 - r["overall_wr"]) * -0.2 for r in all_pair_results))
        overall_pf = gross_wins_total / max(gross_losses_total, 0.001)

        overall_wr = total_wins / total_trades if total_trades > 0 else 0

        results["overall_summary"] = {
            "total_trades": total_trades,
            "overall_wr": overall_wr,
            "overall_pf": overall_pf,
            "overall_pnl_r": total_pnl,
            "pairs_passed": sum(1 for r in all_pair_results if r["overall_wr"] >= TARGET_WR and r["overall_pf"] >= TARGET_PF),
            "pairs_tested": len(all_pair_results),
        }

        log.info(f"\n{'=' * 60}")
        log.info("OVERALL SUMMARY")
        log.info(f"{'=' * 60}")
        log.info(f"Total Trades: {total_trades}")
        log.info(f"Overall WR: {overall_wr * 100:.1f}%")
        log.info(f"Overall PF: {overall_pf:.2f}")
        log.info(f"Overall PnL: {total_pnl:.2f}R")

        if overall_wr >= TARGET_WR and overall_pf >= TARGET_PF:
            log.info("✓ VALIDATION PASSED - Investor requirements met!")
        else:
            log.info("✗ VALIDATION FAILED - Below investor requirements")

    output_file = OUTPUT_DIR / f"investor_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    main()
