"""
Walk-Forward Threshold Optimizer
=================================
Per-pair/per-TF rolling threshold optimization for maximum profitable exposure.

For each walk-forward fold:
  1. Train period: collect model predictions (no trading)
  2. Validation period: grid-search BUY/SELL thresholds to maximize PF or TotalR
  3. Test period: apply optimized thresholds and simulate trades

Uses actual fixed code paths: ensemble.py, constants.py, unified_exits.py.
"""

import os
import sys
import json
import logging
import random
import math
import warnings
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("WF_OPT")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "DATA_MODELS" / "models_live"
RESULTS_DIR = PROJECT_ROOT / "CORE_MODULES" / "results"
OUTPUT_DIR = RESULTS_DIR / "LIVE_EQ_VALIDATION"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "DATA_MODELS" / "training"))
sys.path.insert(0, str(PROJECT_ROOT / "CORE_MODULES"))

from core.config.constants import BASE_BUY_THRESHOLD, BASE_SELL_THRESHOLD, W_SMC
from core.unified_exits import calculate_sl_tp
from core.models.ensemble import model_feature_names, align_df

try:
    from compute_features_ultimate import compute_all_features
except ImportError as e:
    log.error(f"[FEATURES] Cannot import: {e}")
    compute_all_features = None

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

# =============================================================================
# CONFIG
# =============================================================================
PAIRS = os.getenv("PAIRS", "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD").split(",")
TFS = ["M5", "M15", "M30", "H1"]
LOOKBACK_DAYS = 60
WF_FOLDS = 6
LOOKAHEAD_BARS = 100
TRAIN_PCT = 0.50
VAL_PCT = 0.20
TEST_PCT = 0.30

# Threshold search space
THR_GRID = np.round(np.arange(0.52, 0.72, 0.02), 3).tolist()

# =============================================================================
# DATA FETCHING
# =============================================================================
def init_mt5() -> bool:
    if mt5 is None:
        return False
    return mt5.initialize()

def fetch_data(pair: str, tf: str, days: int = 60) -> Optional[pd.DataFrame]:
    if mt5 and mt5.initialize():
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=days)
        tf_map = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1}
        rates = mt5.copy_rates_range(pair, tf_map.get(tf, mt5.TIMEFRAME_M5), start, now)
        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)
            df.sort_index(inplace=True)
            return df
    # Parquet fallback
    p = Path(f"DATA_MODELS/data_1y_backtest/{pair}_{tf}_1Y.parquet")
    if p.exists():
        df = pd.read_parquet(p)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
        df.sort_index(inplace=True)
        return df
    return None

# =============================================================================
# MODEL LOADING
# =============================================================================
def load_models(pair: str, tf: str) -> Dict[str, List[Any]]:
    import joblib
    model_dir = MODELS_DIR / f"{pair}_{tf}"
    if not model_dir.exists():
        return {}
    families = {"cat": [], "lgb": [], "xgb": []}
    for tier in ["full", "core", "minimal"]:
        tier_dir = model_dir / tier
        if not tier_dir.is_dir():
            continue
        for mtype in ["cat", "lgb", "xgb"]:
            mpath = tier_dir / f"{mtype}_model.joblib"
            if mpath.exists():
                try:
                    families[mtype].append(joblib.load(str(mpath)))
                except Exception:
                    pass
    return {k: v for k, v in families.items() if v}

def _batch_predict_family(features_df: pd.DataFrame, models: List[Any], fam: str) -> Optional[np.ndarray]:
    if not models:
        return None
    preds = []
    for m in models:
        try:
            feats = model_feature_names(m)
            Xin = align_df(features_df, feats) if feats else features_df
            p = m.predict_proba(Xin)[:, 1] if hasattr(m, "predict_proba") else m.predict(Xin)
            preds.append(p)
        except Exception:
            continue
    return np.mean(preds, axis=0) if preds else None

def predict_ensemble(features_df: pd.DataFrame, models_by_family: Dict[str, List[Any]]) -> np.ndarray:
    if not models_by_family:
        return np.full(len(features_df), 0.5)
    cat_arr = _batch_predict_family(features_df, models_by_family.get("cat", []), "cat")
    lgb_arr = _batch_predict_family(features_df, models_by_family.get("lgb", []), "lgb")
    xgb_arr = _batch_predict_family(features_df, models_by_family.get("xgb", []), "xgb")
    active = []
    weights = {"cat": 0.5, "lgb": 0.3, "xgb": 0.2}
    wsum = 0.0
    for arr, w in [(cat_arr, weights["cat"]), (lgb_arr, weights["lgb"]), (xgb_arr, weights["xgb"])]:
        if arr is not None:
            active.append((arr, w))
            wsum += w
    if not active:
        return np.full(len(features_df), 0.5)
    fused = np.zeros(len(features_df))
    for arr, w in active:
        fused += arr * w
    fused /= wsum
    return np.clip(fused, 0, 1).astype(float)

# =============================================================================
# FEATURES & REGIME
# =============================================================================
def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, low_p, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - low_p, (h - c.shift(1)).abs(), (low_p - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def detect_regime(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    returns = close.pct_change()
    vol = returns.rolling(50).std()
    ema_f = close.ewm(span=12, adjust=False).mean()
    ema_s = close.ewm(span=26, adjust=False).mean()
    macd = ema_f - ema_s
    macd_hist = macd - macd.ewm(span=9, adjust=False).mean()
    h, low_p = df["high"], df["low"]
    pdm = h.diff().clip(lower=0)
    mdm = (-low_p.diff()).clip(lower=0)
    tr = calculate_atr(df, 14)
    pdi = 100 * (pdm.rolling(14).mean() / tr)
    mdi = 100 * (mdm.rolling(14).mean() / tr)
    dx = 100 * abs(pdi - mdi) / (pdi + mdi)
    adx = dx.rolling(14).mean()
    regime = pd.Series("RANGING", index=df.index)
    regime[vol > vol.quantile(0.8)] = "VOLATILE"
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
# SIGNAL & SIMULATION
# =============================================================================
def directional_fusion(ml_prob: float, smc_conf: float) -> Tuple[float, int]:
    _ml_dir = 1 if ml_prob >= 0.5 else -1
    _confidence = abs(ml_prob - 0.5) * 2.0
    _smc_boost = smc_conf * W_SMC
    _boosted_confidence = min(1.0, _confidence + _smc_boost)
    fused = float(np.clip(0.5 + (_ml_dir * _boosted_confidence * 0.5), 0, 1))
    if fused >= BASE_SELL_THRESHOLD:
        return fused, 1
    elif fused <= BASE_BUY_THRESHOLD:
        return fused, -1
    return fused, 0

def simulate_bar(df: pd.DataFrame, entry_idx: int, direction: int, sl: float, tp: float, atr: float, max_bars: int = 100) -> Tuple[float, str, int]:
    entry_price = float(df.iloc[entry_idx]["close"])
    for i in range(entry_idx + 1, min(entry_idx + max_bars, len(df))):
        high = float(df.iloc[i]["high"])
        low = float(df.iloc[i]["low"])
        if direction == 1:
            if low <= sl:
                return (sl - entry_price) / atr if atr > 0 else -1.0, "LOSS", i
            elif high >= tp:
                return (tp - entry_price) / atr if atr > 0 else 1.0, "WIN", i
        else:
            if high >= sl:
                return (entry_price - sl) / atr if atr > 0 else -1.0, "LOSS", i
            elif low <= tp:
                return (entry_price - tp) / atr if atr > 0 else 1.0, "WIN", i
    exit_price = float(df.iloc[min(entry_idx + max_bars, len(df) - 1)]["close"])
    if direction == 1:
        pnl_r = (exit_price - entry_price) / atr if atr > 0 else 0.0
    else:
        pnl_r = (entry_price - exit_price) / atr if atr > 0 else 0.0
    outcome = "WIN" if pnl_r > 0 else "LOSS" if pnl_r < 0 else "BREAKEVEN"
    return pnl_r, outcome, min(entry_idx + max_bars, len(df) - 1)

def run_backtest_segment(df: pd.DataFrame, proba: np.ndarray, smc_conf: pd.Series, regime: pd.Series, atr: pd.Series, start: int, end: int, buy_thr: float, sell_thr: float) -> List[dict]:
    trades = []
    in_trade = False
    entry_bar = 0
    for i in range(start, end):
        if in_trade:
            if i >= entry_bar + LOOKAHEAD_BARS:
                in_trade = False
            continue
        if pd.isna(atr.iloc[i]) or atr.iloc[i] <= 0:
            continue
        ml_prob = float(proba[i])
        smc = float(smc_conf.iloc[i]) if i < len(smc_conf) else 0.5
        fused, direction = directional_fusion(ml_prob, smc)
        if direction == 0:
            continue
        # Apply optimized thresholds (override constants for this segment)
        if direction == 1 and fused < buy_thr:
            continue
        if direction == -1 and fused > sell_thr:
            continue
        entry_price = float(df.iloc[i]["close"])
        sl, tp = calculate_sl_tp(entry_price, direction, float(atr.iloc[i]), str(regime.iloc[i]) if i < len(regime) else "RANGING")
        pnl_r, outcome, exit_i = simulate_bar(df, i, direction, sl, tp, float(atr.iloc[i]))
        trades.append({"pnl_r": pnl_r, "outcome": outcome, "ml_prob": ml_prob, "fused": fused, "direction": direction, "entry_idx": i, "exit_idx": exit_i})
        in_trade = True
        entry_bar = i
    return trades

def evaluate_thresholds(df: pd.DataFrame, proba: np.ndarray, smc_conf: pd.Series, regime: pd.Series, atr: pd.Series, val_start: int, val_end: int) -> Tuple[float, float, float]:
    """Grid-search BUY/SELL thresholds on validation data. Returns (best_buy_thr, best_sell_thr, best_score)."""
    best_score = -9999.0
    best_buy = BASE_SELL_THRESHOLD
    best_sell = BASE_BUY_THRESHOLD
    
    for buy_thr in THR_GRID:
        for sell_thr in [t for t in THR_GRID if t <= buy_thr - 0.06]:
            trades = run_backtest_segment(df, proba, smc_conf, regime, atr, val_start, val_end, buy_thr, sell_thr)
            if not trades:
                continue
            wins = [t for t in trades if t["outcome"] == "WIN"]
            losses = [t for t in trades if t["outcome"] == "LOSS"]
            total_r = sum(t["pnl_r"] for t in trades)
            n = len(trades)
            len(wins) / n
            win_r = sum(t["pnl_r"] for t in wins)
            loss_r = abs(sum(t["pnl_r"] for t in losses))
            pf = win_r / loss_r if loss_r > 0 else 999
            
            # Score: prefer high PF with decent trade count
            # Use PF * sqrt(n) as objective, but penalize negative total R
            if total_r < 0:
                score = total_r  # raw negative return
            else:
                score = pf * math.sqrt(n)
            
            if score > best_score:
                best_score = score
                best_buy = buy_thr
                best_sell = sell_thr
    
    return best_buy, best_sell, best_score

# =============================================================================
# MAIN WF LOOP
# =============================================================================
def run_pair_tf(pair: str, tf: str) -> Optional[Dict]:
    log.info(f"\n{'='*50}")
    log.info(f"OPTIMIZING: {pair} {tf}")
    log.info(f"{'='*50}")
    
    df = fetch_data(pair, tf, LOOKBACK_DAYS)
    if df is None or len(df) < 500:
        log.warning(f"[SKIP] {pair}_{tf}: insufficient data")
        return None
    
    models = load_models(pair, tf)
    if not models:
        log.warning(f"[SKIP] {pair}_{tf}: no models")
        return None
    
    # Compute features on full history (no leakage in backtest segment logic)
    try:
        features = compute_all_features(df)
    except Exception as e:
        log.warning(f"[SKIP] {pair}_{tf}: feature error {e}")
        return None
    
    regime = detect_regime(df)
    atr = calculate_atr(df)
    if "confluence_score" in features.columns:
        smc_conf = features["confluence_score"].clip(0, 1)
    elif "confluence_net" in features.columns:
        smc_conf = (features["confluence_net"] + 1) / 2
    else:
        smc_conf = pd.Series(0.5, index=features.index)
    
    proba = predict_ensemble(features, models)
    
    n = len(df)
    fold_size = n // WF_FOLDS
    
    all_fold_results = []
    all_test_trades = []
    
    for fold in range(WF_FOLDS):
        test_start = fold * fold_size
        test_end = min((fold + 1) * fold_size, n)
        val_start = max(0, test_start - int(fold_size * 0.5))
        val_end = test_start
        
        if test_end - test_start < 100:
            continue
        
        log.info(f"[FOLD {fold+1}/{WF_FOLDS}] train={val_start}, val={val_end-val_start}, test={test_end-test_start}")
        
        # Optimize thresholds on validation period
        opt_buy, opt_sell, opt_score = evaluate_thresholds(df, proba, smc_conf, regime, atr, val_start, val_end)
        log.info(f"[FOLD {fold+1}] Optimal thresholds: BUY>={opt_buy:.3f}, SELL<={opt_sell:.3f}, score={opt_score:.2f}")
        
        # Run test with optimized thresholds
        test_trades = run_backtest_segment(df, proba, smc_conf, regime, atr, test_start, test_end, opt_buy, opt_sell)
        
        wins = [t for t in test_trades if t["outcome"] == "WIN"]
        losses = [t for t in test_trades if t["outcome"] == "LOSS"]
        wr = len(wins) / len(test_trades) * 100 if test_trades else 0
        total_r = sum(t["pnl_r"] for t in test_trades)
        win_r = sum(t["pnl_r"] for t in wins)
        loss_r = abs(sum(t["pnl_r"] for t in losses))
        pf = win_r / loss_r if loss_r > 0 else 0
        
        log.info(f"[FOLD {fold+1}] TEST: {len(test_trades)} trades, WR={wr:.1f}%, PF={pf:.2f}, TotalR={total_r:.2f}")
        
        all_fold_results.append({
            "fold": fold + 1,
            "opt_buy_thr": opt_buy,
            "opt_sell_thr": opt_sell,
            "test_trades": len(test_trades),
            "test_wins": len(wins),
            "test_losses": len(losses),
            "test_wr": wr,
            "test_pf": pf,
            "test_total_r": total_r,
        })
        all_test_trades.extend(test_trades)
    
    if not all_test_trades:
        return None
    
    wins = [t for t in all_test_trades if t["outcome"] == "WIN"]
    losses = [t for t in all_test_trades if t["outcome"] == "LOSS"]
    wr = len(wins) / len(all_test_trades) * 100
    total_r = sum(t["pnl_r"] for t in all_test_trades)
    win_r = sum(t["pnl_r"] for t in wins)
    loss_r = abs(sum(t["pnl_r"] for t in losses))
    pf = win_r / loss_r if loss_r > 0 else 0
    
    log.info(f"[SUMMARY {pair}_{tf}] {len(all_test_trades)} trades, WR={wr:.1f}%, PF={pf:.2f}, TotalR={total_r:.2f}")
    
    return {
        "pair": pair,
        "tf": tf,
        "folds": all_fold_results,
        "total_trades": len(all_test_trades),
        "win_rate": wr,
        "profit_factor": pf,
        "total_r": total_r,
        "trades": all_test_trades,
    }

# =============================================================================
# MONTE CARLO
# =============================================================================
def run_mc(trades: List[dict], n_sims: int = 2000) -> dict:
    if not trades:
        return {}
    pnls = [t["pnl_r"] for t in trades]
    risk = 0.0117
    final_eqs = []
    max_dds = []
    surv = 0
    prof = 0
    for _ in range(n_sims):
        sim = random.choices(pnls, k=len(pnls))
        eq = [1.0]
        for p in sim:
            eq.append(eq[-1] * (1 + p * risk))
        final_eqs.append(eq[-1])
        rm = np.maximum.accumulate(eq)
        dd = (eq - rm) / rm
        max_dds.append(abs(dd.min()))
        if min(eq) > 0.5:
            surv += 1
        if eq[-1] > 1.0:
            prof += 1
    return {
        "final_equity_p50": float(np.percentile(final_eqs, 50)),
        "max_dd_p95": float(np.percentile(max_dds, 95)),
        "survival_rate": surv / n_sims,
        "prob_profit": prof / n_sims,
    }

# =============================================================================
# MAIN
# =============================================================================
def main():
    log.info("=" * 60)
    log.info("WALK-FORWARD THRESHOLD OPTIMIZER")
    log.info("=" * 60)
    
    init_mt5()
    
    all_results = []
    all_trades = []
    
    for pair in PAIRS:
        pair = pair.strip()
        for tf in TFS:
            res = run_pair_tf(pair, tf)
            if res:
                all_results.append(res)
                all_trades.extend(res["trades"])
    
    # Global summary
    if all_trades:
        wins = [t for t in all_trades if t["outcome"] == "WIN"]
        losses = [t for t in all_trades if t["outcome"] == "LOSS"]
        wr = len(wins) / len(all_trades) * 100
        total_r = sum(t["pnl_r"] for t in all_trades)
        win_r = sum(t["pnl_r"] for t in wins)
        loss_r = abs(sum(t["pnl_r"] for t in losses))
        pf = win_r / loss_r if loss_r > 0 else 0
        
        log.info("\n" + "=" * 60)
        log.info("GLOBAL SUMMARY")
        log.info("=" * 60)
        log.info(f"Total trades: {len(all_trades)}")
        log.info(f"Win Rate: {wr:.1f}%")
        log.info(f"Profit Factor: {pf:.2f}")
        log.info(f"Total R: {total_r:.2f}")
        
        mc = run_mc(all_trades)
        log.info(f"MC Equity p50: {mc.get('final_equity_p50', 0):.3f}")
        log.info(f"MC MaxDD p95: {mc.get('max_dd_p95', 0):.1%}")
        log.info(f"MC Survival: {mc.get('survival_rate', 0):.1%}")
        log.info(f"MC ProbProfit: {mc.get('prob_profit', 0):.1%}")
    
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {"thr_grid": THR_GRID, "wf_folds": WF_FOLDS},
        "pair_results": all_results,
        "global": {
            "total_trades": len(all_trades),
            "win_rate": len(wins) / len(all_trades) * 100 if all_trades else 0,
            "profit_factor": pf if all_trades else 0,
            "total_r": total_r if all_trades else 0,
            "monte_carlo": mc if all_trades else {},
        },
    }
    
    out_path = OUTPUT_DIR / f"wf_threshold_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"\nResults saved to: {out_path}")
    
    if mt5 is not None:
        mt5.shutdown()

if __name__ == "__main__":
    main()
