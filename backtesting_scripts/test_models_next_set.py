#!/usr/bin/env python3
"""
Test Next Set Models — Bar-by-Bar Backtest on Recent Data
===========================================================
Loads all 304 trained models and tests them on the most recent out-of-sample
bars from parquet files. Simulates trades with the same SL/TP used during
training (0.5×ATR SL, 0.5×ATR TP) and records equity curves.

Usage:
    python CORE_MODULES/backtesting/test_models_next_set.py --days 30
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARQUET_DIR = PROJECT_ROOT / "DATA_MODELS" / "data_parquet"
MODELS_DIR = PROJECT_ROOT / "DATA_MODELS" / "models_live"
RESULTS_DIR = PROJECT_ROOT / "CORE_MODULES" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT / "DATA_MODELS" / "training"))
sys.path.insert(0, str(PROJECT_ROOT / "CORE_MODULES" / "training"))

# ── Config ────────────────────────────────────────────────────────────────────
PAIRS = [
    "EURUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "GBPCHF",
    "NZDUSD", "XAUUSD", "XAGUSD", "USOIL", "UKOIL", "HEATOIL",
    "JP225", "US100", "HK50", "UK100", "GBPUSD", "BTCUSD", "ETHUSD",
]
TFS = ["M5", "M15", "M30", "H1"]
TIERS = ["core", "full"]
VARIANTS = ["buy_bias", "sell_bias"]

LOOKBACK_BARS = 600  # Enough for 500-bar momentum + rolling windows
TEST_DAYS_DEFAULT = 30

# Trading params (same as training labels)
SL_ATR = 0.5
TP_ATR = 0.5
MAX_HOLD_BARS = 100

# Minimum probability to take a trade
MIN_BUY_PROB = 0.55
MIN_SELL_PROB = 0.55

# Bars per day by timeframe
BARS_PER_DAY = {"M5": 288, "M15": 96, "M30": 48, "H1": 24}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_parquet(pair: str, tf: str) -> Optional[pd.DataFrame]:
    """Load parquet, preferring the most recent data source."""
    candidates = [
        PARQUET_DIR / f"{pair}_{tf}.parquet",
        PARQUET_DIR / f"{pair}_{tf}_LIVE.parquet",
    ]
    for path in candidates:
        if path.exists():
            try:
                df = pd.read_parquet(path)
                # Handle time column / index
                if df.index.name == "time" or "time" in df.index.names:
                    df = df.reset_index()
                if "time" not in df.columns and "timestamp" in df.columns:
                    df = df.rename(columns={"timestamp": "time"})
                if "time" in df.columns:
                    df["time"] = pd.to_datetime(df["time"])
                    return df.sort_values("time").reset_index(drop=True)
                # No time column: generate synthetic dates for compatibility
                df["time"] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq="min")
                return df.reset_index(drop=True)
            except Exception as e:
                log.warning("Error loading %s: %s", path.name, e)
                continue
    return None


# ============================================================================
# FEATURES & FUSION (same pipeline as training)
# ============================================================================

def compute_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute base features using compute_features_ultimate if available."""
    try:
        from compute_features_ultimate import compute_all_features
        return compute_all_features(df)
    except ImportError as e:
        log.warning("compute_features_ultimate unavailable: %s", e)
        return pd.DataFrame(index=df.index)


def detect_regime(df: pd.DataFrame) -> pd.Series:
    """Simple regime detection matching training."""
    close = df["close"]
    returns = close.pct_change()
    vol = returns.rolling(50).std()
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    regime = pd.Series("RANGING", index=df.index)
    regime[vol > vol.quantile(0.8)] = "VOLATILE"
    regime[(ema_fast > ema_slow) & (vol <= vol.quantile(0.8))] = "TRENDING"
    regime[(ema_fast < ema_slow) & (vol <= vol.quantile(0.8))] = "TRENDING_DOWN"
    return regime


def load_model_and_features(pair: str, tf: str, tier: str, variant: str):
    """Load model + selected feature list for a specific variant."""
    out_dir = MODELS_DIR / f"{pair}_{tf}" / tier / variant
    model_path = out_dir / "lgb_model.joblib"
    if not model_path.exists():
        model_path = out_dir / "cat_model.joblib"
    if not model_path.exists():
        return None, None

    model = joblib.load(model_path)
    features = joblib.load(out_dir / "features.pkl")
    return model, list(features)


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

@dataclass
class SimTrade:
    entry_time: datetime
    entry_price: float
    direction: int
    sl: float
    tp: float
    bars_held: int = 0
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: str = ""
    pnl_pct: float = 0.0


class ModelBacktest:
    def __init__(self, pair: str, tf: str, tier: str,
                 buy_model, buy_features, sell_model, sell_features,
                 prob_threshold: float = 0.55):
        self.pair = pair
        self.tf = tf
        self.tier = tier
        self.buy_model = buy_model
        self.buy_features = buy_features
        self.sell_model = sell_model
        self.sell_features = sell_features
        self.prob_threshold = prob_threshold
        self.trades: List[SimTrade] = []

    def _predict(self, model, features, X_row: pd.DataFrame) -> float:
        """Return probability of class 1."""
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_row)[0]
                return float(proba[1]) if len(proba) == 2 else float(proba[0])
            elif hasattr(model, "predict"):
                return float(model.predict(X_row)[0])
        except Exception as e:
            log.debug("Prediction error: %s", e)
        return 0.5

    def run(self, df: pd.DataFrame, features_df: pd.DataFrame,
            regime_series: pd.Series, test_mask: pd.Series) -> Dict:
        """Walk-forward simulation on test bars."""
        n = len(df)
        test_idx = df[test_mask].index
        if len(test_idx) == 0:
            return {"trades": 0, "win_rate": 0.0, "avg_pnl_pct": 0.0}

        # Ensure alignment
        features_df = features_df.reset_index(drop=True)
        regime_series = regime_series.reset_index(drop=True)
        df = df.reset_index(drop=True)

        # Recompute test mask on reset index
        test_start = n - len(test_idx)

        open_trade: Optional[SimTrade] = None
        sample_probs = []  # For debugging

        for i in range(test_start, n):
            # Update open trade
            if open_trade is not None:
                open_trade.bars_held += 1
                bar = df.iloc[i]
                if open_trade.direction == 1:
                    if bar["low"] <= open_trade.sl:
                        open_trade.exit_price = open_trade.sl
                        open_trade.exit_time = bar["time"]
                        open_trade.exit_reason = "SL"
                        open_trade.pnl_pct = (open_trade.exit_price - open_trade.entry_price) / open_trade.entry_price
                        self.trades.append(open_trade)
                        open_trade = None
                    elif bar["high"] >= open_trade.tp:
                        open_trade.exit_price = open_trade.tp
                        open_trade.exit_time = bar["time"]
                        open_trade.exit_reason = "TP"
                        open_trade.pnl_pct = (open_trade.exit_price - open_trade.entry_price) / open_trade.entry_price
                        self.trades.append(open_trade)
                        open_trade = None
                    elif open_trade.bars_held >= MAX_HOLD_BARS:
                        open_trade.exit_price = bar["close"]
                        open_trade.exit_time = bar["time"]
                        open_trade.exit_reason = "TIME"
                        open_trade.pnl_pct = (open_trade.exit_price - open_trade.entry_price) / open_trade.entry_price
                        self.trades.append(open_trade)
                        open_trade = None
                else:
                    if bar["high"] >= open_trade.sl:
                        open_trade.exit_price = open_trade.sl
                        open_trade.exit_time = bar["time"]
                        open_trade.exit_reason = "SL"
                        open_trade.pnl_pct = (open_trade.entry_price - open_trade.exit_price) / open_trade.entry_price
                        self.trades.append(open_trade)
                        open_trade = None
                    elif bar["low"] <= open_trade.tp:
                        open_trade.exit_price = open_trade.tp
                        open_trade.exit_time = bar["time"]
                        open_trade.exit_reason = "TP"
                        open_trade.pnl_pct = (open_trade.entry_price - open_trade.exit_price) / open_trade.entry_price
                        self.trades.append(open_trade)
                        open_trade = None
                    elif open_trade.bars_held >= MAX_HOLD_BARS:
                        open_trade.exit_price = bar["close"]
                        open_trade.exit_time = bar["time"]
                        open_trade.exit_reason = "TIME"
                        open_trade.pnl_pct = (open_trade.entry_price - open_trade.exit_price) / open_trade.entry_price
                        self.trades.append(open_trade)
                        open_trade = None

            if open_trade is not None:
                continue

            # Need enough history for feature row
            if i < 10:
                continue

            # Get feature row
            try:
                row = features_df.iloc[i:i + 1]
            except Exception:
                continue

            bar = df.iloc[i]
            atr = (df["high"] - df["low"]).rolling(14).mean().iloc[i]
            if pd.isna(atr) or atr <= 0:
                continue

            # Buy signal
            buy_prob = 0.0
            if self.buy_model is not None and self.buy_features:
                X_buy = row[self.buy_features].replace([np.inf, -np.inf], np.nan).fillna(0)
                buy_prob = self._predict(self.buy_model, self.buy_features, X_buy)

            # Sell signal
            sell_prob = 0.0
            if self.sell_model is not None and self.sell_features:
                X_sell = row[self.sell_features].replace([np.inf, -np.inf], np.nan).fillna(0)
                sell_prob = self._predict(self.sell_model, self.sell_features, X_sell)

            if len(sample_probs) < 5:
                sample_probs.append((buy_prob, sell_prob))

            entry = bar["close"]

            if buy_prob >= self.prob_threshold and buy_prob > sell_prob:
                sl = entry - atr * SL_ATR
                tp = entry + atr * TP_ATR
                open_trade = SimTrade(
                    entry_time=bar["time"],
                    entry_price=entry,
                    direction=1,
                    sl=sl,
                    tp=tp,
                )
            elif sell_prob >= self.prob_threshold and sell_prob > buy_prob:
                sl = entry + atr * SL_ATR
                tp = entry - atr * TP_ATR
                open_trade = SimTrade(
                    entry_time=bar["time"],
                    entry_price=entry,
                    direction=-1,
                    sl=sl,
                    tp=tp,
                )

        # Close any open trade at end
        if open_trade is not None:
            last_bar = df.iloc[-1]
            open_trade.exit_price = last_bar["close"]
            open_trade.exit_time = last_bar["time"]
            open_trade.exit_reason = "END"
            if open_trade.direction == 1:
                open_trade.pnl_pct = (open_trade.exit_price - open_trade.entry_price) / open_trade.entry_price
            else:
                open_trade.pnl_pct = (open_trade.entry_price - open_trade.exit_price) / open_trade.entry_price
            self.trades.append(open_trade)

        # Metrics
        total = len(self.trades)
        wins = sum(1 for t in self.trades if t.pnl_pct > 0)
        losses = total - wins
        win_rate = wins / total if total > 0 else 0.0
        avg_pnl = np.mean([t.pnl_pct for t in self.trades]) if total > 0 else 0.0
        avg_win = np.mean([t.pnl_pct for t in self.trades if t.pnl_pct > 0]) if wins > 0 else 0.0
        avg_loss = np.mean([t.pnl_pct for t in self.trades if t.pnl_pct <= 0]) if losses > 0 else 0.0
        total_pnl = sum(t.pnl_pct for t in self.trades)
        gross_profit = sum(t.pnl_pct for t in self.trades if t.pnl_pct > 0)
        gross_loss = sum(t.pnl_pct for t in self.trades if t.pnl_pct <= 0)
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
        max_dd = 0.0
        peak = 0.0
        running = 0.0
        for t in self.trades:
            running += t.pnl_pct
            peak = max(peak, running)
            max_dd = max(max_dd, peak - running)

        return {
            "pair": self.pair,
            "tf": self.tf,
            "tier": self.tier,
            "trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 4),
            "avg_pnl_pct": round(avg_pnl, 6),
            "avg_win_pct": round(avg_win, 6),
            "avg_loss_pct": round(avg_loss, 6),
            "total_pnl_pct": round(total_pnl, 6),
            "profit_factor": round(profit_factor, 2),
            "max_dd_pct": round(max_dd, 6),
            "avg_bars": round(np.mean([t.bars_held for t in self.trades]), 1) if total > 0 else 0.0,
            "sample_probs": sample_probs,
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=TEST_DAYS_DEFAULT, help="Test on last N days of data")
    parser.add_argument("--threshold", type=float, default=0.55, help="Min probability to trade")
    parser.add_argument("--pairs", type=str, default="ALL", help="Comma-separated or ALL")
    parser.add_argument("--tfs", type=str, default="M5,M15,M30,H1", help="Timeframes")
    parser.add_argument("--tiers", type=str, default="core,full", help="Tiers")
    parser.add_argument("--debug", action="store_true", help="Show prediction samples")
    args = parser.parse_args()

    pairs = PAIRS if args.pairs.upper() == "ALL" else args.pairs.split(",")
    tfs = args.tfs.split(",")
    tiers = args.tiers.split(",")

    log.info("=" * 70)
    log.info("NEXT SET MODEL BACKTEST — Last %d days", args.days)
    log.info("Pairs: %s | TFs: %s | Tiers: %s | Threshold: %.2f", pairs, tfs, tiers, args.threshold)
    log.info("=" * 70)

    all_results: List[Dict] = []
    total_trades = 0

    for pair in pairs:
        for tf in tfs:
            df = load_parquet(pair, tf)
            if df is None or len(df) < LOOKBACK_BARS + 100:
                log.warning("Insufficient data for %s %s (%d bars)", pair, tf, len(df) if df is not None else 0)
                continue

            # Limit to last N bars for speed (same as training)
            MAX_BACKTEST_BARS = 50_000
            if len(df) > MAX_BACKTEST_BARS:
                df = df.iloc[-MAX_BACKTEST_BARS:].reset_index(drop=True)

            # Determine test period — always use bar-count based for consistency
            # (date-based fails for synthetic dates and data with recency gaps)
            bars_per_day = BARS_PER_DAY.get(tf, 96)
            test_bars = min(args.days * bars_per_day, len(df) // 3)
            test_mask = pd.Series(False, index=df.index)
            test_mask.iloc[-test_bars:] = True

            if test_mask.sum() < 50:
                log.warning("Not enough test bars for %s %s (%d)", pair, tf, test_mask.sum())
                continue

            log.info("\n--- %s %s ---", pair, tf)
            log.info("Total: %d bars | Test: %d bars | Date range: %s to %s",
                     len(df), test_mask.sum(),
                     str(df["time"].min()) if "time" in df.columns else "N/A",
                     str(df["time"].max()) if "time" in df.columns else "N/A")

            # Compute features once on full window
            features_df = compute_base_features(df)
            if features_df.empty:
                log.warning("Feature computation failed for %s %s", pair, tf)
                continue

            # Regime
            regime = detect_regime(df)

            # Fusion (optional - if feature_fusion is available)
            try:
                from feature_fusion import FeatureFusion
                fusion = FeatureFusion()
                features_df = fusion.inject_features(features_df, symbol=pair, regime_series=regime)
            except Exception as e:
                log.debug("Fusion skipped: %s", e)

            for tier in tiers:
                buy_model, buy_feats = load_model_and_features(pair, tf, tier, "buy_bias")
                sell_model, sell_feats = load_model_and_features(pair, tf, tier, "sell_bias")

                if buy_model is None and sell_model is None:
                    log.warning("  No models for %s %s %s", pair, tf, tier)
                    continue

                bt = ModelBacktest(pair, tf, tier, buy_model, buy_feats, sell_model, sell_feats, args.threshold)
                result = bt.run(df, features_df, regime, test_mask)
                result["buy_model"] = buy_model is not None
                result["sell_model"] = sell_model is not None
                all_results.append(result)
                total_trades += result["trades"]

                log.info("  %s tier | Trades: %d | WR: %.1f%% | Total: %.4f%% | PF: %.2f | MaxDD: %.4f%%",
                         tier, result["trades"], result["win_rate"] * 100,
                         result["total_pnl_pct"] * 100, result["profit_factor"],
                         result["max_dd_pct"] * 100)
                if args.debug and result.get("sample_probs"):
                    log.info("    Sample probs (buy, sell): %s", result["sample_probs"])

    # Summary
    log.info("\n" + "=" * 70)
    log.info("BACKTEST COMPLETE")
    log.info("=" * 70)

    if not all_results:
        log.info("No results generated.")
        return

    # Aggregate
    total_all = sum(r["trades"] for r in all_results)
    wins_all = sum(r["wins"] for r in all_results)
    total_pnl_all = sum(r["total_pnl_pct"] for r in all_results)
    active_models = [r for r in all_results if r["trades"] > 0]
    avg_wr = np.mean([r["win_rate"] for r in active_models]) if active_models else 0.0
    avg_pf = np.mean([r["profit_factor"] for r in active_models if r["profit_factor"] != float('inf')]) if active_models else 0.0
    max_dd_all = max((r["max_dd_pct"] for r in all_results), default=0.0)

    log.info("Total models tested: %d", len(all_results))
    log.info("Models with trades: %d", len(active_models))
    log.info("Total trades: %d", total_all)
    log.info("Overall win rate: %.1f%%", (wins_all / total_all * 100) if total_all > 0 else 0)
    log.info("Average per-model win rate: %.1f%%", avg_wr * 100)
    log.info("Average per-model profit factor: %.2f", avg_pf)
    log.info("Combined P&L (%%): %.4f%%", total_pnl_all * 100)
    log.info("Max drawdown across all models: %.4f%%", max_dd_all * 100)

    # Top / Bottom performers
    if active_models:
        top = sorted(active_models, key=lambda x: x["total_pnl_pct"], reverse=True)[:5]
        bottom = sorted(active_models, key=lambda x: x["total_pnl_pct"])[:5]
        log.info("\nTop 5 performers:")
        for r in top:
            log.info("  %s %s %s | Trades: %d | PnL: %.4f%% | WR: %.1f%%",
                     r["pair"], r["tf"], r["tier"], r["trades"],
                     r["total_pnl_pct"] * 100, r["win_rate"] * 100)
        log.info("\nBottom 5 performers:")
        for r in bottom:
            log.info("  %s %s %s | Trades: %d | PnL: %.4f%% | WR: %.1f%%",
                     r["pair"], r["tf"], r["tier"], r["trades"],
                     r["total_pnl_pct"] * 100, r["win_rate"] * 100)

    # Save report
    report_path = RESULTS_DIR / f"next_set_backtest_{datetime.now():%Y%m%d_%H%M%S}.json"
    report = {
        "config": {
            "days": args.days,
            "threshold": args.threshold,
            "sl_atr": SL_ATR,
            "tp_atr": TP_ATR,
            "max_hold_bars": MAX_HOLD_BARS,
        },
        "summary": {
            "models_tested": len(all_results),
            "models_with_trades": len(active_models),
            "total_trades": total_all,
            "overall_win_rate": round(wins_all / total_all, 4) if total_all > 0 else 0,
            "avg_model_win_rate": round(avg_wr, 4),
            "avg_model_profit_factor": round(avg_pf, 2),
            "combined_pnl_pct": round(total_pnl_all, 6),
            "max_drawdown": round(max_dd_all, 6),
        },
        "results": all_results,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Report: %s", report_path)


if __name__ == "__main__":
    main()
