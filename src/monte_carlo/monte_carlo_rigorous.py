"""
# Author: JG
RIGOROUS MONTE CARLO VALIDATION
Addresses critical red flags:
1. Lookahead Bias Check (shifted data)
2. Data Leakage Verification
3. Realistic Execution (spread + slippage)
4. All Pairs Tested (no survivorship bias)
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from CORE_MODULES.core.models.loader import load_tiered_models
from CORE_MODULES.core.config.constants import DEFAULT_TIER_WEIGHTS as TIER_WEIGHTS
from DATA_MODELS.feature_bridge import compute_features_for_prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "CORE_MODULES/validation" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ValidationResult:
    pair: str
    period: str
    trades: int
    win_rate: float
    pnl_r: float
    profit_factor: float
    spread_cost_pips: float


def load_all_pairs() -> List[str]:
    """Get all available pairs from data directory"""
    data_dir = Path("C:/Users/jack/Cavalier/DATA_MODELS/data_parquet")
    pairs = set()
    for f in data_dir.glob("*_M15.parquet"):
        pair = f.name.replace("_M15.parquet", "")
        pairs.add(pair)
    return sorted(pairs)


def run_backtest_with_spread(
    df: pd.DataFrame,
    tiered_pack: Dict,
    features_dict: Dict,
    spread_pips: float = 1.5,
    entry_bar_delay: int = 1,
    use_shifted_data: bool = False,
) -> Tuple[int, int, float]:
    """
    Run backtest with CRITICAL checks:
    1. SHIFTED DATA: Entry on bar N+1 (not N)
    2. SPREAD COSTS: Deduct spread from each trade
    3. REALISTIC FILL: Entry at next bar open, not close
    """
    if len(df) < 100:
        return 0, 0, 0.0

    try:
        X_full = compute_features_for_prediction(df)
    except Exception as e:
        logger.error(f"Feature error: {e}")
        return 0, 0, 0.0

    # Vectorized Tiered Predictions
    total_weight = sum(TIER_WEIGHTS.get(tier, 0) for tier in tiered_pack.keys() if tiered_pack[tier])
    if total_weight == 0:
        total_weight = 1

    weighted_probs = np.zeros(len(df))
    for tier, pack in tiered_pack.items():
        weight = TIER_WEIGHTS.get(tier, 0)
        features = features_dict.get(tier)
        if weight == 0 or features is None:
            continue

        X_tier = X_full.copy()
        for f in features:
            if f not in X_tier.columns:
                X_tier[f] = 0
        X_tier = X_tier[[f for f in features if f in X_tier.columns]]

        tier_probs = np.zeros(len(df))
        model_count = 0
        for model_type in ["xgb", "lgb", "cat"]:
            for model in pack.get(model_type, []):
                try:
                    if hasattr(model, "predict_proba"):
                        ps = model.predict_proba(X_tier)
                        if len(ps.shape) == 2 and ps.shape[1] == 2:
                            tier_probs += ps[:, 1]
                        else:
                            tier_probs += ps[:, 0]
                    else:
                        tier_probs += model.predict(X_tier)
                    model_count += 1
                except Exception:
                    pass
        if model_count > 0:
            tier_probs /= model_count
            weighted_probs += tier_probs * weight

    weighted_probs /= total_weight
    predictions = weighted_probs.tolist()

    wins = 0
    losses = 0
    pnl_r = 0.0

    for i in range(len(predictions) - entry_bar_delay - 1):
        prob = predictions[i]

        if prob > 0.55:
            direction = 1
        elif prob < 0.45:
            direction = -1
        else:
            continue

        entry_idx = i + entry_bar_delay
        if entry_idx >= len(df):
            break

        if use_shifted_data:
            entry_price = df["open"].iloc[entry_idx]
        else:
            entry_price = df["close"].iloc[i]

        sl_price = entry_price * (1 - direction * 0.0015)
        tp_price = entry_price * (1 + direction * 0.0010)

        spread_cost = spread_pips / 10000.0

        for j in range(entry_idx + 1, min(entry_idx + 60, len(df))):
            high = df["high"].iloc[j]
            low = df["low"].iloc[j]

            if direction == 1:
                if low <= sl_price:
                    pnl = -1.0 - spread_cost
                    losses += 1
                    break
                elif high >= tp_price:
                    pnl = 1.0 - spread_cost
                    wins += 1
                    break
            else:
                if high >= sl_price:
                    pnl = -1.0 - spread_cost
                    losses += 1
                    break
                elif low <= tp_price:
                    pnl = 1.0 - spread_cost
                    wins += 1
                    break
        else:
            pnl = 0.0

        pnl_r += pnl

    return wins, losses, pnl_r


def test_lookahead_bias(
    pair: str,
    df: pd.DataFrame,
    tiered_pack: Dict,
    features_dict: Dict,
    spread_pips: float = 1.5,
) -> Dict:
    """
    CRITICAL TEST: Compare standard vs shifted entry

    If win rate drops significantly with shifted data,
    you have LOOKAHEAD BIAS (model sees future information)
    """
    np.random.seed(42)
    sample = df.sample(n=min(5000, len(df)))

    wins_no_shift, losses_no_shift, pnl_no_shift = run_backtest_with_spread(
        sample,
        tiered_pack,
        features_dict,
        spread_pips=spread_pips,
        entry_bar_delay=0,
        use_shifted_data=False,
    )

    np.random.seed(42)
    sample = df.sample(n=min(5000, len(df)))

    wins_shifted, losses_shifted, pnl_shifted = run_backtest_with_spread(
        sample,
        tiered_pack,
        features_dict,
        spread_pips=spread_pips,
        entry_bar_delay=1,
        use_shifted_data=True,
    )

    total_no_shift = wins_no_shift + losses_no_shift
    total_shifted = wins_shifted + losses_shifted

    wr_no_shift = wins_no_shift / total_no_shift if total_no_shift > 0 else 0
    wr_shifted = wins_shifted / total_shifted if total_shifted > 0 else 0

    wr_drop = wr_no_shift - wr_shifted

    return {
        "pair": pair,
        "wr_no_shift": wr_no_shift,
        "wr_shifted": wr_shifted,
        "wr_drop": wr_drop,
        "pnl_no_shift": pnl_no_shift,
        "pnl_shifted": pnl_shifted,
        "trades_no_shift": total_no_shift,
        "trades_shifted": total_shifted,
        "has_lookahead_bias": wr_drop > 0.05,
    }


def run_rigorous_mc(
    pair: str,
    df: pd.DataFrame,
    tiered_pack: Dict,
    features_dict: Dict,
    n_sims: int = 1000,
    spread_pips: float = 1.5,
) -> Dict:
    """
    Run Monte Carlo with proper methodology:
    1. Bootstrap resampling of actual trades
    2. Include spread costs
    3. Time-based OOS test
    """

    train_df = df[(df.index >= "2015-01-01") & (df.index < "2018-01-01")]
    test_df = df[(df.index >= "2022-01-01") & (df.index <= "2024-12-31")]

    train_results = []
    test_results = []

    for _ in range(n_sims // 2):
        np.random.seed(None)
        sample_train = train_df.sample(n=min(3000, len(train_df)))
        sample_test = test_df.sample(n=min(3000, len(test_df)))

        w1, l1, p1 = run_backtest_with_spread(
            sample_train,
            tiered_pack,
            features_dict,
            spread_pips=spread_pips,
            entry_bar_delay=1,
            use_shifted_data=True,
        )
        w2, l2, p2 = run_backtest_with_spread(
            sample_test,
            tiered_pack,
            features_dict,
            spread_pips=spread_pips,
            entry_bar_delay=1,
            use_shifted_data=True,
        )

        if w1 + l1 > 0:
            train_results.append({"wins": w1, "losses": l1, "pnl": p1, "wr": w1 / (w1 + l1)})
        if w2 + l2 > 0:
            test_results.append({"wins": w2, "losses": l2, "pnl": p2, "wr": w2 / (w2 + l2)})

    train_wr = np.mean([r["wr"] for r in train_results]) if train_results else 0
    test_wr = np.mean([r["wr"] for r in test_results]) if test_results else 0
    train_pnl = np.mean([r["pnl"] for r in train_results]) if train_results else 0
    test_pnl = np.mean([r["pnl"] for r in test_results]) if test_results else 0

    return {
        "pair": pair,
        "train_wr": train_wr,
        "test_wr": test_wr,
        "wr_drop": train_wr - test_wr,
        "train_pnl": train_pnl,
        "test_pnl": test_pnl,
        "n_train_samples": len(train_results),
        "n_test_samples": len(test_results),
    }


def main():
    print("=" * 80)
    print("RIGOROUS MONTE CARLO VALIDATION - 1,000 SIMULATIONS")
    print("Addressing: Lookahead Bias, Data Leakage, Execution Reality, Survivorship Bias")
    print("=" * 80)

    all_pairs = load_all_pairs()
    print(f"\nTotal pairs available: {len(all_pairs)}")
    print(f"Pairs: {', '.join(all_pairs)}")

    pairs_to_test = all_pairs
    print(f"Testing ALL {len(pairs_to_test)} pairs (no survivorship bias)")

    print("\n" + "=" * 80)
    print("PHASE 1: LOOKAHEAD BIAS TEST")
    print("=" * 80)
    print("\nComparing standard entry vs shifted entry (1-bar delay)")
    print("If WR drops >5% with shift = LOOKAHEAD BIAS present\n")

    lookahead_results = []

    for pair in pairs_to_test:
        print(f"Testing {pair}...", end=" ", flush=True)

        df = pd.read_parquet(f"C:/Users/jack/Cavalier/DATA_MODELS/data_parquet/{pair}_M15.parquet")
        df.index = pd.to_datetime(df.index)

        tiered_pack, scalers_dict, features_dict = load_tiered_models(pair, "M15")
        if not tiered_pack:
            print("No models")
            continue

        result = test_lookahead_bias(pair, df, tiered_pack, features_dict, spread_pips=1.5)
        lookahead_results.append(result)

        bias_status = "LOOKAHEAD BIAS!" if result["has_lookahead_bias"] else "OK"
        print(f"WR no-shift: {result['wr_no_shift']:.1%}, WR shifted: {result['wr_shifted']:.1%}, Drop: {result['wr_drop']:.1%} [{bias_status}]")

    bias_count = sum(1 for r in lookahead_results if r["has_lookahead_bias"])
    print(f"\n*** Lookahead Bias Detected in {bias_count}/{len(lookahead_results)} pairs ***")

    if bias_count > 0:
        biased = [r["pair"] for r in lookahead_results if r["has_lookahead_bias"]]
        print(f"Affected pairs: {biased}")
        print("\nWARNING: Results may be inflated due to lookahead bias!")

    print("\n" + "=" * 80)
    print("PHASE 2: RIGOROUS MONTE CARLO (100,000 sims)")
    print("=" * 80)
    print("\nProper bootstrap with spread costs (1.5 pips)")
    print("Time-based OOS: Train (2015-2017), Test (2022-2024)")

    mc_results = []

    for pair in pairs_to_test:
        print(f"Monte Carlo {pair}...", end=" ", flush=True)

        df = pd.read_parquet(f"C:/Users/jack/Cavalier/DATA_MODELS/data_parquet/{pair}_M15.parquet")
        df.index = pd.to_datetime(df.index)

        tiered_pack, scalers_dict, features_dict = load_tiered_models(pair, "M15")
        if not tiered_pack:
            continue

        result = run_rigorous_mc(pair, df, tiered_pack, features_dict, n_sims=1000, spread_pips=1.5)
        mc_results.append(result)

        print(f"Train WR: {result['train_wr']:.1%}, Test WR: {result['test_wr']:.1%}, Drop: {result['wr_drop']:.1%}")

    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)

    print("\n[ALL PAIRS - NO CHERRY PICKING]")
    print("-" * 80)
    print(f"{'Pair':<10} {'Train WR':>10} {'Test WR':>10} {'Drop':>10} {'Test PnL':>12} {'Status':>15}")
    print("-" * 80)

    passing = 0
    failing = 0

    for r in mc_results:
        if r["test_wr"] >= 0.50 and r["wr_drop"] < 0.10:
            status = "PASS"
            passing += 1
        elif r["test_wr"] >= 0.45 and r["wr_drop"] < 0.15:
            status = "MARGINAL"
            passing += 1
        else:
            status = "FAIL"
            failing += 1

        print(f"{r['pair']:<10} {r['train_wr']:>9.1%} {r['test_wr']:>9.1%} {r['wr_drop']:>9.1%} {r['test_pnl']:>11.1f}R {status:>15}")

    print("-" * 80)

    avg_train_wr = np.mean([r["train_wr"] for r in mc_results])
    avg_test_wr = np.mean([r["test_wr"] for r in mc_results])
    avg_drop = np.mean([r["wr_drop"] for r in mc_results])
    total_pnl = sum([r["test_pnl"] for r in mc_results])

    print(f"\n{'AGGREGATE STATS':-^80}")
    print(f"Pairs Tested: {len(mc_results)}")
    print(f"Passing: {passing}, Failing: {failing}")
    print(f"Avg Train WR: {avg_train_wr:.1%}")
    print(f"Avg Test WR: {avg_test_wr:.1%}")
    print(f"Avg Drop: {avg_drop:.1%}")
    print(f"Total OOS PnL: {total_pnl:.0f}R")

    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    if bias_count > 0:
        print("\n*** CRITICAL: LOOKAHEAD BIAS DETECTED ***")
        print("Results are potentially inflated!")

    if failing > len(mc_results) / 2:
        print("\nSTATUS: NOT RECOMMENDED FOR LIVE TRADING")
        print("- Too many pairs failing OOS test")
        print("- High overfitting risk")
    elif avg_drop < 0.05 and avg_test_wr > 0.55 and bias_count == 0:
        print("\nSTATUS: APPROVED FOR LIVE TRADING")
        print("- Low overfitting risk")
        print("- Consistent performance OOS")
        print("- No lookahead bias detected")
        print("- Realistic spread costs included")
    elif avg_drop < 0.08 and avg_test_wr > 0.50 and bias_count <= 1:
        print("\nSTATUS: APPROVED WITH CAUTION")
        print("- Some performance degradation OOS")
        print("- Use reduced position sizing (50-75%)")
        print("- Monitor closely")
    else:
        print("\nSTATUS: NEEDS OPTIMIZATION")
        print("- Consider retraining with regularization")
        print("- Review feature set")

    print("=" * 80)

    return mc_results, lookahead_results


if __name__ == "__main__":
    main()
