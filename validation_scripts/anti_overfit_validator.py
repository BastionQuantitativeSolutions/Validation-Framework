"""
Anti-Overfit Validation Framework
=================================
Ensures all parameters work on unseen data.

Tests:
1. Walk-forward analysis (multiple OOS periods)
2. Parameter stability (small changes shouldn't break performance)
3. Regime robustness (must work in all market conditions)
4. Monte Carlo permutation test

Usage: python anti_overfit_validator.py
"""

import sys
import logging
import random
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("ANTI_OVERFIT")


# ═════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class WalkForwardResult:
    """Result from one walk-forward fold."""
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_trades: int
    win_rate: float
    profit_factor: float
    expectancy: float
    max_drawdown: float
    sharpe: float


class WalkForwardValidator:
    """
    Validates strategy on multiple out-of-sample periods.
    No parameter optimization on test data allowed.
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        train_size: int = 2000,  # bars
        test_size: int = 500,    # bars
        embargo: int = 50,       # bars between train/test
    ):
        self.n_folds = n_folds
        self.train_size = train_size
        self.test_size = test_size
        self.embargo = embargo
        
    def generate_folds(self, n_bars: int) -> List[Tuple[int, int, int, int]]:
        """Generate walk-forward splits."""
        folds = []
        
        # Start from the end and work backwards
        test_end = n_bars
        
        for fold in range(self.n_folds):
            test_start = test_end - self.test_size
            train_end = test_start - self.embargo
            train_start = max(0, train_end - self.train_size)
            
            if train_start < 0 or test_start < 0:
                break
            
            folds.append((train_start, train_end, test_start, test_end))
            test_end = test_start  # Next fold starts here
        
        return list(reversed(folds))  # Chronological order
    
    def validate_fold(
        self,
        fold: int,
        train_start: int,
        train_end: int,
        test_start: int,
        test_end: int,
        df: pd.DataFrame,
        predictions: np.ndarray,
        config: Dict,
    ) -> Optional[WalkForwardResult]:
        """Validate one fold."""
        
        # Training period - only for analysis, NOT for parameter optimization
        train_df = df.iloc[train_start:train_end]
        train_preds = predictions[train_start:train_end]
        
        # Test period - true OOS
        test_df = df.iloc[test_start:test_end]
        test_preds = predictions[test_start:test_end]
        
        # Calculate train metrics (informational only)
        train_metrics = self._calculate_metrics(train_df, train_preds)
        
        # Calculate test metrics (what matters)
        test_metrics = self._calculate_metrics(test_df, test_preds)
        
        log.info(f"  Fold {fold}: Train WR={train_metrics['wr']:.1%}, "
                 f"Test WR={test_metrics['wr']:.1%}, "
                 f"Trades={test_metrics['n_trades']}")
        
        return WalkForwardResult(
            fold=fold,
            train_start=str(df.index[train_start]),
            train_end=str(df.index[train_end]),
            test_start=str(df.index[test_start]),
            test_end=str(df.index[test_end]),
            n_trades=test_metrics['n_trades'],
            win_rate=test_metrics['wr'],
            profit_factor=test_metrics['pf'],
            expectancy=test_metrics['ev'],
            max_drawdown=test_metrics['dd'],
            sharpe=test_metrics['sharpe'],
        )
    
    def _calculate_metrics(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        threshold: float = 0.65,
    ) -> Dict:
        """Calculate simple metrics for validation."""
        signals = np.where(predictions > threshold, 1,
                  np.where(predictions < (1 - threshold), -1, 0))
        
        trades = []
        for i in range(len(df) - 1):
            if signals[i] != 0:
                entry = df['close'].iloc[i]
                exit_p = df['close'].iloc[i + 1]
                pnl = (exit_p - entry) / entry * signals[i]
                trades.append(pnl)
        
        if not trades:
            return {'n_trades': 0, 'wr': 0, 'pf': 0, 'ev': 0, 'dd': 0, 'sharpe': 0}
        
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]
        
        wr = len(wins) / len(trades)
        pf = sum(wins) / abs(sum(losses)) if losses else float('inf')
        ev = np.mean(trades)
        
        # Simple drawdown
        cumsum = np.cumsum(trades)
        peak = np.maximum.accumulate(cumsum)
        dd = peak - cumsum
        max_dd = max(dd) if len(dd) > 0 else 0
        
        sharpe = np.mean(trades) / np.std(trades) * np.sqrt(252) if np.std(trades) > 0 else 0
        
        return {
            'n_trades': len(trades),
            'wr': wr,
            'pf': pf,
            'ev': ev,
            'dd': max_dd,
            'sharpe': sharpe,
        }


# ═════════════════════════════════════════════════════════════════════════════
# PARAMETER STABILITY TEST
# ═════════════════════════════════════════════════════════════════════════════

class ParameterStabilityTest:
    """
    Tests that small parameter changes don't cause large performance swings.
    If they do, the system is overfit.
    """
    
    def __init__(self, n_perturbations: int = 20):
        self.n_perturbations = n_perturbations
    
    def test_stability(
        self,
        base_config: Dict,
        df: pd.DataFrame,
        predictions: np.ndarray,
    ) -> Dict:
        """Test parameter stability."""
        
        # Base performance
        base_perf = self._evaluate(base_config, df, predictions)
        
        results = []
        
        for i in range(self.n_perturbations):
            # Perturb parameters
            perturbed = self._perturb_config(base_config)
            perf = self._evaluate(perturbed, df, predictions)
            results.append(perf)
        
        # Calculate stability metrics
        win_rates = [r['wr'] for r in results]
        pfs = [r['pf'] for r in results]
        
        stability = {
            'base_wr': base_perf['wr'],
            'wr_std': np.std(win_rates),
            'wr_min': np.min(win_rates),
            'wr_max': np.max(win_rates),
            'base_pf': base_perf['pf'],
            'pf_std': np.std(pfs),
            'is_stable': np.std(win_rates) < 0.15,  # WR shouldn't vary > 15%
        }
        
        return stability
    
    def _perturb_config(self, config: Dict, pct: float = 0.1) -> Dict:
        """Randomly perturb config parameters."""
        import copy
        perturbed = copy.deepcopy(config)
        
        # Perturb risk parameters
        if 'risk' in perturbed:
            for key in ['base_risk_per_trade', 'kelly_fraction']:
                if key in perturbed['risk']:
                    perturbed['risk'][key] *= (1 + random.uniform(-pct, pct))
        
        # Perturb thresholds
        if 'signal_quality' in perturbed:
            for key in ['min_confidence', 'min_edge_for_trade']:
                if key in perturbed['signal_quality']:
                    perturbed['signal_quality'][key] *= (1 + random.uniform(-pct, pct))
        
        return perturbed
    
    def _evaluate(self, config: Dict, df: pd.DataFrame, predictions: np.ndarray) -> Dict:
        """Simple evaluation."""
        threshold = config.get('signal_quality', {}).get('min_confidence', 0.65)
        
        signals = np.where(predictions > threshold, 1,
                  np.where(predictions < (1 - threshold), -1, 0))
        
        trades = []
        for i in range(len(df) - 1):
            if signals[i] != 0:
                entry = df['close'].iloc[i]
                exit_p = df['close'].iloc[i + 1]
                pnl = (exit_p - entry) / entry * signals[i]
                trades.append(pnl)
        
        if not trades:
            return {'wr': 0, 'pf': 0}
        
        wins = [t for t in trades if t > 0]
        wr = len(wins) / len(trades)
        pf = sum(wins) / abs(sum([t for t in trades if t <= 0])) if any(t <= 0 for t in trades) else float('inf')
        
        return {'wr': wr, 'pf': pf}


# ═════════════════════════════════════════════════════════════════════════════
# REGIME ROBUSTNESS TEST
# ═════════════════════════════════════════════════════════════════════════════

class RegimeRobustnessTest:
    """
    Tests performance across different market regimes.
    Strategy must work in all conditions (or gracefully degrade).
    """
    
    def test_regime_robustness(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        config: Dict,
    ) -> Dict:
        """Test performance by regime."""
        
        # Detect regimes
        from CORE_MODULES.training.high_performance_optimizer import detect_regime_quality
        regimes, quality, trend = detect_regime_quality(df)
        
        regime_performance = defaultdict(list)
        
        threshold = config.get('signal_quality', {}).get('min_confidence', 0.65)
        signals = np.where(predictions > threshold, 1,
                  np.where(predictions < (1 - threshold), -1, 0))
        
        for i in range(len(df) - 1):
            if signals[i] != 0:
                entry = df['close'].iloc[i]
                exit_p = df['close'].iloc[i + 1]
                pnl = (exit_p - entry) / entry * signals[i]
                
                regime = regimes[i]
                regime_performance[regime].append(pnl)
        
        results = {}
        for regime, trades in regime_performance.items():
            if len(trades) < 10:
                continue
            
            wins = [t for t in trades if t > 0]
            wr = len(wins) / len(trades)
            pf = sum(wins) / abs(sum([t for t in trades if t <= 0])) if any(t <= 0 for t in trades) else 0
            
            results[regime] = {
                'n_trades': len(trades),
                'win_rate': wr,
                'profit_factor': pf,
                'expectancy': np.mean(trades),
            }
        
        # Check for robustness
        wrs = [r['win_rate'] for r in results.values()]
        is_robust = np.std(wrs) < 0.2 if wrs else False  # WR shouldn't vary too much
        
        return {
            'by_regime': results,
            'wr_std_across_regimes': np.std(wrs) if wrs else 0,
            'is_robust': is_robust,
        }


# ═════════════════════════════════════════════════════════════════════════════
# MONTE CARLO PERMUTATION TEST
# ═════════════════════════════════════════════════════════════════════════════

class MonteCarloPermutationTest:
    """
    Tests that performance is better than random chance.
    """
    
    def __init__(self, n_permutations: int = 1000):
        self.n_permutations = n_permutations
    
    def test_significance(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        config: Dict,
    ) -> Dict:
        """Test if performance is statistically significant."""
        
        # Actual performance
        actual_perf = self._evaluate(predictions, df, config)
        
        # Random permutations
        random_perfs = []
        for _ in range(self.n_permutations):
            shuffled = np.random.permutation(predictions)
            perf = self._evaluate(shuffled, df, config)
            random_perfs.append(perf['total_return'])
        
        # Calculate p-value
        better_than_random = sum(1 for r in random_perfs if r >= actual_perf['total_return'])
        p_value = better_than_random / self.n_permutations
        
        return {
            'actual_return': actual_perf['total_return'],
            'random_median': np.median(random_perfs),
            'random_95th': np.percentile(random_perfs, 95),
            'p_value': p_value,
            'is_significant': p_value < 0.05,  # 95% confidence
        }
    
    def _evaluate(self, predictions: np.ndarray, df: pd.DataFrame, config: Dict) -> Dict:
        """Evaluate strategy."""
        threshold = config.get('signal_quality', {}).get('min_confidence', 0.65)
        
        signals = np.where(predictions > threshold, 1,
                  np.where(predictions < (1 - threshold), -1, 0))
        
        returns = []
        for i in range(len(df) - 1):
            if signals[i] != 0:
                entry = df['close'].iloc[i]
                exit_p = df['close'].iloc[i + 1]
                pnl = (exit_p - entry) / entry * signals[i]
                returns.append(pnl)
        
        return {'total_return': sum(returns)}


# ═════════════════════════════════════════════════════════════════════════════
# MAIN VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def run_full_validation(
    df: pd.DataFrame,
    predictions: np.ndarray,
    config: Dict,
) -> Dict:
    """Run complete anti-overfit validation suite."""
    
    log.info("="*60)
    log.info("ANTI-OVERFIT VALIDATION SUITE")
    log.info("="*60)
    
    results = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'n_bars': len(df),
    }
    
    # 1. Walk-forward validation
    log.info("\n[1] Walk-Forward Validation")
    wf = WalkForwardValidator(n_folds=5)
    folds = wf.generate_folds(len(df))
    log.info(f"  Generated {len(folds)} folds")
    
    wf_results = []
    for fold_idx, (train_s, train_e, test_s, test_e) in enumerate(folds):
        result = wf.validate_fold(fold_idx, train_s, train_e, test_s, test_e, df, predictions, config)
        if result:
            wf_results.append(result)
    
    # Check consistency across folds
    if wf_results:
        wrs = [r.win_rate for r in wf_results]
        consistency = {
            'fold_win_rates': wrs,
            'wr_mean': np.mean(wrs),
            'wr_std': np.std(wrs),
            'is_consistent': np.std(wrs) < 0.15,  # Less than 15% variation
        }
        results['walk_forward'] = consistency
        log.info(f"  Consistency: WR std = {consistency['wr_std']:.3f}")
    
    # 2. Parameter stability
    log.info("\n[2] Parameter Stability Test")
    stability = ParameterStabilityTest(n_perturbations=20)
    stab_results = stability.test_stability(config, df, predictions)
    results['parameter_stability'] = stab_results
    log.info(f"  Stable: {stab_results['is_stable']} (WR std = {stab_results['wr_std']:.3f})")
    
    # 3. Regime robustness
    log.info("\n[3] Regime Robustness Test")
    regime = RegimeRobustnessTest()
    regime_results = regime.test_regime_robustness(df, predictions, config)
    results['regime_robustness'] = regime_results
    log.info(f"  Robust: {regime_results['is_robust']} (WR std = {regime_results['wr_std_across_regimes']:.3f})")
    for r, data in regime_results['by_regime'].items():
        log.info(f"    {r}: WR={data['win_rate']:.1%}, PF={data['profit_factor']:.2f}")
    
    # 4. Statistical significance
    log.info("\n[4] Monte Carlo Permutation Test")
    mc = MonteCarloPermutationTest(n_permutations=1000)
    mc_results = mc.test_significance(df, predictions, config)
    results['statistical_significance'] = mc_results
    log.info(f"  Significant: {mc_results['is_significant']} (p-value = {mc_results['p_value']:.3f})")
    
    # Final verdict
    all_passed = (
        consistency.get('is_consistent', False) and
        stab_results['is_stable'] and
        regime_results['is_robust'] and
        mc_results['is_significant']
    )
    
    results['verdict'] = {
        'passed_all': all_passed,
        'can_trade_live': all_passed,
        'warnings': [],
    }
    
    log.info("\n" + "="*60)
    log.info(f"FINAL VERDICT: {'PASS' if all_passed else 'FAIL'}")
    log.info("="*60)
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Anti-overfit validation framework loaded.")
    print("Use run_full_validation(df, predictions, config) to validate.")
