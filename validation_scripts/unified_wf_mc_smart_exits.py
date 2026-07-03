"""
Enhanced Walk-Forward & Monte Carlo Testing with Smart Exits
=============================================================

This extends unified_wf_mc_test.py with FULL smart exit simulation:
- Breakeven mover (SL to entry + offset after R threshold)
- ATR trailing stops (dynamic trail based on volatility)
- Partial take-profits (scale out at R levels)
- Time-based exits (max bars in trade)
- Squeeze protection (exit on low volatility compression)
- Rapid drawdown protection (partial close on adverse moves)

This provides the MOST ACCURATE simulation of live trading performance.

Usage:
    python unified_wf_mc_smart_exits.py

Version: 1.0 (Smart Exit Integration - 2026-03-31)
"""

import sys
import json
import logging
import random
import warnings
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("WF_MC_SMART")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Import from original WF/MC test
from CORE_MODULES.validation.unified_wf_mc_test import (
    LiveSimulator, WalkForwardResult, Trade,
    run_monte_carlo, calculate_wf_stats
)

# Import smart exit manager
from CORE_MODULES.core.risk.smart_exit_manager import SmartExitManager


@dataclass
class SmartExitTrade(Trade):
    """Extended trade record with smart exit details."""
    exit_decisions: List[Dict] = field(default_factory=list)
    peak_r: float = 0.0
    bars_in_trade: int = 0
    partial_closes: List[Tuple[float, float]] = field(default_factory=list)  # (pct_closed, price)


class SmartExitSimulator(LiveSimulator):
    """
    Live simulator with FULL smart exit management.
    Exactly mirrors how live trading manages positions.
    """
    
    def __init__(
        self,
        data_dir: str = "training_data/mt5_m30_comparison",
        pairs: List[str] = None,
        exit_config: Optional[Dict] = None,
    ):
        super().__init__(data_dir, pairs)
        
        # Initialize smart exit manager with config
        self.exit_manager = SmartExitManager()
        if exit_config:
            self.exit_manager.config = exit_config
            
        log.info("[SmartExitSimulator] Initialized with smart exit management")
        log.info(f"  Breakeven: {self.exit_manager.config.get('break_even', {})}")
        log.info(f"  Trailing: {self.exit_manager.config.get('trailing', {})}")
        log.info(f"  Partials: {self.exit_manager.config.get('partials', [])}")
        log.info(f"  Max bars: {self.exit_manager.config.get('max_bars_in_trade', 120)}")

    def simulate_trade_with_smart_exits(
        self,
        entry_idx: int,
        df: pd.DataFrame,
        direction: int,
        sl: float,
        tp: float,
        atr: float,
        regime: str,
        session: str,
        confidence: float,
        pair: str,
        ml_prob: float,
        smc_conf: float,
        gov_mult: float,
    ) -> Optional[SmartExitTrade]:
        """
        Simulate trade with FULL smart exit logic.
        
        Unlike basic simulate_trade, this:
        1. Registers position with SmartExitManager
        2. Evaluates breakeven, trailing, partials each bar
        3. Handles scale-outs and SL adjustments
        4. Tracks peak R and exit decisions
        """
        entry_price = df.iloc[entry_idx]["close"]
        entry_time = df.index[entry_idx]
        
        # Generate unique position ID
        pos_id = f"{pair}_{entry_time.timestamp()}"
        
        # Register with smart exit manager
        self.exit_manager.register_position(
            position_id=pos_id,
            entry_price=entry_price,
            direction=direction,
            sl_price=sl,
            tp_price=tp,
            atr=atr,
            entry_time=entry_time,
        )
        
        current_sl = sl
        current_tp = tp
        position_size = 1.0  # Start with full position
        partial_closes = []
        exit_decisions = []
        peak_r = 0.0
        
        # Simulate bar by bar with smart exit evaluation
        for i in range(entry_idx + 1, min(entry_idx + 200, len(df))):  # Max 200 bars
            bar = df.iloc[i]
            high = bar["high"]
            low = bar["low"]
            bar["close"]
            curr_time = df.index[i]
            
            # Current price (assume worst case for execution)
            current_price = low if direction == 1 else high
            
            # Calculate current ATR for trailing (use rolling)
            curr_atr = atr  # Base ATR
            if i >= 14:
                recent_df = df.iloc[max(0, i-14):i+1]
                tr = pd.concat([
                    recent_df["high"] - recent_df["low"],
                    abs(recent_df["high"] - recent_df["close"].shift(1)),
                    abs(recent_df["low"] - recent_df["close"].shift(1))
                ], axis=1).max(axis=1)
                curr_atr = tr.mean()
            
            # Evaluate smart exits
            decisions = self.exit_manager.evaluate_exits(
                position_id=pos_id,
                current_price=current_price,
                current_atr=curr_atr,
                bars_elapsed=1,
                signal_confidence=confidence,
                regime=regime,
            )
            
            # Process decisions
            for d in decisions:
                exit_decisions.append({
                    "bar": i,
                    "action": d.action,
                    "r_multiple": d.r_multiple,
                    "reason": d.reason,
                })
                
                if d.action == "MOVE_SL" and d.new_sl is not None:
                    # Move SL to breakeven or better
                    if direction == 1 and d.new_sl > current_sl:
                        current_sl = d.new_sl
                    elif direction == -1 and (current_sl == 0 or d.new_sl < current_sl):
                        current_sl = d.new_sl
                        
                elif d.action == "TRAIL_SL" and d.new_sl is not None:
                    # ATR trailing stop
                    if direction == 1 and d.new_sl > current_sl:
                        current_sl = d.new_sl
                    elif direction == -1 and (current_sl == 0 or d.new_sl < current_sl):
                        current_sl = d.new_sl
                        
                elif d.action == "PARTIAL_CLOSE":
                    # Scale out position
                    close_pct = d.close_pct
                    close_price = current_price
                    partial_closes.append((close_pct, close_price))
                    position_size -= close_pct
                    
                    # Move SL if specified
                    if d.new_sl is not None:
                        if direction == 1 and d.new_sl > current_sl:
                            current_sl = d.new_sl
                        elif direction == -1 and (current_sl == 0 or d.new_sl < current_sl):
                            current_sl = d.new_sl
                            
                elif d.action == "FULL_CLOSE":
                    # Time limit or other full close
                    exit_price = current_price
                    exit_time = curr_time
                    
                    pnl_r = self._calculate_pnl_r(
                        entry_price, exit_price, current_sl, direction, atr,
                        partial_closes, position_size
                    )
                    
                    self.exit_manager.unregister_position(pos_id)
                    
                    return SmartExitTrade(
                        entry_time=entry_time,
                        exit_time=exit_time,
                        pair=pair,
                        direction=direction,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        sl=sl,
                        tp=tp,
                        atr=atr,
                        pnl_r=pnl_r,
                        regime=regime,
                        session=session,
                        confidence=confidence,
                        ml_prob=ml_prob,
                        smc_conf=smc_conf,
                        gov_multiplier=gov_mult,
                        exit_decisions=exit_decisions,
                        peak_r=max(d.r_multiple for d in decisions) if decisions else 0,
                        bars_in_trade=i - entry_idx,
                        partial_closes=partial_closes,
                    )
            
            # Check if SL/TP hit (with current SL which may have been moved)
            exit_price = None
            
            if direction == 1:  # BUY
                if low <= current_sl:
                    exit_price = current_sl
                elif high >= current_tp:
                    exit_price = current_tp
            else:  # SELL
                if high >= current_sl:
                    exit_price = current_sl
                elif low <= current_tp:
                    exit_price = current_tp
            
            if exit_price is not None:
                exit_time = curr_time
                
                pnl_r = self._calculate_pnl_r(
                    entry_price, exit_price, sl, direction, atr,
                    partial_closes, position_size
                )
                
                self.exit_manager.unregister_position(pos_id)
                
                return SmartExitTrade(
                    entry_time=entry_time,
                    exit_time=exit_time,
                    pair=pair,
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    sl=sl,
                    tp=tp,
                    atr=atr,
                    pnl_r=pnl_r,
                    regime=regime,
                    session=session,
                    confidence=confidence,
                    ml_prob=ml_prob,
                    smc_conf=smc_conf,
                    gov_multiplier=gov_mult,
                    exit_decisions=exit_decisions,
                    peak_r=peak_r,
                    bars_in_trade=i - entry_idx,
                    partial_closes=partial_closes,
                )
            
            # Update peak R
            r_multiple = self.exit_manager.compute_r_multiple(
                entry_price, current_price, sl, direction, atr
            )
            peak_r = max(peak_r, r_multiple)
        
        # End of data - close at last price
        last_price = df.iloc[-1]["close"]
        exit_time = df.index[-1]
        
        pnl_r = self._calculate_pnl_r(
            entry_price, last_price, sl, direction, atr,
            partial_closes, position_size
        )
        
        self.exit_manager.unregister_position(pos_id)
        
        return SmartExitTrade(
            entry_time=entry_time,
            exit_time=exit_time,
            pair=pair,
            direction=direction,
            entry_price=entry_price,
            exit_price=last_price,
            sl=sl,
            tp=tp,
            atr=atr,
            pnl_r=pnl_r,
            regime=regime,
            session=session,
            confidence=confidence,
            ml_prob=ml_prob,
            smc_conf=smc_conf,
            gov_multiplier=gov_mult,
            exit_decisions=exit_decisions,
            peak_r=peak_r,
            bars_in_trade=len(df) - entry_idx,
            partial_closes=partial_closes,
        )
    
    def _calculate_pnl_r(
        self,
        entry_price: float,
        exit_price: float,
        original_sl: float,
        direction: int,
        atr: float,
        partial_closes: List[Tuple[float, float]],
        remaining_size: float,
    ) -> float:
        """Calculate PnL in R multiples accounting for partial closes."""
        risk_distance = abs(entry_price - original_sl)
        if risk_distance == 0:
            risk_distance = atr * 1.2
        
        total_pnl_r = 0.0
        
        # PnL from partial closes
        for close_pct, close_price in partial_closes:
            if direction == 1:
                pnl = (close_price - entry_price) / risk_distance * close_pct
            else:
                pnl = (entry_price - close_price) / risk_distance * close_pct
            total_pnl_r += pnl
        
        # PnL from remaining position
        if remaining_size > 0:
            if direction == 1:
                pnl = (exit_price - entry_price) / risk_distance * remaining_size
            else:
                pnl = (entry_price - exit_price) / risk_distance * remaining_size
            total_pnl_r += pnl
        
        return max(-2.0, min(3.0, total_pnl_r))  # Cap extremes

    def run_backtest(
        self,
        start_date: str,
        end_date: str,
        generate_realistic_signals: bool = True,
    ) -> List[SmartExitTrade]:
        """Run backtest with smart exit simulation."""
        all_trades = []
        
        self._daily_state = {}
        self._session_state = {}
        self._loss_streak = 0
        
        for pair in self.pairs:
            df = self.load_data(pair)
            if df is None:
                log.warning(f"No data for {pair}")
                continue
            
            df_period = df[(df.index >= start_date) & (df.index <= end_date)]
            if len(df_period) < 50:
                log.warning(f"Insufficient data for {pair}: {len(df_period)} rows")
                continue
            
            atr = self.calculate_atr(df_period)
            regime = self.detect_regime(df_period)
            
            bars_since_signal = 3  # COOLDOWN_BARS + 1
            
            for i in range(30, len(df_period) - 5):
                dt = df_period.index[i]
                
                if bars_since_signal < 3:  # COOLDOWN_BARS
                    bars_since_signal += 1
                    continue
                
                session = self.get_session(dt)
                curr_regime = regime.iloc[i] if i < len(regime) else "RANGING"
                curr_atr = atr.iloc[i] if i < len(atr) else atr.iloc[-1]
                
                if pd.isna(curr_atr) or curr_atr <= 0:
                    continue
                
                if generate_realistic_signals:
                    ml_prob = self._generate_realistic_ml_prob(df_period, i)
                    smc_conf = self._generate_realistic_smc_conf(df_period, i, curr_regime)
                else:
                    ml_prob = random.uniform(0.4, 0.8)
                    smc_conf = random.uniform(0.4, 0.8)
                
                fused, direction = self.calculate_signal(ml_prob, smc_conf)
                
                if direction == 0:
                    bars_since_signal += 1
                    continue
                
                confidence = fused
                
                allowed, reason, gov_mult = self.apply_governance(
                    pair=pair,
                    direction=direction,
                    regime=curr_regime,
                    confidence=confidence,
                    session=session,
                    smc_conf=smc_conf,
                    dt=dt,
                )
                
                if not allowed:
                    bars_since_signal += 1
                    continue
                
                entry_price = df_period.iloc[i]["close"]
                sl, tp = self.calculate_sl_tp(
                    entry_price=entry_price,
                    direction=direction,
                    atr=curr_atr,
                    regime=curr_regime,
                )
                
                # Use smart exit simulation instead of basic
                trade = self.simulate_trade_with_smart_exits(
                    entry_idx=i,
                    df=df_period,
                    direction=direction,
                    sl=sl,
                    tp=tp,
                    atr=curr_atr,
                    regime=curr_regime,
                    session=session,
                    confidence=confidence,
                    pair=pair,
                    ml_prob=ml_prob,
                    smc_conf=smc_conf,
                    gov_mult=gov_mult,
                )
                
                if trade:
                    all_trades.append(trade)
                    bars_since_signal = 0
                    
                    # Update loss streak tracking
                    if trade.pnl_r < 0:
                        self._loss_streak += 1
                    else:
                        self._loss_streak = 0
        
        return all_trades


def run_smart_exit_walk_forward(
    simulator: SmartExitSimulator,
    start_date: str,
    end_date: str,
    n_folds: int = 4,
) -> Tuple[List[WalkForwardResult], List[SmartExitTrade]]:
    """Run walk-forward with smart exits."""
    log.info("=" * 60)
    log.info("WALK-FORWARD ANALYSIS (WITH SMART EXITS)")
    log.info("=" * 60)
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    total_days = max(1, (end_dt - start_dt).days)
    
    train_days = max(1, int(total_days * 0.5))
    test_days = max(1, total_days - train_days)
    
    wf_results = []
    all_trades = []
    
    for fold in range(n_folds):
        offset = fold * max(1, test_days // n_folds)
        
        train_end = start_dt + timedelta(days=train_days)
        test_start = start_dt + timedelta(days=offset)
        test_end = min(test_start + timedelta(days=test_days), end_dt)
        
        if test_start >= end_dt:
            break
        
        log.info(f"\nFold {fold + 1}/{n_folds}:")
        log.info(f"  Train: {start_dt.date()} to {train_end.date()}")
        log.info(f"  Test:  {test_start.date()} to {test_end.date()}")
        
        trades = simulator.run_backtest(
            test_start.strftime("%Y-%m-%d"),
            test_end.strftime("%Y-%m-%d"),
            generate_realistic_signals=True,
        )
        log.info(f"  Found {len(trades)} trades in backtest")
        
        all_trades.extend(trades)
        
        result = calculate_wf_stats(fold + 1, start_dt, train_end, test_start, test_end, trades)
        wf_results.append(result)
        
        log.info(f"  Trades: {result.total_trades}, WR: {result.win_rate:.1%}, E: {result.expectancy:.3f}R")
        log.info(f"  Sharpe: {result.sharpe:.2f}, Sortino: {result.sortino:.2f}, MaxDD: {result.max_dd:.1%}")
    
    return wf_results, all_trades


def analyze_smart_exit_performance(trades: List[SmartExitTrade]) -> Dict:
    """Analyze smart exit effectiveness."""
    if not trades:
        return {}
    
    total_partials = sum(len(t.partial_closes) for t in trades)
    trades_with_partials = sum(1 for t in trades if t.partial_closes)
    
    breakeven_hits = sum(
        1 for t in trades 
        if any("breakeven" in d.get("reason", "").lower() for d in t.exit_decisions)
    )
    
    trailing_hits = sum(
        1 for t in trades
        if any("trail" in d.get("reason", "").lower() for d in t.exit_decisions)
    )
    
    time_limit_exits = sum(
        1 for t in trades
        if any("max bars" in d.get("reason", "").lower() for d in t.exit_decisions)
    )
    
    avg_bars = np.mean([t.bars_in_trade for t in trades])
    avg_peak_r = np.mean([t.peak_r for t in trades])
    
    # Compare to theoretical max (if exited at TP every time)
    theoretical_max = sum(1.5 for t in trades)  # Assumes 1.5R average TP
    actual_pnl = sum(t.pnl_r for t in trades)
    efficiency = actual_pnl / theoretical_max if theoretical_max > 0 else 0
    
    return {
        "total_trades": len(trades),
        "trades_with_partials": trades_with_partials,
        "total_partial_closes": total_partials,
        "breakeven_activations": breakeven_hits,
        "trailing_stop_activations": trailing_hits,
        "time_limit_exits": time_limit_exits,
        "avg_bars_in_trade": avg_bars,
        "avg_peak_r": avg_peak_r,
        "exit_efficiency": efficiency,
        "capture_ratio": actual_pnl / sum(t.peak_r for t in trades) if sum(t.peak_r for t in trades) > 0 else 0,
    }


def main():
    """Run complete smart exit WF + MC analysis."""
    log.info("\n" + "=" * 60)
    log.info("SMART EXIT WALK-FORWARD + MONTE CARLO ANALYSIS")
    log.info("LIVE TRADING SIMULATION WITH INTELLIGENT EXITS")
    log.info("=" * 60)
    
    # Smart exit config matching live system
    smart_exit_config = {
        "atr_length": 14,
        "r_multiple_atr": 1.2,
        "break_even": {"enabled": True, "trigger_r": 1.0, "offset_r": 0.2},
        "partials": [
            {"trigger_r": 1.0, "close_pct": 0.25, "move_sl_to": "breakeven", "min_confidence": 0.0, "max_confidence": 1.0},
            {"trigger_r": 1.5, "close_pct": 0.25, "move_sl_to": None, "min_confidence": 0.0, "max_confidence": 1.0},
        ],
        "trailing": {"enabled": True, "atr_mult": 0.8, "start_after_r": 1.0},
        "max_bars_in_trade": 120,
        "squeeze_protection": {"enabled": False},
        "rapid_drawdown_protection": {"enabled": False},
    }
    
    simulator = SmartExitSimulator(
        data_dir="training_data/mt5_m30_comparison",
        pairs=["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"],
        exit_config=smart_exit_config,
    )
    
    wf_results, trades = run_smart_exit_walk_forward(
        simulator,
        start_date="2026-03-10",
        end_date="2026-03-18",
        n_folds=4,
    )
    
    mc_stats = run_monte_carlo(trades, n_sims=1000)
    
    # Smart exit analysis
    smart_analysis = analyze_smart_exit_performance(trades)
    
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "smart_exit_config": smart_exit_config,
        "walk_forward": [
            {
                "fold": r.fold,
                "train_period": f"{r.train_start} to {r.train_end}",
                "test_period": f"{r.test_start} to {r.test_end}",
                "total_trades": r.total_trades,
                "winning_trades": r.winning_trades,
                "losing_trades": r.losing_trades,
                "win_rate": r.win_rate,
                "expectancy": r.expectancy,
                "sharpe": r.sharpe,
                "sortino": r.sortino,
                "max_dd": r.max_dd,
                "profit_factor": r.profit_factor,
            }
            for r in wf_results
        ],
        "monte_carlo": {
            "n_simulations": mc_stats.n_simulations,
            "final_equity_p5": mc_stats.final_equity_p5,
            "final_equity_p50": mc_stats.final_equity_p50,
            "final_equity_p95": mc_stats.final_equity_p95,
            "max_dd_p95": mc_stats.max_dd_p95,
            "survival_rate": mc_stats.survival_rate,
            "probability_of_ruin": mc_stats.probability_of_ruin,
        },
        "smart_exit_analysis": smart_analysis,
    }
    
    log.info("\n" + "=" * 60)
    log.info("SMART EXIT ANALYSIS SUMMARY")
    log.info("=" * 60)
    
    log.info("\nExit Performance:")
    log.info(f"  Trades with partial closes: {smart_analysis.get('trades_with_partials', 0)}")
    log.info(f"  Total partial closes: {smart_analysis.get('total_partial_closes', 0)}")
    log.info(f"  Breakeven activations: {smart_analysis.get('breakeven_activations', 0)}")
    log.info(f"  Trailing stop hits: {smart_analysis.get('trailing_stop_activations', 0)}")
    log.info(f"  Time limit exits: {smart_analysis.get('time_limit_exits', 0)}")
    log.info(f"  Avg bars in trade: {smart_analysis.get('avg_bars_in_trade', 0):.1f}")
    log.info(f"  Avg peak R: {smart_analysis.get('avg_peak_r', 0):.2f}")
    log.info(f"  Exit efficiency: {smart_analysis.get('exit_efficiency', 0):.1%}")
    
    log.info("\nMonte Carlo Results:")
    log.info(f"  Survival Rate: {mc_stats.survival_rate:.1%}")
    log.info(f"  Probability of Ruin: {mc_stats.probability_of_ruin:.1%}")
    log.info(f"  Final Equity (p50): {mc_stats.final_equity_p50:.3f}")
    log.info(f"  Max Drawdown (p95): {mc_stats.max_dd_p95:.1%}")
    
    output_path = Path(__file__).parent / "wf_mc_smart_exit_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    log.info(f"\nResults saved to: {output_path}")
    
    return output


if __name__ == "__main__":
    main()
