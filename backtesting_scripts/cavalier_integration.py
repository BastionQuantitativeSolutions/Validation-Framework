"""
Cavalier Backtest Signal Integration

Integrates the backtest replay engine with Cavalier's existing
signal generation and governance pipeline for accurate simulation.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import pandas as pd

try:
    from core.backtesting.replay_engine import ReplayEngine, BacktestOrderExecutor, ExecutionMode, BacktestTrade
    from core.backtesting.data_fetcher import TickVaultFetcher
except ImportError:
    from .replay_engine import ReplayEngine, BacktestOrderExecutor, ExecutionMode, BacktestTrade
    from .data_fetcher import TickVaultFetcher

logger = logging.getLogger(__name__)


class CavalierSignalGenerator:
    """
    Wraps Cavalier's signal generation for backtesting.

    Uses the same signal pipeline as live trading but
    processes historical bars instead of live data.
    """

    def __init__(self, regime_detector=None, volatility_detector=None, smc_analyzer=None, model_predictor=None):
        self.regime_detector = regime_detector
        self.volatility_detector = volatility_detector
        self.smc_analyzer = smc_analyzer
        self.model_predictor = model_predictor

        self.bar_cache: Dict[str, pd.DataFrame] = {}

    def set_bar_data(self, symbol: str, bars: pd.DataFrame):
        """Set historical bar data for a symbol."""
        self.bar_cache[symbol] = bars.copy()

    def generate_signal(self, symbol: str, current_bar: pd.Series, lookback_bars: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Generate trading signal for current bar.

        Returns signal dict with:
        - direction: 1 (buy), -1 (sell), 0 (no signal)
        - confidence: 0.0-1.0
        - entry_price: float
        - sl: float
        - tp: float
        - strategy: str
        """
        try:
            regime = self._detect_regime(symbol, lookback_bars)
            volatility = self._detect_volatility(symbol, lookback_bars)

            self._analyze_smc(symbol, lookback_bars)

            ml_prediction = self._get_ml_prediction(symbol, lookback_bars)

            direction = 0
            confidence = 0.5

            if ml_prediction:
                direction = ml_prediction.get("direction", 0)
                confidence = min(ml_prediction.get("confidence", 0.5) * 1.2, 0.99)

                if regime and regime in ["RANGING", "VOLATILE"]:
                    confidence *= 0.85

            if direction == 0:
                return None

            mid = current_bar["close"]
            spread = 0.00015

            if direction > 0:
                entry = mid
                sl = entry - 0.0030
                tp = entry + 0.0090
            else:
                entry = mid
                sl = entry + 0.0030
                tp = entry - 0.0090

            return {
                "symbol": symbol,
                "direction": direction,
                "confidence": confidence,
                "entry_price": entry,
                "sl": sl,
                "tp": tp,
                "spread": spread,
                "regime": regime,
                "volatility": volatility,
                "timeframe": "M5",
                "strategy": "SMC_ML_FUSION",
            }

        except Exception as e:
            logger.debug(f"[backtest-signal] Error generating signal: {e}")
            return None

    def _detect_regime(self, symbol: str, bars: pd.DataFrame) -> Optional[str]:
        """Detect market regime."""
        if self.regime_detector is None:
            return self._simple_regime_detection(bars)

        try:
            result = self.regime_detector.detect(bars)
            return result.get("regime") if result else None
        except Exception:
            return self._simple_regime_detection(bars)

    def _simple_regime_detection(self, bars: pd.DataFrame) -> str:
        """Simple regime detection fallback."""
        if len(bars) < 50:
            return "UNKNOWN"

        close = bars["close"]
        ema20 = close.ewm(span=20).mean()
        ema50 = close.ewm(span=50).mean()

        adx = self._calculate_adx(bars)

        if adx > 25:
            if ema20.iloc[-1] > ema50.iloc[-1]:
                return "TRENDING"
            else:
                return "STRONG_DOWNTREND"
        elif adx < 20:
            return "RANGING"
        else:
            return "VOLATILE"

    def _calculate_adx(self, bars: pd.DataFrame, period: int = 14) -> float:
        """Calculate ADX indicator."""
        if len(bars) < period + 1:
            return 20.0

        high = bars["high"]
        low = bars["low"]
        close = bars["close"]

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()

        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx.iloc[-1] if not adx.empty else 20.0

    def _detect_volatility(self, symbol: str, bars: pd.DataFrame) -> str:
        """Detect volatility regime."""
        if len(bars) < 20:
            return "MEDIUM"

        returns = bars["close"].pct_change()
        vol = returns.std()

        if vol < 0.01:
            return "LOW"
        elif vol > 0.02:
            return "HIGH"
        else:
            return "MEDIUM"

    def _analyze_smc(self, symbol: str, bars: pd.DataFrame) -> List[Dict]:
        """Analyze SMC structures."""
        if len(bars) < 50:
            return []

        zones = []
        high = bars["high"].iloc[-20:]
        low = bars["low"].iloc[-20:]

        recent_high = high.max()
        recent_low = low.min()

        zones.append({"type": "resistance", "price": recent_high, "strength": 0.7})
        zones.append({"type": "support", "price": recent_low, "strength": 0.7})

        return zones

    def _get_ml_prediction(self, symbol: str, bars: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Get ML model prediction using momentum-based signal."""
        if self.model_predictor is None:
            return self._momentum_prediction(bars)

        try:
            features = self._extract_features(bars)
            return self.model_predictor.predict(features)
        except Exception:
            return self._momentum_prediction(bars)

    def _momentum_prediction(self, bars: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Momentum-based prediction using price action."""
        if bars.empty or len(bars) < 20:
            return {"direction": 0, "confidence": 0.45}

        close = bars["close"]
        if len(close) < 20:
            return {"direction": 0, "confidence": 0.45}

        ema8 = close.ewm(span=8).mean().iloc[-1]
        ema21 = close.ewm(span=21).mean().iloc[-1]
        current = close.iloc[-1]

        momentum = (current - close.iloc[-5]) / close.iloc[-5] if len(close) >= 5 else 0
        ema_diff_pct = (ema8 - ema21) / ema21 if ema21 != 0 else 0

        direction = 0
        confidence = 0.45

        if ema8 > ema21:
            direction = 1
            if momentum > 0 and ema_diff_pct > 0.0005:
                confidence = 0.60
            elif momentum > 0:
                confidence = 0.55
            elif momentum < 0:
                confidence = 0.50
            else:
                confidence = 0.52
        elif ema8 < ema21:
            direction = -1
            if momentum < 0 and ema_diff_pct < -0.0005:
                confidence = 0.60
            elif momentum < 0:
                confidence = 0.55
            elif momentum > 0:
                confidence = 0.50
            else:
                confidence = 0.52

        return {"direction": direction, "confidence": min(confidence, 0.70)}

    def _extract_features(self, bars: pd.DataFrame) -> Dict[str, float]:
        """Extract features for ML model."""
        if len(bars) < 20:
            return {}

        close = bars["close"]

        return {
            "returns_mean": close.pct_change().mean(),
            "returns_std": close.pct_change().std(),
            "momentum_5": (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] if len(close) >= 6 else 0,
            "momentum_20": (close.iloc[-1] - close.iloc[-21]) / close.iloc[-21] if len(close) >= 21 else 0,
        }


class CavalierBacktester:
    """
    Full backtester that integrates Cavalier's signal and governance pipeline.
    """

    def __init__(self, initial_balance: float = 10000.0, risk_per_trade: float = 0.01, max_positions: int = 5):
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions

        self.fetcher = TickVaultFetcher()
        self.executor = BacktestOrderExecutor(initial_balance=initial_balance, risk_per_trade=risk_per_trade, max_positions=max_positions)
        self.signal_generator = CavalierSignalGenerator()

        self.results_dir = Path("C:/Users/jack/Cavalier/CORE_MODULES/results/backtest")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        mode: ExecutionMode = ExecutionMode.BAR_CLOSE,
        output_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run complete backtest.

        Args:
            symbols: List of symbols to backtest
            start_date: Start of backtest period
            end_date: End of backtest period
            mode: Execution mode (tick/bar/signal)
            output_name: Optional name for output files

        Returns:
            Dictionary with full backtest results
        """
        logger.info(f"[backtest] Starting backtest: {symbols}")
        logger.info(f"[backtest] Period: {start_date.date()} to {end_date.date()}")

        all_bars = {}
        for symbol in symbols:
            bars = self.fetcher.load_bars(symbol, start_date, end_date, "M5")
            if not bars.empty:
                all_bars[symbol] = bars
                self.signal_generator.set_bar_data(symbol, bars)

        if not all_bars:
            logger.error("[backtest] No data available for any symbol")
            return {"error": "No data available"}

        engine = ReplayEngine(fetcher=self.fetcher, executor=self.executor, mode=mode)

        engine.set_callbacks(bar_callback=self._on_bar)

        engine.load_data(symbols, start_date, end_date)
        stats = engine.run()

        results = {
            "metadata": {
                "symbols": symbols,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "initial_balance": self.initial_balance,
                "risk_per_trade": self.risk_per_trade,
                "mode": mode.value,
            },
            "stats": stats.to_dict(),
            "trades": [self._trade_to_dict(t) for t in self.executor.closed_trades],
        }

        output_name = output_name or f"backtest_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        self._save_results(results, output_name)

        return results

    def _on_bar(self, symbol: str, bar: pd.Series):
        """Process new bar - generate signals."""
        bars = self.signal_generator.bar_cache.get(symbol)
        if bars is None or len(bars) < 20:
            return

        try:
            bar_time = bar["time"]
            if pd.isna(bar_time):
                return

            bars_list = bars[bars["time"] <= bar_time]
            if len(bars_list) < 20:
                return

            lookback = bars_list.tail(20)

            if lookback.empty or len(lookback) < 20:
                return

            signal = self.signal_generator.generate_signal(symbol, bar, lookback)

            if signal and signal.get("confidence", 0) > 0.50 and signal.get("direction", 0) != 0:
                pip_value = 0.0001

                position = self.executor.open_position(
                    symbol=signal["symbol"],
                    direction=signal["direction"],
                    entry_price=float(signal["entry_price"]),
                    sl=float(signal["sl"]),
                    tp=float(signal["tp"]),
                    entry_time=bar_time,
                    pip_value=pip_value,
                )

                if position:
                    logger.info(f"[backtest] OPENED: {symbol} {signal['direction']} @ {signal['entry_price']:.5f} conf={signal['confidence']:.2f}")
        except Exception as e:
            logger.warning(f"[backtest] Signal error for {symbol}: {e}")

    def _trade_to_dict(self, trade: BacktestTrade) -> Dict[str, Any]:
        """Convert trade to dictionary."""
        return {
            "ticket": trade.ticket,
            "symbol": trade.symbol,
            "direction": "BUY" if trade.direction > 0 else "SELL",
            "entry_time": trade.entry_time.isoformat() if trade.entry_time else None,
            "entry_price": trade.entry_price,
            "exit_time": trade.exit_time.isoformat() if trade.exit_time else None,
            "exit_price": trade.exit_price,
            "sl": trade.sl,
            "tp": trade.tp,
            "volume": trade.volume,
            "pnl": trade.pnl,
            "pnl_pips": trade.pnl_pips,
            "exit_reason": trade.exit_reason,
            "duration_bars": trade.duration_bars,
        }

    def _save_results(self, results: Dict[str, Any], name: str):
        """Save results to JSON file."""

        try:
            from core.infra.resilience import atomic_write_json
        except ImportError:

            def atomic_write_json(path, data):
                import json

                with open(path, "w") as f:
                    json.dump(data, f, indent=2, default=str)

        output_file = self.results_dir / f"{name}.json"
        atomic_write_json(output_file, results)

        logger.info(f"[backtest] Results saved to {output_file}")

        html_file = self.results_dir / f"{name}.html"
        self._generate_html_report(results, html_file)

    def _generate_html_report(self, results: Dict[str, Any], output_file: Path):
        """Generate HTML report."""
        stats = results.get("stats", {})
        trades = results.get("trades", [])

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Cavalier Backtest Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }}
        .stat-card {{ background: #f5f5f5; padding: 15px; border-radius: 8px; }}
        .stat-value {{ font-size: 24px; font-weight: bold; }}
        .positive {{ color: green; }}
        .negative {{ color: red; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #4CAF50; color: white; }}
    </style>
</head>
<body>
    <h1>Cavalier Backtest Report</h1>
    <p>Period: {results["metadata"]["start_date"][:10]} to {results["metadata"]["end_date"][:10]}</p>
    
    <h2>Performance Summary</h2>
    <div class="stats">
        <div class="stat-card">
            <div>Total Trades</div>
            <div class="stat-value">{stats.get("total_trades", 0)}</div>
        </div>
        <div class="stat-card">
            <div>Win Rate</div>
            <div class="stat-value">{stats.get("win_rate", 0):.1f}%</div>
        </div>
        <div class="stat-card">
            <div>Total PnL</div>
            <div class="stat-value {"positive" if stats.get("total_pnl", 0) >= 0 else "negative"}">
                ${stats.get("total_pnl", 0):.2f}
            </div>
        </div>
        <div class="stat-card">
            <div>Max Drawdown</div>
            <div class="stat-value negative">
                {stats.get("max_drawdown_pct", 0):.1f}%
            </div>
        </div>
        <div class="stat-card">
            <div>Winning Trades</div>
            <div class="stat-value positive">{stats.get("winning_trades", 0)}</div>
        </div>
        <div class="stat-card">
            <div>Losing Trades</div>
            <div class="stat-value negative">{stats.get("losing_trades", 0)}</div>
        </div>
        <div class="stat-card">
            <div>Profit Factor</div>
            <div class="stat-value">{stats.get("profit_factor", 0):.2f}</div>
        </div>
        <div class="stat-card">
            <div>Sharpe Ratio</div>
            <div class="stat-value">{stats.get("Sharpe_ratio", 0):.2f}</div>
        </div>
    </div>
    
    <h2>Trade History</h2>
    <table>
        <tr>
            <th>Symbol</th>
            <th>Direction</th>
            <th>Entry Time</th>
            <th>Entry</th>
            <th>Exit Time</th>
            <th>Exit</th>
            <th>PnL</th>
            <th>Pips</th>
            <th>Reason</th>
        </tr>
"""

        for trade in trades[:100]:
            pnl_class = "positive" if trade["pnl"] >= 0 else "negative"
            html += f"""
        <tr>
            <td>{trade["symbol"]}</td>
            <td>{trade["direction"]}</td>
            <td>{trade["entry_time"][:16] if trade["entry_time"] else "N/A"}</td>
            <td>{trade["entry_price"]:.5f}</td>
            <td>{trade["exit_time"][:16] if trade["exit_time"] else "Open"}</td>
            <td>{f"{trade['exit_price']:.5f}" if trade["exit_price"] else "N/A"}</td>
            <td class="{pnl_class}">${trade["pnl"]:.2f}</td>
            <td class="{pnl_class}">{trade["pnl_pips"]:.1f}</td>
            <td>{trade["exit_reason"]}</td>
        </tr>
"""

        html += """
    </table>
</body>
</html>
"""

        with open(output_file, "w") as f:
            f.write(html)

        logger.info(f"[backtest] HTML report saved to {output_file}")


async def download_and_backtest(symbols: List[str], start_date: datetime, end_date: datetime, initial_balance: float = 10000.0):
    """
    Download data and run backtest in one command.
    """
    try:
        from core.backtesting.data_fetcher import download_symbols_async
    except ImportError:
        from .data_fetcher import download_symbols_async

    print(f"Downloading data for {symbols}...")
    results = await download_symbols_async(symbols, start_date, end_date)

    for symbol, result in results.items():
        downloaded = len(result.get("dates_downloaded", []))
        cached = len(result.get("dates_cached", []))
        print(f"  {symbol}: {downloaded} downloaded, {cached} cached")

    backtester = CavalierBacktester(initial_balance=initial_balance)

    print("\nRunning backtest...")
    results = backtester.run(symbols, start_date, end_date)

    return results


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    start = datetime(2026, 1, 1)
    end = datetime(2026, 1, 31)

    results = download_and_backtest(symbols, start, end)

    if "error" not in results:
        stats = results["stats"]
        print("\n=== Backtest Results ===")
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Win Rate: {stats['win_rate']:.1f}%")
        print(f"Total PnL: ${stats['total_pnl']:.2f}")
        print(f"Max Drawdown: {stats['max_drawdown_pct']:.1f}%")
