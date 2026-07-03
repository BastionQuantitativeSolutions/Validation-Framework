'# Backtest engine for Validation-Framework.

import pandas as pd
import numpy as np

class BacktestEngine:
    def __init__(self, initial_capital=100000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = 0.0
        self.equity_curve = []

    def run(self, prices, signals):
        """
        Run a backtest on price data with trading signals.

        Parameters:
        prices (pd.Series): Series of prices (close) indexed by datetime.
        signals (pd.Series): Series of signals (1 for long, -1 for short, 0 for exit) indexed by datetime.

        Returns:
        pd.Series: Equity curve.
        """
        # Ensure the indices are aligned
        if not prices.index.equals(signals.index):
            raise ValueError("Prices and signals must have the same index.")

        self.equity_curve = [self.initial_capital]
        self.position = 0
        self.entry_price = 0.0

        for i in range(1, len(prices)):
            prev_price = prices.iloc[i-1]
            curr_price = prices.iloc[i]
            signal = signals.iloc[i]

            # Update equity based on position
            if self.position == 1:  # long
                self.capital += self.position * (curr_price - prev_price)
            elif self.position == -1:  # short
                self.capital += self.position * (prev_price - curr_price)

            # Check for signal to change position
            if signal == 1 and self.position <= 0:  # enter long
                if self.position == -1:  # close short first
                    self.capital += self.position * (prev_price - curr_price)  # pips from close to open?
                self.position = 1
                self.entry_price = curr_price
            elif signal == -1 and self.position >= 0:  # enter short
                if self.position == 1:  # close long first
                    self.capital += self.position * (curr_price - prev_price)
                self.position = -1
                self.entry_price = curr_price
            elif signal == 0:  # exit position
                if self.position != 0:
                    self.capital += self.position * (curr_price - prev_price)
                    self.position = 0
                    self.entry_price = 0.0

            self.equity_curve.append(self.capital)

        return pd.Series(self.equity_curve, index=prices.index)

    def get_stats(self):
        """Calculate basic statistics."""
        if len(self.equity_curve) < 2:
            return {}
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0
        max_drawdown = (np.max(np.maximum.accumulate(self.equity_curve) - self.equity_curve) / np.max(np.maximum.accumulate(self.equity_curve))) if np.max(self.equity_curve) != 0 else 0
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "final_equity": self.equity_curve[-1],
        }
'
