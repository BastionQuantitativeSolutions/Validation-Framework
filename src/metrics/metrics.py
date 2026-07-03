
# Metrics for validation.

import numpy as np

def sharpe_ratio(returns, risk_free=0.0, periods_per_year=252):
    """Calculate the Sharpe ratio."""
    excess_returns = returns - risk_free / periods_per_year
    return np.sqrt(per_per_year) * np.mean(excess_returns) / np.std(excess_returns)

