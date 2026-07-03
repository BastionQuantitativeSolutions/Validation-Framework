# Validation Framework

This repository contains the extracted validation components from the Cavalier trading system, including:
- Backtesting engine
- Walk-forward validation
- Monte Carlo simulation suite
- Performance metrics
- Regime detection
- Reporting utilities

## Structure

- `src/`: Contains the core extracted components.
  - `backtest/`: Backtesting engine components.
  - `monte_carlo/`: Monte Carlo simulation scripts.
  - `regime/`: Regime detection models and utilities.
  - `metrics/`: Financial performance metrics (Sharpe, Sortino, Calmar, max drawdown, profit factor, expectancy).
  - `reporting/: Report generation utilities.
- `validation_scripts/`: Original validation scripts from the Cavalier codebase (for reference).
- `backtesting_scripts/`: Original backtesting scripts from the Cavalier codebase (for reference).
- `config/`: Configuration files for regime detection and other modules.
- `docs/`: Documentation (to be filled).
- `examples/`: Example notebooks and scripts demonstrating usage.
- `tests/`: Unit tests (to be filled).

## Usage

The extracted components are intended to be used as a foundation for building a validation pipeline. Users can import the modules from `src/` and integrate them into their own validation workflows.

## Note

The files in `validation_scripts/` and `backtesting_scripts/` are direct copies from the Cavalier codebase and may contain strategy-specific logic. The `src/` directory contains cleaned and extracted versions where possible, but some files may still contain references to the original system. Users are advised to review and adapt the code to their specific needs.

## License

MIT License - see the LICENSE file for details.
