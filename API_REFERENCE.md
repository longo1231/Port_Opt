# API Reference

Quick reference for programmatic usage of the Portfolio Optimizer.

## Core Components

### DataLoader
```python
from data.loader import DataLoader

loader = DataLoader(random_seed=42)

# Generate simulated data
data = loader.generate_simulated_returns(
    n_days=252,
    mu=np.array([0.08, 0.03, 0.05]),  # SPX, TLT, GLD
    sigma=np.array([0.16, 0.07, 0.15]),
    correlation=0.2,
    risk_free_rate=0.0525
)

# Fetch market data
data = loader.fetch_market_data('2023-01-01', '2024-01-01')
```

### Portfolio Optimization
```python
from opt.optimizer import PortfolioOptimizer
from features.estimators import estimate_all_parameters

# Estimate parameters
mu, cov_matrix, volatilities = estimate_all_parameters(
    returns_data,
    mu_method='ewma',
    mu_halflife=30,
    corr_halflife=15,
    shrinkage=0.2
)

# Optimize portfolio
optimizer = PortfolioOptimizer(
    transaction_cost_bps=2.0,
    turnover_penalty=0.001
)

result = optimizer.optimize(
    mu, cov_matrix, 
    method='quadratic'  # or 'monte_carlo'
)

optimal_weights = result['weights']
```

### Backtesting
```python
from backtest.engine import BacktestEngine, analyze_backtest_performance

engine = BacktestEngine(
    optimizer=optimizer,
    rebalance_freq='weekly'
)

results = engine.run_backtest(
    returns_data,
    start_date='2023-01-01',
    optimization_method='quadratic'
)

performance = analyze_backtest_performance(results)
```

## Configuration

All defaults in `config.py`:
```python
from config import *

# Assets
print(ASSETS)  # ['SPX', 'TLT', 'GLD', 'Cash']

# Default parameters
DEFAULT_RISK_FREE_RATE = 0.0525
DEFAULT_TRANSACTION_COST_BPS = 2.0
DEFAULT_MU_SIGMA_HALFLIFE = 30
DEFAULT_CORR_HALFLIFE = 15
```

## Example Scripts

### Simple Optimization
```python
import numpy as np
from data.loader import DataLoader
from features.estimators import estimate_all_parameters
from opt.optimizer import PortfolioOptimizer

# Load data
loader = DataLoader()
data = loader.generate_simulated_returns(n_days=252)

# Estimate parameters
mu, cov, vol = estimate_all_parameters(data)

# Optimize
optimizer = PortfolioOptimizer()
result = optimizer.optimize(mu, cov, method='quadratic')

print(f"Optimal weights: {result['weights']}")
print(f"Expected growth: {result['objective_value']:.2%}")
```

### Full Backtest
```python
from backtest.engine import BacktestEngine, analyze_backtest_performance

# Create and run backtest
engine = BacktestEngine(optimizer, rebalance_freq='daily')
results = engine.run_backtest(data)

# Analyze performance
perf = analyze_backtest_performance(results)
print(f"Sharpe ratio: {perf['sharpe_ratio']:.3f}")
print(f"Max drawdown: {perf['max_drawdown']:.2%}")
```