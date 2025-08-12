# API Reference

Quick reference for programmatic usage of the Minimum Variance Portfolio Optimizer.

## Core Components

### DataLoader
```python
from data.loader import DataLoader

loader = DataLoader(random_seed=42)

# Fetch market data (simple returns)
data = loader.fetch_market_data('2023-01-01', '2024-01-01')

# Generate simulated data for testing
data = loader.generate_simulated_returns(
    n_days=252,
    mu=np.array([0.08, 0.03, 0.05, 0.02]),  # SPY, TLT, GLD, Cash
    sigma=np.array([0.16, 0.07, 0.15, 0.001]),
    risk_free_rate=0.053
)

# Validate data quality
is_valid, message = validate_returns_data(data)
```

### Parameter Estimation
```python
from estimators import estimate_covariance_matrix, get_estimation_info

# Estimate covariance matrix only (no expected returns needed)
covariance_matrix = estimate_covariance_matrix(returns_data)

# Get information about estimation windows
info = get_estimation_info(n_days=len(returns_data))
print(f"Volatility window: {info['sigma_window']}")
print(f"Correlation window: {info['rho_window']}")
```

### Minimum Variance Optimization
```python
from optimizer import optimize_min_variance, analyze_min_variance_portfolio

# Optimize for minimum variance
result = optimize_min_variance(
    covariance_matrix,
    exclude_cash=True  # Focus on risky assets only
)

if result['success']:
    weights = result['weights']
    portfolio_vol = result['portfolio_volatility']
    diversification_ratio = result['diversification_ratio']
    effective_assets = result['effective_n_assets']
    
    # Detailed analysis
    analysis = analyze_min_variance_portfolio(
        weights, 
        covariance_matrix, 
        asset_names=['SPY', 'TLT', 'GLD', 'Cash']
    )
```

## Configuration

All settings in `config.py`:
```python
from config import *

# Assets
print(ASSETS)  # ['SPY', 'TLT', 'GLD', 'Cash']

# Estimation windows (Breaking the Market)
SIGMA_WINDOW = 60   # Volatility: medium adaptation
RHO_WINDOW = 30     # Correlations: fast adaptation

# Portfolio settings
MIN_WEIGHT = 1e-4   # Minimum position size
EXCLUDE_CASH_FROM_OPTIMIZATION = True

# Risk-free rate
RISK_FREE_RATE = 0.053  # 5.3% annual
```

## Example Scripts

### Simple Minimum Variance Optimization
```python
import numpy as np
from data.loader import DataLoader
from estimators import estimate_covariance_matrix
from optimizer import optimize_min_variance

# Load market data
loader = DataLoader()
data = loader.fetch_market_data('2022-01-01', '2024-01-01')

# Estimate covariance matrix (no expected returns needed)
covariance_matrix = estimate_covariance_matrix(data)

# Optimize for minimum variance
result = optimize_min_variance(covariance_matrix)

if result['success']:
    print(f"Portfolio weights: {result['weights']}")
    print(f"Portfolio volatility: {result['portfolio_volatility']:.2%}")
    print(f"Diversification ratio: {result['diversification_ratio']:.2f}")
    print(f"Effective # assets: {result['effective_n_assets']:.1f}")
else:
    print(f"Optimization failed: {result['error']}")
```

### Detailed Portfolio Analysis
```python
from optimizer import analyze_min_variance_portfolio
from config import ASSETS

# Analyze the optimized portfolio
analysis = analyze_min_variance_portfolio(
    result['weights'], 
    covariance_matrix, 
    ASSETS
)

print("Asset Analysis:")
for item in analysis['asset_analysis']:
    print(f"{item['asset']:4s}: {item['weight']:6.2%} weight, "
          f"{item['risk_contrib_pct']:6.2%} risk contribution")

print(f"\nDiversification metrics:")
print(f"Max weight: {analysis['max_weight']:.2%}")
print(f"Active assets: {analysis['n_nonzero_assets']}")
print(f"Portfolio volatility: {analysis['portfolio_volatility']:.2%}")
```

### Streamlit Dashboard Integration
```python
import streamlit as st
from data.loader import DataLoader
from estimators import estimate_covariance_matrix
from optimizer import optimize_min_variance
from datetime import datetime, timedelta

# Streamlit app structure
st.title("Minimum Variance Portfolio Optimizer")

# Date range selection
end_date = datetime.now()
start_date = end_date - timedelta(days=365*3)
date_range = st.date_input("Select date range", value=(start_date, end_date))

if len(date_range) == 2:
    # Load and optimize
    loader = DataLoader()
    data = loader.fetch_market_data(
        date_range[0].strftime('%Y-%m-%d'),
        date_range[1].strftime('%Y-%m-%d')
    )
    
    covariance_matrix = estimate_covariance_matrix(data)
    result = optimize_min_variance(covariance_matrix)
    
    if result['success']:
        # Display results
        st.metric("Portfolio Volatility", f"{result['portfolio_volatility']:.2%}")
        st.metric("Diversification Ratio", f"{result['diversification_ratio']:.2f}")
        
        # Portfolio weights chart
        import plotly.graph_objects as go
        fig = go.Figure(data=go.Bar(x=ASSETS, y=result['weights']*100))
        st.plotly_chart(fig)
```

## Key Benefits of Minimum Variance Approach

**Mathematical Stability**:
- No expected returns estimation required
- Covariance matrices more stable than expected returns  
- Eliminates main source of optimization instability

**Natural Diversification**:
- Only way to reduce portfolio risk is through correlation benefits
- Automatically balances assets based on their covariance relationships
- Avoids corner solutions (100% single asset allocations)

**Practical Implementation**:
- Simple objective function: minimize w'Σw
- Standard convex optimization problem
- Fast convergence with scipy.optimize.minimize
- Robust to various market conditions

**Validation Metrics**:
- Diversification ratio >1.0 indicates correlation benefits
- Effective number of assets >1.5 shows meaningful diversification
- Risk contributions show how each asset adds to portfolio risk