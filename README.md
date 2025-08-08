# Kelly Criterion Portfolio Optimizer

A production-quality interactive dashboard for optimizing long-only portfolios using the Kelly Criterion across four assets: SPX, TLT, GLD, and Cash.

![Portfolio Optimizer](https://img.shields.io/badge/Status-Production%20Ready-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)

## 🎯 Overview

This application implements the Kelly Criterion for portfolio optimization, maximizing expected geometric growth `E[log(1 + w^T R)]` under long-only and full-investment constraints. It provides both simulated data testing (Phase 1) and live market data integration (Phase 2).

### Key Features

- **Two Optimization Methods:**
  - Quadratic Approximation: `max w^T μ - 0.5 * w^T Σ w` (fast Kelly approximation)
  - Monte Carlo: Exact expected log utility via sampling

- **Flexible Parameter Estimation:**
  - Rolling windows or EWMA with configurable half-lives
  - Separate parameterization for μ/σ vs correlation estimation
  - Covariance shrinkage regularization

- **Realistic Trading Costs:**
  - Transaction costs in basis points
  - Turnover penalties for regularization
  - Daily/weekly/monthly rebalancing frequencies

- **Comprehensive Analysis:**
  - Interactive Streamlit dashboard
  - Real-time sensitivity analysis
  - Detailed backtesting with drawdown analysis
  - Performance attribution and cost decomposition

## 📊 Assets

| Asset | Description | Role |
|-------|-------------|------|
| **SPX** | S&P 500 Index | Equity exposure |
| **TLT** | 20+ Year Treasury Bond ETF | Duration/rates exposure |  
| **GLD** | Gold ETF | Inflation hedge/alternative |
| **Cash** | Risk-free asset | Liquidity/safety |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Port_Optimizer

# Install dependencies
pip install -r requirements.txt
```

### Phase 1: Simulated Data Testing

```bash
# Validate core functionality
python test_phase1.py

# Expected output: All tests should pass ✅
```

### Phase 2: Launch Dashboard

```bash
# Start the Streamlit dashboard
streamlit run ui/app.py

# Open browser to http://localhost:8501
```

### Alternative: Test Market Data Integration

```bash
# Verify Yahoo Finance connectivity
python test_streamlit.py
```

## 📈 Usage Guide

### 1. Control Panel (Sidebar)

**Data Source:**
- Phase 1: Simulated data with configurable parameters
- Phase 2: Live market data from Yahoo Finance

**Estimation Parameters:**
- Method: EWMA (faster adaptation) or Rolling Window
- μ/σ Half-life: 30 days default (return/volatility estimation)
- ρ Half-life: 15 days default (correlation estimation - faster)
- Covariance Shrinkage: 0.2 default (regularization strength)

**Optimization Settings:**
- Method: Quadratic (fast) or Monte Carlo (exact)
- Transaction Costs: 2 bps per side default
- Turnover Penalty: L1 regularization coefficient
- Rebalancing Frequency: Daily/Weekly/Monthly

### 2. Main Dashboard

**Portfolio Optimization:**
- Current optimal weights with expected statistics
- Risk-return visualization and correlation heatmap
- Performance attribution analysis

**Strategy Backtesting:**
- Historical performance simulation
- Cumulative returns (log scale available)
- Drawdown analysis and turnover tracking
- Comprehensive performance metrics

## 🔧 Architecture

```
Port_Optimizer/
├── data/
│   ├── __init__.py
│   └── loader.py          # Data generation & Yahoo Finance integration
├── features/
│   ├── __init__.py
│   └── estimators.py      # μ, σ, ρ estimation with EWMA/rolling
├── opt/
│   ├── __init__.py
│   └── optimizer.py       # Kelly optimization (quadratic + Monte Carlo)
├── backtest/
│   ├── __init__.py
│   └── engine.py          # Portfolio simulation with costs
├── ui/
│   ├── __init__.py
│   └── app.py             # Streamlit dashboard
├── utils/
│   ├── __init__.py
│   └── metrics.py         # Performance & risk metrics
├── config.py              # Default parameters
├── requirements.txt       # Dependencies
├── test_phase1.py         # Simulated data validation
└── test_streamlit.py      # Market data validation
```

## 📊 Mathematical Framework

### Kelly Criterion

For a portfolio with weights **w** and returns **R**, the Kelly criterion maximizes:

```
E[log(1 + w^T R)]
```

**Quadratic Approximation** (small returns):
```
max w^T μ - 0.5 * w^T Σ w - λ * ||w - w_prev||_1
```

**Monte Carlo** (exact):
```
max E[log(1 + w^T R)] via sampling from MVN(μ, Σ)
```

**Constraints:**
- Long-only: `w_i ≥ 0`
- Full investment: `Σ w_i = 1`
- No leverage or shorting

### Parameter Estimation

**Expected Returns (μ):**
- Rolling: Sample mean over window
- EWMA: Exponentially weighted with half-life

**Volatilities (σ):**
- Rolling: Sample standard deviation
- EWMA: Exponentially weighted variance

**Correlations (ρ):**
- Faster adaptation (shorter half-life) vs μ/σ
- Positive definite enforcement with eigenvalue clipping

**Covariance Matrix:**
```
Σ = (1-λ) * σσ^T ⊙ ρ + λ * diag(σ²)
```
where λ is shrinkage intensity.

## ⚙️ Configuration

### Default Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| μ/σ Half-life | 30 days | EWMA adaptation for returns/volatility |
| ρ Half-life | 15 days | Faster correlation adaptation |
| Lookback Window | 252 days | Rolling window size |
| Risk-free Rate | 5.25% | Cash return (annualized) |
| Transaction Cost | 2 bps | Cost per side |
| Turnover Penalty | 0.001 | L1 regularization coefficient |
| Shrinkage | 0.2 | Covariance regularization |
| MC Simulations | 10,000 | Monte Carlo sample size |

### Customization

All parameters are configurable through:
1. **Streamlit UI**: Interactive sliders and inputs
2. **config.py**: Programmatic defaults
3. **Function arguments**: Direct parameter passing

## 📋 Performance Metrics

The system tracks comprehensive performance statistics:

**Return Metrics:**
- Total Return, Annualized Return
- Sharpe Ratio, Sortino Ratio
- Hit Rate (% positive periods)

**Risk Metrics:**
- Volatility (annualized)
- Maximum Drawdown
- Value at Risk (5%)
- Calmar Ratio

**Cost Analysis:**
- Transaction Costs (bps)
- Turnover (% portfolio changed)
- Cost Drag (annualized impact)

**Attribution:**
- Expected Return Contribution
- Risk Penalty
- Turnover Penalty
- Transaction Costs

## 🧪 Testing

The project includes comprehensive test suites:

### Phase 1 Tests (Simulated Data)
```bash
python test_phase1.py
```
Validates:
- Data generation and validation
- Parameter estimation (EWMA & rolling)
- Portfolio optimization (both methods)
- Attribution analysis
- Backtesting engine

### Phase 2 Tests (Market Data)
```bash
python test_streamlit.py
```
Validates:
- Yahoo Finance connectivity
- Market data parsing
- Streamlit app functionality

## 🔍 Troubleshooting

### Common Issues

**1. Yahoo Finance Connection Errors**
- Network connectivity issues
- API rate limiting
- Solution: Retry or use shorter date ranges

**2. Correlation Matrix Issues**
- NaN values in EWMA calculations
- Solution: System uses fallback correlation matrices

**3. CVXPY Solver Not Found**
- Missing optimization solver
- Solution: System falls back to SciPy SLSQP automatically

**4. Memory Issues with Large Backtests**
- Large date ranges or high-frequency rebalancing
- Solution: Use weekly/monthly rebalancing or shorter periods

### Debug Mode

Set environment variables for additional logging:
```bash
export STREAMLIT_LOGGER_LEVEL=debug
streamlit run ui/app.py
```

## 📚 References

1. Kelly Jr, J. L. (1956). "A new interpretation of information rate"
2. Thorp, E. O. (2006). "The Kelly Capital Growth Investment Criterion"
3. MacLean, L. C., Thorp, E. O., & Ziemba, W. T. (2011). "The Kelly Capital Growth Investment Criterion: Theory and Practice"

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not intended as investment advice. Trading and investment decisions carry inherent risks, and users should consult with qualified financial professionals before making investment decisions. Past performance does not guarantee future results.

---

**Built with ❤️ using Python, Streamlit, and modern quantitative finance principles.**