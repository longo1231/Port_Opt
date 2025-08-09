# Portfolio Optimizer - Breaking the Market

A clean Kelly criterion portfolio optimizer following the "Breaking the Market" methodology.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)

## 🎯 Overview

This application implements pure Kelly criterion portfolio optimization for SPY, TLT, GLD, and Cash using the "Breaking the Market" approach:

- **μ (returns)**: 252-day window (slow, uncertain)
- **σ (volatility)**: 60-day window (medium adaptation)  
- **ρ (correlations)**: 30-day window (fast regime changes)

### Key Features

- **Pure Kelly Optimization**: `max w^T μ - 0.5 w^T Σ w`
- **Simple Returns**: Fixed log vs simple returns bug for proper Kelly calculation
- **Real-time Data**: Yahoo Finance integration with proper error handling
- **Interactive Dashboard**: Clean Streamlit interface with minimal controls
- **Breaking the Market Philosophy**: Fast correlations, medium volatility, slow returns

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   python run.py
   ```

3. **Open your browser** to http://localhost:8503

## 📁 Project Structure

```
Port_Optimizer/
├── run.py                  # Main launcher
├── app_simple.py          # Streamlit dashboard  
├── config_simple.py       # Configuration (windows, assets)
├── estimators_simple.py   # Parameter estimation
├── optimizer_simple.py    # Kelly optimization
├── data/
│   └── loader.py          # Yahoo Finance data loading
├── legacy_system/         # Original complex system (archived)
├── debug_archive/         # Debugging scripts (archived)
├── tests_archive/         # Test scripts (archived)
└── docs_archive/          # Documentation (archived)
```

## 🔧 Configuration

Edit `config_simple.py` to modify:

- **Assets**: Currently SPY, TLT, GLD, Cash
- **Estimation windows**: MU_WINDOW=252, SIGMA_WINDOW=60, RHO_WINDOW=30
- **Risk-free rate**: For cash returns
- **Date ranges**: Default lookback period

## 🧮 Methodology

### Breaking the Market Approach

1. **Returns (μ)**: Use longer window (252 days) since returns are uncertain and noisy
2. **Volatility (σ)**: Use medium window (60 days) for moderate adaptation to regime changes  
3. **Correlations (ρ)**: Use short window (30 days) since correlations change quickly

### Kelly Criterion

Maximizes expected log growth: `E[log(1 + w^T R)]`

Quadratic approximation: `max w^T μ - 0.5 w^T Σ w`
Subject to: `Σw = 1, w ≥ 0`

### Data Processing

- **Simple returns**: `r_t = P_t / P_{t-1} - 1` (not log returns)
- **Proper annualization**: Both μ and Σ consistently annualized
- **Covariance construction**: `Σ = D·ρ·D` where D = diag(σ)

## 📊 Dashboard Features

- **Parameter Display**: Shows current μ, σ, Sharpe ratios
- **Correlation Matrix**: Real-time correlation heatmap
- **Optimization Results**: Portfolio weights and statistics
- **Time Period Selection**: Interactive date range picker
- **Methodology Info**: Displays estimation windows used

## 🔍 Debugging

Debug scripts are archived in `debug_archive/` including:

- **Analytical solutions**: Verify optimization math
- **Correlation analysis**: Check correlation calculations
- **Kelly formula tests**: Different Kelly formulations
- **Data validation**: Yahoo Finance data quality checks

## 📈 Recent Fixes

- ✅ Fixed log vs simple returns bug (major impact)
- ✅ Proper covariance matrix usage in optimization
- ✅ Corrected μ estimation to use rolling windows
- ✅ Fixed correlation calculation and display
- ✅ Cleaned up UI and removed unnecessary complexity

## 🚨 Important Notes

**Mathematical Correctness**: The optimizer will sometimes show 100% allocation to a single asset. This is mathematically correct Kelly behavior when:
- One asset has significantly higher risk-adjusted returns
- Return differentials outweigh diversification benefits
- The selected time period favors one asset strongly

**Diversification**: Diversification benefits appear when assets have similar risk-adjusted returns and low/negative correlations.

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

This is a cleaned-up implementation following debugging and optimization work. The original complex system is preserved in `legacy_system/` for reference.