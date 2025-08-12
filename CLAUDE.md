# Claude Code Project Instructions

This file contains instructions for Claude on how to work with this Portfolio Optimizer project.

## Project Overview

This is a **Minimum Variance Portfolio Optimizer** built with:
- **Python 3.8+** with NumPy, Pandas, SciPy stack
- **Streamlit** for the interactive dashboard
- **Plotly** for visualizations  
- **yfinance** for market data
- **SciPy** for quadratic optimization

**Key Innovation**: Eliminates expected returns from the optimization to avoid corner solutions and achieve natural diversification.

## Project Structure

```
Port_Optimizer/
├── data/loader.py          # Yahoo Finance data (simple returns)
├── estimators.py           # Covariance matrix estimation only
├── optimizer.py            # Minimum variance optimization
├── app.py                  # Streamlit dashboard
├── config.py               # Configuration parameters
├── test_phase1.py          # Simulated data validation
└── test_streamlit.py       # Market data validation
```

## Core Methodology: Minimum Variance

**Objective**: Minimize w'Σw (portfolio variance)
**Constraint**: Σw = 1, w ≥ 0 (budget constraint, long-only)

**Why This Works:**
1. **Eliminates expected returns (μ)** - the main source of instability
2. **Focuses on correlation benefits** - the only way to reduce risk is diversification
3. **No corner solutions** - single assets can never be optimal
4. **Stable and robust** - covariance matrices are more stable than expected returns

**Parameter Estimation (Breaking the Market windows):**
- **σ (volatility)**: 60 days (medium adaptation)
- **ρ (correlation)**: 30 days (fast adaptation)
- **No μ needed** - eliminated from optimization

## Development Guidelines

### When making changes:

1. **Always run tests** after modifications:
   ```bash
   python test_phase1.py  # Core functionality
   python test_streamlit.py  # Market data integration
   ```

2. **Follow existing patterns**:
   - NumPy-style docstrings
   - Type hints throughout
   - Error handling with fallbacks
   - Comprehensive input validation

3. **Key design principles**:
   - **Simplicity**: Minimum variance eliminates complexity
   - **Robustness**: Graceful handling of data issues, API failures
   - **Diversification**: Natural diversification through correlation benefits
   - **Stability**: Covariance-only approach is more stable

### Critical Files:

- **config.py**: All parameters - focused on minimum variance approach
- **app.py**: Clean Streamlit interface showing diversification metrics
- **optimizer.py**: Core minimum variance optimization logic
- **estimators.py**: Covariance matrix estimation (no expected returns)
- **data/loader.py**: Simple returns (not log returns) - critical for correct optimization

### Common Tasks:

**Adding new assets:**
1. Update `ASSETS` in config.py
2. Update ticker mapping in `data/loader.py`
3. Test with live data

**Modifying risk model:**
1. Update estimation windows in config.py (`SIGMA_WINDOW`, `RHO_WINDOW`)
2. Modify `estimate_covariance_matrix()` in estimators.py
3. Test with various market conditions

**UI enhancements:**
1. Add metrics to `app.py`
2. Update analysis functions in `optimizer.py`
3. Ensure diversification focus remains clear

### Dependencies Management:

- **Required**: numpy, pandas, scipy, streamlit, plotly, yfinance
- **No complex solvers needed**: Uses scipy.optimize.minimize with SLSQP
- **Development**: pytest, black, mypy for code quality

### Error Handling:

This system has focused fallbacks:
- **Cash handling**: Excludes Cash from optimization to avoid 100% cash allocation
- **Correlation matrices**: Handles NaN correlations gracefully
- **Market data**: Robust Yahoo Finance data handling
- **Numerical stability**: Minimum weight thresholds, proper normalization

### Testing Philosophy:

- **Phase 1**: Validates minimum variance with simulated data
- **Phase 2**: Validates market data integration and diversification
- **Focus**: Ensure no 100% single-asset allocations
- **Validation**: Check diversification ratio and effective number of assets

### Performance Notes:

- **Optimization**: ~10ms for 4 assets using SLSQP
- **Memory**: Efficient - no complex backtesting needed
- **UI**: Fast rendering with clean interface
- **Stability**: Converges reliably, no solver issues

## Key Metrics and Validation

**Diversification Metrics:**
- **Diversification Ratio**: Weighted avg vol / portfolio vol (>1 indicates benefit)
- **Effective # Assets**: 1 / Σ(w²) (4.0 = equal weights, 1.0 = single asset)
- **Max Weight**: Largest position (should be reasonable, not 100%)
- **Risk Contributions**: How much each asset contributes to portfolio risk

**Success Criteria:**
- Portfolio should have 2+ assets with meaningful weights (>5%)
- Diversification ratio should be > 1.0
- Effective number of assets should be > 1.5
- No single asset should dominate (>80% weight) except in extreme cases

## Critical Bug Fix History

**Log vs Simple Returns Issue (SOLVED):**
- **Problem**: Using log returns caused double-penalty on variance in Kelly optimization
- **Solution**: Changed to simple returns using `pct_change()` instead of `np.log()`
- **Impact**: Eliminated bias toward low-volatility assets, enabled proper diversification
- **File**: data/loader.py line 231

**Cash Domination Issue (SOLVED):**
- **Problem**: Minimum variance optimizer chose 100% Cash (zero volatility)
- **Solution**: Exclude Cash from optimization (`EXCLUDE_CASH_FROM_OPTIMIZATION = True`)
- **Impact**: Focuses optimization on risky assets only
- **File**: config.py, optimizer.py

## Maintenance Tasks

### Regular:
- Validate portfolio diversification (no 100% allocations)
- Test market data connectivity
- Check correlation matrices remain reasonable

### When adding features:
- Maintain focus on minimum variance approach
- Preserve diversification benefits
- Update tests to validate new functionality
- Ensure UI clearly shows diversification metrics

### Before releases:
- Verify diversified portfolios across different market conditions
- Test with various date ranges
- Validate all diversification metrics
- Ensure documentation reflects minimum variance approach

## Known Issues & Workarounds

1. **Cash optimization bias**
   - **Solution**: `EXCLUDE_CASH_FROM_OPTIMIZATION = True` in config
   
2. **Yahoo Finance data structure changes**
   - **Solution**: Robust column parsing in data/loader.py
   
3. **Correlation estimation with limited data**
   - **Solution**: Fallback to reasonable windows, handle NaN correlations

## Development Environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py  # Launch the optimizer
```

## Production Deployment

The system is designed for deployment with:
- Clean minimum variance approach
- Clear diversification metrics
- No complex dependencies
- Robust error handling
- User-friendly interface showing "why" diversification works