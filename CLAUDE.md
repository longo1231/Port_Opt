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
├── backtest.py             # Walk-forward backtesting engine
├── app.py                  # Streamlit dashboard
├── config.py               # Configuration parameters
├── test_phase1.py          # Simulated data validation
├── test_streamlit.py       # Market data validation
└── test_backtest.py        # Backtest engine validation
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
- **σ (volatility)**: 60 days (medium adaptation) - user configurable via UI sliders
- **ρ (correlation)**: 30 days (fast adaptation) - user configurable via UI sliders  
- **No μ needed** - eliminated from optimization

**Walk-Forward Backtesting:**
- **True historical analysis** - no look-ahead bias
- **Weekly rebalancing** - realistic transaction costs
- **Rolling window estimation** - uses only data available at each rebalancing date
- **Performance comparison** - dynamic vs static vs SPY benchmark

## Development Guidelines

### When making changes:

1. **Always run tests** after modifications:
   ```bash
   python test_phase1.py      # Core functionality
   python test_streamlit.py   # Market data integration
   python test_backtest.py    # Walk-forward backtest engine
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
- **app.py**: Clean Streamlit interface with walk-forward backtesting always enabled
- **backtest.py**: Walk-forward backtesting engine with no look-ahead bias
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
- **Backtesting**: Walk-forward analysis typically completes in 30-60 seconds
- **Memory**: Efficient with caching for expensive backtest computations  
- **UI**: Fast rendering with clean interface, parameter changes require full recomputation
- **Stability**: Converges reliably, no solver issues

## Current Application State (August 2025 - Pre-Leverage Implementation)

**Major Features:**
- **Always-on Walk-Forward Backtesting**: Replaced misleading perfect foresight analysis with realistic historical backtesting
- **Interactive Rolling Windows**: Users can adjust volatility (σ) and correlation (ρ) estimation windows via UI sliders
- **Comprehensive Performance Comparison**: Three-way comparison (Dynamic vs Static vs SPY) with full metrics
- **Advanced Visualizations**: Portfolio weight evolution, drawdown analysis, rolling risk metrics
- **Realistic Cost Analysis**: Turnover tracking and rebalancing frequency metrics

**Chart Suite:**
- **Current Portfolio Weights**: Bar chart showing optimal allocation
- **Performance Comparison**: Dynamic portfolio performance vs static vs SPY benchmark
- **Weight Evolution**: Stacked area chart showing portfolio changes over time
- **Drawdown Analysis**: Peak-to-trough declines for all portfolios and individual assets
- **Rolling Risk Metrics**: 60-day rolling Sharpe ratio and volatility with dual y-axes
- **Correlation Heatmap**: Asset correlation matrix with color coding

**User Interface:**
- **Sidebar Controls**: Date range selection, rolling window parameters, rebalancing frequency
- **Main Dashboard**: Current portfolio weights, expected returns (display only), historical analysis
- **Performance Sections**: Dynamic comparison, static portfolio, SPY benchmark (5 columns each)
- **Charts**: Clean hover tooltips with unified date display, professional formatting

**Technical Implementation:**
- **Caching**: Expensive backtest computations cached for 1 hour with smart cache keys
- **Data Pipeline**: Extended historical data fetching for proper rolling windows
- **Error Handling**: Graceful fallbacks for optimization failures and data issues
- **Parameter Validation**: Rolling windows properly passed through entire backtest pipeline
- **Chart Architecture**: Plotly-based interactive visualizations with consistent styling

**Core Assets & Data:**
- **Risky Assets**: SPY (US Equity), TLT (Long Treasury), GLD (Gold)
- **Cash Proxy**: BIL (Treasury Bills) for realistic short-term rates
- **Data Source**: Yahoo Finance with robust error handling and fallbacks
- **Returns Calculation**: Simple returns (not log returns) for correct optimization

## Upcoming Feature: Volatility Targeting with Leverage (In Development)

**Objective**: Allow users to apply leverage to achieve target volatility levels
**Implementation Status**: Ready to begin development 

**Core Features:**
- **Target Volatility Slider**: 1% to 25% range, 10% default, 0.5% increments
- **Automatic Leverage Calculation**: System calculates required leverage to hit target
- **Leverage Cap**: Maximum 3x leverage for risk management
- **Negative Cash Display**: Bar chart shows borrowing when leverage > 1.0x
- **Cost Integration**: Uses BIL rate for borrowing costs (simplified approach)

**Technical Approach:**
- **Step 1**: Extract risky assets only from covariance matrix (3x3: SPY, TLT, GLD)
- **Step 2**: Optimize minimum variance on risky assets only (weights sum to 1.0)
- **Step 3**: Calculate MVP volatility for risky portfolio
- **Step 4**: Determine leverage = target_volatility / mvp_volatility (max 3.0x)
- **Step 5**: Scale risky weights by leverage, set cash = 1.0 - leverage
- **Step 6**: Adjust returns for borrowing costs when cash < 0

**Backtesting Integration:**
- **Constant Target Volatility**: Each rebalance maintains user's target volatility
- **Variable Leverage**: Leverage adjusts as MVP volatility changes over time
- **Turnover Tracking**: Monitor both weight changes and leverage adjustments
- **Performance Metrics**: Show leveraged vs unleveraged comparisons

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

**Rolling Window Parameters Not Applied in Backtest (SOLVED):**
- **Problem**: Changing correlation/volatility windows in UI had no effect on backtest results
- **Root Cause**: `walk_forward_backtest()` used hardcoded config values instead of user parameters
- **Solution**: Added `sigma_window` and `rho_window` parameters to backtest function and pipeline
- **Impact**: Rolling window sliders now dramatically affect portfolio allocation and performance
- **Files**: backtest.py, app.py
- **Validation**: Extreme window differences show 20%+ allocation changes and 8%+ return differences

**Portfolio Weight Evolution Chart Display Issues (SOLVED):**
- **Problem**: Stacked area chart showed incorrect cumulative values in hover tooltips
- **Root Cause**: Plotly stackgroup approach caused rendering issues with TLT and GLD areas
- **Solution**: Reimplemented using explicit polygon fills with custom hover templates  
- **Impact**: Visual areas now match numerical weights exactly, hover shows individual weights
- **File**: app.py create_weight_evolution_chart()

**Performance Comparison UI Improvements (COMPLETED):**
- **Added**: Annualized return metrics to all three portfolio comparison sections
- **Removed**: Redundant metrics (turnover, beta, rebalancing activity)  
- **Improved**: Consistent 5-column layout for Dynamic, Static, and SPY portfolios
- **Enhanced**: Color-coded delta comparisons for all metrics
- **File**: app.py performance comparison sections

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

1. **Cash optimization bias (SOLVED)**
   - **Solution**: `EXCLUDE_CASH_FROM_OPTIMIZATION = True` in config
   
2. **Yahoo Finance data structure changes**
   - **Solution**: Robust column parsing in data/loader.py
   
3. **Correlation estimation with limited data**
   - **Solution**: Fallback to reasonable windows, handle NaN correlations

4. **UI scroll position resets on parameter changes**
   - **Limitation**: Streamlit framework behavior - parameter changes require full page recomputation
   - **Workaround**: Users should expect to scroll back to results section after adjusting windows
   - **Considered**: Anchor links, manual update buttons - deemed too complex for benefit

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