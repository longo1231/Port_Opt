# Changelog

All notable changes to the Minimum Variance Portfolio Optimizer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-01-12

### 🎉 Major Architecture Shift - Minimum Variance Approach

**Fundamental Change**: Eliminated Kelly Criterion and expected returns optimization in favor of minimum variance approach for stable, diversified portfolios.

#### Why the Change?
- **Kelly Criterion Issues**: Expected returns (μ) are extremely noisy and unstable
- **Corner Solutions**: Optimizers heavily favored single assets (100% allocations)
- **Volatility**: Small changes in μ caused massive portfolio shifts
- **Solution**: Minimum variance focuses purely on risk reduction through diversification

### Changed

#### Core Optimization
- **Old**: Kelly Criterion `max E[log(1 + r)]` with expected returns
- **New**: Minimum Variance `min w'Σw` eliminating expected returns entirely
- **Benefit**: Naturally diversified portfolios, no corner solutions
- **Stability**: Covariance matrices much more stable than expected returns

#### Parameter Estimation  
- **Retained**: Breaking the Market windows (σ=60d medium, ρ=30d fast)
- **Eliminated**: Expected returns estimation (no longer needed)
- **Enhanced**: Robust covariance matrix estimation with fallbacks
- **Added**: Treasury bill integration (BIL ETF) for realistic cash returns

#### User Interface
- **Enhanced**: Clear minimum variance methodology explanation
- **Added**: Risk contribution analysis (optimized vs equal weight)
- **Added**: Historical performance analysis with quick date buttons
- **Added**: Comprehensive volatility analysis (period vs window)
- **Fixed**: Dynamic chart labeling based on actual slider values

### Fixed

#### Critical Bug Fixes
1. **Yahoo Finance Ticker Mapping** (Major)
   - **Problem**: Column order mismatch caused wrong data assignment
   - **Impact**: SPY data assigned to GLD, completely wrong calculations
   - **Solution**: Proper ticker-to-column mapping in `data/loader.py:219-231`
   - **Detection**: User noticed impossible YTD returns (SPY 25%, GLD 2%)

2. **Simple vs Log Returns**
   - **Problem**: Mixed return types caused optimization inconsistencies  
   - **Solution**: Standardized on simple returns (`pct_change()`) throughout
   - **Impact**: Proper risk calculations and portfolio weights

3. **Cash Domination**
   - **Problem**: Optimizer chose 100% Cash (zero volatility optimal)
   - **Solution**: Exclude cash from optimization, focus on risky asset diversification
   - **Config**: `EXCLUDE_CASH_FROM_OPTIMIZATION = True`

#### UI/UX Improvements
- **Volatility Display**: Separated period vs window volatility appropriately
- **Date Selection**: Quick buttons (1Y, 2Y, 3Y, 5Y, 10Y, YTD) for easy analysis
- **Returns Table**: Show actual period volatility, not window volatility
- **Chart Labels**: Dynamic labeling based on actual slider values

### Added

#### New Features
- **Treasury Integration**: Real Treasury bill returns via BIL ETF
- **Historical Analysis**: Comprehensive backtest showing how current portfolio would have performed
- **Risk Analysis**: Side-by-side comparison of optimized vs equal weight risk contributions
- **Advanced Metrics**: Rolling Sharpe ratio and drawdown analysis
- **Period Selection**: Quick date range buttons with custom option

#### Enhanced Visualizations
- **Risk Contributions**: Shows why minimum variance works (equal marginal risk)
- **Volatility Comparison**: Individual vs portfolio volatility with diversification benefits
- **Performance Charts**: Historical performance with proper benchmarking
- **Dynamic Labels**: Charts update based on actual parameter settings

### Technical Implementation

#### Current Architecture
```
Port_Optimizer/
├── app.py                  # Streamlit dashboard (minimum variance focused)
├── optimizer.py            # Minimum variance optimization  
├── estimators.py           # Covariance estimation (no μ needed)
├── config.py               # Configuration parameters
├── data/loader.py          # Yahoo Finance + Treasury integration
├── run.py                  # Application launcher
└── CLAUDE.md               # Development documentation
```

#### Key Dependencies
- **Core**: NumPy, Pandas, SciPy (for optimization)
- **Data**: yfinance (market data), BIL ETF (Treasury bills)
- **UI**: Streamlit, Plotly (interactive charts)
- **Optimization**: scipy.optimize.minimize (SLSQP method)

### Performance & Reliability

#### Optimization Results
- **Typical Allocation**: SPY ~18%, TLT ~55%, GLD ~27% (varies with conditions)
- **Diversification Ratio**: ~1.4 (40% volatility reduction from correlation benefits)
- **Effective Assets**: ~2.8 (well-diversified, not concentrated)
- **Stability**: Portfolio weights change gradually with market conditions

#### Error Handling
- **Data Quality**: Robust Yahoo Finance parsing with fallbacks
- **Treasury Integration**: Graceful fallback to constant rate if BIL unavailable
- **Correlation Issues**: Fallback correlation matrices for sparse data
- **Numerical Stability**: Positive definite enforcement

### Success Metrics

A properly functioning minimum variance portfolio shows:
- **Multiple Assets**: Meaningful weights (>5%) across assets
- **Diversification Ratio > 1.0**: Clear correlation benefits
- **Effective Assets > 1.5**: Not concentrated in single position  
- **No Domination**: No single asset >80% (except extreme market conditions)

### Known Limitations

1. **Expected Returns**: Still calculated for display but NOT used in optimization
2. **Cash Modeling**: Excluded from optimization to focus on risky asset diversification
3. **Market Regimes**: Correlation-based optimization may not adapt to fundamental shifts
4. **Optimization Method**: Uses SLSQP, may find local optima in complex cases

---

## [1.0.0] - 2025-01-08 (DEPRECATED)

### Initial Kelly Criterion Implementation (Superseded)

This version implemented Kelly Criterion optimization but was found to produce unstable, highly concentrated portfolios due to the noisy nature of expected returns estimation.

**Issues that led to v2.0.0:**
- Frequent 100% single-asset allocations
- Extreme sensitivity to expected returns estimates
- Poor diversification despite sophisticated risk modeling
- High portfolio turnover from parameter changes

---

## Future Roadmap

### Potential Enhancements
- **Black-Litterman**: Incorporate market views with uncertainty
- **Risk Parity**: Alternative to minimum variance for different risk budgets  
- **Factor Models**: Multi-factor risk decomposition
- **Regime Detection**: Adapt correlation windows to market conditions

### Technical Improvements
- **Database**: Persistent storage for historical analysis
- **Performance**: Caching for faster UI updates
- **Testing**: Expanded test suite for edge cases
- **Documentation**: Video tutorials and case studies

---

*The minimum variance approach has proven much more stable and practical than the original Kelly Criterion implementation, providing consistent diversification benefits without the instability of expected returns optimization.*