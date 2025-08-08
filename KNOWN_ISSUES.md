# Known Issues and Future Fixes

This document tracks identified problems and planned improvements for the Portfolio Optimizer.

## 🎯 **EXECUTIVE SUMMARY - TOP 2 PRIORITIES**

### **1. 🔥 CORRELATION MATRIX FAILURES** 
- **Problem**: EWMA correlation estimation constantly fails → dummy fallback matrices
- **Impact**: Portfolio allocation based on fake 0.1 correlations, not real market relationships  
- **Status**: Broken and needs immediate fix with robust shrinkage methods

### **2. 🔥 NO DATA PERSISTENCE**
- **Problem**: All optimization results are ephemeral, no audit trail or performance tracking
- **Impact**: Can't validate model performance, no learning from past decisions
- **Status**: Solution created (`data/persistence.py`) but needs UI integration

**These two issues prevent professional/production use and must be fixed first.**

---

## 🚨 CRITICAL ISSUES - TOP PRIORITIES

### ⭐ **PRIORITY #1: Correlation Matrix Estimation Failures** ⭐

**Problem:**
- EWMA correlation estimation frequently produces NaN values
- System falls back to dummy correlation matrices (0.1 off-diagonal)
- Users are unaware when "sophisticated" correlation adaptation is actually using fallbacks
- Root cause: Insufficient handling of sparse data in EWMA calculations

**Current Workaround:**
```python
# In features/estimators.py:225
if np.isnan(corr_matrix).any():
    warnings.warn("NaN values found in correlation matrix, using fallback")
    fallback_matrix = np.full((n, n), 0.1)
    np.fill_diagonal(fallback_matrix, 1.0)
    return fallback_matrix
```

**Impact:**
- **HIGH** - Correlation structure drives portfolio allocation
- Fallback matrices may not reflect actual market relationships
- Backtests are misleading when using artificial correlations

**Planned Fix:**
1. **Replace EWMA correlation** with robust shrinkage estimators (Ledoit-Wolf)
2. **Add minimum sample size requirements** before attempting EWMA
3. **Implement correlation forecasting models** (DCC-GARCH, factor models)
4. **Add UI indicators** when fallbacks are used
5. **Validate correlation estimates** with statistical tests

```python
# Proposed solution
from sklearn.covariance import LedoitWolf

def estimate_correlation_robust(returns, method='ledoit_wolf'):
    if method == 'ledoit_wolf':
        lw = LedoitWolf()
        cov_matrix = lw.fit(returns).covariance_
        # Extract correlation from covariance
        std_dev = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(std_dev, std_dev)
        return corr_matrix, lw.shrinkage_
```

---

### ⭐ **PRIORITY #2: Data Persistence and Historical Tracking** ⭐

**Problem:**
- **Zero data persistence** - all optimization results are ephemeral 
- **No audit trail** - can't track what parameters were used when
- **No performance monitoring** - can't validate if model predictions were accurate
- **No historical analysis** - can't see how different parameters performed over time

**Current State:**
- Browser refresh = lose all results
- No database storage of optimization decisions
- Can't compare strategies or track model degradation
- No record of which parameters worked in different market conditions

**Impact:**
- **CRITICAL** - Can't validate model performance in production
- No way to learn from past decisions
- Can't detect when model starts failing
- Makes professional/institutional use impossible

**Solution Created:**
- `data/persistence.py` - SQLite database for all results
- `ui/portfolio_history.py` - UI for viewing historical analysis  
- Automatic saving of every optimization with parameters
- Performance tracking and backtest result storage

**Implementation Status:**
- [ ] Database schema created
- [ ] UI components built  
- [ ] Integration with main dashboard (TODO)
- [ ] Automatic result saving (TODO)

---

### 3. **Transaction Cost Model Oversimplification**

**Problem:**
- Fixed 2 bps cost regardless of market conditions
- No market impact modeling (price moves against large orders)
- No bid-ask spread modeling
- No liquidity-dependent costs

**Current Implementation:**
```python
daily_cost = (transaction_cost_bps / 10000) * daily_turnover
```

**Reality:**
- TLT costs can spike to 20+ bps during stress periods
- Large rebalances have higher costs than small ones
- Costs vary by time of day, market volatility, asset liquidity

**Planned Fix:**
1. **Market impact model**: Cost = base_cost + impact_cost × sqrt(turnover)
2. **Regime-dependent costs**: Higher costs during high-volatility periods
3. **Asset-specific costs**: Different cost structures for SPY vs TLT vs GLD
4. **Time-varying bid-ask spreads** from market data

---

### 4. **Kelly Criterion Implementation Issues**

**Problem:**
- Full Kelly optimization without fractional sizing
- No leverage constraints (forces sum(w) = 1)
- Missing parameter uncertainty in Kelly formula

**Current Issues:**
- Full Kelly leads to extreme positions and guaranteed ruin
- Industry standard is 0.1x to 0.25x Kelly for safety
- No accounting for estimation error in μ and Σ

**Planned Fix:**
1. **Fractional Kelly**: `optimal_weights *= kelly_fraction` (default 0.25)
2. **Parameter uncertainty**: Bayesian approach to μ/Σ estimation  
3. **Dynamic sizing**: Reduce position size during high uncertainty periods
4. **Leverage option**: Allow leveraged positions with risk controls

---

## ⚠️ Medium Priority Issues

### 5. **Risk Model Limitations**

**Problems:**
- Only 4 assets (underdiversified)
- No factor model decomposition  
- Normal distribution assumption (no tail risk)
- No regime detection

**Planned Improvements:**
1. **Expand universe**: Add international equities, REITs, commodities
2. **Factor models**: Fama-French factors, principal components
3. **Heavy-tailed distributions**: Student-t, skewed-t distributions
4. **Regime switching models**: Hidden Markov models for parameter changes

### 6. **Backtesting Biases**

**Problems:**
- Survivorship bias (delisted assets disappear)
- Look-ahead bias (using future data for parameter selection)
- No benchmark comparison
- Transaction costs too optimistic

**Planned Improvements:**
1. **Point-in-time data**: Use only data available at decision time
2. **Walk-forward analysis**: Re-estimate parameters out-of-sample
3. **Multiple benchmarks**: 60/40, equal weight, momentum strategies
4. **Realistic execution**: Include slippage, partial fills, timing delays

---

## 🔧 Technical Debt

### 7. **Error Handling and Logging**

**Problems:**
- Silent fallbacks hide system failures
- Warning spam in console (50+ warnings during backtest)
- No structured logging for production use
- Users unaware when degraded functionality is used

**Planned Improvements:**
1. **Structured logging**: Replace warnings with proper log levels
2. **User notifications**: Dashboard alerts when fallbacks are used
3. **Health monitoring**: System status indicators in UI
4. **Graceful degradation**: Clear communication about reduced functionality

### 8. **Performance Optimization**

**Problems:**
- Pandas operations for numerical computing (slow)
- No caching of intermediate results
- UI blocks during optimization
- Memory usage grows over time

**Planned Improvements:**
1. **NumPy acceleration**: Replace pandas with NumPy for numerical operations
2. **Async optimization**: Non-blocking UI updates
3. **Result caching**: Cache parameter estimates and optimization results
4. **Memory management**: Periodic cleanup of old results

---

## 📊 Data and Infrastructure Issues

### 9. **Yahoo Finance Dependency**

**Problems:**
- Rate limiting and API instability
- Data quality issues (missing adjustments, corporate actions)
- Single data source risk
- No real-time data

**Planned Improvements:**
1. **Multiple data sources**: Alpha Vantage, IEX Cloud, Quandl backup
2. **Data validation**: Cross-check prices between sources  
3. **Corporate action handling**: Dividend adjustments, stock splits
4. **Real-time feeds**: WebSocket connections for live data

### 10. **[RESOLVED] No Data Persistence**

**Problems:**
- All results are ephemeral (lost on browser refresh)
- No audit trail of decisions
- Can't track model performance over time
- No historical parameter analysis

**Solution Created:**
- ✅ `data/persistence.py` module created 
- ✅ SQLite database for local storage
- ✅ `ui/portfolio_history.py` UI components built
- 🔄 Integration with main dashboard (in progress)

---

## 🚀 Future Enhancements

### 11. **Advanced Portfolio Techniques**

**Planned Features:**
1. **Black-Litterman model**: Incorporate market views
2. **Risk parity**: Equal risk contribution weighting
3. **Minimum variance**: Pure risk-based allocation
4. **Factor investing**: Style tilts (value, momentum, quality)

### 12. **Production Readiness**

**Requirements for Live Trading:**
1. **Real-time execution**: Order management system integration
2. **Risk management**: Position limits, drawdown controls, stress tests  
3. **Monitoring**: Alerting, performance tracking, model validation
4. **Compliance**: Audit trails, regulatory reporting, documentation

---

## 📈 Implementation Priority

### Phase 1 (CRITICAL - Next 1-2 weeks) 🔥
- [ ] **PRIORITY #1**: Fix correlation matrix estimation (replace EWMA fallbacks)
- [ ] **PRIORITY #2**: Integrate data persistence layer (complete UI integration)
- [ ] Implement fractional Kelly sizing
- [ ] Improve error handling and user feedback

### Phase 2 (Medium Priority - Next month)
- [ ] Enhanced transaction cost model
- [ ] Factor model implementation
- [ ] Backtesting improvements
- [ ] Performance optimization

### Phase 3 (Long-term - Next quarter)  
- [ ] Additional asset classes
- [ ] Advanced portfolio methods
- [ ] Production infrastructure
- [ ] Real-time data integration

---

*This document is updated regularly as issues are discovered and fixed. Last updated: 2025-01-08*