# Claude Code Project Instructions

This file contains instructions for Claude on how to work with this Portfolio Optimizer project.

## Project Overview

This is a Kelly Criterion portfolio optimizer built with:
- **Python 3.8+** with NumPy, Pandas, SciPy stack
- **Streamlit** for the interactive dashboard
- **Plotly** for visualizations  
- **yfinance** for market data
- **CVXPY** for optimization (with SciPy fallback)

## Project Structure

```
Port_Optimizer/
├── data/loader.py          # Data generation & Yahoo Finance
├── features/estimators.py  # μ, σ, ρ estimation (EWMA/rolling)
├── opt/optimizer.py        # Kelly optimization (quadratic + MC)
├── backtest/engine.py      # Portfolio simulation with costs
├── ui/app.py              # Streamlit dashboard
├── utils/metrics.py        # Performance metrics
├── config.py              # Default parameters
├── test_phase1.py         # Simulated data validation
└── test_streamlit.py      # Market data validation
```

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
   - **Modularity**: Each module has clear responsibilities
   - **Robustness**: Graceful handling of data issues, API failures
   - **Flexibility**: User-configurable parameters throughout
   - **Performance**: Efficient pandas/numpy operations

### Critical Files:

- **config.py**: All default parameters - change here affects entire system
- **ui/app.py**: Main user interface - any UI changes go here
- **opt/optimizer.py**: Core Kelly optimization logic
- **data/loader.py**: Data handling - both simulated and live market data

### Common Tasks:

**Adding new assets:**
1. Update `ASSETS` in config.py
2. Update ticker mapping in `data/loader.py`
3. Test with both simulated and live data

**Adding new optimization methods:**
1. Add method to `opt/optimizer.py`
2. Update UI controls in `ui/app.py`
3. Add validation tests

**Adding new metrics:**
1. Implement in `utils/metrics.py`
2. Add to backtest analysis in `backtest/engine.py`
3. Display in dashboard `ui/app.py`

### Dependencies Management:

- **Required**: All packages in requirements.txt must work
- **Optional solvers**: CVXPY solvers (ECOS, etc.) - system falls back to SciPy if missing
- **Development**: pytest, black, mypy for code quality (not in requirements.txt)

### Error Handling:

This system has extensive fallbacks:
- **Correlation matrices**: Falls back to reasonable defaults if EWMA fails
- **Optimization**: Falls back from CVXPY to SciPy if needed  
- **Market data**: Handles Yahoo Finance API changes gracefully
- **Numerical issues**: Positive definite enforcement, condition number monitoring

### Testing Philosophy:

- **Phase 1**: Validates all core functionality with simulated data
- **Phase 2**: Validates market data integration
- **Unit-level**: Each estimator and optimizer validated independently  
- **Integration**: Full end-to-end backtesting validation

### Performance Notes:

- **Optimization**: Quadratic method ~10ms, Monte Carlo ~100ms
- **Memory**: Efficient for multi-year daily backtests
- **UI**: Streamlit caching used where appropriate
- **Data**: Pandas operations optimized for time series

## Maintenance Tasks

### Regular:
- Test market data connectivity (Yahoo Finance API changes)
- Validate optimization results remain sensible
- Check for package updates and compatibility

### When adding features:
- Update both Phase 1 and Phase 2 tests
- Add comprehensive error handling
- Document in module docstrings
- Update UI if user-facing

### Before releases:
- Run full test suite
- Update CHANGELOG.md
- Verify all documentation is current
- Test with fresh virtual environment

## Known Issues & Workarounds

1. **EWMA correlation matrices** can produce NaN with sparse data
   - **Solution**: Fallback correlation matrices implemented
   
2. **CVXPY solvers** may not be available on all systems
   - **Solution**: Automatic fallback to SciPy SLSQP
   
3. **Yahoo Finance rate limiting** possible with frequent requests
   - **Solution**: Reasonable delays and error retry logic
   
4. **Memory usage** with large backtests
   - **Solution**: Use weekly/monthly rebalancing for long periods

## Development Environment

Recommended setup:
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

For better performance during development:
```bash
pip install watchdog  # Faster Streamlit reloading
```

## Production Deployment

The system is designed to be deployment-ready with:
- Comprehensive error handling
- Input validation throughout
- Graceful degradation of functionality
- Clear user feedback on issues
- No hardcoded paths or credentials