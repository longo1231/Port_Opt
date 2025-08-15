"""
Configuration for Minimum Variance Portfolio Optimizer.

Simple and focused configuration for the minimum variance approach.
"""

import numpy as np

# Portfolio Templates
PORTFOLIO_TEMPLATES = {
    'Current': ['SPY', 'TLT', 'GLD', 'Cash'],  # Original diversified portfolio
    'MAG7': ['AMZN', 'AAPL', 'MSFT', 'GOOG', 'META', 'TSLA', 'NVDA', 'Cash']  # Magnificent 7 tech stocks
}

# Default assets (for backward compatibility)
ASSETS = PORTFOLIO_TEMPLATES['Current']

# Time series parameters
TRADING_DAYS_PER_YEAR = 252

# Estimation windows (Breaking the Market philosophy)
MU_WINDOW = 252    # Not used in min variance (no expected returns needed)
SIGMA_WINDOW = 60  # Medium adaptation for volatility
RHO_WINDOW = 30    # Fast adaptation for correlations

# Risk-free rate (for Cash)
RISK_FREE_RATE = 0.053  # 5.3% annual

# Portfolio constraints
MIN_WEIGHT = 1e-4  # Minimum position size (0.01%)

# UI defaults
DEFAULT_DATE_RANGE_YEARS = 3

# Minimum variance specific settings
EXCLUDE_CASH_FROM_OPTIMIZATION = True  # Focus on risky assets only