"""
Simplified configuration following Breaking the Market philosophy.

Core approach:
- 4 assets: SPY, TLT, GLD, Cash
- Long-only, no leverage, daily rebalancing
- Rolling windows: μ slow (252d), σ medium (60d), ρ fast (15d)
- Pure quadratic Kelly optimization
"""

# Asset universe
ASSETS = ["SPY", "TLT", "GLD", "Cash"]
N_ASSETS = len(ASSETS)

# Breaking the Market windows - fixed, no user choice
MU_WINDOW = 252     # Mean: slow (1 year) - means are uncertain
SIGMA_WINDOW = 60   # Volatility: medium (3 months) 
RHO_WINDOW = 30     # Correlation: fast but stable (6 weeks) - balance speed vs stability

# Risk-free rate
RISK_FREE_RATE = 0.0525  # 5.25% annualized for Cash

# Market constants
TRADING_DAYS_PER_YEAR = 252

# Rebalancing
REBALANCE_FREQUENCY = "daily"  # Core philosophy: frequent rebalancing

# UI defaults
DEFAULT_DATE_RANGE_YEARS = 2
MIN_WEIGHT = 1e-6  # Numerical tolerance for zero weights

# Display precision
DISPLAY_DECIMALS = 3