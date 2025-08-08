"""
Configuration file for Portfolio Optimizer.

Contains default parameters for the portfolio optimization system including
estimation windows, risk parameters, costs, and optimization settings.
"""

from typing import List
import numpy as np

# Asset configuration
ASSETS = ["SPX", "TLT", "GLD", "Cash"]
N_ASSETS = len(ASSETS)

# Estimation parameters
DEFAULT_LOOKBACK_DAYS = 252  # 1 year for mu/sigma estimation
DEFAULT_MU_SIGMA_HALFLIFE = 30  # EWMA half-life for returns/volatility
DEFAULT_CORR_HALFLIFE = 15  # Faster adaptation for correlations

# Risk parameters
DEFAULT_RISK_FREE_RATE = 0.0525  # 5.25% annualized
DEFAULT_SHRINKAGE = 0.2  # Covariance shrinkage towards diagonal

# Transaction costs
DEFAULT_TRANSACTION_COST_BPS = 2.0  # 2 bps per side
DEFAULT_TURNOVER_PENALTY = 0.001  # Lambda for turnover penalty

# Optimization settings
DEFAULT_OPTIMIZER_METHOD = "quadratic"  # "quadratic" or "monte_carlo"
DEFAULT_MC_SIMULATIONS = 10000
DEFAULT_REBALANCE_FREQ = "daily"  # "daily", "weekly", "monthly"

# Simulation parameters (Phase 1)
DEFAULT_SIM_MU = [0.08, 0.03, 0.05]  # Annualized expected returns for SPX, TLT, GLD
DEFAULT_SIM_SIGMA = [0.16, 0.07, 0.15]  # Annualized volatilities
DEFAULT_SIM_CORR = 0.2  # Pairwise correlation between risky assets

# UI settings
DEFAULT_DATE_RANGE_YEARS = 2
TRADING_DAYS_PER_YEAR = 252

# Numerical tolerances
MIN_WEIGHT = 1e-6  # Minimum portfolio weight
CONDITION_NUMBER_THRESHOLD = 1e12  # Warning threshold for covariance matrix