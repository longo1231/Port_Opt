"""
Simplified parameter estimation following Breaking the Market methodology.

Uses fixed rolling windows:
- μ (returns): 252 days (slow, uncertain)
- σ (volatility): 60 days (medium) 
- ρ (correlation): 15 days (fast, adapts to regime changes)
"""

import numpy as np
import pandas as pd
from config_simple import MU_WINDOW, SIGMA_WINDOW, RHO_WINDOW, TRADING_DAYS_PER_YEAR, ASSETS


def estimate_parameters(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate portfolio parameters using Breaking the Market approach.
    
    Uses the full date range provided, applying Breaking the Market philosophy:
    - μ: Uses full period (slow, uncertain)
    - σ: Uses shorter window or full period if shorter (medium adaptation)
    - ρ: Uses shortest window or full period if shorter (fast adaptation)
    
    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns with assets as columns
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (expected_returns, covariance_matrix) - both annualized
    """
    n_days = len(returns)
    
    if n_days < 20:  # Minimum reasonable sample
        raise ValueError(f"Need at least 20 days of data, got {n_days}")
    
    # Expected returns - use appropriate window (Breaking the Market: μ slow/uncertain)
    mu_window = min(MU_WINDOW, n_days)
    mu_daily = returns.tail(mu_window).mean().values
    expected_returns = mu_daily * TRADING_DAYS_PER_YEAR
    
    # Volatilities - use appropriate window (Breaking the Market: σ medium adaptation)
    vol_window = min(SIGMA_WINDOW, n_days)
    sigma_daily = returns.tail(vol_window).std().values
    volatilities = sigma_daily * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    # Handle Cash (zero volatility)
    if 'Cash' in returns.columns:
        cash_idx = list(returns.columns).index('Cash')
        volatilities[cash_idx] = 1e-8  # Tiny non-zero for numerical stability
    
    # Correlations - use fastest adaptation (Breaking the Market: ρ fast)
    corr_window = min(RHO_WINDOW, n_days)
    corr_data = returns.tail(corr_window)
    correlation_matrix = corr_data.corr().values
    
    # Handle Cash correlations (set to zero with other assets, 1.0 with itself)
    if 'Cash' in returns.columns:
        cash_idx = list(returns.columns).index('Cash')
        correlation_matrix[cash_idx, :] = 0.0
        correlation_matrix[:, cash_idx] = 0.0
        correlation_matrix[cash_idx, cash_idx] = 1.0
    
    # Handle any remaining NaN correlations
    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
    np.fill_diagonal(correlation_matrix, 1.0)
    
    # Build covariance matrix: Σ = D·ρ·D where D = diag(σ)
    covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix
    
    return expected_returns, covariance_matrix


def get_estimation_info(n_days: int = None) -> dict:
    """Return information about the estimation windows used."""
    if n_days is None:
        return {
            'mu_window': 'full period',
            'sigma_window': f'min({SIGMA_WINDOW}, period_length)',
            'rho_window': f'min({RHO_WINDOW}, period_length)',
            'philosophy': 'Breaking the Market: μ slow, σ medium, ρ fast'
        }
    else:
        return {
            'mu_window': f'full period ({n_days} days)',
            'sigma_window': f'{min(SIGMA_WINDOW, n_days)} days',
            'rho_window': f'{min(RHO_WINDOW, n_days)} days',
            'philosophy': 'Breaking the Market: μ slow, σ medium, ρ fast'
        }