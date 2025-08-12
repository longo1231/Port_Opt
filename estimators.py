"""
Parameter estimation for Minimum Variance portfolio optimization.

Only estimates covariance matrix since minimum variance optimization
doesn't require expected returns.
"""

import numpy as np
import pandas as pd
from config import SIGMA_WINDOW, RHO_WINDOW, TRADING_DAYS_PER_YEAR, ASSETS


def estimate_covariance_matrix(returns: pd.DataFrame, 
                             sigma_window: int = None, 
                             rho_window: int = None) -> np.ndarray:
    """
    Estimate covariance matrix for minimum variance optimization.
    
    Uses Breaking the Market windowing philosophy:
    - σ (volatility): Medium adaptation (default 60 days)  
    - ρ (correlation): Fast adaptation (default 30 days)
    
    Parameters
    ----------
    returns : pd.DataFrame
        Daily simple returns with assets as columns
    sigma_window : int, optional
        Window for volatility estimation (uses config default if None)
    rho_window : int, optional
        Window for correlation estimation (uses config default if None)
        
    Returns
    -------
    np.ndarray
        Annualized covariance matrix
    """
    n_days = len(returns)
    
    if n_days < 20:  # Minimum reasonable sample
        raise ValueError(f"Need at least 20 days of data, got {n_days}")
    
    # Use provided windows or defaults from config
    if sigma_window is None:
        sigma_window = SIGMA_WINDOW
    if rho_window is None:
        rho_window = RHO_WINDOW
    
    # Volatilities - use appropriate window (Breaking the Market: σ medium adaptation)
    vol_window = min(sigma_window, n_days)
    sigma_daily = returns.tail(vol_window).std().values
    volatilities = sigma_daily * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    # Handle Cash (zero volatility)
    if 'Cash' in returns.columns:
        cash_idx = list(returns.columns).index('Cash')
        volatilities[cash_idx] = 1e-8  # Tiny non-zero for numerical stability
    
    # Correlations - use fastest adaptation (Breaking the Market: ρ fast)
    corr_window = min(rho_window, n_days)
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
    
    return covariance_matrix


def estimate_expected_returns(returns: pd.DataFrame, mu_window: int = None) -> np.ndarray:
    """
    Estimate expected returns for display purposes.
    
    Note: These are NOT used in minimum variance optimization,
    but are useful for analysis and validation.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Daily simple returns with assets as columns
    mu_window : int, optional
        Window for return estimation (uses full period if None)
        
    Returns
    -------
    np.ndarray
        Annualized expected returns
    """
    n_days = len(returns)
    
    if mu_window is None:
        # Use full period for expected returns display
        mu_data = returns
    else:
        # Use specified window
        window = min(mu_window, n_days)
        mu_data = returns.tail(window)
    
    # Calculate annualized returns with proper compounding
    daily_returns = mu_data.mean().values
    
    # Proper compounding: (1 + daily_return)^252 - 1
    # This gives the true expected annual return for simple returns
    expected_returns = np.power(1 + daily_returns, TRADING_DAYS_PER_YEAR) - 1
    
    return expected_returns


def get_estimation_info(n_days: int = None, sigma_window: int = None, rho_window: int = None) -> dict:
    """Return information about the estimation windows used."""
    if sigma_window is None:
        sigma_window = SIGMA_WINDOW
    if rho_window is None:
        rho_window = RHO_WINDOW
        
    if n_days is None:
        return {
            'sigma_window': f'min({sigma_window}, period_length)',
            'rho_window': f'min({rho_window}, period_length)',
            'philosophy': 'Breaking the Market: σ medium, ρ fast',
            'approach': 'Minimum Variance (no expected returns needed)'
        }
    else:
        return {
            'sigma_window': f'{min(sigma_window, n_days)} days',
            'rho_window': f'{min(rho_window, n_days)} days',
            'sigma_actual': min(sigma_window, n_days),
            'rho_actual': min(rho_window, n_days),
            'philosophy': 'Breaking the Market: σ medium, ρ fast',
            'approach': 'Minimum Variance (no expected returns needed)'
        }