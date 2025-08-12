"""
Minimum Variance Portfolio Optimizer.

Simple and stable approach that eliminates the problematic expected returns
input by focusing solely on risk reduction through diversification.

Objective: minimize w'Σw
Subject to: Σw = 1, w ≥ 0

This approach naturally creates diversified portfolios without corner solutions.
"""

import numpy as np
from scipy.optimize import minimize
from config import MIN_WEIGHT


def optimize_min_variance(covariance_matrix: np.ndarray, exclude_cash: bool = True) -> dict:
    """
    Find the portfolio with minimum possible variance.
    
    Solves: minimize w'Σw
    Subject to: Σw = 1, w ≥ 0
    
    By removing expected returns from the objective, this eliminates
    the main source of instability that causes corner solutions.
    
    Parameters
    ----------
    covariance_matrix : np.ndarray
        Annualized covariance matrix
    exclude_cash : bool, default True
        If True, excludes Cash from optimization (assumes Cash is last asset)
        
    Returns
    -------
    dict
        Optimization results with weights and portfolio statistics
    """
    n_assets = covariance_matrix.shape[0]
    
    if exclude_cash and n_assets > 1:
        # Optimize only risky assets (exclude last asset assumed to be Cash)
        risky_cov = covariance_matrix[:-1, :-1]
        n_risky = risky_cov.shape[0]
        
        # Objective function: minimize portfolio variance for risky assets
        def objective(weights_risky):
            return weights_risky @ risky_cov @ weights_risky
        
        # Constraints: risky weights sum to 1 (no cash allocation)
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        # Bounds: long-only constraint w ≥ 0
        bounds = [(0, 1) for _ in range(n_risky)]
        
        # Initial guess: equal weighting among risky assets
        x0 = np.ones(n_risky) / n_risky
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        
        if not result.success:
            return {
                'success': False,
                'error': f"Optimization failed: {result.message}",
                'weights': np.zeros(n_assets)
            }
        
        # Clean up tiny weights for risky assets
        risky_weights = np.maximum(result.x, 0)
        risky_weights[risky_weights < MIN_WEIGHT] = 0
        risky_weights = risky_weights / risky_weights.sum()  # Renormalize
        
        # Create full weights vector (including cash = 0)
        full_weights = np.zeros(n_assets)
        full_weights[:-1] = risky_weights
        
    else:
        # Standard optimization including all assets
        def objective(weights):
            return weights @ covariance_matrix @ weights
        
        # Constraints: budget constraint Σw = 1
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        # Bounds: long-only constraint w ≥ 0
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess: equal weighting
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        
        if not result.success:
            return {
                'success': False,
                'error': f"Optimization failed: {result.message}",
                'weights': np.zeros(n_assets)
            }
        
        # Clean up tiny weights
        full_weights = np.maximum(result.x, 0)
        full_weights[full_weights < MIN_WEIGHT] = 0
        full_weights = full_weights / full_weights.sum()  # Renormalize
    
    # Calculate portfolio statistics
    portfolio_variance = full_weights @ covariance_matrix @ full_weights
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    # Calculate diversification ratio
    # DR = weighted average volatility / portfolio volatility
    individual_vols = np.sqrt(np.diag(covariance_matrix))
    weighted_avg_vol = full_weights @ individual_vols
    diversification_ratio = weighted_avg_vol / portfolio_volatility if portfolio_volatility > 0 else 1.0
    
    # Calculate effective number of assets (inverse Herfindahl index)
    effective_assets = 1.0 / np.sum(full_weights ** 2) if np.sum(full_weights ** 2) > 0 else 1.0
    
    return {
        'success': True,
        'method': 'Minimum Variance',
        'weights': full_weights,
        'portfolio_variance': portfolio_variance,
        'portfolio_volatility': portfolio_volatility,
        'diversification_ratio': diversification_ratio,
        'effective_n_assets': effective_assets,
        'objective_value': portfolio_variance,  # The minimized objective
        'exclude_cash': exclude_cash,
        'optimization_result': result
    }


def analyze_min_variance_portfolio(weights: np.ndarray, 
                                 covariance_matrix: np.ndarray,
                                 asset_names: list = None) -> dict:
    """
    Analyze a minimum variance portfolio in detail.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    covariance_matrix : np.ndarray
        Asset covariance matrix
    asset_names : list, optional
        Names of assets
        
    Returns
    -------
    dict
        Detailed portfolio analysis
    """
    if asset_names is None:
        asset_names = [f"Asset_{i}" for i in range(len(weights))]
    
    # Portfolio statistics
    portfolio_var = weights @ covariance_matrix @ weights
    portfolio_vol = np.sqrt(portfolio_var)
    
    # Individual asset statistics
    individual_vols = np.sqrt(np.diag(covariance_matrix))
    
    # Risk contribution analysis
    marginal_risk = covariance_matrix @ weights
    risk_contributions = weights * marginal_risk
    risk_contrib_pct = risk_contributions / portfolio_var if portfolio_var > 0 else np.zeros_like(weights)
    
    # Correlation with portfolio
    portfolio_correlations = []
    for i in range(len(weights)):
        if individual_vols[i] > 0 and portfolio_vol > 0:
            port_corr = marginal_risk[i] / (individual_vols[i] * portfolio_vol)
            portfolio_correlations.append(port_corr)
        else:
            portfolio_correlations.append(0.0)
    
    # Create detailed analysis
    asset_analysis = []
    for i, asset in enumerate(asset_names):
        asset_analysis.append({
            'asset': asset,
            'weight': weights[i],
            'individual_vol': individual_vols[i],
            'risk_contribution': risk_contributions[i],
            'risk_contrib_pct': risk_contrib_pct[i],
            'correlation_with_portfolio': portfolio_correlations[i]
        })
    
    return {
        'portfolio_volatility': portfolio_vol,
        'portfolio_variance': portfolio_var,
        'diversification_ratio': (weights @ individual_vols) / portfolio_vol if portfolio_vol > 0 else 1.0,
        'effective_n_assets': 1.0 / np.sum(weights ** 2),
        'max_weight': np.max(weights),
        'min_positive_weight': np.min(weights[weights > 0]) if np.any(weights > 0) else 0.0,
        'n_nonzero_assets': np.sum(weights > MIN_WEIGHT),
        'asset_analysis': asset_analysis
    }