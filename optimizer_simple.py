"""
Simplified portfolio optimizer following Breaking the Market approach.

Pure quadratic Kelly optimization:
- Maximize: w'μ - 0.5 w'Σw  
- Subject to: Σw = 1, w ≥ 0
- No leverage, no Kelly fractions, no position limits
"""

import numpy as np
from scipy.optimize import minimize
from config_simple import MIN_WEIGHT


def optimize_portfolio(expected_returns: np.ndarray, covariance_matrix: np.ndarray) -> dict:
    """
    Optimize portfolio using pure quadratic Kelly criterion.
    
    Solves: maximize w'μ - 0.5 w'Σw
    Subject to: Σw = 1, w ≥ 0
    
    Parameters
    ----------
    expected_returns : np.ndarray
        Annualized expected returns
    covariance_matrix : np.ndarray  
        Annualized covariance matrix
        
    Returns
    -------
    dict
        Optimization results with weights and portfolio statistics
    """
    n_assets = len(expected_returns)
    
    # Objective function: minimize -(w'μ - 0.5 w'Σw)
    def objective(weights):
        portfolio_return = weights @ expected_returns
        portfolio_risk = 0.5 * weights @ covariance_matrix @ weights
        return -(portfolio_return - portfolio_risk)
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Budget: Σw = 1
    ]
    
    # Bounds: w ≥ 0 (long-only)
    bounds = [(0, 1) for _ in range(n_assets)]
    
    # Initial guess
    x0 = np.ones(n_assets) / n_assets
    
    # Optimize
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9}
    )
    
    if not result.success:
        return {
            'success': False,
            'error': f"Optimization failed: {result.message}"
        }
    
    # Clean up tiny weights
    weights = np.maximum(result.x, 0)
    weights[weights < MIN_WEIGHT] = 0
    weights = weights / weights.sum()  # Renormalize
    
    # Portfolio statistics
    portfolio_return = weights @ expected_returns
    portfolio_variance = weights @ covariance_matrix @ weights  
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    # Sharpe ratio (excess return over volatility)
    sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
    
    return {
        'success': True,
        'weights': weights,
        'expected_return': portfolio_return,
        'volatility': portfolio_volatility,
        'variance': portfolio_variance,
        'sharpe_ratio': sharpe_ratio,
        'objective_value': -result.fun  # Convert back to maximization value
    }