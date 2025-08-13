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


def optimize_min_variance_risky_only(risky_covariance_matrix):
    """
    Optimize minimum variance portfolio using only risky assets.
    
    This function is used for leverage calculations where we need the optimal
    allocation among risky assets only (weights sum to 1.0), before applying
    leverage and determining cash position.
    
    Parameters
    ----------
    risky_covariance_matrix : np.ndarray or pd.DataFrame
        Covariance matrix for risky assets only (e.g., 3x3 for SPY, TLT, GLD)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'success': bool, whether optimization succeeded
        - 'weights': np.ndarray, optimal weights for risky assets (sum to 1.0)
        - 'portfolio_volatility': float, annualized volatility of risky portfolio
        - 'error': str, error message if optimization failed
    """
    try:
        # Convert to numpy if DataFrame
        if hasattr(risky_covariance_matrix, 'values'):
            cov_matrix = risky_covariance_matrix.values
        else:
            cov_matrix = risky_covariance_matrix
            
        n_assets = cov_matrix.shape[0]
        
        # Objective: minimize w'Σw (portfolio variance)
        def objective(weights):
            return weights.T @ cov_matrix @ weights
            
        # Constraint: weights sum to 1 (fully invested in risky assets)
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        
        # Bounds: long-only positions (0 ≤ w ≤ 1)
        bounds = [(0.0, 1.0) for _ in range(n_assets)]
        
        # Initial guess: equal weights
        initial_weights = np.array([1.0/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9}
        )
        
        if result.success:
            weights = result.x
            portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
            
            return {
                'success': True,
                'weights': weights,
                'portfolio_volatility': portfolio_volatility,
                'error': None
            }
        else:
            return {
                'success': False,
                'weights': initial_weights,
                'portfolio_volatility': 0.0,
                'error': f"Optimization failed: {result.message}"
            }
            
    except Exception as e:
        n_assets = risky_covariance_matrix.shape[0]
        return {
            'success': False,
            'weights': np.array([1.0/n_assets] * n_assets),
            'portfolio_volatility': 0.0,
            'error': f"Optimization error: {str(e)}"
        }


def calculate_leveraged_portfolio(risky_covariance_matrix, target_volatility, max_leverage=3.0):
    """
    Calculate leveraged portfolio to achieve target volatility.
    
    This implements the 6-step process:
    1. Optimize minimum variance on risky assets only
    2. Calculate MVP volatility 
    3. Determine required leverage
    4. Cap leverage at maximum
    5. Calculate final weights (risky assets scaled, cash = 1 - leverage)
    6. Return results with leverage info
    
    Parameters
    ----------
    risky_covariance_matrix : np.ndarray
        Covariance matrix for risky assets only
    target_volatility : float
        Desired portfolio volatility (e.g., 0.10 for 10%)
    max_leverage : float, default 3.0
        Maximum allowed leverage ratio
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'success': bool, whether calculation succeeded
        - 'final_weights': np.ndarray, final portfolio weights [SPY, TLT, GLD, Cash]
        - 'calculated_leverage': float, actual leverage applied
        - 'mvp_volatility': float, volatility of minimum variance risky portfolio
        - 'final_volatility': float, final portfolio volatility (should match target)
        - 'risky_weights': np.ndarray, optimal weights for risky assets only
        - 'error': str, error message if failed
    """
    try:
        # Step 1 & 2: Optimize risky assets and get MVP volatility
        mvp_result = optimize_min_variance_risky_only(risky_covariance_matrix)
        
        if not mvp_result['success']:
            return {
                'success': False,
                'final_weights': np.array([0.33, 0.33, 0.34, 0.0]),  # Equal weight fallback
                'calculated_leverage': 1.0,
                'mvp_volatility': 0.0,
                'final_volatility': 0.0,
                'risky_weights': np.array([0.33, 0.33, 0.34]),
                'error': f"MVP optimization failed: {mvp_result['error']}"
            }
        
        risky_weights = mvp_result['weights']
        mvp_volatility = mvp_result['portfolio_volatility']
        
        # Step 3 & 4: Calculate leverage with cap
        if mvp_volatility <= 1e-8:  # Avoid division by zero
            calculated_leverage = 1.0
        else:
            calculated_leverage = min(target_volatility / mvp_volatility, max_leverage)
        
        # Step 5: Calculate final weights
        final_risky_weights = risky_weights * calculated_leverage
        final_cash_weight = 1.0 - calculated_leverage
        
        # Combine into full portfolio weights [SPY, TLT, GLD, Cash]
        final_weights = np.append(final_risky_weights, final_cash_weight)
        
        # Calculate actual final portfolio volatility
        # For leveraged portfolio: vol = leverage * mvp_vol
        final_volatility = calculated_leverage * mvp_volatility
        
        return {
            'success': True,
            'final_weights': final_weights,
            'calculated_leverage': calculated_leverage,
            'mvp_volatility': mvp_volatility,
            'final_volatility': final_volatility,
            'risky_weights': risky_weights,
            'error': None
        }
        
    except Exception as e:
        return {
            'success': False,
            'final_weights': np.array([0.33, 0.33, 0.34, 0.0]),
            'calculated_leverage': 1.0,
            'mvp_volatility': 0.0,
            'final_volatility': 0.0,
            'risky_weights': np.array([0.33, 0.33, 0.34]),
            'error': f"Leverage calculation error: {str(e)}"
        }


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