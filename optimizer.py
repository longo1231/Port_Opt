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


def calculate_portfolio_diagnostics(data, weights, assets, covariance_matrix):
    """
    Calculate comprehensive portfolio diagnostics for minimum variance suitability.
    
    Analyzes why certain allocations occur and provides actionable insights about
    portfolio construction effectiveness.
    
    Parameters
    ----------
    data : pd.DataFrame
        Historical returns data
    weights : np.ndarray
        Portfolio weights (including Cash)
    assets : list
        Asset names
    covariance_matrix : np.ndarray
        Asset covariance matrix
        
    Returns
    -------
    dict
        Comprehensive diagnostics including suitability score and explanations
    """
    # Filter out Cash for risky asset analysis
    risky_assets = [a for a in assets if a != 'Cash']
    n_risky = len(risky_assets)
    risky_weights = weights[:n_risky]
    risky_data = data[risky_assets]
    risky_cov = covariance_matrix[:n_risky, :n_risky]
    
    # 1. Volatility Analysis
    individual_vols = np.sqrt(np.diag(risky_cov)) * 100  # Already annualized in estimators.py
    vol_range = individual_vols.max() - individual_vols.min()
    vol_ratio = individual_vols.max() / individual_vols.min() if individual_vols.min() > 0 else np.inf
    
    # 2. Correlation Analysis
    risky_corr = risky_data.corr()
    correlations = []
    for i in range(len(risky_assets)):
        for j in range(i+1, len(risky_assets)):
            correlations.append(risky_corr.iloc[i, j])
    
    avg_correlation = np.mean(correlations)
    min_correlation = np.min(correlations)
    max_correlation = np.max(correlations)
    negative_correlations = sum(1 for corr in correlations if corr < -0.1)
    
    # 3. Asset Class Diversity (simple heuristic)
    # Count different "types" based on ticker patterns
    asset_types = set()
    for asset in risky_assets:
        if asset in ['SPY', 'VTI', 'IWM', 'QQQ']:
            asset_types.add('equity')
        elif asset in ['TLT', 'IEF', 'SHY', 'BND']:
            asset_types.add('bonds')
        elif asset in ['GLD', 'SLV', 'IAU']:
            asset_types.add('commodities')
        elif asset in ['VNQ', 'REZ']:
            asset_types.add('reits')
        else:
            asset_types.add('individual_stock')  # Tech stocks, etc.
    
    asset_class_diversity = len(asset_types)
    
    # 4. Calculate Suitability Score (0-100)
    score_components = {}
    
    # Correlation Score (40 points max)
    if negative_correlations > 0:
        corr_score = min(40, 20 + negative_correlations * 10)  # Bonus for negative correlations
    elif avg_correlation < 0.3:
        corr_score = 30
    elif avg_correlation < 0.5:
        corr_score = 20
    elif avg_correlation < 0.7:
        corr_score = 10
    else:
        corr_score = 0
    score_components['correlation'] = corr_score
    
    # Volatility Balance Score (30 points max)
    if vol_ratio < 1.5:
        vol_score = 30  # Very balanced
    elif vol_ratio < 2.0:
        vol_score = 20  # Moderately balanced
    elif vol_ratio < 3.0:
        vol_score = 10  # Some imbalance
    else:
        vol_score = 0   # High imbalance
    score_components['volatility_balance'] = vol_score
    
    # Asset Class Diversity Score (30 points max)
    if asset_class_diversity >= 3:
        diversity_score = 30
    elif asset_class_diversity == 2:
        diversity_score = 20
    else:
        diversity_score = 5  # All same class
    score_components['asset_class_diversity'] = diversity_score
    
    total_score = sum(score_components.values())
    
    # 5. Individual Asset Analysis
    asset_diagnostics = []
    for i, asset in enumerate(risky_assets):
        weight = risky_weights[i]
        vol = individual_vols[i]
        avg_corr_with_others = np.mean([risky_corr.loc[asset, other] for other in risky_assets if other != asset])
        
        # Determine allocation reason
        vol_rank = np.argsort(individual_vols)[i] + 1  # 1 = lowest vol
        
        if weight > 0.4:
            reason = f"🥇 Dominant (lowest volatility: {vol:.1f}%)"
            category = "dominant"
        elif weight > 0.15:
            if vol_rank <= 3:
                reason = f"⚖️ Good balance (vol rank #{vol_rank}, {vol:.1f}%)"
            else:
                reason = f"⚖️ Moderate allocation despite higher volatility ({vol:.1f}%)"
            category = "balanced"
        elif weight > 0.05:
            reason = f"🔹 Minor role (vol: {vol:.1f}%, avg corr: {avg_corr_with_others:.2f})"
            category = "minor"
        elif weight > 0.001:
            reason = f"🔸 Minimal (too volatile: {vol:.1f}%)"
            category = "minimal"
        else:
            reason = f"❌ Excluded (dominated by lower-vol options)"
            category = "excluded"
        
        asset_diagnostics.append({
            'asset': asset,
            'weight': weight,
            'volatility': vol,
            'vol_rank': vol_rank,
            'avg_correlation': avg_corr_with_others,
            'reason': reason,
            'category': category
        })
    
    # 6. Overall Assessment
    if total_score >= 70:
        assessment = "Excellent"
        recommendation = "Well-suited for minimum variance optimization"
        color = "green"
    elif total_score >= 40:
        assessment = "Fair"
        recommendation = "Acceptable but limited diversification benefits"
        color = "yellow"
    else:
        assessment = "Poor"
        recommendation = "Consider Current template for better diversification"
        color = "red"
    
    # 7. Issues Detected
    issues = []
    if vol_ratio > 2.5:
        issues.append(f"High volatility spread ({vol_ratio:.1f}x range)")
    if avg_correlation > 0.6:
        issues.append("High average correlation limits diversification")
    if asset_class_diversity == 1:
        if 'individual_stock' in asset_types:
            issues.append("Single sector concentration (all individual stocks)")
        else:
            issues.append("Single asset class concentration")
    if negative_correlations == 0:
        issues.append("No negative correlations found")
    
    return {
        'suitability_score': total_score,
        'assessment': assessment,
        'recommendation': recommendation,
        'color': color,
        'score_components': score_components,
        'issues': issues,
        'volatility_stats': {
            'individual_vols': individual_vols,
            'vol_range': vol_range,
            'vol_ratio': vol_ratio,
            'lowest_vol_asset': risky_assets[np.argmin(individual_vols)],
            'highest_vol_asset': risky_assets[np.argmax(individual_vols)]
        },
        'correlation_stats': {
            'avg_correlation': avg_correlation,
            'min_correlation': min_correlation,
            'max_correlation': max_correlation,
            'negative_correlations': negative_correlations,
            'correlation_matrix': risky_corr
        },
        'diversity_stats': {
            'asset_classes': list(asset_types),
            'n_asset_classes': asset_class_diversity
        },
        'asset_diagnostics': asset_diagnostics
    }