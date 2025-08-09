"""
Portfolio optimization module implementing Kelly criterion maximization.

Provides two methods for maximizing expected geometric growth E[log(1 + w^T R)]:
1. Quadratic approximation: max w^T μ - 0.5 * w^T Σ w (small-return Kelly)
2. Monte Carlo simulation: exact expected log utility via sampling

Both methods enforce long-only constraints (w_i >= 0) and full investment (sum(w_i) = 1).
Includes transaction cost penalties and turnover regularization.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, Union
from scipy.optimize import minimize, Bounds, LinearConstraint
import cvxpy as cp
from scipy.stats import multivariate_normal
import warnings

from config import (
    N_ASSETS, DEFAULT_MC_SIMULATIONS, DEFAULT_TRANSACTION_COST_BPS,
    DEFAULT_TURNOVER_PENALTY, DEFAULT_KELLY_FRACTION, DEFAULT_LEVERAGE_CAP,
    MIN_WEIGHT, ASSETS, DEFAULT_MAX_WEIGHT, DEFAULT_MIN_WEIGHT
)


class PortfolioOptimizer:
    """
    Portfolio optimizer implementing Kelly criterion with constraints.
    
    Maximizes expected geometric growth under long-only and budget constraints
    with optional transaction costs and turnover penalties.
    """
    
    def __init__(
        self,
        transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
        turnover_penalty: float = DEFAULT_TURNOVER_PENALTY,
        kelly_fraction: float = DEFAULT_KELLY_FRACTION,
        leverage_cap: float = DEFAULT_LEVERAGE_CAP,
        max_weight: float = DEFAULT_MAX_WEIGHT,
        min_weight: float = DEFAULT_MIN_WEIGHT,
        random_seed: Optional[int] = 42
    ):
        """
        Initialize portfolio optimizer.
        
        Parameters
        ----------
        transaction_cost_bps : float, default from config
            Transaction cost in basis points per side
        turnover_penalty : float, default from config
            L1 penalty coefficient for turnover regularization
        kelly_fraction : float, default from config
            Partial Kelly fraction (0-1) to scale risk exposure
        leverage_cap : float, default from config
            Maximum leverage allowed (1.0 = no leverage)
        random_seed : int, optional
            Random seed for Monte Carlo simulations
        """
        self.transaction_cost_bps = transaction_cost_bps
        self.turnover_penalty = turnover_penalty
        self.kelly_fraction = np.clip(kelly_fraction, 0.0, 1.0)
        self.leverage_cap = max(leverage_cap, 0.0)
        self.max_weight = np.clip(max_weight, 0.0, 1.0)
        self.min_weight = np.clip(min_weight, 0.0, max_weight)
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def _apply_kelly_fraction_and_leverage(self, kelly_weights: np.ndarray) -> np.ndarray:
        """
        Apply partial Kelly fraction and leverage constraints to optimal Kelly weights.
        
        Parameters
        ----------
        kelly_weights : np.ndarray
            Optimal Kelly weights (may be leveraged)
            
        Returns
        -------
        np.ndarray
            Final portfolio weights with Kelly fraction and leverage applied
        """
        # Apply partial Kelly: scale risky positions, allocate remainder to cash
        if 'Cash' in ASSETS:
            cash_idx = ASSETS.index('Cash')
            risky_weights = kelly_weights.copy()
            risky_weights[cash_idx] = 0  # Separate out cash
            
            # Scale risky weights by Kelly fraction
            scaled_risky = risky_weights * self.kelly_fraction
            
            # Allocate remainder to cash
            final_weights = scaled_risky
            final_weights[cash_idx] = 1 - scaled_risky.sum()
            
            # Ensure cash weight is non-negative
            if final_weights[cash_idx] < 0:
                # If we need to borrow cash, scale down risky positions
                scale_factor = 1.0 / scaled_risky.sum()
                final_weights = scaled_risky * scale_factor
                final_weights[cash_idx] = 0
        else:
            # No cash asset, just scale all weights
            final_weights = kelly_weights * self.kelly_fraction
            final_weights = final_weights / final_weights.sum()
        
        # Apply leverage constraint (sum of absolute weights)
        total_leverage = np.sum(np.abs(final_weights))
        if total_leverage > self.leverage_cap:
            final_weights = final_weights * (self.leverage_cap / total_leverage)
        
        # Ensure weights are normalized and non-negative for long-only
        final_weights = np.maximum(final_weights, 0)
        if final_weights.sum() > 0:
            final_weights = final_weights / final_weights.sum()
        else:
            # Fallback to equal weights
            final_weights = np.ones(len(kelly_weights)) / len(kelly_weights)
            
        return final_weights
    
    def optimize_quadratic(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        current_weights: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Optimize portfolio using quadratic Kelly approximation.
        
        Solves: max w^T μ - 0.5 * w^T Σ w - λ * ||w - w_prev||_1
        subject to: w_i >= 0, sum(w_i) = 1
        
        Uses CVXPY for convex quadratic programming when available,
        falls back to SLSQP if needed.
        
        Parameters
        ----------
        expected_returns : np.ndarray
            Expected returns for each asset (annualized)
        covariance_matrix : np.ndarray
            Covariance matrix (annualized)
        current_weights : np.ndarray, optional
            Current portfolio weights for turnover penalty
            
        Returns
        -------
        Dict[str, Any]
            Optimization result including optimal weights, objective value, 
            and solver status
        """
        n_assets = len(expected_returns)
        
        if current_weights is None:
            current_weights = np.ones(n_assets) / n_assets
        
        # Try CVXPY first (more robust for QP)
        try:
            return self._solve_quadratic_cvxpy(
                expected_returns, covariance_matrix, current_weights
            )
        except Exception as e:
            warnings.warn(f"CVXPY solver failed, falling back to SLSQP: {str(e)}")
            return self._solve_quadratic_scipy(
                expected_returns, covariance_matrix, current_weights
            )
    
    def _solve_quadratic_cvxpy(
        self, 
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray, 
        current_weights: np.ndarray
    ) -> Dict[str, Any]:
        """Solve quadratic problem using CVXPY."""
        n_assets = len(expected_returns)
        w = cp.Variable(n_assets, nonneg=True)
        
        # Objective: maximize utility minus turnover penalty
        portfolio_return = w @ expected_returns
        portfolio_risk = cp.quad_form(w, covariance_matrix)
        turnover_cost = self.turnover_penalty * cp.norm(w - current_weights, 1)
        
        objective = cp.Maximize(portfolio_return - 0.5 * portfolio_risk - turnover_cost)
        
        # Constraints
        constraints = [cp.sum(w) == 1.0]  # Budget constraint
        
        # Create and solve problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS)
        
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"CVXPY solver failed with status: {problem.status}")
        
        optimal_weights = w.value
        if optimal_weights is None:
            raise RuntimeError("CVXPY returned None for optimal weights")
        
        # Clean up tiny weights
        kelly_weights = np.maximum(optimal_weights, 0)
        kelly_weights = kelly_weights / kelly_weights.sum()
        
        # Apply partial Kelly fraction and leverage constraints
        final_weights = self._apply_kelly_fraction_and_leverage(kelly_weights)
        
        # Calculate final objective value (based on Kelly weights, not final weights)
        obj_value = (kelly_weights @ expected_returns - 
                    0.5 * kelly_weights @ covariance_matrix @ kelly_weights -
                    self.turnover_penalty * np.sum(np.abs(kelly_weights - current_weights)))
        
        return {
            'weights': final_weights,
            'kelly_weights': kelly_weights,  # Store original Kelly weights
            'objective_value': float(obj_value),
            'success': True,
            'method': 'cvxpy',
            'solver_status': problem.status
        }
    
    def _solve_quadratic_scipy(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        current_weights: np.ndarray
    ) -> Dict[str, Any]:
        """Solve quadratic problem using SciPy SLSQP."""
        n_assets = len(expected_returns)
        
        def objective(w):
            portfolio_return = w @ expected_returns
            portfolio_risk = w @ covariance_matrix @ w
            turnover_cost = self.turnover_penalty * np.sum(np.abs(w - current_weights))
            return -(portfolio_return - 0.5 * portfolio_risk - turnover_cost)
        
        def objective_grad(w):
            grad_return = expected_returns
            grad_risk = covariance_matrix @ w
            grad_turnover = self.turnover_penalty * np.sign(w - current_weights)
            return -(grad_return - grad_risk - grad_turnover)
        
        # Constraints and bounds
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Budget constraint
        ]
        bounds = Bounds(lb=self.min_weight, ub=self.max_weight)  # Position size constraints
        
        # Initial guess - respect weight constraints
        x0 = np.ones(n_assets) / n_assets
        x0 = np.clip(x0, self.min_weight, self.max_weight)
        x0 = x0 / x0.sum()  # Renormalize
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            jac=objective_grad,
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )
        
        if not result.success:
            warnings.warn(f"SciPy optimization failed: {result.message}")
        
        kelly_weights = np.maximum(result.x, 0)
        kelly_weights = kelly_weights / kelly_weights.sum()
        
        # Apply partial Kelly fraction and leverage constraints
        final_weights = self._apply_kelly_fraction_and_leverage(kelly_weights)
        
        return {
            'weights': final_weights,
            'kelly_weights': kelly_weights,  # Store original Kelly weights
            'objective_value': float(-result.fun),
            'success': result.success,
            'method': 'scipy_slsqp',
            'solver_status': result.message
        }
    
    def optimize_monte_carlo(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        n_simulations: int = DEFAULT_MC_SIMULATIONS,
        current_weights: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Optimize portfolio using Monte Carlo expected log utility.
        
        Exactly maximizes E[log(1 + w^T R)] by sampling from multivariate normal
        distribution and computing sample average of log returns.
        
        Parameters
        ----------
        expected_returns : np.ndarray
            Expected returns for each asset (annualized)  
        covariance_matrix : np.ndarray
            Covariance matrix (annualized)
        n_simulations : int, default from config
            Number of Monte Carlo samples
        current_weights : np.ndarray, optional
            Current portfolio weights for turnover penalty
            
        Returns
        -------
        Dict[str, Any]
            Optimization result including optimal weights and objective value
        """
        n_assets = len(expected_returns)
        
        if current_weights is None:
            current_weights = np.ones(n_assets) / n_assets
        
        # Convert to daily parameters for simulation
        daily_mu = expected_returns / 252
        daily_cov = covariance_matrix / 252
        
        # Generate random samples
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        samples = np.random.multivariate_normal(
            mean=daily_mu,
            cov=daily_cov,
            size=n_simulations
        )
        
        def objective(w):
            portfolio_returns = samples @ w
            # Avoid log(negative) by clamping portfolio returns
            portfolio_returns = np.maximum(portfolio_returns, -0.99)
            log_returns = np.log(1 + portfolio_returns)
            expected_log_return = np.mean(log_returns)
            
            # Add turnover penalty
            turnover_cost = self.turnover_penalty * np.sum(np.abs(w - current_weights))
            
            return -(expected_log_return - turnover_cost)
        
        def objective_grad(w):
            portfolio_returns = samples @ w
            portfolio_returns = np.maximum(portfolio_returns, -0.99)
            
            # Gradient of log(1 + r) is 1/(1 + r)
            weights_grad = np.mean(samples / (1 + portfolio_returns[:, np.newaxis]), axis=0)
            turnover_grad = self.turnover_penalty * np.sign(w - current_weights)
            
            return -(weights_grad - turnover_grad)
        
        # Constraints and bounds
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        bounds = Bounds(lb=self.min_weight, ub=self.max_weight)
        
        # Initial guess - respect weight constraints
        x0 = np.ones(n_assets) / n_assets
        x0 = np.clip(x0, self.min_weight, self.max_weight)
        x0 = x0 / x0.sum()  # Renormalize
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            jac=objective_grad,
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )
        
        if not result.success:
            warnings.warn(f"Monte Carlo optimization failed: {result.message}")
        
        kelly_weights = np.maximum(result.x, 0)
        kelly_weights = kelly_weights / kelly_weights.sum()
        
        # Apply partial Kelly fraction and leverage constraints
        final_weights = self._apply_kelly_fraction_and_leverage(kelly_weights)
        
        # Calculate final objective (annualized expected log return, using Kelly weights)
        portfolio_returns = samples @ kelly_weights
        portfolio_returns = np.maximum(portfolio_returns, -0.99)
        expected_log_return = np.mean(np.log(1 + portfolio_returns)) * 252
        turnover_cost = self.turnover_penalty * np.sum(np.abs(kelly_weights - current_weights))
        
        return {
            'weights': final_weights,
            'kelly_weights': kelly_weights,  # Store original Kelly weights
            'objective_value': float(expected_log_return - turnover_cost),
            'success': result.success,
            'method': 'monte_carlo',
            'n_simulations': n_simulations,
            'solver_status': result.message
        }
    
    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        method: str = "quadratic",
        current_weights: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Unified optimization interface.
        
        Parameters
        ----------
        expected_returns : np.ndarray
            Expected returns for each asset
        covariance_matrix : np.ndarray  
            Covariance matrix
        method : str, default "quadratic"
            Optimization method: "quadratic" or "monte_carlo"
        current_weights : np.ndarray, optional
            Current portfolio weights
        **kwargs
            Additional method-specific parameters
            
        Returns
        -------
        Dict[str, Any]
            Optimization result
        """
        if method == "quadratic":
            return self.optimize_quadratic(expected_returns, covariance_matrix, current_weights)
        elif method == "monte_carlo":
            return self.optimize_monte_carlo(
                expected_returns, covariance_matrix,
                current_weights=current_weights,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")


def calculate_expected_geometric_growth(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    method: str = "quadratic"
) -> float:
    """
    Calculate expected geometric growth rate for given portfolio weights.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    expected_returns : np.ndarray
        Expected returns  
    covariance_matrix : np.ndarray
        Covariance matrix
    method : str, default "quadratic"
        Calculation method: "quadratic" for approximation or "exact" for MC
        
    Returns
    -------
    float
        Annualized expected geometric growth rate
    """
    if method == "quadratic":
        # Kelly approximation: E[log(1 + R)] ≈ μ - 0.5 * σ²
        portfolio_return = weights @ expected_returns
        portfolio_variance = weights @ covariance_matrix @ weights
        return portfolio_return - 0.5 * portfolio_variance
    else:
        raise NotImplementedError("Exact calculation requires Monte Carlo simulation")


def compute_portfolio_attribution(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    current_weights: Optional[np.ndarray] = None,
    turnover_penalty: float = DEFAULT_TURNOVER_PENALTY,
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS
) -> Dict[str, float]:
    """
    Decompose portfolio expected growth into components.
    
    Breaks down the expected geometric growth into:
    - Expected return contribution (w^T μ)
    - Risk penalty (-0.5 * w^T Σ w)  
    - Turnover penalty
    - Transaction costs
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    expected_returns : np.ndarray
        Expected returns
    covariance_matrix : np.ndarray
        Covariance matrix
    current_weights : np.ndarray, optional
        Current weights for cost calculation
    turnover_penalty : float
        Turnover penalty coefficient
    transaction_cost_bps : float
        Transaction cost in basis points
        
    Returns
    -------
    Dict[str, float]
        Attribution dictionary with component contributions
    """
    # Core components
    return_contribution = weights @ expected_returns
    risk_penalty = 0.5 * weights @ covariance_matrix @ weights
    
    # Cost components
    if current_weights is not None:
        turnover = np.sum(np.abs(weights - current_weights))
        turnover_cost = turnover_penalty * turnover
        transaction_costs = (transaction_cost_bps / 10000) * turnover
    else:
        turnover = 0.0
        turnover_cost = 0.0
        transaction_costs = 0.0
    
    # Net expected growth
    net_growth = return_contribution - risk_penalty - turnover_cost - transaction_costs
    
    return {
        'expected_return': float(return_contribution),
        'risk_penalty': float(risk_penalty),
        'turnover_penalty': float(turnover_cost),
        'transaction_costs': float(transaction_costs),
        'turnover': float(turnover),
        'net_expected_growth': float(net_growth)
    }