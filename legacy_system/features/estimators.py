"""
Feature estimation module for portfolio optimizer.

Provides statistical estimators for expected returns (mu), volatilities (sigma),
and correlation matrices (rho) using both rolling window and EWMA approaches.
Includes covariance matrix construction with shrinkage regularization.

Key features:
- Separate parameterization for mu/sigma vs correlation estimation
- EWMA with configurable half-lives for faster correlation adaptation  
- Ledoit-Wolf shrinkage for covariance regularization
- Numerical stability checks and condition number monitoring
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, Dict
from scipy.linalg import sqrtm
from sklearn.covariance import LedoitWolf
import warnings

from config import (
    TRADING_DAYS_PER_YEAR, DEFAULT_LOOKBACK_DAYS, DEFAULT_MU_HALFLIFE,
    DEFAULT_SIGMA_HALFLIFE, DEFAULT_CORR_HALFLIFE, DEFAULT_SHRINKAGE, 
    CONDITION_NUMBER_THRESHOLD, ASSETS, DEFAULT_RISK_FREE_RATE
)


class ExpectedReturnEstimator:
    """
    Expected return estimator using structural risk premium approach.
    
    Separates long-term expected returns (structural parameters) from
    short-term risk estimation. Expected returns should be based on
    fundamental risk premiums, not noisy historical samples.
    """
    
    def __init__(
        self, 
        method: str = "risk_premium",
        expected_returns: Optional[Dict[str, float]] = None,
        historical_window: Optional[int] = None,
        min_history_years: int = 5
    ):
        """
        Initialize expected return estimator.
        
        Parameters
        ----------
        method : str, default "risk_premium"
            Method: "risk_premium", "factor_model", "long_term_historical"
        expected_returns : dict, optional
            User-specified expected returns for each asset
        historical_window : int, optional
            Window for historical estimation (if using historical method)
        min_history_years : int, default 5
            Minimum years of data required for historical estimation
        """
        if method not in ["risk_premium", "factor_model", "long_term_historical"]:
            raise ValueError("Method must be 'risk_premium', 'factor_model', or 'long_term_historical'")
            
        self.method = method
        self.expected_returns = expected_returns
        self.historical_window = historical_window
        self.min_history_years = min_history_years
    
    def estimate(self, returns: pd.DataFrame, risk_free_rate: float = None) -> np.ndarray:
        """
        Estimate expected returns using structural approach.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Daily log returns with assets as columns (used for validation/historical)
        risk_free_rate : float, optional
            Current risk-free rate for factor models
            
        Returns
        -------
        np.ndarray
            Annualized expected returns for each asset
        """
        if self.method == "risk_premium":
            return self._estimate_risk_premium(risk_free_rate)
            
        elif self.method == "factor_model":
            return self._estimate_factor_model(risk_free_rate or DEFAULT_RISK_FREE_RATE)
            
        elif self.method == "long_term_historical":
            return self._estimate_long_term_historical(returns)
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _estimate_risk_premium(self, risk_free_rate: Optional[float]) -> np.ndarray:
        """Estimate returns using risk premium assumptions."""
        if self.expected_returns:
            # User-specified returns
            returns = np.array([self.expected_returns.get(asset, 0.05) for asset in ASSETS])
        else:
            # Default risk premiums
            from config import DEFAULT_EXPECTED_RETURNS
            returns = np.array([DEFAULT_EXPECTED_RETURNS[asset] for asset in ASSETS])
        
        # Update cash return with current risk-free rate if provided
        if risk_free_rate is not None and 'Cash' in ASSETS:
            cash_idx = ASSETS.index('Cash')
            returns[cash_idx] = risk_free_rate
            
        return returns
    
    def _estimate_factor_model(self, risk_free_rate: float) -> np.ndarray:
        """Estimate returns using factor model (CAPM-style)."""
        from config import FACTOR_BASED_RETURNS
        
        market_premium = FACTOR_BASED_RETURNS['market_premium']
        betas = FACTOR_BASED_RETURNS['betas']
        
        returns = []
        for asset in ASSETS:
            if asset == 'Cash':
                expected_return = risk_free_rate
            else:
                beta = betas.get(asset, 0.5)  # Default beta if not specified
                expected_return = risk_free_rate + beta * market_premium
            returns.append(expected_return)
            
        return np.array(returns)
    
    def _estimate_long_term_historical(self, returns: pd.DataFrame) -> np.ndarray:
        """Estimate returns using long-term historical averages."""
        min_observations = self.min_history_years * TRADING_DAYS_PER_YEAR
        
        if len(returns) >= min_observations:
            # Use entire available history for stable estimates
            if self.historical_window and len(returns) > self.historical_window:
                # Use specified window from the end
                historical_returns = returns.tail(self.historical_window).mean().values
            else:
                # Use all available data
                historical_returns = returns.mean().values
                
            return historical_returns * TRADING_DAYS_PER_YEAR
        else:
            # Insufficient history, fall back to risk premiums
            warnings.warn(f"Insufficient history ({len(returns)} days), using risk premium assumptions")
            return self._estimate_risk_premium(None)


class ReturnEstimator:
    """
    SPEC-COMPLIANT: Adaptive expected return estimator with slow adaptation.
    
    Uses slow adaptation (long window/halflife) to minimize noise in μ estimates
    while still being responsive to genuine regime changes. Per Breaking the Market
    philosophy: slow μ adaptation + partial Kelly handles uncertainty.
    """
    
    def __init__(
        self, 
        method: str = "ewma",
        window: int = DEFAULT_LOOKBACK_DAYS,
        halflife: Optional[int] = DEFAULT_MU_HALFLIFE
    ):
        
        if method not in ["rolling", "ewma"]:
            raise ValueError("Method must be 'rolling' or 'ewma'")
        if method == "ewma" and halflife is None:
            raise ValueError("Half-life must be specified for EWMA method")
            
        self.method = method
        self.window = window
        self.halflife = halflife
    
    def estimate(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Estimate expected returns using slow adaptation (SPEC-COMPLIANT).
        
        Uses slow adaptation to minimize noise while still being responsive
        to genuine regime changes. Designed to work with partial Kelly sizing.
        """
        if len(returns) < max(2, self.window if self.method == "rolling" else 1):
            raise ValueError("Insufficient data for return estimation")
        
        if self.method == "rolling":
            mu_daily = returns.tail(self.window).mean().values
        else:  # EWMA
            mu_daily = returns.ewm(halflife=self.halflife).mean().iloc[-1].values
            
        return mu_daily * TRADING_DAYS_PER_YEAR


class VolatilityEstimator:
    """
    Volatility estimator using rolling window or EWMA methods.
    
    Computes sample standard deviations with proper annualization
    for portfolio risk modeling.
    """
    
    def __init__(
        self,
        method: str = "ewma", 
        window: int = DEFAULT_LOOKBACK_DAYS,
        halflife: Optional[int] = DEFAULT_SIGMA_HALFLIFE
    ):
        """
        Initialize volatility estimator.
        
        Parameters
        ----------
        method : str, default "ewma"
            Estimation method: "rolling" or "ewma"
        window : int, default from config
            Rolling window size in days
        halflife : int, optional
            EWMA half-life in days, required for EWMA method
        """
        if method not in ["rolling", "ewma"]:
            raise ValueError("Method must be 'rolling' or 'ewma'")
        if method == "ewma" and halflife is None:
            raise ValueError("Half-life must be specified for EWMA method")
            
        self.method = method
        self.window = window
        self.halflife = halflife
    
    def estimate(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Estimate volatilities from historical returns.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Daily log returns with assets as columns
            
        Returns
        -------
        np.ndarray
            Annualized volatilities for each asset
        """
        if len(returns) < max(2, self.window if self.method == "rolling" else 2):
            raise ValueError("Insufficient data for volatility estimation")
        
        if self.method == "rolling":
            sigma_daily = returns.tail(self.window).std().values
        else:  # EWMA
            sigma_daily = returns.ewm(halflife=self.halflife).std().iloc[-1].values
        
        # Handle zero volatility assets (like cash) by setting minimum volatility
        min_vol = 1e-6 / np.sqrt(TRADING_DAYS_PER_YEAR)  # Very small daily volatility
        sigma_daily = np.maximum(sigma_daily, min_vol)
            
        return sigma_daily * np.sqrt(TRADING_DAYS_PER_YEAR)


class CorrelationEstimator:
    """
    Correlation matrix estimator with fast adaptation capability.
    
    Uses shorter half-lives than mu/sigma estimation to capture
    changing market regimes and correlations more quickly.
    """
    
    def __init__(
        self,
        method: str = "ewma",
        window: int = DEFAULT_LOOKBACK_DAYS,
        halflife: int = DEFAULT_CORR_HALFLIFE
    ):
        """
        Initialize correlation estimator.
        
        Parameters
        ----------
        method : str, default "ewma"
            Estimation method: "rolling" or "ewma"
        window : int, default from config
            Rolling window size in days
        halflife : int, default from config
            EWMA half-life in days (typically shorter than mu/sigma)
        """
        if method not in ["rolling", "ewma"]:
            raise ValueError("Method must be 'rolling' or 'ewma'")
            
        self.method = method
        self.window = window
        self.halflife = halflife
    
    def estimate(self, returns: pd.DataFrame) -> tuple[np.ndarray, dict]:
        """
        Estimate correlation matrix from returns with robust NA handling.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Daily log returns with assets as columns
            
        Returns
        -------
        tuple[np.ndarray, dict]
            Correlation matrix (n_assets x n_assets) and status information
        """
        # Track status information
        status = {
            'method': self.method,
            'original_observations': len(returns),
            'clean_observations': 0,
            'data_dropped': 0,
            'estimation_method': 'normal',
            'warnings': [],
            'fallback_used': False
        }
        
        # Clean data: drop rows with any NaN values
        clean_returns = returns.dropna()
        status['clean_observations'] = len(clean_returns)
        status['data_dropped'] = len(returns) - len(clean_returns)
        
        if status['data_dropped'] > 0:
            status['warnings'].append(f"Dropped {status['data_dropped']} rows with missing data")
        
        min_periods = max(10, self.window if self.method == "rolling" else 10)
        if len(clean_returns) < min_periods:
            status['fallback_used'] = True
            status['warnings'].append(f"Insufficient clean data: {len(clean_returns)} observations (need {min_periods}+)")
            
            # Simple fallback: use sample correlation on whatever data we have
            if len(clean_returns) >= 2:
                corr_matrix = clean_returns.corr().values
                status['estimation_method'] = 'sample_correlation_fallback'
                status['warnings'].append("Using sample correlation on limited data")
            else:
                # Last resort: identity matrix (no correlation)
                n_assets = len(returns.columns)
                corr_matrix = np.eye(n_assets)
                status['estimation_method'] = 'identity_matrix_fallback'
                status['warnings'].append("Using identity matrix - no correlations assumed")
        else:
            # Normal estimation with clean data
            if self.method == "rolling":
                subset = clean_returns.tail(min(self.window, len(clean_returns)))
                corr_matrix = subset.corr().values
                status['estimation_method'] = f"rolling_window_{len(subset)}_days"
            else:  # EWMA
                # Use EWMA on clean data only
                corr_matrix = clean_returns.ewm(halflife=self.halflife).corr().iloc[-len(returns.columns):].values
                status['estimation_method'] = f"ewma_halflife_{self.halflife}_days"
            
            # Handle zero-volatility assets (like Cash) that cause NaN correlations
            corr_matrix = self._handle_zero_volatility_correlations(corr_matrix, clean_returns, status)
        
        # Handle any remaining NaNs and ensure positive definiteness  
        corr_matrix, pd_status = self._ensure_positive_definite(corr_matrix)
        status.update(pd_status)
        
        return corr_matrix, status
    
    def _handle_zero_volatility_correlations(self, corr_matrix: np.ndarray, returns: pd.DataFrame, status: dict) -> np.ndarray:
        """
        Handle correlations involving zero-volatility assets (like Cash).
        
        Zero-volatility assets cause NaN correlations. We set:
        - Self-correlation to 1.0 (diagonal)
        - Cross-correlations to 0.0 (uncorrelated with other assets)
        """
        # Identify zero-volatility assets
        volatilities = returns.std()
        zero_vol_mask = (volatilities == 0) | (volatilities < 1e-10)
        zero_vol_indices = np.where(zero_vol_mask)[0]
        
        if len(zero_vol_indices) > 0:
            status['warnings'].append(f"Found {len(zero_vol_indices)} zero-volatility assets: {[returns.columns[i] for i in zero_vol_indices]}")
            
            # Fix NaN correlations for zero-volatility assets
            for idx in zero_vol_indices:
                # Set diagonal to 1.0 (self-correlation)
                corr_matrix[idx, idx] = 1.0
                # Set off-diagonal to 0.0 (uncorrelated)
                corr_matrix[idx, :] = 0.0
                corr_matrix[:, idx] = 0.0
                corr_matrix[idx, idx] = 1.0  # Restore diagonal
        
        return corr_matrix
    
    def _ensure_positive_definite(self, corr_matrix: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Ensure correlation matrix is positive definite with minimal intervention.
        
        Uses eigenvalue clipping to fix numerical issues while preserving 
        the actual correlation structure as much as possible.
        """
        pd_status = {
            'positive_definite_issues': [],
            'eigenvalue_clipping': False,
            'final_method': 'original'
        }
        
        # Handle NaN values - should be rare with clean data approach
        if np.isnan(corr_matrix).any():
            pd_status['positive_definite_issues'].append("NaN values found")
            pd_status['final_method'] = 'identity_matrix'
            warnings.warn("NaN values found in correlation matrix, using identity matrix")
            return np.eye(corr_matrix.shape[0]), pd_status
        
        try:
            eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
        except np.linalg.LinAlgError:
            pd_status['positive_definite_issues'].append("Eigenvalue decomposition failed")
            pd_status['final_method'] = 'identity_matrix'
            warnings.warn("Eigenvalue decomposition failed, using identity matrix")
            return np.eye(corr_matrix.shape[0]), pd_status
            
        min_eigenval = 1e-8
        
        if eigenvals.min() < min_eigenval:
            pd_status['eigenvalue_clipping'] = True
            pd_status['positive_definite_issues'].append(f"Clipped {(eigenvals < min_eigenval).sum()} negative eigenvalues")
            pd_status['final_method'] = 'eigenvalue_clipping'
            
            eigenvals = np.maximum(eigenvals, min_eigenval)
            corr_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            # Rescale diagonal to 1.0
            diag_vals = np.diag(corr_matrix)
            if np.any(diag_vals <= 0):
                pd_status['positive_definite_issues'].append("Invalid diagonal values after clipping")
                pd_status['final_method'] = 'identity_matrix'
                warnings.warn("Invalid diagonal values in correlation matrix")
                return np.eye(corr_matrix.shape[0]), pd_status
                
            diag_sqrt = np.sqrt(diag_vals)
            corr_matrix = corr_matrix / np.outer(diag_sqrt, diag_sqrt)
            
        return corr_matrix, pd_status


class CovarianceEstimator:
    """
    Covariance matrix estimator with shrinkage regularization.
    
    Combines volatility and correlation estimates to build covariance matrices
    with optional Ledoit-Wolf shrinkage for improved numerical properties.
    """
    
    def __init__(
        self,
        volatility_estimator: VolatilityEstimator,
        correlation_estimator: CorrelationEstimator,
        shrinkage: float = DEFAULT_SHRINKAGE
    ):
        """
        Initialize covariance estimator.
        
        Parameters
        ----------
        volatility_estimator : VolatilityEstimator
            Configured volatility estimator
        correlation_estimator : CorrelationEstimator
            Configured correlation estimator
        shrinkage : float, default from config
            Shrinkage intensity towards diagonal matrix (0=no shrinkage, 1=diagonal)
        """
        self.vol_estimator = volatility_estimator
        self.corr_estimator = correlation_estimator
        self.shrinkage = np.clip(shrinkage, 0.0, 1.0)
    
    def estimate(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Estimate covariance matrix from returns.
        
        Combines separate volatility and correlation estimates with optional
        shrinkage towards diagonal matrix for regularization.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Daily log returns with assets as columns
            
        Returns
        -------
        np.ndarray  
            Annualized covariance matrix
        """
        # Get individual estimates
        volatilities = self.vol_estimator.estimate(returns)
        correlation_matrix, corr_status = self.corr_estimator.estimate(returns)
        
        # Store correlation status for potential UI display
        self._last_correlation_status = corr_status
        
        # Build covariance matrix: Σ = D * R * D where D is diag(σ), R is correlation
        vol_matrix = np.outer(volatilities, volatilities)
        cov_matrix = vol_matrix * correlation_matrix
        
        # Apply shrinkage if specified
        if self.shrinkage > 0:
            cov_matrix = self._apply_shrinkage(cov_matrix, volatilities)
        
        # Validate numerical properties
        self._validate_covariance(cov_matrix)
        
        return cov_matrix
    
    def _apply_shrinkage(self, cov_matrix: np.ndarray, volatilities: np.ndarray) -> np.ndarray:
        """Apply shrinkage towards diagonal covariance matrix."""
        diagonal_cov = np.diag(volatilities ** 2)
        shrunk_cov = (1 - self.shrinkage) * cov_matrix + self.shrinkage * diagonal_cov
        return shrunk_cov
    
    def _validate_covariance(self, cov_matrix: np.ndarray) -> None:
        """Validate covariance matrix properties."""
        eigenvals = np.linalg.eigvals(cov_matrix)
        
        if eigenvals.min() <= 0:
            warnings.warn("Covariance matrix is not positive definite")
        
        condition_number = np.linalg.cond(cov_matrix)
        if condition_number > CONDITION_NUMBER_THRESHOLD:
            warnings.warn(f"Covariance matrix is ill-conditioned (cond={condition_number:.2e})")


def estimate_all_parameters(
    returns: pd.DataFrame,
    mu_method: str = "ewma",
    vol_method: str = "ewma", 
    corr_method: str = "ewma",
    mu_window: int = DEFAULT_LOOKBACK_DAYS,
    vol_window: int = DEFAULT_LOOKBACK_DAYS,
    corr_window: int = DEFAULT_LOOKBACK_DAYS,
    mu_halflife: int = DEFAULT_MU_HALFLIFE,
    vol_halflife: int = DEFAULT_SIGMA_HALFLIFE,
    corr_halflife: int = DEFAULT_CORR_HALFLIFE,
    shrinkage: float = DEFAULT_SHRINKAGE,
    # Legacy parameters (deprecated)
    return_method: Optional[str] = None,
    expected_returns: Optional[Dict[str, float]] = None,
    risk_free_rate: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate all portfolio parameters using SPEC-COMPLIANT adaptive approach.
    
    PHILOSOPHY: Slow μ adaptation to avoid noise, medium σ adaptation, 
    fast ρ adaptation to capture regime changes. Use partial Kelly 
    to handle μ uncertainty.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns with assets as columns
    mu_method, vol_method, corr_method : str
        Estimation methods: "rolling" or "ewma"
    mu_window, vol_window, corr_window : int
        Rolling window sizes (days)
    mu_halflife, vol_halflife, corr_halflife : int
        EWMA half-lives (days) - μ slow, σ medium, ρ fast
    shrinkage : float
        Covariance shrinkage intensity
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (expected_returns, covariance_matrix, volatilities)
    """
    # Handle legacy parameters
    if return_method is not None:
        warnings.warn(
            "The structural risk premium approach has been removed per spec requirements. "
            "Now using adaptive μ estimation with slow adaptation to minimize noise.",
            DeprecationWarning
        )
    
    # Initialize estimators with SPEC-COMPLIANT speed hierarchy
    # μ: Slow adaptation to avoid noise (legacy ReturnEstimator approach)
    mu_estimator = ReturnEstimator(mu_method, mu_window, mu_halflife)
    
    # σ: Medium adaptation  
    vol_estimator = VolatilityEstimator(vol_method, vol_window, vol_halflife)
    
    # ρ: Fast adaptation to capture regime changes
    corr_estimator = CorrelationEstimator(corr_method, corr_window, corr_halflife)
    
    # Combined covariance estimation
    cov_estimator = CovarianceEstimator(vol_estimator, corr_estimator, shrinkage)
    
    # Compute estimates using adaptive approach (spec compliant)
    expected_returns = mu_estimator.estimate(returns)
    covariance_matrix = cov_estimator.estimate(returns)
    volatilities = vol_estimator.estimate(returns)
    
    # Get correlation status for UI display
    correlation_status = getattr(cov_estimator, '_last_correlation_status', {'warnings': []})
    
    return expected_returns, covariance_matrix, volatilities, correlation_status


def estimate_all_parameters_legacy(
    returns: pd.DataFrame,
    mu_method: str = "ewma",
    vol_method: str = "ewma", 
    corr_method: str = "ewma",
    mu_window: int = DEFAULT_LOOKBACK_DAYS,
    vol_window: int = DEFAULT_LOOKBACK_DAYS,
    corr_window: int = DEFAULT_LOOKBACK_DAYS,
    mu_halflife: int = DEFAULT_MU_HALFLIFE,
    vol_halflife: int = DEFAULT_SIGMA_HALFLIFE,
    corr_halflife: int = DEFAULT_CORR_HALFLIFE,
    shrinkage: float = DEFAULT_SHRINKAGE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    DEPRECATED: Legacy parameter estimation function.
    
    This function produces unstable return forecasts. Use estimate_all_parameters() instead.
    """
    warnings.warn(
        "estimate_all_parameters_legacy produces unstable return forecasts. "
        "Use estimate_all_parameters() with return_method='risk_premium' instead.",
        DeprecationWarning
    )
    
    # Initialize estimators (legacy approach)
    mu_estimator = ReturnEstimator(mu_method, mu_window, mu_halflife)
    vol_estimator = VolatilityEstimator(vol_method, vol_window, vol_halflife)
    corr_estimator = CorrelationEstimator(corr_method, corr_window, corr_halflife)
    cov_estimator = CovarianceEstimator(vol_estimator, corr_estimator, shrinkage)
    
    # Compute estimates
    expected_returns = mu_estimator.estimate(returns)
    covariance_matrix = cov_estimator.estimate(returns)
    volatilities = vol_estimator.estimate(returns)
    
    return expected_returns, covariance_matrix, volatilities


def compute_portfolio_statistics(
    weights: np.ndarray,
    expected_returns: np.ndarray, 
    covariance_matrix: np.ndarray
) -> dict:
    """
    Compute portfolio-level risk and return statistics.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    expected_returns : np.ndarray
        Expected returns for each asset
    covariance_matrix : np.ndarray
        Asset covariance matrix
        
    Returns
    -------
    dict
        Dictionary with portfolio statistics including expected return,
        volatility, and Sharpe ratio
    """
    portfolio_return = weights @ expected_returns
    portfolio_variance = weights @ covariance_matrix @ weights
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    # Assume risk-free rate is already incorporated in expected returns
    sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
    
    return {
        'expected_return': float(portfolio_return),
        'volatility': float(portfolio_volatility),
        'variance': float(portfolio_variance),
        'sharpe_ratio': float(sharpe_ratio)
    }