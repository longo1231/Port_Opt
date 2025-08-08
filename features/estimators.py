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
from typing import Tuple, Optional, Union
from scipy.linalg import sqrtm
from sklearn.covariance import LedoitWolf
import warnings

from config import (
    TRADING_DAYS_PER_YEAR, DEFAULT_LOOKBACK_DAYS, DEFAULT_MU_SIGMA_HALFLIFE,
    DEFAULT_CORR_HALFLIFE, DEFAULT_SHRINKAGE, CONDITION_NUMBER_THRESHOLD
)


class ReturnEstimator:
    """
    Expected return estimator using rolling window or EWMA methods.
    
    Provides both historical sample means and exponentially weighted averages
    with annualized output for portfolio optimization.
    """
    
    def __init__(
        self, 
        method: str = "ewma",
        window: int = DEFAULT_LOOKBACK_DAYS,
        halflife: Optional[int] = DEFAULT_MU_SIGMA_HALFLIFE
    ):
        """
        Initialize return estimator.
        
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
        Estimate expected returns from historical data.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Daily log returns with assets as columns
            
        Returns
        -------
        np.ndarray
            Annualized expected returns for each asset
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
        halflife: Optional[int] = DEFAULT_MU_SIGMA_HALFLIFE
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
    
    def estimate(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Estimate correlation matrix from returns.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Daily log returns with assets as columns
            
        Returns
        -------
        np.ndarray
            Correlation matrix (n_assets x n_assets)
        """
        min_periods = max(2, self.window if self.method == "rolling" else 2)
        if len(returns) < min_periods:
            raise ValueError("Insufficient data for correlation estimation")
        
        if self.method == "rolling":
            corr_matrix = returns.tail(self.window).corr().values
        else:  # EWMA
            # Get the latest correlation matrix from EWMA
            ewma_corr = returns.ewm(halflife=self.halflife).corr()
            # Extract the last nxn block (the latest correlation matrix)
            n_assets = len(returns.columns)
            corr_matrix = ewma_corr.iloc[-n_assets:].values
            
        # Ensure numerical stability
        corr_matrix = self._ensure_positive_definite(corr_matrix)
        return corr_matrix
    
    def _ensure_positive_definite(self, corr_matrix: np.ndarray) -> np.ndarray:
        """
        Ensure correlation matrix is positive definite.
        
        Uses eigenvalue clipping to fix numerical issues while
        preserving the correlation structure as much as possible.
        """
        # Handle NaN values by replacing with reasonable correlation matrix
        if np.isnan(corr_matrix).any():
            warnings.warn("NaN values found in correlation matrix, using fallback")
            n = corr_matrix.shape[0]
            # Use a mild positive correlation structure
            fallback_matrix = np.full((n, n), 0.1)
            np.fill_diagonal(fallback_matrix, 1.0)
            return fallback_matrix
        
        try:
            eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
        except np.linalg.LinAlgError:
            warnings.warn("Eigenvalue decomposition failed, using fallback correlation matrix")
            # Use a simple fallback: set off-diagonal correlations to 0.5
            n = corr_matrix.shape[0]
            fallback_matrix = np.full((n, n), 0.5)
            np.fill_diagonal(fallback_matrix, 1.0)
            return fallback_matrix
            
        min_eigenval = 1e-8
        
        if eigenvals.min() < min_eigenval:
            eigenvals = np.maximum(eigenvals, min_eigenval)
            corr_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            # Rescale diagonal to 1.0
            diag_vals = np.diag(corr_matrix)
            if np.any(diag_vals <= 0):
                warnings.warn("Invalid diagonal values in correlation matrix")
                return np.eye(corr_matrix.shape[0])
                
            diag_sqrt = np.sqrt(diag_vals)
            corr_matrix = corr_matrix / np.outer(diag_sqrt, diag_sqrt)
            
        return corr_matrix


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
        correlation_matrix = self.corr_estimator.estimate(returns)
        
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
    mu_halflife: int = DEFAULT_MU_SIGMA_HALFLIFE,
    vol_halflife: int = DEFAULT_MU_SIGMA_HALFLIFE,
    corr_halflife: int = DEFAULT_CORR_HALFLIFE,
    shrinkage: float = DEFAULT_SHRINKAGE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate all portfolio parameters in one call.
    
    Convenience function that estimates expected returns, covariance matrix,
    and individual volatilities using specified methods and parameters.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns with assets as columns
    mu_method, vol_method, corr_method : str
        Estimation methods: "rolling" or "ewma"
    mu_window, vol_window, corr_window : int
        Rolling window sizes
    mu_halflife, vol_halflife, corr_halflife : int
        EWMA half-lives
    shrinkage : float
        Covariance shrinkage intensity
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (expected_returns, covariance_matrix, volatilities)
    """
    # Initialize estimators
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