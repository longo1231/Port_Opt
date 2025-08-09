"""
Portfolio performance metrics and risk analysis utilities.

Provides comprehensive performance measurement tools including:
- Standard risk-return metrics (Sharpe, Sortino, Calmar ratios)
- Drawdown analysis and risk measurement  
- Attribution analysis for portfolio components
- Statistical tests and confidence intervals
- Rolling performance analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union
from scipy import stats
import warnings

from config import TRADING_DAYS_PER_YEAR


def calculate_returns_metrics(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> Dict[str, float]:
    """
    Calculate comprehensive return-based performance metrics.
    
    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Time series of portfolio returns
    risk_free_rate : float, default 0.0
        Annualized risk-free rate for excess return calculations
    periods_per_year : int, default from config
        Number of return periods per year for annualization
        
    Returns
    -------
    Dict[str, float]
        Dictionary of performance metrics
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    returns = returns[~np.isnan(returns)]  # Remove NaN values
    
    if len(returns) == 0:
        return _empty_metrics_dict()
    
    # Basic statistics
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns, fisher=True)  # Excess kurtosis
    
    # Annualized metrics
    annualized_return = mean_return * periods_per_year
    annualized_vol = std_return * np.sqrt(periods_per_year)
    
    # Risk-adjusted metrics
    excess_return = annualized_return - risk_free_rate
    sharpe_ratio = excess_return / annualized_vol if annualized_vol > 0 else 0
    
    # Downside risk metrics
    negative_returns = returns[returns < 0]
    downside_deviation = np.std(negative_returns, ddof=1) if len(negative_returns) > 0 else 0
    annualized_downside_vol = downside_deviation * np.sqrt(periods_per_year)
    sortino_ratio = excess_return / annualized_downside_vol if annualized_downside_vol > 0 else np.inf
    
    # Hit rate
    positive_periods = np.sum(returns > 0)
    hit_rate = positive_periods / len(returns)
    
    # Value at Risk (5%)
    var_95 = np.percentile(returns, 5)
    cvar_95 = np.mean(returns[returns <= var_95]) if np.any(returns <= var_95) else var_95
    
    return {
        'total_periods': len(returns),
        'mean_return': float(mean_return),
        'std_return': float(std_return),
        'annualized_return': float(annualized_return),
        'annualized_volatility': float(annualized_vol),
        'sharpe_ratio': float(sharpe_ratio),
        'sortino_ratio': float(sortino_ratio),
        'hit_rate': float(hit_rate),
        'skewness': float(skewness),
        'kurtosis': float(kurtosis),
        'var_95': float(var_95),
        'cvar_95': float(cvar_95),
        'annualized_downside_vol': float(annualized_downside_vol)
    }


def calculate_drawdown_metrics(
    cumulative_returns: Union[pd.Series, np.ndarray],
    returns: Optional[Union[pd.Series, np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Calculate drawdown-based risk metrics.
    
    Parameters
    ----------
    cumulative_returns : pd.Series or np.ndarray
        Cumulative return series (wealth index)
    returns : pd.Series or np.ndarray, optional
        Period returns for recovery time calculation
        
    Returns
    -------
    Dict[str, Any]
        Drawdown metrics including max drawdown, recovery times, and Calmar ratio
    """
    if isinstance(cumulative_returns, pd.Series):
        cum_returns = cumulative_returns.values
        dates = cumulative_returns.index if hasattr(cumulative_returns, 'index') else None
    else:
        cum_returns = cumulative_returns
        dates = None
    
    if len(cum_returns) == 0:
        return _empty_drawdown_dict()
    
    # Calculate running maximum (peak)
    running_max = np.maximum.accumulate(cum_returns)
    
    # Drawdown series
    drawdown = (cum_returns - running_max) / running_max
    
    # Maximum drawdown
    max_drawdown = np.min(drawdown)
    max_dd_idx = np.argmin(drawdown)
    
    # Current drawdown
    current_drawdown = drawdown[-1]
    
    # Drawdown duration analysis
    in_drawdown = drawdown < -1e-6  # Tolerance for floating point
    drawdown_periods = _calculate_drawdown_periods(in_drawdown)
    
    max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
    avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
    
    # Recovery analysis
    recovery_info = _calculate_recovery_times(drawdown, dates)
    
    # Calmar ratio (annualized return / max drawdown)
    if returns is not None:
        if isinstance(returns, pd.Series):
            returns = returns.values
        annual_return = np.mean(returns) * TRADING_DAYS_PER_YEAR
        calmar_ratio = abs(annual_return / max_drawdown) if max_drawdown < 0 else np.inf
    else:
        calmar_ratio = np.nan
    
    return {
        'max_drawdown': float(max_drawdown),
        'current_drawdown': float(current_drawdown), 
        'max_drawdown_duration': int(max_drawdown_duration),
        'avg_drawdown_duration': float(avg_drawdown_duration),
        'num_drawdown_periods': len(drawdown_periods),
        'calmar_ratio': float(calmar_ratio),
        'recovery_info': recovery_info,
        'drawdown_series': drawdown
    }


def calculate_rolling_metrics(
    returns: pd.Series,
    window: int = 252,
    metrics: list = ['sharpe_ratio', 'volatility', 'max_drawdown']
) -> pd.DataFrame:
    """
    Calculate rolling performance metrics over time.
    
    Parameters
    ----------
    returns : pd.Series
        Return time series with date index
    window : int, default 252
        Rolling window size in periods
    metrics : list, default ['sharpe_ratio', 'volatility', 'max_drawdown']
        Metrics to calculate
        
    Returns
    -------
    pd.DataFrame
        Rolling metrics with same date index as returns
    """
    if len(returns) < window:
        warnings.warn(f"Insufficient data for rolling metrics: {len(returns)} < {window}")
        return pd.DataFrame(index=returns.index)
    
    results = {}
    
    if 'sharpe_ratio' in metrics:
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        results['sharpe_ratio'] = (rolling_mean * TRADING_DAYS_PER_YEAR) / (rolling_std * np.sqrt(TRADING_DAYS_PER_YEAR))
    
    if 'volatility' in metrics:
        results['volatility'] = returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    if 'max_drawdown' in metrics:
        cum_returns = (1 + returns).cumprod()
        results['max_drawdown'] = cum_returns.rolling(window).apply(
            lambda x: _rolling_max_drawdown(x.values), raw=False
        )
    
    if 'hit_rate' in metrics:
        results['hit_rate'] = (returns > 0).rolling(window).mean()
    
    return pd.DataFrame(results, index=returns.index)


def calculate_attribution_metrics(
    portfolio_weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    benchmark_weights: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Calculate return attribution metrics.
    
    Decomposes portfolio performance into asset contributions
    and active management effects.
    
    Parameters
    ---------- 
    portfolio_weights : pd.DataFrame
        Portfolio weights over time
    asset_returns : pd.DataFrame
        Asset returns over same period
    benchmark_weights : pd.DataFrame, optional
        Benchmark weights for active attribution
        
    Returns
    -------
    Dict[str, Any]
        Attribution analysis results
    """
    # Align data
    common_dates = portfolio_weights.index.intersection(asset_returns.index)
    weights = portfolio_weights.loc[common_dates]
    returns = asset_returns.loc[common_dates]
    
    if len(common_dates) == 0:
        return {'error': 'No overlapping dates between weights and returns'}
    
    # Asset contribution = weight * return
    asset_contributions = weights.shift(1) * returns  # Use previous day's weights
    asset_contributions = asset_contributions.dropna()
    
    # Total portfolio return
    portfolio_returns = asset_contributions.sum(axis=1)
    
    # Average asset contributions
    avg_contributions = asset_contributions.mean()
    contribution_vol = asset_contributions.std()
    
    results = {
        'portfolio_returns': portfolio_returns,
        'asset_contributions': asset_contributions,
        'avg_contributions': avg_contributions,
        'contribution_volatility': contribution_vol,
        'contribution_sharpe': avg_contributions / contribution_vol
    }
    
    # Active attribution vs benchmark
    if benchmark_weights is not None:
        bench_weights = benchmark_weights.loc[common_dates]
        active_weights = weights - bench_weights.reindex_like(weights, fill_value=0)
        active_returns = (active_weights.shift(1) * returns).dropna().sum(axis=1)
        
        results.update({
            'active_weights': active_weights,
            'active_returns': active_returns,
            'tracking_error': active_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR),
            'information_ratio': (active_returns.mean() * TRADING_DAYS_PER_YEAR) / (active_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        })
    
    return results


def statistical_tests(
    returns1: np.ndarray,
    returns2: np.ndarray,
    test_type: str = 'ttest'
) -> Dict[str, Any]:
    """
    Perform statistical tests comparing two return series.
    
    Parameters
    ----------
    returns1, returns2 : np.ndarray
        Return series to compare
    test_type : str, default 'ttest'
        Type of test: 'ttest', 'wilcoxon', or 'ks'
        
    Returns
    -------
    Dict[str, Any]
        Test results including statistics and p-values
    """
    returns1 = returns1[~np.isnan(returns1)]
    returns2 = returns2[~np.isnan(returns2)]
    
    results = {
        'n1': len(returns1),
        'n2': len(returns2),
        'mean_diff': np.mean(returns1) - np.mean(returns2)
    }
    
    if test_type == 'ttest':
        stat, pvalue = stats.ttest_ind(returns1, returns2, equal_var=False)
        results.update({'statistic': stat, 'p_value': pvalue, 'test': 'Welch t-test'})
        
    elif test_type == 'wilcoxon':
        if len(returns1) == len(returns2):
            stat, pvalue = stats.wilcoxon(returns1, returns2)
            results.update({'statistic': stat, 'p_value': pvalue, 'test': 'Wilcoxon signed-rank'})
        else:
            stat, pvalue = stats.mannwhitneyu(returns1, returns2)
            results.update({'statistic': stat, 'p_value': pvalue, 'test': 'Mann-Whitney U'})
            
    elif test_type == 'ks':
        stat, pvalue = stats.ks_2samp(returns1, returns2)
        results.update({'statistic': stat, 'p_value': pvalue, 'test': 'Kolmogorov-Smirnov'})
        
    return results


# Helper functions

def _empty_metrics_dict() -> Dict[str, float]:
    """Return empty metrics dictionary with NaN values."""
    return {
        'total_periods': 0,
        'mean_return': np.nan,
        'std_return': np.nan,
        'annualized_return': np.nan,
        'annualized_volatility': np.nan,
        'sharpe_ratio': np.nan,
        'sortino_ratio': np.nan,
        'hit_rate': np.nan,
        'skewness': np.nan,
        'kurtosis': np.nan,
        'var_95': np.nan,
        'cvar_95': np.nan,
        'annualized_downside_vol': np.nan
    }


def _empty_drawdown_dict() -> Dict[str, Any]:
    """Return empty drawdown dictionary."""
    return {
        'max_drawdown': np.nan,
        'current_drawdown': np.nan,
        'max_drawdown_duration': 0,
        'avg_drawdown_duration': np.nan,
        'num_drawdown_periods': 0,
        'calmar_ratio': np.nan,
        'recovery_info': {},
        'drawdown_series': np.array([])
    }


def _calculate_drawdown_periods(in_drawdown: np.ndarray) -> list:
    """Calculate lengths of consecutive drawdown periods."""
    periods = []
    current_length = 0
    
    for is_dd in in_drawdown:
        if is_dd:
            current_length += 1
        else:
            if current_length > 0:
                periods.append(current_length)
                current_length = 0
    
    # Add final period if still in drawdown
    if current_length > 0:
        periods.append(current_length)
    
    return periods


def _calculate_recovery_times(drawdown: np.ndarray, dates: Optional[pd.DatetimeIndex]) -> Dict[str, Any]:
    """Calculate recovery times from drawdown periods."""
    recovery_info = {
        'avg_recovery_time': np.nan,
        'max_recovery_time': np.nan,
        'current_recovery_time': np.nan
    }
    
    # Find all local minima (drawdown troughs)
    troughs = []
    for i in range(1, len(drawdown) - 1):
        if drawdown[i] < drawdown[i-1] and drawdown[i] < drawdown[i+1]:
            troughs.append(i)
    
    recovery_times = []
    for trough_idx in troughs:
        # Find next recovery to previous high
        for j in range(trough_idx + 1, len(drawdown)):
            if drawdown[j] >= -1e-6:  # Back to high (tolerance for floating point)
                recovery_times.append(j - trough_idx)
                break
    
    if recovery_times:
        recovery_info['avg_recovery_time'] = np.mean(recovery_times)
        recovery_info['max_recovery_time'] = max(recovery_times)
    
    return recovery_info


def _rolling_max_drawdown(cum_returns: np.ndarray) -> float:
    """Calculate max drawdown for a rolling window."""
    if len(cum_returns) == 0:
        return np.nan
    
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - running_max) / running_max
    return np.min(drawdown)