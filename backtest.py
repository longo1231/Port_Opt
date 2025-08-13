"""
Walk-Forward Backtest Engine for Minimum Variance Portfolio Optimizer.

Implements true historical backtesting with periodic rebalancing based on 
information available at each point in time. No look-ahead bias.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings

from estimators import estimate_covariance_matrix
from optimizer import optimize_min_variance, analyze_min_variance_portfolio, calculate_leveraged_portfolio
from config import ASSETS, SIGMA_WINDOW, RHO_WINDOW, TRADING_DAYS_PER_YEAR


def generate_rebalance_dates(start_date: str, end_date: str, 
                           frequency: str = 'weekly') -> List[str]:
    """
    Generate rebalancing dates for the backtest period.
    
    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str  
        End date in 'YYYY-MM-DD' format
    frequency : str, default 'weekly'
        Rebalancing frequency ('weekly', 'monthly', 'quarterly')
        
    Returns
    -------
    List[str]
        List of rebalancing dates in 'YYYY-MM-DD' format
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    if frequency == 'weekly':
        # Rebalance every Friday (or last available day of week)
        dates = pd.date_range(start=start, end=end, freq='W-FRI')
    elif frequency == 'monthly':
        # Rebalance on last day of each month
        dates = pd.date_range(start=start, end=end, freq='ME')
    elif frequency == 'quarterly':
        # Rebalance on last day of each quarter
        dates = pd.date_range(start=start, end=end, freq='Q')
    else:
        raise ValueError(f"Unsupported frequency: {frequency}")
    
    # Convert to string format
    return [date.strftime('%Y-%m-%d') for date in dates]


def calculate_period_performance(returns_data: pd.DataFrame, weights: np.ndarray,
                               start_date: str, end_date: str) -> Dict:
    """
    Calculate portfolio performance for a specific period.
    
    Parameters
    ----------
    returns_data : pd.DataFrame
        Daily returns data with assets as columns
    weights : np.ndarray
        Portfolio weights for the period
    start_date : str
        Period start date
    end_date : str
        Period end date
        
    Returns
    -------
    Dict
        Performance metrics for the period
    """
    # Filter data for the specific period
    period_data = returns_data.loc[start_date:end_date]
    
    if len(period_data) == 0:
        return {
            'period_return': 0.0,
            'period_volatility': 0.0,
            'n_days': 0
        }
    
    # Calculate portfolio returns for the period
    portfolio_returns = (period_data * weights).sum(axis=1)
    
    # Period performance metrics
    period_return = (1 + portfolio_returns).prod() - 1  # Compound return
    period_volatility = portfolio_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if len(portfolio_returns) > 1 else 0.0
    
    return {
        'period_return': period_return,
        'period_volatility': period_volatility,
        'n_days': len(period_data),
        'daily_returns': portfolio_returns
    }


def walk_forward_backtest(returns_data: pd.DataFrame, start_date: str, end_date: str,
                         rebalance_freq: str = 'weekly', min_history_days: int = None,
                         sigma_window: int = None, rho_window: int = None, 
                         target_volatility: float = None) -> Dict:
    """
    Execute walk-forward backtest with periodic rebalancing.
    
    For each rebalancing period:
    1. Use only data available up to that date
    2. Estimate covariance matrix with rolling windows  
    3. Optimize portfolio weights
    4. Apply weights to next period until next rebalance
    5. Track performance and portfolio evolution
    
    Parameters
    ----------
    returns_data : pd.DataFrame
        Daily returns data with assets as columns, dates as index
    start_date : str
        Backtest start date in 'YYYY-MM-DD' format
    end_date : str
        Backtest end date in 'YYYY-MM-DD' format
    rebalance_freq : str, default 'weekly'
        Rebalancing frequency ('weekly', 'monthly', 'quarterly')
    min_history_days : int, optional
        Minimum days of history required before first rebalance
    sigma_window : int, optional
        Rolling window for volatility estimation, defaults to config value
    rho_window : int, optional
        Rolling window for correlation estimation, defaults to config value
        
    Returns
    -------
    Dict
        Comprehensive backtest results including performance metrics,
        weight history, and comparison with static approach
    """
    # Set default windows if not provided
    if sigma_window is None:
        sigma_window = SIGMA_WINDOW
    if rho_window is None:
        rho_window = RHO_WINDOW
        
    # Set minimum history requirement
    if min_history_days is None:
        min_history_days = max(sigma_window, rho_window)
    
    # Generate rebalancing dates
    rebalance_dates = generate_rebalance_dates(start_date, end_date, rebalance_freq)
    
    # Find actual backtest start date (need sufficient history)
    backtest_start = pd.to_datetime(start_date)
    data_start = returns_data.index[0]
    
    # Ensure we have enough historical data
    required_start = backtest_start - timedelta(days=min_history_days + 10)  # Buffer for weekends
    if data_start > required_start:
        warnings.warn(f"Insufficient historical data. Need data from {required_start}, "
                     f"but earliest available is {data_start}")
        # Adjust min_history_days to what's actually available
        available_days = (backtest_start - data_start).days - 10  # Buffer for weekends
        if available_days > 20:  # Minimum reasonable window
            min_history_days = min(min_history_days, available_days)
            print(f"Adjusting minimum history requirement to {min_history_days} days based on available data")
    
    # Initialize tracking structures
    weight_history = []
    performance_periods = []
    portfolio_values = [1.0]  # Start with $1 invested
    portfolio_dates = [backtest_start]
    optimization_errors = []
    
    # Main backtest loop
    print(f"Starting backtest from {start_date} to {end_date} with {len(rebalance_dates)} rebalancing dates")
    for i, rebal_date in enumerate(rebalance_dates):
        rebal_datetime = pd.to_datetime(rebal_date)
        
        # Skip if rebalance date is before our start date
        if rebal_datetime < backtest_start:
            continue
            
        # Get all data available up to this rebalancing date (no look-ahead bias)
        available_data = returns_data[returns_data.index <= rebal_datetime]
        
        # Check if we have sufficient history for optimization
        if len(available_data) < min_history_days:
            print(f"Skipping {rebal_date}: insufficient data ({len(available_data)} < {min_history_days} days)")
            continue
            
        try:
            # Estimate covariance matrix using only available data
            covariance_matrix = estimate_covariance_matrix(
                available_data, 
                sigma_window=sigma_window, 
                rho_window=rho_window
            )
            
            # Optimize portfolio weights
            optimization_result = optimize_min_variance(covariance_matrix, exclude_cash=True)
            
            if not optimization_result['success']:
                optimization_errors.append({
                    'date': rebal_date,
                    'error': optimization_result.get('error', 'Unknown error')
                })
                # Use equal weights as fallback
                weights = np.array([0.33, 0.33, 0.34, 0.0])  # Equal weight risky assets
                print(f"Optimization failed on {rebal_date}, using equal weights")
            else:
                weights = optimization_result['weights']
                
                # Apply leverage if target volatility is specified
                if target_volatility is not None and target_volatility > 0:
                    try:
                        # Extract risky covariance matrix (SPY, TLT, GLD only)
                        risky_cov_matrix = covariance_matrix[:3, :3]
                        
                        # Calculate leveraged portfolio
                        leverage_result = calculate_leveraged_portfolio(
                            risky_cov_matrix, 
                            target_volatility, 
                            max_leverage=3.0
                        )
                        
                        if leverage_result['success']:
                            weights = leverage_result['final_weights']
                        else:
                            print(f"Leverage calculation failed on {rebal_date}: {leverage_result['error']}")
                            # Keep original weights as fallback
                            
                    except Exception as lever_e:
                        print(f"Leverage error on {rebal_date}: {lever_e}")
                        # Keep original weights as fallback
                
        except Exception as e:
            optimization_errors.append({
                'date': rebal_date,
                'error': str(e)
            })
            # Use equal weights as fallback
            weights = np.array([0.33, 0.33, 0.34, 0.0])
            print(f"Error on {rebal_date}: {e}, using equal weights")
        
        # Store weight history
        weight_history.append({
            'date': rebal_date,
            'weights': weights.copy(),
            'SPY': weights[0],
            'TLT': weights[1], 
            'GLD': weights[2],
            'Cash': weights[3]
        })
        
        # Calculate performance until next rebalance (or end date)
        period_start = rebal_datetime
        if i < len(rebalance_dates) - 1:
            period_end = pd.to_datetime(rebalance_dates[i + 1]) - timedelta(days=1)
        else:
            period_end = pd.to_datetime(end_date)
            
        # Get performance for this holding period
        period_perf = calculate_period_performance(
            returns_data, weights, 
            period_start.strftime('%Y-%m-%d'), 
            period_end.strftime('%Y-%m-%d')
        )
        
        period_perf['rebalance_date'] = rebal_date
        period_perf['weights'] = weights.copy()
        performance_periods.append(period_perf)
        
        # Update portfolio value
        if len(period_perf['daily_returns']) > 0:
            # Compound the daily returns for this period using actual data dates
            period_returns = period_perf['daily_returns']
            for date, daily_return in period_returns.items():
                new_value = portfolio_values[-1] * (1 + daily_return)
                portfolio_values.append(new_value)
                portfolio_dates.append(date)
    
    # Create portfolio value series
    portfolio_series = pd.Series(portfolio_values[1:], index=portfolio_dates[1:])
    
    print(f"Backtest completed. Portfolio series from {portfolio_series.index[0] if len(portfolio_series) > 0 else 'N/A'} to {portfolio_series.index[-1] if len(portfolio_series) > 0 else 'N/A'}")
    
    # Calculate overall backtest metrics
    total_return = portfolio_series.iloc[-1] - 1 if len(portfolio_series) > 0 else 0.0
    
    # Calculate daily returns from portfolio values
    daily_returns = portfolio_series.pct_change().dropna()
    
    if len(daily_returns) > 0:
        annualized_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / len(daily_returns)) - 1
        volatility = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        sharpe_ratio = (daily_returns.mean() * TRADING_DAYS_PER_YEAR) / volatility if volatility > 0 else 0.0
        
        # Calculate max drawdown
        cumulative = (1 + daily_returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()
    else:
        annualized_return = 0.0
        volatility = 0.0
        sharpe_ratio = 0.0
        max_drawdown = 0.0
    
    # Calculate turnover (average weight change per rebalance)
    turnover_values = []
    for i in range(1, len(weight_history)):
        prev_weights = weight_history[i-1]['weights']
        curr_weights = weight_history[i]['weights']
        turnover = np.sum(np.abs(curr_weights - prev_weights)) / 2
        turnover_values.append(turnover)
    
    average_turnover = np.mean(turnover_values) if turnover_values else 0.0
    
    # Prepare results
    results = {
        # Core performance data
        'portfolio_values': portfolio_series,
        'portfolio_returns': daily_returns,
        'weight_history': pd.DataFrame(weight_history),
        'rebalance_dates': rebalance_dates,
        
        # Performance metrics
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.inf,
        
        # Portfolio analytics
        'average_turnover': average_turnover,
        'n_rebalances': len(weight_history),
        'n_optimization_errors': len(optimization_errors),
        'optimization_errors': optimization_errors,
        
        # Backtest metadata
        'backtest_start': start_date,
        'backtest_end': end_date,
        'rebalance_frequency': rebalance_freq,
        'min_history_days': min_history_days,
        'performance_periods': performance_periods
    }
    
    return results


def compare_with_static_weights(backtest_results: Dict, returns_data: pd.DataFrame, 
                              static_weights: np.ndarray) -> Dict:
    """
    Compare walk-forward backtest results with static weight approach.
    
    Parameters
    ----------
    backtest_results : Dict
        Results from walk_forward_backtest()
    returns_data : pd.DataFrame
        Same returns data used for backtest
    static_weights : np.ndarray
        Static weights to compare against
        
    Returns
    -------
    Dict
        Comparison metrics between dynamic and static approaches
    """
    # Get backtest period
    start_date = backtest_results['backtest_start']
    end_date = backtest_results['backtest_end']
    
    # Calculate static portfolio performance over same period
    period_data = returns_data.loc[start_date:end_date]
    static_returns = (period_data * static_weights).sum(axis=1)
    static_cumulative = (1 + static_returns).cumprod()
    
    # Static performance metrics
    static_total_return = static_cumulative.iloc[-1] - 1
    static_volatility = static_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    static_sharpe = (static_returns.mean() * TRADING_DAYS_PER_YEAR) / static_volatility if static_volatility > 0 else 0.0
    
    # Static max drawdown
    static_peak = static_cumulative.cummax()
    static_drawdown = (static_cumulative - static_peak) / static_peak
    static_max_drawdown = static_drawdown.min()
    
    # Comparison
    comparison = {
        'dynamic_total_return': backtest_results['total_return'],
        'static_total_return': static_total_return,
        'return_difference': backtest_results['total_return'] - static_total_return,
        
        'dynamic_volatility': backtest_results['volatility'],
        'static_volatility': static_volatility,
        'volatility_difference': backtest_results['volatility'] - static_volatility,
        
        'dynamic_sharpe': backtest_results['sharpe_ratio'],
        'static_sharpe': static_sharpe,
        'sharpe_difference': backtest_results['sharpe_ratio'] - static_sharpe,
        
        'dynamic_max_drawdown': backtest_results['max_drawdown'],
        'static_max_drawdown': static_max_drawdown,
        'drawdown_difference': backtest_results['max_drawdown'] - static_max_drawdown,
        
        'turnover_cost': backtest_results['average_turnover'],
        'n_rebalances': backtest_results['n_rebalances']
    }
    
    return comparison