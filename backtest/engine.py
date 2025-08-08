"""
Backtesting engine for portfolio optimization strategies.

Simulates daily portfolio rebalancing with transaction costs, tracks performance
metrics, and provides detailed analysis of strategy effectiveness.

Key features:
- Daily/weekly/monthly rebalancing frequencies
- Realistic transaction costs and turnover tracking  
- Performance attribution and drawdown analysis
- Support for different estimation windows and methods
- Out-of-sample validation with rolling parameters
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
import warnings

from data.loader import DataLoader
from features.estimators import estimate_all_parameters
from opt.optimizer import PortfolioOptimizer, compute_portfolio_attribution
from config import (
    ASSETS, DEFAULT_LOOKBACK_DAYS, DEFAULT_MU_SIGMA_HALFLIFE,
    DEFAULT_CORR_HALFLIFE, DEFAULT_SHRINKAGE, DEFAULT_TRANSACTION_COST_BPS,
    DEFAULT_TURNOVER_PENALTY, DEFAULT_OPTIMIZER_METHOD, TRADING_DAYS_PER_YEAR
)


class BacktestEngine:
    """
    Portfolio backtesting engine with realistic transaction costs.
    
    Simulates the complete portfolio management process including
    parameter estimation, optimization, and rebalancing with costs.
    """
    
    def __init__(
        self,
        optimizer: PortfolioOptimizer,
        estimation_params: Optional[Dict[str, Any]] = None,
        rebalance_freq: str = "daily",
        min_history_days: int = 50
    ):
        """
        Initialize backtesting engine.
        
        Parameters
        ----------
        optimizer : PortfolioOptimizer
            Configured portfolio optimizer
        estimation_params : dict, optional
            Parameters for return/risk estimation
        rebalance_freq : str, default "daily"
            Rebalancing frequency: "daily", "weekly", or "monthly"  
        min_history_days : int, default 50
            Minimum history required before starting optimization
        """
        self.optimizer = optimizer
        self.rebalance_freq = rebalance_freq
        self.min_history_days = min_history_days
        
        # Default estimation parameters
        self.estimation_params = {
            'mu_method': 'ewma',
            'vol_method': 'ewma',
            'corr_method': 'ewma', 
            'mu_window': DEFAULT_LOOKBACK_DAYS,
            'vol_window': DEFAULT_LOOKBACK_DAYS,
            'corr_window': DEFAULT_LOOKBACK_DAYS,
            'mu_halflife': DEFAULT_MU_SIGMA_HALFLIFE,
            'vol_halflife': DEFAULT_MU_SIGMA_HALFLIFE,
            'corr_halflife': DEFAULT_CORR_HALFLIFE,
            'shrinkage': DEFAULT_SHRINKAGE
        }
        
        if estimation_params:
            self.estimation_params.update(estimation_params)
    
    def run_backtest(
        self,
        returns_data: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        optimization_method: str = DEFAULT_OPTIMIZER_METHOD
    ) -> Dict[str, Any]:
        """
        Run complete backtest simulation.
        
        Parameters
        ----------
        returns_data : pd.DataFrame
            Historical returns data with assets as columns
        start_date : str, optional
            Backtest start date (YYYY-MM-DD format)
        end_date : str, optional
            Backtest end date (YYYY-MM-DD format)
        optimization_method : str, default from config
            Optimization method: "quadratic" or "monte_carlo"
            
        Returns
        -------
        Dict[str, Any]
            Complete backtest results including portfolio weights,
            returns, costs, and performance metrics
        """
        # Filter date range
        if start_date:
            returns_data = returns_data.loc[returns_data.index >= start_date]
        if end_date:
            returns_data = returns_data.loc[returns_data.index <= end_date]
        
        if len(returns_data) < self.min_history_days:
            raise ValueError(f"Insufficient data: {len(returns_data)} days < {self.min_history_days}")
        
        # Initialize tracking variables
        n_assets = len(ASSETS)
        dates = returns_data.index[self.min_history_days:]  # Start after minimum history
        
        # Results storage
        portfolio_weights = pd.DataFrame(index=dates, columns=ASSETS)
        portfolio_returns = pd.Series(index=dates, dtype=float)
        transaction_costs = pd.Series(index=dates, dtype=float)  
        turnover = pd.Series(index=dates, dtype=float)
        optimization_results = []
        
        # Current portfolio state
        current_weights = np.ones(n_assets) / n_assets  # Equal weight start
        
        # Rebalancing schedule
        rebalance_dates = self._get_rebalance_dates(dates)
        
        # Main simulation loop
        for i, date in enumerate(dates):
            # Get historical data up to current date
            hist_data = returns_data.loc[:date].iloc[:-1]  # Exclude current day
            
            # Check if we should rebalance
            should_rebalance = (date in rebalance_dates) or (i == 0)
            
            if should_rebalance and len(hist_data) >= self.min_history_days:
                # Estimate parameters
                try:
                    mu, cov_matrix, volatilities = estimate_all_parameters(
                        hist_data, **self.estimation_params
                    )
                    
                    # Optimize portfolio
                    opt_result = self.optimizer.optimize(
                        mu, cov_matrix, optimization_method, current_weights
                    )
                    
                    if opt_result['success']:
                        new_weights = opt_result['weights']
                        
                        # Calculate transaction costs and turnover
                        weight_change = np.abs(new_weights - current_weights)
                        daily_turnover = np.sum(weight_change)
                        daily_cost = (self.optimizer.transaction_cost_bps / 10000) * daily_turnover
                        
                        # Update current weights
                        current_weights = new_weights.copy()
                        
                        # Store optimization results
                        optimization_results.append({
                            'date': date,
                            'objective_value': opt_result['objective_value'],
                            'method': opt_result['method'],
                            'success': True
                        })
                    else:
                        # Optimization failed, keep current weights
                        daily_turnover = 0.0
                        daily_cost = 0.0
                        warnings.warn(f"Optimization failed on {date}")
                        
                        optimization_results.append({
                            'date': date,
                            'objective_value': np.nan,
                            'method': optimization_method,
                            'success': False
                        })
                        
                except Exception as e:
                    # Parameter estimation or optimization error
                    daily_turnover = 0.0
                    daily_cost = 0.0
                    warnings.warn(f"Error on {date}: {str(e)}")
                    
                    optimization_results.append({
                        'date': date,
                        'objective_value': np.nan,
                        'method': optimization_method,
                        'success': False
                    })
            else:
                # No rebalancing
                daily_turnover = 0.0
                daily_cost = 0.0
            
            # Calculate portfolio return for the day
            day_returns = returns_data.loc[date]
            portfolio_return = current_weights @ day_returns.values - daily_cost
            
            # Store results
            portfolio_weights.loc[date] = current_weights
            portfolio_returns.loc[date] = portfolio_return
            transaction_costs.loc[date] = daily_cost
            turnover.loc[date] = daily_turnover
        
        # Compile results
        results = {
            'portfolio_weights': portfolio_weights,
            'portfolio_returns': portfolio_returns,
            'transaction_costs': transaction_costs,
            'turnover': turnover,
            'optimization_results': optimization_results,
            'cumulative_returns': (1 + portfolio_returns).cumprod(),
            'settings': {
                'optimization_method': optimization_method,
                'rebalance_freq': self.rebalance_freq,
                'estimation_params': self.estimation_params,
                'transaction_cost_bps': self.optimizer.transaction_cost_bps,
                'turnover_penalty': self.optimizer.turnover_penalty
            }
        }
        
        return results
    
    def _get_rebalance_dates(self, all_dates: pd.DatetimeIndex) -> List[pd.Timestamp]:
        """
        Generate rebalancing dates based on frequency.
        
        Parameters
        ----------
        all_dates : pd.DatetimeIndex
            All available trading dates
            
        Returns
        -------
        List[pd.Timestamp]
            Dates on which portfolio should be rebalanced
        """
        if self.rebalance_freq == "daily":
            return list(all_dates)
        
        elif self.rebalance_freq == "weekly":
            # Rebalance on Mondays (or first trading day of week)
            weekly_dates = []
            current_week = None
            
            for date in all_dates:
                week = date.isocalendar()[1]  # Week number
                if week != current_week:
                    weekly_dates.append(date)
                    current_week = week
            
            return weekly_dates
        
        elif self.rebalance_freq == "monthly":
            # Rebalance on first trading day of each month
            monthly_dates = []
            current_month = None
            
            for date in all_dates:
                month = date.month
                if month != current_month:
                    monthly_dates.append(date)
                    current_month = month
            
            return monthly_dates
        
        else:
            raise ValueError(f"Unknown rebalance frequency: {self.rebalance_freq}")


def analyze_backtest_performance(backtest_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute comprehensive performance metrics from backtest results.
    
    Parameters
    ----------
    backtest_results : dict
        Results from BacktestEngine.run_backtest()
        
    Returns
    -------
    Dict[str, Any]
        Performance analysis including returns, risk, and cost metrics
    """
    portfolio_returns = backtest_results['portfolio_returns']
    transaction_costs = backtest_results['transaction_costs']
    turnover = backtest_results['turnover']
    cum_returns = backtest_results['cumulative_returns']
    
    # Basic return metrics
    total_return = cum_returns.iloc[-1] - 1
    annualized_return = (cum_returns.iloc[-1] ** (TRADING_DAYS_PER_YEAR / len(portfolio_returns))) - 1
    annualized_vol = portfolio_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
    
    # Drawdown analysis
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Cost analysis
    total_costs = transaction_costs.sum()
    avg_daily_turnover = turnover.mean()
    cost_drag = total_costs / len(portfolio_returns) * TRADING_DAYS_PER_YEAR
    
    # Risk metrics
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if len(downside_returns) > 0 else 0
    sortino_ratio = annualized_return / downside_vol if downside_vol > 0 else np.inf
    
    # Hit rate
    positive_days = (portfolio_returns > 0).sum()
    hit_rate = positive_days / len(portfolio_returns)
    
    return {
        'total_return': float(total_return),
        'annualized_return': float(annualized_return),
        'annualized_volatility': float(annualized_vol),
        'sharpe_ratio': float(sharpe_ratio),
        'sortino_ratio': float(sortino_ratio),
        'max_drawdown': float(max_drawdown),
        'hit_rate': float(hit_rate),
        'total_transaction_costs': float(total_costs),
        'annualized_cost_drag': float(cost_drag),
        'avg_daily_turnover': float(avg_daily_turnover),
        'n_observations': len(portfolio_returns)
    }


def compare_strategies(
    returns_data: pd.DataFrame,
    strategies: Dict[str, Dict[str, Any]],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Compare multiple portfolio strategies side-by-side.
    
    Parameters
    ----------
    returns_data : pd.DataFrame
        Historical returns data
    strategies : dict
        Dictionary mapping strategy names to their configuration parameters
    start_date, end_date : str, optional
        Date range for comparison
        
    Returns
    -------
    pd.DataFrame
        Comparison table with performance metrics for each strategy
    """
    results = {}
    
    for name, config in strategies.items():
        # Create optimizer and engine
        optimizer = PortfolioOptimizer(
            transaction_cost_bps=config.get('transaction_cost_bps', DEFAULT_TRANSACTION_COST_BPS),
            turnover_penalty=config.get('turnover_penalty', DEFAULT_TURNOVER_PENALTY)
        )
        
        engine = BacktestEngine(
            optimizer=optimizer,
            estimation_params=config.get('estimation_params', {}),
            rebalance_freq=config.get('rebalance_freq', 'daily')
        )
        
        # Run backtest
        try:
            backtest_results = engine.run_backtest(
                returns_data,
                start_date=start_date,
                end_date=end_date,
                optimization_method=config.get('optimization_method', 'quadratic')
            )
            
            # Analyze performance
            performance = analyze_backtest_performance(backtest_results)
            results[name] = performance
            
        except Exception as e:
            warnings.warn(f"Strategy {name} failed: {str(e)}")
            results[name] = {metric: np.nan for metric in [
                'total_return', 'annualized_return', 'annualized_volatility',
                'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'hit_rate',
                'total_transaction_costs', 'annualized_cost_drag', 'avg_daily_turnover'
            ]}
    
    return pd.DataFrame(results).T