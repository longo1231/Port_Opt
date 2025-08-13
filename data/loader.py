"""
Data loading module for portfolio optimizer.

Provides functionality to generate simulated returns data for Phase 1 testing
and real market data fetching from Yahoo Finance for Phase 2.

The module handles both:
1. Multivariate normal simulation of correlated asset returns
2. Historical data fetching and preprocessing for SPY, TLT, GLD
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
import yfinance as yf
from datetime import datetime, timedelta
import warnings

from config import (
    ASSETS, RISK_FREE_RATE as DEFAULT_RISK_FREE_RATE, 
    TRADING_DAYS_PER_YEAR
)
# Default simulation parameters for risky assets only (SPY, TLT, GLD)
DEFAULT_SIM_MU = np.array([0.08, 0.03, 0.05])  # SPY, TLT, GLD
DEFAULT_SIM_SIGMA = np.array([0.16, 0.05, 0.20])  # SPY, TLT, GLD 
DEFAULT_SIM_CORR = 0.2  # Use scalar correlation for simplicity


class DataLoader:
    """
    Data loading and simulation class for portfolio optimization.
    
    Handles both simulated data generation for testing and real market
    data fetching for live optimization.
    """
    
    def __init__(self, random_seed: Optional[int] = 42):
        """
        Initialize data loader.
        
        Parameters
        ----------
        random_seed : int, optional
            Random seed for reproducible simulations, by default 42
        """
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def generate_simulated_returns(
        self, 
        n_days: int = 500,
        mu: Optional[np.ndarray] = None,
        sigma: Optional[np.ndarray] = None,
        correlation: float = DEFAULT_SIM_CORR,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        start_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate simulated daily returns for portfolio assets.
        
        Creates multivariate normal returns for SPX, TLT, GLD with specified
        correlation structure, plus risk-free cash returns.
        
        Parameters
        ----------
        n_days : int, default 500
            Number of trading days to simulate
        mu : np.ndarray, optional
            Annualized expected returns for risky assets [SPX, TLT, GLD]
        sigma : np.ndarray, optional
            Annualized volatilities for risky assets [SPX, TLT, GLD]
        correlation : float, default from config
            Pairwise correlation between risky assets
        risk_free_rate : float, default from config
            Annualized risk-free rate for cash
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns [SPX, TLT, GLD, Cash] and DatetimeIndex
            containing daily log returns
        """
        if mu is None:
            mu = np.array(DEFAULT_SIM_MU)
        if sigma is None:
            sigma = np.array(DEFAULT_SIM_SIGMA)
            
        # Convert annualized parameters to daily
        daily_mu = mu / TRADING_DAYS_PER_YEAR
        daily_sigma = sigma / np.sqrt(TRADING_DAYS_PER_YEAR)
        daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
        
        # Build correlation matrix for risky assets
        n_risky = len(mu)
        corr_matrix = np.full((n_risky, n_risky), correlation)
        np.fill_diagonal(corr_matrix, 1.0)
        
        # Build covariance matrix
        cov_matrix = np.outer(daily_sigma, daily_sigma) * corr_matrix
        
        # Generate correlated returns
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
        risky_returns = np.random.multivariate_normal(
            mean=daily_mu,
            cov=cov_matrix,
            size=n_days
        )
        
        # Add cash returns (constant risk-free rate)
        cash_returns = np.full((n_days, 1), daily_rf)
        
        # Combine all returns
        all_returns = np.hstack([risky_returns, cash_returns])
        
        # Create date index
        if start_date is None:
            start_date = '2020-01-01'
        dates = pd.date_range(start=start_date, periods=n_days, freq='B')
        
        return pd.DataFrame(
            data=all_returns,
            index=dates,
            columns=ASSETS
        )
    
    def fetch_market_data(
        self,
        start_date: str,
        end_date: str,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        use_treasury_bills: bool = True
    ) -> pd.DataFrame:
        """
        Fetch real market data from Yahoo Finance.
        
        Downloads adjusted closing prices for SPY, TLT, GLD and computes
        simple returns. Uses Treasury bill ETF (BIL) for cash returns if available.
        
        Parameters
        ----------
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str  
            End date in 'YYYY-MM-DD' format
        risk_free_rate : float, default from config
            Annualized risk-free rate fallback for cash
        use_treasury_bills : bool, default True
            Whether to use Treasury bill ETF data for cash returns
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns [SPY, TLT, GLD, Cash] containing daily simple returns
            
        Raises
        ------
        Exception
            If data fetching fails or insufficient data is available
        """
        # Ticker mapping (now using SPY directly per spec)
        ticker_map = {
            'SPY': 'SPY',  # SPY ETF per spec
            'TLT': 'TLT',  # 20+ year Treasury ETF
            'GLD': 'GLD'   # Gold ETF
        }
        
        try:
            # Download data
            tickers = list(ticker_map.values())
            data = yf.download(
                tickers, 
                start=start_date, 
                end=end_date,
                progress=False
            )
            
            if data.empty:
                raise ValueError("No data returned from Yahoo Finance")
            
            # Handle different data structures
            try:
                if len(tickers) == 1:
                    # Single ticker case
                    if hasattr(data.columns, 'levels'):
                        # Multi-level columns even for single ticker
                        if 'Adj Close' in data.columns.levels[0]:
                            prices = data['Adj Close'].iloc[:, 0].to_frame()
                        else:
                            prices = data['Close'].iloc[:, 0].to_frame()
                        prices.columns = [list(ticker_map.keys())[0]]
                    else:
                        # Simple columns
                        if 'Adj Close' in data.columns:
                            prices = data['Adj Close'].to_frame()
                        else:
                            prices = data['Close'].to_frame()
                        prices.columns = [list(ticker_map.keys())[0]]
                else:
                    # Multiple tickers case
                    if hasattr(data.columns, 'levels') and 'Adj Close' in data.columns.levels[0]:
                        prices = data['Adj Close']
                    elif hasattr(data.columns, 'levels') and 'Close' in data.columns.levels[0]:
                        prices = data['Close']
                    elif 'Adj Close' in data.columns:
                        prices = data[['Adj Close']]
                    else:
                        prices = data[['Close']]
                    
                    # CRITICAL FIX: Map the actual Yahoo column order to our desired order
                    # Yahoo may return columns in different order than we requested
                    original_columns = list(prices.columns)
                    
                    # Create properly ordered DataFrame with our asset names
                    ordered_prices = pd.DataFrame(index=prices.index)
                    for our_name, yahoo_ticker in ticker_map.items():
                        if yahoo_ticker in original_columns:
                            ordered_prices[our_name] = prices[yahoo_ticker]
                        else:
                            print(f"Warning: {yahoo_ticker} not found in Yahoo data")
                    
                    prices = ordered_prices
            
            except Exception as col_error:
                # If column handling fails, try a different approach
                print(f"Column structure: {data.columns}")
                print(f"Data shape: {data.shape}")
                raise ValueError(f"Failed to parse Yahoo Finance data structure: {str(col_error)}")
            
            # Remove any rows with NaN values
            prices = prices.dropna()
            
            if len(prices) < 2:
                raise ValueError("Insufficient data points after cleaning")
            
            # Calculate simple returns: r_t = P_t / P_{t-1} - 1
            simple_returns = prices.pct_change().dropna()
            
            # Add cash returns - try Treasury bills first, fallback to constant rate
            if use_treasury_bills:
                try:
                    # Fetch Treasury bill ETF data (BIL = SPDR 1-3 Month Treasury ETF)
                    treasury_data = yf.download(
                        'BIL', 
                        start=start_date, 
                        end=end_date,
                        progress=False
                    )
                    
                    if not treasury_data.empty:
                        # Handle multi-level columns from yfinance
                        if hasattr(treasury_data.columns, 'levels'):
                            treasury_prices = treasury_data['Close'].iloc[:, 0] if len(treasury_data['Close'].shape) > 1 else treasury_data['Close']
                        else:
                            treasury_prices = treasury_data['Close']
                        
                        # Calculate Treasury bill returns and align with main data
                        treasury_returns = treasury_prices.pct_change().dropna()
                        
                        # Align dates with main returns data
                        aligned_treasury = treasury_returns.reindex(simple_returns.index, method='ffill')
                        
                        # Use Treasury returns where available, fallback to constant rate
                        cash_returns = aligned_treasury.fillna(risk_free_rate / TRADING_DAYS_PER_YEAR)
                        simple_returns['Cash'] = cash_returns
                        
                        print(f"✅ Using Treasury bill returns (BIL ETF) for Cash - {len(treasury_returns)} days available")
                    else:
                        # Fallback to constant rate
                        daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
                        simple_returns['Cash'] = daily_rf
                        print(f"⚠️  Treasury data empty, using constant {risk_free_rate:.1%} rate")
                        
                except Exception as treasury_error:
                    # Fallback to constant rate
                    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
                    simple_returns['Cash'] = daily_rf
                    print(f"⚠️  Treasury fetch failed: {str(treasury_error)[:50]}... Using constant {risk_free_rate:.1%} rate")
            else:
                # Use constant risk-free rate
                daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
                simple_returns['Cash'] = daily_rf
                print(f"ℹ️  Using constant risk-free rate: {risk_free_rate:.1%}")
            
            # Ensure column order matches ASSETS
            return simple_returns[ASSETS]
            
        except Exception as e:
            raise Exception(f"Failed to fetch market data: {str(e)}")
    
    def get_latest_prices(self, tickers: list = None) -> Dict[str, float]:
        """
        Get latest closing prices for market data validation.
        
        Parameters
        ----------
        tickers : list, optional
            List of tickers to fetch, defaults to ['SPY', 'TLT', 'GLD']
            
        Returns
        -------
        Dict[str, float]
            Dictionary mapping tickers to latest prices
        """
        if tickers is None:
            tickers = ['SPY', 'TLT', 'GLD']
            
        try:
            latest_data = yf.download(
                tickers,
                period='5d',
                progress=False
            )
            
            if latest_data.empty:
                return {}
            
            if len(tickers) == 1:
                # Single ticker
                if 'Adj Close' in latest_data.columns:
                    latest_price = latest_data['Adj Close'].iloc[-1]
                else:
                    latest_price = latest_data['Close'].iloc[-1]
                return {tickers[0]: float(latest_price)}
            else:
                # Multiple tickers
                if 'Adj Close' in latest_data.columns.levels[0]:
                    latest_prices = latest_data['Adj Close'].iloc[-1]
                else:
                    latest_prices = latest_data['Close'].iloc[-1]
                return {ticker: float(price) for ticker, price in latest_prices.items() if not pd.isna(price)}
                
        except Exception as e:
            warnings.warn(f"Failed to fetch latest prices: {str(e)}")
            return {}


def validate_returns_data(returns: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate returns data for optimization requirements.
    
    Checks for missing values, infinite values, and reasonable return ranges.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Returns data to validate
        
    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message) where is_valid indicates if data passes validation
    """
    if returns.empty:
        return False, "Returns data is empty"
    
    if returns.isnull().any().any():
        return False, "Returns data contains NaN values"
    
    if np.isinf(returns.values).any():
        return False, "Returns data contains infinite values"
    
    # Check for reasonable return ranges (daily log returns should be < 50%)
    if (np.abs(returns.iloc[:, :-1]) > 0.5).any().any():  # Exclude cash column
        return False, "Returns data contains unrealistic values (> 50% daily)"
    
    if len(returns) < 10:
        return False, "Insufficient data points for optimization (< 10 days)"
    
    return True, "Data validation passed"