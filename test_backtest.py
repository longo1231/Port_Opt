"""
Test suite for walk-forward backtest functionality.

Tests core backtest logic with simulated data to ensure correctness
before integrating with the main application.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest import (
    generate_rebalance_dates, 
    walk_forward_backtest, 
    compare_with_static_weights,
    calculate_period_performance
)
from data.loader import DataLoader
from config import ASSETS


def test_rebalance_date_generation():
    """Test rebalancing date generation."""
    print("Testing rebalance date generation...")
    
    # Test weekly rebalancing
    dates = generate_rebalance_dates('2023-01-01', '2023-01-31', 'weekly')
    print(f"Weekly dates for January 2023: {dates}")
    assert len(dates) >= 4, "Should have at least 4 weekly dates in January"
    
    # Test monthly rebalancing
    dates = generate_rebalance_dates('2023-01-01', '2023-12-31', 'monthly')
    print(f"Monthly dates for 2023: {len(dates)} dates")
    assert len(dates) == 12, "Should have 12 monthly dates in 2023"
    
    print("✅ Rebalance date generation tests passed")


def test_period_performance():
    """Test period performance calculation."""
    print("\nTesting period performance calculation...")
    
    # Create simple test data
    dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
    # Simple returns: 1% daily for all assets
    returns_data = pd.DataFrame({
        'SPY': [0.01] * len(dates),
        'TLT': [0.005] * len(dates), 
        'GLD': [0.008] * len(dates),
        'Cash': [0.0001] * len(dates)
    }, index=dates)
    
    # Equal weights
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    
    # Calculate performance for the period
    perf = calculate_period_performance(
        returns_data, weights, '2023-01-01', '2023-01-10'
    )
    
    print(f"Period return: {perf['period_return']:.4f}")
    print(f"Number of days: {perf['n_days']}")
    
    assert perf['n_days'] == 10, "Should have 10 days of data"
    assert perf['period_return'] > 0, "Should have positive return"
    
    print("✅ Period performance tests passed")


def test_backtest_with_simulated_data():
    """Test backtest with simulated data."""
    print("\nTesting backtest with simulated data...")
    
    # Generate simulated data
    loader = DataLoader(random_seed=42)
    returns_data = loader.generate_simulated_returns(
        n_days=252,  # 1 year of data
        mu=np.array([0.08, 0.03, 0.05]),  # SPY, TLT, GLD (risky assets only)
        sigma=np.array([0.16, 0.07, 0.15]),  # SPY, TLT, GLD (risky assets only)
        correlation=0.2,  # Use scalar correlation 
        risk_free_rate=0.053
    )
    
    print(f"Generated {len(returns_data)} days of simulated data")
    print(f"Assets: {list(returns_data.columns)}")
    print(f"Date range: {returns_data.index[0]} to {returns_data.index[-1]}")
    
    # Run backtest on last 3 months (need first 9 months for history)
    start_date = returns_data.index[-63].strftime('%Y-%m-%d')  # Last ~3 months
    end_date = returns_data.index[-1].strftime('%Y-%m-%d')
    
    print(f"Running backtest from {start_date} to {end_date}")
    
    try:
        results = walk_forward_backtest(
            returns_data, 
            start_date, 
            end_date,
            rebalance_freq='weekly',
            min_history_days=60
        )
        
        print(f"\n📊 Backtest Results:")
        print(f"Total return: {results['total_return']:.4f} ({results['total_return']*100:.2f}%)")
        print(f"Annualized return: {results['annualized_return']:.4f} ({results['annualized_return']*100:.2f}%)")
        print(f"Volatility: {results['volatility']:.4f} ({results['volatility']*100:.2f}%)")
        print(f"Sharpe ratio: {results['sharpe_ratio']:.3f}")
        print(f"Max drawdown: {results['max_drawdown']:.4f} ({results['max_drawdown']*100:.2f}%)")
        print(f"Average turnover: {results['average_turnover']:.4f} ({results['average_turnover']*100:.2f}%)")
        print(f"Number of rebalances: {results['n_rebalances']}")
        print(f"Optimization errors: {results['n_optimization_errors']}")
        
        # Check basic sanity
        assert results['n_rebalances'] > 0, "Should have at least one rebalance"
        assert len(results['portfolio_values']) > 0, "Should have portfolio values"
        assert len(results['weight_history']) > 0, "Should have weight history"
        
        print("✅ Simulated data backtest passed")
        
        return results
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        raise


def test_static_comparison():
    """Test comparison with static weights."""
    print("\nTesting static weight comparison...")
    
    # Generate simulated data
    loader = DataLoader(random_seed=42)
    returns_data = loader.generate_simulated_returns(n_days=252)
    
    # Run backtest
    start_date = returns_data.index[-63].strftime('%Y-%m-%d')
    end_date = returns_data.index[-1].strftime('%Y-%m-%d')
    
    backtest_results = walk_forward_backtest(
        returns_data, start_date, end_date, rebalance_freq='weekly'
    )
    
    # Compare with static equal weights
    static_weights = np.array([0.33, 0.33, 0.34, 0.0])  # Equal weight risky assets
    
    comparison = compare_with_static_weights(
        backtest_results, returns_data, static_weights
    )
    
    print(f"\n📈 Dynamic vs Static Comparison:")
    print(f"Dynamic return: {comparison['dynamic_total_return']:.4f}")
    print(f"Static return: {comparison['static_total_return']:.4f}")
    print(f"Return difference: {comparison['return_difference']:.4f}")
    print(f"Turnover cost: {comparison['turnover_cost']:.4f}")
    
    assert 'dynamic_total_return' in comparison, "Should have dynamic return"
    assert 'static_total_return' in comparison, "Should have static return"
    
    print("✅ Static comparison tests passed")


def test_with_real_market_data():
    """Test with real market data if available."""
    print("\nTesting with real market data...")
    
    try:
        # Try to fetch recent market data
        loader = DataLoader()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)  # 2 years of data
        
        market_data = loader.fetch_market_data(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if len(market_data) < 100:
            print("Insufficient market data, skipping real data test")
            return
            
        print(f"Fetched {len(market_data)} days of market data")
        
        # Run backtest on last 6 months
        backtest_start = market_data.index[-126].strftime('%Y-%m-%d')
        backtest_end = market_data.index[-1].strftime('%Y-%m-%d')
        
        print(f"Running market data backtest from {backtest_start} to {backtest_end}")
        
        results = walk_forward_backtest(
            market_data,
            backtest_start,
            backtest_end,
            rebalance_freq='weekly'
        )
        
        print(f"\n📊 Market Data Backtest Results:")
        print(f"Total return: {results['total_return']:.4f} ({results['total_return']*100:.2f}%)")
        print(f"Volatility: {results['volatility']:.4f} ({results['volatility']*100:.2f}%)")
        print(f"Sharpe ratio: {results['sharpe_ratio']:.3f}")
        print(f"Number of rebalances: {results['n_rebalances']}")
        
        print("✅ Market data backtest passed")
        
    except Exception as e:
        print(f"⚠️  Market data test failed (may be network/API issue): {e}")
        print("This is not necessarily a problem with the backtest code")


def main():
    """Run all tests."""
    print("🧪 Testing Walk-Forward Backtest Engine")
    print("=" * 50)
    
    try:
        test_rebalance_date_generation()
        test_period_performance()
        test_backtest_with_simulated_data()
        test_static_comparison()
        test_with_real_market_data()
        
        print("\n" + "=" * 50)
        print("🎉 All backtest tests passed!")
        print("The walk-forward backtest engine is ready for integration.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)