"""
Phase 1 testing script for portfolio optimizer with simulated data.

Validates end-to-end functionality including:
- Data generation and validation
- Parameter estimation with different methods
- Portfolio optimization (both quadratic and Monte Carlo)
- Backtesting with realistic costs
- Performance attribution analysis
"""

import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append('.')

from data.loader import DataLoader, validate_returns_data
from features.estimators import estimate_all_parameters
from opt.optimizer import PortfolioOptimizer, compute_portfolio_attribution
from backtest.engine import BacktestEngine, analyze_backtest_performance
from config import *


def test_data_generation():
    """Test simulated data generation."""
    print("Testing data generation...")
    
    loader = DataLoader(random_seed=42)
    
    # Test with default parameters
    data = loader.generate_simulated_returns(n_days=252)
    
    # Validate data
    is_valid, msg = validate_returns_data(data)
    assert is_valid, f"Data validation failed: {msg}"
    
    # Check dimensions
    assert data.shape == (252, 4), f"Wrong data shape: {data.shape}"
    assert list(data.columns) == ASSETS, f"Wrong columns: {data.columns}"
    
    # Check cash returns are constant
    cash_returns = data['Cash'].values
    assert np.allclose(cash_returns, cash_returns[0]), "Cash returns should be constant"
    
    print("✓ Data generation test passed")
    return data


def test_parameter_estimation(data):
    """Test parameter estimation methods."""
    print("Testing parameter estimation...")
    
    # Test EWMA method
    mu_ewma, cov_ewma, vol_ewma = estimate_all_parameters(
        data,
        mu_method='ewma',
        vol_method='ewma', 
        corr_method='ewma',
        mu_halflife=30,
        vol_halflife=30,
        corr_halflife=15
    )
    
    # Validate dimensions
    assert len(mu_ewma) == 4, f"Wrong mu dimensions: {len(mu_ewma)}"
    assert cov_ewma.shape == (4, 4), f"Wrong covariance shape: {cov_ewma.shape}"
    assert len(vol_ewma) == 4, f"Wrong volatility dimensions: {len(vol_ewma)}"
    
    # Check covariance matrix properties
    eigenvals = np.linalg.eigvals(cov_ewma)
    assert np.all(eigenvals > 0), "Covariance matrix not positive definite"
    
    # Test rolling window method
    mu_roll, cov_roll, vol_roll = estimate_all_parameters(
        data,
        mu_method='rolling',
        vol_method='rolling',
        corr_method='rolling',
        mu_window=60,
        vol_window=60,
        corr_window=30
    )
    
    # Validate rolling results
    assert len(mu_roll) == 4, "Wrong rolling mu dimensions"
    assert cov_roll.shape == (4, 4), "Wrong rolling covariance shape"
    
    print("✓ Parameter estimation test passed")
    return mu_ewma, cov_ewma, vol_ewma


def test_optimization(mu, cov_matrix):
    """Test portfolio optimization methods."""
    print("Testing portfolio optimization...")
    
    optimizer = PortfolioOptimizer(
        transaction_cost_bps=2.0,
        turnover_penalty=0.001,
        random_seed=42
    )
    
    # Test quadratic optimization
    result_quad = optimizer.optimize_quadratic(mu, cov_matrix)
    
    assert result_quad['success'], "Quadratic optimization failed"
    assert len(result_quad['weights']) == 4, "Wrong weights dimensions"
    assert abs(result_quad['weights'].sum() - 1.0) < 1e-6, "Weights don't sum to 1"
    assert np.all(result_quad['weights'] >= -1e-6), "Negative weights found"
    
    # Test Monte Carlo optimization
    result_mc = optimizer.optimize_monte_carlo(mu, cov_matrix, n_simulations=5000)
    
    assert result_mc['success'], "Monte Carlo optimization failed"  
    assert len(result_mc['weights']) == 4, "Wrong MC weights dimensions"
    assert abs(result_mc['weights'].sum() - 1.0) < 1e-6, "MC weights don't sum to 1"
    assert np.all(result_mc['weights'] >= -1e-6), "MC negative weights found"
    
    print("✓ Optimization test passed")
    print(f"  Quadratic weights: {result_quad['weights'].round(3)}")
    print(f"  Monte Carlo weights: {result_mc['weights'].round(3)}")
    
    return result_quad, result_mc


def test_attribution(weights, mu, cov_matrix):
    """Test portfolio attribution analysis."""
    print("Testing attribution analysis...")
    
    current_weights = np.ones(4) / 4  # Equal weight baseline
    
    attribution = compute_portfolio_attribution(
        weights, mu, cov_matrix, current_weights
    )
    
    # Check required fields
    required_fields = [
        'expected_return', 'risk_penalty', 'turnover_penalty', 
        'transaction_costs', 'net_expected_growth'
    ]
    
    for field in required_fields:
        assert field in attribution, f"Missing attribution field: {field}"
        assert not np.isnan(attribution[field]), f"NaN value in {field}"
    
    # Check that components sum approximately to net growth
    components_sum = (attribution['expected_return'] - 
                     attribution['risk_penalty'] -
                     attribution['turnover_penalty'] - 
                     attribution['transaction_costs'])
    
    assert abs(components_sum - attribution['net_expected_growth']) < 1e-10, "Attribution doesn't balance"
    
    print("✓ Attribution test passed")
    return attribution


def test_backtesting(data):
    """Test backtesting engine."""
    print("Testing backtesting engine...")
    
    # Create optimizer and engine
    optimizer = PortfolioOptimizer(
        transaction_cost_bps=2.0,
        turnover_penalty=0.001
    )
    
    engine = BacktestEngine(
        optimizer=optimizer,
        rebalance_freq='weekly'  # Less frequent for faster testing
    )
    
    # Run backtest
    results = engine.run_backtest(data, optimization_method='quadratic')
    
    # Validate results
    assert 'portfolio_returns' in results, "Missing portfolio returns"
    assert 'portfolio_weights' in results, "Missing portfolio weights"
    assert 'transaction_costs' in results, "Missing transaction costs"
    
    portfolio_returns = results['portfolio_returns']
    portfolio_weights = results['portfolio_weights']
    
    # Check dimensions
    expected_length = len(data) - engine.min_history_days
    assert len(portfolio_returns) == expected_length, f"Wrong return series length: {len(portfolio_returns)} vs {expected_length}"
    assert len(portfolio_weights) == expected_length, "Wrong weights series length"
    
    # Analyze performance
    performance = analyze_backtest_performance(results)
    
    # Check performance metrics
    required_metrics = [
        'total_return', 'annualized_return', 'annualized_volatility',
        'sharpe_ratio', 'max_drawdown'
    ]
    
    for metric in required_metrics:
        assert metric in performance, f"Missing performance metric: {metric}"
        assert not np.isnan(performance[metric]), f"NaN value in {metric}"
    
    print("✓ Backtesting test passed")
    print(f"  Total return: {performance['total_return']:.2%}")
    print(f"  Sharpe ratio: {performance['sharpe_ratio']:.3f}")
    print(f"  Max drawdown: {performance['max_drawdown']:.2%}")
    
    return results, performance


def main():
    """Run all Phase 1 tests."""
    print("=" * 50)
    print("PHASE 1 TESTING: SIMULATED DATA VALIDATION")
    print("=" * 50)
    
    try:
        # Test 1: Data generation
        data = test_data_generation()
        
        # Test 2: Parameter estimation  
        mu, cov_matrix, volatilities = test_parameter_estimation(data)
        
        # Test 3: Portfolio optimization
        result_quad, result_mc = test_optimization(mu, cov_matrix)
        
        # Test 4: Attribution analysis
        attribution = test_attribution(result_quad['weights'], mu, cov_matrix)
        
        # Test 5: Backtesting
        backtest_results, performance = test_backtesting(data)
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED! 🎉")
        print("=" * 50)
        
        # Summary statistics
        print("\nSUMMARY STATISTICS:")
        print(f"Data period: {len(data)} days")
        print(f"Optimal weights (Quadratic): {result_quad['weights'].round(3)}")  
        print(f"Expected return: {attribution['expected_return']:.2%}")
        print(f"Risk penalty: {attribution['risk_penalty']:.2%}")
        print(f"Net expected growth: {attribution['net_expected_growth']:.2%}")
        print(f"Backtest Sharpe ratio: {performance['sharpe_ratio']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)