"""
Streamlit dashboard for interactive portfolio optimization.

Provides comprehensive UI for:
- Parameter control panel with sliders and inputs
- Real-time portfolio optimization with multiple methods
- Risk-return visualization and correlation heatmaps
- Backtesting with detailed performance analysis
- Sensitivity analysis and scenario testing
- Cost attribution and turnover analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from data.loader import DataLoader, validate_returns_data
from features.estimators import estimate_all_parameters, compute_portfolio_statistics
from opt.optimizer import PortfolioOptimizer, compute_portfolio_attribution
from backtest.engine import BacktestEngine, analyze_backtest_performance
from utils.metrics import calculate_returns_metrics, calculate_drawdown_metrics
from config import *


# Page configuration
st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Kelly Criterion Portfolio Optimizer")
st.markdown("*Long-only optimization for SPX, TLT, GLD, and Cash*")

# Initialize session state
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = DataLoader(random_seed=42)
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'current_weights' not in st.session_state:
    st.session_state.current_weights = np.ones(N_ASSETS) / N_ASSETS


def create_sidebar():
    """Create the parameter control sidebar."""
    st.sidebar.header("🎛️ Control Panel")
    
    # Data source selection
    data_phase = st.sidebar.radio(
        "Data Source",
        ["Phase 1: Simulated Data", "Phase 2: Market Data"],
        help="Phase 1 uses simulated returns for testing. Phase 2 uses real market data."
    )
    
    st.sidebar.divider()
    
    if data_phase == "Phase 1: Simulated Data":
        st.sidebar.subheader("📊 Simulation Parameters")
        
        # Simulated data parameters
        n_days = st.sidebar.slider("Simulation Days", 100, 1000, 500)
        
        st.sidebar.write("**Expected Returns (Annualized)**")
        mu_spx = st.sidebar.slider("SPX Return", 0.0, 0.20, 0.08, 0.01, format="%.2f")
        mu_tlt = st.sidebar.slider("TLT Return", 0.0, 0.10, 0.03, 0.01, format="%.2f") 
        mu_gld = st.sidebar.slider("GLD Return", 0.0, 0.15, 0.05, 0.01, format="%.2f")
        
        st.sidebar.write("**Volatilities (Annualized)**")
        sig_spx = st.sidebar.slider("SPX Volatility", 0.05, 0.50, 0.16, 0.01, format="%.2f")
        sig_tlt = st.sidebar.slider("TLT Volatility", 0.02, 0.30, 0.07, 0.01, format="%.2f")
        sig_gld = st.sidebar.slider("GLD Volatility", 0.05, 0.40, 0.15, 0.01, format="%.2f")
        
        correlation = st.sidebar.slider("Pairwise Correlation", -0.5, 0.8, 0.2, 0.05, format="%.2f")
        
        sim_params = {
            'n_days': n_days,
            'mu': np.array([mu_spx, mu_tlt, mu_gld]),
            'sigma': np.array([sig_spx, sig_tlt, sig_gld]),
            'correlation': correlation
        }
    else:
        st.sidebar.subheader("📈 Market Data")
        
        # Date range for market data
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=2*365)  # 2 years default
        
        date_range = st.sidebar.date_input(
            "Date Range",
            value=[start_date, end_date],
            max_value=end_date
        )
        
        sim_params = None
        if len(date_range) == 2:
            market_params = {
                'start_date': date_range[0].strftime('%Y-%m-%d'),
                'end_date': date_range[1].strftime('%Y-%m-%d')
            }
        else:
            market_params = None
    
    # Risk-free rate
    risk_free_rate = st.sidebar.slider(
        "Risk-Free Rate",
        0.0, 0.10, DEFAULT_RISK_FREE_RATE, 0.0025,
        format="%.2f%%",
        help="Annualized risk-free rate for cash returns"
    ) 
    
    st.sidebar.divider()
    st.sidebar.subheader("🧮 Estimation Parameters")
    
    # Estimation method
    estimation_method = st.sidebar.selectbox(
        "Estimation Method",
        ["EWMA", "Rolling Window"],
        help="EWMA adapts faster to regime changes"
    )
    
    if estimation_method == "EWMA":
        mu_halflife = st.sidebar.slider("μ/σ Half-life (days)", 10, 100, DEFAULT_MU_SIGMA_HALFLIFE)
        corr_halflife = st.sidebar.slider("ρ Half-life (days)", 5, 50, DEFAULT_CORR_HALFLIFE)
        mu_window = vol_window = corr_window = DEFAULT_LOOKBACK_DAYS
    else:
        mu_window = st.sidebar.slider("μ/σ Window (days)", 50, 500, DEFAULT_LOOKBACK_DAYS)
        corr_window = st.sidebar.slider("ρ Window (days)", 20, 200, 60)
        mu_halflife = vol_halflife = corr_halflife = None
        vol_window = mu_window
    
    # Shrinkage
    shrinkage = st.sidebar.slider(
        "Covariance Shrinkage", 0.0, 1.0, DEFAULT_SHRINKAGE, 0.05,
        help="Regularization towards diagonal matrix"
    )
    
    st.sidebar.divider()
    st.sidebar.subheader("⚖️ Optimization")
    
    # Optimization method
    opt_method = st.sidebar.selectbox(
        "Method",
        ["Quadratic Approximation", "Monte Carlo"],
        help="Quadratic is faster, Monte Carlo is exact"
    )
    
    if opt_method == "Monte Carlo":
        mc_sims = st.sidebar.slider("MC Simulations", 1000, 50000, DEFAULT_MC_SIMULATIONS, 1000)
    else:
        mc_sims = DEFAULT_MC_SIMULATIONS
    
    # Transaction costs
    transaction_cost = st.sidebar.slider(
        "Transaction Cost (bps)", 0.0, 20.0, DEFAULT_TRANSACTION_COST_BPS, 0.5,
        help="Cost per side in basis points"
    )
    
    turnover_penalty = st.sidebar.slider(
        "Turnover Penalty", 0.0, 0.01, DEFAULT_TURNOVER_PENALTY, 0.0005,
        format="%.4f",
        help="L1 penalty for portfolio changes"
    )
    
    # Rebalancing frequency
    rebalance_freq = st.sidebar.selectbox(
        "Rebalance Frequency",
        ["Daily", "Weekly", "Monthly"]
    )
    
    # Package parameters
    params = {
        'data_phase': data_phase,
        'risk_free_rate': risk_free_rate,
        'estimation_method': estimation_method.lower().replace(' ', '_'),
        'mu_window': mu_window,
        'vol_window': vol_window, 
        'corr_window': corr_window,
        'mu_halflife': mu_halflife,
        'vol_halflife': mu_halflife,  # Same as mu
        'corr_halflife': corr_halflife,
        'shrinkage': shrinkage,
        'opt_method': opt_method.lower().replace(' ', '_'),
        'mc_sims': mc_sims,
        'transaction_cost': transaction_cost,
        'turnover_penalty': turnover_penalty,
        'rebalance_freq': rebalance_freq.lower()
    }
    
    if data_phase == "Phase 1: Simulated Data":
        params['sim_params'] = sim_params
    else:
        params['market_params'] = market_params
    
    return params


def load_data(params):
    """Load data based on selected parameters."""
    try:
        if params['data_phase'] == "Phase 1: Simulated Data":
            # Generate simulated data
            data = st.session_state.data_loader.generate_simulated_returns(
                risk_free_rate=params['risk_free_rate'],
                **params['sim_params']
            )
        else:
            # Load market data
            if params.get('market_params') is None:
                return None, "Please select a valid date range"
                
            data = st.session_state.data_loader.fetch_market_data(
                risk_free_rate=params['risk_free_rate'],
                **params['market_params']
            )
        
        # Validate data
        is_valid, error_msg = validate_returns_data(data)
        if not is_valid:
            return None, f"Data validation failed: {error_msg}"
            
        return data, "Data loaded successfully"
        
    except Exception as e:
        return None, f"Error loading data: {str(e)}"


def optimize_portfolio(data, params):
    """Optimize portfolio with current parameters."""
    try:
        # Estimate parameters
        method_map = {
            'ewma': 'ewma',
            'rolling_window': 'rolling'
        }
        
        est_method = method_map[params['estimation_method']]
        
        mu, cov_matrix, volatilities = estimate_all_parameters(
            data,
            mu_method=est_method,
            vol_method=est_method,
            corr_method=est_method,
            mu_window=params['mu_window'],
            vol_window=params['vol_window'],
            corr_window=params['corr_window'],
            mu_halflife=params['mu_halflife'],
            vol_halflife=params['vol_halflife'],
            corr_halflife=params['corr_halflife'],
            shrinkage=params['shrinkage']
        )
        
        # Create optimizer
        optimizer = PortfolioOptimizer(
            transaction_cost_bps=params['transaction_cost'],
            turnover_penalty=params['turnover_penalty']
        )
        
        # Optimize
        opt_method = params['opt_method']
        if opt_method == "monte_carlo":
            result = optimizer.optimize_monte_carlo(
                mu, cov_matrix, 
                n_simulations=params['mc_sims'],
                current_weights=st.session_state.current_weights
            )
        else:
            result = optimizer.optimize_quadratic(
                mu, cov_matrix,
                current_weights=st.session_state.current_weights
            )
        
        if result['success']:
            # Update current weights
            st.session_state.current_weights = result['weights'].copy()
            
            # Compute additional statistics
            portfolio_stats = compute_portfolio_statistics(result['weights'], mu, cov_matrix)
            attribution = compute_portfolio_attribution(
                result['weights'], mu, cov_matrix,
                st.session_state.current_weights,
                params['turnover_penalty'],
                params['transaction_cost']
            )
            
            return {
                'success': True,
                'weights': result['weights'],
                'objective_value': result['objective_value'],
                'method': result['method'],
                'mu': mu,
                'cov_matrix': cov_matrix,
                'volatilities': volatilities,
                'portfolio_stats': portfolio_stats,
                'attribution': attribution
            }
        else:
            return {'success': False, 'error': 'Optimization failed'}
            
    except Exception as e:
        return {'success': False, 'error': str(e)}


def create_weights_chart(weights, asset_names=ASSETS):
    """Create portfolio weights bar chart."""
    fig = go.Figure(data=[
        go.Bar(x=asset_names, y=weights, 
               marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ])
    
    fig.update_layout(
        title="Optimal Portfolio Weights",
        xaxis_title="Assets",
        yaxis_title="Weight",
        yaxis=dict(range=[0, 1], tickformat='.1%'),
        height=400
    )
    
    return fig


def create_correlation_heatmap(cov_matrix, volatilities):
    """Create correlation matrix heatmap."""
    # Extract correlation matrix from covariance
    corr_matrix = cov_matrix / np.outer(volatilities, volatilities)
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=ASSETS,
        y=ASSETS,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(corr_matrix, 3),
        texttemplate="%{text}",
        textfont={"size": 12},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Asset Correlation Matrix",
        height=400
    )
    
    return fig


def create_risk_return_chart(mu, volatilities, weights):
    """Create risk-return scatter plot."""
    # Portfolio metrics
    port_return = weights @ mu
    port_vol = np.sqrt(weights @ (volatilities[:, np.newaxis] @ volatilities[np.newaxis, :] * 
                                  (volatilities[:, np.newaxis] @ volatilities[np.newaxis, :] != 0)) @ weights)
    
    fig = go.Figure()
    
    # Individual assets
    fig.add_trace(go.Scatter(
        x=volatilities, y=mu,
        mode='markers+text',
        text=ASSETS,
        textposition="top center",
        marker=dict(size=12, color='lightblue'),
        name="Individual Assets"
    ))
    
    # Portfolio
    fig.add_trace(go.Scatter(
        x=[port_vol], y=[port_return],
        mode='markers',
        marker=dict(size=15, color='red', symbol='star'),
        name="Portfolio"
    ))
    
    fig.update_layout(
        title="Risk-Return Profile",
        xaxis_title="Volatility (Annualized)",
        yaxis_title="Expected Return (Annualized)",
        xaxis=dict(tickformat='.1%'),
        yaxis=dict(tickformat='.1%'),
        height=400
    )
    
    return fig


def run_backtest(data, params):
    """Run portfolio backtest."""
    try:
        # Setup estimation parameters
        est_params = {
            'mu_method': 'ewma' if params['estimation_method'] == 'ewma' else 'rolling',
            'vol_method': 'ewma' if params['estimation_method'] == 'ewma' else 'rolling', 
            'corr_method': 'ewma' if params['estimation_method'] == 'ewma' else 'rolling',
            'mu_window': params['mu_window'],
            'vol_window': params['vol_window'],
            'corr_window': params['corr_window'],
            'mu_halflife': params['mu_halflife'],
            'vol_halflife': params['vol_halflife'],
            'corr_halflife': params['corr_halflife'],
            'shrinkage': params['shrinkage']
        }
        
        # Create optimizer and engine
        optimizer = PortfolioOptimizer(
            transaction_cost_bps=params['transaction_cost'],
            turnover_penalty=params['turnover_penalty']
        )
        
        engine = BacktestEngine(
            optimizer=optimizer,
            estimation_params=est_params,
            rebalance_freq=params['rebalance_freq']
        )
        
        # Run backtest
        opt_method = params['opt_method']
        results = engine.run_backtest(
            data,
            optimization_method=opt_method
        )
        
        # Analyze performance
        performance = analyze_backtest_performance(results)
        
        return results, performance
        
    except Exception as e:
        st.error(f"Backtesting failed: {str(e)}")
        return None, None


def create_backtest_charts(backtest_results, performance):
    """Create backtest visualization charts."""
    if backtest_results is None:
        return None, None, None
    
    cum_returns = backtest_results['cumulative_returns']
    portfolio_returns = backtest_results['portfolio_returns']
    weights = backtest_results['portfolio_weights']
    
    # Cumulative returns chart
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=cum_returns.index, y=cum_returns.values,
        mode='lines', name='Portfolio',
        line=dict(color='blue', width=2)
    ))
    
    fig1.update_layout(
        title="Cumulative Returns",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        yaxis=dict(tickformat='.1%'),
        height=400,
        yaxis_type="log" if cum_returns.max() > 2 else "linear"
    )
    
    # Drawdown chart
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        mode='lines', fill='tonexty',
        fillcolor='rgba(255,0,0,0.3)',
        line=dict(color='red'),
        name='Drawdown'
    ))
    
    fig2.update_layout(
        title="Portfolio Drawdown",
        xaxis_title="Date", 
        yaxis_title="Drawdown",
        yaxis=dict(tickformat='.1%'),
        height=400
    )
    
    # Weights evolution
    fig3 = go.Figure()
    for i, asset in enumerate(ASSETS):
        fig3.add_trace(go.Scatter(
            x=weights.index, y=weights[asset],
            mode='lines', name=asset,
            line=dict(width=2)
        ))
    
    fig3.update_layout(
        title="Portfolio Weights Over Time",
        xaxis_title="Date",
        yaxis_title="Weight",
        yaxis=dict(tickformat='.1%', range=[0, 1]),
        height=400
    )
    
    return fig1, fig2, fig3


def main():
    """Main application logic."""
    # Sidebar controls
    params = create_sidebar()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col2:
        # Load data button
        if st.button("🔄 Load Data", use_container_width=True):
            with st.spinner("Loading data..."):
                data, message = load_data(params)
                st.session_state.current_data = data
                
                if data is not None:
                    st.success(message)
                else:
                    st.error(message)
    
    # Main dashboard
    if st.session_state.current_data is not None:
        data = st.session_state.current_data
        
        # Optimization section
        st.header("🎯 Portfolio Optimization")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("⚡ Optimize Portfolio", use_container_width=True):
                with st.spinner("Optimizing..."):
                    opt_result = optimize_portfolio(data, params)
                    
                    if opt_result['success']:
                        st.success("Optimization completed!")
                        
                        # Display results
                        st.subheader("Optimal Weights")
                        weights_df = pd.DataFrame({
                            'Asset': ASSETS,
                            'Weight': opt_result['weights'],
                            'Expected Return': opt_result['mu'],
                            'Volatility': opt_result['volatilities']
                        })
                        weights_df['Weight'] = weights_df['Weight'].map('{:.1%}'.format)
                        weights_df['Expected Return'] = weights_df['Expected Return'].map('{:.1%}'.format)
                        weights_df['Volatility'] = weights_df['Volatility'].map('{:.1%}'.format)
                        
                        st.dataframe(weights_df, use_container_width=True)
                        
                        # Key metrics
                        stats = opt_result['portfolio_stats']
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Expected Return", f"{stats['expected_return']:.1%}")
                        with col2:
                            st.metric("Volatility", f"{stats['volatility']:.1%}")
                        with col3:
                            st.metric("Sharpe Ratio", f"{stats['sharpe_ratio']:.2f}")
                        
                        # Charts
                        st.plotly_chart(
                            create_weights_chart(opt_result['weights']),
                            use_container_width=True
                        )
                        
                        # Risk-return and correlation charts
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(
                                create_risk_return_chart(
                                    opt_result['mu'], 
                                    opt_result['volatilities'],
                                    opt_result['weights']
                                ),
                                use_container_width=True
                            )
                        
                        with col2:
                            st.plotly_chart(
                                create_correlation_heatmap(
                                    opt_result['cov_matrix'],
                                    opt_result['volatilities']
                                ),
                                use_container_width=True
                            )
                        
                        # Attribution analysis
                        st.subheader("Performance Attribution")
                        attr = opt_result['attribution']
                        attr_df = pd.DataFrame({
                            'Component': ['Expected Return', 'Risk Penalty', 'Turnover Cost', 'Transaction Cost'],
                            'Contribution': [
                                attr['expected_return'],
                                -attr['risk_penalty'], 
                                -attr['turnover_penalty'],
                                -attr['transaction_costs']
                            ]
                        })
                        attr_df['Contribution'] = attr_df['Contribution'].map('{:.2%}'.format)
                        st.dataframe(attr_df, use_container_width=True)
                        
                    else:
                        st.error(f"Optimization failed: {opt_result['error']}")
        
        with col2:
            # Data summary
            st.subheader("📋 Data Summary")
            st.write(f"**Date Range:** {data.index[0].date()} to {data.index[-1].date()}")
            st.write(f"**Observations:** {len(data)}")
            st.write(f"**Assets:** {', '.join(ASSETS)}")
            
            # Recent returns
            st.write("**Recent Performance (5 days)**")
            recent_returns = data.tail(5).mean()
            for asset in ASSETS:
                st.write(f"{asset}: {recent_returns[asset]*252:.1%}")
        
        # Backtesting section
        st.divider()
        st.header("📊 Strategy Backtesting")
        
        if st.button("🚀 Run Backtest", use_container_width=True):
            with st.spinner("Running backtest..."):
                backtest_results, performance = run_backtest(data, params)
                
                if backtest_results is not None:
                    st.success("Backtest completed!")
                    
                    # Performance metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Return", f"{performance['total_return']:.1%}")
                    with col2:
                        st.metric("Annualized Return", f"{performance['annualized_return']:.1%}")
                    with col3:
                        st.metric("Sharpe Ratio", f"{performance['sharpe_ratio']:.2f}")
                    with col4:
                        st.metric("Max Drawdown", f"{performance['max_drawdown']:.1%}")
                    
                    # Charts
                    fig1, fig2, fig3 = create_backtest_charts(backtest_results, performance)
                    
                    if fig1:
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(fig2, use_container_width=True)
                        with col2:
                            st.plotly_chart(fig3, use_container_width=True)
                    
                    # Detailed performance table
                    st.subheader("Detailed Performance Metrics")
                    perf_df = pd.DataFrame({
                        'Metric': [
                            'Total Return', 'Annualized Return', 'Volatility', 
                            'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown',
                            'Hit Rate', 'Total Costs', 'Avg Turnover'
                        ],
                        'Value': [
                            f"{performance['total_return']:.2%}",
                            f"{performance['annualized_return']:.2%}",
                            f"{performance['annualized_volatility']:.2%}",
                            f"{performance['sharpe_ratio']:.3f}",
                            f"{performance['sortino_ratio']:.3f}",
                            f"{performance['max_drawdown']:.2%}",
                            f"{performance['hit_rate']:.2%}",
                            f"{performance['total_transaction_costs']:.4%}",
                            f"{performance['avg_daily_turnover']:.2%}"
                        ]
                    })
                    st.dataframe(perf_df, use_container_width=True)
    
    else:
        # Welcome message
        st.info("👆 Please configure parameters in the sidebar and click 'Load Data' to begin.")
        
        st.subheader("About This Application")
        st.write("""
        This portfolio optimizer implements the Kelly Criterion for long-only portfolios containing:
        - **SPX** (S&P 500 Index)
        - **TLT** (20+ Year Treasury Bonds) 
        - **GLD** (Gold ETF)
        - **Cash** (Risk-free asset)
        
        **Key Features:**
        - Two optimization methods: Quadratic approximation and Monte Carlo
        - Flexible parameter estimation with EWMA and rolling windows
        - Transaction costs and turnover penalties
        - Comprehensive backtesting and performance analysis
        - Real-time sensitivity analysis and scenario testing
        """)


if __name__ == "__main__":
    main()