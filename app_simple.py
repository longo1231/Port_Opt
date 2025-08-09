"""
Simplified Portfolio Optimizer UI - Breaking the Market approach.

Clean interface with minimal controls, showing methodology choices.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Import simplified components
from data.loader import DataLoader
from estimators_simple import estimate_parameters, get_estimation_info
from optimizer_simple import optimize_portfolio
from config_simple import ASSETS, RISK_FREE_RATE, DEFAULT_DATE_RANGE_YEARS


def create_correlation_heatmap(expected_returns, covariance_matrix):
    """Create correlation matrix heatmap."""
    # Extract volatilities and correlation matrix
    volatilities = np.sqrt(np.diag(covariance_matrix))
    corr_matrix = covariance_matrix / np.outer(volatilities, volatilities)
    
    # Format text for display
    text_array = [[f"{val:.3f}" for val in row] for row in corr_matrix]
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=ASSETS,
        y=ASSETS,
        colorscale='RdBu_r',
        zmid=0,
        text=text_array,
        texttemplate="%{text}",
        textfont={"size": 12},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Asset Correlation Matrix",
        height=400
    )
    
    return fig


def create_weights_chart(weights, expected_returns):
    """Create portfolio weights bar chart."""
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    
    fig = go.Figure(data=go.Bar(
        x=ASSETS,
        y=weights * 100,  # Convert to percentages
        marker_color=colors,
        text=[f"{w:.1f}%" for w in weights * 100],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Optimal Portfolio Weights",
        xaxis_title="Assets",
        yaxis_title="Weight (%)",
        height=400
    )
    
    return fig


def display_methodology(n_days: int = None):
    """Display the methodology being used."""
    st.sidebar.header("📋 Methodology")
    
    info = get_estimation_info(n_days)
    
    st.sidebar.markdown("""
    **Breaking the Market Approach:**
    - 4 assets: SPY, TLT, GLD, Cash
    - Long-only, no leverage
    - Daily rebalancing philosophy
    """)
    
    st.sidebar.markdown(f"""
    **Rolling Windows:**
    - μ (returns): {info['mu_window']} (slow)
    - σ (volatility): {info['sigma_window']} (medium)
    - ρ (correlation): {info['rho_window']} (fast)
    """)
    
    st.sidebar.markdown(f"""
    **Optimization:**
    - Pure quadratic Kelly criterion
    - Maximize: w'μ - 0.5 w'Σw
    - Subject to: Σw = 1, w ≥ 0
    - Risk-free rate: {RISK_FREE_RATE:.2%}
    """)


def load_market_data(start_date, end_date):
    """Load market data for the specified period."""
    try:
        loader = DataLoader()
        data = loader.fetch_market_data(start_date.strftime("%Y-%m-%d"), 
                                       end_date.strftime("%Y-%m-%d"))
        return data, None
    except Exception as e:
        return None, str(e)


def main():
    st.set_page_config(
        page_title="Portfolio Optimizer - Breaking the Market",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 Breaking the Market Portfolio Optimizer")
    st.markdown("*Simple, frequent rebalancing with adaptive correlations*")
    
    # Display methodology in sidebar (will be updated after data load)
    display_methodology()
    
    # Date selection (only essential control)
    st.sidebar.divider()
    st.sidebar.subheader("📅 Data Range")
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=DEFAULT_DATE_RANGE_YEARS * 365)
    
    date_range = st.sidebar.date_input(
        "Select date range:",
        value=[start_date, end_date],
        max_value=end_date,
        help="Minimum 1 year of data recommended"
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        
        # Load data
        if st.sidebar.button("🔄 Load & Optimize", use_container_width=True):
            with st.spinner("Loading market data and optimizing..."):
                data, error = load_market_data(start_date, end_date)
                
                if data is not None:
                    st.success(f"✅ Loaded {len(data)} days of market data")
                    
                    # Update methodology display with actual data size
                    display_methodology(len(data))
                    
                    # Display data summary  
                    st.subheader("📋 Data Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Start Date", start_date.strftime("%Y-%m-%d"))
                    with col2:
                        st.metric("End Date", end_date.strftime("%Y-%m-%d"))
                    with col3:
                        st.metric("Observations", f"{len(data):,} days")
                    
                    # Recent performance
                    st.write("**Recent Performance (5 days, annualized):**")
                    recent_returns = data.tail(5).mean() * 252
                    perf_text = " • ".join([f"**{asset}:** {recent_returns[asset]:+.1%}" for asset in ASSETS])
                    st.write(perf_text)
                    
                    # Estimate parameters
                    try:
                        expected_returns, covariance_matrix = estimate_parameters(data)
                        
                        # Optimize portfolio
                        result = optimize_portfolio(expected_returns, covariance_matrix)
                        
                        if result['success']:
                            st.divider()
                            st.subheader("🎯 Optimization Results")
                            
                            # Portfolio metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Expected Return", f"{result['expected_return']:.1%}")
                            with col2:
                                st.metric("Volatility", f"{result['volatility']:.1%}")
                            with col3:
                                st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")
                            with col4:
                                st.metric("Kelly Objective", f"{result['objective_value']:.3f}")
                            
                            # Weights table and chart
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                st.subheader("Portfolio Weights")
                                weights_df = pd.DataFrame({
                                    'Asset': ASSETS,
                                    'Weight': [f"{w:.1%}" for w in result['weights']],
                                    'Expected Return': [f"{er:.1%}" for er in expected_returns]
                                })
                                st.dataframe(weights_df, use_container_width=True, hide_index=True)
                            
                            with col2:
                                st.plotly_chart(
                                    create_weights_chart(result['weights'], expected_returns),
                                    use_container_width=True
                                )
                            
                            # Correlation matrix
                            st.subheader("Asset Correlations")
                            st.plotly_chart(
                                create_correlation_heatmap(expected_returns, covariance_matrix),
                                use_container_width=True
                            )
                            
                            # Parameter details
                            with st.expander("📊 Parameter Details"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("**Expected Returns (Annualized)**")
                                    for i, asset in enumerate(ASSETS):
                                        st.write(f"• {asset}: {expected_returns[i]:+.1%}")
                                
                                with col2:
                                    st.write("**Volatilities (Annualized)**")
                                    volatilities = np.sqrt(np.diag(covariance_matrix))
                                    for i, asset in enumerate(ASSETS):
                                        st.write(f"• {asset}: {volatilities[i]:.1%}")
                        
                        else:
                            st.error(f"❌ Optimization failed: {result['error']}")
                    
                    except Exception as e:
                        st.error(f"❌ Error in parameter estimation: {str(e)}")
                
                else:
                    st.error(f"❌ Failed to load data: {error}")
    
    # Footer
    st.divider()
    st.markdown("""
    **About This Optimizer**
    
    This implements the "Breaking the Market" approach to portfolio optimization:
    - **Philosophy**: Frequent rebalancing with adaptive correlations beats single-asset strategies
    - **Methodology**: Pure Kelly criterion without leverage or artificial constraints
    - **Simplicity**: Fixed windows, no parameter tuning, just optimal risk-adjusted growth
    
    *Built with Streamlit • Data via Yahoo Finance*
    """)


if __name__ == "__main__":
    main()