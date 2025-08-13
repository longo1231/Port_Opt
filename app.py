"""
Minimum Variance Portfolio Optimizer - Streamlit Dashboard.

Clean implementation focusing purely on risk reduction through diversification.
Eliminates problematic expected returns to avoid corner solutions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import hashlib
import pickle

from data.loader import DataLoader
from estimators import estimate_covariance_matrix, estimate_expected_returns, get_estimation_info
from optimizer import optimize_min_variance, analyze_min_variance_portfolio
from backtest import walk_forward_backtest, compare_with_static_weights
from config import ASSETS, DEFAULT_DATE_RANGE_YEARS, SIGMA_WINDOW, RHO_WINDOW


def get_data_hash(data, start_str, end_str, sigma_window, rho_window, rebalance_freq):
    """Generate a hash for caching based on data and parameters."""
    # Create a string representation of key parameters
    cache_key = f"{start_str}_{end_str}_{sigma_window}_{rho_window}_{rebalance_freq}"
    
    # Add data hash (sample of data to avoid large computation)
    data_sample = data.iloc[::max(1, len(data)//100)]  # Sample every 1% of data
    data_str = str(data_sample.values.tolist())
    
    return hashlib.md5((cache_key + data_str).encode()).hexdigest()


@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def get_cached_backtest_result(cache_key, data_pickle, start_str, end_str, rebalance_freq, min_history_days, sigma_window, rho_window):
    """
    Cached backtest execution. Returns cached result if available, otherwise computes new result.
    """
    # Unpickle data
    data = pickle.loads(data_pickle)
    
    # Run backtest
    return walk_forward_backtest(
        data, start_str, end_str,
        rebalance_freq=rebalance_freq,
        min_history_days=min_history_days,
        sigma_window=sigma_window,
        rho_window=rho_window
    )


def create_correlation_heatmap(covariance_matrix):
    """Create correlation matrix heatmap."""
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


def create_weights_chart(weights, diversification_ratio, effective_assets):
    """Create portfolio weights bar chart."""
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    
    fig = go.Figure(data=go.Bar(
        x=ASSETS,
        y=weights * 100,
        marker_color=colors,
        text=[f"{w:.1f}%" for w in weights * 100],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=f"Minimum Variance Portfolio Weights<br>" + 
              f"<sub>Diversification Ratio: {diversification_ratio:.2f} | " +
              f"Effective Assets: {effective_assets:.1f}</sub>",
        xaxis_title="Assets",
        yaxis_title="Weight (%)",
        height=400
    )
    
    return fig


def create_risk_contribution_chart(analysis_data, covariance_matrix):
    """Create risk contribution analysis chart showing optimized vs naive allocations."""
    assets = [item['asset'] for item in analysis_data['asset_analysis']]
    current_weights = [item['weight'] * 100 for item in analysis_data['asset_analysis']]
    current_risk_contribs = [item['risk_contrib_pct'] * 100 for item in analysis_data['asset_analysis']]
    
    # Calculate naive (equal weight) risk contributions
    n_risky_assets = 3  # SPY, TLT, GLD (excluding Cash)
    equal_weights = np.zeros(len(ASSETS))
    equal_weights[:n_risky_assets] = 1.0 / n_risky_assets
    equal_weights[-1] = 0.01  # Tiny cash position
    
    # Calculate risk contributions for equal weights
    equal_portfolio_var = equal_weights @ covariance_matrix @ equal_weights
    equal_marginal_risks = covariance_matrix @ equal_weights
    equal_risk_contribs = equal_weights * equal_marginal_risks
    equal_risk_contribs_pct = (equal_risk_contribs / equal_portfolio_var) * 100
    
    fig = go.Figure()
    
    # Current risk contributions (same as weights for min-var)
    fig.add_trace(go.Bar(
        name='Optimized Risk Contrib',
        x=assets,
        y=current_risk_contribs,
        marker_color='crimson',
        opacity=0.9
    ))
    
    # Equal weight risk contributions (showing vol effect)
    fig.add_trace(go.Bar(
        name='Equal Weight Risk Contrib',
        x=assets,
        y=equal_risk_contribs_pct[:len(assets)],
        marker_color='orange',
        opacity=0.8
    ))
    
    fig.update_layout(
        title="Risk Contributions: Optimized vs Equal Weight",
        xaxis_title="Assets",
        yaxis_title="Percentage (%)",
        barmode='group',
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig


def create_volatility_comparison_chart(analysis_data, sigma_window):
    """Compare individual vs portfolio-weighted volatilities with window volatility annotations."""
    assets = [item['asset'] for item in analysis_data['asset_analysis'] if item['weight'] > 0.001]
    individual_vols = [item['individual_vol'] * 100 for item in analysis_data['asset_analysis'] if item['weight'] > 0.001]
    weights = [item['weight'] for item in analysis_data['asset_analysis'] if item['weight'] > 0.001]
    
    # Calculate weighted average volatility
    weighted_avg_vol = sum(w * vol for w, vol in zip(weights, individual_vols))
    portfolio_vol = analysis_data['portfolio_volatility'] * 100
    
    fig = go.Figure()
    
    # Individual volatilities with text annotations showing the window vol values
    fig.add_trace(go.Bar(
        name=f'Window Volatility ({sigma_window}d)',
        x=assets,
        y=individual_vols,
        marker_color='orange',
        opacity=0.7,
        text=[f"{vol:.1f}%" for vol in individual_vols],
        textposition='auto',
        textfont=dict(color='white', size=12, family='Arial Black')
    ))
    
    # Add horizontal lines for comparison
    fig.add_hline(
        y=weighted_avg_vol,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Weighted Avg: {weighted_avg_vol:.1f}%",
        annotation_position="top right"
    )
    
    fig.add_hline(
        y=portfolio_vol,
        line_dash="solid",
        line_color="red",
        annotation_text=f"Portfolio: {portfolio_vol:.1f}% (Window)",
        annotation_position="bottom right"
    )
    
    fig.update_layout(
        title=f"Window Volatility Comparison ({sigma_window}-day rolling)",
        xaxis_title="Assets (with positive weights)",
        yaxis_title="Volatility (%)",
        height=400
    )
    
    return fig


def calculate_portfolio_performance(data, weights):
    """Calculate historical performance of portfolio using static weights."""
    # Calculate daily portfolio returns
    portfolio_daily_returns = (data * weights).sum(axis=1)
    
    # Calculate cumulative returns (assumes simple returns)
    portfolio_cumulative = (1 + portfolio_daily_returns).cumprod()
    individual_cumulative = (1 + data).cumprod()
    
    # Calculate performance metrics over the period
    portfolio_total_return = portfolio_cumulative.iloc[-1] - 1
    portfolio_vol = portfolio_daily_returns.std() * np.sqrt(252)
    portfolio_sharpe = (portfolio_daily_returns.mean() * 252) / portfolio_vol if portfolio_vol > 0 else 0
    
    # Calculate max drawdown
    portfolio_peak = portfolio_cumulative.cummax()
    portfolio_drawdown = (portfolio_cumulative - portfolio_peak) / portfolio_peak
    max_drawdown = portfolio_drawdown.min()
    
    return {
        'portfolio_daily_returns': portfolio_daily_returns,
        'portfolio_cumulative': portfolio_cumulative,
        'individual_cumulative': individual_cumulative,
        'portfolio_total_return': portfolio_total_return,
        'portfolio_vol': portfolio_vol,
        'portfolio_sharpe': portfolio_sharpe,
        'max_drawdown': max_drawdown,
        'drawdowns': portfolio_drawdown
    }


def create_performance_chart(performance_data, assets):
    """Create historical performance comparison chart."""
    fig = go.Figure()
    
    # Add individual asset performance
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    
    for i, asset in enumerate(assets):
        if asset != 'Cash' or performance_data['individual_cumulative'][asset].iloc[-1] > 1.001:  # Skip Cash if flat
            fig.add_trace(go.Scatter(
                x=performance_data['individual_cumulative'].index,
                y=(performance_data['individual_cumulative'][asset] - 1) * 100,
                mode='lines',
                name=asset,
                line=dict(color=colors[i % len(colors)], width=2),
                opacity=0.7
            ))
    
    # Add portfolio performance (highlighted)
    fig.add_trace(go.Scatter(
        x=performance_data['portfolio_cumulative'].index,
        y=(performance_data['portfolio_cumulative'] - 1) * 100,
        mode='lines',
        name='Portfolio',
        line=dict(color='crimson', width=4),
        opacity=1.0
    ))
    
    fig.update_layout(
        title="Historical Performance: Individual Assets vs Optimized Portfolio",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode='x unified'
    )
    
    return fig


def create_backtest_performance_chart(backtest_results, original_data, static_weights):
    """Create walk-forward backtest performance comparison chart."""
    fig = go.Figure()
    
    # Get portfolio values from backtest
    portfolio_values = backtest_results['portfolio_values']
    
    # Calculate static portfolio performance over same period
    backtest_period_data = original_data.loc[backtest_results['backtest_start']:backtest_results['backtest_end']]
    static_returns = (backtest_period_data * static_weights).sum(axis=1)
    static_cumulative = (1 + static_returns).cumprod()
    
    # Add individual asset performance for reference
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    individual_cumulative = (1 + backtest_period_data).cumprod()
    
    for i, asset in enumerate(ASSETS):
        if asset != 'Cash' or individual_cumulative[asset].iloc[-1] > 1.001:
            fig.add_trace(go.Scatter(
                x=individual_cumulative.index,
                y=(individual_cumulative[asset] - 1) * 100,
                mode='lines',
                name=asset,
                line=dict(color=colors[i % len(colors)], width=1.5),
                opacity=0.5
            ))
    
    # Add static portfolio performance
    fig.add_trace(go.Scatter(
        x=static_cumulative.index,
        y=(static_cumulative - 1) * 100,
        mode='lines',
        name='Static Portfolio',
        line=dict(color='orange', width=3, dash='dash'),
        opacity=0.8
    ))
    
    # Add dynamic backtest performance (highlighted)
    fig.add_trace(go.Scatter(
        x=portfolio_values.index,
        y=(portfolio_values - 1) * 100,
        mode='lines',
        name='Dynamic Portfolio (Backtest)',
        line=dict(color='crimson', width=4),
        opacity=1.0
    ))
    
    fig.update_layout(
        title="Walk-Forward Backtest: Dynamic vs Static Portfolio Performance",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode='x unified'
    )
    
    return fig


def create_rolling_metrics_chart(backtest_results, window_days=30):
    """Create rolling Sharpe ratio and volatility chart for dynamic portfolio."""
    
    # Get dynamic portfolio returns
    dynamic_returns = backtest_results['portfolio_returns']
    
    if len(dynamic_returns) < window_days:
        # Not enough data for rolling calculation
        fig = go.Figure()
        fig.add_annotation(
            text=f"Insufficient data for {window_days}-day rolling metrics<br>Need at least {window_days} days, have {len(dynamic_returns)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(
            title="Rolling Risk Metrics - Insufficient Data",
            height=400
        )
        return fig
    
    # Calculate rolling metrics
    rolling_vol = dynamic_returns.rolling(window=window_days).std() * np.sqrt(252) * 100  # Annualized %
    rolling_mean = dynamic_returns.rolling(window=window_days).mean() * 252  # Annualized return
    rolling_sharpe = rolling_mean / (rolling_vol / 100)  # Sharpe ratio
    
    # Remove NaN values
    rolling_vol = rolling_vol.dropna()
    rolling_sharpe = rolling_sharpe.dropna()
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add rolling volatility (primary y-axis)
    fig.add_trace(go.Scatter(
        x=rolling_vol.index,
        y=rolling_vol,
        mode='lines',
        name='Rolling Volatility',
        line=dict(color='#E74C3C', width=2),
        hovertemplate='<b>Volatility</b><br>Vol: %{y:.1f}%<extra></extra>',
        yaxis='y'
    ))
    
    # Add rolling Sharpe ratio (secondary y-axis)  
    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index,
        y=rolling_sharpe,
        mode='lines',
        name='Rolling Sharpe Ratio',
        line=dict(color='#3498DB', width=2),
        hovertemplate='<b>Sharpe Ratio</b><br>Sharpe: %{y:.2f}<extra></extra>',
        yaxis='y2'
    ))
    
    # Update layout with dual y-axes
    fig.update_layout(
        title=f"Dynamic Portfolio Rolling Risk Metrics ({window_days}-Day Window)",
        xaxis_title="Date",
        height=400,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis=dict(
            title="Volatility (%)",
            side="left",
            color='#E74C3C',
            tickformat='.1f'
        ),
        yaxis2=dict(
            title="Sharpe Ratio",
            side="right",
            overlaying="y",
            color='#3498DB',
            tickformat='.2f',
            dtick=1  # Show increments of 1
        )
    )
    
    return fig


def create_drawdown_chart(backtest_results, data, static_weights):
    """Create comprehensive drawdown chart for all portfolios and individual assets."""
    fig = go.Figure()
    
    # Get the backtest period data
    start_date = backtest_results['backtest_start']
    end_date = backtest_results['backtest_end']
    period_data = data.loc[start_date:end_date]
    
    # Colors for different series
    colors = {
        'Dynamic Portfolio': '#2E86C1',  # Blue
        'Static Portfolio': '#28B463',   # Green  
        'SPY': '#E74C3C',                # Red
        'TLT': '#F39C12',                # Orange
        'GLD': '#8E44AD'                 # Purple
    }
    
    # 1. Dynamic Portfolio Drawdown (from backtest results)
    dynamic_returns = backtest_results['portfolio_returns']
    if len(dynamic_returns) > 0:
        dynamic_cumulative = (1 + dynamic_returns).cumprod()
        dynamic_peak = dynamic_cumulative.cummax()
        dynamic_drawdown = (dynamic_cumulative - dynamic_peak) / dynamic_peak
        
        fig.add_trace(go.Scatter(
            x=dynamic_drawdown.index,
            y=dynamic_drawdown * 100,
            mode='lines',
            name='Dynamic Portfolio',
            line=dict(color=colors['Dynamic Portfolio'], width=2),
            hovertemplate='<b>Dynamic Portfolio</b><br>Drawdown: %{y:.2f}%<extra></extra>'
        ))
    
    # 2. Static Portfolio Drawdown
    static_returns = (period_data * static_weights).sum(axis=1)
    static_cumulative = (1 + static_returns).cumprod()
    static_peak = static_cumulative.cummax()
    static_drawdown = (static_cumulative - static_peak) / static_peak
    
    fig.add_trace(go.Scatter(
        x=static_drawdown.index,
        y=static_drawdown * 100,
        mode='lines',
        name='Static Portfolio',
        line=dict(color=colors['Static Portfolio'], width=2),
        hovertemplate='<b>Static Portfolio</b><br>Drawdown: %{y:.2f}%<extra></extra>'
    ))
    
    # 3. Individual Assets Drawdowns
    for asset in ['SPY', 'TLT', 'GLD']:
        asset_returns = period_data[asset]
        asset_cumulative = (1 + asset_returns).cumprod()
        asset_peak = asset_cumulative.cummax()
        asset_drawdown = (asset_cumulative - asset_peak) / asset_peak
        
        fig.add_trace(go.Scatter(
            x=asset_drawdown.index,
            y=asset_drawdown * 100,
            mode='lines',
            name=asset,
            line=dict(color=colors[asset], width=1.5, dash='dot'),
            hovertemplate=f'<b>{asset}</b><br>Drawdown: %{{y:.2f}}%<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title="Portfolio Drawdown Analysis",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis=dict(
            tickformat='.1f',
            range=[None, 5]  # Start from 5% to show small drawdowns, auto-scale down for larger ones
        )
    )
    
    # Add horizontal line at 0%
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig


def create_weight_evolution_chart(weight_history):
    """Create portfolio weight evolution chart for backtest."""
    fig = go.Figure()
    
    # Convert weight history to DataFrame for easier plotting
    weight_df = weight_history.set_index('date')
    
    # Colors for each asset
    colors = {'SPY': '#1f77b4', 'TLT': '#ff7f0e', 'GLD': '#2ca02c', 'Cash': '#d62728'}
    
    # Create stacked area chart using explicit fill approach
    # SPY: Fill from 0 to SPY%
    fig.add_trace(go.Scatter(
        x=weight_df.index,
        y=weight_df['SPY'] * 100,
        mode='lines',
        name='SPY',
        fill='tozeroy',
        line=dict(color=colors['SPY'], width=1),
        fillcolor=colors['SPY'],
        hovertemplate='<b>SPY</b>: %{y:.1f}%<extra></extra>'
    ))
    
    # TLT: Fill between SPY and SPY+TLT
    spy_cumulative = weight_df['SPY'] * 100
    tlt_cumulative = (weight_df['SPY'] + weight_df['TLT']) * 100
    
    # Create TLT area by combining upper and lower bounds
    x_vals = list(weight_df.index) + list(reversed(weight_df.index))
    y_vals = list(tlt_cumulative) + list(reversed(spy_cumulative))
    
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines',
        name='TLT',
        fill='toself',
        line=dict(color=colors['TLT'], width=0),
        fillcolor=colors['TLT'],
        showlegend=True,
        hoverinfo='skip'  # We'll add hover info separately
    ))
    
    # Add invisible line for TLT hover
    fig.add_trace(go.Scatter(
        x=weight_df.index,
        y=(weight_df['SPY'] + weight_df['TLT']/2) * 100,  # Middle of TLT area
        mode='lines',
        name='TLT_hover',
        line=dict(color='rgba(0,0,0,0)', width=0),
        showlegend=False,
        customdata=weight_df['TLT'] * 100,
        hovertemplate='<b>TLT</b>: %{customdata:.1f}%<extra></extra>'
    ))
    
    # GLD: Fill between SPY+TLT and SPY+TLT+GLD
    gld_cumulative = (weight_df['SPY'] + weight_df['TLT'] + weight_df['GLD']) * 100
    
    x_vals_gld = list(weight_df.index) + list(reversed(weight_df.index))
    y_vals_gld = list(gld_cumulative) + list(reversed(tlt_cumulative))
    
    fig.add_trace(go.Scatter(
        x=x_vals_gld,
        y=y_vals_gld,
        mode='lines',
        name='GLD',
        fill='toself',
        line=dict(color=colors['GLD'], width=0),
        fillcolor=colors['GLD'],
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Add invisible line for GLD hover
    fig.add_trace(go.Scatter(
        x=weight_df.index,
        y=(tlt_cumulative + weight_df['GLD']/2 * 100),  # Middle of GLD area
        mode='lines',
        name='GLD_hover',
        line=dict(color='rgba(0,0,0,0)', width=0),
        showlegend=False,
        customdata=weight_df['GLD'] * 100,
        hovertemplate='<b>GLD</b>: %{customdata:.1f}%<extra></extra>'
    ))
    
    if weight_df['Cash'].max() > 0.001:  # Only show Cash if it has meaningful allocation
        # Cash: Fill between SPY+TLT+GLD and 100%
        cash_cumulative = (weight_df['SPY'] + weight_df['TLT'] + weight_df['GLD'] + weight_df['Cash']) * 100
        
        x_vals_cash = list(weight_df.index) + list(reversed(weight_df.index))
        y_vals_cash = list(cash_cumulative) + list(reversed(gld_cumulative))
        
        fig.add_trace(go.Scatter(
            x=x_vals_cash,
            y=y_vals_cash,
            mode='lines',
            name='Cash',
            fill='toself',
            line=dict(color=colors['Cash'], width=0),
            fillcolor=colors['Cash'],
            showlegend=True,
            hoverinfo='skip'
        ))
        
        # Add invisible line for Cash hover
        fig.add_trace(go.Scatter(
            x=weight_df.index,
            y=(gld_cumulative + weight_df['Cash']/2 * 100),  # Middle of Cash area
            mode='lines',
            name='Cash_hover',
            line=dict(color='rgba(0,0,0,0)', width=0),
            showlegend=False,
            customdata=weight_df['Cash'] * 100,
            hovertemplate='<b>Cash</b>: %{customdata:.1f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title="Portfolio Weight Evolution During Backtest",
        xaxis_title="Date",
        yaxis_title="Weight (%)",
        height=400,
        yaxis=dict(range=[0, 100]),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode='x unified'
    )
    
    return fig




def create_returns_comparison_chart(expected_returns, weights):
    """Compare individual asset returns vs portfolio return."""
    # Portfolio return is weighted average of individual returns
    portfolio_return = np.sum(weights * expected_returns) * 100
    individual_returns = expected_returns * 100
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    fig = go.Figure()
    
    # Individual returns
    fig.add_trace(go.Bar(
        name='Individual Returns',
        x=ASSETS,
        y=individual_returns,
        marker_color=colors,
        opacity=0.7,
        text=[f"{r:.1f}%" for r in individual_returns],
        textposition='auto'
    ))
    
    # Portfolio return line
    fig.add_hline(
        y=portfolio_return,
        line_dash="solid",
        line_color="darkred",
        line_width=3,
        annotation_text=f"Portfolio: {portfolio_return:.1f}%",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title="Expected Returns Comparison: Individual Assets vs Portfolio (Annualized)",
        xaxis_title="Assets",
        yaxis_title="Expected Return (% Annual)",
        height=400,
        showlegend=False
    )
    
    return fig


def main():
    st.set_page_config(
        page_title="Minimum Variance Optimizer",
        page_icon="📉",
        layout="wide"
    )
    
    st.title("📉 Minimum Variance Portfolio Optimizer")
    st.markdown("**Simple and stable** approach focusing purely on risk reduction through diversification")
    
    # Sidebar controls
    st.sidebar.header("📊 Data Configuration")
    
    # Quick date selection buttons
    st.sidebar.markdown("**📅 Analysis Period:**")
    
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        if st.button("1Y", help="Past 1 year"):
            st.session_state.date_selection = "1Y"
        if st.button("5Y", help="Past 5 years"):
            st.session_state.date_selection = "5Y"
    
    with col2:
        if st.button("2Y", help="Past 2 years"):
            st.session_state.date_selection = "2Y"
        if st.button("10Y", help="Past 10 years"):
            st.session_state.date_selection = "10Y"
    
    with col3:
        if st.button("3Y", help="Past 3 years"):
            st.session_state.date_selection = "3Y"
        if st.button("YTD", help="Year to date"):
            st.session_state.date_selection = "YTD"
    
    # Initialize default if not set
    if 'date_selection' not in st.session_state:
        st.session_state.date_selection = "3Y"
    
    # Calculate dates based on selection
    end_date = datetime.now()
    
    if st.session_state.date_selection == "YTD":
        start_date = datetime(end_date.year, 1, 1)
    elif st.session_state.date_selection == "1Y":
        start_date = end_date - timedelta(days=365)
    elif st.session_state.date_selection == "2Y":
        start_date = end_date - timedelta(days=365 * 2)
    elif st.session_state.date_selection == "3Y":
        start_date = end_date - timedelta(days=365 * 3)
    elif st.session_state.date_selection == "5Y":
        start_date = end_date - timedelta(days=365 * 5)
    elif st.session_state.date_selection == "10Y":
        start_date = end_date - timedelta(days=365 * 10)
    else:
        start_date = end_date - timedelta(days=365 * 3)  # Default fallback
    
    # Show selected period
    st.sidebar.markdown(f"**Selected:** {st.session_state.date_selection} ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")
    
    # Option for custom dates
    use_custom = st.sidebar.checkbox("Use custom date range")
    if use_custom:
        date_range = st.sidebar.date_input(
            "Custom date range",
            value=(start_date, end_date),
            min_value=datetime(2010, 1, 1),
            max_value=end_date
        )
        if len(date_range) == 2:
            start_date, end_date = date_range
    
    st.sidebar.header("⚙️ Optimization Parameters")
    st.sidebar.markdown("*These windows determine current portfolio allocation*")
    
    # Rolling window controls
    sigma_window = st.sidebar.slider(
        "Volatility Window (days)",
        min_value=10,
        max_value=252,
        value=SIGMA_WINDOW,
        step=5,
        help="Window for volatility estimation (Breaking the Market: medium adaptation)"
    )
    
    rho_window = st.sidebar.slider(
        "Correlation Window (days)", 
        min_value=5,
        max_value=120,
        value=RHO_WINDOW,
        step=5,
        help="Window for correlation estimation (Breaking the Market: fast adaptation)"
    )
    
    
    # Backtest configuration (always enabled)
    st.sidebar.header("🧪 Backtest Settings")
    st.sidebar.markdown("*Historical performance uses realistic walk-forward backtesting*")
    
    use_walkforward = True  # Always use walk-forward backtest
    
    rebalance_freq = st.sidebar.selectbox(
        "Rebalancing Frequency",
        options=['weekly', 'monthly'],
        index=0,
        help="How often to rebalance the portfolio during historical analysis"
    )
    
    st.sidebar.markdown("⏱️ **Note**: Analysis uses realistic backtesting with no look-ahead bias")
    
    # Convert dates to strings for API calls  
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # Load and process data
    try:
        with st.spinner("Loading market data..."):
            loader = DataLoader()
            
            # Always fetch extra historical data for proper backtest windows
            buffer_days = max(sigma_window, rho_window) + 30  # Extra buffer for weekends/holidays
            extended_start = (pd.to_datetime(start_str) - timedelta(days=buffer_days)).strftime('%Y-%m-%d')
            data = loader.fetch_market_data(extended_start, end_str, use_treasury_bills=True)
            print(f"Fetched extended data from {extended_start} to {end_str} for backtesting")
            
            # Estimate covariance matrix using user-specified windows
            with st.spinner("Estimating covariance matrix..."):
                covariance_matrix = estimate_covariance_matrix(data, sigma_window, rho_window)
            
            # Also calculate expected returns for display
            with st.spinner("Calculating expected returns for display..."):
                expected_returns = estimate_expected_returns(data)
            
            # Optimize portfolio
            with st.spinner("Optimizing minimum variance portfolio..."):
                result = optimize_min_variance(covariance_matrix)
            
            if result['success']:
                # Detailed analysis
                analysis = analyze_min_variance_portfolio(
                    result['weights'], 
                    covariance_matrix, 
                    ASSETS
                )
                
                # Get actual window info
                window_info = get_estimation_info(len(data), sigma_window, rho_window)
                
                # Display methodology info  
                st.sidebar.markdown("### 📉 Current Settings")
                st.sidebar.markdown("**Objective**: Minimize w'Σw")
                st.sidebar.markdown("**Constraint**: Σw = 1, w ≥ 0")
                st.sidebar.markdown("**Cash Returns (Treasury Bills)**:")
                cash_vol = data['Cash'].std() * np.sqrt(252)
                cash_ret = ((1 + data['Cash'].mean()) ** 252) - 1
                st.sidebar.markdown(f"- BIL ETF: {cash_ret:.1%} return, {cash_vol:.2%} vol")
                
                # Main results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(
                        create_weights_chart(
                            result['weights'], 
                            result['diversification_ratio'],
                            result['effective_n_assets']
                        ),
                        use_container_width=True
                    )
                
                with col2:
                    st.plotly_chart(
                        create_correlation_heatmap(covariance_matrix),
                        use_container_width=True
                    )
                
                # Risk and returns analysis charts
                col3, col4 = st.columns(2)
                
                with col3:
                    st.plotly_chart(
                        create_risk_contribution_chart(analysis, covariance_matrix),
                        use_container_width=True
                    )
                    st.caption("Crimson: Current optimized risk contribution. Orange: Equal weight risk contribution (shows raw volatility effects).")
                
                with col4:
                    st.plotly_chart(
                        create_volatility_comparison_chart(analysis, sigma_window),
                        use_container_width=True
                    )
                
                # Analysis comparison: Rolling window vs Full period
                col5, col6 = st.columns(2)
                
                with col5:
                    # Current portfolio statistics based on rolling windows
                    st.subheader("📊 Current Portfolio Statistics")
                    st.markdown(f"*Based on rolling windows (σ={sigma_window}d, ρ={rho_window}d)*")
                    
                    # Get rolling window metrics for display
                    rolling_volatilities = np.sqrt(np.diag(covariance_matrix))
                    portfolio_vol = np.sqrt(result['weights'] @ covariance_matrix @ result['weights'])
                    
                    # Create compact table matching the right side format
                    current_stats_data = []
                    
                    # Portfolio volatility first
                    current_stats_data.append({
                        'Metric': 'Portfolio Volatility',
                        'Value': f"{portfolio_vol:.1%}"
                    })
                    
                    # Individual asset volatilities (exclude Cash)
                    for i, asset in enumerate(ASSETS):
                        if asset != 'Cash':  # Skip Cash volatility
                            current_stats_data.append({
                                'Metric': f'{asset} Volatility',
                                'Value': f"{rolling_volatilities[i]:.1%}"
                            })
                    
                    # Portfolio characteristics at bottom
                    current_stats_data.append({
                        'Metric': 'Diversification Ratio', 
                        'Value': f"{result['diversification_ratio']:.2f}"
                    })
                    current_stats_data.append({
                        'Metric': 'Effective # Assets',
                        'Value': f"{result['effective_n_assets']:.1f}"
                    })
                    
                    # Display as compact table
                    current_df = pd.DataFrame(current_stats_data)
                    st.dataframe(current_df, use_container_width=True, hide_index=True)
                
                with col6:
                    # Full period analysis table
                    st.subheader("📈 Full Period Analysis")
                    st.markdown(f"*Based on entire {st.session_state.date_selection} period*")
                    
                    returns_data = []
                    portfolio_return = np.sum(result['weights'] * expected_returns)
                    
                    # Calculate ACTUAL period volatility (matching the expected returns period)
                    actual_period_vols = data.std() * np.sqrt(252)  # Annualized vol over full period
                    portfolio_actual_vol = (data * result['weights']).sum(axis=1).std() * np.sqrt(252)
                    
                    # Individual assets
                    for i, asset in enumerate(ASSETS):
                        vol = actual_period_vols[asset]  # Use actual period volatility
                        ret = expected_returns[i]
                        
                        # Calculate Sharpe ratio (return / volatility)
                        if vol > 1e-6:  # Avoid division by zero
                            sharpe = ret / vol
                        else:
                            sharpe = None  # For Cash with near-zero vol
                        
                        returns_data.append({
                            'Asset': asset,
                            'Expected Return': f"{ret:.2%}",
                            'Volatility': f"{vol:.2%}",
                            'Sharpe Ratio': f"{sharpe:.2f}" if sharpe is not None else "N/A",
                            'Weight': f"{result['weights'][i]:.2%}",
                            'Contribution': f"{result['weights'][i] * ret:.2%}"
                        })
                    
                    # Add portfolio row (highlighted) - use actual period volatility
                    portfolio_sharpe = portfolio_return / portfolio_actual_vol if portfolio_actual_vol > 0 else 0
                    returns_data.append({
                        'Asset': "📊 PORTFOLIO",
                        'Expected Return': f"{portfolio_return:.2%}",
                        'Volatility': f"{portfolio_actual_vol:.2%}",
                        'Sharpe Ratio': f"{portfolio_sharpe:.2f}",
                        'Weight': "100.0%",
                        'Contribution': f"{portfolio_return:.2%}"
                    })
                    
                    returns_df = pd.DataFrame(returns_data)
                    
                    # Style the dataframe to highlight portfolio row
                    def highlight_portfolio(row):
                        if "PORTFOLIO" in row['Asset']:
                            return ['background-color: #ffeb3b; font-weight: bold; color: #000000'] * len(row)
                        return [''] * len(row)
                    
                    styled_df = returns_df.style.apply(highlight_portfolio, axis=1)
                    st.dataframe(styled_df, use_container_width=True)
                    
                    st.caption("*Expected returns for display only - NOT used in minimum variance optimization*")
                
                
                # Historical Performance Analysis
                st.subheader("📈 Historical Performance Analysis")
                
                # Run walk-forward backtest analysis
                cache_key = get_data_hash(data, start_str, end_str, sigma_window, rho_window, rebalance_freq)
                
                with st.spinner(f"Running walk-forward backtest... (this may take 30-60 seconds for {st.session_state.date_selection})"):
                    try:
                        # Pickle data for caching
                        data_pickle = pickle.dumps(data)
                        
                        # Use cached backtest
                        backtest_results = get_cached_backtest_result(
                            cache_key,
                            data_pickle,
                            start_str, 
                            end_str,
                            rebalance_freq,
                            max(sigma_window, rho_window),
                            sigma_window,
                            rho_window
                        )
                        
                        # Compare with static weights
                        static_comparison = compare_with_static_weights(
                            backtest_results, data, result['weights']
                        )
                        
                    except Exception as e:
                        st.error(f"Backtest failed: {str(e)}")
                        st.info("Please try adjusting the date range or parameters.")
                        st.stop()
                
                # Main performance chart showing dynamic vs static
                backtest_chart = create_backtest_performance_chart(backtest_results, data, result['weights'])
                st.plotly_chart(backtest_chart, use_container_width=True)
                
                # Show weight evolution chart
                st.subheader("⚖️ Portfolio Weight Evolution")
                weight_evolution_chart = create_weight_evolution_chart(backtest_results['weight_history'])
                st.plotly_chart(weight_evolution_chart, use_container_width=True)
                
                # Show backtest insights
                st.markdown("### 🔍 Backtest Insights")
                if backtest_results['n_optimization_errors'] > 0:
                    st.warning(f"⚠️ {backtest_results['n_optimization_errors']} optimization errors occurred during backtest")
                
                st.info(f"📊 **Rebalancing Summary**: Portfolio was rebalanced {backtest_results['n_rebalances']} times "
                       f"with an average turnover of {backtest_results['average_turnover']:.1%} per rebalance. "
                       f"This represents the realistic costs and benefits of dynamic portfolio management.")
                
                # Performance metrics summary
                st.markdown("### 🔬 Dynamic vs Static Comparison")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric(
                        "Dynamic Total Return", 
                        f"{backtest_results['total_return']:.1%}",
                        f"{static_comparison['return_difference']:+.1%} vs static"
                    )
                
                with col2:
                    # Calculate static annualized return for comparison
                    static_period_days = len(data.loc[backtest_results['backtest_start']:backtest_results['backtest_end']])
                    static_annualized_return = (1 + static_comparison['static_total_return']) ** (252 / static_period_days) - 1
                    
                    st.metric(
                        "Dynamic Annualized Return", 
                        f"{backtest_results['annualized_return']:.1%}",
                        f"{backtest_results['annualized_return'] - static_annualized_return:+.1%} vs static"
                    )
                
                with col3:
                    st.metric(
                        "Dynamic Volatility", 
                        f"{backtest_results['volatility']:.1%}",
                        f"{static_comparison['volatility_difference']:+.1%} vs static"
                    )
                
                with col4:
                    st.metric(
                        "Dynamic Sharpe Ratio", 
                        f"{backtest_results['sharpe_ratio']:.2f}",
                        f"{static_comparison['sharpe_difference']:+.2f} vs static"
                    )
                
                with col5:
                    st.metric(
                        "Dynamic Max Drawdown", 
                        f"{backtest_results['max_drawdown']:.1%}",
                        f"{static_comparison['drawdown_difference']:+.1%} vs static"
                    )
                
                # Calculate SPY performance for comparison
                spy_period_data = data.loc[backtest_results['backtest_start']:backtest_results['backtest_end']]
                spy_returns = spy_period_data['SPY']
                spy_cumulative = (1 + spy_returns).cumprod()
                spy_total_return = spy_cumulative.iloc[-1] - 1
                spy_volatility = spy_returns.std() * np.sqrt(252)
                spy_sharpe = (spy_returns.mean() * 252) / spy_volatility if spy_volatility > 0 else 0
                spy_peak = spy_cumulative.cummax()
                spy_drawdown = (spy_cumulative - spy_peak) / spy_peak
                spy_max_drawdown = spy_drawdown.min()
                
                # Static Portfolio Performance (horizontal format)
                st.markdown("---")
                st.markdown("### ⚖️ Static Portfolio Performance")
                
                # Calculate static annualized return
                static_period_days = len(data.loc[backtest_results['backtest_start']:backtest_results['backtest_end']])
                static_annualized_return = (1 + static_comparison['static_total_return']) ** (252 / static_period_days) - 1
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric(
                        "Static Total Return",
                        f"{static_comparison['static_total_return']:.1%}",
                        f"{-static_comparison['return_difference']:+.1%} vs dynamic"
                    )
                
                with col2:
                    st.metric(
                        "Static Annualized Return",
                        f"{static_annualized_return:.1%}",
                        f"{static_annualized_return - backtest_results['annualized_return']:+.1%} vs dynamic"
                    )
                
                with col3:
                    st.metric(
                        "Static Volatility",
                        f"{static_comparison['static_volatility']:.1%}",
                        f"{-static_comparison['volatility_difference']:+.1%} vs dynamic"
                    )
                
                with col4:
                    st.metric(
                        "Static Sharpe Ratio",
                        f"{static_comparison['static_sharpe']:.2f}",
                        f"{-static_comparison['sharpe_difference']:+.2f} vs dynamic"
                    )
                
                with col5:
                    st.metric(
                        "Static Max Drawdown",
                        f"{static_comparison['static_max_drawdown']:.1%}",
                        f"{-static_comparison['drawdown_difference']:+.1%} vs dynamic"
                    )
                
                # SPY Benchmark Performance (horizontal format)
                st.markdown("---")
                st.markdown("### 📈 SPY Benchmark Performance")
                
                # Calculate differences vs dynamic portfolio
                spy_return_diff = spy_total_return - backtest_results['total_return']
                spy_vol_diff = spy_volatility - backtest_results['volatility']
                spy_sharpe_diff = spy_sharpe - backtest_results['sharpe_ratio']
                spy_drawdown_diff = spy_max_drawdown - backtest_results['max_drawdown']
                
                # Calculate SPY annualized return
                spy_period_days = len(spy_period_data)
                spy_annualized_return = (1 + spy_total_return) ** (252 / spy_period_days) - 1
                spy_annualized_diff = spy_annualized_return - backtest_results['annualized_return']
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric(
                        "SPY Total Return",
                        f"{spy_total_return:.1%}",
                        f"{spy_return_diff:+.1%} vs dynamic"
                    )
                
                with col2:
                    st.metric(
                        "SPY Annualized Return",
                        f"{spy_annualized_return:.1%}",
                        f"{spy_annualized_diff:+.1%} vs dynamic"
                    )
                
                with col3:
                    st.metric(
                        "SPY Volatility", 
                        f"{spy_volatility:.1%}",
                        f"{spy_vol_diff:+.1%} vs dynamic"
                    )
                
                with col4:
                    st.metric(
                        "SPY Sharpe Ratio",
                        f"{spy_sharpe:.2f}",
                        f"{spy_sharpe_diff:+.2f} vs dynamic"
                    )
                
                with col5:
                    st.metric(
                        "SPY Max Drawdown",
                        f"{spy_max_drawdown:.1%}",
                        f"{spy_drawdown_diff:+.1%} vs dynamic"
                    )
                
                # Drawdown Analysis Chart
                st.markdown("---")
                st.subheader("📉 Drawdown Analysis")
                st.markdown("*Shows peak-to-trough declines for portfolios and individual assets*")
                
                drawdown_chart = create_drawdown_chart(backtest_results, data, result['weights'])
                st.plotly_chart(drawdown_chart, use_container_width=True)
                
                # Rolling Risk Metrics Chart
                st.markdown("---")
                st.subheader("📊 Rolling Risk Metrics")
                st.markdown("*60-day rolling volatility and Sharpe ratio for dynamic portfolio*")
                
                rolling_metrics_chart = create_rolling_metrics_chart(backtest_results, window_days=60)
                st.plotly_chart(rolling_metrics_chart, use_container_width=True)
                
                # Note: Advanced rolling metrics charts removed - backtest provides comprehensive analysis
                
                
            else:
                st.error("❌ Optimization failed!")
                st.error(result.get('error', 'Unknown error'))
                
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()