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

from data.loader import DataLoader
from estimators import estimate_covariance_matrix, estimate_expected_returns, get_estimation_info
from optimizer import optimize_min_variance, analyze_min_variance_portfolio
from config import ASSETS, DEFAULT_DATE_RANGE_YEARS, SIGMA_WINDOW, RHO_WINDOW


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


def create_rolling_metrics_chart(performance_data, window_days=252):
    """Create rolling Sharpe ratio and volatility chart."""
    portfolio_returns = performance_data['portfolio_daily_returns']
    
    # Calculate rolling metrics
    rolling_sharpe = (portfolio_returns.rolling(window_days).mean() * 252) / (portfolio_returns.rolling(window_days).std() * np.sqrt(252))
    rolling_vol = portfolio_returns.rolling(window_days).std() * np.sqrt(252)
    
    fig = go.Figure()
    
    # Rolling Sharpe ratio
    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index,
        y=rolling_sharpe,
        mode='lines',
        name='Rolling Sharpe Ratio (1Y)',
        line=dict(color='blue', width=2),
        yaxis='y'
    ))
    
    # Rolling volatility on secondary y-axis
    fig.add_trace(go.Scatter(
        x=rolling_vol.index,
        y=rolling_vol * 100,  # Convert to percentage
        mode='lines',
        name='Rolling Volatility (1Y)',
        line=dict(color='orange', width=2),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Rolling Risk-Adjusted Performance (1-Year Window)",
        xaxis_title="Date",
        yaxis=dict(title="Sharpe Ratio", side="left"),
        yaxis2=dict(title="Volatility (%)", side="right", overlaying="y"),
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig


def create_drawdown_chart(performance_data):
    """Create drawdown analysis chart."""
    drawdowns = performance_data['drawdowns'] * 100  # Convert to percentage
    
    fig = go.Figure()
    
    # Drawdown area chart
    fig.add_trace(go.Scatter(
        x=drawdowns.index,
        y=drawdowns,
        fill='tonexty',
        mode='lines',
        name='Drawdown',
        line=dict(color='red', width=1),
        fillcolor='rgba(255, 0, 0, 0.3)'
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    
    fig.update_layout(
        title=f"Portfolio Drawdowns (Max: {performance_data['max_drawdown']:.1%})",
        xaxis_title="Date", 
        yaxis_title="Drawdown (%)",
        height=300,
        showlegend=False
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
    
    st.sidebar.markdown("**Note**: Only the most recent data within these windows affects the current portfolio allocation, regardless of the total date range selected.")
    
    # Convert dates to strings for API calls  
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # Load and process data
    try:
        with st.spinner("Loading market data..."):
            loader = DataLoader()
            data = loader.fetch_market_data(start_str, end_str, use_treasury_bills=True)
            
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
                st.sidebar.markdown("**Active Windows**:")
                st.sidebar.markdown(f"- Volatility: {window_info['sigma_actual']} days")
                st.sidebar.markdown(f"- Correlation: {window_info['rho_actual']} days")
                st.sidebar.markdown("**Cash Returns (Treasury Bills)**:")
                cash_vol = data['Cash'].std() * np.sqrt(252)
                cash_ret = ((1 + data['Cash'].mean()) ** 252) - 1
                st.sidebar.markdown(f"- BIL ETF: {cash_ret:.1%} return, {cash_vol:.2%} vol")
                st.sidebar.markdown("**Key Benefits**:")
                st.sidebar.markdown("- No expected returns needed")
                st.sidebar.markdown("- Naturally diversifies") 
                st.sidebar.markdown("- Stable and robust")
                st.sidebar.markdown("- Eliminates corner solutions")
                
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
                
                # Returns analysis
                col5, col6 = st.columns(2)
                
                with col5:
                    st.plotly_chart(
                        create_returns_comparison_chart(expected_returns, result['weights']),
                        use_container_width=True
                    )
                
                with col6:
                    # Expected returns metrics table
                    st.subheader("📈 Returns, Risk & Sharpe Analysis")
                    
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
                
                # Portfolio statistics
                st.subheader("📊 Portfolio Statistics")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Portfolio Volatility", f"{result['portfolio_volatility']:.2%}")
                
                with col2:
                    st.metric("Diversification Ratio", f"{result['diversification_ratio']:.2f}")
                    st.caption("Weighted avg vol / portfolio vol. >1.0 = diversification benefit")
                
                with col3:
                    st.metric("Effective # Assets", f"{result['effective_n_assets']:.1f}")
                    st.caption("1/Σ(w²). 4.0=equal weights, 1.0=single asset")
                
                with col4:
                    st.metric("Max Weight", f"{analysis['max_weight']:.1%}")
                
                with col5:
                    st.metric("Active Assets", f"{analysis['n_nonzero_assets']}")
                
                # Asset details table
                st.subheader("🔍 Detailed Asset Analysis")
                
                # Prepare table data
                table_data = []
                individual_vols = np.sqrt(np.diag(covariance_matrix))
                
                for i, asset in enumerate(ASSETS):
                    item = analysis['asset_analysis'][i]
                    
                    table_data.append({
                        'Asset': asset,
                        'Weight': f"{item['weight']:.2%}",
                        'Individual Vol': f"{item['individual_vol']:.2%}",
                        'Risk Contribution': f"{item['risk_contrib_pct']:.2%}",
                        'Portfolio Correlation': f"{item['correlation_with_portfolio']:.3f}",
                    })
                
                df = pd.DataFrame(table_data)
                st.dataframe(df, use_container_width=True)
                
                # Historical Performance Analysis
                st.subheader("📈 Historical Performance Analysis")
                
                # Calculate portfolio performance over the selected period
                with st.spinner("Calculating historical performance..."):
                    performance_data = calculate_portfolio_performance(data, result['weights'])
                
                # Main performance chart (full width)
                st.plotly_chart(
                    create_performance_chart(performance_data, ASSETS),
                    use_container_width=True
                )
                
                # Performance metrics summary
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric(
                        "Portfolio Total Return", 
                        f"{performance_data['portfolio_total_return']:.1%}",
                        f"{st.session_state.date_selection} period"
                    )
                
                with col2:
                    st.metric(
                        "Portfolio Volatility", 
                        f"{performance_data['portfolio_vol']:.1%}",
                        "Annualized"
                    )
                
                with col3:
                    st.metric(
                        "Portfolio Sharpe Ratio", 
                        f"{performance_data['portfolio_sharpe']:.2f}",
                        "Risk-adjusted return"
                    )
                
                with col4:
                    st.metric(
                        "Max Drawdown", 
                        f"{performance_data['max_drawdown']:.1%}",
                        "Worst peak-to-trough"
                    )
                
                with col5:
                    # Compare to best individual asset
                    individual_returns = {}
                    for asset in ASSETS:
                        if asset in data.columns:
                            total_return = performance_data['individual_cumulative'][asset].iloc[-1] - 1
                            individual_returns[asset] = total_return
                    
                    best_asset = max(individual_returns, key=individual_returns.get) if individual_returns else "N/A"
                    best_return = individual_returns.get(best_asset, 0)
                    outperformance = performance_data['portfolio_total_return'] - best_return
                    
                    st.metric(
                        f"vs Best Asset ({best_asset})", 
                        f"{outperformance:+.1%}",
                        "Outperformance" if outperformance >= 0 else "Underperformance"
                    )
                
                # Advanced metrics charts (Phase 3)
                if len(data) > 252:  # Only show if we have enough data for rolling metrics
                    col_left, col_right = st.columns(2)
                    
                    with col_left:
                        st.plotly_chart(
                            create_rolling_metrics_chart(performance_data),
                            use_container_width=True
                        )
                    
                    with col_right:
                        st.plotly_chart(
                            create_drawdown_chart(performance_data),
                            use_container_width=True
                        )
                
                # Key insights summary
                st.subheader("🎯 Portfolio Insights")
                
                st.markdown("""
                **Key Metrics from Your Portfolio:**
                - **Diversification Ratio = {:.2f}**: Ratio of weighted-average volatility to portfolio volatility (>1 indicates diversification benefit)
                - **Effective Assets = {:.1f}**: Concentration measure (4.0 = equal weights, 1.0 = single asset)
                - **Max Weight = {:.1%}**: Largest position (shows how concentrated the portfolio is)
                
                This minimum variance approach focuses purely on **"What mix of assets gives me the smoothest ride?"**
                by optimizing the covariance structure without requiring expected return predictions.
                """.format(
                    result['diversification_ratio'],
                    result['effective_n_assets'],
                    analysis['max_weight']
                ))
                
            else:
                st.error("❌ Optimization failed!")
                st.error(result.get('error', 'Unknown error'))
                
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()