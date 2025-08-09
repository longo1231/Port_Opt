"""
Portfolio history and persistence UI components.

Provides Streamlit components for viewing and managing historical
optimization results and backtest performance.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date, timedelta
import json

from data.persistence import PortfolioDatabase, get_database
from config import ASSETS


def show_optimization_history():
    """Display historical optimization results."""
    st.subheader("📊 Optimization History")
    
    db = get_database()
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "From Date", 
            value=datetime.now().date() - timedelta(days=30),
            key="hist_start"
        )
    with col2:
        end_date = st.date_input(
            "To Date",
            value=datetime.now().date(),
            key="hist_end"
        )
    
    # Get data
    history = db.get_optimization_history(start_date, end_date)
    
    if history.empty:
        st.info("No optimization history found for the selected date range.")
        return
    
    # Summary statistics
    st.write(f"**{len(history)} optimization records found**")
    
    # Method breakdown
    method_counts = history['optimization_method'].value_counts()
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Methods Used:**")
        for method, count in method_counts.items():
            st.write(f"- {method}: {count} times")
    
    with col2:
        avg_metrics = history[['expected_return', 'portfolio_volatility', 'sharpe_ratio']].mean()
        st.write("**Average Performance:**")
        st.write(f"- Expected Return: {avg_metrics['expected_return']:.2%}")
        st.write(f"- Volatility: {avg_metrics['portfolio_volatility']:.2%}")  
        st.write(f"- Sharpe Ratio: {avg_metrics['sharpe_ratio']:.3f}")
    
    # Time series charts
    if len(history) > 1:
        history['date'] = pd.to_datetime(history['date'])
        history = history.sort_values('date')
        
        # Performance over time
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history['date'],
            y=history['expected_return'],
            mode='lines+markers',
            name='Expected Return',
            line=dict(color='green')
        ))
        
        fig.add_trace(go.Scatter(
            x=history['date'],
            y=history['sharpe_ratio'],
            mode='lines+markers', 
            name='Sharpe Ratio',
            yaxis='y2',
            line=dict(color='blue')
        ))
        
        fig.update_layout(
            title="Expected Performance Over Time",
            xaxis_title="Date",
            yaxis=dict(title="Expected Return", side="left", tickformat=".1%"),
            yaxis2=dict(title="Sharpe Ratio", side="right", overlaying="y"),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Portfolio weights over time
    weights_history = db.get_portfolio_weights_history(start_date, end_date)
    
    if not weights_history.empty:
        st.subheader("Portfolio Allocation Over Time")
        
        fig = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, asset in enumerate(ASSETS):
            if asset in weights_history.columns:
                fig.add_trace(go.Scatter(
                    x=weights_history.index,
                    y=weights_history[asset],
                    mode='lines',
                    name=asset,
                    stackgroup='one',
                    fillcolor=colors[i % len(colors)]
                ))
        
        fig.update_layout(
            title="Portfolio Weights Evolution",
            xaxis_title="Date",
            yaxis_title="Weight",
            yaxis=dict(tickformat='.0%'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    if st.checkbox("Show Detailed Records"):
        display_cols = [
            'date', 'optimization_method', 'expected_return', 
            'portfolio_volatility', 'sharpe_ratio', 'turnover', 
            'transaction_costs', 'objective_value'
        ]
        
        display_history = history[display_cols].copy()
        display_history['expected_return'] = display_history['expected_return'].map('{:.2%}'.format)
        display_history['portfolio_volatility'] = display_history['portfolio_volatility'].map('{:.2%}'.format)
        display_history['sharpe_ratio'] = display_history['sharpe_ratio'].map('{:.3f}'.format)
        
        st.dataframe(display_history, use_container_width=True)


def show_backtest_history():
    """Display historical backtest results."""
    st.subheader("🧪 Backtest History")
    
    db = get_database()
    backtest_history = db.get_backtest_history()
    
    if backtest_history.empty:
        st.info("No backtest results found. Run some backtests to see history here.")
        return
    
    # Summary table
    display_cols = [
        'name', 'start_date', 'end_date', 'optimization_method',
        'annualized_return', 'volatility', 'sharpe_ratio', 
        'max_drawdown', 'total_costs', 'created_at'
    ]
    
    display_df = backtest_history[display_cols].copy()
    display_df['annualized_return'] = display_df['annualized_return'].map('{:.2%}'.format)
    display_df['volatility'] = display_df['volatility'].map('{:.2%}'.format) 
    display_df['sharpe_ratio'] = display_df['sharpe_ratio'].map('{:.3f}'.format)
    display_df['max_drawdown'] = display_df['max_drawdown'].map('{:.2%}'.format)
    display_df['total_costs'] = display_df['total_costs'].map('{:.2%}'.format)
    
    st.dataframe(display_df, use_container_width=True)
    
    # Performance comparison
    if len(backtest_history) > 1:
        st.subheader("Backtest Comparison")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=backtest_history['volatility'],
            y=backtest_history['annualized_return'], 
            mode='markers+text',
            text=backtest_history['name'],
            textposition="top center",
            marker=dict(
                size=10,
                color=backtest_history['sharpe_ratio'],
                colorscale='Viridis',
                colorbar=dict(title="Sharpe Ratio")
            ),
            name="Backtests"
        ))
        
        fig.update_layout(
            title="Risk-Return Comparison of Backtests",
            xaxis_title="Volatility",
            yaxis_title="Annualized Return", 
            xaxis=dict(tickformat='.1%'),
            yaxis=dict(tickformat='.1%'),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)


def save_current_backtest(backtest_results, performance, parameters):
    """Save current backtest results to database."""
    if backtest_results is None or performance is None:
        return False
    
    # Get backtest name from user
    name = st.text_input(
        "Backtest Name",
        value=f"Backtest_{datetime.now().strftime('%Y%m%d_%H%M')}",
        help="Give this backtest a memorable name"
    )
    
    if st.button("💾 Save Backtest Results"):
        try:
            db = get_database()
            
            # Extract date range from results
            start_date = backtest_results['portfolio_returns'].index[0].date()
            end_date = backtest_results['portfolio_returns'].index[-1].date() 
            
            backtest_id = db.save_backtest_result(
                name=name,
                start_date=start_date,
                end_date=end_date,
                method=backtest_results['settings']['optimization_method'],
                parameters=parameters,
                performance_metrics=performance
            )
            
            st.success(f"✅ Backtest saved with ID: {backtest_id}")
            return True
            
        except Exception as e:
            st.error(f"Failed to save backtest: {str(e)}")
            return False
    
    return False


def manage_database():
    """Database management tools."""
    st.subheader("🗄️ Database Management")
    
    db = get_database()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Database Stats"):
            with st.spinner("Querying database..."):
                try:
                    import sqlite3
                    with sqlite3.connect(db.db_path) as conn:
                        # Count records in each table
                        tables = ['daily_optimizations', 'backtest_results', 'portfolio_weights', 'model_performance']
                        
                        stats = {}
                        for table in tables:
                            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                            stats[table] = count
                        
                        st.write("**Record Counts:**")
                        for table, count in stats.items():
                            st.write(f"- {table}: {count:,}")
                        
                        # Database size
                        size_mb = db.db_path.stat().st_size / (1024 * 1024)
                        st.write(f"**Database Size:** {size_mb:.2f} MB")
                        
                except Exception as e:
                    st.error(f"Error querying database: {e}")
    
    with col2:
        if st.button("🧹 Cleanup Old Records"):
            days_to_keep = st.number_input("Days to Keep", min_value=30, max_value=365*5, value=365)
            
            if st.confirm(f"Delete records older than {days_to_keep} days?"):
                try:
                    db.cleanup_old_records(days_to_keep)
                    st.success("✅ Old records cleaned up")
                except Exception as e:
                    st.error(f"Cleanup failed: {e}")
    
    with col3:
        if st.button("📁 Database Location"):
            st.info(f"Database file: `{db.db_path.absolute()}`")
            st.write("You can back up this file to preserve your optimization history.")


def show_portfolio_persistence_tab():
    """Main portfolio history tab content."""
    st.header("📈 Portfolio History & Persistence")
    
    tab1, tab2, tab3 = st.tabs(["Optimization History", "Backtest History", "Database Management"])
    
    with tab1:
        show_optimization_history()
    
    with tab2:
        show_backtest_history()
    
    with tab3:
        manage_database()