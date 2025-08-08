"""
Data persistence layer for portfolio optimizer.

Handles storage and retrieval of optimization results, backtest history,
and live trading records. Provides audit trail and performance monitoring.
"""

import sqlite3
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, date
from pathlib import Path
import warnings

from config import ASSETS


class PortfolioDatabase:
    """
    SQLite database for storing portfolio optimization results and history.
    
    Provides persistent storage for:
    - Daily optimization results and portfolio weights
    - Backtest performance and parameters  
    - Live trading decisions and execution records
    - Model performance tracking over time
    """
    
    def __init__(self, db_path: str = "portfolio_optimizer.db"):
        """
        Initialize database connection and create tables if needed.
        
        Parameters
        ----------
        db_path : str
            Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables for storing optimization results."""
        with sqlite3.connect(self.db_path) as conn:
            # Daily optimization results
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_optimizations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    optimization_method TEXT NOT NULL,
                    expected_returns TEXT NOT NULL,  -- JSON array
                    covariance_matrix TEXT NOT NULL, -- JSON 2D array  
                    optimal_weights TEXT NOT NULL,   -- JSON array
                    objective_value REAL NOT NULL,
                    expected_return REAL NOT NULL,
                    portfolio_volatility REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    turnover REAL,
                    transaction_costs REAL,
                    parameters TEXT,  -- JSON of all parameters used
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, optimization_method, parameters)
                )
            """)
            
            # Backtest results
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    start_date DATE NOT NULL,
                    end_date DATE NOT NULL,
                    optimization_method TEXT NOT NULL,
                    parameters TEXT NOT NULL,  -- JSON
                    total_return REAL NOT NULL,
                    annualized_return REAL NOT NULL,
                    volatility REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    calmar_ratio REAL,
                    total_costs REAL,
                    avg_turnover REAL,
                    n_rebalances INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Daily portfolio weights (for live tracking)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_weights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    asset TEXT NOT NULL,
                    weight REAL NOT NULL,
                    optimization_id INTEGER,
                    FOREIGN KEY (optimization_id) REFERENCES daily_optimizations(id),
                    UNIQUE(date, asset, optimization_id)
                )
            """)
            
            # Model performance tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    realized_return REAL,
                    predicted_return REAL,
                    realized_volatility REAL,
                    predicted_volatility REAL,
                    attribution TEXT,  -- JSON breakdown
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date)
                )
            """)
    
    def save_optimization_result(
        self,
        date: date,
        method: str,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        optimal_weights: np.ndarray,
        objective_value: float,
        portfolio_stats: Dict[str, float],
        parameters: Dict[str, Any],
        turnover: Optional[float] = None,
        transaction_costs: Optional[float] = None
    ) -> int:
        """
        Save daily optimization result to database.
        
        Returns
        -------
        int
            Database ID of saved record
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT OR REPLACE INTO daily_optimizations 
                (date, optimization_method, expected_returns, covariance_matrix,
                 optimal_weights, objective_value, expected_return, 
                 portfolio_volatility, sharpe_ratio, turnover, 
                 transaction_costs, parameters)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                date,
                method,
                json.dumps(expected_returns.tolist()),
                json.dumps(covariance_matrix.tolist()),
                json.dumps(optimal_weights.tolist()),
                objective_value,
                portfolio_stats.get('expected_return', 0),
                portfolio_stats.get('volatility', 0),
                portfolio_stats.get('sharpe_ratio', 0),
                turnover,
                transaction_costs,
                json.dumps(parameters)
            ))
            
            opt_id = cursor.lastrowid
            
            # Save individual asset weights
            for i, asset in enumerate(ASSETS):
                conn.execute("""
                    INSERT OR REPLACE INTO portfolio_weights
                    (date, asset, weight, optimization_id)
                    VALUES (?, ?, ?, ?)
                """, (date, asset, float(optimal_weights[i]), opt_id))
            
            return opt_id
    
    def save_backtest_result(
        self,
        name: str,
        start_date: date,
        end_date: date,
        method: str,
        parameters: Dict[str, Any],
        performance_metrics: Dict[str, float]
    ) -> int:
        """Save backtest results to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO backtest_results
                (name, start_date, end_date, optimization_method, parameters,
                 total_return, annualized_return, volatility, sharpe_ratio,
                 max_drawdown, calmar_ratio, total_costs, avg_turnover, n_rebalances)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                name,
                start_date,
                end_date, 
                method,
                json.dumps(parameters),
                performance_metrics.get('total_return', 0),
                performance_metrics.get('annualized_return', 0),
                performance_metrics.get('annualized_volatility', 0),
                performance_metrics.get('sharpe_ratio', 0),
                performance_metrics.get('max_drawdown', 0),
                performance_metrics.get('calmar_ratio', 0),
                performance_metrics.get('total_transaction_costs', 0),
                performance_metrics.get('avg_daily_turnover', 0),
                performance_metrics.get('n_observations', 0)
            ))
            
            return cursor.lastrowid
    
    def get_optimization_history(
        self, 
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        method: Optional[str] = None
    ) -> pd.DataFrame:
        """Retrieve historical optimization results."""
        query = "SELECT * FROM daily_optimizations WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        if method:
            query += " AND optimization_method = ?"
            params.append(method)
            
        query += " ORDER BY date"
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def get_portfolio_weights_history(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """Get historical portfolio weights in wide format."""
        query = """
            SELECT pw.date, pw.asset, pw.weight
            FROM portfolio_weights pw
            JOIN daily_optimizations do ON pw.optimization_id = do.id
            WHERE 1=1
        """
        params = []
        
        if start_date:
            query += " AND pw.date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND pw.date <= ?"
            params.append(end_date)
            
        query += " ORDER BY pw.date, pw.asset"
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        # Pivot to wide format
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            return df.pivot(index='date', columns='asset', values='weight')
        else:
            return pd.DataFrame()
    
    def get_backtest_history(self) -> pd.DataFrame:
        """Get all backtest results."""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(
                "SELECT * FROM backtest_results ORDER BY created_at DESC", 
                conn
            )
    
    def get_model_performance(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """Get model performance tracking data."""
        query = "SELECT * FROM model_performance WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
            
        query += " ORDER BY date"
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def cleanup_old_records(self, days_to_keep: int = 365):
        """Remove optimization records older than specified days."""
        cutoff_date = datetime.now().date() - pd.Timedelta(days=days_to_keep)
        
        with sqlite3.connect(self.db_path) as conn:
            # Clean up old optimizations and their weights
            conn.execute("""
                DELETE FROM portfolio_weights 
                WHERE optimization_id IN (
                    SELECT id FROM daily_optimizations WHERE date < ?
                )
            """, (cutoff_date,))
            
            conn.execute("""
                DELETE FROM daily_optimizations WHERE date < ?
            """, (cutoff_date,))
            
            conn.execute("""
                DELETE FROM model_performance WHERE date < ?
            """, (cutoff_date,))


# Convenience functions for common operations
def get_database(db_path: str = "portfolio_optimizer.db") -> PortfolioDatabase:
    """Get database instance (singleton pattern)."""
    if not hasattr(get_database, '_instance'):
        get_database._instance = PortfolioDatabase(db_path)
    return get_database._instance


def save_daily_optimization(
    date: date,
    optimization_result: Dict[str, Any],
    parameters: Dict[str, Any]
) -> int:
    """Convenience function to save optimization result."""
    db = get_database()
    
    return db.save_optimization_result(
        date=date,
        method=optimization_result.get('method', 'unknown'),
        expected_returns=optimization_result['mu'],
        covariance_matrix=optimization_result['cov_matrix'], 
        optimal_weights=optimization_result['weights'],
        objective_value=optimization_result['objective_value'],
        portfolio_stats=optimization_result['portfolio_stats'],
        parameters=parameters,
        turnover=optimization_result['attribution'].get('turnover'),
        transaction_costs=optimization_result['attribution'].get('transaction_costs')
    )