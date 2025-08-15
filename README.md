# Minimum Variance Portfolio Optimizer

A clean and stable minimum variance portfolio optimizer that achieves natural diversification by eliminating expected returns from the optimization.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)

## 🎯 Overview

This application implements minimum variance portfolio optimization with multiple portfolio templates. By eliminating expected returns from the objective function, it avoids corner solutions and naturally creates diversified portfolios.

**Key Innovation**: Focuses purely on **risk reduction through diversification** rather than trying to predict which assets will perform best.

### Core Features

- **Minimum Variance Optimization**: `min w^T Σ w` (no expected returns needed)
- **📊 Portfolio Templates**: Switch between Current (SPY/TLT/GLD) and MAG7 tech stocks
- **🎯 Volatility Targeting with Leverage**: Professional-grade leverage system
- **Natural Diversification**: Only way to reduce risk is through correlation benefits
- **Stable Results**: Covariance matrices are more stable than expected returns
- **Real-time Data**: Yahoo Finance integration with robust error handling
- **Revolutionary Visualizations**: See exactly how leverage works
- **Smart Chart Scaling**: Handles extreme performers (like NVDA's 865% AI boom gains)
- **Clean Interface**: Clear visualization of diversification benefits

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   streamlit run app.py
   ```

3. **Open your browser** to http://localhost:8501

## 📁 Project Structure

```
Port_Optimizer/
├── app.py                  # Streamlit dashboard
├── optimizer.py            # Minimum variance optimization
├── estimators.py           # Covariance matrix estimation
├── config.py               # Configuration parameters
├── data/
│   └── loader.py          # Yahoo Finance data (simple returns)
├── test_phase1.py          # Core functionality tests
├── test_streamlit.py       # Market data integration tests
└── CLAUDE.md               # Development documentation
```

## 📊 Portfolio Templates

Choose between different asset universes in the sidebar:

**Current Portfolio**: `SPY, TLT, GLD, Cash`
- **SPY**: S&P 500 ETF (broad market exposure)
- **TLT**: 20+ Year Treasury ETF (safe haven, negative correlation with SPY)  
- **GLD**: Gold ETF (inflation hedge, portfolio diversifier)
- **Cash**: Risk-free Treasury bills via BIL ETF

**MAG7 Portfolio**: `AMZN, AAPL, MSFT, GOOG, META, TSLA, NVDA, Cash`
- **Focus**: Magnificent 7 technology stocks
- **Characteristics**: Higher growth potential, higher volatility
- **Note**: Includes extreme AI boom performers (NVDA 865% over 3 years!)

## 🔧 Configuration

Edit `config.py` to modify:

- **Portfolio Templates**: Add new templates to `PORTFOLIO_TEMPLATES` dictionary
- **Estimation windows**: SIGMA_WINDOW=60, RHO_WINDOW=30 (Breaking the Market)
- **Cash exclusion**: EXCLUDE_CASH_FROM_OPTIMIZATION=True
- **Risk-free rate**: For cash returns

## 🧮 Methodology

### Why Minimum Variance Works

**The Problem with Expected Returns (μ):**
- Expected returns are extremely noisy and unstable
- Small changes in μ cause massive portfolio shifts  
- Optimizers chase the highest historical performer → 100% allocations

**The Minimum Variance Solution:**
- **Eliminates μ entirely** - only uses covariance matrix (Σ)
- **Covariance is more stable** than expected returns over time
- **Natural diversification** - only way to reduce risk is correlation benefits
- **No corner solutions** - single assets can never be optimal

### Parameter Estimation

Following "Breaking the Market" philosophy:
- **σ (volatility)**: 60 days (medium adaptation)
- **ρ (correlations)**: 30 days (fast regime changes)
- **No μ needed** - eliminated from optimization

### Mathematical Formulation

**Objective**: Minimize portfolio variance
```
min w^T Σ w
```

**Subject to**:
- Budget constraint: Σw = 1
- Long-only: w ≥ 0
- Exclude Cash: Optimize only risky assets

## 🎯 Leverage Feature: Volatility Targeting

### What is Volatility Targeting?

The leverage system allows you to specify a **target portfolio volatility** (1-25%) and automatically applies leverage to achieve it:

```
Leverage = Target_Volatility ÷ MVP_Volatility (capped at 3x)
```

**Example**: If your minimum variance portfolio has 8% volatility but you want 12%:
- **Leverage applied**: 12% ÷ 8% = 1.5x
- **Result**: All risky assets scaled by 1.5x, cash becomes -50% (borrowed)

### UI Controls

**⚙️ Optimization Parameters** (all in sidebar):
- **Volatility Window**: 10-252 days (σ estimation)
- **Correlation Window**: 5-120 days (ρ estimation)
- **Target Volatility**: 1-25% slider 🎯
- **"No Leverage" Checkbox**: Forces Cash = 0%, disables slider

**⚡ Smart Behavior**:
- Slider automatically **disables** when "No Leverage" is checked
- Real-time leverage calculation and display
- Dynamic feedback showing applied leverage ratio

### Revolutionary Visualizations

**📈 Portfolio Weight Evolution Chart**:
- **Cash moved to bottom** of stack (can go negative!)
- **Red area below 0%** = borrowed amount
- **Dynamic Y-axis** accommodates negative values
- **Reference line at 0%** for clear visual guidance

**📊 Portfolio Weights Chart**:
- **Leverage ratio** displayed in title
- **Negative cash annotation** with arrow
- **"Borrowed: X%" indicator** for leveraged positions

### Example Scenarios

**🔳 No Leverage (checkbox checked)**:
- Cash: exactly 0%
- SPY + TLT + GLD = 100%
- Target volatility ignored

**📈 Moderate Leverage (15% target, 10% MVP)**:
- Leverage: 1.5x
- SPY: 45%, TLT: 52%, GLD: 33%
- Cash: -30% (borrowed to fund positions)
- **Visual**: Red area extends 30% below 0% line

**⚡ High Leverage (20% target, 8% MVP, capped at 3x)**:
- Leverage: 2.5x (reasonable limit applied)
- All risky positions scaled significantly
- Cash: -150% (substantial borrowing)
- **Visual**: Deep red area below 0%, risky assets well above 100%

## 📊 Dashboard Features

### Key Visualizations
- **Portfolio Weights**: Bar chart with leverage ratio and negative cash support
- **Weight Evolution**: Revolutionary chart showing leverage over time (cash can go negative!)
- **Correlation Matrix**: Real-time correlation heatmap (risky assets only)
- **Risk Analysis**: Risk contributions vs portfolio weights
- **Volatility Comparison**: Shows unleveraged MVP volatility vs individual assets

### Diversification Metrics
- **Diversification Ratio**: Weighted avg vol / portfolio vol (>1 = diversification benefit)
- **Effective # Assets**: 1/Σ(w²) - concentration measure (4.0 = equal weights)
- **Max Weight**: Largest position (shows concentration level)
- **Risk Contributions**: How much each asset contributes to total risk

## ✅ Success Criteria

A well-functioning minimum variance portfolio should show:
- **Multiple assets** with meaningful weights (>5%)
- **Diversification ratio > 1.0** (indicates correlation benefits)
- **Effective assets > 1.5** (not concentrated in single position)
- **No domination** (no single asset >80% except in extreme cases)

## 🔧 Critical Bug Fixes

### 1. Log vs Simple Returns (SOLVED)
- **Problem**: Log returns caused double-penalty on variance
- **Solution**: Use simple returns (`pct_change()`) in data/loader.py:231
- **Impact**: Eliminated bias toward low-volatility assets

### 2. Cash Domination (SOLVED)  
- **Problem**: Optimizer chose 100% Cash (zero volatility)
- **Solution**: `EXCLUDE_CASH_FROM_OPTIMIZATION = True`
- **Impact**: Focuses on risky asset diversification

## 🧪 Testing

Run tests to validate functionality:

```bash
python test_phase1.py      # Simulated data validation
python test_streamlit.py   # Market data integration
```

**Test Focus**: Ensure diversified portfolios, no 100% single-asset allocations

## 📈 Key Insights

**Typical Results**: 
- SPY: ~18%, TLT: ~55%, GLD: ~27% (varies with market conditions)
- Diversification Ratio: ~1.4 (40% volatility reduction from diversification)
- Effective Assets: ~2.8 (well-diversified across assets)

**Why This Works**:
- TLT-SPY negative correlation provides major diversification benefit
- GLD adds further diversification and inflation protection
- Portfolio volatility significantly lower than individual assets
- Focus on "smoothest ride" rather than "highest returns"

## 🚨 When to Be Concerned

The optimizer should **NOT** show:
- 100% allocation to any single asset (indicates correlation issues)
- Diversification ratio close to 1.0 (no diversification benefit)
- Effective assets close to 1.0 (concentrated portfolio)

These indicate data or methodology issues that need investigation.

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Development

See `CLAUDE.md` for detailed development instructions, including:
- Adding new assets or metrics
- Modifying risk models
- Understanding the minimum variance approach
- Testing and validation procedures