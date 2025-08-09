# Correlation Matrix Display Issue - Debug Report

## 🐛 **PROBLEM DESCRIPTION**

**Issue:** Correlation matrix heatmap is displaying correlations as integers (`1`, `0`) instead of proper decimals (`1.000`, `0.142`).

**Current Behavior:**
- Diagonal elements show as `1` instead of `1.000`
- Off-diagonal correlations show as `0` instead of actual values like `0.142`
- This suggests a **display formatting issue**, not a calculation problem

**Confirmed Working:**
- Backend correlation calculations are correct (verified in testing)
- Real market data shows realistic correlations in code tests
- 2 years of data (Aug 2023 - Aug 2024) with 35-day correlation halflife should show meaningful correlations

## 🔧 **ATTEMPTED FIX**

**What was changed:**
```python
# Changed in ui/app.py line 390
# FROM:
texttemplate="%{text}",

# TO:
texttemplate="%{text:.3f}",  # Force 3 decimal places
```

**Expected result:** All correlation values should display with 3 decimal places.

## 🔍 **DEBUGGING STEPS TO TRY**

### **Step 1: Verify Plotly Template Fix**
1. Restart Streamlit completely
2. Load market data and run optimization
3. Check if correlation heatmap now shows `1.000` and `0.142` instead of `1` and `0`

### **Step 2: Check Data Flow**
Add debug logging to see what values are actually being passed to the heatmap:

```python
def create_correlation_heatmap(cov_matrix, volatilities):
    # Extract correlation matrix from covariance
    corr_matrix = cov_matrix / np.outer(volatilities, volatilities)
    
    # DEBUG: Print actual correlation values
    print("DEBUG - Correlation matrix values:")
    print(corr_matrix)
    print(f"DEBUG - SPY-TLT correlation: {corr_matrix[0,1]}")
    print(f"DEBUG - SPY-GLD correlation: {corr_matrix[0,2]}")
    
    # ... rest of function
```

### **Step 3: Alternative Formatting Approaches**
If the `texttemplate="%{text:.3f}"` doesn't work, try:

**Option A: Pre-format the text array**
```python
# Replace this:
text=np.round(corr_matrix, 3),
texttemplate="%{text:.3f}",

# With this:
text=[[f"{val:.3f}" for val in row] for row in corr_matrix],
texttemplate="%{text}",
```

**Option B: Use customdata instead**
```python
fig = go.Figure(data=go.Heatmap(
    z=corr_matrix,
    x=ASSETS,
    y=ASSETS,
    colorscale='RdBu_r',
    zmid=0,
    customdata=corr_matrix,
    texttemplate="%{customdata:.3f}",
    textfont={"size": 12},
    colorbar=dict(title="Correlation")
))
```

**Option C: Force hover formatting**
```python
fig = go.Figure(data=go.Heatmap(
    z=corr_matrix,
    x=ASSETS,
    y=ASSETS,
    colorscale='RdBu_r',
    zmid=0,
    text=corr_matrix,
    texttemplate="%{z:.3f}",  # Use z values directly
    textfont={"size": 12},
    colorbar=dict(title="Correlation")
))
```

### **Step 4: Test with Simple Data**
Create a minimal test to isolate the issue:

```python
import plotly.graph_objects as go
import numpy as np

# Simple test matrix
test_matrix = np.array([
    [1.000, 0.142, -0.087],
    [0.142, 1.000, 0.256],
    [-0.087, 0.256, 1.000]
])

fig = go.Figure(data=go.Heatmap(
    z=test_matrix,
    text=test_matrix,
    texttemplate="%{text:.3f}",
    textfont={"size": 12}
))

fig.show()
```

### **Step 5: Check Plotly Version**
Version compatibility might be an issue:
```python
import plotly
print(f"Plotly version: {plotly.__version__}")

# If version is old, try updating:
# pip install --upgrade plotly
```

## 🎯 **EXPECTED CORRELATION RANGES**

With 2 years of real market data, realistic correlations should be:

- **SPY-TLT**: `-0.3` to `+0.3` (varies by market regime, often negative during risk-off)
- **SPY-GLD**: `-0.4` to `+0.1` (usually negative, gold hedges equities)
- **TLT-GLD**: `-0.2` to `+0.4` (both are risk-off assets, usually positive)
- **All-Cash**: `~0.0` (cash should be uncorrelated with risky assets)

**If seeing all zeros:** This indicates display formatting issue, not calculation issue.

## 📋 **VERIFICATION CHECKLIST**

Before proceeding with more complex fixes:

- [ ] Streamlit fully restarted after code changes
- [ ] Using Phase 2 market data (not simulated)
- [ ] Date range is substantial (6+ months)
- [ ] Correlation halflife is reasonable (15-60 days)
- [ ] Check Plotly version compatibility
- [ ] Test with debug prints to see actual correlation values
- [ ] Try alternative texttemplate formats

## 🚨 **NEXT STEPS**

1. **First Priority:** Verify the `texttemplate="%{text:.3f}"` fix worked after restart
2. **If still broken:** Add debug prints to see actual values being passed
3. **If values are correct:** Try alternative Plotly formatting approaches
4. **If values are wrong:** Investigate correlation calculation pipeline

---

*This debug report created: 2025-08-09*  
*Context: Correlation calculations verified correct in backend testing, issue appears to be Plotly display formatting*