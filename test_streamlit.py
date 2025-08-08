"""
Test script to validate Streamlit app can be imported and basic functionality works.
"""

import sys
sys.path.append('.')

def test_streamlit_import():
    """Test that the Streamlit app can be imported without errors."""
    try:
        from ui.app import main
        print("✓ Streamlit app imported successfully")
        return True
    except Exception as e:
        print(f"❌ Streamlit app import failed: {str(e)}")
        return False

def test_data_loader_yahoo():
    """Test Yahoo Finance data loading functionality."""
    try:
        from data.loader import DataLoader
        
        loader = DataLoader()
        
        # Test latest prices fetch
        prices = loader.get_latest_prices(['SPY', 'TLT', 'GLD'])
        
        if prices:
            print("✓ Yahoo Finance connection working")
            print(f"  Latest prices: {prices}")
            return True
        else:
            print("⚠️ Yahoo Finance returned empty results")
            return False
            
    except Exception as e:
        print(f"❌ Yahoo Finance test failed: {str(e)}")
        return False

def test_market_data_fetch():
    """Test full market data fetching."""
    try:
        from data.loader import DataLoader
        
        loader = DataLoader()
        
        # Test market data fetch for a short period
        data = loader.fetch_market_data('2024-01-01', '2024-01-31')
        
        if len(data) > 0:
            print(f"✓ Market data fetch successful - {len(data)} days")
            print(f"  Assets: {list(data.columns)}")
            print(f"  Date range: {data.index[0].date()} to {data.index[-1].date()}")
            return True
        else:
            print("❌ Market data fetch returned empty results")
            return False
            
    except Exception as e:
        print(f"❌ Market data fetch failed: {str(e)}")
        return False

def main():
    """Run Phase 2 validation tests."""
    print("=" * 50)
    print("PHASE 2 TESTING: MARKET DATA INTEGRATION")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Streamlit import
    if test_streamlit_import():
        tests_passed += 1
    
    # Test 2: Yahoo Finance connection
    if test_data_loader_yahoo():
        tests_passed += 1
    
    # Test 3: Market data fetch
    if test_market_data_fetch():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    if tests_passed == total_tests:
        print("ALL PHASE 2 TESTS PASSED! 🎉")
        print("Ready to launch Streamlit dashboard!")
    else:
        print(f"TESTS PASSED: {tests_passed}/{total_tests}")
        if tests_passed < total_tests:
            print("⚠️ Some functionality may be limited")
    print("=" * 50)
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)