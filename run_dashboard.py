#!/usr/bin/env python3
"""
Launch script for the Portfolio Optimizer dashboard.

This script provides a convenient way to start the Streamlit application
with proper configuration and error handling.
"""

import subprocess
import sys
import os
from pathlib import Path


def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'streamlit', 'numpy', 'pandas', 'plotly', 
        'yfinance', 'scipy', 'cvxpy', 'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\nPlease install missing packages:")
        print("   pip install -r requirements.txt")
        return False
    
    return True


def run_validation_tests():
    """Run validation tests before launching dashboard."""
    print("🧪 Running validation tests...")
    
    # Run Phase 1 tests
    try:
        result = subprocess.run([sys.executable, 'test_phase1.py'], 
                               capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print("⚠️ Phase 1 tests failed, but dashboard will still launch")
            print("   Some functionality may be limited")
        else:
            print("✅ Phase 1 tests passed")
    
    except subprocess.TimeoutExpired:
        print("⚠️ Phase 1 tests timed out")
    except Exception as e:
        print(f"⚠️ Could not run Phase 1 tests: {e}")
    
    # Run Phase 2 tests
    try:
        result = subprocess.run([sys.executable, 'test_streamlit.py'], 
                               capture_output=True, text=True, timeout=15)
        
        if result.returncode != 0:
            print("⚠️ Phase 2 tests indicate market data issues")
            print("   Dashboard will launch but market data may be unavailable")
        else:
            print("✅ Phase 2 tests passed - market data available")
    
    except subprocess.TimeoutExpired:
        print("⚠️ Phase 2 tests timed out")
    except Exception as e:
        print(f"⚠️ Could not run Phase 2 tests: {e}")


def launch_dashboard():
    """Launch the Streamlit dashboard."""
    print("\n🚀 Launching Portfolio Optimizer Dashboard...")
    print("📊 Dashboard will open in your default browser")
    print("🔗 URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    print("\n" + "="*50)
    
    try:
        # Change to the script directory
        script_dir = Path(__file__).parent
        os.chdir(script_dir)
        
        # Launch Streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'ui/app.py',
            '--server.headless', 'false',
            '--server.port', '8501',
            '--server.address', 'localhost',
            '--browser.gatherUsageStats', 'false'
        ])
        
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching dashboard: {e}")
        print("Try running manually: streamlit run ui/app.py")


def main():
    """Main launch sequence."""
    print("📈 Kelly Criterion Portfolio Optimizer")
    print("="*40)
    
    # Check if we're in the right directory
    if not Path('ui/app.py').exists():
        print("❌ Please run this script from the Port_Optimizer directory")
        print("   Current directory should contain ui/app.py")
        sys.exit(1)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    print("✅ All dependencies available")
    
    # Optional: Run validation tests
    run_tests = input("\n🧪 Run validation tests first? (y/N): ").lower().strip()
    if run_tests in ['y', 'yes']:
        run_validation_tests()
    
    # Launch dashboard
    launch_dashboard()


if __name__ == "__main__":
    main()