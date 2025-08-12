#!/usr/bin/env python3
"""
Minimum Variance Portfolio Optimizer

Run the Streamlit application for minimum variance portfolio optimization.
"""

import subprocess
import sys
import os


def main():
    """Launch the Portfolio Optimizer Streamlit app."""
    print("🚀 Starting Minimum Variance Portfolio Optimizer...")
    print("   Pure risk reduction through diversification")
    print("   Access the app at: http://localhost:8503")
    print()
    
    try:
        # Run the Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "app.py", 
            "--server.port", "8503",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ], check=True)
        
    except KeyboardInterrupt:
        print("\n👋 Portfolio Optimizer stopped by user")
    except FileNotFoundError:
        print("❌ Error: Streamlit not found. Please install with:")
        print("   pip install streamlit")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()