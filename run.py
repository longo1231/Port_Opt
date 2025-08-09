#!/usr/bin/env python3
"""
Portfolio Optimizer - Breaking the Market Approach

Run the Streamlit application for Kelly criterion portfolio optimization.
"""

import subprocess
import sys
import os


def main():
    """Launch the Portfolio Optimizer Streamlit app."""
    print("🚀 Starting Portfolio Optimizer...")
    print("   Breaking the Market methodology with Kelly criterion")
    print("   Access the app at: http://localhost:8503")
    print()
    
    try:
        # Run the Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "app_simple.py", 
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