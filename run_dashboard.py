#!/usr/bin/env python3
"""
Dashboard Launcher for Stock AI Project

Simple script to run my stock dashboard.
This is for my CS final year project!
"""

import sys
import os
from pathlib import Path

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent / "src"))

def main():
    print("=" * 50)
    print("🚀 Stock AI Dashboard - Final Year Project")
    print("=" * 50)
    print("📊 Starting dashboard server...")
    print("🌐 Will open at: http://localhost:8050")
    print("💡 Press Ctrl+C to stop")
    print()
    
    try:
        # Try the simple dashboard first (more reliable)
        from src.frontend.simple_dashboard import StockDashboard
        dashboard = StockDashboard()
        dashboard.run(debug=True, port=8050)
        
    except ImportError:
        print("❌ Could not import dashboard modules")
        print("💡 Make sure you've installed the requirements:")
        print("   pip install dash plotly pandas yfinance")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped. Thanks for checking out my project!")
    except Exception as e:
        print(f"❌ Something went wrong: {e}")
        print("💡 This is still a work in progress!")

if __name__ == "__main__":
    main()