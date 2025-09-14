#!/usr/bin/env python3
"""
Stock AI Dashboard Launcher

Quick launcher script for the interactive web dashboard.
Run this to start the dashboard server.
"""

import sys
import os
import logging
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.frontend.dashboard import StockAIDashboard
    
    def main():
        """Main launcher function"""
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        print("🚀 Starting Stock AI Interactive Dashboard...")
        print("📊 Features: Real-time charts, AI predictions, portfolio monitoring")
        print("🌐 Dashboard URL: http://localhost:8050")
        print("📱 Mobile responsive design")
        print("⚡ Auto-refresh every 30 seconds")
        print("-" * 60)
        
        # Initialize and run dashboard
        dashboard = StockAIDashboard()
        
        try:
            dashboard.run_server(
                debug=False,  # Set to False for production-like behavior
                port=8050,
                dev_tools_hot_reload=True,
                dev_tools_ui=True
            )
        except KeyboardInterrupt:
            print("\n👋 Dashboard stopped by user")
        except Exception as e:
            print(f"❌ Error starting dashboard: {e}")
            sys.exit(1)
    
    if __name__ == "__main__":
        main()

except ImportError as e:
    print("❌ Missing dependencies. Please install frontend requirements:")
    print("pip install -r requirements_frontend.txt")
    print(f"Error: {e}")
    sys.exit(1)