#!/usr/bin/env python3
"""
Stock AI API Server Launch Script
Simple script to launch the production API server
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

try:
    from services.api import run_api
    
    if __name__ == "__main__":
        print("🚀 Launching Stock AI Production API...")
        print("📚 Documentation will be available at: http://localhost:8000/docs")
        print("🔧 Health check: http://localhost:8000/health")
        print("📊 Status: http://localhost:8000/status")
        print()
        print("💡 Example API calls:")
        print("  POST /predict - Stock predictions")
        print("  POST /portfolio/optimize - Portfolio optimization")
        print("  POST /regime/analyze - Market regime analysis")
        print("  GET /models/list - List available models")
        print()
        print("🔑 Don't forget to set your API_KEY environment variable!")
        print("   export API_KEY='your-secret-key-here'")
        print()
        
        run_api()

except ImportError as e:
    print(f"❌ Failed to import API components: {e}")
    print("📦 Please install required dependencies:")
    print("   pip install fastapi uvicorn redis cachetools psutil")
    sys.exit(1)
except Exception as e:
    print(f"❌ Failed to start API: {e}")
    sys.exit(1)