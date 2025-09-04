#!/usr/bin/env python3
"""
Professional Dashboard Launcher
Ultra-simple launcher for the professional stock prediction dashboard
"""

import sys
import os
from pathlib import Path
import subprocess

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'plotly', 
        'streamlit-autorefresh'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages for dashboard:")
        for package in missing_packages:
            print(f"   • {package}")
        print("\n💡 Install with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def launch_dashboard():
    """Launch the professional dashboard"""
    dashboard_path = Path(__file__).parent / 'src' / 'professional' / 'professional_dashboard.py'
    
    if not dashboard_path.exists():
        print("❌ Dashboard file not found!")
        return False
    
    print("🚀 Starting Professional Stock AI Dashboard...")
    print("📊 Target: 99% Accuracy with Advanced Analytics")
    print("🌐 Dashboard will open in your browser automatically")
    print()
    print("⏹️  Press Ctrl+C to stop the dashboard")
    print("=" * 60)
    
    try:
        # Launch Streamlit dashboard
        cmd = [
            sys.executable, 
            '-m', 'streamlit', 
            'run', 
            str(dashboard_path),
            '--server.port=8501',
            '--server.address=localhost',
            '--server.headless=false'
        ]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n⏹️  Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("🎯 PROFESSIONAL STOCK AI DASHBOARD")
    print("=" * 50)
    print("Ultra-Professional Interface for Stock Prediction")
    print("Targeting 99% Accuracy with Advanced Analytics")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        return 1
    
    # Launch dashboard
    if not launch_dashboard():
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())