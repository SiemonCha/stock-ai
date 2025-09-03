#!/usr/bin/env python3

import os
import sys
import subprocess

def print_banner():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                Neural Network Stock Predictor                ║
║               LSTM-based Stock Price Prediction              ║
╚═══════════════════════════════════════════════════════════════╝
    """)

def check_python_version():
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {sys.version}")
    return True

def install_dependencies():
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def setup_environment():
    print("\n🔧 Setting up environment...")
    
    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        import shutil
        shutil.copy('.env.example', '.env')
        print("✅ Created .env file from template")
    
    # Create necessary directories
    directories = ['models', 'plots', 'data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✅ Created project directories")

def show_next_steps():
    print("""
🚀 Setup Complete! Here's what you can do next:

1. CONFIGURE API KEYS (Optional but recommended):
   - Edit .env file and add your Alpha Vantage API key
   - Get free key at: https://www.alphavantage.co/support/#api-key

2. TRAIN YOUR FIRST MODEL:
   python train.py --symbol AAPL --epochs 50

3. START THE API SERVER:
   python start_api.py
   
   Then visit: http://localhost:8000/docs for API documentation

4. EXAMPLE TRAINING COMMANDS:
   python train.py --symbol AAPL    # Train Apple stock predictor
   python train.py --symbol GOOGL   # Train Google stock predictor
   python train.py --symbol MSFT    # Train Microsoft stock predictor

5. DOCKER DEPLOYMENT:
   docker-compose up -d

📚 Key Features:
   - ✅ LSTM Neural Network with 60-day sequences
   - ✅ 22 technical indicators (RSI, MACD, Bollinger Bands, etc.)
   - ✅ FastAPI REST API for predictions
   - ✅ Backtesting framework with portfolio optimization
   - ✅ Real-time predictions via yfinance (no API key needed)
   - ✅ Docker deployment ready

💡 Tips:
   - Start with popular stocks (AAPL, GOOGL, MSFT, TSLA)
   - More training epochs = better accuracy (but longer training time)
   - Use the API to get real-time predictions after training models

Happy Trading! 📈
    """)

def main():
    print_banner()
    
    if not check_python_version():
        return 1
    
    print("🔍 Checking current setup...")
    
    # Check if requirements.txt exists
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found")
        return 1
    
    # Install dependencies
    if not install_dependencies():
        return 1
    
    # Setup environment
    setup_environment()
    
    show_next_steps()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())