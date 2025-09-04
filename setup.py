#!/usr/bin/env python3
"""
Setup script for Stock AI system
"""

import os
import subprocess
import sys

def create_directories():
    """Create necessary directories"""
    directories = [
        'models/saved',
        'data/cache',
        'plots',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'numpy',
        'pandas', 
        'yfinance',
        'scikit-learn',
        'matplotlib',
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"[OK] {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"[MISSING] {package} is missing")
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies"""
    try:
        print("Installing dependencies from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        return False

def main():
    """Main setup function"""
    print("Setting up Stock AI system...")
    
    # Create directories
    print("\nCreating directories...")
    create_directories()
    
    # Check dependencies
    print("\nChecking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        response = input("Would you like to install missing dependencies? (y/n): ")
        if response.lower() == 'y':
            if install_dependencies():
                print("Setup completed successfully!")
            else:
                print("Setup failed. Please install dependencies manually.")
                return 1
        else:
            print("Setup completed but some dependencies are missing.")
    else:
        print("All dependencies are satisfied!")
    
    print("\nStock AI setup completed!")
    print("\nNext steps:")
    print("1. Train a model: python train.py --symbol AAPL --model lstm")
    print("2. Analyze results: python analyze.py --symbol AAPL --model lstm")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())