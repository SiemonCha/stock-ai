"""
Configuration module for Stock AI system
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class Config:
    """Main configuration class"""
    
    # Data settings
    DEFAULT_SEQUENCE_LENGTH: int = 60
    DEFAULT_PERIOD: str = "5y"
    DEFAULT_INTERVAL: str = "1d"
    
    # Training settings
    DEFAULT_EPOCHS: int = 100
    DEFAULT_BATCH_SIZE: int = 32
    DEFAULT_TEST_SIZE: float = 0.2
    DEFAULT_VAL_SIZE: float = 0.1
    
    # Model settings
    DEFAULT_LEARNING_RATE: float = 0.001
    DEFAULT_DROPOUT_RATE: float = 0.2
    
    # File paths
    MODELS_DIR: str = "models/saved"
    DATA_DIR: str = "data/cache"
    PLOTS_DIR: str = "plots"
    
    # API settings (optional)
    ALPHA_VANTAGE_KEY: str = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    QUANDL_KEY: str = os.getenv('QUANDL_API_KEY', '')
    
    # Advanced settings
    ENABLE_GPU: bool = True
    ENABLE_MIXED_PRECISION: bool = False
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        directories = [cls.MODELS_DIR, cls.DATA_DIR, cls.PLOTS_DIR]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

# Global config instance
config = Config()