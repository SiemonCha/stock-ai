"""
Pydantic models for API requests and responses
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class ModelType(str, Enum):
    """Available model types"""
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    CNN_LSTM = "cnn_lstm"
    ENSEMBLE = "ensemble"

class PredictionRequest(BaseModel):
    """Request model for stock predictions"""
    symbol: str = Field(..., description="Stock symbol (e.g., 'AAPL')")
    model_type: ModelType = Field(default=ModelType.LSTM, description="Model type to use")
    days: int = Field(default=30, ge=1, le=365, description="Number of days to predict")
    confidence: bool = Field(default=False, description="Include confidence intervals")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Symbol cannot be empty')
        return v.strip().upper()

class TrainingRequest(BaseModel):
    """Request model for model training"""
    symbol: str = Field(..., description="Stock symbol to train on")
    model_type: ModelType = Field(default=ModelType.LSTM, description="Model type to train")
    epochs: int = Field(default=100, ge=1, le=1000, description="Number of training epochs")
    years: int = Field(default=5, ge=1, le=20, description="Years of historical data")
    sequence_length: int = Field(default=60, ge=10, le=200, description="Sequence length")
    quick_mode: bool = Field(default=False, description="Quick training mode")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Symbol cannot be empty')
        return v.strip().upper()

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    symbol: str
    model_type: str
    predictions: List[float]
    dates: List[str]
    confidence_intervals: Optional[Dict[str, List[float]]] = None
    metadata: Dict[str, Any]

class TrainingResponse(BaseModel):
    """Response model for training"""
    symbol: str
    model_type: str
    training_completed: bool
    metrics: Dict[str, float]
    training_time: float
    model_id: str

class ModelPerformance(BaseModel):
    """Model performance metrics"""
    rmse: float = Field(..., description="Root Mean Square Error")
    mae: float = Field(..., description="Mean Absolute Error")
    directional_accuracy: float = Field(..., ge=0, le=1, description="Directional accuracy (0-1)")
    mse: float = Field(..., description="Mean Square Error")

class StockAnalysisResponse(BaseModel):
    """Response model for stock analysis"""
    symbol: str
    analysis_date: datetime
    current_price: float
    predicted_price: float
    price_change: float
    price_change_percent: float
    trend: str  # "upward", "downward", "sideways"
    volatility: str  # "low", "medium", "high"
    performance: ModelPerformance
    technical_indicators: Dict[str, float]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    version: str
    models_loaded: List[str]
    system_info: Dict[str, Any]

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    message: str
    timestamp: datetime
    request_id: Optional[str] = None

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    symbols: List[str] = Field(..., min_items=1, max_items=50, description="List of stock symbols")
    model_type: ModelType = Field(default=ModelType.ENSEMBLE, description="Model type to use")
    days: int = Field(default=30, ge=1, le=365, description="Number of days to predict")
    
    @validator('symbols')
    def validate_symbols(cls, v):
        if not v:
            raise ValueError('Symbols list cannot be empty')
        return [symbol.strip().upper() for symbol in v if symbol.strip()]

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    results: Dict[str, PredictionResponse]
    failed: Dict[str, str]  # symbol -> error message
    processing_time: float
    total_symbols: int
    successful: int
    failed_count: int

class ModelListResponse(BaseModel):
    """Response model for available models"""
    available_models: List[str]
    trained_models: Dict[str, List[str]]  # model_type -> [symbols]
    default_model: str

class DataStatusResponse(BaseModel):
    """Response model for data status"""
    symbol: str
    last_updated: datetime
    data_points: int
    date_range: Dict[str, str]  # start_date, end_date
    data_quality: str  # "good", "fair", "poor"
    missing_days: int