"""
FastAPI main application for Stock AI API
"""

import os
import sys
import time
import logging
import traceback
from datetime import datetime
from typing import Dict, List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from api.models import (
    PredictionRequest, PredictionResponse, TrainingRequest, TrainingResponse,
    StockAnalysisResponse, HealthResponse, ErrorResponse, BatchPredictionRequest,
    BatchPredictionResponse, ModelListResponse, DataStatusResponse
)
from api.endpoints.predictions import router as predictions_router
from api.endpoints.training import router as training_router
from api.endpoints.analysis import router as analysis_router
from api.middleware.auth import AuthMiddleware
from api.middleware.rate_limiting import RateLimitMiddleware

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Stock AI API",
    description="Professional-grade AI stock prediction and analysis API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Custom middleware
app.add_middleware(AuthMiddleware)
app.add_middleware(RateLimitMiddleware)

# Include routers
app.include_router(predictions_router, prefix="/api/v1/predictions", tags=["Predictions"])
app.include_router(training_router, prefix="/api/v1/training", tags=["Training"])
app.include_router(analysis_router, prefix="/api/v1/analysis", tags=["Analysis"])

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "message": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check system components
        system_info = {
            "cpu_count": os.cpu_count(),
            "python_version": sys.version,
            "platform": sys.platform
        }
        
        # Check loaded models (placeholder)
        models_loaded = ["lstm", "gru", "ensemble"]  # This would be dynamic
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            version="1.0.0",
            models_loaded=models_loaded,
            system_info=system_info
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Stock AI API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "documentation": "/docs",
        "health": "/health"
    }

# Models endpoint
@app.get("/api/v1/models", response_model=ModelListResponse)
async def list_models():
    """List available and trained models"""
    try:
        available_models = ["lstm", "gru", "transformer", "cnn_lstm", "ensemble"]
        
        # This would dynamically check for trained models
        trained_models = {
            "lstm": ["AAPL", "GOOGL"],
            "ensemble": ["AAPL", "MSFT"]
        }
        
        return ModelListResponse(
            available_models=available_models,
            trained_models=trained_models,
            default_model="lstm"
        )
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model information")

# Data status endpoint
@app.get("/api/v1/data/{symbol}/status", response_model=DataStatusResponse)
async def get_data_status(symbol: str):
    """Get data status for a symbol"""
    try:
        symbol = symbol.upper()
        
        # This would check actual data status
        return DataStatusResponse(
            symbol=symbol,
            last_updated=datetime.now(),
            data_points=1000,
            date_range={"start_date": "2020-01-01", "end_date": "2024-01-01"},
            data_quality="good",
            missing_days=0
        )
    except Exception as e:
        logger.error(f"Failed to get data status for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve data status for {symbol}")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    logger.info("Starting Stock AI API...")
    
    # Initialize system components
    try:
        # Load models, setup database connections, etc.
        logger.info("System initialization completed")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks"""
    logger.info("Shutting down Stock AI API...")
    
    # Cleanup resources
    try:
        # Save state, close connections, etc.
        logger.info("Shutdown completed")
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")

# Main execution
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )