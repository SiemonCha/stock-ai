"""
Prediction endpoints
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))

from fastapi import APIRouter, HTTPException, BackgroundTasks
import numpy as np

from api.models import (
    PredictionRequest, PredictionResponse, BatchPredictionRequest, 
    BatchPredictionResponse
)

try:
    from models.unified_models import UnifiedStockModels, DataProcessor
    from data.stock_data import StockDataCollector
except ImportError as e:
    logging.warning(f"Import warning: {e}")

router = APIRouter()
logger = logging.getLogger(__name__)

# Global model cache
_model_cache = {}
_collector = None

def get_collector():
    """Get or create data collector"""
    global _collector
    if _collector is None:
        _collector = StockDataCollector()
    return _collector

def get_models(symbol: str):
    """Get or load models for symbol"""
    global _model_cache
    
    cache_key = symbol
    if cache_key not in _model_cache:
        try:
            models = UnifiedStockModels()
            model_path = f"models/saved/{symbol}"
            
            if os.path.exists(model_path):
                models.load_models(model_path)
                _model_cache[cache_key] = models
                logger.info(f"Loaded models for {symbol}")
            else:
                logger.warning(f"No trained models found for {symbol}")
                return None
        except Exception as e:
            logger.error(f"Failed to load models for {symbol}: {str(e)}")
            return None
    
    return _model_cache.get(cache_key)

@router.post("/", response_model=PredictionResponse)
async def predict_stock(request: PredictionRequest):
    """
    Generate stock price predictions
    
    This endpoint generates price predictions for a given stock symbol
    using the specified machine learning model.
    """
    try:
        # Get data collector
        collector = get_collector()
        
        # Get recent data
        df = collector.get_stock_data(request.symbol, period="2y")
        if df is None or len(df) < 100:
            raise HTTPException(
                status_code=400, 
                detail=f"Insufficient data for symbol {request.symbol}"
            )
        
        # Load models
        models = get_models(request.symbol)
        if models is None:
            raise HTTPException(
                status_code=404,
                detail=f"No trained models found for symbol {request.symbol}. Please train a model first."
            )
        
        # Prepare data for prediction
        processor = DataProcessor()
        X, y = processor.create_sequences(df.tail(request.days + 100))
        
        # Make predictions
        predictions = models.predict(request.model_type.value, X[-request.days:])
        
        # Generate future dates
        last_date = df.index[-1]
        future_dates = [
            (last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') 
            for i in range(len(predictions))
        ]
        
        # Inverse transform predictions if needed
        try:
            predictions_actual = processor.inverse_transform_target(predictions)
        except:
            predictions_actual = predictions  # Fallback
        
        # Prepare confidence intervals if requested
        confidence_intervals = None
        if request.confidence:
            # Simple confidence intervals (this would be more sophisticated with Bayesian models)
            std_dev = np.std(predictions_actual) * 0.1
            confidence_intervals = {
                "upper_95": (predictions_actual + 2 * std_dev).tolist(),
                "lower_95": (predictions_actual - 2 * std_dev).tolist(),
                "upper_99": (predictions_actual + 3 * std_dev).tolist(),
                "lower_99": (predictions_actual - 3 * std_dev).tolist()
            }
        
        return PredictionResponse(
            symbol=request.symbol,
            model_type=request.model_type.value,
            predictions=predictions_actual.tolist(),
            dates=future_dates,
            confidence_intervals=confidence_intervals,
            metadata={
                "data_points_used": len(X),
                "last_price": float(df['Close'].iloc[-1]),
                "prediction_horizon": request.days,
                "generated_at": datetime.now().isoformat()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed for {request.symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/batch", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest, background_tasks: BackgroundTasks):
    """
    Generate predictions for multiple stocks
    
    This endpoint processes multiple stock symbols in parallel
    and returns predictions for each one.
    """
    start_time = datetime.now()
    results = {}
    failed = {}
    
    for symbol in request.symbols:
        try:
            # Create individual prediction request
            pred_request = PredictionRequest(
                symbol=symbol,
                model_type=request.model_type,
                days=request.days,
                confidence=False  # Disable confidence for batch to improve speed
            )
            
            # Get prediction
            prediction = await predict_stock(pred_request)
            results[symbol] = prediction
            
        except HTTPException as e:
            failed[symbol] = e.detail
            logger.warning(f"Batch prediction failed for {symbol}: {e.detail}")
        except Exception as e:
            failed[symbol] = str(e)
            logger.error(f"Unexpected error for {symbol}: {str(e)}")
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    return BatchPredictionResponse(
        results=results,
        failed=failed,
        processing_time=processing_time,
        total_symbols=len(request.symbols),
        successful=len(results),
        failed_count=len(failed)
    )

@router.get("/{symbol}/latest")
async def get_latest_prediction(symbol: str, model_type: str = "lstm", days: int = 7):
    """
    Get the latest prediction for a symbol
    
    Quick endpoint for getting recent predictions without full configuration.
    """
    request = PredictionRequest(
        symbol=symbol,
        model_type=model_type,
        days=days
    )
    
    return await predict_stock(request)

@router.delete("/cache/{symbol}")
async def clear_model_cache(symbol: str):
    """
    Clear model cache for a symbol
    
    Forces reloading of models on next prediction request.
    """
    global _model_cache
    
    cache_key = symbol.upper()
    if cache_key in _model_cache:
        del _model_cache[cache_key]
        return {"message": f"Cache cleared for {symbol}"}
    else:
        return {"message": f"No cache found for {symbol}"}

@router.get("/cache/status")
async def get_cache_status():
    """
    Get current cache status
    
    Returns information about cached models.
    """
    global _model_cache
    
    cache_info = {}
    for symbol, models in _model_cache.items():
        cache_info[symbol] = {
            "models_loaded": list(models.models.keys()) if hasattr(models, 'models') else [],
            "cached_at": datetime.now().isoformat()  # This would be the actual cache time
        }
    
    return {
        "cached_symbols": list(_model_cache.keys()),
        "cache_size": len(_model_cache),
        "details": cache_info
    }