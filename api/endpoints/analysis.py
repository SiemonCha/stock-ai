"""
Analysis endpoints
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))

from fastapi import APIRouter, HTTPException
import numpy as np

from api.models import StockAnalysisResponse, ModelPerformance

try:
    from models.unified_models import UnifiedStockModels, DataProcessor
    from data.stock_data import StockDataCollector
except ImportError as e:
    logging.warning(f"Import warning: {e}")

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/{symbol}", response_model=StockAnalysisResponse)
async def analyze_stock(
    symbol: str, 
    model_type: str = "ensemble", 
    detailed: bool = False
):
    """
    Comprehensive stock analysis
    
    Provides detailed analysis including current price, predictions, 
    technical indicators, and model performance metrics.
    """
    try:
        symbol = symbol.upper()
        
        # Get data collector
        collector = StockDataCollector()
        
        # Get recent data
        df = collector.get_stock_data(symbol, period="1y")
        if df is None or len(df) < 100:
            raise HTTPException(
                status_code=400, 
                detail=f"Insufficient data for symbol {symbol}"
            )
        
        current_price = float(df['Close'].iloc[-1])
        
        # Load models
        models = UnifiedStockModels()
        model_path = f"models/saved/{symbol}"
        
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=404,
                detail=f"No trained models found for {symbol}. Please train a model first."
            )
        
        models.load_models(model_path)
        
        # Prepare data for prediction
        processor = DataProcessor()
        X, y = processor.create_sequences(df.tail(100))
        
        # Make prediction for next day
        prediction = models.predict(model_type, X[-1:])
        predicted_price = float(processor.inverse_transform_target(prediction)[0])
        
        # Calculate price changes
        price_change = predicted_price - current_price
        price_change_percent = (price_change / current_price) * 100
        
        # Determine trend
        recent_prices = df['Close'].tail(5).values
        if len(recent_prices) > 1:
            price_trend = np.mean(np.diff(recent_prices))
            if price_trend > 0.01:
                trend = "upward"
            elif price_trend < -0.01:
                trend = "downward"
            else:
                trend = "sideways"
        else:
            trend = "unknown"
        
        # Calculate volatility
        returns = df['Close'].pct_change().dropna()
        volatility_score = np.std(returns) * np.sqrt(252)  # Annualized volatility
        
        if volatility_score > 0.3:
            volatility = "high"
        elif volatility_score > 0.15:
            volatility = "medium"
        else:
            volatility = "low"
        
        # Evaluate model performance
        test_X = X[-20:] if len(X) > 20 else X
        test_y = y[-20:] if len(y) > 20 else y
        
        metrics_dict = models.evaluate_model(model_type, test_X, test_y)
        performance = ModelPerformance(
            rmse=metrics_dict['rmse'],
            mae=metrics_dict['mae'],
            directional_accuracy=metrics_dict['directional_accuracy'],
            mse=metrics_dict['mse']
        )
        
        # Technical indicators
        technical_indicators = calculate_technical_indicators(df) if detailed else {}
        
        return StockAnalysisResponse(
            symbol=symbol,
            analysis_date=datetime.now(),
            current_price=current_price,
            predicted_price=predicted_price,
            price_change=price_change,
            price_change_percent=price_change_percent,
            trend=trend,
            volatility=volatility,
            performance=performance,
            technical_indicators=technical_indicators
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def calculate_technical_indicators(df) -> Dict[str, float]:
    """Calculate technical indicators"""
    try:
        indicators = {}
        
        # Simple Moving Averages
        indicators['sma_20'] = float(df['Close'].rolling(window=20).mean().iloc[-1])
        indicators['sma_50'] = float(df['Close'].rolling(window=50).mean().iloc[-1])
        
        # Price ratios
        indicators['price_to_sma_20'] = float(df['Close'].iloc[-1] / indicators['sma_20'])
        indicators['price_to_sma_50'] = float(df['Close'].iloc[-1] / indicators['sma_50'])
        
        # Volatility
        returns = df['Close'].pct_change().dropna()
        indicators['volatility_20d'] = float(returns.rolling(window=20).std().iloc[-1])
        
        # Volume indicators
        indicators['avg_volume_20d'] = float(df['Volume'].rolling(window=20).mean().iloc[-1])
        indicators['volume_ratio'] = float(df['Volume'].iloc[-1] / indicators['avg_volume_20d'])
        
        # Price range
        indicators['high_low_ratio'] = float(df['High'].iloc[-1] / df['Low'].iloc[-1])
        
        return indicators
        
    except Exception as e:
        logger.warning(f"Failed to calculate technical indicators: {str(e)}")
        return {}

@router.get("/{symbol}/performance")
async def get_model_performance(symbol: str, model_type: str = "ensemble"):
    """
    Get detailed model performance metrics
    
    Returns comprehensive performance analysis for the specified model.
    """
    try:
        symbol = symbol.upper()
        
        # Load models
        models = UnifiedStockModels()
        model_path = f"models/saved/{symbol}"
        
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=404,
                detail=f"No trained models found for {symbol}"
            )
        
        models.load_models(model_path)
        
        # Get test data
        collector = StockDataCollector()
        df = collector.get_stock_data(symbol, period="2y")
        
        processor = DataProcessor()
        X, y = processor.create_sequences(df)
        
        # Use last 20% for testing
        test_size = int(len(X) * 0.2)
        X_test = X[-test_size:]
        y_test = y[-test_size:]
        
        # Evaluate model
        metrics = models.evaluate_model(model_type, X_test, y_test)
        
        # Get predictions for analysis
        predictions = models.predict(model_type, X_test)
        
        # Additional metrics
        prediction_errors = y_test - predictions
        
        additional_metrics = {
            "mean_error": float(np.mean(prediction_errors)),
            "std_error": float(np.std(prediction_errors)),
            "max_error": float(np.max(np.abs(prediction_errors))),
            "r2_score": float(1 - (np.sum(prediction_errors**2) / np.sum((y_test - np.mean(y_test))**2))),
            "test_samples": len(X_test)
        }
        
        return {
            "symbol": symbol,
            "model_type": model_type,
            "basic_metrics": metrics,
            "additional_metrics": additional_metrics,
            "performance_grade": get_performance_grade(metrics['directional_accuracy']),
            "evaluated_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Performance analysis failed for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Performance analysis failed: {str(e)}")

def get_performance_grade(directional_accuracy: float) -> str:
    """Get performance grade based on directional accuracy"""
    if directional_accuracy >= 0.9:
        return "A+"
    elif directional_accuracy >= 0.85:
        return "A"
    elif directional_accuracy >= 0.8:
        return "B+"
    elif directional_accuracy >= 0.75:
        return "B"
    elif directional_accuracy >= 0.7:
        return "C+"
    elif directional_accuracy >= 0.65:
        return "C"
    elif directional_accuracy >= 0.6:
        return "D"
    else:
        return "F"

@router.get("/{symbol}/comparison")
async def compare_models(symbol: str):
    """
    Compare performance of different models for a symbol
    
    Returns performance comparison across all available model types.
    """
    try:
        symbol = symbol.upper()
        
        # Load models
        models = UnifiedStockModels()
        model_path = f"models/saved/{symbol}"
        
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=404,
                detail=f"No trained models found for {symbol}"
            )
        
        models.load_models(model_path)
        
        # Get test data
        collector = StockDataCollector()
        df = collector.get_stock_data(symbol, period="1y")
        
        processor = DataProcessor()
        X, y = processor.create_sequences(df)
        
        # Use last portion for testing
        test_size = min(100, int(len(X) * 0.2))
        X_test = X[-test_size:]
        y_test = y[-test_size:]
        
        # Compare available models
        model_comparison = {}
        available_models = list(models.models.keys())
        
        for model_type in available_models:
            try:
                metrics = models.evaluate_model(model_type, X_test, y_test)
                model_comparison[model_type] = {
                    "metrics": metrics,
                    "grade": get_performance_grade(metrics['directional_accuracy']),
                    "available": True
                }
            except Exception as e:
                model_comparison[model_type] = {
                    "error": str(e),
                    "available": False
                }
        
        # Find best model
        best_model = None
        best_accuracy = 0
        
        for model_type, data in model_comparison.items():
            if data.get('available') and data['metrics']['directional_accuracy'] > best_accuracy:
                best_accuracy = data['metrics']['directional_accuracy']
                best_model = model_type
        
        return {
            "symbol": symbol,
            "model_comparison": model_comparison,
            "best_model": best_model,
            "best_accuracy": best_accuracy,
            "total_models": len(available_models),
            "evaluated_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model comparison failed for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model comparison failed: {str(e)}")