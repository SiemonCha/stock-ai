from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import sys
import asyncio
from typing import List, Dict
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.collector import StockDataCollector
from data.preprocessor import StockDataPreprocessor
from models.lstm_model import StockLSTMModel

app = FastAPI(title="Stock Price Predictor API", version="1.0.0")

# Global variables for model and components
models: Dict[str, StockLSTMModel] = {}
collector = StockDataCollector()

class PredictionRequest(BaseModel):
    symbol: str
    days_ahead: int = 1

class PredictionResponse(BaseModel):
    symbol: str
    current_price: float
    predicted_price: float
    price_change: float
    price_change_percent: float
    confidence_score: float

class HealthResponse(BaseModel):
    status: str
    message: str

@app.on_event("startup")
async def startup_event():
    """
    Load pre-trained models on startup
    """
    print("Loading pre-trained models...")
    
    # Check for available models
    models_dir = "models"
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith("_final_model.h5"):
                symbol = file.replace("_final_model.h5", "")
                try:
                    model = StockLSTMModel()
                    model.load_model(os.path.join(models_dir, file))
                    models[symbol] = model
                    print(f"Loaded model for {symbol}")
                except Exception as e:
                    print(f"Error loading model for {symbol}: {str(e)}")
    
    print(f"Loaded {len(models)} models")

@app.get("/", response_model=HealthResponse)
async def root():
    """
    Health check endpoint
    """
    return HealthResponse(
        status="healthy",
        message="Stock Predictor API is running"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Detailed health check
    """
    return HealthResponse(
        status="healthy",
        message=f"API is running with {len(models)} loaded models"
    )

@app.get("/models")
async def list_models():
    """
    List available models
    """
    return {"available_models": list(models.keys())}

@app.post("/predict", response_model=PredictionResponse)
async def predict_stock_price(request: PredictionRequest):
    """
    Predict stock price for a given symbol
    """
    symbol = request.symbol.upper()
    
    # Check if model exists
    if symbol not in models:
        raise HTTPException(
            status_code=404, 
            detail=f"Model for {symbol} not found. Available models: {list(models.keys())}"
        )
    
    try:
        # Get recent stock data
        stock_data = collector.get_stock_data_yfinance(symbol, period="1y")
        
        if len(stock_data) < 60:
            raise HTTPException(
                status_code=400,
                detail="Insufficient data for prediction"
            )
        
        # Preprocess data
        preprocessor = StockDataPreprocessor()
        X_pred = preprocessor.preprocess_for_prediction(stock_data)
        
        # Make prediction
        model = models[symbol]
        prediction_scaled = model.predict(X_pred)
        
        # Convert back to original scale
        predicted_price = preprocessor.inverse_transform_prediction(prediction_scaled[0][0])
        
        # Get current price
        current_price = float(stock_data['close'].iloc[-1])
        
        # Calculate changes
        price_change = predicted_price - current_price
        price_change_percent = (price_change / current_price) * 100
        
        # Simple confidence score based on recent volatility
        recent_volatility = stock_data['close'].pct_change().tail(20).std()
        confidence_score = max(0.5, 1 - min(recent_volatility * 10, 0.5))
        
        return PredictionResponse(
            symbol=symbol,
            current_price=current_price,
            predicted_price=predicted_price,
            price_change=price_change,
            price_change_percent=price_change_percent,
            confidence_score=confidence_score
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )

@app.post("/batch_predict")
async def batch_predict(symbols: List[str]):
    """
    Batch prediction for multiple symbols
    """
    results = []
    
    for symbol in symbols:
        try:
            prediction = await predict_stock_price(PredictionRequest(symbol=symbol))
            results.append(prediction.dict())
        except Exception as e:
            results.append({
                "symbol": symbol,
                "error": str(e)
            })
    
    return {"predictions": results}

@app.get("/symbol/{symbol}/info")
async def get_symbol_info(symbol: str):
    """
    Get basic information about a stock symbol
    """
    try:
        stock_data = collector.get_stock_data_yfinance(symbol.upper(), period="1mo")
        
        if stock_data.empty:
            raise HTTPException(status_code=404, detail="Symbol not found")
        
        current_price = float(stock_data['close'].iloc[-1])
        prev_price = float(stock_data['close'].iloc[-2])
        daily_change = current_price - prev_price
        daily_change_percent = (daily_change / prev_price) * 100
        
        volume = int(stock_data['volume'].iloc[-1])
        avg_volume = int(stock_data['volume'].tail(30).mean())
        
        return {
            "symbol": symbol.upper(),
            "current_price": current_price,
            "daily_change": daily_change,
            "daily_change_percent": daily_change_percent,
            "volume": volume,
            "avg_volume_30d": avg_volume,
            "model_available": symbol.upper() in models
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching symbol info: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)