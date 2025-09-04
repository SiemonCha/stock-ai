"""
Production-Ready Stock AI API
FastAPI-based API with monitoring, authentication, and scalability features
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager
import asyncio
from functools import wraps
import warnings
warnings.filterwarnings('ignore')

try:
    from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.responses import JSONResponse
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel, validator
    FASTAPI_AVAILABLE = True
except ImportError:
    print("Warning: FastAPI not available. Install with: pip install fastapi uvicorn")
    FASTAPI_AVAILABLE = False
    FastAPI = HTTPBearer = HTTPAuthorizationCredentials = None
    BaseModel = object

try:
    import redis
    from cachetools import TTLCache
    CACHING_AVAILABLE = True
except ImportError:
    print("Warning: Redis/cachetools not available. Caching disabled.")
    redis = None
    TTLCache = None
    CACHING_AVAILABLE = False

try:
    import psutil
    import GPUtil
    MONITORING_AVAILABLE = True
except ImportError:
    print("Warning: psutil/GPUtil not available. System monitoring limited.")
    psutil = None
    GPUtil = None
    MONITORING_AVAILABLE = False

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from ..core.models import UnifiedStockModels, DataProcessor
    from ..core.data_collector import StockDataCollector
    from ..ai.feature_engineering import IntelligentFeatureEngine
    from ..ai.realtime_analysis import RealTimeMarketIntelligence
    from ..ai.market_regimes import create_regime_detection_pipeline
    from ..ai.portfolio_optimizer import MultiAssetOptimizer
    MODEL_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some model imports not available: {e}")
    MODEL_IMPORTS_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
class APIConfig:
    """API Configuration"""
    
    # Server settings
    HOST = os.getenv("API_HOST", "0.0.0.0")
    PORT = int(os.getenv("API_PORT", 8000))
    WORKERS = int(os.getenv("API_WORKERS", 4))
    
    # Security
    API_KEY = os.getenv("API_KEY", "your-secret-api-key-here")
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # Rate limiting
    RATE_LIMIT_CALLS = int(os.getenv("RATE_LIMIT_CALLS", 100))
    RATE_LIMIT_PERIOD = int(os.getenv("RATE_LIMIT_PERIOD", 3600))  # 1 hour
    
    # Caching
    CACHE_TTL = int(os.getenv("CACHE_TTL", 300))  # 5 minutes
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Model settings
    MODEL_CACHE_SIZE = int(os.getenv("MODEL_CACHE_SIZE", 10))
    MAX_PREDICTION_DAYS = int(os.getenv("MAX_PREDICTION_DAYS", 90))
    DEFAULT_PREDICTION_DAYS = int(os.getenv("DEFAULT_PREDICTION_DAYS", 30))

config = APIConfig()

# Pydantic models for API
class PredictionRequest(BaseModel):
    """Stock prediction request model"""
    symbol: str
    model_type: str = "ensemble"
    days: int = 30
    include_uncertainty: bool = True
    include_regime_analysis: bool = False
    feature_level: str = "intelligent"
    
    @validator('symbol')
    def validate_symbol(cls, v):
        if not v or len(v) < 1 or len(v) > 10:
            raise ValueError('Symbol must be 1-10 characters')
        return v.upper()
    
    @validator('days')
    def validate_days(cls, v):
        if v < 1 or v > config.MAX_PREDICTION_DAYS:
            raise ValueError(f'Days must be between 1 and {config.MAX_PREDICTION_DAYS}')
        return v
    
    @validator('model_type')
    def validate_model_type(cls, v):
        allowed_models = ['lstm', 'gru', 'transformer', 'cnn_lstm', 'ensemble', 
                         'nextgen_ensemble', 'advanced_ensemble']
        if v not in allowed_models:
            raise ValueError(f'Model type must be one of: {allowed_models}')
        return v

class PortfolioOptimizationRequest(BaseModel):
    """Portfolio optimization request model"""
    symbols: List[str]
    method: str = "mean_reversion"
    lookback_days: int = 252
    risk_tolerance: float = 0.5
    include_alternatives: bool = False
    
    @validator('symbols')
    def validate_symbols(cls, v):
        if len(v) < 2 or len(v) > 20:
            raise ValueError('Portfolio must contain 2-20 symbols')
        return [s.upper() for s in v]
    
    @validator('method')
    def validate_method(cls, v):
        allowed_methods = ['mean_reversion', 'momentum', 'mean_variance', 'risk_parity', 
                          'black_litterman', 'hrp', 'kelly_criterion']
        if v not in allowed_methods:
            raise ValueError(f'Method must be one of: {allowed_methods}')
        return v

class RegimeAnalysisRequest(BaseModel):
    """Market regime analysis request model"""
    symbol: str
    lookback_days: int = 252
    include_forecast: bool = True
    forecast_horizon: int = 5
    
    @validator('symbol')
    def validate_symbol(cls, v):
        return v.upper()

# Cache and rate limiting
class CacheManager:
    """Manages caching for API responses"""
    
    def __init__(self):
        self.memory_cache = TTLCache(maxsize=1000, ttl=config.CACHE_TTL) if TTLCache else {}
        self.redis_client = None
        
        if redis and CACHING_AVAILABLE:
            try:
                self.redis_client = redis.from_url(config.REDIS_URL, decode_responses=True)
                self.redis_client.ping()
                logger.info("Redis cache connected successfully")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self.redis_client = None
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            # Try Redis first
            if self.redis_client:
                value = self.redis_client.get(key)
                if value:
                    return json.loads(value)
            
            # Fallback to memory cache
            if TTLCache:
                return self.memory_cache.get(key)
            else:
                return self.memory_cache.get(key)
                
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            ttl = ttl or config.CACHE_TTL
            json_value = json.dumps(value, default=str)
            
            # Try Redis first
            if self.redis_client:
                self.redis_client.setex(key, ttl, json_value)
                return True
            
            # Fallback to memory cache
            if TTLCache:
                self.memory_cache[key] = value
            else:
                self.memory_cache[key] = value
                
            return True
            
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False

class RateLimiter:
    """Simple rate limiter using cache"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.calls_limit = config.RATE_LIMIT_CALLS
        self.time_window = config.RATE_LIMIT_PERIOD
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if client is within rate limits"""
        key = f"rate_limit:{client_id}"
        current_calls = self.cache.get(key) or 0
        
        if current_calls >= self.calls_limit:
            return False
        
        self.cache.set(key, current_calls + 1, self.time_window)
        return True

# Global instances
cache_manager = CacheManager()
rate_limiter = RateLimiter(cache_manager)
model_cache = {}

# Monitoring
class SystemMonitor:
    """System monitoring and health checks"""
    
    @staticmethod
    def get_system_stats() -> Dict[str, Any]:
        """Get system statistics"""
        stats = {
            'timestamp': datetime.utcnow().isoformat(),
            'uptime': time.time(),
            'api_version': '2.0.0'
        }
        
        if MONITORING_AVAILABLE and psutil:
            # CPU and Memory
            stats.update({
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
            })
            
            # GPU if available
            try:
                if GPUtil:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        stats['gpu'] = {
                            'utilization': f"{gpu.load * 100:.1f}%",
                            'memory': f"{gpu.memoryUtil * 100:.1f}%",
                            'temperature': f"{gpu.temperature}°C"
                        }
            except Exception as e:
                logger.debug(f"GPU monitoring not available: {e}")
        
        return stats
    
    @staticmethod
    def get_model_stats() -> Dict[str, Any]:
        """Get model performance statistics"""
        return {
            'loaded_models': len(model_cache),
            'model_types': list(set(model_cache.values())) if model_cache else [],
            'cache_size': len(cache_manager.memory_cache) if hasattr(cache_manager, 'memory_cache') else 0
        }

monitor = SystemMonitor()

# Security
def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> str:
    """Validate API key"""
    if not credentials or credentials.credentials != config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

def rate_limit_check(request: Request):
    """Check rate limits"""
    client_id = request.client.host
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    return True

# Model management
async def load_model(symbol: str, model_type: str) -> Optional[UnifiedStockModels]:
    """Load or get cached model"""
    cache_key = f"{symbol}_{model_type}"
    
    if cache_key in model_cache:
        return model_cache[cache_key]
    
    try:
        if MODEL_IMPORTS_AVAILABLE:
            models = UnifiedStockModels()
            model_path = f"models/saved/{symbol}"
            
            if os.path.exists(model_path):
                models.load_models(model_path)
                model_cache[cache_key] = models
                logger.info(f"Loaded model for {symbol} ({model_type})")
                return models
        
        logger.warning(f"No trained model found for {symbol}")
        return None
        
    except Exception as e:
        logger.error(f"Failed to load model for {symbol}: {e}")
        return None

# API Routes
if FASTAPI_AVAILABLE:
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan manager"""
        logger.info("🚀 Stock AI API starting up...")
        yield
        logger.info("🛑 Stock AI API shutting down...")
    
    app = FastAPI(
        title="Stock AI Production API",
        description="Advanced Stock Market Prediction and Analysis API",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all requests"""
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
        
        response.headers["X-Process-Time"] = str(process_time)
        return response
    
    # Health check endpoints
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "2.0.0"
        }
    
    @app.get("/health/detailed")
    async def detailed_health_check():
        """Detailed health check with system stats"""
        system_stats = monitor.get_system_stats()
        model_stats = monitor.get_model_stats()
        
        return {
            "status": "healthy",
            "system": system_stats,
            "models": model_stats,
            "cache": {
                "redis_available": cache_manager.redis_client is not None,
                "memory_cache_size": len(cache_manager.memory_cache) if hasattr(cache_manager, 'memory_cache') else 0
            }
        }
    
    # Prediction endpoints
    @app.post("/predict")
    async def predict_stock(
        request: PredictionRequest,
        background_tasks: BackgroundTasks,
        api_key: str = Depends(get_api_key),
        rate_check: bool = Depends(rate_limit_check)
    ):
        """Stock price prediction endpoint"""
        
        # Check cache first
        cache_key = f"predict:{request.symbol}:{request.model_type}:{request.days}:{request.feature_level}"
        cached_result = cache_manager.get(cache_key)
        if cached_result:
            return JSONResponse(content=cached_result, headers={"X-Cache": "HIT"})
        
        try:
            # Load model
            models = await load_model(request.symbol, request.model_type)
            if not models:
                raise HTTPException(status_code=404, detail=f"No trained model found for {request.symbol}")
            
            # Get data
            collector = StockDataCollector()
            df = collector.get_stock_data(request.symbol, period="1y")
            
            if df is None or len(df) < 100:
                raise HTTPException(status_code=400, detail="Insufficient data for prediction")
            
            # Process data
            processor = DataProcessor(feature_level=request.feature_level)
            
            if request.feature_level != 'standard' and MODEL_IMPORTS_AVAILABLE:
                feature_engine = IntelligentFeatureEngine()
                enhanced_df = feature_engine.create_comprehensive_features(df)
                X, y = processor.create_sequences(enhanced_df)
            else:
                X, y = processor.create_sequences(df)
            
            # Make prediction
            test_size = min(request.days, len(X) // 4)
            X_test = X[-test_size:]
            y_test = y[-test_size:]
            
            # Get predictions with uncertainty if available
            if request.include_uncertainty and hasattr(models, 'predict_with_uncertainty'):
                try:
                    predictions, uncertainty = models.predict_with_uncertainty(request.model_type, X_test)
                    
                    # Calculate confidence intervals
                    z_score = 1.96  # 95% confidence
                    confidence_intervals = {
                        'lower': (predictions - z_score * np.sqrt(uncertainty)).tolist(),
                        'upper': (predictions + z_score * np.sqrt(uncertainty)).tolist()
                    }
                except:
                    predictions = models.predict(request.model_type, X_test)
                    uncertainty = np.zeros_like(predictions)
                    confidence_intervals = None
            else:
                predictions = models.predict(request.model_type, X_test)
                uncertainty = np.zeros_like(predictions)
                confidence_intervals = None
            
            # Calculate metrics
            mse = np.mean((y_test - predictions) ** 2)
            mae = np.mean(np.abs(y_test - predictions))
            
            # Directional accuracy
            directional_accuracy = None
            if len(y_test) > 1:
                actual_direction = np.diff(y_test) > 0
                pred_direction = np.diff(predictions) > 0
                directional_accuracy = float(np.mean(actual_direction == pred_direction))
            
            # Prepare response
            response = {
                "symbol": request.symbol,
                "model_type": request.model_type,
                "predictions": predictions.tolist(),
                "dates": df.index[-test_size:].strftime('%Y-%m-%d').tolist(),
                "metrics": {
                    "rmse": float(np.sqrt(mse)),
                    "mae": float(mae),
                    "directional_accuracy": directional_accuracy
                },
                "metadata": {
                    "prediction_days": test_size,
                    "feature_level": request.feature_level,
                    "data_points": len(df),
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            if request.include_uncertainty and confidence_intervals:
                response["uncertainty"] = {
                    "values": uncertainty.tolist(),
                    "confidence_intervals": confidence_intervals,
                    "mean_uncertainty": float(np.mean(uncertainty))
                }
            
            # Cache result
            cache_manager.set(cache_key, response, config.CACHE_TTL)
            
            return JSONResponse(content=response, headers={"X-Cache": "MISS"})
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Prediction error for {request.symbol}: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    @app.post("/portfolio/optimize")
    async def optimize_portfolio(
        request: PortfolioOptimizationRequest,
        api_key: str = Depends(get_api_key),
        rate_check: bool = Depends(rate_limit_check)
    ):
        """Portfolio optimization endpoint"""
        
        cache_key = f"portfolio:{'-'.join(request.symbols)}:{request.method}:{request.lookback_days}"
        cached_result = cache_manager.get(cache_key)
        if cached_result:
            return JSONResponse(content=cached_result, headers={"X-Cache": "HIT"})
        
        try:
            if not MODEL_IMPORTS_AVAILABLE:
                raise HTTPException(status_code=503, detail="Portfolio optimization not available")
            
            optimizer = MultiAssetOptimizer(request.symbols)
            
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=request.lookback_days)
            
            data = {}
            collector = StockDataCollector()
            
            for symbol in request.symbols:
                df = collector.get_stock_data(symbol, start=start_date.strftime('%Y-%m-%d'), 
                                            end=end_date.strftime('%Y-%m-%d'))
                if df is not None and len(df) > 50:
                    data[symbol] = df['Close']
            
            if len(data) < 2:
                raise HTTPException(status_code=400, detail="Insufficient data for portfolio optimization")
            
            # Optimize portfolio
            result = optimizer.optimize_portfolio(
                data=data,
                method=request.method,
                risk_tolerance=request.risk_tolerance
            )
            
            response = {
                "symbols": request.symbols,
                "method": request.method,
                "optimization_result": result,
                "metadata": {
                    "lookback_days": request.lookback_days,
                    "risk_tolerance": request.risk_tolerance,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            cache_manager.set(cache_key, response, config.CACHE_TTL * 2)  # Cache longer
            return JSONResponse(content=response, headers={"X-Cache": "MISS"})
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Portfolio optimization error: {e}")
            raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")
    
    @app.post("/regime/analyze")
    async def analyze_regime(
        request: RegimeAnalysisRequest,
        api_key: str = Depends(get_api_key),
        rate_check: bool = Depends(rate_limit_check)
    ):
        """Market regime analysis endpoint"""
        
        cache_key = f"regime:{request.symbol}:{request.lookback_days}:{request.forecast_horizon}"
        cached_result = cache_manager.get(cache_key)
        if cached_result:
            return JSONResponse(content=cached_result, headers={"X-Cache": "HIT"})
        
        try:
            if not MODEL_IMPORTS_AVAILABLE:
                raise HTTPException(status_code=503, detail="Regime analysis not available")
            
            # Get data
            collector = StockDataCollector()
            end_date = datetime.now()
            start_date = end_date - timedelta(days=request.lookback_days)
            
            df = collector.get_stock_data(request.symbol, 
                                        start=start_date.strftime('%Y-%m-%d'),
                                        end=end_date.strftime('%Y-%m-%d'))
            
            if df is None or len(df) < 100:
                raise HTTPException(status_code=400, detail="Insufficient data for regime analysis")
            
            # Run regime detection
            regime_pipeline = create_regime_detection_pipeline(df)
            
            detector = regime_pipeline['detector']
            regimes = regime_pipeline['regimes']
            characteristics = regime_pipeline['characteristics']
            regime_labels = regime_pipeline['regime_labels']
            transition_matrix = regime_pipeline['transition_matrix']
            
            # Current regime
            current_regime = int(regimes[-1])
            current_regime_name = regime_labels.get(current_regime, f"Regime {current_regime}")
            
            # Forecast if requested
            forecast = None
            if request.include_forecast:
                try:
                    future_probs = detector.predict_regime_probability(
                        regime_pipeline['features'], 
                        horizon=request.forecast_horizon
                    )
                    if future_probs:
                        forecast = {
                            regime_labels.get(regime_id, f"Regime {regime_id}"): prob 
                            for regime_id, prob in sorted(future_probs.items())
                        }
                except Exception as e:
                    logger.warning(f"Regime forecasting failed: {e}")
            
            # Strategy recommendations
            strategy = regime_pipeline['strategy']
            strategy_params = strategy.get_strategy_parameters(current_regime)
            should_trade = strategy.should_trade(current_regime)
            
            response = {
                "symbol": request.symbol,
                "current_regime": {
                    "id": current_regime,
                    "name": current_regime_name,
                    "characteristics": characteristics.get(current_regime, {})
                },
                "regime_distribution": {
                    regime_labels.get(int(regime_id), f"Regime {regime_id}"): int(count)
                    for regime_id, count in pd.Series(regimes).value_counts().items()
                },
                "transition_matrix": transition_matrix.tolist(),
                "strategy_recommendation": {
                    "should_trade": should_trade,
                    "position_size_multiplier": strategy_params['position_size'],
                    "stop_loss": strategy_params['stop_loss'],
                    "take_profit": strategy_params['take_profit']
                },
                "forecast": forecast,
                "metadata": {
                    "lookback_days": request.lookback_days,
                    "total_regimes": len(characteristics),
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            cache_manager.set(cache_key, response, config.CACHE_TTL)
            return JSONResponse(content=response, headers={"X-Cache": "MISS"})
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Regime analysis error for {request.symbol}: {e}")
            raise HTTPException(status_code=500, detail=f"Regime analysis failed: {str(e)}")
    
    @app.get("/models/list")
    async def list_models(api_key: str = Depends(get_api_key)):
        """List available trained models"""
        models_dir = Path("models/saved")
        available_models = []
        
        if models_dir.exists():
            for symbol_dir in models_dir.iterdir():
                if symbol_dir.is_dir():
                    symbol = symbol_dir.name
                    model_files = list(symbol_dir.glob("*.h5")) + list(symbol_dir.glob("*.pkl"))
                    if model_files:
                        available_models.append({
                            "symbol": symbol,
                            "model_files": [f.name for f in model_files],
                            "last_updated": max(f.stat().st_mtime for f in model_files)
                        })
        
        return {
            "available_models": available_models,
            "total_symbols": len(available_models),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @app.get("/status")
    async def get_status(api_key: str = Depends(get_api_key)):
        """Get API status and statistics"""
        return {
            "status": "operational",
            "version": "2.0.0",
            "uptime": time.time(),
            "system": monitor.get_system_stats(),
            "models": monitor.get_model_stats(),
            "rate_limits": {
                "calls_per_hour": config.RATE_LIMIT_CALLS,
                "current_period": config.RATE_LIMIT_PERIOD
            },
            "features": {
                "caching": CACHING_AVAILABLE,
                "monitoring": MONITORING_AVAILABLE,
                "advanced_models": MODEL_IMPORTS_AVAILABLE
            }
        }
    
    # Error handlers
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=404,
            content={"error": "Endpoint not found", "detail": str(exc.detail)}
        )
    
    @app.exception_handler(500)
    async def internal_error_handler(request: Request, exc: Exception):
        logger.error(f"Internal server error: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": "Please try again later"}
        )

else:
    # Fallback when FastAPI not available
    class MockAPI:
        """Mock API when FastAPI is not available"""
        
        def __init__(self):
            logger.warning("FastAPI not available. API functionality disabled.")
        
        def run(self):
            print("API not available. Install FastAPI: pip install fastapi uvicorn")
    
    app = MockAPI()

def run_api():
    """Run the API server"""
    if not FASTAPI_AVAILABLE:
        print("Cannot start API: FastAPI not installed")
        print("Install with: pip install fastapi uvicorn")
        return
    
    try:
        import uvicorn
        
        logger.info(f"🚀 Starting Stock AI API on {config.HOST}:{config.PORT}")
        logger.info(f"📚 API Documentation: http://{config.HOST}:{config.PORT}/docs")
        
        uvicorn.run(
            "src.api.production_api:app",
            host=config.HOST,
            port=config.PORT,
            workers=config.WORKERS,
            reload=False,
            access_log=True,
            log_level="info"
        )
        
    except ImportError:
        print("Uvicorn not available. Install with: pip install uvicorn")
    except Exception as e:
        logger.error(f"Failed to start API: {e}")

if __name__ == "__main__":
    run_api()