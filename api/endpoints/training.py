"""
Training endpoints
"""

import os
import sys
import logging
import time
import uuid
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))

from fastapi import APIRouter, HTTPException, BackgroundTasks

from api.models import TrainingRequest, TrainingResponse

try:
    from models.unified_models import UnifiedStockModels, DataProcessor
    from data.stock_data import StockDataCollector
except ImportError as e:
    logging.warning(f"Import warning: {e}")

router = APIRouter()
logger = logging.getLogger(__name__)

# Global training status tracking
_training_jobs = {}

class TrainingJob:
    """Training job tracker"""
    def __init__(self, job_id: str, symbol: str, model_type: str):
        self.job_id = job_id
        self.symbol = symbol
        self.model_type = model_type
        self.status = "queued"
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.progress = 0
        self.metrics = {}
        self.error = None

async def train_model_background(job_id: str, request: TrainingRequest):
    """Background training task"""
    job = _training_jobs[job_id]
    
    try:
        job.status = "running"
        job.started_at = datetime.now()
        
        # Initialize components
        collector = StockDataCollector()
        processor = DataProcessor(sequence_length=request.sequence_length)
        models = UnifiedStockModels()
        
        # Collect data
        logger.info(f"Collecting data for {request.symbol}...")
        df = collector.get_stock_data(request.symbol, period=f"{request.years}y")
        
        if df is None or len(df) < 100:
            raise ValueError(f"Insufficient data for symbol {request.symbol}")
        
        job.progress = 20
        
        # Process data
        logger.info("Processing data...")
        X, y = processor.create_sequences(df)
        
        # Split data
        train_size = int(len(X) * 0.7)
        val_size = int(len(X) * 0.2)
        
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
        X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
        
        job.progress = 40
        
        # Train model
        logger.info(f"Training {request.model_type} model...")
        epochs = 5 if request.quick_mode else request.epochs
        
        model_path = f"models/saved/{request.symbol}_{request.model_type}_{job_id}.h5"
        history = models.train_model(
            model_type=request.model_type.value,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            epochs=epochs,
            model_path=model_path
        )
        
        job.progress = 80
        
        # Evaluate model
        logger.info("Evaluating model...")
        metrics = models.evaluate_model(request.model_type.value, X_test, y_test)
        
        # Save models
        models.save_models(f"models/saved/{request.symbol}")
        
        job.progress = 100
        job.status = "completed"
        job.completed_at = datetime.now()
        job.metrics = metrics
        
        logger.info(f"Training completed for {request.symbol} - {request.model_type}")
        
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.completed_at = datetime.now()
        logger.error(f"Training failed for {job_id}: {str(e)}")

@router.post("/", response_model=TrainingResponse)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Train a machine learning model
    
    This endpoint starts training a model for the specified stock symbol.
    Training runs in the background and can be monitored via the job ID.
    """
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Create training job
        job = TrainingJob(job_id, request.symbol, request.model_type.value)
        _training_jobs[job_id] = job
        
        # Start background training
        background_tasks.add_task(train_model_background, job_id, request)
        
        return TrainingResponse(
            symbol=request.symbol,
            model_type=request.model_type.value,
            training_completed=False,
            metrics={},
            training_time=0.0,
            model_id=job_id
        )
        
    except Exception as e:
        logger.error(f"Failed to start training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

@router.get("/job/{job_id}")
async def get_training_status(job_id: str):
    """
    Get training job status
    
    Monitor the progress of a training job.
    """
    if job_id not in _training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    job = _training_jobs[job_id]
    
    training_time = 0.0
    if job.started_at:
        end_time = job.completed_at if job.completed_at else datetime.now()
        training_time = (end_time - job.started_at).total_seconds()
    
    return {
        "job_id": job_id,
        "symbol": job.symbol,
        "model_type": job.model_type,
        "status": job.status,
        "progress": job.progress,
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "training_time": training_time,
        "metrics": job.metrics,
        "error": job.error
    }

@router.get("/jobs")
async def list_training_jobs():
    """
    List all training jobs
    
    Get a list of all training jobs and their current status.
    """
    jobs = []
    for job_id, job in _training_jobs.items():
        training_time = 0.0
        if job.started_at:
            end_time = job.completed_at if job.completed_at else datetime.now()
            training_time = (end_time - job.started_at).total_seconds()
        
        jobs.append({
            "job_id": job_id,
            "symbol": job.symbol,
            "model_type": job.model_type,
            "status": job.status,
            "progress": job.progress,
            "created_at": job.created_at.isoformat(),
            "training_time": training_time
        })
    
    return {
        "jobs": jobs,
        "total": len(jobs),
        "active": len([j for j in _training_jobs.values() if j.status == "running"]),
        "completed": len([j for j in _training_jobs.values() if j.status == "completed"]),
        "failed": len([j for j in _training_jobs.values() if j.status == "failed"])
    }

@router.delete("/job/{job_id}")
async def cancel_training_job(job_id: str):
    """
    Cancel a training job
    
    Attempt to cancel a running training job.
    """
    if job_id not in _training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    job = _training_jobs[job_id]
    
    if job.status in ["completed", "failed"]:
        return {"message": f"Job {job_id} already finished with status: {job.status}"}
    
    # In a real implementation, this would signal the training process to stop
    job.status = "cancelled"
    job.completed_at = datetime.now()
    
    return {"message": f"Job {job_id} cancelled"}

@router.post("/quick/{symbol}")
async def quick_train(symbol: str, model_type: str = "lstm", background_tasks: BackgroundTasks):
    """
    Quick training endpoint
    
    Start a quick training session with reduced parameters for testing.
    """
    request = TrainingRequest(
        symbol=symbol,
        model_type=model_type,
        epochs=10,
        years=2,
        sequence_length=30,
        quick_mode=True
    )
    
    return await train_model(request, background_tasks)

@router.get("/models/{symbol}")
async def list_trained_models(symbol: str):
    """
    List trained models for a symbol
    
    Get information about available trained models for a stock symbol.
    """
    symbol = symbol.upper()
    model_dir = f"models/saved/{symbol}"
    
    if not os.path.exists(model_dir):
        return {
            "symbol": symbol,
            "trained_models": [],
            "message": "No trained models found"
        }
    
    try:
        # Check for model files
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
        
        models_info = []
        for model_file in model_files:
            model_path = os.path.join(model_dir, model_file)
            stat = os.stat(model_path)
            
            models_info.append({
                "model_file": model_file,
                "model_type": model_file.replace('.h5', '').replace(f'{symbol}_', ''),
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat()
            })
        
        return {
            "symbol": symbol,
            "trained_models": models_info,
            "total_models": len(models_info)
        }
        
    except Exception as e:
        logger.error(f"Failed to list models for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@router.delete("/models/{symbol}")
async def delete_trained_models(symbol: str, model_type: str = None):
    """
    Delete trained models for a symbol
    
    Remove trained model files for a stock symbol.
    """
    symbol = symbol.upper()
    model_dir = f"models/saved/{symbol}"
    
    if not os.path.exists(model_dir):
        raise HTTPException(status_code=404, detail=f"No models found for {symbol}")
    
    try:
        deleted_files = []
        
        if model_type:
            # Delete specific model type
            model_file = f"{model_type}.h5"
            model_path = os.path.join(model_dir, model_file)
            if os.path.exists(model_path):
                os.remove(model_path)
                deleted_files.append(model_file)
        else:
            # Delete all models for symbol
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
            for model_file in model_files:
                model_path = os.path.join(model_dir, model_file)
                os.remove(model_path)
                deleted_files.append(model_file)
            
            # Remove directory if empty
            try:
                os.rmdir(model_dir)
            except OSError:
                pass  # Directory not empty
        
        return {
            "symbol": symbol,
            "deleted_files": deleted_files,
            "message": f"Deleted {len(deleted_files)} model files"
        }
        
    except Exception as e:
        logger.error(f"Failed to delete models for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete models: {str(e)}")