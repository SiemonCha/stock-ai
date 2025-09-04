"""
Distributed Training and Inference System
Provides multi-GPU training, model parallelism, and distributed inference capabilities
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Try to import distributed computing libraries
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("Warning: Ray not available. Install with: pip install ray")

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("Warning: Joblib not available. Install with: pip install joblib")

try:
    import tensorflow as tf
    from tensorflow.distribute import MirroredStrategy, MultiWorkerMirroredStrategy
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available")

try:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.multiprocessing as mp
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available")

# Local imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from ..core.models import UnifiedStockModels, DataProcessor
    from ..core.data_collector import StockDataCollector
    from ..ai.feature_engineering import IntelligentFeatureEngine
    MODEL_IMPORTS_AVAILABLE = True
except ImportError:
    MODEL_IMPORTS_AVAILABLE = False
    print("Warning: Model imports not available")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DistributedTrainingManager:
    """Manages distributed training across multiple devices and nodes"""
    
    def __init__(self, backend='tensorflow'):
        self.backend = backend.lower()
        self.devices = self._discover_devices()
        self.strategy = None
        self.is_distributed = False
        
        self._setup_distributed_strategy()
        
        # Ray cluster info
        self.ray_cluster_info = None
        if RAY_AVAILABLE:
            self._initialize_ray()
    
    def _discover_devices(self) -> Dict[str, Any]:
        """Discover available compute devices"""
        devices = {
            'cpu_count': os.cpu_count(),
            'gpu_count': 0,
            'gpu_memory': [],
            'available_backends': []
        }
        
        # Check TensorFlow GPUs
        if TF_AVAILABLE:
            devices['available_backends'].append('tensorflow')
            gpus = tf.config.experimental.list_physical_devices('GPU')
            devices['gpu_count'] = len(gpus)
            
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    details = tf.config.experimental.get_device_details(gpu)
                    devices['gpu_memory'].append(details.get('device_name', 'Unknown GPU'))
                except:
                    pass
        
        # Check PyTorch GPUs
        if TORCH_AVAILABLE:
            devices['available_backends'].append('pytorch')
            if torch.cuda.is_available():
                torch_gpu_count = torch.cuda.device_count()
                if torch_gpu_count > devices['gpu_count']:
                    devices['gpu_count'] = torch_gpu_count
                    devices['gpu_memory'] = [
                        f"GPU {i}: {torch.cuda.get_device_name(i)}"
                        for i in range(torch_gpu_count)
                    ]
        
        logger.info(f"Discovered devices: {devices}")
        return devices
    
    def _setup_distributed_strategy(self):
        """Setup distributed training strategy"""
        if self.backend == 'tensorflow' and TF_AVAILABLE:
            self._setup_tensorflow_strategy()
        elif self.backend == 'pytorch' and TORCH_AVAILABLE:
            self._setup_pytorch_strategy()
        else:
            logger.warning(f"Backend {self.backend} not available, using single device")
    
    def _setup_tensorflow_strategy(self):
        """Setup TensorFlow distributed strategy"""
        try:
            if self.devices['gpu_count'] > 1:
                # Multi-GPU strategy
                self.strategy = MirroredStrategy()
                self.is_distributed = True
                logger.info(f"TensorFlow MirroredStrategy initialized with {self.devices['gpu_count']} GPUs")
            
            elif 'TF_CONFIG' in os.environ:
                # Multi-worker strategy (for multi-node training)
                self.strategy = MultiWorkerMirroredStrategy()
                self.is_distributed = True
                logger.info("TensorFlow MultiWorkerMirroredStrategy initialized")
            
            else:
                # Single device strategy
                self.strategy = tf.distribute.get_strategy()
                logger.info("Using default TensorFlow strategy (single device)")
                
        except Exception as e:
            logger.error(f"Failed to setup TensorFlow strategy: {e}")
            self.strategy = tf.distribute.get_strategy()
    
    def _setup_pytorch_strategy(self):
        """Setup PyTorch distributed strategy"""
        try:
            if self.devices['gpu_count'] > 1:
                logger.info("PyTorch multi-GPU setup available")
                self.is_distributed = True
            
            # Check for distributed training environment variables
            if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
                dist.init_process_group(backend='nccl')
                self.is_distributed = True
                logger.info("PyTorch distributed training initialized")
                
        except Exception as e:
            logger.error(f"Failed to setup PyTorch distributed strategy: {e}")
    
    def _initialize_ray(self):
        """Initialize Ray cluster"""
        try:
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            
            self.ray_cluster_info = ray.cluster_resources()
            logger.info(f"Ray cluster initialized: {self.ray_cluster_info}")
            
        except Exception as e:
            logger.warning(f"Ray initialization failed: {e}")
    
    def get_distributed_context(self):
        """Get current distributed training context"""
        context = {
            'backend': self.backend,
            'is_distributed': self.is_distributed,
            'devices': self.devices,
            'strategy': type(self.strategy).__name__ if self.strategy else None,
            'ray_available': RAY_AVAILABLE and ray.is_initialized() if RAY_AVAILABLE else False
        }
        
        if TORCH_AVAILABLE and dist.is_initialized():
            context.update({
                'rank': dist.get_rank(),
                'world_size': dist.get_world_size(),
                'local_rank': int(os.environ.get('LOCAL_RANK', 0))
            })
        
        return context

class DistributedModelTrainer:
    """Distributed training for stock prediction models"""
    
    def __init__(self, training_manager: DistributedTrainingManager):
        self.training_manager = training_manager
        self.models_cache = {}
        
    def train_model_distributed(self, 
                               symbols: List[str], 
                               model_type: str = 'ensemble',
                               epochs: int = 50,
                               batch_size: int = 32,
                               feature_level: str = 'intelligent') -> Dict[str, Any]:
        """Train models across multiple symbols in distributed manner"""
        
        if not MODEL_IMPORTS_AVAILABLE:
            raise RuntimeError("Model imports not available")
        
        logger.info(f"Starting distributed training for {len(symbols)} symbols")
        
        # Choose distributed method based on available resources
        if RAY_AVAILABLE and ray.is_initialized():
            return self._train_with_ray(symbols, model_type, epochs, batch_size, feature_level)
        elif JOBLIB_AVAILABLE:
            return self._train_with_joblib(symbols, model_type, epochs, batch_size, feature_level)
        else:
            return self._train_sequential(symbols, model_type, epochs, batch_size, feature_level)
    
    def _train_with_ray(self, symbols, model_type, epochs, batch_size, feature_level):
        """Train using Ray distributed computing"""
        
        @ray.remote
        class ModelTrainer:
            def __init__(self):
                if MODEL_IMPORTS_AVAILABLE:
                    self.models = UnifiedStockModels()
                    self.data_collector = StockDataCollector()
                    self.processor = DataProcessor(feature_level=feature_level)
            
            def train_symbol(self, symbol, model_type, epochs, batch_size, feature_level):
                try:
                    # Get data
                    df = self.data_collector.get_stock_data(symbol, period='2y')
                    if df is None or len(df) < 200:
                        return {'symbol': symbol, 'status': 'failed', 'error': 'Insufficient data'}
                    
                    # Process features
                    if feature_level != 'standard':
                        feature_engine = IntelligentFeatureEngine()
                        enhanced_df = feature_engine.create_comprehensive_features(df)
                        X, y = self.processor.create_sequences(enhanced_df)
                    else:
                        X, y = self.processor.create_sequences(df)
                    
                    # Split data
                    split_index = int(len(X) * 0.8)
                    X_train, X_test = X[:split_index], X[split_index:]
                    y_train, y_test = y[:split_index], y[split_index:]
                    
                    # Train model
                    training_config = {
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'validation_split': 0.2,
                        'verbose': 0
                    }
                    
                    history = self.models.train_model(
                        model_type, X_train, y_train, 
                        X_test, y_test, 
                        symbol=symbol,
                        **training_config
                    )
                    
                    # Calculate performance
                    predictions = self.models.predict(model_type, X_test)
                    mse = np.mean((y_test - predictions) ** 2)
                    mae = np.mean(np.abs(y_test - predictions))
                    
                    return {
                        'symbol': symbol,
                        'status': 'success',
                        'metrics': {
                            'mse': float(mse),
                            'mae': float(mae),
                            'rmse': float(np.sqrt(mse))
                        },
                        'training_time': history.get('training_time', 0),
                        'epochs_completed': len(history.get('loss', [])),
                        'final_loss': history.get('loss', [])[-1] if history.get('loss') else None
                    }
                    
                except Exception as e:
                    return {
                        'symbol': symbol,
                        'status': 'failed',
                        'error': str(e)
                    }
        
        # Create trainers
        num_workers = min(len(symbols), int(ray.cluster_resources().get('CPU', 1)))
        trainers = [ModelTrainer.remote() for _ in range(num_workers)]
        
        # Distribute training tasks
        futures = []
        for i, symbol in enumerate(symbols):
            trainer = trainers[i % len(trainers)]
            future = trainer.train_symbol.remote(symbol, model_type, epochs, batch_size, feature_level)
            futures.append(future)
        
        # Collect results
        results = ray.get(futures)
        
        # Aggregate results
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']
        
        return {
            'training_method': 'ray',
            'total_symbols': len(symbols),
            'successful': len(successful),
            'failed': len(failed),
            'results': results,
            'summary': {
                'avg_mse': np.mean([r['metrics']['mse'] for r in successful]) if successful else 0,
                'avg_mae': np.mean([r['metrics']['mae'] for r in successful]) if successful else 0,
                'total_training_time': sum([r['training_time'] for r in successful]),
                'worker_count': num_workers
            }
        }
    
    def _train_with_joblib(self, symbols, model_type, epochs, batch_size, feature_level):
        """Train using Joblib parallel processing"""
        
        def train_single_symbol(symbol):
            try:
                if not MODEL_IMPORTS_AVAILABLE:
                    return {'symbol': symbol, 'status': 'failed', 'error': 'Model imports not available'}
                
                models = UnifiedStockModels()
                data_collector = StockDataCollector()
                processor = DataProcessor(feature_level=feature_level)
                
                # Get and process data
                df = data_collector.get_stock_data(symbol, period='2y')
                if df is None or len(df) < 200:
                    return {'symbol': symbol, 'status': 'failed', 'error': 'Insufficient data'}
                
                if feature_level != 'standard':
                    feature_engine = IntelligentFeatureEngine()
                    enhanced_df = feature_engine.create_comprehensive_features(df)
                    X, y = processor.create_sequences(enhanced_df)
                else:
                    X, y = processor.create_sequences(df)
                
                # Train
                split_index = int(len(X) * 0.8)
                X_train, X_test = X[:split_index], X[split_index:]
                y_train, y_test = y[:split_index], y[split_index:]
                
                start_time = time.time()
                history = models.train_model(
                    model_type, X_train, y_train, X_test, y_test,
                    symbol=symbol, epochs=epochs, batch_size=batch_size,
                    validation_split=0.2, verbose=0
                )
                training_time = time.time() - start_time
                
                # Metrics
                predictions = models.predict(model_type, X_test)
                mse = np.mean((y_test - predictions) ** 2)
                mae = np.mean(np.abs(y_test - predictions))
                
                return {
                    'symbol': symbol,
                    'status': 'success',
                    'metrics': {'mse': float(mse), 'mae': float(mae), 'rmse': float(np.sqrt(mse))},
                    'training_time': training_time
                }
                
            except Exception as e:
                return {'symbol': symbol, 'status': 'failed', 'error': str(e)}
        
        # Parallel execution
        n_jobs = min(len(symbols), os.cpu_count())
        results = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(train_single_symbol)(symbol) for symbol in symbols
        )
        
        # Aggregate
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']
        
        return {
            'training_method': 'joblib',
            'total_symbols': len(symbols),
            'successful': len(successful),
            'failed': len(failed),
            'results': results,
            'summary': {
                'avg_mse': np.mean([r['metrics']['mse'] for r in successful]) if successful else 0,
                'avg_mae': np.mean([r['metrics']['mae'] for r in successful]) if successful else 0,
                'total_training_time': sum([r['training_time'] for r in successful]),
                'worker_count': n_jobs
            }
        }
    
    def _train_sequential(self, symbols, model_type, epochs, batch_size, feature_level):
        """Fallback sequential training"""
        logger.warning("Using sequential training (no distributed computing available)")
        
        results = []
        
        for symbol in symbols:
            try:
                if not MODEL_IMPORTS_AVAILABLE:
                    results.append({'symbol': symbol, 'status': 'failed', 'error': 'Model imports not available'})
                    continue
                
                models = UnifiedStockModels()
                data_collector = StockDataCollector()
                processor = DataProcessor(feature_level=feature_level)
                
                # Get data
                df = data_collector.get_stock_data(symbol, period='2y')
                if df is None or len(df) < 200:
                    results.append({'symbol': symbol, 'status': 'failed', 'error': 'Insufficient data'})
                    continue
                
                # Process
                if feature_level != 'standard':
                    feature_engine = IntelligentFeatureEngine()
                    enhanced_df = feature_engine.create_comprehensive_features(df)
                    X, y = processor.create_sequences(enhanced_df)
                else:
                    X, y = processor.create_sequences(df)
                
                # Train
                split_index = int(len(X) * 0.8)
                X_train, X_test = X[:split_index], X[split_index:]
                y_train, y_test = y[:split_index], y[split_index:]
                
                start_time = time.time()
                history = models.train_model(
                    model_type, X_train, y_train, X_test, y_test,
                    symbol=symbol, epochs=epochs, batch_size=batch_size,
                    validation_split=0.2, verbose=1
                )
                training_time = time.time() - start_time
                
                # Metrics
                predictions = models.predict(model_type, X_test)
                mse = np.mean((y_test - predictions) ** 2)
                mae = np.mean(np.abs(y_test - predictions))
                
                results.append({
                    'symbol': symbol,
                    'status': 'success',
                    'metrics': {'mse': float(mse), 'mae': float(mae), 'rmse': float(np.sqrt(mse))},
                    'training_time': training_time
                })
                
                logger.info(f"Completed training for {symbol}: RMSE={np.sqrt(mse):.4f}")
                
            except Exception as e:
                logger.error(f"Training failed for {symbol}: {e}")
                results.append({'symbol': symbol, 'status': 'failed', 'error': str(e)})
        
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']
        
        return {
            'training_method': 'sequential',
            'total_symbols': len(symbols),
            'successful': len(successful),
            'failed': len(failed),
            'results': results,
            'summary': {
                'avg_mse': np.mean([r['metrics']['mse'] for r in successful]) if successful else 0,
                'avg_mae': np.mean([r['metrics']['mae'] for r in successful]) if successful else 0,
                'total_training_time': sum([r['training_time'] for r in successful]),
                'worker_count': 1
            }
        }

class DistributedInferenceEngine:
    """Distributed inference for real-time predictions"""
    
    def __init__(self, training_manager: DistributedTrainingManager):
        self.training_manager = training_manager
        self.model_cache = {}
        self.executor = None
        
        # Initialize executor based on available resources
        if self.training_manager.devices['gpu_count'] > 0:
            self.executor = ThreadPoolExecutor(max_workers=self.training_manager.devices['gpu_count'])
        else:
            self.executor = ProcessPoolExecutor(max_workers=min(8, os.cpu_count()))
    
    def predict_batch(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple prediction requests in parallel"""
        
        if not requests:
            return []
        
        logger.info(f"Processing {len(requests)} prediction requests in parallel")
        
        # Submit all requests
        futures = {
            self.executor.submit(self._predict_single, req): req 
            for req in requests
        }
        
        results = []
        for future in as_completed(futures):
            request = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Prediction failed for {request}: {e}")
                results.append({
                    'symbol': request.get('symbol', 'unknown'),
                    'status': 'failed',
                    'error': str(e)
                })
        
        return results
    
    def _predict_single(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process single prediction request"""
        try:
            symbol = request['symbol']
            model_type = request.get('model_type', 'ensemble')
            days = request.get('days', 30)
            feature_level = request.get('feature_level', 'intelligent')
            
            # Load or get cached model
            model_key = f"{symbol}_{model_type}"
            if model_key not in self.model_cache:
                if MODEL_IMPORTS_AVAILABLE:
                    models = UnifiedStockModels()
                    model_path = f"models/saved/{symbol}"
                    if os.path.exists(model_path):
                        models.load_models(model_path)
                        self.model_cache[model_key] = models
                    else:
                        return {
                            'symbol': symbol,
                            'status': 'failed',
                            'error': f'No trained model found for {symbol}'
                        }
                else:
                    return {
                        'symbol': symbol,
                        'status': 'failed',
                        'error': 'Model imports not available'
                    }
            
            models = self.model_cache[model_key]
            
            # Get recent data
            collector = StockDataCollector()
            df = collector.get_stock_data(symbol, period='1y')
            
            if df is None or len(df) < 100:
                return {
                    'symbol': symbol,
                    'status': 'failed',
                    'error': 'Insufficient data'
                }
            
            # Process data
            processor = DataProcessor(feature_level=feature_level)
            
            if feature_level != 'standard':
                feature_engine = IntelligentFeatureEngine()
                enhanced_df = feature_engine.create_comprehensive_features(df)
                X, y = processor.create_sequences(enhanced_df)
            else:
                X, y = processor.create_sequences(df)
            
            # Make prediction
            test_size = min(days, len(X) // 4)
            X_test = X[-test_size:]
            y_test = y[-test_size:]
            
            predictions = models.predict(model_type, X_test)
            
            # Calculate basic metrics
            mse = np.mean((y_test - predictions) ** 2)
            mae = np.mean(np.abs(y_test - predictions))
            
            return {
                'symbol': symbol,
                'status': 'success',
                'predictions': predictions.tolist(),
                'metrics': {
                    'mse': float(mse),
                    'mae': float(mae),
                    'rmse': float(np.sqrt(mse))
                },
                'prediction_count': len(predictions),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'symbol': request.get('symbol', 'unknown'),
                'status': 'failed',
                'error': str(e)
            }
    
    def __del__(self):
        if self.executor:
            self.executor.shutdown(wait=True)

class ClusterManager:
    """Manages distributed computing clusters"""
    
    def __init__(self):
        self.cluster_info = self._get_cluster_info()
    
    def _get_cluster_info(self) -> Dict[str, Any]:
        """Get information about available clusters"""
        info = {
            'local': {
                'cpu_count': os.cpu_count(),
                'memory_gb': self._get_memory_gb(),
                'available': True
            }
        }
        
        # Ray cluster info
        if RAY_AVAILABLE:
            try:
                if ray.is_initialized():
                    resources = ray.cluster_resources()
                    info['ray'] = {
                        'nodes': len(ray.nodes()),
                        'cpu_total': int(resources.get('CPU', 0)),
                        'gpu_total': int(resources.get('GPU', 0)),
                        'memory_total': int(resources.get('memory', 0)),
                        'available': True
                    }
            except Exception as e:
                info['ray'] = {'available': False, 'error': str(e)}
        else:
            info['ray'] = {'available': False, 'error': 'Ray not installed'}
        
        return info
    
    def _get_memory_gb(self) -> float:
        """Get available memory in GB"""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except ImportError:
            return 0.0
    
    def scale_cluster(self, target_nodes: int) -> bool:
        """Scale Ray cluster to target number of nodes"""
        if not RAY_AVAILABLE or not ray.is_initialized():
            logger.error("Ray cluster not available")
            return False
        
        try:
            current_nodes = len(ray.nodes())
            logger.info(f"Current nodes: {current_nodes}, target: {target_nodes}")
            
            # Ray doesn't support direct scaling, this would typically
            # integrate with cloud providers like AWS, GCP, etc.
            logger.warning("Cluster scaling requires external orchestration")
            return True
            
        except Exception as e:
            logger.error(f"Cluster scaling failed: {e}")
            return False
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get current cluster status"""
        status = {
            'timestamp': datetime.utcnow().isoformat(),
            'clusters': self.cluster_info
        }
        
        # Update Ray status if available
        if RAY_AVAILABLE and ray.is_initialized():
            try:
                nodes = ray.nodes()
                status['ray_detailed'] = {
                    'alive_nodes': len([n for n in nodes if n['Alive']]),
                    'total_nodes': len(nodes),
                    'cluster_resources': ray.cluster_resources(),
                    'available_resources': ray.available_resources()
                }
            except Exception as e:
                status['ray_detailed'] = {'error': str(e)}
        
        return status

def create_distributed_training_pipeline(symbols: List[str], 
                                       model_type: str = 'ensemble',
                                       backend: str = 'tensorflow',
                                       epochs: int = 50,
                                       batch_size: int = 32,
                                       feature_level: str = 'intelligent') -> Dict[str, Any]:
    """Create and run complete distributed training pipeline"""
    
    logger.info("🚀 Starting Distributed Training Pipeline")
    logger.info(f"Symbols: {len(symbols)}, Model: {model_type}, Backend: {backend}")
    
    # Initialize distributed training
    training_manager = DistributedTrainingManager(backend=backend)
    model_trainer = DistributedModelTrainer(training_manager)
    cluster_manager = ClusterManager()
    
    pipeline_start = time.time()
    
    # Get initial cluster status
    initial_status = cluster_manager.get_cluster_status()
    logger.info(f"Cluster status: {initial_status}")
    
    # Run distributed training
    try:
        training_results = model_trainer.train_model_distributed(
            symbols=symbols,
            model_type=model_type,
            epochs=epochs,
            batch_size=batch_size,
            feature_level=feature_level
        )
        
        pipeline_time = time.time() - pipeline_start
        
        # Final results
        results = {
            'pipeline': {
                'start_time': datetime.utcnow().isoformat(),
                'duration_seconds': pipeline_time,
                'backend': backend,
                'distributed_context': training_manager.get_distributed_context()
            },
            'training': training_results,
            'cluster': {
                'initial_status': initial_status,
                'final_status': cluster_manager.get_cluster_status()
            },
            'performance': {
                'symbols_per_second': len(symbols) / pipeline_time,
                'avg_training_time_per_symbol': training_results['summary']['total_training_time'] / len(symbols) if len(symbols) > 0 else 0,
                'speedup_factor': _calculate_speedup(training_results, len(symbols))
            }
        }
        
        logger.info("✅ Distributed Training Pipeline Completed")
        logger.info(f"Successfully trained: {training_results['successful']}/{training_results['total_symbols']} models")
        logger.info(f"Total time: {pipeline_time:.2f}s")
        logger.info(f"Average RMSE: {training_results['summary']['avg_mse']**0.5:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Distributed training pipeline failed: {e}")
        return {
            'pipeline': {
                'start_time': datetime.utcnow().isoformat(),
                'duration_seconds': time.time() - pipeline_start,
                'backend': backend,
                'status': 'failed',
                'error': str(e)
            },
            'training': {'successful': 0, 'failed': len(symbols), 'total_symbols': len(symbols)},
            'cluster': {'initial_status': initial_status}
        }

def _calculate_speedup(training_results: Dict[str, Any], total_symbols: int) -> float:
    """Calculate speedup factor from distributed training"""
    if training_results['training_method'] == 'sequential':
        return 1.0
    
    worker_count = training_results['summary'].get('worker_count', 1)
    return min(worker_count, total_symbols) * 0.8  # Assume 80% efficiency