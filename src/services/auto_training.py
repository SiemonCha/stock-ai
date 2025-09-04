"""
Automated Model Retraining Pipeline
Provides scheduled retraining, performance monitoring, and automatic model updates
"""

import os
import json
import time
import schedule
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

# Try to import scheduling libraries
try:
    import crontab
    CRONTAB_AVAILABLE = True
except ImportError:
    CRONTAB_AVAILABLE = False
    print("Warning: python-crontab not available. Install with: pip install python-crontab")

try:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

# Local imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    from ..core.models import UnifiedStockModels, DataProcessor
    from ..core.data_collector import StockDataCollector
    from ..ai.feature_engineering import IntelligentFeatureEngine
    from .distributed_compute import create_distributed_training_pipeline
    MODEL_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Model imports not available: {e}")
    MODEL_IMPORTS_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RetrainingConfig:
    """Configuration for automated retraining"""
    # Symbols to monitor and retrain
    symbols: List[str]
    
    # Model configuration
    model_type: str = "ensemble"
    feature_level: str = "intelligent"
    epochs: int = 50
    batch_size: int = 32
    
    # Retraining triggers
    schedule_cron: str = "0 2 * * 0"  # Weekly at 2 AM on Sunday
    performance_threshold: float = 0.15  # RMSE threshold for retraining
    data_freshness_days: int = 30  # Retrain if no new data for X days
    
    # Performance monitoring
    performance_window_days: int = 7  # Days to evaluate performance
    min_samples_for_evaluation: int = 10
    
    # Distributed training
    use_distributed_training: bool = True
    max_concurrent_retrains: int = 4
    
    # Notifications
    email_notifications: bool = False
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_recipients: List[str] = None
    
    # Storage
    model_backup_days: int = 30  # Keep backups for X days
    log_retention_days: int = 90
    
    def __post_init__(self):
        if self.email_recipients is None:
            self.email_recipients = []

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    symbol: str
    model_type: str
    timestamp: datetime
    rmse: float
    mae: float
    directional_accuracy: float
    prediction_count: int
    data_age_days: int
    last_training_date: Optional[datetime] = None
    needs_retraining: bool = False
    retraining_reason: str = ""

class PerformanceMonitor:
    """Monitors model performance and triggers retraining"""
    
    def __init__(self, config: RetrainingConfig):
        self.config = config
        self.performance_history = []
        self.model_cache = {}
    
    def evaluate_model_performance(self, symbol: str, model_type: str) -> ModelPerformance:
        """Evaluate current model performance"""
        try:
            if not MODEL_IMPORTS_AVAILABLE:
                raise RuntimeError("Model imports not available")
            
            # Load model
            model_key = f"{symbol}_{model_type}"
            if model_key not in self.model_cache:
                models = UnifiedStockModels()
                model_path = f"models/saved/{symbol}"
                
                if os.path.exists(model_path):
                    models.load_models(model_path)
                    self.model_cache[model_key] = models
                    
                    # Get last training date
                    info_file = os.path.join(model_path, 'training_info.json')
                    last_training_date = None
                    if os.path.exists(info_file):
                        with open(info_file, 'r') as f:
                            info = json.load(f)
                            last_training_date = datetime.fromisoformat(
                                info.get('last_training_date', datetime.now().isoformat())
                            )
                else:
                    # No model exists, needs training
                    return ModelPerformance(
                        symbol=symbol,
                        model_type=model_type,
                        timestamp=datetime.now(),
                        rmse=float('inf'),
                        mae=float('inf'),
                        directional_accuracy=0.0,
                        prediction_count=0,
                        data_age_days=0,
                        last_training_date=None,
                        needs_retraining=True,
                        retraining_reason="No trained model found"
                    )
            
            models = self.model_cache[model_key]
            
            # Get recent data
            collector = StockDataCollector()
            df = collector.get_stock_data(symbol, period='3mo')  # 3 months for evaluation
            
            if df is None or len(df) < self.config.min_samples_for_evaluation:
                return ModelPerformance(
                    symbol=symbol,
                    model_type=model_type,
                    timestamp=datetime.now(),
                    rmse=float('inf'),
                    mae=float('inf'),
                    directional_accuracy=0.0,
                    prediction_count=0,
                    data_age_days=float('inf'),
                    needs_retraining=True,
                    retraining_reason="Insufficient data for evaluation"
                )
            
            # Check data freshness
            latest_data_date = df.index[-1]
            data_age_days = (datetime.now() - latest_data_date).days
            
            # Process data for evaluation
            processor = DataProcessor(feature_level=self.config.feature_level)
            
            if self.config.feature_level != 'standard':
                feature_engine = IntelligentFeatureEngine()
                enhanced_df = feature_engine.create_comprehensive_features(df)
                X, y = processor.create_sequences(enhanced_df)
            else:
                X, y = processor.create_sequences(df)
            
            if len(X) < self.config.min_samples_for_evaluation:
                return ModelPerformance(
                    symbol=symbol,
                    model_type=model_type,
                    timestamp=datetime.now(),
                    rmse=float('inf'),
                    mae=float('inf'),
                    directional_accuracy=0.0,
                    prediction_count=0,
                    data_age_days=data_age_days,
                    needs_retraining=True,
                    retraining_reason="Insufficient processed data"
                )
            
            # Use recent data for evaluation
            eval_size = min(self.config.performance_window_days, len(X) // 2)
            X_eval = X[-eval_size:]
            y_eval = y[-eval_size:]
            
            # Make predictions
            predictions = models.predict(model_type, X_eval)
            
            # Calculate metrics
            mse = np.mean((y_eval - predictions) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_eval - predictions))
            
            # Directional accuracy
            directional_accuracy = 0.0
            if len(y_eval) > 1:
                actual_direction = np.diff(y_eval) > 0
                pred_direction = np.diff(predictions) > 0
                directional_accuracy = float(np.mean(actual_direction == pred_direction))
            
            # Determine if retraining is needed
            needs_retraining = False
            retraining_reason = ""
            
            if rmse > self.config.performance_threshold:
                needs_retraining = True
                retraining_reason = f"RMSE ({rmse:.4f}) exceeds threshold ({self.config.performance_threshold})"
            elif data_age_days > self.config.data_freshness_days:
                needs_retraining = True
                retraining_reason = f"Data is {data_age_days} days old (threshold: {self.config.data_freshness_days})"
            elif last_training_date and (datetime.now() - last_training_date).days > 30:
                needs_retraining = True
                retraining_reason = f"Model hasn't been retrained for {(datetime.now() - last_training_date).days} days"
            
            return ModelPerformance(
                symbol=symbol,
                model_type=model_type,
                timestamp=datetime.now(),
                rmse=rmse,
                mae=mae,
                directional_accuracy=directional_accuracy,
                prediction_count=len(predictions),
                data_age_days=data_age_days,
                last_training_date=last_training_date,
                needs_retraining=needs_retraining,
                retraining_reason=retraining_reason
            )
            
        except Exception as e:
            logger.error(f"Performance evaluation failed for {symbol}: {e}")
            return ModelPerformance(
                symbol=symbol,
                model_type=model_type,
                timestamp=datetime.now(),
                rmse=float('inf'),
                mae=float('inf'),
                directional_accuracy=0.0,
                prediction_count=0,
                data_age_days=float('inf'),
                needs_retraining=True,
                retraining_reason=f"Evaluation error: {str(e)}"
            )
    
    def evaluate_all_models(self) -> List[ModelPerformance]:
        """Evaluate all configured models"""
        performances = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self.evaluate_model_performance, symbol, self.config.model_type): symbol
                for symbol in self.config.symbols
            }
            
            for future in futures:
                try:
                    performance = future.result()
                    performances.append(performance)
                    self.performance_history.append(performance)
                except Exception as e:
                    symbol = futures[future]
                    logger.error(f"Failed to evaluate {symbol}: {e}")
                    performances.append(ModelPerformance(
                        symbol=symbol,
                        model_type=self.config.model_type,
                        timestamp=datetime.now(),
                        rmse=float('inf'),
                        mae=float('inf'),
                        directional_accuracy=0.0,
                        prediction_count=0,
                        data_age_days=float('inf'),
                        needs_retraining=True,
                        retraining_reason=f"Evaluation failed: {str(e)}"
                    ))
        
        return performances

class RetrainingExecutor:
    """Executes model retraining tasks"""
    
    def __init__(self, config: RetrainingConfig):
        self.config = config
        self.active_retrainings = set()
        
    def retrain_model(self, symbol: str, backup_existing: bool = True) -> Dict[str, Any]:
        """Retrain a single model"""
        if symbol in self.active_retrainings:
            return {
                'symbol': symbol,
                'status': 'skipped',
                'reason': 'Already retraining'
            }
        
        self.active_retrainings.add(symbol)
        
        try:
            logger.info(f"Starting retraining for {symbol}")
            
            # Backup existing model
            if backup_existing:
                self._backup_model(symbol)
            
            # Run training
            if self.config.use_distributed_training:
                result = self._retrain_distributed([symbol])
                training_result = result['training']
            else:
                training_result = self._retrain_single(symbol)
            
            # Update training info
            self._update_training_info(symbol, training_result)
            
            # Log result
            if training_result.get('status') == 'success':
                logger.info(f"✅ Retraining completed for {symbol}: RMSE={training_result.get('rmse', 0):.4f}")
            else:
                logger.error(f"❌ Retraining failed for {symbol}: {training_result.get('error', 'Unknown error')}")
            
            return training_result
            
        except Exception as e:
            logger.error(f"Retraining error for {symbol}: {e}")
            return {
                'symbol': symbol,
                'status': 'failed',
                'error': str(e)
            }
        finally:
            self.active_retrainings.discard(symbol)
    
    def retrain_batch(self, symbols: List[str]) -> Dict[str, Any]:
        """Retrain multiple models"""
        if len(symbols) == 0:
            return {'status': 'success', 'results': []}
        
        logger.info(f"Starting batch retraining for {len(symbols)} symbols")
        
        if self.config.use_distributed_training and len(symbols) > 1:
            return self._retrain_distributed(symbols)
        else:
            # Sequential retraining with concurrency limit
            results = []
            with ThreadPoolExecutor(max_workers=self.config.max_concurrent_retrains) as executor:
                futures = {
                    executor.submit(self.retrain_model, symbol): symbol
                    for symbol in symbols
                }
                
                for future in futures:
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        symbol = futures[future]
                        logger.error(f"Batch retraining failed for {symbol}: {e}")
                        results.append({
                            'symbol': symbol,
                            'status': 'failed',
                            'error': str(e)
                        })
            
            successful = len([r for r in results if r.get('status') == 'success'])
            return {
                'status': 'completed',
                'total': len(symbols),
                'successful': successful,
                'failed': len(symbols) - successful,
                'results': results
            }
    
    def _retrain_distributed(self, symbols: List[str]) -> Dict[str, Any]:
        """Retrain using distributed training"""
        try:
            result = create_distributed_training_pipeline(
                symbols=symbols,
                model_type=self.config.model_type,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                feature_level=self.config.feature_level
            )
            
            # Update training info for all symbols
            for symbol in symbols:
                symbol_result = next((r for r in result['training']['results'] 
                                    if r.get('symbol') == symbol), None)
                if symbol_result:
                    self._update_training_info(symbol, symbol_result)
            
            return result
            
        except Exception as e:
            logger.error(f"Distributed retraining failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'training': {
                    'total_symbols': len(symbols),
                    'successful': 0,
                    'failed': len(symbols),
                    'results': [
                        {'symbol': s, 'status': 'failed', 'error': str(e)}
                        for s in symbols
                    ]
                }
            }
    
    def _retrain_single(self, symbol: str) -> Dict[str, Any]:
        """Retrain single model sequentially"""
        try:
            if not MODEL_IMPORTS_AVAILABLE:
                return {'symbol': symbol, 'status': 'failed', 'error': 'Model imports not available'}
            
            # Initialize components
            models = UnifiedStockModels()
            collector = StockDataCollector()
            processor = DataProcessor(feature_level=self.config.feature_level)
            
            # Get data
            df = collector.get_stock_data(symbol, period='2y')
            if df is None or len(df) < 200:
                return {'symbol': symbol, 'status': 'failed', 'error': 'Insufficient data'}
            
            # Process features
            if self.config.feature_level != 'standard':
                feature_engine = IntelligentFeatureEngine()
                enhanced_df = feature_engine.create_comprehensive_features(df)
                X, y = processor.create_sequences(enhanced_df)
            else:
                X, y = processor.create_sequences(df)
            
            # Split data
            split_index = int(len(X) * 0.8)
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]
            
            # Train model
            start_time = time.time()
            history = models.train_model(
                self.config.model_type, X_train, y_train, X_test, y_test,
                symbol=symbol, epochs=self.config.epochs, batch_size=self.config.batch_size,
                validation_split=0.2, verbose=0
            )
            training_time = time.time() - start_time
            
            # Calculate performance
            predictions = models.predict(self.config.model_type, X_test)
            mse = np.mean((y_test - predictions) ** 2)
            mae = np.mean(np.abs(y_test - predictions))
            
            return {
                'symbol': symbol,
                'status': 'success',
                'rmse': float(np.sqrt(mse)),
                'mae': float(mae),
                'training_time': training_time,
                'epochs_completed': len(history.get('loss', [])),
                'data_points': len(df)
            }
            
        except Exception as e:
            return {
                'symbol': symbol,
                'status': 'failed',
                'error': str(e)
            }
    
    def _backup_model(self, symbol: str):
        """Backup existing model before retraining"""
        model_path = Path(f"models/saved/{symbol}")
        if not model_path.exists():
            return
        
        backup_dir = Path(f"models/backups/{symbol}")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = backup_dir / f"backup_{timestamp}"
        
        try:
            import shutil
            shutil.copytree(model_path, backup_path)
            logger.info(f"Model backup created: {backup_path}")
            
            # Clean old backups
            self._cleanup_old_backups(backup_dir)
            
        except Exception as e:
            logger.warning(f"Failed to backup model for {symbol}: {e}")
    
    def _cleanup_old_backups(self, backup_dir: Path):
        """Clean up old model backups"""
        cutoff_date = datetime.now() - timedelta(days=self.config.model_backup_days)
        
        for backup_path in backup_dir.iterdir():
            if backup_path.is_dir():
                try:
                    # Extract timestamp from backup folder name
                    timestamp_str = backup_path.name.replace('backup_', '')
                    backup_date = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    
                    if backup_date < cutoff_date:
                        import shutil
                        shutil.rmtree(backup_path)
                        logger.info(f"Removed old backup: {backup_path}")
                        
                except Exception as e:
                    logger.warning(f"Failed to process backup {backup_path}: {e}")
    
    def _update_training_info(self, symbol: str, training_result: Dict[str, Any]):
        """Update training information file"""
        model_dir = Path(f"models/saved/{symbol}")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        info_file = model_dir / 'training_info.json'
        
        info = {
            'last_training_date': datetime.now().isoformat(),
            'model_type': self.config.model_type,
            'feature_level': self.config.feature_level,
            'epochs': self.config.epochs,
            'batch_size': self.config.batch_size,
            'training_result': training_result
        }
        
        try:
            with open(info_file, 'w') as f:
                json.dump(info, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to update training info for {symbol}: {e}")

class NotificationManager:
    """Handles notifications for retraining events"""
    
    def __init__(self, config: RetrainingConfig):
        self.config = config
    
    def send_retraining_notification(self, performances: List[ModelPerformance], 
                                   retraining_results: Dict[str, Any]):
        """Send notification about retraining results"""
        if not self.config.email_notifications or not EMAIL_AVAILABLE:
            return
        
        try:
            subject = f"Stock AI Retraining Report - {datetime.now().strftime('%Y-%m-%d')}"
            
            # Create email content
            content = self._create_notification_content(performances, retraining_results)
            
            # Send email
            self._send_email(subject, content)
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    def _create_notification_content(self, performances: List[ModelPerformance], 
                                   results: Dict[str, Any]) -> str:
        """Create notification email content"""
        content = f"""
Stock AI Automated Retraining Report
====================================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Model Performance Summary:
-------------------------
"""
        
        needs_retraining = [p for p in performances if p.needs_retraining]
        
        content += f"Total Models Monitored: {len(performances)}\n"
        content += f"Models Requiring Retraining: {len(needs_retraining)}\n\n"
        
        if needs_retraining:
            content += "Models Flagged for Retraining:\n"
            for perf in needs_retraining:
                content += f"  - {perf.symbol}: {perf.retraining_reason} (RMSE: {perf.rmse:.4f})\n"
            content += "\n"
        
        # Retraining results
        if 'results' in results:
            successful = len([r for r in results['results'] if r.get('status') == 'success'])
            total = len(results['results'])
            
            content += f"Retraining Results:\n"
            content += f"Successfully Retrained: {successful}/{total}\n\n"
            
            for result in results['results']:
                status = "✅" if result.get('status') == 'success' else "❌"
                symbol = result.get('symbol', 'Unknown')
                rmse = result.get('rmse', 0)
                content += f"  {status} {symbol}: RMSE={rmse:.4f}\n"
        
        content += f"\n\nConfiguration:\n"
        content += f"Model Type: {self.config.model_type}\n"
        content += f"Feature Level: {self.config.feature_level}\n"
        content += f"Performance Threshold: {self.config.performance_threshold}\n"
        content += f"Data Freshness Threshold: {self.config.data_freshness_days} days\n"
        
        return content
    
    def _send_email(self, subject: str, content: str):
        """Send email notification"""
        if not self.config.email_recipients:
            return
        
        msg = MIMEMultipart()
        msg['From'] = self.config.email_username
        msg['Subject'] = subject
        
        msg.attach(MIMEText(content, 'plain'))
        
        server = smtplib.SMTP(self.config.email_smtp_server, self.config.email_smtp_port)
        server.starttls()
        server.login(self.config.email_username, self.config.email_password)
        
        for recipient in self.config.email_recipients:
            msg['To'] = recipient
            text = msg.as_string()
            server.sendmail(self.config.email_username, recipient, text)
        
        server.quit()
        logger.info(f"Email notification sent to {len(self.config.email_recipients)} recipients")

class AutomatedRetrainingPipeline:
    """Main automated retraining pipeline"""
    
    def __init__(self, config: RetrainingConfig):
        self.config = config
        self.monitor = PerformanceMonitor(config)
        self.executor = RetrainingExecutor(config)
        self.notifier = NotificationManager(config)
        
        self.is_running = False
        self.scheduler_thread = None
        
        # Setup logging
        self._setup_logging()
        
        # Schedule jobs
        self._setup_schedule()
    
    def _setup_logging(self):
        """Setup logging for retraining pipeline"""
        log_dir = Path('logs/retraining')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f'retraining_{datetime.now().strftime("%Y%m%d")}.log'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
    
    def _setup_schedule(self):
        """Setup automated retraining schedule"""
        # Parse cron schedule (simplified)
        cron_parts = self.config.schedule_cron.split()
        if len(cron_parts) >= 5:
            # For simplicity, we'll use a weekly schedule
            # In production, you'd want proper cron parsing
            schedule.every().sunday.at("02:00").do(self.run_retraining_cycle)
        
        # Add daily performance monitoring
        schedule.every().day.at("01:00").do(self.monitor_performance)
    
    def start(self):
        """Start the automated retraining pipeline"""
        if self.is_running:
            logger.warning("Pipeline is already running")
            return
        
        self.is_running = True
        
        logger.info("🚀 Starting Automated Retraining Pipeline")
        logger.info(f"Monitoring {len(self.config.symbols)} symbols")
        logger.info(f"Schedule: {self.config.schedule_cron}")
        
        # Start scheduler in separate thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        # Run initial performance check
        self.monitor_performance()
    
    def stop(self):
        """Stop the automated retraining pipeline"""
        self.is_running = False
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        logger.info("🛑 Automated Retraining Pipeline Stopped")
    
    def _run_scheduler(self):
        """Run the scheduler loop"""
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def monitor_performance(self):
        """Monitor model performance and trigger retraining if needed"""
        try:
            logger.info("🔍 Monitoring model performance...")
            
            performances = self.monitor.evaluate_all_models()
            
            # Log performance summary
            needs_retraining = [p for p in performances if p.needs_retraining]
            logger.info(f"Performance check completed: {len(needs_retraining)}/{len(performances)} models need retraining")
            
            # Trigger immediate retraining for critical cases
            critical_symbols = [
                p.symbol for p in needs_retraining 
                if p.rmse > self.config.performance_threshold * 2  # Very poor performance
            ]
            
            if critical_symbols:
                logger.warning(f"Critical performance detected for {critical_symbols}")
                self._trigger_immediate_retraining(critical_symbols)
            
            return performances
            
        except Exception as e:
            logger.error(f"Performance monitoring failed: {e}")
            return []
    
    def run_retraining_cycle(self):
        """Run complete retraining cycle"""
        try:
            logger.info("🔄 Starting scheduled retraining cycle")
            
            # Evaluate all models
            performances = self.monitor.evaluate_all_models()
            
            # Get symbols that need retraining
            symbols_to_retrain = [p.symbol for p in performances if p.needs_retraining]
            
            if not symbols_to_retrain:
                logger.info("✅ No models require retraining")
                return
            
            logger.info(f"Retraining {len(symbols_to_retrain)} models: {symbols_to_retrain}")
            
            # Execute retraining
            retraining_results = self.executor.retrain_batch(symbols_to_retrain)
            
            # Send notification
            self.notifier.send_retraining_notification(performances, retraining_results)
            
            # Log summary
            successful = retraining_results.get('successful', 0)
            total = retraining_results.get('total', 0)
            logger.info(f"✅ Retraining cycle completed: {successful}/{total} models successfully retrained")
            
        except Exception as e:
            logger.error(f"Retraining cycle failed: {e}")
    
    def _trigger_immediate_retraining(self, symbols: List[str]):
        """Trigger immediate retraining for critical performance issues"""
        logger.info(f"🚨 Triggering immediate retraining for: {symbols}")
        
        try:
            retraining_results = self.executor.retrain_batch(symbols)
            
            successful = retraining_results.get('successful', 0)
            logger.info(f"Immediate retraining completed: {successful}/{len(symbols)} models")
            
        except Exception as e:
            logger.error(f"Immediate retraining failed: {e}")
    
    def force_retrain_all(self):
        """Force retraining of all configured models"""
        logger.info("🔧 Force retraining all models")
        
        try:
            retraining_results = self.executor.retrain_batch(self.config.symbols)
            
            successful = retraining_results.get('successful', 0)
            total = len(self.config.symbols)
            
            logger.info(f"Force retraining completed: {successful}/{total} models")
            return retraining_results
            
        except Exception as e:
            logger.error(f"Force retraining failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        try:
            recent_performances = self.monitor.evaluate_all_models()
            needs_retraining = [p for p in recent_performances if p.needs_retraining]
            
            return {
                'is_running': self.is_running,
                'monitored_symbols': len(self.config.symbols),
                'symbols_needing_retraining': len(needs_retraining),
                'last_check': datetime.now().isoformat(),
                'config': asdict(self.config),
                'performance_summary': [
                    {
                        'symbol': p.symbol,
                        'rmse': p.rmse,
                        'needs_retraining': p.needs_retraining,
                        'reason': p.retraining_reason
                    }
                    for p in recent_performances
                ]
            }
            
        except Exception as e:
            return {
                'is_running': self.is_running,
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }

def create_default_config(symbols: List[str]) -> RetrainingConfig:
    """Create default retraining configuration"""
    return RetrainingConfig(
        symbols=symbols,
        model_type="ensemble",
        feature_level="intelligent",
        epochs=50,
        batch_size=32,
        schedule_cron="0 2 * * 0",  # Weekly on Sunday at 2 AM
        performance_threshold=0.15,
        data_freshness_days=30,
        performance_window_days=7,
        use_distributed_training=True,
        max_concurrent_retrains=4,
        email_notifications=False,
        model_backup_days=30,
        log_retention_days=90
    )