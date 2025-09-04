#!/usr/bin/env python3
"""
Stock AI Training Script
Unified training interface for all model types and complexity levels
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
import yfinance as yf
from typing import Dict, List, Tuple, Optional

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from core.models import UnifiedStockModels, DataProcessor
    from core.data_collector import StockDataCollector
    from ai.feature_engineering import IntelligentFeatureEngine
    from ai.hyperparameter_tuning import AdaptiveHyperparameterOptimizer
    from visualization.charts import create_prediction_chart, create_performance_dashboard
    ADVANCED_FEATURES = True
except ImportError as e:
    print(f"Import warning: {e}")
    print("Some advanced features may not be available. Using basic mode.")
    ADVANCED_FEATURES = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_training_history(history: dict, save_path: str = 'plots/training_history.png'):
    """
    Plot training history
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # MAE plot
    ax1.plot(history['mae'], label='Training MAE')
    ax1.plot(history['val_mae'], label='Validation MAE')
    ax2.set_title('Model MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_stock_model(symbol: str, 
                     sequence_length: int = 60,
                     epochs: int = 50,
                     batch_size: int = 32,
                     test_size: float = 0.2):
    """
    Complete training pipeline for a stock symbol
    """
    print(f"Training model for {symbol}")
    
    # Initialize components
    collector = StockDataCollector()
    preprocessor = StockDataPreprocessor(sequence_length=sequence_length)
    
    # Collect data
    print("Collecting stock data...")
    data = collector.get_stock_data_yfinance(symbol, period="5y")
    print(f"Collected {len(data)} days of data")
    
    # Preprocess data
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocessor.preprocess_for_training(
        data, test_size=test_size
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[2]}")
    
    # Initialize model
    model = StockLSTMModel(
        sequence_length=sequence_length,
        n_features=X_train.shape[2]
    )
    
    # Train model
    print("Training model...")
    model_path = f'models/{symbol}_best_model.h5'
    history = model.train(
        X_train, y_train,
        X_test, y_test,
        epochs=epochs,
        batch_size=batch_size,
        model_path=model_path
    )
    
    # Evaluate model
    print("Evaluating model...")
    metrics = model.evaluate(X_test, y_test)
    
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot training history
    plot_path = f'plots/{symbol}_training_history.png'
    plot_training_history(history, plot_path)
    print(f"Training plots saved to {plot_path}")
    
    # Save model
    final_model_path = f'models/{symbol}_final_model.h5'
    model.save_model(final_model_path)
    print(f"Model saved to {final_model_path}")
    
    return model, metrics

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Stock AI Training System')
    
    # Basic parameters
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol to train on')
    parser.add_argument('--model', type=str, default='lstm', 
                       choices=['lstm', 'gru', 'transformer', 'cnn_lstm', 'ensemble', 
                               'nextgen_ensemble', 'advanced_ensemble', 'legacy'],
                       help='Model type to train')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--years', type=int, default=5, help='Years of historical data')
    parser.add_argument('--sequence-length', type=int, default=60, help='Sequence length for training')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    
    # Advanced options
    parser.add_argument('--quick', action='store_true', help='Quick training mode (reduced parameters)')
    parser.add_argument('--advanced', action='store_true', help='Use advanced ensemble training with intelligent features')
    parser.add_argument('--optimize', action='store_true', help='Enable hyperparameter optimization')
    parser.add_argument('--features', type=str, default='standard', 
                       choices=['standard', 'advanced', 'intelligent', 'all'],
                       help='Feature engineering level')
    parser.add_argument('--regime-aware', action='store_true', help='Enable market regime-aware training')
    
    return parser.parse_args()

def train_unified_model(args):
    """Train model using unified models system"""
    logger.info(f"🚀 Training {args.model.upper()} model for {args.symbol}")
    
    try:
        # Initialize data collector
        collector = StockDataCollector()
        
        # Get data
        logger.info("📊 Collecting stock data...")
        df = collector.get_stock_data(args.symbol, period=f"{args.years}y")
        
        if df is None or len(df) < 100:
            logger.error("❌ Insufficient data collected")
            return False
        
        # Process data
        processor = DataProcessor(sequence_length=args.sequence_length)
        X, y = processor.create_sequences(df)
        
        # Split data
        train_size = int(len(X) * (1 - args.test_size - 0.1))
        val_size = int(len(X) * 0.1)
        
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
        X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
        
        # Initialize models
        models = UnifiedStockModels()
        
        # Train model
        logger.info(f"🔥 Training {args.model} model...")
        history = models.train_model(
            model_type=args.model,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            epochs=args.epochs,
            model_path=f"models/saved/{args.symbol}_{args.model}.h5"
        )
        
        # Evaluate
        metrics = models.evaluate_model(args.model, X_test, y_test)
        
        logger.info("📈 Training Results:")
        logger.info(f"   🎯 RMSE: {metrics['rmse']:.4f}")
        logger.info(f"   📊 MAE: {metrics['mae']:.4f}")
        logger.info(f"   🧭 Directional Accuracy: {metrics['directional_accuracy']:.2%}")
        
        # Save models
        models.save_models(f"models/saved/{args.symbol}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        return False

def train_advanced_ensemble(args):
    """Train advanced ensemble models with intelligent features"""
    if not ADVANCED_FEATURES:
        logger.warning("⚠️ Advanced features not available, falling back to basic training")
        return train_unified_model(args)
    
    logger.info(f"🚀 Advanced Ensemble Training for {args.symbol}")
    
    try:
        # Initialize components
        collector = StockDataCollector()
        feature_engine = IntelligentFeatureEngine()
        
        # Collect base data
        logger.info("📊 Collecting stock data...")
        df = collector.get_stock_data(args.symbol, period=f"{args.years}y")
        
        if df is None or len(df) < 200:
            logger.error("❌ Insufficient data for advanced training")
            return False
        
        # Create intelligent features
        logger.info("🧠 Creating intelligent features...")
        enhanced_df = feature_engine.create_comprehensive_features(df)
        
        # Optional: Feature selection for performance
        if len(enhanced_df.columns) > 100:
            logger.info("🎯 Performing intelligent feature selection...")
            target = enhanced_df['Close'].shift(-1).dropna()
            features = enhanced_df[:-1]
            top_features = feature_engine.intelligent_feature_selection(
                features, target, top_k=80
            )
            enhanced_df = enhanced_df[top_features + ['Close']]
        
        # Process data with enhanced features
        processor = DataProcessor(
            sequence_length=args.sequence_length, 
            feature_level=args.features
        )
        X, y = processor.create_sequences(enhanced_df)
        
        logger.info(f"📈 Dataset shape: {X.shape}")
        logger.info(f"🎛️ Features: {X.shape[2]}")
        
        # Split data
        train_size = int(len(X) * 0.7)
        val_size = int(len(X) * 0.15)
        
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
        X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
        
        # Initialize advanced models
        models = UnifiedStockModels(use_advanced=True)
        
        # Hyperparameter optimization (optional)
        if not args.quick and args.optimize:
            logger.info("🎛️ Optimizing hyperparameters...")
            optimizer = AdaptiveHyperparameterOptimizer(n_trials=20)
            
            # Market regime detection for adaptive optimization
            market_regime = detect_market_regime(df)
            
            def objective(trial):
                config = {
                    'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.5),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                    'sequence_length': args.sequence_length,
                    'n_features': X.shape[2]
                }
                return optimizer.market_aware_objective(
                    trial, models, X_train, y_train, X_val, y_val, market_regime
                )
            
            study = optimizer.multi_objective_optimization(
                models, X_train, y_train, X_val, y_val
            )
            best_config = study.best_params
            logger.info(f"🏆 Best config: {best_config}")
        else:
            best_config = None
        
        # Train advanced ensemble
        logger.info(f"🔥 Training {args.model} with advanced techniques...")
        
        if args.model in ['nextgen_ensemble', 'advanced_ensemble']:
            history = models.train_advanced_ensemble(
                model_type=args.model,
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                epochs=args.epochs,
                config=best_config
            )
        else:
            # Use standard training with advanced features
            history = models.train_model(
                model_type=args.model,
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                epochs=args.epochs,
                model_path=f"models/saved/{args.symbol}_{args.model}_advanced.h5"
            )
        
        # Evaluate with uncertainty quantification
        if args.model in ['nextgen_ensemble', 'advanced_ensemble']:
            predictions, uncertainty = models.predict_with_uncertainty(args.model, X_test)
            
            # Calculate enhanced metrics
            metrics = calculate_advanced_metrics(y_test, predictions, uncertainty)
            
            logger.info("🎯 Advanced Training Results:")
            logger.info(f"   📊 RMSE: {metrics['rmse']:.4f}")
            logger.info(f"   📈 MAE: {metrics['mae']:.4f}")
            logger.info(f"   🧭 Directional Accuracy: {metrics['directional_accuracy']:.2%}")
            logger.info(f"   🎲 Mean Uncertainty: {metrics['mean_uncertainty']:.4f}")
            logger.info(f"   📊 Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}")
            
        else:
            metrics = models.evaluate_model(args.model, X_test, y_test)
            logger.info("📈 Training Results:")
            logger.info(f"   🎯 RMSE: {metrics['rmse']:.4f}")
            logger.info(f"   📊 MAE: {metrics['mae']:.4f}")
            logger.info(f"   🧭 Directional Accuracy: {metrics['directional_accuracy']:.2%}")
        
        # Save models and results
        models.save_models(f"models/saved/{args.symbol}_advanced")
        
        # Plot advanced results
        if history:
            plot_advanced_training_history(history, f"plots/{args.symbol}_{args.model}_advanced.png")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Advanced training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def detect_market_regime(df: pd.DataFrame) -> str:
    """Simple market regime detection"""
    returns = df['Close'].pct_change().dropna()
    volatility = returns.rolling(20).std().iloc[-1]
    vol_percentile = returns.rolling(252).std().quantile(0.75)
    
    if volatility > vol_percentile:
        return 'volatile'
    
    trend = df['Close'].iloc[-20:].pct_change().mean()
    if trend > 0.001:
        return 'trending'
    
    return 'normal'

def calculate_advanced_metrics(y_true, y_pred, uncertainty=None):
    """Calculate advanced performance metrics"""
    metrics = {}
    
    # Basic metrics
    mse = np.mean((y_true - y_pred) ** 2)
    metrics['rmse'] = np.sqrt(mse)
    metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    
    # Directional accuracy
    if len(y_true) > 1:
        actual_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        metrics['directional_accuracy'] = np.mean(actual_direction == pred_direction)
    else:
        metrics['directional_accuracy'] = 0
    
    # Uncertainty metrics
    if uncertainty is not None:
        metrics['mean_uncertainty'] = np.mean(uncertainty)
        metrics['uncertainty_correlation'] = np.corrcoef(
            np.abs(y_true - y_pred), 
            uncertainty.flatten()
        )[0, 1] if len(uncertainty.shape) > 1 else np.corrcoef(
            np.abs(y_true - y_pred), uncertainty
        )[0, 1]
    
    # Sharpe ratio approximation
    returns = np.diff(y_pred) / y_pred[:-1]
    if len(returns) > 1 and np.std(returns) > 0:
        metrics['sharpe_ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252)
    
    return metrics

def plot_advanced_training_history(history, save_path):
    """Plot advanced training history with multiple metrics"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    if 'loss' in history:
        axes[0,0].plot(history['loss'], label='Training Loss')
        axes[0,0].plot(history['val_loss'], label='Validation Loss')
        axes[0,0].set_title('Model Loss')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
    
    # MAE plot
    if 'mae' in history or 'main_prediction_mae' in history:
        mae_key = 'mae' if 'mae' in history else 'main_prediction_mae'
        val_mae_key = 'val_mae' if 'val_mae' in history else 'val_main_prediction_mae'
        
        axes[0,1].plot(history[mae_key], label='Training MAE')
        axes[0,1].plot(history[val_mae_key], label='Validation MAE')
        axes[0,1].set_title('Model MAE')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('MAE')
        axes[0,1].legend()
    
    # Learning rate plot (if available)
    if 'lr' in history:
        axes[1,0].plot(history['lr'], label='Learning Rate')
        axes[1,0].set_title('Learning Rate Schedule')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Learning Rate')
        axes[1,0].set_yscale('log')
        axes[1,0].legend()
    
    # Additional metrics
    if 'uncertainty_loss' in history:
        axes[1,1].plot(history['uncertainty_loss'], label='Uncertainty Loss')
        axes[1,1].plot(history['val_uncertainty_loss'], label='Val Uncertainty Loss')
        axes[1,1].set_title('Uncertainty Estimation')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Uncertainty Loss')
        axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main training function"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    args = parse_arguments()
    
    logger.info(f"🚀 Starting Stock AI Training")
    logger.info(f"   📊 Symbol: {args.symbol}")
    logger.info(f"   🤖 Model: {args.model}")
    logger.info(f"   ⏱️  Years: {args.years}")
    logger.info(f"   🔄 Epochs: {args.epochs}")
    
    # Create necessary directories
    os.makedirs("models/saved", exist_ok=True)
    os.makedirs("data/cache", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    success = False
    
    try:
        if args.model == 'legacy':
            # Use legacy training for backward compatibility
            model, metrics = train_stock_model(
                symbol=args.symbol,
                sequence_length=args.sequence_length,
                epochs=args.epochs,
                batch_size=args.batch_size,
                test_size=args.test_size
            )
            success = True
        elif args.advanced or args.model in ['nextgen_ensemble', 'advanced_ensemble'] or args.features != 'standard':
            # Use advanced training pipeline
            logger.info("🧠 Using advanced training pipeline...")
            success = train_advanced_ensemble(args)
        else:
            # Use unified models system
            success = train_unified_model(args)
        
        if success:
            logger.info("✅ Training completed successfully!")
        else:
            logger.error("❌ Training failed!")
            
    except KeyboardInterrupt:
        logger.info("⏹️ Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        # Try legacy training as fallback
        try:
            logger.info("🔄 Attempting legacy training as fallback...")
            model, metrics = train_stock_model(
                symbol=args.symbol,
                sequence_length=args.sequence_length,
                epochs=args.epochs,
                batch_size=args.batch_size,
                test_size=args.test_size
            )
            success = True
            logger.info("✅ Legacy training completed successfully!")
        except Exception as fallback_error:
            logger.error(f"❌ Fallback training also failed: {str(fallback_error)}")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())