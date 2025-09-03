#!/usr/bin/env python3
"""
Stock AI Training Script
Unified training interface for all model types and complexity levels
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from models.unified_models import UnifiedStockModels, DataProcessor
    from data.stock_data import StockDataCollector
    # Fallback imports for backward compatibility
    from data.collector import StockDataCollector as LegacyCollector
    from data.preprocessor import StockDataPreprocessor
    from models.lstm_model import StockLSTMModel
except ImportError as e:
    print(f"Import warning: {e}")
    print("Some advanced features may not be available.")

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
                       choices=['lstm', 'gru', 'transformer', 'cnn_lstm', 'ensemble', 'legacy'],
                       help='Model type to train')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--years', type=int, default=5, help='Years of historical data')
    parser.add_argument('--sequence-length', type=int, default=60, help='Sequence length for training')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    
    # Advanced options
    parser.add_argument('--quick', action='store_true', help='Quick training mode (reduced parameters)')
    
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