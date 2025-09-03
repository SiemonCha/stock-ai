#!/usr/bin/env python3
"""
Stock AI Analysis Script
Unified analysis interface for predictions and performance evaluation
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from models.unified_models import UnifiedStockModels, DataProcessor
    from data.stock_data import StockDataCollector
    from visualization.plotter import create_prediction_plot, create_performance_chart
except ImportError as e:
    print(f"Import warning: {e}")
    print("Some advanced features may not be available.")
    try:
        from data.collector import StockDataCollector
        from data.preprocessor import StockDataPreprocessor
        from models.lstm_model import StockLSTMModel
    except ImportError:
        print("Legacy imports also failed. Please check your environment.")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Stock AI Analysis System')
    
    # Basic parameters
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol to analyze')
    parser.add_argument('--model', type=str, default='ensemble', 
                       choices=['lstm', 'gru', 'transformer', 'cnn_lstm', 'ensemble'],
                       help='Model type to use for analysis')
    
    # Analysis options
    parser.add_argument('--days', type=int, default=30, help='Number of days to predict')
    parser.add_argument('--confidence', action='store_true', help='Include confidence intervals')
    parser.add_argument('--detailed', action='store_true', help='Detailed analysis report')
    parser.add_argument('--save-plots', action='store_true', help='Save plots to files')
    
    # Data options
    parser.add_argument('--years', type=int, default=1, help='Years of recent data for context')
    
    return parser.parse_args()

def load_trained_models(symbol: str, model_type: str):
    """Load trained models for analysis"""
    try:
        models = UnifiedStockModels()
        model_path = f"models/saved/{symbol}"
        
        if os.path.exists(model_path):
            models.load_models(model_path)
            return models
        else:
            logger.warning(f"No trained models found at {model_path}")
            return None
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        return None

def analyze_stock_performance(symbol: str, model_type: str, days: int = 30, years: int = 1):
    """Analyze stock performance and make predictions"""
    logger.info(f"📊 Analyzing {symbol} with {model_type} model")
    
    try:
        # Load trained models
        models = load_trained_models(symbol, model_type)
        if models is None:
            logger.error("❌ No trained models available. Please train models first.")
            return None
        
        # Get recent data
        collector = StockDataCollector()
        df = collector.get_stock_data(symbol, period=f"{years}y")
        
        if df is None or len(df) < 100:
            logger.error("❌ Insufficient data for analysis")
            return None
        
        # Process data for prediction
        processor = DataProcessor()
        X, y = processor.create_sequences(df)
        
        # Use recent data for prediction
        recent_X = X[-days:]
        recent_y = y[-days:]
        
        # Make predictions
        predictions = models.predict(model_type, recent_X)
        
        # Calculate metrics
        metrics = models.evaluate_model(model_type, recent_X, recent_y)
        
        # Prepare results
        results = {
            'symbol': symbol,
            'model_type': model_type,
            'predictions': predictions,
            'actual': recent_y,
            'metrics': metrics,
            'dates': df.index[-days:],
            'recent_data': df.tail(days)
        }
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        return None

def generate_prediction_report(results: dict, detailed: bool = False):
    """Generate analysis report"""
    if results is None:
        return
    
    symbol = results['symbol']
    model_type = results['model_type']
    metrics = results['metrics']
    predictions = results['predictions']
    actual = results['actual']
    
    logger.info("📈 ANALYSIS REPORT")
    logger.info(f"   📊 Symbol: {symbol}")
    logger.info(f"   🤖 Model: {model_type.upper()}")
    logger.info(f"   📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    logger.info("🎯 PERFORMANCE METRICS:")
    logger.info(f"   📊 RMSE: {metrics['rmse']:.4f}")
    logger.info(f"   📈 MAE: {metrics['mae']:.4f}")
    logger.info(f"   🧭 Directional Accuracy: {metrics['directional_accuracy']:.2%}")
    
    if detailed:
        logger.info("📊 DETAILED STATISTICS:")
        logger.info(f"   📈 Mean Prediction: {np.mean(predictions):.2f}")
        logger.info(f"   📊 Std Prediction: {np.std(predictions):.2f}")
        logger.info(f"   🎯 Min Prediction: {np.min(predictions):.2f}")
        logger.info(f"   📈 Max Prediction: {np.max(predictions):.2f}")
        
        # Recent trend analysis
        if len(predictions) > 1:
            trend = "📈 Upward" if predictions[-1] > predictions[0] else "📉 Downward"
            logger.info(f"   🔄 Recent Trend: {trend}")
        
        # Volatility analysis
        volatility = np.std(predictions) / np.mean(predictions) * 100
        volatility_level = "High" if volatility > 5 else "Medium" if volatility > 2 else "Low"
        logger.info(f"   📊 Volatility: {volatility:.2f}% ({volatility_level})")

def create_analysis_plots(results: dict, save_plots: bool = False):
    """Create visualization plots"""
    if results is None:
        return
    
    try:
        import matplotlib.pyplot as plt
        
        symbol = results['symbol']
        predictions = results['predictions']
        actual = results['actual']
        dates = results['dates']
        
        # Create prediction vs actual plot
        plt.figure(figsize=(12, 6))
        plt.plot(dates, actual, label='Actual', color='blue', linewidth=2)
        plt.plot(dates, predictions, label='Predicted', color='red', linewidth=2, linestyle='--')
        plt.title(f'{symbol} - Actual vs Predicted Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_plots:
            plot_path = f'plots/{symbol}_analysis.png'
            os.makedirs('plots', exist_ok=True)
            plt.savefig(plot_path)
            logger.info(f"📊 Plot saved to {plot_path}")
        else:
            plt.show()
        
        plt.close()
        
        # Create error distribution plot
        errors = actual - predictions
        
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(f'{symbol} - Prediction Error Distribution')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        if save_plots:
            error_plot_path = f'plots/{symbol}_error_distribution.png'
            plt.savefig(error_plot_path)
            logger.info(f"📊 Error plot saved to {error_plot_path}")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        logger.warning(f"⚠️ Could not create plots: {str(e)}")

def main():
    """Main analysis function"""
    args = parse_arguments()
    
    logger.info(f"🔍 Starting Stock AI Analysis")
    logger.info(f"   📊 Symbol: {args.symbol}")
    logger.info(f"   🤖 Model: {args.model}")
    logger.info(f"   📅 Days: {args.days}")
    
    try:
        # Analyze stock performance
        results = analyze_stock_performance(
            symbol=args.symbol,
            model_type=args.model,
            days=args.days,
            years=args.years
        )
        
        if results:
            # Generate report
            generate_prediction_report(results, detailed=args.detailed)
            
            # Create plots
            create_analysis_plots(results, save_plots=args.save_plots)
            
            logger.info("✅ Analysis completed successfully!")
        else:
            logger.error("❌ Analysis failed!")
            return 1
            
    except KeyboardInterrupt:
        logger.info("⏹️ Analysis interrupted by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())