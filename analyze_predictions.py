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
    from core.models import UnifiedStockModels, DataProcessor
    from core.data_collector import StockDataCollector
    from ai.feature_engineering import IntelligentFeatureEngine
    from visualization.charts import create_prediction_chart, create_performance_dashboard
    ADVANCED_FEATURES = True
except ImportError as e:
    print(f"Import warning: {e}")
    print("Some advanced features may not be available.")
    ADVANCED_FEATURES = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Stock AI Analysis System')
    
    # Basic parameters
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol to analyze')
    parser.add_argument('--model', type=str, default='ensemble', 
                       choices=['lstm', 'gru', 'transformer', 'cnn_lstm', 'ensemble', 
                               'nextgen_ensemble', 'advanced_ensemble'],
                       help='Model type to use for analysis')
    
    # Analysis options
    parser.add_argument('--days', type=int, default=30, help='Number of days to predict')
    parser.add_argument('--confidence', action='store_true', help='Include confidence intervals')
    parser.add_argument('--detailed', action='store_true', help='Detailed analysis report')
    parser.add_argument('--save-plots', action='store_true', help='Save plots to files')
    parser.add_argument('--uncertainty', action='store_true', help='Show prediction uncertainty')
    parser.add_argument('--regime-analysis', action='store_true', help='Include market regime analysis')
    
    # Data options
    parser.add_argument('--years', type=int, default=1, help='Years of recent data for context')
    parser.add_argument('--features', type=str, default='standard', 
                       choices=['standard', 'advanced', 'intelligent', 'all'],
                       help='Feature engineering level for analysis')
    
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

def analyze_with_uncertainty(models, X_test, y_test, model_type):
    """Advanced analysis with uncertainty quantification"""
    results = {}
    
    if model_type in ['advanced_ensemble', 'nextgen_ensemble']:
        try:
            # Get predictions with uncertainty
            predictions, uncertainty = models.predict_with_uncertainty(model_type, X_test)
            
            # Calculate comprehensive metrics
            results['predictions'] = predictions
            results['uncertainty'] = uncertainty
            results['confidence_intervals'] = calculate_confidence_intervals(predictions, uncertainty)
            
            # Basic metrics
            mse = np.mean((y_test - predictions) ** 2)
            results['rmse'] = np.sqrt(mse)
            results['mae'] = np.mean(np.abs(y_test - predictions))
            
            # Directional accuracy
            if len(y_test) > 1:
                actual_direction = np.diff(y_test) > 0
                pred_direction = np.diff(predictions) > 0
                results['directional_accuracy'] = np.mean(actual_direction == pred_direction)
            
            # Uncertainty metrics
            results['mean_uncertainty'] = np.mean(uncertainty)
            results['uncertainty_calibration'] = calculate_uncertainty_calibration(y_test, predictions, uncertainty)
            
            # Risk metrics
            results['sharpe_ratio'] = calculate_sharpe_ratio(predictions)
            results['max_drawdown'] = calculate_max_drawdown(predictions)
            
            return results
            
        except Exception as e:
            logger.warning(f"Advanced analysis failed: {e}, falling back to standard analysis")
    
    # Standard analysis
    predictions = models.predict(model_type, X_test)
    mse = np.mean((y_test - predictions) ** 2)
    
    results['predictions'] = predictions
    results['uncertainty'] = np.zeros_like(predictions)
    results['rmse'] = np.sqrt(mse)
    results['mae'] = np.mean(np.abs(y_test - predictions))
    
    if len(y_test) > 1:
        actual_direction = np.diff(y_test) > 0
        pred_direction = np.diff(predictions) > 0
        results['directional_accuracy'] = np.mean(actual_direction == pred_direction)
    
    return results

def calculate_confidence_intervals(predictions, uncertainty, confidence_level=0.95):
    """Calculate confidence intervals from uncertainty estimates"""
    z_score = 1.96 if confidence_level == 0.95 else 2.58
    
    if len(uncertainty.shape) > 1:
        uncertainty = uncertainty.flatten()
    if len(predictions.shape) > 1:
        predictions = predictions.flatten()
    
    lower_bound = predictions - z_score * np.sqrt(uncertainty)
    upper_bound = predictions + z_score * np.sqrt(uncertainty)
    
    return {'lower': lower_bound, 'upper': upper_bound}

def calculate_uncertainty_calibration(y_true, y_pred, uncertainty):
    """Calculate how well uncertainty estimates are calibrated"""
    try:
        residuals = np.abs(y_true - y_pred.flatten())
        uncertainty_flat = uncertainty.flatten() if len(uncertainty.shape) > 1 else uncertainty
        
        correlation = np.corrcoef(residuals, uncertainty_flat)[0, 1]
        return correlation
    except:
        return 0.0

def calculate_sharpe_ratio(predictions):
    """Calculate approximate Sharpe ratio from predictions"""
    try:
        returns = np.diff(predictions) / predictions[:-1]
        if len(returns) > 1 and np.std(returns) > 0:
            return np.mean(returns) / np.std(returns) * np.sqrt(252)
        return 0.0
    except:
        return 0.0

def calculate_max_drawdown(predictions):
    """Calculate maximum drawdown from prediction series"""
    try:
        cumulative = np.cumprod(1 + np.diff(predictions) / predictions[:-1])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    except:
        return 0.0

def create_advanced_visualizations(symbol, results, args):
    """Create advanced visualization plots"""
    if not args.save_plots:
        return
    
    os.makedirs('plots', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Predictions with uncertainty
    if 'predictions' in results:
        axes[0,0].plot(results['predictions'], label='Predictions', color='blue')
        
        if 'confidence_intervals' in results and args.uncertainty:
            ci = results['confidence_intervals']
            axes[0,0].fill_between(
                range(len(results['predictions'])),
                ci['lower'], ci['upper'],
                alpha=0.3, color='blue', label='95% Confidence Interval'
            )
        
        axes[0,0].set_title(f'{symbol} Price Predictions')
        axes[0,0].set_xlabel('Time')
        axes[0,0].set_ylabel('Price')
        axes[0,0].legend()
    
    # Plot 2: Uncertainty over time
    if 'uncertainty' in results and args.uncertainty:
        axes[0,1].plot(results['uncertainty'], color='red')
        axes[0,1].set_title('Prediction Uncertainty')
        axes[0,1].set_xlabel('Time')
        axes[0,1].set_ylabel('Uncertainty')
    
    # Plot 3: Performance metrics
    if 'directional_accuracy' in results:
        metrics = ['RMSE', 'MAE', 'Dir. Acc.', 'Sharpe']
        values = [
            results.get('rmse', 0),
            results.get('mae', 0),
            results.get('directional_accuracy', 0),
            results.get('sharpe_ratio', 0)
        ]
        
        axes[1,0].bar(metrics, values, color=['red', 'orange', 'green', 'blue'])
        axes[1,0].set_title('Performance Metrics')
        axes[1,0].set_ylabel('Value')
    
    # Plot 4: Risk analysis
    if 'max_drawdown' in results:
        returns = np.diff(results['predictions']) / results['predictions'][:-1]
        axes[1,1].hist(returns, bins=50, alpha=0.7, color='purple')
        axes[1,1].axvline(x=np.mean(returns), color='red', linestyle='--', label='Mean Return')
        axes[1,1].set_title('Return Distribution')
        axes[1,1].set_xlabel('Returns')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(f'plots/{symbol}_{args.model}_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📊 Advanced visualizations saved to plots/{symbol}_{args.model}_analysis.png")

def analyze_stock_performance(args):
    """Enhanced stock performance analysis with advanced features"""
    logger.info(f"📊 Advanced Analysis: {args.symbol} with {args.model} model")
    
    try:
        # Load trained models
        models = load_trained_models(args.symbol, args.model)
        if models is None:
            logger.error("❌ No trained models available. Please train models first.")
            return None
        
        # Get recent data
        collector = StockDataCollector()
        df = collector.get_stock_data(args.symbol, period=f"{args.years}y")
        
        if df is None or len(df) < 100:
            logger.error("❌ Insufficient data for analysis")
            return None
        
        # Enhanced data processing with intelligent features
        feature_level = args.features if ADVANCED_FEATURES else 'standard'
        processor = DataProcessor(feature_level=feature_level)
        
        # Process data with appropriate feature level
        if feature_level != 'standard' and ADVANCED_FEATURES:
            logger.info("🧠 Applying intelligent feature engineering...")
            feature_engine = IntelligentFeatureEngine()
            enhanced_df = feature_engine.create_comprehensive_features(df)
            X, y = processor.create_sequences(enhanced_df)
        else:
            X, y = processor.create_sequences(df)
        
        logger.info(f"📊 Data shape: {X.shape}, Features: {X.shape[2]}")
        
        # Use recent data for analysis
        test_size = min(args.days, len(X) // 4)  # Use up to 25% of data for testing
        X_test = X[-test_size:]
        y_test = y[-test_size:]
        
        # Advanced analysis with uncertainty quantification
        logger.info("🔍 Performing advanced analysis...")
        results = analyze_with_uncertainty(models, X_test, y_test, args.model)
        
        # Add metadata
        results.update({
            'symbol': args.symbol,
            'model_type': args.model,
            'dates': df.index[-test_size:],
            'recent_data': df.tail(test_size),
            'feature_level': feature_level
        })
        
        # Advanced market regime analysis (if requested)
        if args.regime_analysis:
            logger.info("🏛️ Analyzing advanced market regimes...")
            results['market_regime'] = analyze_advanced_market_regime(df, args)
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def analyze_advanced_market_regime(df, args):
    """Advanced market regime analysis with comprehensive detection system"""
    
    if args.regime_analysis:
        print("\n" + "="*50)
        print("ADVANCED MARKET REGIME ANALYSIS")
        print("="*50)
        
        try:
            from ai.market_regimes import create_regime_detection_pipeline
            
            # Run comprehensive regime detection
            regime_pipeline = create_regime_detection_pipeline(df)
            
            detector = regime_pipeline['detector']
            regimes = regime_pipeline['regimes']
            characteristics = regime_pipeline['characteristics']
            regime_labels = regime_pipeline['regime_labels']
            transition_matrix = regime_pipeline['transition_matrix']
            
            # Add regime to data
            regime_features = regime_pipeline['features']
            df_aligned = df.loc[regime_features.index]
            df_aligned['regime'] = regimes
            
            print("Advanced Regime Detection Results:")
            print("="*40)
            
            # Regime distribution
            regime_counts = pd.Series(regimes).value_counts().sort_index()
            total_days = len(regimes)
            
            for regime_id in sorted(regime_counts.index):
                count = regime_counts[regime_id]
                percentage = (count / total_days) * 100
                regime_name = regime_labels.get(regime_id, f"Regime {regime_id}")
                print(f"  {regime_name}: {count} days ({percentage:.1f}%)")
            
            # Current regime
            current_regime = regimes[-1]
            current_regime_name = regime_labels.get(current_regime, f"Regime {current_regime}")
            print(f"\nCurrent Market Regime: {current_regime_name}")
            
            # Detailed regime characteristics
            print("\nRegime Characteristics Analysis:")
            print("="*40)
            
            for regime_id, chars in characteristics.items():
                regime_name = regime_labels.get(regime_id, f"Regime {regime_id}")
                print(f"\n{regime_name}:")
                print(f"  Average Return: {chars['avg_return']*100:.2f}% per day")
                print(f"  Volatility: {chars['volatility']*100:.2f}%")
                print(f"  Trend Strength: {chars['trend_strength']*100:.2f}%")
                print(f"  Volume Activity: {chars['avg_volume_ratio']:.2f}x normal")
                print(f"  Duration: {chars['duration_days']} days ({chars['frequency']*100:.1f}% of time)")
                print(f"  Max Drawdown: {chars['max_drawdown']*100:.2f}%")
                print(f"  Sharpe Ratio: {chars['sharpe_ratio']:.2f}")
            
            # Transition analysis
            print("\nRegime Transition Matrix:")
            print("="*30)
            print("From → To probabilities:")
            
            for i, from_regime in enumerate(sorted(regime_labels.keys())):
                from_name = regime_labels[from_regime][:12].ljust(12)
                print(f"{from_name}:", end=" ")
                
                for j, to_regime in enumerate(sorted(regime_labels.keys())):
                    if i < len(transition_matrix) and j < len(transition_matrix[0]):
                        prob = transition_matrix[i][j]
                        print(f"{prob:.2f}", end="  ")
                    else:
                        print("0.00", end="  ")
                print()
            
            # Regime prediction
            try:
                future_probs = detector.predict_regime_probability(regime_features, horizon=5)
                if future_probs:
                    print("\nRegime Forecast (5-day ahead):")
                    print("="*35)
                    for regime_id, prob in sorted(future_probs.items()):
                        regime_name = regime_labels.get(regime_id, f"Regime {regime_id}")
                        print(f"  {regime_name}: {prob*100:.1f}% probability")
            except Exception as e:
                print(f"\nRegime prediction unavailable: {str(e)[:50]}")
            
            # Strategy recommendations
            strategy = regime_pipeline['strategy']
            current_params = strategy.get_strategy_parameters(current_regime)
            should_trade = strategy.should_trade(current_regime)
            
            print(f"\nStrategy Recommendations for {current_regime_name}:")
            print("="*45)
            print(f"  Trade Recommendation: {'TRADE' if should_trade else 'HOLD/AVOID'}")
            print(f"  Position Size Multiplier: {current_params['position_size']:.1f}x")
            print(f"  Recommended Stop Loss: {current_params['stop_loss']*100:.1f}%")
            print(f"  Recommended Take Profit: {current_params['take_profit']*100:.1f}%")
            
            return {
                'regime': current_regime_name,
                'regime_id': current_regime,
                'characteristics': characteristics,
                'transition_matrix': transition_matrix.tolist(),
                'strategy_params': current_params,
                'should_trade': should_trade,
                'regime_distribution': regime_counts.to_dict(),
                'advanced': True
            }
            
        except Exception as e:
            print(f"Advanced regime analysis failed: {str(e)}")
            print("Falling back to simple regime detection...")
            return analyze_simple_market_regime(df)
    
    return {}

def analyze_simple_market_regime(df):
    """Fallback simple market regime analysis"""
    try:
        # Add technical indicators
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['trend'] = (df['Close'] - df['sma_20']) / df['sma_20']
        
        # Simple regime detection
        df['regime'] = 0  # Default regime
        volatility_threshold = df['volatility'].quantile(0.7)
        trend_threshold = 0.02
        
        # High volatility regime
        high_vol_mask = df['volatility'] > volatility_threshold
        df.loc[high_vol_mask, 'regime'] = 1
        
        # Trending regimes
        strong_uptrend = (df['trend'] > trend_threshold) & (df['volatility'] <= volatility_threshold)
        strong_downtrend = (df['trend'] < -trend_threshold) & (df['volatility'] <= volatility_threshold)
        
        df.loc[strong_uptrend, 'regime'] = 2  # Bull market
        df.loc[strong_downtrend, 'regime'] = 3  # Bear market
        
        regime_names = {0: 'Consolidation', 1: 'High Volatility', 2: 'Bull Market', 3: 'Bear Market'}
        
        print("Simple Regime Distribution:")
        regime_counts = df['regime'].value_counts().sort_index()
        for regime, count in regime_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {regime_names[regime]}: {count} days ({percentage:.1f}%)")
        
        # Current regime
        current_regime = df['regime'].iloc[-1]
        print(f"\nCurrent Market Regime: {regime_names[current_regime]}")
        
        # Regime performance
        print("\nRegime Performance Analysis:")
        for regime in sorted(df['regime'].unique()):
            regime_data = df[df['regime'] == regime]
            if len(regime_data) > 0:
                avg_return = regime_data['returns'].mean() * 100
                volatility = regime_data['volatility'].mean() * 100
                print(f"  {regime_names[regime]}:")
                print(f"    Average Return: {avg_return:.2f}%")
                print(f"    Average Volatility: {volatility:.2f}%")
                if len(regime_data) > 1:
                    sharpe = (regime_data['returns'].mean() / regime_data['returns'].std()) * np.sqrt(252)
                    print(f"    Sharpe Ratio: {sharpe:.2f}")
        
        return {
            'regime': regime_names[current_regime],
            'regime_id': current_regime,
            'volatility_percentile': (df['volatility'] < df['volatility'].iloc[-1]).mean(),
            'trend_20d': (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100 if len(df) >= 20 else 0,
            'trend_60d': (df['Close'].iloc[-1] / df['Close'].iloc[-60] - 1) * 100 if len(df) >= 60 else 0,
            'current_volatility': df['volatility'].iloc[-1],
            'advanced': False
        }
    
    except Exception as e:
        logger.warning(f"Simple regime analysis failed: {e}")
        return {'regime': 'Unknown', 'error': str(e), 'advanced': False}

def generate_prediction_report(results: dict, detailed: bool = False, uncertainty: bool = False):
    """Generate enhanced analysis report"""
    if results is None:
        return
    
    symbol = results['symbol']
    model_type = results['model_type']
    predictions = results['predictions']
    
    logger.info("📈 ADVANCED ANALYSIS REPORT")
    logger.info(f"   📊 Symbol: {symbol}")
    logger.info(f"   🤖 Model: {model_type.upper()}")
    logger.info(f"   🧠 Features: {results.get('feature_level', 'standard').upper()}")
    logger.info(f"   📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    logger.info("🎯 PERFORMANCE METRICS:")
    logger.info(f"   📊 RMSE: {results['rmse']:.4f}")
    logger.info(f"   📈 MAE: {results['mae']:.4f}")
    logger.info(f"   🧭 Directional Accuracy: {results.get('directional_accuracy', 0):.2%}")
    
    # Advanced metrics
    if 'sharpe_ratio' in results:
        logger.info(f"   📊 Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    if 'max_drawdown' in results:
        logger.info(f"   📉 Max Drawdown: {results['max_drawdown']:.2%}")
    
    # Uncertainty metrics
    if uncertainty and 'uncertainty' in results:
        logger.info("🎲 UNCERTAINTY ANALYSIS:")
        logger.info(f"   📊 Mean Uncertainty: {results['mean_uncertainty']:.4f}")
        if 'uncertainty_calibration' in results:
            logger.info(f"   🎯 Uncertainty Calibration: {results['uncertainty_calibration']:.3f}")
        
        # Confidence intervals
        if 'confidence_intervals' in results:
            ci = results['confidence_intervals']
            logger.info(f"   📈 95% CI Width: {np.mean(ci['upper'] - ci['lower']):.4f}")
    
    # Market regime analysis
    if 'market_regime' in results:
        regime = results['market_regime']
        logger.info("🏛️ MARKET REGIME ANALYSIS:")
        logger.info(f"   📊 Current Regime: {regime['regime']}")
        logger.info(f"   📈 Volatility Percentile: {regime['volatility_percentile']:.1%}")
        logger.info(f"   🔄 20-Day Trend: {regime['trend_20d']:+.2f}%")
        logger.info(f"   📊 60-Day Trend: {regime['trend_60d']:+.2f}%")
    
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
        
        # Risk assessment
        if len(predictions) > 10:
            var_95 = np.percentile(np.diff(predictions) / predictions[:-1], 5) * 100
            logger.info(f"   ⚠️ Value at Risk (95%): {var_95:.2f}%")

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
        # Analyze stock performance with advanced features
        results = analyze_stock_performance(args)
        
        if results:
            # Generate enhanced report
            generate_prediction_report(
                results, 
                detailed=args.detailed,
                uncertainty=args.uncertainty
            )
            
            # Create advanced visualizations
            if args.save_plots:
                create_advanced_visualizations(args.symbol, results, args)
            
            # Create standard plots if available
            create_analysis_plots(results, save_plots=args.save_plots)
            
            logger.info("✅ Advanced analysis completed successfully!")
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