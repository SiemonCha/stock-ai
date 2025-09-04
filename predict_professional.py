#!/usr/bin/env python3
"""
Professional Stock Predictor Interface
Ultra-simple interface for professional investors targeting 99% accuracy
"""

import sys
import os
from pathlib import Path
import argparse
import logging
from datetime import datetime

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

try:
    from professional.ultra_predictor import ProfessionalStockPredictor
    from professional.market_data_engine import ComprehensiveMarketDataEngine, create_professional_data_config
    from professional.risk_manager import ProfessionalRiskManager
    PROFESSIONAL_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Professional modules not available: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install xgboost catboost lightgbm tensorflow torch scikit-learn")
    PROFESSIONAL_IMPORTS_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Professional Stock Predictor - 99% Accuracy Target')
    
    # Basic prediction
    parser.add_argument('symbol', help='Stock symbol to analyze (e.g., AAPL)')
    parser.add_argument('--days', type=int, default=30, help='Days ahead to predict (default: 30)')
    
    # Analysis type
    parser.add_argument('--quick', action='store_true', help='Quick analysis for immediate decisions')
    parser.add_argument('--detailed', action='store_true', help='Detailed professional analysis')
    parser.add_argument('--position-size', type=float, help='Calculate position size for portfolio value')
    
    # Risk management
    parser.add_argument('--portfolio-value', type=float, default=100000, 
                       help='Portfolio value for position sizing (default: $100,000)')
    parser.add_argument('--max-risk', type=float, default=0.02, 
                       help='Maximum risk per position (default: 2%)')
    
    # Professional features
    parser.add_argument('--comprehensive-data', action='store_true',
                       help='Use all available data sources (slower but more accurate)')
    parser.add_argument('--save-analysis', help='Save detailed analysis to file')
    
    return parser.parse_args()

def main():
    """Main professional prediction interface"""
    
    if not PROFESSIONAL_IMPORTS_AVAILABLE:
        print("❌ Professional prediction system not available")
        print("Please install required dependencies and try again")
        return 1
    
    args = parse_arguments()
    
    print("🎯 PROFESSIONAL STOCK AI PREDICTOR")
    print("=" * 50)
    print(f"Target: 99% Directional Accuracy")
    print(f"Symbol: {args.symbol}")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    try:
        # Initialize professional predictor
        predictor = ProfessionalStockPredictor()
        
        # Quick analysis for immediate decisions
        if args.quick:
            print("\n🚀 QUICK ANALYSIS")
            print("-" * 30)
            
            quick_result = predictor.quick_analysis(args.symbol)
            print(quick_result)
            
            return 0
        
        # Comprehensive prediction
        print(f"\n🔮 Making ultra-high accuracy prediction for {args.symbol}...")
        print("⏳ This may take 2-3 minutes for maximum accuracy...")
        
        prediction_result = predictor.predict_stock(args.symbol, args.days)
        
        # Display results
        print("\n" + "=" * 60)
        print("🎯 PROFESSIONAL PREDICTION RESULTS")
        print("=" * 60)
        
        current_price = prediction_result['current_price']
        future_price = prediction_result['predicted_prices'][-1]
        change_pct = ((future_price - current_price) / current_price) * 100
        confidence = prediction_result['confidence_score'] * 100
        accuracy = prediction_result['directional_accuracy'] * 100
        
        print(f"📊 Current Price: ${current_price:.2f}")
        print(f"🎯 {args.days}-Day Target: ${future_price:.2f}")
        print(f"📈 Expected Move: {change_pct:+.1f}%")
        print(f"🎯 Confidence Score: {confidence:.1f}%")
        print(f"📊 Model Accuracy: {accuracy:.1f}%")
        print(f"🤖 Models Used: {prediction_result['model_count']}")
        
        # Signal strength
        if abs(change_pct) > 10:
            signal = "🔥 VERY STRONG"
        elif abs(change_pct) > 5:
            signal = "💪 STRONG"
        elif abs(change_pct) > 2:
            signal = "📊 MODERATE"
        else:
            signal = "😴 WEAK"
        
        direction = "📈 BULLISH" if change_pct > 0 else "📉 BEARISH" if change_pct < 0 else "➡️ NEUTRAL"
        
        print(f"\n🚦 TRADING SIGNAL")
        print(f"Direction: {direction}")
        print(f"Strength: {signal}")
        
        # Trading recommendation
        if confidence > 80 and abs(change_pct) > 3:
            if change_pct > 0:
                recommendation = "🟢 STRONG BUY"
            else:
                recommendation = "🔴 STRONG SELL"
        elif confidence > 70 and abs(change_pct) > 2:
            if change_pct > 0:
                recommendation = "🟡 BUY"
            else:
                recommendation = "🟠 SELL"
        else:
            recommendation = "⚪ HOLD"
        
        print(f"Recommendation: {recommendation}")
        
        # Position sizing if requested
        if args.position_size or args.portfolio_value:
            print(f"\n💰 POSITION SIZING ANALYSIS")
            print("-" * 40)
            
            # Prepare prediction data for risk manager
            prediction_data = {
                'expected_return': change_pct / 100,
                'confidence_score': confidence / 100,
                'volatility': 0.25,  # Default volatility
                'directional_accuracy': accuracy / 100,
                'price_history': [current_price] * 50  # Placeholder
            }
            
            risk_manager = ProfessionalRiskManager()
            position_analysis = risk_manager.calculate_position_size(
                args.symbol, prediction_data, args.portfolio_value, current_price
            )
            
            print(f"💼 Portfolio Value: ${args.portfolio_value:,.0f}")
            print(f"📊 Optimal Position: {position_analysis['optimal_shares']:,} shares")
            print(f"💵 Position Value: ${position_analysis['position_value']:,.0f}")
            print(f"📈 Position Weight: {position_analysis['position_weight']:.1%}")
            print(f"🛑 Stop Loss: ${position_analysis['stop_loss_price']:.2f}")
            print(f"🎯 Take Profit: ${position_analysis['take_profit_price']:.2f}")
            print(f"⚖️ Risk/Reward: {position_analysis['risk_metrics']['risk_reward_ratio']:.1f}:1")
        
        # Detailed breakdown if requested
        if args.detailed:
            print(f"\n📊 MODEL BREAKDOWN")
            print("-" * 30)
            
            for model_name, prediction in prediction_result.get('individual_predictions', {}).items():
                print(f"{model_name}: ${current_price * (1 + prediction):.2f} ({prediction * 100:+.1f}%)")
            
            print(f"\n📈 VALIDATION METRICS")
            print("-" * 30)
            
            for model_name, metrics in prediction_result.get('validation_metrics', {}).items():
                print(f"{model_name}:")
                print(f"  RMSE: {metrics.get('rmse', 0):.4f}")
                print(f"  Accuracy: {metrics.get('directional_accuracy', 0):.1%}")
        
        # Price trajectory
        print(f"\n📅 PRICE TRAJECTORY ({args.days} days)")
        print("-" * 40)
        
        prices = [current_price] + prediction_result['predicted_prices']
        for i, price in enumerate(prices[:min(8, len(prices))]):  # Show first 8 days
            if i == 0:
                print(f"Day 0 (Today): ${price:.2f}")
            else:
                change = ((price - current_price) / current_price) * 100
                print(f"Day {i * (args.days // 7) if args.days > 7 else i}: ${price:.2f} ({change:+.1f}%)")
        
        # Risk warnings
        print(f"\n⚠️ RISK CONSIDERATIONS")
        print("-" * 30)
        
        if confidence < 60:
            print("🟡 LOW CONFIDENCE - Consider smaller position size")
        
        if abs(change_pct) > 20:
            print("🟠 HIGH VOLATILITY EXPECTED - Use appropriate stops")
        
        if prediction_result['model_count'] < 5:
            print("🟡 LIMITED MODEL CONSENSUS - Exercise caution")
        
        print("🔴 Past performance does not guarantee future results")
        print("🔴 This is not financial advice - consult professionals")
        
        # Save detailed analysis if requested
        if args.save_analysis:
            import json
            
            with open(args.save_analysis, 'w') as f:
                json.dump(prediction_result, f, indent=2, default=str)
            
            print(f"\n💾 Detailed analysis saved to: {args.save_analysis}")
        
        print(f"\n✅ Professional analysis complete!")
        print(f"📊 Prediction made with {prediction_result['model_count']} advanced models")
        print(f"🎯 Target accuracy: 99% (Current model accuracy: {accuracy:.1f}%)")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️ Analysis interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"❌ Professional prediction failed: {e}")
        print(f"\n❌ Error: {e}")
        print("Please check your internet connection and symbol validity")
        return 1

if __name__ == "__main__":
    sys.exit(main())