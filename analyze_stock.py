#!/usr/bin/env python3

import sys
import os
import argparse
from datetime import datetime
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from intelligence.stock_intelligence import StockIntelligenceSystem

def print_banner():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                  🤖 AI STOCK INTELLIGENCE SYSTEM             ║
║              Advanced Stock Analysis & Prediction            ║
╚═══════════════════════════════════════════════════════════════╝
    """)

def print_simple_report(analysis: dict):
    """Print a simple, user-friendly report"""
    symbol = analysis.get("symbol", "Unknown")
    
    print(f"\n{'='*60}")
    print(f"📈 STOCK ANALYSIS REPORT: {symbol}")
    print(f"{'='*60}")
    
    # QUICK SUMMARY
    recommendation = analysis.get("simple_recommendation", {})
    if recommendation:
        print(f"\n🎯 QUICK RECOMMENDATION:")
        print(f"   {recommendation.get('emoji', '')} Rating: {recommendation.get('rating', 'N/A')}")
        print(f"   💡 Action: {recommendation.get('action', 'N/A')}")
        print(f"   📝 Reason: {recommendation.get('explanation', 'N/A')}")
        print(f"   🎲 Confidence: {recommendation.get('confidence', 'N/A')}")
    
    # CURRENT STATUS
    current = analysis.get("current_status", {})
    if current:
        print(f"\n📊 CURRENT STATUS:")
        print(f"   💰 Price: ${current.get('current_price', 'N/A')}")
        print(f"   📈 Daily Change: {current.get('daily_change', 'N/A')} ({current.get('daily_change_percent', 'N/A')}%)")
        print(f"   📅 Week Change: {current.get('week_change_percent', 'N/A')}%")
        print(f"   🗓️  Month Change: {current.get('month_change_percent', 'N/A')}%")
        print(f"   🏢 Sector: {current.get('sector', 'N/A')}")
        print(f"   🔢 52W High: ${current.get('52_week_high', 'N/A')} | Low: ${current.get('52_week_low', 'N/A')}")
    
    # OVERALL SCORES
    overall = analysis.get("overall_assessment", {})
    if overall:
        print(f"\n🏆 OVERALL SCORES:")
        print(f"   🎯 Overall Score: {overall.get('overall_score', 'N/A')}/100")
        component_scores = overall.get("component_scores", {})
        if component_scores:
            print(f"   📈 Technical: {component_scores.get('technical_score', 'N/A')}/100")
            print(f"   💼 Fundamental: {component_scores.get('fundamental_score', 'N/A')}/100")
            print(f"   ⚠️  Risk: {component_scores.get('risk_score', 'N/A')}/100")
    
    # TECHNICAL SIGNALS
    technical = analysis.get("technical_analysis", {})
    if technical:
        print(f"\n🔍 TECHNICAL SIGNALS:")
        trend = technical.get("trend_signals", {})
        momentum = technical.get("momentum_signals", {})
        
        if trend:
            print(f"   📈 Trend: {trend.get('short_term', 'N/A')} ({trend.get('short_term_strength', 'N/A')})")
        
        if momentum:
            rsi = momentum.get("rsi_value", "N/A")
            rsi_signal = momentum.get("rsi", "N/A")
            print(f"   ⚡ RSI: {rsi} ({rsi_signal})")
            print(f"   📊 MACD: {momentum.get('macd', 'N/A')}")
    
    # FUNDAMENTAL HEALTH
    fundamental = analysis.get("fundamental_analysis", {})
    if fundamental and "financial_health" in fundamental:
        health = fundamental["financial_health"]
        print(f"\n💼 FINANCIAL HEALTH:")
        print(f"   🏥 Health Score: {health.get('percentage', 'N/A')}/100")
        print(f"   🏅 Rating: {health.get('rating', 'N/A')}")
        
        # Key metrics
        valuation = fundamental.get("valuation_assessment", {})
        if valuation:
            pe = valuation.get("pe_ratio")
            if pe:
                print(f"   💰 P/E Ratio: {pe} ({valuation.get('pe_interpretation', 'N/A')})")
    
    # RISK ANALYSIS
    risk = analysis.get("risk_analysis", {})
    if risk:
        print(f"\n⚠️  RISK ANALYSIS:")
        print(f"   🎯 Risk Level: {risk.get('risk_level', 'N/A')}")
        print(f"   📊 Annual Volatility: {risk.get('annual_volatility', 'N/A')}%")
        print(f"   📉 Max Drawdown: {risk.get('max_drawdown', 'N/A')}%")
        
        risk_interp = risk.get("risk_interpretation", {})
        if risk_interp:
            print(f"   💡 In Simple Terms: {risk_interp.get('volatility', 'N/A')}")
    
    # PREDICTIONS
    predictions = analysis.get("predictions", {})
    if predictions and "next_day" in predictions:
        print(f"\n🔮 PREDICTIONS:")
        next_day = predictions["next_day"]
        next_week = predictions.get("next_week", {})
        
        print(f"   📅 Tomorrow: ${next_day.get('price', 'N/A')} ({next_day.get('change_percent', 'N/A')}%)")
        if next_week:
            print(f"   📅 Next Week: ${next_week.get('price', 'N/A')} ({next_week.get('change_percent', 'N/A')}%)")
    
    print(f"\n{'='*60}")
    print(f"⏰ Analysis completed at: {analysis.get('analysis_date', 'Unknown')}")
    print(f"{'='*60}\n")

def print_detailed_report(analysis: dict, save_to_file: bool = False):
    """Print detailed JSON report"""
    if save_to_file:
        filename = f"analysis_{analysis.get('symbol', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"📁 Detailed report saved to: {filename}")
    else:
        print(json.dumps(analysis, indent=2, default=str))

def main():
    parser = argparse.ArgumentParser(description='AI Stock Intelligence System')
    parser.add_argument('symbol', type=str, help='Stock symbol to analyze (e.g., AAPL, GOOGL)')
    parser.add_argument('--period', type=str, default='2y', help='Analysis period (default: 2y)')
    parser.add_argument('--detailed', action='store_true', help='Show detailed JSON report')
    parser.add_argument('--save', action='store_true', help='Save detailed report to file')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Initialize the intelligence system
    print("🚀 Initializing AI Stock Intelligence System...")
    intelligence_system = StockIntelligenceSystem()
    
    # Perform comprehensive analysis
    print(f"🔍 Analyzing {args.symbol.upper()}...")
    analysis = intelligence_system.analyze_stock_comprehensive(
        symbol=args.symbol.upper(),
        period=args.period
    )
    
    # Check for errors
    if "error" in analysis:
        print(f"❌ Error: {analysis['error']}")
        return 1
    
    # Display results
    if args.detailed or args.save:
        print_detailed_report(analysis, save_to_file=args.save)
    else:
        print_simple_report(analysis)
    
    # Additional suggestions
    print("💡 NEXT STEPS:")
    print("   • Use --detailed flag for complete analysis")
    print("   • Use --save flag to save report as JSON")
    print("   • Train ML model: python train_advanced.py --symbol", args.symbol.upper())
    print("   • Start API server: python start_api.py")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⛔ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)