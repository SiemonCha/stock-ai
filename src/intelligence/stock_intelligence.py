import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ..data.advanced_features import AdvancedFeatureEngineer
from ..data.fundamental_analysis import FundamentalAnalyzer
from ..models.advanced_ensemble import AdvancedEnsembleModel

class StockIntelligenceSystem:
    def __init__(self):
        self.feature_engineer = AdvancedFeatureEngineer()
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.models = {}
        
    def analyze_stock_comprehensive(self, symbol: str, period: str = "2y") -> Dict:
        """
        Comprehensive stock analysis - everything you need to know
        """
        print(f"🔍 Performing comprehensive analysis for {symbol}...")
        
        try:
            # Get basic stock data
            stock = yf.Ticker(symbol)
            price_data = stock.history(period=period)
            info = stock.info
            
            if price_data.empty:
                return {"error": f"No data available for {symbol}"}
            
            # 1. CURRENT STATUS
            current_analysis = self._analyze_current_status(symbol, price_data, info)
            
            # 2. TECHNICAL ANALYSIS
            technical_analysis = self._analyze_technical_signals(price_data)
            
            # 3. FUNDAMENTAL ANALYSIS  
            fundamental_analysis = self._analyze_fundamentals(symbol)
            
            # 4. RISK ANALYSIS
            risk_analysis = self._analyze_risk(price_data)
            
            # 5. PRICE PREDICTION
            prediction_analysis = self._generate_predictions(symbol, price_data)
            
            # 6. OVERALL RECOMMENDATION
            overall_score = self._calculate_overall_score(
                technical_analysis, fundamental_analysis, risk_analysis
            )
            
            comprehensive_report = {
                "symbol": symbol,
                "analysis_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "current_status": current_analysis,
                "technical_analysis": technical_analysis,
                "fundamental_analysis": fundamental_analysis,
                "risk_analysis": risk_analysis,
                "predictions": prediction_analysis,
                "overall_assessment": overall_score,
                "simple_recommendation": self._generate_simple_recommendation(overall_score)
            }
            
            return comprehensive_report
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _analyze_current_status(self, symbol: str, price_data: pd.DataFrame, info: Dict) -> Dict:
        """Current stock status and key metrics"""
        current_price = price_data['Close'].iloc[-1]
        prev_close = price_data['Close'].iloc[-2]
        
        # Performance metrics
        daily_change = current_price - prev_close
        daily_change_pct = (daily_change / prev_close) * 100
        
        week_change_pct = ((current_price - price_data['Close'].iloc[-5]) / price_data['Close'].iloc[-5]) * 100
        month_change_pct = ((current_price - price_data['Close'].iloc[-20]) / price_data['Close'].iloc[-20]) * 100
        
        # Volume analysis
        current_volume = price_data['Volume'].iloc[-1]
        avg_volume = price_data['Volume'].tail(20).mean()
        volume_ratio = current_volume / avg_volume
        
        # Price levels
        week_52_high = price_data['High'].tail(252).max()
        week_52_low = price_data['Low'].tail(252).min()
        distance_from_high = ((week_52_high - current_price) / week_52_high) * 100
        distance_from_low = ((current_price - week_52_low) / week_52_low) * 100
        
        return {
            "current_price": round(current_price, 2),
            "daily_change": round(daily_change, 2),
            "daily_change_percent": round(daily_change_pct, 2),
            "week_change_percent": round(week_change_pct, 2),
            "month_change_percent": round(month_change_pct, 2),
            "volume_vs_average": round(volume_ratio, 2),
            "52_week_high": round(week_52_high, 2),
            "52_week_low": round(week_52_low, 2),
            "distance_from_high_percent": round(distance_from_high, 2),
            "distance_from_low_percent": round(distance_from_low, 2),
            "market_cap": info.get('marketCap', 'N/A'),
            "sector": info.get('sector', 'N/A'),
            "industry": info.get('industry', 'N/A')
        }
    
    def _analyze_technical_signals(self, price_data: pd.DataFrame) -> Dict:
        """Technical analysis with clear signals"""
        
        # Create enhanced features
        enhanced_data = self.feature_engineer.create_comprehensive_features(price_data)
        
        signals = {
            "trend_signals": {},
            "momentum_signals": {},
            "volume_signals": {},
            "volatility_signals": {},
            "pattern_signals": {},
            "overall_technical_score": 0
        }
        
        latest_data = enhanced_data.iloc[-1]
        recent_data = enhanced_data.tail(5)
        
        # TREND SIGNALS
        if 'sma_20' in enhanced_data.columns and 'sma_50' in enhanced_data.columns:
            price = latest_data['close']
            sma_20 = latest_data['sma_20']
            sma_50 = latest_data['sma_50']
            
            if price > sma_20 > sma_50:
                signals["trend_signals"]["short_term"] = "BULLISH"
                signals["trend_signals"]["short_term_strength"] = "Strong"
            elif price > sma_20:
                signals["trend_signals"]["short_term"] = "BULLISH"
                signals["trend_signals"]["short_term_strength"] = "Moderate"
            elif price < sma_20 < sma_50:
                signals["trend_signals"]["short_term"] = "BEARISH"
                signals["trend_signals"]["short_term_strength"] = "Strong"
            else:
                signals["trend_signals"]["short_term"] = "NEUTRAL"
                signals["trend_signals"]["short_term_strength"] = "Weak"
        
        # MOMENTUM SIGNALS
        if 'rsi' in enhanced_data.columns:
            rsi = latest_data['rsi']
            if rsi > 70:
                signals["momentum_signals"]["rsi"] = "OVERBOUGHT"
                signals["momentum_signals"]["rsi_value"] = round(rsi, 2)
            elif rsi < 30:
                signals["momentum_signals"]["rsi"] = "OVERSOLD"
                signals["momentum_signals"]["rsi_value"] = round(rsi, 2)
            else:
                signals["momentum_signals"]["rsi"] = "NEUTRAL"
                signals["momentum_signals"]["rsi_value"] = round(rsi, 2)
        
        if 'macd' in enhanced_data.columns and 'macd_signal' in enhanced_data.columns:
            macd = latest_data['macd']
            macd_signal = latest_data['macd_signal']
            
            if macd > macd_signal and recent_data['macd'].iloc[-2] <= recent_data['macd_signal'].iloc[-2]:
                signals["momentum_signals"]["macd"] = "BULLISH_CROSSOVER"
            elif macd < macd_signal and recent_data['macd'].iloc[-2] >= recent_data['macd_signal'].iloc[-2]:
                signals["momentum_signals"]["macd"] = "BEARISH_CROSSOVER"
            elif macd > macd_signal:
                signals["momentum_signals"]["macd"] = "BULLISH"
            else:
                signals["momentum_signals"]["macd"] = "BEARISH"
        
        # VOLUME SIGNALS
        if 'volume_ratio' in enhanced_data.columns:
            vol_ratio = latest_data['volume_ratio']
            if vol_ratio > 2:
                signals["volume_signals"]["activity"] = "VERY_HIGH"
            elif vol_ratio > 1.5:
                signals["volume_signals"]["activity"] = "HIGH"
            elif vol_ratio > 0.8:
                signals["volume_signals"]["activity"] = "NORMAL"
            else:
                signals["volume_signals"]["activity"] = "LOW"
            
            signals["volume_signals"]["ratio"] = round(vol_ratio, 2)
        
        # VOLATILITY SIGNALS
        if 'volatility' in enhanced_data.columns:
            volatility = latest_data['volatility']
            volatility_pct = volatility * 100
            
            if volatility_pct > 5:
                signals["volatility_signals"]["level"] = "VERY_HIGH"
            elif volatility_pct > 3:
                signals["volatility_signals"]["level"] = "HIGH"  
            elif volatility_pct > 1:
                signals["volatility_signals"]["level"] = "MODERATE"
            else:
                signals["volatility_signals"]["level"] = "LOW"
                
            signals["volatility_signals"]["value"] = round(volatility_pct, 2)
        
        # PATTERN SIGNALS
        if 'is_support' in enhanced_data.columns and 'is_resistance' in enhanced_data.columns:
            if latest_data['is_support']:
                signals["pattern_signals"]["support_level"] = "AT_SUPPORT"
            elif latest_data['is_resistance']:
                signals["pattern_signals"]["resistance_level"] = "AT_RESISTANCE"
        
        # Calculate technical score
        score = 50  # Start neutral
        
        # Trend contribution (30 points)
        if signals["trend_signals"].get("short_term") == "BULLISH":
            score += 15 if signals["trend_signals"].get("short_term_strength") == "Strong" else 10
        elif signals["trend_signals"].get("short_term") == "BEARISH":
            score -= 15 if signals["trend_signals"].get("short_term_strength") == "Strong" else 10
        
        # Momentum contribution (20 points)
        rsi_signal = signals["momentum_signals"].get("rsi")
        if rsi_signal == "OVERSOLD":
            score += 10
        elif rsi_signal == "OVERBOUGHT":
            score -= 10
            
        macd_signal = signals["momentum_signals"].get("macd")
        if "BULLISH" in str(macd_signal):
            score += 10
        elif "BEARISH" in str(macd_signal):
            score -= 10
        
        signals["overall_technical_score"] = max(0, min(100, score))
        
        return signals
    
    def _analyze_fundamentals(self, symbol: str) -> Dict:
        """Fundamental analysis with clear interpretation"""
        try:
            fundamental_data = self.fundamental_analyzer.get_financial_metrics(symbol)
            health_score = self.fundamental_analyzer.calculate_financial_health_score(symbol)
            
            # Interpret key metrics
            interpretation = {
                "financial_health": health_score,
                "valuation_assessment": {},
                "profitability_assessment": {},
                "growth_assessment": {},
                "key_metrics": fundamental_data
            }
            
            # Valuation Assessment
            pe = fundamental_data.get('pe_ratio', None)
            if pe and pe > 0:
                if pe < 15:
                    interpretation["valuation_assessment"]["pe_interpretation"] = "UNDERVALUED"
                elif pe < 25:
                    interpretation["valuation_assessment"]["pe_interpretation"] = "FAIRLY_VALUED"
                else:
                    interpretation["valuation_assessment"]["pe_interpretation"] = "OVERVALUED"
                interpretation["valuation_assessment"]["pe_ratio"] = round(pe, 2)
            
            pb = fundamental_data.get('price_to_book', None)
            if pb and pb > 0:
                if pb < 1:
                    interpretation["valuation_assessment"]["pb_interpretation"] = "UNDERVALUED"
                elif pb < 3:
                    interpretation["valuation_assessment"]["pb_interpretation"] = "FAIRLY_VALUED"
                else:
                    interpretation["valuation_assessment"]["pb_interpretation"] = "OVERVALUED"
                interpretation["valuation_assessment"]["pb_ratio"] = round(pb, 2)
            
            # Profitability Assessment
            roe = fundamental_data.get('return_on_equity', None)
            if roe:
                if roe > 0.15:
                    interpretation["profitability_assessment"]["roe_quality"] = "EXCELLENT"
                elif roe > 0.10:
                    interpretation["profitability_assessment"]["roe_quality"] = "GOOD"
                elif roe > 0.05:
                    interpretation["profitability_assessment"]["roe_quality"] = "FAIR"
                else:
                    interpretation["profitability_assessment"]["roe_quality"] = "POOR"
                interpretation["profitability_assessment"]["roe_value"] = round(roe * 100, 2)
            
            # Growth Assessment
            revenue_growth = fundamental_data.get('revenue_growth', None)
            if revenue_growth:
                if revenue_growth > 0.20:
                    interpretation["growth_assessment"]["revenue_growth_quality"] = "EXCELLENT"
                elif revenue_growth > 0.10:
                    interpretation["growth_assessment"]["revenue_growth_quality"] = "GOOD"
                elif revenue_growth > 0.05:
                    interpretation["growth_assessment"]["revenue_growth_quality"] = "MODERATE"
                else:
                    interpretation["growth_assessment"]["revenue_growth_quality"] = "SLOW"
                interpretation["growth_assessment"]["revenue_growth_value"] = round(revenue_growth * 100, 2)
            
            return interpretation
            
        except Exception as e:
            return {"error": f"Fundamental analysis failed: {str(e)}"}
    
    def _analyze_risk(self, price_data: pd.DataFrame) -> Dict:
        """Risk analysis with clear metrics"""
        returns = price_data['Close'].pct_change().dropna()
        
        # Calculate risk metrics
        daily_volatility = returns.std()
        annual_volatility = daily_volatility * np.sqrt(252)
        
        # VaR calculation (95% confidence)
        var_95 = returns.quantile(0.05)
        
        # Maximum drawdown
        rolling_max = price_data['Close'].expanding().max()
        drawdown = (price_data['Close'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Beta calculation (vs SPY if possible)
        try:
            spy = yf.download("SPY", start=price_data.index[0], end=price_data.index[-1])['Close']
            spy_returns = spy.pct_change().dropna()
            
            # Align returns
            aligned_returns = returns.reindex(spy_returns.index).dropna()
            aligned_spy_returns = spy_returns.reindex(aligned_returns.index).dropna()
            
            if len(aligned_returns) > 20 and len(aligned_spy_returns) > 20:
                beta = np.cov(aligned_returns, aligned_spy_returns)[0][1] / np.var(aligned_spy_returns)
            else:
                beta = None
        except:
            beta = None
        
        # Risk interpretation
        risk_level = "MODERATE"
        if annual_volatility > 0.40:
            risk_level = "VERY_HIGH"
        elif annual_volatility > 0.30:
            risk_level = "HIGH"
        elif annual_volatility > 0.15:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"
        
        return {
            "risk_level": risk_level,
            "annual_volatility": round(annual_volatility * 100, 2),
            "daily_volatility": round(daily_volatility * 100, 2),
            "var_95_percent": round(var_95 * 100, 2),
            "max_drawdown": round(max_drawdown * 100, 2),
            "beta": round(beta, 2) if beta else None,
            "risk_interpretation": {
                "volatility": f"Stock moves {round(annual_volatility * 100, 1)}% annually on average",
                "worst_case": f"Could lose {round(abs(var_95) * 100, 1)}% in a day (5% chance)",
                "max_loss": f"Worst historical loss was {round(abs(max_drawdown) * 100, 1)}%"
            }
        }
    
    def _generate_predictions(self, symbol: str, price_data: pd.DataFrame) -> Dict:
        """Generate price predictions with confidence intervals"""
        try:
            current_price = price_data['Close'].iloc[-1]
            
            # Simple trend-based prediction (can be enhanced with trained models)
            returns = price_data['Close'].pct_change().tail(20)
            avg_return = returns.mean()
            volatility = returns.std()
            
            # Predict next few periods
            predictions = {
                "next_day": {
                    "price": round(current_price * (1 + avg_return), 2),
                    "change_percent": round(avg_return * 100, 2),
                    "confidence": "Moderate"
                },
                "next_week": {
                    "price": round(current_price * (1 + avg_return * 5), 2),
                    "change_percent": round(avg_return * 5 * 100, 2),
                    "confidence": "Low"
                },
                "next_month": {
                    "price": round(current_price * (1 + avg_return * 22), 2),
                    "change_percent": round(avg_return * 22 * 100, 2),
                    "confidence": "Very Low"
                }
            }
            
            # Add confidence intervals
            for period in predictions:
                price = predictions[period]["price"]
                lower = round(price * (1 - 2 * volatility), 2)
                upper = round(price * (1 + 2 * volatility), 2)
                predictions[period]["confidence_interval"] = [lower, upper]
            
            return predictions
            
        except Exception as e:
            return {"error": f"Prediction generation failed: {str(e)}"}
    
    def _calculate_overall_score(self, technical: Dict, fundamental: Dict, risk: Dict) -> Dict:
        """Calculate overall investment score"""
        
        scores = {
            "technical_score": technical.get("overall_technical_score", 50),
            "fundamental_score": fundamental.get("financial_health", {}).get("percentage", 50),
            "risk_score": 50  # Default middle score
        }
        
        # Risk score adjustment
        risk_level = risk.get("risk_level", "MODERATE")
        if risk_level == "LOW":
            scores["risk_score"] = 80
        elif risk_level == "MODERATE":
            scores["risk_score"] = 60
        elif risk_level == "HIGH":
            scores["risk_score"] = 40
        else:  # VERY_HIGH
            scores["risk_score"] = 20
        
        # Weighted overall score
        overall_score = (
            scores["technical_score"] * 0.4 +
            scores["fundamental_score"] * 0.4 +
            scores["risk_score"] * 0.2
        )
        
        return {
            "overall_score": round(overall_score, 1),
            "component_scores": scores,
            "rating": self._score_to_rating(overall_score)
        }
    
    def _score_to_rating(self, score: float) -> str:
        """Convert score to rating"""
        if score >= 80:
            return "STRONG BUY"
        elif score >= 70:
            return "BUY"
        elif score >= 60:
            return "HOLD"
        elif score >= 50:
            return "WEAK HOLD"
        else:
            return "SELL"
    
    def _generate_simple_recommendation(self, overall_assessment: Dict) -> Dict:
        """Generate simple, user-friendly recommendation"""
        score = overall_assessment["overall_score"]
        rating = overall_assessment["rating"]
        
        # Simple explanation based on score
        if score >= 75:
            explanation = "Strong fundamentals and positive technical signals suggest good upside potential."
            action = "Consider buying"
            emoji = "🟢"
        elif score >= 65:
            explanation = "Decent fundamentals with mixed technical signals. Suitable for moderate risk investors."
            action = "Consider buying on dips"
            emoji = "🟡"
        elif score >= 55:
            explanation = "Mixed signals. Good for current holders but wait for better entry for new positions."
            action = "Hold current position"
            emoji = "🟡"
        else:
            explanation = "Weak fundamentals or negative technical signals suggest limited upside."
            action = "Consider selling or avoid"
            emoji = "🔴"
        
        return {
            "rating": rating,
            "emoji": emoji,
            "action": action,
            "explanation": explanation,
            "confidence": "High" if score > 70 or score < 40 else "Moderate"
        }