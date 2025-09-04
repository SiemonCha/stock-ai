"""
Real-Time Market Scanner and Screener
Scans thousands of stocks for trading opportunities with professional-grade filtering
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Financial data
try:
    import yfinance as yf
    import requests
    FINANCIAL_DATA_AVAILABLE = True
except ImportError:
    FINANCIAL_DATA_AVAILABLE = False

# Technical analysis
try:
    import talib
    import ta
    TECHNICAL_ANALYSIS_AVAILABLE = True
except ImportError:
    TECHNICAL_ANALYSIS_AVAILABLE = False

# Import professional modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from professional.ultra_predictor import ProfessionalStockPredictor
    from professional.risk_manager import ProfessionalRiskManager
    PROFESSIONAL_MODULES_AVAILABLE = True
except ImportError:
    PROFESSIONAL_MODULES_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProfessionalMarketScanner:
    """
    Professional market scanner for finding high-probability trading opportunities
    
    Features:
    - Real-time scanning of 1000+ stocks
    - Technical pattern recognition
    - Fundamental screening
    - Momentum detection
    - Breakout identification
    - Professional-grade filters
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        
        # Initialize data sources
        self.stock_universes = self._load_stock_universes()
        self.predictor = ProfessionalStockPredictor() if PROFESSIONAL_MODULES_AVAILABLE else None
        self.risk_manager = ProfessionalRiskManager() if PROFESSIONAL_MODULES_AVAILABLE else None
        
        # Scanner results
        self.scan_results = {}
        self.alert_history = []
        
        logger.info(f"🔍 Market Scanner initialized - {sum(len(universe) for universe in self.stock_universes.values())} stocks to scan")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default scanner configuration"""
        return {
            # Scanning parameters
            'max_concurrent_requests': 50,
            'scan_interval_minutes': 15,
            'data_freshness_minutes': 30,
            
            # Technical filters
            'min_volume': 500000,  # Minimum daily volume
            'min_price': 5.0,      # Minimum stock price
            'max_price': 1000.0,   # Maximum stock price
            'min_market_cap': 100_000_000,  # $100M minimum
            
            # Momentum filters
            'min_momentum_days': 5,
            'momentum_threshold': 0.03,  # 3% minimum move
            'breakout_volume_multiplier': 1.5,
            
            # Pattern recognition
            'enable_candlestick_patterns': True,
            'enable_chart_patterns': True,
            'enable_support_resistance': True,
            
            # Fundamental filters
            'max_pe_ratio': 50,
            'min_revenue_growth': -0.1,  # -10% minimum
            'max_debt_to_equity': 2.0,
            
            # Prediction filters
            'min_prediction_confidence': 0.70,  # 70% confidence
            'min_expected_return': 0.02,        # 2% expected return
            'max_risk_score': 7.0,              # Risk score out of 10
            
            # Alert thresholds
            'strong_buy_threshold': 0.08,   # 8%+ expected return
            'buy_threshold': 0.04,          # 4%+ expected return
            'sell_threshold': -0.04,        # -4% or worse
            'strong_sell_threshold': -0.08, # -8% or worse
        }
    
    def _load_stock_universes(self) -> Dict[str, List[str]]:
        """Load different stock universes to scan"""
        
        universes = {}
        
        # S&P 500 stocks
        universes['sp500'] = self._get_sp500_stocks()
        
        # NASDAQ 100 stocks
        universes['nasdaq100'] = self._get_nasdaq100_stocks()
        
        # Russell 2000 small caps (sample)
        universes['russell2000'] = self._get_russell2000_sample()
        
        # Popular growth stocks
        universes['growth'] = [
            'NVDA', 'TSLA', 'AMZN', 'GOOGL', 'MSFT', 'AAPL', 'META', 'NFLX',
            'AMD', 'CRM', 'ADBE', 'PYPL', 'SQ', 'ROKU', 'ZOOM', 'SHOP',
            'SNOW', 'PLTR', 'ARKK', 'TQQQ', 'SOXL', 'UPRO'
        ]
        
        # Value stocks
        universes['value'] = [
            'BRK.B', 'JPM', 'BAC', 'WFC', 'V', 'JNJ', 'PG', 'KO', 'PFE',
            'XOM', 'CVX', 'T', 'VZ', 'IBM', 'INTC', 'GE', 'F', 'GM'
        ]
        
        # High beta / momentum stocks
        universes['momentum'] = [
            'GME', 'AMC', 'MEME', 'COIN', 'HOOD', 'RIVN', 'LCID', 'SOFI',
            'WISH', 'CLOV', 'BB', 'NOK', 'SNDL', 'TLRY', 'SPCE', 'PLTR'
        ]
        
        # Crypto-related stocks
        universes['crypto'] = [
            'COIN', 'RIOT', 'MARA', 'HUT', 'BITF', 'CAN', 'HOOD', 'SQ',
            'PYPL', 'MSTR', 'TSLA', 'NVDA'
        ]
        
        return universes
    
    def _get_sp500_stocks(self) -> List[str]:
        """Get S&P 500 stock list"""
        # Top 50 S&P 500 stocks by market cap (simplified)
        return [
            'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'TSLA', 'META', 'GOOG',
            'BRK.B', 'UNH', 'XOM', 'JNJ', 'JPM', 'V', 'PG', 'MA', 'CVX',
            'HD', 'PFE', 'ABBV', 'BAC', 'KO', 'AVGO', 'PEP', 'TMO', 'COST',
            'MRK', 'DIS', 'ABT', 'ACN', 'VZ', 'ADBE', 'WMT', 'CRM', 'NFLX',
            'AMD', 'LLY', 'NKE', 'ORCL', 'DHR', 'T', 'BMY', 'INTC', 'PM',
            'TXN', 'HON', 'UPS', 'QCOM', 'SBUX', 'LOW'
        ]
    
    def _get_nasdaq100_stocks(self) -> List[str]:
        """Get NASDAQ 100 stock list"""
        # Top NASDAQ 100 stocks (simplified)
        return [
            'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA',
            'AVGO', 'COST', 'ADBE', 'NFLX', 'PEP', 'TMUS', 'CSCO', 'COMCAST',
            'TXN', 'QCOM', 'HON', 'SBUX', 'AMAT', 'INTU', 'AMD', 'ISRG',
            'BKNG', 'ADP', 'GILD', 'MU', 'ADI', 'MDLZ', 'LRCX', 'REGN',
            'PYPL', 'ATVI', 'FISV', 'CSX', 'MRVL', 'ORLY', 'FTNT', 'NXPI'
        ]
    
    def _get_russell2000_sample(self) -> List[str]:
        """Get sample of Russell 2000 small cap stocks"""
        return [
            'IWM', 'SIRI', 'PLUG', 'FCEL', 'RIDE', 'WKHS', 'SKLZ', 'CLOV',
            'SOFI', 'WISH', 'ROOT', 'OPEN', 'RKT', 'UWMC', 'COUR', 'HOOD',
            'RBLX', 'COIN', 'RIVN', 'LCID', 'TTCF', 'BYND', 'CVNA', 'ZM'
        ]
    
    async def scan_market(self, universes: List[str] = None, 
                         filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive market scan for trading opportunities
        
        Returns opportunities ranked by potential and filtered by risk
        """
        
        logger.info("🔍 Starting comprehensive market scan...")
        
        # Select universes to scan
        if universes is None:
            universes = ['sp500', 'nasdaq100', 'growth']
        
        # Combine all symbols from selected universes
        all_symbols = set()
        for universe in universes:
            all_symbols.update(self.stock_universes.get(universe, []))
        
        all_symbols = list(all_symbols)
        logger.info(f"📊 Scanning {len(all_symbols)} stocks across {len(universes)} universes")
        
        # Apply basic filters first to reduce workload
        filtered_symbols = await self._apply_basic_filters(all_symbols)
        logger.info(f"📋 {len(filtered_symbols)} stocks passed basic filters")
        
        # Scan remaining stocks in parallel
        scan_results = await self._parallel_stock_scan(filtered_symbols)
        
        # Apply advanced filters and ranking
        opportunities = self._rank_opportunities(scan_results, filters)
        
        # Generate alerts
        alerts = self._generate_alerts(opportunities)
        
        final_results = {
            'scan_timestamp': datetime.now().isoformat(),
            'total_stocks_scanned': len(all_symbols),
            'stocks_analyzed': len(filtered_symbols),
            'opportunities_found': len(opportunities),
            'top_opportunities': opportunities[:20],  # Top 20
            'alerts': alerts,
            'scan_summary': {
                'strong_buys': len([o for o in opportunities if o['signal'] == 'STRONG_BUY']),
                'buys': len([o for o in opportunities if o['signal'] == 'BUY']),
                'holds': len([o for o in opportunities if o['signal'] == 'HOLD']),
                'sells': len([o for o in opportunities if o['signal'] == 'SELL']),
                'strong_sells': len([o for o in opportunities if o['signal'] == 'STRONG_SELL']),
            },
            'universes_scanned': universes
        }
        
        # Cache results
        self.scan_results = final_results
        
        logger.info(f"✅ Market scan complete - {len(opportunities)} opportunities found")
        
        return final_results
    
    async def _apply_basic_filters(self, symbols: List[str]) -> List[str]:
        """Apply basic filters to reduce the symbol list"""
        
        logger.info("🔍 Applying basic filters...")
        
        filtered_symbols = []
        
        # Process in chunks to avoid overwhelming the API
        chunk_size = 50
        
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]
            
            try:
                # Download basic data for the chunk
                tickers = yf.Tickers(' '.join(chunk))
                
                for symbol in chunk:
                    try:
                        ticker = tickers.tickers[symbol]
                        info = ticker.info
                        
                        # Basic filters
                        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
                        volume = info.get('volume', info.get('regularMarketVolume', 0))
                        market_cap = info.get('marketCap', 0)
                        
                        if (self.config['min_price'] <= current_price <= self.config['max_price'] and
                            volume >= self.config['min_volume'] and
                            market_cap >= self.config['min_market_cap']):
                            
                            filtered_symbols.append(symbol)
                            
                    except Exception as e:
                        logger.debug(f"Basic filter error for {symbol}: {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Chunk processing error: {e}")
                # Add all symbols if filtering fails
                filtered_symbols.extend(chunk)
            
            # Rate limiting
            await asyncio.sleep(0.1)
        
        return filtered_symbols
    
    async def _parallel_stock_scan(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Scan multiple stocks in parallel"""
        
        logger.info(f"🔍 Analyzing {len(symbols)} stocks in parallel...")
        
        semaphore = asyncio.Semaphore(self.config['max_concurrent_requests'])
        
        async def scan_single_stock(symbol: str) -> Optional[Dict[str, Any]]:
            async with semaphore:
                try:
                    return await self._analyze_stock(symbol)
                except Exception as e:
                    logger.debug(f"Analysis error for {symbol}: {e}")
                    return None
        
        # Process all stocks concurrently
        tasks = [scan_single_stock(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out None results and exceptions
        valid_results = [r for r in results if r is not None and not isinstance(r, Exception)]
        
        logger.info(f"✅ Successfully analyzed {len(valid_results)} stocks")
        
        return valid_results
    
    async def _analyze_stock(self, symbol: str) -> Dict[str, Any]:
        """Comprehensive analysis of a single stock"""
        
        # Get stock data
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='6mo', interval='1d')
        
        if hist.empty or len(hist) < 30:
            return None
        
        info = ticker.info
        
        # Basic metrics
        current_price = hist['Close'].iloc[-1]
        volume = hist['Volume'].iloc[-1]
        avg_volume = hist['Volume'].rolling(20).mean().iloc[-1]
        
        # Technical analysis
        technical_analysis = self._technical_analysis(hist)
        
        # Fundamental analysis
        fundamental_analysis = self._fundamental_analysis(info)
        
        # Get prediction if available
        prediction_analysis = {}
        if self.predictor:
            try:
                prediction_result = self.predictor.predict_stock(symbol, 5)  # 5-day prediction
                prediction_analysis = {
                    'expected_return': ((prediction_result['predicted_prices'][-1] - current_price) / current_price),
                    'confidence': prediction_result['confidence_score'],
                    'directional_accuracy': prediction_result['directional_accuracy']
                }
            except:
                prediction_analysis = {
                    'expected_return': 0,
                    'confidence': 0.5,
                    'directional_accuracy': 0.6
                }
        
        # Calculate overall score and signal
        overall_score = self._calculate_overall_score(technical_analysis, fundamental_analysis, prediction_analysis)
        signal = self._determine_signal(overall_score, prediction_analysis)
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'volume': volume,
            'volume_ratio': volume / avg_volume if avg_volume > 0 else 1,
            'market_cap': info.get('marketCap', 0),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'technical': technical_analysis,
            'fundamental': fundamental_analysis,
            'prediction': prediction_analysis,
            'overall_score': overall_score,
            'signal': signal,
            'scan_timestamp': datetime.now().isoformat()
        }
    
    def _technical_analysis(self, hist: pd.DataFrame) -> Dict[str, Any]:
        """Perform technical analysis on stock data"""
        
        if not TECHNICAL_ANALYSIS_AVAILABLE:
            return {'score': 5.0, 'signals': []}
        
        analysis = {'signals': [], 'score': 5.0}
        
        try:
            # Price and volume data
            close = hist['Close']
            high = hist['High']
            low = hist['Low']
            volume = hist['Volume']
            
            # Moving averages
            sma_20 = talib.SMA(close, 20)
            sma_50 = talib.SMA(close, 50)
            ema_12 = talib.EMA(close, 12)
            
            current_price = close.iloc[-1]
            
            # Trend analysis
            if current_price > sma_20.iloc[-1]:
                analysis['signals'].append('Above 20-day SMA')
                analysis['score'] += 0.5
            
            if current_price > sma_50.iloc[-1]:
                analysis['signals'].append('Above 50-day SMA')
                analysis['score'] += 0.5
            
            if sma_20.iloc[-1] > sma_50.iloc[-1]:
                analysis['signals'].append('Golden Cross (20>50 SMA)')
                analysis['score'] += 1.0
            
            # Momentum indicators
            rsi = talib.RSI(close)
            current_rsi = rsi.iloc[-1]
            
            if 30 <= current_rsi <= 70:
                analysis['signals'].append(f'RSI Normal Range ({current_rsi:.1f})')
                analysis['score'] += 0.3
            elif current_rsi < 30:
                analysis['signals'].append(f'RSI Oversold ({current_rsi:.1f})')
                analysis['score'] += 0.8  # Potential bounce
            elif current_rsi > 70:
                analysis['signals'].append(f'RSI Overbought ({current_rsi:.1f})')
                analysis['score'] -= 0.5
            
            # Volume analysis
            volume_sma = talib.SMA(volume, 20)
            current_volume_ratio = volume.iloc[-1] / volume_sma.iloc[-1]
            
            if current_volume_ratio > 1.5:
                analysis['signals'].append(f'High Volume ({current_volume_ratio:.1f}x)')
                analysis['score'] += 0.5
            
            # MACD
            macd, macd_signal, _ = talib.MACD(close)
            if macd.iloc[-1] > macd_signal.iloc[-1]:
                analysis['signals'].append('MACD Bullish')
                analysis['score'] += 0.3
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
            bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
            
            if bb_position < 0.2:
                analysis['signals'].append('Near Lower Bollinger Band')
                analysis['score'] += 0.4
            elif bb_position > 0.8:
                analysis['signals'].append('Near Upper Bollinger Band')
                analysis['score'] -= 0.3
            
            # Price momentum
            momentum_5d = (current_price / close.iloc[-6] - 1) * 100
            momentum_20d = (current_price / close.iloc[-21] - 1) * 100
            
            if momentum_5d > 3:
                analysis['signals'].append(f'Strong 5D Momentum (+{momentum_5d:.1f}%)')
                analysis['score'] += 0.5
            elif momentum_5d < -3:
                analysis['signals'].append(f'Negative 5D Momentum ({momentum_5d:.1f}%)')
                analysis['score'] -= 0.3
            
            # Candlestick patterns
            patterns = {
                'Hammer': talib.CDLHAMMER(hist['Open'], high, low, close),
                'Doji': talib.CDLDOJI(hist['Open'], high, low, close),
                'Engulfing': talib.CDLENGULFING(hist['Open'], high, low, close),
                'Morning Star': talib.CDLMORNINGSTAR(hist['Open'], high, low, close)
            }
            
            for pattern_name, pattern_data in patterns.items():
                if not pattern_data.empty and pattern_data.iloc[-1] != 0:
                    signal_type = 'Bullish' if pattern_data.iloc[-1] > 0 else 'Bearish'
                    analysis['signals'].append(f'{signal_type} {pattern_name}')
                    analysis['score'] += 0.3 if pattern_data.iloc[-1] > 0 else -0.3
            
            # Cap score between 0 and 10
            analysis['score'] = max(0, min(10, analysis['score']))
            
            # Add specific metrics
            analysis['rsi'] = current_rsi
            analysis['volume_ratio'] = current_volume_ratio
            analysis['momentum_5d'] = momentum_5d
            analysis['momentum_20d'] = momentum_20d
            
        except Exception as e:
            logger.debug(f"Technical analysis error: {e}")
            analysis['score'] = 5.0  # Neutral if analysis fails
        
        return analysis
    
    def _fundamental_analysis(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Perform fundamental analysis"""
        
        analysis = {'signals': [], 'score': 5.0}
        
        try:
            # Key fundamental metrics
            pe_ratio = info.get('trailingPE')
            forward_pe = info.get('forwardPE')
            peg_ratio = info.get('pegRatio')
            price_to_book = info.get('priceToBook')
            debt_to_equity = info.get('debtToEquity')
            roe = info.get('returnOnEquity')
            revenue_growth = info.get('revenueGrowth')
            earnings_growth = info.get('earningsGrowth')
            
            # PE Ratio analysis
            if pe_ratio:
                if 10 <= pe_ratio <= 25:
                    analysis['signals'].append(f'Reasonable P/E ({pe_ratio:.1f})')
                    analysis['score'] += 0.3
                elif pe_ratio < 10:
                    analysis['signals'].append(f'Low P/E - Value ({pe_ratio:.1f})')
                    analysis['score'] += 0.5
                elif pe_ratio > 50:
                    analysis['signals'].append(f'High P/E - Growth Premium ({pe_ratio:.1f})')
                    analysis['score'] -= 0.3
            
            # PEG Ratio (Price/Earnings to Growth)
            if peg_ratio and 0.5 <= peg_ratio <= 1.5:
                analysis['signals'].append(f'Good PEG Ratio ({peg_ratio:.1f})')
                analysis['score'] += 0.4
            
            # Return on Equity
            if roe and roe > 0.15:
                analysis['signals'].append(f'Strong ROE ({roe:.1%})')
                analysis['score'] += 0.3
            
            # Revenue Growth
            if revenue_growth:
                if revenue_growth > 0.1:
                    analysis['signals'].append(f'Strong Revenue Growth ({revenue_growth:.1%})')
                    analysis['score'] += 0.5
                elif revenue_growth < -0.1:
                    analysis['signals'].append(f'Declining Revenue ({revenue_growth:.1%})')
                    analysis['score'] -= 0.5
            
            # Earnings Growth
            if earnings_growth:
                if earnings_growth > 0.15:
                    analysis['signals'].append(f'Strong Earnings Growth ({earnings_growth:.1%})')
                    analysis['score'] += 0.5
                elif earnings_growth < -0.1:
                    analysis['signals'].append(f'Declining Earnings ({earnings_growth:.1%})')
                    analysis['score'] -= 0.5
            
            # Debt Analysis
            if debt_to_equity:
                if debt_to_equity < 0.5:
                    analysis['signals'].append('Low Debt Levels')
                    analysis['score'] += 0.2
                elif debt_to_equity > 2.0:
                    analysis['signals'].append('High Debt Levels')
                    analysis['score'] -= 0.3
            
            # Cap score
            analysis['score'] = max(0, min(10, analysis['score']))
            
            # Add specific metrics
            analysis['pe_ratio'] = pe_ratio
            analysis['peg_ratio'] = peg_ratio
            analysis['roe'] = roe
            analysis['revenue_growth'] = revenue_growth
            analysis['earnings_growth'] = earnings_growth
            
        except Exception as e:
            logger.debug(f"Fundamental analysis error: {e}")
        
        return analysis
    
    def _calculate_overall_score(self, technical: Dict, fundamental: Dict, 
                               prediction: Dict) -> float:
        """Calculate overall opportunity score"""
        
        # Weighted combination of different analyses
        weights = {
            'technical': 0.4,
            'fundamental': 0.3,
            'prediction': 0.3
        }
        
        technical_score = technical.get('score', 5.0)
        fundamental_score = fundamental.get('score', 5.0)
        
        # Convert prediction metrics to score
        prediction_score = 5.0
        if prediction:
            expected_return = prediction.get('expected_return', 0)
            confidence = prediction.get('confidence', 0.5)
            
            # Score based on expected return and confidence
            return_score = min(10, 5 + expected_return * 50)  # Scale expected return
            confidence_score = confidence * 10
            prediction_score = (return_score + confidence_score) / 2
        
        overall_score = (
            technical_score * weights['technical'] +
            fundamental_score * weights['fundamental'] +
            prediction_score * weights['prediction']
        )
        
        return max(0, min(10, overall_score))
    
    def _determine_signal(self, overall_score: float, prediction: Dict) -> str:
        """Determine trading signal based on analysis"""
        
        expected_return = prediction.get('expected_return', 0)
        confidence = prediction.get('confidence', 0.5)
        
        # Strong signals require both high score and high confidence
        if overall_score >= 8 and confidence >= 0.8 and expected_return >= self.config['strong_buy_threshold']:
            return 'STRONG_BUY'
        elif overall_score >= 7 and confidence >= 0.7 and expected_return >= self.config['buy_threshold']:
            return 'BUY'
        elif overall_score <= 3 and confidence >= 0.7 and expected_return <= self.config['strong_sell_threshold']:
            return 'STRONG_SELL'
        elif overall_score <= 4 and confidence >= 0.6 and expected_return <= self.config['sell_threshold']:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _rank_opportunities(self, scan_results: List[Dict], filters: Dict = None) -> List[Dict]:
        """Rank and filter opportunities"""
        
        if not scan_results:
            return []
        
        # Apply additional filters if provided
        if filters:
            filtered_results = []
            for result in scan_results:
                if self._passes_custom_filters(result, filters):
                    filtered_results.append(result)
            scan_results = filtered_results
        
        # Sort by overall score descending
        ranked_opportunities = sorted(scan_results, key=lambda x: x['overall_score'], reverse=True)
        
        # Add rank
        for i, opportunity in enumerate(ranked_opportunities, 1):
            opportunity['rank'] = i
        
        return ranked_opportunities
    
    def _passes_custom_filters(self, result: Dict, filters: Dict) -> bool:
        """Check if result passes custom filters"""
        
        # Minimum confidence filter
        if 'min_confidence' in filters:
            if result['prediction']['confidence'] < filters['min_confidence']:
                return False
        
        # Minimum expected return filter
        if 'min_expected_return' in filters:
            if result['prediction']['expected_return'] < filters['min_expected_return']:
                return False
        
        # Sector filter
        if 'sectors' in filters:
            if result['sector'] not in filters['sectors']:
                return False
        
        # Market cap filter
        if 'min_market_cap' in filters:
            if result['market_cap'] < filters['min_market_cap']:
                return False
        
        # Volume filter
        if 'min_volume_ratio' in filters:
            if result['volume_ratio'] < filters['min_volume_ratio']:
                return False
        
        return True
    
    def _generate_alerts(self, opportunities: List[Dict]) -> List[Dict]:
        """Generate trading alerts for top opportunities"""
        
        alerts = []
        
        # Strong buy alerts
        strong_buys = [o for o in opportunities if o['signal'] == 'STRONG_BUY'][:5]
        for opportunity in strong_buys:
            alerts.append({
                'type': 'STRONG_BUY_ALERT',
                'symbol': opportunity['symbol'],
                'current_price': opportunity['current_price'],
                'expected_return': opportunity['prediction']['expected_return'],
                'confidence': opportunity['prediction']['confidence'],
                'overall_score': opportunity['overall_score'],
                'key_signals': opportunity['technical']['signals'][:3],
                'timestamp': datetime.now().isoformat(),
                'priority': 'HIGH'
            })
        
        # Strong sell alerts
        strong_sells = [o for o in opportunities if o['signal'] == 'STRONG_SELL'][:3]
        for opportunity in strong_sells:
            alerts.append({
                'type': 'STRONG_SELL_ALERT',
                'symbol': opportunity['symbol'],
                'current_price': opportunity['current_price'],
                'expected_return': opportunity['prediction']['expected_return'],
                'confidence': opportunity['prediction']['confidence'],
                'overall_score': opportunity['overall_score'],
                'key_signals': opportunity['technical']['signals'][:3],
                'timestamp': datetime.now().isoformat(),
                'priority': 'HIGH'
            })
        
        # Unusual volume alerts
        high_volume = [o for o in opportunities if o.get('volume_ratio', 1) > 2.0][:3]
        for opportunity in high_volume:
            alerts.append({
                'type': 'VOLUME_SPIKE_ALERT',
                'symbol': opportunity['symbol'],
                'volume_ratio': opportunity['volume_ratio'],
                'current_price': opportunity['current_price'],
                'signal': opportunity['signal'],
                'timestamp': datetime.now().isoformat(),
                'priority': 'MEDIUM'
            })
        
        return alerts
    
    def get_watchlist_recommendations(self, portfolio_sectors: List[str] = None, 
                                   risk_tolerance: str = 'moderate') -> List[Dict]:
        """Get personalized watchlist recommendations"""
        
        if not self.scan_results:
            return []
        
        opportunities = self.scan_results.get('top_opportunities', [])
        
        # Filter based on risk tolerance
        risk_filters = {
            'conservative': {'min_confidence': 0.8, 'max_expected_volatility': 0.3},
            'moderate': {'min_confidence': 0.7, 'max_expected_volatility': 0.5},
            'aggressive': {'min_confidence': 0.6, 'max_expected_volatility': 1.0}
        }
        
        filter_config = risk_filters.get(risk_tolerance, risk_filters['moderate'])
        
        # Sector diversification
        if portfolio_sectors:
            underweight_sectors = self._identify_underweight_sectors(opportunities, portfolio_sectors)
            opportunities = [o for o in opportunities if o['sector'] in underweight_sectors]
        
        # Apply risk filters
        filtered_opportunities = []
        for opp in opportunities[:15]:  # Top 15
            confidence = opp['prediction']['confidence']
            if confidence >= filter_config['min_confidence']:
                filtered_opportunities.append(opp)
        
        return filtered_opportunities
    
    def _identify_underweight_sectors(self, opportunities: List[Dict], 
                                    current_sectors: List[str]) -> List[str]:
        """Identify sectors that are underrepresented in current portfolio"""
        
        # Count current sector allocation
        sector_counts = {}
        for sector in current_sectors:
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        # Find underweight sectors from opportunities
        opportunity_sectors = set(o['sector'] for o in opportunities)
        underweight = []
        
        for sector in opportunity_sectors:
            current_weight = sector_counts.get(sector, 0) / len(current_sectors) if current_sectors else 0
            if current_weight < 0.2:  # Less than 20% allocation
                underweight.append(sector)
        
        return underweight if underweight else list(opportunity_sectors)

# Convenience functions for easy usage
async def scan_for_opportunities(universes: List[str] = None) -> Dict[str, Any]:
    """Simple interface to scan for trading opportunities"""
    scanner = ProfessionalMarketScanner()
    return await scanner.scan_market(universes)

async def get_top_picks(count: int = 10) -> List[Dict]:
    """Get top stock picks from latest scan"""
    scanner = ProfessionalMarketScanner()
    results = await scanner.scan_market(['sp500', 'nasdaq100'])
    return results.get('top_opportunities', [])[:count]