import asyncio
import aiohttp
import websockets
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import threading
import queue
import time
from typing import Dict, List, Tuple, Optional, Callable
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup
import re
import feedparser
import yfinance as yf

@dataclass
class MarketEvent:
    """Real-time market event structure"""
    timestamp: datetime
    event_type: str  # 'price', 'news', 'volume', 'sentiment', 'economic'
    symbol: str
    data: Dict
    confidence: float
    impact_score: float

class RealTimeDataStream:
    """Real-time data streaming manager"""
    
    def __init__(self, symbols: List[str], update_interval: float = 1.0):
        self.symbols = symbols
        self.update_interval = update_interval
        self.active = False
        self.data_queue = queue.Queue()
        self.subscribers = []
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Data storage
        self.current_prices = {}
        self.volume_data = {}
        self.news_cache = {}
        self.sentiment_data = {}
        
        # Market state tracking
        self.market_state = {
            'volatility': 0.0,
            'trend': 'neutral',
            'volume_surge': False,
            'news_sentiment': 0.0,
            'regime': 'normal'
        }
        
    async def start_streaming(self):
        """Start real-time data streaming"""
        self.active = True
        
        # Start different data streams concurrently
        tasks = [
            asyncio.create_task(self._stream_price_data()),
            asyncio.create_task(self._stream_news_data()),
            asyncio.create_task(self._stream_social_sentiment()),
            asyncio.create_task(self._stream_economic_data()),
            asyncio.create_task(self._process_market_events())
        ]
        
        await asyncio.gather(*tasks)
    
    async def _stream_price_data(self):
        """Stream real-time price and volume data"""
        while self.active:
            try:
                # Simulate real-time price updates (replace with actual API)
                for symbol in self.symbols:
                    # In production, use WebSocket APIs from exchanges
                    price_data = await self._fetch_current_price(symbol)
                    
                    if price_data:
                        event = MarketEvent(
                            timestamp=datetime.now(),
                            event_type='price',
                            symbol=symbol,
                            data=price_data,
                            confidence=0.95,
                            impact_score=self._calculate_price_impact(symbol, price_data)
                        )
                        
                        self.data_queue.put(event)
                        self.current_prices[symbol] = price_data
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logging.error(f"Price streaming error: {e}")
                await asyncio.sleep(5)
    
    async def _fetch_current_price(self, symbol: str) -> Optional[Dict]:
        """Fetch current price data for a symbol"""
        try:
            # Using yfinance for real data (in production, use faster APIs)
            ticker = yf.Ticker(symbol)
            info = ticker.info
            history = ticker.history(period='1d', interval='1m')
            
            if len(history) > 0:
                current = history.iloc[-1]
                return {
                    'symbol': symbol,
                    'price': float(current['Close']),
                    'volume': int(current['Volume']),
                    'high': float(current['High']),
                    'low': float(current['Low']),
                    'change': float(current['Close'] - current['Open']),
                    'change_pct': float((current['Close'] - current['Open']) / current['Open'] * 100),
                    'timestamp': current.name.timestamp()
                }
        except Exception as e:
            logging.error(f"Error fetching price for {symbol}: {e}")
            return None
    
    def _calculate_price_impact(self, symbol: str, price_data: Dict) -> float:
        """Calculate impact score of price movement"""
        if symbol not in self.current_prices:
            return 0.5
        
        prev_data = self.current_prices[symbol]
        price_change = abs(price_data['change_pct'])
        volume_ratio = price_data['volume'] / max(prev_data.get('volume', 1), 1)
        
        # Impact score based on price change and volume
        impact = min(price_change * 0.1 + np.log(volume_ratio) * 0.05, 1.0)
        return max(0.0, impact)
    
    async def _stream_news_data(self):
        """Stream real-time news and events"""
        news_sources = [
            'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'https://feeds.bloomberg.com/markets/news.rss',
            'https://www.cnbc.com/id/100003114/device/rss/rss.html'
        ]
        
        while self.active:
            try:
                for symbol in self.symbols:
                    news_items = await self._fetch_news_for_symbol(symbol)
                    
                    for news in news_items:
                        sentiment = self._analyze_news_sentiment(news['title'] + ' ' + news['description'])
                        
                        event = MarketEvent(
                            timestamp=datetime.now(),
                            event_type='news',
                            symbol=symbol,
                            data={
                                'title': news['title'],
                                'description': news['description'],
                                'sentiment': sentiment,
                                'url': news['url'],
                                'source': news['source']
                            },
                            confidence=0.8,
                            impact_score=abs(sentiment) * 0.7
                        )
                        
                        self.data_queue.put(event)
                
                await asyncio.sleep(300)  # Check news every 5 minutes
                
            except Exception as e:
                logging.error(f"News streaming error: {e}")
                await asyncio.sleep(60)
    
    async def _fetch_news_for_symbol(self, symbol: str) -> List[Dict]:
        """Fetch news articles for a specific symbol"""
        try:
            # Use multiple news sources
            all_news = []
            
            # Yahoo Finance news
            yahoo_url = f"https://query1.finance.yahoo.com/v1/finance/search?q={symbol}&quotesCount=0&newsCount=10"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(yahoo_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'news' in data:
                            for item in data['news'][:5]:  # Top 5 news
                                all_news.append({
                                    'title': item.get('title', ''),
                                    'description': item.get('summary', ''),
                                    'url': item.get('link', ''),
                                    'source': 'Yahoo Finance',
                                    'timestamp': datetime.now()
                                })
            
            # Filter news relevant to the symbol
            relevant_news = []
            for news in all_news:
                if symbol.lower() in news['title'].lower() or symbol.lower() in news['description'].lower():
                    relevant_news.append(news)
            
            return relevant_news[:3]  # Return top 3 most relevant
            
        except Exception as e:
            logging.error(f"Error fetching news for {symbol}: {e}")
            return []
    
    def _analyze_news_sentiment(self, text: str) -> float:
        """Analyze sentiment of news text"""
        # Simple sentiment analysis (in production, use advanced NLP models)
        positive_words = ['gain', 'rise', 'up', 'bull', 'positive', 'strong', 'growth', 'profit', 'beat', 'exceed']
        negative_words = ['fall', 'drop', 'down', 'bear', 'negative', 'weak', 'loss', 'miss', 'below', 'decline']
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        sentiment = (positive_count - negative_count) / (positive_count + negative_count)
        return sentiment
    
    async def _stream_social_sentiment(self):
        """Stream social media sentiment (simulated)"""
        while self.active:
            try:
                for symbol in self.symbols:
                    # Simulate social sentiment data
                    sentiment_score = np.random.normal(0, 0.3)  # Neutral with some variation
                    
                    event = MarketEvent(
                        timestamp=datetime.now(),
                        event_type='sentiment',
                        symbol=symbol,
                        data={
                            'social_sentiment': sentiment_score,
                            'mention_count': np.random.randint(10, 100),
                            'trending': sentiment_score > 0.5 or sentiment_score < -0.5
                        },
                        confidence=0.6,
                        impact_score=abs(sentiment_score) * 0.4
                    )
                    
                    self.data_queue.put(event)
                
                await asyncio.sleep(600)  # Update every 10 minutes
                
            except Exception as e:
                logging.error(f"Social sentiment error: {e}")
                await asyncio.sleep(120)
    
    async def _stream_economic_data(self):
        """Stream economic indicators and events"""
        economic_indicators = [
            'GDP', 'CPI', 'unemployment', 'interest_rate', 'PMI'
        ]
        
        while self.active:
            try:
                # Check for scheduled economic releases
                for indicator in economic_indicators:
                    if self._is_economic_release_day(indicator):
                        impact = np.random.uniform(0.3, 0.9)
                        
                        event = MarketEvent(
                            timestamp=datetime.now(),
                            event_type='economic',
                            symbol='MARKET',  # Market-wide impact
                            data={
                                'indicator': indicator,
                                'expected_impact': impact,
                                'release_time': datetime.now()
                            },
                            confidence=0.9,
                            impact_score=impact
                        )
                        
                        self.data_queue.put(event)
                
                await asyncio.sleep(3600)  # Check hourly
                
            except Exception as e:
                logging.error(f"Economic data error: {e}")
                await asyncio.sleep(600)
    
    def _is_economic_release_day(self, indicator: str) -> bool:
        """Check if today has an economic release"""
        # Simplified: random chance for demonstration
        return np.random.random() < 0.05  # 5% chance per hour
    
    async def _process_market_events(self):
        """Process and analyze market events"""
        while self.active:
            try:
                events_batch = []
                
                # Collect events for batch processing
                while not self.data_queue.empty() and len(events_batch) < 50:
                    try:
                        event = self.data_queue.get_nowait()
                        events_batch.append(event)
                    except queue.Empty:
                        break
                
                if events_batch:
                    await self._analyze_market_events(events_batch)
                    await self._update_market_state(events_batch)
                    await self._notify_subscribers(events_batch)
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logging.error(f"Event processing error: {e}")
                await asyncio.sleep(5)
    
    async def _analyze_market_events(self, events: List[MarketEvent]):
        """Analyze batch of market events for patterns"""
        if not events:
            return
        
        # Group events by type and symbol
        event_groups = {}
        for event in events:
            key = f"{event.symbol}_{event.event_type}"
            if key not in event_groups:
                event_groups[key] = []
            event_groups[key].append(event)
        
        # Analyze patterns
        for key, group in event_groups.items():
            symbol, event_type = key.split('_', 1)
            
            if event_type == 'price':
                await self._analyze_price_patterns(symbol, group)
            elif event_type == 'news':
                await self._analyze_news_patterns(symbol, group)
            elif event_type == 'sentiment':
                await self._analyze_sentiment_patterns(symbol, group)
    
    async def _analyze_price_patterns(self, symbol: str, price_events: List[MarketEvent]):
        """Analyze price movement patterns"""
        if len(price_events) < 3:
            return
        
        prices = [event.data['price'] for event in price_events]
        volumes = [event.data['volume'] for event in price_events]
        
        # Calculate moving statistics
        price_trend = np.polyfit(range(len(prices)), prices, 1)[0]
        price_volatility = np.std(prices) / np.mean(prices)
        volume_surge = max(volumes) > np.mean(volumes) * 2
        
        # Detect patterns
        patterns = {
            'strong_trend': abs(price_trend) > np.mean(prices) * 0.01,
            'high_volatility': price_volatility > 0.02,
            'volume_surge': volume_surge,
            'momentum': price_trend
        }
        
        # Store pattern analysis
        self.current_prices[symbol]['patterns'] = patterns
    
    async def _analyze_news_patterns(self, symbol: str, news_events: List[MarketEvent]):
        """Analyze news sentiment patterns"""
        if not news_events:
            return
        
        sentiments = [event.data['sentiment'] for event in news_events]
        avg_sentiment = np.mean(sentiments)
        sentiment_consistency = 1 - np.std(sentiments) if len(sentiments) > 1 else 1
        
        self.news_cache[symbol] = {
            'avg_sentiment': avg_sentiment,
            'consistency': sentiment_consistency,
            'news_count': len(news_events),
            'latest_news': news_events[-1].data if news_events else None
        }
    
    async def _analyze_sentiment_patterns(self, symbol: str, sentiment_events: List[MarketEvent]):
        """Analyze social sentiment patterns"""
        if not sentiment_events:
            return
        
        sentiments = [event.data['social_sentiment'] for event in sentiment_events]
        mentions = [event.data['mention_count'] for event in sentiment_events]
        
        self.sentiment_data[symbol] = {
            'avg_sentiment': np.mean(sentiments),
            'sentiment_trend': np.polyfit(range(len(sentiments)), sentiments, 1)[0],
            'total_mentions': sum(mentions),
            'trending': any(event.data['trending'] for event in sentiment_events)
        }
    
    async def _update_market_state(self, events: List[MarketEvent]):
        """Update overall market state based on events"""
        if not events:
            return
        
        # Calculate aggregate metrics
        total_impact = sum(event.impact_score for event in events)
        avg_confidence = np.mean([event.confidence for event in events])
        
        # Update market volatility
        price_events = [e for e in events if e.event_type == 'price']
        if price_events:
            price_changes = [abs(e.data['change_pct']) for e in price_events]
            self.market_state['volatility'] = np.mean(price_changes)
        
        # Update trend
        if price_events:
            recent_changes = [e.data['change_pct'] for e in price_events[-10:]]
            avg_change = np.mean(recent_changes)
            
            if avg_change > 0.5:
                self.market_state['trend'] = 'bullish'
            elif avg_change < -0.5:
                self.market_state['trend'] = 'bearish'
            else:
                self.market_state['trend'] = 'neutral'
        
        # Update regime classification
        if self.market_state['volatility'] > 2.0:
            self.market_state['regime'] = 'high_volatility'
        elif total_impact > 5.0:
            self.market_state['regime'] = 'event_driven'
        else:
            self.market_state['regime'] = 'normal'
    
    async def _notify_subscribers(self, events: List[MarketEvent]):
        """Notify subscribers of market events"""
        for subscriber in self.subscribers:
            try:
                if callable(subscriber):
                    await asyncio.get_event_loop().run_in_executor(
                        self.executor, 
                        subscriber, 
                        events, 
                        self.market_state
                    )
            except Exception as e:
                logging.error(f"Subscriber notification error: {e}")
    
    def subscribe(self, callback: Callable):
        """Subscribe to real-time market events"""
        self.subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from market events"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    def get_current_market_state(self) -> Dict:
        """Get current market state snapshot"""
        return {
            'market_state': self.market_state.copy(),
            'current_prices': self.current_prices.copy(),
            'news_sentiment': self.news_cache.copy(),
            'social_sentiment': self.sentiment_data.copy(),
            'timestamp': datetime.now()
        }
    
    def stop_streaming(self):
        """Stop real-time data streaming"""
        self.active = False
        self.executor.shutdown(wait=False)

class MarketIntelligenceEngine:
    """Advanced market intelligence analysis engine"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.data_stream = RealTimeDataStream(symbols)
        self.intelligence_models = {}
        self.alert_thresholds = {
            'price_spike': 5.0,  # 5% price change
            'volume_surge': 3.0,  # 3x normal volume
            'sentiment_extreme': 0.7,  # Strong sentiment
            'news_impact': 0.8  # High impact news
        }
        
        # Historical pattern database
        self.pattern_database = {}
        
        # Subscribe to real-time events
        self.data_stream.subscribe(self._process_intelligence)
    
    async def start_intelligence(self):
        """Start market intelligence system"""
        # Start real-time data streaming
        intelligence_task = asyncio.create_task(self.data_stream.start_streaming())
        
        # Start intelligence analysis
        analysis_task = asyncio.create_task(self._run_continuous_analysis())
        
        await asyncio.gather(intelligence_task, analysis_task)
    
    async def _run_continuous_analysis(self):
        """Run continuous market intelligence analysis"""
        while True:
            try:
                # Get current market state
                market_state = self.data_stream.get_current_market_state()
                
                # Perform intelligence analysis
                intelligence_report = await self._generate_intelligence_report(market_state)
                
                # Check for alerts
                alerts = self._check_alert_conditions(market_state, intelligence_report)
                
                if alerts:
                    await self._handle_alerts(alerts)
                
                # Update pattern database
                self._update_pattern_database(market_state, intelligence_report)
                
                await asyncio.sleep(30)  # Run analysis every 30 seconds
                
            except Exception as e:
                logging.error(f"Intelligence analysis error: {e}")
                await asyncio.sleep(60)
    
    async def _generate_intelligence_report(self, market_state: Dict) -> Dict:
        """Generate comprehensive market intelligence report"""
        report = {
            'timestamp': datetime.now(),
            'market_regime': self._classify_market_regime(market_state),
            'risk_assessment': self._assess_market_risk(market_state),
            'opportunities': self._identify_opportunities(market_state),
            'predictions': await self._generate_short_term_predictions(market_state),
            'correlation_analysis': self._analyze_cross_asset_correlations(market_state),
            'anomaly_detection': self._detect_market_anomalies(market_state)
        }
        
        return report
    
    def _classify_market_regime(self, market_state: Dict) -> Dict:
        """Classify current market regime"""
        volatility = market_state['market_state']['volatility']
        trend = market_state['market_state']['trend']
        
        # Advanced regime classification
        if volatility > 3.0:
            regime = 'crisis'
            confidence = 0.9
        elif volatility > 1.5:
            if trend in ['bullish', 'bearish']:
                regime = 'trending_volatile'
            else:
                regime = 'high_volatility'
            confidence = 0.8
        elif trend == 'neutral':
            regime = 'consolidation'
            confidence = 0.7
        else:
            regime = f'trending_{trend}'
            confidence = 0.8
        
        return {
            'regime': regime,
            'confidence': confidence,
            'characteristics': self._get_regime_characteristics(regime)
        }
    
    def _get_regime_characteristics(self, regime: str) -> List[str]:
        """Get characteristics of market regime"""
        characteristics_map = {
            'crisis': ['high_volatility', 'flight_to_safety', 'correlation_breakdown'],
            'trending_volatile': ['directional_bias', 'high_volatility', 'momentum_driven'],
            'high_volatility': ['uncertain_direction', 'news_driven', 'risk_averse'],
            'consolidation': ['range_bound', 'low_volatility', 'mean_reversion'],
            'trending_bullish': ['upward_momentum', 'risk_on', 'growth_focus'],
            'trending_bearish': ['downward_pressure', 'risk_off', 'defensive_rotation']
        }
        
        return characteristics_map.get(regime, ['unknown'])
    
    def _assess_market_risk(self, market_state: Dict) -> Dict:
        """Assess current market risk levels"""
        volatility = market_state['market_state']['volatility']
        
        # Calculate risk scores
        volatility_risk = min(volatility / 5.0, 1.0)  # Normalize to 0-1
        
        # Sentiment risk
        sentiment_scores = []
        for symbol_data in market_state.get('news_sentiment', {}).values():
            if 'avg_sentiment' in symbol_data:
                sentiment_scores.append(abs(symbol_data['avg_sentiment']))
        
        sentiment_risk = np.mean(sentiment_scores) if sentiment_scores else 0.5
        
        # Overall risk assessment
        overall_risk = (volatility_risk * 0.6 + sentiment_risk * 0.4)
        
        risk_level = 'low'
        if overall_risk > 0.7:
            risk_level = 'high'
        elif overall_risk > 0.4:
            risk_level = 'medium'
        
        return {
            'overall_risk': overall_risk,
            'risk_level': risk_level,
            'volatility_risk': volatility_risk,
            'sentiment_risk': sentiment_risk,
            'recommendations': self._get_risk_recommendations(risk_level)
        }
    
    def _get_risk_recommendations(self, risk_level: str) -> List[str]:
        """Get risk management recommendations"""
        recommendations_map = {
            'low': ['normal_position_sizing', 'growth_opportunities', 'momentum_strategies'],
            'medium': ['reduced_leverage', 'diversification', 'hedging_consideration'],
            'high': ['defensive_positioning', 'cash_preservation', 'volatility_protection']
        }
        
        return recommendations_map.get(risk_level, [])
    
    def _identify_opportunities(self, market_state: Dict) -> List[Dict]:
        """Identify trading opportunities"""
        opportunities = []
        
        for symbol in self.symbols:
            if symbol in market_state.get('current_prices', {}):
                price_data = market_state['current_prices'][symbol]
                
                # Pattern-based opportunities
                if 'patterns' in price_data:
                    patterns = price_data['patterns']
                    
                    if patterns.get('volume_surge') and patterns.get('strong_trend'):
                        opportunities.append({
                            'symbol': symbol,
                            'type': 'momentum_breakout',
                            'confidence': 0.8,
                            'direction': 'long' if patterns['momentum'] > 0 else 'short',
                            'rationale': 'Strong trend with volume confirmation'
                        })
                    
                    if patterns.get('high_volatility') and not patterns.get('strong_trend'):
                        opportunities.append({
                            'symbol': symbol,
                            'type': 'mean_reversion',
                            'confidence': 0.6,
                            'direction': 'contrarian',
                            'rationale': 'High volatility without clear trend'
                        })
        
        return opportunities
    
    async def _generate_short_term_predictions(self, market_state: Dict) -> Dict:
        """Generate short-term market predictions"""
        predictions = {}
        
        for symbol in self.symbols:
            if symbol in market_state.get('current_prices', {}):
                current_price = market_state['current_prices'][symbol]['price']
                
                # Simple prediction based on momentum and sentiment
                momentum = 0
                if symbol in market_state.get('current_prices', {}):
                    price_data = market_state['current_prices'][symbol]
                    if 'patterns' in price_data:
                        momentum = price_data['patterns'].get('momentum', 0)
                
                sentiment_boost = 0
                if symbol in market_state.get('news_sentiment', {}):
                    sentiment_boost = market_state['news_sentiment'][symbol].get('avg_sentiment', 0) * 0.5
                
                # 1-hour prediction
                prediction_change = (momentum * 0.7 + sentiment_boost * 0.3) * np.random.normal(1, 0.1)
                predicted_price = current_price * (1 + prediction_change / 100)
                
                predictions[symbol] = {
                    'current_price': current_price,
                    'predicted_price_1h': predicted_price,
                    'confidence': 0.6,
                    'factors': {
                        'momentum': momentum,
                        'sentiment': sentiment_boost
                    }
                }
        
        return predictions
    
    def _analyze_cross_asset_correlations(self, market_state: Dict) -> Dict:
        """Analyze correlations between assets"""
        correlations = {}
        prices = []
        symbols = []
        
        # Collect current prices
        for symbol in self.symbols:
            if symbol in market_state.get('current_prices', {}):
                prices.append(market_state['current_prices'][symbol]['price'])
                symbols.append(symbol)
        
        if len(symbols) > 1:
            # Calculate simple correlation (in production, use historical data)
            correlation_matrix = np.corrcoef(prices)
            
            for i, symbol1 in enumerate(symbols):
                correlations[symbol1] = {}
                for j, symbol2 in enumerate(symbols):
                    if i != j:
                        correlations[symbol1][symbol2] = float(correlation_matrix[i, j])
        
        return correlations
    
    def _detect_market_anomalies(self, market_state: Dict) -> List[Dict]:
        """Detect market anomalies"""
        anomalies = []
        
        for symbol in self.symbols:
            if symbol in market_state.get('current_prices', {}):
                price_data = market_state['current_prices'][symbol]
                
                # Check for unusual price movements
                if abs(price_data.get('change_pct', 0)) > 10:
                    anomalies.append({
                        'symbol': symbol,
                        'type': 'extreme_price_move',
                        'severity': 'high',
                        'value': price_data['change_pct'],
                        'description': f"Unusual {price_data['change_pct']:.2f}% price movement"
                    })
                
                # Check for volume anomalies
                if 'patterns' in price_data and price_data['patterns'].get('volume_surge'):
                    anomalies.append({
                        'symbol': symbol,
                        'type': 'volume_anomaly',
                        'severity': 'medium',
                        'description': 'Unusual volume surge detected'
                    })
        
        return anomalies
    
    def _check_alert_conditions(self, market_state: Dict, intelligence_report: Dict) -> List[Dict]:
        """Check for alert conditions"""
        alerts = []
        
        # Price spike alerts
        for symbol in self.symbols:
            if symbol in market_state.get('current_prices', {}):
                change_pct = abs(market_state['current_prices'][symbol].get('change_pct', 0))
                
                if change_pct > self.alert_thresholds['price_spike']:
                    alerts.append({
                        'type': 'price_spike',
                        'symbol': symbol,
                        'severity': 'high',
                        'value': change_pct,
                        'message': f"{symbol} moved {change_pct:.2f}% - exceeds threshold"
                    })
        
        # Risk level alerts
        risk_assessment = intelligence_report.get('risk_assessment', {})
        if risk_assessment.get('risk_level') == 'high':
            alerts.append({
                'type': 'high_risk',
                'symbol': 'MARKET',
                'severity': 'high',
                'message': 'Market risk level elevated to HIGH'
            })
        
        return alerts
    
    async def _handle_alerts(self, alerts: List[Dict]):
        """Handle market alerts"""
        for alert in alerts:
            logging.warning(f"MARKET ALERT: {alert['message']}")
            # In production, send notifications, emails, etc.
    
    def _update_pattern_database(self, market_state: Dict, intelligence_report: Dict):
        """Update historical pattern database"""
        timestamp = datetime.now()
        
        pattern_entry = {
            'timestamp': timestamp,
            'market_state': market_state['market_state'].copy(),
            'regime': intelligence_report['market_regime'],
            'risk_level': intelligence_report['risk_assessment']['risk_level'],
            'opportunities': len(intelligence_report['opportunities']),
            'anomalies': len(intelligence_report['anomaly_detection'])
        }
        
        # Store in pattern database (simplified)
        date_key = timestamp.strftime('%Y-%m-%d')
        if date_key not in self.pattern_database:
            self.pattern_database[date_key] = []
        
        self.pattern_database[date_key].append(pattern_entry)
        
        # Keep only last 30 days
        cutoff_date = (timestamp - timedelta(days=30)).strftime('%Y-%m-%d')
        keys_to_remove = [k for k in self.pattern_database.keys() if k < cutoff_date]
        for key in keys_to_remove:
            del self.pattern_database[key]
    
    def get_intelligence_summary(self) -> Dict:
        """Get current intelligence summary"""
        market_state = self.data_stream.get_current_market_state()
        
        return {
            'market_state': market_state,
            'active_symbols': self.symbols,
            'intelligence_status': 'active' if self.data_stream.active else 'inactive',
            'pattern_database_size': sum(len(patterns) for patterns in self.pattern_database.values()),
            'last_updated': datetime.now()
        }
    
    def _process_intelligence(self, events: List[MarketEvent], market_state: Dict):
        """Process intelligence from market events (sync callback)"""
        # This is called by the data stream for each batch of events
        # Can be used for immediate response to critical events
        
        critical_events = [e for e in events if e.impact_score > 0.8]
        if critical_events:
            logging.info(f"Detected {len(critical_events)} critical market events")
    
    async def stop_intelligence(self):
        """Stop market intelligence system"""
        self.data_stream.stop_streaming()
        logging.info("Market intelligence system stopped")