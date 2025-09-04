import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup
import feedparser
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import logging
import json
import re
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

@dataclass
class NewsItem:
    """News item structure"""
    title: str
    content: str
    source: str
    url: str
    timestamp: datetime
    sentiment_score: float
    relevance_score: float
    symbols: List[str]
    keywords: List[str]

@dataclass
class SocialMentionItem:
    """Social media mention structure"""
    platform: str
    content: str
    author: str
    timestamp: datetime
    sentiment_score: float
    engagement_score: float
    symbols: List[str]
    hashtags: List[str]

@dataclass
class EconomicIndicator:
    """Economic indicator structure"""
    indicator_name: str
    value: float
    previous_value: float
    forecast: float
    release_date: datetime
    importance: str  # 'high', 'medium', 'low'
    country: str
    frequency: str  # 'daily', 'weekly', 'monthly', 'quarterly'

class NewsDataProvider:
    """Advanced news data collection and analysis"""
    
    def __init__(self):
        self.news_sources = {
            'yahoo_finance': 'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'bloomberg': 'https://feeds.bloomberg.com/markets/news.rss',
            'reuters': 'http://feeds.reuters.com/reuters/businessNews',
            'cnbc': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
            'marketwatch': 'http://feeds.marketwatch.com/marketwatch/StockstoWatch/',
            'seeking_alpha': 'https://seekingalpha.com/feed.xml'
        }
        
        self.sentiment_keywords = {
            'positive': [
                'gain', 'rise', 'up', 'bull', 'bullish', 'positive', 'strong', 'growth',
                'profit', 'beat', 'exceed', 'outperform', 'surge', 'rally', 'boom',
                'optimistic', 'confident', 'upgrade', 'buy', 'recommend'
            ],
            'negative': [
                'fall', 'drop', 'down', 'bear', 'bearish', 'negative', 'weak', 'loss',
                'miss', 'below', 'decline', 'crash', 'plunge', 'slump', 'pessimistic',
                'concerned', 'downgrade', 'sell', 'warning', 'risk'
            ],
            'volatility': [
                'volatile', 'uncertainty', 'fluctuation', 'swing', 'dramatic', 'sharp',
                'sudden', 'unexpected', 'surprise', 'shock'
            ]
        }
    
    async def fetch_news_data(self, symbols: List[str], hours_back: int = 24) -> List[NewsItem]:
        """Fetch news data for given symbols"""
        all_news = []
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            # Fetch from RSS feeds
            for source, url in self.news_sources.items():
                tasks.append(self._fetch_rss_news(session, source, url, symbols))
            
            # Fetch symbol-specific news
            for symbol in symbols:
                tasks.append(self._fetch_symbol_news(session, symbol))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, list):
                    all_news.extend(result)
                elif isinstance(result, Exception):
                    logging.error(f"News fetch error: {result}")
        
        # Filter by time
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        recent_news = [news for news in all_news if news.timestamp >= cutoff_time]
        
        # Remove duplicates
        unique_news = self._remove_duplicate_news(recent_news)
        
        # Sort by relevance and timestamp
        unique_news.sort(key=lambda x: (x.relevance_score, x.timestamp), reverse=True)
        
        return unique_news[:100]  # Return top 100 most relevant news items
    
    async def _fetch_rss_news(self, session: aiohttp.ClientSession, source: str, url: str, symbols: List[str]) -> List[NewsItem]:
        """Fetch news from RSS feed"""
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    news_items = []
                    for entry in feed.entries[:20]:  # Top 20 items per source
                        # Extract relevant symbols
                        relevant_symbols = self._extract_symbols_from_text(
                            entry.get('title', '') + ' ' + entry.get('summary', ''),
                            symbols
                        )
                        
                        if relevant_symbols:  # Only include if relevant to tracked symbols
                            news_item = NewsItem(
                                title=entry.get('title', ''),
                                content=entry.get('summary', ''),
                                source=source,
                                url=entry.get('link', ''),
                                timestamp=self._parse_date(entry.get('published', '')),
                                sentiment_score=self._calculate_sentiment(entry.get('title', '') + ' ' + entry.get('summary', '')),
                                relevance_score=len(relevant_symbols) / len(symbols),
                                symbols=relevant_symbols,
                                keywords=self._extract_keywords(entry.get('title', '') + ' ' + entry.get('summary', ''))
                            )
                            news_items.append(news_item)
                    
                    return news_items
        
        except Exception as e:
            logging.error(f"Error fetching RSS news from {source}: {e}")
            return []
    
    async def _fetch_symbol_news(self, session: aiohttp.ClientSession, symbol: str) -> List[NewsItem]:
        """Fetch news specific to a symbol"""
        try:
            # Yahoo Finance news API
            yahoo_url = f"https://query1.finance.yahoo.com/v1/finance/search?q={symbol}&quotesCount=0&newsCount=20"
            
            async with session.get(yahoo_url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    news_items = []
                    if 'news' in data:
                        for item in data['news']:
                            news_item = NewsItem(
                                title=item.get('title', ''),
                                content=item.get('summary', ''),
                                source='Yahoo Finance',
                                url=item.get('link', ''),
                                timestamp=datetime.fromtimestamp(item.get('providerPublishTime', 0)),
                                sentiment_score=self._calculate_sentiment(item.get('title', '') + ' ' + item.get('summary', '')),
                                relevance_score=1.0,  # Symbol-specific news is highly relevant
                                symbols=[symbol],
                                keywords=self._extract_keywords(item.get('title', '') + ' ' + item.get('summary', ''))
                            )
                            news_items.append(news_item)
                    
                    return news_items
        
        except Exception as e:
            logging.error(f"Error fetching news for {symbol}: {e}")
            return []
    
    def _extract_symbols_from_text(self, text: str, symbols: List[str]) -> List[str]:
        """Extract relevant symbols from text"""
        relevant_symbols = []
        text_upper = text.upper()
        
        for symbol in symbols:
            if symbol.upper() in text_upper:
                relevant_symbols.append(symbol)
        
        return relevant_symbols
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date from various formats"""
        try:
            # Try different date formats
            for fmt in ['%a, %d %b %Y %H:%M:%S %Z', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d %H:%M:%S']:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            
            # If all formats fail, return current time
            return datetime.now()
            
        except:
            return datetime.now()
    
    def _calculate_sentiment(self, text: str) -> float:
        """Calculate sentiment score from text"""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.sentiment_keywords['positive'] if word in text_lower)
        negative_count = sum(1 for word in self.sentiment_keywords['negative'] if word in text_lower)
        volatility_count = sum(1 for word in self.sentiment_keywords['volatility'] if word in text_lower)
        
        total_sentiment_words = positive_count + negative_count + volatility_count
        
        if total_sentiment_words == 0:
            return 0.0
        
        sentiment = (positive_count - negative_count) / total_sentiment_words
        
        # Adjust for volatility (neutral but important)
        if volatility_count > 0:
            sentiment *= (1 - volatility_count / total_sentiment_words * 0.5)
        
        return np.clip(sentiment, -1.0, 1.0)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Simple keyword extraction (in production, use NLP libraries)
        financial_keywords = [
            'earnings', 'revenue', 'profit', 'loss', 'guidance', 'outlook', 'forecast',
            'merger', 'acquisition', 'ipo', 'dividend', 'split', 'buyback',
            'regulation', 'approval', 'lawsuit', 'investigation', 'partnership',
            'expansion', 'growth', 'debt', 'cash', 'bankruptcy', 'restructuring'
        ]
        
        text_lower = text.lower()
        found_keywords = [keyword for keyword in financial_keywords if keyword in text_lower]
        
        return found_keywords
    
    def _remove_duplicate_news(self, news_items: List[NewsItem]) -> List[NewsItem]:
        """Remove duplicate news items"""
        seen_titles = set()
        unique_news = []
        
        for news in news_items:
            # Use first 50 characters of title as duplicate check
            title_key = news.title[:50].lower()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_news.append(news)
        
        return unique_news

class SocialSentimentProvider:
    """Social media sentiment analysis (simulated)"""
    
    def __init__(self):
        self.platforms = ['twitter', 'reddit', 'stocktwits', 'discord']
        
    async def fetch_social_sentiment(self, symbols: List[str], hours_back: int = 24) -> List[SocialMentionItem]:
        """Fetch social media sentiment for symbols"""
        # In production, integrate with Twitter API, Reddit API, etc.
        # For now, simulate social sentiment data
        
        social_mentions = []
        
        for symbol in symbols:
            # Simulate mentions across platforms
            for platform in self.platforms:
                num_mentions = np.random.randint(5, 50)  # Random number of mentions
                
                for i in range(num_mentions):
                    mention = SocialMentionItem(
                        platform=platform,
                        content=f"Simulated {platform} mention about {symbol}",
                        author=f"user_{np.random.randint(1000, 9999)}",
                        timestamp=datetime.now() - timedelta(
                            hours=np.random.randint(0, hours_back)
                        ),
                        sentiment_score=np.random.normal(0, 0.5),  # Random sentiment
                        engagement_score=np.random.exponential(10),  # Random engagement
                        symbols=[symbol],
                        hashtags=[f"#{symbol}", f"#{symbol.lower()}stocks"]
                    )
                    social_mentions.append(mention)
        
        return social_mentions
    
    def aggregate_social_sentiment(self, mentions: List[SocialMentionItem]) -> Dict[str, Dict]:
        """Aggregate social sentiment by symbol"""
        symbol_sentiment = {}
        
        for mention in mentions:
            for symbol in mention.symbols:
                if symbol not in symbol_sentiment:
                    symbol_sentiment[symbol] = {
                        'mentions': [],
                        'avg_sentiment': 0.0,
                        'total_engagement': 0.0,
                        'platform_breakdown': {},
                        'trending_score': 0.0
                    }
                
                symbol_sentiment[symbol]['mentions'].append(mention)
                symbol_sentiment[symbol]['total_engagement'] += mention.engagement_score
                
                platform = mention.platform
                if platform not in symbol_sentiment[symbol]['platform_breakdown']:
                    symbol_sentiment[symbol]['platform_breakdown'][platform] = {
                        'count': 0, 'sentiment': 0.0
                    }
                
                symbol_sentiment[symbol]['platform_breakdown'][platform]['count'] += 1
                symbol_sentiment[symbol]['platform_breakdown'][platform]['sentiment'] += mention.sentiment_score
        
        # Calculate aggregated metrics
        for symbol, data in symbol_sentiment.items():
            mentions = data['mentions']
            if mentions:
                data['avg_sentiment'] = np.mean([m.sentiment_score for m in mentions])
                
                # Calculate trending score based on recent mentions
                recent_mentions = [m for m in mentions if m.timestamp > datetime.now() - timedelta(hours=6)]
                data['trending_score'] = len(recent_mentions) / len(mentions) if mentions else 0
                
                # Average platform sentiment
                for platform, stats in data['platform_breakdown'].items():
                    if stats['count'] > 0:
                        stats['sentiment'] /= stats['count']
        
        return symbol_sentiment

class EconomicDataProvider:
    """Economic indicators and macro data provider"""
    
    def __init__(self):
        self.indicators = {
            'US': {
                'GDP': {'frequency': 'quarterly', 'importance': 'high'},
                'CPI': {'frequency': 'monthly', 'importance': 'high'},
                'unemployment': {'frequency': 'monthly', 'importance': 'high'},
                'federal_funds_rate': {'frequency': 'monthly', 'importance': 'high'},
                'retail_sales': {'frequency': 'monthly', 'importance': 'medium'},
                'industrial_production': {'frequency': 'monthly', 'importance': 'medium'},
                'consumer_confidence': {'frequency': 'monthly', 'importance': 'medium'},
                'durable_goods': {'frequency': 'monthly', 'importance': 'low'},
                'housing_starts': {'frequency': 'monthly', 'importance': 'low'},
                'pmi_manufacturing': {'frequency': 'monthly', 'importance': 'medium'}
            }
        }
    
    async def fetch_economic_indicators(self, country: str = 'US', days_back: int = 30) -> List[EconomicIndicator]:
        """Fetch recent economic indicators"""
        indicators = []
        
        if country in self.indicators:
            for indicator_name, details in self.indicators[country].items():
                # Simulate economic data (in production, use FRED API, etc.)
                indicator = self._simulate_economic_indicator(
                    indicator_name, details, country, days_back
                )
                indicators.append(indicator)
        
        return indicators
    
    def _simulate_economic_indicator(self, name: str, details: Dict, country: str, days_back: int) -> EconomicIndicator:
        """Simulate economic indicator data"""
        # Base values for different indicators
        base_values = {
            'GDP': 2.5,  # GDP growth %
            'CPI': 3.2,  # CPI %
            'unemployment': 4.1,  # Unemployment %
            'federal_funds_rate': 5.25,  # Fed funds rate %
            'retail_sales': 1.8,  # Retail sales growth %
            'industrial_production': 0.5,  # Industrial production %
            'consumer_confidence': 110.0,  # Consumer confidence index
            'durable_goods': 2.1,  # Durable goods %
            'housing_starts': 1350000,  # Housing starts (units)
            'pmi_manufacturing': 52.3  # PMI Manufacturing index
        }
        
        base_value = base_values.get(name, 50.0)
        current_value = base_value + np.random.normal(0, base_value * 0.05)
        previous_value = current_value + np.random.normal(0, base_value * 0.03)
        forecast = current_value + np.random.normal(0, base_value * 0.02)
        
        # Generate release date based on frequency
        if details['frequency'] == 'monthly':
            release_date = datetime.now() - timedelta(days=np.random.randint(1, 31))
        elif details['frequency'] == 'quarterly':
            release_date = datetime.now() - timedelta(days=np.random.randint(1, 90))
        else:
            release_date = datetime.now() - timedelta(days=np.random.randint(1, days_back))
        
        return EconomicIndicator(
            indicator_name=name,
            value=current_value,
            previous_value=previous_value,
            forecast=forecast,
            release_date=release_date,
            importance=details['importance'],
            country=country,
            frequency=details['frequency']
        )
    
    def calculate_economic_surprise_index(self, indicators: List[EconomicIndicator]) -> float:
        """Calculate economic surprise index"""
        surprises = []
        
        for indicator in indicators:
            if indicator.forecast != 0:
                surprise = (indicator.value - indicator.forecast) / abs(indicator.forecast)
                
                # Weight by importance
                importance_weights = {'high': 3.0, 'medium': 2.0, 'low': 1.0}
                weight = importance_weights.get(indicator.importance, 1.0)
                
                surprises.append(surprise * weight)
        
        return np.mean(surprises) if surprises else 0.0

class AlternativeDataIntegrator:
    """Main class for integrating alternative data sources"""
    
    def __init__(self):
        self.news_provider = NewsDataProvider()
        self.social_provider = SocialSentimentProvider()
        self.economic_provider = EconomicDataProvider()
        
        # Data cache
        self.news_cache = {}
        self.social_cache = {}
        self.economic_cache = {}
        
        # Update intervals (in minutes)
        self.update_intervals = {
            'news': 15,  # Update news every 15 minutes
            'social': 30,  # Update social data every 30 minutes
            'economic': 60  # Update economic data every hour
        }
    
    async def get_comprehensive_alternative_data(self, symbols: List[str]) -> Dict:
        """Get comprehensive alternative data for symbols"""
        # Fetch all data sources concurrently
        tasks = [
            self.news_provider.fetch_news_data(symbols, hours_back=48),
            self.social_provider.fetch_social_sentiment(symbols, hours_back=24),
            self.economic_provider.fetch_economic_indicators('US', days_back=30)
        ]
        
        news_data, social_data, economic_data = await asyncio.gather(*tasks)
        
        # Process and integrate data
        integrated_data = {
            'news_analysis': self._analyze_news_data(news_data, symbols),
            'social_sentiment': self.social_provider.aggregate_social_sentiment(social_data),
            'economic_indicators': self._analyze_economic_data(economic_data),
            'alternative_signals': self._generate_alternative_signals(news_data, social_data, economic_data, symbols),
            'risk_factors': self._identify_risk_factors(news_data, economic_data),
            'timestamp': datetime.now()
        }
        
        return integrated_data
    
    def _analyze_news_data(self, news_data: List[NewsItem], symbols: List[str]) -> Dict:
        """Analyze news data for insights"""
        analysis = {}
        
        for symbol in symbols:
            symbol_news = [news for news in news_data if symbol in news.symbols]
            
            if symbol_news:
                analysis[symbol] = {
                    'news_count': len(symbol_news),
                    'avg_sentiment': np.mean([news.sentiment_score for news in symbol_news]),
                    'sentiment_consistency': 1 - np.std([news.sentiment_score for news in symbol_news]),
                    'recent_news_trend': self._calculate_news_trend(symbol_news),
                    'key_themes': self._extract_news_themes(symbol_news),
                    'source_diversity': len(set([news.source for news in symbol_news])),
                    'latest_news': {
                        'title': symbol_news[0].title,
                        'sentiment': symbol_news[0].sentiment_score,
                        'timestamp': symbol_news[0].timestamp
                    } if symbol_news else None
                }
            else:
                analysis[symbol] = {
                    'news_count': 0,
                    'avg_sentiment': 0.0,
                    'sentiment_consistency': 0.0,
                    'recent_news_trend': 0.0,
                    'key_themes': [],
                    'source_diversity': 0,
                    'latest_news': None
                }
        
        return analysis
    
    def _calculate_news_trend(self, news_items: List[NewsItem]) -> float:
        """Calculate news sentiment trend"""
        if len(news_items) < 2:
            return 0.0
        
        # Sort by timestamp
        sorted_news = sorted(news_items, key=lambda x: x.timestamp)
        
        # Calculate trend using linear regression
        sentiments = [news.sentiment_score for news in sorted_news]
        x_values = list(range(len(sentiments)))
        
        if len(x_values) > 1:
            trend = np.polyfit(x_values, sentiments, 1)[0]
            return float(trend)
        
        return 0.0
    
    def _extract_news_themes(self, news_items: List[NewsItem]) -> List[str]:
        """Extract key themes from news"""
        all_keywords = []
        for news in news_items:
            all_keywords.extend(news.keywords)
        
        # Count keyword frequency
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        # Return top themes
        sorted_themes = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        return [theme[0] for theme in sorted_themes[:5]]
    
    def _analyze_economic_data(self, economic_data: List[EconomicIndicator]) -> Dict:
        """Analyze economic indicators"""
        analysis = {
            'surprise_index': self.economic_provider.calculate_economic_surprise_index(economic_data),
            'indicators_by_importance': {
                'high': [],
                'medium': [],
                'low': []
            },
            'recent_releases': [],
            'upcoming_releases': []
        }
        
        # Categorize indicators
        for indicator in economic_data:
            analysis['indicators_by_importance'][indicator.importance].append({
                'name': indicator.indicator_name,
                'value': indicator.value,
                'previous': indicator.previous_value,
                'forecast': indicator.forecast,
                'surprise': (indicator.value - indicator.forecast) / abs(indicator.forecast) if indicator.forecast != 0 else 0,
                'release_date': indicator.release_date
            })
        
        # Recent releases (last 7 days)
        recent_cutoff = datetime.now() - timedelta(days=7)
        analysis['recent_releases'] = [
            {
                'name': ind.indicator_name,
                'value': ind.value,
                'surprise': (ind.value - ind.forecast) / abs(ind.forecast) if ind.forecast != 0 else 0,
                'importance': ind.importance,
                'release_date': ind.release_date
            }
            for ind in economic_data if ind.release_date >= recent_cutoff
        ]
        
        return analysis
    
    def _generate_alternative_signals(self, 
                                     news_data: List[NewsItem], 
                                     social_data: List[SocialMentionItem],
                                     economic_data: List[EconomicIndicator],
                                     symbols: List[str]) -> Dict:
        """Generate trading signals from alternative data"""
        signals = {}
        
        for symbol in symbols:
            symbol_signals = {
                'news_signal': 0.0,
                'social_signal': 0.0,
                'economic_signal': 0.0,
                'combined_signal': 0.0,
                'confidence': 0.0,
                'signal_strength': 'neutral'
            }
            
            # News signal
            symbol_news = [news for news in news_data if symbol in news.symbols]
            if symbol_news:
                recent_news = [news for news in symbol_news if news.timestamp > datetime.now() - timedelta(hours=12)]
                if recent_news:
                    news_sentiment = np.mean([news.sentiment_score for news in recent_news])
                    news_volume = len(recent_news)
                    symbol_signals['news_signal'] = news_sentiment * np.log(1 + news_volume) * 0.1
            
            # Social signal
            symbol_social = [social for social in social_data if symbol in social.symbols]
            if symbol_social:
                recent_social = [social for social in symbol_social if social.timestamp > datetime.now() - timedelta(hours=6)]
                if recent_social:
                    social_sentiment = np.mean([social.sentiment_score for social in recent_social])
                    social_engagement = np.mean([social.engagement_score for social in recent_social])
                    symbol_signals['social_signal'] = social_sentiment * np.log(1 + social_engagement) * 0.05
            
            # Economic signal (market-wide impact)
            high_importance_indicators = [ind for ind in economic_data if ind.importance == 'high']
            if high_importance_indicators:
                surprise_index = self.economic_provider.calculate_economic_surprise_index(high_importance_indicators)
                symbol_signals['economic_signal'] = surprise_index * 0.1
            
            # Combined signal
            weights = [0.5, 0.3, 0.2]  # news, social, economic
            signal_components = [
                symbol_signals['news_signal'],
                symbol_signals['social_signal'],
                symbol_signals['economic_signal']
            ]
            
            symbol_signals['combined_signal'] = np.dot(weights, signal_components)
            
            # Calculate confidence based on data availability and consistency
            data_availability = sum([
                1 if abs(symbol_signals['news_signal']) > 0.01 else 0,
                1 if abs(symbol_signals['social_signal']) > 0.01 else 0,
                1 if abs(symbol_signals['economic_signal']) > 0.01 else 0
            ]) / 3.0
            
            signal_consistency = 1 - np.std([abs(s) for s in signal_components if s != 0]) if any(signal_components) else 0
            symbol_signals['confidence'] = (data_availability + signal_consistency) / 2.0
            
            # Signal strength classification
            abs_signal = abs(symbol_signals['combined_signal'])
            if abs_signal > 0.1:
                symbol_signals['signal_strength'] = 'strong'
            elif abs_signal > 0.05:
                symbol_signals['signal_strength'] = 'moderate'
            elif abs_signal > 0.02:
                symbol_signals['signal_strength'] = 'weak'
            else:
                symbol_signals['signal_strength'] = 'neutral'
            
            signals[symbol] = symbol_signals
        
        return signals
    
    def _identify_risk_factors(self, news_data: List[NewsItem], economic_data: List[EconomicIndicator]) -> Dict:
        """Identify potential risk factors from alternative data"""
        risk_factors = {
            'news_risks': [],
            'economic_risks': [],
            'overall_risk_level': 'low',
            'risk_score': 0.0
        }
        
        # News-based risks
        negative_news = [news for news in news_data if news.sentiment_score < -0.3]
        high_impact_news = [news for news in news_data if 
                           any(keyword in news.keywords for keyword in 
                               ['lawsuit', 'investigation', 'bankruptcy', 'regulation'])]
        
        for news in negative_news + high_impact_news:
            risk_factors['news_risks'].append({
                'title': news.title,
                'sentiment': news.sentiment_score,
                'symbols': news.symbols,
                'source': news.source,
                'timestamp': news.timestamp
            })
        
        # Economic risks
        negative_surprises = [ind for ind in economic_data if 
                             ind.importance == 'high' and 
                             (ind.value - ind.forecast) / abs(ind.forecast) < -0.02 if ind.forecast != 0]
        
        for indicator in negative_surprises:
            risk_factors['economic_risks'].append({
                'indicator': indicator.indicator_name,
                'surprise': (indicator.value - indicator.forecast) / abs(indicator.forecast),
                'importance': indicator.importance,
                'release_date': indicator.release_date
            })
        
        # Overall risk assessment
        news_risk = len(risk_factors['news_risks']) * 0.1
        economic_risk = len(risk_factors['economic_risks']) * 0.2
        total_risk = news_risk + economic_risk
        
        risk_factors['risk_score'] = min(total_risk, 1.0)
        
        if total_risk > 0.6:
            risk_factors['overall_risk_level'] = 'high'
        elif total_risk > 0.3:
            risk_factors['overall_risk_level'] = 'medium'
        else:
            risk_factors['overall_risk_level'] = 'low'
        
        return risk_factors
    
    async def get_real_time_alternative_feed(self, symbols: List[str]) -> Dict:
        """Get real-time alternative data feed"""
        # This would be used for continuous monitoring
        current_data = await self.get_comprehensive_alternative_data(symbols)
        
        # Add real-time specific metrics
        current_data['real_time_metrics'] = {
            'data_freshness': {
                'news': (datetime.now() - max([news.timestamp for news in current_data.get('news_analysis', {}).values() if news is not None], default=datetime.now())).total_seconds() / 60,
                'social': 'live',  # Simulated real-time
                'economic': 'scheduled'
            },
            'alert_conditions': self._check_alert_conditions(current_data),
            'trending_symbols': self._identify_trending_symbols(current_data, symbols)
        }
        
        return current_data
    
    def _check_alert_conditions(self, data: Dict) -> List[Dict]:
        """Check for alert conditions in alternative data"""
        alerts = []
        
        # Check for extreme sentiment
        for symbol, signals in data.get('alternative_signals', {}).items():
            if abs(signals['combined_signal']) > 0.15 and signals['confidence'] > 0.7:
                alerts.append({
                    'type': 'extreme_sentiment',
                    'symbol': symbol,
                    'signal': signals['combined_signal'],
                    'confidence': signals['confidence'],
                    'message': f"Extreme sentiment signal for {symbol}: {signals['combined_signal']:.3f}"
                })
        
        # Check for high-impact news
        if data.get('risk_factors', {}).get('overall_risk_level') == 'high':
            alerts.append({
                'type': 'high_risk',
                'symbol': 'MARKET',
                'message': f"High risk level detected: {data['risk_factors']['risk_score']:.2f}"
            })
        
        return alerts
    
    def _identify_trending_symbols(self, data: Dict, symbols: List[str]) -> List[Dict]:
        """Identify trending symbols based on alternative data"""
        trending = []
        
        for symbol in symbols:
            # Social trending
            social_data = data.get('social_sentiment', {}).get(symbol, {})
            social_trend_score = social_data.get('trending_score', 0)
            
            # News volume
            news_data = data.get('news_analysis', {}).get(symbol, {})
            news_count = news_data.get('news_count', 0)
            
            # Combined trending score
            trend_score = social_trend_score * 0.6 + min(news_count / 10, 1.0) * 0.4
            
            if trend_score > 0.5:
                trending.append({
                    'symbol': symbol,
                    'trend_score': trend_score,
                    'drivers': {
                        'social': social_trend_score,
                        'news': news_count
                    }
                })
        
        return sorted(trending, key=lambda x: x['trend_score'], reverse=True)
    
    def get_data_quality_metrics(self) -> Dict:
        """Get data quality and coverage metrics"""
        return {
            'news_sources': len(self.news_provider.news_sources),
            'social_platforms': len(self.social_provider.platforms),
            'economic_indicators': sum(len(indicators) for indicators in self.economic_provider.indicators.values()),
            'last_update': datetime.now(),
            'data_coverage': {
                'news': 'comprehensive',
                'social': 'simulated',  # In production, would be real
                'economic': 'US_focused'
            }
        }