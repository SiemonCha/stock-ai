"""
Robust Multi-Source Data Collection System
Professional-grade data collection with failover and redundancy
"""

import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import json
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests

try:
    import alpha_vantage
    from alpha_vantage.timeseries import TimeSeries
    from alpha_vantage.fundamentaldata import FundamentalData
    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    ALPHA_VANTAGE_AVAILABLE = False

try:
    import quandl
    QUANDL_AVAILABLE = True
except ImportError:
    QUANDL_AVAILABLE = False

try:
    import polygon
    from polygon import RESTClient
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False

try:
    import finnhub
    FINNHUB_AVAILABLE = True
except ImportError:
    FINNHUB_AVAILABLE = False

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """Data source configuration"""
    name: str
    priority: int  # 1 = highest priority
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    rate_limit: int = 100  # requests per minute
    timeout: int = 30
    retry_count: int = 3
    active: bool = True
    last_error: Optional[str] = None
    error_count: int = 0
    success_count: int = 0
    avg_response_time: float = 0.0

@dataclass
class DataRequest:
    """Data request specification"""
    symbol: str
    data_type: str  # 'prices', 'fundamentals', 'news', 'options', etc.
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    interval: str = "1d"
    additional_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataResponse:
    """Data response with metadata"""
    data: pd.DataFrame
    source: str
    timestamp: datetime
    symbol: str
    data_type: str
    quality_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class RobustDataCollector:
    """
    Professional-grade data collector with multiple sources and failover
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.sources = {}
        self.session_pool = {}
        self.rate_limiters = {}
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize data sources
        self._initialize_sources()
        
        # Performance tracking
        self.collection_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'failover_count': 0,
            'avg_response_time': 0.0
        }
        
        logger.info(f"RobustDataCollector initialized with {len(self.sources)} sources")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        
        default_config = {
            'sources': {
                'yfinance': {
                    'priority': 1,
                    'rate_limit': 200,
                    'timeout': 30
                },
                'alpha_vantage': {
                    'priority': 2,
                    'rate_limit': 5,  # Free tier limit
                    'timeout': 30,
                    'api_key': None
                },
                'polygon': {
                    'priority': 3,
                    'rate_limit': 60,
                    'timeout': 30,
                    'api_key': None
                },
                'finnhub': {
                    'priority': 4,
                    'rate_limit': 60,
                    'timeout': 30,
                    'api_key': None
                },
                'fred': {
                    'priority': 5,
                    'rate_limit': 120,
                    'timeout': 30,
                    'api_key': None
                }
            },
            'failover': {
                'max_retries': 3,
                'retry_delay': 1.0,
                'circuit_breaker_threshold': 10
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                # Merge with defaults
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def _initialize_sources(self):
        """Initialize all available data sources"""
        
        source_configs = self.config.get('sources', {})
        
        # YFinance (always available)
        if 'yfinance' in source_configs:
            config = source_configs['yfinance']
            self.sources['yfinance'] = DataSource(
                name='yfinance',
                priority=config.get('priority', 1),
                rate_limit=config.get('rate_limit', 200),
                timeout=config.get('timeout', 30)
            )
        
        # Alpha Vantage
        if ALPHA_VANTAGE_AVAILABLE and 'alpha_vantage' in source_configs:
            config = source_configs['alpha_vantage']
            api_key = config.get('api_key') or self._get_env_api_key('ALPHA_VANTAGE_API_KEY')
            if api_key:
                self.sources['alpha_vantage'] = DataSource(
                    name='alpha_vantage',
                    priority=config.get('priority', 2),
                    api_key=api_key,
                    rate_limit=config.get('rate_limit', 5),
                    timeout=config.get('timeout', 30)
                )
        
        # Polygon.io
        if POLYGON_AVAILABLE and 'polygon' in source_configs:
            config = source_configs['polygon']
            api_key = config.get('api_key') or self._get_env_api_key('POLYGON_API_KEY')
            if api_key:
                self.sources['polygon'] = DataSource(
                    name='polygon',
                    priority=config.get('priority', 3),
                    api_key=api_key,
                    rate_limit=config.get('rate_limit', 60),
                    timeout=config.get('timeout', 30)
                )
        
        # Finnhub
        if FINNHUB_AVAILABLE and 'finnhub' in source_configs:
            config = source_configs['finnhub']
            api_key = config.get('api_key') or self._get_env_api_key('FINNHUB_API_KEY')
            if api_key:
                self.sources['finnhub'] = DataSource(
                    name='finnhub',
                    priority=config.get('priority', 4),
                    api_key=api_key,
                    rate_limit=config.get('rate_limit', 60),
                    timeout=config.get('timeout', 30)
                )
        
        # FRED
        if FRED_AVAILABLE and 'fred' in source_configs:
            config = source_configs['fred']
            api_key = config.get('api_key') or self._get_env_api_key('FRED_API_KEY')
            if api_key:
                self.sources['fred'] = DataSource(
                    name='fred',
                    priority=config.get('priority', 5),
                    api_key=api_key,
                    rate_limit=config.get('rate_limit', 120),
                    timeout=config.get('timeout', 30)
                )
        
        # Initialize HTTP sessions with retry logic
        self._setup_http_sessions()
        
        # Initialize rate limiters
        self._setup_rate_limiters()
        
        logger.info(f"Initialized {len(self.sources)} data sources")
    
    def _get_env_api_key(self, key_name: str) -> Optional[str]:
        """Get API key from environment variables"""
        import os
        return os.getenv(key_name)
    
    def _setup_http_sessions(self):
        """Setup HTTP sessions with retry logic and connection pooling"""
        
        for source_name, source in self.sources.items():
            session = requests.Session()
            
            # Configure retry strategy
            retry_strategy = Retry(
                total=source.retry_count,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "OPTIONS"]
            )
            
            adapter = HTTPAdapter(
                max_retries=retry_strategy,
                pool_connections=10,
                pool_maxsize=20
            )
            
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            # Set timeout
            session.timeout = source.timeout
            
            self.session_pool[source_name] = session
    
    def _setup_rate_limiters(self):
        """Setup rate limiters for each data source"""
        
        for source_name, source in self.sources.items():
            self.rate_limiters[source_name] = RateLimiter(
                calls=source.rate_limit,
                period=60  # per minute
            )
    
    async def collect_data(self, request: DataRequest) -> Optional[DataResponse]:
        """Collect data with automatic failover"""
        
        self.collection_stats['total_requests'] += 1
        
        # Get sources sorted by priority
        available_sources = [
            (name, source) for name, source in self.sources.items()
            if source.active and self._can_handle_request(source, request)
        ]
        available_sources.sort(key=lambda x: x[1].priority)
        
        if not available_sources:
            logger.error(f"No available sources for {request.symbol} {request.data_type}")
            self.collection_stats['failed_requests'] += 1
            return None
        
        # Try each source in priority order
        for source_name, source in available_sources:
            try:
                logger.info(f"Attempting to collect {request.symbol} from {source_name}")
                
                # Check rate limit
                await self.rate_limiters[source_name].acquire()
                
                start_time = time.time()
                response = await self._collect_from_source(source_name, request)
                
                if response:
                    # Update source statistics
                    elapsed_time = time.time() - start_time
                    self._update_source_stats(source_name, True, elapsed_time)
                    
                    self.collection_stats['successful_requests'] += 1
                    logger.info(f"Successfully collected {request.symbol} from {source_name}")
                    
                    return response
                
            except Exception as e:
                logger.warning(f"Failed to collect from {source_name}: {e}")
                self._update_source_stats(source_name, False, 0)
                
                # Check if we should disable this source temporarily
                if source.error_count > self.config['failover']['circuit_breaker_threshold']:
                    logger.warning(f"Disabling {source_name} due to high error rate")
                    source.active = False
                
                self.collection_stats['failover_count'] += 1
                continue
        
        logger.error(f"All sources failed for {request.symbol} {request.data_type}")
        self.collection_stats['failed_requests'] += 1
        return None
    
    async def collect_multiple(self, requests: List[DataRequest]) -> Dict[str, DataResponse]:
        """Collect multiple data requests concurrently"""
        
        logger.info(f"Collecting data for {len(requests)} requests")
        
        # Create async tasks
        tasks = [self.collect_data(req) for req in requests]
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        response_dict = {}
        for request, result in zip(requests, results):
            if isinstance(result, DataResponse):
                key = f"{request.symbol}_{request.data_type}"
                response_dict[key] = result
            elif isinstance(result, Exception):
                logger.error(f"Error collecting {request.symbol}: {result}")
        
        logger.info(f"Successfully collected {len(response_dict)} out of {len(requests)} requests")
        
        return response_dict
    
    def _can_handle_request(self, source: DataSource, request: DataRequest) -> bool:
        """Check if source can handle the request"""
        
        # Define source capabilities
        capabilities = {
            'yfinance': ['prices', 'fundamentals', 'dividends', 'splits'],
            'alpha_vantage': ['prices', 'fundamentals', 'technical', 'economic'],
            'polygon': ['prices', 'fundamentals', 'options', 'forex'],
            'finnhub': ['prices', 'fundamentals', 'news', 'earnings'],
            'fred': ['economic', 'rates']
        }
        
        source_caps = capabilities.get(source.name, [])
        return request.data_type in source_caps
    
    async def _collect_from_source(self, source_name: str, request: DataRequest) -> Optional[DataResponse]:
        """Collect data from specific source"""
        
        if source_name == 'yfinance':
            return await self._collect_yfinance(request)
        elif source_name == 'alpha_vantage':
            return await self._collect_alpha_vantage(request)
        elif source_name == 'polygon':
            return await self._collect_polygon(request)
        elif source_name == 'finnhub':
            return await self._collect_finnhub(request)
        elif source_name == 'fred':
            return await self._collect_fred(request)
        else:
            raise ValueError(f"Unknown source: {source_name}")
    
    async def _collect_yfinance(self, request: DataRequest) -> Optional[DataResponse]:
        """Collect data from Yahoo Finance"""
        
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            def _fetch_yf_data():
                ticker = yf.Ticker(request.symbol)
                
                if request.data_type == 'prices':
                    data = ticker.history(
                        start=request.start_date,
                        end=request.end_date,
                        interval=request.interval
                    )
                elif request.data_type == 'fundamentals':
                    # Get basic info and financials
                    info = ticker.info
                    financials = ticker.financials
                    
                    # Convert to DataFrame format
                    fundamental_data = pd.DataFrame([info])
                    data = fundamental_data
                else:
                    data = pd.DataFrame()
                
                return data
            
            data = await loop.run_in_executor(None, _fetch_yf_data)
            
            if data.empty:
                return None
            
            return DataResponse(
                data=data,
                source='yfinance',
                timestamp=datetime.now(),
                symbol=request.symbol,
                data_type=request.data_type,
                quality_score=0.9,  # YFinance is generally reliable
                metadata={'interval': request.interval}
            )
            
        except Exception as e:
            logger.error(f"YFinance error for {request.symbol}: {e}")
            return None
    
    async def _collect_alpha_vantage(self, request: DataRequest) -> Optional[DataResponse]:
        """Collect data from Alpha Vantage"""
        
        if not ALPHA_VANTAGE_AVAILABLE:
            return None
        
        try:
            source = self.sources['alpha_vantage']
            
            if request.data_type == 'prices':
                ts = TimeSeries(key=source.api_key, output_format='pandas')
                
                if request.interval == '1d':
                    data, meta_data = ts.get_daily_adjusted(
                        symbol=request.symbol,
                        outputsize='full'
                    )
                elif request.interval == '1h':
                    data, meta_data = ts.get_intraday(
                        symbol=request.symbol,
                        interval='60min',
                        outputsize='full'
                    )
                else:
                    data = pd.DataFrame()
                
            elif request.data_type == 'fundamentals':
                fd = FundamentalData(key=source.api_key, output_format='pandas')
                data, meta_data = fd.get_company_overview(symbol=request.symbol)
                
            else:
                data = pd.DataFrame()
            
            if data.empty:
                return None
            
            return DataResponse(
                data=data,
                source='alpha_vantage',
                timestamp=datetime.now(),
                symbol=request.symbol,
                data_type=request.data_type,
                quality_score=0.95,  # Alpha Vantage is very reliable
                metadata={'source_meta': meta_data}
            )
            
        except Exception as e:
            logger.error(f"Alpha Vantage error for {request.symbol}: {e}")
            return None
    
    async def _collect_polygon(self, request: DataRequest) -> Optional[DataResponse]:
        """Collect data from Polygon.io"""
        
        if not POLYGON_AVAILABLE:
            return None
        
        try:
            source = self.sources['polygon']
            client = RESTClient(api_key=source.api_key)
            
            if request.data_type == 'prices':
                # Get aggregates (OHLCV data)
                aggs = client.get_aggs(
                    ticker=request.symbol,
                    multiplier=1,
                    timespan='day',
                    from_=request.start_date.strftime('%Y-%m-%d') if request.start_date else '2023-01-01',
                    to=request.end_date.strftime('%Y-%m-%d') if request.end_date else datetime.now().strftime('%Y-%m-%d')
                )
                
                # Convert to DataFrame
                data_list = []
                for agg in aggs:
                    data_list.append({
                        'Date': datetime.fromtimestamp(agg.timestamp / 1000),
                        'Open': agg.open,
                        'High': agg.high,
                        'Low': agg.low,
                        'Close': agg.close,
                        'Volume': agg.volume
                    })
                
                data = pd.DataFrame(data_list).set_index('Date')
                
            else:
                data = pd.DataFrame()
            
            if data.empty:
                return None
            
            return DataResponse(
                data=data,
                source='polygon',
                timestamp=datetime.now(),
                symbol=request.symbol,
                data_type=request.data_type,
                quality_score=0.92,
                metadata={'timespan': 'day'}
            )
            
        except Exception as e:
            logger.error(f"Polygon error for {request.symbol}: {e}")
            return None
    
    async def _collect_finnhub(self, request: DataRequest) -> Optional[DataResponse]:
        """Collect data from Finnhub"""
        
        if not FINNHUB_AVAILABLE:
            return None
        
        try:
            source = self.sources['finnhub']
            client = finnhub.Client(api_key=source.api_key)
            
            if request.data_type == 'prices':
                # Get stock candles
                from_ts = int(request.start_date.timestamp()) if request.start_date else int((datetime.now() - timedelta(days=365)).timestamp())
                to_ts = int(request.end_date.timestamp()) if request.end_date else int(datetime.now().timestamp())
                
                candles = client.stock_candles(
                    request.symbol, 
                    'D',  # Daily resolution
                    from_ts,
                    to_ts
                )
                
                if candles['s'] == 'ok':
                    data = pd.DataFrame({
                        'Date': [datetime.fromtimestamp(ts) for ts in candles['t']],
                        'Open': candles['o'],
                        'High': candles['h'],
                        'Low': candles['l'],
                        'Close': candles['c'],
                        'Volume': candles['v']
                    }).set_index('Date')
                else:
                    data = pd.DataFrame()
                
            elif request.data_type == 'fundamentals':
                # Get company profile
                profile = client.company_profile2(symbol=request.symbol)
                data = pd.DataFrame([profile])
                
            else:
                data = pd.DataFrame()
            
            if data.empty:
                return None
            
            return DataResponse(
                data=data,
                source='finnhub',
                timestamp=datetime.now(),
                symbol=request.symbol,
                data_type=request.data_type,
                quality_score=0.88,
                metadata={'resolution': 'D'}
            )
            
        except Exception as e:
            logger.error(f"Finnhub error for {request.symbol}: {e}")
            return None
    
    async def _collect_fred(self, request: DataRequest) -> Optional[DataResponse]:
        """Collect data from FRED (Federal Reserve Economic Data)"""
        
        if not FRED_AVAILABLE:
            return None
        
        try:
            source = self.sources['fred']
            fred = Fred(api_key=source.api_key)
            
            if request.data_type == 'economic':
                # Map symbols to FRED series IDs
                fred_series_map = {
                    'GDP': 'GDP',
                    'INFLATION': 'CPIAUCSL',
                    'UNEMPLOYMENT': 'UNRATE',
                    'INTEREST_RATE': 'FEDFUNDS',
                    'VIX': 'VIXCLS'
                }
                
                series_id = fred_series_map.get(request.symbol.upper())
                if not series_id:
                    return None
                
                data = fred.get_series(
                    series_id,
                    start=request.start_date,
                    end=request.end_date
                )
                
                # Convert to DataFrame
                data = pd.DataFrame({
                    'Date': data.index,
                    'Value': data.values
                }).set_index('Date')
                
            else:
                data = pd.DataFrame()
            
            if data.empty:
                return None
            
            return DataResponse(
                data=data,
                source='fred',
                timestamp=datetime.now(),
                symbol=request.symbol,
                data_type=request.data_type,
                quality_score=0.98,  # FRED data is extremely reliable
                metadata={'series_id': series_id}
            )
            
        except Exception as e:
            logger.error(f"FRED error for {request.symbol}: {e}")
            return None
    
    def _update_source_stats(self, source_name: str, success: bool, response_time: float):
        """Update source performance statistics"""
        
        source = self.sources[source_name]
        
        if success:
            source.success_count += 1
            source.error_count = max(0, source.error_count - 1)  # Decay errors on success
            
            # Update average response time
            total_responses = source.success_count
            source.avg_response_time = (
                (source.avg_response_time * (total_responses - 1) + response_time) / total_responses
            )
            
            # Re-enable source if it was disabled
            if not source.active and source.error_count < 5:
                source.active = True
                logger.info(f"Re-enabled source {source_name}")
            
        else:
            source.error_count += 1
            source.last_error = datetime.now().isoformat()
    
    def get_source_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all data sources"""
        
        health_report = {}
        
        for name, source in self.sources.items():
            total_requests = source.success_count + source.error_count
            success_rate = (source.success_count / total_requests * 100) if total_requests > 0 else 0
            
            health_report[name] = {
                'active': source.active,
                'priority': source.priority,
                'success_count': source.success_count,
                'error_count': source.error_count,
                'success_rate': success_rate,
                'avg_response_time': source.avg_response_time,
                'last_error': source.last_error,
                'health_status': self._get_health_status(source)
            }
        
        return health_report
    
    def _get_health_status(self, source: DataSource) -> str:
        """Determine health status of a source"""
        
        if not source.active:
            return 'DISABLED'
        
        total_requests = source.success_count + source.error_count
        if total_requests == 0:
            return 'UNTESTED'
        
        success_rate = source.success_count / total_requests
        
        if success_rate >= 0.95:
            return 'EXCELLENT'
        elif success_rate >= 0.90:
            return 'GOOD'
        elif success_rate >= 0.75:
            return 'FAIR'
        else:
            return 'POOR'
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get overall collection statistics"""
        
        success_rate = (
            self.collection_stats['successful_requests'] / 
            self.collection_stats['total_requests'] * 100
        ) if self.collection_stats['total_requests'] > 0 else 0
        
        return {
            **self.collection_stats,
            'success_rate': success_rate,
            'source_count': len(self.sources),
            'active_sources': sum(1 for s in self.sources.values() if s.active)
        }

class RateLimiter:
    """Async rate limiter for API calls"""
    
    def __init__(self, calls: int, period: int = 60):
        self.calls = calls
        self.period = period
        self.timestamps = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire a rate limit slot"""
        async with self.lock:
            now = time.time()
            
            # Remove old timestamps
            self.timestamps = [ts for ts in self.timestamps if now - ts < self.period]
            
            # Check if we need to wait
            if len(self.timestamps) >= self.calls:
                sleep_time = self.period - (now - self.timestamps[0]) + 0.1
                await asyncio.sleep(sleep_time)
                
                # Remove the oldest timestamp
                self.timestamps.pop(0)
            
            # Add current timestamp
            self.timestamps.append(now)

# Example usage and testing
if __name__ == "__main__":
    print("🏗️ Robust Data Collection System")
    print("=" * 50)
    
    async def test_data_collection():
        # Initialize collector
        collector = RobustDataCollector()
        
        # Test single data request
        request = DataRequest(
            symbol='AAPL',
            data_type='prices',
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            interval='1d'
        )
        
        print(f"📊 Testing single data collection for {request.symbol}...")
        response = await collector.collect_data(request)
        
        if response:
            print(f"✅ Successfully collected {len(response.data)} rows from {response.source}")
            print(f"   Quality Score: {response.quality_score}")
            print(f"   Data columns: {list(response.data.columns)}")
        else:
            print("❌ Failed to collect data")
        
        # Test multiple requests
        requests = [
            DataRequest('AAPL', 'prices', datetime(2024, 1, 1), datetime(2024, 12, 31)),
            DataRequest('MSFT', 'prices', datetime(2024, 1, 1), datetime(2024, 12, 31)),
            DataRequest('GOOGL', 'prices', datetime(2024, 1, 1), datetime(2024, 12, 31))
        ]
        
        print(f"\n📈 Testing multiple data collection...")
        responses = await collector.collect_multiple(requests)
        
        print(f"✅ Collected data for {len(responses)} symbols")
        for key, response in responses.items():
            print(f"   {key}: {len(response.data)} rows from {response.source}")
        
        # Print health report
        print(f"\n🏥 Source Health Report:")
        health = collector.get_source_health()
        for source, stats in health.items():
            print(f"   {source}: {stats['health_status']} ({stats['success_rate']:.1f}% success)")
        
        # Print collection stats
        print(f"\n📊 Collection Statistics:")
        stats = collector.get_collection_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")
    
    # Run test
    asyncio.run(test_data_collection())
    
    print(f"\n✅ Robust data collection system ready!")
    print(f"📋 Features:")
    print(f"   • Multi-source failover")
    print(f"   • Rate limiting")
    print(f"   • Health monitoring")
    print(f"   • Performance tracking")
    print(f"   • Circuit breaker protection")