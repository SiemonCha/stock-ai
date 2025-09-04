"""
Advanced API Rate Limiting and Failover Management System
Professional-grade API management with circuit breakers and intelligent routing
"""

import asyncio
import aiohttp
from aiohttp import ClientTimeout, ClientError
import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from collections import deque, defaultdict

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Circuit breaker tripped
    HALF_OPEN = "half_open"  # Testing if service is back

class APIProvider(Enum):
    """Supported API providers"""
    YFINANCE = "yfinance"
    ALPHA_VANTAGE = "alpha_vantage"
    POLYGON = "polygon"
    FINNHUB = "finnhub"
    IEX = "iex"
    TIINGO = "tiingo"
    QUANDL = "quandl"

@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests_per_minute: int
    requests_per_hour: int = None
    requests_per_day: int = None
    burst_limit: int = None
    concurrent_limit: int = 10

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    success_threshold: int = 3  # successes needed to close circuit
    timeout: float = 30.0

@dataclass
class APIEndpoint:
    """API endpoint configuration"""
    provider: APIProvider
    base_url: str
    api_key: Optional[str] = None
    priority: int = 1  # 1 = highest priority
    rate_limit: RateLimitConfig = field(default_factory=lambda: RateLimitConfig(100))
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0
    retry_config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class APIRequest:
    """API request specification"""
    endpoint: str
    method: str = "GET"
    params: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    data: Optional[Any] = None
    timeout: Optional[float] = None
    priority: int = 1
    retry_attempts: int = 3

@dataclass
class APIResponse:
    """API response with metadata"""
    data: Any
    status_code: int
    provider: APIProvider
    endpoint: str
    response_time: float
    timestamp: datetime
    from_cache: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class TokenBucketLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from the bucket"""
        async with self.lock:
            now = time.time()
            
            # Refill tokens based on time passed
            time_passed = now - self.last_refill
            new_tokens = time_passed * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last_refill = now
            
            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    async def wait_for_tokens(self, tokens: int = 1):
        """Wait until tokens are available"""
        while not await self.acquire(tokens):
            wait_time = tokens / self.refill_rate
            await asyncio.sleep(min(wait_time, 1.0))

class SlidingWindowLimiter:
    """Sliding window rate limiter"""
    
    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests = deque()
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """Check if request can be made within rate limit"""
        async with self.lock:
            now = time.time()
            
            # Remove old requests outside the window
            while self.requests and self.requests[0] <= now - self.window_size:
                self.requests.popleft()
            
            # Check if we can make a request
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            
            return False
    
    async def wait_for_slot(self):
        """Wait for a rate limit slot to become available"""
        while not await self.acquire():
            # Wait until the oldest request expires
            if self.requests:
                wait_time = self.window_size - (time.time() - self.requests[0])
                await asyncio.sleep(max(wait_time, 0.1))
            else:
                await asyncio.sleep(0.1)

class CircuitBreaker:
    """Circuit breaker implementation"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        async with self.lock:
            if self.state == CircuitState.OPEN:
                # Check if we should transition to half-open
                if (time.time() - self.last_failure_time) > self.config.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
            
        except Exception as e:
            await self._on_failure()
            raise e
    
    async def _on_success(self):
        """Handle successful request"""
        async with self.lock:
            self.failure_count = 0
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    logger.info("Circuit breaker CLOSED after successful recovery")
    
    async def _on_failure(self):
        """Handle failed request"""
        async with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")
    
    def get_state(self) -> CircuitState:
        """Get current circuit breaker state"""
        return self.state

class AdvancedAPIManager:
    """
    Advanced API management with rate limiting, failover, and circuit breaking
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        self.endpoints: Dict[APIProvider, APIEndpoint] = {}
        self.rate_limiters: Dict[APIProvider, List] = {}
        self.circuit_breakers: Dict[APIProvider, CircuitBreaker] = {}
        self.session_pool: Dict[APIProvider, aiohttp.ClientSession] = {}
        
        # Performance tracking
        self.performance_stats = defaultdict(lambda: {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'rate_limit_hits': 0,
            'circuit_breaker_trips': 0
        })
        
        # Initialize Redis for distributed rate limiting if available
        self.redis_client = None
        if REDIS_AVAILABLE and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info("Redis client initialized for distributed rate limiting")
            except Exception as e:
                logger.warning(f"Redis initialization failed: {e}")
        
        # Response cache
        self.response_cache: Dict[str, Tuple[APIResponse, datetime]] = {}
        self.cache_ttl = 300  # 5 minutes default TTL
        
        logger.info("AdvancedAPIManager initialized")
    
    def register_endpoint(self, endpoint: APIEndpoint):
        """Register an API endpoint"""
        self.endpoints[endpoint.provider] = endpoint
        
        # Initialize rate limiters
        rate_limiters = []
        
        # Per-minute limiter
        if endpoint.rate_limit.requests_per_minute:
            limiter = TokenBucketLimiter(
                capacity=endpoint.rate_limit.requests_per_minute,
                refill_rate=endpoint.rate_limit.requests_per_minute / 60.0
            )
            rate_limiters.append(('minute', limiter))
        
        # Per-hour limiter
        if endpoint.rate_limit.requests_per_hour:
            limiter = SlidingWindowLimiter(
                window_size=3600,
                max_requests=endpoint.rate_limit.requests_per_hour
            )
            rate_limiters.append(('hour', limiter))
        
        # Per-day limiter
        if endpoint.rate_limit.requests_per_day:
            limiter = SlidingWindowLimiter(
                window_size=86400,
                max_requests=endpoint.rate_limit.requests_per_day
            )
            rate_limiters.append(('day', limiter))
        
        self.rate_limiters[endpoint.provider] = rate_limiters
        
        # Initialize circuit breaker
        self.circuit_breakers[endpoint.provider] = CircuitBreaker(endpoint.circuit_breaker)
        
        # Initialize HTTP session
        timeout = ClientTimeout(total=endpoint.timeout)
        headers = {'User-Agent': 'Professional-Stock-AI/1.0'}
        headers.update(endpoint.headers)
        
        self.session_pool[endpoint.provider] = aiohttp.ClientSession(
            timeout=timeout,
            headers=headers,
            connector=aiohttp.TCPConnector(limit=endpoint.rate_limit.concurrent_limit)
        )
        
        logger.info(f"Registered endpoint: {endpoint.provider.value}")
    
    async def make_request(self, 
                          provider: APIProvider, 
                          request: APIRequest,
                          use_fallback: bool = True) -> APIResponse:
        """Make API request with rate limiting and circuit breaking"""
        
        if provider not in self.endpoints:
            raise ValueError(f"Provider {provider} not registered")
        
        endpoint = self.endpoints[provider]
        
        # Check cache first
        cache_key = self._generate_cache_key(provider, request)
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response
        
        # Try primary provider
        try:
            response = await self._make_single_request(provider, request)
            
            # Cache successful response
            self._cache_response(cache_key, response)
            
            return response
            
        except Exception as e:
            logger.warning(f"Request to {provider.value} failed: {e}")
            
            if not use_fallback:
                raise e
            
            # Try fallback providers
            return await self._try_fallback_providers(request, exclude=[provider])
    
    async def make_batch_requests(self, 
                                 requests: List[Tuple[APIProvider, APIRequest]],
                                 max_concurrent: int = 10) -> List[Optional[APIResponse]]:
        """Make multiple API requests concurrently with rate limiting"""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_request(provider: APIProvider, request: APIRequest):
            async with semaphore:
                try:
                    return await self.make_request(provider, request)
                except Exception as e:
                    logger.error(f"Batch request failed for {provider.value}: {e}")
                    return None
        
        # Create tasks
        tasks = [bounded_request(provider, request) for provider, request in requests]
        
        # Execute all requests
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch request exception: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _make_single_request(self, provider: APIProvider, request: APIRequest) -> APIResponse:
        """Make single API request with all protections"""
        
        endpoint = self.endpoints[provider]
        circuit_breaker = self.circuit_breakers[provider]
        
        # Apply circuit breaker
        async def _protected_request():
            # Apply rate limiting
            await self._apply_rate_limits(provider)
            
            # Make actual request
            return await self._execute_request(provider, request)
        
        start_time = time.time()
        
        try:
            response = await circuit_breaker.call(_protected_request)
            response_time = time.time() - start_time
            
            # Update performance stats
            self._update_stats(provider, success=True, response_time=response_time)
            
            return response
            
        except Exception as e:
            response_time = time.time() - start_time
            self._update_stats(provider, success=False, response_time=response_time)
            
            # Check if it's a rate limit error
            if "rate limit" in str(e).lower() or "429" in str(e):
                self.performance_stats[provider]['rate_limit_hits'] += 1
            
            # Check if circuit breaker tripped
            if circuit_breaker.get_state() == CircuitState.OPEN:
                self.performance_stats[provider]['circuit_breaker_trips'] += 1
            
            raise e
    
    async def _apply_rate_limits(self, provider: APIProvider):
        """Apply all rate limits for provider"""
        
        rate_limiters = self.rate_limiters.get(provider, [])
        
        for limiter_type, limiter in rate_limiters:
            if isinstance(limiter, TokenBucketLimiter):
                await limiter.wait_for_tokens()
            elif isinstance(limiter, SlidingWindowLimiter):
                await limiter.wait_for_slot()
        
        # Apply distributed rate limiting if Redis is available
        if self.redis_client:
            await self._apply_distributed_rate_limit(provider)
    
    async def _apply_distributed_rate_limit(self, provider: APIProvider):
        """Apply distributed rate limiting using Redis"""
        
        endpoint = self.endpoints[provider]
        key = f"rate_limit:{provider.value}"
        
        try:
            # Use Redis sliding window counter
            pipe = self.redis_client.pipeline()
            now = int(time.time())
            window = 60  # 1 minute window
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, now - window)
            
            # Count current requests
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(now): now})
            
            # Set expiration
            pipe.expire(key, window)
            
            results = pipe.execute()
            current_count = results[1]
            
            if current_count >= endpoint.rate_limit.requests_per_minute:
                raise Exception(f"Distributed rate limit exceeded for {provider.value}")
                
        except redis.RedisError as e:
            logger.warning(f"Redis rate limiting error: {e}")
            # Continue without distributed rate limiting
    
    async def _execute_request(self, provider: APIProvider, request: APIRequest) -> APIResponse:
        """Execute the actual HTTP request"""
        
        endpoint = self.endpoints[provider]
        session = self.session_pool[provider]
        
        # Build URL
        url = f"{endpoint.base_url.rstrip('/')}/{request.endpoint.lstrip('/')}"
        
        # Prepare headers
        headers = dict(endpoint.headers)
        headers.update(request.headers)
        
        # Add API key if needed
        if endpoint.api_key:
            if provider == APIProvider.ALPHA_VANTAGE:
                request.params['apikey'] = endpoint.api_key
            elif provider == APIProvider.POLYGON:
                request.params['apiKey'] = endpoint.api_key
            elif provider == APIProvider.FINNHUB:
                request.params['token'] = endpoint.api_key
            else:
                headers['Authorization'] = f"Bearer {endpoint.api_key}"
        
        # Prepare request parameters
        timeout = request.timeout or endpoint.timeout
        
        start_time = time.time()
        
        try:
            async with session.request(
                method=request.method,
                url=url,
                params=request.params,
                headers=headers,
                json=request.data if request.method != 'GET' else None,
                timeout=ClientTimeout(total=timeout)
            ) as response:
                
                response_time = time.time() - start_time
                
                # Check response status
                if response.status >= 400:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
                
                # Parse response
                content_type = response.headers.get('content-type', '')
                
                if 'application/json' in content_type:
                    data = await response.json()
                else:
                    data = await response.text()
                
                return APIResponse(
                    data=data,
                    status_code=response.status,
                    provider=provider,
                    endpoint=request.endpoint,
                    response_time=response_time,
                    timestamp=datetime.now(),
                    metadata={
                        'headers': dict(response.headers),
                        'url': str(response.url)
                    }
                )
                
        except asyncio.TimeoutError:
            raise Exception(f"Request timeout after {timeout}s")
        except ClientError as e:
            raise Exception(f"Request error: {e}")
    
    async def _try_fallback_providers(self, 
                                    request: APIRequest, 
                                    exclude: List[APIProvider] = None) -> APIResponse:
        """Try fallback providers in priority order"""
        
        exclude = exclude or []
        
        # Get available providers sorted by priority
        available_providers = [
            (provider, endpoint) for provider, endpoint in self.endpoints.items()
            if provider not in exclude and self.circuit_breakers[provider].get_state() != CircuitState.OPEN
        ]
        
        available_providers.sort(key=lambda x: x[1].priority)
        
        for provider, endpoint in available_providers:
            try:
                logger.info(f"Trying fallback provider: {provider.value}")
                response = await self._make_single_request(provider, request)
                
                # Mark as fallback response
                response.metadata['is_fallback'] = True
                response.metadata['original_provider'] = exclude[0] if exclude else None
                
                return response
                
            except Exception as e:
                logger.warning(f"Fallback provider {provider.value} failed: {e}")
                continue
        
        raise Exception("All providers failed")
    
    def _generate_cache_key(self, provider: APIProvider, request: APIRequest) -> str:
        """Generate cache key for request"""
        
        key_data = {
            'provider': provider.value,
            'endpoint': request.endpoint,
            'method': request.method,
            'params': sorted(request.params.items()),
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[APIResponse]:
        """Get cached response if valid"""
        
        if cache_key not in self.response_cache:
            return None
        
        response, cached_time = self.response_cache[cache_key]
        
        # Check if cache is still valid
        if (datetime.now() - cached_time).total_seconds() > self.cache_ttl:
            del self.response_cache[cache_key]
            return None
        
        # Mark as cached
        cached_response = APIResponse(
            data=response.data,
            status_code=response.status_code,
            provider=response.provider,
            endpoint=response.endpoint,
            response_time=0.0,  # Instant from cache
            timestamp=datetime.now(),
            from_cache=True,
            metadata=response.metadata
        )
        
        return cached_response
    
    def _cache_response(self, cache_key: str, response: APIResponse):
        """Cache response"""
        
        # Only cache successful responses
        if response.status_code == 200:
            self.response_cache[cache_key] = (response, datetime.now())
            
            # Limit cache size
            if len(self.response_cache) > 1000:
                # Remove oldest entries
                oldest_keys = sorted(
                    self.response_cache.keys(),
                    key=lambda k: self.response_cache[k][1]
                )[:100]
                
                for key in oldest_keys:
                    del self.response_cache[key]
    
    def _update_stats(self, provider: APIProvider, success: bool, response_time: float):
        """Update performance statistics"""
        
        stats = self.performance_stats[provider]
        stats['total_requests'] += 1
        
        if success:
            stats['successful_requests'] += 1
        else:
            stats['failed_requests'] += 1
        
        # Update average response time
        total_successful = stats['successful_requests']
        if total_successful > 0:
            current_avg = stats['avg_response_time']
            stats['avg_response_time'] = (
                (current_avg * (total_successful - 1) + response_time) / total_successful
            )
    
    def get_performance_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all providers"""
        
        stats_with_health = {}
        
        for provider, stats in self.performance_stats.items():
            circuit_state = self.circuit_breakers[provider].get_state()
            
            # Calculate success rate
            total_requests = stats['total_requests']
            success_rate = (
                stats['successful_requests'] / total_requests * 100
                if total_requests > 0 else 0
            )
            
            # Determine health status
            if circuit_state == CircuitState.OPEN:
                health = "CIRCUIT_OPEN"
            elif success_rate >= 95:
                health = "EXCELLENT"
            elif success_rate >= 85:
                health = "GOOD"
            elif success_rate >= 70:
                health = "FAIR"
            else:
                health = "POOR"
            
            stats_with_health[provider.value] = {
                **dict(stats),
                'success_rate': success_rate,
                'circuit_state': circuit_state.value,
                'health_status': health
            }
        
        return stats_with_health
    
    def clear_cache(self):
        """Clear response cache"""
        self.response_cache.clear()
        logger.info("Response cache cleared")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all registered endpoints"""
        
        health_results = {}
        
        for provider, endpoint in self.endpoints.items():
            try:
                # Simple health check request
                health_request = APIRequest(
                    endpoint="",  # Root endpoint
                    method="GET",
                    timeout=10.0
                )
                
                start_time = time.time()
                response = await self._execute_request(provider, health_request)
                response_time = time.time() - start_time
                
                health_results[provider.value] = {
                    'status': 'healthy',
                    'response_time': response_time,
                    'circuit_state': self.circuit_breakers[provider].get_state().value
                }
                
            except Exception as e:
                health_results[provider.value] = {
                    'status': 'unhealthy',
                    'error': str(e),
                    'circuit_state': self.circuit_breakers[provider].get_state().value
                }
        
        return health_results
    
    async def close(self):
        """Close all resources"""
        
        # Close HTTP sessions
        for session in self.session_pool.values():
            if not session.closed:
                await session.close()
        
        # Close Redis connection
        if self.redis_client:
            try:
                self.redis_client.close()
            except:
                pass
        
        logger.info("APIManager closed")

# Example usage and testing
if __name__ == "__main__":
    print("🌐 Advanced API Management System")
    print("=" * 50)
    
    async def test_api_manager():
        # Initialize API manager
        manager = AdvancedAPIManager()
        
        # Register test endpoints
        yf_endpoint = APIEndpoint(
            provider=APIProvider.YFINANCE,
            base_url="https://query1.finance.yahoo.com",
            priority=1,
            rate_limit=RateLimitConfig(requests_per_minute=200),
            circuit_breaker=CircuitBreakerConfig(failure_threshold=3)
        )
        
        av_endpoint = APIEndpoint(
            provider=APIProvider.ALPHA_VANTAGE,
            base_url="https://www.alphavantage.co",
            priority=2,
            rate_limit=RateLimitConfig(requests_per_minute=5),  # Free tier
            api_key="demo"  # Demo key
        )
        
        manager.register_endpoint(yf_endpoint)
        manager.register_endpoint(av_endpoint)
        
        # Test single request
        request = APIRequest(
            endpoint="v8/finance/chart/AAPL",
            params={'interval': '1d', 'range': '1mo'}
        )
        
        print("📊 Testing single API request...")
        try:
            response = await manager.make_request(APIProvider.YFINANCE, request)
            print(f"✅ Success: {response.status_code}, Time: {response.response_time:.3f}s")
            print(f"   From cache: {response.from_cache}")
        except Exception as e:
            print(f"❌ Request failed: {e}")
        
        # Test batch requests
        print(f"\n📈 Testing batch requests...")
        batch_requests = [
            (APIProvider.YFINANCE, APIRequest("v8/finance/chart/AAPL")),
            (APIProvider.YFINANCE, APIRequest("v8/finance/chart/MSFT")),
            (APIProvider.YFINANCE, APIRequest("v8/finance/chart/GOOGL"))
        ]
        
        batch_responses = await manager.make_batch_requests(batch_requests, max_concurrent=3)
        successful_responses = [r for r in batch_responses if r is not None]
        
        print(f"✅ Batch completed: {len(successful_responses)}/{len(batch_requests)} successful")
        
        # Test performance stats
        print(f"\n📊 Performance Statistics:")
        stats = manager.get_performance_stats()
        
        for provider, provider_stats in stats.items():
            print(f"\n{provider.upper()}:")
            print(f"   Total Requests: {provider_stats['total_requests']}")
            print(f"   Success Rate: {provider_stats['success_rate']:.1f}%")
            print(f"   Avg Response Time: {provider_stats['avg_response_time']:.3f}s")
            print(f"   Health: {provider_stats['health_status']}")
            print(f"   Circuit State: {provider_stats['circuit_state']}")
        
        # Test health check
        print(f"\n🏥 Health Check:")
        health = await manager.health_check()
        
        for provider, health_info in health.items():
            status_emoji = "✅" if health_info['status'] == 'healthy' else "❌"
            print(f"   {status_emoji} {provider}: {health_info['status']}")
        
        # Cleanup
        await manager.close()
    
    # Run test
    asyncio.run(test_api_manager())
    
    print(f"\n🎯 API management system ready!")
    print(f"📋 Features:")
    print(f"   • Multi-level rate limiting")
    print(f"   • Circuit breaker protection")
    print(f"   • Automatic failover")
    print(f"   • Response caching")
    print(f"   • Performance monitoring")
    print(f"   • Health checking")
    print(f"   • Distributed rate limiting (Redis)")