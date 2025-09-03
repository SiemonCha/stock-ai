"""
Rate limiting middleware
"""

import time
import logging
from collections import defaultdict, deque
from typing import Dict, Tuple

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from datetime import datetime

logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using sliding window"""
    
    def __init__(
        self, 
        app,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_limit: int = 10
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_limit = burst_limit
        
        # Storage for request tracking
        self._requests: Dict[str, deque] = defaultdict(deque)
        self._last_cleanup = time.time()
        
        # Different limits for different endpoints
        self.endpoint_limits = {
            "/api/v1/predictions/": {"rpm": 30, "rph": 500},
            "/api/v1/training/": {"rpm": 5, "rph": 50},
            "/api/v1/analysis/": {"rpm": 20, "rph": 300},
        }
    
    async def dispatch(self, request: Request, call_next):
        """Process request and apply rate limiting"""
        
        # Skip rate limiting for health checks and docs
        if request.url.path in ["/health", "/", "/docs", "/redoc"]:
            return await call_next(request)
        
        # Get client identifier
        client_id = self._get_client_id(request)
        current_time = time.time()
        
        # Clean up old requests periodically
        if current_time - self._last_cleanup > 300:  # 5 minutes
            self._cleanup_old_requests(current_time)
            self._last_cleanup = current_time
        
        # Check rate limits
        if self._is_rate_limited(client_id, request.url.path, current_time):
            return self._rate_limit_response(client_id)
        
        # Record the request
        self._record_request(client_id, current_time)
        
        # Process the request
        response = await call_next(request)
        
        # Add rate limit headers
        self._add_rate_limit_headers(response, client_id, current_time)
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting"""
        # Try API key first
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api_key:{api_key[:8]}..."
        
        # Try JWT token
        auth_header = request.headers.get("Authorization")
        if auth_header and hasattr(request.state, 'user'):
            user_id = request.state.user.get('user_id', 'unknown')
            return f"user:{user_id}"
        
        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        
        return f"ip:{forwarded_for or client_ip}"
    
    def _is_rate_limited(self, client_id: str, path: str, current_time: float) -> bool:
        """Check if client is rate limited"""
        requests = self._requests[client_id]
        
        # Get limits for this endpoint
        limits = self._get_endpoint_limits(path)
        rpm_limit = limits["rpm"]
        rph_limit = limits["rph"]
        
        # Count requests in last minute
        minute_ago = current_time - 60
        minute_requests = sum(1 for req_time in requests if req_time > minute_ago)
        
        # Count requests in last hour
        hour_ago = current_time - 3600
        hour_requests = sum(1 for req_time in requests if req_time > hour_ago)
        
        # Check burst limit (last 10 seconds)
        burst_ago = current_time - 10
        burst_requests = sum(1 for req_time in requests if req_time > burst_ago)
        
        # Apply limits
        if burst_requests >= self.burst_limit:
            logger.warning(f"Burst limit exceeded for {client_id}: {burst_requests}/{self.burst_limit}")
            return True
        
        if minute_requests >= rpm_limit:
            logger.warning(f"Minute limit exceeded for {client_id}: {minute_requests}/{rpm_limit}")
            return True
        
        if hour_requests >= rph_limit:
            logger.warning(f"Hour limit exceeded for {client_id}: {hour_requests}/{rph_limit}")
            return True
        
        return False
    
    def _get_endpoint_limits(self, path: str) -> Dict[str, int]:
        """Get rate limits for specific endpoint"""
        for endpoint_pattern, limits in self.endpoint_limits.items():
            if path.startswith(endpoint_pattern):
                return limits
        
        # Default limits
        return {"rpm": self.requests_per_minute, "rph": self.requests_per_hour}
    
    def _record_request(self, client_id: str, current_time: float):
        """Record a new request"""
        requests = self._requests[client_id]
        requests.append(current_time)
        
        # Keep only requests from last hour
        hour_ago = current_time - 3600
        while requests and requests[0] < hour_ago:
            requests.popleft()
    
    def _cleanup_old_requests(self, current_time: float):
        """Clean up old request records"""
        hour_ago = current_time - 3600
        
        for client_id in list(self._requests.keys()):
            requests = self._requests[client_id]
            
            # Remove old requests
            while requests and requests[0] < hour_ago:
                requests.popleft()
            
            # Remove empty deques
            if not requests:
                del self._requests[client_id]
    
    def _add_rate_limit_headers(self, response: Response, client_id: str, current_time: float):
        """Add rate limit headers to response"""
        try:
            requests = self._requests[client_id]
            
            # Count current requests
            minute_ago = current_time - 60
            hour_ago = current_time - 3600
            
            minute_requests = sum(1 for req_time in requests if req_time > minute_ago)
            hour_requests = sum(1 for req_time in requests if req_time > hour_ago)
            
            # Add headers
            response.headers["X-RateLimit-Limit-Minute"] = str(self.requests_per_minute)
            response.headers["X-RateLimit-Remaining-Minute"] = str(max(0, self.requests_per_minute - minute_requests))
            response.headers["X-RateLimit-Limit-Hour"] = str(self.requests_per_hour)
            response.headers["X-RateLimit-Remaining-Hour"] = str(max(0, self.requests_per_hour - hour_requests))
            response.headers["X-RateLimit-Reset-Minute"] = str(int(current_time + 60))
            response.headers["X-RateLimit-Reset-Hour"] = str(int(current_time + 3600))
            
        except Exception as e:
            logger.error(f"Failed to add rate limit headers: {str(e)}")
    
    def _rate_limit_response(self, client_id: str) -> JSONResponse:
        """Return rate limit exceeded response"""
        logger.info(f"Rate limit exceeded for {client_id}")
        
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate Limit Exceeded",
                "message": "Too many requests. Please slow down.",
                "timestamp": datetime.now().isoformat(),
                "retry_after": 60
            },
            headers={
                "Retry-After": "60",
                "X-RateLimit-Limit": str(self.requests_per_minute),
                "X-RateLimit-Reset": str(int(time.time() + 60))
            }
        )