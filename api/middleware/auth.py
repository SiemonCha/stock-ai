"""
Authentication middleware
"""

import os
import jwt
import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

class AuthMiddleware(BaseHTTPMiddleware):
    """JWT Authentication middleware"""
    
    def __init__(self, app, secret_key: Optional[str] = None, algorithm: str = "HS256"):
        super().__init__(app)
        self.secret_key = secret_key or os.getenv("JWT_SECRET_KEY", "stock-ai-secret-key-change-in-production")
        self.algorithm = algorithm
        
        # Paths that don't require authentication
        self.public_paths = {
            "/", "/health", "/docs", "/redoc", "/openapi.json"
        }
    
    async def dispatch(self, request: Request, call_next):
        """Process request and check authentication"""
        
        # Skip authentication for public paths
        if request.url.path in self.public_paths:
            return await call_next(request)
        
        # Skip authentication in development mode
        if os.getenv("ENVIRONMENT", "development") == "development":
            return await call_next(request)
        
        # Check for API key or JWT token
        auth_header = request.headers.get("Authorization")
        api_key = request.headers.get("X-API-Key")
        
        # API Key authentication (simple)
        if api_key:
            if self._validate_api_key(api_key):
                return await call_next(request)
            else:
                return self._unauthorized_response("Invalid API key")
        
        # JWT Token authentication
        if auth_header:
            try:
                token = auth_header.replace("Bearer ", "")
                payload = self._validate_jwt_token(token)
                
                if payload:
                    # Add user info to request state
                    request.state.user = payload
                    return await call_next(request)
                else:
                    return self._unauthorized_response("Invalid token")
                    
            except Exception as e:
                logger.warning(f"Auth error: {str(e)}")
                return self._unauthorized_response("Authentication failed")
        
        # No authentication provided
        return self._unauthorized_response("Authentication required")
    
    def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key"""
        # In production, this would check against a database
        valid_api_keys = [
            os.getenv("STOCK_AI_API_KEY", "demo-api-key"),
            "demo-api-key"  # For demo purposes
        ]
        
        return api_key in valid_api_keys
    
    def _validate_jwt_token(self, token: str) -> Optional[dict]:
        """Validate JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.utcnow().timestamp() > exp:
                return None
            
            return payload
            
        except jwt.InvalidTokenError:
            return None
    
    def _unauthorized_response(self, message: str) -> JSONResponse:
        """Return unauthorized response"""
        return JSONResponse(
            status_code=401,
            content={
                "error": "Unauthorized",
                "message": message,
                "timestamp": datetime.now().isoformat()
            }
        )

def create_jwt_token(user_data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT token for user"""
    secret_key = os.getenv("JWT_SECRET_KEY", "stock-ai-secret-key-change-in-production")
    
    to_encode = user_data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=24)
    
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm="HS256")
    return encoded_jwt

# Demo token generation endpoint (for testing)
def get_demo_token() -> str:
    """Generate demo token for testing"""
    user_data = {
        "user_id": "demo-user",
        "email": "demo@stockai.com",
        "plan": "demo"
    }
    
    return create_jwt_token(user_data, expires_delta=timedelta(hours=24))