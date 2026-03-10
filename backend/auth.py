"""
Authentication middleware for backend API
Protects admin endpoints while allowing frontend access
"""

from fastapi import HTTPException, Security, Depends, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials, APIKeyHeader
import secrets
import os
from typing import Optional

# Security schemes
security = HTTPBasic()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Admin credentials from environment variables
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "bishaladmin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "plbishal3268")

# API key for frontend access (set this in Render environment variables)
FRONTEND_API_KEY = os.getenv("FRONTEND_API_KEY", "bhakundo_frontend_key_2026")

def verify_admin_credentials(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """
    Verify admin username and password for protected endpoints
    Used for /docs, /redoc, and other admin-only routes
    """
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> bool:
    """
    Verify API key for frontend access
    This allows frontend to access endpoints without basic auth
    """
    if api_key and secrets.compare_digest(api_key, FRONTEND_API_KEY):
        return True
    return False

def get_current_user(
    request: Request,
    api_key: Optional[str] = Security(api_key_header),
    credentials: Optional[HTTPBasicCredentials] = Depends(security)
) -> str:
    """
    Flexible authentication: Accept either API key (for frontend) or admin credentials
    """
    # Check if API key is valid (frontend access)
    if api_key and secrets.compare_digest(api_key, FRONTEND_API_KEY):
        return "frontend"
    
    # Check if admin credentials are valid
    if credentials:
        correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
        correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
        if correct_username and correct_password:
            return "admin"
    
    # Neither valid, raise error
    raise HTTPException(
        status_code=401,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Basic"},
    )

# Optional: Rate limiting helper
class RateLimiter:
    def __init__(self):
        self.requests = {}
    
    def check_rate_limit(self, identifier: str, max_requests: int = 100, window_seconds: int = 60):
        """Simple rate limiting check"""
        from datetime import datetime, timedelta
        
        now = datetime.now()
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        # Clean old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if now - req_time < timedelta(seconds=window_seconds)
        ]
        
        # Check limit
        if len(self.requests[identifier]) >= max_requests:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {max_requests} requests per {window_seconds} seconds"
            )
        
        # Add current request
        self.requests[identifier].append(now)

rate_limiter = RateLimiter()
