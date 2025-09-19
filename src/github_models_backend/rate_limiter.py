"""
Rate Limiter for GitHub Models Backend

Provides rate limiting functionality to respect API limits and ensure
fair usage for public showcase applications.
"""

import time
import asyncio
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimit:
    """Rate limit configuration for different API endpoints."""
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_limit: int = 10  # Allow burst of requests


class RateLimiter:
    """
    Rate limiter for managing API requests to prevent exceeding limits.
    
    Features:
    - Per-endpoint rate limiting
    - Burst handling
    - Automatic reset windows
    - Async support
    """
    
    def __init__(self):
        self.requests: Dict[str, list] = {}
        self.limits: Dict[str, RateLimit] = {
            # GitHub Models API limits (conservative for public showcase)
            "github_models": RateLimit(
                requests_per_minute=30,
                requests_per_hour=300,
                requests_per_day=3000,
                burst_limit=10
            ),
            # General model evaluation limits
            "evaluation": RateLimit(
                requests_per_minute=30,
                requests_per_hour=300,
                requests_per_day=3000,
                burst_limit=10
            ),
            # Default limits for unknown endpoints
            "default": RateLimit(
                requests_per_minute=50,
                requests_per_hour=500,
                requests_per_day=5000,
                burst_limit=15
            )
        }
    
    def _get_limit(self, endpoint: str) -> RateLimit:
        """Get rate limit configuration for an endpoint."""
        return self.limits.get(endpoint, self.limits["default"])
    
    def _cleanup_old_requests(self, endpoint: str, window_seconds: int):
        """Remove requests older than the specified window."""
        cutoff_time = time.time() - window_seconds
        self.requests[endpoint] = [
            req_time for req_time in self.requests[endpoint]
            if req_time > cutoff_time
        ]
    
    def _can_make_request(self, endpoint: str) -> tuple[bool, Optional[float]]:
        """
        Check if a request can be made for the given endpoint.
        
        Returns:
            tuple: (can_make_request, wait_time_seconds)
        """
        if endpoint not in self.requests:
            self.requests[endpoint] = []
        
        limit = self._get_limit(endpoint)
        current_time = time.time()
        
        # Clean up old requests
        self._cleanup_old_requests(endpoint, 3600)  # 1 hour window
        
        # Check minute limit
        minute_requests = [
            req_time for req_time in self.requests[endpoint]
            if req_time > current_time - 60
        ]
        
        if len(minute_requests) >= limit.requests_per_minute:
            wait_time = 60 - (current_time - minute_requests[0])
            return False, wait_time
        
        # Check hour limit
        hour_requests = [
            req_time for req_time in self.requests[endpoint]
            if req_time > current_time - 3600
        ]
        
        if len(hour_requests) >= limit.requests_per_hour:
            wait_time = 3600 - (current_time - hour_requests[0])
            return False, wait_time
        
        # Check day limit
        day_requests = [
            req_time for req_time in self.requests[endpoint]
            if req_time > current_time - 86400
        ]
        
        if len(day_requests) >= limit.requests_per_day:
            wait_time = 86400 - (current_time - day_requests[0])
            return False, wait_time
        
        return True, None
    
    def record_request(self, endpoint: str):
        """Record a successful request for rate limiting."""
        if endpoint not in self.requests:
            self.requests[endpoint] = []
        
        self.requests[endpoint].append(time.time())
        logger.debug(f"Recorded request for {endpoint}. Total requests: {len(self.requests[endpoint])}")
    
    def wait_if_needed(self, endpoint: str) -> float:
        """
        Check rate limits and wait if necessary.
        
        Returns:
            float: Time waited in seconds
        """
        can_make, wait_time = self._can_make_request(endpoint)
        
        if not can_make and wait_time:
            logger.info(f"Rate limit reached for {endpoint}. Waiting {wait_time:.2f} seconds...")
            time.sleep(wait_time)
            return wait_time
        
        return 0.0
    
    async def async_wait_if_needed(self, endpoint: str) -> float:
        """
        Async version of wait_if_needed.
        
        Returns:
            float: Time waited in seconds
        """
        can_make, wait_time = self._can_make_request(endpoint)
        
        if not can_make and wait_time:
            logger.info(f"Rate limit reached for {endpoint}. Waiting {wait_time:.2f} seconds...")
            await asyncio.sleep(wait_time)
            return wait_time
        
        return 0.0
    
    def get_status(self, endpoint: str) -> Dict[str, int]:
        """Get current rate limit status for an endpoint."""
        if endpoint not in self.requests:
            return {
                "requests_last_minute": 0,
                "requests_last_hour": 0,
                "requests_last_day": 0,
                "limit_per_minute": self._get_limit(endpoint).requests_per_minute,
                "limit_per_hour": self._get_limit(endpoint).requests_per_hour,
                "limit_per_day": self._get_limit(endpoint).requests_per_day
            }
        
        current_time = time.time()
        
        minute_requests = len([
            req_time for req_time in self.requests[endpoint]
            if req_time > current_time - 60
        ])
        
        hour_requests = len([
            req_time for req_time in self.requests[endpoint]
            if req_time > current_time - 3600
        ])
        
        day_requests = len([
            req_time for req_time in self.requests[endpoint]
            if req_time > current_time - 86400
        ])
        
        limit = self._get_limit(endpoint)
        
        return {
            "requests_last_minute": minute_requests,
            "requests_last_hour": hour_requests,
            "requests_last_day": day_requests,
            "limit_per_minute": limit.requests_per_minute,
            "limit_per_hour": limit.requests_per_hour,
            "limit_per_day": limit.requests_per_day
        }
    
    def reset_endpoint(self, endpoint: str):
        """Reset rate limit counters for a specific endpoint."""
        if endpoint in self.requests:
            del self.requests[endpoint]
            logger.info(f"Reset rate limit counters for {endpoint}")
    
    def reset_all(self):
        """Reset all rate limit counters."""
        self.requests.clear()
        logger.info("Reset all rate limit counters")


# Global rate limiter instance
rate_limiter = RateLimiter()
