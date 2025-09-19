"""
Cache Manager for GitHub Models Backend

Provides caching functionality to optimize API calls and reduce costs.
"""

import json
import hashlib
import time
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cache entry with metadata."""
    data: Any
    timestamp: float
    ttl_seconds: int
    hit_count: int = 0
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return time.time() - self.timestamp > self.ttl_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "data": self.data,
            "timestamp": self.timestamp,
            "ttl_seconds": self.ttl_seconds,
            "hit_count": self.hit_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary."""
        return cls(
            data=data["data"],
            timestamp=data["timestamp"],
            ttl_seconds=data["ttl_seconds"],
            hit_count=data.get("hit_count", 0)
        )


class CacheManager:
    """
    Cache manager for GitHub Models API responses.
    
    Features:
    - In-memory caching with TTL
    - Request deduplication
    - Cache statistics
    - Automatic cleanup
    """
    
    def __init__(self, default_ttl: int = 3600):  # 1 hour default
        self.cache: Dict[str, CacheEntry] = {}
        self.default_ttl = default_ttl
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0
        }
    
    def _generate_key(self, model_id: str, messages: list, parameters: Dict[str, Any]) -> str:
        """Generate a cache key from request parameters."""
        # Create a deterministic key from the request
        key_data = {
            "model_id": model_id,
            "messages": messages,
            "parameters": parameters
        }
        
        # Sort parameters for consistent hashing
        if "parameters" in key_data:
            key_data["parameters"] = dict(sorted(key_data["parameters"].items()))
        
        # Create hash
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, model_id: str, messages: list, parameters: Dict[str, Any]) -> Optional[Any]:
        """
        Get cached response for a request.
        
        Args:
            model_id: Model identifier
            messages: List of messages
            parameters: Request parameters
            
        Returns:
            Cached response or None if not found/expired
        """
        key = self._generate_key(model_id, messages, parameters)
        self.stats["total_requests"] += 1
        
        if key in self.cache:
            entry = self.cache[key]
            
            if entry.is_expired():
                # Remove expired entry
                del self.cache[key]
                self.stats["evictions"] += 1
                self.stats["misses"] += 1
                logger.debug(f"Cache miss (expired): {key[:8]}...")
                return None
            
            # Update hit count and return data
            entry.hit_count += 1
            self.stats["hits"] += 1
            logger.debug(f"Cache hit: {key[:8]}... (hit count: {entry.hit_count})")
            return entry.data
        
        self.stats["misses"] += 1
        logger.debug(f"Cache miss: {key[:8]}...")
        return None
    
    def set(self, model_id: str, messages: list, parameters: Dict[str, Any], 
            data: Any, ttl_seconds: Optional[int] = None) -> None:
        """
        Cache a response.
        
        Args:
            model_id: Model identifier
            messages: List of messages
            parameters: Request parameters
            data: Response data to cache
            ttl_seconds: Time to live in seconds (uses default if None)
        """
        key = self._generate_key(model_id, messages, parameters)
        ttl = ttl_seconds or self.default_ttl
        
        entry = CacheEntry(
            data=data,
            timestamp=time.time(),
            ttl_seconds=ttl
        )
        
        self.cache[key] = entry
        logger.debug(f"Cached response: {key[:8]}... (TTL: {ttl}s)")
    
    def invalidate(self, model_id: str, messages: list, parameters: Dict[str, Any]) -> bool:
        """
        Invalidate a specific cache entry.
        
        Returns:
            True if entry was found and removed, False otherwise
        """
        key = self._generate_key(model_id, messages, parameters)
        
        if key in self.cache:
            del self.cache[key]
            logger.debug(f"Invalidated cache entry: {key[:8]}...")
            return True
        
        return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        count = len(self.cache)
        self.cache.clear()
        logger.info(f"Cleared {count} cache entries")
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries from cache.
        
        Returns:
            Number of entries removed
        """
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            del self.cache[key]
            self.stats["evictions"] += 1
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = 0.0
        if self.stats["total_requests"] > 0:
            hit_rate = self.stats["hits"] / self.stats["total_requests"]
        
        return {
            "cache_size": len(self.cache),
            "hit_rate": hit_rate,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "evictions": self.stats["evictions"],
            "total_requests": self.stats["total_requests"]
        }
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        now = time.time()
        entries_info = []
        
        for key, entry in self.cache.items():
            age_seconds = now - entry.timestamp
            remaining_ttl = entry.ttl_seconds - age_seconds
            
            entries_info.append({
                "key": key[:8] + "...",
                "age_seconds": age_seconds,
                "remaining_ttl": max(0, remaining_ttl),
                "hit_count": entry.hit_count,
                "is_expired": entry.is_expired()
            })
        
        return {
            "total_entries": len(self.cache),
            "entries": entries_info,
            "stats": self.get_stats()
        }


# Global cache manager instance
cache_manager = CacheManager()

