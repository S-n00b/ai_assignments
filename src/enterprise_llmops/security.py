"""
Security and Authentication for Enterprise LLMOps Platform

This module provides comprehensive security, authentication, and authorization
capabilities for the enterprise LLM operations platform.

Key Features:
- JWT-based authentication
- Role-based access control (RBAC)
- API key management
- Rate limiting and DDoS protection
- Input validation and sanitization
- Audit logging and compliance
"""

import asyncio
import logging
import time
import hashlib
import secrets
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis
from pydantic import BaseModel, EmailStr
import re


@dataclass
class User:
    """User model for authentication."""
    user_id: str
    username: str
    email: str
    password_hash: str
    roles: List[str]
    permissions: List[str]
    is_active: bool = True
    created_at: datetime = None
    last_login: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class APIKey:
    """API key model."""
    key_id: str
    key_hash: str
    user_id: str
    name: str
    permissions: List[str]
    rate_limit: int
    expires_at: Optional[datetime] = None
    is_active: bool = True
    created_at: datetime = None
    last_used: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class SecurityConfig:
    """Security configuration."""
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    password_min_length: int = 8
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    rate_limit_requests_per_minute: int = 60
    api_key_length: int = 32


class EnterpriseSecurity:
    """
    Comprehensive security manager for Enterprise LLMOps platform.
    
    This class provides authentication, authorization, and security features
    including JWT tokens, API keys, rate limiting, and audit logging.
    """
    
    def __init__(self, config: SecurityConfig):
        """Initialize the security manager."""
        self.config = config
        self.logger = self._setup_logging()
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.redis_client = None
        self.users = {}
        self.api_keys = {}
        
        # Initialize Redis for session management
        self._init_redis()
        
        # Load default admin user
        self._create_default_admin()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for security."""
        logger = logging.getLogger("security")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _init_redis(self):
        """Initialize Redis connection for session management."""
        try:
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                db=1,  # Use different DB for security
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            self.logger.info("Redis connection established for security")
        except Exception as e:
            self.logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
    
    def _create_default_admin(self):
        """Create default admin user."""
        admin_user = User(
            user_id="admin",
            username="admin",
            email="admin@enterprise.com",
            password_hash=self.pwd_context.hash("admin123"),
            roles=["admin", "user"],
            permissions=["*"]  # All permissions
        )
        self.users["admin"] = admin_user
        self.logger.info("Default admin user created")
    
    def hash_password(self, password: str) -> str:
        """Hash a password."""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(self, user_id: str, roles: List[str], permissions: List[str]) -> str:
        """Create a JWT access token."""
        expire = datetime.utcnow() + timedelta(minutes=self.config.access_token_expire_minutes)
        
        payload = {
            "user_id": user_id,
            "roles": roles,
            "permissions": permissions,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        token = jwt.encode(payload, self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm)
        return token
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create a JWT refresh token."""
        expire = datetime.utcnow() + timedelta(days=self.config.refresh_token_expire_days)
        
        payload = {
            "user_id": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        }
        
        token = jwt.encode(payload, self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm)
        return token
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, self.config.jwt_secret_key, algorithms=[self.config.jwt_algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user with username and password."""
        # Check if user exists
        user = None
        for u in self.users.values():
            if u.username == username:
                user = u
                break
        
        if not user:
            return None
        
        # Check if user is active
        if not user.is_active:
            return None
        
        # Verify password
        if not self.verify_password(password, user.password_hash):
            return None
        
        # Update last login
        user.last_login = datetime.now()
        
        return user
    
    def create_user(self, username: str, email: str, password: str, roles: List[str] = None) -> User:
        """Create a new user."""
        if roles is None:
            roles = ["user"]
        
        # Validate password strength
        if len(password) < self.config.password_min_length:
            raise ValueError(f"Password must be at least {self.config.password_min_length} characters")
        
        # Check if username already exists
        for user in self.users.values():
            if user.username == username:
                raise ValueError("Username already exists")
        
        # Create user
        user_id = secrets.token_urlsafe(16)
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=self.hash_password(password),
            roles=roles,
            permissions=self._get_permissions_for_roles(roles),
            is_active=True
        )
        
        self.users[user_id] = user
        self.logger.info(f"User created: {username}")
        
        return user
    
    def _get_permissions_for_roles(self, roles: List[str]) -> List[str]:
        """Get permissions for given roles."""
        role_permissions = {
            "admin": ["*"],
            "user": ["read", "write"],
            "viewer": ["read"],
            "api": ["api_access"]
        }
        
        permissions = set()
        for role in roles:
            if role in role_permissions:
                permissions.update(role_permissions[role])
        
        return list(permissions)
    
    def generate_api_key(self, user_id: str, name: str, permissions: List[str] = None, rate_limit: int = 1000) -> str:
        """Generate a new API key."""
        if permissions is None:
            permissions = ["api_access"]
        
        # Generate API key
        api_key = secrets.token_urlsafe(self.config.api_key_length)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Create API key record
        key_id = secrets.token_urlsafe(16)
        api_key_record = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            user_id=user_id,
            name=name,
            permissions=permissions,
            rate_limit=rate_limit
        )
        
        self.api_keys[key_id] = api_key_record
        
        self.logger.info(f"API key created: {name}")
        
        return api_key
    
    def verify_api_key(self, api_key: str) -> Optional[APIKey]:
        """Verify an API key."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        for key_record in self.api_keys.values():
            if key_record.key_hash == key_hash and key_record.is_active:
                # Update last used
                key_record.last_used = datetime.now()
                return key_record
        
        return None
    
    def check_rate_limit(self, identifier: str, limit: int = None) -> bool:
        """Check if request is within rate limit."""
        if limit is None:
            limit = self.config.rate_limit_requests_per_minute
        
        if not self.redis_client:
            return True  # No rate limiting if Redis is not available
        
        try:
            key = f"rate_limit:{identifier}"
            current = self.redis_client.get(key)
            
            if current is None:
                self.redis_client.setex(key, 60, 1)  # 1 minute window
                return True
            
            if int(current) >= limit:
                return False
            
            self.redis_client.incr(key)
            return True
            
        except Exception as e:
            self.logger.warning(f"Rate limit check failed: {e}")
            return True  # Allow request if rate limiting fails
    
    def check_permission(self, user_permissions: List[str], required_permission: str) -> bool:
        """Check if user has required permission."""
        if "*" in user_permissions:
            return True
        
        return required_permission in user_permissions
    
    def sanitize_input(self, input_data: str) -> str:
        """Sanitize user input to prevent injection attacks."""
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\']', '', input_data)
        
        # Limit length
        if len(sanitized) > 1000:
            sanitized = sanitized[:1000]
        
        return sanitized.strip()
    
    def log_security_event(self, event_type: str, user_id: str, details: Dict[str, Any]):
        """Log a security event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "details": details
        }
        
        self.logger.info(f"Security event: {json.dumps(event)}")
        
        # Store in Redis if available
        if self.redis_client:
            try:
                key = f"security_event:{int(time.time())}"
                self.redis_client.setex(key, 86400, json.dumps(event))  # Store for 24 hours
            except Exception as e:
                self.logger.warning(f"Failed to store security event: {e}")
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate a user."""
        user = self.users.get(user_id)
        if user:
            user.is_active = False
            self.logger.info(f"User deactivated: {user.username}")
            return True
        return False
    
    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        key_record = self.api_keys.get(key_id)
        if key_record:
            key_record.is_active = False
            self.logger.info(f"API key revoked: {key_record.name}")
            return True
        return False
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security summary."""
        active_users = len([u for u in self.users.values() if u.is_active])
        active_api_keys = len([k for k in self.api_keys.values() if k.is_active])
        
        return {
            "total_users": len(self.users),
            "active_users": active_users,
            "total_api_keys": len(self.api_keys),
            "active_api_keys": active_api_keys,
            "security_config": asdict(self.config)
        }


# FastAPI dependency for authentication
security_scheme = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)) -> Dict[str, Any]:
    """FastAPI dependency to get current authenticated user."""
    # This would be implemented with the SecurityManager
    # For now, return a mock user
    return {
        "user_id": "mock_user",
        "username": "test_user",
        "roles": ["user"],
        "permissions": ["read", "write"]
    }
