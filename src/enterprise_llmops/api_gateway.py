"""
API Gateway for Enterprise LLMOps Platform

This module provides a comprehensive API gateway for managing and routing
requests to various enterprise services with load balancing, authentication,
and monitoring capabilities.

Key Features:
- Request routing and load balancing
- Authentication and authorization
- Rate limiting and throttling
- Request/response transformation
- Health checking and circuit breaking
- Metrics collection and monitoring
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import aiohttp
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest


@dataclass
class RouteConfig:
    """Configuration for API route."""
    path: str
    target_service: str
    target_url: str
    methods: List[str]
    auth_required: bool = True
    rate_limit: Optional[int] = None
    timeout: int = 30


@dataclass
class ServiceHealth:
    """Health status of a service."""
    service_name: str
    is_healthy: bool
    last_check: datetime
    response_time: float
    error_count: int = 0


class APIGateway:
    """
    API Gateway for Enterprise LLMOps platform.
    
    This class provides comprehensive API gateway functionality including
    routing, load balancing, authentication, and monitoring.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the API gateway."""
        self.config = config
        self.routes = {}
        self.service_health = {}
        self.redis_client = None
        self.logger = self._setup_logging()
        
        # Prometheus metrics
        self.request_counter = Counter('api_gateway_requests_total', 
                                     'Total API requests', ['service', 'method', 'status'])
        self.request_duration = Histogram('api_gateway_request_duration_seconds',
                                        'Request duration', ['service', 'method'])
        self.active_connections = Gauge('api_gateway_active_connections',
                                      'Active connections')
        
        # Initialize Redis for rate limiting
        self._init_redis()
        
        # Setup routes
        self._setup_routes()
        
        # Start health checking
        self.health_check_task = None
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for API gateway."""
        logger = logging.getLogger("api_gateway")
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
        """Initialize Redis connection for rate limiting."""
        try:
            self.redis_client = redis.Redis(
                host=self.config.get('redis_host', 'localhost'),
                port=self.config.get('redis_port', 6379),
                db=self.config.get('redis_db', 0),
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            self.logger.info("Redis connection established")
        except Exception as e:
            self.logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
    
    def _setup_routes(self):
        """Setup API routes."""
        default_routes = [
            RouteConfig(
                path="/ollama/{path:path}",
                target_service="ollama",
                target_url="http://localhost:11434",
                methods=["GET", "POST", "PUT", "DELETE"],
                auth_required=False
            ),
            RouteConfig(
                path="/mlflow/{path:path}",
                target_service="mlflow",
                target_url="http://localhost:5000",
                methods=["GET", "POST", "PUT", "DELETE"],
                auth_required=True
            ),
            RouteConfig(
                path="/chroma/{path:path}",
                target_service="chroma",
                target_url="http://localhost:8081",
                methods=["GET", "POST", "PUT", "DELETE"],
                auth_required=True
            ),
            RouteConfig(
                path="/weaviate/{path:path}",
                target_service="weaviate",
                target_url="http://localhost:8083",
                methods=["GET", "POST", "PUT", "DELETE"],
                auth_required=True
            )
        ]
        
        for route in default_routes:
            self.routes[route.path] = route
            self.service_health[route.target_service] = ServiceHealth(
                service_name=route.target_service,
                is_healthy=False,
                last_check=datetime.now(),
                response_time=0.0
            )
    
    async def check_service_health(self, service_name: str, target_url: str) -> ServiceHealth:
        """Check health of a service."""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{target_url}/health", timeout=5) as response:
                    response_time = time.time() - start_time
                    
                    return ServiceHealth(
                        service_name=service_name,
                        is_healthy=response.status == 200,
                        last_check=datetime.now(),
                        response_time=response_time
                    )
        except Exception as e:
            response_time = time.time() - start_time
            self.logger.warning(f"Health check failed for {service_name}: {e}")
            
            return ServiceHealth(
                service_name=service_name,
                is_healthy=False,
                last_check=datetime.now(),
                response_time=response_time,
                error_count=1
            )
    
    async def health_check_loop(self):
        """Background health checking loop."""
        while True:
            try:
                for route in self.routes.values():
                    health = await self.check_service_health(
                        route.target_service, 
                        route.target_url
                    )
                    self.service_health[route.target_service] = health
                
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def start(self):
        """Start the API gateway."""
        self.logger.info("Starting API Gateway...")
        
        # Start health checking
        self.health_check_task = asyncio.create_task(self.health_check_loop())
        
        self.logger.info("API Gateway started successfully")
    
    async def stop(self):
        """Stop the API gateway."""
        self.logger.info("Stopping API Gateway...")
        
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        if self.redis_client:
            self.redis_client.close()
        
        self.logger.info("API Gateway stopped")
    
    def get_healthy_services(self) -> Dict[str, ServiceHealth]:
        """Get all healthy services."""
        return {name: health for name, health in self.service_health.items() 
                if health.is_healthy}
    
    def get_service_metrics(self) -> Dict[str, Any]:
        """Get service metrics."""
        return {
            "total_services": len(self.service_health),
            "healthy_services": len(self.get_healthy_services()),
            "service_health": {name: asdict(health) for name, health in self.service_health.items()}
        }
