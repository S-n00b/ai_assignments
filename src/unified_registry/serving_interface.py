"""
Model Serving Interface

This module provides a unified interface for model serving with support for
both local (Ollama) and remote (GitHub Models) serving capabilities.
"""

import asyncio
import aiohttp
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from .model_objects import UnifiedModelObject
from .registry_manager import UnifiedRegistryManager

logger = logging.getLogger(__name__)


@dataclass
class ServingRequest:
    """Request for model serving."""
    model_id: str
    messages: List[Dict[str, str]]
    parameters: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    timeout: int = 30


@dataclass
class ServingResponse:
    """Response from model serving."""
    request_id: str
    model_id: str
    response: Any
    latency_ms: float
    timestamp: datetime
    success: bool = True
    error: Optional[str] = None
    serving_type: str = "unknown"


class ModelServingInterface:
    """Interface for model serving with local/remote abstraction."""
    
    def __init__(self, registry_manager: UnifiedRegistryManager):
        """Initialize the serving interface."""
        self.registry_manager = registry_manager
        self.serving_endpoints: Dict[str, str] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.serving_history = []
        self.active_requests = {}
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def serve_model(self, model_id: str) -> str:
        """
        Start serving a model and return endpoint.
        
        Args:
            model_id: Model ID to serve
            
        Returns:
            Serving endpoint URL
        """
        try:
            logger.info(f"Starting serving for model: {model_id}")
            
            # Get model from registry
            model = await self.registry_manager.get_model(model_id)
            if not model:
                raise ValueError(f"Model {model_id} not found in registry")
            
            # Determine serving endpoint
            endpoint = model.get_serving_endpoint()
            if not endpoint:
                raise ValueError(f"No serving endpoint available for model {model_id}")
            
            # Store serving endpoint
            self.serving_endpoints[model_id] = endpoint
            
            # Update model status
            model.update_status("available")
            await self.registry_manager.register_model(model)
            
            logger.info(f"Started serving model {model_id} at {endpoint}")
            
            return endpoint
            
        except Exception as e:
            logger.error(f"Error starting serving for {model_id}: {e}")
            raise
    
    async def stop_serving(self, model_id: str) -> bool:
        """
        Stop serving a model.
        
        Args:
            model_id: Model ID to stop serving
            
        Returns:
            True if successfully stopped, False otherwise
        """
        try:
            if model_id in self.serving_endpoints:
                del self.serving_endpoints[model_id]
                
                # Update model status
                model = await self.registry_manager.get_model(model_id)
                if model:
                    model.update_status("unavailable")
                    await self.registry_manager.register_model(model)
                
                logger.info(f"Stopped serving model: {model_id}")
                return True
            else:
                logger.warning(f"Model {model_id} was not being served")
                return False
                
        except Exception as e:
            logger.error(f"Error stopping serving for {model_id}: {e}")
            return False
    
    async def make_inference(self, request: ServingRequest) -> ServingResponse:
        """
        Make inference request to a model.
        
        Args:
            request: Serving request
            
        Returns:
            Serving response
        """
        request_id = request.request_id or f"{request.model_id}_{int(time.time() * 1000)}"
        start_time = time.time()
        
        try:
            logger.info(f"Making inference request to {request.model_id} (request: {request_id})")
            
            # Get model from registry
            model = await self.registry_manager.get_model(request.model_id)
            if not model:
                return ServingResponse(
                    request_id=request_id,
                    model_id=request.model_id,
                    response={"error": f"Model {request.model_id} not found"},
                    latency_ms=(time.time() - start_time) * 1000,
                    timestamp=datetime.now(),
                    success=False,
                    error=f"Model {request.model_id} not found",
                    serving_type="unknown"
                )
            
            # Check if model is available
            if not model.is_available():
                return ServingResponse(
                    request_id=request_id,
                    model_id=request.model_id,
                    response={"error": f"Model {request.model_id} is not available"},
                    latency_ms=(time.time() - start_time) * 1000,
                    timestamp=datetime.now(),
                    success=False,
                    error=f"Model {request.model_id} is not available",
                    serving_type=model.serving_type
                )
            
            # Make inference based on serving type
            if model.serving_type == "local":
                response = await self._make_local_inference(model, request)
            elif model.serving_type == "remote":
                response = await self._make_remote_inference(model, request)
            elif model.serving_type == "hybrid":
                # Try local first, fallback to remote
                try:
                    response = await self._make_local_inference(model, request)
                except Exception as e:
                    logger.warning(f"Local inference failed for {request.model_id}, trying remote: {e}")
                    response = await self._make_remote_inference(model, request)
            else:
                raise ValueError(f"Unknown serving type: {model.serving_type}")
            
            # Update response metadata
            response.request_id = request_id
            response.model_id = request.model_id
            response.latency_ms = (time.time() - start_time) * 1000
            response.timestamp = datetime.now()
            response.serving_type = model.serving_type
            
            # Record serving history
            self.serving_history.append(response)
            
            # Update model performance metrics
            model.update_performance_metrics({
                "last_request_latency_ms": response.latency_ms,
                "total_requests": model.performance_metrics.get("total_requests", 0) + 1
            })
            await self.registry_manager.register_model(model)
            
            logger.info(f"Inference request {request_id} completed in {response.latency_ms:.2f}ms")
            
            return response
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error_msg = f"Inference request failed: {str(e)}"
            logger.error(f"Request {request_id}: {error_msg}")
            
            response = ServingResponse(
                request_id=request_id,
                model_id=request.model_id,
                response={"error": error_msg},
                latency_ms=latency_ms,
                timestamp=datetime.now(),
                success=False,
                error=error_msg,
                serving_type="unknown"
            )
            
            # Record failed request
            self.serving_history.append(response)
            
            return response
    
    async def _make_local_inference(self, model: UnifiedModelObject, request: ServingRequest) -> ServingResponse:
        """Make inference request to local Ollama model."""
        if not self.session:
            raise Exception("Session not initialized. Use async context manager.")
        
        # Prepare Ollama API request
        payload = {
            "model": model.ollama_name,
            "messages": request.messages,
            "stream": False
        }
        
        # Add parameters
        if request.parameters:
            payload.update(request.parameters)
        
        # Make request to Ollama
        url = model.local_endpoint or "http://localhost:11434/api/generate"
        
        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return ServingResponse(
                    request_id="",
                    model_id="",
                    response=data,
                    latency_ms=0.0,
                    timestamp=datetime.now(),
                    success=True,
                    serving_type="local"
                )
            else:
                error_text = await response.text()
                raise Exception(f"Ollama API error: {response.status} - {error_text}")
    
    async def _make_remote_inference(self, model: UnifiedModelObject, request: ServingRequest) -> ServingResponse:
        """Make inference request to remote GitHub Models API."""
        if not self.session:
            raise Exception("Session not initialized. Use async context manager.")
        
        # Prepare GitHub Models API request
        payload = {
            "model": model.github_models_id,
            "messages": request.messages,
            "stream": False
        }
        
        # Add parameters
        if request.parameters:
            if "temperature" in request.parameters:
                payload["temperature"] = request.parameters["temperature"]
            if "max_tokens" in request.parameters:
                payload["max_tokens"] = request.parameters["max_tokens"]
            if "top_p" in request.parameters:
                payload["top_p"] = request.parameters["top_p"]
        
        # Get GitHub token
        import os
        github_token = os.getenv("GITHUB_TOKEN")
        
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json"
        }
        
        if github_token:
            headers["Authorization"] = f"Bearer {github_token}"
        
        # Make request to GitHub Models API
        url = model.remote_endpoint or "https://models.github.ai/inference/chat/completions"
        
        async with self.session.post(url, json=payload, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                return ServingResponse(
                    request_id="",
                    model_id="",
                    response=data,
                    latency_ms=0.0,
                    timestamp=datetime.now(),
                    success=True,
                    serving_type="remote"
                )
            else:
                error_text = await response.text()
                raise Exception(f"GitHub Models API error: {response.status} - {error_text}")
    
    async def get_serving_status(self, model_id: str) -> Dict[str, Any]:
        """
        Get serving status for a model.
        
        Args:
            model_id: Model ID to check
            
        Returns:
            Dictionary with serving status information
        """
        model = await self.registry_manager.get_model(model_id)
        if not model:
            return {
                "model_id": model_id,
                "serving": False,
                "status": "not_found",
                "message": "Model not found in registry"
            }
        
        is_serving = model_id in self.serving_endpoints
        
        return {
            "model_id": model_id,
            "serving": is_serving,
            "status": model.status,
            "serving_type": model.serving_type,
            "endpoint": self.serving_endpoints.get(model_id),
            "local_endpoint": model.local_endpoint,
            "remote_endpoint": model.remote_endpoint,
            "capabilities": model.capabilities,
            "performance_metrics": model.performance_metrics
        }
    
    async def health_check(self, model_id: str) -> bool:
        """
        Perform health check on a serving model.
        
        Args:
            model_id: Model ID to check
            
        Returns:
            True if model is healthy, False otherwise
        """
        try:
            # Create a simple test request
            test_request = ServingRequest(
                model_id=model_id,
                messages=[{"role": "user", "content": "Hello"}],
                parameters={"max_tokens": 10},
                request_id=f"health_check_{int(time.time())}"
            )
            
            # Make test inference
            response = await self.make_inference(test_request)
            
            return response.success
            
        except Exception as e:
            logger.error(f"Health check failed for {model_id}: {e}")
            return False
    
    def list_serving_models(self) -> List[Dict[str, Any]]:
        """List all currently serving models."""
        serving_models = []
        
        for model_id in self.serving_endpoints:
            model = self.registry_manager.model_objects.get(model_id)
            if model:
                serving_models.append({
                    "model_id": model_id,
                    "name": model.name,
                    "category": model.category,
                    "serving_type": model.serving_type,
                    "endpoint": self.serving_endpoints[model_id],
                    "status": model.status,
                    "capabilities": model.capabilities
                })
        
        return serving_models
    
    def get_serving_statistics(self) -> Dict[str, Any]:
        """Get serving statistics."""
        total_requests = len(self.serving_history)
        successful_requests = sum(1 for r in self.serving_history if r.success)
        failed_requests = total_requests - successful_requests
        
        avg_latency = 0.0
        if successful_requests > 0:
            successful_latencies = [r.latency_ms for r in self.serving_history if r.success]
            avg_latency = sum(successful_latencies) / len(successful_latencies)
        
        # Count by serving type
        serving_type_counts = {}
        for response in self.serving_history:
            serving_type_counts[response.serving_type] = serving_type_counts.get(response.serving_type, 0) + 1
        
        return {
            "total_models_serving": len(self.serving_endpoints),
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "average_latency_ms": avg_latency,
            "success_rate": successful_requests / total_requests if total_requests > 0 else 0.0,
            "serving_type_counts": serving_type_counts
        }
    
    def get_serving_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get serving history."""
        history = []
        
        for response in self.serving_history[-limit:] if limit else self.serving_history:
            history.append({
                "request_id": response.request_id,
                "model_id": response.model_id,
                "latency_ms": response.latency_ms,
                "timestamp": response.timestamp.isoformat(),
                "success": response.success,
                "error": response.error,
                "serving_type": response.serving_type
            })
        
        return history
    
    def clear_serving_history(self):
        """Clear serving history."""
        self.serving_history.clear()
        logger.info("Serving history cleared")
