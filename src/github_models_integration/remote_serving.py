"""
GitHub Models Remote Serving

This module provides remote model serving capabilities via GitHub Models API,
including endpoint management and inference handling.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from .api_client import GitHubModelsAPIClient

logger = logging.getLogger(__name__)


@dataclass
class ServingEndpoint:
    """Represents a serving endpoint for a remote model."""
    model_id: str
    endpoint_url: str
    status: str  # "active", "inactive", "error"
    created_at: datetime
    request_count: int = 0
    last_request: Optional[datetime] = None
    average_latency_ms: float = 0.0


@dataclass
class InferenceRequest:
    """Request for model inference."""
    model_id: str
    messages: List[Dict[str, str]]
    parameters: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    timeout: int = 30


@dataclass
class InferenceResponse:
    """Response from model inference."""
    request_id: str
    model_id: str
    response: Any
    latency_ms: float
    timestamp: datetime
    success: bool = True
    error: Optional[str] = None


class GitHubModelsRemoteServing:
    """Remote model serving via GitHub Models API."""
    
    def __init__(self, api_client: GitHubModelsAPIClient):
        """Initialize the remote serving manager."""
        self.api_client = api_client
        self.serving_endpoints: Dict[str, ServingEndpoint] = {}
        self.request_history = []
        self.active_requests = {}
        
    async def serve_model(self, model_id: str) -> ServingEndpoint:
        """
        Create a serving endpoint for a remote model.
        
        Args:
            model_id: Model ID to serve
            
        Returns:
            ServingEndpoint object
        """
        try:
            logger.info(f"Creating serving endpoint for model: {model_id}")
            
            # Check if endpoint already exists
            if model_id in self.serving_endpoints:
                endpoint = self.serving_endpoints[model_id]
                endpoint.status = "active"
                logger.info(f"Reactivated existing endpoint for {model_id}")
                return endpoint
            
            # Validate model exists
            model_info = await self.api_client.get_model_details(model_id)
            if not model_info:
                raise ValueError(f"Model {model_id} not found")
            
            # Create serving endpoint
            endpoint = ServingEndpoint(
                model_id=model_id,
                endpoint_url=f"https://models.github.ai/inference/chat/completions",
                status="active",
                created_at=datetime.now()
            )
            
            self.serving_endpoints[model_id] = endpoint
            
            logger.info(f"Created serving endpoint for {model_id}")
            
            return endpoint
            
        except Exception as e:
            logger.error(f"Error creating serving endpoint for {model_id}: {e}")
            raise
    
    async def make_inference(
        self,
        model_id: str,
        input_data: Dict[str, Any]
    ) -> InferenceResponse:
        """
        Make inference call to remote model.
        
        Args:
            model_id: Model ID to call
            input_data: Input data for inference
            
        Returns:
            InferenceResponse with model output
        """
        request_id = f"{model_id}_{int(time.time() * 1000)}"
        start_time = time.time()
        
        try:
            logger.info(f"Making inference request to {model_id} (request: {request_id})")
            
            # Check if model is being served
            if model_id not in self.serving_endpoints:
                await self.serve_model(model_id)
            
            endpoint = self.serving_endpoints[model_id]
            
            if endpoint.status != "active":
                raise Exception(f"Model {model_id} endpoint is not active")
            
            # Prepare request data
            messages = input_data.get("messages", [])
            parameters = input_data.get("parameters", {})
            
            # Make API call
            request_data = {
                "messages": messages,
                "parameters": parameters
            }
            
            response = await self.api_client.evaluate_model(model_id, request_data)
            
            # Process response
            if response.get("success"):
                inference_response = InferenceResponse(
                    request_id=request_id,
                    model_id=model_id,
                    response=response["response"],
                    latency_ms=(time.time() - start_time) * 1000,
                    timestamp=datetime.now(),
                    success=True
                )
            else:
                inference_response = InferenceResponse(
                    request_id=request_id,
                    model_id=model_id,
                    response={"error": response.get("error")},
                    latency_ms=(time.time() - start_time) * 1000,
                    timestamp=datetime.now(),
                    success=False,
                    error=response.get("error")
                )
            
            # Update endpoint statistics
            endpoint.request_count += 1
            endpoint.last_request = datetime.now()
            
            # Update average latency
            if endpoint.average_latency_ms == 0.0:
                endpoint.average_latency_ms = inference_response.latency_ms
            else:
                # Exponential moving average
                endpoint.average_latency_ms = (
                    endpoint.average_latency_ms * 0.9 + inference_response.latency_ms * 0.1
                )
            
            # Record request history
            self.request_history.append(inference_response)
            
            logger.info(f"Inference request {request_id} completed in {inference_response.latency_ms:.2f}ms")
            
            return inference_response
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error_msg = f"Inference request failed: {str(e)}"
            logger.error(f"Request {request_id}: {error_msg}")
            
            inference_response = InferenceResponse(
                request_id=request_id,
                model_id=model_id,
                response={"error": error_msg},
                latency_ms=latency_ms,
                timestamp=datetime.now(),
                success=False,
                error=error_msg
            )
            
            # Record failed request
            self.request_history.append(inference_response)
            
            return inference_response
    
    async def get_serving_status(self, model_id: str) -> Dict[str, Any]:
        """
        Get serving status for a remote model.
        
        Args:
            model_id: Model ID to check
            
        Returns:
            Dictionary with serving status information
        """
        if model_id not in self.serving_endpoints:
            return {
                "model_id": model_id,
                "serving": False,
                "status": "not_served",
                "message": "Model is not being served"
            }
        
        endpoint = self.serving_endpoints[model_id]
        
        return {
            "model_id": model_id,
            "serving": endpoint.status == "active",
            "status": endpoint.status,
            "endpoint_url": endpoint.endpoint_url,
            "created_at": endpoint.created_at.isoformat(),
            "request_count": endpoint.request_count,
            "last_request": endpoint.last_request.isoformat() if endpoint.last_request else None,
            "average_latency_ms": endpoint.average_latency_ms
        }
    
    async def stop_serving(self, model_id: str) -> bool:
        """
        Stop serving a remote model.
        
        Args:
            model_id: Model ID to stop serving
            
        Returns:
            True if successfully stopped, False otherwise
        """
        try:
            if model_id in self.serving_endpoints:
                self.serving_endpoints[model_id].status = "inactive"
                logger.info(f"Stopped serving model: {model_id}")
                return True
            else:
                logger.warning(f"Model {model_id} was not being served")
                return False
                
        except Exception as e:
            logger.error(f"Error stopping serving for {model_id}: {e}")
            return False
    
    def list_serving_models(self) -> List[Dict[str, Any]]:
        """List all currently serving models."""
        serving_models = []
        
        for model_id, endpoint in self.serving_endpoints.items():
            if endpoint.status == "active":
                serving_models.append({
                    "model_id": model_id,
                    "endpoint_url": endpoint.endpoint_url,
                    "created_at": endpoint.created_at.isoformat(),
                    "request_count": endpoint.request_count,
                    "last_request": endpoint.last_request.isoformat() if endpoint.last_request else None,
                    "average_latency_ms": endpoint.average_latency_ms
                })
        
        return serving_models
    
    def get_serving_statistics(self) -> Dict[str, Any]:
        """Get serving statistics."""
        total_requests = len(self.request_history)
        successful_requests = sum(1 for r in self.request_history if r.success)
        failed_requests = total_requests - successful_requests
        
        avg_latency = 0.0
        if successful_requests > 0:
            successful_latencies = [r.latency_ms for r in self.request_history if r.success]
            avg_latency = sum(successful_latencies) / len(successful_latencies)
        
        active_endpoints = sum(1 for endpoint in self.serving_endpoints.values() if endpoint.status == "active")
        
        return {
            "total_endpoints": len(self.serving_endpoints),
            "active_endpoints": active_endpoints,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "average_latency_ms": avg_latency,
            "success_rate": successful_requests / total_requests if total_requests > 0 else 0.0
        }
    
    def get_request_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get request history."""
        history = []
        
        for request in self.request_history[-limit:] if limit else self.request_history:
            history.append({
                "request_id": request.request_id,
                "model_id": request.model_id,
                "latency_ms": request.latency_ms,
                "timestamp": request.timestamp.isoformat(),
                "success": request.success,
                "error": request.error
            })
        
        return history
    
    def clear_serving_endpoints(self):
        """Clear all serving endpoints."""
        self.serving_endpoints.clear()
        logger.info("Cleared all serving endpoints")
    
    def clear_request_history(self):
        """Clear request history."""
        self.request_history.clear()
        logger.info("Cleared request history")
