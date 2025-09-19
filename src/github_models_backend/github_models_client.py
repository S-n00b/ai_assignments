"""
GitHub Models Client

A client for integrating with the official GitHub Models API.
Provides a unified interface for model evaluation and inference using GitHub credentials.

Based on: https://docs.github.com/en/rest/models/inference?apiVersion=2022-11-28
"""

import asyncio
import aiohttp
import json
import logging
import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import time

from .rate_limiter import rate_limiter

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Supported model providers via GitHub Models."""
    OPENAI = "openai"
    META = "meta"
    DEEPSEEK = "deepseek"
    MICROSOFT = "microsoft"
    LLAMA = "llama"
    ANTHROPIC = "anthropic"


@dataclass
class ModelInfo:
    """Information about a model."""
    id: str
    name: str
    provider: ModelProvider
    description: str
    capabilities: List[str]
    max_tokens: Optional[int] = None
    cost_per_token: Optional[float] = None
    is_free: bool = True


@dataclass
class EvaluationRequest:
    """Request for model evaluation."""
    model_id: str
    task_type: str
    input_data: Union[str, List[str], Dict[str, Any]]
    parameters: Optional[Dict[str, Any]] = None
    evaluation_metrics: Optional[List[str]] = None


@dataclass
class EvaluationResult:
    """Result of model evaluation."""
    model_id: str
    task_type: str
    input_data: Union[str, List[str], Dict[str, Any]]
    output: Union[str, List[str], Dict[str, Any]]
    metrics: Dict[str, float]
    latency_ms: float
    timestamp: float
    provider: ModelProvider


class GitHubModelsClient:
    """
    Client for GitHub Models-inspired backend integration.
    
    Provides access to free AI models for evaluation and prototyping.
    """
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.models = self._initialize_models()
        self.cache = {}  # Simple in-memory cache
        
    def _initialize_models(self) -> Dict[str, ModelInfo]:
        """Initialize available models from GitHub Models catalog."""
        return {
            # OpenAI models via GitHub Models
            "openai/gpt-4.1": ModelInfo(
                id="openai/gpt-4.1",
                name="GPT-4.1",
                provider=ModelProvider.OPENAI,
                description="Latest GPT-4 model with enhanced capabilities",
                capabilities=["text-generation", "completion", "chat", "reasoning"],
                max_tokens=128000
            ),
            "openai/gpt-4o": ModelInfo(
                id="openai/gpt-4o",
                name="GPT-4o",
                provider=ModelProvider.OPENAI,
                description="Multimodal GPT-4 model with vision capabilities",
                capabilities=["text-generation", "completion", "chat", "vision"],
                max_tokens=128000
            ),
            "openai/gpt-4o-mini": ModelInfo(
                id="openai/gpt-4o-mini",
                name="GPT-4o Mini",
                provider=ModelProvider.OPENAI,
                description="Faster, more cost-effective GPT-4o variant",
                capabilities=["text-generation", "completion", "chat"],
                max_tokens=128000
            ),
            "openai/gpt-3.5-turbo": ModelInfo(
                id="openai/gpt-3.5-turbo",
                name="GPT-3.5 Turbo",
                provider=ModelProvider.OPENAI,
                description="Fast and efficient language model",
                capabilities=["text-generation", "completion", "chat"],
                max_tokens=16384
            ),
            # Meta models via GitHub Models
            "meta/llama-3.1-8b": ModelInfo(
                id="meta/llama-3.1-8b",
                name="Llama 3.1 8B",
                provider=ModelProvider.META,
                description="Open-source large language model",
                capabilities=["text-generation", "completion", "chat"],
                max_tokens=128000
            ),
            "meta/llama-3.1-70b": ModelInfo(
                id="meta/llama-3.1-70b",
                name="Llama 3.1 70B",
                provider=ModelProvider.META,
                description="Large-scale open-source language model",
                capabilities=["text-generation", "completion", "chat", "reasoning"],
                max_tokens=128000
            ),
            # DeepSeek models via GitHub Models
            "deepseek/deepseek-chat": ModelInfo(
                id="deepseek/deepseek-chat",
                name="DeepSeek Chat",
                provider=ModelProvider.DEEPSEEK,
                description="Advanced reasoning and coding model",
                capabilities=["text-generation", "completion", "chat", "coding", "reasoning"],
                max_tokens=32000
            ),
            # Microsoft models via GitHub Models
            "microsoft/phi-3-medium": ModelInfo(
                id="microsoft/phi-3-medium",
                name="Phi-3 Medium",
                provider=ModelProvider.MICROSOFT,
                description="Efficient small language model",
                capabilities=["text-generation", "completion", "chat"],
                max_tokens=128000
            )
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models."""
        return list(self.models.values())
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        return self.models.get(model_id)
    
    def _get_github_token(self) -> Optional[str]:
        """Get GitHub token from environment or GitHub CLI."""
        # Try environment variable first
        token = os.getenv("GITHUB_TOKEN")
        if token:
            logger.info("Using GitHub token from GITHUB_TOKEN environment variable")
            return token
        
        # Try GitHub CLI token
        try:
            import subprocess
            result = subprocess.run(
                ["gh", "auth", "token"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            token = result.stdout.strip()
            if token:
                logger.info("Using GitHub token from GitHub CLI")
                return token
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # No token found
        logger.error("""
❌ GitHub Models API requires authentication!

To use GitHub Models, you need a Personal Access Token with 'models' scope.

Setup options:
1. Create a PAT: https://github.com/settings/tokens
   - Select 'models' scope
   - Set GITHUB_TOKEN environment variable

2. Use GitHub CLI: gh auth login
   - Automatically sets up token with correct scopes

3. For demo mode: Set GITHUB_TOKEN=demo_token
   - Limited functionality, no real API calls
        """)
        return None
    
    async def _call_github_models_api(self, model_id: str, messages: List[Dict[str, str]], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call GitHub Models API."""
        url = "https://models.github.ai/inference/chat/completions"
        
        # Get GitHub token
        token = self._get_github_token()
        
        if not token:
            return {
                "error": "No GitHub token available. Please set up authentication.",
                "demo_response": f"Demo response from {model_id}: This is a mock response. Set up GitHub token for real API calls."
            }
        
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json"
        }
        
        # Prepare payload according to GitHub Models API spec
        payload = {
            "model": model_id,
            "messages": messages,
            "stream": False
        }
        
        # Add parameters if provided
        if parameters:
            # Map common parameters to GitHub Models format
            if "temperature" in parameters:
                payload["temperature"] = parameters["temperature"]
            if "max_tokens" in parameters:
                payload["max_tokens"] = parameters["max_tokens"]
            if "top_p" in parameters:
                payload["top_p"] = parameters["top_p"]
            if "frequency_penalty" in parameters:
                payload["frequency_penalty"] = parameters["frequency_penalty"]
            if "presence_penalty" in parameters:
                payload["presence_penalty"] = parameters["presence_penalty"]
        
        # Apply rate limiting
        await rate_limiter.async_wait_if_needed("github_models")
        
        try:
            async with self.session.post(url, json=payload, headers=headers) as response:
                rate_limiter.record_request("github_models")
                
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"GitHub Models API error: {response.status} - {error_text}")
                    return {"error": f"API error: {response.status} - {error_text}"}
                    
        except Exception as e:
            logger.error(f"Error calling GitHub Models API: {e}")
            return {"error": str(e)}
    
    async def evaluate_model(self, request: EvaluationRequest) -> EvaluationResult:
        """
        Evaluate a model with the given request using GitHub Models API.
        
        Args:
            request: Evaluation request containing model, task, and input data
            
        Returns:
            EvaluationResult with model output and metrics
        """
        start_time = time.time()
        
        model_info = self.get_model_info(request.model_id)
        if not model_info:
            raise ValueError(f"Model {request.model_id} not found")
        
        # Prepare input data for GitHub Models API format
        if isinstance(request.input_data, dict):
            # If it's a dict, assume it contains role and content
            if "role" in request.input_data and "content" in request.input_data:
                messages = [request.input_data]
            else:
                # Convert dict to user message
                messages = [{"role": "user", "content": json.dumps(request.input_data)}]
        elif isinstance(request.input_data, list):
            # If it's a list, assume it's a list of messages or strings
            if len(request.input_data) > 0 and isinstance(request.input_data[0], dict):
                messages = request.input_data
            else:
                # Convert list of strings to user message
                content = "\n".join(str(item) for item in request.input_data)
                messages = [{"role": "user", "content": content}]
        else:
            # Single string input
            messages = [{"role": "user", "content": str(request.input_data)}]
        
        # Add system message for task context
        if request.task_type:
            system_message = {
                "role": "system",
                "content": f"You are an AI assistant helping with {request.task_type}. Please provide a helpful response."
            }
            messages = [system_message] + messages
        
        # Prepare parameters
        parameters = request.parameters or {}
        if model_info.max_tokens:
            parameters.setdefault("max_tokens", min(parameters.get("max_tokens", 1000), model_info.max_tokens))
        
        # Call GitHub Models API
        response = await self._call_github_models_api(request.model_id, messages, parameters)
        
        # Process response
        if "error" in response:
            output = {"error": response["error"]}
        else:
            output = self._extract_output(response, model_info.provider)
        
        # Calculate metrics
        latency_ms = (time.time() - start_time) * 1000
        input_text = " ".join(msg.get("content", "") for msg in messages)
        metrics = self._calculate_metrics(input_text, output, latency_ms, model_info)
        
        return EvaluationResult(
            model_id=request.model_id,
            task_type=request.task_type,
            input_data=request.input_data,
            output=output,
            metrics=metrics,
            latency_ms=latency_ms,
            timestamp=time.time(),
            provider=model_info.provider
        )
    
    def _extract_output(self, response: Dict[str, Any], provider: ModelProvider) -> Union[str, List[str], Dict[str, Any]]:
        """Extract output from GitHub Models API response."""
        # GitHub Models API returns responses in OpenAI-compatible format
        if "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
            elif "text" in choice:
                return choice["text"]
        
        # Fallback to full response if structure is unexpected
        return response
    
    def _calculate_metrics(self, inputs: str, output: Union[str, List[str], Dict[str, Any]], 
                          latency_ms: float, model_info: ModelInfo) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {
            "latency_ms": latency_ms,
            "input_length": len(inputs),
            "output_length": len(str(output)),
            "tokens_per_second": len(str(output)) / (latency_ms / 1000) if latency_ms > 0 else 0
        }
        
        # Add provider-specific metrics
        if model_info.provider == ModelProvider.HUGGINGFACE:
            metrics["provider"] = "huggingface"
        elif model_info.provider == ModelProvider.OPENAI_FREE:
            metrics["provider"] = "openai"
        
        return metrics
    
    async def batch_evaluate(self, requests: List[EvaluationRequest]) -> List[EvaluationResult]:
        """Evaluate multiple models in batch."""
        tasks = [self.evaluate_model(request) for request in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_rate_limit_status(self) -> Dict[str, Dict[str, int]]:
        """Get rate limit status for all endpoints."""
        return {
            "github_models": rate_limiter.get_status("github_models"),
            "evaluation": rate_limiter.get_status("evaluation")
        }
    
    def clear_cache(self):
        """Clear the model cache."""
        self.cache.clear()
        logger.info("Model cache cleared")


# Global client instance
github_models_client = GitHubModelsClient()
