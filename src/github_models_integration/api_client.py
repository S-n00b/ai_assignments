"""
GitHub Models API Client

This module provides a client for GitHub Models API integration with
authentication, rate limiting, and model discovery capabilities.
"""

import asyncio
import aiohttp
import logging
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
import subprocess

logger = logging.getLogger(__name__)


@dataclass
class GitHubModel:
    """Represents a GitHub Models API model."""
    id: str
    name: str
    provider: str
    description: str
    capabilities: List[str]
    max_tokens: Optional[int] = None
    cost_per_token: Optional[float] = None
    is_free: bool = True
    category: str = "cloud_general"


@dataclass
class RateLimitStatus:
    """Rate limit status for GitHub Models API."""
    remaining: int
    reset_time: int
    limit: int


class GitHubModelsAPIClient:
    """Client for GitHub Models API integration."""
    
    # GitHub Models provider categories
    PROVIDER_CATEGORIES = {
        "openai/*": "cloud_text",
        "meta/*": "cloud_text", 
        "deepseek/*": "cloud_code",
        "microsoft/*": "cloud_multimodal",
        "default": "cloud_general"
    }
    
    def __init__(self, github_token: Optional[str] = None):
        """
        Initialize the GitHub Models API client.
        
        Args:
            github_token: GitHub token (optional, will be auto-detected)
        """
        self.github_token = github_token or self._get_github_token()
        self.base_url = "https://api.github.com"
        self.models_url = "https://models.github.ai"
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit_status = {}
        self.request_count = 0
        self.last_request_time = 0.0
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers=self._get_headers()
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def _get_github_token(self) -> Optional[str]:
        """Get GitHub token from environment or GitHub CLI."""
        # Try environment variable first
        token = os.getenv("GITHUB_TOKEN")
        if token:
            logger.info("Using GitHub token from GITHUB_TOKEN environment variable")
            return token
        
        # Try GitHub CLI token
        try:
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
        logger.warning("""
⚠️  GitHub Models API requires authentication!

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
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests."""
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json"
        }
        
        if self.github_token:
            headers["Authorization"] = f"Bearer {self.github_token}"
        
        return headers
    
    async def get_available_models(self) -> List[GitHubModel]:
        """
        Get all available models from GitHub Models API.
        
        Returns:
            List of GitHubModel objects
        """
        try:
            # For now, return a curated list of available models
            # In a real implementation, this would call the GitHub Models API
            return self._get_curated_models()
            
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []
    
    def _get_curated_models(self) -> List[GitHubModel]:
        """Get curated list of available GitHub Models."""
        return [
            # OpenAI models via GitHub Models
            GitHubModel(
                id="openai/gpt-4.1",
                name="GPT-4.1",
                provider="openai",
                description="Latest GPT-4 model with enhanced capabilities",
                capabilities=["text-generation", "completion", "chat", "reasoning"],
                max_tokens=128000,
                category="cloud_text"
            ),
            GitHubModel(
                id="openai/gpt-4o",
                name="GPT-4o",
                provider="openai",
                description="Multimodal GPT-4 model with vision capabilities",
                capabilities=["text-generation", "completion", "chat", "vision"],
                max_tokens=128000,
                category="cloud_multimodal"
            ),
            GitHubModel(
                id="openai/gpt-4o-mini",
                name="GPT-4o Mini",
                provider="openai",
                description="Faster, more cost-effective GPT-4o variant",
                capabilities=["text-generation", "completion", "chat"],
                max_tokens=128000,
                category="cloud_text"
            ),
            GitHubModel(
                id="openai/gpt-3.5-turbo",
                name="GPT-3.5 Turbo",
                provider="openai",
                description="Fast and efficient language model",
                capabilities=["text-generation", "completion", "chat"],
                max_tokens=16384,
                category="cloud_text"
            ),
            # Meta models via GitHub Models
            GitHubModel(
                id="meta/llama-3.1-8b",
                name="Llama 3.1 8B",
                provider="meta",
                description="Open-source large language model",
                capabilities=["text-generation", "completion", "chat"],
                max_tokens=128000,
                category="cloud_text"
            ),
            GitHubModel(
                id="meta/llama-3.1-70b",
                name="Llama 3.1 70B",
                provider="meta",
                description="Large-scale open-source language model",
                capabilities=["text-generation", "completion", "chat", "reasoning"],
                max_tokens=128000,
                category="cloud_text"
            ),
            # DeepSeek models via GitHub Models
            GitHubModel(
                id="deepseek/deepseek-chat",
                name="DeepSeek Chat",
                provider="deepseek",
                description="Advanced reasoning and coding model",
                capabilities=["text-generation", "completion", "chat", "coding", "reasoning"],
                max_tokens=32000,
                category="cloud_code"
            ),
            # Microsoft models via GitHub Models
            GitHubModel(
                id="microsoft/phi-3-medium",
                name="Phi-3 Medium",
                provider="microsoft",
                description="Efficient small language model",
                capabilities=["text-generation", "completion", "chat"],
                max_tokens=128000,
                category="cloud_text"
            )
        ]
    
    async def get_models_by_provider(self, provider: str) -> List[GitHubModel]:
        """
        Get models by specific provider.
        
        Args:
            provider: Provider name (openai, meta, etc.)
            
        Returns:
            List of GitHubModel objects from the provider
        """
        try:
            all_models = await self.get_available_models()
            return [model for model in all_models if model.provider == provider]
            
        except Exception as e:
            logger.error(f"Error getting models by provider {provider}: {e}")
            return []
    
    async def evaluate_model(self, model_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a model using GitHub Models API.
        
        Args:
            model_id: Model ID to evaluate
            request_data: Evaluation request data
            
        Returns:
            Evaluation result dictionary
        """
        try:
            if not self.session:
                raise Exception("Session not initialized. Use async context manager.")
            
            # Prepare evaluation request
            url = f"{self.models_url}/inference/chat/completions"
            
            # Map request data to GitHub Models API format
            payload = {
                "model": model_id,
                "messages": request_data.get("messages", []),
                "stream": False
            }
            
            # Add parameters if provided
            if "parameters" in request_data:
                params = request_data["parameters"]
                if "temperature" in params:
                    payload["temperature"] = params["temperature"]
                if "max_tokens" in params:
                    payload["max_tokens"] = params["max_tokens"]
                if "top_p" in params:
                    payload["top_p"] = params["top_p"]
            
            # Apply rate limiting
            await self._apply_rate_limiting()
            
            async with self.session.post(url, json=payload) as response:
                self.request_count += 1
                
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "model_id": model_id,
                        "response": result,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"GitHub Models API error: {response.status} - {error_text}")
                    return {
                        "success": False,
                        "error": f"API error: {response.status} - {error_text}",
                        "model_id": model_id
                    }
                    
        except Exception as e:
            logger.error(f"Error evaluating model {model_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_id": model_id
            }
    
    async def get_model_details(self, model_id: str) -> Optional[GitHubModel]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_id: Model ID to get details for
            
        Returns:
            GitHubModel object or None if not found
        """
        try:
            all_models = await self.get_available_models()
            
            for model in all_models:
                if model.id == model_id:
                    return model
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting model details for {model_id}: {e}")
            return None
    
    async def _apply_rate_limiting(self):
        """Apply rate limiting between requests."""
        current_time = asyncio.get_event_loop().time()
        time_since_last_request = current_time - self.last_request_time
        
        # Minimum 1 second between requests to be respectful
        if time_since_last_request < 1.0:
            await asyncio.sleep(1.0 - time_since_last_request)
        
        self.last_request_time = asyncio.get_event_loop().time()
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status."""
        return {
            "request_count": self.request_count,
            "last_request_time": self.last_request_time,
            "has_token": self.github_token is not None,
            "token_source": "environment" if os.getenv("GITHUB_TOKEN") else "github_cli" if self.github_token else "none"
        }
    
    def get_provider_categories(self) -> Dict[str, str]:
        """Get mapping of providers to categories."""
        return self.PROVIDER_CATEGORIES.copy()
    
    def categorize_model(self, model_id: str) -> str:
        """
        Categorize a model based on its ID.
        
        Args:
            model_id: Model ID to categorize
            
        Returns:
            Category string
        """
        for provider_pattern, category in self.PROVIDER_CATEGORIES.items():
            if provider_pattern == "default":
                continue
            
            provider = provider_pattern.replace("/*", "")
            if model_id.startswith(provider + "/"):
                return category
        
        return self.PROVIDER_CATEGORIES["default"]
