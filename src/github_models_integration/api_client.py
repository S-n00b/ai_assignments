"""
GitHub Models API Client for Brantwood Organization

This module provides comprehensive integration with GitHub Models API for
accessing and managing models in the Brantwood organization repository.
"""

import asyncio
import aiohttp
import json
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import yaml

from .inference_client import GitHubModelsInferenceClient


@dataclass
class GitHubModel:
    """Represents a GitHub Models package."""
    name: str
    full_name: str
    description: str
    package_id: int
    version: str
    visibility: str
    created_at: datetime
    updated_at: datetime
    download_count: int
    metadata: Dict[str, Any]


@dataclass
class GitHubModelsResponse:
    """Represents a response from GitHub Models API."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    models: Optional[List[GitHubModel]] = None
    error: Optional[str] = None
    rate_limit_remaining: Optional[int] = None


class GitHubModelsAPIClient:
    """Client for GitHub Models API integration with Brantwood organization."""
    
    def __init__(self, config_path: str = "config/small_models_config.yaml"):
        """Initialize the GitHub Models API client."""
        self.config = self._load_config(config_path)
        self.session = None
        self.logger = self._setup_logging()
        
        # GitHub API configuration
        self.base_url = "https://api.github.com"
        self.organization = self.config["github_models_api"]["organization"]
        self.token = self._get_auth_token()
        
        # Initialize inference client
        self.inference_client = GitHubModelsInferenceClient(organization=self.organization)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                return self._get_default_config()
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "github_models_api": {
                "organization": "Brantwood",
                "rate_limits": {
                    "requests_per_hour": 5000,
                    "requests_per_minute": 100
                }
            }
        }
    
    def _get_auth_token(self) -> Optional[str]:
        """Get authentication token from environment."""
        return os.getenv("GITHUB_MODELS_TOKEN")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for GitHub Models API client."""
        logger = logging.getLogger("github_models_api")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def initialize(self):
        """Initialize the GitHub Models API client."""
        try:
            # Initialize HTTP session with authentication
            headers = {}
            if self.token:
                headers["Authorization"] = f"token {self.token}"
            
            headers.update({
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Lenovo-AAITC-Models-Client/1.0"
            })
            
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            # Initialize inference client
            await self.inference_client.initialize()
            
            self.logger.info("GitHub Models API client initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize GitHub Models API client: {e}")
            raise
    
    async def list_organization_packages(self, package_type: str = "container") -> GitHubModelsResponse:
        """List all packages in the organization."""
        try:
            url = f"{self.base_url}/orgs/{self.organization}/packages"
            params = {"package_type": package_type}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    models = await self._parse_packages_data(data)
                    
                    return GitHubModelsResponse(
                        success=True,
                        models=models,
                        rate_limit_remaining=self._get_rate_limit_remaining(response)
                    )
                else:
                    error_msg = f"Failed to list packages: HTTP {response.status}"
                    return GitHubModelsResponse(success=False, error=error_msg)
                    
        except Exception as e:
            self.logger.error(f"Error listing organization packages: {e}")
            return GitHubModelsResponse(success=False, error=str(e))
    
    async def _parse_packages_data(self, data: List[Dict[str, Any]]) -> List[GitHubModel]:
        """Parse packages data from GitHub API response."""
        models = []
        
        for package_data in data:
            try:
                model = GitHubModel(
                    name=package_data["name"],
                    full_name=package_data["full_name"],
                    description=package_data.get("description", ""),
                    package_id=package_data["id"],
                    version=package_data.get("version", "latest"),
                    visibility=package_data.get("visibility", "private"),
                    created_at=datetime.fromisoformat(
                        package_data["created_at"].replace("Z", "+00:00")
                    ),
                    updated_at=datetime.fromisoformat(
                        package_data["updated_at"].replace("Z", "+00:00")
                    ),
                    download_count=package_data.get("download_count", 0),
                    metadata=package_data.get("metadata", {})
                )
                models.append(model)
                
            except Exception as e:
                self.logger.warning(f"Failed to parse package data: {e}")
                continue
        
        return models
    
    def _get_rate_limit_remaining(self, response: aiohttp.ClientResponse) -> Optional[int]:
        """Get remaining rate limit from response headers."""
        remaining = response.headers.get("X-RateLimit-Remaining")
        return int(remaining) if remaining else None
    
    async def get_model_catalog(self):
        """Get the catalog of available models from GitHub Models."""
        try:
            return await self.inference_client.get_model_catalog()
        except Exception as e:
            self.logger.error(f"Error getting model catalog: {e}")
            return []
    
    async def test_small_models_inference(self):
        """Test inference with small models for Phase 2."""
        try:
            return await self.inference_client.test_small_models()
        except Exception as e:
            self.logger.error(f"Error testing small models inference: {e}")
            return {"error": str(e)}
    
    async def run_phase2_evaluation(self, model_id: str, prompt: str):
        """Run Phase 2 evaluation with a specific model."""
        try:
            return await self.inference_client.run_phase2_evaluation(model_id, prompt)
        except Exception as e:
            self.logger.error(f"Error running Phase 2 evaluation: {e}")
            return {"error": str(e)}
    
    async def shutdown(self):
        """Shutdown the GitHub Models API client."""
        try:
            if self.session:
                await self.session.close()
            
            # Shutdown inference client
            await self.inference_client.shutdown()
            
            self.logger.info("GitHub Models API client shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")