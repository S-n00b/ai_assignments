"""
GitHub Models Loader

This module provides model loading functionality for GitHub Models API,
including provider-based loading and model validation.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from .api_client import GitHubModelsAPIClient, GitHubModel

logger = logging.getLogger(__name__)


@dataclass
class ModelLoadResult:
    """Result of model loading operation."""
    success: bool
    model: Optional[GitHubModel] = None
    error: Optional[str] = None
    load_time_ms: float = 0.0


class GitHubModelsLoader:
    """Loads models from GitHub Models API."""
    
    def __init__(self, api_client: GitHubModelsAPIClient):
        """Initialize the GitHub Models loader."""
        self.api_client = api_client
        self.loaded_models = {}
        self.load_history = []
        
    async def load_models_by_provider(self, provider: str) -> List[GitHubModel]:
        """
        Load models for a specific provider.
        
        Args:
            provider: Provider name (openai, meta, etc.)
            
        Returns:
            List of GitHubModel objects
        """
        try:
            logger.info(f"Loading models for provider: {provider}")
            
            models = await self.api_client.get_models_by_provider(provider)
            
            # Cache loaded models
            for model in models:
                self.loaded_models[model.id] = model
            
            logger.info(f"Loaded {len(models)} models for provider: {provider}")
            return models
            
        except Exception as e:
            logger.error(f"Error loading models for provider {provider}: {e}")
            return []
    
    async def load_all_models(self) -> List[GitHubModel]:
        """
        Load all available models.
        
        Returns:
            List of all GitHubModel objects
        """
        try:
            logger.info("Loading all GitHub Models")
            
            models = await self.api_client.get_available_models()
            
            # Cache loaded models
            for model in models:
                self.loaded_models[model.id] = model
            
            logger.info(f"Loaded {len(models)} total models")
            return models
            
        except Exception as e:
            logger.error(f"Error loading all models: {e}")
            return []
    
    async def load_model(self, model_id: str) -> ModelLoadResult:
        """
        Load a specific model with validation.
        
        Args:
            model_id: Model ID to load
            
        Returns:
            ModelLoadResult with model data and load status
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Loading model: {model_id}")
            
            # Check if model is already loaded
            if model_id in self.loaded_models:
                logger.info(f"Model {model_id} already loaded")
                return ModelLoadResult(
                    success=True,
                    model=self.loaded_models[model_id],
                    load_time_ms=0.0
                )
            
            # Validate model exists
            if not await self.validate_model_access(model_id):
                return ModelLoadResult(
                    success=False,
                    error=f"Model {model_id} not accessible"
                )
            
            # Get model details
            model_details = await self.api_client.get_model_details(model_id)
            if not model_details:
                return ModelLoadResult(
                    success=False,
                    error=f"Failed to get details for model {model_id}"
                )
            
            # Cache the model
            self.loaded_models[model_id] = model_details
            
            # Record load history
            load_time = (datetime.now() - start_time).total_seconds() * 1000
            self.load_history.append({
                "model_id": model_id,
                "load_time_ms": load_time,
                "timestamp": datetime.now(),
                "success": True
            })
            
            logger.info(f"Successfully loaded model {model_id} in {load_time:.2f}ms")
            
            return ModelLoadResult(
                success=True,
                model=model_details,
                load_time_ms=load_time
            )
            
        except Exception as e:
            load_time = (datetime.now() - start_time).total_seconds() * 1000
            error_msg = f"Error loading model {model_id}: {str(e)}"
            logger.error(error_msg)
            
            # Record failed load
            self.load_history.append({
                "model_id": model_id,
                "load_time_ms": load_time,
                "timestamp": datetime.now(),
                "success": False,
                "error": str(e)
            })
            
            return ModelLoadResult(
                success=False,
                error=error_msg,
                load_time_ms=load_time
            )
    
    async def load_models_batch(self, model_ids: List[str]) -> List[ModelLoadResult]:
        """
        Load multiple models efficiently.
        
        Args:
            model_ids: List of model IDs to load
            
        Returns:
            List of ModelLoadResult objects
        """
        logger.info(f"Loading {len(model_ids)} models in batch")
        
        # Create tasks for concurrent loading
        tasks = [self.load_model(model_id) for model_id in model_ids]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ModelLoadResult(
                    success=False,
                    error=f"Exception loading {model_ids[i]}: {str(result)}"
                ))
            else:
                processed_results.append(result)
        
        successful_loads = sum(1 for r in processed_results if r.success)
        logger.info(f"Batch load completed: {successful_loads}/{len(model_ids)} models loaded successfully")
        
        return processed_results
    
    async def validate_model_access(self, model_id: str) -> bool:
        """
        Validate that a model is accessible via API.
        
        Args:
            model_id: Model ID to validate
            
        Returns:
            True if model is accessible, False otherwise
        """
        try:
            # Check if model exists in available models
            model_details = await self.api_client.get_model_details(model_id)
            return model_details is not None
            
        except Exception as e:
            logger.error(f"Error validating model access for {model_id}: {e}")
            return False
    
    def get_loaded_models(self) -> Dict[str, GitHubModel]:
        """Get all currently loaded models."""
        return self.loaded_models.copy()
    
    def get_model_by_provider(self, provider: str) -> List[GitHubModel]:
        """Get loaded models by provider."""
        return [model for model in self.loaded_models.values() if model.provider == provider]
    
    def get_models_by_category(self, category: str) -> List[GitHubModel]:
        """Get loaded models by category."""
        return [model for model in self.loaded_models.values() if model.category == category]
    
    def get_load_statistics(self) -> Dict[str, Any]:
        """Get loading statistics."""
        if not self.load_history:
            return {
                "total_loads": 0,
                "successful_loads": 0,
                "failed_loads": 0,
                "average_load_time_ms": 0.0,
                "total_models_loaded": 0
            }
        
        successful_loads = [h for h in self.load_history if h["success"]]
        failed_loads = [h for h in self.load_history if not h["success"]]
        
        avg_load_time = 0.0
        if successful_loads:
            avg_load_time = sum(h["load_time_ms"] for h in successful_loads) / len(successful_loads)
        
        return {
            "total_loads": len(self.load_history),
            "successful_loads": len(successful_loads),
            "failed_loads": len(failed_loads),
            "average_load_time_ms": avg_load_time,
            "total_models_loaded": len(self.loaded_models),
            "success_rate": len(successful_loads) / len(self.load_history) if self.load_history else 0.0
        }
    
    def clear_loaded_models(self):
        """Clear all loaded models from memory."""
        self.loaded_models.clear()
        logger.info("Cleared all loaded models")
    
    def get_model_info(self, model_id: str) -> Optional[GitHubModel]:
        """Get information about a loaded model."""
        return self.loaded_models.get(model_id)
    
    def is_model_loaded(self, model_id: str) -> bool:
        """Check if a model is currently loaded."""
        return model_id in self.loaded_models
    
    def get_provider_statistics(self) -> Dict[str, int]:
        """Get statistics by provider."""
        stats = {}
        for model in self.loaded_models.values():
            stats[model.provider] = stats.get(model.provider, 0) + 1
        return stats
    
    def get_category_statistics(self) -> Dict[str, int]:
        """Get statistics by category."""
        stats = {}
        for model in self.loaded_models.values():
            stats[model.category] = stats.get(model.category, 0) + 1
        return stats
