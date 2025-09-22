"""
Ollama Registry Sync

This module provides synchronization between Ollama models and the unified registry,
ensuring that local Ollama models are properly integrated into the system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json

from .category_loader import OllamaCategoryLoader
from .model_loader import OllamaModelLoader, OllamaModel

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    """Result of registry synchronization operation."""
    success: bool
    total_models: int = 0
    synced_models: int = 0
    failed_models: int = 0
    categories: Dict[str, int] = None
    duration_seconds: float = 0.0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.categories is None:
            self.categories = {}
        if self.errors is None:
            self.errors = []


class OllamaRegistrySync:
    """Synchronizes Ollama models with the unified registry."""
    
    def __init__(self, model_loader: OllamaModelLoader, unified_registry=None):
        """
        Initialize the registry sync.
        
        Args:
            model_loader: OllamaModelLoader instance
            unified_registry: Unified registry instance (optional for now)
        """
        self.model_loader = model_loader
        self.unified_registry = unified_registry
        self.category_loader = model_loader.category_loader
        self.sync_history = []
        
    async def sync_models(self) -> SyncResult:
        """
        Sync all Ollama models with unified registry.
        
        Returns:
            SyncResult with sync statistics
        """
        start_time = datetime.now()
        logger.info("Starting Ollama models sync with unified registry")
        
        try:
            # Get all categories and their models
            categories = await self.category_loader.sync_all_categories()
            
            total_models = sum(len(models) for models in categories.values())
            synced_models = 0
            failed_models = 0
            errors = []
            
            # Sync models by category
            for category, models in categories.items():
                category_result = await self.sync_category(category)
                synced_models += category_result.synced_models
                failed_models += category_result.failed_models
                errors.extend(category_result.errors)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            result = SyncResult(
                success=failed_models == 0,
                total_models=total_models,
                synced_models=synced_models,
                failed_models=failed_models,
                categories={cat: len(models) for cat, models in categories.items()},
                duration_seconds=duration,
                errors=errors
            )
            
            # Record sync history
            self.sync_history.append({
                "timestamp": start_time,
                "result": result,
                "success": result.success
            })
            
            logger.info(f"Sync completed: {synced_models}/{total_models} models synced in {duration:.2f}s")
            
            return result
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = f"Sync failed: {str(e)}"
            logger.error(error_msg)
            
            return SyncResult(
                success=False,
                duration_seconds=duration,
                errors=[error_msg]
            )
    
    async def sync_category(self, category: str) -> SyncResult:
        """
        Sync models for a specific category.
        
        Args:
            category: Category to sync
            
        Returns:
            SyncResult for the category
        """
        logger.info(f"Syncing models for category: {category}")
        
        try:
            # Load models for the category
            models = await self.category_loader.load_models_by_category(category)
            
            if not models:
                logger.info(f"No models found for category: {category}")
                return SyncResult(
                    success=True,
                    total_models=0,
                    synced_models=0,
                    failed_models=0,
                    categories={category: 0}
                )
            
            # Load models with full metadata
            model_names = [model.name for model in models]
            load_results = await self.model_loader.load_models_batch(model_names)
            
            synced_models = 0
            failed_models = 0
            errors = []
            
            # Process each model
            for i, load_result in enumerate(load_results):
                if load_result.success and load_result.model:
                    # Convert to unified model format
                    unified_model = self._convert_to_unified_model(load_result.model)
                    
                    # Register with unified registry (if available)
                    if self.unified_registry:
                        try:
                            await self.unified_registry.register_model(unified_model)
                            synced_models += 1
                            logger.debug(f"Synced model: {unified_model.name}")
                        except Exception as e:
                            failed_models += 1
                            error_msg = f"Failed to register {model_names[i]}: {str(e)}"
                            errors.append(error_msg)
                            logger.error(error_msg)
                    else:
                        # Just count as synced if no registry available
                        synced_models += 1
                        logger.debug(f"Model loaded (no registry): {model_names[i]}")
                else:
                    failed_models += 1
                    error_msg = f"Failed to load {model_names[i]}: {load_result.error}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            
            result = SyncResult(
                success=failed_models == 0,
                total_models=len(models),
                synced_models=synced_models,
                failed_models=failed_models,
                categories={category: len(models)},
                errors=errors
            )
            
            logger.info(f"Category {category} sync: {synced_models}/{len(models)} models synced")
            
            return result
            
        except Exception as e:
            error_msg = f"Error syncing category {category}: {str(e)}"
            logger.error(error_msg)
            
            return SyncResult(
                success=False,
                errors=[error_msg]
            )
    
    async def update_model(self, model_name: str) -> bool:
        """
        Update a specific model in the registry.
        
        Args:
            model_name: Name of the model to update
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            logger.info(f"Updating model: {model_name}")
            
            # Load the model with fresh metadata
            load_result = await self.model_loader.load_model(model_name)
            
            if not load_result.success or not load_result.model:
                logger.error(f"Failed to load model {model_name} for update")
                return False
            
            # Convert to unified model format
            unified_model = self._convert_to_unified_model(load_result.model)
            
            # Update in unified registry (if available)
            if self.unified_registry:
                try:
                    await self.unified_registry.update_model(model_name, unified_model)
                    logger.info(f"Updated model: {model_name}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to update model {model_name}: {str(e)}")
                    return False
            else:
                logger.info(f"Model loaded (no registry to update): {model_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating model {model_name}: {str(e)}")
            return False
    
    def _convert_to_unified_model(self, ollama_model: OllamaModel) -> Dict[str, Any]:
        """
        Convert OllamaModel to unified model format.
        
        Args:
            ollama_model: OllamaModel object
            
        Returns:
            Dictionary in unified model format
        """
        return {
            "id": f"ollama_{ollama_model.name.replace(':', '_')}",
            "name": ollama_model.name,
            "version": "1.0.0",  # Default version
            "ollama_name": ollama_model.name,
            "category": ollama_model.category,
            "model_type": "base",
            "source": "ollama",
            "serving_type": "local",
            "capabilities": ollama_model.capabilities,
            "parameters": ollama_model.parameters,
            "local_endpoint": f"http://localhost:11434/api/generate",
            "remote_endpoint": None,
            "status": ollama_model.status,
            "created_at": ollama_model.modified_at.isoformat(),
            "updated_at": datetime.now().isoformat(),
            "description": f"Ollama model: {ollama_model.name}",
            "size_bytes": ollama_model.size,
            "digest": ollama_model.digest
        }
    
    def get_sync_statistics(self) -> Dict[str, Any]:
        """Get synchronization statistics."""
        if not self.sync_history:
            return {
                "total_syncs": 0,
                "successful_syncs": 0,
                "failed_syncs": 0,
                "last_sync": None,
                "average_sync_time": 0.0
            }
        
        successful_syncs = [h for h in self.sync_history if h["success"]]
        failed_syncs = [h for h in self.sync_history if not h["success"]]
        
        avg_sync_time = 0.0
        if successful_syncs:
            avg_sync_time = sum(h["result"].duration_seconds for h in successful_syncs) / len(successful_syncs)
        
        last_sync = self.sync_history[-1]["timestamp"] if self.sync_history else None
        
        return {
            "total_syncs": len(self.sync_history),
            "successful_syncs": len(successful_syncs),
            "failed_syncs": len(failed_syncs),
            "last_sync": last_sync.isoformat() if last_sync else None,
            "average_sync_time": avg_sync_time,
            "success_rate": len(successful_syncs) / len(self.sync_history) if self.sync_history else 0.0
        }
    
    def get_last_sync_result(self) -> Optional[SyncResult]:
        """Get the result of the last sync operation."""
        if self.sync_history:
            return self.sync_history[-1]["result"]
        return None
    
    def clear_sync_history(self):
        """Clear sync history."""
        self.sync_history.clear()
        logger.info("Sync history cleared")
