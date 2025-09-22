"""
Ollama Model Loader

This module provides individual model loading functionality for Ollama models,
including metadata extraction, validation, and performance optimization.
"""

import asyncio
import aiohttp
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import json

from .category_loader import OllamaModel, OllamaCategoryLoader

logger = logging.getLogger(__name__)


@dataclass
class ModelLoadResult:
    """Result of model loading operation."""
    success: bool
    model: Optional[OllamaModel] = None
    error: Optional[str] = None
    load_time_ms: float = 0.0


class OllamaModelLoader:
    """Loads individual models from Ollama with metadata extraction."""
    
    def __init__(self, category_loader: OllamaCategoryLoader):
        """Initialize the model loader."""
        self.category_loader = category_loader
        self.session: Optional[aiohttp.ClientSession] = None
        self.loaded_models = {}
        self.load_history = []
        
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
    
    async def load_model(self, model_name: str) -> ModelLoadResult:
        """
        Load a specific model with full metadata.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            ModelLoadResult with model data and load status
        """
        start_time = time.time()
        
        try:
            logger.info(f"Loading model: {model_name}")
            
            # Check if model is already loaded
            if model_name in self.loaded_models:
                logger.info(f"Model {model_name} already loaded")
                return ModelLoadResult(
                    success=True,
                    model=self.loaded_models[model_name],
                    load_time_ms=0.0
                )
            
            # Validate model exists
            if not await self.validate_model(model_name):
                return ModelLoadResult(
                    success=False,
                    error=f"Model {model_name} not found or not available"
                )
            
            # Get detailed model information
            model_details = await self.category_loader.get_model_details(model_name)
            if not model_details:
                return ModelLoadResult(
                    success=False,
                    error=f"Failed to get details for model {model_name}"
                )
            
            # Test model inference capability
            inference_test = await self._test_model_inference(model_name)
            if not inference_test["success"]:
                logger.warning(f"Model {model_name} inference test failed: {inference_test['error']}")
                # Don't fail loading, but mark as potentially problematic
            
            # Store loaded model
            self.loaded_models[model_name] = model_details
            
            # Record load history
            load_time = (time.time() - start_time) * 1000
            self.load_history.append({
                "model_name": model_name,
                "load_time_ms": load_time,
                "timestamp": datetime.now(),
                "success": True
            })
            
            logger.info(f"Successfully loaded model {model_name} in {load_time:.2f}ms")
            
            return ModelLoadResult(
                success=True,
                model=model_details,
                load_time_ms=load_time
            )
            
        except Exception as e:
            load_time = (time.time() - start_time) * 1000
            error_msg = f"Error loading model {model_name}: {str(e)}"
            logger.error(error_msg)
            
            # Record failed load
            self.load_history.append({
                "model_name": model_name,
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
    
    async def load_models_batch(self, model_names: List[str]) -> List[ModelLoadResult]:
        """
        Load multiple models efficiently.
        
        Args:
            model_names: List of model names to load
            
        Returns:
            List of ModelLoadResult objects
        """
        logger.info(f"Loading {len(model_names)} models in batch")
        
        # Create tasks for concurrent loading
        tasks = [self.load_model(model_name) for model_name in model_names]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ModelLoadResult(
                    success=False,
                    error=f"Exception loading {model_names[i]}: {str(result)}"
                ))
            else:
                processed_results.append(result)
        
        successful_loads = sum(1 for r in processed_results if r.success)
        logger.info(f"Batch load completed: {successful_loads}/{len(model_names)} models loaded successfully")
        
        return processed_results
    
    async def validate_model(self, model_name: str) -> bool:
        """
        Validate that a model is available and working.
        
        Args:
            model_name: Name of the model to validate
            
        Returns:
            True if model is valid, False otherwise
        """
        try:
            if not self.session:
                raise Exception("Session not initialized. Use async context manager.")
            
            # Check if model exists in Ollama
            url = f"{self.category_loader.ollama_base_url}/api/tags"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    model_names = [model["name"] for model in data.get("models", [])]
                    return model_name in model_names
                else:
                    logger.error(f"Failed to validate model {model_name}: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error validating model {model_name}: {e}")
            return False
    
    async def _test_model_inference(self, model_name: str) -> Dict[str, Any]:
        """
        Test model inference capability with a simple prompt.
        
        Args:
            model_name: Name of the model to test
            
        Returns:
            Dictionary with test results
        """
        try:
            if not self.session:
                raise Exception("Session not initialized. Use async context manager.")
            
            # Simple test prompt
            test_prompt = "Hello, this is a test. Please respond with 'Test successful'."
            
            payload = {
                "model": model_name,
                "prompt": test_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "max_tokens": 50
                }
            }
            
            url = f"{self.category_loader.ollama_base_url}/api/generate"
            
            async with self.session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    response_text = data.get("response", "")
                    
                    return {
                        "success": True,
                        "response": response_text,
                        "response_time_ms": data.get("total_duration", 0) / 1000000  # Convert from nanoseconds
                    }
                else:
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}"
                    }
                    
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "Inference test timeout"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_loaded_models(self) -> Dict[str, OllamaModel]:
        """Get all currently loaded models."""
        return self.loaded_models.copy()
    
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
    
    def get_model_info(self, model_name: str) -> Optional[OllamaModel]:
        """Get information about a loaded model."""
        return self.loaded_models.get(model_name)
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is currently loaded."""
        return model_name in self.loaded_models
