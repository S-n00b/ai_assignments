"""
Ollama Category Loader

This module provides category-based model loading for Ollama models,
leveraging Ollama's native categorization system for streamlined model discovery.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class OllamaModel:
    """Represents an Ollama model with category information."""
    name: str
    size: int
    digest: str
    modified_at: datetime
    category: str
    capabilities: List[str]
    parameters: Dict[str, Any]
    status: str = "available"


class OllamaCategoryLoader:
    """Loads and manages Ollama model categories."""
    
    # Ollama native categories mapping
    OLLAMA_CATEGORIES = {
        "embedding": "embedding",
        "vision": "multimodal", 
        "tools": "function_calling",
        "thinking": "reasoning"
    }
    
    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        """Initialize the Ollama category loader."""
        self.ollama_base_url = ollama_base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.categories = {
            "embedding": [],
            "vision": [],
            "tools": [],
            "thinking": [],
            "cloud": []
        }
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = 300  # 5 minutes
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def load_models_by_category(self, category: str) -> List[OllamaModel]:
        """
        Load models for a specific category.
        
        Args:
            category: Category to load models for
            
        Returns:
            List of OllamaModel objects
        """
        try:
            # Check cache first
            if self._is_cache_valid(category):
                logger.info(f"Using cached models for category: {category}")
                return self.cache.get(category, [])
            
            # Load all models and filter by category
            all_models = await self._load_all_models()
            category_models = []
            
            for model in all_models:
                if self._categorize_model(model.name) == category:
                    category_models.append(model)
            
            # Cache results
            self.cache[category] = category_models
            self.cache_expiry[category] = datetime.now().timestamp() + self.cache_duration
            
            logger.info(f"Loaded {len(category_models)} models for category: {category}")
            return category_models
            
        except Exception as e:
            logger.error(f"Error loading models for category {category}: {e}")
            return []
    
    async def sync_all_categories(self) -> Dict[str, List[OllamaModel]]:
        """
        Sync all categories with Ollama.
        
        Returns:
            Dictionary mapping categories to their models
        """
        try:
            logger.info("Syncing all categories with Ollama...")
            
            # Load all models
            all_models = await self._load_all_models()
            
            # Clear existing categories
            for category in self.categories:
                self.categories[category] = []
            
            # Categorize models
            for model in all_models:
                category = self._categorize_model(model.name)
                if category in self.categories:
                    self.categories[category].append(model)
                else:
                    # Add to cloud category for unknown models
                    self.categories["cloud"].append(model)
            
            # Update cache
            for category, models in self.categories.items():
                self.cache[category] = models
                self.cache_expiry[category] = datetime.now().timestamp() + self.cache_duration
            
            total_models = sum(len(models) for models in self.categories.values())
            logger.info(f"Synced {total_models} models across {len(self.categories)} categories")
            
            return self.categories
            
        except Exception as e:
            logger.error(f"Error syncing categories: {e}")
            return {}
    
    async def get_model_details(self, model_name: str) -> Optional[OllamaModel]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            OllamaModel object or None if not found
        """
        try:
            if not self.session:
                raise Exception("Session not initialized. Use async context manager.")
            
            # Get model info from Ollama API
            url = f"{self.ollama_base_url}/api/show"
            payload = {"name": model_name}
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Create OllamaModel object
                    model = OllamaModel(
                        name=data.get("modelfile", {}).get("name", model_name),
                        size=data.get("size", 0),
                        digest=data.get("digest", ""),
                        modified_at=datetime.now(),
                        category=self._categorize_model(model_name),
                        capabilities=self._extract_capabilities(data),
                        parameters=self._extract_parameters(data)
                    )
                    
                    return model
                else:
                    logger.error(f"Failed to get model details for {model_name}: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting model details for {model_name}: {e}")
            return None
    
    async def _load_all_models(self) -> List[OllamaModel]:
        """Load all available models from Ollama."""
        try:
            if not self.session:
                raise Exception("Session not initialized. Use async context manager.")
            
            url = f"{self.ollama_base_url}/api/tags"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    models = []
                    
                    for model_data in data.get("models", []):
                        model = OllamaModel(
                            name=model_data["name"],
                            size=model_data.get("size", 0),
                            digest=model_data.get("digest", ""),
                            modified_at=datetime.fromisoformat(
                                model_data["modified_at"].replace("Z", "+00:00")
                            ),
                            category=self._categorize_model(model_data["name"]),
                            capabilities=self._infer_capabilities(model_data["name"]),
                            parameters={}
                        )
                        models.append(model)
                    
                    return models
                else:
                    logger.error(f"Failed to load models: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error loading all models: {e}")
            return []
    
    def _categorize_model(self, model_name: str) -> str:
        """
        Categorize a model based on its name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Category string
        """
        model_lower = model_name.lower()
        
        # Check for embedding models
        if any(keyword in model_lower for keyword in ["embed", "sentence", "all-minilm"]):
            return "embedding"
        
        # Check for vision models
        if any(keyword in model_lower for keyword in ["llava", "vision", "instructblip", "blip"]):
            return "vision"
        
        # Check for tool models
        if any(keyword in model_lower for keyword in ["function", "tool", "code"]):
            return "tools"
        
        # Check for thinking models
        if any(keyword in model_lower for keyword in ["llama", "mistral", "gemma", "qwen", "thinking"]):
            return "thinking"
        
        # Default to thinking for general language models
        return "thinking"
    
    def _infer_capabilities(self, model_name: str) -> List[str]:
        """Infer model capabilities from name."""
        capabilities = []
        model_lower = model_name.lower()
        
        if any(keyword in model_lower for keyword in ["embed", "sentence"]):
            capabilities.append("text-embedding")
        
        if any(keyword in model_lower for keyword in ["llava", "vision", "blip"]):
            capabilities.extend(["text-generation", "vision"])
        elif any(keyword in model_lower for keyword in ["code", "function"]):
            capabilities.extend(["text-generation", "code-generation", "function-calling"])
        else:
            capabilities.append("text-generation")
        
        # Add common capabilities
        capabilities.extend(["completion", "chat"])
        
        return capabilities
    
    def _extract_capabilities(self, model_data: Dict[str, Any]) -> List[str]:
        """Extract capabilities from detailed model data."""
        capabilities = []
        
        # Extract from modelfile template if available
        template = model_data.get("modelfile", {}).get("template", "")
        if "{{ if .System }}" in template:
            capabilities.append("system-message")
        
        # Add basic capabilities
        capabilities.extend(["text-generation", "completion", "chat"])
        
        return capabilities
    
    def _extract_parameters(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters from detailed model data."""
        parameters = {}
        
        # Extract from modelfile parameters
        modelfile = model_data.get("modelfile", {})
        
        if "temperature" in modelfile:
            parameters["temperature"] = modelfile["temperature"]
        
        if "top_p" in modelfile:
            parameters["top_p"] = modelfile["top_p"]
        
        if "top_k" in modelfile:
            parameters["top_k"] = modelfile["top_k"]
        
        return parameters
    
    def _is_cache_valid(self, category: str) -> bool:
        """Check if cache is valid for a category."""
        if category not in self.cache or category not in self.cache_expiry:
            return False
        
        return datetime.now().timestamp() < self.cache_expiry[category]
    
    def get_available_categories(self) -> List[str]:
        """Get list of available categories."""
        return list(self.OLLAMA_CATEGORIES.keys())
    
    def get_category_mapping(self) -> Dict[str, str]:
        """Get mapping of Ollama categories to system categories."""
        return self.OLLAMA_CATEGORIES.copy()
    
    def clear_cache(self):
        """Clear the model cache."""
        self.cache.clear()
        self.cache_expiry.clear()
        logger.info("Model cache cleared")
