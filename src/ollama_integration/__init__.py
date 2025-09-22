"""
Ollama Integration Module

This module provides Ollama-centric integration for local model management,
leveraging Ollama's native categorization (embedding, vision, tools, thinking)
and creating a streamlined interface for local model serving.

Key Features:
- Category-based model loading (embedding, vision, tools, thinking)
- Model metadata extraction and caching
- Registry synchronization with Ollama API
- Performance optimization with local caching
- Integration with unified model registry
"""

from .category_loader import OllamaCategoryLoader
from .model_loader import OllamaModelLoader
from .registry_sync import OllamaRegistrySync

__all__ = [
    "OllamaCategoryLoader",
    "OllamaModelLoader", 
    "OllamaRegistrySync"
]
