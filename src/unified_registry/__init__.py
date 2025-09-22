"""
Unified Registry Module

This module provides unified model management for both local (Ollama) and remote
(GitHub Models) model sources, creating a streamlined interface for model discovery,
serving, and lifecycle management.

Key Features:
- Unified model object structure for local and remote models
- Model serving abstraction with local/remote capabilities
- Registry management with dual-source support
- Model discovery and filtering by category
- Integration with both Ollama and GitHub Models
"""

from .model_objects import UnifiedModelObject, ModelCapability, ModelServingConfig
from .registry_manager import UnifiedRegistryManager
from .serving_interface import ModelServingInterface

__all__ = [
    "UnifiedModelObject",
    "ModelCapability", 
    "ModelServingConfig",
    "UnifiedRegistryManager",
    "ModelServingInterface"
]
