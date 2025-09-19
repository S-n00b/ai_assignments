"""
Enterprise LLMOps Platform for Lenovo AAITC Solutions - Assignment 2

This module provides a comprehensive enterprise-grade LLM Operations platform
featuring Ollama integration, advanced model management, and robust frontend
capabilities for production AI model deployment and monitoring.

Key Components:
- Ollama Integration for local model management
- Enterprise Model Registry and Lifecycle Management
- Advanced Monitoring and Observability
- Robust API Gateway and Load Balancing
- Enterprise Security and Compliance
- Scalable Infrastructure Management
"""

__version__ = "2.0.0"
__author__ = "Lenovo AAITC Team"

from .ollama_manager import OllamaManager
from .model_registry import EnterpriseModelRegistry
from .api_gateway import APIGateway
from .monitoring import LLMOpsMonitoring
from .security import EnterpriseSecurity
from .infrastructure import InfrastructureManager

__all__ = [
    "OllamaManager",
    "EnterpriseModelRegistry", 
    "APIGateway",
    "LLMOpsMonitoring",
    "EnterpriseSecurity",
    "InfrastructureManager"
]
