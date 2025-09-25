"""
Lenovo AAITC - AI Architecture Framework

This package provides comprehensive AI architecture capabilities for enterprise-scale
AI systems, including hybrid platforms, model lifecycle management, agentic computing,
and advanced RAG systems.

Key Components:
- HybridAIPlatform: Enterprise hybrid AI platform architecture
- ModelLifecycleManager: Complete MLOps pipeline and model lifecycle management
- AgenticComputingFramework: Multi-agent systems and intelligent orchestration
- RAGSystem: Advanced retrieval-augmented generation with enterprise features

Author: Lenovo AAITC Technical Assignment
Date: Q3 2025
"""

# Import core modules with optional dependencies
try:
    from .platform import HybridAIPlatform, DeploymentTarget, InfrastructureConfig
    PLATFORM_AVAILABLE = True
except ImportError as e:
    PLATFORM_AVAILABLE = False
    logging.warning(f"Platform module not available: {e}")

try:
    from .lifecycle import ModelLifecycleManager, ModelVersion, DeploymentStrategy
    LIFECYCLE_AVAILABLE = True
except ImportError as e:
    LIFECYCLE_AVAILABLE = False
    logging.warning(f"Lifecycle module not available: {e}")

try:
    from .agents import AgenticComputingFramework, BaseAgent, AgentMessage, MessageType
    AGENTS_AVAILABLE = True
except ImportError as e:
    AGENTS_AVAILABLE = False
    logging.warning(f"Agents module not available: {e}")

try:
    from .rag_system import RAGSystem, DocumentChunk, DocumentMetadata, ChunkingStrategy
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    logging.warning(f"RAG system module not available: {e}")

__version__ = "1.0.0"
__author__ = "Lenovo AAITC"

__all__ = [
    "HybridAIPlatform",
    "DeploymentTarget", 
    "InfrastructureConfig",
    "ModelLifecycleManager",
    "ModelVersion",
    "DeploymentStrategy",
    "AgenticComputingFramework",
    "BaseAgent",
    "AgentMessage",
    "MessageType",
    "RAGSystem",
    "DocumentChunk",
    "DocumentMetadata",
    "ChunkingStrategy"
]
