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

from .platform import HybridAIPlatform, DeploymentTarget, InfrastructureConfig
from .lifecycle import ModelLifecycleManager, ModelVersion, DeploymentStrategy
from .agents import AgenticComputingFramework, BaseAgent, AgentMessage, MessageType
from .rag_system import RAGSystem, DocumentChunk, DocumentMetadata, ChunkingStrategy

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
