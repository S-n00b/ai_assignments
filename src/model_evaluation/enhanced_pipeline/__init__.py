"""
Enhanced Model Evaluation Pipeline for Phase 5

This module provides comprehensive model evaluation capabilities for:
- Raw foundation model testing
- Custom AI Architect model testing  
- Mobile/edge specific evaluation
- Agentic workflow evaluation (SmolAgent/LangGraph)
- Retrieval workflow evaluation (LangChain/LlamaIndex)
- Factory roster integration and management

Key Features:
- Unified evaluation framework for all model types
- MLflow experiment integration
- Performance benchmarking
- Stress testing capabilities
- Production deployment validation
"""

from .raw_foundation_evaluation import RawFoundationEvaluator
from .custom_model_evaluation import CustomModelEvaluator
from .mobile_model_evaluation import MobileModelEvaluator
from .agentic_workflow_evaluation import AgenticWorkflowEvaluator
from .retrieval_workflow_evaluation import RetrievalWorkflowEvaluator
from .factory_roster_management import FactoryRosterManager

__all__ = [
    "RawFoundationEvaluator",
    "CustomModelEvaluator", 
    "MobileModelEvaluator",
    "AgenticWorkflowEvaluator",
    "RetrievalWorkflowEvaluator",
    "FactoryRosterManager"
]
