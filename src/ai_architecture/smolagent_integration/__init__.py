"""
SmolAgent Integration Package

This package provides SmolAgent integration for agentic workflows with small models,
optimized for mobile and edge deployment scenarios.

Key Components:
- SmolAgent workflow designer for creating agentic workflows
- Mobile agent optimization for edge deployment
- Agent performance monitoring and analytics
- MLflow integration for agent experiment tracking
"""

from .smolagent_workflow_designer import SmolAgentWorkflowDesigner
from .mobile_agent_optimization import MobileAgentOptimizer
from .agent_performance_monitor import AgentPerformanceMonitor
from .mlflow_agent_tracking import MLflowAgentTracker

__all__ = [
    "SmolAgentWorkflowDesigner",
    "MobileAgentOptimizer", 
    "AgentPerformanceMonitor",
    "MLflowAgentTracker"
]

__version__ = "1.0.0"
__author__ = "AI Architecture Team"
