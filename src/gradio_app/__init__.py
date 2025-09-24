"""
Gradio Frontend for Lenovo AAITC Solutions

This package provides a production-ready Gradio frontend for both Assignment 1
(Model Evaluation) and Assignment 2 (AI Architecture) with MCP server integration.

Key Features:
- Interactive model evaluation interface
- Real-time performance monitoring
- MCP server integration for advanced capabilities
- Comprehensive visualization dashboards
- Export capabilities for reports and data
"""

from .main import create_gradio_app
from .mcp_server import MCPServer
from .components import (
    ModelEvaluationInterface,
    ModelProfilingInterface,
    ModelFactoryInterface,
    VisualizationDashboard,
    ReportGenerator
)

__version__ = "1.0.0"
__author__ = "Lenovo AAITC"

__all__ = [
    "create_gradio_app",
    "MCPServer",
    "ModelEvaluationInterface",
    "ModelProfilingInterface",
    "ModelFactoryInterface",
    "VisualizationDashboard",
    "ReportGenerator"
]
