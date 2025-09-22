"""
Enterprise LLMOps Frontend Package

This package contains the frontend components for the Enterprise LLMOps platform,
including FastAPI applications, modern dashboards, and copilot integrations.
"""

from .fastapi_app import app
from .modern_dashboard import ModernDashboard
from .copilot_integration import CopilotIntegration

__all__ = ["app", "ModernDashboard", "CopilotIntegration"]

