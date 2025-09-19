"""
GitHub Models Backend Integration

This module provides a backend service inspired by GitHub Models for model evaluation,
prototyping, and monitoring. It integrates with free AI APIs and provides rate limiting
for public showcase usage.

Key Features:
- Free AI model integration (Hugging Face, OpenAI free tier, etc.)
- Rate limiting for public showcase
- Model evaluation and monitoring
- No API key requirements for basic functionality
- Caching and optimization
"""

from .github_models_client import GitHubModelsClient
from .rate_limiter import RateLimiter
from .model_evaluator import ModelEvaluator
from .cache_manager import CacheManager

__all__ = [
    "GitHubModelsClient",
    "RateLimiter", 
    "ModelEvaluator",
    "CacheManager"
]

