"""
Lenovo AAITC - Model Evaluation Framework

This package provides comprehensive model evaluation capabilities for foundation models,
including the latest versions (GPT-5, GPT-5-Codex, Claude 3.5 Sonnet, Llama 3.3).

Key Components:
- ModelConfig: Configuration management for latest model versions
- EvaluationPipeline: Comprehensive evaluation framework
- RobustnessTesting: Adversarial and edge case testing
- BiasDetection: Multi-dimensional bias analysis
- PromptRegistries: Integration with open-source prompt databases

Author: Lenovo AAITC Technical Assignment
Date: Q3 2025
"""

from .config import ModelConfig, EvaluationMetrics, TaskType
from .pipeline import ComprehensiveEvaluationPipeline
from .robustness import RobustnessTestingSuite
from .bias_detection import BiasDetectionSystem
from .prompt_registries import PromptRegistryManager

__version__ = "1.0.0"
__author__ = "Lenovo AAITC"

__all__ = [
    "ModelConfig",
    "EvaluationMetrics", 
    "TaskType",
    "ComprehensiveEvaluationPipeline",
    "RobustnessTestingSuite",
    "BiasDetectionSystem",
    "PromptRegistryManager"
]
