"""
Utilities Package for Lenovo AAITC Solutions

This package provides shared utilities and common functionality for both
Assignment 1 (Model Evaluation) and Assignment 2 (AI Architecture) solutions.

Key Components:
- LoggingSystem: Comprehensive multi-layer logging architecture
- VisualizationUtils: Plotting and chart generation utilities
- DataUtils: Data processing and manipulation utilities
- ConfigUtils: Configuration management utilities
"""

from .logging_system import LoggingSystem, LogLevel, LogCategory
from .visualization import VisualizationUtils
from .data_utils import DataUtils
from .config_utils import ConfigUtils

__version__ = "1.0.0"
__author__ = "Lenovo AAITC"

__all__ = [
    "LoggingSystem",
    "LogLevel", 
    "LogCategory",
    "VisualizationUtils",
    "DataUtils",
    "ConfigUtils"
]
