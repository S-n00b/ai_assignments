"""
Minimal pytest configuration for testing without heavy dependencies.
This configuration focuses on testing basic functionality and structure.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator
from unittest.mock import Mock, AsyncMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import only basic modules that don't require heavy dependencies
from src.model_evaluation.config import ModelConfig
from src.utils.config_utils import ConfigUtils
from src.utils.data_utils import DataValidator

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_model_config():
    """Sample model configuration for testing."""
    return ModelConfig(
        model_name="gpt-3.5-turbo",
        model_version="2024-01-01",
        api_key="test-key",
        max_tokens=1000,
        temperature=0.7,
        top_p=0.9
    )


@pytest.fixture
def sample_evaluation_data():
    """Sample evaluation data for testing."""
    return {
        "prompts": [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "Write a Python function to calculate fibonacci numbers."
        ],
        "expected_outputs": [
            "Paris",
            "Quantum computing uses quantum mechanics...",
            "def fibonacci(n): ..."
        ],
        "task_types": ["qa", "explanation", "code_generation"]
    }


@pytest.fixture
def sample_metrics_data():
    """Sample metrics data for testing."""
    return pd.DataFrame({
        "model": ["gpt-3.5-turbo", "claude-3-sonnet", "llama-2-70b"],
        "bleu_score": [0.85, 0.82, 0.78],
        "rouge_score": [0.88, 0.85, 0.81],
        "bert_score": [0.92, 0.89, 0.86],
        "latency_ms": [1200, 1500, 2000],
        "cost_per_1k_tokens": [0.002, 0.003, 0.001]
    })


@pytest.fixture
def mock_api_client():
    """Mock API client for testing without actual API calls."""
    mock_client = Mock()
    mock_client.generate = AsyncMock(return_value={
        "choices": [{"message": {"content": "Test response"}}],
        "usage": {"total_tokens": 100}
    })
    mock_client.chat = AsyncMock(return_value={
        "choices": [{"message": {"content": "Test chat response"}}],
        "usage": {"total_tokens": 150}
    })
    return mock_client


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "id": "doc1",
            "content": "Artificial intelligence is transforming industries worldwide.",
            "metadata": {"source": "tech_article", "date": "2024-01-01"}
        },
        {
            "id": "doc2", 
            "content": "Machine learning algorithms require large datasets for training.",
            "metadata": {"source": "research_paper", "date": "2024-01-02"}
        },
        {
            "id": "doc3",
            "content": "Natural language processing enables computers to understand human language.",
            "metadata": {"source": "textbook", "date": "2024-01-03"}
        }
    ]


@pytest.fixture
def sample_performance_metrics():
    """Sample performance metrics for testing."""
    return {
        "throughput": 100,  # requests per minute
        "latency_p50": 500,  # milliseconds
        "latency_p95": 1200,
        "latency_p99": 2000,
        "error_rate": 0.01,
        "cpu_usage": 0.75,
        "memory_usage": 0.60,
        "gpu_usage": 0.85
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "api: mark test as requiring API access"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        
        # Add slow marker for tests that take > 5 seconds
        if "slow" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.slow)
