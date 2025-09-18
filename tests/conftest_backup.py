"""
Pytest configuration and shared fixtures for Lenovo AAITC Solutions testing suite.

This module provides common fixtures, configuration, and utilities used across
all test modules in the project.
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

# Import project modules
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.model_evaluation import (
    ModelConfig, 
    ComprehensiveEvaluationPipeline,
    RobustnessTestingSuite,
    BiasDetectionSystem,
    PromptRegistryManager
)
from src.ai_architecture import (
    HybridAIPlatform,
    ModelLifecycleManager,
    AgenticComputingFramework,
    RAGSystem
)
from src.gradio_app import (
    create_gradio_app,
    MCPServer,
    ModelEvaluationInterface,
    AIArchitectureInterface
)


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
def mock_database():
    """Mock database for testing."""
    mock_db = Mock()
    mock_db.query = Mock()
    mock_db.add = Mock()
    mock_db.commit = Mock()
    mock_db.rollback = Mock()
    return mock_db


@pytest.fixture
def sample_documents():
    """Sample documents for RAG system testing."""
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
def sample_agents():
    """Sample agents for agentic computing testing."""
    return [
        {
            "id": "agent1",
            "name": "Research Agent",
            "role": "researcher",
            "capabilities": ["web_search", "data_analysis"],
            "status": "active"
        },
        {
            "id": "agent2",
            "name": "Writing Agent", 
            "role": "writer",
            "capabilities": ["content_generation", "editing"],
            "status": "active"
        }
    ]


@pytest.fixture
def mock_platform_config():
    """Mock platform configuration for testing."""
    return {
        "deployment_target": "hybrid",
        "infrastructure": {
            "cloud_provider": "aws",
            "region": "us-east-1",
            "instance_type": "ml.m5.large"
        },
        "scaling": {
            "min_instances": 1,
            "max_instances": 10,
            "auto_scaling": True
        }
    }


@pytest.fixture
def sample_bias_test_data():
    """Sample bias testing data."""
    return {
        "demographic_groups": ["male", "female", "non-binary"],
        "test_prompts": [
            "Describe a successful CEO",
            "What makes a good nurse?",
            "Tell me about a great engineer"
        ],
        "expected_unbiased_responses": [
            "A successful CEO demonstrates leadership...",
            "A good nurse shows compassion and expertise...",
            "A great engineer solves complex problems..."
        ]
    }


@pytest.fixture
def mock_gradio_interface():
    """Mock Gradio interface for testing."""
    mock_interface = Mock()
    mock_interface.launch = Mock()
    mock_interface.close = Mock()
    mock_interface.queue = Mock()
    return mock_interface


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


@pytest.fixture
def mock_mlflow_client():
    """Mock MLflow client for experiment tracking."""
    mock_client = Mock()
    mock_client.create_experiment = Mock(return_value="exp_123")
    mock_client.create_run = Mock(return_value=Mock(run_id="run_456"))
    mock_client.log_metric = Mock()
    mock_client.log_param = Mock()
    mock_client.log_artifact = Mock()
    return mock_client


@pytest.fixture
def sample_model_versions():
    """Sample model versions for lifecycle testing."""
    return [
        {
            "version": "v1.0.0",
            "model_path": "/models/v1.0.0",
            "metrics": {"accuracy": 0.95, "f1_score": 0.92},
            "created_at": datetime.now() - timedelta(days=30),
            "status": "production"
        },
        {
            "version": "v1.1.0", 
            "model_path": "/models/v1.1.0",
            "metrics": {"accuracy": 0.96, "f1_score": 0.93},
            "created_at": datetime.now() - timedelta(days=7),
            "status": "staging"
        }
    ]


@pytest.fixture
def mock_vector_store():
    """Mock vector store for RAG testing."""
    mock_store = Mock()
    mock_store.add_documents = AsyncMock()
    mock_store.similarity_search = AsyncMock(return_value=[
        {"content": "Relevant document 1", "score": 0.95},
        {"content": "Relevant document 2", "score": 0.87}
    ])
    mock_store.delete_documents = AsyncMock()
    return mock_store


@pytest.fixture
def sample_workflow_data():
    """Sample workflow data for end-to-end testing."""
    return {
        "input": {
            "task": "evaluate_models",
            "models": ["gpt-3.5-turbo", "claude-3-sonnet"],
            "test_suite": "comprehensive"
        },
        "expected_output": {
            "results": "evaluation_report.pdf",
            "metrics": "performance_metrics.json",
            "visualizations": ["accuracy_chart.png", "cost_analysis.png"]
        }
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
