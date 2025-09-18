"""
Mock objects and fixtures for testing.

Provides reusable mock objects and fixtures for consistent testing
across the Lenovo AAITC Solutions test suite.
"""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class MockAPIClient:
    """Mock API client for testing without actual API calls."""
    
    def __init__(self, model_name: str = "test-model"):
        self.model_name = model_name
        self.generate = AsyncMock(return_value={
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {"total_tokens": 100, "prompt_tokens": 50, "completion_tokens": 50}
        })
        self.chat = AsyncMock(return_value={
            "choices": [{"message": {"content": "Test chat response"}}],
            "usage": {"total_tokens": 150, "prompt_tokens": 75, "completion_tokens": 75}
        })
        self.embeddings = AsyncMock(return_value={
            "data": [{"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}],
            "usage": {"total_tokens": 10}
        })


class MockDatabase:
    """Mock database for testing."""
    
    def __init__(self):
        self.query = Mock()
        self.add = Mock()
        self.commit = Mock()
        self.rollback = Mock()
        self.execute = Mock()
        self.fetchall = Mock(return_value=[])
        self.fetchone = Mock(return_value=None)


class MockVectorStore:
    """Mock vector store for RAG system testing."""
    
    def __init__(self):
        self.add_documents = AsyncMock(return_value=True)
        self.similarity_search = AsyncMock(return_value=[
            {"content": "Relevant document 1", "score": 0.95, "metadata": {"source": "doc1"}},
            {"content": "Relevant document 2", "score": 0.87, "metadata": {"source": "doc2"}}
        ])
        self.delete_documents = AsyncMock(return_value=True)
        self.update_documents = AsyncMock(return_value=True)
        self.get_document = AsyncMock(return_value={"content": "Document content", "metadata": {}})


class MockMLflowClient:
    """Mock MLflow client for experiment tracking."""
    
    def __init__(self):
        self.create_experiment = Mock(return_value="exp_123")
        self.create_run = Mock(return_value=Mock(run_id="run_456"))
        self.log_metric = Mock()
        self.log_param = Mock()
        self.log_artifact = Mock()
        self.get_experiment = Mock(return_value={"experiment_id": "exp_123", "name": "test_experiment"})
        self.get_run = Mock(return_value={"run_id": "run_456", "status": "FINISHED"})


class MockGradioInterface:
    """Mock Gradio interface for testing."""
    
    def __init__(self):
        self.launch = Mock()
        self.close = Mock()
        self.queue = Mock()
        self.load = Mock()
        self.reload = Mock()
        self.share = Mock()


class MockFileSystem:
    """Mock file system for testing file operations."""
    
    def __init__(self):
        self.files = {}
        self.directories = set()
    
    def exists(self, path: str) -> bool:
        return path in self.files or path in self.directories
    
    def read_file(self, path: str) -> str:
        return self.files.get(path, "")
    
    def write_file(self, path: str, content: str) -> None:
        self.files[path] = content
    
    def create_directory(self, path: str) -> None:
        self.directories.add(path)
    
    def delete_file(self, path: str) -> None:
        self.files.pop(path, None)


@pytest.fixture
def mock_api_client():
    """Fixture providing a mock API client."""
    return MockAPIClient()


@pytest.fixture
def mock_database():
    """Fixture providing a mock database."""
    return MockDatabase()


@pytest.fixture
def mock_vector_store():
    """Fixture providing a mock vector store."""
    return MockVectorStore()


@pytest.fixture
def mock_mlflow_client():
    """Fixture providing a mock MLflow client."""
    return MockMLflowClient()


@pytest.fixture
def mock_gradio_interface():
    """Fixture providing a mock Gradio interface."""
    return MockGradioInterface()


@pytest.fixture
def mock_file_system():
    """Fixture providing a mock file system."""
    return MockFileSystem()


@pytest.fixture
def sample_model_responses():
    """Fixture providing sample model responses for testing."""
    return {
        "gpt-3.5-turbo": {
            "response": "This is a response from GPT-3.5-turbo",
            "tokens": 25,
            "latency": 1.2
        },
        "claude-3-sonnet": {
            "response": "This is a response from Claude-3-Sonnet",
            "tokens": 30,
            "latency": 1.5
        },
        "llama-2-70b": {
            "response": "This is a response from Llama-2-70B",
            "tokens": 28,
            "latency": 2.1
        }
    }


@pytest.fixture
def sample_evaluation_metrics():
    """Fixture providing sample evaluation metrics."""
    return {
        "bleu_score": 0.85,
        "rouge_score": 0.88,
        "bert_score": 0.92,
        "semantic_similarity": 0.90,
        "human_evaluation": 4.2,
        "latency_ms": 1200,
        "cost_per_1k_tokens": 0.002
    }


@pytest.fixture
def sample_performance_metrics():
    """Fixture providing sample performance metrics."""
    return {
        "throughput_rps": 100,
        "latency_p50_ms": 500,
        "latency_p95_ms": 1200,
        "latency_p99_ms": 2000,
        "error_rate": 0.01,
        "availability": 0.999,
        "cpu_usage_percent": 75,
        "memory_usage_percent": 60,
        "gpu_usage_percent": 85
    }


@pytest.fixture
def sample_bias_metrics():
    """Fixture providing sample bias detection metrics."""
    return {
        "overall_bias_score": 0.12,
        "gender_bias": 0.08,
        "racial_bias": 0.15,
        "age_bias": 0.10,
        "socioeconomic_bias": 0.18,
        "demographic_parity": 0.85,
        "equalized_odds": 0.88,
        "calibration": 0.92
    }


@pytest.fixture
def sample_robustness_metrics():
    """Fixture providing sample robustness testing metrics."""
    return {
        "adversarial_success_rate": 0.95,
        "noise_tolerance_score": 0.87,
        "edge_case_handling": 0.90,
        "prompt_injection_resistance": 0.88,
        "jailbreak_resistance": 0.92,
        "safety_score": 0.94
    }


@pytest.fixture
def sample_cost_metrics():
    """Fixture providing sample cost analysis metrics."""
    return {
        "monthly_compute_cost": 8500,
        "monthly_storage_cost": 1200,
        "monthly_data_transfer_cost": 800,
        "monthly_monitoring_cost": 300,
        "total_monthly_cost": 10800,
        "cost_per_request": 0.15,
        "cost_per_1k_tokens": 0.002,
        "cost_efficiency_score": 0.85
    }


@pytest.fixture
def sample_deployment_metrics():
    """Fixture providing sample deployment metrics."""
    return {
        "deployment_id": "deploy_123",
        "status": "active",
        "instances": 5,
        "auto_scaling_enabled": True,
        "min_instances": 2,
        "max_instances": 10,
        "current_load_percent": 65,
        "deployment_time_minutes": 8.5,
        "rollback_count": 0
    }


@pytest.fixture
def sample_agent_metrics():
    """Fixture providing sample agent performance metrics."""
    return {
        "agent_id": "agent_001",
        "tasks_completed": 150,
        "success_rate": 0.96,
        "average_response_time_seconds": 2.3,
        "collaboration_score": 0.89,
        "error_count": 6,
        "uptime_percent": 99.5,
        "last_activity": datetime.now() - timedelta(minutes=5)
    }


@pytest.fixture
def sample_rag_metrics():
    """Fixture providing sample RAG system metrics."""
    return {
        "retrieval_accuracy": 0.95,
        "generation_quality": 0.92,
        "response_relevance": 0.88,
        "document_coverage": 0.85,
        "query_processing_time_ms": 250,
        "embedding_generation_time_ms": 50,
        "similarity_search_time_ms": 30,
        "cache_hit_rate": 0.65
    }


@pytest.fixture
def mock_async_context_manager():
    """Fixture providing a mock async context manager."""
    context_manager = AsyncMock()
    context_manager.__aenter__ = AsyncMock(return_value=context_manager)
    context_manager.__aexit__ = AsyncMock(return_value=None)
    return context_manager


@pytest.fixture
def mock_http_response():
    """Fixture providing a mock HTTP response."""
    response = Mock()
    response.status_code = 200
    response.json.return_value = {"status": "success", "data": "test_data"}
    response.text = '{"status": "success", "data": "test_data"}'
    response.headers = {"Content-Type": "application/json"}
    return response


@pytest.fixture
def mock_configuration():
    """Fixture providing a mock configuration object."""
    return {
        "model_evaluation": {
            "default_models": ["gpt-3.5-turbo", "claude-3-sonnet"],
            "evaluation_metrics": ["bleu", "rouge", "bert_score"],
            "test_suite_size": 1000
        },
        "ai_architecture": {
            "default_deployment_target": "hybrid",
            "auto_scaling": True,
            "monitoring_enabled": True
        },
        "gradio_app": {
            "host": "0.0.0.0",
            "port": 7860,
            "share": False,
            "debug": False
        },
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "test_db",
            "pool_size": 10
        }
    }


@pytest.fixture
def mock_environment_variables():
    """Fixture providing mock environment variables."""
    return {
        "OPENAI_API_KEY": "test_openai_key",
        "ANTHROPIC_API_KEY": "test_anthropic_key",
        "HUGGINGFACE_API_KEY": "test_hf_key",
        "DATABASE_URL": "postgresql://test:test@localhost:5432/test_db",
        "REDIS_URL": "redis://localhost:6379/0",
        "MLFLOW_TRACKING_URI": "http://localhost:5000"
    }


@pytest.fixture
def mock_logger():
    """Fixture providing a mock logger."""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    logger.critical = Mock()
    return logger


@pytest.fixture
def mock_metrics_collector():
    """Fixture providing a mock metrics collector."""
    collector = Mock()
    collector.collect_metrics = AsyncMock(return_value={
        "timestamp": datetime.now(),
        "metrics": {
            "cpu_usage": 0.75,
            "memory_usage": 0.60,
            "disk_usage": 0.45,
            "network_io": 1024
        }
    })
    collector.start_collection = AsyncMock()
    collector.stop_collection = AsyncMock()
    return collector


@pytest.fixture
def mock_alert_manager():
    """Fixture providing a mock alert manager."""
    manager = Mock()
    manager.send_alert = AsyncMock(return_value=True)
    manager.create_alert_rule = Mock(return_value="rule_123")
    manager.update_alert_rule = Mock(return_value=True)
    manager.delete_alert_rule = Mock(return_value=True)
    manager.get_active_alerts = Mock(return_value=[])
    return manager
