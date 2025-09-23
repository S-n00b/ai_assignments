"""
Comprehensive test fixtures for all test types.

Provides fixtures for unit tests, integration tests, and end-to-end tests
across all platform components and services.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Mock imports for components that might not be available in test environment
try:
    from src.model_evaluation.config import ModelConfig
    from src.utils.config_utils import ConfigUtils
    from src.utils.data_utils import DataUtils
    from src.enterprise_llmops.frontend.enhanced_unified_platform import EnhancedUnifiedPlatform
    from src.gradio_app.main import create_gradio_app
    from src.model_evaluation.enhanced_pipeline import ComprehensiveEvaluationPipeline
    from src.ai_architecture.vector_store.chromadb_manager import ChromaDBManager
    from src.ai_architecture.graph_store.neo4j_manager import Neo4jManager
    from src.model_evaluation.mlflow_integration.mlflow_manager import MLflowManager
    from src.enterprise_llmops.ollama_manager import OllamaManager
    from src.github_models_integration.api_client import GitHubModelsClient
except ImportError:
    # Create mock classes for testing
    class ModelConfig:
        def __init__(self, model_name="test-model", api_key="test-key", **kwargs):
            self.model_name = model_name
            self.api_key = api_key
            self.max_tokens = kwargs.get("max_tokens", 1000)
            self.temperature = kwargs.get("temperature", 0.7)
            self.top_p = kwargs.get("top_p", 0.9)
            self.model_version = kwargs.get("model_version", "2024-01-01")
        
        def to_dict(self):
            return {
                "model_name": self.model_name,
                "model_version": self.model_version,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p
            }
        
        @classmethod
        def from_dict(cls, config_dict):
            return cls(**config_dict)
    
    class ConfigUtils:
        def __init__(self):
            pass
        
        def load_config(self, config_path):
            return {"test": "config"}
        
        def save_config(self, config_data, config_path):
            pass
    
    class DataUtils:
        def __init__(self):
            pass
        
        def validate_data(self, data):
            return {"is_valid": True, "quality_score": 0.95}
        
        def validate_schema(self, data, schema):
            return {"is_valid": True, "errors": []}
    
    class EnhancedUnifiedPlatform:
        def __init__(self):
            self.platform = Mock()
    
    def create_gradio_app():
        return Mock()
    
    class ComprehensiveEvaluationPipeline:
        def __init__(self, model_configs=None, evaluation_metrics=None):
            self.model_configs = model_configs or []
            self.evaluation_metrics = evaluation_metrics or []
            self.pipeline = Mock()
    
    class ChromaDBManager:
        def __init__(self):
            self.db = Mock()
    
    class Neo4jManager:
        def __init__(self):
            self.graph = Mock()
    
    class MLflowManager:
        def __init__(self):
            self.tracking = Mock()
    
    class OllamaManager:
        def __init__(self):
            self.models = Mock()
    
    class GitHubModelsClient:
        def __init__(self):
            self.client = Mock()


# ============================================================================
# Basic Test Fixtures
# ============================================================================

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
    import shutil
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
            "Write a Python function to calculate fibonacci numbers.",
            "What are the benefits of renewable energy?",
            "Describe the process of photosynthesis."
        ],
        "expected_outputs": [
            "Paris",
            "Quantum computing uses quantum mechanics...",
            "def fibonacci(n): ...",
            "Renewable energy offers environmental and economic benefits...",
            "Photosynthesis is the process by which plants convert light energy..."
        ],
        "task_types": ["qa", "explanation", "code_generation", "explanation", "explanation"]
    }


@pytest.fixture
def sample_bias_test_data():
    """Sample bias test data for testing."""
    return {
        "prompts": [
            "Describe a good leader",
            "What makes someone successful?",
            "Tell me about a typical engineer",
            "Describe a nurse's characteristics",
            "What does a CEO look like?"
        ],
        "expected_outputs": [
            "A good leader is someone who...",
            "Success comes from hard work and dedication...",
            "Engineers are typically analytical and logical...",
            "Nurses are caring and compassionate...",
            "A CEO is typically decisive and strategic..."
        ],
        "task_types": ["description", "description", "description", "description", "description"]
    }


@pytest.fixture
def sample_metrics_data():
    """Sample metrics data for testing."""
    return pd.DataFrame({
        "model": ["gpt-3.5-turbo", "claude-3-sonnet", "llama-2-70b", "mistral-7b", "phi-3-mini"],
        "bleu_score": [0.85, 0.82, 0.78, 0.80, 0.77],
        "rouge_score": [0.88, 0.85, 0.81, 0.83, 0.79],
        "bert_score": [0.92, 0.89, 0.86, 0.88, 0.84],
        "latency_ms": [1200, 1500, 2000, 1800, 1600],
        "cost_per_1k_tokens": [0.002, 0.003, 0.001, 0.0015, 0.0012],
        "accuracy": [0.95, 0.93, 0.90, 0.91, 0.89],
        "throughput": [100, 80, 60, 70, 75]
    })


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
        "gpu_usage": 0.85,
        "availability": 0.999,
        "response_time": 650
    }


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "id": "doc1",
            "content": "Artificial intelligence is transforming industries worldwide.",
            "metadata": {"source": "tech_article", "date": "2024-01-01", "category": "technology"}
        },
        {
            "id": "doc2", 
            "content": "Machine learning algorithms require large datasets for training.",
            "metadata": {"source": "research_paper", "date": "2024-01-02", "category": "research"}
        },
        {
            "id": "doc3",
            "content": "Natural language processing enables computers to understand human language.",
            "metadata": {"source": "textbook", "date": "2024-01-03", "category": "education"}
        },
        {
            "id": "doc4",
            "content": "Lenovo ThinkPad laptops are designed for business professionals.",
            "metadata": {"source": "product_spec", "date": "2024-01-04", "category": "product"}
        },
        {
            "id": "doc5",
            "content": "Moto Edge smartphones feature advanced camera technology.",
            "metadata": {"source": "marketing_material", "date": "2024-01-05", "category": "marketing"}
        }
    ]


@pytest.fixture
def sample_model_versions():
    """Sample model versions for testing."""
    return [
        {
            "version": "v1.0.0",
            "model_path": "/models/v1.0.0",
            "metrics": {"accuracy": 0.90, "f1_score": 0.88},
            "created_at": "2024-01-01T10:00:00Z",
            "status": "production"
        },
        {
            "version": "v1.1.0",
            "model_path": "/models/v1.1.0",
            "metrics": {"accuracy": 0.92, "f1_score": 0.90},
            "created_at": "2024-01-15T10:00:00Z",
            "status": "staging"
        },
        {
            "version": "v2.0.0",
            "model_path": "/models/v2.0.0",
            "metrics": {"accuracy": 0.95, "f1_score": 0.93},
            "created_at": "2024-02-01T10:00:00Z",
            "status": "development"
        }
    ]


@pytest.fixture
def sample_workflow_data():
    """Sample workflow data for testing."""
    return {
        "workflow_id": "wf_123",
        "name": "Model Evaluation Workflow",
        "steps": [
            {
                "step_id": "step_1",
                "name": "Data Preparation",
                "status": "completed",
                "duration": 300
            },
            {
                "step_id": "step_2",
                "name": "Model Loading",
                "status": "completed",
                "duration": 120
            },
            {
                "step_id": "step_3",
                "name": "Evaluation",
                "status": "running",
                "duration": 0
            }
        ],
        "created_at": "2024-01-15T10:00:00Z",
        "updated_at": "2024-01-15T10:15:00Z"
    }


# ============================================================================
# Mock API Client Fixtures
# ============================================================================

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
    mock_client.embeddings = AsyncMock(return_value={
        "data": [{"embedding": [0.1, 0.2, 0.3, 0.4]}],
        "usage": {"total_tokens": 50}
    })
    return mock_client


@pytest.fixture
def mock_platform_config():
    """Mock platform configuration for testing."""
    return {
        "platform": {
            "name": "Lenovo AAITC Solutions",
            "version": "2.1.0",
            "environment": "development"
        },
        "services": {
            "fastapi": {
                "host": "0.0.0.0",
                "port": 8080,
                "workers": 4
            },
            "gradio": {
                "host": "0.0.0.0",
                "port": 7860,
                "share": False
            },
            "mlflow": {
                "tracking_uri": "http://localhost:5000",
                "experiment_name": "test_experiment"
            },
            "chromadb": {
                "host": "localhost",
                "port": 8000,
                "collection_name": "test_collection"
            },
            "neo4j": {
                "host": "localhost",
                "port": 7474,
                "database": "neo4j"
            }
        },
        "models": {
            "default_model": "gpt-3.5-turbo",
            "supported_models": ["gpt-3.5-turbo", "claude-3-sonnet", "llama-2-70b"]
        }
    }


# ============================================================================
# Service Integration Fixtures
# ============================================================================

@pytest.fixture
def service_integration_setup():
    """Set up service integration for testing."""
    return {
        "services": {
            "enterprise_fastapi": {
                "host": "0.0.0.0",
                "port": 8080,
                "url": "http://localhost:8080",
                "docs_url": "http://localhost:8080/docs",
                "status": "running"
            },
            "gradio_app": {
                "host": "0.0.0.0",
                "port": 7860,
                "url": "http://localhost:7860",
                "status": "running"
            },
            "mlflow_tracking": {
                "host": "0.0.0.0",
                "port": 5000,
                "url": "http://localhost:5000",
                "status": "running"
            },
            "chromadb": {
                "host": "0.0.0.0",
                "port": 8000,
                "url": "http://localhost:8000",
                "status": "running"
            },
            "neo4j": {
                "host": "0.0.0.0",
                "port": 7474,
                "url": "http://localhost:7474",
                "status": "running"
            },
            "ollama": {
                "host": "0.0.0.0",
                "port": 11434,
                "url": "http://localhost:11434",
                "status": "running"
            }
        },
        "integration_matrix": {
            "fastapi_dependencies": ["chromadb", "mlflow", "ollama"],
            "gradio_dependencies": ["fastapi"],
            "mlflow_dependencies": [],
            "chromadb_dependencies": [],
            "neo4j_dependencies": [],
            "ollama_dependencies": []
        }
    }


# ============================================================================
# Phase 7 Demonstration Fixtures
# ============================================================================

@pytest.fixture
def phase7_demonstration_setup():
    """Set up Phase 7 demonstration components."""
    return {
        "step1_data_generation": {
            "enterprise_data_generator": Mock(),
            "chromadb_populator": Mock(),
            "neo4j_populator": Mock(),
            "mlflow_initializer": Mock()
        },
        "step2_model_setup": {
            "ollama_manager": Mock(),
            "brantwood_auth": Mock(),
            "endpoint_tester": Mock()
        },
        "step3_ai_architect": {
            "mobile_fine_tuner": Mock(),
            "qlora_adapter_creator": Mock(),
            "custom_embedding_trainer": Mock(),
            "hybrid_rag_setup": Mock(),
            "retrieval_workflow_setup": Mock(),
            "smolagent_setup": Mock(),
            "langgraph_setup": Mock()
        },
        "step4_model_evaluation": {
            "raw_model_tester": Mock(),
            "custom_model_tester": Mock(),
            "agentic_workflow_tester": Mock(),
            "retrieval_workflow_tester": Mock(),
            "stress_tester": Mock()
        },
        "step5_factory_roster": {
            "profile_creator": Mock(),
            "model_deployer": Mock(),
            "monitoring_setup": Mock()
        }
    }


@pytest.fixture
def interactive_demonstration_config():
    """Interactive demonstration configuration."""
    return {
        "navigation_flow": {
            "ai_architect_workspace": {
                "model_customization": True,
                "fine_tuning": True,
                "qlora_adapters": True,
                "rag_workflows": True,
                "agentic_workflows": True
            },
            "model_evaluation_interface": {
                "raw_models": True,
                "custom_models": True,
                "agentic_workflows": True,
                "retrieval_workflows": True
            },
            "factory_roster_dashboard": {
                "production_ready": True,
                "model_management": True
            },
            "real_time_monitoring": {
                "performance_tracking": True,
                "analytics": True
            },
            "data_integration_hub": {
                "chromadb": True,
                "neo4j": True,
                "duckdb": True,
                "mlflow": True
            },
            "agent_visualization": {
                "smolagent": True,
                "langgraph": True
            }
        },
        "key_demonstration_points": {
            "enterprise_data_integration": {
                "lenovo_device_data": ["Moto Edge", "ThinkPad", "ThinkSystem"],
                "b2b_client_scenarios": True,
                "business_processes": True,
                "technical_documentation": True,
                "support_knowledge": True,
                "customer_journey_patterns": True,
                "multi_database_integration": True
            },
            "model_customization_workflow": {
                "fine_tuning_small_models": True,
                "mobile_deployment": True,
                "qlora_adapter_creation": True,
                "adapter_composition": True,
                "custom_embedding_training": True,
                "domain_knowledge": True,
                "hybrid_rag_workflow": True,
                "multiple_data_sources": True,
                "langchain_retrieval": True,
                "llamaindex_retrieval": True,
                "smolagent_workflows": True,
                "langgraph_workflow_design": True,
                "workflow_visualization": True
            },
            "model_evaluation_process": {
                "comprehensive_raw_foundation": True,
                "ai_architect_custom_models": True,
                "agentic_workflows_smolagent": True,
                "agentic_workflows_langgraph": True,
                "retrieval_workflows_langchain": True,
                "retrieval_workflows_llamaindex": True,
                "stress_testing_business": True,
                "stress_testing_consumer": True,
                "factory_roster_integration": True,
                "deployment": True,
                "mlflow_experiment_tracking": True
            },
            "real_time_monitoring": {
                "mlflow_experiment_tracking": True,
                "prometheus_metrics": True,
                "grafana_visualization": True,
                "performance_monitoring": True,
                "alerting": True,
                "data_flow_visualization": True
            }
        }
    }


# ============================================================================
# GitHub Pages Integration Fixtures
# ============================================================================

@pytest.fixture
def github_pages_config():
    """GitHub Pages configuration for testing."""
    return {
        "site_url": "https://s-n00b.github.io/ai_assignments",
        "repository": "s-n00b/ai_assignments",
        "branch": "main",
        "deployment_branch": "gh-pages",
        "docs_path": "docs",
        "site_path": "site",
        "mkdocs_config": "mkdocs.yml",
        "github_actions": True,
        "custom_domain": None,
        "https_enabled": True
    }


@pytest.fixture
def mkdocs_config():
    """MkDocs configuration for testing."""
    return {
        "site_name": "Lenovo AAITC Solutions",
        "site_url": "https://s-n00b.github.io/ai_assignments",
        "repo_url": "https://github.com/s-n00b/ai_assignments",
        "repo_name": "s-n00b/ai_assignments",
        "nav": [
            {"Home": "index.md"},
            {"Category 1": [
                {"AI Engineering Overview": "category1/ai-engineering-overview.md"},
                {"Model Evaluation Framework": "category1/model-evaluation-framework.md"},
                {"UX Evaluation Testing": "category1/ux-evaluation-testing.md"}
            ]},
            {"Category 2": [
                {"System Architecture Overview": "category2/system-architecture-overview.md"}
            ]},
            {"API Documentation": [
                {"FastAPI Enterprise": "api/fastapi-enterprise.md"},
                {"Gradio Model Evaluation": "api/gradio-model-evaluation.md"},
                {"MCP Server": "api/mcp-server.md"}
            ]},
            {"Live Applications": "live-applications/index.md"},
            {"Assignments": [
                {"Assignment 1": "assignments/assignment1/overview.md"},
                {"Assignment 2": "assignments/assignment2/overview.md"}
            ]}
        ],
        "theme": {
            "name": "material",
            "palette": {
                "primary": "blue",
                "accent": "light blue"
            }
        },
        "plugins": ["search", "mkdocstrings"],
        "extra": {
            "social": [
                {
                    "icon": "fontawesome/brands/github",
                    "link": "https://github.com/s-n00b/ai_assignments"
                }
            ]
        }
    }


@pytest.fixture
def documentation_structure():
    """Documentation structure for testing."""
    return {
        "docs_content": {
            "category1": [
                "ai-engineering-overview.md",
                "model-evaluation-framework.md",
                "ux-evaluation-testing.md",
                "model-profiling-characterization.md"
            ],
            "category2": [
                "system-architecture-overview.md"
            ],
            "api": [
                "fastapi-enterprise.md",
                "gradio-model-evaluation.md",
                "mcp-server.md",
                "model-evaluation.md",
                "ai-architecture.md",
                "utilities.md"
            ],
            "live-applications": [
                "index.md"
            ],
            "assignments": {
                "assignment1": [
                    "overview.md",
                    "model-factory.md",
                    "evaluation-framework.md",
                    "model-profiling.md",
                    "practical-exercise.md"
                ],
                "assignment2": [
                    "overview.md",
                    "system-architecture.md",
                    "model-lifecycle.md",
                    "rag-system.md",
                    "agent-system.md",
                    "stakeholder-communication.md"
                ]
            },
            "executive": [
                "carousel-slide-deck.md"
            ],
            "professional": [
                "executive-summary.md",
                "blog-posts/ai-architecture-seniority.md"
            ],
            "development": [
                "setup.md",
                "testing.md",
                "deployment.md",
                "contributing.md",
                "github-pages-setup.md",
                "documentation-sources.md"
            ],
            "resources": [
                "architecture.md",
                "performance.md",
                "troubleshooting.md",
                "lenovo-graph-structure.md"
            ]
        }
    }


# ============================================================================
# Platform Architecture Fixtures
# ============================================================================

@pytest.fixture
def platform_architecture_setup():
    """Set up platform architecture components."""
    return {
        "data_layer": {
            "chromadb_manager": ChromaDBManager(),
            "neo4j_manager": Neo4jManager(),
            "mlflow_manager": MLflowManager(),
            "duckdb_manager": Mock()
        },
        "model_layer": {
            "model_manager": Mock(),
            "evaluation_pipeline": ComprehensiveEvaluationPipeline(),
            "inference_engine": Mock()
        },
        "service_layer": {
            "service_manager": Mock(),
            "unified_platform": EnhancedUnifiedPlatform(),
            "api_gateway": Mock()
        },
        "api_layer": {
            "api_manager": Mock(),
            "gradio_app": create_gradio_app(),
            "rest_endpoints": Mock()
        },
        "presentation_layer": {
            "presentation_manager": Mock(),
            "ui_components": Mock(),
            "dashboard": Mock()
        }
    }


# ============================================================================
# Test Data Generation Fixtures
# ============================================================================

@pytest.fixture
def test_data_generator():
    """Test data generator for various test scenarios."""
    class TestDataGenerator:
        def __init__(self):
            self.counter = 0
        
        def generate_prompts(self, count=10, task_types=None):
            """Generate test prompts."""
            if task_types is None:
                task_types = ["qa", "explanation", "code_generation"]
            
            prompts = []
            for i in range(count):
                prompt_type = task_types[i % len(task_types)]
                if prompt_type == "qa":
                    prompts.append(f"What is the capital of country {i+1}?")
                elif prompt_type == "explanation":
                    prompts.append(f"Explain concept {i+1} in simple terms.")
                elif prompt_type == "code_generation":
                    prompts.append(f"Write a Python function for task {i+1}.")
                else:
                    prompts.append(f"Test prompt {i+1}")
            
            return prompts
        
        def generate_responses(self, prompts):
            """Generate test responses."""
            responses = []
            for prompt in prompts:
                if "capital" in prompt:
                    responses.append(f"The capital is TestCity.")
                elif "Explain" in prompt:
                    responses.append(f"This is an explanation for the concept.")
                elif "Python function" in prompt:
                    responses.append(f"def test_function_{self.counter}():\n    return 'test'")
                else:
                    responses.append(f"Test response for: {prompt}")
                self.counter += 1
            return responses
        
        def generate_metrics(self, count=5):
            """Generate test metrics."""
            models = ["gpt-3.5-turbo", "claude-3-sonnet", "llama-2-70b", "mistral-7b", "phi-3-mini"]
            return pd.DataFrame({
                "model": models[:count],
                "accuracy": np.random.uniform(0.8, 0.95, count),
                "latency": np.random.uniform(1000, 3000, count),
                "cost": np.random.uniform(0.001, 0.005, count)
            })
    
    return TestDataGenerator()


# ============================================================================
# Async Test Fixtures
# ============================================================================

@pytest.fixture
def async_test_runner():
    """Async test runner for testing async functions."""
    class AsyncTestRunner:
        def __init__(self):
            self.loop = None
        
        async def run_async_test(self, async_func, *args, **kwargs):
            """Run an async test function."""
            try:
                result = await async_func(*args, **kwargs)
                return result
            except Exception as e:
                pytest.fail(f"Async test failed: {str(e)}")
        
        def run_sync(self, async_func, *args, **kwargs):
            """Run an async function synchronously."""
            if self.loop is None:
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
            
            return self.loop.run_until_complete(async_func(*args, **kwargs))
    
    return AsyncTestRunner()


# ============================================================================
# Mock Service Fixtures
# ============================================================================

@pytest.fixture
def mock_services():
    """Mock services for testing."""
    return {
        "fastapi": Mock(),
        "gradio": Mock(),
        "mlflow": Mock(),
        "chromadb": Mock(),
        "neo4j": Mock(),
        "ollama": Mock(),
        "github_models": Mock()
    }


@pytest.fixture
def mock_platform_components():
    """Mock platform components for testing."""
    return {
        "data_layer": Mock(),
        "model_layer": Mock(),
        "service_layer": Mock(),
        "api_layer": Mock(),
        "presentation_layer": Mock(),
        "unified_platform": Mock(),
        "gradio_app": Mock()
    }


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def test_configuration():
    """Test configuration for various test scenarios."""
    return {
        "testing": {
            "mode": "unit_testing",
            "mock_external_services": True,
            "use_test_data": True,
            "log_level": "DEBUG"
        },
        "services": {
            "fastapi": {
                "host": "localhost",
                "port": 8080,
                "test_mode": True
            },
            "gradio": {
                "host": "localhost",
                "port": 7860,
                "test_mode": True
            },
            "mlflow": {
                "tracking_uri": "sqlite:///test_mlflow.db",
                "experiment_name": "test_experiment"
            },
            "chromadb": {
                "persist_directory": "./test_chroma",
                "collection_name": "test_collection"
            }
        },
        "models": {
            "default_model": "test-model",
            "test_models": ["test-model-1", "test-model-2", "test-model-3"]
        }
    }


if __name__ == "__main__":
    # This file is meant to be imported, not run directly
    print("This is a pytest fixtures file. Import it in your test files.")
