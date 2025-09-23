"""
Comprehensive pytest configuration for the complete test suite.

This configuration extends the existing conftest.py with additional
fixtures and settings for comprehensive testing across all platform
components and services.
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
import logging
import os
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import comprehensive fixtures
from tests.fixtures.comprehensive_test_fixtures import (
    # Basic fixtures
    event_loop, temp_dir, sample_model_config, sample_evaluation_data,
    sample_bias_test_data, sample_metrics_data, sample_performance_metrics,
    sample_documents, sample_model_versions, sample_workflow_data,
    
    # Mock fixtures
    mock_api_client, mock_platform_config,
    
    # Service integration fixtures
    service_integration_setup,
    
    # Phase 7 fixtures
    phase7_demonstration_setup, interactive_demonstration_config,
    
    # GitHub Pages fixtures
    github_pages_config, mkdocs_config, documentation_structure,
    
    # Platform architecture fixtures
    platform_architecture_setup,
    
    # Test data generation fixtures
    test_data_generator, async_test_runner,
    
    # Mock service fixtures
    mock_services, mock_platform_components,
    
    # Configuration fixtures
    test_configuration
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Additional Comprehensive Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def test_session_config():
    """Test session configuration."""
    return {
        "session_id": f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "start_time": datetime.now(),
        "test_types": ["unit", "integration", "e2e"],
        "platform_components": [
            "data_layer", "model_layer", "service_layer", 
            "api_layer", "presentation_layer"
        ],
        "services": [
            "fastapi", "gradio", "mlflow", "chromadb", 
            "neo4j", "ollama", "github_models"
        ]
    }


@pytest.fixture(scope="session")
def test_environment_setup():
    """Test environment setup."""
    return {
        "python_version": sys.version,
        "test_directory": Path(__file__).parent,
        "project_root": Path(__file__).parent.parent,
        "src_directory": Path(__file__).parent.parent / "src",
        "temp_base": Path(tempfile.gettempdir()) / "ai_assignments_tests",
        "mock_external_services": True,
        "use_test_data": True,
        "log_level": "INFO"
    }


@pytest.fixture
def comprehensive_test_data():
    """Comprehensive test data for all test types."""
    return {
        "unit_test_data": {
            "model_configs": [
                {"model_name": "gpt-3.5-turbo", "api_key": "test-key-1"},
                {"model_name": "claude-3-sonnet", "api_key": "test-key-2"},
                {"model_name": "llama-2-70b", "api_key": "test-key-3"}
            ],
            "evaluation_data": {
                "prompts": ["Test prompt 1", "Test prompt 2", "Test prompt 3"],
                "expected_outputs": ["Response 1", "Response 2", "Response 3"],
                "task_types": ["qa", "explanation", "code_generation"]
            },
            "metrics_data": pd.DataFrame({
                "model": ["gpt-3.5-turbo", "claude-3-sonnet"],
                "accuracy": [0.95, 0.93],
                "latency": [1200, 1500]
            })
        },
        "integration_test_data": {
            "service_endpoints": {
                "fastapi": ["/health", "/api/status", "/api/models"],
                "gradio": ["/", "/api/predict", "/api/evaluate"],
                "mlflow": ["/", "/api/2.0/mlflow/experiments/list"],
                "chromadb": ["/api/v1/heartbeat", "/api/v1/collections"]
            },
            "service_communication": {
                "gradio_to_fastapi": {
                    "endpoint": "/api/evaluate",
                    "method": "POST",
                    "payload": {"model": "test-model", "prompt": "test"}
                },
                "fastapi_to_mlflow": {
                    "endpoint": "/api/2.0/mlflow/runs/create",
                    "method": "POST",
                    "payload": {"experiment_id": "123", "run_name": "test-run"}
                }
            }
        },
        "e2e_test_data": {
            "phase7_workflows": {
                "data_generation": {
                    "enterprise_data": True,
                    "chromadb_population": True,
                    "neo4j_population": True,
                    "mlflow_initialization": True
                },
                "model_setup": {
                    "ollama_setup": True,
                    "github_models_auth": True,
                    "endpoint_testing": True
                },
                "ai_architect_customization": {
                    "fine_tuning": True,
                    "qlora_adapters": True,
                    "custom_embeddings": True,
                    "rag_workflows": True
                },
                "model_evaluation": {
                    "raw_models": True,
                    "custom_models": True,
                    "agentic_workflows": True,
                    "stress_testing": True
                },
                "factory_roster": {
                    "profile_creation": True,
                    "model_deployment": True,
                    "monitoring_setup": True
                }
            },
            "github_pages_workflows": {
                "local_development": {
                    "mkdocs_serve": True,
                    "documentation_build": True,
                    "service_integration": True
                },
                "hosted_deployment": {
                    "github_actions": True,
                    "site_generation": True,
                    "live_applications": True
                }
            }
        }
    }


@pytest.fixture
def test_coverage_config():
    """Test coverage configuration."""
    return {
        "target_coverage": 85.0,
        "exclude_patterns": [
            "*/tests/*",
            "*/venv/*",
            "*/__pycache__/*",
            "*/site-packages/*"
        ],
        "include_patterns": [
            "src/*",
            "*.py"
        ],
        "coverage_modules": [
            "src.model_evaluation",
            "src.ai_architecture",
            "src.enterprise_llmops",
            "src.gradio_app",
            "src.utils"
        ]
    }


@pytest.fixture
def performance_test_config():
    """Performance test configuration."""
    return {
        "load_testing": {
            "concurrent_users": [10, 50, 100, 500],
            "duration_minutes": [5, 10, 30],
            "ramp_up_time": 60  # seconds
        },
        "stress_testing": {
            "max_concurrent_users": 1000,
            "max_duration_minutes": 60,
            "failure_threshold": 0.05  # 5% error rate
        },
        "performance_benchmarks": {
            "api_response_time_p95": 2000,  # milliseconds
            "api_response_time_p99": 5000,  # milliseconds
            "throughput_min": 100,  # requests per second
            "availability_min": 0.999  # 99.9%
        }
    }


@pytest.fixture
def security_test_config():
    """Security test configuration."""
    return {
        "authentication_tests": {
            "test_invalid_tokens": True,
            "test_expired_tokens": True,
            "test_missing_authentication": True
        },
        "authorization_tests": {
            "test_unauthorized_access": True,
            "test_privilege_escalation": True,
            "test_resource_isolation": True
        },
        "input_validation_tests": {
            "test_sql_injection": True,
            "test_xss_attacks": True,
            "test_injection_attacks": True
        },
        "data_protection_tests": {
            "test_data_encryption": True,
            "test_pii_protection": True,
            "test_data_anonymization": True
        }
    }


# ============================================================================
# Test Environment Management
# ============================================================================

@pytest.fixture(scope="session")
def test_environment_manager():
    """Test environment manager for setup and teardown."""
    class TestEnvironmentManager:
        def __init__(self):
            self.temp_directories = []
            self.mock_services = {}
            self.test_data = {}
        
        def setup_test_environment(self):
            """Set up test environment."""
            # Create temporary directories
            temp_base = Path(tempfile.gettempdir()) / "ai_assignments_tests"
            temp_base.mkdir(exist_ok=True)
            self.temp_directories.append(temp_base)
            
            # Set up logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            logger.info("Test environment setup completed")
        
        def teardown_test_environment(self):
            """Tear down test environment."""
            # Clean up temporary directories
            for temp_dir in self.temp_directories:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
            
            logger.info("Test environment teardown completed")
        
        def create_temp_directory(self, name: str) -> Path:
            """Create a temporary directory."""
            temp_base = self.temp_directories[0]
            temp_dir = temp_base / name
            temp_dir.mkdir(exist_ok=True)
            return temp_dir
    
    manager = TestEnvironmentManager()
    manager.setup_test_environment()
    yield manager
    manager.teardown_test_environment()


# ============================================================================
# Mock Service Management
# ============================================================================

@pytest.fixture
def mock_service_manager():
    """Mock service manager for testing."""
    class MockServiceManager:
        def __init__(self):
            self.services = {}
            self.service_status = {}
        
        def register_service(self, name: str, mock_service: Mock):
            """Register a mock service."""
            self.services[name] = mock_service
            self.service_status[name] = "running"
        
        def get_service(self, name: str) -> Mock:
            """Get a mock service."""
            return self.services.get(name)
        
        def start_service(self, name: str) -> bool:
            """Start a mock service."""
            if name in self.services:
                self.service_status[name] = "running"
                return True
            return False
        
        def stop_service(self, name: str) -> bool:
            """Stop a mock service."""
            if name in self.services:
                self.service_status[name] = "stopped"
                return True
            return False
        
        def get_service_status(self, name: str) -> str:
            """Get service status."""
            return self.service_status.get(name, "unknown")
    
    return MockServiceManager()


# ============================================================================
# Test Data Management
# ============================================================================

@pytest.fixture
def test_data_manager():
    """Test data manager for managing test data."""
    class TestDataManager:
        def __init__(self):
            self.test_datasets = {}
            self.data_generators = {}
        
        def register_dataset(self, name: str, dataset: Dict[str, Any]):
            """Register a test dataset."""
            self.test_datasets[name] = dataset
        
        def get_dataset(self, name: str) -> Dict[str, Any]:
            """Get a test dataset."""
            return self.test_datasets.get(name, {})
        
        def register_generator(self, name: str, generator_func):
            """Register a data generator."""
            self.data_generators[name] = generator_func
        
        def generate_data(self, generator_name: str, **kwargs):
            """Generate data using a registered generator."""
            if generator_name in self.data_generators:
                return self.data_generators[generator_name](**kwargs)
            return None
        
        def create_sample_evaluation_data(self, count: int = 10):
            """Create sample evaluation data."""
            prompts = [f"Test prompt {i+1}" for i in range(count)]
            expected_outputs = [f"Expected response {i+1}" for i in range(count)]
            task_types = ["qa"] * count
            
            return {
                "prompts": prompts,
                "expected_outputs": expected_outputs,
                "task_types": task_types
            }
        
        def create_sample_metrics_data(self, models: List[str] = None):
            """Create sample metrics data."""
            if models is None:
                models = ["gpt-3.5-turbo", "claude-3-sonnet", "llama-2-70b"]
            
            return pd.DataFrame({
                "model": models,
                "accuracy": np.random.uniform(0.8, 0.95, len(models)),
                "latency": np.random.uniform(1000, 3000, len(models)),
                "cost": np.random.uniform(0.001, 0.005, len(models))
            })
    
    return TestDataManager()


# ============================================================================
# Test Execution Tracking
# ============================================================================

@pytest.fixture
def test_execution_tracker():
    """Test execution tracker for monitoring test runs."""
    class TestExecutionTracker:
        def __init__(self):
            self.test_results = {}
            self.execution_times = {}
            self.failed_tests = []
            self.passed_tests = []
        
        def start_test(self, test_name: str):
            """Start tracking a test."""
            self.test_results[test_name] = {
                "status": "running",
                "start_time": datetime.now(),
                "end_time": None,
                "duration": None,
                "error": None
            }
        
        def end_test(self, test_name: str, status: str, error: str = None):
            """End tracking a test."""
            if test_name in self.test_results:
                self.test_results[test_name]["status"] = status
                self.test_results[test_name]["end_time"] = datetime.now()
                self.test_results[test_name]["duration"] = (
                    self.test_results[test_name]["end_time"] - 
                    self.test_results[test_name]["start_time"]
                ).total_seconds()
                self.test_results[test_name]["error"] = error
                
                if status == "passed":
                    self.passed_tests.append(test_name)
                elif status == "failed":
                    self.failed_tests.append(test_name)
        
        def get_test_summary(self) -> Dict[str, Any]:
            """Get test execution summary."""
            total_tests = len(self.test_results)
            passed_count = len(self.passed_tests)
            failed_count = len(self.failed_tests)
            
            return {
                "total_tests": total_tests,
                "passed": passed_count,
                "failed": failed_count,
                "success_rate": passed_count / total_tests if total_tests > 0 else 0,
                "failed_tests": self.failed_tests,
                "execution_times": self.execution_times
            }
    
    return TestExecutionTracker()


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Add custom markers
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Integration tests for module interactions")
    config.addinivalue_line("markers", "e2e: End-to-end tests for complete workflows")
    config.addinivalue_line("markers", "slow: Tests that take longer than 5 seconds to run")
    config.addinivalue_line("markers", "api: Tests that require API access")
    config.addinivalue_line("markers", "performance: Performance and load tests")
    config.addinivalue_line("markers", "security: Security-related tests")
    config.addinivalue_line("markers", "regression: Regression tests")
    config.addinivalue_line("markers", "smoke: Smoke tests for basic functionality")
    config.addinivalue_line("markers", "stress: Stress tests for system limits")
    config.addinivalue_line("markers", "github_pages: GitHub Pages specific tests")
    config.addinivalue_line("markers", "phase7: Phase 7 demonstration tests")
    config.addinivalue_line("markers", "platform_architecture: Platform architecture tests")
    config.addinivalue_line("markers", "service_integration: Service integration tests")
    config.addinivalue_line("markers", "frontend: Frontend integration tests")
    config.addinivalue_line("markers", "backend: Backend integration tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location and name."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        
        # Add markers based on test name
        if "github_pages" in item.name:
            item.add_marker(pytest.mark.github_pages)
        if "phase7" in item.name:
            item.add_marker(pytest.mark.phase7)
        if "platform_architecture" in item.name:
            item.add_marker(pytest.mark.platform_architecture)
        if "service_integration" in item.name:
            item.add_marker(pytest.mark.service_integration)
        if "frontend" in item.name:
            item.add_marker(pytest.mark.frontend)
        if "backend" in item.name:
            item.add_marker(pytest.mark.backend)
        
        # Add slow marker for tests that take > 5 seconds
        if any(keyword in item.name.lower() for keyword in ["slow", "performance", "stress", "load"]):
            item.add_marker(pytest.mark.slow)
        
        # Add API marker for tests that require API access
        if any(keyword in item.name.lower() for keyword in ["api", "endpoint", "http"]):
            item.add_marker(pytest.mark.api)


def pytest_runtest_setup(item):
    """Set up test execution."""
    logger.info(f"Starting test: {item.name}")


def pytest_runtest_teardown(item):
    """Tear down test execution."""
    logger.info(f"Completed test: {item.name}")


def pytest_runtest_logreport(report):
    """Log test report."""
    if report.when == "call":
        if report.outcome == "passed":
            logger.info(f"✓ PASSED: {report.nodeid}")
        elif report.outcome == "failed":
            logger.error(f"✗ FAILED: {report.nodeid} - {report.longrepr}")
        elif report.outcome == "skipped":
            logger.warning(f"- SKIPPED: {report.nodeid}")


# ============================================================================
# Test Session Hooks
# ============================================================================

def pytest_sessionstart(session):
    """Called after the Session object has been created."""
    logger.info("=" * 80)
    logger.info("Starting AI Assignments Test Suite")
    logger.info("=" * 80)
    logger.info(f"Test session started at: {datetime.now()}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Pytest version: {pytest.__version__}")


def pytest_sessionfinish(session):
    """Called after whole test run finished, right before returning the exit status."""
    logger.info("=" * 80)
    logger.info("AI Assignments Test Suite Completed")
    logger.info("=" * 80)
    logger.info(f"Test session finished at: {datetime.now()}")
    
    # Print test summary
    if hasattr(session, 'testscollected'):
        logger.info(f"Tests collected: {session.testscollected}")
    if hasattr(session, 'testfailed'):
        logger.info(f"Tests failed: {session.testfailed}")
    if hasattr(session, 'testsfailed'):
        logger.info(f"Tests failed: {session.testsfailed}")


# ============================================================================
# Custom Assertions
# ============================================================================

def pytest_assertrepr_compare(config, op, left, right):
    """Custom assertion representation for better error messages."""
    if op == "==":
        if isinstance(left, dict) and isinstance(right, dict):
            return [
                f"Dict comparison failed:",
                f"Left:  {left}",
                f"Right: {right}",
                f"Differences: {set(left.items()) - set(right.items())}"
            ]
        elif isinstance(left, list) and isinstance(right, list):
            return [
                f"List comparison failed:",
                f"Left:  {left}",
                f"Right: {right}",
                f"Length difference: {len(left)} vs {len(right)}"
            ]


# ============================================================================
# Test Utilities
# ============================================================================

@pytest.fixture
def test_utilities():
    """Test utilities for common test operations."""
    class TestUtilities:
        @staticmethod
        def create_temp_file(content: str, suffix: str = ".txt") -> Path:
            """Create a temporary file with content."""
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False)
            temp_file.write(content)
            temp_file.close()
            return Path(temp_file.name)
        
        @staticmethod
        def create_temp_directory() -> Path:
            """Create a temporary directory."""
            return Path(tempfile.mkdtemp())
        
        @staticmethod
        def mock_async_function(return_value=None, side_effect=None):
            """Create a mock async function."""
            async def mock_func(*args, **kwargs):
                if side_effect:
                    if isinstance(side_effect, Exception):
                        raise side_effect
                    return side_effect(*args, **kwargs)
                return return_value
            return mock_func
        
        @staticmethod
        def create_sample_json_data(data: Dict[str, Any]) -> str:
            """Create sample JSON data."""
            return json.dumps(data, indent=2)
        
        @staticmethod
        def create_sample_yaml_data(data: Dict[str, Any]) -> str:
            """Create sample YAML data."""
            import yaml
            return yaml.dump(data, default_flow_style=False)
    
    return TestUtilities()


if __name__ == "__main__":
    # This file is meant to be imported, not run directly
    print("This is a comprehensive pytest configuration file.")
    print("Import it in your test files or use it with pytest.")
