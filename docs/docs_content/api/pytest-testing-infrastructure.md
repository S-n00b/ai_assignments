# Pytest Testing Infrastructure Documentation

## 🎯 Overview

Comprehensive testing infrastructure for GitHub Pages frontend, platform architecture layers, and Phase 7 demonstration flow. This infrastructure provides unit, integration, and end-to-end testing capabilities following pytest best practices.

## 🚀 Key Features

### Core Testing Capabilities

- **Unit Tests**: Individual component testing with isolation
- **Integration Tests**: Service-level and platform architecture testing
- **End-to-End Tests**: Complete workflow and Phase 7 demonstration testing
- **GitHub Pages Testing**: Frontend integration testing (local and hosted)
- **Mock & Fixture Support**: Comprehensive test data and mocking infrastructure

### Integration Features

- **FastAPI Integration**: Backend service testing and validation
- **Gradio Integration**: Frontend interface testing and user interaction simulation
- **MLflow Integration**: Experiment tracking and model registry testing
- **ChromaDB Integration**: Vector database testing and validation
- **Phase 7 Flow Testing**: Complete demonstration workflow validation

## 📊 Test Structure

### Test Organization

```
tests/
├── unit/                           # Unit tests for individual components
│   ├── test_basic_functionality.py # Core ModelConfig and utility tests
│   ├── test_github_pages_integration.py # GitHub Pages frontend tests
│   └── test_phase7_demonstration_flow.py # Phase 7 component tests
├── integration/                    # Integration tests
│   ├── test_platform_architecture_layers.py # Platform architecture tests
│   ├── test_service_level_interactions.py # Service interaction tests
│   └── test_model_evaluation_integration.py # Model evaluation integration
├── e2e/                          # End-to-end tests
│   ├── test_phase7_complete_demonstration.py # Complete Phase 7 flow
│   ├── test_github_pages_frontend_integration.py # Frontend E2E tests
│   └── test_complete_workflows.py # Complete system workflows
├── fixtures/                      # Test fixtures and utilities
│   └── comprehensive_test_fixtures.py # Advanced test fixtures
├── conftest_minimal.py           # Minimal pytest configuration
├── conftest_comprehensive.py     # Comprehensive pytest configuration
└── test_runner.py               # Test execution runner
```

### Test Categories

| Category              | Description                           | Examples                                      |
| --------------------- | ------------------------------------- | --------------------------------------------- |
| **Unit Tests**        | Individual component testing          | ModelConfig, utilities, basic functionality   |
| **Integration Tests** | Service and layer interaction testing | Platform architecture, service communications |
| **E2E Tests**         | Complete workflow testing             | Phase 7 demonstration, full system workflows  |
| **Frontend Tests**    | GitHub Pages and UI testing           | User interactions, interface validation       |

## 🌐 Service Integration

### Testing Service Dependencies

| Service             | Port   | Test Coverage          | Description                        |
| ------------------- | ------ | ---------------------- | ---------------------------------- |
| **FastAPI Backend** | 8080   | Unit, Integration, E2E | Enterprise platform testing        |
| **Gradio Frontend** | 7860   | Unit, Integration, E2E | Model evaluation interface testing |
| **MLflow Tracking** | 5000   | Integration, E2E       | Experiment tracking validation     |
| **ChromaDB**        | 8000   | Integration, E2E       | Vector database testing            |
| **GitHub Pages**    | Hosted | E2E                    | Production frontend testing        |

### Test Data Flow

1. **Test Setup** → Fixtures and mock data initialization
2. **Component Testing** → Individual unit test execution
3. **Service Integration** → Cross-service communication testing
4. **End-to-End Validation** → Complete workflow testing
5. **Results Aggregation** → Test report generation and analysis

## 🔧 Configuration

### Pytest Configuration

Reference: [config/pytest.ini](mdc:config/pytest.ini)

#### Test Discovery Settings

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

#### Custom Markers

```ini
markers =
    unit: Unit tests for individual components
    integration: Integration tests for service interactions
    e2e: End-to-end tests for complete workflows
    slow: Tests that take longer to execute
    api: Tests for API endpoints
```

### Virtual Environment Setup

```bash
# Activate virtual environment
& C:\Users\samne\PycharmProjects\ai_assignments\venv\Scripts\Activate.ps1

# Install test dependencies
pip install -r config/requirements-testing.txt
```

## 📚 Test Execution

### Quick Start Testing

```bash
# Run all unit tests
python -m pytest tests/unit/ -v

# Run specific test category
python -m pytest tests/unit/test_basic_functionality.py::TestModelConfig -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run integration tests
python -m pytest tests/integration/ -v -m integration

# Run end-to-end tests
python -m pytest tests/e2e/ -v -m e2e
```

### Phase 7 Demonstration Testing

```bash
# Test complete Phase 7 demonstration flow
python -m pytest tests/e2e/test_phase7_complete_demonstration.py -v

# Test individual Phase 7 components
python -m pytest tests/unit/test_phase7_demonstration_flow.py -v

# Test with live services
python -m pytest tests/e2e/ -v --live-services
```

### GitHub Pages Testing

```bash
# Test GitHub Pages integration (local)
python -m pytest tests/unit/test_github_pages_integration.py -v

# Test GitHub Pages integration (hosted)
python -m pytest tests/e2e/test_github_pages_frontend_integration.py -v

# Test with production URLs
python -m pytest tests/e2e/ -v --production-urls
```

## 🛠️ Development

### Test Fixtures

Reference: [tests/fixtures/comprehensive_test_fixtures.py](mdc:tests/fixtures/comprehensive_test_fixtures.py)

#### Core Fixtures

```python
@pytest.fixture
def sample_model_config():
    """Sample model configuration for testing."""
    return ModelConfig(
        name="test-model",
        provider="test-provider",
        model_id="test-model-id"
    )

@pytest.fixture
def mock_api_client():
    """Mock API client for testing."""
    with patch('requests.Session') as mock_session:
        yield mock_session.return_value
```

#### Advanced Fixtures

```python
@pytest.fixture
def live_services():
    """Fixture for testing with live services."""
    # Setup live service connections
    yield services
    # Cleanup after tests

@pytest.fixture
def phase7_demonstration_data():
    """Fixture with Phase 7 demonstration data."""
    return {
        "data_generation": {...},
        "model_setup": {...},
        "ai_architect_customization": {...}
    }
```

### Test Utilities

```python
# Test data generators
def generate_test_model_config():
    """Generate test model configuration."""
    pass

def create_mock_evaluation_data():
    """Create mock evaluation data."""
    pass

# Assertion helpers
def assert_model_config_valid(config):
    """Assert model configuration is valid."""
    pass
```

## 🚨 Troubleshooting

### Common Test Issues

#### Import Errors

```bash
# Fix import path issues
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python -m pytest tests/unit/ -v
```

#### Service Connection Issues

```bash
# Test service connectivity
curl http://localhost:8080/health
curl http://localhost:7860
curl http://localhost:5000

# Run tests with service validation
python -m pytest tests/integration/ -v --validate-services
```

#### Mock Configuration Issues

```bash
# Run tests with mock debugging
python -m pytest tests/unit/ -v -s --pdb

# Check mock call history
python -m pytest tests/unit/test_basic_functionality.py::TestMockFunctionality -v -s
```

### Debug Procedures

```bash
# Enable verbose test output
python -m pytest tests/ -v -s

# Run specific test with debugging
python -m pytest tests/unit/test_basic_functionality.py::TestModelConfig::test_model_config_creation -v -s --pdb

# Generate test coverage report
python -m pytest tests/ --cov=src --cov-report=html --cov-report=term
```

## 📞 Support

### Test Infrastructure Support

- **Test Runner**: [tests/test_runner.py](mdc:tests/test_runner.py)
- **Configuration**: [config/pytest.ini](mdc:config/pytest.ini)
- **Fixtures**: [tests/fixtures/comprehensive_test_fixtures.py](mdc:tests/fixtures/comprehensive_test_fixtures.py)

### Related Documentation

- [FastAPI Enterprise Platform](fastapi-enterprise.md)
- [Gradio Model Evaluation](gradio-model-evaluation.md)
- [Troubleshooting Guide](../resources/troubleshooting.md)
- [Progress Bulletin](../progress-bulletin.md)

### GitHub Pages Integration

- **Local Development**: http://localhost:8000 (MkDocs serve)
- **GitHub Pages**: https://s-n00b.github.io/ai_assignments
- **FastAPI Docs**: http://localhost:8080/docs
- **Gradio App**: http://localhost:7860

---

**Last Updated**: December 2024  
**Version**: 1.0.0  
**Status**: Production Ready  
**Integration**: Full FastAPI & Gradio Backend Integration
