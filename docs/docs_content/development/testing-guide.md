# Comprehensive Testing Guide

## 🎯 Overview

Complete guide for executing and managing the comprehensive test infrastructure for GitHub Pages frontend, platform architecture layers, and Phase 7 demonstration flow.

## 🚀 Key Features

### Core Testing Capabilities

- **Comprehensive Coverage**: Unit, integration, and end-to-end testing
- **Service Integration**: FastAPI, Gradio, MLflow, and ChromaDB testing
- **GitHub Pages Testing**: Local and hosted frontend validation
- **Phase 7 Flow Testing**: Complete demonstration workflow validation
- **Automated Execution**: Test runner and CI/CD integration

### Integration Features

- **Live Service Testing**: Real service integration validation
- **Mock Testing**: Isolated component testing with mocks
- **Performance Testing**: Load and stress testing capabilities
- **Coverage Reporting**: Comprehensive test coverage analysis

## 📊 Test Execution Strategy

### Test Execution Hierarchy

```bash
# 1. Unit Tests (Fastest)
python -m pytest tests/unit/ -v

# 2. Integration Tests (Medium)
python -m pytest tests/integration/ -v

# 3. End-to-End Tests (Slowest)
python -m pytest tests/e2e/ -v

# 4. Complete Test Suite
python -m pytest tests/ -v
```

### Test Categories by Execution Time

| Category              | Execution Time | Coverage        | Purpose                       |
| --------------------- | -------------- | --------------- | ----------------------------- |
| **Unit Tests**        | < 30 seconds   | Component-level | Fast feedback loop            |
| **Integration Tests** | 1-5 minutes    | Service-level   | Cross-component validation    |
| **E2E Tests**         | 5-15 minutes   | System-level    | Complete workflow validation  |
| **Performance Tests** | 10-30 minutes  | Load testing    | System performance validation |

## 🌐 Service Integration Testing

### Service Startup Sequence

```bash
# Terminal 1: Activate virtual environment
& C:\Users\samne\PycharmProjects\ai_assignments\venv\Scripts\Activate.ps1

# Terminal 2: Start ChromaDB
chroma run --host 0.0.0.0 --port 8000 --path chroma_data

# Terminal 3: Start MLflow
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

# Terminal 4: Start FastAPI Backend (Demo Mode)
python -m src.enterprise_llmops.main --host 0.0.0.0 --port 8080

# Terminal 5: Start Gradio Frontend
python -m src.gradio_app.main --host 0.0.0.0 --port 7860

# Terminal 6: Run Tests
python -m pytest tests/ -v
```

### Service Health Validation

```bash
# Validate all services are running
curl http://localhost:8080/health
curl http://localhost:7860
curl http://localhost:5000
curl http://localhost:8000

# Run service integration tests
python -m pytest tests/integration/test_service_level_interactions.py -v
```

## 🔧 Configuration Management

### Environment Setup

```bash
# 1. Virtual Environment Activation
& C:\Users\samne\PycharmProjects\ai_assignments\venv\Scripts\Activate.ps1

# 2. Install Test Dependencies
pip install -r config/requirements-testing.txt

# 3. Set Environment Variables
set PYTHONPATH=%PYTHONPATH%;%CD%
set TEST_ENVIRONMENT=development
set LOG_LEVEL=INFO
```

### Test Configuration Files

| File                          | Purpose                | Location                                                               |
| ----------------------------- | ---------------------- | ---------------------------------------------------------------------- |
| **pytest.ini**                | Pytest configuration   | [config/pytest.ini](mdc:config/pytest.ini)                             |
| **conftest_minimal.py**       | Basic test fixtures    | [tests/conftest_minimal.py](mdc:tests/conftest_minimal.py)             |
| **conftest_comprehensive.py** | Advanced test fixtures | [tests/conftest_comprehensive.py](mdc:tests/conftest_comprehensive.py) |

## 📚 Test Execution Commands

### Unit Testing Commands

```bash
# Run all unit tests
python -m pytest tests/unit/ -v

# Run specific test file
python -m pytest tests/unit/test_basic_functionality.py -v

# Run specific test class
python -m pytest tests/unit/test_basic_functionality.py::TestModelConfig -v

# Run specific test method
python -m pytest tests/unit/test_basic_functionality.py::TestModelConfig::test_model_config_creation -v

# Run with coverage
python -m pytest tests/unit/ --cov=src --cov-report=html
```

### Integration Testing Commands

```bash
# Run all integration tests
python -m pytest tests/integration/ -v

# Run platform architecture tests
python -m pytest tests/integration/test_platform_architecture_layers.py -v

# Run service interaction tests
python -m pytest tests/integration/test_service_level_interactions.py -v

# Run with live services
python -m pytest tests/integration/ -v --live-services
```

### End-to-End Testing Commands

```bash
# Run all E2E tests
python -m pytest tests/e2e/ -v

# Run Phase 7 demonstration tests
python -m pytest tests/e2e/test_phase7_complete_demonstration.py -v

# Run GitHub Pages integration tests
python -m pytest tests/e2e/test_github_pages_frontend_integration.py -v

# Run with production URLs
python -m pytest tests/e2e/ -v --production-urls
```

### Phase 7 Demonstration Testing

```bash
# Test complete Phase 7 flow
python -m pytest tests/e2e/test_phase7_complete_demonstration.py -v

# Test individual Phase 7 components
python -m pytest tests/unit/test_phase7_demonstration_flow.py -v

# Test with demonstration data
python -m pytest tests/e2e/ -v --demo-data
```

## 🛠️ Development Workflow

### Test-Driven Development Cycle

```bash
# 1. Write failing test
python -m pytest tests/unit/test_new_feature.py::test_feature -v

# 2. Implement feature
# Edit source code

# 3. Run test until it passes
python -m pytest tests/unit/test_new_feature.py::test_feature -v

# 4. Run all related tests
python -m pytest tests/unit/test_new_feature.py -v

# 5. Run integration tests
python -m pytest tests/integration/ -v

# 6. Run E2E tests if applicable
python -m pytest tests/e2e/ -v
```

### Continuous Integration Testing

```bash
# Run full test suite (CI equivalent)
python -m pytest tests/ -v --tb=short --maxfail=5

# Run with parallel execution
python -m pytest tests/ -n auto

# Generate comprehensive report
python -m pytest tests/ --cov=src --cov-report=html --cov-report=xml --junitxml=test-results.xml
```

## 🚨 Troubleshooting

### Common Issues and Solutions

#### Import Errors

```bash
# Issue: ModuleNotFoundError
# Solution: Set PYTHONPATH
set PYTHONPATH=%PYTHONPATH%;%CD%
python -m pytest tests/unit/ -v
```

#### Service Connection Failures

```bash
# Issue: Connection refused errors
# Solution: Verify services are running
curl http://localhost:8080/health
curl http://localhost:7860
curl http://localhost:5000
curl http://localhost:8000

# Start missing services
python -m src.enterprise_llmops.main --host 0.0.0.0 --port 8080
python -m src.gradio_app.main --host 0.0.0.0 --port 7860
```

#### Test Timeout Issues

```bash
# Issue: Tests timing out
# Solution: Increase timeout
python -m pytest tests/ --timeout=300

# Run slow tests separately
python -m pytest tests/ -m "slow" --timeout=600
```

#### Mock Configuration Issues

```bash
# Issue: Mock not working as expected
# Solution: Debug with verbose output
python -m pytest tests/unit/test_basic_functionality.py::TestMockFunctionality -v -s

# Check mock call history
python -m pytest tests/unit/test_basic_functionality.py::TestMockFunctionality -v -s --pdb
```

### Debug Procedures

```bash
# Enable debug mode
python -m pytest tests/ -v -s --pdb

# Run specific test with debugging
python -m pytest tests/unit/test_basic_functionality.py::TestModelConfig::test_model_config_creation -v -s --pdb

# Generate detailed failure report
python -m pytest tests/ --tb=long -v

# Run tests with logging
python -m pytest tests/ -v -s --log-cli-level=DEBUG
```

## 📞 Support and Resources

### Test Infrastructure Documentation

- **API Documentation**: [pytest-testing-infrastructure.md](../api/pytest-testing-infrastructure.md)
- **Test Runner**: [tests/test_runner.py](mdc:tests/test_runner.py)
- **Configuration**: [config/pytest.ini](mdc:config/pytest.ini)

### Related Services

- **FastAPI Enterprise**: [fastapi-enterprise.md](../api/fastapi-enterprise.md)
- **Gradio Model Evaluation**: [gradio-model-evaluation.md](../api/gradio-model-evaluation.md)
- **Troubleshooting Guide**: [troubleshooting.md](../resources/troubleshooting.md)

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
