---
layout: post
title: "Testing Guide"
date: 2025-09-18 10:00:00 -0400
categories: [Documentation, Testing]
tags: [Testing, Quality Assurance, Unit Tests, Integration Tests, E2E Tests]
author: Lenovo AAITC Team
---

# Testing Guide - Lenovo AAITC Solutions

## Overview

This guide provides comprehensive testing instructions for the Lenovo AAITC Solutions framework, covering unit tests, integration tests, end-to-end tests, and performance testing.

## Table of Contents

1. [Testing Strategy](#testing-strategy)
2. [Test Environment Setup](#test-environment-setup)
3. [Unit Testing](#unit-testing)
4. [Integration Testing](#integration-testing)
5. [End-to-End Testing](#end-to-end-testing)
6. [Performance Testing](#performance-testing)
7. [Test Coverage](#test-coverage)
8. [Continuous Integration](#continuous-integration)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Testing Strategy

### Test Pyramid

Our testing strategy follows the test pyramid approach:

1. **Unit Tests (70%)**: Fast, isolated tests for individual components
2. **Integration Tests (20%)**: Tests for component interactions
3. **End-to-End Tests (10%)**: Full system tests

### Test Categories

- **Unit Tests**: Individual function and class testing
- **Integration Tests**: API and service integration testing
- **End-to-End Tests**: Complete workflow testing
- **Performance Tests**: Load and stress testing
- **Security Tests**: Vulnerability and penetration testing

---

## Test Environment Setup

### Prerequisites

```bash
# Install testing dependencies
pip install -r config/requirements-testing.txt

# Install additional testing tools
pip install pytest pytest-cov pytest-xdist pytest-benchmark
pip install pytest-asyncio pytest-mock
pip install coverage bandit safety
```

### Test Configuration

The project uses `pytest.ini` for configuration:

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    -v
    --tb=short
    --strict-markers
    --disable-warnings
    --cov=src
    --cov-report=html
    --cov-report=term-missing
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow running tests
    api: API tests
    performance: Performance tests
```

---

## Unit Testing

### Running Unit Tests

```bash
# Run all unit tests
python -m pytest tests/unit/ -v

# Run specific unit test file
python -m pytest tests/unit/test_model_evaluation.py -v

# Run with coverage
python -m pytest tests/unit/ --cov=src --cov-report=html
```

### Example Unit Test

```python
# tests/unit/test_model_evaluation.py
import pytest
from src.model_evaluation.config import ModelConfig, LATEST_MODEL_CONFIGS
from src.model_evaluation.pipeline import ComprehensiveEvaluationPipeline

class TestModelConfig:
    def test_model_config_creation(self):
        """Test ModelConfig creation with valid data."""
        config = ModelConfig(
            name="Test Model",
            provider="openai",
            model_id="gpt-4",
            max_tokens=1000,
            temperature=0.7
        )

        assert config.name == "Test Model"
        assert config.provider == "openai"
        assert config.model_id == "gpt-4"
        assert config.max_tokens == 1000
        assert config.temperature == 0.7

    def test_model_config_validation(self):
        """Test ModelConfig validation."""
        config = ModelConfig(
            name="Test Model",
            provider="openai",
            model_id="gpt-4"
        )

        assert config.validate() == True

    def test_invalid_model_config(self):
        """Test ModelConfig with invalid data."""
        with pytest.raises(ValueError):
            ModelConfig(
                name="",  # Invalid empty name
                provider="openai",
                model_id="gpt-4"
            )

class TestComprehensiveEvaluationPipeline:
    @pytest.fixture
    def pipeline(self):
        """Create a test pipeline instance."""
        models = [LATEST_MODEL_CONFIGS["gpt-5"]]
        return ComprehensiveEvaluationPipeline(models)

    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline is not None
        assert len(pipeline.models) == 1
        assert pipeline.models[0].name == "GPT-5"

    @pytest.mark.asyncio
    async def test_model_evaluation(self, pipeline):
        """Test model evaluation functionality."""
        # Mock test data
        test_data = pd.DataFrame({
            'prompt': ['Test prompt 1', 'Test prompt 2'],
            'expected_output': ['Expected 1', 'Expected 2']
        })

        # This would be mocked in real tests
        result = await pipeline.evaluate_model_comprehensive(
            pipeline.models[0],
            test_data,
            TaskType.TEXT_GENERATION
        )

        assert result is not None
        assert 'metrics' in result
```

---

## Integration Testing

### Running Integration Tests

```bash
# Run all integration tests
python -m pytest tests/integration/ -v

# Run specific integration test
python -m pytest tests/integration/test_model_evaluation_integration.py -v

# Run with timeout for long-running tests
python -m pytest tests/integration/ --timeout=300
```

### Example Integration Test

```python
# tests/integration/test_model_evaluation_integration.py
import pytest
import asyncio
from src.model_evaluation.pipeline import ComprehensiveEvaluationPipeline
from src.model_evaluation.config import LATEST_MODEL_CONFIGS
from src.utils.logging_system import LoggingSystem

class TestModelEvaluationIntegration:
    @pytest.fixture
    async def evaluation_pipeline(self):
        """Create evaluation pipeline for integration testing."""
        models = [
            LATEST_MODEL_CONFIGS["gpt-5"],
            LATEST_MODEL_CONFIGS["claude-3.5-sonnet"]
        ]
        pipeline = ComprehensiveEvaluationPipeline(models)
        return pipeline

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multi_model_evaluation(self, evaluation_pipeline):
        """Test evaluation across multiple models."""
        # Create test dataset
        test_data = pd.DataFrame({
            'prompt': [
                'Explain quantum computing in simple terms.',
                'Write a Python function to sort a list.',
                'What are the benefits of renewable energy?'
            ],
            'category': ['reasoning', 'code', 'knowledge']
        })

        # Run evaluation
        results = await evaluation_pipeline.run_multi_task_evaluation({
            TaskType.TEXT_GENERATION: test_data
        })

        # Verify results
        assert results is not None
        assert len(results) > 0
        assert 'model_name' in results.columns
        assert 'accuracy' in results.columns

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_robustness_integration(self, evaluation_pipeline):
        """Test robustness testing integration."""
        from src.model_evaluation.robustness import RobustnessTestingSuite

        robustness_suite = RobustnessTestingSuite()
        test_prompts = [
            "Normal prompt",
            "Prompt with typos: helo world",
            "PROMPT IN ALL CAPS",
            "Prompt with special characters: @#$%^&*()"
        ]

        results = await robustness_suite.test_noise_tolerance(
            evaluation_pipeline.models[0],
            test_prompts
        )

        assert results is not None
        assert 'noise_tolerance_score' in results
```

---

## End-to-End Testing

### Running E2E Tests

```bash
# Run all E2E tests
python -m pytest tests/e2e/ -v

# Run specific E2E test
python -m pytest tests/e2e/test_complete_workflows.py -v

# Run with longer timeout
python -m pytest tests/e2e/ --timeout=600
```

### Example E2E Test

```python
# tests/e2e/test_complete_workflows.py
import pytest
import asyncio
from src.gradio_app.main import LenovoAAITCApp
from src.model_evaluation.pipeline import ComprehensiveEvaluationPipeline

class TestCompleteWorkflows:
    @pytest.fixture
    async def app(self):
        """Create application instance for E2E testing."""
        app = LenovoAAITCApp()
        return app

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_complete_model_evaluation_workflow(self, app):
        """Test complete model evaluation workflow."""
        # Initialize pipeline
        models = [LATEST_MODEL_CONFIGS["gpt-5"]]
        pipeline = ComprehensiveEvaluationPipeline(models)

        # Create test dataset
        test_data = pd.DataFrame({
            'prompt': [
                'Write a haiku about artificial intelligence.',
                'Solve this math problem: 2x + 5 = 15',
                'Explain the concept of machine learning.'
            ],
            'expected_output': [
                'AI learns and grows,',
                'x = 5',
                'Machine learning is...'
            ]
        })

        # Run complete evaluation
        results = await pipeline.run_multi_task_evaluation({
            TaskType.TEXT_GENERATION: test_data
        }, include_robustness=True, include_bias_detection=True)

        # Generate report
        report = pipeline.generate_evaluation_report(results, "html")

        # Verify complete workflow
        assert results is not None
        assert len(results) > 0
        assert report is not None
        assert len(report) > 0

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_ai_architecture_deployment_workflow(self, app):
        """Test complete AI architecture deployment workflow."""
        from src.ai_architecture.platform import HybridAIPlatform
        from src.ai_architecture.lifecycle import ModelLifecycleManager

        # Initialize platform
        platform = HybridAIPlatform()
        lifecycle_manager = ModelLifecycleManager()

        # Register model
        model_version = await lifecycle_manager.register_model(
            model_id="test-model",
            version="1.0.0",
            stage=ModelStage.DEVELOPMENT,
            created_by="test-user",
            description="Test model for E2E testing"
        )

        # Deploy model
        deployment_result = await platform.deploy_model(
            model_config=ModelDeploymentConfig(
                model_id="test-model",
                version="1.0.0"
            ),
            target_environment=DeploymentTarget.CLOUD
        )

        # Verify deployment
        assert deployment_result is not None
        assert deployment_result['status'] == 'success'
        assert 'deployment_id' in deployment_result
```

---

## Performance Testing

### Running Performance Tests

```bash
# Run performance tests
python -m pytest tests/unit/ --benchmark-only --benchmark-save=baseline

# Compare with baseline
python -m pytest tests/unit/ --benchmark-compare --benchmark-compare-fail=mean:5%

# Run load tests
python -m pytest tests/performance/ -v
```

### Example Performance Test

```python
# tests/performance/test_performance.py
import pytest
import asyncio
import time
from src.model_evaluation.pipeline import ComprehensiveEvaluationPipeline

class TestPerformance:
    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_model_evaluation_performance(self, benchmark):
        """Benchmark model evaluation performance."""
        models = [LATEST_MODEL_CONFIGS["gpt-5"]]
        pipeline = ComprehensiveEvaluationPipeline(models)

        test_data = pd.DataFrame({
            'prompt': ['Test prompt'] * 100,
            'expected_output': ['Expected output'] * 100
        })

        def run_evaluation():
            return asyncio.run(pipeline.run_multi_task_evaluation({
                TaskType.TEXT_GENERATION: test_data
            }))

        result = benchmark(run_evaluation)
        assert result is not None

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_evaluations(self):
        """Test concurrent model evaluations."""
        models = [LATEST_MODEL_CONFIGS["gpt-5"]]
        pipeline = ComprehensiveEvaluationPipeline(models)

        test_data = pd.DataFrame({
            'prompt': ['Concurrent test prompt'],
            'expected_output': ['Expected output']
        })

        # Run 10 concurrent evaluations
        tasks = []
        for _ in range(10):
            task = pipeline.run_multi_task_evaluation({
                TaskType.TEXT_GENERATION: test_data
            })
            tasks.append(task)

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        # Verify all evaluations completed
        assert len(results) == 10
        assert all(result is not None for result in results)

        # Verify performance (should complete within reasonable time)
        execution_time = end_time - start_time
        assert execution_time < 60  # Should complete within 60 seconds
```

---

## Test Coverage

### Coverage Requirements

- **Minimum Coverage**: 80% overall
- **Critical Components**: 95% coverage
- **New Code**: 90% coverage

### Running Coverage Analysis

```bash
# Generate coverage report
python -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# View coverage report
Start-Process htmlcov/index.html  # Windows
open htmlcov/index.html          # macOS
xdg-open htmlcov/index.html      # Linux
```

### Coverage Configuration

```ini
# .coveragerc
[run]
source = src
omit =
    */tests/*
    */venv/*
    */__pycache__/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
```

---

## Continuous Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r config/requirements.txt
          pip install -r config/requirements-testing.txt

      - name: Run unit tests
        run: |
          python -m pytest tests/unit/ -v --cov=src --cov-report=xml

      - name: Run integration tests
        run: |
          python -m pytest tests/integration/ -v --timeout=300

      - name: Run E2E tests
        run: |
          python -m pytest tests/e2e/ -v --timeout=600

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

---

## Best Practices

### Test Organization

1. **Test Structure**: Mirror source code structure
2. **Naming Convention**: Use descriptive test names
3. **Test Isolation**: Each test should be independent
4. **Test Data**: Use fixtures for reusable test data

### Test Quality

1. **AAA Pattern**: Arrange, Act, Assert
2. **Single Responsibility**: One assertion per test
3. **Clear Assertions**: Use descriptive assertion messages
4. **Mock External Dependencies**: Isolate units under test

### Performance Considerations

1. **Fast Tests**: Unit tests should run quickly
2. **Parallel Execution**: Use pytest-xdist for parallel testing
3. **Test Data Size**: Use minimal test data
4. **Cleanup**: Clean up resources after tests

---

## Troubleshooting

### Common Test Issues

#### 1. Import Errors

```bash
# Add project root to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use pytest with proper path
python -m pytest tests/ -v
```

#### 2. Async Test Issues

```python
# Use pytest-asyncio for async tests
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result is not None
```

#### 3. Mock Issues

```python
# Use pytest-mock for mocking
def test_with_mock(mocker):
    mock_api = mocker.patch('src.api.external_api')
    mock_api.return_value = "mocked response"

    result = function_that_uses_api()
    assert result == "expected result"
```

#### 4. Timeout Issues

```bash
# Increase timeout for slow tests
python -m pytest tests/ --timeout=600

# Or mark slow tests
@pytest.mark.slow
def test_slow_function():
    # Slow test implementation
    pass
```

### Test Debugging

```bash
# Run tests with verbose output
python -m pytest tests/ -v -s

# Run specific test with debugging
python -m pytest tests/unit/test_specific.py::test_function -v -s

# Use pdb for debugging
python -m pytest tests/ --pdb
```

---

## Test Maintenance

### Regular Tasks

1. **Weekly**: Review test failures and fix flaky tests
2. **Monthly**: Update test data and fixtures
3. **Quarterly**: Review and update test coverage requirements
4. **Annually**: Refactor and optimize test suite

### Test Metrics

- **Test Coverage**: Track coverage trends
- **Test Execution Time**: Monitor test performance
- **Test Failure Rate**: Track test reliability
- **Flaky Test Rate**: Identify and fix unstable tests

---

**Testing Guide - Lenovo AAITC Solutions**  
_Comprehensive testing instructions for quality assurance_
