# Testing Guide for Lenovo AAITC Solutions

## Overview

This document provides comprehensive guidance for testing the Lenovo AAITC Solutions project. The testing suite includes unit tests, integration tests, and end-to-end tests to ensure the reliability and quality of the AI model evaluation and architecture framework.

## Table of Contents

- [Testing Philosophy](#testing-philosophy)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Categories](#test-categories)
- [Writing Tests](#writing-tests)
- [Test Data and Fixtures](#test-data-and-fixtures)
- [Mocking and Stubbing](#mocking-and-stubbing)
- [Performance Testing](#performance-testing)
- [CI/CD Integration](#cicd-integration)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Testing Philosophy

Our testing approach follows these principles:

1. **Comprehensive Coverage**: Tests cover all major components and user scenarios
2. **Fast Feedback**: Unit tests provide quick feedback during development
3. **Reliable Integration**: Integration tests verify component interactions
4. **Real-world Scenarios**: E2E tests validate complete user workflows
5. **Maintainable**: Tests are well-organized, documented, and easy to maintain

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                 # Shared fixtures and configuration
├── unit/                       # Unit tests for individual components
│   ├── test_model_evaluation.py
│   ├── test_ai_architecture.py
│   ├── test_gradio_app.py
│   └── test_utils.py
├── integration/                # Integration tests for component interactions
│   ├── test_model_evaluation_integration.py
│   ├── test_ai_architecture_integration.py
│   └── test_gradio_integration.py
├── e2e/                        # End-to-end tests for complete workflows
│   ├── test_complete_workflows.py
│   └── test_user_scenarios.py
└── fixtures/                   # Shared test fixtures and utilities
    ├── mock_objects.py
    └── test_data.py
```

## Running Tests

### Prerequisites

1. Install dependencies:

   ```bash
   pip install -r config/requirements.txt
   ```

2. Set up environment variables (for integration tests):
   ```bash
   export OPENAI_API_KEY="your_key_here"
   export ANTHROPIC_API_KEY="your_key_here"
   ```

### Basic Test Commands

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/              # Unit tests only
pytest tests/integration/       # Integration tests only
pytest tests/e2e/              # End-to-end tests only

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test files
pytest tests/unit/test_model_evaluation.py

# Run specific test methods
pytest tests/unit/test_model_evaluation.py::TestModelConfig::test_model_config_creation
```

### Using Make Commands

```bash
# Quick commands
make test                      # Run all tests
make test-unit                 # Unit tests only
make test-integration          # Integration tests only
make test-e2e                  # End-to-end tests only
make test-all                  # All tests with coverage

# Development commands
make lint                      # Run linting checks
make format                    # Format code
make security                  # Run security checks
make clean                     # Clean up generated files
```

## Test Categories

### Unit Tests

Unit tests verify individual components in isolation:

- **Model Evaluation**: Configuration, pipeline, robustness testing, bias detection
- **AI Architecture**: Platform, lifecycle management, agents, RAG systems
- **Gradio App**: Interfaces, components, MCP server integration
- **Utils**: Logging, visualization, data processing, configuration

**Example:**

```python
def test_model_config_creation():
    """Test basic model configuration creation."""
    config = ModelConfig(
        model_name="gpt-3.5-turbo",
        model_version="2024-01-01",
        api_key="test-key"
    )

    assert config.model_name == "gpt-3.5-turbo"
    assert config.max_tokens == 1000  # default value
```

### Integration Tests

Integration tests verify component interactions:

- **Model Evaluation Integration**: Pipeline with robustness and bias detection
- **AI Architecture Integration**: Platform with lifecycle management and agents
- **Gradio Integration**: Frontend with backend systems

**Example:**

```python
@pytest.mark.asyncio
async def test_complete_evaluation_workflow():
    """Test complete evaluation workflow integration."""
    pipeline = ComprehensiveEvaluationPipeline(...)
    robustness_suite = RobustnessTestingSuite(...)

    # Test integrated workflow
    results = await pipeline.evaluate_all_models(test_data)
    assert "robustness" in results[0]
```

### End-to-End Tests

E2E tests validate complete user workflows:

- **Complete Workflows**: Model evaluation, AI architecture, MLOps, RAG systems
- **User Scenarios**: Data scientist, ML engineer, business user perspectives

**Example:**

```python
@pytest.mark.asyncio
async def test_data_scientist_model_comparison_scenario():
    """Test data scientist comparing multiple models."""
    # Step 1: User logs in
    # Step 2: Selects models
    # Step 3: Configures evaluation
    # Step 4: Runs evaluation
    # Step 5: Analyzes results
    # Step 6: Generates report
```

## Writing Tests

### Test Naming Conventions

- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`
- Descriptive names that explain what is being tested

### Test Structure

```python
class TestComponentName:
    """Test cases for ComponentName class."""

    @pytest.fixture
    def component(self):
        """Create component instance for testing."""
        return ComponentName()

    def test_basic_functionality(self, component):
        """Test basic component functionality."""
        # Arrange
        input_data = "test_input"

        # Act
        result = component.process(input_data)

        # Assert
        assert result == "expected_output"

    @pytest.mark.asyncio
    async def test_async_functionality(self, component):
        """Test async component functionality."""
        result = await component.async_process("test_input")
        assert result is not None
```

### Test Markers

Use pytest markers to categorize tests:

```python
@pytest.mark.unit
def test_unit_functionality():
    pass

@pytest.mark.integration
def test_integration_functionality():
    pass

@pytest.mark.e2e
def test_e2e_functionality():
    pass

@pytest.mark.slow
def test_slow_functionality():
    pass

@pytest.mark.api
def test_api_functionality():
    pass
```

## Test Data and Fixtures

### Using Fixtures

Fixtures provide reusable test data and setup:

```python
@pytest.fixture
def sample_model_config():
    """Sample model configuration for testing."""
    return ModelConfig(
        model_name="gpt-3.5-turbo",
        api_key="test-key"
    )

def test_with_fixture(sample_model_config):
    assert sample_model_config.model_name == "gpt-3.5-turbo"
```

### Test Data Files

Sample data is provided in `tests/fixtures/test_data.py`:

- `sample_evaluation_dataset`: Test prompts and expected outputs
- `sample_bias_test_dataset`: Bias testing data
- `sample_robustness_test_dataset`: Robustness testing data
- `sample_model_configurations`: Model configurations
- `sample_architecture_configurations`: Architecture configurations

## Mocking and Stubbing

### Mock Objects

Use mock objects to isolate components:

```python
from unittest.mock import Mock, patch, AsyncMock

def test_with_mock():
    mock_client = Mock()
    mock_client.generate.return_value = {"response": "test"}

    with patch('module.api_client', mock_client):
        result = module.call_api()
        assert result == "test"
```

### Async Mocking

For async functions, use `AsyncMock`:

```python
@pytest.mark.asyncio
async def test_async_with_mock():
    mock_client = AsyncMock()
    mock_client.generate.return_value = {"response": "test"}

    with patch('module.async_client', mock_client):
        result = await module.async_call_api()
        assert result == "test"
```

## Performance Testing

### Benchmarking

Use pytest-benchmark for performance testing:

```python
def test_performance(benchmark):
    result = benchmark(expensive_function, large_dataset)
    assert result is not None
```

### Performance Baselines

Run benchmarks to establish baselines:

```bash
pytest --benchmark-only --benchmark-save=baseline
```

Compare against baselines:

```bash
pytest --benchmark-compare=baseline
```

## CI/CD Integration

### GitHub Actions

Tests run automatically on:

- Push to main/develop branches
- Pull requests
- Daily scheduled runs

### Test Reports

CI generates:

- Unit test results
- Integration test results
- E2E test results
- Coverage reports
- Security scan results
- Performance benchmarks

### Local CI Simulation

Run CI checks locally:

```bash
make ci-test      # Run all tests
make ci-lint      # Run linting
make ci-security  # Run security checks
```

## Best Practices

### Test Organization

1. **One test per concept**: Each test should verify one specific behavior
2. **Descriptive names**: Test names should clearly describe what is being tested
3. **Arrange-Act-Assert**: Structure tests with clear setup, execution, and verification
4. **Independent tests**: Tests should not depend on each other

### Test Data

1. **Use fixtures**: Reuse common test data through fixtures
2. **Minimal data**: Use the smallest dataset that tests the functionality
3. **Realistic data**: Use data that represents real-world scenarios
4. **Clean data**: Ensure test data is consistent and predictable

### Error Handling

1. **Test error cases**: Verify that errors are handled correctly
2. **Test edge cases**: Include boundary conditions and edge cases
3. **Test validation**: Verify input validation and error messages

### Async Testing

1. **Use pytest-asyncio**: Mark async tests with `@pytest.mark.asyncio`
2. **Mock async dependencies**: Use `AsyncMock` for async dependencies
3. **Test timeouts**: Include timeout handling in async tests

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Async Test Failures**: Check that async tests are properly marked
3. **Mock Issues**: Verify mock objects are properly configured
4. **Timeout Errors**: Increase timeout for slow tests

### Debug Commands

```bash
# Run tests with verbose output
pytest -v -s

# Run specific test with debugging
pytest tests/unit/test_model_evaluation.py::TestModelConfig::test_model_config_creation -v -s

# Run tests with coverage and show missing lines
pytest --cov=. --cov-report=term-missing

# Run tests and stop on first failure
pytest -x

# Run tests with maximum failures
pytest --maxfail=3
```

### Test Environment

Ensure your test environment matches CI:

```bash
# Use same Python version as CI
python --version

# Install exact dependencies
pip install -r config/requirements.txt

# Set environment variables
export PYTHONPATH=$PWD
```

## Contributing

When adding new tests:

1. Follow the existing test structure and naming conventions
2. Add appropriate test markers
3. Include docstrings explaining what is being tested
4. Use fixtures for common test data
5. Mock external dependencies
6. Ensure tests are fast and reliable
7. Update this documentation if needed

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/)
- [unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
- [pytest-benchmark Documentation](https://pytest-benchmark.readthedocs.io/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
