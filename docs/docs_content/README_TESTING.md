# Testing Suite Overview

## Quick Start

```bash
# Install dependencies
pip install -r config/requirements.txt

# Run all tests
make test

# Run specific test categories
make test-unit
make test-integration
make test-e2e

# Run with coverage
make test-all
```

## Test Structure

- **Unit Tests** (`tests/unit/`): Test individual components in isolation
- **Integration Tests** (`tests/integration/`): Test component interactions
- **End-to-End Tests** (`tests/e2e/`): Test complete user workflows
- **Fixtures** (`tests/fixtures/`): Shared test data and mock objects

## Key Features

✅ **Comprehensive Coverage**: Unit, integration, and E2E tests  
✅ **Async Support**: Full async/await testing with pytest-asyncio  
✅ **Mock Objects**: Extensive mocking for external dependencies  
✅ **Performance Testing**: Benchmarking with pytest-benchmark  
✅ **CI/CD Integration**: GitHub Actions workflows  
✅ **Security Testing**: Bandit and Safety checks  
✅ **Code Quality**: Black, isort, flake8, mypy integration

## Test Categories

### Unit Tests

- Model evaluation components
- AI architecture modules
- Gradio application components
- Utility functions

### Integration Tests

- Model evaluation pipeline integration
- AI architecture component interactions
- Gradio frontend-backend integration

### End-to-End Tests

- Complete model evaluation workflows
- AI architecture design and deployment
- User scenarios (data scientist, ML engineer, business user)

## Running Tests

### Basic Commands

```bash
pytest                    # Run all tests
pytest tests/unit/        # Unit tests only
pytest tests/integration/ # Integration tests only
pytest tests/e2e/         # End-to-end tests only
```

### With Coverage

```bash
pytest --cov=. --cov-report=html
```

### Specific Tests

```bash
pytest tests/unit/test_model_evaluation.py
pytest tests/unit/test_model_evaluation.py::TestModelConfig
```

### Using Make

```bash
make test                 # All tests
make test-unit           # Unit tests
make test-integration    # Integration tests
make test-e2e           # End-to-end tests
make test-all           # All tests with coverage
make lint               # Linting checks
make format             # Code formatting
make security           # Security checks
```

## Test Markers

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.e2e`: End-to-end tests
- `@pytest.mark.slow`: Slow-running tests
- `@pytest.mark.api`: Tests requiring API access

## Fixtures and Mock Objects

### Available Fixtures

- `mock_api_client`: Mock API client for testing
- `mock_database`: Mock database for testing
- `mock_vector_store`: Mock vector store for RAG testing
- `sample_evaluation_data`: Sample evaluation data
- `sample_model_config`: Sample model configuration
- `sample_metrics_data`: Sample performance metrics

### Mock Objects

- `MockAPIClient`: Mock API client with async support
- `MockDatabase`: Mock database with common operations
- `MockVectorStore`: Mock vector store for RAG systems
- `MockMLflowClient`: Mock MLflow client for experiment tracking

## CI/CD Integration

### GitHub Actions Workflows

- **CI Pipeline** (`.github/workflows/ci.yml`): Full CI/CD pipeline
- **Test Suite** (`.github/workflows/test.yml`): Comprehensive testing

### Automated Checks

- Unit, integration, and E2E tests
- Code linting and formatting
- Security scanning
- Performance benchmarking
- Coverage reporting

## Performance Testing

### Benchmarking

```bash
pytest --benchmark-only --benchmark-save=baseline
pytest --benchmark-compare=baseline
```

### Performance Baselines

- Response time benchmarks
- Throughput measurements
- Memory usage tracking
- CPU utilization monitoring

## Security Testing

### Automated Security Checks

- **Bandit**: Security linting
- **Safety**: Dependency vulnerability scanning
- **Custom security tests**: API security, data protection

### Running Security Tests

```bash
make security
bandit -r .
safety check
```

## Test Data

### Sample Datasets

- Evaluation prompts and expected outputs
- Bias testing data
- Robustness testing scenarios
- Model configurations
- Architecture configurations
- Performance metrics

### Data Management

- Fixtures for consistent test data
- Mock objects for external dependencies
- Isolated test environments
- Cleanup after tests

## Best Practices

### Test Writing

1. **One concept per test**: Each test should verify one specific behavior
2. **Descriptive names**: Clear, descriptive test names
3. **Arrange-Act-Assert**: Clear test structure
4. **Independent tests**: Tests should not depend on each other

### Async Testing

1. Use `@pytest.mark.asyncio` for async tests
2. Use `AsyncMock` for async dependencies
3. Include proper timeout handling

### Mocking

1. Mock external dependencies
2. Use realistic mock data
3. Verify mock interactions
4. Clean up mocks after tests

## Troubleshooting

### Common Issues

- **Import errors**: Check dependencies are installed
- **Async test failures**: Verify async tests are properly marked
- **Mock issues**: Ensure mocks are properly configured
- **Timeout errors**: Increase timeout for slow tests

### Debug Commands

```bash
pytest -v -s                    # Verbose output
pytest -x                       # Stop on first failure
pytest --maxfail=3              # Stop after 3 failures
pytest --tb=short               # Short traceback format
```

## Documentation

- **TESTING.md**: Comprehensive testing guide
- **pytest.ini**: Pytest configuration
- **Makefile**: Common test commands
- **CI/CD workflows**: Automated testing setup

## Contributing

When adding new tests:

1. Follow existing naming conventions
2. Add appropriate test markers
3. Include descriptive docstrings
4. Use fixtures for common data
5. Mock external dependencies
6. Ensure tests are fast and reliable

## Support

For testing-related questions:

1. Check this documentation
2. Review existing test examples
3. Check CI/CD logs for issues
4. Consult pytest documentation
