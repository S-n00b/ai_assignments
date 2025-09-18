# Makefile for Lenovo AAITC Solutions Testing Suite

.PHONY: help install test test-unit test-integration test-e2e test-all lint format security clean coverage benchmark

# Default target
help:
	@echo "Available targets:"
	@echo "  install          Install dependencies"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-e2e         Run end-to-end tests only"
	@echo "  test-all         Run all tests with coverage"
	@echo "  lint             Run linting checks"
	@echo "  format           Format code with black and isort"
	@echo "  security         Run security checks"
	@echo "  coverage         Generate coverage report"
	@echo "  benchmark        Run performance benchmarks"
	@echo "  clean            Clean up generated files"

# Installation
install:
	pip install -r requirements.txt
	pre-commit install

# Testing
test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v --tb=short

test-integration:
	pytest tests/integration/ -v --tb=short

test-e2e:
	pytest tests/e2e/ -v --tb=short --timeout=600

test-all:
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term-missing --cov-report=xml

# Linting and formatting
lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	mypy . --ignore-missing-imports
	black --check .
	isort --check-only .

format:
	black .
	isort .

# Security
security:
	bandit -r . -f json -o bandit-report.json
	safety check --json --output safety-report.json

# Coverage
coverage:
	pytest tests/ --cov=. --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/index.html"

# Performance benchmarks
benchmark:
	pytest tests/unit/ --benchmark-only --benchmark-save=baseline --benchmark-histogram=benchmark-histogram

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .benchmarks/
	rm -rf dist/
	rm -rf build/
	rm -f bandit-report.json
	rm -f safety-report.json
	rm -f coverage.xml
	rm -f *.xml

# Development helpers
dev-setup: install
	@echo "Setting up development environment..."
	pre-commit install
	@echo "Development environment ready!"

# Quick test (smoke tests only)
test-smoke:
	pytest tests/unit/ -v -m "not slow" --tb=short

# Test with specific markers
test-api:
	pytest tests/ -v -m api --tb=short

test-slow:
	pytest tests/ -v -m slow --tb=short

# Parallel testing
test-parallel:
	pytest tests/ -v -n auto

# Test specific module
test-model-eval:
	pytest tests/unit/test_model_evaluation.py -v

test-ai-arch:
	pytest tests/unit/test_ai_architecture.py -v

test-gradio:
	pytest tests/unit/test_gradio_app.py -v

test-utils:
	pytest tests/unit/test_utils.py -v

# Integration test specific modules
test-model-eval-integration:
	pytest tests/integration/test_model_evaluation_integration.py -v

test-ai-arch-integration:
	pytest tests/integration/test_ai_architecture_integration.py -v

test-gradio-integration:
	pytest tests/integration/test_gradio_integration.py -v

# E2E test specific modules
test-workflows:
	pytest tests/e2e/test_complete_workflows.py -v

test-user-scenarios:
	pytest tests/e2e/test_user_scenarios.py -v

# Documentation
docs:
	@echo "Generating documentation..."
	# Add documentation generation commands here

# Docker helpers
docker-build:
	docker build -t lenovo-aaitc-solutions .

docker-test:
	docker run --rm lenovo-aaitc-solutions make test

# CI/CD helpers
ci-test:
	pytest tests/unit/ -v --junitxml=unit-test-results.xml
	pytest tests/integration/ -v --junitxml=integration-test-results.xml
	pytest tests/e2e/ -v --junitxml=e2e-test-results.xml

ci-lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	black --check .
	isort --check-only .
	mypy . --ignore-missing-imports

ci-security:
	bandit -r . -f json -o bandit-report.json
	safety check --json --output safety-report.json
