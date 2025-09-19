# Contributing Guide

## Overview

Thank you for your interest in contributing to the AI Assignments project! This guide outlines the contribution process, coding standards, and best practices for developers.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Virtual environment setup
- Basic understanding of AI/ML concepts

### Setup Development Environment

#### 1. Fork and Clone Repository
```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/ai_assignments.git
cd ai_assignments

# Add upstream remote
git remote add upstream https://github.com/s-n00b/ai_assignments.git
```

#### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
# Install development dependencies
pip install -r config/requirements.txt
pip install -r config/requirements-testing.txt

# Install pre-commit hooks
pre-commit install
```

## Development Workflow

### 1. Create Feature Branch
```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes
- Write code following the coding standards
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Commit Changes
```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add new model evaluation metric

- Implement F1 score calculation
- Add unit tests for metric
- Update documentation"
```

### 4. Push and Create Pull Request
```bash
# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
```

## Coding Standards

### Python Code Style

#### PEP 8 Compliance
- Maximum line length: 88 characters (Black formatter)
- Use meaningful variable and function names
- Add docstrings for all public functions and classes
- Use type hints where appropriate

#### Example Code Style
```python
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates model performance using various metrics."""
    
    def __init__(self, model_config: Dict[str, str]) -> None:
        """Initialize the model evaluator.
        
        Args:
            model_config: Configuration dictionary for the model
        """
        self.config = model_config
        self.metrics: Dict[str, float] = {}
    
    def calculate_accuracy(self, predictions: List[int], 
                          labels: List[int]) -> float:
        """Calculate model accuracy.
        
        Args:
            predictions: List of predicted labels
            labels: List of true labels
            
        Returns:
            Accuracy score between 0 and 1
            
        Raises:
            ValueError: If predictions and labels have different lengths
        """
        if len(predictions) != len(labels):
            raise ValueError("Predictions and labels must have same length")
        
        correct = sum(p == l for p, l in zip(predictions, labels))
        return correct / len(predictions)
```

#### Code Formatting Tools
```bash
# Format code with Black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Lint with flake8
flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics

# Type checking with mypy
mypy src/ --ignore-missing-imports
```

### Documentation Standards

#### Docstring Format
Use Google-style docstrings:
```python
def process_data(data: List[Dict], config: Dict) -> List[Dict]:
    """Process input data according to configuration.
    
    Args:
        data: List of dictionaries containing raw data
        config: Configuration dictionary specifying processing steps
        
    Returns:
        List of processed data dictionaries
        
    Raises:
        ValueError: If data format is invalid
        ConfigurationError: If config contains invalid settings
        
    Example:
        >>> data = [{"text": "Hello world"}]
        >>> config = {"tokenize": True, "lowercase": True}
        >>> result = process_data(data, config)
        >>> print(result[0]["text"])
        hello world
    """
```

#### Markdown Documentation
- Use clear headings and structure
- Include code examples where helpful
- Add diagrams for complex concepts
- Keep documentation up-to-date with code changes

### Testing Standards

#### Test Structure
```python
# tests/unit/test_model_evaluation.py
import pytest
from unittest.mock import Mock, patch
from src.model_evaluation.pipeline import EvaluationPipeline


class TestEvaluationPipeline:
    """Test cases for EvaluationPipeline class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.config = {
            "model_path": "test_model.pt",
            "test_dataset": "test_data.csv"
        }
        self.pipeline = EvaluationPipeline(self.config)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization with valid config."""
        assert self.pipeline.config == self.config
        assert self.pipeline.model is None  # Not loaded yet
    
    def test_load_model_success(self):
        """Test successful model loading."""
        with patch('torch.load') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            result = self.pipeline.load_model()
            
            assert result is True
            assert self.pipeline.model == mock_model
            mock_load.assert_called_once_with(self.config["model_path"])
    
    def test_load_model_file_not_found(self):
        """Test model loading when file doesn't exist."""
        with patch('torch.load', side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                self.pipeline.load_model()
    
    @pytest.mark.parametrize("predictions,labels,expected", [
        ([1, 0, 1], [1, 0, 0], 0.67),  # 2/3 correct
        ([0, 0, 0], [0, 0, 0], 1.0),   # All correct
        ([1, 1, 1], [0, 0, 0], 0.0),   # None correct
    ])
    def test_calculate_accuracy(self, predictions, labels, expected):
        """Test accuracy calculation with various inputs."""
        result = self.pipeline.calculate_accuracy(predictions, labels)
        assert abs(result - expected) < 0.01
```

#### Test Coverage Requirements
- Aim for 80%+ code coverage
- Test all public methods and functions
- Include edge cases and error conditions
- Mock external dependencies

#### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v --tb=short

# Run specific test file
python -m pytest tests/unit/test_model_evaluation.py -v

# Run with coverage
python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# Run only fast tests (exclude slow integration tests)
python -m pytest tests/ -v -m "not slow"
```

## Pull Request Process

### 1. Before Submitting
- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New features have tests
- [ ] Documentation is updated
- [ ] No merge conflicts with main branch

### 2. Pull Request Template
```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed
- [ ] All existing tests pass

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No hardcoded values or secrets
- [ ] Error handling implemented
- [ ] Logging added where appropriate

## Related Issues
Closes #(issue number)

## Screenshots (if applicable)
Add screenshots to help explain your changes.
```

### 3. Code Review Process
- Assign appropriate reviewers
- Address all review comments
- Ensure CI/CD checks pass
- Get approval from maintainers before merging

## Issue Reporting

### Bug Reports
When reporting bugs, include:
```markdown
## Bug Description
Clear and concise description of the bug.

## Steps to Reproduce
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- OS: [e.g. Windows 10, macOS 12.0, Ubuntu 20.04]
- Python version: [e.g. 3.9.7]
- Package versions: [e.g. torch 1.12.0, transformers 4.20.0]

## Additional Context
Add any other context about the problem here.
```

### Feature Requests
For feature requests, include:
```markdown
## Feature Description
Clear and concise description of the feature.

## Use Case
Describe the use case and why this feature would be valuable.

## Proposed Solution
Describe how you would like this feature to work.

## Alternatives Considered
Describe any alternative solutions you've considered.

## Additional Context
Add any other context or screenshots about the feature request.
```

## Development Guidelines

### Git Commit Messages
Follow conventional commit format:
```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(evaluation): add F1 score metric
fix(api): handle missing authentication token
docs(readme): update installation instructions
test(model): add unit tests for prediction pipeline
```

### Branch Naming Convention
- `feature/description`: New features
- `bugfix/description`: Bug fixes
- `hotfix/description`: Critical bug fixes
- `docs/description`: Documentation updates
- `refactor/description`: Code refactoring

### File Organization
```
src/
├── ai_architecture/          # AI architecture components
│   ├── __init__.py
│   ├── agents.py            # Agent implementations
│   ├── lifecycle.py         # Model lifecycle management
│   └── platform.py          # Platform abstractions
├── gradio_app/              # Gradio web application
│   ├── __init__.py
│   ├── main.py             # Main application entry point
│   ├── components.py        # UI components
│   └── mcp_server.py       # MCP server integration
├── model_evaluation/        # Model evaluation framework
│   ├── __init__.py
│   ├── pipeline.py         # Evaluation pipeline
│   ├── bias_detection.py   # Bias detection algorithms
│   └── robustness.py       # Robustness testing
└── utils/                   # Utility functions
    ├── __init__.py
    ├── config_utils.py     # Configuration management
    ├── data_utils.py       # Data processing utilities
    └── logging_system.py   # Logging configuration
```

### Error Handling
```python
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def process_data(data: Dict) -> Optional[Dict]:
    """Process data with proper error handling."""
    try:
        # Validate input
        if not data:
            raise ValueError("Data cannot be empty")
        
        # Process data
        result = perform_processing(data)
        
        logger.info(f"Successfully processed data: {len(data)} items")
        return result
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing data: {e}")
        # Return None or raise depending on use case
        return None
```

### Logging Best Practices
```python
import logging

# Use module-level logger
logger = logging.getLogger(__name__)

# Different log levels
logger.debug("Detailed debugging information")
logger.info("General information about program execution")
logger.warning("Something unexpected happened")
logger.error("A serious error occurred")
logger.critical("A critical error occurred")

# Include context in log messages
logger.info(f"Processing user {user_id} request: {request_type}")
logger.error(f"Failed to load model {model_id}: {error_message}")
```

## Performance Guidelines

### Code Optimization
- Use appropriate data structures
- Avoid unnecessary computations
- Cache expensive operations
- Use async/await for I/O operations

### Memory Management
- Close file handles properly
- Use context managers
- Avoid memory leaks in long-running processes
- Monitor memory usage

## Security Guidelines

### Input Validation
```python
from typing import Any, Dict

def validate_input(data: Any) -> Dict:
    """Validate and sanitize input data."""
    if not isinstance(data, dict):
        raise ValueError("Input must be a dictionary")
    
    # Sanitize string inputs
    for key, value in data.items():
        if isinstance(value, str):
            data[key] = value.strip()
    
    return data
```

### Secret Management
- Never commit secrets to version control
- Use environment variables for configuration
- Implement proper authentication and authorization
- Validate all inputs and outputs

## Community Guidelines

### Code of Conduct
- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow
- Follow the project's code of conduct

### Getting Help
- Check existing documentation first
- Search existing issues and discussions
- Ask questions in GitHub discussions
- Join community channels if available

## Release Process

### Version Numbering
Follow semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes (backward compatible)

### Release Checklist
- [ ] Update version numbers
- [ ] Update CHANGELOG.md
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create release notes
- [ ] Tag release in Git
- [ ] Deploy to production (if applicable)

Thank you for contributing to the AI Assignments project! Your contributions help make this project better for everyone.
