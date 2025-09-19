# Setup Guide - Lenovo AAITC Solutions

## Overview

This guide provides step-by-step instructions for setting up the Lenovo AAITC Solutions development environment, including Python dependencies, virtual environment, and documentation system.

## Prerequisites

- **Python 3.8+**: Required for the framework
- **PowerShell**: For Windows development commands
- **Git**: For version control
- **Node.js** (optional): For frontend development tools

## Quick Start

=== "Windows (PowerShell)"

    ```powershell
    # Clone repository
    git clone https://github.com/s-n00b/ai_assignments.git
    cd ai_assignments

    # Activate virtual environment
    .\venv\Scripts\Activate.ps1

    # Install dependencies
    pip install -r config\requirements.txt
    pip install -r config\requirements-testing.txt

    # Install documentation dependencies
    pip install -r docs\requirements-docs.txt

    # Run tests
    python -m pytest tests\ -v

    # Launch application
    python -m src.gradio_app.main
    ```

=== "Linux/macOS"

    ```bash
    # Clone repository
    git clone https://github.com/s-n00b/ai_assignments.git
    cd ai_assignments

    # Create and activate virtual environment
    python -m venv venv
    source venv/bin/activate

    # Install dependencies
    pip install -r config/requirements.txt
    pip install -r config/requirements-testing.txt

    # Install documentation dependencies
    pip install -r docs/requirements-docs.txt

    # Run tests
    python -m pytest tests/ -v

    # Launch application
    python -m src.gradio_app.main
    ```

## Detailed Setup Instructions

### 1. Environment Setup

#### Virtual Environment

```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Verify activation
python --version
pip --version
```

#### Dependencies Installation

```powershell
# Core dependencies
pip install -r config\requirements.txt

# Testing dependencies
pip install -r config\requirements-testing.txt

# Documentation dependencies
pip install -r docs\requirements-docs.txt

# Development tools (optional)
pip install black isort flake8 mypy
```

### 2. Configuration Setup

#### Environment Variables

Create a `.env` file in the project root:

```bash
# API Keys (add your actual keys)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
META_API_KEY=your_meta_api_key_here

# Application Configuration
GRADIO_HOST=0.0.0.0
GRADIO_PORT=7860
MCP_SERVER_PORT=8000

# Logging Configuration
LOG_LEVEL=INFO
LOG_DIR=./logs

# Cache Configuration
CACHE_DIR=./cache
ENABLE_CACHING=true
```

#### Model Configuration

The framework includes pre-configured model settings in `src/model_evaluation/config.py`:

```python
LATEST_MODEL_CONFIGS = {
    "gpt-5": ModelConfig(
        name="GPT-5",
        provider="openai",
        model_id="gpt-5",
        max_tokens=4000,
        temperature=0.7,
        context_window=128000,
        parameters=175,  # billion parameters
        capabilities=["text_generation", "reasoning", "multimodal"]
    ),
    # ... other models
}
```

### 3. Documentation Setup

#### MkDocs Installation

```powershell
# Install MkDocs with Material theme
pip install -r docs\requirements-docs.txt

# Verify installation
mkdocs --version
```

#### Serve Documentation

```powershell
# Navigate to docs directory
cd docs

# Clean up Jekyll files (one-time)
.\cleanup-jekyll.ps1

# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build

# Deploy to GitHub Pages
mkdocs gh-deploy
```

### 4. Testing Setup

#### Run Test Suite

```powershell
# Run all tests
python -m pytest tests\ -v

# Run specific test categories
python -m pytest tests\unit\ -v
python -m pytest tests\integration\ -v
python -m pytest tests\e2e\ -v --timeout=600

# Run with coverage
python -m pytest tests\ --cov=src --cov-report=html --cov-report=term-missing
```

#### Test Configuration

The project uses `config/pytest.ini` for test configuration:

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short --strict-markers --disable-warnings
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow running tests
    api: API tests
    performance: Performance tests
```

### 5. Application Launch

#### Gradio Application

```powershell
# Basic launch
python -m src.gradio_app.main

# Launch with MCP server
python -m src.gradio_app.main --mcp-server

# Launch with custom host/port
python -m src.gradio_app.main --host 0.0.0.0 --port 7860

# Launch with specific configuration
python -m src.gradio_app.main --config config/production.yaml
```

#### MCP Server Only

```powershell
# Launch MCP server
python -m src.gradio_app.mcp_server

# Launch with custom configuration
python -m src.gradio_app.mcp_server --host 0.0.0.0 --port 8000
```

### 6. Development Tools

#### Code Quality

```powershell
# Format code
black src\ tests\
isort src\ tests\

# Lint code
flake8 src\ tests\ --count --select=E9,F63,F7,F82 --show-source --statistics

# Type checking
mypy src\ --ignore-missing-imports
```

#### Git Hooks (Optional)

```powershell
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### 7. Troubleshooting

#### Common Issues

**1. Import Errors**

```powershell
# Add project root to Python path
$env:PYTHONPATH = "$PWD;$env:PYTHONPATH"

# Or use relative imports
python -m src.gradio_app.main
```

**2. Virtual Environment Issues**

```powershell
# Recreate virtual environment
Remove-Item -Recurse -Force venv
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r config\requirements.txt
```

**3. Permission Issues (Windows)**

```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**4. Port Already in Use**

```powershell
# Find process using port
netstat -ano | findstr :7860

# Kill process (replace PID with actual process ID)
taskkill /PID <PID> /F
```

#### Log Files

Check log files for detailed error information:

```powershell
# View recent logs
Get-Content logs\application.log -Tail 50

# View error logs
Get-Content logs\error.log -Tail 20
```

### 8. Production Deployment

#### Docker Deployment

```dockerfile
# Dockerfile example
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -r config/requirements.txt
RUN pip install -r docs/requirements-docs.txt

EXPOSE 7860 8000

CMD ["python", "-m", "src.gradio_app.main", "--mcp-server"]
```

#### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lenovo-aaitc-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: lenovo-aaitc-app
  template:
    metadata:
      labels:
        app: lenovo-aaitc-app
    spec:
      containers:
        - name: app
          image: lenovo-aaitc:latest
          ports:
            - containerPort: 7860
            - containerPort: 8000
          env:
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: api-keys
                  key: openai-key
```

### 9. Development Workflow

#### Daily Development

1. **Start Development Environment**

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. **Run Tests Before Changes**

   ```powershell
   python -m pytest tests\unit\ -v
   ```

3. **Make Changes and Test**

   ```powershell
   python -m pytest tests\ -v
   ```

4. **Format and Lint Code**

   ```powershell
   black src\ tests\
   flake8 src\ tests\
   ```

5. **Update Documentation**
   ```powershell
   cd docs
   mkdocs serve
   ```

#### Git Workflow

```powershell
# Create feature branch
git checkout -b feature/new-feature

# Make changes and commit
git add .
git commit -m "Add new feature"

# Push and create PR
git push origin feature/new-feature
```

### 10. Performance Optimization

#### Development Mode

```powershell
# Enable development mode
$env:FLASK_ENV = "development"
$env:DEBUG = "true"

# Run with hot reload
python -m src.gradio_app.main --reload
```

#### Production Mode

```powershell
# Enable production mode
$env:FLASK_ENV = "production"
$env:DEBUG = "false"

# Run with optimizations
python -m src.gradio_app.main --workers 4
```

---

**Setup Guide - Lenovo AAITC Solutions**  
_Complete development environment setup instructions_
