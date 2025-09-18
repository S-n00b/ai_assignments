# =============================================================================
# AI Assignments - Development Commands (PowerShell)
# =============================================================================
# Comprehensive PowerShell command reference for the refactored project structure
# Updated for: ai_assignments/ (root) with src/, tests/, docs/, config/, scripts/

# =============================================================================
# PROJECT NAVIGATION & SETUP
# =============================================================================

# Change to project root directory:
cd C:\Users\samne\PycharmProjects\ai_assignments

# Activate virtual environment (from project root):
.\venv\Scripts\Activate.ps1

# Deactivate virtual environment:
deactivate

# Check current directory and Python environment:
pwd
python --version
pip --version

# =============================================================================
# DEPENDENCY MANAGEMENT
# =============================================================================

# Install all dependencies:
pip install -r config\requirements.txt

# Install testing dependencies:
pip install -r config\requirements-testing.txt

# Install development dependencies:
pip install -r config\requirements.txt
pip install black isort flake8 mypy pre-commit

# Update all packages:
pip list --outdated
pip install --upgrade pip

# Check for security vulnerabilities:
pip install safety
safety check

# =============================================================================
# TESTING COMMANDS
# =============================================================================

# Run all tests:
python -m pytest tests\ -v --tb=short

# Run tests with coverage:
python -m pytest tests\ -v --cov=src --cov-report=html --cov-report=term-missing

# Run specific test categories:
python -m pytest tests\unit\ -v --tb=short
python -m pytest tests\integration\ -v --tb=short
python -m pytest tests\e2e\ -v --tb=short --timeout=600

# Run tests with specific markers:
python -m pytest tests\ -v -m "not slow"
python -m pytest tests\ -v -m "api"
python -m pytest tests\ -v -m "performance"

# Run tests in parallel:
python -m pytest tests\ -v -n auto

# Run specific test files:
python -m pytest tests\unit\test_model_evaluation.py -v
python -m pytest tests\unit\test_ai_architecture.py -v
python -m pytest tests\unit\test_gradio_app.py -v
python -m pytest tests\unit\test_utils.py -v

# Run integration tests:
python -m pytest tests\integration\test_model_evaluation_integration.py -v
python -m pytest tests\integration\test_ai_architecture_integration.py -v
python -m pytest tests\integration\test_gradio_integration.py -v

# Run end-to-end tests:
python -m pytest tests\e2e\test_complete_workflows.py -v
python -m pytest tests\e2e\test_user_scenarios.py -v

# Generate test coverage report:
python -m pytest tests\ --cov=src --cov-report=html
Start-Process htmlcov\index.html

# =============================================================================
# CODE QUALITY & FORMATTING
# =============================================================================

# Format code with Black:
black src\ tests\

# Sort imports with isort:
isort src\ tests\

# Run linting with flake8:
flake8 src\ tests\ --count --select=E9,F63,F7,F82 --show-source --statistics

# Run type checking with mypy:
mypy src\ --ignore-missing-imports

# Run all code quality checks:
black --check src\ tests\
isort --check-only src\ tests\
flake8 src\ tests\
mypy src\ --ignore-missing-imports

# =============================================================================
# SECURITY & VULNERABILITY SCANNING
# =============================================================================

# Run security scan with bandit:
bandit -r src\ -f json -o bandit-report.json

# Check for known security vulnerabilities:
safety check --json --output safety-report.json

# Run comprehensive security audit:
bandit -r src\ -f json -o bandit-report.json
safety check --json --output safety-report.json

# =============================================================================
# APPLICATION LAUNCH & DEVELOPMENT
# =============================================================================

# Launch Gradio application:
python -m src.gradio_app.main

# Launch with specific host and port:
python -m src.gradio_app.main --host 0.0.0.0 --port 7860

# Launch with MCP server:
python -m src.gradio_app.main --mcp-server

# Launch in development mode with auto-reload:
python -m src.gradio_app.main --reload

# =============================================================================
# PROJECT STRUCTURE & EXPLORATION
# =============================================================================

# View project structure:
tree /f

# List all Python files:
Get-ChildItem -Recurse -Filter "*.py" | Select-Object FullName

# Count lines of code:
Get-ChildItem -Recurse -Filter "*.py" | Get-Content | Measure-Object -Line

# Find files by pattern:
Get-ChildItem -Recurse -Filter "*test*.py"
Get-ChildItem -Recurse -Filter "*config*.py"

# Search for specific text in files:
Select-String -Path "src\*.py" -Pattern "class.*Manager"
Select-String -Path "tests\*.py" -Pattern "def test_"

# =============================================================================
# GIT & VERSION CONTROL
# =============================================================================

# Check git status:
git status

# Add all changes:
git add .

# Commit changes:
git commit -m "Update: Refactor project structure"

# Push to remote:
git push origin main

# Create new branch:
git checkout -b feature/new-feature

# Switch branches:
git checkout main
git checkout feature/new-feature

# View git log:
git log --oneline -10

# =============================================================================
# VIRTUAL ENVIRONMENT MANAGEMENT
# =============================================================================

# Create new virtual environment:
python -m venv venv

# Activate virtual environment:
.\venv\Scripts\Activate.ps1

# Install package in development mode:
pip install -e .

# Export requirements:
pip freeze > config\requirements-current.txt

# =============================================================================
# DOCUMENTATION & HELP
# =============================================================================

# Generate documentation:
python -m pydoc src.model_evaluation
python -m pydoc src.ai_architecture
python -m pydoc src.gradio_app

# View help for specific modules:
python -c "import src.model_evaluation; help(src.model_evaluation)"

# =============================================================================
# PERFORMANCE & MONITORING
# =============================================================================

# Run performance benchmarks:
python -m pytest tests\unit\ --benchmark-only --benchmark-save=baseline

# Profile application:
python -m cProfile -o profile_output.prof -m src.gradio_app.main

# Monitor memory usage:
python -m memory_profiler src\gradio_app\main.py

# =============================================================================
# CLEANUP & MAINTENANCE
# =============================================================================

# Clean Python cache files:
Get-ChildItem -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force

# Clean test artifacts:
Remove-Item -Recurse -Force .pytest_cache\
Remove-Item -Recurse -Force .coverage
Remove-Item -Recurse -Force htmlcov\
Remove-Item -Recurse -Force .benchmarks\

# Clean build artifacts:
Remove-Item -Recurse -Force dist\
Remove-Item -Recurse -Force build\
Remove-Item -Recurse -Force *.egg-info\

# Clean all generated files:
Get-ChildItem -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force
Remove-Item -Recurse -Force .pytest_cache\
Remove-Item -Recurse -Force .coverage
Remove-Item -Recurse -Force htmlcov\
Remove-Item -Recurse -Force .benchmarks\

# =============================================================================
# QUICK DEVELOPMENT WORKFLOW
# =============================================================================

# Complete development setup (run once):
# cd C:\Users\samne\PycharmProjects\ai_assignments
# .\venv\Scripts\Activate.ps1
# pip install -r config\requirements.txt
# pip install -r config\requirements-testing.txt

# Daily development workflow:
# .\venv\Scripts\Activate.ps1
# python -m pytest tests\unit\ -v --tb=short
# black src\ tests\
# isort src\ tests\
# python -m src.gradio_app.main

# Pre-commit workflow:
# git add .
# git commit -m "Your commit message"

# =============================================================================
# TROUBLESHOOTING
# =============================================================================

# Check Python path:
python -c "import sys; print('\n'.join(sys.path))"

# Check installed packages:
pip list

# Check for import errors:
python -c "import src.model_evaluation; print('Model evaluation imports OK')"
python -c "import src.ai_architecture; print('AI architecture imports OK')"
python -c "import src.gradio_app; print('Gradio app imports OK')"

# Check test discovery:
python -m pytest --collect-only tests\

# =============================================================================
# USEFUL ALIASES (Add to PowerShell Profile)
# =============================================================================

# To add these aliases to your PowerShell profile, run:
# notepad $PROFILE

# Then add these lines:
# Set-Alias -Name "ai-test" -Value "python -m pytest tests\ -v --tb=short"
# Set-Alias -Name "ai-format" -Value "black src\ tests\ && isort src\ tests\"
# Set-Alias -Name "ai-lint" -Value "flake8 src\ tests\"
# Set-Alias -Name "ai-run" -Value "python -m src.gradio_app.main"
# Set-Alias -Name "ai-clean" -Value "Get-ChildItem -Recurse -Filter '__pycache__' | Remove-Item -Recurse -Force"

# =============================================================================
# NOTES
# =============================================================================

# 1. Always activate virtual environment before running commands
# 2. Use relative paths from project root (ai_assignments/)
# 3. Configuration files are in config/ directory
# 4. Source code is in src/ directory
# 5. Tests are in tests/ directory
# 6. Documentation is in docs/ directory
# 7. Use PowerShell ISE or VS Code for better development experience
