#!/bin/bash
set -euo pipefail

echo "🧹 AI Platform Codebase Cleanup Script"
echo "======================================"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if we're in the project root
if [ ! -f "README.md" ] || [ ! -d "src" ]; then
    print_error "This script must be run from the project root directory"
    exit 1
fi

# Backup important files
print_step "Creating backup of important files..."
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -r .env* "$BACKUP_DIR/" 2>/dev/null || true
cp -r config/ "$BACKUP_DIR/" 2>/dev/null || true
print_success "Backup created in $BACKUP_DIR"

# Clean Python cache and compiled files
print_step "Cleaning Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
find . -type f -name "*.pyd" -delete 2>/dev/null || true
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
print_success "Python cache files cleaned"

# Clean JavaScript/Node files
print_step "Cleaning JavaScript/Node files..."
find . -type d -name "node_modules" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "package-lock.json" -delete 2>/dev/null || true
find . -type f -name "yarn.lock" -delete 2>/dev/null || true
find . -type d -name ".next" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
print_success "JavaScript/Node files cleaned"

# Clean logs and temporary files
print_step "Cleaning logs and temporary files..."
find logs/ -type f -name "*.log" -mtime +7 -delete 2>/dev/null || true
find . -type f -name "*.tmp" -delete 2>/dev/null || true
find . -type f -name "*.temp" -delete 2>/dev/null || true
find . -type f -name ".DS_Store" -delete 2>/dev/null || true
find . -type f -name "Thumbs.db" -delete 2>/dev/null || true
print_success "Logs and temporary files cleaned"

# Clean up Jupyter notebook checkpoints
print_step "Cleaning Jupyter notebook checkpoints..."
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
print_success "Jupyter checkpoints cleaned"

# Clean up Git ignored files (optional)
if [ "${CLEAN_GITIGNORE:-false}" = "true" ]; then
    print_step "Cleaning Git ignored files..."
    git clean -fdX
    print_success "Git ignored files cleaned"
else
    print_warning "Skipping Git ignored files (set CLEAN_GITIGNORE=true to clean)"
fi

# Remove duplicate files
if command -v fdupes &> /dev/null; then
    print_step "Checking for duplicate files..."
    DUPLICATES=$(fdupes -r src/ | grep -v "^$" | wc -l)
    if [ "$DUPLICATES" -gt 0 ]; then
        print_warning "Found $DUPLICATES duplicate files. Run 'fdupes -r -d src/' to remove them interactively"
    else
        print_success "No duplicate files found"
    fi
else
    print_warning "fdupes not installed. Install it to check for duplicate files"
fi

# Format Python code
if command -v black &> /dev/null; then
    print_step "Formatting Python code with Black..."
    black src/ --line-length 100 --quiet
    print_success "Python code formatted"
else
    print_warning "Black not installed. Run 'pip install black' to format Python code"
fi

# Sort Python imports
if command -v isort &> /dev/null; then
    print_step "Sorting Python imports with isort..."
    isort src/ --quiet
    print_success "Python imports sorted"
else
    print_warning "isort not installed. Run 'pip install isort' to sort imports"
fi

# Run Python linting
if command -v flake8 &> /dev/null; then
    print_step "Running flake8 linting..."
    flake8 src/ --max-line-length=100 --extend-ignore=E203,W503 --count --statistics || true
else
    print_warning "flake8 not installed. Run 'pip install flake8' to lint Python code"
fi

# Clean up and optimize requirements
print_step "Optimizing Python requirements..."
if [ -f "requirements.txt" ]; then
    # Remove comments and empty lines
    grep -v '^#' requirements.txt | grep -v '^$' > requirements_clean.txt
    mv requirements_clean.txt requirements.txt
    print_success "Requirements file cleaned"
else
    print_warning "No requirements.txt found"
fi

# Create/update .gitignore
print_step "Updating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv
pip-log.txt
pip-delete-this-directory.txt
.pytest_cache/
.mypy_cache/
.coverage
htmlcov/
.tox/
*.egg-info/
dist/
build/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~
.project
.pydevproject

# OS
.DS_Store
Thumbs.db
ehthumbs.db

# Logs
logs/
*.log

# Data
data/
mlruns/
*.db
*.sqlite
*.sqlite3

# Environment
.env
.env.*

# Node
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Documentation
site/
.mkdocs_cache/

# Temporary
*.tmp
*.temp
*.bak
*.cache

# Podman/Docker
.container-cache/
EOF
print_success ".gitignore updated"

# Generate directory structure documentation
print_step "Generating directory structure documentation..."
cat > STRUCTURE.md << 'EOF'
# Project Structure

```
ai_assignments/
├── .devcontainer/              # GitHub Codespaces configuration
│   ├── devcontainer.json       # Dev container configuration
│   ├── Containerfile           # Base container image
│   └── scripts/                # Setup and startup scripts
├── containers/                 # Service container definitions
│   ├── fastapi/                # FastAPI service container
│   ├── gradio/                 # Gradio app container
│   ├── mlflow/                 # MLflow tracking container
│   ├── chromadb/               # ChromaDB vector store container
│   ├── neo4j/                  # Neo4j graph database container
│   └── gateway/                # NGINX API gateway container
├── frontend/                   # GitHub Pages static site
│   ├── index.html              # Enhanced unified platform
│   ├── js/                     # JavaScript files
│   │   ├── api-client.js       # API client library
│   │   └── service-config.js   # Service configuration
│   └── css/                    # Stylesheets
├── src/                        # Source code
│   ├── enterprise_llmops/      # Enterprise LLMOps platform
│   ├── gradio_app/             # Gradio model evaluation app
│   ├── ai_architecture/        # AI architecture components
│   ├── model_evaluation/       # Model evaluation framework
│   └── utils/                  # Utility functions
├── scripts/                    # Utility scripts
│   ├── cleanup.sh              # This cleanup script
│   ├── build-containers.sh     # Container build script
│   └── start-services.sh       # Service startup script
├── config/                     # Configuration files
│   ├── requirements.txt        # Python dependencies
│   └── requirements-*.txt      # Environment-specific dependencies
├── docs/                       # MkDocs documentation
├── tests/                      # Test suites
└── data/                       # Data storage (git-ignored)
```
EOF
print_success "Directory structure documented in STRUCTURE.md"

# Summary
echo ""
echo "🎉 Cleanup Summary"
echo "=================="
echo ""

# Calculate space saved
SPACE_BEFORE=$(du -sh . 2>/dev/null | cut -f1)
echo "Space usage: $SPACE_BEFORE"

# Count files
PYTHON_FILES=$(find src/ -name "*.py" | wc -l)
echo "Python files: $PYTHON_FILES"

# Count lines of code
if command -v cloc &> /dev/null; then
    echo ""
    echo "Code statistics:"
    cloc src/ --quiet
else
    TOTAL_LINES=$(find src/ -name "*.py" -exec wc -l {} + | tail -1 | awk '{print $1}')
    echo "Total Python lines: $TOTAL_LINES"
fi

echo ""
print_success "Cleanup completed successfully!"
echo ""
echo "📝 Next steps:"
echo "  1. Review and commit changes: git add -A && git commit -m 'Cleanup codebase'"
echo "  2. Build containers: ./scripts/build-containers.sh"
echo "  3. Start services: .devcontainer/scripts/start-services.sh"
echo ""