# Complete AI Assignments Project - Server Commands & Workflows

## 🚀 Core Server Startup Commands

### 1. Virtual Environment Activation

```powershell
# Always activate first
& C:\Users\samne\PycharmProjects\ai_assignments\venv\Scripts\Activate.ps1
```

### 2. Essential Services (Start in Order)

```powershell
# Terminal 1: ChromaDB Vector Store
chroma run --host 0.0.0.0 --port 8081 --path chroma_data

# Terminal 2: MLflow Experiment Tracking
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

# Terminal 3: Enterprise LLMOps Platform (Main) - Demo Mode (No Auth)
python -m src.enterprise_llmops.main --host 0.0.0.0 --port 8080

# Terminal 4: Gradio Model Evaluation App
python -m src.gradio_app.main --host 0.0.0.0 --port 7860

# Terminal 5: LangGraph Studio (Optional - Agent Visualization & Debugging)
langgraph dev --host 0.0.0.0 --port 8083
```

## 🔐 Authentication & Token Management

### Demo Mode (Default - No Authentication Required)

The server runs in demo mode by default with authentication disabled for easy testing.

### Test API Endpoints (No Auth Required)

```powershell
# Test models endpoint
curl http://localhost:8080/api/models

# Test experiments endpoint
curl http://localhost:8080/api/experiments

# Test health endpoint
curl http://localhost:8080/health

# Test LangGraph Studio status
curl http://localhost:8080/api/langgraph/studios/status
```

### Production Mode (Enable Authentication)

```powershell
# Start server with authentication enabled
python -m src.enterprise_llmops.main --host 0.0.0.0 --port 8080 --enable-auth

# Get authentication token
.\scripts\get-auth-token.ps1

# Test with authentication (PowerShell format)
$headers = @{ 'Authorization' = 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTc1ODQ4NjUxN30.NNyvpouaVMUc7wqLCQP7k-YDOHcBS0bIJAg7Wdk-zk0' }
Invoke-RestMethod -Uri 'http://localhost:8080/api/models' -Headers $headers
```

## 📊 Model & Prompt Sync Workflows

### Basic Sync (Quick - Existing)

```powershell
# Basic sync with authentication (existing script)
.\scripts\basic-sync.ps1
```

### Comprehensive Sync (New - Recommended)

```powershell
# Comprehensive sync without authentication (demo mode)
.\scripts\comprehensive-sync.ps1

# Or run the Python version directly
python scripts\sync_registries.py
```

### Generate Evaluation Dataset

```powershell
python scripts\generate_evaluation_dataset.py
```

### Check Prompt Cache Status

```powershell
# Check cached AI tool prompts
ls cache\ai_tool_prompts\
```

### Manual API Sync (No Auth Required)

```powershell
# Sync prompts via API
curl -X POST http://localhost:8080/api/prompts/sync

# Get prompt cache summary
curl http://localhost:8080/api/prompts/cache/summary

# Get registry statistics
curl http://localhost:8080/api/prompts/registries/statistics
```

## 🔄 Comprehensive Registry Sync Features

### What the Comprehensive Sync Does

The comprehensive sync script (`.\scripts\comprehensive-sync.ps1`) performs a complete synchronization of all registries:

1. **🤖 Model Registry Sync**: Registers all Ollama models in the enterprise model registry
2. **📝 Prompt Registry Sync**: Syncs prompts with model capabilities and generates model-specific datasets
3. **🧪 Experiment Sync**: Creates baseline experiments for each available model
4. **🎯 Gradio Config Update**: Updates Gradio app configuration with available models
5. **📊 Dataset Generation**: Generates enhanced evaluation datasets
6. **🔍 Verification**: Tests all endpoints to ensure everything is working

### Sync Results

After running comprehensive sync, you'll have:

- ✅ All Ollama models registered in the model registry
- ✅ Model-specific prompt datasets generated
- ✅ Baseline experiments created for each model
- ✅ Gradio app configured with available models
- ✅ Enhanced evaluation dataset ready for testing

### Available Models (Your Current Setup)

Based on your Ollama installation:

- `llama3.1:8b` (4.9 GB) - General purpose LLM
- `codellama:7b` (3.8 GB) - Code generation specialist
- `mistral:7b` (4.4 GB) - General purpose LLM
- `gemma3:1b` (815 MB) - Lightweight general purpose LLM

## 🔧 Development & Testing Commands

### Run Tests

```powershell
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# All tests
pytest tests/ -v --cov=src/
```

### Code Quality Checks

```powershell
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/

# Security scan
bandit -r src/
```

## 📚 Documentation Commands

### Build Documentation

```powershell
cd docs/
mkdocs build
mkdocs serve --dev-addr 0.0.0.0:8082  # View at http://localhost:8082
```

### Generate API Docs

```powershell
# FastAPI auto-docs available at:
# http://localhost:8080/docs
# http://localhost:8080/redoc
```

## 🌐 Service URLs & Endpoints

### Core Services

- **Enterprise Platform**: http://localhost:8080
- **Gradio App**: http://localhost:7860
- **MLflow UI**: http://localhost:5000
- **ChromaDB**: http://localhost:8081
- **Documentation**: http://localhost:8082 (when mkdocs serve)
- **LangGraph Studio**: http://localhost:8083 (when running)

### API Endpoints (Demo Mode - No Auth Required)

- **Models**: http://localhost:8080/api/models
- **Experiments**: http://localhost:8080/api/experiments
- **Ollama Models**: http://localhost:8080/api/ollama/models
- **Prompt Cache**: http://localhost:8080/api/prompts/cache/summary
- **Health Check**: http://localhost:8080/health
- **API Documentation**: http://localhost:8080/docs
- **LangGraph Studio API**: http://localhost:8080/api/langgraph/studios/status

## 🔄 Complete Workflow Examples

### Workflow 1: Full Model Evaluation Setup (Demo Mode)

```powershell
# 1. Start all services (4 terminals)
# 2. Activate venv in each terminal
& C:\Users\samne\PycharmProjects\ai_assignments\venv\Scripts\Activate.ps1

# 3. Start services in order:
# Terminal 1: ChromaDB
chroma run --host 0.0.0.0 --port 8081 --path chroma_data

# Terminal 2: MLflow
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

# Terminal 3: Enterprise Platform (Demo Mode - No Auth)
python -m src.enterprise_llmops.main --host 0.0.0.0 --port 8080

# Terminal 4: Gradio App
python -m src.gradio_app.main --host 0.0.0.0 --port 7860

# Terminal 5: LangGraph Studio (Optional)
langgraph dev --host 0.0.0.0 --port 8083

# 4. Test API endpoints (no authentication required)
curl http://localhost:8080/health
curl http://localhost:8080/api/models
curl http://localhost:8080/api/experiments
curl http://localhost:8080/api/langgraph/studios/status

# 5. Comprehensive sync (recommended)
.\scripts\comprehensive-sync.ps1

# OR use basic sync (existing)
# .\scripts\basic-sync.ps1

# 7. Access services
# - Enterprise Platform: http://localhost:8080/docs
# - Model Evaluation: http://localhost:7860
# - MLflow Tracking: http://localhost:5000
# - LangGraph Studio: http://localhost:8083
```

### Workflow 2: Development & Testing (Demo Mode)

```powershell
# 1. Start services (demo mode - no auth)
python -m src.enterprise_llmops.main --host 0.0.0.0 --port 8080

# 2. Test API endpoints
curl http://localhost:8080/health
curl http://localhost:8080/api/models

# 3. Run tests
pytest tests/ -v

# 4. Check code quality
black src/ && flake8 src/ && mypy src/

# 5. Build docs
cd docs/ && mkdocs build
```

### Workflow 3: Prompt Registry Management (Demo Mode)

```powershell
# 1. Check cached prompts
ls cache\ai_tool_prompts\

# 2. Comprehensive sync (recommended)
.\scripts\comprehensive-sync.ps1

# OR manual sync steps:
# 2a. Load prompt statistics (no auth required)
curl http://localhost:8080/api/prompts/cache/summary

# 2b. Sync prompts (no auth required)
curl -X POST http://localhost:8080/api/prompts/sync

# 2c. Generate dataset
python scripts\generate_evaluation_dataset.py
```

## 🛠️ Troubleshooting Commands

### Check Service Status

```powershell
# Check if ports are in use
netstat -an | findstr ":8080"
netstat -an | findstr ":7860"
netstat -an | findstr ":5000"
netstat -an | findstr ":8081"
netstat -an | findstr ":8083"

# Check Python processes
tasklist /fi "imagename eq python.exe"
```

### Reset Services

```powershell
# Stop all Python processes
taskkill /f /im python.exe

# Clear MLflow database
del mlflow.db

# Clear ChromaDB data
rmdir /s chroma_data
```

### Log Analysis

```powershell
# Check application logs
type logs\llmops.log

# Real-time log monitoring
Get-Content logs\llmops.log -Wait
```

## 📁 Key Directories & Files

### Project Structure

```
ai_assignments/
├── src/                    # Source code
├── tests/                  # Test suites
├── docs/                   # Documentation
├── scripts/               # Automation scripts
├── cache/                 # Cached data
│   └── ai_tool_prompts/   # Cached AI prompts
├── data/                  # Generated data
│   └── evaluation_datasets/ # Evaluation datasets
└── logs/                  # Application logs
```

### Important Files

- `config/requirements.txt` - Dependencies
- `scripts/basic-sync.ps1` - Basic model sync script (existing)
- `scripts/comprehensive-sync.ps1` - Comprehensive registry sync (new)
- `scripts/sync_registries.py` - Python registry sync script (new)
- `scripts/generate_evaluation_dataset.py` - Dataset generation
- `config/gradio_models.json` - Gradio app model configuration (generated)
- `data/evaluation_datasets/enhanced_evaluation_dataset.csv` - Generated dataset

## 🎯 Quick Start Checklist

1. ✅ Activate virtual environment
2. ✅ Start ChromaDB (port 8081)
3. ✅ Start MLflow (port 5000)
4. ✅ Start Enterprise Platform (port 8080)
5. ✅ Start Gradio App (port 7860)
6. ✅ Start LangGraph Studio (port 8083) - Optional
7. ✅ Run comprehensive sync script (recommended)
   - OR run basic sync script (existing)
8. ✅ Test API endpoints (no auth required in demo mode)
9. ✅ Access services via URLs
10. ✅ Use interactive landing page for testing

## 🔗 Integration Points

### Model Evaluation Flow

```
Ollama Models → Enterprise Registry → MLflow Tracking → Evaluation Dataset → Gradio App
```

### Prompt Management Flow

```
AI Tool Cache → Prompt Registry → Enhanced Dataset → Model Evaluation → MLflow Experiments
```

### Development Flow

```
Code Changes → Tests → Quality Checks → Documentation → Deployment
```

## 🚀 Advanced Commands

### Ollama Model Management

```powershell
# List Ollama models
ollama list

# Pull new model
ollama pull llama3.1:8b

# Check model status
curl http://localhost:11434/api/tags
```

### MLflow Advanced Operations

```powershell
# List experiments
mlflow experiments list

# Create new experiment
mlflow experiments create --experiment-name "custom_experiment"

# Run tracking
mlflow run . --experiment-name "llmops_enterprise"
```

### ChromaDB Operations

```powershell
# Check ChromaDB status
curl http://localhost:8081/api/v1/heartbeat

# List collections
curl http://localhost:8081/api/v1/collections
```

### LangGraph Studio Operations

```powershell
# Install LangGraph CLI (if not already installed)
pip install langgraph-cli

# Start LangGraph Studio
langgraph dev --host 0.0.0.0 --port 8083

# Check LangGraph Studio status via API
curl http://localhost:8080/api/langgraph/studios/status

# Get studio information
curl http://localhost:8080/api/langgraph/studios/info

# Create a new studio session
curl -X POST http://localhost:8080/api/langgraph/studios/sessions \
  -H "Content-Type: application/json" \
  -d '{"mode": "graph", "metadata": {"project": "ai-architecture"}}'

# Get available assistants
curl http://localhost:8080/api/langgraph/studios/assistants

# Get available threads
curl http://localhost:8080/api/langgraph/studios/threads

# Access LangGraph Studio dashboard
# Open http://localhost:8083 in browser
```

## 🔧 Configuration Management

### Environment Variables

```powershell
# Set environment variables
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:CHROMA_SERVER_HOST = "localhost"
$env:CHROMA_SERVER_HTTP_PORT = "8081"
```

### Configuration Files

- `config/frontend_config.yaml` - Frontend configuration
- `config/pytest.ini` - Test configuration
- `docs/mkdocs.yml` - Documentation configuration

## 📊 Monitoring & Metrics

### Service Health Checks

```powershell
# Enterprise Platform
curl http://localhost:8080/health

# MLflow
curl http://localhost:5000/health

# ChromaDB
curl http://localhost:8081/api/v1/heartbeat

# Gradio App
curl http://localhost:7860/health

# LangGraph Studio
curl http://localhost:8080/api/langgraph/studios/status
```

### Performance Monitoring

```powershell
# Check CPU usage
Get-Process python | Select-Object ProcessName,CPU

# Check memory usage
Get-Process python | Select-Object ProcessName,WorkingSet

# Check disk usage
Get-ChildItem . -Recurse | Measure-Object -Property Length -Sum
```

## 🎯 Production Deployment Commands

### Production Mode with Authentication

```powershell
# Start server with authentication enabled
python -m src.enterprise_llmops.main --host 0.0.0.0 --port 8080 --enable-auth

# Get authentication token
.\scripts\get-auth-token.ps1

# Test with authentication (PowerShell format)
$headers = @{ 'Authorization' = 'Bearer YOUR_TOKEN_HERE' }
Invoke-RestMethod -Uri 'http://localhost:8080/api/models' -Headers $headers

# Or using curl with proper PowerShell syntax
$token = "YOUR_TOKEN_HERE"
curl -Headers @{Authorization="Bearer $token"} http://localhost:8080/api/models
```

### Docker Commands (if using Docker)

```powershell
# Build images
docker build -t enterprise-llmops .

# Run containers
docker-compose up -d

# Check container status
docker ps
```

### Service Management

```powershell
# Install as Windows Service (if needed)
sc create "EnterpriseLLMOps" binPath="python -m src.enterprise_llmops.main --enable-auth"

# Start service
sc start "EnterpriseLLMOps"

# Stop service
sc stop "EnterpriseLLMOps"
```
