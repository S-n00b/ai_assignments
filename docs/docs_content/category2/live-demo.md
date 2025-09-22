# Live Demo

## 🎯 Overview

Interactive live demonstrations showcasing the complete Lenovo AAITC platform capabilities and enterprise features.

## 🚀 Demo Access Points

### Main Enterprise Platform

- **URL**: http://localhost:8080
- **Features**: Unified dashboard with all services
- **Capabilities**: Service integration, monitoring, management

### Model Evaluation Interface

- **URL**: http://localhost:7860
- **Features**: Interactive model testing and evaluation
- **Capabilities**: Model selection, task execution, result analysis

### Documentation Site

- **URL**: http://localhost:8082
- **Features**: Complete documentation with diagrams
- **Capabilities**: Interactive documentation, search, navigation

## 📊 Demo Scenarios

### Scenario 1: Complete Model Evaluation Workflow

1. **Access Platform** - Navigate to FastAPI enterprise platform
2. **Model Selection** - Choose from available models
3. **Task Configuration** - Set up evaluation parameters
4. **Execution** - Run comprehensive evaluation
5. **Analysis** - Review results and metrics
6. **Factory Roster** - Add to production roster

### Scenario 2: Enterprise Service Integration

1. **Service Overview** - Explore integrated services
2. **iframe Integration** - Test embedded services
3. **Cross-Service Communication** - Validate data flow
4. **Monitoring** - Check system health and metrics
5. **Management** - Configure and optimize services

### Scenario 3: Advanced AI Features

1. **QLoRA Fine-Tuning** - Custom model adaptation
2. **LangGraph Studio** - Agent workflow visualization
3. **Neo4j GraphRAG** - Knowledge graph exploration
4. **Faker Data Generation** - Realistic data scenarios

## 🔧 Demo Setup

### Prerequisites

- Virtual environment activated
- All services running
- Sample data loaded
- Demo models available

### Quick Start Commands

```bash
# Terminal 1: ChromaDB
chroma run --host 0.0.0.0 --port 8081 --path chroma_data

# Terminal 2: MLflow
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

# Terminal 3: Enterprise Platform
python -m src.enterprise_llmops.main --host 0.0.0.0 --port 8080

# Terminal 4: Gradio App
python -m src.gradio_app.main --host 0.0.0.0 --port 7860

# Terminal 5: Documentation
cd docs && mkdocs serve --dev-addr 0.0.0.0:8082
```

---

**Last Updated**: January 19, 2025  
**Version**: 2.1.0  
**Status**: Production Ready  
**Integration**: Live Demo System
