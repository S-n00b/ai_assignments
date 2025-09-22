# Live Demo

## 🎯 Overview

Interactive live demonstrations of the Lenovo AAITC platform capabilities.

## 🚀 Demo Access

### Main Platform

- **URL**: http://localhost:8080
- **Features**: Unified dashboard with all services

### Model Evaluation

- **URL**: http://localhost:7860
- **Features**: Interactive model testing interface

### Documentation

- **URL**: http://localhost:8082
- **Features**: Complete documentation site

## 📊 Demo Scenarios

### Scenario 1: Model Evaluation

1. Access Gradio interface
2. Select model and task
3. Run evaluation
4. Review results

### Scenario 2: Enterprise Platform

1. Access FastAPI platform
2. Explore service integration
3. Test iframe embedding
4. Monitor system status

### Scenario 3: Knowledge Graph

1. Access Neo4j interface
2. Explore graph data
3. Test RAG capabilities
4. Visualize relationships

## 🔧 Demo Configuration

### Prerequisites

- All services running
- Sample data loaded
- Demo models available

### Setup Commands

```bash
# Start all services
python -m src.enterprise_llmops.main --host 0.0.0.0 --port 8080
python -m src.gradio_app.main --host 0.0.0.0 --port 7860
cd docs && mkdocs serve --dev-addr 0.0.0.0:8082
```

---

**Last Updated**: January 19, 2025  
**Version**: 2.1.0  
**Status**: Production Ready  
**Integration**: Live Demo System
