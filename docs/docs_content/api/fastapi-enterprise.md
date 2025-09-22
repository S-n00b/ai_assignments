# FastAPI Enterprise Platform Documentation

## 🚀 Overview

The Enterprise LLMOps Platform is a comprehensive FastAPI-based backend that provides enterprise-grade AI operations capabilities for Lenovo AAITC assignments. This platform serves as the foundation for Assignment 2 and integrates with all enterprise components.

## 📊 Key Features

### Core Capabilities

- **Enterprise API Endpoints**: RESTful API with comprehensive model management
- **Real-time Monitoring**: WebSocket connections for live system updates
- **Model Registry**: Centralized model lifecycle management
- **Experiment Tracking**: MLflow integration for reproducible ML workflows
- **AutoML**: Optuna hyperparameter optimization
- **Vector Databases**: Chroma, Weaviate, Pinecone integration
- **Graph Database**: Neo4j service integration with GraphRAG capabilities
- **Authentication**: JWT-based security with demo mode support

### Service Integration

- **Ollama Manager**: Local LLM serving and management
- **MLflow Manager**: Experiment tracking and model registry
- **Optuna Optimizer**: Automated hyperparameter tuning
- **Prompt Integration**: AI tool prompt management and caching
- **ChromaDB Client**: Vector database operations
- **Neo4j Service**: Graph database operations and GraphRAG queries

## 🌐 API Endpoints

### Core Endpoints

#### Health & Status

- `GET /health` - System health check
- `GET /api/status` - Detailed component status
- `GET /api/info` - Comprehensive API information

#### Model Management

- `GET /api/models` - List registered models
- `POST /api/models/register` - Register new model
- `PUT /api/models/{model_name}/versions/{version}/status` - Update model status

#### Ollama Integration

- `GET /api/ollama/models` - List Ollama models
- `POST /api/ollama/models/{model_name}/pull` - Pull new model
- `POST /api/ollama/generate` - Generate response using Ollama

#### Experiment Tracking

- `GET /api/experiments` - List MLflow experiments
- `POST /api/experiments/start` - Start new experiment
- `POST /api/experiments/{run_id}/log-metrics` - Log metrics
- `POST /api/experiments/{run_id}/log-params` - Log parameters
- `POST /api/experiments/{run_id}/end` - End experiment

#### AutoML & Optimization

- `POST /api/optimization/start` - Start hyperparameter optimization

#### Monitoring & Metrics

- `GET /api/monitoring/status` - System monitoring status
- `GET /api/monitoring/metrics` - Monitoring metrics

#### Vector Databases

- `GET /api/chromadb/health` - ChromaDB health check
- `GET /api/chromadb/collections` - List ChromaDB collections
- `POST /api/chromadb/collections` - Create new collection
- `GET /api/chromadb/collections/{collection_name}` - Get collection details
- `POST /api/chromadb/collections/{collection_name}/add` - Add documents to collection
- `POST /api/chromadb/collections/{collection_name}/query` - Query collection

#### Knowledge Graph

- `GET /api/knowledge-graph/status` - Neo4j status
- `GET /api/knowledge-graph/query` - Query knowledge graph

#### LangGraph Studio

- `GET /api/langgraph/studios` - List LangGraph Studio instances

#### Prompt Integration

- `GET /api/prompts/cache/summary` - Prompt cache summary
- `POST /api/prompts/sync` - Sync AI tool prompts
- `POST /api/prompts/dataset/generate` - Generate evaluation dataset
- `GET /api/prompts/registries/statistics` - Registry statistics

#### Neo4j Graph Database

- `GET /api/neo4j/health` - Neo4j service health status
- `GET /api/neo4j/info` - Database information and statistics
- `POST /api/neo4j/query` - Execute custom Cypher queries
- `POST /api/neo4j/graphrag` - GraphRAG semantic search
- `GET /api/neo4j/org-structure` - Lenovo organizational data
- `GET /api/neo4j/b2b-clients` - B2B client relationships
- `GET /api/neo4j/project-dependencies` - Project network analysis
- `GET /api/neo4j/employees` - Employee information
- `GET /api/neo4j/departments` - Department data
- `GET /api/neo4j/projects` - Project information
- `GET /api/neo4j/skills` - Skills and certifications
- `GET /api/neo4j/analytics/*` - Analytics endpoints

### WebSocket Endpoints

- `WS /ws` - Real-time updates and monitoring

## 🔐 Authentication

### Demo Mode (Default)

Authentication is disabled by default for easy testing and development:

```bash
python -m src.enterprise_llmops.main --host 0.0.0.0 --port 8080
```

### Production Mode

Enable authentication for production deployments:

```bash
python -m src.enterprise_llmops.main --host 0.0.0.0 --port 8080 --enable-auth
```

### Authentication Endpoints

- `POST /api/auth/login` - User login (returns JWT token)

## 📚 API Documentation

### Interactive Documentation

- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc
- **OpenAPI Schema**: http://localhost:8080/openapi.json

### API Information

- **API Info**: http://localhost:8080/api/info
- **System Status**: http://localhost:8080/api/status

## 🚀 Quick Start

### 1. Start the Enterprise Platform

```bash
# Activate virtual environment
& C:\Users\samne\PycharmProjects\ai_assignments\venv\Scripts\Activate.ps1

# Start the platform
python -m src.enterprise_llmops.main --host 0.0.0.0 --port 8080
```

### 2. Test API Endpoints

```bash
# Health check
curl http://localhost:8080/health

# API information
curl http://localhost:8080/api/info

# System status
curl http://localhost:8080/api/status
```

### 3. Access Documentation

- **Interactive Docs**: http://localhost:8080/docs
- **API Information**: http://localhost:8080/api/info

## 🔧 Configuration

### Command Line Options

```bash
python -m src.enterprise_llmops.main [options]

Options:
  --host HOST              Host to bind to (default: 0.0.0.0)
  --port PORT              Port to bind to (default: 8080)
  --workers WORKERS        Number of worker processes (default: 1)
  --config CONFIG          Path to configuration file
  --log-level LEVEL        Logging level (default: info)
  --enable-gpu             Enable GPU support
  --enable-monitoring      Enable monitoring stack
  --enable-automl          Enable AutoML features
  --disable-ollama         Disable Ollama integration
  --disable-mlflow         Disable MLflow integration
  --disable-model-registry Disable model registry
  --enable-auth            Enable authentication
  --minimal                Start with minimal configuration
```

### Configuration File

The platform supports YAML configuration files:

```yaml
# config/enterprise-config.yaml
host: "0.0.0.0"
port: 8080
workers: 1
log_level: "info"
enable_ollama: true
enable_model_registry: true
enable_mlflow: true
enable_automl: true
enable_monitoring: true
enable_gpu: false
mlflow_tracking_uri: "http://localhost:5000"
optuna_n_trials: 100
optuna_pruning: true
vector_databases:
  chroma:
    enabled: true
    url: "http://localhost:8080"
  weaviate:
    enabled: true
    url: "http://localhost:8080"
  pinecone:
    enabled: false
    api_key: null
monitoring:
  prometheus:
    enabled: true
    url: "http://localhost:9090"
  grafana:
    enabled: true
    url: "http://localhost:3000"
  langfuse:
    enabled: true
    url: "http://localhost:3000"
integrations:
  langgraph_studio:
    enabled: true
    url: "http://localhost:8080/api/langgraph/studios"
    studio_port: 8083
  neo4j:
    enabled: true
    url: "http://localhost:7474"
```

## 🔄 Service Integration

### External Services

The platform integrates with several external services:

| Service                | Port | URL                             | Description                  |
| ---------------------- | ---- | ------------------------------- | ---------------------------- |
| **Enterprise FastAPI** | 8080 | http://localhost:8080           | Main enterprise platform     |
| **Gradio App**         | 7860 | http://localhost:7860           | Model evaluation interface   |
| **MLflow Tracking**    | 5000 | http://localhost:5000           | Experiment tracking          |
| **ChromaDB**           | 8081 | http://localhost:8081           | Vector database              |
| **MkDocs**             | 8082 | http://localhost:8082           | Master documentation site    |
| **Weaviate**           | 8083 | http://localhost:8083           | Alternative vector database  |
| **Neo4j Browser**      | 7474 | http://localhost:7474           | Neo4j graph database browser |
| **Neo4j API**          | 8080 | http://localhost:8080/api/neo4j | Neo4j service endpoints      |

### Service Dependencies

- **ChromaDB**: Vector database for embeddings and retrieval
  - **API Endpoints**: `/api/v2/heartbeat`, `/api/v2/version`, `/api/v1/collections`
  - **Interactive Docs**: http://localhost:8080/docs (Swagger UI)
  - **Status**: ✅ Operational (v1.0.0)
- **MLflow**: Experiment tracking and model registry
- **Ollama**: Local LLM serving (optional)
- **Neo4j**: Knowledge graph database with GraphRAG capabilities
  - **API Endpoints**: `/api/neo4j/health`, `/api/neo4j/query`, `/api/neo4j/graphrag`
  - **Browser**: http://localhost:7474
  - **Service API**: http://localhost:8080/api/neo4j
  - **Status**: ✅ Operational (v1.0.0)
- **Prometheus/Grafana**: Monitoring stack (optional)

### ChromaDB Integration Details

- **Server Status**: ✅ Running on port 8081
- **API Version**: v2 (v1 deprecated)
- **Data Storage**: `chroma_data/` directory
- **Collections**: Test collection created and operational
- **Embeddings**: Auto-generated using all-MiniLM-L6-v2 model
- **Web Interface**: Available at http://localhost:8081/docs

## 📊 Monitoring & Observability

### Real-time Updates

The platform provides WebSocket connections for real-time monitoring:

```javascript
const ws = new WebSocket("ws://localhost:8080/ws");
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log("Real-time update:", data);
};
```

### Background Monitoring

Automatic background monitoring tasks provide:

- System status updates every 30 seconds
- Model pull progress notifications
- Optimization progress updates
- Error alerts and notifications

### Metrics Collection

The platform collects metrics for:

- Model inference performance
- System resource usage
- API endpoint performance
- Error rates and patterns

## 🛠️ Development

### Code Structure

```
src/enterprise_llmops/
├── main.py                    # Main entry point
├── frontend/
│   ├── fastapi_app.py         # FastAPI application
│   ├── modern_dashboard.py    # Dashboard components
│   └── copilot_integration.py # CopilotKit integration
├── automl/
│   └── optuna_optimizer.py    # Optuna optimization
├── mlops/
│   └── mlflow_manager.py      # MLflow integration
├── infrastructure/            # Kubernetes, Docker, Terraform
├── ollama_manager.py          # Ollama integration
├── model_registry.py          # Model registry
├── prompt_integration.py      # Prompt management
├── security.py                # Security components
└── monitoring.py              # Monitoring utilities
```

### Adding New Endpoints

1. Define the endpoint function in `fastapi_app.py`
2. Add appropriate authentication if needed
3. Include error handling and logging
4. Update documentation
5. Add tests

### Testing

```bash
# Run tests
pytest tests/ -v

# Test specific endpoints
curl http://localhost:8080/health
curl http://localhost:8080/api/info
curl http://localhost:8080/api/status

# Test Neo4j endpoints
curl http://localhost:8080/api/neo4j/health
curl http://localhost:8080/api/neo4j/info
curl -X POST http://localhost:8080/api/neo4j/query \
  -H "Content-Type: application/json" \
  -d '{"query": "RETURN 1 as test"}'
```

## 🔗 Integration with Gradio App

The FastAPI platform integrates seamlessly with the Gradio Model Evaluation app:

### Data Flow

1. **Gradio App** (port 7860) - User interface for model evaluation
2. **FastAPI Platform** (port 8080) - Backend services and data processing
3. **MLflow** (port 5000) - Experiment tracking and model registry
4. **ChromaDB** (port 8081) - Vector database for embeddings

### API Integration

The Gradio app can interact with the FastAPI platform through:

- REST API calls for model management
- WebSocket connections for real-time updates
- Shared data storage in MLflow and ChromaDB

## 📈 Performance & Scaling

### Production Considerations

- Use multiple workers: `--workers 4`
- Enable GPU support: `--enable-gpu`
- Configure proper authentication: `--enable-auth`
- Set up monitoring stack: `--enable-monitoring`

### Kubernetes Deployment

The platform includes Kubernetes deployment configurations:

```bash
# Deploy to Kubernetes
./src/enterprise_llmops/scripts/deploy.sh
```

## 🚨 Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure ports 8080, 7860, 5000, 8081, 8082 are available
   - **Solution**: All services now use unique ports - no conflicts
2. **Service dependencies**: Start ChromaDB (8081) and MLflow (5000) before the platform
3. **ChromaDB connection**: Verify server is running with `netstat -an | findstr :8081`
4. **Authentication errors**: Check if auth is enabled and tokens are valid
5. **Memory issues**: Monitor resource usage with `/api/monitoring/status`
6. **ChromaDB v1 API**: Use v2 endpoints, v1 is deprecated (returns 410)
7. **Documentation access**: Use correct ports (MkDocs: 8082, FastAPI: 8080, ChromaDB: 8081)

### Debug Mode

```bash
# Enable debug logging
python -m src.enterprise_llmops.main --log-level debug
```

### Logs

Check application logs in:

- `logs/llmops.log` - Application logs
- Console output - Real-time debugging

## 📞 Support

For issues and questions:

1. Check the [API documentation](http://localhost:8080/docs)
2. Review the [troubleshooting guide](../resources/troubleshooting.md)
3. Check the [progress bulletin](../progress-bulletin.md)
4. Examine application logs in `logs/llmops.log`

---

**Last Updated**: January 19, 2025  
**Version**: 2.1.0  
**Status**: Production Ready
