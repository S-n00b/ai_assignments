# FastAPI Enterprise Platform - Interactive Documentation

## 🎯 Overview

This page provides direct access to the FastAPI Enterprise Platform's interactive API documentation. The documentation below is embedded from the live FastAPI application running on port 8080.

## 📊 Live API Documentation

<div class="api-docs-container">
    <iframe 
        src="http://localhost:8080/docs" 
        width="100%" 
        height="800px"
        frameborder="0"
        title="FastAPI Enterprise Platform API Documentation">
        <p>Your browser does not support iframes. 
           Please visit <a href="http://localhost:8080/docs" target="_blank">http://localhost:8080/docs</a> 
           to view the API documentation.</p>
    </iframe>
</div>

## 🔗 Direct Access Links

- **Interactive API Docs (Swagger UI)**: [http://localhost:8080/docs](http://localhost:8080/docs)
- **ReDoc Documentation**: [http://localhost:8080/redoc](http://localhost:8080/redoc)
- **OpenAPI Schema**: [http://localhost:8080/openapi.json](http://localhost:8080/openapi.json)
- **API Information**: [http://localhost:8080/api/info](http://localhost:8080/api/info)
- **System Status**: [http://localhost:8080/api/status](http://localhost:8080/api/status)

## 📋 Source Information

<div class="source-attribution">
    <strong>Source:</strong> 
    <code>src/enterprise_llmops/frontend/fastapi_app.py</code>
    <br>
    <strong>API Endpoint:</strong> 
    <code>http://localhost:8080/docs</code>
    <br>
    <strong>Documentation Type:</strong> 
    Auto-generated from FastAPI OpenAPI specification
    <br>
    <strong>Last Updated:</strong> 
    January 19, 2025
    <br>
    <strong>Port Assignment:</strong> 
    8080 (FastAPI Enterprise Platform)
</div>

## 🚀 Quick Start

### Prerequisites

- FastAPI Enterprise Platform must be running on port 8080
- Virtual environment activated

### Starting the FastAPI Platform

```bash
# Activate virtual environment
& C:\Users\samne\PycharmProjects\ai_assignments\venv\Scripts\Activate.ps1

# Start the platform
python -m src.enterprise_llmops.main --host 0.0.0.0 --port 8080
```

### Accessing Documentation

1. **Embedded View**: Use the iframe above (requires FastAPI running)
2. **Direct Access**: Visit http://localhost:8080/docs in a new tab
3. **ReDoc Format**: Visit http://localhost:8080/redoc for alternative format

## 🔧 API Features

### Interactive Testing

- Test all endpoints directly in the browser
- View request/response schemas
- Execute real API calls
- View authentication requirements

### Available Endpoints

- **Health & Status**: System health checks and status information
- **Model Management**: Model registry and lifecycle management
- **Ollama Integration**: Local LLM serving and management
- **Experiment Tracking**: MLflow integration for ML workflows
- **AutoML & Optimization**: Optuna hyperparameter optimization
- **Vector Databases**: ChromaDB integration (port 8081)
- **Monitoring & Metrics**: System monitoring and performance metrics
- **Authentication**: JWT-based security with demo mode support

## 🌐 Service Integration

The FastAPI platform integrates with:

| Service        | Port | Integration Type           |
| -------------- | ---- | -------------------------- |
| **ChromaDB**   | 8081 | Vector database operations |
| **MLflow**     | 5000 | Experiment tracking        |
| **Gradio App** | 7860 | Frontend interface         |
| **MkDocs**     | 8082 | Documentation site         |

## 🚨 Troubleshooting

### Common Issues

1. **Iframe not loading**

   - Ensure FastAPI platform is running on port 8080
   - Check firewall settings
   - Try direct access via http://localhost:8080/docs

2. **API endpoints not responding**

   - Verify all dependent services are running
   - Check service health at http://localhost:8080/api/status
   - Review application logs

3. **Authentication errors**
   - Check if authentication is enabled
   - Verify JWT tokens are valid
   - Use demo mode for testing

### Debug Commands

```bash
# Check if FastAPI is running
netstat -an | findstr :8080

# Test basic connectivity
curl http://localhost:8080/health

# Check API info
curl http://localhost:8080/api/info
```

## 📚 Related Documentation

- [FastAPI Enterprise Platform Overview](fastapi-enterprise.md)
- [ChromaDB Integration](chromadb-integration.md)
- [Gradio Model Evaluation](gradio-model-evaluation.md)
- [Documentation Sources](../development/documentation-sources.md)

---

**Note**: This embedded documentation requires the FastAPI Enterprise Platform to be running. If the iframe above is not loading, please ensure the platform is started on port 8080.
