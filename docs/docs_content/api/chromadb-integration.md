# ChromaDB Integration Documentation

## 🎯 Overview

ChromaDB is the vector database solution integrated into the Enterprise LLMOps Platform. This document provides comprehensive information about ChromaDB setup, configuration, and integration with the FastAPI platform.

## 🚀 Quick Status

**Current Status**: ✅ **FULLY OPERATIONAL**

- **Server**: Running on port 8081
- **Version**: 1.0.0
- **API**: v2 (v1 deprecated)
- **Data Storage**: `chroma_data/` directory
- **Collections**: Test collection operational

## 🌐 Service Endpoints

### Working Endpoints

| Endpoint            | Method | Status | Description                   |
| ------------------- | ------ | ------ | ----------------------------- |
| `/api/v2/heartbeat` | GET    | ✅ 200 | Health check                  |
| `/api/v2/version`   | GET    | ✅ 200 | Server version                |
| `/docs`             | GET    | ✅ 200 | Interactive API documentation |
| `/openapi.json`     | GET    | ✅ 200 | OpenAPI specification         |

### Deprecated Endpoints (v1 API)

| Endpoint              | Status | Response                                        |
| --------------------- | ------ | ----------------------------------------------- |
| `/api/v1/heartbeat`   | ❌ 410 | "The v1 API is deprecated. Please use /v2 apis" |
| `/api/v1/version`     | ❌ 410 | "The v1 API is deprecated. Please use /v2 apis" |
| `/api/v1/collections` | ❌ 410 | "The v1 API is deprecated. Please use /v2 apis" |

## 🔧 Configuration

### Server Configuration

```bash
# Start ChromaDB server
chroma run --host 0.0.0.0 --port 8081 --path chroma_data
```

### Python Client Configuration

```python
import chromadb
from chromadb.config import Settings

# Connect to ChromaDB server
client = chromadb.HttpClient(
    host="localhost",
    port=8081,
    settings=Settings(allow_reset=True)
)

# Verify connection
version = client.get_version()
print(f"ChromaDB Version: {version}")
```

### Enterprise Configuration

In `config/enterprise-config.yaml`:

```yaml
services:
  chroma:
    url: "http://localhost:8081"
    health_endpoint: "/api/v2/heartbeat"
    required: true
    timeout: 10
    start_command: "chroma run --host 0.0.0.0 --port 8081 --path chroma_data"
```

## 📊 Integration with FastAPI Platform

### FastAPI Endpoints

The Enterprise FastAPI platform provides these ChromaDB integration endpoints:

- `GET /api/chromadb/health` - ChromaDB health check
- `GET /api/chromadb/collections` - List ChromaDB collections
- `POST /api/chromadb/collections` - Create new collection
- `GET /api/chromadb/collections/{collection_name}` - Get collection details
- `POST /api/chromadb/collections/{collection_name}/add` - Add documents to collection
- `POST /api/chromadb/collections/{collection_name}/query` - Query collection

### Integration Code

```python
# In FastAPI application
try:
    import chromadb
    from chromadb.config import Settings
    chroma_client = chromadb.HttpClient(
        host="localhost",
        port=8081,
        settings=Settings(allow_reset=True)
    )
    # Test connection
    version = chroma_client.get_version()
    logging.info(f"ChromaDB client initialized successfully (v{version})")
except Exception as e:
    logging.warning(f"Failed to initialize ChromaDB client: {e}")
    chroma_client = None
```

## 🗄️ Data Management

### Collections

ChromaDB stores data in collections. Each collection can contain:

- **Documents**: Text content for vectorization
- **Metadatas**: Structured metadata for filtering
- **Embeddings**: Vector representations (auto-generated)
- **IDs**: Unique identifiers for each document

### Example Collection Usage

```python
# Create collection
collection = client.create_collection(
    name="test_collection",
    metadata={"description": "Test collection for verification"}
)

# Add documents
collection.add(
    documents=[
        "This is a test document about AI and machine learning.",
        "ChromaDB is a vector database for building AI applications."
    ],
    metadatas=[
        {"topic": "AI", "category": "technology"},
        {"topic": "ChromaDB", "category": "database"}
    ],
    ids=["doc1", "doc2"]
)

# Query collection
results = collection.query(
    query_texts=["What is ChromaDB?"],
    n_results=2
)
```

## 🌐 Web Interface

### Interactive API Documentation

ChromaDB provides an interactive Swagger UI interface:

**URL**: http://localhost:8081/docs

**Features**:

- Browse all available API endpoints
- Test API calls directly in the browser
- View request/response schemas
- Execute real API calls

### Access Instructions

1. Ensure ChromaDB server is running on port 8081
2. Open browser to http://localhost:8081/docs
3. Explore available endpoints
4. Test API functionality

## 🔄 Service Integration

### Port Configuration

| Service      | Port | URL                   | Status    |
| ------------ | ---- | --------------------- | --------- |
| **ChromaDB** | 8081 | http://localhost:8081 | ✅ Active |
| **MkDocs**   | 8082 | http://localhost:8082 | ✅ Active |

**Port Assignment**:

- ChromaDB: Port 8081 (no conflicts)
- MkDocs: Port 8082 (no conflicts)
- All services use unique ports

### Service Dependencies

- **ChromaDB**: Vector database for embeddings and retrieval
- **FastAPI Platform**: Backend services integration
- **Gradio App**: Frontend interface for vector operations
- **MLflow**: Experiment tracking integration

## 🧪 Testing & Validation

### Health Check

```bash
# Check if ChromaDB is running
netstat -an | findstr :8081

# Test API endpoint
curl http://localhost:8081/api/v2/heartbeat
```

### Python Client Test

```python
def test_chromadb_connection():
    try:
        client = chromadb.HttpClient(host="localhost", port=8081)
        version = client.get_version()
        print(f"✅ ChromaDB Version: {version}")

        collections = client.list_collections()
        print(f"✅ Collections: {len(collections)} found")

        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
```

## 🚨 Troubleshooting

### Common Issues

1. **Port 8081 not accessible**

   - Check if ChromaDB server is running on port 8081
   - Verify port is not blocked by firewall
   - Ensure no other service is using port 8081

2. **v1 API errors**

   - Use v2 endpoints instead of v1
   - Update client code to use v2 API
   - Check endpoint URLs in documentation

3. **Connection timeouts**

   - Verify ChromaDB server is responsive
   - Check network connectivity
   - Increase timeout values in configuration

4. **OpenTelemetry warnings**
   - These are informational only
   - ChromaDB works fine without OpenTelemetry
   - Can be ignored or configured if needed

### Debug Commands

```bash
# Check server status
netstat -an | findstr :8081

# Test basic connectivity
curl http://localhost:8081/api/v2/heartbeat

# Check server logs
# Look for ChromaDB startup messages in terminal
```

## 📈 Performance & Scaling

### Production Considerations

- **Memory Usage**: Monitor ChromaDB memory consumption
- **Storage**: Ensure adequate disk space for vector data
- **Network**: Optimize network latency for API calls
- **Concurrent Connections**: Monitor connection limits

### Optimization Tips

- Use appropriate embedding models for your use case
- Batch document operations when possible
- Monitor collection sizes and query performance
- Implement proper error handling and retry logic

## 🔗 Related Documentation

- [FastAPI Enterprise Platform](fastapi-enterprise.md)
- [Gradio Model Evaluation](gradio-model-evaluation.md)
- [Model Evaluation API](model-evaluation.md)
- [Troubleshooting Guide](../resources/troubleshooting.md)

## 📞 Support

For ChromaDB-specific issues:

1. Check ChromaDB server logs
2. Verify API endpoint responses
3. Test with Python client
4. Review integration code in FastAPI platform
5. Check port conflicts with other services

---

**Last Updated**: January 19, 2025  
**Version**: 1.0.0  
**Status**: Production Ready
