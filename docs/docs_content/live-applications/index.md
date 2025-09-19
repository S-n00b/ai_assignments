# Live Applications & Demo Access

## 🚀 **Real-Time Platform Access**

This section provides direct access to all live applications and services running in the Lenovo AAITC AI Assignments platform. Each service is designed to demonstrate specific capabilities and can be accessed through the URLs below.

## 🌐 **Primary Applications**

### **Enterprise LLMOps Platform**

- **URL**: [http://localhost:8080](http://localhost:8080)
- **Description**: Main enterprise platform with comprehensive AI operations
- **Features**:
  - FastAPI backend with full enterprise features
  - Real-time monitoring and metrics
  - API documentation and testing interface
  - WebSocket support for live updates

### **Model Evaluation Interface**

- **URL**: [http://localhost:7860](http://localhost:7860)
- **Description**: Gradio interface for model evaluation and prototyping
- **Features**:
  - Interactive model comparison tools
  - Real-time evaluation metrics
  - Custom prompt testing interface
  - Visualization dashboards

### **Documentation Site**

- **URL**: [http://localhost:8000](http://localhost:8000)
- **Description**: This comprehensive MkDocs documentation site
- **Features**:
  - Complete project documentation
  - Interactive navigation and search
  - Code examples and tutorials
  - Architecture diagrams and guides

## 🔧 **Development & Testing Tools**

### **API Documentation**

- **URL**: [http://localhost:8080/docs](http://localhost:8080/docs)
- **Description**: FastAPI auto-generated documentation
- **Features**:
  - Interactive API testing interface
  - Complete endpoint documentation
  - Request/response examples
  - Authentication and authorization guides

### **Health Check Endpoint**

- **URL**: [http://localhost:8080/health](http://localhost:8080/health)
- **Description**: System health monitoring endpoint
- **Features**:
  - Real-time system status
  - Service availability checks
  - Performance metrics
  - Error reporting

## 🗄️ **Data & Model Management**

### **MLflow UI**

- **URL**: [http://localhost:5000](http://localhost:5000)
- **Description**: MLflow experiment tracking and model registry
- **Features**:
  - Experiment tracking and comparison
  - Model versioning and registry
  - Artifact storage and management
  - Performance metrics visualization

### **Ollama LLM Server**

- **URL**: [http://localhost:11434](http://localhost:11434)
- **Description**: Local LLM server for model serving
- **Features**:
  - Model management and deployment
  - API endpoints for inference
  - Model performance monitoring
  - Resource utilization tracking

## 📊 **Monitoring & Analytics**

### **Grafana Dashboards**

- **URL**: [http://localhost:3000](http://localhost:3000)
- **Description**: Monitoring dashboards and visualization
- **Features**:
  - System performance metrics
  - Application monitoring
  - Custom dashboard creation
  - Alert management

### **Prometheus Metrics**

- **URL**: [http://localhost:9090](http://localhost:9090)
- **Description**: Metrics collection and querying
- **Features**:
  - Time-series data collection
  - Custom metrics and alerts
  - Query interface for metrics
  - Integration with Grafana

### **LangFuse Observability**

- **URL**: [http://localhost:3000](http://localhost:3000) (Alternative port)
- **Description**: LLM observability and performance tracking
- **Features**:
  - LLM performance monitoring
  - Trace analysis and debugging
  - Cost tracking and optimization
  - Quality metrics and evaluation

## 🗃️ **Data Storage & Management**

### **Neo4j Browser**

- **URL**: [http://localhost:7474](http://localhost:7474)
- **Description**: Knowledge graph database browser
- **Features**:
  - Graph data visualization
  - Cypher query interface
  - Relationship mapping
  - Data exploration tools

### **Redis Cache**

- **URL**: [redis://localhost:6379](redis://localhost:6379)
- **Description**: In-memory data store for caching
- **Features**:
  - Session management
  - Cache performance optimization
  - Data persistence options
  - Cluster management

## 🔌 **Additional Services**

### **MCP Server**

- **URL**: [http://localhost:8001](http://localhost:8001)
- **Description**: Model Context Protocol server
- **Features**:
  - Tool integration interface
  - Agent communication protocols
  - Context management
  - Service discovery

### **Additional Service**

- **URL**: [http://localhost:8002](http://localhost:8002)
- **Description**: Secondary service for extended functionality
- **Features**:
  - Extended API endpoints
  - Additional processing capabilities
  - Integration with main platform
  - Custom functionality

## 🚀 **Quick Start Guide**

### **Starting All Services**

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Start Enterprise Platform
python -m src.enterprise_llmops.main --host 0.0.0.0 --port 8080

# Start Model Evaluation Interface
python -m src.gradio_app.main --host 0.0.0.0 --port 7860

# Start Documentation Site
cd docs
mkdocs serve
```

### **Accessing Services**

1. **Primary Platform**: Navigate to [http://localhost:8080](http://localhost:8080)
2. **Model Evaluation**: Access [http://localhost:7860](http://localhost:7860)
3. **Documentation**: Visit [http://localhost:8000](http://localhost:8000)
4. **API Testing**: Use [http://localhost:8080/docs](http://localhost:8080/docs)

## 🔧 **Troubleshooting**

### **Common Issues**

**Port Conflicts**:

- Ensure no other services are using the required ports
- Check for port 3000 conflicts between Grafana and LangFuse
- Use different ports if conflicts occur

**Service Unavailable**:

- Verify all services are running
- Check virtual environment activation
- Ensure all dependencies are installed

**Connection Issues**:

- Verify localhost connectivity
- Check firewall settings
- Ensure services are bound to 0.0.0.0

### **Health Checks**

```powershell
# Check Enterprise Platform
curl http://localhost:8080/health

# Check Model Evaluation Interface
curl http://localhost:7860

# Check MLflow
curl http://localhost:5000

# Check Ollama
curl http://localhost:11434/api/tags
```

## 📱 **Mobile & Cross-Device Access**

### **Network Access**

To access services from other devices on the same network:

1. Replace `localhost` with your machine's IP address
2. Ensure firewall allows connections on required ports
3. Update URLs accordingly (e.g., `http://192.168.1.100:8080`)

### **Remote Access**

For remote access, consider:

- VPN connection to your network
- Port forwarding configuration
- Secure tunneling solutions
- Cloud deployment options

---

_This live applications section provides comprehensive access to all platform services, enabling hands-on exploration of the Lenovo AAITC AI Assignments platform capabilities._
