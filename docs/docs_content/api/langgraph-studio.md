# LangGraph Studio Integration

## 🎯 Overview

LangGraph Studio is a specialized agent IDE that enables visualization, interaction, and debugging of agentic systems that implement the LangGraph Server API protocol. This integration provides comprehensive support for building, testing, and monitoring complex agent workflows.

## 🚀 Key Features

### Core Capabilities

- **Visual Graph Architecture**: Visualize your graph architecture with interactive node and edge representations
- **Agent Interaction**: Run and interact with your agents in real-time
- **Assistant Management**: Manage assistants and their configurations
- **Thread Management**: Handle conversation threads and state management
- **Prompt Engineering**: Iterate on prompts with live feedback
- **Experiment Management**: Run experiments over datasets with comprehensive tracking
- **Long-term Memory**: Manage persistent memory across agent sessions
- **Time Travel Debugging**: Debug agent state via time travel capabilities

### Integration Features

- **LangSmith Integration**: Seamless integration with LangSmith for tracing, evaluation, and prompt engineering
- **Graph Mode**: Full feature-set with detailed execution information
- **Chat Mode**: Simplified UI for chat-specific agents
- **Real-time Monitoring**: Live agent state monitoring and performance metrics
- **API Access**: Programmatic access through REST API endpoints

## 📊 Structure/Architecture

### Service Configuration

```yaml
# LangGraph Studio Configuration
langgraph_studio:
  enabled: true
  host: "localhost"
  port: 8080
  studio_port: 8083 # Changed from 8081 to avoid conflict with ChromaDB
  mode: "graph" # or "chat"
  enable_langsmith: true
  langsmith_api_key: "${LANGSMITH_API_KEY}"
  langsmith_project: "ai-architecture"
  log_level: "INFO"
```

### API Endpoints

| Endpoint                               | Method | Description                          |
| -------------------------------------- | ------ | ------------------------------------ |
| `/api/langgraph/studios/status`        | GET    | Get studio service status            |
| `/api/langgraph/studios/start`         | POST   | Start LangGraph Studio service       |
| `/api/langgraph/studios/stop`          | POST   | Stop LangGraph Studio service        |
| `/api/langgraph/studios/info`          | GET    | Get comprehensive studio information |
| `/api/langgraph/studios/sessions`      | POST   | Create new studio session            |
| `/api/langgraph/studios/sessions/{id}` | GET    | Get session information              |
| `/api/langgraph/studios/sessions/{id}` | DELETE | Delete studio session                |
| `/api/langgraph/studios/assistants`    | GET    | Get available assistants             |
| `/api/langgraph/studios/threads`       | GET    | Get available threads                |
| `/api/langgraph/studios/experiments`   | GET    | Get available experiments            |
| `/api/langgraph/studios/datasets`      | GET    | Get available datasets               |
| `/api/langgraph/studios/memory`        | GET    | Get long-term memory information     |
| `/api/langgraph/studios/dashboard`     | GET    | Studio dashboard UI                  |

## 🌐 Service Integration

### Port Configuration

| Service                | Port | URL                                         | Description      |
| ---------------------- | ---- | ------------------------------------------- | ---------------- |
| **LangGraph Studio**   | 8083 | http://localhost:8083                       | Studio interface |
| **LangGraph API**      | 8080 | http://localhost:8080/api/langgraph/studios | API endpoints    |
| **Enterprise FastAPI** | 8080 | http://localhost:8080                       | Main platform    |

### Integration Points

1. **FastAPI Backend**: Integrated through dedicated API endpoints
2. **Model Registry**: Seamless access to registered models
3. **MLflow Tracking**: Experiment tracking and model versioning
4. **ChromaDB**: Vector database for agent memory and context
5. **LangSmith**: Tracing and evaluation integration

## 🔧 Configuration

### Prerequisites

```bash
# Install LangGraph CLI
pip install langgraph-cli

# Verify installation
langgraph --version
```

### Environment Variables

```bash
# LangSmith Integration (Optional)
export LANGSMITH_API_KEY="your-api-key"
export LANGSMITH_PROJECT="ai-architecture"

# Studio Configuration
export LANGGRAPH_STUDIO_HOST="localhost"
export LANGGRAPH_STUDIO_PORT="8083"
export LANGGRAPH_STUDIO_MODE="graph"
```

### Quick Start

```bash
# 1. Activate virtual environment
& C:\Users\samne\PycharmProjects\ai_assignments\venv\Scripts\Activate.ps1

# 2. Start LangGraph Studio
langgraph dev --host 0.0.0.0 --port 8083

# 3. Access Studio Interface
# Open http://localhost:8083 in browser

# 4. Access API Endpoints
curl http://localhost:8080/api/langgraph/studios/status
```

## 📚 Documentation

### Studio Modes

#### Graph Mode

- **Purpose**: Full feature-set with detailed execution information
- **Use Case**: Complex agent workflows with multiple nodes and edges
- **Features**:
  - Visual graph representation
  - Node-by-node execution tracking
  - Intermediate state inspection
  - LangSmith integration for detailed tracing

#### Chat Mode

- **Purpose**: Simplified UI for chat-specific agents
- **Use Case**: Conversational agents with MessagesState
- **Features**:
  - Clean chat interface
  - Message history management
  - Simplified debugging
  - Business user friendly

### Troubleshooting

#### Common Issues

1. **LangGraph CLI Not Found**

   ```bash
   # Install LangGraph CLI
   pip install langgraph-cli
   ```

2. **Port Conflicts**

   ```bash

   ```

# Check port availability

netstat -an | findstr :8083

# Use different port if needed

langgraph dev --port 8084

````

3. **LangSmith Connection Issues**

```bash
# Verify API key
echo $LANGSMITH_API_KEY

# Test connection
curl -H "Authorization: Bearer $LANGSMITH_API_KEY" https://api.smith.langchain.com/projects
````

#### Debug Commands

```bash
# Check studio status
curl http://localhost:8080/api/langgraph/studios/status

# View studio logs
langgraph dev --log-level debug

# Test API endpoints
curl http://localhost:8080/api/langgraph/studios/info
```

## 🛠️ Development

### API Integration Example

```python
import requests

# Get studio status
response = requests.get("http://localhost:8080/api/langgraph/studios/status")
status = response.json()

# Create new session
session_data = {
    "mode": "graph",
    "metadata": {"project": "ai-architecture"}
}
response = requests.post(
    "http://localhost:8080/api/langgraph/studios/sessions",
    json=session_data
)
session = response.json()

# Get assistants
response = requests.get("http://localhost:8080/api/langgraph/studios/assistants")
assistants = response.json()
```

### Studio Session Management

```python
from src.ai_architecture.langgraph_studio_integration import (
    LangGraphStudioManager,
    StudioConfig,
    StudioMode
)

# Initialize manager
config = StudioConfig(
    host="localhost",
    port=8080,
    studio_port=8083,
    mode=StudioMode.GRAPH,
    enable_langsmith=True
)

manager = LangGraphStudioManager(config)

# Start studio
await manager.start_studio()

# Create session
session_id = await manager.create_session(
    mode=StudioMode.GRAPH,
    metadata={"project": "ai-architecture"}
)

# Get studio information
info = await manager.get_studio_info()
```

## 🚨 Troubleshooting

### Service Status

Check the current status of LangGraph Studio:

```bash
# API endpoint
curl http://localhost:8080/api/langgraph/studios/status

# Expected response
{
  "status": "running",
  "base_url": "http://localhost:8083",
  "api_url": "http://localhost:8080/api/langgraph/studios",
  "sessions_count": 0,
  "uptime": 120.5
}
```

### Common Error Messages

1. **"LangGraph CLI not found"**

   - Solution: Install with `pip install langgraph-cli`

2. **"Failed to start LangGraph Studio"**

   - Check port availability
   - Verify LangGraph CLI installation
   - Check system permissions

3. **"Studio service not responding"**
   - Restart the service
   - Check firewall settings
   - Verify network connectivity

## 📞 Support

### Resources

- **LangGraph Studio Documentation**: [studio.langchain.com](https://studio.langchain.com){:target="\_blank"}
- **LangGraph CLI Documentation**: [LangGraph CLI Guide](https://langchain-ai.github.io/langgraph/cli/){:target="\_blank"}
- **LangSmith Integration**: [LangSmith Documentation](https://docs.smith.langchain.com/){:target="\_blank"}

### Getting Help

1. **Check Logs**: Review application logs for detailed error information
2. **API Status**: Use `/api/langgraph/studios/status` endpoint
3. **Documentation**: Refer to LangGraph Studio official documentation
4. **Community**: Join LangChain community for support

---

**Last Updated**: 2025-01-27  
**Version**: 2.1.0  
**Status**: Production Ready  
**Integration**: Full LangGraph Studio Integration
