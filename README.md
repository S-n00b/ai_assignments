# Lenovo AAITC AI Assignments - Enterprise LLMOps Platform

A comprehensive enterprise-grade AI operations platform built for Lenovo AAITC assignments, featuring advanced model evaluation, MLOps pipeline, and modern AI architecture.

## 🚀 **Quick Start**

### Prerequisites

- Python 3.11+
- PowerShell (Windows)
- Virtual Environment
- Docker (optional)
- Kubernetes (optional)

### Installation

1. **Clone and Setup**

   ```powershell
   git clone <repository-url>
   cd ai_assignments
   .\venv\Scripts\Activate.ps1
   ```

2. **Install Dependencies**

   ```powershell
   pip install -r config/requirements.txt
   pip install -r config/requirements-testing.txt
   ```

### 🚨 **Port Configuration & Quick Start**

**⚠️ IMPORTANT: Port 3000 Conflict**

- **Grafana** and **LangFuse** both use port 3000
- Only one can run at a time in local development
- Consider using different ports for production deployment

#### **All Service Ports**

| Service                | Port  | URL                    | Description                |
| ---------------------- | ----- | ---------------------- | -------------------------- |
| **Enterprise FastAPI** | 8080  | http://localhost:8080  | Main enterprise platform   |
| **Gradio App**         | 7860  | http://localhost:7860  | Model evaluation interface |
| **MkDocs Docs**        | 8082  | http://localhost:8082  | Documentation site         |
| **Chroma Vector DB**   | 8081  | http://localhost:8081  | Embeddings database        |
| **LangGraph Studio**   | 8080  | http://localhost:8080  | Workflow visualization     |
| **MLflow Tracking**    | 5000  | http://localhost:5000  | Experiment tracking        |
| **Ollama LLM**         | 11434 | http://localhost:11434 | Local LLM server           |
| **Grafana**            | 3000  | http://localhost:3000  | Monitoring dashboards      |
| **LangFuse**           | 3000  | http://localhost:3000  | LLM observability          |
| **Prometheus**         | 9090  | http://localhost:9090  | Metrics collection         |
| **Neo4j**              | 7474  | http://localhost:7474  | Knowledge graph database   |
| **Redis**              | 6379  | redis://localhost:6379 | Caching and sessions       |
| **MCP Server**         | 8001  | http://localhost:8001  | Model Context Protocol     |
| **Additional Service** | 8002  | http://localhost:8002  | Secondary service          |

#### **Launch Applications**

```powershell
# 1. Enterprise FastAPI Platform (Main Service)
python -m src.enterprise_llmops.main --host 0.0.0.0 --port 8080

# 2. Gradio Model Evaluation Interface
python -m src.gradio_app.main --host 0.0.0.0 --port 7860

# 3. MkDocs Documentation
cd docs
mkdocs serve

# 4. Optional: Simplified FastAPI App (Alternative)
python -m src.enterprise_llmops.simple_app

# 5. Optional: Individual Services (if running separately)
# MLflow: python -m mlflow server --host 0.0.0.0 --port 5000
# Ollama: ollama serve (runs on 11434)
```

#### **Quick Access URLs**

- **Main Platform**: http://localhost:8080
- **API Documentation**: http://localhost:8080/docs
- **Health Check**: http://localhost:8080/health
- **Model Evaluation**: http://localhost:7860
- **Documentation**: http://localhost:8082
- **MLflow UI**: http://localhost:5000 (when MLflow is running)
- **Ollama API**: http://localhost:11434 (when Ollama is running)

## 📋 **Project Structure**

```
ai_assignments/
├── src/
│   ├── enterprise_llmops/          # Enterprise LLMOps Platform (Assignment 2)
│   │   ├── infrastructure/         # Kubernetes, Docker, Terraform
│   │   ├── automl/                 # Optuna hyperparameter optimization
│   │   ├── mlops/                  # MLflow experiment tracking
│   │   ├── frontend/               # FastAPI enterprise frontend
│   │   └── scripts/                # Deployment scripts
│   ├── gradio_app/                 # Model Evaluation Interface (Assignment 1)
│   ├── ai_architecture/            # AI system architecture
│   ├── model_evaluation/           # Model evaluation pipeline
│   └── utils/                      # Utility functions
├── tests/                          # Comprehensive test suites
├── docs/                           # MkDocs documentation
├── config/                         # Configuration files
└── scripts/                        # Development scripts
```

## 🏗️ **Architecture Overview**

### **Assignment 1: Model Evaluation Prototyping**

- **Gradio Interface**: Interactive model evaluation and comparison
- **MCP Server**: Model Context Protocol for tool integration
- **Evaluation Pipeline**: Comprehensive model testing framework
- **Visualization**: Real-time metrics and performance dashboards

### **Assignment 2: Enterprise LLMOps Platform**

- **FastAPI Backend**: Production-ready REST API
- **Kubernetes Orchestration**: Container orchestration and scaling
- **MLflow Integration**: Experiment tracking and model registry
- **Vector Databases**: Chroma, Weaviate, Pinecone integration
- **Monitoring Stack**: Prometheus, Grafana, LangFuse observability
- **AutoML**: Optuna hyperparameter optimization

## 🔧 **Key Features**

### **Enterprise Infrastructure**

- ✅ **Kubernetes Deployment**: Production-ready container orchestration
- ✅ **Docker Containers**: Optimized multi-service architecture
- ✅ **Terraform IaC**: Infrastructure as code automation
- ✅ **Monitoring**: Comprehensive observability stack

### **AI/ML Operations**

- ✅ **Model Management**: Ollama integration for local LLM serving
- ✅ **Experiment Tracking**: MLflow for reproducible ML workflows
- ✅ **AutoML**: Automated hyperparameter optimization with Optuna
- ✅ **Vector Search**: Advanced semantic search capabilities

### **Modern UI/UX**

- ✅ **FastAPI Backend**: RESTful API with WebSocket support
- ✅ **Real-time Monitoring**: Live system status and metrics
- ✅ **Knowledge Graphs**: Neo4j integration for relationship mapping
- ✅ **Workflow Visualization**: LangGraph Studio-style interfaces
- ✅ **Chat Playground**: Side-by-side Ollama & GitHub Models comparison with Google AI Studio-like UX

## 🌐 **Application URLs**

> **Note**: For complete port information, see the [Port Configuration](#-port-configuration--quick-start) section above.

| Service                 | URL                          | Description                                   |
| ----------------------- | ---------------------------- | --------------------------------------------- |
| **Enterprise Platform** | http://localhost:8080        | FastAPI backend with full enterprise features |
| **Chat Playground**     | http://localhost:8080        | Ollama & GitHub Models side-by-side comparison |
| **Model Evaluation**    | http://localhost:7860        | Gradio interface for model prototyping        |
| **Documentation**       | http://localhost:8082        | MkDocs documentation site                     |
| **API Docs**            | http://localhost:8080/docs   | FastAPI auto-generated documentation          |
| **Health Check**        | http://localhost:8080/health | System health monitoring                      |

### **Port Conflict Resolution**

**Issue**: Grafana and LangFuse both use port 3000

**Solutions**:

1. **Local Development**: Run only one service at a time
2. **Production**: Use Kubernetes port-forwarding or different ports
3. **Docker**: Map different host ports to container port 3000

## 📊 **Technology Stack**

### **Backend & Infrastructure**

- **FastAPI**: Modern, fast web framework
- **Kubernetes**: Container orchestration
- **Docker**: Containerization
- **Terraform**: Infrastructure as code
- **PostgreSQL**: Database for MLflow
- **MinIO**: Object storage for artifacts

### **AI/ML Frameworks**

- **PyTorch**: Deep learning framework
- **LangChain**: LLM application development
- **LangGraph**: Agentic workflows
- **AutoGen**: Multi-agent systems
- **MLflow**: Experiment tracking
- **Optuna**: Hyperparameter optimization

### **Vector Databases**

- **Chroma**: Embeddings database
- **Weaviate**: Vector search engine
- **Pinecone**: Managed vector database

### **Monitoring & Observability**

- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **LangFuse**: LLM observability
- **WebSocket**: Real-time updates

## 🚀 **Deployment Options**

### **Local Development**

```powershell
# Start all services locally
python -m src.enterprise_llmops.simple_app
python -m src.gradio_app.main --host 0.0.0.0 --port 7860
```

### **Kubernetes Production**

```bash
# Deploy complete enterprise stack
./src/enterprise_llmops/scripts/deploy.sh

# Access services (all ports are properly configured in Kubernetes)
kubectl port-forward service/llmops-frontend-service 8080:8080 -n llmops-enterprise
kubectl port-forward service/grafana-service 3000:3000 -n llmops-enterprise
kubectl port-forward service/langfuse-service 3001:3000 -n llmops-enterprise  # Different host port
```

**Note**: In Kubernetes, port conflicts are resolved by:

- Using different host ports for port-forwarding
- Services communicate internally via service names
- LoadBalancer/Ingress handles external routing

### **Docker Compose**

```yaml
# docker-compose.yml (available in infrastructure/docker/)
docker-compose up -d
```

## 📚 **Documentation**

### **Enhanced MkDocs Site Structure**

Our documentation is organized into two distinct categories with professional content:

#### **Category 1: Model Enablement & UX Evaluation**

- **AI Engineering Overview**: Foundation AI engineering principles
- **Model Evaluation Framework**: Comprehensive evaluation methodologies
- **UX Evaluation & Testing**: User experience design and testing
- **Model Profiling & Characterization**: Performance analysis and optimization
- **Model Factory Architecture**: Automated model selection and deployment
- **Practical Evaluation Exercise**: Hands-on implementation examples

#### **Category 2: AI System Architecture & MLOps**

- **System Architecture Overview**: Enterprise-scale AI system design
- **MLOps & CI/CD Lifecycle**: Complete model lifecycle management
- **Post-Training Optimization**: Advanced model optimization techniques
- **Frontier Model Experimentation**: Cutting-edge AI research and development
- **Stakeholder Vision Scoping**: Executive communication and strategy
- **Project Management & Skills**: Professional development and leadership

### **Executive & Professional Content**

- **Carousel Slide Deck**: Comprehensive stakeholder presentations
- **Executive Summary**: Strategic overview and business impact
- **AI Architecture Seniority Blog**: Medium-style professional content
- **ROI Analysis**: Financial projections and business metrics

### **Live Applications & Demos**

- **Real-time Platform Access**: Direct links to all running services
- **Interactive Documentation**: Embedded iframes and live demonstrations
- **GitHub Pages Deployment**: Public access at `https://s-n00b.github.io/ai_assignments`

### **Documentation URLs**

- **Local Development**: [http://localhost:8082](http://localhost:8082)
- **GitHub Pages**: [https://s-n00b.github.io/ai_assignments](https://s-n00b.github.io/ai_assignments)
- **API Documentation**: [http://localhost:8080/docs](http://localhost:8080/docs)
- **Live Applications**: [http://localhost:8080](http://localhost:8080)

## 🧪 **Testing**

```powershell
# Run all tests
pytest

# Run specific test suites
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run with coverage
pytest --cov=src tests/
```

## 🔧 **Development Commands**

```powershell
# Use development commands
.\scripts\dev-commands.ps1

# Common operations
.\scripts\dev-commands.ps1 -Build
.\scripts\dev-commands.ps1 -Test
.\scripts\dev-commands.ps1 -Deploy
```

## 📈 **Progress Status**

- **Overall Completion**: 95%
- **Infrastructure**: 100% Complete
- **Core Services**: 100% Complete
- **API Endpoints**: 100% Complete
- **Documentation**: 100% Complete
- **Integration**: 100% Complete
- **Chat Playground**: 100% Complete

## 🎯 **Next Steps**

### **Immediate (1-2 days)**

1. **LangGraph Studio Integration**: Integrate LangGraph Studio for agent visualization and debugging
2. **QLoRA Fine-Tuning**: Implement QLoRA adapter management and fine-tuning capabilities
3. **Neo4j UI with Faker**: Create realistic GraphRAG demo with Faker-generated data
4. **End-to-End Testing**: Validate complete enterprise workflow

### **Short Term (1-2 weeks)**

1. **Production Deployment**: Deploy to Kubernetes cluster
2. **Integration Testing**: Comprehensive testing of all components
3. **Performance Optimization**: Scale and optimize for production loads
4. **Security Hardening**: Implement enterprise security features

## 🤝 **Contributing**

1. Follow the project structure and coding standards
2. Run tests before submitting changes
3. Update documentation for new features
4. Use the development scripts for common operations

## 📄 **License**

This project is developed for Lenovo AAITC assignments and educational purposes.

## 📞 **Support**

For questions and support:

- Check the documentation at http://localhost:8082
- Review the API documentation at http://localhost:8080/docs
- See [completed.md](completed.md) for recent achievements
- Check [TODO.md](TODO.md) for remaining tasks

---

**Last Updated**: January 19, 2025  
**Status**: Enterprise LLMOps Platform 95% Complete  
**Ready for**: Final integration testing and production deployment
