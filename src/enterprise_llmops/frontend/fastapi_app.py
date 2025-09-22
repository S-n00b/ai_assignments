"""
Enterprise LLMOps Frontend - FastAPI Application

This module provides a comprehensive enterprise-grade frontend for LLM operations
featuring real-time monitoring, model management, experiment tracking, and
integration with all the enterprise stack components.

Key Features:
- FastAPI-based REST API
- Real-time WebSocket connections for monitoring
- Integration with Ollama, MLflow, Optuna, and monitoring stack
- Modern React frontend with TypeScript
- LangGraph Studio integration
- Neo4j knowledge graph visualization
- Comprehensive dashboard and analytics
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import uvicorn
from contextlib import asynccontextmanager
import os
from pathlib import Path
import jwt
from pydantic import BaseModel
from dataclasses import asdict

# Import our enterprise components
from ..ollama_manager import OllamaManager
from ..model_registry import EnterpriseModelRegistry, ModelStatus, ModelType
from ..automl.optuna_optimizer import OptunaOptimizer, OptimizationConfig, LLMHyperparameterSpace
from ..mlops.mlflow_manager import MLflowManager, ExperimentConfig, ModelInfo, DeploymentConfig
from ..prompt_integration import PromptIntegrationManager

# Import GitHub Models integration
try:
    from ...github_models_integration.api_client import GitHubModelsAPIClient
    from ...github_models_integration.remote_serving import GitHubModelsRemoteServing
    GITHUB_MODELS_AVAILABLE = True
except ImportError:
    GITHUB_MODELS_AVAILABLE = False
    logging.warning("GitHub Models integration not available")

# Import LangGraph Studio integration
try:
    from ...ai_architecture.langgraph_studio_integration import (
        LangGraphStudioManager, 
        initialize_langgraph_studio_manager,
        get_langgraph_studio_manager,
        create_langgraph_studio_endpoints,
        StudioConfig
    )
    LANGGRAPH_STUDIO_AVAILABLE = True
except ImportError:
    LANGGRAPH_STUDIO_AVAILABLE = False
    logging.warning("LangGraph Studio integration not available")

# Import QLoRA integration
try:
    from ...ai_architecture.qlora_integration import (
        QLoRAManager,
        initialize_qlora_manager,
        get_qlora_manager,
        create_qlora_endpoints
    )
    QLORA_AVAILABLE = True
except ImportError:
    QLORA_AVAILABLE = False
    logging.warning("QLoRA integration not available")

# Import Neo4j Faker integration
try:
    from ...ai_architecture.neo4j_faker_integration import (
        Neo4jFakerManager,
        initialize_neo4j_faker_manager,
        get_neo4j_faker_manager,
        create_neo4j_faker_endpoints
    )
    NEO4J_FAKER_AVAILABLE = True
except ImportError:
    NEO4J_FAKER_AVAILABLE = False
    logging.warning("Neo4j Faker integration not available")


# Authentication models
class UserLoginRequest(BaseModel):
    """User login request model."""
    username: str
    password: str


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str = "bearer"


# WebSocket connection manager
class ConnectionManager:
    """Manage WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.logger = logging.getLogger("connection_manager")
    
    async def connect(self, websocket: WebSocket):
        """Accept a WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        self.logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            return
        
        message_json = json.dumps(message)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                self.logger.warning(f"Failed to send message to WebSocket: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)


# Global managers
ollama_manager: Optional[OllamaManager] = None
model_registry: Optional[EnterpriseModelRegistry] = None
mlflow_manager: Optional[MLflowManager] = None
optuna_optimizer: Optional[OptunaOptimizer] = None
prompt_integration_manager: Optional[PromptIntegrationManager] = None
langgraph_studio_manager: Optional[LangGraphStudioManager] = None
qlora_manager: Optional[QLoRAManager] = None
neo4j_faker_manager: Optional[Neo4jFakerManager] = None
chroma_client = None
connection_manager = ConnectionManager()

# GitHub Models integration
github_models_client: Optional[GitHubModelsAPIClient] = None
github_models_serving: Optional[GitHubModelsRemoteServing] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global ollama_manager, model_registry, mlflow_manager, optuna_optimizer, prompt_integration_manager, langgraph_studio_manager, qlora_manager, neo4j_faker_manager, chroma_client, github_models_client, github_models_serving
    
    # Startup
    logging.info("Starting Enterprise LLMOps Frontend...")
    
    try:
        # Initialize Ollama manager (optional)
        enable_ollama = app.state.config.get("enable_ollama", True)
        if enable_ollama:
            try:
                ollama_manager = OllamaManager()
                await ollama_manager.initialize()
                logging.info("Ollama manager initialized")
            except Exception as e:
                logging.warning(f"Failed to initialize Ollama manager: {e}")
                logging.info("Continuing without Ollama support...")
                ollama_manager = None
        else:
            logging.info("Ollama support disabled")
            ollama_manager = None
        
        # Initialize model registry
        model_registry = EnterpriseModelRegistry()
        logging.info("Model registry initialized")
        
        # Initialize MLflow manager
        mlflow_config = ExperimentConfig(
            experiment_name="llmops_enterprise",
            tracking_uri="http://localhost:5000",
            description="Enterprise LLMOps Experiment Tracking"
        )
        mlflow_manager = MLflowManager(mlflow_config)
        logging.info("MLflow manager initialized")
        
        # Initialize Optuna optimizer
        optuna_config = OptimizationConfig(
            study_name="llm_optimization",
            direction="maximize",
            n_trials=100,
            pruning_enabled=True
        )
        optuna_optimizer = OptunaOptimizer(optuna_config)
        logging.info("Optuna optimizer initialized")
        
        # Initialize prompt integration manager
        prompt_integration_manager = PromptIntegrationManager()
        logging.info("Prompt integration manager initialized")
        
        # Initialize LangGraph Studio manager
        if LANGGRAPH_STUDIO_AVAILABLE:
            try:
                langgraph_studio_manager = initialize_langgraph_studio_manager()
                logging.info("LangGraph Studio manager initialized")
            except Exception as e:
                logging.warning(f"Failed to initialize LangGraph Studio manager: {e}")
                langgraph_studio_manager = None
        else:
            logging.info("LangGraph Studio integration not available")
            langgraph_studio_manager = None
        
        # Initialize QLoRA manager
        if QLORA_AVAILABLE:
            try:
                qlora_manager = initialize_qlora_manager(model_registry)
                logging.info("QLoRA manager initialized")
            except Exception as e:
                logging.warning(f"Failed to initialize QLoRA manager: {e}")
                qlora_manager = None
        else:
            logging.info("QLoRA integration not available")
            qlora_manager = None
        
        # Initialize Neo4j Faker manager
        if NEO4J_FAKER_AVAILABLE:
            try:
                neo4j_faker_manager = initialize_neo4j_faker_manager()
                logging.info("Neo4j Faker manager initialized")
            except Exception as e:
                logging.warning(f"Failed to initialize Neo4j Faker manager: {e}")
                neo4j_faker_manager = None
        else:
            logging.info("Neo4j Faker integration not available")
            neo4j_faker_manager = None
        
        # Initialize GitHub Models integration
        if GITHUB_MODELS_AVAILABLE:
            try:
                github_models_client = GitHubModelsAPIClient()
                github_models_serving = GitHubModelsRemoteServing(github_models_client)
                logging.info("GitHub Models integration initialized")
            except Exception as e:
                logging.error(f"Failed to initialize GitHub Models integration: {e}")
                github_models_client = None
                github_models_serving = None
        else:
            logging.warning("GitHub Models integration not available")
        
        # Initialize ChromaDB client with v2 API
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Configure for ChromaDB v2 API
            chroma_client = chromadb.HttpClient(
                host="localhost",
                port=8081,
                settings=Settings(
                    allow_reset=True,
                    anonymized_telemetry=False
                )
            )
            # Test connection with v2 API - use heartbeat
            try:
                heartbeat_response = chroma_client.heartbeat()
                logging.info(f"ChromaDB v2 client initialized successfully")
                logging.info(f"ChromaDB heartbeat: {heartbeat_response}")
            except Exception as heartbeat_error:
                # Fallback to version check
                version = chroma_client.get_version()
                logging.info(f"ChromaDB client initialized (v{version}) - heartbeat failed but version check passed")
        except Exception as e:
            logging.warning(f"Failed to initialize ChromaDB client: {e}")
            logging.info("Continuing without ChromaDB support...")
            chroma_client = None
        
        # Initialize LangGraph Studio manager
        try:
            from ..ai_architecture.langgraph_studio_integration import initialize_langgraph_studio_manager
            langgraph_studio_manager = initialize_langgraph_studio_manager()
            logging.info("LangGraph Studio manager initialized")
        except Exception as e:
            logging.warning(f"Failed to initialize LangGraph Studio manager: {e}")
            langgraph_studio_manager = None
        
        # Initialize Neo4j Faker manager
        try:
            from ..ai_architecture.neo4j_faker_integration import initialize_neo4j_faker_manager
            neo4j_faker_manager = initialize_neo4j_faker_manager()
            logging.info("Neo4j Faker manager initialized")
        except Exception as e:
            logging.warning(f"Failed to initialize Neo4j Faker manager: {e}")
            neo4j_faker_manager = None
        
        # Start background monitoring
        asyncio.create_task(background_monitoring())
        
        logging.info("Enterprise LLMOps Frontend started successfully")
        
    except Exception as e:
        logging.error(f"Failed to initialize application: {e}")
        raise
    
    yield
    
    # Shutdown
    logging.info("Shutting down Enterprise LLMOps Frontend...")
    
    if ollama_manager:
        await ollama_manager.shutdown()
    
    if model_registry:
        model_registry.close()
    
    logging.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Enterprise LLMOps Platform",
    description="""
    ## 🚀 Enterprise LLMOps Platform for Lenovo AAITC
    
    A comprehensive enterprise-grade AI operations platform featuring:
    
    ### 🎯 Core Features
    - **Model Management**: Ollama integration with local LLM serving
    - **Experiment Tracking**: MLflow for reproducible ML workflows  
    - **AutoML**: Optuna hyperparameter optimization
    - **Vector Databases**: Chroma, Weaviate, Pinecone integration
    - **Monitoring**: Prometheus, Grafana, LangFuse observability
    - **Knowledge Graphs**: Neo4j integration for relationship mapping
    - **Real-time Updates**: WebSocket connections for live monitoring
    
    ### 📊 Applications
    - **Assignment 1**: Model Evaluation Engineer (Gradio app on port 7860)
    - **Assignment 2**: Enterprise LLMOps Platform (This FastAPI app)
    
    ### 🔗 External Services
    - **MLflow UI**: http://localhost:5000
    - **ChromaDB**: http://localhost:8081  
    - **Gradio App**: http://localhost:7860
    - **Documentation**: http://localhost:8082 (MkDocs)
    
    ### 🔐 Authentication
    - **Demo Mode**: Authentication disabled by default
    - **Production Mode**: Enable with `--enable-auth` flag
    """,
    version="2.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security configuration
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
ENABLE_AUTH = False  # Set to True for production

security = HTTPBearer(auto_error=False)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token."""
    # Skip authentication if disabled (demo mode)
    # Check both global setting and app config
    enable_auth = ENABLE_AUTH
    if hasattr(app.state, 'config'):
        enable_auth = app.state.config.get("enable_auth", ENABLE_AUTH)
    
    if not enable_auth:
        return {"user_id": "demo_user", "role": "admin"}
    
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"user_id": username, "role": "admin"}
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# Static files - only mount if directory exists
import os
static_dir = "frontend/build/static"
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
else:
    # Create a simple static file handler for development
    @app.get("/static/{file_path:path}")
    async def serve_static_dev(file_path: str):
        """Serve static files in development mode."""
        raise HTTPException(status_code=404, detail="Static files not available in development mode")


@app.get("/", response_class=HTMLResponse)
async def serve_unified_platform():
    """Serve the unified platform with Lenovo branding and embedded services."""
    unified_platform_file = "src/enterprise_llmops/frontend/unified_platform.html"
    if os.path.exists(unified_platform_file):
        return FileResponse(unified_platform_file)
    else:
        # Fallback to original pitch page if available
        pitch_file = "lenovo_ai_architecture_pitch.html"
        if os.path.exists(pitch_file):
            return FileResponse(pitch_file)
        else:
            # Serve a simple development landing page
            return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Enterprise LLMOps Platform</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                .container { max-width: 1000px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h1 { color: #333; text-align: center; }
                .status { background: #e8f5e8; padding: 20px; border-radius: 4px; margin: 20px 0; }
                .endpoint { background: #f0f8ff; padding: 15px; margin: 10px 0; border-radius: 4px; border-left: 4px solid #0066cc; }
                .endpoint:hover { background: #e6f3ff; }
                a { color: #0066cc; text-decoration: none; font-weight: 500; }
                a:hover { text-decoration: underline; }
                .section { margin: 30px 0; }
                .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
                .auth-info { background: #fff3cd; padding: 15px; border-radius: 4px; border-left: 4px solid #ffc107; margin: 20px 0; }
                .test-button { background: #28a745; color: white; padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; margin-left: 10px; }
                .test-button:hover { background: #218838; }
            </style>
            <script>
                async function testEndpoint(url, button) {
                    button.textContent = 'Testing...';
                    button.disabled = true;
                    try {
                        const response = await fetch(url);
                        if (response.ok) {
                            button.textContent = '✅ OK';
                            button.style.background = '#28a745';
                        } else {
                            button.textContent = '❌ Error';
                            button.style.background = '#dc3545';
                        }
                    } catch (error) {
                        button.textContent = '❌ Failed';
                        button.style.background = '#dc3545';
                    }
                    setTimeout(() => {
                        button.textContent = 'Test';
                        button.style.background = '#28a745';
                        button.disabled = false;
                    }, 3000);
                }
            </script>
        </head>
        <body>
            <div class="container">
                <h1>🚀 Enterprise LLMOps Platform</h1>
                <div class="status">
                    <h2>✅ Demo Mode Active (No Authentication Required)</h2>
                    <p>The Enterprise LLMOps platform is running in demo mode. All API endpoints are available without authentication for easy testing and development.</p>
                </div>
                
                <div class="auth-info">
                    <strong>🔐 Authentication:</strong> Disabled by default for demo purposes. 
                    To enable authentication for production, start the server with <code>--enable-auth</code> flag.
                </div>
                
                <div class="section">
                    <h2>📖 API Documentation & Schema</h2>
                    <div class="endpoint">
                        <strong>📚 Interactive API Docs (Swagger UI):</strong> 
                        <a href="/docs" target="_blank">/docs</a>
                        <button class="test-button" onclick="testEndpoint('/docs', this)">Test</button>
                    </div>
                    <div class="endpoint">
                        <strong>📋 ReDoc Documentation:</strong> 
                        <a href="/redoc" target="_blank">/redoc</a>
                        <button class="test-button" onclick="testEndpoint('/redoc', this)">Test</button>
                    </div>
                    <div class="endpoint">
                        <strong>📋 OpenAPI Schema (JSON):</strong> 
                        <a href="/openapi.json" target="_blank">/openapi.json</a>
                        <button class="test-button" onclick="testEndpoint('/openapi.json', this)">Test</button>
                    </div>
                    <div class="endpoint">
                        <strong>📊 API Information:</strong> 
                        <a href="/api/info" target="_blank">/api/info</a>
                        <button class="test-button" onclick="testEndpoint('/api/info', this)">Test</button>
                    </div>
                    <div class="endpoint">
                        <strong>🔍 System Status:</strong> 
                        <a href="/api/status" target="_blank">/api/status</a>
                        <button class="test-button" onclick="testEndpoint('/api/status', this)">Test</button>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📚 Unified Documentation Site</h2>
                    <div class="endpoint">
                        <strong>📖 MkDocs Documentation:</strong> 
                        <a href="http://localhost:8082" target="_blank">http://localhost:8082</a>
                        <button class="test-button" onclick="testEndpoint('http://localhost:8082', this)">Test</button>
                    </div>
                    <div class="endpoint">
                        <strong>🌐 GitHub Pages:</strong> 
                        <a href="https://s-n00b.github.io/ai_assignments" target="_blank">https://s-n00b.github.io/ai_assignments</a>
                        <button class="test-button" onclick="testEndpoint('https://s-n00b.github.io/ai_assignments', this)">Test</button>
                    </div>
                    <div style="background: #e8f4fd; padding: 15px; border-radius: 4px; margin: 10px 0;">
                        <strong>📋 Documentation Features:</strong>
                        <ul style="margin: 10px 0; padding-left: 20px;">
                            <li>Two-category organization (Model Enablement & AI System Architecture)</li>
                            <li>Executive presentations and professional content</li>
                            <li>Live application integration with iframe embedding</li>
                            <li>API documentation auto-generated from FastAPI</li>
                            <li>Assignment-specific documentation and guides</li>
                        </ul>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🔗 Core API Endpoints</h2>
                    <div class="grid">
                <div class="endpoint">
                            <strong>💚 Health Check:</strong> 
                            <a href="/health" target="_blank">/health</a>
                            <button class="test-button" onclick="testEndpoint('/health', this)">Test</button>
                </div>
                <div class="endpoint">
                            <strong>🤖 Model Registry:</strong> 
                            <a href="/api/models" target="_blank">/api/models</a>
                            <button class="test-button" onclick="testEndpoint('/api/models', this)">Test</button>
                </div>
                <div class="endpoint">
                            <strong>🧪 Experiments:</strong> 
                            <a href="/api/experiments" target="_blank">/api/experiments</a>
                            <button class="test-button" onclick="testEndpoint('/api/experiments', this)">Test</button>
                        </div>
                        <div class="endpoint">
                            <strong>🦙 Ollama Models:</strong> 
                            <a href="/api/ollama/models" target="_blank">/api/ollama/models</a>
                            <button class="test-button" onclick="testEndpoint('/api/ollama/models', this)">Test</button>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📊 Prompt & Dataset Management</h2>
                    <div class="grid">
                <div class="endpoint">
                            <strong>💾 Prompt Cache Summary:</strong> 
                            <a href="/api/prompts/cache/summary" target="_blank">/api/prompts/cache/summary</a>
                            <button class="test-button" onclick="testEndpoint('/api/prompts/cache/summary', this)">Test</button>
                </div>
                <div class="endpoint">
                            <strong>🔄 Sync Prompts:</strong> 
                            <a href="/api/prompts/sync" target="_blank">/api/prompts/sync</a>
                            <button class="test-button" onclick="testEndpoint('/api/prompts/sync', this)">Test</button>
                </div>
                <div class="endpoint">
                            <strong>📈 Registry Statistics:</strong> 
                            <a href="/api/prompts/registries/statistics" target="_blank">/api/prompts/registries/statistics</a>
                            <button class="test-button" onclick="testEndpoint('/api/prompts/registries/statistics', this)">Test</button>
                        </div>
                        <div class="endpoint">
                            <strong>🗄️ ChromaDB Health:</strong> 
                            <a href="/api/chromadb/health" target="_blank">/api/chromadb/health</a>
                            <button class="test-button" onclick="testEndpoint('/api/chromadb/health', this)">Test</button>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🌐 External Services</h2>
                    <div class="endpoint">
                        <strong>🎯 Model Evaluation (Gradio):</strong> 
                        <a href="http://localhost:7860" target="_blank">http://localhost:7860</a>
                        <button class="test-button" onclick="testEndpoint('http://localhost:7860', this)">Test</button>
                    </div>
                    <div class="endpoint">
                        <strong>📈 MLflow Tracking UI:</strong> 
                        <a href="http://localhost:5000" target="_blank">http://localhost:5000</a>
                        <button class="test-button" onclick="testEndpoint('http://localhost:5000', this)">Test</button>
                    </div>
                    <div class="endpoint">
                        <strong>🗄️ ChromaDB Vector Store:</strong> 
                        <a href="http://localhost:8081" target="_blank">http://localhost:8081</a>
                        <button class="test-button" onclick="testEndpoint('http://localhost:8081', this)">Test</button>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🖼️ Unified Service Integration (iframe)</h2>
                    <div style="background: #e8f4fd; padding: 15px; border-radius: 4px; margin: 10px 0;">
                        <strong>🎯 Unified Dashboard:</strong> Access all services through a single interface with tabbed navigation and service status indicators.
                    </div>
                    <div class="grid">
                        <div class="endpoint">
                            <strong>🖥️ Unified Dashboard:</strong> 
                            <a href="/iframe/dashboard" target="_blank">/iframe/dashboard</a>
                            <button class="test-button" onclick="testEndpoint('/iframe/dashboard', this)">Test</button>
                        </div>
                        <div class="endpoint">
                            <strong>🎯 Lenovo Pitch (iframe):</strong> 
                            <a href="/iframe/lenovo-pitch" target="_blank">/iframe/lenovo-pitch</a>
                            <button class="test-button" onclick="testEndpoint('/iframe/lenovo-pitch', this)">Test</button>
                        </div>
                        <div class="endpoint">
                            <strong>🤖 Gradio App (iframe):</strong> 
                            <a href="/iframe/gradio" target="_blank">/iframe/gradio</a>
                            <button class="test-button" onclick="testEndpoint('/iframe/gradio', this)">Test</button>
                        </div>
                        <div class="endpoint">
                            <strong>📈 MLflow (iframe):</strong> 
                            <a href="/iframe/mlflow" target="_blank">/iframe/mlflow</a>
                            <button class="test-button" onclick="testEndpoint('/iframe/mlflow', this)">Test</button>
                        </div>
                        <div class="endpoint">
                            <strong>🗄️ ChromaDB (iframe):</strong> 
                            <a href="/iframe/chromadb" target="_blank">/iframe/chromadb</a>
                            <button class="test-button" onclick="testEndpoint('/iframe/chromadb', this)">Test</button>
                        </div>
                        <div class="endpoint">
                            <strong>📚 Documentation (iframe):</strong> 
                            <a href="/iframe/docs" target="_blank">/iframe/docs</a>
                            <button class="test-button" onclick="testEndpoint('/iframe/docs', this)">Test</button>
                        </div>
                        <div class="endpoint">
                            <strong>🎯 LangGraph Studio (iframe):</strong> 
                            <a href="/iframe/langgraph-studio" target="_blank">/iframe/langgraph-studio</a>
                            <button class="test-button" onclick="testEndpoint('/iframe/langgraph-studio', this)">Test</button>
                        </div>
                        <div class="endpoint">
                            <strong>🔧 QLoRA Fine-Tuning (iframe):</strong> 
                            <a href="/iframe/qlora" target="_blank">/iframe/qlora</a>
                            <button class="test-button" onclick="testEndpoint('/iframe/qlora', this)">Test</button>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🔧 Advanced Features</h2>
                    <div class="grid">
                        <div class="endpoint">
                            <strong>📊 Monitoring Status:</strong> 
                            <a href="/api/monitoring/status" target="_blank">/api/monitoring/status</a>
                            <button class="test-button" onclick="testEndpoint('/api/monitoring/status', this)">Test</button>
                        </div>
                        <div class="endpoint">
                            <strong>📈 Monitoring Metrics:</strong> 
                            <a href="/api/monitoring/metrics" target="_blank">/api/monitoring/metrics</a>
                            <button class="test-button" onclick="testEndpoint('/api/monitoring/metrics', this)">Test</button>
                        </div>
                        <div class="endpoint">
                            <strong>🕸️ LangGraph Studios:</strong> 
                            <a href="/api/langgraph/studios" target="_blank">/api/langgraph/studios</a>
                            <button class="test-button" onclick="testEndpoint('/api/langgraph/studios', this)">Test</button>
                        </div>
                        <div class="endpoint">
                            <strong>🧠 Knowledge Graph Status:</strong> 
                            <a href="/api/knowledge-graph/status" target="_blank">/api/knowledge-graph/status</a>
                            <button class="test-button" onclick="testEndpoint('/api/knowledge-graph/status', this)">Test</button>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🚀 Quick Start Commands</h2>
                    <div style="background: #f8f9fa; padding: 20px; border-radius: 4px; font-family: monospace; font-size: 14px;">
                        <div><strong>Test API endpoints:</strong></div>
                        <div>curl http://localhost:8080/health</div>
                        <div>curl http://localhost:8080/api/models</div>
                        <div>curl http://localhost:8080/api/experiments</div>
                        <br>
                        <div><strong>Start with authentication:</strong></div>
                        <div>python -m src.enterprise_llmops.main --enable-auth</div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """)


@app.get("/about", response_class=HTMLResponse)
async def serve_about_page():
    """Serve the Lenovo AI Architecture about page."""
    about_file = "src/enterprise_llmops/frontend/about_page.html"
    if os.path.exists(about_file):
        return FileResponse(about_file)
    else:
        # Fallback to original pitch page if available
        pitch_file = "lenovo_ai_architecture_pitch.html"
        if os.path.exists(pitch_file):
            return FileResponse(pitch_file)
        else:
            return HTMLResponse("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Assignment - Lenovo AI Architecture</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; background: #1a1a1a; color: white; }
                    .container { max-width: 1000px; margin: 0 auto; background: #2a2a2a; padding: 40px; border-radius: 8px; border: 1px solid #404040; }
                    h1 { color: #E2231A; text-align: center; }
                    .lenovo-gradient { background: linear-gradient(135deg, #E2231A, #C01E17); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1 class="lenovo-gradient">Lenovo AI Architecture</h1>
                    <p>Meta-Assignment Portfolio: AI Architect Model Customization → Model Evaluation Engineer Testing → Lenovo Factory Roster</p>
                </div>
            </body>
            </html>
            """)


@app.get("/assignment-pdf")
async def serve_assignment_pdf():
    """Serve the Lenovo AAITC Technical Assignments PDF."""
    # Get the project root directory (3 levels up from this file)
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent
    pdf_file = project_root / "docs" / "Lenovo AAITC Technical Assignments.pdf"
    
    if pdf_file.exists():
        return FileResponse(
            str(pdf_file),
            media_type="application/pdf",
            filename="Lenovo AAITC Technical Assignments.pdf",
            headers={
                "Content-Disposition": "inline; filename=Lenovo AAITC Technical Assignments.pdf",
                "Cache-Control": "public, max-age=3600"
            }
        )
    else:
        raise HTTPException(status_code=404, detail=f"PDF file not found at {pdf_file}")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "ollama": ollama_manager is not None,
            "model_registry": model_registry is not None,
            "mlflow": mlflow_manager is not None,
            "optuna": optuna_optimizer is not None,
            "chromadb": chroma_client is not None
        }
    }


# Authentication endpoints
@app.post("/api/auth/login", response_model=TokenResponse)
async def login(request: UserLoginRequest):
    """User login endpoint."""
    # In production, validate against database
    if request.username == "admin" and request.password == "admin":
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": request.username}, expires_delta=access_token_expires
        )
        return TokenResponse(access_token=access_token, token_type="bearer")
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )


# ChromaDB endpoints
@app.get("/api/chromadb/health")
async def chromadb_health():
    """ChromaDB health check endpoint."""
    if chroma_client is None:
        return {"status": "not_available", "error": "ChromaDB client not initialized"}
    
    try:
        # Use v2 API for health check
        version = chroma_client.get_version()
        collections = chroma_client.list_collections()
        return {
            "status": "healthy",
            "version": version,
            "collections_count": len(collections),
            "collections": [{"name": col.name, "metadata": col.metadata} for col in collections],
            "api_version": "v2"
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/api/chromadb/collections")
async def list_chromadb_collections():
    """List ChromaDB collections using v2 API."""
    if chroma_client is None:
        raise HTTPException(status_code=503, detail="ChromaDB client not available")
    
    try:
        # ChromaDB v2 API - list collections
        collections = chroma_client.list_collections()
        return {
            "collections": [
                {
                    "name": col.name,
                    "id": col.id,
                    "metadata": col.metadata if hasattr(col, 'metadata') else {}
                } for col in collections
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {e}")


# ChromaDB service management endpoints
@app.post("/api/chroma/start")
async def start_chroma():
    """Start ChromaDB service"""
    try:
        # This would start ChromaDB in a subprocess
        # For now, we'll just return a success message
        return {"message": "ChromaDB start command issued", "status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start ChromaDB: {str(e)}")

@app.get("/api/chroma/health")
async def chroma_health():
    """Check ChromaDB health using v2 API"""
    try:
        if chroma_client:
            # ChromaDB v2 API - use heartbeat endpoint
            try:
                response = chroma_client.heartbeat()
                return {"status": "healthy", "version": response.get("nanosecond heartbeat", "unknown")}
            except Exception as heartbeat_error:
                # Fallback to version check
                try:
                    version = chroma_client.get_version()
                    return {"status": "healthy", "version": version}
                except:
                    return {"status": "connected", "message": "ChromaDB connected but heartbeat failed"}
        else:
            return {"status": "unavailable", "message": "ChromaDB client not initialized"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ChromaDB document operations endpoints
@app.post("/api/chromadb/collections")
async def create_chromadb_collection(request: dict):
    """Create a new ChromaDB collection using v2 API"""
    try:
        if not chroma_client:
            raise HTTPException(status_code=503, detail="ChromaDB client not available")
        
        collection_name = request.get("name", "default")
        collection = chroma_client.create_collection(name=collection_name)
        
        return {
            "message": f"Collection '{collection_name}' created successfully",
            "collection": {
                "name": collection.name,
                "id": collection.id,
                "metadata": collection.metadata
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create collection: {str(e)}")

@app.post("/api/chromadb/documents")
async def add_document_to_collection(request: dict):
    """Add document to ChromaDB collection using v2 API"""
    try:
        if not chroma_client:
            raise HTTPException(status_code=503, detail="ChromaDB client not available")
        
        text = request.get("text", "")
        collection_name = request.get("collection", "default")
        
        # Get or create collection
        try:
            collection = chroma_client.get_collection(name=collection_name)
        except:
            collection = chroma_client.create_collection(name=collection_name)
        
        # Add document
        collection.add(
            documents=[text],
            ids=[f"doc_{len(collection.get()['ids'])}"]
        )
        
        return {"message": "Document added successfully", "collection": collection_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add document: {str(e)}")

@app.post("/api/chromadb/search")
async def search_similar_documents(request: dict):
    """Search for similar documents using ChromaDB v2 API"""
    try:
        if not chroma_client:
            raise HTTPException(status_code=503, detail="ChromaDB client not available")
        
        query = request.get("query", "")
        collection_name = request.get("collection", "default")
        limit = request.get("limit", 5)
        
        # Get collection
        try:
            collection = chroma_client.get_collection(name=collection_name)
        except:
            return {"documents": [], "distances": [], "message": "Collection not found"}
        
        # Search for similar documents
        results = collection.query(
            query_texts=[query],
            n_results=limit
        )
        
        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "ids": results["ids"][0] if results["ids"] else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search documents: {str(e)}")


# LangGraph Studio endpoints
@app.post("/api/langgraph/studios/start")
async def start_langgraph_studio():
    """Start LangGraph Studio service"""
    try:
        if langgraph_studio_manager:
            success = await langgraph_studio_manager.start_studio()
            if success:
                return {"message": "LangGraph Studio started successfully"}
            else:
                raise HTTPException(status_code=500, detail="Failed to start LangGraph Studio")
        else:
            raise HTTPException(status_code=500, detail="LangGraph Studio manager not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start LangGraph Studio: {str(e)}")

@app.get("/api/langgraph/studios/status")
async def langgraph_studio_status():
    """Get LangGraph Studio status"""
    try:
        if langgraph_studio_manager:
            status = await langgraph_studio_manager.get_studio_status()
            return status
        else:
            return {"status": "unavailable", "message": "LangGraph Studio manager not initialized"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# Ollama management endpoints
@app.get("/api/ollama/models")
async def list_ollama_models(user: dict = Depends(get_current_user)):
    """List all Ollama models."""
    try:
        models = await ollama_manager.list_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ollama/models/{model_name}/pull")
async def pull_ollama_model(
    model_name: str,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user)
):
    """Pull a new Ollama model."""
    try:
        # Start pulling in background
        background_tasks.add_task(pull_model_task, model_name)
        
        return {
            "message": f"Started pulling model: {model_name}",
            "status": "started"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def pull_model_task(model_name: str):
    """Background task to pull Ollama model."""
    try:
        success = await ollama_manager.pull_model(model_name)
        
        # Broadcast update
        await connection_manager.broadcast({
            "type": "model_pull_complete",
            "model_name": model_name,
            "success": success,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logging.error(f"Failed to pull model {model_name}: {e}")
        await connection_manager.broadcast({
            "type": "model_pull_error",
            "model_name": model_name,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })


@app.post("/api/ollama/generate")
async def generate_response(
    request: Dict[str, Any],
    user: dict = Depends(get_current_user)
):
    """Generate response using Ollama."""
    try:
        model_name = request.get("model_name")
        prompt = request.get("prompt")
        parameters = request.get("parameters", {})
        
        if not model_name or not prompt:
            raise HTTPException(status_code=400, detail="model_name and prompt are required")
        
        response = await ollama_manager.generate_response(
            model_name=model_name,
            prompt=prompt,
            parameters=parameters,
            user_id=user["user_id"]
        )
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# GitHub Models endpoints
@app.get("/api/github-models/available")
async def list_github_models(user: dict = Depends(get_current_user)):
    """List available GitHub Models."""
    try:
        if not GITHUB_MODELS_AVAILABLE or not github_models_client:
            raise HTTPException(status_code=503, detail="GitHub Models integration not available")
        
        models = await github_models_client.get_available_models()
        return {"models": [{"id": model.id, "name": model.name, "provider": model.provider, "description": model.description} for model in models]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/github-models/generate")
async def generate_github_response(
    request: Dict[str, Any],
    user: dict = Depends(get_current_user)
):
    """Generate response using GitHub Models."""
    try:
        if not GITHUB_MODELS_AVAILABLE or not github_models_serving:
            raise HTTPException(status_code=503, detail="GitHub Models integration not available")
        
        model_id = request.get("model_id")
        prompt = request.get("prompt")
        parameters = request.get("parameters", {})
        
        if not model_id or not prompt:
            raise HTTPException(status_code=400, detail="model_id and prompt are required")
        
        # Create input data for GitHub Models API
        input_data = {
            "model": model_id,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": parameters.get("temperature", 0.7),
            "max_tokens": parameters.get("max_tokens", 1000)
        }
        
        # Make inference request
        response = await github_models_serving.make_inference(model_id, input_data)
        
        if response.success:
            return {
                "response": response.content,
                "model": model_id,
                "usage": response.usage,
                "timestamp": response.timestamp.isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail=response.error or "Unknown error")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Model registry endpoints
@app.get("/api/models")
async def list_models(
    model_type: Optional[str] = None,
    status: Optional[str] = None,
    tags: Optional[str] = None,
    user: dict = Depends(get_current_user)
):
    """List registered models."""
    try:
        # Parse tags if provided
        tags_list = tags.split(",") if tags else None
        
        # Convert string enums to enum objects
        model_type_enum = None
        if model_type:
            try:
                model_type_enum = ModelType(model_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid model type: {model_type}")
        
        status_enum = None
        if status:
            try:
                status_enum = ModelStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        models = await model_registry.list_models(
            model_type=model_type_enum,
            status=status_enum,
            tags=tags_list
        )
        
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/models/register")
async def register_model(
    model_data: Dict[str, Any],
    user: dict = Depends(get_current_user)
):
    """Register a new model."""
    try:
        model_name = model_data.get("name")
        model_type_str = model_data.get("model_type")
        description = model_data.get("description", "")
        
        if not model_name or not model_type_str:
            raise HTTPException(status_code=400, detail="name and model_type are required")
        
        try:
            model_type = ModelType(model_type_str)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid model type: {model_type_str}")
        
        model_version = await model_registry.register_model(
            model_name=model_name,
            model_type=model_type,
            description=description,
            created_by=user["user_id"],
            **model_data.get("additional_params", {})
        )
        
        return {"model_version": asdict(model_version)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/models/{model_name}/versions/{version}/status")
async def update_model_status(
    model_name: str,
    version: str,
    status_data: Dict[str, str],
    user: dict = Depends(get_current_user)
):
    """Update model status."""
    try:
        status_str = status_data.get("status")
        if not status_str:
            raise HTTPException(status_code=400, detail="status is required")
        
        try:
            status = ModelStatus(status_str)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status_str}")
        
        success = await model_registry.update_model_status(
            model_name=model_name,
            version=version,
            status=status,
            user_id=user["user_id"]
        )
        
        if success:
            return {"message": "Status updated successfully"}
        else:
            raise HTTPException(status_code=404, detail="Model not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# MLflow experiment tracking endpoints
@app.get("/api/experiments")
async def list_experiments(user: dict = Depends(get_current_user)):
    """List MLflow experiments."""
    try:
        # Get run history
        runs = mlflow_manager.get_run_history(max_results=100)
        
        # Group runs by experiment
        experiments = {}
        for run in runs:
            experiment_name = run.get("tags", {}).get("experiment_name", "default")
            if experiment_name not in experiments:
                experiments[experiment_name] = {
                    "name": experiment_name,
                    "runs": [],
                    "latest_run": None,
                    "best_metric": None
                }
            
            experiments[experiment_name]["runs"].append(run)
            
            # Track latest run
            if (not experiments[experiment_name]["latest_run"] or 
                run["start_time"] > experiments[experiment_name]["latest_run"]["start_time"]):
                experiments[experiment_name]["latest_run"] = run
        
        return {"experiments": list(experiments.values())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/experiments/start")
async def start_experiment(
    experiment_data: Dict[str, Any],
    user: dict = Depends(get_current_user)
):
    """Start a new experiment run."""
    try:
        run_name = experiment_data.get("run_name")
        description = experiment_data.get("description", "")
        tags = experiment_data.get("tags", {})
        
        run_id = mlflow_manager.start_run(
            run_name=run_name,
            tags=tags,
            description=description
        )
        
        return {"run_id": run_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/experiments/{run_id}/log-metrics")
async def log_experiment_metrics(
    run_id: str,
    metrics: Dict[str, float],
    user: dict = Depends(get_current_user)
):
    """Log metrics for an experiment."""
    try:
        mlflow_manager.log_llm_metrics(metrics, run_id)
        return {"message": "Metrics logged successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/experiments/{run_id}/log-params")
async def log_experiment_params(
    run_id: str,
    params: Dict[str, Any],
    user: dict = Depends(get_current_user)
):
    """Log parameters for an experiment."""
    try:
        mlflow_manager.log_llm_params(params, run_id)
        return {"message": "Parameters logged successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/experiments/{run_id}/end")
async def end_experiment(
    run_id: str,
    status: str = "FINISHED",
    user: dict = Depends(get_current_user)
):
    """End an experiment run."""
    try:
        mlflow_manager.end_run(run_id, status)
        return {"message": "Experiment ended successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Optuna optimization endpoints
@app.post("/api/optimization/start")
async def start_optimization(
    optimization_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user)
):
    """Start hyperparameter optimization."""
    try:
        study_name = optimization_data.get("study_name", "llm_optimization")
        n_trials = optimization_data.get("n_trials", 100)
        
        # Start optimization in background
        background_tasks.add_task(optimization_task, study_name, n_trials, optimization_data)
        
        return {
            "message": "Optimization started",
            "study_name": study_name,
            "status": "started"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def optimization_task(study_name: str, n_trials: int, optimization_data: Dict[str, Any]):
    """Background task for hyperparameter optimization."""
    try:
        # Create optimization config
        config = OptimizationConfig(
            study_name=study_name,
            direction="maximize",
            n_trials=n_trials
        )
        
        # Create optimizer
        optimizer = OptunaOptimizer(config)
        
        # Broadcast optimization start
        await connection_manager.broadcast({
            "type": "optimization_started",
            "study_name": study_name,
            "timestamp": datetime.now().isoformat()
        })
        
        # Run optimization (this would be implemented with actual model evaluation)
        # For now, we'll simulate the process
        
        for trial in range(n_trials):
            # Simulate trial
            await asyncio.sleep(1)
            
            # Broadcast progress
            await connection_manager.broadcast({
                "type": "optimization_progress",
                "study_name": study_name,
                "trial": trial + 1,
                "total_trials": n_trials,
                "timestamp": datetime.now().isoformat()
            })
        
        # Broadcast completion
        await connection_manager.broadcast({
            "type": "optimization_completed",
            "study_name": study_name,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logging.error(f"Optimization failed: {e}")
        await connection_manager.broadcast({
            "type": "optimization_error",
            "study_name": study_name,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })


# Monitoring and metrics endpoints
@app.get("/api/monitoring/status")
async def get_monitoring_status(user: dict = Depends(get_current_user)):
    """Get overall system monitoring status."""
    try:
        # Get Ollama status
        ollama_status = await ollama_manager.get_instance_status()
        
        # Get system metrics
        system_metrics = {
            "ollama": ollama_status,
            "model_registry": {
                "total_models": len(await model_registry.list_models()),
                "status": "healthy"
            },
            "mlflow": {
                "status": "healthy",
                "experiments": len(mlflow_manager.get_run_history())
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return system_metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/monitoring/metrics")
async def get_monitoring_metrics(
    time_range: str = "1h",
    user: dict = Depends(get_current_user)
):
    """Get monitoring metrics for the specified time range."""
    try:
        # This would integrate with Prometheus/Grafana
        # For now, return mock data
        
        metrics = {
            "cpu_usage": [{"timestamp": datetime.now().isoformat(), "value": 65.2}],
            "memory_usage": [{"timestamp": datetime.now().isoformat(), "value": 78.5}],
            "gpu_usage": [{"timestamp": datetime.now().isoformat(), "value": 45.8}],
            "model_inference_time": [{"timestamp": datetime.now().isoformat(), "value": 142.3}],
            "throughput": [{"timestamp": datetime.now().isoformat(), "value": 8.7}],
            "error_rate": [{"timestamp": datetime.now().isoformat(), "value": 0.02}]
        }
        
        return {"metrics": metrics, "time_range": time_range}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await connection_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)


# Background monitoring task
async def background_monitoring():
    """Background task for system monitoring."""
    while True:
        try:
            # Get system status
            if ollama_manager:
                status = await ollama_manager.get_instance_status()
                
                # Broadcast status update
                await connection_manager.broadcast({
                    "type": "system_status",
                    "data": status,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Wait before next update
            await asyncio.sleep(30)  # Update every 30 seconds
            
        except Exception as e:
            logging.error(f"Background monitoring error: {e}")
            await asyncio.sleep(60)  # Wait longer on error


# LangGraph Studio integration endpoint
@app.get("/api/langgraph/studios")
async def list_langgraph_studios(user: dict = Depends(get_current_user)):
    """List available LangGraph Studio instances."""
    try:
        # This would integrate with actual LangGraph Studio instances
        studios = [
            {
                "id": "studio-1",
                "name": "Model Evaluation Workflow",
                "status": "running",
                "url": "http://langgraph-studio-1:8080",
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "studio-2", 
                "name": "RAG System Workflow",
                "status": "running",
                "url": "http://langgraph-studio-2:8080",
                "created_at": datetime.now().isoformat()
            }
        ]
        
        return {"studios": studios}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Neo4j knowledge graph endpoints
@app.get("/api/knowledge-graph/status")
async def get_knowledge_graph_status(user: dict = Depends(get_current_user)):
    """Get Neo4j knowledge graph status."""
    try:
        # This would integrate with actual Neo4j instance
        status = {
            "status": "healthy",
            "nodes": 1247,
            "relationships": 2389,
            "database": "llmops_knowledge",
            "url": "http://neo4j-service:7474",
            "last_updated": datetime.now().isoformat()
        }
        
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/knowledge-graph/query")
async def query_knowledge_graph(
    query: str,
    limit: int = 100,
    user: dict = Depends(get_current_user)
):
    """Query the knowledge graph."""
    try:
        # This would execute actual Neo4j queries
        # For now, return mock data
        
        results = {
            "query": query,
            "results": [
                {
                    "type": "node",
                    "labels": ["Model"],
                    "properties": {
                        "name": "GPT-5",
                        "type": "LLM",
                        "performance_score": 0.95
                    }
                },
                {
                    "type": "relationship",
                    "start": "GPT-5",
                    "end": "Text Generation",
                    "type": "PERFORMS",
                    "properties": {"confidence": 0.92}
                }
            ],
            "execution_time_ms": 45,
            "timestamp": datetime.now().isoformat()
        }
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Documentation and API Information endpoints
@app.get("/api/info")
async def get_api_info():
    """Get comprehensive API information and service status."""
    return {
        "platform": "Enterprise LLMOps Platform",
        "version": "2.1.0",
        "description": "Comprehensive LLM Operations Platform for Lenovo AAITC",
        "services": {
            "fastapi": {
                "status": "running",
                "port": 8080,
                "docs_url": "/docs",
                "redoc_url": "/redoc"
            },
            "gradio": {
                "status": "available",
                "port": 7860,
                "url": "http://localhost:7860",
                "description": "Model Evaluation Engineer Interface"
            },
            "mlflow": {
                "status": "available", 
                "port": 5000,
                "url": "http://localhost:5000",
                "description": "Experiment Tracking UI"
            },
            "chromadb": {
                "status": "available",
                "port": 8081,
                "url": "http://localhost:8081",
                "description": "Vector Database"
            },
            "mkdocs": {
                "status": "available",
                "port": 8082,
                "url": "http://localhost:8082",
                "description": "Documentation Site"
            }
        },
        "assignments": {
            "assignment1": {
                "title": "Model Evaluation Engineer",
                "description": "Comprehensive model evaluation framework with Gradio interface",
                "url": "http://localhost:7860",
                "features": [
                    "Model evaluation pipeline",
                    "Model profiling & characterization", 
                    "Model factory architecture",
                    "Practical evaluation exercise",
                    "Real-time dashboard",
                    "Report generation"
                ]
            },
            "assignment2": {
                "title": "Enterprise LLMOps Platform",
                "description": "Production-ready enterprise platform with FastAPI backend",
                "url": "http://localhost:8080",
                "features": [
                    "Enterprise API endpoints",
                    "Model registry management",
                    "Experiment tracking",
                    "AutoML optimization",
                    "Vector database integration",
                    "Real-time monitoring",
                    "WebSocket updates"
                ]
            }
        },
        "documentation": {
            "mkdocs": {
                "url": "http://localhost:8082",
                "github_pages": "https://s-n00b.github.io/ai_assignments",
                "structure": "Two-category organization with professional content"
            },
            "api_docs": {
                "swagger_ui": "/docs",
                "redoc": "/redoc",
                "openapi_schema": "/openapi.json"
            }
        }
    }


@app.get("/api/status")
async def get_system_status():
    """Get detailed system status for all components."""
    status = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "healthy",
        "components": {
            "ollama": {
                "status": "healthy" if ollama_manager else "not_available",
                "details": await ollama_manager.get_instance_status() if ollama_manager else None
            },
            "model_registry": {
                "status": "healthy" if model_registry else "not_available",
                "total_models": len(await model_registry.list_models()) if model_registry else 0
            },
            "mlflow": {
                "status": "healthy" if mlflow_manager else "not_available",
                "experiments": len(mlflow_manager.get_run_history()) if mlflow_manager else 0
            },
            "optuna": {
                "status": "healthy" if optuna_optimizer else "not_available"
            },
            "chromadb": {
                "status": "healthy" if chroma_client else "not_available",
                "collections": len(chroma_client.list_collections()) if chroma_client else 0
            },
            "prompt_integration": {
                "status": "healthy" if prompt_integration_manager else "not_available"
            }
        }
    }
    
    # Determine overall status
    component_statuses = [comp["status"] for comp in status["components"].values()]
    if any(status == "error" for status in component_statuses):
        status["overall_status"] = "degraded"
    elif any(status == "not_available" for status in component_statuses):
        status["overall_status"] = "partial"
    
    return status


# Prompt Integration endpoints
@app.get("/api/prompts/cache/summary")
async def get_prompt_cache_summary(user: dict = Depends(get_current_user)):
    """Get summary of cached AI tool prompts."""
    try:
        if not prompt_integration_manager:
            raise HTTPException(status_code=503, detail="Prompt integration manager not initialized")
        
        summary = await prompt_integration_manager.load_cached_prompts_summary()
        return {"success": True, "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/prompts/sync")
async def sync_ai_tool_prompts(user: dict = Depends(get_current_user)):
    """Sync AI tool prompts from cache."""
    try:
        if not prompt_integration_manager:
            raise HTTPException(status_code=503, detail="Prompt integration manager not initialized")
        
        result = await prompt_integration_manager.sync_ai_tool_prompts()
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/prompts/dataset/generate")
async def generate_evaluation_dataset(
    request_data: Dict[str, Any],
    user: dict = Depends(get_current_user)
):
    """Generate enhanced evaluation dataset."""
    try:
        if not prompt_integration_manager:
            raise HTTPException(status_code=503, detail="Prompt integration manager not initialized")
        
        model_capabilities = request_data.get("model_capabilities", {})
        target_size = request_data.get("target_size", 1000)
        
        dataset = await prompt_integration_manager.generate_model_specific_dataset(
            model_capabilities=model_capabilities,
            target_size=target_size
        )
        
        # Get metrics for the dataset
        metrics = await prompt_integration_manager.get_prompt_evaluation_metrics(dataset)
        
        return {
            "success": True,
            "dataset_size": len(dataset),
            "metrics": metrics,
            "model_capabilities": model_capabilities
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/prompts/registries/statistics")
async def get_prompt_registry_statistics(user: dict = Depends(get_current_user)):
    """Get statistics about prompt registries."""
    try:
        if not prompt_integration_manager:
            raise HTTPException(status_code=503, detail="Prompt integration manager not initialized")
        
        stats = prompt_integration_manager.prompt_manager.get_registry_statistics()
        return {"success": True, "statistics": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# IFRAME SERVICE INTEGRATION ENDPOINTS
# ============================================================================

@app.get("/lenovo-pitch", response_class=HTMLResponse)
async def serve_lenovo_pitch_direct():
    """Serve the Lenovo AI Architecture pitch page directly."""
    pitch_file = "lenovo_ai_architecture_pitch.html"
    if os.path.exists(pitch_file):
        return FileResponse(pitch_file)
    else:
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Lenovo AI Architecture Pitch</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h1 { color: #333; text-align: center; }
                .status { background: #fff3cd; padding: 20px; border-radius: 4px; margin: 20px 0; border-left: 4px solid #ffc107; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎯 Lenovo AI Architecture Pitch</h1>
                <div class="status">
                    <strong>⚠️ Pitch Page Not Found</strong><br>
                    The Lenovo AI Architecture pitch page (lenovo_ai_architecture_pitch.html) is not available in the current directory.
                </div>
                <p>This endpoint serves the Lenovo pitch page for the AI Architect's enterprise platform.</p>
            </div>
        </body>
        </html>
        """)


@app.get("/iframe/lenovo-pitch", response_class=HTMLResponse)
async def serve_lenovo_pitch():
    """Serve the Lenovo AI Architecture pitch page in iframe."""
    pitch_file = "lenovo_ai_architecture_pitch.html"
    if os.path.exists(pitch_file):
        return FileResponse(pitch_file)
    else:
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Lenovo AI Architecture Pitch</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h1 { color: #333; text-align: center; }
                .status { background: #fff3cd; padding: 20px; border-radius: 4px; margin: 20px 0; border-left: 4px solid #ffc107; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎯 Lenovo AI Architecture Pitch</h1>
                <div class="status">
                    <strong>⚠️ Pitch Page Not Found</strong><br>
                    The Lenovo AI Architecture pitch page (lenovo_ai_architecture_pitch.html) is not available in the current directory.
                </div>
                <p>This iframe endpoint is designed to serve the Lenovo pitch page for the AI Architect's enterprise platform.</p>
            </div>
        </body>
        </html>
        """)


@app.get("/iframe/mlflow", response_class=HTMLResponse)
async def serve_mlflow_iframe():
    """Serve MLflow UI in an iframe with Lenovo branding and dark theme."""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Lenovo AI Architecture - MLflow Tracking</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&display=swap" rel="stylesheet">
        <style>
            :root {
                --lenovo-red: #E2231A;
                --lenovo-dark-red: #C01E17;
                --lenovo-black: #000000;
                --lenovo-gray: #666666;
                --lenovo-light-gray: #F5F5F5;
                --lenovo-white: #FFFFFF;
                --lenovo-blue: #0066CC;
                --lenovo-dark: #1A1A1A;
                --lenovo-card: #2A2A2A;
                --lenovo-border: #404040;
            }
            
            * { box-sizing: border-box; }
            
            html, body { 
                margin: 0; 
                padding: 0; 
                height: 100vh; 
                font-family: 'Inter', sans-serif;
                background-color: var(--lenovo-dark);
                color: var(--lenovo-white);
                overflow: hidden;
            }
            
            .lenovo-gradient {
                background: linear-gradient(135deg, var(--lenovo-red), var(--lenovo-dark-red));
            }
            
            .lenovo-text-gradient {
                background: linear-gradient(90deg, var(--lenovo-red), var(--lenovo-blue));
                background-clip: text;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            
            .mlflow-header {
                background: var(--lenovo-gradient);
                color: white;
                padding: 16px 20px;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                position: relative;
                z-index: 1000;
            }
            
            .mlflow-header h1 {
                margin: 0;
                font-size: 1.5rem;
                font-weight: 700;
            }
            
            .mlflow-header p {
                margin: 4px 0 0 0;
                opacity: 0.9;
                font-size: 0.9rem;
            }
            
            .mlflow-container {
                height: calc(100vh - 80px);
                position: relative;
            }
            
            iframe { 
                width: 100%; 
                height: 100%; 
                border: none;
                background: var(--lenovo-dark);
            }
            
            .loading { 
                position: absolute; 
                top: 50%; 
                left: 50%; 
                transform: translate(-50%, -50%); 
                font-family: 'Inter', sans-serif;
                color: var(--lenovo-white);
                text-align: center;
                z-index: 999;
            }
            
            .loading-spinner {
                width: 40px;
                height: 40px;
                border: 4px solid var(--lenovo-border);
                border-top: 4px solid var(--lenovo-red);
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 16px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .lenovo-logo {
                background: var(--lenovo-text-gradient);
                background-clip: text;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 900;
            }
        </style>
    </head>
    <body>
        <div class="mlflow-header">
            <h1><span class="lenovo-logo">Lenovo</span> AI Architecture - MLflow Tracking</h1>
            <p>Experiment Tracking & Model Registry</p>
        </div>
        
        <div class="mlflow-container">
            <div class="loading" id="loading">
                <div class="loading-spinner"></div>
                <div>Loading MLflow UI...</div>
            </div>
            <iframe 
                src="http://localhost:5000" 
                title="MLflow Tracking"
                sandbox="allow-same-origin allow-scripts allow-forms allow-popups"
                onload="document.getElementById('loading').style.display='none'"
                onerror="document.getElementById('loading').innerHTML='<div class=&quot;loading-spinner&quot;></div><div>Failed to load MLflow UI. Please ensure MLflow is running on port 5000.</div>'">
            </iframe>
        </div>
    </body>
    </html>
    """)


@app.get("/iframe/gradio", response_class=HTMLResponse)
async def serve_gradio_iframe():
    """Serve Gradio Model Evaluation interface in iframe with Lenovo branding."""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Lenovo AI Architecture - Model Evaluation</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&display=swap" rel="stylesheet">
        <style>
            :root {
                --lenovo-red: #E2231A;
                --lenovo-dark-red: #C01E17;
                --lenovo-black: #000000;
                --lenovo-gray: #666666;
                --lenovo-light-gray: #F5F5F5;
                --lenovo-white: #FFFFFF;
                --lenovo-blue: #0066CC;
                --lenovo-dark: #1A1A1A;
                --lenovo-card: #2A2A2A;
                --lenovo-border: #404040;
            }
            
            * { box-sizing: border-box; }
            
            html, body { 
                margin: 0; 
                padding: 0; 
                height: 100vh; 
                font-family: 'Inter', sans-serif;
                background-color: var(--lenovo-dark);
                color: var(--lenovo-white);
                overflow: hidden;
            }
            
            .lenovo-gradient {
                background: linear-gradient(135deg, var(--lenovo-red), var(--lenovo-dark-red));
            }
            
            .lenovo-text-gradient {
                background: linear-gradient(90deg, var(--lenovo-red), var(--lenovo-blue));
                background-clip: text;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            
            .gradio-header {
                background: var(--lenovo-gradient);
                color: white;
                padding: 16px 20px;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                position: relative;
                z-index: 1000;
            }
            
            .gradio-header h1 {
                margin: 0;
                font-size: 1.5rem;
                font-weight: 700;
            }
            
            .gradio-header p {
                margin: 4px 0 0 0;
                opacity: 0.9;
                font-size: 0.9rem;
            }
            
            .gradio-container {
                height: calc(100vh - 80px);
                position: relative;
            }
            
            iframe { 
                width: 100%; 
                height: 100%; 
                border: none;
                background: var(--lenovo-dark);
            }
            
            .loading { 
                position: absolute; 
                top: 50%; 
                left: 50%; 
                transform: translate(-50%, -50%); 
                font-family: 'Inter', sans-serif;
                color: var(--lenovo-white);
                text-align: center;
                z-index: 999;
            }
            
            .loading-spinner {
                width: 40px;
                height: 40px;
                border: 4px solid var(--lenovo-border);
                border-top: 4px solid var(--lenovo-red);
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 16px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .lenovo-logo {
                background: var(--lenovo-text-gradient);
                background-clip: text;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 900;
            }
        </style>
    </head>
    <body>
        <div class="gradio-header">
            <h1><span class="lenovo-logo">Lenovo</span> AI Architecture - Model Evaluation</h1>
            <p>Model Testing & Factory Roster Framework</p>
        </div>
        
        <div class="gradio-container">
            <div class="loading" id="loading">
                <div class="loading-spinner"></div>
                <div>Loading Model Evaluation Interface...</div>
            </div>
            <iframe 
                src="http://localhost:7860" 
                title="Gradio Model Evaluation"
                sandbox="allow-same-origin allow-scripts allow-forms allow-popups"
                onload="document.getElementById('loading').style.display='none'"
                onerror="document.getElementById('loading').innerHTML='<div class=&quot;loading-spinner&quot;></div><div>Failed to load Gradio app. Please ensure the Gradio app is running on port 7860.</div>'">
            </iframe>
        </div>
    </body>
    </html>
    """)


@app.get("/iframe/chromadb", response_class=HTMLResponse)
async def serve_chromadb_iframe():
    """Serve ChromaDB UI in iframe with Lenovo branding."""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Lenovo AI Architecture - ChromaDB Vector Store</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&display=swap" rel="stylesheet">
        <style>
            :root {
                --lenovo-red: #E2231A;
                --lenovo-dark-red: #C01E17;
                --lenovo-black: #000000;
                --lenovo-gray: #666666;
                --lenovo-light-gray: #F5F5F5;
                --lenovo-white: #FFFFFF;
                --lenovo-blue: #0066CC;
                --lenovo-dark: #1A1A1A;
                --lenovo-card: #2A2A2A;
                --lenovo-border: #404040;
            }
            
            * { box-sizing: border-box; }
            
            html, body { 
                margin: 0; 
                padding: 0; 
                height: 100vh; 
                font-family: 'Inter', sans-serif;
                background-color: var(--lenovo-dark);
                color: var(--lenovo-white);
                overflow: hidden;
            }
            
            .lenovo-gradient {
                background: linear-gradient(135deg, var(--lenovo-red), var(--lenovo-dark-red));
            }
            
            .lenovo-text-gradient {
                background: linear-gradient(90deg, var(--lenovo-red), var(--lenovo-blue));
                background-clip: text;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            
            .chromadb-header {
                background: var(--lenovo-gradient);
                color: white;
                padding: 16px 20px;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                position: relative;
                z-index: 1000;
            }
            
            .chromadb-header h1 {
                margin: 0;
                font-size: 1.5rem;
                font-weight: 700;
            }
            
            .chromadb-header p {
                margin: 4px 0 0 0;
                opacity: 0.9;
                font-size: 0.9rem;
            }
            
            .chromadb-container {
                height: calc(100vh - 80px);
                position: relative;
            }
            
            iframe { 
                width: 100%; 
                height: 100%; 
                border: none;
                background: var(--lenovo-dark);
            }
            
            .loading { 
                position: absolute; 
                top: 50%; 
                left: 50%; 
                transform: translate(-50%, -50%); 
                font-family: 'Inter', sans-serif;
                color: var(--lenovo-white);
                text-align: center;
                z-index: 999;
            }
            
            .loading-spinner {
                width: 40px;
                height: 40px;
                border: 4px solid var(--lenovo-border);
                border-top: 4px solid var(--lenovo-red);
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 16px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .lenovo-logo {
                background: var(--lenovo-text-gradient);
                background-clip: text;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 900;
            }
        </style>
    </head>
    <body>
        <div class="chromadb-header">
            <h1><span class="lenovo-logo">Lenovo</span> AI Architecture - ChromaDB Vector Store</h1>
            <p>Vector Database for Embeddings & Knowledge Retrieval</p>
        </div>
        
        <div class="chromadb-container">
            <div class="loading" id="loading">
                <div class="loading-spinner"></div>
                <div>Loading ChromaDB UI...</div>
            </div>
            <iframe 
                src="http://localhost:8081" 
                title="ChromaDB Vector Database"
                sandbox="allow-same-origin allow-scripts allow-forms allow-popups"
                onload="document.getElementById('loading').style.display='none'"
                onerror="document.getElementById('loading').innerHTML='<div class=&quot;loading-spinner&quot;></div><div>Failed to load ChromaDB UI. Please ensure ChromaDB is running on port 8081.</div>'">
            </iframe>
        </div>
    </body>
    </html>
    """)


@app.get("/iframe/docs", response_class=HTMLResponse)
async def serve_docs_iframe():
    """Serve MkDocs documentation in iframe with Lenovo branding."""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Lenovo AI Architecture - Documentation Hub</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&display=swap" rel="stylesheet">
        <style>
            :root {
                --lenovo-red: #E2231A;
                --lenovo-dark-red: #C01E17;
                --lenovo-black: #000000;
                --lenovo-gray: #666666;
                --lenovo-light-gray: #F5F5F5;
                --lenovo-white: #FFFFFF;
                --lenovo-blue: #0066CC;
                --lenovo-dark: #1A1A1A;
                --lenovo-card: #2A2A2A;
                --lenovo-border: #404040;
            }
            
            * { box-sizing: border-box; }
            
            html, body { 
                margin: 0; 
                padding: 0; 
                height: 100vh; 
                font-family: 'Inter', sans-serif;
                background-color: var(--lenovo-dark);
                color: var(--lenovo-white);
                overflow: hidden;
            }
            
            .lenovo-gradient {
                background: linear-gradient(135deg, var(--lenovo-red), var(--lenovo-dark-red));
            }
            
            .lenovo-text-gradient {
                background: linear-gradient(90deg, var(--lenovo-red), var(--lenovo-blue));
                background-clip: text;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            
            .docs-header {
                background: var(--lenovo-gradient);
                color: white;
                padding: 16px 20px;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                position: relative;
                z-index: 1000;
            }
            
            .docs-header h1 {
                margin: 0;
                font-size: 1.5rem;
                font-weight: 700;
            }
            
            .docs-header p {
                margin: 4px 0 0 0;
                opacity: 0.9;
                font-size: 0.9rem;
            }
            
            .docs-container {
                height: calc(100vh - 80px);
                position: relative;
            }
            
            iframe { 
                width: 100%; 
                height: 100%; 
                border: none;
                background: var(--lenovo-dark);
            }
            
            .loading { 
                position: absolute; 
                top: 50%; 
                left: 50%; 
                transform: translate(-50%, -50%); 
                font-family: 'Inter', sans-serif;
                color: var(--lenovo-white);
                text-align: center;
                z-index: 999;
            }
            
            .loading-spinner {
                width: 40px;
                height: 40px;
                border: 4px solid var(--lenovo-border);
                border-top: 4px solid var(--lenovo-red);
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 16px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .lenovo-logo {
                background: var(--lenovo-text-gradient);
                background-clip: text;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 900;
            }
        </style>
    </head>
    <body>
        <div class="docs-header">
            <h1><span class="lenovo-logo">Lenovo</span> AI Architecture - Documentation Hub</h1>
            <p>Global Documentation Platform with Unified API Reference</p>
        </div>
        
        <div class="docs-container">
            <div class="loading" id="loading">
                <div class="loading-spinner"></div>
                <div>Loading Documentation Hub...</div>
            </div>
            <iframe 
                src="http://localhost:8082" 
                title="MkDocs Documentation"
                sandbox="allow-same-origin allow-scripts allow-forms allow-popups"
                onload="document.getElementById('loading').style.display='none'"
                onerror="document.getElementById('loading').innerHTML='<div class=&quot;loading-spinner&quot;></div><div>Failed to load MkDocs. Please ensure MkDocs is running on port 8082.</div>'">
            </iframe>
        </div>
    </body>
    </html>
    """)


@app.get("/iframe/dashboard", response_class=HTMLResponse)
async def serve_unified_dashboard():
    """Serve the unified dashboard with all embedded services."""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Lenovo AI Architecture - Unified Dashboard</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&display=swap" rel="stylesheet">
        <style>
            :root {
                --lenovo-red: #E2231A;
                --lenovo-dark-red: #C01E17;
                --lenovo-black: #000000;
                --lenovo-gray: #666666;
                --lenovo-light-gray: #F5F5F5;
                --lenovo-white: #FFFFFF;
                --lenovo-blue: #0066CC;
                --lenovo-dark: #1A1A1A;
                --lenovo-card: #2A2A2A;
                --lenovo-border: #404040;
            }
            
            * { box-sizing: border-box; }
            
            html, body { 
                margin: 0; 
                padding: 0; 
                height: 100%; 
                font-family: 'Inter', sans-serif;
                background-color: var(--lenovo-dark);
                color: var(--lenovo-white);
                overflow: hidden;
            }
            
            .lenovo-gradient {
                background: linear-gradient(135deg, var(--lenovo-red), var(--lenovo-dark-red));
            }
            
            .lenovo-text-gradient {
                background: linear-gradient(90deg, var(--lenovo-red), var(--lenovo-blue));
                background-clip: text;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            
            .header { 
                background: var(--lenovo-gradient);
                color: white; 
                padding: 20px; 
                text-align: center;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            .header h1 {
                margin: 0 0 8px 0;
                font-size: 1.8rem;
                font-weight: 700;
            }
            
            .header p {
                margin: 0;
                opacity: 0.9;
                font-size: 1rem;
            }
            
            .nav-tabs { 
                background: var(--lenovo-card); 
                display: flex; 
                border-bottom: 2px solid var(--lenovo-red);
                overflow-x: auto;
            }
            
            .nav-tab { 
                background: transparent; 
                color: var(--lenovo-white); 
                padding: 16px 20px; 
                cursor: pointer; 
                border: none; 
                border-right: 1px solid var(--lenovo-border);
                font-family: 'Inter', sans-serif;
                font-weight: 500;
                transition: all 0.3s ease;
                white-space: nowrap;
                display: flex;
                align-items: center;
            }
            
            .nav-tab:hover { 
                background: rgba(226, 35, 26, 0.1);
                color: var(--lenovo-red);
            }
            
            .nav-tab.active { 
                background: var(--lenovo-red);
                color: white;
                font-weight: 600;
            }
            
            .content { height: calc(100vh - 140px); }
            .tab-content { display: none; height: 100%; }
            .tab-content.active { display: block; }
            iframe { width: 100%; height: 100%; border: none; }
            
            .status-indicator { 
                display: inline-block; 
                width: 8px; 
                height: 8px; 
                border-radius: 50%; 
                margin-right: 8px;
                transition: all 0.3s ease;
            }
            
            .status-online { background: #22c55e; }
            .status-offline { background: #ef4444; }
            .status-warning { background: #f59e0b; }
            
            .lenovo-logo {
                background: var(--lenovo-text-gradient);
                background-clip: text;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 900;
            }
        </style>
    </head>
    <body class="lenovo-dark-theme">
        <div class="header">
            <h1><span class="lenovo-logo">Lenovo</span> AI Architecture - Unified Dashboard</h1>
            <p>Enterprise LLMOps Platform with Integrated Services</p>
        </div>
        
        <div class="nav-tabs">
            <button class="nav-tab active" onclick="showTab('main')">
                <span class="status-indicator status-online"></span>Main Platform
            </button>
            <button class="nav-tab" onclick="showTab('gradio')">
                <span class="status-indicator" id="gradio-status"></span>Model Evaluation (Gradio)
            </button>
            <button class="nav-tab" onclick="showTab('mlflow')">
                <span class="status-indicator" id="mlflow-status"></span>MLflow Tracking
            </button>
            <button class="nav-tab" onclick="showTab('chromadb')">
                <span class="status-indicator" id="chromadb-status"></span>ChromaDB Vector Store
            </button>
            <button class="nav-tab" onclick="showTab('docs')">
                <span class="status-indicator" id="docs-status"></span>Documentation
            </button>
            <button class="nav-tab" onclick="showTab('pitch')">
                <span class="status-indicator status-online"></span>Lenovo Pitch
            </button>
            <button class="nav-tab" onclick="showTab('langgraph-studio')">
                <span class="status-indicator" id="langgraph-studio-status"></span>LangGraph Studio
            </button>
            <button class="nav-tab" onclick="showTab('qlora')">
                <span class="status-indicator" id="qlora-status"></span>QLoRA Fine-Tuning
            </button>
        </div>
        
        <div class="content">
            <div id="main" class="tab-content active">
                <iframe src="/" title="Main Enterprise Platform"></iframe>
            </div>
            <div id="gradio" class="tab-content">
                <iframe src="http://localhost:7860" title="Gradio Model Evaluation" onload="updateStatus('gradio', true)" onerror="updateStatus('gradio', false)"></iframe>
            </div>
            <div id="mlflow" class="tab-content">
                <iframe src="http://localhost:5000" title="MLflow Tracking" onload="updateStatus('mlflow', true)" onerror="updateStatus('mlflow', false)"></iframe>
            </div>
            <div id="chromadb" class="tab-content">
                <iframe src="http://localhost:8081" title="ChromaDB Vector Store" onload="updateStatus('chromadb', true)" onerror="updateStatus('chromadb', false)"></iframe>
            </div>
            <div id="docs" class="tab-content">
                <iframe src="http://localhost:8082" title="MkDocs Documentation" onload="updateStatus('docs', true)" onerror="updateStatus('docs', false)"></iframe>
            </div>
            <div id="pitch" class="tab-content">
                <iframe src="/iframe/lenovo-pitch" title="Lenovo AI Architecture Pitch"></iframe>
            </div>
            <div id="langgraph-studio" class="tab-content">
                <iframe src="/iframe/langgraph-studio" title="LangGraph Studio Dashboard" onload="updateStatus('langgraph-studio', true)" onerror="updateStatus('langgraph-studio', false)"></iframe>
            </div>
            <div id="qlora" class="tab-content">
                <iframe src="/iframe/qlora" title="QLoRA Fine-Tuning Dashboard" onload="updateStatus('qlora', true)" onerror="updateStatus('qlora', false)"></iframe>
            </div>
        </div>
        
        <script>
            function showTab(tabName) {
                // Hide all tab contents
                const contents = document.querySelectorAll('.tab-content');
                contents.forEach(content => content.classList.remove('active'));
                
                // Remove active class from all tabs
                const tabs = document.querySelectorAll('.nav-tab');
                tabs.forEach(tab => tab.classList.remove('active'));
                
                // Show selected tab content
                document.getElementById(tabName).classList.add('active');
                
                // Add active class to clicked tab
                event.target.classList.add('active');
            }
            
            function updateStatus(service, isOnline) {
                const statusElement = document.getElementById(service + '-status');
                if (statusElement) {
                    statusElement.className = 'status-indicator ' + (isOnline ? 'status-online' : 'status-offline');
                }
            }
            
            // Check service status on load
            window.addEventListener('load', function() {
                // The iframe onload/onerror events will update the status indicators
            });
        </script>
    </body>
    </html>
    """)


# ============================================================================
# LANGGRAPH STUDIO INTEGRATION ENDPOINTS
# ============================================================================

if LANGGRAPH_STUDIO_AVAILABLE:
    # Create LangGraph Studio endpoints
    create_langgraph_studio_endpoints(app)
    
    @app.get("/iframe/langgraph-studio", response_class=HTMLResponse)
    async def serve_langgraph_studio_iframe():
        """Serve LangGraph Studio dashboard in iframe."""
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LangGraph Studio Dashboard - Agent Visualization and Debugging</title>
            <style>
                body { margin: 0; padding: 0; height: 100vh; overflow: hidden; }
                iframe { width: 100%; height: 100vh; border: none; }
                .loading { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-family: Arial, sans-serif; }
            </style>
        </head>
        <body>
            <div class="loading" id="loading">Loading LangGraph Studio Dashboard...</div>
            <iframe 
                src="/api/langgraph/studios/dashboard" 
                title="LangGraph Studio Dashboard"
                sandbox="allow-same-origin allow-scripts allow-forms allow-popups"
                onload="document.getElementById('loading').style.display='none'"
                onerror="document.getElementById('loading').innerHTML='Failed to load LangGraph Studio Dashboard. Please ensure LangGraph Studio is properly configured.'">
            </iframe>
        </body>
        </html>
        """)
else:
    @app.get("/api/langgraph/studios/status")
    async def langgraph_studio_status():
        """Get LangGraph Studio integration status."""
        return {
            "status": "unavailable",
            "message": "LangGraph Studio integration not available. Install required dependencies.",
            "dependencies": ["langgraph-cli", "langgraph", "langchain", "fastapi"],
            "setup_instructions": "Install LangGraph CLI with: pip install langgraph-cli"
        }

# ============================================================================
# QLORA INTEGRATION ENDPOINTS
# ============================================================================

if QLORA_AVAILABLE:
    # Create QLoRA endpoints
    create_qlora_endpoints(app)
    
    @app.get("/iframe/qlora", response_class=HTMLResponse)
    async def serve_qlora_iframe():
        """Serve QLoRA dashboard in iframe."""
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>QLoRA Fine-Tuning Dashboard</title>
            <style>
                body { margin: 0; padding: 0; height: 100vh; overflow: hidden; }
                iframe { width: 100%; height: 100vh; border: none; }
                .loading { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-family: Arial, sans-serif; }
            </style>
        </head>
        <body>
            <div class="loading" id="loading">Loading QLoRA Dashboard...</div>
            <iframe 
                src="/api/qlora/dashboard" 
                title="QLoRA Fine-Tuning Dashboard"
                sandbox="allow-same-origin allow-scripts allow-forms allow-popups"
                onload="document.getElementById('loading').style.display='none'"
                onerror="document.getElementById('loading').innerHTML='Failed to load QLoRA Dashboard. Please ensure QLoRA is properly configured.'">
            </iframe>
        </body>
        </html>
        """)
else:
    @app.get("/api/qlora/status")
    async def qlora_status():
        """Get QLoRA integration status."""
        return {
            "status": "unavailable",
            "message": "QLoRA integration not available. Install required dependencies.",
            "dependencies": ["torch", "transformers", "peft", "fastapi"]
        }


# ============================================================================
# NEO4J FAKER INTEGRATION ENDPOINTS
# ============================================================================

if NEO4J_FAKER_AVAILABLE:
    # Create Neo4j Faker endpoints
    create_neo4j_faker_endpoints(app)
    
    @app.get("/iframe/neo4j-faker", response_class=HTMLResponse)
    async def serve_neo4j_faker_iframe():
        """Serve Neo4j Faker dashboard in iframe."""
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Neo4j Faker Dashboard - GraphRAG Demo</title>
            <style>
                body { margin: 0; padding: 0; height: 100vh; overflow: hidden; }
                iframe { width: 100%; height: 100vh; border: none; }
                .loading { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-family: Arial, sans-serif; }
            </style>
        </head>
        <body>
            <div class="loading" id="loading">Loading Neo4j Faker Dashboard...</div>
            <iframe 
                src="/api/neo4j-faker/dashboard" 
                title="Neo4j Faker Dashboard"
                sandbox="allow-same-origin allow-scripts allow-forms allow-popups"
                onload="document.getElementById('loading').style.display='none'"
                onerror="document.getElementById('loading').innerHTML='Failed to load Neo4j Faker Dashboard. Please ensure Neo4j and Faker are properly configured.'">
            </iframe>
        </body>
        </html>
        """)
else:
    @app.get("/api/neo4j-faker/status")
    async def neo4j_faker_status():
        """Get Neo4j Faker integration status."""
        return {
            "status": "unavailable",
            "message": "Neo4j Faker integration not available. Install required dependencies.",
            "dependencies": ["neo4j", "py2neo", "faker", "fastapi"]
        }


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the application
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
