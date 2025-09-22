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
chroma_client = None
connection_manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global ollama_manager, model_registry, mlflow_manager, optuna_optimizer, prompt_integration_manager, chroma_client
    
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
        
        # Initialize ChromaDB client
        try:
            import chromadb
            from chromadb.config import Settings
            
            chroma_client = chromadb.HttpClient(
                host="localhost",
                port=8081,
                settings=Settings(allow_reset=True)
            )
            # Test connection with v2 API
            version = chroma_client.get_version()
            logging.info(f"ChromaDB client initialized successfully (v{version})")
        except Exception as e:
            logging.warning(f"Failed to initialize ChromaDB client: {e}")
            logging.info("Continuing without ChromaDB support...")
            chroma_client = None
        
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
async def serve_frontend():
    """Serve the React frontend or development landing page."""
    frontend_file = "frontend/build/index.html"
    if os.path.exists(frontend_file):
        return FileResponse(frontend_file)
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
    """List ChromaDB collections."""
    if chroma_client is None:
        raise HTTPException(status_code=503, detail="ChromaDB client not available")
    
    try:
        collections = chroma_client.list_collections()
        return {
            "collections": [
                {
                    "name": col.name,
                    "id": col.id,
                    "metadata": col.metadata
                } for col in collections
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {e}")


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
