"""
Enterprise Frontend for LLMOps Platform

This module provides a comprehensive, modern web frontend for the enterprise
LLMOps platform using FastAPI, WebSockets, and modern JavaScript frameworks.
Designed for production use with enterprise-grade features.

Key Features:
- FastAPI backend with WebSocket support
- Modern responsive UI with real-time updates
- Model management and deployment interface
- Real-time monitoring and alerting
- User authentication and authorization
- API documentation and testing interface
- Export and reporting capabilities
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, status
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn
from pydantic import BaseModel, Field
import jwt
from passlib.context import CryptContext

from .ollama_manager import OllamaManager, ModelRequest
from .model_registry import EnterpriseModelRegistry, ModelType, ModelStatus


# Pydantic models for API
class ModelRegistrationRequest(BaseModel):
    """Request model for model registration."""
    name: str = Field(..., description="Model name")
    model_type: ModelType = Field(..., description="Model type")
    description: str = Field(..., description="Model description")
    tags: List[str] = Field(default_factory=list, description="Model tags")
    version: str = Field(default="1.0.0", description="Model version")
    created_by: str = Field(default="system", description="Creator user ID")


class ModelInferenceRequest(BaseModel):
    """Request model for model inference."""
    model_name: str = Field(..., description="Model name")
    prompt: str = Field(..., description="Input prompt")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Model parameters")
    user_id: str = Field(default="default", description="User ID")
    priority: int = Field(default=2, description="Request priority (1=high, 2=medium, 3=low)")
    timeout: int = Field(default=30, description="Request timeout in seconds")


class ModelDeploymentRequest(BaseModel):
    """Request model for model deployment."""
    model_name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    environment: str = Field(..., description="Deployment environment")
    endpoint: str = Field(..., description="Deployment endpoint")
    user_id: str = Field(default="system", description="User ID")
    deployment_config: Dict[str, Any] = Field(default_factory=dict, description="Deployment configuration")


class UserLoginRequest(BaseModel):
    """Request model for user login."""
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")


class WebSocketMessage(BaseModel):
    """WebSocket message model."""
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(default_factory=dict, description="Message data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str = "anonymous"):
        """Accept WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        if user_id not in self.user_connections:
            self.user_connections[user_id] = []
        self.user_connections[user_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, user_id: str = "anonymous"):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        if user_id in self.user_connections and websocket in self.user_connections[user_id]:
            self.user_connections[user_id].remove(websocket)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific WebSocket."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logging.error(f"Error sending personal message: {e}")
    
    async def broadcast(self, message: str):
        """Broadcast message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logging.error(f"Error broadcasting message: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.active_connections.remove(connection)
    
    async def send_to_user(self, message: str, user_id: str):
        """Send message to specific user."""
        if user_id in self.user_connections:
            disconnected = []
            for connection in self.user_connections[user_id]:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logging.error(f"Error sending to user {user_id}: {e}")
                    disconnected.append(connection)
            
            # Remove disconnected connections
            for connection in disconnected:
                self.user_connections[user_id].remove(connection)


class EnterpriseLLMOpsFrontend:
    """
    Enterprise LLMOps Frontend using FastAPI.
    
    This class provides a comprehensive web interface for enterprise
    AI model operations including management, deployment, and monitoring.
    """
    
    def __init__(self, config_path: str = "config/frontend_config.yaml"):
        """Initialize the enterprise frontend."""
        self.config = self._load_config(config_path)
        self.app = self._create_fastapi_app()
        self.manager = ConnectionManager()
        self.logger = self._setup_logging()
        
        # Initialize components
        self.ollama_manager = None
        self.model_registry = None
        
        # Security
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.security = HTTPBearer()
        self.secret_key = self.config.get("security", {}).get("secret_key", "your-secret-key")
        
        # Setup routes
        self._setup_routes()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load frontend configuration."""
        default_config = {
            "app": {
                "title": "Enterprise LLMOps Platform",
                "description": "Lenovo AAITC Enterprise AI Model Operations",
                "version": "2.0.0",
                "host": "0.0.0.0",
                "port": 8000
            },
            "security": {
                "secret_key": "your-secret-key-change-in-production",
                "algorithm": "HS256",
                "access_token_expire_minutes": 30
            },
            "static_files": {
                "directory": "static",
                "mount_path": "/static"
            },
            "templates": {
                "directory": "templates"
            },
            "cors": {
                "allow_origins": ["*"],
                "allow_credentials": True,
                "allow_methods": ["*"],
                "allow_headers": ["*"]
            }
        }
        
        # In production, load from YAML file
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for frontend."""
        logger = logging.getLogger("enterprise_frontend")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _create_fastapi_app(self) -> FastAPI:
        """Create FastAPI application."""
        app = FastAPI(
            title=self.config["app"]["title"],
            description=self.config["app"]["description"],
            version=self.config["app"]["version"],
            docs_url="/api/docs",
            redoc_url="/api/redoc"
        )
        
        # Add middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config["cors"]["allow_origins"],
            allow_credentials=self.config["cors"]["allow_credentials"],
            allow_methods=self.config["cors"]["allow_methods"],
            allow_headers=self.config["cors"]["allow_headers"]
        )
        
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]
        )
        
        return app
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        # Static files
        static_dir = Path(self.config["static_files"]["directory"])
        static_dir.mkdir(exist_ok=True)
        
        self.app.mount(
            "/static",
            StaticFiles(directory=str(static_dir)),
            name="static"
        )
        
        # Templates
        templates_dir = Path(self.config["templates"]["directory"])
        templates_dir.mkdir(exist_ok=True)
        
        self.templates = Jinja2Templates(directory=str(templates_dir))
        
        # WebSocket endpoint
        @self.app.websocket("/ws/{user_id}")
        async def websocket_endpoint(websocket: WebSocket, user_id: str):
            await self.manager.connect(websocket, user_id)
            try:
                while True:
                    # Keep connection alive and handle incoming messages
                    data = await websocket.receive_text()
                    message = WebSocketMessage.parse_raw(data)
                    await self._handle_websocket_message(message, websocket, user_id)
            except WebSocketDisconnect:
                self.manager.disconnect(websocket, user_id)
        
        # API Routes
        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            """Serve main application page."""
            return self.templates.TemplateResponse("index.html", {
                "request": {},
                "title": "Enterprise LLMOps Platform"
            })
        
        # Authentication routes
        @self.app.post("/api/auth/login")
        async def login(request: UserLoginRequest):
            """User login endpoint."""
            # In production, validate against database
            if request.username == "admin" and request.password == "admin":
                access_token = self._create_access_token({"sub": request.username})
                return {"access_token": access_token, "token_type": "bearer"}
            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect username or password"
                )
        
        # Model management routes
        @self.app.get("/api/models")
        async def list_models(
            model_type: Optional[str] = None,
            status: Optional[str] = None,
            limit: int = 100
        ):
            """List all models."""
            try:
                if self.model_registry:
                    models = await self.model_registry.list_models(
                        model_type=ModelType(model_type) if model_type else None,
                        status=ModelStatus(status) if status else None,
                        limit=limit
                    )
                    return {"models": models}
                else:
                    return {"models": []}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/models/register")
        async def register_model(request: ModelRegistrationRequest):
            """Register a new model."""
            try:
                if not self.model_registry:
                    raise HTTPException(status_code=500, detail="Model registry not initialized")
                
                model_version = await self.model_registry.register_model(
                    name=request.name,
                    model_type=request.model_type,
                    description=request.description,
                    tags=request.tags,
                    version=request.version,
                    created_by=request.created_by
                )
                
                # Broadcast model registration event
                await self.manager.broadcast(json.dumps({
                    "type": "model_registered",
                    "data": {
                        "model_name": request.name,
                        "version": request.version,
                        "timestamp": datetime.now().isoformat()
                    }
                }))
                
                return {"message": "Model registered successfully", "model": model_version}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/models/inference")
        async def model_inference(request: ModelInferenceRequest):
            """Perform model inference."""
            try:
                if not self.ollama_manager:
                    raise HTTPException(status_code=500, detail="Ollama manager not initialized")
                
                response = await self.ollama_manager.generate_response(
                    model_name=request.model_name,
                    prompt=request.prompt,
                    parameters=request.parameters,
                    user_id=request.user_id
                )
                
                # Broadcast inference event
                await self.manager.broadcast(json.dumps({
                    "type": "inference_completed",
                    "data": {
                        "model_name": request.model_name,
                        "user_id": request.user_id,
                        "success": response.get("success", False),
                        "timestamp": datetime.now().isoformat()
                    }
                }))
                
                return response
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/models/deploy")
        async def deploy_model(request: ModelDeploymentRequest):
            """Deploy a model."""
            try:
                if not self.model_registry:
                    raise HTTPException(status_code=500, detail="Model registry not initialized")
                
                deployment = await self.model_registry.deploy_model(
                    model_name=request.model_name,
                    version=request.version,
                    environment=request.environment,
                    endpoint=request.endpoint,
                    user_id=request.user_id,
                    **request.deployment_config
                )
                
                # Broadcast deployment event
                await self.manager.broadcast(json.dumps({
                    "type": "model_deployed",
                    "data": {
                        "model_name": request.model_name,
                        "version": request.version,
                        "environment": request.environment,
                        "deployment_id": deployment.deployment_id,
                        "timestamp": datetime.now().isoformat()
                    }
                }))
                
                return {"message": "Model deployed successfully", "deployment": deployment}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/models/{model_name}/versions")
        async def list_model_versions(model_name: str):
            """List versions of a specific model."""
            try:
                if not self.model_registry:
                    raise HTTPException(status_code=500, detail="Model registry not initialized")
                
                versions = await self.model_registry.list_model_versions(model_name)
                return {"model_name": model_name, "versions": versions}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/deployments")
        async def list_deployments(
            model_name: Optional[str] = None,
            environment: Optional[str] = None,
            status: Optional[str] = None
        ):
            """List model deployments."""
            try:
                if not self.model_registry:
                    raise HTTPException(status_code=500, detail="Model registry not initialized")
                
                deployments = await self.model_registry.list_deployments(
                    model_name=model_name,
                    environment=environment,
                    status=status
                )
                return {"deployments": deployments}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Ollama management routes
        @self.app.get("/api/ollama/models")
        async def list_ollama_models():
            """List Ollama models."""
            try:
                if not self.ollama_manager:
                    raise HTTPException(status_code=500, detail="Ollama manager not initialized")
                
                models = await self.ollama_manager.list_models()
                return {"models": models}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/ollama/models/pull")
        async def pull_ollama_model(model_name: str):
            """Pull a model from Ollama registry."""
            try:
                if not self.ollama_manager:
                    raise HTTPException(status_code=500, detail="Ollama manager not initialized")
                
                success = await self.ollama_manager.pull_model(model_name)
                
                # Broadcast model pull event
                await self.manager.broadcast(json.dumps({
                    "type": "model_pulled",
                    "data": {
                        "model_name": model_name,
                        "success": success,
                        "timestamp": datetime.now().isoformat()
                    }
                }))
                
                return {"message": f"Model {model_name} pulled successfully", "success": success}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/api/ollama/models/{model_name}")
        async def delete_ollama_model(model_name: str):
            """Delete an Ollama model."""
            try:
                if not self.ollama_manager:
                    raise HTTPException(status_code=500, detail="Ollama manager not initialized")
                
                success = await self.ollama_manager.delete_model(model_name)
                
                # Broadcast model deletion event
                await self.manager.broadcast(json.dumps({
                    "type": "model_deleted",
                    "data": {
                        "model_name": model_name,
                        "success": success,
                        "timestamp": datetime.now().isoformat()
                    }
                }))
                
                return {"message": f"Model {model_name} deleted successfully", "success": success}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/ollama/status")
        async def get_ollama_status():
            """Get Ollama service status."""
            try:
                if not self.ollama_manager:
                    raise HTTPException(status_code=500, detail="Ollama manager not initialized")
                
                status = await self.ollama_manager.get_instance_status()
                return status
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Monitoring routes
        @self.app.get("/api/monitoring/metrics")
        async def get_monitoring_metrics():
            """Get monitoring metrics."""
            try:
                if not self.ollama_manager:
                    raise HTTPException(status_code=500, detail="Ollama manager not initialized")
                
                status = await self.ollama_manager.get_instance_status()
                return {
                    "ollama_status": status,
                    "timestamp": datetime.now().isoformat(),
                    "system_health": "healthy"  # In production, get real system metrics
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Audit and compliance routes
        @self.app.get("/api/audit/log")
        async def get_audit_log(
            model_name: Optional[str] = None,
            action: Optional[str] = None,
            user_id: Optional[str] = None,
            limit: int = 100
        ):
            """Get audit log."""
            try:
                if not self.model_registry:
                    raise HTTPException(status_code=500, detail="Model registry not initialized")
                
                audit_log = await self.model_registry.get_audit_log(
                    model_name=model_name,
                    action=action,
                    user_id=user_id,
                    limit=limit
                )
                return {"audit_log": audit_log}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _handle_websocket_message(self, message: WebSocketMessage, websocket: WebSocket, user_id: str):
        """Handle incoming WebSocket messages."""
        try:
            if message.type == "ping":
                # Respond to ping
                response = WebSocketMessage(
                    type="pong",
                    data={"timestamp": datetime.now().isoformat()}
                )
                await self.manager.send_personal_message(response.json(), websocket)
            
            elif message.type == "subscribe":
                # Handle subscription to specific events
                event_type = message.data.get("event_type")
                # In production, implement event subscription logic
                
            elif message.type == "get_status":
                # Send current status
                if self.ollama_manager:
                    status = await self.ollama_manager.get_instance_status()
                    response = WebSocketMessage(
                        type="status_update",
                        data=status
                    )
                    await self.manager.send_personal_message(response.json(), websocket)
            
        except Exception as e:
            self.logger.error(f"Error handling WebSocket message: {e}")
    
    def _create_access_token(self, data: Dict[str, Any]):
        """Create JWT access token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.config["security"]["access_token_expire_minutes"])
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.config["security"]["algorithm"])
        return encoded_jwt
    
    async def initialize_components(self):
        """Initialize Ollama manager and model registry."""
        try:
            # Initialize Ollama manager
            self.ollama_manager = OllamaManager()
            await self.ollama_manager.initialize()
            
            # Initialize model registry
            self.model_registry = EnterpriseModelRegistry()
            
            self.logger.info("Enterprise frontend components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def start_background_tasks(self):
        """Start background monitoring tasks."""
        # Start periodic status updates
        asyncio.create_task(self._periodic_status_updates())
    
    async def _periodic_status_updates(self):
        """Send periodic status updates via WebSocket."""
        while True:
            try:
                if self.ollama_manager and self.manager.active_connections:
                    status = await self.ollama_manager.get_instance_status()
                    
                    message = WebSocketMessage(
                        type="status_update",
                        data=status
                    )
                    
                    await self.manager.broadcast(message.json())
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in periodic status updates: {e}")
                await asyncio.sleep(30)
    
    def run(self, host: str = None, port: int = None):
        """Run the FastAPI application."""
        host = host or self.config["app"]["host"]
        port = port or self.config["app"]["port"]
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
    
    async def shutdown(self):
        """Shutdown the frontend and cleanup resources."""
        try:
            if self.ollama_manager:
                await self.ollama_manager.shutdown()
            
            if self.model_registry:
                self.model_registry.close()
            
            self.logger.info("Enterprise frontend shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


def create_enterprise_frontend(config_path: str = "config/frontend_config.yaml") -> EnterpriseLLMOpsFrontend:
    """
    Create and return the enterprise frontend instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        EnterpriseLLMOpsFrontend instance
    """
    return EnterpriseLLMOpsFrontend(config_path)


if __name__ == "__main__":
    # Create and run the enterprise frontend
    frontend = create_enterprise_frontend()
    
    async def main():
        await frontend.initialize_components()
        await frontend.start_background_tasks()
        frontend.run()
    
    asyncio.run(main())
