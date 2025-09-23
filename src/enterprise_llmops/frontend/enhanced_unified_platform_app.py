"""
Enhanced Unified Platform FastAPI Application

This module provides the main FastAPI application for the Enhanced Unified Platform
integrating all Phase 6 components including AI Architect workspace, Model Evaluation
workspace, Factory Roster Dashboard, Real-time Monitoring, and Unified Data Flow Visualization.

Key Features:
- Complete Phase 6 integration
- AI Architect and Model Evaluation workspaces
- Factory roster management and deployment
- Real-time monitoring and analytics
- Unified data flow visualization
- WebSocket support for real-time updates
- Comprehensive API documentation
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import yaml
from pathlib import Path
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import Phase 6 components
from .ai_architect_workspace import router as ai_architect_router, AIArchitectWorkspace
from .model_evaluation_workspace import router as model_evaluation_router, ModelEvaluationWorkspace
from .factory_roster_dashboard import router as factory_roster_router, FactoryRosterDashboard
from .real_time_monitoring import router as monitoring_router, RealTimeMonitoring
from .unified_data_flow_visualization import router as data_flow_router, UnifiedDataFlowVisualization

# Import existing components
from ..main import app as enterprise_app
from ..ollama_manager import OllamaManager
from ..mlops.mlflow_manager import MLflowManager
from ..model_registry import EnterpriseModelRegistry


class EnhancedUnifiedPlatform:
    """
    Enhanced Unified Platform integrating all Phase 6 components.
    
    This class provides the main application logic for the Enhanced Unified Platform
    with comprehensive integration of all workspaces and services.
    """
    
    def __init__(self, config_path: str = "config/enhanced_platform_config.yaml"):
        """Initialize the Enhanced Unified Platform."""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Initialize workspaces
        self.ai_architect_workspace = AIArchitectWorkspace()
        self.model_evaluation_workspace = ModelEvaluationWorkspace()
        self.factory_roster_dashboard = FactoryRosterDashboard()
        self.real_time_monitoring = RealTimeMonitoring()
        self.data_flow_visualization = UnifiedDataFlowVisualization()
        
        # Initialize existing components
        self.ollama_manager = OllamaManager()
        self.mlflow_manager = MLflowManager()
        self.model_registry = EnterpriseModelRegistry()
        
        # Application state
        self.is_initialized = False
        self.startup_time = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load Enhanced Platform configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "platform": {
                "name": "Lenovo AI Architecture - Enhanced Unified Platform",
                "version": "2.0.0",
                "description": "Complete Phase 6 implementation with AI Architect and Model Evaluation workspaces",
                "startup_timeout": 30,  # seconds
                "health_check_interval": 30  # seconds
            },
            "workspaces": {
                "ai_architect": {
                    "enabled": True,
                    "port": 8080,
                    "path": "/ai-architect"
                },
                "model_evaluation": {
                    "enabled": True,
                    "port": 8080,
                    "path": "/model-evaluation"
                },
                "factory_roster": {
                    "enabled": True,
                    "port": 8080,
                    "path": "/factory-roster"
                },
                "monitoring": {
                    "enabled": True,
                    "port": 8080,
                    "path": "/monitoring"
                },
                "data_flow": {
                    "enabled": True,
                    "port": 8080,
                    "path": "/data-flow"
                }
            },
            "services": {
                "fastapi_platform": {
                    "port": 8080,
                    "url": "http://localhost:8080",
                    "health_endpoint": "/health"
                },
                "gradio_app": {
                    "port": 7860,
                    "url": "http://localhost:7860",
                    "health_endpoint": "/health"
                },
                "mlflow_tracking": {
                    "port": 5000,
                    "url": "http://localhost:5000",
                    "health_endpoint": "/health"
                },
                "chromadb": {
                    "port": 8081,
                    "url": "http://localhost:8081",
                    "health_endpoint": "/health"
                },
                "neo4j": {
                    "port": 7687,
                    "url": "http://localhost:7687",
                    "health_endpoint": "/health"
                },
                "langgraph_studio": {
                    "port": 8083,
                    "url": "http://localhost:8083",
                    "health_endpoint": "/health"
                }
            },
            "integration": {
                "auto_start_services": True,
                "service_dependencies": True,
                "health_monitoring": True,
                "performance_tracking": True
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for Enhanced Unified Platform."""
        logger = logging.getLogger("enhanced_unified_platform")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def initialize(self) -> bool:
        """Initialize the Enhanced Unified Platform."""
        try:
            self.logger.info("Initializing Enhanced Unified Platform")
            self.startup_time = datetime.now()
            
            # Initialize existing components
            await self.ollama_manager.initialize()
            await self.mlflow_manager.initialize()
            await self.model_registry.initialize()
            
            # Initialize workspaces
            await self.ai_architect_workspace.initialize()
            await self.model_evaluation_workspace.initialize()
            await self.factory_roster_dashboard.initialize()
            await self.real_time_monitoring.initialize()
            await self.data_flow_visualization.initialize()
            
            # Start monitoring and visualization
            if self.config["integration"]["health_monitoring"]:
                await self.real_time_monitoring.start_monitoring()
            
            if self.config["integration"]["performance_tracking"]:
                await self.data_flow_visualization.start_visualization()
            
            self.is_initialized = True
            self.logger.info("Enhanced Unified Platform initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Enhanced Unified Platform: {e}")
            return False
    
    async def get_platform_status(self) -> Dict[str, Any]:
        """Get comprehensive platform status."""
        try:
            if not self.is_initialized:
                return {"status": "not_initialized", "error": "Platform not initialized"}
            
            # Get workspace statuses
            ai_architect_status = await self.ai_architect_workspace.get_workspace_status()
            model_evaluation_status = await self.model_evaluation_workspace.get_evaluation_results()
            factory_roster_status = await self.factory_roster_dashboard.get_dashboard_metrics()
            monitoring_status = await self.real_time_monitoring.get_system_metrics()
            data_flow_status = await self.data_flow_visualization.get_system_architecture()
            
            # Get service health
            service_health = await self._check_service_health()
            
            # Calculate overall health score
            overall_health = self._calculate_overall_health_score(
                ai_architect_status,
                model_evaluation_status,
                factory_roster_status,
                monitoring_status,
                data_flow_status,
                service_health
            )
            
            return {
                "platform": {
                    "name": self.config["platform"]["name"],
                    "version": self.config["platform"]["version"],
                    "status": "operational" if overall_health > 0.8 else "degraded" if overall_health > 0.5 else "critical",
                    "health_score": overall_health,
                    "uptime": (datetime.now() - self.startup_time).total_seconds() if self.startup_time else 0,
                    "initialized": self.is_initialized
                },
                "workspaces": {
                    "ai_architect": ai_architect_status,
                    "model_evaluation": model_evaluation_status,
                    "factory_roster": factory_roster_status,
                    "monitoring": monitoring_status,
                    "data_flow": data_flow_status
                },
                "services": service_health,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get platform status: {e}")
            return {"error": str(e)}
    
    async def _check_service_health(self) -> Dict[str, Any]:
        """Check health of all integrated services."""
        try:
            service_health = {}
            
            for service_name, service_config in self.config["services"].items():
                try:
                    # Simulate health check (in real implementation, would make actual HTTP requests)
                    health_score = 0.8 + (hash(str(datetime.now())) % 20) / 100  # 0.8-1.0
                    
                    service_health[service_name] = {
                        "name": service_name,
                        "url": service_config["url"],
                        "port": service_config["port"],
                        "status": "online" if health_score > 0.9 else "degraded" if health_score > 0.7 else "offline",
                        "health_score": health_score,
                        "last_check": datetime.now(),
                        "response_time": 50 + (hash(str(datetime.now())) % 100),  # 50-150ms
                        "error_rate": max(0, (hash(str(datetime.now())) % 5) / 100)  # 0-5%
                    }
                    
                except Exception as e:
                    service_health[service_name] = {
                        "name": service_name,
                        "url": service_config["url"],
                        "port": service_config["port"],
                        "status": "offline",
                        "health_score": 0.0,
                        "last_check": datetime.now(),
                        "error": str(e)
                    }
            
            return service_health
            
        except Exception as e:
            self.logger.error(f"Failed to check service health: {e}")
            return {}
    
    def _calculate_overall_health_score(self, *statuses) -> float:
        """Calculate overall platform health score."""
        try:
            # Extract health scores from statuses
            health_scores = []
            
            for status in statuses:
                if isinstance(status, dict):
                    if "health_score" in status:
                        health_scores.append(status["health_score"])
                    elif "performance_metrics" in status:
                        # Calculate health score from performance metrics
                        metrics = status["performance_metrics"]
                        if metrics:
                            # Simple health score calculation
                            health_score = 0.8  # Base score
                            if "error_rate" in metrics and metrics["error_rate"] < 0.05:
                                health_score += 0.1
                            if "response_time" in metrics and metrics["response_time"] < 1000:
                                health_score += 0.1
                            health_scores.append(min(1.0, health_score))
            
            # Calculate average health score
            if health_scores:
                return sum(health_scores) / len(health_scores)
            else:
                return 0.5  # Default score if no health data available
                
        except Exception as e:
            self.logger.error(f"Failed to calculate overall health score: {e}")
            return 0.0


# Create FastAPI application
app = FastAPI(
    title="Lenovo AI Architecture - Enhanced Unified Platform",
    description="Complete Phase 6 implementation with AI Architect and Model Evaluation workspaces",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global platform instance
platform = EnhancedUnifiedPlatform()


# Include routers
app.include_router(ai_architect_router)
app.include_router(model_evaluation_router)
app.include_router(factory_roster_router)
app.include_router(monitoring_router)
app.include_router(data_flow_router)

# Include existing enterprise app routes
app.include_router(enterprise_app.router)


@app.on_event("startup")
async def startup_event():
    """Initialize platform on startup."""
    try:
        success = await platform.initialize()
        if not success:
            raise Exception("Failed to initialize Enhanced Unified Platform")
    except Exception as e:
        print(f"Startup error: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint with platform overview."""
    try:
        status = await platform.get_platform_status()
        return JSONResponse(content=status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        if not platform.is_initialized:
            raise HTTPException(status_code=503, detail="Platform not initialized")
        
        status = await platform.get_platform_status()
        health_score = status.get("platform", {}).get("health_score", 0.0)
        
        if health_score > 0.8:
            return JSONResponse(content={"status": "healthy", "health_score": health_score})
        elif health_score > 0.5:
            return JSONResponse(content={"status": "degraded", "health_score": health_score})
        else:
            raise HTTPException(status_code=503, detail=f"Platform unhealthy: {health_score}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def get_status():
    """Get comprehensive platform status."""
    try:
        status = await platform.get_platform_status()
        return JSONResponse(content=status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/workspace")
async def get_unified_workspace():
    """Get unified workspace interface."""
    try:
        with open("src/enterprise_llmops/frontend/enhanced_unified_platform.html", "r") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/info")
async def get_api_info():
    """Get comprehensive API information."""
    try:
        info = {
            "platform": {
                "name": "Lenovo AI Architecture - Enhanced Unified Platform",
                "version": "2.0.0",
                "description": "Complete Phase 6 implementation with AI Architect and Model Evaluation workspaces",
                "phase": "Phase 6 - Unified Platform Integration with Clear Data Flow"
            },
            "workspaces": {
                "ai_architect": {
                    "name": "AI Architect Workspace",
                    "description": "Model customization, fine-tuning, and deployment management",
                    "endpoints": [
                        "GET /ai-architect/status",
                        "POST /ai-architect/customize-model",
                        "POST /ai-architect/create-workflow",
                        "GET /ai-architect/customizations",
                        "GET /ai-architect/workflows"
                    ]
                },
                "model_evaluation": {
                    "name": "Model Evaluation Engineer Workspace",
                    "description": "Comprehensive model testing and evaluation framework",
                    "endpoints": [
                        "GET /model-evaluation/status",
                        "POST /model-evaluation/evaluate-model",
                        "POST /model-evaluation/deploy-to-factory",
                        "GET /model-evaluation/evaluations",
                        "GET /model-evaluation/deployments"
                    ]
                },
                "factory_roster": {
                    "name": "Factory Roster Dashboard",
                    "description": "Production model deployment and management",
                    "endpoints": [
                        "GET /factory-roster/dashboard",
                        "POST /factory-roster/deploy",
                        "POST /factory-roster/stop/{deployment_id}",
                        "POST /factory-roster/rollback/{deployment_id}",
                        "GET /factory-roster/deployments"
                    ]
                },
                "monitoring": {
                    "name": "Real-time Monitoring Dashboard",
                    "description": "Performance metrics collection and alerting",
                    "endpoints": [
                        "GET /monitoring/metrics",
                        "GET /monitoring/services",
                        "GET /monitoring/alerts",
                        "POST /monitoring/start",
                        "POST /monitoring/stop",
                        "WS /monitoring/ws"
                    ]
                },
                "data_flow": {
                    "name": "Unified Data Flow Visualization",
                    "description": "Real-time data flow monitoring and visualization",
                    "endpoints": [
                        "GET /data-flow/diagram",
                        "GET /data-flow/services",
                        "GET /data-flow/metrics",
                        "GET /data-flow/architecture",
                        "WS /data-flow/ws"
                    ]
                }
            },
            "services": {
                "fastapi_platform": {"port": 8080, "url": "http://localhost:8080"},
                "gradio_app": {"port": 7860, "url": "http://localhost:7860"},
                "mlflow_tracking": {"port": 5000, "url": "http://localhost:5000"},
                "chromadb": {"port": 8081, "url": "http://localhost:8081"},
                "neo4j": {"port": 7687, "url": "http://localhost:7687"},
                "langgraph_studio": {"port": 8083, "url": "http://localhost:8083"}
            },
            "integration": {
                "data_flow": "Unified data flow across all components",
                "mlflow_tracking": "All experiments tracked in MLflow",
                "real_time_monitoring": "Performance metrics collection and alerting",
                "service_integration": "Complete service integration matrix",
                "factory_roster": "Production-ready model deployment pipeline"
            },
            "documentation": {
                "api_docs": "http://localhost:8080/docs",
                "redoc": "http://localhost:8080/redoc",
                "workspace": "http://localhost:8080/workspace"
            }
        }
        
        return JSONResponse(content=info)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "src.enterprise_llmops.frontend.enhanced_unified_platform_app:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
