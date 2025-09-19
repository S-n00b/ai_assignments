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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, BackgroundTasks
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

# Import our enterprise components
from ..ollama_manager import OllamaManager
from ..model_registry import EnterpriseModelRegistry, ModelStatus, ModelType
from ..automl.optuna_optimizer import OptunaOptimizer, OptimizationConfig, LLMHyperparameterSpace
from ..mlops.mlflow_manager import MLflowManager, ExperimentConfig, ModelInfo, DeploymentConfig


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
connection_manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global ollama_manager, model_registry, mlflow_manager, optuna_optimizer
    
    # Startup
    logging.info("Starting Enterprise LLMOps Frontend...")
    
    try:
        # Initialize Ollama manager
        ollama_manager = OllamaManager()
        await ollama_manager.initialize()
        logging.info("Ollama manager initialized")
        
        # Initialize model registry
        model_registry = EnterpriseModelRegistry()
        logging.info("Model registry initialized")
        
        # Initialize MLflow manager
        mlflow_config = ExperimentConfig(
            experiment_name="llmops_enterprise",
            tracking_uri="http://mlflow-tracking-service:5000",
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
    description="Comprehensive LLM Operations Platform for Lenovo AAITC",
    version="2.0.0",
    lifespan=lifespan
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

# Security
security = HTTPBearer(auto_error=False)


# Dependency to get current user (simplified for demo)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from token (simplified implementation)."""
    # In production, implement proper JWT token validation
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    return {"user_id": "demo_user", "role": "admin"}


# Static files
app.mount("/static", StaticFiles(directory="frontend/build/static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the React frontend."""
    return FileResponse("frontend/build/index.html")


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
            "optuna": optuna_optimizer is not None
        }
    }


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
                "url": "http://langgraph-studio-1:8000",
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "studio-2", 
                "name": "RAG System Workflow",
                "status": "running",
                "url": "http://langgraph-studio-2:8000",
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
