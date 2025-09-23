"""
Unified Data Flow Visualization for Enhanced Unified Platform

This module provides data flow visualization functionality including
real-time data flow monitoring, service integration tracking, and
visual representation of the entire system architecture.

Key Features:
- Real-time data flow visualization
- Service integration matrix and status
- Data flow performance analytics
- Interactive system architecture diagrams
- Integration with all platform components
- Mermaid diagram generation and rendering
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import yaml
from pathlib import Path
import aiohttp
import requests
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import websockets
from collections import defaultdict, deque
import statistics

# Import data flow components
from ...ai_architecture.unified_data_flow import (
    ChromaDBVectorIntegration,
    Neo4jGraphIntegration,
    DuckDBAnalyticsIntegration,
    MLflowExperimentIntegration,
    DataSynchronizationManager
)


@dataclass
class DataFlowNode:
    """Data flow node representation."""
    node_id: str
    node_type: str  # "service", "database", "model", "workflow"
    name: str
    status: str  # "online", "offline", "degraded"
    position: Tuple[int, int]
    metadata: Dict[str, Any]


@dataclass
class DataFlowEdge:
    """Data flow edge representation."""
    edge_id: str
    source_node: str
    target_node: str
    data_type: str
    flow_rate: float  # data units per second
    latency: float  # milliseconds
    status: str  # "active", "inactive", "degraded"


@dataclass
class ServiceIntegration:
    """Service integration information."""
    service_name: str
    port: int
    url: str
    purpose: str
    data_flow: str
    integration_status: str
    health_score: float
    last_check: datetime
    metrics: Dict[str, Any]


@dataclass
class DataFlowMetrics:
    """Data flow performance metrics."""
    total_flow_rate: float
    average_latency: float
    data_throughput: float
    error_rate: float
    active_connections: int
    timestamp: datetime


class UnifiedDataFlowVisualization:
    """
    Unified Data Flow Visualization for system architecture.
    
    This class provides comprehensive functionality for visualizing
    data flow, service integration, and system architecture.
    """
    
    def __init__(self, config_path: str = "config/data_flow_config.yaml"):
        """Initialize the Unified Data Flow Visualization."""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Initialize data flow components
        self.chromadb_integration = ChromaDBVectorIntegration()
        self.neo4j_integration = Neo4jGraphIntegration()
        self.duckdb_integration = DuckDBAnalyticsIntegration()
        self.mlflow_integration = MLflowExperimentIntegration()
        self.sync_manager = DataSynchronizationManager()
        
        # Data flow visualization data
        self.data_flow_nodes = {}
        self.data_flow_edges = {}
        self.service_integrations = {}
        self.data_flow_metrics = {}
        
        # WebSocket connections
        self.active_connections: List[WebSocket] = []
        
        # Visualization tasks
        self.visualization_tasks = {}
        self.is_visualizing = False
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load data flow configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "data_flow": {
                "enabled": True,
                "update_interval": 5,  # seconds
                "retention_hours": 24,
                "services": [
                    {
                        "name": "FastAPI Platform",
                        "port": 8080,
                        "url": "http://localhost:8080",
                        "purpose": "Main enterprise platform",
                        "data_flow": "Central hub",
                        "node_type": "service"
                    },
                    {
                        "name": "Gradio Evaluation",
                        "port": 7860,
                        "url": "http://localhost:7860",
                        "purpose": "Model evaluation interface",
                        "data_flow": "Direct integration",
                        "node_type": "service"
                    },
                    {
                        "name": "MLflow Tracking",
                        "port": 5000,
                        "url": "http://localhost:5000",
                        "purpose": "Experiment tracking",
                        "data_flow": "All experiments",
                        "node_type": "service"
                    },
                    {
                        "name": "ChromaDB",
                        "port": 8081,
                        "url": "http://localhost:8081",
                        "purpose": "Vector database",
                        "data_flow": "RAG workflows",
                        "node_type": "database"
                    },
                    {
                        "name": "Neo4j",
                        "port": 7687,
                        "url": "http://localhost:7687",
                        "purpose": "Graph database",
                        "data_flow": "Knowledge graphs",
                        "node_type": "database"
                    },
                    {
                        "name": "LangGraph Studio",
                        "port": 8083,
                        "url": "http://localhost:8083",
                        "purpose": "Agent visualization",
                        "data_flow": "Agent workflows",
                        "node_type": "service"
                    }
                ]
            },
            "visualization": {
                "mermaid_enabled": True,
                "interactive_diagrams": True,
                "real_time_updates": True,
                "auto_layout": True
            },
            "metrics": {
                "flow_rate_tracking": True,
                "latency_monitoring": True,
                "throughput_analysis": True,
                "error_tracking": True
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for Unified Data Flow Visualization."""
        logger = logging.getLogger("unified_data_flow_visualization")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def start_visualization(self) -> None:
        """Start data flow visualization."""
        try:
            if self.is_visualizing:
                self.logger.warning("Visualization is already running")
                return
            
            self.logger.info("Starting unified data flow visualization")
            self.is_visualizing = True
            
            # Initialize data flow nodes and edges
            await self._initialize_data_flow_structure()
            
            # Start visualization tasks
            self.visualization_tasks["data_flow_monitoring"] = asyncio.create_task(
                self._monitor_data_flow_loop()
            )
            self.visualization_tasks["service_integration_check"] = asyncio.create_task(
                self._service_integration_check_loop()
            )
            self.visualization_tasks["metrics_collection"] = asyncio.create_task(
                self._collect_metrics_loop()
            )
            self.visualization_tasks["websocket_broadcast"] = asyncio.create_task(
                self._websocket_broadcast_loop()
            )
            
            self.logger.info("Unified data flow visualization started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start visualization: {e}")
            self.is_visualizing = False
    
    async def stop_visualization(self) -> None:
        """Stop data flow visualization."""
        try:
            self.logger.info("Stopping unified data flow visualization")
            self.is_visualizing = False
            
            # Cancel all visualization tasks
            for task_name, task in self.visualization_tasks.items():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            self.visualization_tasks.clear()
            self.logger.info("Unified data flow visualization stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop visualization: {e}")
    
    async def get_data_flow_diagram(self) -> str:
        """Generate Mermaid data flow diagram."""
        try:
            # Generate Mermaid diagram
            diagram = await self._generate_mermaid_diagram()
            return diagram
            
        except Exception as e:
            self.logger.error(f"Failed to generate data flow diagram: {e}")
            return "graph TB\n    A[Error] --> B[Failed to generate diagram]"
    
    async def get_service_integration_matrix(self) -> Dict[str, ServiceIntegration]:
        """Get service integration matrix."""
        try:
            return self.service_integrations
            
        except Exception as e:
            self.logger.error(f"Failed to get service integration matrix: {e}")
            return {}
    
    async def get_data_flow_metrics(self) -> DataFlowMetrics:
        """Get current data flow metrics."""
        try:
            # Calculate current metrics
            total_flow_rate = sum(edge.flow_rate for edge in self.data_flow_edges.values())
            average_latency = statistics.mean([edge.latency for edge in self.data_flow_edges.values()]) if self.data_flow_edges else 0
            data_throughput = await self._calculate_data_throughput()
            error_rate = await self._calculate_error_rate()
            active_connections = len(self.data_flow_edges)
            
            metrics = DataFlowMetrics(
                total_flow_rate=total_flow_rate,
                average_latency=average_latency,
                data_throughput=data_throughput,
                error_rate=error_rate,
                active_connections=active_connections,
                timestamp=datetime.now()
            )
            
            self.data_flow_metrics[str(datetime.now())] = metrics
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get data flow metrics: {e}")
            return DataFlowMetrics(
                total_flow_rate=0.0,
                average_latency=0.0,
                data_throughput=0.0,
                error_rate=0.0,
                active_connections=0,
                timestamp=datetime.now()
            )
    
    async def get_system_architecture(self) -> Dict[str, Any]:
        """Get complete system architecture information."""
        try:
            architecture = {
                "nodes": {node_id: asdict(node) for node_id, node in self.data_flow_nodes.items()},
                "edges": {edge_id: asdict(edge) for edge_id, edge in self.data_flow_edges.items()},
                "services": {name: asdict(service) for name, service in self.service_integrations.items()},
                "metrics": asdict(await self.get_data_flow_metrics()),
                "diagram": await self.get_data_flow_diagram(),
                "timestamp": datetime.now()
            }
            
            return architecture
            
        except Exception as e:
            self.logger.error(f"Failed to get system architecture: {e}")
            return {"error": str(e)}
    
    async def websocket_endpoint(self, websocket: WebSocket):
        """WebSocket endpoint for real-time data flow updates."""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        try:
            while True:
                # Send real-time data flow updates
                await websocket.send_text(json.dumps({
                    "type": "data_flow_update",
                    "timestamp": datetime.now().isoformat(),
                    "nodes": {node_id: asdict(node) for node_id, node in self.data_flow_nodes.items()},
                    "edges": {edge_id: asdict(edge) for edge_id, edge in self.data_flow_edges.items()},
                    "metrics": asdict(await self.get_data_flow_metrics())
                }))
                await asyncio.sleep(5)  # Update every 5 seconds
                
        except WebSocketDisconnect:
            self.active_connections.remove(websocket)
        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
    
    async def _initialize_data_flow_structure(self) -> None:
        """Initialize data flow nodes and edges."""
        try:
            # Initialize service nodes
            for service_config in self.config["data_flow"]["services"]:
                node_id = f"node_{service_config['name'].lower().replace(' ', '_')}"
                node = DataFlowNode(
                    node_id=node_id,
                    node_type=service_config.get("node_type", "service"),
                    name=service_config["name"],
                    status="online",
                    position=(0, 0),  # Will be calculated by auto-layout
                    metadata={
                        "port": service_config["port"],
                        "url": service_config["url"],
                        "purpose": service_config["purpose"],
                        "data_flow": service_config["data_flow"]
                    }
                )
                self.data_flow_nodes[node_id] = node
                
                # Create service integration
                service_integration = ServiceIntegration(
                    service_name=service_config["name"],
                    port=service_config["port"],
                    url=service_config["url"],
                    purpose=service_config["purpose"],
                    data_flow=service_config["data_flow"],
                    integration_status="online",
                    health_score=1.0,
                    last_check=datetime.now(),
                    metrics={}
                )
                self.service_integrations[service_config["name"]] = service_integration
            
            # Initialize data flow edges
            await self._initialize_data_flow_edges()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize data flow structure: {e}")
    
    async def _initialize_data_flow_edges(self) -> None:
        """Initialize data flow edges between services."""
        try:
            # Define data flow connections
            connections = [
                # FastAPI Platform connections
                ("node_fastapi_platform", "node_chromadb", "vector_data", 100.0, 50.0),
                ("node_fastapi_platform", "node_neo4j", "graph_data", 50.0, 75.0),
                ("node_fastapi_platform", "node_mlflow_tracking", "experiment_data", 25.0, 100.0),
                ("node_fastapi_platform", "node_gradio_evaluation", "evaluation_requests", 200.0, 30.0),
                
                # Gradio Evaluation connections
                ("node_gradio_evaluation", "node_fastapi_platform", "evaluation_results", 150.0, 40.0),
                ("node_gradio_evaluation", "node_mlflow_tracking", "evaluation_metrics", 75.0, 60.0),
                
                # ChromaDB connections
                ("node_chromadb", "node_fastapi_platform", "retrieved_vectors", 80.0, 45.0),
                
                # Neo4j connections
                ("node_neo4j", "node_fastapi_platform", "graph_queries", 60.0, 70.0),
                
                # MLflow connections
                ("node_mlflow_tracking", "node_fastapi_platform", "experiment_updates", 30.0, 90.0),
                
                # LangGraph Studio connections
                ("node_langgraph_studio", "node_fastapi_platform", "workflow_data", 40.0, 80.0),
                ("node_fastapi_platform", "node_langgraph_studio", "workflow_requests", 35.0, 85.0)
            ]
            
            # Create edges
            for i, (source, target, data_type, flow_rate, latency) in enumerate(connections):
                edge_id = f"edge_{i+1}"
                edge = DataFlowEdge(
                    edge_id=edge_id,
                    source_node=source,
                    target_node=target,
                    data_type=data_type,
                    flow_rate=flow_rate,
                    latency=latency,
                    status="active"
                )
                self.data_flow_edges[edge_id] = edge
                
        except Exception as e:
            self.logger.error(f"Failed to initialize data flow edges: {e}")
    
    async def _generate_mermaid_diagram(self) -> str:
        """Generate Mermaid diagram for data flow visualization."""
        try:
            diagram_lines = [
                "graph TB",
                "    %% Data Generation Layer",
                "    subgraph \"Data Generation Layer\"",
                "        A[Enterprise Data Generators] --> B[ChromaDB Vector Store]",
                "        A --> C[Neo4j Graph Database]",
                "        A --> D[DuckDB Analytics]",
                "        A --> E[MLflow Experiments]",
                "    end",
                "",
                "    %% AI Architect Layer",
                "    subgraph \"AI Architect Layer\"",
                "        F[AI Architect] --> G[Model Customization]",
                "        G --> H[Fine-tuning Pipeline]",
                "        G --> I[QLoRA Adapters]",
                "        G --> J[Custom Embeddings]",
                "        G --> K[Hybrid RAG]",
                "        G --> L[LangChain/LlamaIndex]",
                "        G --> M[SmolAgent Workflows]",
                "        G --> N[LangGraph Workflows]",
                "    end",
                "",
                "    %% Model Evaluation Layer",
                "    subgraph \"Model Evaluation Layer\"",
                "        O[Model Evaluation Engineer] --> P[Raw Model Testing]",
                "        O --> Q[Custom Model Testing]",
                "        O --> R[Agentic Workflow Testing]",
                "        O --> S[Retrieval Workflow Testing]",
                "        P --> T[Factory Roster]",
                "        Q --> T",
                "        R --> T",
                "        S --> T",
                "    end",
                "",
                "    %% MLflow Experiment Tracking",
                "    subgraph \"MLflow Experiment Tracking\"",
                "        E --> U[All Experiments]",
                "        H --> U",
                "        I --> U",
                "        J --> U",
                "        K --> U",
                "        L --> U",
                "        M --> U",
                "        N --> U",
                "        P --> U",
                "        Q --> U",
                "        R --> U",
                "        S --> U",
                "    end",
                "",
                "    %% Production Deployment",
                "    subgraph \"Production Deployment\"",
                "        T --> V[Production Models]",
                "        V --> W[Real-time Monitoring]",
                "        W --> X[Performance Analytics]",
                "    end",
                "",
                "    %% Data Flow Connections",
                "    B --> K",
                "    C --> K",
                "    D --> K",
                "    K --> L",
                "    L --> M",
                "    M --> N",
                "    N --> R",
                "    L --> S"
            ]
            
            return "\\n".join(diagram_lines)
            
        except Exception as e:
            self.logger.error(f"Failed to generate Mermaid diagram: {e}")
            return "graph TB\\n    A[Error] --> B[Failed to generate diagram]"
    
    async def _monitor_data_flow_loop(self) -> None:
        """Continuous data flow monitoring loop."""
        while self.is_visualizing:
            try:
                # Update data flow metrics
                for edge_id, edge in self.data_flow_edges.items():
                    # Simulate real-time flow rate variations
                    base_flow_rate = edge.flow_rate
                    variation = (hash(str(datetime.now())) % 20 - 10) / 100  # ±10% variation
                    edge.flow_rate = max(0, base_flow_rate * (1 + variation))
                    
                    # Simulate latency variations
                    base_latency = edge.latency
                    latency_variation = (hash(str(datetime.now())) % 10 - 5) / 100  # ±5% variation
                    edge.latency = max(1, base_latency * (1 + latency_variation))
                
                # Update node statuses
                for node_id, node in self.data_flow_nodes.items():
                    # Simulate occasional status changes
                    if hash(str(datetime.now())) % 100 < 5:  # 5% chance of status change
                        statuses = ["online", "degraded", "offline"]
                        current_index = statuses.index(node.status)
                        node.status = statuses[(current_index + 1) % len(statuses)]
                
                # Wait for next update
                await asyncio.sleep(self.config["data_flow"]["update_interval"])
                
            except Exception as e:
                self.logger.error(f"Error in data flow monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def _service_integration_check_loop(self) -> None:
        """Continuous service integration check loop."""
        while self.is_visualizing:
            try:
                # Check service health
                for service_name, service in self.service_integrations.items():
                    try:
                        # Simulate health check
                        health_score = 0.8 + (hash(str(datetime.now())) % 20) / 100  # 0.8-1.0
                        service.health_score = health_score
                        service.last_check = datetime.now()
                        
                        # Update integration status
                        if health_score > 0.9:
                            service.integration_status = "online"
                        elif health_score > 0.7:
                            service.integration_status = "degraded"
                        else:
                            service.integration_status = "offline"
                        
                        # Update metrics
                        service.metrics = {
                            "response_time": 50 + (hash(str(datetime.now())) % 100),
                            "error_rate": max(0, (hash(str(datetime.now())) % 5) / 100),
                            "throughput": 100 + (hash(str(datetime.now())) % 50)
                        }
                        
                    except Exception as e:
                        self.logger.error(f"Health check failed for {service_name}: {e}")
                        service.integration_status = "offline"
                        service.health_score = 0.0
                
                # Wait for next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in service integration check loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def _collect_metrics_loop(self) -> None:
        """Continuous metrics collection loop."""
        while self.is_visualizing:
            try:
                # Collect data flow metrics
                metrics = await self.get_data_flow_metrics()
                self.data_flow_metrics[str(datetime.now())] = metrics
                
                # Clean up old metrics (keep last 24 hours)
                cutoff_time = datetime.now() - timedelta(hours=24)
                metrics_to_remove = []
                for timestamp_str, metric in self.data_flow_metrics.items():
                    if datetime.fromisoformat(timestamp_str) < cutoff_time:
                        metrics_to_remove.append(timestamp_str)
                
                for timestamp_str in metrics_to_remove:
                    del self.data_flow_metrics[timestamp_str]
                
                # Wait for next collection
                await asyncio.sleep(60)  # Collect every minute
                
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def _websocket_broadcast_loop(self) -> None:
        """Continuous WebSocket broadcast loop."""
        while self.is_visualizing:
            try:
                if self.active_connections:
                    # Prepare real-time data flow data
                    real_time_data = {
                        "type": "data_flow_update",
                        "timestamp": datetime.now().isoformat(),
                        "nodes": {node_id: asdict(node) for node_id, node in self.data_flow_nodes.items()},
                        "edges": {edge_id: asdict(edge) for edge_id, edge in self.data_flow_edges.items()},
                        "services": {name: asdict(service) for name, service in self.service_integrations.items()},
                        "metrics": asdict(await self.get_data_flow_metrics())
                    }
                    
                    # Broadcast to all connected clients
                    disconnected = []
                    for websocket in self.active_connections:
                        try:
                            await websocket.send_text(json.dumps(real_time_data))
                        except Exception:
                            disconnected.append(websocket)
                    
                    # Remove disconnected clients
                    for websocket in disconnected:
                        self.active_connections.remove(websocket)
                
                # Wait for next broadcast
                await asyncio.sleep(5)  # Broadcast every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in WebSocket broadcast loop: {e}")
                await asyncio.sleep(10)  # Wait 10 seconds before retry
    
    async def _calculate_data_throughput(self) -> float:
        """Calculate current data throughput."""
        try:
            total_throughput = sum(edge.flow_rate for edge in self.data_flow_edges.values())
            return total_throughput
            
        except Exception as e:
            self.logger.error(f"Failed to calculate data throughput: {e}")
            return 0.0
    
    async def _calculate_error_rate(self) -> float:
        """Calculate current error rate."""
        try:
            # Simulate error rate calculation
            total_connections = len(self.data_flow_edges)
            if total_connections == 0:
                return 0.0
            
            # Count degraded/offline connections
            error_connections = sum(1 for edge in self.data_flow_edges.values() if edge.status != "active")
            return error_connections / total_connections
            
        except Exception as e:
            self.logger.error(f"Failed to calculate error rate: {e}")
            return 0.0


# FastAPI Router for Unified Data Flow Visualization
router = APIRouter(prefix="/data-flow", tags=["Unified Data Flow Visualization"])

# Global visualization instance
visualization = UnifiedDataFlowVisualization()


@router.get("/diagram")
async def get_data_flow_diagram():
    """Get Mermaid data flow diagram."""
    try:
        diagram = await visualization.get_data_flow_diagram()
        return JSONResponse(content={"diagram": diagram})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/services")
async def get_service_integration_matrix():
    """Get service integration matrix."""
    try:
        services = await visualization.get_service_integration_matrix()
        return JSONResponse(content={name: asdict(service) for name, service in services.items()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_data_flow_metrics():
    """Get current data flow metrics."""
    try:
        metrics = await visualization.get_data_flow_metrics()
        return JSONResponse(content=asdict(metrics))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/architecture")
async def get_system_architecture():
    """Get complete system architecture."""
    try:
        architecture = await visualization.get_system_architecture()
        return JSONResponse(content=architecture)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data flow updates."""
    await visualization.websocket_endpoint(websocket)


@router.post("/start")
async def start_visualization():
    """Start data flow visualization."""
    try:
        await visualization.start_visualization()
        return JSONResponse(content={"success": True, "message": "Visualization started"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_visualization():
    """Stop data flow visualization."""
    try:
        await visualization.stop_visualization()
        return JSONResponse(content={"success": True, "message": "Visualization stopped"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workspace")
async def get_workspace_interface():
    """Get Unified Data Flow workspace interface."""
    try:
        with open("src/enterprise_llmops/frontend/enhanced_unified_platform.html", "r") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
