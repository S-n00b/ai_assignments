"""
Real-time Monitoring Dashboard for Enhanced Unified Platform

This module provides real-time monitoring functionality including
performance metrics collection, alerting, and visualization.

Key Features:
- Real-time performance metrics collection
- Service health monitoring and alerting
- Performance analytics and trend analysis
- Custom dashboard creation and management
- Integration with Prometheus and Grafana
- WebSocket-based real-time updates
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

# Import monitoring components
from ...enterprise_llmops.monitoring import (
    PrometheusMetricsCollector,
    GrafanaDashboardManager,
    AlertingManager,
    PerformanceAnalyzer,
    ServiceHealthChecker
)


@dataclass
class MetricData:
    """Real-time metric data."""
    metric_name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str]
    unit: str = ""


@dataclass
class ServiceStatus:
    """Service status information."""
    service_name: str
    status: str  # "online", "offline", "degraded"
    health_score: float
    last_check: datetime
    metrics: Dict[str, Any]
    alerts: List[str]


@dataclass
class Alert:
    """Alert information."""
    alert_id: str
    service_name: str
    metric_name: str
    severity: str  # "critical", "warning", "info"
    message: str
    value: float
    threshold: float
    timestamp: datetime
    acknowledged: bool = False


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    dashboard_id: str
    name: str
    widgets: List[Dict[str, Any]]
    refresh_interval: int  # seconds
    auto_refresh: bool = True


class RealTimeMonitoring:
    """
    Real-time Monitoring Dashboard for system monitoring.
    
    This class provides comprehensive functionality for real-time
    monitoring, alerting, and performance analytics.
    """
    
    def __init__(self, config_path: str = "config/monitoring_config.yaml"):
        """Initialize the Real-time Monitoring Dashboard."""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Initialize components
        self.metrics_collector = PrometheusMetricsCollector()
        self.dashboard_manager = GrafanaDashboardManager()
        self.alerting_manager = AlertingManager()
        self.performance_analyzer = PerformanceAnalyzer()
        self.health_checker = ServiceHealthChecker()
        
        # Real-time data storage
        self.metric_history = defaultdict(lambda: deque(maxlen=1000))  # Store last 1000 data points
        self.service_statuses = {}
        self.active_alerts = {}
        self.dashboard_configs = {}
        
        # WebSocket connections
        self.active_connections: List[WebSocket] = []
        
        # Monitoring tasks
        self.monitoring_tasks = {}
        self.is_monitoring = False
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load monitoring configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "monitoring": {
                "enabled": True,
                "collection_interval": 5,  # seconds
                "retention_hours": 24,
                "services": [
                    {"name": "FastAPI Platform", "port": 8080, "health_endpoint": "/health"},
                    {"name": "Gradio App", "port": 7860, "health_endpoint": "/health"},
                    {"name": "MLflow Tracking", "port": 5000, "health_endpoint": "/health"},
                    {"name": "ChromaDB", "port": 8081, "health_endpoint": "/health"},
                    {"name": "Neo4j", "port": 7687, "health_endpoint": "/health"},
                    {"name": "LangGraph Studio", "port": 8083, "health_endpoint": "/health"}
                ]
            },
            "metrics": {
                "system_metrics": ["cpu_usage", "memory_usage", "disk_usage", "network_io"],
                "application_metrics": ["request_rate", "response_time", "error_rate", "throughput"],
                "model_metrics": ["inference_latency", "model_accuracy", "prediction_confidence"]
            },
            "alerting": {
                "enabled": True,
                "thresholds": {
                    "cpu_usage": 0.9,
                    "memory_usage": 0.9,
                    "response_time": 1000,  # ms
                    "error_rate": 0.05
                },
                "channels": ["email", "slack", "webhook"]
            },
            "dashboards": {
                "default_dashboard": {
                    "name": "System Overview",
                    "widgets": [
                        {"type": "metric", "title": "CPU Usage", "metric": "cpu_usage"},
                        {"type": "metric", "title": "Memory Usage", "metric": "memory_usage"},
                        {"type": "graph", "title": "Request Rate", "metric": "request_rate"},
                        {"type": "graph", "title": "Response Time", "metric": "response_time"}
                    ]
                }
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for Real-time Monitoring."""
        logger = logging.getLogger("real_time_monitoring")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def start_monitoring(self) -> None:
        """Start real-time monitoring."""
        try:
            if self.is_monitoring:
                self.logger.warning("Monitoring is already running")
                return
            
            self.logger.info("Starting real-time monitoring")
            self.is_monitoring = True
            
            # Start monitoring tasks
            self.monitoring_tasks["metrics_collection"] = asyncio.create_task(
                self._collect_metrics_loop()
            )
            self.monitoring_tasks["health_checks"] = asyncio.create_task(
                self._health_check_loop()
            )
            self.monitoring_tasks["alert_processing"] = asyncio.create_task(
                self._alert_processing_loop()
            )
            self.monitoring_tasks["websocket_broadcast"] = asyncio.create_task(
                self._websocket_broadcast_loop()
            )
            
            self.logger.info("Real-time monitoring started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
            self.is_monitoring = False
    
    async def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        try:
            self.logger.info("Stopping real-time monitoring")
            self.is_monitoring = False
            
            # Cancel all monitoring tasks
            for task_name, task in self.monitoring_tasks.items():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            self.monitoring_tasks.clear()
            self.logger.info("Real-time monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop monitoring: {e}")
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            metrics = {}
            
            # Get system metrics
            system_metrics = await self.metrics_collector.collect_system_metrics()
            metrics["system"] = system_metrics
            
            # Get application metrics
            app_metrics = await self.metrics_collector.collect_application_metrics()
            metrics["application"] = app_metrics
            
            # Get model metrics
            model_metrics = await self.metrics_collector.collect_model_metrics()
            metrics["models"] = model_metrics
            
            # Calculate aggregated metrics
            metrics["summary"] = await self._calculate_summary_metrics()
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get system metrics: {e}")
            return {"error": str(e)}
    
    async def get_service_status(self) -> Dict[str, ServiceStatus]:
        """Get status of all services."""
        try:
            statuses = {}
            
            for service_config in self.config["monitoring"]["services"]:
                service_name = service_config["name"]
                port = service_config["port"]
                health_endpoint = service_config.get("health_endpoint", "/health")
                
                # Check service health
                health_result = await self.health_checker.check_service_health(
                    service_name=service_name,
                    port=port,
                    health_endpoint=health_endpoint
                )
                
                # Get service metrics
                service_metrics = await self.metrics_collector.collect_service_metrics(service_name)
                
                # Get active alerts for this service
                service_alerts = [
                    alert for alert in self.active_alerts.values()
                    if alert.service_name == service_name
                ]
                
                # Create service status
                status = ServiceStatus(
                    service_name=service_name,
                    status=health_result["status"],
                    health_score=health_result["health_score"],
                    last_check=datetime.now(),
                    metrics=service_metrics,
                    alerts=[alert.message for alert in service_alerts]
                )
                
                statuses[service_name] = status
                self.service_statuses[service_name] = status
            
            return statuses
            
        except Exception as e:
            self.logger.error(f"Failed to get service status: {e}")
            return {}
    
    async def get_active_alerts(self) -> Dict[str, Alert]:
        """Get all active alerts."""
        try:
            return self.active_alerts
            
        except Exception as e:
            self.logger.error(f"Failed to get active alerts: {e}")
            return {}
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        try:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].acknowledged = True
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to acknowledge alert: {e}")
            return False
    
    async def create_dashboard(self, config: DashboardConfig) -> bool:
        """Create a custom dashboard."""
        try:
            # Store dashboard configuration
            self.dashboard_configs[config.dashboard_id] = config
            
            # Create dashboard in Grafana if enabled
            if self.config.get("grafana", {}).get("enabled", False):
                await self.dashboard_manager.create_dashboard(config)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create dashboard: {e}")
            return False
    
    async def get_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """Get data for a specific dashboard."""
        try:
            if dashboard_id not in self.dashboard_configs:
                raise ValueError(f"Dashboard {dashboard_id} not found")
            
            config = self.dashboard_configs[dashboard_id]
            dashboard_data = {}
            
            # Get data for each widget
            for widget in config.widgets:
                widget_type = widget["type"]
                metric_name = widget["metric"]
                
                if widget_type == "metric":
                    # Get current metric value
                    current_value = await self._get_current_metric_value(metric_name)
                    dashboard_data[widget["title"]] = current_value
                
                elif widget_type == "graph":
                    # Get metric history
                    history = await self._get_metric_history(metric_name)
                    dashboard_data[widget["title"]] = history
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"Failed to get dashboard data: {e}")
            return {}
    
    async def get_performance_analytics(self, time_range: str = "1h") -> Dict[str, Any]:
        """Get performance analytics for the specified time range."""
        try:
            # Get performance data
            performance_data = await self.performance_analyzer.analyze_performance(
                time_range=time_range
            )
            
            # Calculate trends
            trends = await self.performance_analyzer.calculate_trends(
                time_range=time_range
            )
            
            # Generate insights
            insights = await self.performance_analyzer.generate_insights(
                performance_data=performance_data
            )
            
            return {
                "performance_data": performance_data,
                "trends": trends,
                "insights": insights,
                "time_range": time_range,
                "generated_at": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get performance analytics: {e}")
            return {"error": str(e)}
    
    async def websocket_endpoint(self, websocket: WebSocket):
        """WebSocket endpoint for real-time updates."""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        try:
            while True:
                # Send real-time updates
                await websocket.send_text(json.dumps({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                }))
                await asyncio.sleep(1)
                
        except WebSocketDisconnect:
            self.active_connections.remove(websocket)
        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
    
    async def _collect_metrics_loop(self) -> None:
        """Continuous metrics collection loop."""
        while self.is_monitoring:
            try:
                # Collect all metrics
                system_metrics = await self.metrics_collector.collect_system_metrics()
                app_metrics = await self.metrics_collector.collect_application_metrics()
                model_metrics = await self.metrics_collector.collect_model_metrics()
                
                # Store metrics in history
                current_time = datetime.now()
                
                for metric_name, value in system_metrics.items():
                    metric_data = MetricData(
                        metric_name=metric_name,
                        value=value,
                        timestamp=current_time,
                        tags={"source": "system"}
                    )
                    self.metric_history[metric_name].append(metric_data)
                
                for metric_name, value in app_metrics.items():
                    metric_data = MetricData(
                        metric_name=metric_name,
                        value=value,
                        timestamp=current_time,
                        tags={"source": "application"}
                    )
                    self.metric_history[metric_name].append(metric_data)
                
                for metric_name, value in model_metrics.items():
                    metric_data = MetricData(
                        metric_name=metric_name,
                        value=value,
                        timestamp=current_time,
                        tags={"source": "model"}
                    )
                    self.metric_history[metric_name].append(metric_data)
                
                # Wait for next collection interval
                await asyncio.sleep(self.config["monitoring"]["collection_interval"])
                
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def _health_check_loop(self) -> None:
        """Continuous health check loop."""
        while self.is_monitoring:
            try:
                # Check all services
                for service_config in self.config["monitoring"]["services"]:
                    service_name = service_config["name"]
                    port = service_config["port"]
                    health_endpoint = service_config.get("health_endpoint", "/health")
                    
                    # Check service health
                    health_result = await self.health_checker.check_service_health(
                        service_name=service_name,
                        port=port,
                        health_endpoint=health_endpoint
                    )
                    
                    # Update service status
                    if service_name in self.service_statuses:
                        self.service_statuses[service_name].status = health_result["status"]
                        self.service_statuses[service_name].health_score = health_result["health_score"]
                        self.service_statuses[service_name].last_check = datetime.now()
                
                # Wait for next health check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def _alert_processing_loop(self) -> None:
        """Continuous alert processing loop."""
        while self.is_monitoring:
            try:
                # Check for new alerts
                for metric_name, history in self.metric_history.items():
                    if not history:
                        continue
                    
                    # Get latest value
                    latest_metric = history[-1]
                    value = latest_metric.value
                    
                    # Check against thresholds
                    thresholds = self.config["alerting"]["thresholds"]
                    if metric_name in thresholds:
                        threshold = thresholds[metric_name]
                        
                        if value > threshold:
                            # Check if alert already exists
                            alert_exists = any(
                                alert.metric_name == metric_name and 
                                alert.service_name == latest_metric.tags.get("source", "unknown") and
                                not alert.acknowledged
                                for alert in self.active_alerts.values()
                            )
                            
                            if not alert_exists:
                                # Create new alert
                                alert_id = f"alert_{metric_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                alert = Alert(
                                    alert_id=alert_id,
                                    service_name=latest_metric.tags.get("source", "unknown"),
                                    metric_name=metric_name,
                                    severity="critical" if value > threshold * 1.5 else "warning",
                                    message=f"{metric_name} is {value:.2f}, exceeds threshold {threshold}",
                                    value=value,
                                    threshold=threshold,
                                    timestamp=datetime.now()
                                )
                                
                                self.active_alerts[alert_id] = alert
                                
                                # Send alert notification
                                await self.alerting_manager.send_alert(alert)
                
                # Clean up old alerts
                await self._cleanup_old_alerts()
                
                # Wait for next alert check
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in alert processing loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def _websocket_broadcast_loop(self) -> None:
        """Continuous WebSocket broadcast loop."""
        while self.is_monitoring:
            try:
                if self.active_connections:
                    # Prepare real-time data
                    real_time_data = {
                        "type": "metrics_update",
                        "timestamp": datetime.now().isoformat(),
                        "system_metrics": await self._get_current_system_metrics(),
                        "service_statuses": {name: asdict(status) for name, status in self.service_statuses.items()},
                        "active_alerts": len([alert for alert in self.active_alerts.values() if not alert.acknowledged])
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
    
    async def _calculate_summary_metrics(self) -> Dict[str, Any]:
        """Calculate summary metrics from collected data."""
        try:
            summary = {}
            
            # Calculate averages for key metrics
            key_metrics = ["cpu_usage", "memory_usage", "request_rate", "response_time"]
            
            for metric_name in key_metrics:
                if metric_name in self.metric_history and self.metric_history[metric_name]:
                    values = [metric.value for metric in self.metric_history[metric_name]]
                    summary[metric_name] = {
                        "current": values[-1] if values else 0,
                        "average": statistics.mean(values),
                        "min": min(values),
                        "max": max(values),
                        "trend": "up" if len(values) > 1 and values[-1] > values[-2] else "down"
                    }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to calculate summary metrics: {e}")
            return {}
    
    async def _get_current_metric_value(self, metric_name: str) -> float:
        """Get current value of a specific metric."""
        try:
            if metric_name in self.metric_history and self.metric_history[metric_name]:
                return self.metric_history[metric_name][-1].value
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to get current metric value: {e}")
            return 0.0
    
    async def _get_metric_history(self, metric_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get history of a specific metric."""
        try:
            if metric_name not in self.metric_history:
                return []
            
            history = list(self.metric_history[metric_name])[-limit:]
            return [
                {
                    "value": metric.value,
                    "timestamp": metric.timestamp.isoformat(),
                    "tags": metric.tags
                }
                for metric in history
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to get metric history: {e}")
            return []
    
    async def _get_current_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics for WebSocket broadcast."""
        try:
            metrics = {}
            
            # Get current values for key metrics
            key_metrics = ["cpu_usage", "memory_usage", "request_rate", "response_time"]
            for metric_name in key_metrics:
                metrics[metric_name] = await self._get_current_metric_value(metric_name)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get current system metrics: {e}")
            return {}
    
    async def _cleanup_old_alerts(self) -> None:
        """Clean up old acknowledged alerts."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=1)  # Keep alerts for 1 hour
            
            alerts_to_remove = []
            for alert_id, alert in self.active_alerts.items():
                if alert.acknowledged and alert.timestamp < cutoff_time:
                    alerts_to_remove.append(alert_id)
            
            for alert_id in alerts_to_remove:
                del self.active_alerts[alert_id]
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup old alerts: {e}")


# FastAPI Router for Real-time Monitoring
router = APIRouter(prefix="/monitoring", tags=["Real-time Monitoring"])

# Global monitoring instance
monitoring = RealTimeMonitoring()


@router.get("/metrics")
async def get_system_metrics():
    """Get current system metrics."""
    try:
        metrics = await monitoring.get_system_metrics()
        return JSONResponse(content=metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/services")
async def get_service_status():
    """Get status of all services."""
    try:
        statuses = await monitoring.get_service_status()
        return JSONResponse(content={name: asdict(status) for name, status in statuses.items()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_active_alerts():
    """Get all active alerts."""
    try:
        alerts = await monitoring.get_active_alerts()
        return JSONResponse(content={alert_id: asdict(alert) for alert_id, alert in alerts.items()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert."""
    try:
        success = await monitoring.acknowledge_alert(alert_id)
        return JSONResponse(content={"success": success})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dashboards")
async def create_dashboard(config: DashboardConfig):
    """Create a custom dashboard."""
    try:
        success = await monitoring.create_dashboard(config)
        return JSONResponse(content={"success": success})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboards/{dashboard_id}")
async def get_dashboard_data(dashboard_id: str):
    """Get data for a specific dashboard."""
    try:
        data = await monitoring.get_dashboard_data(dashboard_id)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics")
async def get_performance_analytics(time_range: str = "1h"):
    """Get performance analytics."""
    try:
        analytics = await monitoring.get_performance_analytics(time_range)
        return JSONResponse(content=analytics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await monitoring.websocket_endpoint(websocket)


@router.post("/start")
async def start_monitoring():
    """Start real-time monitoring."""
    try:
        await monitoring.start_monitoring()
        return JSONResponse(content={"success": True, "message": "Monitoring started"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_monitoring():
    """Stop real-time monitoring."""
    try:
        await monitoring.stop_monitoring()
        return JSONResponse(content={"success": True, "message": "Monitoring stopped"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workspace")
async def get_workspace_interface():
    """Get Real-time Monitoring workspace interface."""
    try:
        with open("src/enterprise_llmops/frontend/enhanced_unified_platform.html", "r") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
