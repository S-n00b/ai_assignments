"""
Factory Roster Dashboard for Enhanced Unified Platform

This module provides factory roster management functionality including
production model deployment, monitoring, and performance analytics.

Key Features:
- Production model deployment and management
- Real-time monitoring and alerting
- Performance analytics and reporting
- Model versioning and rollback capabilities
- Environment management (production, staging, testing)
- Integration with MLflow model registry
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
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse

# Import Factory Roster components
from ...model_evaluation.factory_roster_management import (
    ProductionDeploymentManager,
    ModelVersioningManager,
    EnvironmentManager,
    PerformanceAnalytics,
    AlertingSystem,
    ModelRegistryIntegration
)


@dataclass
class DeploymentRequest:
    """Request for model deployment."""
    model_id: str
    model_name: str
    version: str
    deployment_type: str  # "production", "staging", "testing"
    target_environment: str
    configuration: Dict[str, Any]
    monitoring_enabled: bool = True
    auto_scaling: bool = False


@dataclass
class MonitoringConfig:
    """Configuration for model monitoring."""
    model_id: str
    metrics_enabled: List[str]  # ["latency", "throughput", "accuracy", "memory", "cpu"]
    alerting_enabled: bool
    alert_thresholds: Dict[str, float]
    reporting_interval: int  # seconds


@dataclass
class DeploymentStatus:
    """Status of model deployment."""
    deployment_id: str
    model_id: str
    model_name: str
    version: str
    deployment_type: str
    target_environment: str
    status: str  # "deployed", "deploying", "failed", "stopped"
    health_status: str  # "healthy", "degraded", "unhealthy"
    performance_metrics: Dict[str, Any]
    created_at: datetime
    last_updated: datetime
    monitoring_url: Optional[str] = None


@dataclass
class PerformanceReport:
    """Performance report for deployed models."""
    model_id: str
    model_name: str
    report_period: str
    metrics: Dict[str, Any]
    trends: Dict[str, Any]
    recommendations: List[str]
    generated_at: datetime


class FactoryRosterDashboard:
    """
    Factory Roster Dashboard for production model management.
    
    This class provides comprehensive functionality for managing
    production model deployments, monitoring, and analytics.
    """
    
    def __init__(self, config_path: str = "config/factory_roster_config.yaml"):
        """Initialize the Factory Roster Dashboard."""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Initialize components
        self.deployment_manager = ProductionDeploymentManager()
        self.versioning_manager = ModelVersioningManager()
        self.environment_manager = EnvironmentManager()
        self.performance_analytics = PerformanceAnalytics()
        self.alerting_system = AlertingSystem()
        self.model_registry = ModelRegistryIntegration()
        
        # Active deployments and monitoring
        self.active_deployments = {}
        self.monitoring_configs = {}
        self.performance_reports = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load Factory Roster configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "deployment": {
                "environments": ["production", "staging", "testing"],
                "auto_scaling": {
                    "enabled": True,
                    "min_instances": 1,
                    "max_instances": 10,
                    "scale_up_threshold": 0.8,
                    "scale_down_threshold": 0.3
                },
                "health_checks": {
                    "enabled": True,
                    "interval": 30,  # seconds
                    "timeout": 10,  # seconds
                    "retries": 3
                }
            },
            "monitoring": {
                "metrics": ["latency", "throughput", "accuracy", "memory", "cpu"],
                "alerting": {
                    "enabled": True,
                    "channels": ["email", "slack", "webhook"],
                    "thresholds": {
                        "latency": 1000,  # ms
                        "error_rate": 0.05,  # 5%
                        "memory_usage": 0.9,  # 90%
                        "cpu_usage": 0.9  # 90%
                    }
                },
                "reporting": {
                    "enabled": True,
                    "interval": 3600,  # 1 hour
                    "retention_days": 30
                }
            },
            "model_registry": {
                "integration": True,
                "auto_versioning": True,
                "rollback_enabled": True
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for Factory Roster Dashboard."""
        logger = logging.getLogger("factory_roster_dashboard")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def deploy_model(self, request: DeploymentRequest) -> DeploymentStatus:
        """Deploy model to factory roster."""
        try:
            self.logger.info(f"Deploying {request.model_name} to {request.deployment_type}")
            
            # Validate deployment request
            await self._validate_deployment_request(request)
            
            # Configure deployment
            deployment_config = {
                "model_id": request.model_id,
                "model_name": request.model_name,
                "version": request.version,
                "deployment_type": request.deployment_type,
                "target_environment": request.target_environment,
                "configuration": request.configuration,
                "monitoring_enabled": request.monitoring_enabled,
                "auto_scaling": request.auto_scaling
            }
            
            # Deploy model
            deployment_result = await self.deployment_manager.deploy_model(deployment_config)
            
            # Setup monitoring if enabled
            monitoring_url = None
            if request.monitoring_enabled:
                monitoring_config = MonitoringConfig(
                    model_id=request.model_id,
                    metrics_enabled=self.config["monitoring"]["metrics"],
                    alerting_enabled=self.config["monitoring"]["alerting"]["enabled"],
                    alert_thresholds=self.config["monitoring"]["alerting"]["thresholds"],
                    reporting_interval=self.config["monitoring"]["reporting"]["interval"]
                )
                monitoring_result = await self._setup_monitoring(monitoring_config)
                monitoring_url = monitoring_result.get("monitoring_url")
            
            # Register in model registry
            if self.config["model_registry"]["integration"]:
                await self.model_registry.register_deployment(
                    model_id=request.model_id,
                    version=request.version,
                    deployment_type=request.deployment_type,
                    target_environment=request.target_environment
                )
            
            # Create deployment status
            deployment_id = f"deploy_{request.model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            deployment_status = DeploymentStatus(
                deployment_id=deployment_id,
                model_id=request.model_id,
                model_name=request.model_name,
                version=request.version,
                deployment_type=request.deployment_type,
                target_environment=request.target_environment,
                status="deployed",
                health_status="healthy",
                performance_metrics=deployment_result.get("metrics", {}),
                created_at=datetime.now(),
                last_updated=datetime.now(),
                monitoring_url=monitoring_url
            )
            
            # Store deployment
            self.active_deployments[deployment_id] = deployment_status
            
            # Start monitoring if enabled
            if request.monitoring_enabled:
                asyncio.create_task(self._start_monitoring(deployment_id))
            
            return deployment_status
            
        except Exception as e:
            self.logger.error(f"Model deployment failed: {e}")
            return DeploymentStatus(
                deployment_id="",
                model_id=request.model_id,
                model_name=request.model_name,
                version=request.version,
                deployment_type=request.deployment_type,
                target_environment=request.target_environment,
                status="failed",
                health_status="unhealthy",
                performance_metrics={},
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
    
    async def stop_deployment(self, deployment_id: str) -> bool:
        """Stop model deployment."""
        try:
            self.logger.info(f"Stopping deployment: {deployment_id}")
            
            if deployment_id not in self.active_deployments:
                raise ValueError(f"Deployment {deployment_id} not found")
            
            # Stop deployment
            await self.deployment_manager.stop_deployment(deployment_id)
            
            # Update status
            deployment = self.active_deployments[deployment_id]
            deployment.status = "stopped"
            deployment.last_updated = datetime.now()
            
            # Stop monitoring
            if deployment_id in self.monitoring_configs:
                await self._stop_monitoring(deployment_id)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop deployment: {e}")
            return False
    
    async def rollback_deployment(self, deployment_id: str, target_version: str) -> bool:
        """Rollback deployment to previous version."""
        try:
            self.logger.info(f"Rolling back {deployment_id} to version {target_version}")
            
            if deployment_id not in self.active_deployments:
                raise ValueError(f"Deployment {deployment_id} not found")
            
            # Get current deployment
            deployment = self.active_deployments[deployment_id]
            
            # Rollback to target version
            rollback_result = await self.deployment_manager.rollback_deployment(
                deployment_id=deployment_id,
                target_version=target_version
            )
            
            if rollback_result["success"]:
                # Update deployment status
                deployment.version = target_version
                deployment.last_updated = datetime.now()
                
                # Update model registry
                if self.config["model_registry"]["integration"]:
                    await self.model_registry.update_deployment_version(
                        deployment_id=deployment_id,
                        new_version=target_version
                    )
                
                return True
            else:
                return False
            
        except Exception as e:
            self.logger.error(f"Failed to rollback deployment: {e}")
            return False
    
    async def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentStatus]:
        """Get status of specific deployment."""
        try:
            if deployment_id not in self.active_deployments:
                return None
            
            deployment = self.active_deployments[deployment_id]
            
            # Update performance metrics
            if deployment.monitoring_url:
                metrics = await self._get_latest_metrics(deployment_id)
                deployment.performance_metrics = metrics
                deployment.last_updated = datetime.now()
                
                # Update health status based on metrics
                deployment.health_status = self._calculate_health_status(metrics)
            
            return deployment
            
        except Exception as e:
            self.logger.error(f"Failed to get deployment status: {e}")
            return None
    
    async def get_all_deployments(self) -> Dict[str, DeploymentStatus]:
        """Get all active deployments."""
        try:
            # Update all deployments with latest metrics
            for deployment_id in self.active_deployments:
                deployment = self.active_deployments[deployment_id]
                if deployment.monitoring_url:
                    metrics = await self._get_latest_metrics(deployment_id)
                    deployment.performance_metrics = metrics
                    deployment.health_status = self._calculate_health_status(metrics)
                    deployment.last_updated = datetime.now()
            
            return self.active_deployments
            
        except Exception as e:
            self.logger.error(f"Failed to get all deployments: {e}")
            return {}
    
    async def generate_performance_report(self, model_id: str, report_period: str = "24h") -> PerformanceReport:
        """Generate performance report for deployed model."""
        try:
            self.logger.info(f"Generating performance report for {model_id}")
            
            # Get performance data
            performance_data = await self.performance_analytics.get_performance_data(
                model_id=model_id,
                period=report_period
            )
            
            # Calculate trends
            trends = await self.performance_analytics.calculate_trends(
                model_id=model_id,
                period=report_period
            )
            
            # Generate recommendations
            recommendations = await self.performance_analytics.generate_recommendations(
                model_id=model_id,
                performance_data=performance_data
            )
            
            # Create performance report
            report = PerformanceReport(
                model_id=model_id,
                model_name=performance_data.get("model_name", "Unknown"),
                report_period=report_period,
                metrics=performance_data.get("metrics", {}),
                trends=trends,
                recommendations=recommendations,
                generated_at=datetime.now()
            )
            
            # Store report
            report_id = f"report_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.performance_reports[report_id] = report
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {e}")
            return PerformanceReport(
                model_id=model_id,
                model_name="Unknown",
                report_period=report_period,
                metrics={},
                trends={},
                recommendations=[],
                generated_at=datetime.now()
            )
    
    async def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get dashboard metrics and status."""
        try:
            # Get deployment statistics
            total_deployments = len(self.active_deployments)
            healthy_deployments = sum(1 for d in self.active_deployments.values() if d.health_status == "healthy")
            degraded_deployments = sum(1 for d in self.active_deployments.values() if d.health_status == "degraded")
            unhealthy_deployments = sum(1 for d in self.active_deployments.values() if d.health_status == "unhealthy")
            
            # Get environment statistics
            environment_stats = {}
            for deployment in self.active_deployments.values():
                env = deployment.target_environment
                if env not in environment_stats:
                    environment_stats[env] = {"total": 0, "healthy": 0, "degraded": 0, "unhealthy": 0}
                
                environment_stats[env]["total"] += 1
                environment_stats[env][deployment.health_status] += 1
            
            # Get recent deployments (last 24 hours)
            recent_deployments = []
            cutoff_time = datetime.now() - timedelta(hours=24)
            for deployment in self.active_deployments.values():
                if deployment.created_at >= cutoff_time:
                    recent_deployments.append({
                        "deployment_id": deployment.deployment_id,
                        "model_name": deployment.model_name,
                        "deployment_type": deployment.deployment_type,
                        "target_environment": deployment.target_environment,
                        "status": deployment.status,
                        "health_status": deployment.health_status,
                        "created_at": deployment.created_at
                    })
            
            # Get alert statistics
            alert_stats = await self.alerting_system.get_alert_statistics()
            
            return {
                "deployment_statistics": {
                    "total": total_deployments,
                    "healthy": healthy_deployments,
                    "degraded": degraded_deployments,
                    "unhealthy": unhealthy_deployments,
                    "health_percentage": (healthy_deployments / total_deployments * 100) if total_deployments > 0 else 0
                },
                "environment_statistics": environment_stats,
                "recent_deployments": recent_deployments,
                "alert_statistics": alert_stats,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get dashboard metrics: {e}")
            return {
                "deployment_statistics": {"total": 0, "healthy": 0, "degraded": 0, "unhealthy": 0, "health_percentage": 0},
                "environment_statistics": {},
                "recent_deployments": [],
                "alert_statistics": {},
                "error": str(e),
                "timestamp": datetime.now()
            }
    
    async def _validate_deployment_request(self, request: DeploymentRequest) -> None:
        """Validate deployment request."""
        # Check if model exists in registry
        if self.config["model_registry"]["integration"]:
            model_exists = await self.model_registry.model_exists(request.model_id)
            if not model_exists:
                raise ValueError(f"Model {request.model_id} not found in registry")
        
        # Check environment availability
        available_environments = self.config["deployment"]["environments"]
        if request.target_environment not in available_environments:
            raise ValueError(f"Environment {request.target_environment} not available")
        
        # Check deployment type
        if request.deployment_type not in available_environments:
            raise ValueError(f"Deployment type {request.deployment_type} not supported")
    
    async def _setup_monitoring(self, config: MonitoringConfig) -> Dict[str, Any]:
        """Setup monitoring for deployment."""
        try:
            # Configure monitoring
            monitoring_result = await self.performance_analytics.setup_monitoring(config)
            
            # Store monitoring configuration
            self.monitoring_configs[config.model_id] = config
            
            return monitoring_result
            
        except Exception as e:
            self.logger.error(f"Failed to setup monitoring: {e}")
            return {"success": False, "error": str(e)}
    
    async def _start_monitoring(self, deployment_id: str) -> None:
        """Start monitoring for deployment."""
        try:
            deployment = self.active_deployments[deployment_id]
            model_id = deployment.model_id
            
            if model_id in self.monitoring_configs:
                config = self.monitoring_configs[model_id]
                
                # Start continuous monitoring
                while deployment.status in ["deployed", "deploying"]:
                    try:
                        # Collect metrics
                        metrics = await self._collect_metrics(deployment_id)
                        
                        # Update deployment with metrics
                        deployment.performance_metrics = metrics
                        deployment.health_status = self._calculate_health_status(metrics)
                        deployment.last_updated = datetime.now()
                        
                        # Check for alerts
                        await self._check_alerts(deployment_id, metrics, config)
                        
                        # Wait for next collection interval
                        await asyncio.sleep(config.reporting_interval)
                        
                    except Exception as e:
                        self.logger.error(f"Monitoring error for {deployment_id}: {e}")
                        await asyncio.sleep(60)  # Wait 1 minute before retry
                
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
    
    async def _stop_monitoring(self, deployment_id: str) -> None:
        """Stop monitoring for deployment."""
        try:
            deployment = self.active_deployments[deployment_id]
            model_id = deployment.model_id
            
            if model_id in self.monitoring_configs:
                # Stop monitoring
                await self.performance_analytics.stop_monitoring(model_id)
                
                # Remove monitoring configuration
                del self.monitoring_configs[model_id]
                
        except Exception as e:
            self.logger.error(f"Failed to stop monitoring: {e}")
    
    async def _collect_metrics(self, deployment_id: str) -> Dict[str, Any]:
        """Collect metrics for deployment."""
        try:
            deployment = self.active_deployments[deployment_id]
            
            # Collect performance metrics
            metrics = await self.performance_analytics.collect_metrics(
                model_id=deployment.model_id,
                deployment_id=deployment_id
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics: {e}")
            return {}
    
    async def _get_latest_metrics(self, deployment_id: str) -> Dict[str, Any]:
        """Get latest metrics for deployment."""
        try:
            deployment = self.active_deployments[deployment_id]
            
            # Get latest metrics from performance analytics
            metrics = await self.performance_analytics.get_latest_metrics(
                model_id=deployment.model_id,
                deployment_id=deployment_id
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get latest metrics: {e}")
            return {}
    
    def _calculate_health_status(self, metrics: Dict[str, Any]) -> str:
        """Calculate health status based on metrics."""
        try:
            # Check key health indicators
            health_indicators = {
                "latency": metrics.get("latency", 0),
                "error_rate": metrics.get("error_rate", 0),
                "memory_usage": metrics.get("memory_usage", 0),
                "cpu_usage": metrics.get("cpu_usage", 0)
            }
            
            # Determine health status
            if (health_indicators["latency"] > 1000 or  # > 1 second
                health_indicators["error_rate"] > 0.05 or  # > 5%
                health_indicators["memory_usage"] > 0.9 or  # > 90%
                health_indicators["cpu_usage"] > 0.9):  # > 90%
                return "unhealthy"
            elif (health_indicators["latency"] > 500 or  # > 500ms
                  health_indicators["error_rate"] > 0.02 or  # > 2%
                  health_indicators["memory_usage"] > 0.8 or  # > 80%
                  health_indicators["cpu_usage"] > 0.8):  # > 80%
                return "degraded"
            else:
                return "healthy"
                
        except Exception as e:
            self.logger.error(f"Failed to calculate health status: {e}")
            return "unknown"
    
    async def _check_alerts(self, deployment_id: str, metrics: Dict[str, Any], config: MonitoringConfig) -> None:
        """Check for alerts based on metrics."""
        try:
            if not config.alerting_enabled:
                return
            
            # Check alert thresholds
            for metric, threshold in config.alert_thresholds.items():
                if metric in metrics and metrics[metric] > threshold:
                    # Trigger alert
                    await self.alerting_system.trigger_alert(
                        deployment_id=deployment_id,
                        metric=metric,
                        value=metrics[metric],
                        threshold=threshold
                    )
                    
        except Exception as e:
            self.logger.error(f"Failed to check alerts: {e}")


# FastAPI Router for Factory Roster Dashboard
router = APIRouter(prefix="/factory-roster", tags=["Factory Roster Dashboard"])

# Global dashboard instance
dashboard = FactoryRosterDashboard()


@router.get("/dashboard")
async def get_dashboard():
    """Get factory roster dashboard."""
    try:
        metrics = await dashboard.get_dashboard_metrics()
        return JSONResponse(content=metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deploy")
async def deploy_model(request: DeploymentRequest):
    """Deploy model to factory roster."""
    try:
        result = await dashboard.deploy_model(request)
        return JSONResponse(content=asdict(result))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop/{deployment_id}")
async def stop_deployment(deployment_id: str):
    """Stop model deployment."""
    try:
        success = await dashboard.stop_deployment(deployment_id)
        return JSONResponse(content={"success": success})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rollback/{deployment_id}")
async def rollback_deployment(deployment_id: str, target_version: str):
    """Rollback deployment to previous version."""
    try:
        success = await dashboard.rollback_deployment(deployment_id, target_version)
        return JSONResponse(content={"success": success})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deployments")
async def get_deployments():
    """Get all deployments."""
    try:
        deployments = await dashboard.get_all_deployments()
        return JSONResponse(content=deployments)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deployments/{deployment_id}")
async def get_deployment_status(deployment_id: str):
    """Get specific deployment status."""
    try:
        status = await dashboard.get_deployment_status(deployment_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Deployment not found")
        return JSONResponse(content=asdict(status))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reports/{model_id}")
async def generate_performance_report(model_id: str, period: str = "24h"):
    """Generate performance report for model."""
    try:
        report = await dashboard.generate_performance_report(model_id, period)
        return JSONResponse(content=asdict(report))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workspace")
async def get_workspace_interface():
    """Get Factory Roster workspace interface."""
    try:
        with open("src/enterprise_llmops/frontend/enhanced_unified_platform.html", "r") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
