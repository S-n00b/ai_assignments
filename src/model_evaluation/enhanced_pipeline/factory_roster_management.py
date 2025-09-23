"""
Factory Roster Management for Production Deployment

This module provides comprehensive factory roster management including
model deployment, monitoring, and production lifecycle management.

Key Features:
- Model deployment to production roster
- Production monitoring and alerting
- Model lifecycle management
- Performance tracking in production
- MLflow production integration
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid

# MLflow imports
try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
except ImportError:
    mlflow = None
    MlflowClient = None

logger = logging.getLogger(__name__)


@dataclass
class ProductionModel:
    """Production model configuration."""
    model_id: str
    model_name: str
    model_type: str  # "foundation", "custom", "agentic", "retrieval"
    version: str
    deployment_status: str  # "pending", "deploying", "active", "inactive", "failed"
    performance_metrics: Dict[str, float]
    resource_requirements: Dict[str, Any]
    monitoring_config: Dict[str, Any]
    created_at: datetime
    deployed_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None


@dataclass
class ProductionAlert:
    """Production alert configuration."""
    alert_id: str
    model_id: str
    alert_type: str  # "performance", "error", "resource", "quality"
    severity: str  # "low", "medium", "high", "critical"
    message: str
    threshold: float
    current_value: float
    created_at: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class ProductionMetrics:
    """Production metrics for a model."""
    model_id: str
    timestamp: datetime
    performance_metrics: Dict[str, float]
    resource_metrics: Dict[str, float]
    quality_metrics: Dict[str, float]
    business_metrics: Dict[str, float]


class FactoryRosterManager:
    """
    Factory Roster Manager for Production Deployment
    
    This class provides comprehensive factory roster management including
    model deployment, monitoring, and production lifecycle management.
    """
    
    def __init__(self, 
                 mlflow_tracking_uri: str = "http://localhost:5000"):
        """
        Initialize the Factory Roster Manager.
        
        Args:
            mlflow_tracking_uri: MLflow tracking server URI
        """
        self.mlflow_tracking_uri = mlflow_tracking_uri
        
        # Initialize MLflow
        if mlflow:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            self.client = MlflowClient(tracking_uri=mlflow_tracking_uri)
            self._ensure_experiment()
        else:
            self.client = None
            logger.warning("MLflow not available, experiment tracking disabled")
        
        # Production models tracking
        self.production_models: Dict[str, ProductionModel] = {}
        self.active_alerts: Dict[str, ProductionAlert] = {}
        self.production_metrics: List[ProductionMetrics] = []
        
        # Monitoring configuration
        self.monitoring_config = {
            "performance_thresholds": {
                "latency_p95": 5.0,
                "error_rate": 0.05,
                "throughput": 5.0
            },
            "resource_thresholds": {
                "memory_usage": 0.9,
                "cpu_usage": 0.8,
                "disk_usage": 0.85
            },
            "quality_thresholds": {
                "accuracy": 0.8,
                "relevance": 0.75,
                "coherence": 0.7
            }
        }
        
        logger.info("Factory Roster Manager initialized")
    
    def _ensure_experiment(self):
        """Ensure production experiment exists."""
        if not self.client:
            return
            
        try:
            experiment_name = "production_deployment"
            experiment = self.client.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = self.client.create_experiment(experiment_name)
                logger.info(f"Created experiment: {experiment_name}")
            else:
                logger.info(f"Using existing experiment: {experiment_name}")
        except Exception as e:
            logger.warning(f"Could not ensure experiment: {e}")
    
    def deploy_model_to_production(self,
                                 model_name: str,
                                 model_type: str,
                                 version: str,
                                 performance_metrics: Dict[str, float],
                                 resource_requirements: Dict[str, Any],
                                 monitoring_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Deploy a model to production roster.
        
        Args:
            model_name: Name of the model
            model_type: Type of the model
            version: Model version
            performance_metrics: Performance metrics
            resource_requirements: Resource requirements
            monitoring_config: Monitoring configuration
            
        Returns:
            Model ID
        """
        model_id = str(uuid.uuid4())
        
        if monitoring_config is None:
            monitoring_config = {
                "monitoring_enabled": True,
                "alert_thresholds": self.monitoring_config["performance_thresholds"],
                "collection_interval": 60,  # seconds
                "retention_days": 30
            }
        
        production_model = ProductionModel(
            model_id=model_id,
            model_name=model_name,
            model_type=model_type,
            version=version,
            deployment_status="pending",
            performance_metrics=performance_metrics,
            resource_requirements=resource_requirements,
            monitoring_config=monitoring_config,
            created_at=datetime.now()
        )
        
        self.production_models[model_id] = production_model
        logger.info(f"Deployed model {model_name} to production roster with ID {model_id}")
        
        return model_id
    
    async def activate_model(self, model_id: str) -> bool:
        """
        Activate a model in production.
        
        Args:
            model_id: Model ID to activate
            
        Returns:
            Success status
        """
        if model_id not in self.production_models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.production_models[model_id]
        model.deployment_status = "deploying"
        model.last_updated = datetime.now()
        
        # Simulate deployment process
        await asyncio.sleep(2)  # Mock deployment time
        
        # Check deployment readiness
        deployment_ready = await self._check_deployment_readiness(model)
        
        if deployment_ready:
            model.deployment_status = "active"
            model.deployed_at = datetime.now()
            logger.info(f"Model {model.model_name} activated in production")
            
            # Start monitoring
            await self._start_model_monitoring(model_id)
            
            return True
        else:
            model.deployment_status = "failed"
            logger.error(f"Model {model.model_name} deployment failed")
            return False
    
    async def deactivate_model(self, model_id: str) -> bool:
        """
        Deactivate a model in production.
        
        Args:
            model_id: Model ID to deactivate
            
        Returns:
            Success status
        """
        if model_id not in self.production_models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.production_models[model_id]
        model.deployment_status = "inactive"
        model.last_updated = datetime.now()
        
        # Stop monitoring
        await self._stop_model_monitoring(model_id)
        
        logger.info(f"Model {model.model_name} deactivated in production")
        return True
    
    async def _check_deployment_readiness(self, model: ProductionModel) -> bool:
        """Check if model is ready for deployment."""
        # Check performance metrics
        if model.performance_metrics.get("latency_p95", 0) > self.monitoring_config["performance_thresholds"]["latency_p95"]:
            return False
        
        if model.performance_metrics.get("error_rate", 1.0) > self.monitoring_config["performance_thresholds"]["error_rate"]:
            return False
        
        if model.performance_metrics.get("throughput", 0) < self.monitoring_config["performance_thresholds"]["throughput"]:
            return False
        
        # Check resource requirements
        if model.resource_requirements.get("memory", 0) > 0.9:
            return False
        
        if model.resource_requirements.get("cpu", 0) > 0.8:
            return False
        
        return True
    
    async def _start_model_monitoring(self, model_id: str):
        """Start monitoring a model."""
        model = self.production_models[model_id]
        
        if model.monitoring_config.get("monitoring_enabled", False):
            logger.info(f"Started monitoring for model {model.model_name}")
            
            # Start background monitoring task
            asyncio.create_task(self._monitor_model(model_id))
    
    async def _stop_model_monitoring(self, model_id: str):
        """Stop monitoring a model."""
        model = self.production_models[model_id]
        logger.info(f"Stopped monitoring for model {model.model_name}")
    
    async def _monitor_model(self, model_id: str):
        """Monitor a model in production."""
        model = self.production_models[model_id]
        collection_interval = model.monitoring_config.get("collection_interval", 60)
        
        while model.deployment_status == "active":
            try:
                # Collect metrics
                metrics = await self._collect_production_metrics(model_id)
                self.production_metrics.append(metrics)
                
                # Check for alerts
                await self._check_alerts(model_id, metrics)
                
                # Log to MLflow
                if self.client:
                    await self._log_production_metrics(model_id, metrics)
                
                await asyncio.sleep(collection_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring model {model_id}: {e}")
                await asyncio.sleep(collection_interval)
    
    async def _collect_production_metrics(self, model_id: str) -> ProductionMetrics:
        """Collect production metrics for a model."""
        model = self.production_models[model_id]
        
        # Mock production metrics collection
        performance_metrics = {
            "latency_p95": 2.5 + (hash(model_id) % 20) / 10,  # 2.5-4.4
            "error_rate": 0.02 + (hash(model_id) % 5) / 100,   # 0.02-0.06
            "throughput": 8.0 + (hash(model_id) % 10) / 2,    # 8.0-12.9
            "response_time": 1.8 + (hash(model_id) % 15) / 10  # 1.8-3.2
        }
        
        resource_metrics = {
            "memory_usage": 0.6 + (hash(model_id) % 30) / 100,  # 0.6-0.89
            "cpu_usage": 0.5 + (hash(model_id) % 25) / 100,      # 0.5-0.74
            "disk_usage": 0.4 + (hash(model_id) % 20) / 100,     # 0.4-0.59
            "network_usage": 0.3 + (hash(model_id) % 15) / 100   # 0.3-0.44
        }
        
        quality_metrics = {
            "accuracy": 0.85 + (hash(model_id) % 15) / 100,     # 0.85-0.99
            "relevance": 0.80 + (hash(model_id) % 20) / 100,    # 0.80-0.99
            "coherence": 0.75 + (hash(model_id) % 25) / 100,   # 0.75-0.99
            "completeness": 0.78 + (hash(model_id) % 22) / 100  # 0.78-0.99
        }
        
        business_metrics = {
            "user_satisfaction": 0.8 + (hash(model_id) % 20) / 100,  # 0.8-0.99
            "adoption_rate": 0.7 + (hash(model_id) % 30) / 100,      # 0.7-0.99
            "retention_rate": 0.75 + (hash(model_id) % 25) / 100,   # 0.75-0.99
            "revenue_impact": 0.6 + (hash(model_id) % 40) / 100     # 0.6-0.99
        }
        
        return ProductionMetrics(
            model_id=model_id,
            timestamp=datetime.now(),
            performance_metrics=performance_metrics,
            resource_metrics=resource_metrics,
            quality_metrics=quality_metrics,
            business_metrics=business_metrics
        )
    
    async def _check_alerts(self, model_id: str, metrics: ProductionMetrics):
        """Check for alerts based on metrics."""
        model = self.production_models[model_id]
        thresholds = model.monitoring_config.get("alert_thresholds", self.monitoring_config["performance_thresholds"])
        
        # Check performance alerts
        if metrics.performance_metrics.get("latency_p95", 0) > thresholds.get("latency_p95", 5.0):
            await self._create_alert(
                model_id=model_id,
                alert_type="performance",
                severity="high",
                message=f"High latency detected: {metrics.performance_metrics['latency_p95']:.2f}s",
                threshold=thresholds.get("latency_p95", 5.0),
                current_value=metrics.performance_metrics.get("latency_p95", 0)
            )
        
        if metrics.performance_metrics.get("error_rate", 0) > thresholds.get("error_rate", 0.05):
            await self._create_alert(
                model_id=model_id,
                alert_type="error",
                severity="critical",
                message=f"High error rate detected: {metrics.performance_metrics['error_rate']:.2%}",
                threshold=thresholds.get("error_rate", 0.05),
                current_value=metrics.performance_metrics.get("error_rate", 0)
            )
        
        # Check resource alerts
        if metrics.resource_metrics.get("memory_usage", 0) > self.monitoring_config["resource_thresholds"]["memory_usage"]:
            await self._create_alert(
                model_id=model_id,
                alert_type="resource",
                severity="medium",
                message=f"High memory usage: {metrics.resource_metrics['memory_usage']:.1%}",
                threshold=self.monitoring_config["resource_thresholds"]["memory_usage"],
                current_value=metrics.resource_metrics.get("memory_usage", 0)
            )
        
        # Check quality alerts
        if metrics.quality_metrics.get("accuracy", 0) < self.monitoring_config["quality_thresholds"]["accuracy"]:
            await self._create_alert(
                model_id=model_id,
                alert_type="quality",
                severity="high",
                message=f"Low accuracy detected: {metrics.quality_metrics['accuracy']:.2%}",
                threshold=self.monitoring_config["quality_thresholds"]["accuracy"],
                current_value=metrics.quality_metrics.get("accuracy", 0)
            )
    
    async def _create_alert(self, model_id: str, alert_type: str, severity: str, 
                           message: str, threshold: float, current_value: float):
        """Create a production alert."""
        alert_id = str(uuid.uuid4())
        
        alert = ProductionAlert(
            alert_id=alert_id,
            model_id=model_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            threshold=threshold,
            current_value=current_value,
            created_at=datetime.now()
        )
        
        self.active_alerts[alert_id] = alert
        logger.warning(f"Alert created: {message}")
    
    async def _log_production_metrics(self, model_id: str, metrics: ProductionMetrics):
        """Log production metrics to MLflow."""
        if not self.client:
            return
        
        try:
            with mlflow.start_run(experiment_id=self._get_experiment_id()):
                # Log metrics
                mlflow.log_metrics({
                    **metrics.performance_metrics,
                    **metrics.resource_metrics,
                    **metrics.quality_metrics,
                    **metrics.business_metrics
                })
                
                # Log parameters
                mlflow.log_params({
                    "model_id": model_id,
                    "timestamp": metrics.timestamp.isoformat()
                })
                
        except Exception as e:
            logger.error(f"Failed to log production metrics to MLflow: {e}")
    
    def get_production_status(self) -> Dict[str, Any]:
        """Get overall production status."""
        active_models = [m for m in self.production_models.values() if m.deployment_status == "active"]
        failed_models = [m for m in self.production_models.values() if m.deployment_status == "failed"]
        active_alerts = [a for a in self.active_alerts.values() if not a.resolved]
        
        return {
            "total_models": len(self.production_models),
            "active_models": len(active_models),
            "failed_models": len(failed_models),
            "active_alerts": len(active_alerts),
            "critical_alerts": len([a for a in active_alerts if a.severity == "critical"]),
            "high_alerts": len([a for a in active_alerts if a.severity == "high"]),
            "models": [
                {
                    "model_id": m.model_id,
                    "model_name": m.model_name,
                    "model_type": m.model_type,
                    "status": m.deployment_status,
                    "version": m.version,
                    "deployed_at": m.deployed_at.isoformat() if m.deployed_at else None
                }
                for m in self.production_models.values()
            ],
            "alerts": [
                {
                    "alert_id": a.alert_id,
                    "model_id": a.model_id,
                    "alert_type": a.alert_type,
                    "severity": a.severity,
                    "message": a.message,
                    "created_at": a.created_at.isoformat()
                }
                for a in active_alerts
            ]
        }
    
    def get_model_metrics(self, model_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get metrics for a specific model."""
        if model_id not in self.production_models:
            raise ValueError(f"Model {model_id} not found")
        
        # Filter metrics by time range
        cutoff_time = datetime.now() - timedelta(hours=hours)
        model_metrics = [
            m for m in self.production_metrics 
            if m.model_id == model_id and m.timestamp >= cutoff_time
        ]
        
        return [
            {
                "timestamp": m.timestamp.isoformat(),
                "performance_metrics": m.performance_metrics,
                "resource_metrics": m.resource_metrics,
                "quality_metrics": m.quality_metrics,
                "business_metrics": m.business_metrics
            }
            for m in model_metrics
        ]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.resolved = True
        alert.resolved_at = datetime.now()
        
        logger.info(f"Alert {alert_id} resolved")
        return True
    
    def _get_experiment_id(self) -> str:
        """Get the production experiment ID."""
        if not self.client:
            return None
            
        try:
            experiment = self.client.get_experiment_by_name("production_deployment")
            return experiment.experiment_id if experiment else None
        except Exception:
            return None
    
    def get_production_dashboard_data(self) -> Dict[str, Any]:
        """Get data for production dashboard."""
        status = self.get_production_status()
        
        # Calculate trends
        recent_metrics = self.production_metrics[-24:] if len(self.production_metrics) >= 24 else self.production_metrics
        
        if recent_metrics:
            avg_latency = sum(m.performance_metrics.get("latency_p95", 0) for m in recent_metrics) / len(recent_metrics)
            avg_throughput = sum(m.performance_metrics.get("throughput", 0) for m in recent_metrics) / len(recent_metrics)
            avg_accuracy = sum(m.quality_metrics.get("accuracy", 0) for m in recent_metrics) / len(recent_metrics)
        else:
            avg_latency = 0
            avg_throughput = 0
            avg_accuracy = 0
        
        return {
            "status": status,
            "trends": {
                "avg_latency": avg_latency,
                "avg_throughput": avg_throughput,
                "avg_accuracy": avg_accuracy
            },
            "health_score": self._calculate_health_score(status, recent_metrics)
        }
    
    def _calculate_health_score(self, status: Dict[str, Any], recent_metrics: List[ProductionMetrics]) -> float:
        """Calculate overall health score."""
        if not recent_metrics:
            return 0.0
        
        # Base score
        score = 100.0
        
        # Penalize for failed models
        score -= status["failed_models"] * 20
        
        # Penalize for alerts
        score -= status["critical_alerts"] * 15
        score -= status["high_alerts"] * 10
        
        # Penalize for poor performance
        avg_latency = sum(m.performance_metrics.get("latency_p95", 0) for m in recent_metrics) / len(recent_metrics)
        if avg_latency > 5.0:
            score -= 20
        elif avg_latency > 3.0:
            score -= 10
        
        return max(0.0, min(100.0, score))


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Initialize factory roster manager
        manager = FactoryRosterManager()
        
        # Deploy a model to production
        model_id = manager.deploy_model_to_production(
            model_name="lenovo-device-support-v1",
            model_type="custom",
            version="1.0.0",
            performance_metrics={
                "latency_p95": 2.5,
                "error_rate": 0.02,
                "throughput": 10.0
            },
            resource_requirements={
                "memory": 0.6,
                "cpu": 0.5,
                "storage": 0.3
            }
        )
        
        print(f"Deployed model with ID: {model_id}")
        
        # Activate model
        success = await manager.activate_model(model_id)
        print(f"Model activation: {success}")
        
        # Get production status
        status = manager.get_production_status()
        print(f"Production status: {status}")
    
    asyncio.run(main())
