"""
Model Lifecycle Management Module

This module implements comprehensive MLOps pipeline and model lifecycle management
for enterprise AI systems, including versioning, deployment strategies, monitoring,
and automated retraining pipelines.

Key Features:
- Model versioning and registry
- Automated deployment pipelines
- Performance monitoring and drift detection
- Automated retraining and rollback
- A/B testing and canary deployments
- Compliance and audit trails
"""

import json
import time
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Import fine-tuning and quantization modules
from .adapter_registry import CustomAdapterRegistry, AdapterMetadata, AdapterType, AdapterStatus
from .finetuning_quantization import (
    AdvancedFineTuner, AdvancedQuantizer, MultiTaskFineTuner,
    FineTuningConfig, QuantizationConfig, FineTuningStrategy, QuantizationStrategy
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model lifecycle status"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class DeploymentStrategy(Enum):
    """Deployment strategies"""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    RECREATE = "recreate"
    A_B_TESTING = "a_b_testing"


class ModelStage(Enum):
    """Model development stages"""
    DATA_PREPARATION = "data_preparation"
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    RETRAINING = "retraining"


@dataclass
class ModelVersion:
    """Model version information"""
    model_id: str
    version: str
    stage: ModelStage
    status: ModelStatus
    created_at: datetime
    created_by: str
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    deployment_config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    strategy: DeploymentStrategy
    target_environment: str
    resource_requirements: Dict[str, Any]
    scaling_config: Dict[str, Any]
    health_check_config: Dict[str, Any]
    rollback_config: Dict[str, Any]
    monitoring_config: Dict[str, Any]


@dataclass
class PerformanceMetrics:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    latency_ms: float
    throughput_rps: float
    error_rate: float
    data_drift_score: float
    model_drift_score: float
    timestamp: datetime


class ModelLifecycleManager:
    """
    Comprehensive Model Lifecycle Management System.
    
    This class provides end-to-end model lifecycle management including:
    - Model versioning and registry
    - Automated deployment pipelines
    - Performance monitoring and drift detection
    - Automated retraining and rollback
    - A/B testing and canary deployments
    - Compliance and audit trails
    
    The system is designed for enterprise-scale AI operations with robust
    monitoring, security, and compliance capabilities.
    """
    
    def __init__(self, registry_path: str = "./model_registry"):
        """
        Initialize the Model Lifecycle Manager.
        
        Args:
            registry_path: Path to the model registry storage
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        
        # Initialize components
        self.model_registry = {}
        self.deployment_history = []
        self.performance_history = []
        self.retraining_pipelines = {}
        self.monitoring_alerts = []
        
        # Initialize fine-tuning and quantization components
        self.adapter_registry = CustomAdapterRegistry(str(self.registry_path / "adapters"))
        self.fine_tuner = AdvancedFineTuner(str(self.registry_path / "adapters"))
        self.quantizer = AdvancedQuantizer()
        self.multi_task_fine_tuner = MultiTaskFineTuner()
        
        # Load existing registry
        self._load_model_registry()
        
        logger.info("Model Lifecycle Manager initialized")
    
    def _load_model_registry(self):
        """Load existing model registry from storage"""
        registry_file = self.registry_path / "registry.json"
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    registry_data = json.load(f)
                    self.model_registry = registry_data
                logger.info(f"Loaded {len(self.model_registry)} models from registry")
            except Exception as e:
                logger.error(f"Failed to load model registry: {str(e)}")
    
    def _save_model_registry(self):
        """Save model registry to storage"""
        registry_file = self.registry_path / "registry.json"
        try:
            # Convert datetime objects to strings for JSON serialization
            registry_data = {}
            for model_id, versions in self.model_registry.items():
                registry_data[model_id] = {}
                for version, model_version in versions.items():
                    model_dict = asdict(model_version)
                    model_dict['created_at'] = model_version.created_at.isoformat()
                    registry_data[model_id][version] = model_dict
            
            with open(registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
            logger.info("Model registry saved successfully")
        except Exception as e:
            logger.error(f"Failed to save model registry: {str(e)}")
    
    async def register_model(
        self,
        model_id: str,
        version: str,
        stage: ModelStage,
        created_by: str,
        description: str,
        metadata: Dict[str, Any] = None,
        performance_metrics: Dict[str, float] = None,
        dependencies: List[str] = None,
        tags: List[str] = None
    ) -> ModelVersion:
        """
        Register a new model version in the registry.
        
        Args:
            model_id: Unique model identifier
            version: Model version (e.g., "1.0.0", "v2.1")
            stage: Current development stage
            created_by: User who created the model
            description: Model description
            metadata: Additional model metadata
            performance_metrics: Model performance metrics
            dependencies: Model dependencies
            tags: Model tags for categorization
            
        Returns:
            Registered ModelVersion object
        """
        
        # Initialize model registry entry if not exists
        if model_id not in self.model_registry:
            self.model_registry[model_id] = {}
        
        # Check if version already exists
        if version in self.model_registry[model_id]:
            raise ValueError(f"Model version {model_id}:{version} already exists")
        
        # Create model version
        model_version = ModelVersion(
            model_id=model_id,
            version=version,
            stage=stage,
            status=ModelStatus.DEVELOPMENT,
            created_at=datetime.now(),
            created_by=created_by,
            description=description,
            metadata=metadata or {},
            performance_metrics=performance_metrics or {},
            dependencies=dependencies or [],
            tags=tags or []
        )
        
        # Register in registry
        self.model_registry[model_id][version] = model_version
        
        # Save registry
        self._save_model_registry()
        
        logger.info(f"Registered model {model_id}:{version} in stage {stage.value}")
        
        return model_version
    
    async def promote_model(
        self,
        model_id: str,
        version: str,
        target_stage: ModelStage,
        deployment_config: DeploymentConfig = None
    ) -> Dict[str, Any]:
        """
        Promote a model to the next stage in the lifecycle.
        
        Args:
            model_id: Model identifier
            version: Model version
            target_stage: Target stage for promotion
            deployment_config: Deployment configuration for production stages
            
        Returns:
            Promotion result with status and metadata
        """
        
        try:
            # Get model version
            if model_id not in self.model_registry or version not in self.model_registry[model_id]:
                raise ValueError(f"Model {model_id}:{version} not found in registry")
            
            model_version = self.model_registry[model_id][version]
            
            # Validate promotion path
            if not self._validate_promotion_path(model_version.stage, target_stage):
                raise ValueError(f"Invalid promotion path from {model_version.stage.value} to {target_stage.value}")
            
            # Update model stage and status
            model_version.stage = target_stage
            model_version.status = self._get_status_for_stage(target_stage)
            
            # Add deployment configuration for production stages
            if target_stage in [ModelStage.DEPLOYMENT, ModelStage.MONITORING]:
                if deployment_config:
                    model_version.deployment_config = asdict(deployment_config)
                else:
                    # Create default deployment configuration
                    model_version.deployment_config = asdict(DeploymentConfig(
                        strategy=DeploymentStrategy.BLUE_GREEN,
                        target_environment="production",
                        resource_requirements={"cpu": "2", "memory": "4Gi"},
                        scaling_config={"min_replicas": 1, "max_replicas": 10},
                        health_check_config={"timeout": 30, "interval": 10},
                        rollback_config={"enabled": True, "threshold": 0.1},
                        monitoring_config={"enabled": True, "metrics": ["accuracy", "latency"]}
                    ))
            
            # Save registry
            self._save_model_registry()
            
            # Record promotion in deployment history
            self.deployment_history.append({
                "model_id": model_id,
                "version": version,
                "action": "promotion",
                "from_stage": model_version.stage.value,
                "to_stage": target_stage.value,
                "timestamp": datetime.now().isoformat(),
                "deployment_config": model_version.deployment_config
            })
            
            logger.info(f"Promoted model {model_id}:{version} to {target_stage.value}")
            
            return {
                "status": "success",
                "model_id": model_id,
                "version": version,
                "stage": target_stage.value,
                "status": model_version.status.value,
                "promoted_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Model promotion failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _validate_promotion_path(self, current_stage: ModelStage, target_stage: ModelStage) -> bool:
        """Validate if promotion path is valid"""
        
        # Define valid promotion paths
        valid_paths = {
            ModelStage.DATA_PREPARATION: [ModelStage.TRAINING],
            ModelStage.TRAINING: [ModelStage.VALIDATION],
            ModelStage.VALIDATION: [ModelStage.TESTING, ModelStage.TRAINING],
            ModelStage.TESTING: [ModelStage.DEPLOYMENT, ModelStage.TRAINING],
            ModelStage.DEPLOYMENT: [ModelStage.MONITORING],
            ModelStage.MONITORING: [ModelStage.RETRAINING, ModelStage.DEPLOYMENT],
            ModelStage.RETRAINING: [ModelStage.TRAINING]
        }
        
        return target_stage in valid_paths.get(current_stage, [])
    
    def _get_status_for_stage(self, stage: ModelStage) -> ModelStatus:
        """Get appropriate status for a given stage"""
        
        status_mapping = {
            ModelStage.DATA_PREPARATION: ModelStatus.DEVELOPMENT,
            ModelStage.TRAINING: ModelStatus.DEVELOPMENT,
            ModelStage.VALIDATION: ModelStatus.TESTING,
            ModelStage.TESTING: ModelStatus.TESTING,
            ModelStage.DEPLOYMENT: ModelStatus.STAGING,
            ModelStage.MONITORING: ModelStatus.PRODUCTION,
            ModelStage.RETRAINING: ModelStatus.DEVELOPMENT
        }
        
        return status_mapping.get(stage, ModelStatus.DEVELOPMENT)
    
    async def deploy_model(
        self,
        model_id: str,
        version: str,
        deployment_strategy: DeploymentStrategy = DeploymentStrategy.BLUE_GREEN,
        target_environment: str = "production"
    ) -> Dict[str, Any]:
        """
        Deploy a model using the specified deployment strategy.
        
        Args:
            model_id: Model identifier
            version: Model version
            deployment_strategy: Deployment strategy to use
            target_environment: Target deployment environment
            
        Returns:
            Deployment result with status and metadata
        """
        
        try:
            # Get model version
            if model_id not in self.model_registry or version not in self.model_registry[model_id]:
                raise ValueError(f"Model {model_id}:{version} not found in registry")
            
            model_version = self.model_registry[model_id][version]
            
            # Validate model is ready for deployment
            if model_version.stage not in [ModelStage.DEPLOYMENT, ModelStage.MONITORING]:
                raise ValueError(f"Model {model_id}:{version} is not ready for deployment (current stage: {model_version.stage.value})")
            
            # Execute deployment based on strategy
            deployment_result = await self._execute_deployment(
                model_version, deployment_strategy, target_environment
            )
            
            # Record deployment in history
            self.deployment_history.append({
                "model_id": model_id,
                "version": version,
                "action": "deployment",
                "strategy": deployment_strategy.value,
                "target_environment": target_environment,
                "timestamp": datetime.now().isoformat(),
                "deployment_result": deployment_result
            })
            
            logger.info(f"Deployed model {model_id}:{version} using {deployment_strategy.value}")
            
            return {
                "status": "success",
                "model_id": model_id,
                "version": version,
                "deployment_strategy": deployment_strategy.value,
                "target_environment": target_environment,
                "deployed_at": datetime.now().isoformat(),
                "deployment_result": deployment_result
            }
            
        except Exception as e:
            logger.error(f"Model deployment failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _execute_deployment(
        self,
        model_version: ModelVersion,
        deployment_strategy: DeploymentStrategy,
        target_environment: str
    ) -> Dict[str, Any]:
        """Execute the actual deployment based on strategy"""
        
        # Simulate deployment process
        await asyncio.sleep(2)
        
        # Generate deployment endpoints based on strategy
        base_url = f"https://api.{target_environment}.com/v1/models/{model_version.model_id}"
        
        if deployment_strategy == DeploymentStrategy.BLUE_GREEN:
            endpoints = {
                "blue": f"{base_url}/v{model_version.version}/predict",
                "green": f"{base_url}/v{model_version.version}/predict",
                "switch_endpoint": f"{base_url}/switch"
            }
        elif deployment_strategy == DeploymentStrategy.CANARY:
            endpoints = {
                "canary": f"{base_url}/v{model_version.version}/predict",
                "stable": f"{base_url}/stable/predict",
                "traffic_split": "10% canary, 90% stable"
            }
        elif deployment_strategy == DeploymentStrategy.A_B_TESTING:
            endpoints = {
                "variant_a": f"{base_url}/v{model_version.version}/predict",
                "variant_b": f"{base_url}/stable/predict",
                "traffic_split": "50% A, 50% B"
            }
        else:
            endpoints = {
                "primary": f"{base_url}/v{model_version.version}/predict"
            }
        
        return {
            "endpoints": endpoints,
            "deployment_id": f"{model_version.model_id}_{model_version.version}_{int(time.time())}",
            "health_check_url": f"{base_url}/v{model_version.version}/health",
            "metrics_url": f"{base_url}/v{model_version.version}/metrics"
        }
    
    async def monitor_model_performance(
        self,
        model_id: str,
        version: str,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Monitor model performance and detect drift.
        
        Args:
            model_id: Model identifier
            version: Model version
            time_window_hours: Time window for monitoring
            
        Returns:
            Performance monitoring results
        """
        
        try:
            # Get model version
            if model_id not in self.model_registry or version not in self.model_registry[model_id]:
                raise ValueError(f"Model {model_id}:{version} not found in registry")
            
            model_version = self.model_registry[model_id][version]
            
            # Generate simulated performance metrics
            performance_metrics = PerformanceMetrics(
                accuracy=np.random.normal(0.85, 0.05),
                precision=np.random.normal(0.82, 0.05),
                recall=np.random.normal(0.88, 0.05),
                f1_score=np.random.normal(0.85, 0.05),
                latency_ms=np.random.normal(150, 30),
                throughput_rps=np.random.normal(1000, 200),
                error_rate=np.random.normal(0.02, 0.01),
                data_drift_score=np.random.normal(0.1, 0.05),
                model_drift_score=np.random.normal(0.05, 0.02),
                timestamp=datetime.now()
            )
            
            # Store performance metrics
            self.performance_history.append({
                "model_id": model_id,
                "version": version,
                "metrics": asdict(performance_metrics),
                "timestamp": datetime.now().isoformat()
            })
            
            # Check for performance degradation
            alerts = []
            if performance_metrics.accuracy < 0.8:
                alerts.append("Accuracy below threshold")
            if performance_metrics.latency_ms > 200:
                alerts.append("Latency above threshold")
            if performance_metrics.data_drift_score > 0.2:
                alerts.append("Data drift detected")
            if performance_metrics.model_drift_score > 0.1:
                alerts.append("Model drift detected")
            
            # Update model performance metrics
            model_version.performance_metrics = asdict(performance_metrics)
            self._save_model_registry()
            
            return {
                "model_id": model_id,
                "version": version,
                "performance_metrics": asdict(performance_metrics),
                "alerts": alerts,
                "monitoring_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Performance monitoring failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def trigger_retraining(
        self,
        model_id: str,
        trigger_reason: str,
        retraining_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Trigger automated retraining for a model.
        
        Args:
            model_id: Model identifier
            trigger_reason: Reason for retraining
            retraining_config: Retraining configuration
            
        Returns:
            Retraining trigger result
        """
        
        try:
            # Get latest production model
            if model_id not in self.model_registry:
                raise ValueError(f"Model {model_id} not found in registry")
            
            # Find latest production version
            production_versions = [
                (version, model_version) for version, model_version in self.model_registry[model_id].items()
                if model_version.status == ModelStatus.PRODUCTION
            ]
            
            if not production_versions:
                raise ValueError(f"No production version found for model {model_id}")
            
            latest_version, latest_model = max(production_versions, key=lambda x: x[1].created_at)
            
            # Create new version for retraining
            new_version = f"{latest_version.split('.')[0]}.{int(latest_version.split('.')[1]) + 1}.0"
            
            # Register new retraining version
            retraining_model = await self.register_model(
                model_id=model_id,
                version=new_version,
                stage=ModelStage.RETRAINING,
                created_by="automated_retraining",
                description=f"Automated retraining triggered by: {trigger_reason}",
                metadata={
                    "trigger_reason": trigger_reason,
                    "parent_version": latest_version,
                    "retraining_config": retraining_config or {}
                }
            )
            
            # Record retraining trigger
            self.deployment_history.append({
                "model_id": model_id,
                "version": new_version,
                "action": "retraining_triggered",
                "trigger_reason": trigger_reason,
                "parent_version": latest_version,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"Triggered retraining for model {model_id}, new version: {new_version}")
            
            return {
                "status": "success",
                "model_id": model_id,
                "new_version": new_version,
                "trigger_reason": trigger_reason,
                "parent_version": latest_version,
                "triggered_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Retraining trigger failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def get_model_versions(self, model_id: str) -> Dict[str, ModelVersion]:
        """Get all versions of a model"""
        return self.model_registry.get(model_id, {})
    
    def get_deployment_history(self, model_id: str = None) -> List[Dict[str, Any]]:
        """Get deployment history, optionally filtered by model"""
        if model_id:
            return [entry for entry in self.deployment_history if entry.get("model_id") == model_id]
        return self.deployment_history
    
    def get_performance_history(self, model_id: str = None, version: str = None) -> List[Dict[str, Any]]:
        """Get performance history, optionally filtered by model and version"""
        filtered_history = self.performance_history
        
        if model_id:
            filtered_history = [entry for entry in filtered_history if entry.get("model_id") == model_id]
        
        if version:
            filtered_history = [entry for entry in filtered_history if entry.get("version") == version]
        
        return filtered_history
    
    def get_model_registry_summary(self) -> Dict[str, Any]:
        """Get summary of the model registry"""
        total_models = len(self.model_registry)
        total_versions = sum(len(versions) for versions in self.model_registry.values())
        
        status_distribution = {}
        stage_distribution = {}
        
        for model_id, versions in self.model_registry.items():
            for version, model_version in versions.items():
                status = model_version.status.value
                stage = model_version.stage.value
                
                status_distribution[status] = status_distribution.get(status, 0) + 1
                stage_distribution[stage] = stage_distribution.get(stage, 0) + 1
        
        return {
            "total_models": total_models,
            "total_versions": total_versions,
            "status_distribution": status_distribution,
            "stage_distribution": stage_distribution,
            "last_updated": datetime.now().isoformat()
        }
