"""
MLflow Manager for Enterprise LLMOps

This module provides comprehensive MLflow integration for experiment tracking,
model registry, and deployment management in enterprise LLM operations.

Key Features:
- Comprehensive experiment tracking for LLM models
- Model versioning and registry management
- Model deployment and serving
- Performance monitoring and drift detection
- Integration with Ollama and other model serving platforms
- Custom metrics and logging for LLM-specific evaluations
"""

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import mlflow.tensorflow
import mlflow.pytorch
import mlflow.transformers
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path
import pickle
import joblib
import requests
import aiohttp
from concurrent.futures import ThreadPoolExecutor


@dataclass
class ExperimentConfig:
    """Configuration for MLflow experiments."""
    experiment_name: str
    tracking_uri: str = "http://localhost:5000"
    registry_uri: Optional[str] = None
    artifact_location: Optional[str] = None
    description: str = ""
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class ModelInfo:
    """Information about a registered model."""
    name: str
    version: str
    stage: str = "None"
    description: str = ""
    tags: Dict[str, str] = None
    metadata: Dict[str, Any] = None
    run_id: str = ""
    model_uri: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""
    model_name: str
    model_version: str
    target_uri: str
    config: Dict[str, Any] = None
    env_manager: str = "local"
    workers: int = 1
    timeout: int = 300
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}


class MLflowManager:
    """
    MLflow manager for enterprise LLM operations.
    
    This class provides comprehensive MLflow integration including experiment
    tracking, model registry, and deployment management.
    """
    
    def __init__(self, config: ExperimentConfig):
        """Initialize MLflow manager."""
        self.config = config
        self.logger = self._setup_logging()
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(config.tracking_uri)
        
        # Set registry URI if provided
        if config.registry_uri:
            mlflow.set_registry_uri(config.registry_uri)
        
        # Initialize experiment
        self.experiment_id = self._setup_experiment()
        
        # Active run tracking
        self.active_runs = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for MLflow manager."""
        logger = logging.getLogger("mlflow_manager")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _setup_experiment(self) -> str:
        """Setup MLflow experiment."""
        try:
            # Try to get existing experiment
            experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
            
            if experiment is None:
                # Create new experiment
                experiment_id = mlflow.create_experiment(
                    name=self.config.experiment_name,
                    artifact_location=self.config.artifact_location,
                    tags=self.config.tags
                )
                self.logger.info(f"Created experiment: {self.config.experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                self.logger.info(f"Using existing experiment: {self.config.experiment_name}")
            
            return experiment_id
            
        except Exception as e:
            self.logger.error(f"Failed to setup experiment: {e}")
            raise
    
    def start_run(
        self,
        run_name: str = None,
        tags: Dict[str, str] = None,
        description: str = ""
    ) -> str:
        """Start a new MLflow run."""
        try:
            # Set experiment
            mlflow.set_experiment(experiment_id=self.experiment_id)
            
            # Start run
            run = mlflow.start_run(run_name=run_name)
            run_id = run.info.run_id
            
            # Log run metadata
            if tags:
                mlflow.set_tags(tags)
            
            if description:
                mlflow.set_tag("description", description)
            
            # Log experiment config
            mlflow.log_params(asdict(self.config))
            
            # Track active run
            self.active_runs[run_id] = {
                "run_name": run_name,
                "start_time": datetime.now(),
                "tags": tags or {},
                "description": description
            }
            
            self.logger.info(f"Started run: {run_id}")
            return run_id
            
        except Exception as e:
            self.logger.error(f"Failed to start run: {e}")
            raise
    
    def end_run(self, run_id: str = None, status: str = "FINISHED"):
        """End an MLflow run."""
        try:
            if run_id is None:
                mlflow.end_run(status=status)
                self.logger.info("Ended current run")
            else:
                # End specific run (if it's active)
                if run_id in self.active_runs:
                    mlflow.end_run(status=status)
                    del self.active_runs[run_id]
                    self.logger.info(f"Ended run: {run_id}")
                else:
                    self.logger.warning(f"Run {run_id} not found in active runs")
                    
        except Exception as e:
            self.logger.error(f"Failed to end run: {e}")
    
    def log_llm_metrics(
        self,
        metrics: Dict[str, float],
        run_id: str = None
    ):
        """Log LLM-specific metrics."""
        try:
            # LLM-specific metrics
            llm_metrics = {
                "perplexity": metrics.get("perplexity", 0.0),
                "bleu_score": metrics.get("bleu_score", 0.0),
                "rouge_l": metrics.get("rouge_l", 0.0),
                "rouge_1": metrics.get("rouge_1", 0.0),
                "rouge_2": metrics.get("rouge_2", 0.0),
                "meteor_score": metrics.get("meteor_score", 0.0),
                "bert_score": metrics.get("bert_score", 0.0),
                "semantic_similarity": metrics.get("semantic_similarity", 0.0),
                "response_time_ms": metrics.get("response_time_ms", 0.0),
                "tokens_per_second": metrics.get("tokens_per_second", 0.0),
                "cost_per_token": metrics.get("cost_per_token", 0.0),
                "memory_usage_mb": metrics.get("memory_usage_mb", 0.0),
                "gpu_utilization": metrics.get("gpu_utilization", 0.0),
                "throughput_qps": metrics.get("throughput_qps", 0.0),
                "latency_p95": metrics.get("latency_p95", 0.0),
                "latency_p99": metrics.get("latency_p99", 0.0),
                "error_rate": metrics.get("error_rate", 0.0),
                "accuracy": metrics.get("accuracy", 0.0),
                "f1_score": metrics.get("f1_score", 0.0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0)
            }
            
            # Log metrics
            mlflow.log_metrics(llm_metrics)
            
            # Log additional metrics if provided
            for key, value in metrics.items():
                if key not in llm_metrics:
                    mlflow.log_metric(key, value)
            
            self.logger.info(f"Logged LLM metrics for run: {run_id or 'current'}")
            
        except Exception as e:
            self.logger.error(f"Failed to log LLM metrics: {e}")
    
    def log_llm_params(
        self,
        params: Dict[str, Any],
        run_id: str = None
    ):
        """Log LLM-specific parameters."""
        try:
            # LLM-specific parameters
            llm_params = {
                "model_name": params.get("model_name", ""),
                "model_size": params.get("model_size", ""),
                "model_type": params.get("model_type", ""),
                "temperature": params.get("temperature", 0.0),
                "max_tokens": params.get("max_tokens", 0),
                "top_p": params.get("top_p", 0.0),
                "top_k": params.get("top_k", 0),
                "frequency_penalty": params.get("frequency_penalty", 0.0),
                "presence_penalty": params.get("presence_penalty", 0.0),
                "learning_rate": params.get("learning_rate", 0.0),
                "batch_size": params.get("batch_size", 0),
                "num_epochs": params.get("num_epochs", 0),
                "optimizer": params.get("optimizer", ""),
                "scheduler": params.get("scheduler", ""),
                "dataset_name": params.get("dataset_name", ""),
                "dataset_size": params.get("dataset_size", 0),
                "validation_split": params.get("validation_split", 0.0),
                "test_split": params.get("test_split", 0.0),
                "preprocessing": params.get("preprocessing", ""),
                "augmentation": params.get("augmentation", ""),
                "hardware": params.get("hardware", ""),
                "gpu_count": params.get("gpu_count", 0),
                "gpu_memory_gb": params.get("gpu_memory_gb", 0.0)
            }
            
            # Log parameters
            mlflow.log_params(llm_params)
            
            # Log additional parameters if provided
            for key, value in params.items():
                if key not in llm_params:
                    mlflow.log_param(key, value)
            
            self.logger.info(f"Logged LLM parameters for run: {run_id or 'current'}")
            
        except Exception as e:
            self.logger.error(f"Failed to log LLM parameters: {e}")
    
    def log_model_artifacts(
        self,
        artifacts: Dict[str, Union[str, Path]],
        run_id: str = None
    ):
        """Log model artifacts."""
        try:
            for artifact_name, artifact_path in artifacts.items():
                if isinstance(artifact_path, Path):
                    artifact_path = str(artifact_path)
                
                if os.path.isfile(artifact_path):
                    mlflow.log_artifact(artifact_path, artifact_name)
                elif os.path.isdir(artifact_path):
                    mlflow.log_artifacts(artifact_path, artifact_name)
                else:
                    self.logger.warning(f"Artifact path not found: {artifact_path}")
            
            self.logger.info(f"Logged artifacts for run: {run_id or 'current'}")
            
        except Exception as e:
            self.logger.error(f"Failed to log artifacts: {e}")
    
    def log_model(
        self,
        model: Any,
        model_name: str,
        model_type: str = "custom",
        signature: mlflow.models.ModelSignature = None,
        input_example: Any = None,
        conda_env: Dict[str, Any] = None,
        code_paths: List[str] = None,
        run_id: str = None
    ):
        """Log model to MLflow."""
        try:
            # Create conda environment if not provided
            if conda_env is None:
                conda_env = {
                    "name": f"{model_name}_env",
                    "channels": ["conda-forge", "defaults"],
                    "dependencies": [
                        "python=3.9",
                        "pip",
                        {
                            "pip": [
                                "mlflow",
                                "pandas",
                                "numpy",
                                "scikit-learn",
                                "torch",
                                "transformers",
                                "ollama"
                            ]
                        }
                    ]
                }
            
            # Log model based on type
            if model_type == "sklearn":
                mlflow.sklearn.log_model(
                    model, model_name,
                    signature=signature,
                    input_example=input_example,
                    conda_env=conda_env,
                    code_paths=code_paths
                )
            elif model_type == "pytorch":
                mlflow.pytorch.log_model(
                    model, model_name,
                    signature=signature,
                    input_example=input_example,
                    conda_env=conda_env,
                    code_paths=code_paths
                )
            elif model_type == "tensorflow":
                mlflow.tensorflow.log_model(
                    model, model_name,
                    signature=signature,
                    input_example=input_example,
                    conda_env=conda_env,
                    code_paths=code_paths
                )
            elif model_type == "transformers":
                mlflow.transformers.log_model(
                    model, model_name,
                    signature=signature,
                    input_example=input_example,
                    conda_env=conda_env,
                    code_paths=code_paths
                )
            else:
                # Custom model using pyfunc
                mlflow.pyfunc.log_model(
                    model, model_name,
                    signature=signature,
                    input_example=input_example,
                    conda_env=conda_env,
                    code_paths=code_paths
                )
            
            self.logger.info(f"Logged model {model_name} for run: {run_id or 'current'}")
            
        except Exception as e:
            self.logger.error(f"Failed to log model: {e}")
            raise
    
    def register_model(
        self,
        model_name: str,
        model_uri: str,
        description: str = "",
        tags: Dict[str, str] = None,
        run_id: str = None
    ) -> ModelInfo:
        """Register model in MLflow Model Registry."""
        try:
            # Register model
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags=tags or {}
            )
            
            # Create model info
            model_info = ModelInfo(
                name=model_name,
                version=model_version.version,
                stage="None",
                description=description,
                tags=tags or {},
                run_id=run_id or mlflow.active_run().info.run_id if mlflow.active_run() else "",
                model_uri=model_uri
            )
            
            # Update description if provided
            if description:
                client = mlflow.tracking.MlflowClient()
                client.update_model_version(
                    name=model_name,
                    version=model_version.version,
                    description=description
                )
            
            self.logger.info(f"Registered model: {model_name} v{model_version.version}")
            return model_info
            
        except Exception as e:
            self.logger.error(f"Failed to register model: {e}")
            raise
    
    def get_model_info(self, model_name: str, version: str = None) -> ModelInfo:
        """Get information about a registered model."""
        try:
            client = mlflow.tracking.MlflowClient()
            
            if version is None:
                # Get latest version
                model_version = client.get_latest_versions(
                    name=model_name,
                    stages=["None", "Staging", "Production", "Archived"]
                )[0]
                version = model_version.version
            
            # Get model version details
            model_version = client.get_model_version(
                name=model_name,
                version=version
            )
            
            model_info = ModelInfo(
                name=model_name,
                version=version,
                stage=model_version.current_stage,
                description=model_version.description or "",
                tags=model_version.tags or {},
                run_id=model_version.run_id,
                model_uri=model_version.source
            )
            
            return model_info
            
        except Exception as e:
            self.logger.error(f"Failed to get model info: {e}")
            raise
    
    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing_versions: bool = False
    ) -> bool:
        """Transition model to a new stage."""
        try:
            client = mlflow.tracking.MlflowClient()
            
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing_versions
            )
            
            self.logger.info(f"Transitioned {model_name} v{version} to {stage}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to transition model stage: {e}")
            return False
    
    def list_models(self, stage: str = None) -> List[ModelInfo]:
        """List all registered models."""
        try:
            client = mlflow.tracking.MlflowClient()
            
            # Get registered models
            registered_models = client.search_registered_models()
            
            models = []
            for registered_model in registered_models:
                # Get latest version
                latest_version = client.get_latest_versions(
                    name=registered_model.name,
                    stages=[stage] if stage else ["None", "Staging", "Production", "Archived"]
                )
                
                if latest_version:
                    model_version = latest_version[0]
                    model_info = ModelInfo(
                        name=registered_model.name,
                        version=model_version.version,
                        stage=model_version.current_stage,
                        description=model_version.description or "",
                        tags=model_version.tags or {},
                        run_id=model_version.run_id,
                        model_uri=model_version.source
                    )
                    models.append(model_info)
            
            return models
            
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            return []
    
    def load_model(self, model_name: str, version: str = None, stage: str = None):
        """Load a registered model."""
        try:
            if stage:
                model_uri = f"models:/{model_name}/{stage}"
            elif version:
                model_uri = f"models:/{model_name}/{version}"
            else:
                model_uri = f"models:/{model_name}/latest"
            
            model = mlflow.pyfunc.load_model(model_uri)
            
            self.logger.info(f"Loaded model: {model_uri}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def deploy_model(
        self,
        deployment_config: DeploymentConfig,
        run_id: str = None
    ) -> Dict[str, Any]:
        """Deploy a registered model."""
        try:
            # Get model URI
            if deployment_config.model_version:
                model_uri = f"models:/{deployment_config.model_name}/{deployment_config.model_version}"
            else:
                model_uri = f"models:/{deployment_config.model_name}/latest"
            
            # Deploy model using MLflow
            deployment = mlflow.deployments.get_deployment_client(deployment_config.target_uri)
            
            deployment_config_dict = {
                "model_uri": model_uri,
                "name": f"{deployment_config.model_name}_{deployment_config.model_version}",
                "config": deployment_config.config,
                "env_manager": deployment_config.env_manager,
                "workers": deployment_config.workers,
                "timeout": deployment_config.timeout
            }
            
            # Create deployment
            deployment_result = deployment.create_deployment(
                name=deployment_config_dict["name"],
                model_uri=model_uri,
                config=deployment_config.config
            )
            
            # Log deployment info
            if run_id:
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_param("deployment_name", deployment_result["name"])
                    mlflow.log_param("deployment_uri", deployment_result["url"])
                    mlflow.log_param("deployment_config", deployment_config_dict)
            
            self.logger.info(f"Deployed model: {deployment_result['name']}")
            
            return deployment_result
            
        except Exception as e:
            self.logger.error(f"Failed to deploy model: {e}")
            raise
    
    def get_run_history(
        self,
        experiment_id: str = None,
        max_results: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get run history for analysis."""
        try:
            client = mlflow.tracking.MlflowClient()
            
            # Get experiment ID
            if experiment_id is None:
                experiment_id = self.experiment_id
            
            # Search runs
            runs = client.search_runs(
                experiment_ids=[experiment_id],
                max_results=max_results
            )
            
            run_history = []
            for run in runs:
                run_data = {
                    "run_id": run.info.run_id,
                    "run_name": run.data.tags.get("mlflow.runName", ""),
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "user_id": run.info.user_id,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "tags": run.data.tags,
                    "artifact_uri": run.info.artifact_uri
                }
                run_history.append(run_data)
            
            return run_history
            
        except Exception as e:
            self.logger.error(f"Failed to get run history: {e}")
            return []
    
    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """Compare multiple runs."""
        try:
            client = mlflow.tracking.MlflowClient()
            
            # Get run data
            runs_data = []
            for run_id in run_ids:
                run = client.get_run(run_id)
                run_data = {
                    "run_id": run_id,
                    "run_name": run.data.tags.get("mlflow.runName", ""),
                    **run.data.metrics,
                    **run.data.params
                }
                runs_data.append(run_data)
            
            # Create comparison DataFrame
            comparison_df = pd.DataFrame(runs_data)
            
            return comparison_df
            
        except Exception as e:
            self.logger.error(f"Failed to compare runs: {e}")
            return pd.DataFrame()
    
    def export_model_artifacts(
        self,
        model_name: str,
        version: str,
        export_path: str
    ) -> bool:
        """Export model artifacts to local path."""
        try:
            # Get model info
            model_info = self.get_model_info(model_name, version)
            
            # Download artifacts
            client = mlflow.tracking.MlflowClient()
            client.download_artifacts(
                run_id=model_info.run_id,
                path="",
                dst_path=export_path
            )
            
            self.logger.info(f"Exported model artifacts to: {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export model artifacts: {e}")
            return False
    
    def cleanup_old_runs(self, days_to_keep: int = 30) -> int:
        """Cleanup old runs to save storage."""
        try:
            client = mlflow.tracking.MlflowClient()
            
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            cutoff_timestamp = int(cutoff_date.timestamp() * 1000)
            
            # Search for old runs
            old_runs = client.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=f"start_time < {cutoff_timestamp}",
                max_results=1000
            )
            
            # Delete old runs
            deleted_count = 0
            for run in old_runs:
                try:
                    client.delete_run(run.info.run_id)
                    deleted_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to delete run {run.info.run_id}: {e}")
            
            self.logger.info(f"Cleaned up {deleted_count} old runs")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old runs: {e}")
            return 0
