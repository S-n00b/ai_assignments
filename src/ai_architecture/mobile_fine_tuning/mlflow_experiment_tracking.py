"""
MLflow Experiment Tracking for Mobile Fine-tuning

This module provides MLflow integration for tracking fine-tuning experiments,
model performance, and deployment metrics for mobile-optimized models.
"""

import mlflow
import mlflow.pytorch
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import torch
import json
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import time

logger = logging.getLogger(__name__)

class MLflowFineTuningTracker:
    """
    MLflow tracker for mobile fine-tuning experiments.
    
    Provides comprehensive tracking of fine-tuning experiments, model performance,
    and deployment metrics for mobile-optimized models.
    """
    
    def __init__(self, 
                 experiment_name: str = "mobile_fine_tuning",
                 tracking_uri: str = "http://localhost:5000"):
        """
        Initialize MLflow fine-tuning tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.client = MlflowClient(tracking_uri=tracking_uri)
        
        # Setup MLflow
        mlflow.set_tracking_uri(tracking_uri)
        self.experiment_id = self._get_or_create_experiment()
        
    def _get_or_create_experiment(self) -> str:
        """Get or create MLflow experiment."""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created experiment: {self.experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {self.experiment_name}")
            
            return experiment_id
            
        except Exception as e:
            logger.error(f"Failed to setup experiment: {e}")
            raise
    
    def start_fine_tuning_run(self, 
                             run_name: str,
                             model_name: str,
                             base_model: str,
                             fine_tuning_config: Dict) -> str:
        """
        Start a new fine-tuning run.
        
        Args:
            run_name: Name of the run
            model_name: Name of the model being fine-tuned
            base_model: Base model name
            fine_tuning_config: Fine-tuning configuration
            
        Returns:
            Run ID
        """
        try:
            with mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name) as run:
                # Log run parameters
                mlflow.log_params({
                    "model_name": model_name,
                    "base_model": base_model,
                    "fine_tuning_method": fine_tuning_config.get("method", "full_fine_tuning"),
                    "learning_rate": fine_tuning_config.get("learning_rate", 2e-5),
                    "batch_size": fine_tuning_config.get("batch_size", 4),
                    "num_epochs": fine_tuning_config.get("num_epochs", 3),
                    "optimizer": fine_tuning_config.get("optimizer", "AdamW"),
                    "scheduler": fine_tuning_config.get("scheduler", "cosine"),
                    "warmup_steps": fine_tuning_config.get("warmup_steps", 100),
                    "weight_decay": fine_tuning_config.get("weight_decay", 0.01),
                    "max_grad_norm": fine_tuning_config.get("max_grad_norm", 1.0),
                    "fp16": fine_tuning_config.get("fp16", True),
                    "gradient_accumulation_steps": fine_tuning_config.get("gradient_accumulation_steps", 2)
                })
                
                # Log model configuration
                mlflow.log_params({
                    "model_size_mb": fine_tuning_config.get("model_size_mb", 0),
                    "num_parameters": fine_tuning_config.get("num_parameters", 0),
                    "vocab_size": fine_tuning_config.get("vocab_size", 0),
                    "max_length": fine_tuning_config.get("max_length", 512),
                    "target_modules": str(fine_tuning_config.get("target_modules", [])),
                    "adapter_dim": fine_tuning_config.get("adapter_dim", 16),
                    "rank": fine_tuning_config.get("rank", 8),
                    "alpha": fine_tuning_config.get("alpha", 16.0)
                })
                
                # Log mobile-specific parameters
                mlflow.log_params({
                    "mobile_optimization": fine_tuning_config.get("mobile_optimization", False),
                    "quantization": fine_tuning_config.get("quantization", "none"),
                    "pruning_ratio": fine_tuning_config.get("pruning_ratio", 0.0),
                    "target_device": fine_tuning_config.get("target_device", "mobile"),
                    "deployment_platform": fine_tuning_config.get("deployment_platform", "android"),
                    "memory_constraint_mb": fine_tuning_config.get("memory_constraint_mb", 512),
                    "latency_constraint_ms": fine_tuning_config.get("latency_constraint_ms", 100)
                })
                
                # Log dataset information
                mlflow.log_params({
                    "dataset_name": fine_tuning_config.get("dataset_name", "lenovo_domain"),
                    "dataset_size": fine_tuning_config.get("dataset_size", 0),
                    "train_samples": fine_tuning_config.get("train_samples", 0),
                    "eval_samples": fine_tuning_config.get("eval_samples", 0),
                    "test_samples": fine_tuning_config.get("test_samples", 0),
                    "data_quality_score": fine_tuning_config.get("data_quality_score", 0.0)
                })
                
                logger.info(f"Started fine-tuning run: {run_name}")
                return run.info.run_id
                
        except Exception as e:
            logger.error(f"Failed to start fine-tuning run: {e}")
            raise
    
    def log_training_metrics(self, 
                            run_id: str,
                            epoch: int,
                            train_loss: float,
                            eval_loss: float,
                            learning_rate: float,
                            additional_metrics: Optional[Dict] = None):
        """
        Log training metrics for a run.
        
        Args:
            run_id: MLflow run ID
            epoch: Current epoch
            train_loss: Training loss
            eval_loss: Evaluation loss
            learning_rate: Current learning rate
            additional_metrics: Additional metrics to log
        """
        try:
            with mlflow.start_run(run_id=run_id):
                # Log basic metrics
                mlflow.log_metrics({
                    f"train_loss_epoch_{epoch}": train_loss,
                    f"eval_loss_epoch_{epoch}": eval_loss,
                    f"learning_rate_epoch_{epoch}": learning_rate,
                    "epoch": epoch
                })
                
                # Log additional metrics if provided
                if additional_metrics:
                    mlflow.log_metrics(additional_metrics)
                
                logger.info(f"Logged training metrics for epoch {epoch}")
                
        except Exception as e:
            logger.error(f"Failed to log training metrics: {e}")
            raise
    
    def log_model_performance(self, 
                            run_id: str,
                            performance_metrics: Dict[str, float],
                            test_results: Optional[Dict] = None):
        """
        Log model performance metrics.
        
        Args:
            run_id: MLflow run ID
            performance_metrics: Performance metrics dictionary
            test_results: Additional test results
        """
        try:
            with mlflow.start_run(run_id=run_id):
                # Log performance metrics
                mlflow.log_metrics(performance_metrics)
                
                # Log test results if provided
                if test_results:
                    mlflow.log_metrics(test_results)
                
                logger.info(f"Logged model performance metrics")
                
        except Exception as e:
            logger.error(f"Failed to log model performance: {e}")
            raise
    
    def log_mobile_metrics(self, 
                          run_id: str,
                          mobile_metrics: Dict[str, float]):
        """
        Log mobile-specific metrics.
        
        Args:
            run_id: MLflow run ID
            mobile_metrics: Mobile deployment metrics
        """
        try:
            with mlflow.start_run(run_id=run_id):
                # Log mobile metrics
                mlflow.log_metrics(mobile_metrics)
                
                logger.info(f"Logged mobile metrics")
                
        except Exception as e:
            logger.error(f"Failed to log mobile metrics: {e}")
            raise
    
    def log_model_artifacts(self, 
                           run_id: str,
                           model_path: str,
                           artifacts: Optional[Dict[str, str]] = None):
        """
        Log model artifacts.
        
        Args:
            run_id: MLflow run ID
            model_path: Path to the model
            artifacts: Additional artifacts to log
        """
        try:
            with mlflow.start_run(run_id=run_id):
                # Log model
                mlflow.pytorch.log_model(
                    pytorch_model=torch.load(model_path),
                    artifact_path="model",
                    registered_model_name=f"mobile_{run_id}"
                )
                
                # Log additional artifacts
                if artifacts:
                    for name, path in artifacts.items():
                        mlflow.log_artifact(path, name)
                
                logger.info(f"Logged model artifacts for run {run_id}")
                
        except Exception as e:
            logger.error(f"Failed to log model artifacts: {e}")
            raise
    
    def log_deployment_info(self, 
                           run_id: str,
                           deployment_config: Dict,
                           deployment_metrics: Dict[str, float]):
        """
        Log deployment information.
        
        Args:
            run_id: MLflow run ID
            deployment_config: Deployment configuration
            deployment_metrics: Deployment performance metrics
        """
        try:
            with mlflow.start_run(run_id=run_id):
                # Log deployment configuration
                mlflow.log_params({
                    "deployment_platform": deployment_config.get("platform", "unknown"),
                    "deployment_method": deployment_config.get("method", "unknown"),
                    "optimization_level": deployment_config.get("optimization_level", "none"),
                    "quantization_method": deployment_config.get("quantization", "none"),
                    "pruning_ratio": deployment_config.get("pruning_ratio", 0.0),
                    "target_device": deployment_config.get("target_device", "unknown")
                })
                
                # Log deployment metrics
                mlflow.log_metrics(deployment_metrics)
                
                logger.info(f"Logged deployment information for run {run_id}")
                
        except Exception as e:
            logger.error(f"Failed to log deployment info: {e}")
            raise
    
    def compare_runs(self, 
                    run_ids: List[str],
                    metric_name: str = "eval_loss") -> Dict[str, Any]:
        """
        Compare multiple runs.
        
        Args:
            run_ids: List of run IDs to compare
            metric_name: Metric to compare
            
        Returns:
            Comparison results
        """
        try:
            comparison_results = {}
            
            for run_id in run_ids:
                run = self.client.get_run(run_id)
                metrics = run.data.metrics
                
                comparison_results[run_id] = {
                    "run_name": run.data.tags.get("mlflow.runName", run_id),
                    "metric_value": metrics.get(metric_name, 0.0),
                    "all_metrics": metrics
                }
            
            # Sort by metric value
            sorted_runs = sorted(
                comparison_results.items(),
                key=lambda x: x[1]["metric_value"]
            )
            
            logger.info(f"Compared {len(run_ids)} runs by {metric_name}")
            return {
                "comparison_results": comparison_results,
                "sorted_runs": sorted_runs,
                "best_run": sorted_runs[0] if sorted_runs else None
            }
            
        except Exception as e:
            logger.error(f"Failed to compare runs: {e}")
            raise
    
    def get_best_model(self, 
                      experiment_id: Optional[str] = None,
                      metric_name: str = "eval_loss",
                      ascending: bool = True) -> Dict[str, Any]:
        """
        Get the best model from experiment.
        
        Args:
            experiment_id: Experiment ID (uses current if None)
            metric_name: Metric to optimize
            ascending: Whether lower values are better
            
        Returns:
            Best model information
        """
        try:
            exp_id = experiment_id or self.experiment_id
            runs = self.client.search_runs(
                experiment_ids=[exp_id],
                order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"]
            )
            
            if not runs:
                raise ValueError("No runs found in experiment")
            
            best_run = runs[0]
            
            return {
                "run_id": best_run.info.run_id,
                "run_name": best_run.data.tags.get("mlflow.runName", best_run.info.run_id),
                "metric_value": best_run.data.metrics.get(metric_name, 0.0),
                "model_uri": f"runs:/{best_run.info.run_id}/model",
                "all_metrics": best_run.data.metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to get best model: {e}")
            raise
    
    def create_model_registry_entry(self, 
                                   run_id: str,
                                   model_name: str,
                                   version: str = "1.0.0",
                                   description: str = "") -> str:
        """
        Create model registry entry.
        
        Args:
            run_id: MLflow run ID
            model_name: Name for the model in registry
            version: Model version
            description: Model description
            
        Returns:
            Model version URI
        """
        try:
            # Get model URI
            model_uri = f"runs:/{run_id}/model"
            
            # Create model version
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags={
                    "version": version,
                    "description": description,
                    "mobile_optimized": True,
                    "fine_tuned": True
                }
            )
            
            logger.info(f"Created model registry entry: {model_name} v{version}")
            return model_version.version
            
        except Exception as e:
            logger.error(f"Failed to create model registry entry: {e}")
            raise
    
    def generate_experiment_report(self, 
                                 experiment_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive experiment report.
        
        Args:
            experiment_id: Experiment ID (uses current if None)
            
        Returns:
            Experiment report
        """
        try:
            exp_id = experiment_id or self.experiment_id
            runs = self.client.search_runs(experiment_ids=[exp_id])
            
            if not runs:
                return {"error": "No runs found in experiment"}
            
            # Collect run statistics
            run_count = len(runs)
            metrics_summary = {}
            
            # Aggregate metrics across all runs
            for run in runs:
                for metric_name, metric_value in run.data.metrics.items():
                    if metric_name not in metrics_summary:
                        metrics_summary[metric_name] = []
                    metrics_summary[metric_name].append(metric_value)
            
            # Calculate statistics
            metrics_stats = {}
            for metric_name, values in metrics_summary.items():
                metrics_stats[metric_name] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
            
            # Get best run
            best_run = self.get_best_model(exp_id)
            
            report = {
                "experiment_id": exp_id,
                "experiment_name": self.experiment_name,
                "run_count": run_count,
                "metrics_summary": metrics_stats,
                "best_run": best_run,
                "generated_at": time.time()
            }
            
            logger.info(f"Generated experiment report for {run_count} runs")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate experiment report: {e}")
            raise
