"""
Unified Experiment Tracking for Model Evaluation

This module provides unified experiment tracking for all model evaluation
components with comprehensive MLflow integration.

Key Features:
- Unified experiment tracking across all components
- Cross-component experiment correlation
- Comprehensive metrics collection
- Experiment comparison and analysis
"""

import logging
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid

# MLflow imports
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pytorch
    from mlflow.tracking import MlflowClient
    from mlflow.entities import Run, Experiment
except ImportError:
    mlflow = None
    MlflowClient = None
    Run = None
    Experiment = None

logger = logging.getLogger(__name__)


@dataclass
class UnifiedExperiment:
    """Unified experiment configuration."""
    experiment_id: str
    experiment_name: str
    experiment_type: str  # "foundation", "custom", "mobile", "agentic", "retrieval", "production"
    components: List[str]
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    tags: Dict[str, str]
    artifacts: List[str]
    created_at: datetime
    status: str = "RUNNING"  # RUNNING, COMPLETED, FAILED
    run_id: Optional[str] = None


@dataclass
class CrossComponentCorrelation:
    """Cross-component experiment correlation."""
    correlation_id: str
    primary_experiment_id: str
    related_experiment_ids: List[str]
    correlation_type: str  # "dependency", "comparison", "pipeline"
    correlation_metrics: Dict[str, float]
    created_at: datetime


class UnifiedExperimentTracker:
    """
    Unified Experiment Tracker for Model Evaluation
    
    This class provides unified experiment tracking for all model evaluation
    components with comprehensive MLflow integration.
    """
    
    def __init__(self, 
                 tracking_uri: str = "http://localhost:5000",
                 registry_uri: Optional[str] = None):
        """
        Initialize the Unified Experiment Tracker.
        
        Args:
            tracking_uri: MLflow tracking server URI
            registry_uri: MLflow model registry URI
        """
        if mlflow is None:
            raise ImportError("MLflow is not installed. Please install it with: pip install mlflow")
        
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri
        
        # Setup MLflow
        mlflow.set_tracking_uri(tracking_uri)
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)
        
        self.client = MlflowClient(tracking_uri=tracking_uri)
        
        # Experiment tracking
        self.active_experiments: Dict[str, UnifiedExperiment] = {}
        self.completed_experiments: Dict[str, UnifiedExperiment] = {}
        self.correlations: Dict[str, CrossComponentCorrelation] = {}
        
        # Ensure default experiments exist
        self._ensure_default_experiments()
        
        logger.info("Unified Experiment Tracker initialized")
    
    def _ensure_default_experiments(self):
        """Ensure default experiments exist."""
        experiments = [
            "foundation_model_evaluation",
            "custom_model_evaluation", 
            "mobile_model_evaluation",
            "agentic_workflow_evaluation",
            "retrieval_workflow_evaluation",
            "production_deployment"
        ]
        
        for exp_name in experiments:
            try:
                experiment = self.client.get_experiment_by_name(exp_name)
                if experiment is None:
                    experiment_id = self.client.create_experiment(exp_name)
                    logger.info(f"Created experiment: {exp_name}")
                else:
                    logger.info(f"Using existing experiment: {exp_name}")
            except Exception as e:
                logger.warning(f"Could not ensure experiment {exp_name}: {e}")
    
    def start_unified_experiment(self,
                               experiment_name: str,
                               experiment_type: str,
                               components: List[str],
                               parameters: Dict[str, Any],
                               tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start a unified experiment.
        
        Args:
            experiment_name: Name of the experiment
            experiment_type: Type of the experiment
            components: List of components involved
            parameters: Experiment parameters
            tags: Experiment tags
            
        Returns:
            Experiment ID
        """
        experiment_id = str(uuid.uuid4())
        
        if tags is None:
            tags = {}
        
        # Add default tags
        tags.update({
            "experiment_type": experiment_type,
            "components": ",".join(components),
            "created_at": datetime.now().isoformat()
        })
        
        experiment = UnifiedExperiment(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            experiment_type=experiment_type,
            components=components,
            parameters=parameters,
            metrics={},
            tags=tags,
            artifacts=[],
            created_at=datetime.now()
        )
        
        self.active_experiments[experiment_id] = experiment
        
        # Start MLflow run
        try:
            with mlflow.start_run(experiment_id=self._get_experiment_id(experiment_type)):
                run_id = mlflow.active_run().info.run_id
                experiment.run_id = run_id
                
                # Log parameters
                mlflow.log_params(parameters)
                
                # Log tags
                mlflow.set_tags(tags)
                
                logger.info(f"Started unified experiment {experiment_name} with ID {experiment_id}")
                
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
        
        return experiment_id
    
    def log_component_metrics(self,
                             experiment_id: str,
                             component: str,
                             metrics: Dict[str, float]) -> bool:
        """
        Log metrics for a specific component.
        
        Args:
            experiment_id: Experiment ID
            component: Component name
            metrics: Metrics to log
            
        Returns:
            Success status
        """
        if experiment_id not in self.active_experiments:
            logger.error(f"Experiment {experiment_id} not found")
            return False
        
        experiment = self.active_experiments[experiment_id]
        
        # Update experiment metrics
        for key, value in metrics.items():
            experiment.metrics[f"{component}_{key}"] = value
        
        # Log to MLflow
        try:
            if experiment.run_id:
                with mlflow.start_run(run_id=experiment.run_id):
                    mlflow.log_metrics(metrics)
                    logger.info(f"Logged metrics for component {component} in experiment {experiment_id}")
                    return True
        except Exception as e:
            logger.error(f"Failed to log metrics to MLflow: {e}")
            return False
        
        return False
    
    def log_component_artifacts(self,
                               experiment_id: str,
                               component: str,
                               artifacts: List[str]) -> bool:
        """
        Log artifacts for a specific component.
        
        Args:
            experiment_id: Experiment ID
            component: Component name
            artifacts: List of artifact paths
            
        Returns:
            Success status
        """
        if experiment_id not in self.active_experiments:
            logger.error(f"Experiment {experiment_id} not found")
            return False
        
        experiment = self.active_experiments[experiment_id]
        
        # Update experiment artifacts
        experiment.artifacts.extend(artifacts)
        
        # Log to MLflow
        try:
            if experiment.run_id:
                with mlflow.start_run(run_id=experiment.run_id):
                    for artifact_path in artifacts:
                        mlflow.log_artifacts(artifact_path)
                    logger.info(f"Logged artifacts for component {component} in experiment {experiment_id}")
                    return True
        except Exception as e:
            logger.error(f"Failed to log artifacts to MLflow: {e}")
            return False
        
        return False
    
    def create_cross_component_correlation(self,
                                          primary_experiment_id: str,
                                          related_experiment_ids: List[str],
                                          correlation_type: str,
                                          correlation_metrics: Dict[str, float]) -> str:
        """
        Create cross-component correlation.
        
        Args:
            primary_experiment_id: Primary experiment ID
            related_experiment_ids: Related experiment IDs
            correlation_type: Type of correlation
            correlation_metrics: Correlation metrics
            
        Returns:
            Correlation ID
        """
        correlation_id = str(uuid.uuid4())
        
        correlation = CrossComponentCorrelation(
            correlation_id=correlation_id,
            primary_experiment_id=primary_experiment_id,
            related_experiment_ids=related_experiment_ids,
            correlation_type=correlation_type,
            correlation_metrics=correlation_metrics,
            created_at=datetime.now()
        )
        
        self.correlations[correlation_id] = correlation
        
        logger.info(f"Created cross-component correlation {correlation_id}")
        
        return correlation_id
    
    def complete_experiment(self,
                           experiment_id: str,
                           final_metrics: Optional[Dict[str, float]] = None,
                           final_artifacts: Optional[List[str]] = None) -> bool:
        """
        Complete an experiment.
        
        Args:
            experiment_id: Experiment ID
            final_metrics: Final metrics
            final_artifacts: Final artifacts
            
        Returns:
            Success status
        """
        if experiment_id not in self.active_experiments:
            logger.error(f"Experiment {experiment_id} not found")
            return False
        
        experiment = self.active_experiments[experiment_id]
        
        # Update final metrics
        if final_metrics:
            experiment.metrics.update(final_metrics)
        
        # Update final artifacts
        if final_artifacts:
            experiment.artifacts.extend(final_artifacts)
        
        # Complete MLflow run
        try:
            if experiment.run_id:
                with mlflow.start_run(run_id=experiment.run_id):
                    if final_metrics:
                        mlflow.log_metrics(final_metrics)
                    if final_artifacts:
                        for artifact_path in final_artifacts:
                            mlflow.log_artifacts(artifact_path)
                    
                    # Mark run as completed
                    mlflow.end_run()
                    logger.info(f"Completed experiment {experiment_id}")
        
        except Exception as e:
            logger.error(f"Failed to complete MLflow run: {e}")
        
        # Move to completed experiments
        experiment.status = "COMPLETED"
        self.completed_experiments[experiment_id] = experiment
        del self.active_experiments[experiment_id]
        
        return True
    
    def fail_experiment(self,
                       experiment_id: str,
                       error_message: str) -> bool:
        """
        Mark an experiment as failed.
        
        Args:
            experiment_id: Experiment ID
            error_message: Error message
            
        Returns:
            Success status
        """
        if experiment_id not in self.active_experiments:
            logger.error(f"Experiment {experiment_id} not found")
            return False
        
        experiment = self.active_experiments[experiment_id]
        
        # Update experiment status
        experiment.status = "FAILED"
        experiment.tags["error_message"] = error_message
        
        # End MLflow run
        try:
            if experiment.run_id:
                with mlflow.start_run(run_id=experiment.run_id):
                    mlflow.set_tag("status", "FAILED")
                    mlflow.set_tag("error_message", error_message)
                    mlflow.end_run()
                    logger.info(f"Failed experiment {experiment_id}: {error_message}")
        
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {e}")
        
        # Move to completed experiments
        self.completed_experiments[experiment_id] = experiment
        del self.active_experiments[experiment_id]
        
        return True
    
    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment status."""
        if experiment_id in self.active_experiments:
            experiment = self.active_experiments[experiment_id]
            return {
                "experiment_id": experiment_id,
                "status": experiment.status,
                "experiment_name": experiment.experiment_name,
                "experiment_type": experiment.experiment_type,
                "components": experiment.components,
                "created_at": experiment.created_at.isoformat()
            }
        elif experiment_id in self.completed_experiments:
            experiment = self.completed_experiments[experiment_id]
            return {
                "experiment_id": experiment_id,
                "status": experiment.status,
                "experiment_name": experiment.experiment_name,
                "experiment_type": experiment.experiment_type,
                "components": experiment.components,
                "created_at": experiment.created_at.isoformat(),
                "metrics": experiment.metrics,
                "artifacts": experiment.artifacts
            }
        else:
            return {"error": "Experiment not found"}
    
    def get_cross_component_analysis(self, experiment_id: str) -> Dict[str, Any]:
        """Get cross-component analysis for an experiment."""
        # Find correlations involving this experiment
        correlations = [
            corr for corr in self.correlations.values()
            if experiment_id in [corr.primary_experiment_id] + corr.related_experiment_ids
        ]
        
        analysis = {
            "experiment_id": experiment_id,
            "correlations": [],
            "related_experiments": [],
            "correlation_metrics": {}
        }
        
        for correlation in correlations:
            analysis["correlations"].append({
                "correlation_id": correlation.correlation_id,
                "correlation_type": correlation.correlation_type,
                "related_experiments": correlation.related_experiment_ids,
                "metrics": correlation.correlation_metrics
            })
            
            analysis["related_experiments"].extend(correlation.related_experiment_ids)
            analysis["correlation_metrics"].update(correlation.correlation_metrics)
        
        return analysis
    
    def _get_experiment_id(self, experiment_type: str) -> str:
        """Get MLflow experiment ID for a type."""
        try:
            experiment = self.client.get_experiment_by_name(experiment_type)
            return experiment.experiment_id if experiment else None
        except Exception:
            return None
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments."""
        experiments = []
        
        # Active experiments
        for experiment_id, experiment in self.active_experiments.items():
            experiments.append({
                "experiment_id": experiment_id,
                "status": experiment.status,
                "experiment_name": experiment.experiment_name,
                "experiment_type": experiment.experiment_type,
                "components": experiment.components,
                "created_at": experiment.created_at.isoformat()
            })
        
        # Completed experiments
        for experiment_id, experiment in self.completed_experiments.items():
            experiments.append({
                "experiment_id": experiment_id,
                "status": experiment.status,
                "experiment_name": experiment.experiment_name,
                "experiment_type": experiment.experiment_type,
                "components": experiment.components,
                "created_at": experiment.created_at.isoformat(),
                "metrics": experiment.metrics,
                "artifacts": experiment.artifacts
            })
        
        return experiments


# Example usage and testing
if __name__ == "__main__":
    # Initialize tracker
    tracker = UnifiedExperimentTracker()
    
    # Start unified experiment
    experiment_id = tracker.start_unified_experiment(
        experiment_name="comprehensive_model_evaluation",
        experiment_type="custom",
        components=["foundation", "custom", "mobile", "agentic", "retrieval"],
        parameters={
            "model_name": "lenovo-device-support-v1",
            "evaluation_type": "comprehensive"
        },
        tags={
            "project": "lenovo_ai_architecture",
            "phase": "phase5"
        }
    )
    
    print(f"Started experiment: {experiment_id}")
    
    # Log component metrics
    tracker.log_component_metrics(
        experiment_id=experiment_id,
        component="foundation",
        metrics={"accuracy": 0.85, "latency": 2.1}
    )
    
    # Complete experiment
    tracker.complete_experiment(
        experiment_id=experiment_id,
        final_metrics={"overall_score": 0.88}
    )
    
    print("Experiment completed")
