"""
MLflow Agent Tracking

This module provides MLflow integration for tracking SmolAgent workflows,
including experiment tracking, model registry, and performance metrics.
"""

import logging
import json
import time
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
    from mlflow.models import Model
except ImportError:
    mlflow = None
    MlflowClient = None
    Run = None
    Experiment = None
    Model = None

logger = logging.getLogger(__name__)


@dataclass
class AgentExperiment:
    """Agent experiment configuration."""
    experiment_name: str
    workflow_name: str
    agent_name: str
    model_name: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    tags: Dict[str, str]
    artifacts: List[str]
    run_id: Optional[str] = None
    status: str = "RUNNING"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


@dataclass
class AgentModel:
    """Agent model configuration."""
    model_name: str
    model_version: str
    model_stage: str
    model_path: str
    model_type: str
    description: str
    tags: Dict[str, str]
    created_time: datetime
    last_updated: datetime


class MLflowAgentTracker:
    """
    MLflow Agent Tracker for SmolAgent workflows.
    
    This class provides comprehensive MLflow integration for tracking
    agent experiments, model registry, and performance metrics.
    """
    
    def __init__(self, tracking_uri: Optional[str] = None, 
                 registry_uri: Optional[str] = None):
        """
        Initialize the MLflow Agent Tracker.
        
        Args:
            tracking_uri: MLflow tracking URI
            registry_uri: MLflow model registry URI
        """
        if mlflow is None:
            raise ImportError("MLflow is not installed. Please install it with: pip install mlflow")
        
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Initialize MLflow client
        self.client = MlflowClient()
        
        # Set registry URI
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)
        
        # Agent experiments tracking
        self.active_experiments: Dict[str, AgentExperiment] = {}
        self.completed_experiments: Dict[str, AgentExperiment] = {}
        
        # Model registry
        self.registered_models: Dict[str, AgentModel] = {}
        
        # Create default experiment if it doesn't exist
        self._ensure_default_experiment()
        
        logger.info("MLflow Agent Tracker initialized")
    
    def _ensure_default_experiment(self):
        """Ensure default experiment exists."""
        try:
            experiment_name = "smolagent_workflows"
            experiment = self.client.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = self.client.create_experiment(experiment_name)
                logger.info(f"Created default experiment: {experiment_name}")
            else:
                logger.info(f"Using existing experiment: {experiment_name}")
        except Exception as e:
            logger.warning(f"Could not ensure default experiment: {e}")
    
    def start_experiment(self, experiment_name: str, 
                        workflow_name: str, 
                        agent_name: str,
                        model_name: str,
                        parameters: Dict[str, Any],
                        tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start a new agent experiment.
        
        Args:
            experiment_name: Name of the experiment
            workflow_name: Name of the workflow
            agent_name: Name of the agent
            model_name: Name of the model
            parameters: Experiment parameters
            tags: Optional tags
            
        Returns:
            Experiment ID
        """
        try:
            # Create experiment if it doesn't exist
            try:
                experiment = self.client.get_experiment_by_name(experiment_name)
                if experiment is None:
                    experiment_id = self.client.create_experiment(experiment_name)
                else:
                    experiment_id = experiment.experiment_id
            except Exception:
                experiment_id = self.client.create_experiment(experiment_name)
            
            # Start MLflow run
            with mlflow.start_run(experiment_id=experiment_id) as run:
                run_id = run.info.run_id
                
                # Log parameters
                mlflow.log_params(parameters)
                
                # Log tags
                if tags:
                    mlflow.set_tags(tags)
                
                # Log workflow and agent information
                mlflow.set_tag("workflow_name", workflow_name)
                mlflow.set_tag("agent_name", agent_name)
                mlflow.set_tag("model_name", model_name)
                mlflow.set_tag("experiment_type", "agent_workflow")
                
                # Create agent experiment
                agent_experiment = AgentExperiment(
                    experiment_name=experiment_name,
                    workflow_name=workflow_name,
                    agent_name=agent_name,
                    model_name=model_name,
                    parameters=parameters,
                    metrics={},
                    tags=tags or {},
                    artifacts=[],
                    run_id=run_id,
                    status="RUNNING",
                    start_time=datetime.now()
                )
                
                # Store active experiment
                self.active_experiments[run_id] = agent_experiment
                
                logger.info(f"Started experiment: {experiment_name} (run_id: {run_id})")
                return run_id
                
        except Exception as e:
            logger.error(f"Failed to start experiment: {e}")
            raise
    
    def log_metrics(self, run_id: str, metrics: Dict[str, float]):
        """
        Log metrics for an experiment.
        
        Args:
            run_id: Experiment run ID
            metrics: Metrics to log
        """
        try:
            if run_id not in self.active_experiments:
                raise ValueError(f"Experiment {run_id} not found")
            
            # Log metrics to MLflow
            mlflow.log_metrics(metrics)
            
            # Update agent experiment
            agent_experiment = self.active_experiments[run_id]
            agent_experiment.metrics.update(metrics)
            
            logger.debug(f"Logged metrics for experiment {run_id}: {metrics}")
            
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
            raise
    
    def log_artifacts(self, run_id: str, artifacts: List[str]):
        """
        Log artifacts for an experiment.
        
        Args:
            run_id: Experiment run ID
            artifacts: List of artifact paths
        """
        try:
            if run_id not in self.active_experiments:
                raise ValueError(f"Experiment {run_id} not found")
            
            # Log artifacts to MLflow
            for artifact_path in artifacts:
                mlflow.log_artifact(artifact_path)
            
            # Update agent experiment
            agent_experiment = self.active_experiments[run_id]
            agent_experiment.artifacts.extend(artifacts)
            
            logger.debug(f"Logged artifacts for experiment {run_id}: {artifacts}")
            
        except Exception as e:
            logger.error(f"Failed to log artifacts: {e}")
            raise
    
    def end_experiment(self, run_id: str, status: str = "FINISHED"):
        """
        End an experiment.
        
        Args:
            run_id: Experiment run ID
            status: Final status
        """
        try:
            if run_id not in self.active_experiments:
                raise ValueError(f"Experiment {run_id} not found")
            
            # Update agent experiment
            agent_experiment = self.active_experiments[run_id]
            agent_experiment.status = status
            agent_experiment.end_time = datetime.now()
            
            # Move to completed experiments
            self.completed_experiments[run_id] = agent_experiment
            del self.active_experiments[run_id]
            
            # End MLflow run
            mlflow.end_run()
            
            logger.info(f"Ended experiment {run_id} with status: {status}")
            
        except Exception as e:
            logger.error(f"Failed to end experiment: {e}")
            raise
    
    def get_experiment(self, run_id: str) -> Optional[AgentExperiment]:
        """Get experiment by run ID."""
        if run_id in self.active_experiments:
            return self.active_experiments[run_id]
        elif run_id in self.completed_experiments:
            return self.completed_experiments[run_id]
        return None
    
    def list_experiments(self, status: Optional[str] = None) -> List[AgentExperiment]:
        """
        List experiments.
        
        Args:
            status: Filter by status (optional)
            
        Returns:
            List of experiments
        """
        experiments = []
        
        # Add active experiments
        for experiment in self.active_experiments.values():
            if status is None or experiment.status == status:
                experiments.append(experiment)
        
        # Add completed experiments
        for experiment in self.completed_experiments.values():
            if status is None or experiment.status == status:
                experiments.append(experiment)
        
        return experiments
    
    def register_model(self, model_name: str, 
                     model_path: str,
                     model_type: str = "agent_model",
                     description: str = "",
                     tags: Optional[Dict[str, str]] = None) -> str:
        """
        Register a model in MLflow model registry.
        
        Args:
            model_name: Name of the model
            model_path: Path to the model
            model_type: Type of the model
            description: Model description
            tags: Optional tags
            
        Returns:
            Model version
        """
        try:
            # Create model if it doesn't exist
            try:
                model = self.client.get_registered_model(model_name)
            except Exception:
                self.client.create_registered_model(
                    name=model_name,
                    description=description,
                    tags=tags or {}
                )
            
            # Create model version
            model_version = self.client.create_model_version(
                name=model_name,
                source=model_path,
                description=description
            )
            
            # Create agent model
            agent_model = AgentModel(
                model_name=model_name,
                model_version=model_version.version,
                model_stage="None",
                model_path=model_path,
                model_type=model_type,
                description=description,
                tags=tags or {},
                created_time=datetime.now(),
                last_updated=datetime.now()
            )
            
            # Store registered model
            model_key = f"{model_name}_{model_version.version}"
            self.registered_models[model_key] = agent_model
            
            logger.info(f"Registered model: {model_name} version {model_version.version}")
            return model_version.version
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    def get_model(self, model_name: str, version: Optional[str] = None) -> Optional[AgentModel]:
        """
        Get a registered model.
        
        Args:
            model_name: Name of the model
            version: Model version (optional, latest if not specified)
            
        Returns:
            Agent model
        """
        try:
            if version is None:
                # Get latest version
                model_versions = self.client.get_latest_versions(model_name)
                if not model_versions:
                    return None
                version = model_versions[0].version
            
            model_key = f"{model_name}_{version}"
            return self.registered_models.get(model_key)
            
        except Exception as e:
            logger.error(f"Failed to get model: {e}")
            return None
    
    def list_models(self, model_name: Optional[str] = None) -> List[AgentModel]:
        """
        List registered models.
        
        Args:
            model_name: Filter by model name (optional)
            
        Returns:
            List of registered models
        """
        try:
            if model_name:
                # Get specific model versions
                model_versions = self.client.get_latest_versions(model_name)
                models = []
                for version in model_versions:
                    model_key = f"{model_name}_{version.version}"
                    if model_key in self.registered_models:
                        models.append(self.registered_models[model_key])
                return models
            else:
                # Get all models
                return list(self.registered_models.values())
                
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def transition_model_stage(self, model_name: str, 
                             version: str, 
                             stage: str) -> bool:
        """
        Transition model to a new stage.
        
        Args:
            model_name: Name of the model
            version: Model version
            stage: New stage
            
        Returns:
            True if successful
        """
        try:
            # Transition model stage
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            
            # Update agent model
            model_key = f"{model_name}_{version}"
            if model_key in self.registered_models:
                self.registered_models[model_key].model_stage = stage
                self.registered_models[model_key].last_updated = datetime.now()
            
            logger.info(f"Transitioned model {model_name} version {version} to {stage}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to transition model stage: {e}")
            return False
    
    def get_experiment_metrics(self, run_id: str) -> Dict[str, Any]:
        """
        Get metrics for an experiment.
        
        Args:
            run_id: Experiment run ID
            
        Returns:
            Experiment metrics
        """
        try:
            # Get run data from MLflow
            run = self.client.get_run(run_id)
            
            metrics = {
                'run_id': run_id,
                'experiment_id': run.info.experiment_id,
                'status': run.info.status,
                'start_time': run.info.start_time,
                'end_time': run.info.end_time,
                'metrics': run.data.metrics,
                'parameters': run.data.params,
                'tags': run.data.tags,
                'artifacts': [artifact.path for artifact in run.data.artifacts]
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get experiment metrics: {e}")
            return {}
    
    def compare_experiments(self, run_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple experiments.
        
        Args:
            run_ids: List of run IDs to compare
            
        Returns:
            Comparison results
        """
        try:
            comparison = {
                'run_ids': run_ids,
                'experiments': [],
                'metrics_comparison': {},
                'parameters_comparison': {},
                'best_experiment': None
            }
            
            # Get experiment data
            for run_id in run_ids:
                experiment = self.get_experiment(run_id)
                if experiment:
                    comparison['experiments'].append(experiment)
            
            # Compare metrics
            if len(comparison['experiments']) > 1:
                # Find common metrics
                common_metrics = set()
                for experiment in comparison['experiments']:
                    common_metrics.update(experiment.metrics.keys())
                
                # Compare each metric
                for metric in common_metrics:
                    values = []
                    for experiment in comparison['experiments']:
                        if metric in experiment.metrics:
                            values.append(experiment.metrics[metric])
                    
                    if values:
                        comparison['metrics_comparison'][metric] = {
                            'values': values,
                            'min': min(values),
                            'max': max(values),
                            'avg': sum(values) / len(values)
                        }
            
            # Find best experiment (highest success rate or lowest error rate)
            best_experiment = None
            best_score = -1
            
            for experiment in comparison['experiments']:
                # Calculate score based on available metrics
                score = 0
                if 'success_rate' in experiment.metrics:
                    score += experiment.metrics['success_rate']
                if 'accuracy' in experiment.metrics:
                    score += experiment.metrics['accuracy']
                if 'error_rate' in experiment.metrics:
                    score -= experiment.metrics['error_rate']
                
                if score > best_score:
                    best_score = score
                    best_experiment = experiment
            
            comparison['best_experiment'] = best_experiment
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare experiments: {e}")
            return {}
    
    def export_experiment_data(self, run_id: str, export_path: str) -> bool:
        """
        Export experiment data to file.
        
        Args:
            run_id: Experiment run ID
            export_path: Path to export file
            
        Returns:
            True if successful
        """
        try:
            # Get experiment data
            experiment = self.get_experiment(run_id)
            if not experiment:
                return False
            
            # Export data
            export_data = {
                'experiment': asdict(experiment),
                'mlflow_metrics': self.get_experiment_metrics(run_id),
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported experiment data for {run_id} to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export experiment data: {e}")
            return False
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Get summary of all experiments.
        
        Returns:
            Experiment summary
        """
        try:
            total_experiments = len(self.active_experiments) + len(self.completed_experiments)
            active_experiments = len(self.active_experiments)
            completed_experiments = len(self.completed_experiments)
            
            # Count by status
            status_counts = {}
            for experiment in self.completed_experiments.values():
                status = experiment.status
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Count by workflow
            workflow_counts = {}
            for experiment in self.completed_experiments.values():
                workflow = experiment.workflow_name
                workflow_counts[workflow] = workflow_counts.get(workflow, 0) + 1
            
            # Count by agent
            agent_counts = {}
            for experiment in self.completed_experiments.values():
                agent = experiment.agent_name
                agent_counts[agent] = agent_counts.get(agent, 0) + 1
            
            summary = {
                'total_experiments': total_experiments,
                'active_experiments': active_experiments,
                'completed_experiments': completed_experiments,
                'status_breakdown': status_counts,
                'workflow_breakdown': workflow_counts,
                'agent_breakdown': agent_counts,
                'registered_models': len(self.registered_models),
                'tracking_uri': mlflow.get_tracking_uri(),
                'registry_uri': mlflow.get_registry_uri()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get experiment summary: {e}")
            return {}


# Factory functions for common MLflow tracking scenarios
def create_workflow_tracker() -> MLflowAgentTracker:
    """Create a tracker for workflow experiments."""
    return MLflowAgentTracker()


def create_model_registry_tracker() -> MLflowAgentTracker:
    """Create a tracker focused on model registry."""
    return MLflowAgentTracker()


def create_mobile_tracker() -> MLflowAgentTracker:
    """Create a tracker optimized for mobile experiments."""
    return MLflowAgentTracker()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create MLflow tracker
    tracker = create_workflow_tracker()
    
    # Start an experiment
    run_id = tracker.start_experiment(
        experiment_name="lenovo_device_support",
        workflow_name="device_support_workflow",
        agent_name="device_analyzer",
        model_name="phi-4-mini",
        parameters={
            "model_size": "3.8B",
            "optimization_level": "medium",
            "mobile_optimized": True
        },
        tags={"platform": "mobile", "use_case": "device_support"}
    )
    
    # Log some metrics
    tracker.log_metrics(run_id, {
        "accuracy": 0.95,
        "latency_ms": 150,
        "memory_usage_mb": 256,
        "success_rate": 0.98
    })
    
    # Log artifacts
    tracker.log_artifacts(run_id, ["model_weights.pth", "config.json"])
    
    # End experiment
    tracker.end_experiment(run_id, "FINISHED")
    
    # Register model
    model_version = tracker.register_model(
        model_name="lenovo_device_support_model",
        model_path="models/device_support_model.pth",
        description="Lenovo device support model optimized for mobile"
    )
    
    # Get experiment summary
    summary = tracker.get_experiment_summary()
    print(f"Experiment summary: {summary}")
    
    # Get experiment metrics
    metrics = tracker.get_experiment_metrics(run_id)
    print(f"Experiment metrics: {metrics}")
