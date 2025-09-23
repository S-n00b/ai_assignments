"""
MLflow Agent Experiment Tracking

This module provides MLflow integration for tracking agentic workflow
experiments including SmolAgent and LangGraph workflows.
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
except ImportError:
    mlflow = None
    MlflowClient = None
    Run = None
    Experiment = None

logger = logging.getLogger(__name__)


@dataclass
class AgentExperiment:
    """Agent experiment configuration."""
    experiment_id: str
    workflow_name: str
    agent_name: str
    model_name: str
    experiment_type: str  # smolagent, langgraph
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    tags: Dict[str, str]
    run_id: Optional[str] = None
    status: str = "RUNNING"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class MLflowAgentExperimentTracking:
    """
    MLflow Agent Experiment Tracking for agentic workflows.
    
    This class provides comprehensive MLflow integration for tracking
    agent experiments including SmolAgent and LangGraph workflows.
    """
    
    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize the MLflow Agent Experiment Tracking.
        
        Args:
            tracking_uri: MLflow tracking URI
        """
        if mlflow is None:
            raise ImportError("MLflow is not installed. Please install it with: pip install mlflow")
        
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Initialize MLflow client
        self.client = MlflowClient()
        
        # Agent experiments tracking
        self.active_experiments: Dict[str, AgentExperiment] = {}
        self.completed_experiments: Dict[str, AgentExperiment] = {}
        
        # Create default experiment if it doesn't exist
        self._ensure_default_experiment()
        
        logger.info("MLflow Agent Experiment Tracking initialized")
    
    def _ensure_default_experiment(self):
        """Ensure default experiment exists."""
        try:
            experiment_name = "agentic_workflows"
            experiment = self.client.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = self.client.create_experiment(experiment_name)
                logger.info(f"Created default experiment: {experiment_name}")
            else:
                logger.info(f"Using existing experiment: {experiment_name}")
        except Exception as e:
            logger.warning(f"Could not ensure default experiment: {e}")
    
    def start_experiment(self, workflow_name: str, 
                        agent_name: str,
                        model_name: str,
                        experiment_type: str,
                        parameters: Dict[str, Any],
                        tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start a new agent experiment.
        
        Args:
            workflow_name: Name of the workflow
            agent_name: Name of the agent
            model_name: Name of the model
            experiment_type: Type of experiment (smolagent, langgraph)
            parameters: Experiment parameters
            tags: Optional tags
            
        Returns:
            Experiment ID
        """
        try:
            experiment_id = str(uuid.uuid4())
            
            # Start MLflow run
            with mlflow.start_run() as run:
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
                mlflow.set_tag("experiment_type", experiment_type)
                mlflow.set_tag("experiment_id", experiment_id)
                
                # Create agent experiment
                agent_experiment = AgentExperiment(
                    experiment_id=experiment_id,
                    workflow_name=workflow_name,
                    agent_name=agent_name,
                    model_name=model_name,
                    experiment_type=experiment_type,
                    parameters=parameters,
                    metrics={},
                    tags=tags or {},
                    run_id=run_id,
                    status="RUNNING",
                    start_time=datetime.now()
                )
                
                # Store active experiment
                self.active_experiments[experiment_id] = agent_experiment
                
                logger.info(f"Started experiment: {workflow_name}.{agent_name} (ID: {experiment_id})")
                return experiment_id
                
        except Exception as e:
            logger.error(f"Failed to start experiment: {e}")
            raise
    
    def log_metrics(self, experiment_id: str, metrics: Dict[str, float]):
        """
        Log metrics for an experiment.
        
        Args:
            experiment_id: Experiment ID
            metrics: Metrics to log
        """
        try:
            if experiment_id not in self.active_experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            # Log metrics to MLflow
            mlflow.log_metrics(metrics)
            
            # Update agent experiment
            agent_experiment = self.active_experiments[experiment_id]
            agent_experiment.metrics.update(metrics)
            
            logger.debug(f"Logged metrics for experiment {experiment_id}: {metrics}")
            
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
            raise
    
    def log_artifacts(self, experiment_id: str, artifacts: List[str]):
        """
        Log artifacts for an experiment.
        
        Args:
            experiment_id: Experiment ID
            artifacts: List of artifact paths
        """
        try:
            if experiment_id not in self.active_experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            # Log artifacts to MLflow
            for artifact_path in artifacts:
                mlflow.log_artifact(artifact_path)
            
            logger.debug(f"Logged artifacts for experiment {experiment_id}: {artifacts}")
            
        except Exception as e:
            logger.error(f"Failed to log artifacts: {e}")
            raise
    
    def end_experiment(self, experiment_id: str, status: str = "COMPLETED"):
        """
        End an experiment.
        
        Args:
            experiment_id: Experiment ID
            status: Final status
        """
        try:
            if experiment_id not in self.active_experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            # Update agent experiment
            agent_experiment = self.active_experiments[experiment_id]
            agent_experiment.status = status
            agent_experiment.end_time = datetime.now()
            
            # Move to completed experiments
            self.completed_experiments[experiment_id] = agent_experiment
            del self.active_experiments[experiment_id]
            
            # End MLflow run
            mlflow.end_run()
            
            logger.info(f"Ended experiment {experiment_id} with status: {status}")
            
        except Exception as e:
            logger.error(f"Failed to end experiment: {e}")
            raise
    
    def get_experiment(self, experiment_id: str) -> Optional[AgentExperiment]:
        """Get experiment by ID."""
        if experiment_id in self.active_experiments:
            return self.active_experiments[experiment_id]
        elif experiment_id in self.completed_experiments:
            return self.completed_experiments[experiment_id]
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
    
    def get_experiment_metrics(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get metrics for an experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment metrics
        """
        try:
            experiment = self.get_experiment(experiment_id)
            if not experiment:
                return {"error": "Experiment not found"}
            
            if not experiment.run_id:
                return {"error": "No run ID found for experiment"}
            
            # Get run data from MLflow
            run = self.client.get_run(experiment.run_id)
            
            metrics = {
                'experiment_id': experiment_id,
                'workflow_name': experiment.workflow_name,
                'agent_name': experiment.agent_name,
                'model_name': experiment.model_name,
                'experiment_type': experiment.experiment_type,
                'status': experiment.status,
                'start_time': experiment.start_time.isoformat() if experiment.start_time else None,
                'end_time': experiment.end_time.isoformat() if experiment.end_time else None,
                'run_id': experiment.run_id,
                'experiment_id_mlflow': run.info.experiment_id,
                'metrics': run.data.metrics,
                'parameters': run.data.params,
                'tags': run.data.tags,
                'artifacts': [artifact.path for artifact in run.data.artifacts]
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get experiment metrics: {e}")
            return {"error": str(e)}
    
    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            
        Returns:
            Comparison results
        """
        try:
            comparison = {
                'experiment_ids': experiment_ids,
                'experiments': [],
                'metrics_comparison': {},
                'parameters_comparison': {},
                'best_experiment': None
            }
            
            # Get experiment data
            for experiment_id in experiment_ids:
                experiment = self.get_experiment(experiment_id)
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
            return {"error": str(e)}
    
    def export_experiment_data(self, experiment_id: str, export_path: str) -> bool:
        """
        Export experiment data to file.
        
        Args:
            experiment_id: Experiment ID
            export_path: Path to export file
            
        Returns:
            True if successful
        """
        try:
            experiment = self.get_experiment(experiment_id)
            if not experiment:
                return False
            
            # Get MLflow metrics
            mlflow_metrics = self.get_experiment_metrics(experiment_id)
            
            # Export data
            export_data = {
                'experiment': asdict(experiment),
                'mlflow_metrics': mlflow_metrics,
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported experiment data for {experiment_id} to {export_path}")
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
            
            # Count by experiment type
            type_counts = {}
            for experiment in self.completed_experiments.values():
                exp_type = experiment.experiment_type
                type_counts[exp_type] = type_counts.get(exp_type, 0) + 1
            
            summary = {
                'total_experiments': total_experiments,
                'active_experiments': active_experiments,
                'completed_experiments': completed_experiments,
                'status_breakdown': status_counts,
                'workflow_breakdown': workflow_counts,
                'agent_breakdown': agent_counts,
                'type_breakdown': type_counts,
                'tracking_uri': mlflow.get_tracking_uri()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get experiment summary: {e}")
            return {"error": str(e)}
    
    def get_workflow_experiments(self, workflow_name: str) -> List[AgentExperiment]:
        """
        Get experiments for a specific workflow.
        
        Args:
            workflow_name: Name of the workflow
            
        Returns:
            List of experiments for the workflow
        """
        try:
            experiments = []
            
            # Check active experiments
            for experiment in self.active_experiments.values():
                if experiment.workflow_name == workflow_name:
                    experiments.append(experiment)
            
            # Check completed experiments
            for experiment in self.completed_experiments.values():
                if experiment.workflow_name == workflow_name:
                    experiments.append(experiment)
            
            return experiments
            
        except Exception as e:
            logger.error(f"Failed to get workflow experiments: {e}")
            return []
    
    def get_agent_experiments(self, agent_name: str) -> List[AgentExperiment]:
        """
        Get experiments for a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            List of experiments for the agent
        """
        try:
            experiments = []
            
            # Check active experiments
            for experiment in self.active_experiments.values():
                if experiment.agent_name == agent_name:
                    experiments.append(experiment)
            
            # Check completed experiments
            for experiment in self.completed_experiments.values():
                if experiment.agent_name == agent_name:
                    experiments.append(experiment)
            
            return experiments
            
        except Exception as e:
            logger.error(f"Failed to get agent experiments: {e}")
            return []


# Factory functions for common experiment tracking scenarios
def create_workflow_tracker() -> MLflowAgentExperimentTracking:
    """Create a workflow-focused experiment tracker."""
    return MLflowAgentExperimentTracking()


def create_agent_tracker() -> MLflowAgentExperimentTracking:
    """Create an agent-focused experiment tracker."""
    return MLflowAgentExperimentTracking()


def create_mobile_tracker() -> MLflowAgentExperimentTracking:
    """Create a mobile-optimized experiment tracker."""
    return MLflowAgentExperimentTracking()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create experiment tracker
    tracker = create_workflow_tracker()
    
    # Start an experiment
    experiment_id = tracker.start_experiment(
        workflow_name="lenovo_device_support",
        agent_name="device_analyzer",
        model_name="phi-4-mini",
        experiment_type="smolagent",
        parameters={
            "model_size": "3.8B",
            "optimization_level": "medium",
            "mobile_optimized": True
        },
        tags={"platform": "mobile", "use_case": "device_support"}
    )
    
    # Log some metrics
    tracker.log_metrics(experiment_id, {
        "accuracy": 0.95,
        "latency_ms": 150,
        "memory_usage_mb": 256,
        "success_rate": 0.98
    })
    
    # Log artifacts
    tracker.log_artifacts(experiment_id, ["model_weights.pth", "config.json"])
    
    # End experiment
    tracker.end_experiment(experiment_id, "COMPLETED")
    
    # Get experiment summary
    summary = tracker.get_experiment_summary()
    print(f"Experiment summary: {summary}")
    
    # Get experiment metrics
    metrics = tracker.get_experiment_metrics(experiment_id)
    print(f"Experiment metrics: {metrics}")
    
    # Get workflow experiments
    workflow_experiments = tracker.get_workflow_experiments("lenovo_device_support")
    print(f"Workflow experiments: {len(workflow_experiments)}")
    
    # Get agent experiments
    agent_experiments = tracker.get_agent_experiments("device_analyzer")
    print(f"Agent experiments: {len(agent_experiments)}")
