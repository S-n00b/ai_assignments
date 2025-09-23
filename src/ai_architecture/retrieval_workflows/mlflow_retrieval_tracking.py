"""
MLflow Retrieval Tracking for Hybrid RAG

This module provides MLflow integration for tracking retrieval experiments,
evaluation metrics, and retrieval system performance.
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)

class MLflowRetrievalTracking:
    """
    MLflow tracking for retrieval experiments.
    
    Provides MLflow integration for tracking retrieval experiments,
    evaluation metrics, and retrieval system performance.
    """
    
    def __init__(self, 
                 experiment_name: str = "retrieval_experiments",
                 tracking_uri: str = "http://localhost:5000"):
        """
        Initialize MLflow retrieval tracking.
        
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
    
    def start_retrieval_experiment(self, 
                                 run_name: str,
                                 retrieval_system_name: str,
                                 system_config: Dict[str, Any],
                                 evaluation_config: Dict[str, Any]) -> str:
        """
        Start a new retrieval experiment.
        
        Args:
            run_name: Name of the run
            retrieval_system_name: Name of the retrieval system
            system_config: System configuration
            evaluation_config: Evaluation configuration
            
        Returns:
            Run ID
        """
        try:
            with mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name) as run:
                # Log run parameters
                mlflow.log_params({
                    'retrieval_system_name': retrieval_system_name,
                    'experiment_type': 'retrieval_evaluation',
                    'evaluation_metrics': str(evaluation_config.get('metrics', [])),
                    'k_values': str(evaluation_config.get('k_values', [])),
                    'similarity_threshold': evaluation_config.get('similarity_threshold', 0.5),
                    'timeout_seconds': evaluation_config.get('timeout_seconds', 30)
                })
                
                # Log system configuration
                mlflow.log_params({
                    'system_type': system_config.get('system_type', 'unknown'),
                    'fusion_method': system_config.get('fusion_method', 'weighted'),
                    'components_available': str(system_config.get('components_available', {})),
                    'weights': str(system_config.get('weights', {})),
                    'embedding_model': system_config.get('embedding_model', 'unknown'),
                    'chunk_size': system_config.get('chunk_size', 1000),
                    'chunk_overlap': system_config.get('chunk_overlap', 200)
                })
                
                # Log evaluation configuration
                mlflow.log_params({
                    'test_queries_count': evaluation_config.get('test_queries_count', 0),
                    'ground_truth_count': evaluation_config.get('ground_truth_count', 0),
                    'evaluation_method': evaluation_config.get('evaluation_method', 'standard'),
                    'cross_validation': evaluation_config.get('cross_validation', False)
                })
                
                logger.info(f"Started retrieval experiment: {run_name}")
                return run.info.run_id
                
        except Exception as e:
            logger.error(f"Failed to start retrieval experiment: {e}")
            raise
    
    def log_retrieval_metrics(self, 
                            run_id: str,
                            evaluation_results: Dict[str, Any]) -> None:
        """
        Log retrieval evaluation metrics.
        
        Args:
            run_id: MLflow run ID
            evaluation_results: Evaluation results
        """
        try:
            with mlflow.start_run(run_id=run_id):
                # Log aggregate metrics
                metrics = evaluation_results.get('metrics', {})
                mlflow.log_metrics(metrics)
                
                # Log per-query metrics summary
                per_query_results = evaluation_results.get('per_query_results', [])
                if per_query_results:
                    # Calculate summary statistics
                    execution_times = [r.get('execution_time', 0) for r in per_query_results]
                    if execution_times:
                        mlflow.log_metrics({
                            'avg_query_execution_time': sum(execution_times) / len(execution_times),
                            'max_query_execution_time': max(execution_times),
                            'min_query_execution_time': min(execution_times)
                        })
                    
                    # Log success rate
                    successful_queries = len([r for r in per_query_results if 'error' not in r])
                    mlflow.log_metrics({
                        'successful_queries': successful_queries,
                        'failed_queries': len(per_query_results) - successful_queries,
                        'success_rate': successful_queries / len(per_query_results) if per_query_results else 0
                    })
                
                logger.info(f"Logged retrieval metrics for run {run_id}")
                
        except Exception as e:
            logger.error(f"Failed to log retrieval metrics: {e}")
            raise
    
    def log_system_performance(self, 
                             run_id: str,
                             performance_metrics: Dict[str, Any]) -> None:
        """
        Log system performance metrics.
        
        Args:
            run_id: MLflow run ID
            performance_metrics: Performance metrics
        """
        try:
            with mlflow.start_run(run_id=run_id):
                # Log performance metrics
                mlflow.log_metrics(performance_metrics)
                
                logger.info(f"Logged system performance for run {run_id}")
                
        except Exception as e:
            logger.error(f"Failed to log system performance: {e}")
            raise
    
    def log_retrieval_artifacts(self, 
                              run_id: str,
                              artifacts: Dict[str, str]) -> None:
        """
        Log retrieval artifacts.
        
        Args:
            run_id: MLflow run ID
            artifacts: Dictionary of artifact paths and names
        """
        try:
            with mlflow.start_run(run_id=run_id):
                # Log artifacts
                for artifact_name, artifact_path in artifacts.items():
                    mlflow.log_artifact(artifact_path, artifact_name)
                
                logger.info(f"Logged {len(artifacts)} artifacts for run {run_id}")
                
        except Exception as e:
            logger.error(f"Failed to log retrieval artifacts: {e}")
            raise
    
    def log_comparison_results(self, 
                             run_id: str,
                             comparison_results: Dict[str, Any]) -> None:
        """
        Log comparison results between systems.
        
        Args:
            run_id: MLflow run ID
            comparison_results: Comparison results
        """
        try:
            with mlflow.start_run(run_id=run_id):
                # Log comparison metrics
                comparison_metrics = comparison_results.get('comparison_metrics', {})
                for metric_name, metric_data in comparison_metrics.items():
                    if isinstance(metric_data, dict) and 'best_system' in metric_data:
                        mlflow.log_metrics({
                            f'{metric_name}_best_system': metric_data['best_system'],
                            f'{metric_name}_best_value': metric_data['best_value']
                        })
                
                # Log system comparison summary
                system_results = comparison_results.get('system_results', {})
                mlflow.log_params({
                    'systems_compared': comparison_results.get('systems_compared', 0),
                    'queries_evaluated': comparison_results.get('queries_evaluated', 0)
                })
                
                logger.info(f"Logged comparison results for run {run_id}")
                
        except Exception as e:
            logger.error(f"Failed to log comparison results: {e}")
            raise
    
    def create_retrieval_model_registry_entry(self, 
                                            run_id: str,
                                            model_name: str,
                                            version: str = "1.0.0",
                                            description: str = "") -> str:
        """
        Create model registry entry for retrieval system.
        
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
                    'version': version,
                    'description': description,
                    'retrieval_system': True,
                    'evaluation_completed': True
                }
            )
            
            logger.info(f"Created retrieval model registry entry: {model_name} v{version}")
            return model_version.version
            
        except Exception as e:
            logger.error(f"Failed to create model registry entry: {e}")
            raise
    
    def get_best_retrieval_system(self, 
                                experiment_id: Optional[str] = None,
                                metric_name: str = "avg_f1",
                                ascending: bool = False) -> Dict[str, Any]:
        """
        Get the best retrieval system from experiment.
        
        Args:
            experiment_id: Experiment ID (uses current if None)
            metric_name: Metric to optimize
            ascending: Whether lower values are better
            
        Returns:
            Best retrieval system information
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
                'run_id': best_run.info.run_id,
                'run_name': best_run.data.tags.get("mlflow.runName", best_run.info.run_id),
                'metric_value': best_run.data.metrics.get(metric_name, 0.0),
                'model_uri': f"runs:/{best_run.info.run_id}/model",
                'all_metrics': best_run.data.metrics,
                'system_config': best_run.data.params
            }
            
        except Exception as e:
            logger.error(f"Failed to get best retrieval system: {e}")
            raise
    
    def compare_retrieval_systems(self, 
                                run_ids: List[str],
                                metric_name: str = "avg_f1") -> Dict[str, Any]:
        """
        Compare multiple retrieval systems.
        
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
                    'run_name': run.data.tags.get("mlflow.runName", run_id),
                    'metric_value': metrics.get(metric_name, 0.0),
                    'all_metrics': metrics,
                    'system_config': run.data.params
                }
            
            # Sort by metric value
            sorted_runs = sorted(
                comparison_results.items(),
                key=lambda x: x[1]["metric_value"],
                reverse=True
            )
            
            logger.info(f"Compared {len(run_ids)} retrieval systems by {metric_name}")
            return {
                'comparison_results': comparison_results,
                'sorted_runs': sorted_runs,
                'best_run': sorted_runs[0] if sorted_runs else None
            }
            
        except Exception as e:
            logger.error(f"Failed to compare retrieval systems: {e}")
            raise
    
    def generate_retrieval_experiment_report(self, 
                                           experiment_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive retrieval experiment report.
        
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
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
            
            # Get best run
            best_run = self.get_best_retrieval_system(exp_id)
            
            report = {
                'experiment_id': exp_id,
                'experiment_name': self.experiment_name,
                'run_count': run_count,
                'metrics_summary': metrics_stats,
                'best_run': best_run,
                'generated_at': time.time()
            }
            
            logger.info(f"Generated retrieval experiment report for {run_count} runs")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate experiment report: {e}")
            raise
    
    def export_retrieval_results(self, 
                               run_id: str,
                               output_path: str) -> str:
        """
        Export retrieval results to file.
        
        Args:
            run_id: MLflow run ID
            output_path: Path to save results
            
        Returns:
            Path to saved results
        """
        try:
            # Get run data
            run = self.client.get_run(run_id)
            
            # Create export data
            export_data = {
                'run_id': run_id,
                'run_name': run.data.tags.get("mlflow.runName", run_id),
                'metrics': run.data.metrics,
                'params': run.data.params,
                'tags': run.data.tags,
                'export_timestamp': time.time()
            }
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported retrieval results to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export retrieval results: {e}")
            raise
