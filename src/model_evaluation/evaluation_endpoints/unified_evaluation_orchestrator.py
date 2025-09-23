"""
Unified Evaluation Orchestrator for All Model Types

This module provides unified evaluation orchestration for all model types
including foundation models, custom models, agentic workflows, and retrieval systems.

Key Features:
- Unified evaluation orchestration
- Cross-model comparison
- Comprehensive evaluation reporting
- Production readiness assessment
- MLflow integration
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid

# Import evaluation components
from ..enhanced_pipeline.raw_foundation_evaluation import RawFoundationEvaluator
from ..enhanced_pipeline.custom_model_evaluation import CustomModelEvaluator
from ..enhanced_pipeline.mobile_model_evaluation import MobileModelEvaluator
from ..enhanced_pipeline.agentic_workflow_evaluation import AgenticWorkflowEvaluator
from ..enhanced_pipeline.retrieval_workflow_evaluation import RetrievalWorkflowEvaluator
from ..enhanced_pipeline.factory_roster_management import FactoryRosterManager
from ..mlflow_integration.experiment_tracking import UnifiedExperimentTracker

logger = logging.getLogger(__name__)


@dataclass
class UnifiedEvaluationRequest:
    """Unified evaluation request configuration."""
    request_id: str
    evaluation_type: str  # "comprehensive", "foundation", "custom", "mobile", "agentic", "retrieval"
    model_configs: List[Dict[str, Any]]
    evaluation_parameters: Dict[str, Any]
    comparison_requirements: Dict[str, Any]
    production_requirements: Dict[str, Any]
    created_at: datetime
    status: str = "PENDING"  # PENDING, RUNNING, COMPLETED, FAILED


@dataclass
class UnifiedEvaluationResult:
    """Unified evaluation result."""
    request_id: str
    evaluation_type: str
    success: bool
    model_results: Dict[str, Dict[str, Any]]
    comparison_analysis: Dict[str, Any]
    production_readiness: Dict[str, Any]
    recommendations: List[str]
    overall_score: float
    error_message: Optional[str] = None
    completed_at: Optional[datetime] = None


class UnifiedEvaluationOrchestrator:
    """
    Unified Evaluation Orchestrator for All Model Types
    
    This class provides unified evaluation orchestration for all model types
    including foundation models, custom models, agentic workflows, and retrieval systems.
    """
    
    def __init__(self, 
                 mlflow_tracking_uri: str = "http://localhost:5000",
                 github_models_token: Optional[str] = None):
        """
        Initialize the Unified Evaluation Orchestrator.
        
        Args:
            mlflow_tracking_uri: MLflow tracking server URI
            github_models_token: GitHub Models API token
        """
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.github_models_token = github_models_token
        
        # Initialize evaluation components
        self.foundation_evaluator = RawFoundationEvaluator(
            mlflow_tracking_uri=mlflow_tracking_uri,
            github_models_token=github_models_token
        )
        self.custom_evaluator = CustomModelEvaluator(
            mlflow_tracking_uri=mlflow_tracking_uri
        )
        self.mobile_evaluator = MobileModelEvaluator(
            mlflow_tracking_uri=mlflow_tracking_uri
        )
        self.agentic_evaluator = AgenticWorkflowEvaluator(
            mlflow_tracking_uri=mlflow_tracking_uri
        )
        self.retrieval_evaluator = RetrievalWorkflowEvaluator(
            mlflow_tracking_uri=mlflow_tracking_uri
        )
        self.factory_manager = FactoryRosterManager(
            mlflow_tracking_uri=mlflow_tracking_uri
        )
        self.experiment_tracker = UnifiedExperimentTracker(
            tracking_uri=mlflow_tracking_uri
        )
        
        # Request tracking
        self.active_requests: Dict[str, UnifiedEvaluationRequest] = {}
        self.completed_requests: Dict[str, UnifiedEvaluationResult] = {}
        
        logger.info("Unified Evaluation Orchestrator initialized")
    
    def create_comprehensive_evaluation(self,
                                      model_configs: List[Dict[str, Any]],
                                      evaluation_parameters: Optional[Dict[str, Any]] = None,
                                      comparison_requirements: Optional[Dict[str, Any]] = None,
                                      production_requirements: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a comprehensive evaluation request.
        
        Args:
            model_configs: List of model configurations
            evaluation_parameters: Evaluation parameters
            comparison_requirements: Comparison requirements
            production_requirements: Production requirements
            
        Returns:
            Request ID
        """
        request_id = str(uuid.uuid4())
        
        if evaluation_parameters is None:
            evaluation_parameters = {
                "test_duration": 300,  # 5 minutes
                "concurrent_tests": 5,
                "performance_thresholds": {
                    "latency_p95": 5.0,
                    "error_rate": 0.05,
                    "throughput": 5.0
                }
            }
        
        if comparison_requirements is None:
            comparison_requirements = {
                "enable_comparison": True,
                "comparison_metrics": ["accuracy", "latency", "throughput", "resource_usage"],
                "statistical_significance": 0.95
            }
        
        if production_requirements is None:
            production_requirements = {
                "production_readiness_threshold": 0.8,
                "deployment_requirements": {
                    "min_accuracy": 0.8,
                    "max_latency": 3.0,
                    "min_throughput": 5.0
                }
            }
        
        request = UnifiedEvaluationRequest(
            request_id=request_id,
            evaluation_type="comprehensive",
            model_configs=model_configs,
            evaluation_parameters=evaluation_parameters,
            comparison_requirements=comparison_requirements,
            production_requirements=production_requirements,
            created_at=datetime.now()
        )
        
        self.active_requests[request_id] = request
        logger.info(f"Created comprehensive evaluation request {request_id}")
        
        return request_id
    
    def create_foundation_evaluation(self,
                                   model_configs: List[Dict[str, Any]],
                                   evaluation_parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a foundation model evaluation request.
        
        Args:
            model_configs: List of model configurations
            evaluation_parameters: Evaluation parameters
            
        Returns:
            Request ID
        """
        request_id = str(uuid.uuid4())
        
        if evaluation_parameters is None:
            evaluation_parameters = {
                "test_duration": 180,  # 3 minutes
                "performance_thresholds": {
                    "latency_p95": 5.0,
                    "error_rate": 0.05,
                    "throughput": 5.0
                }
            }
        
        request = UnifiedEvaluationRequest(
            request_id=request_id,
            evaluation_type="foundation",
            model_configs=model_configs,
            evaluation_parameters=evaluation_parameters,
            comparison_requirements={"enable_comparison": False},
            production_requirements={"production_readiness_threshold": 0.8},
            created_at=datetime.now()
        )
        
        self.active_requests[request_id] = request
        logger.info(f"Created foundation evaluation request {request_id}")
        
        return request_id
    
    async def run_evaluation(self, request_id: str) -> UnifiedEvaluationResult:
        """
        Run a unified evaluation.
        
        Args:
            request_id: Request ID to run
            
        Returns:
            Evaluation result
        """
        if request_id not in self.active_requests:
            raise ValueError(f"Request {request_id} not found")
        
        request = self.active_requests[request_id]
        request.status = "RUNNING"
        
        logger.info(f"Starting unified evaluation {request_id}")
        
        # Start unified experiment
        experiment_id = self.experiment_tracker.start_unified_experiment(
            experiment_name=f"unified_evaluation_{request_id}",
            experiment_type=request.evaluation_type,
            components=["foundation", "custom", "mobile", "agentic", "retrieval"],
            parameters=request.evaluation_parameters,
            tags={
                "request_id": request_id,
                "evaluation_type": request.evaluation_type
            }
        )
        
        try:
            # Run evaluation based on type
            if request.evaluation_type == "comprehensive":
                result = await self._run_comprehensive_evaluation(request, experiment_id)
            elif request.evaluation_type == "foundation":
                result = await self._run_foundation_evaluation(request, experiment_id)
            else:
                result = await self._run_specific_evaluation(request, experiment_id)
            
            # Complete experiment
            self.experiment_tracker.complete_experiment(experiment_id)
            
        except Exception as e:
            logger.error(f"Error in unified evaluation {request_id}: {e}")
            self.experiment_tracker.fail_experiment(experiment_id, str(e))
            
            result = UnifiedEvaluationResult(
                request_id=request_id,
                evaluation_type=request.evaluation_type,
                success=False,
                model_results={},
                comparison_analysis={},
                production_readiness={},
                recommendations=[],
                overall_score=0.0,
                error_message=str(e),
                completed_at=datetime.now()
            )
        
        # Update request status
        request.status = "COMPLETED" if result.success else "FAILED"
        self.completed_requests[request_id] = result
        del self.active_requests[request_id]
        
        logger.info(f"Completed unified evaluation {request_id}, success: {result.success}")
        
        return result
    
    async def _run_comprehensive_evaluation(self, request: UnifiedEvaluationRequest, experiment_id: str) -> UnifiedEvaluationResult:
        """Run comprehensive evaluation."""
        model_results = {}
        all_metrics = {}
        
        # Evaluate each model configuration
        for i, model_config in enumerate(request.model_configs):
            model_name = model_config.get("name", f"model_{i}")
            model_type = model_config.get("type", "foundation")
            
            logger.info(f"Evaluating model {model_name} of type {model_type}")
            
            # Run appropriate evaluation based on model type
            if model_type == "foundation":
                result = await self._evaluate_foundation_model(model_config, experiment_id)
            elif model_type == "custom":
                result = await self._evaluate_custom_model(model_config, experiment_id)
            elif model_type == "mobile":
                result = await self._evaluate_mobile_model(model_config, experiment_id)
            elif model_type == "agentic":
                result = await self._evaluate_agentic_workflow(model_config, experiment_id)
            elif model_type == "retrieval":
                result = await self._evaluate_retrieval_workflow(model_config, experiment_id)
            else:
                logger.warning(f"Unknown model type: {model_type}")
                continue
            
            model_results[model_name] = result
            all_metrics[model_name] = result.get("metrics", {})
        
        # Perform comparison analysis
        comparison_analysis = self._perform_comparison_analysis(model_results, request.comparison_requirements)
        
        # Assess production readiness
        production_readiness = self._assess_production_readiness(model_results, request.production_requirements)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(model_results, comparison_analysis, production_readiness)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(model_results, comparison_analysis, production_readiness)
        
        return UnifiedEvaluationResult(
            request_id=request.request_id,
            evaluation_type=request.evaluation_type,
            success=True,
            model_results=model_results,
            comparison_analysis=comparison_analysis,
            production_readiness=production_readiness,
            recommendations=recommendations,
            overall_score=overall_score,
            completed_at=datetime.now()
        )
    
    async def _run_foundation_evaluation(self, request: UnifiedEvaluationRequest, experiment_id: str) -> UnifiedEvaluationResult:
        """Run foundation model evaluation."""
        model_results = {}
        
        for i, model_config in enumerate(request.model_configs):
            model_name = model_config.get("name", f"foundation_model_{i}")
            result = await self._evaluate_foundation_model(model_config, experiment_id)
            model_results[model_name] = result
        
        # Foundation-specific analysis
        comparison_analysis = self._perform_foundation_comparison(model_results)
        production_readiness = self._assess_foundation_production_readiness(model_results)
        recommendations = self._generate_foundation_recommendations(model_results)
        overall_score = self._calculate_foundation_score(model_results)
        
        return UnifiedEvaluationResult(
            request_id=request.request_id,
            evaluation_type=request.evaluation_type,
            success=True,
            model_results=model_results,
            comparison_analysis=comparison_analysis,
            production_readiness=production_readiness,
            recommendations=recommendations,
            overall_score=overall_score,
            completed_at=datetime.now()
        )
    
    async def _run_specific_evaluation(self, request: UnifiedEvaluationRequest, experiment_id: str) -> UnifiedEvaluationResult:
        """Run specific evaluation based on type."""
        # Implementation for specific evaluation types
        # This would be similar to comprehensive but focused on specific model types
        return await self._run_comprehensive_evaluation(request, experiment_id)
    
    async def _evaluate_foundation_model(self, model_config: Dict[str, Any], experiment_id: str) -> Dict[str, Any]:
        """Evaluate a foundation model."""
        model_name = model_config.get("name", "foundation_model")
        
        # Create performance test
        test_id = self.foundation_evaluator.create_performance_test(
            model_name=model_name,
            test_prompts=model_config.get("test_prompts", ["Test prompt"]),
            parameters=model_config.get("parameters", {})
        )
        
        # Run test
        result = await self.foundation_evaluator.run_test(test_id)
        
        # Log metrics to experiment tracker
        self.experiment_tracker.log_component_metrics(
            experiment_id=experiment_id,
            component="foundation",
            metrics=result.metrics
        )
        
        return {
            "model_name": model_name,
            "model_type": "foundation",
            "success": result.success,
            "metrics": result.metrics,
            "performance_metrics": {
                "inference_time": result.inference_time_ms,
                "throughput": result.throughput,
                "latency_p95": result.latency_p95,
                "error_rate": result.error_rate
            }
        }
    
    async def _evaluate_custom_model(self, model_config: Dict[str, Any], experiment_id: str) -> Dict[str, Any]:
        """Evaluate a custom model."""
        model_name = model_config.get("name", "custom_model")
        model_type = model_config.get("model_type", "fine_tuned")
        
        # Create custom model test
        if model_type == "fine_tuned":
            test_id = self.custom_evaluator.create_fine_tuned_model_test(
                model_name=model_name,
                base_model=model_config.get("base_model", "base_model"),
                domain=model_config.get("domain", "general"),
                test_prompts=model_config.get("test_prompts", ["Test prompt"])
            )
        elif model_type == "qlora":
            test_id = self.custom_evaluator.create_qlora_adapter_test(
                adapter_name=model_name,
                base_model=model_config.get("base_model", "base_model"),
                domain=model_config.get("domain", "general"),
                test_prompts=model_config.get("test_prompts", ["Test prompt"])
            )
        else:
            test_id = self.custom_evaluator.create_custom_embedding_test(
                embedding_name=model_name,
                base_model=model_config.get("base_model", "base_model"),
                domain=model_config.get("domain", "general"),
                test_queries=model_config.get("test_queries", ["Test query"])
            )
        
        # Run test
        result = await self.custom_evaluator.run_test(test_id)
        
        # Log metrics to experiment tracker
        self.experiment_tracker.log_component_metrics(
            experiment_id=experiment_id,
            component="custom",
            metrics=result.performance_metrics
        )
        
        return {
            "model_name": model_name,
            "model_type": model_type,
            "success": result.success,
            "metrics": result.performance_metrics,
            "quality_scores": result.quality_scores,
            "production_readiness": result.production_readiness
        }
    
    async def _evaluate_mobile_model(self, model_config: Dict[str, Any], experiment_id: str) -> Dict[str, Any]:
        """Evaluate a mobile model."""
        model_name = model_config.get("name", "mobile_model")
        platform = model_config.get("platform", "android")
        
        # Create mobile performance test
        test_id = self.mobile_evaluator.create_mobile_performance_test(
            model_name=model_name,
            platform=platform,
            optimization_level=model_config.get("optimization_level", "medium")
        )
        
        # Run test
        result = await self.mobile_evaluator.run_test(test_id)
        
        # Log metrics to experiment tracker
        self.experiment_tracker.log_component_metrics(
            experiment_id=experiment_id,
            component="mobile",
            metrics=result.performance_metrics
        )
        
        return {
            "model_name": model_name,
            "model_type": "mobile",
            "platform": platform,
            "success": result.success,
            "metrics": result.performance_metrics,
            "deployment_readiness": result.deployment_readiness
        }
    
    async def _evaluate_agentic_workflow(self, model_config: Dict[str, Any], experiment_id: str) -> Dict[str, Any]:
        """Evaluate an agentic workflow."""
        workflow_name = model_config.get("name", "agentic_workflow")
        workflow_type = model_config.get("workflow_type", "smolagent")
        
        # Create agentic workflow test
        if workflow_type == "smolagent":
            test_id = self.agentic_evaluator.create_smolagent_workflow_test(
                workflow_name=workflow_name,
                agents=model_config.get("agents", []),
                tasks=model_config.get("tasks", [])
            )
        else:
            test_id = self.agentic_evaluator.create_langgraph_workflow_test(
                workflow_name=workflow_name,
                agents=model_config.get("agents", []),
                tasks=model_config.get("tasks", [])
            )
        
        # Run test
        result = await self.agentic_evaluator.run_test(test_id)
        
        # Log metrics to experiment tracker
        self.experiment_tracker.log_component_metrics(
            experiment_id=experiment_id,
            component="agentic",
            metrics=result.workflow_metrics
        )
        
        return {
            "model_name": workflow_name,
            "model_type": "agentic",
            "workflow_type": workflow_type,
            "success": result.success,
            "metrics": result.workflow_metrics,
            "production_readiness": result.production_readiness
        }
    
    async def _evaluate_retrieval_workflow(self, model_config: Dict[str, Any], experiment_id: str) -> Dict[str, Any]:
        """Evaluate a retrieval workflow."""
        workflow_name = model_config.get("name", "retrieval_workflow")
        retrieval_type = model_config.get("retrieval_type", "hybrid")
        
        # Create retrieval workflow test
        if retrieval_type == "langchain":
            test_id = self.retrieval_evaluator.create_langchain_retrieval_test(
                workflow_name=workflow_name,
                data_sources=model_config.get("data_sources", []),
                test_queries=model_config.get("test_queries", [])
            )
        elif retrieval_type == "llamaindex":
            test_id = self.retrieval_evaluator.create_llamaindex_retrieval_test(
                workflow_name=workflow_name,
                data_sources=model_config.get("data_sources", []),
                test_queries=model_config.get("test_queries", [])
            )
        else:
            test_id = self.retrieval_evaluator.create_hybrid_retrieval_test(
                workflow_name=workflow_name,
                data_sources=model_config.get("data_sources", []),
                test_queries=model_config.get("test_queries", [])
            )
        
        # Run test
        result = await self.retrieval_evaluator.run_test(test_id)
        
        # Log metrics to experiment tracker
        self.experiment_tracker.log_component_metrics(
            experiment_id=experiment_id,
            component="retrieval",
            metrics=result.retrieval_metrics
        )
        
        return {
            "model_name": workflow_name,
            "model_type": "retrieval",
            "retrieval_type": retrieval_type,
            "success": result.success,
            "metrics": result.retrieval_metrics,
            "production_readiness": result.production_readiness
        }
    
    def _perform_comparison_analysis(self, model_results: Dict[str, Dict[str, Any]], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comparison analysis across models."""
        if not requirements.get("enable_comparison", False):
            return {}
        
        comparison_metrics = requirements.get("comparison_metrics", ["accuracy", "latency", "throughput"])
        analysis = {
            "comparison_metrics": comparison_metrics,
            "model_rankings": {},
            "statistical_analysis": {},
            "recommendations": []
        }
        
        # Rank models by each metric
        for metric in comparison_metrics:
            rankings = []
            for model_name, result in model_results.items():
                value = result.get("metrics", {}).get(metric, 0)
                rankings.append((model_name, value))
            
            # Sort by metric value (higher is better for accuracy, lower is better for latency)
            if metric in ["latency", "error_rate"]:
                rankings.sort(key=lambda x: x[1])  # Ascending
            else:
                rankings.sort(key=lambda x: x[1], reverse=True)  # Descending
            
            analysis["model_rankings"][metric] = rankings
        
        # Statistical analysis
        for metric in comparison_metrics:
            values = [result.get("metrics", {}).get(metric, 0) for result in model_results.values()]
            if values:
                analysis["statistical_analysis"][metric] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "std": (sum((x - sum(values) / len(values))**2 for x in values) / len(values))**0.5
                }
        
        return analysis
    
    def _assess_production_readiness(self, model_results: Dict[str, Dict[str, Any]], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Assess production readiness across models."""
        threshold = requirements.get("production_readiness_threshold", 0.8)
        deployment_reqs = requirements.get("deployment_requirements", {})
        
        readiness = {
            "overall_readiness": 0.0,
            "model_readiness": {},
            "deployment_ready_models": [],
            "recommendations": []
        }
        
        ready_count = 0
        total_models = len(model_results)
        
        for model_name, result in model_results.items():
            model_readiness = result.get("production_readiness", {})
            overall_score = model_readiness.get("overall_score", 0.0)
            
            readiness["model_readiness"][model_name] = {
                "score": overall_score,
                "ready": overall_score >= threshold,
                "details": model_readiness
            }
            
            if overall_score >= threshold:
                ready_count += 1
                readiness["deployment_ready_models"].append(model_name)
        
        readiness["overall_readiness"] = ready_count / total_models if total_models > 0 else 0
        
        # Generate recommendations
        if readiness["overall_readiness"] < 0.5:
            readiness["recommendations"].append("Most models are not production ready. Focus on improving performance and quality.")
        elif readiness["overall_readiness"] < 0.8:
            readiness["recommendations"].append("Some models are production ready. Consider optimizing underperforming models.")
        else:
            readiness["recommendations"].append("Most models are production ready. Consider deploying the best performing models.")
        
        return readiness
    
    def _generate_recommendations(self, model_results: Dict[str, Dict[str, Any]], 
                                comparison_analysis: Dict[str, Any], 
                                production_readiness: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        # Performance-based recommendations
        if comparison_analysis:
            for metric, rankings in comparison_analysis.get("model_rankings", {}).items():
                if rankings:
                    best_model = rankings[0][0]
                    recommendations.append(f"Best performing model for {metric}: {best_model}")
        
        # Production readiness recommendations
        if production_readiness.get("overall_readiness", 0) < 0.8:
            recommendations.append("Improve production readiness by optimizing underperforming models")
        
        # Model-specific recommendations
        for model_name, result in model_results.items():
            if not result.get("success", False):
                recommendations.append(f"Fix issues with {model_name} before deployment")
        
        return recommendations
    
    def _calculate_overall_score(self, model_results: Dict[str, Dict[str, Any]], 
                               comparison_analysis: Dict[str, Any], 
                               production_readiness: Dict[str, Any]) -> float:
        """Calculate overall evaluation score."""
        if not model_results:
            return 0.0
        
        # Calculate average success rate
        success_count = sum(1 for result in model_results.values() if result.get("success", False))
        success_rate = success_count / len(model_results)
        
        # Calculate average production readiness
        readiness_scores = [
            result.get("production_readiness", {}).get("overall_score", 0.0)
            for result in model_results.values()
        ]
        avg_readiness = sum(readiness_scores) / len(readiness_scores) if readiness_scores else 0.0
        
        # Calculate overall score
        overall_score = (success_rate * 0.6) + (avg_readiness * 0.4)
        
        return min(1.0, max(0.0, overall_score))
    
    def _perform_foundation_comparison(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Perform foundation-specific comparison."""
        return self._perform_comparison_analysis(model_results, {"enable_comparison": True})
    
    def _assess_foundation_production_readiness(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Assess foundation-specific production readiness."""
        return self._assess_production_readiness(model_results, {"production_readiness_threshold": 0.8})
    
    def _generate_foundation_recommendations(self, model_results: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate foundation-specific recommendations."""
        recommendations = []
        
        for model_name, result in model_results.items():
            if result.get("success", False):
                recommendations.append(f"Foundation model {model_name} is ready for production")
            else:
                recommendations.append(f"Foundation model {model_name} needs optimization")
        
        return recommendations
    
    def _calculate_foundation_score(self, model_results: Dict[str, Dict[str, Any]]) -> float:
        """Calculate foundation-specific score."""
        return self._calculate_overall_score(model_results, {}, {})
    
    def get_evaluation_status(self, request_id: str) -> Dict[str, Any]:
        """Get evaluation status."""
        if request_id in self.active_requests:
            request = self.active_requests[request_id]
            return {
                "request_id": request_id,
                "status": request.status,
                "evaluation_type": request.evaluation_type,
                "created_at": request.created_at.isoformat()
            }
        elif request_id in self.completed_requests:
            result = self.completed_requests[request_id]
            return {
                "request_id": request_id,
                "status": "COMPLETED",
                "evaluation_type": result.evaluation_type,
                "success": result.success,
                "overall_score": result.overall_score,
                "completed_at": result.completed_at.isoformat()
            }
        else:
            return {"error": "Request not found"}
    
    def list_evaluations(self) -> List[Dict[str, Any]]:
        """List all evaluations."""
        evaluations = []
        
        # Active requests
        for request_id, request in self.active_requests.items():
            evaluations.append({
                "request_id": request_id,
                "status": request.status,
                "evaluation_type": request.evaluation_type,
                "created_at": request.created_at.isoformat()
            })
        
        # Completed requests
        for request_id, result in self.completed_requests.items():
            evaluations.append({
                "request_id": request_id,
                "status": "COMPLETED",
                "evaluation_type": result.evaluation_type,
                "success": result.success,
                "overall_score": result.overall_score,
                "completed_at": result.completed_at.isoformat()
            })
        
        return evaluations


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Initialize orchestrator
        orchestrator = UnifiedEvaluationOrchestrator()
        
        # Create comprehensive evaluation
        model_configs = [
            {
                "name": "phi-4-mini",
                "type": "foundation",
                "test_prompts": ["What is AI?", "Explain machine learning."]
            },
            {
                "name": "lenovo-device-support",
                "type": "custom",
                "model_type": "fine_tuned",
                "test_prompts": ["How do I fix display issues?"]
            }
        ]
        
        request_id = orchestrator.create_comprehensive_evaluation(
            model_configs=model_configs
        )
        
        print(f"Created evaluation request: {request_id}")
        
        # Run evaluation
        result = await orchestrator.run_evaluation(request_id)
        print(f"Evaluation completed: {result.success}")
        print(f"Overall score: {result.overall_score}")
        print(f"Recommendations: {result.recommendations}")
    
    asyncio.run(main())
