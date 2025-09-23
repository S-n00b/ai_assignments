"""
Model Evaluation Engineer Workspace for Enhanced Unified Platform

This module provides the Model Evaluation Engineer workspace functionality including
comprehensive model testing, evaluation frameworks, and factory roster management.

Key Features:
- Raw foundation model testing and evaluation
- Custom model testing and validation
- Agentic workflow testing and evaluation
- Retrieval workflow testing and evaluation
- Factory roster management and deployment
- Real-time performance monitoring and analytics
- MLflow experiment tracking integration
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import yaml
from pathlib import Path
import aiohttp
import requests
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse

# Import Model Evaluation components
from ...model_evaluation.enhanced_pipeline import (
    RawFoundationEvaluation,
    CustomModelEvaluation,
    MobileModelEvaluation,
    AgenticWorkflowEvaluation,
    RetrievalWorkflowEvaluation,
    FactoryRosterManagement
)
from ...model_evaluation.mlflow_integration import (
    ExperimentTracking,
    ModelRegistryIntegration,
    PerformanceMetricsTracking,
    AgentExperimentTracking,
    RetrievalExperimentTracking
)
from ...model_evaluation.evaluation_endpoints import (
    FoundationModelEndpoint,
    CustomModelEndpoint,
    FineTunedModelEndpoint,
    QLoRAAdapterEndpoint,
    AgenticWorkflowEndpoint,
    RetrievalWorkflowEndpoint,
    UnifiedEvaluationOrchestrator
)


@dataclass
class EvaluationRequest:
    """Request for model evaluation."""
    model_name: str
    evaluation_type: str  # "raw", "custom", "agentic", "retrieval"
    test_suite: str
    parameters: Dict[str, Any]
    target_platform: str
    stress_testing: bool = False


@dataclass
class FactoryRosterRequest:
    """Request for factory roster management."""
    model_id: str
    deployment_type: str  # "production", "staging", "testing"
    target_environment: str
    configuration: Dict[str, Any]
    monitoring_enabled: bool = True


@dataclass
class EvaluationResult:
    """Result of model evaluation."""
    success: bool
    model_name: str
    evaluation_type: str
    test_suite: str
    metrics: Dict[str, Any]
    performance_score: float
    recommendations: List[str]
    created_at: datetime
    error_message: Optional[str] = None


@dataclass
class FactoryRosterResult:
    """Result of factory roster operation."""
    success: bool
    model_id: str
    deployment_type: str
    deployment_status: str
    monitoring_url: Optional[str] = None
    created_at: datetime
    error_message: Optional[str] = None


class ModelEvaluationWorkspace:
    """
    Model Evaluation Engineer Workspace for comprehensive model testing.
    
    This class provides comprehensive functionality for Model Evaluation Engineers
    to test models, evaluate performance, and manage factory roster deployments.
    """
    
    def __init__(self, config_path: str = "config/model_evaluation_config.yaml"):
        """Initialize the Model Evaluation workspace."""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Initialize evaluation components
        self.raw_evaluation = RawFoundationEvaluation()
        self.custom_evaluation = CustomModelEvaluation()
        self.mobile_evaluation = MobileModelEvaluation()
        self.agentic_evaluation = AgenticWorkflowEvaluation()
        self.retrieval_evaluation = RetrievalWorkflowEvaluation()
        self.factory_roster = FactoryRosterManagement()
        
        # Initialize MLflow integration
        self.experiment_tracking = ExperimentTracking()
        self.model_registry = ModelRegistryIntegration()
        self.performance_tracking = PerformanceMetricsTracking()
        self.agent_tracking = AgentExperimentTracking()
        self.retrieval_tracking = RetrievalExperimentTracking()
        
        # Initialize evaluation endpoints
        self.foundation_endpoint = FoundationModelEndpoint()
        self.custom_endpoint = CustomModelEndpoint()
        self.fine_tuned_endpoint = FineTunedModelEndpoint()
        self.qlora_endpoint = QLoRAAdapterEndpoint()
        self.agentic_endpoint = AgenticWorkflowEndpoint()
        self.retrieval_endpoint = RetrievalWorkflowEndpoint()
        self.evaluation_orchestrator = UnifiedEvaluationOrchestrator()
        
        # Active evaluations and deployments
        self.active_evaluations = {}
        self.factory_deployments = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load Model Evaluation configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "evaluation": {
                "raw_models": {
                    "enabled": True,
                    "test_suites": ["performance", "accuracy", "latency", "memory"],
                    "stress_testing": True
                },
                "custom_models": {
                    "enabled": True,
                    "test_suites": ["custom_metrics", "domain_specific", "edge_cases"],
                    "comparison_baseline": True
                },
                "agentic_workflows": {
                    "enabled": True,
                    "test_suites": ["workflow_execution", "agent_performance", "task_completion"],
                    "mobile_optimization": True
                },
                "retrieval_workflows": {
                    "enabled": True,
                    "test_suites": ["retrieval_accuracy", "response_relevance", "context_quality"],
                    "multi_source": True
                }
            },
            "factory_roster": {
                "deployment_types": ["production", "staging", "testing"],
                "monitoring": {
                    "enabled": True,
                    "real_time_metrics": True,
                    "alerting": True
                },
                "environments": ["mobile", "edge", "cloud", "embedded"]
            },
            "mlflow": {
                "experiment_tracking": True,
                "model_registry": True,
                "performance_metrics": True
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for Model Evaluation workspace."""
        logger = logging.getLogger("model_evaluation_workspace")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def test_raw_models(self, request: EvaluationRequest) -> EvaluationResult:
        """Test raw foundation models."""
        try:
            self.logger.info(f"Testing raw foundation models: {request.model_name}")
            
            # Initialize MLflow experiment
            experiment_id = await self.experiment_tracking.start_experiment(
                name=f"raw_evaluation_{request.model_name}",
                tags={"type": "raw_evaluation", "model": request.model_name}
            )
            
            # Configure evaluation
            evaluation_config = {
                "model_name": request.model_name,
                "test_suite": request.test_suite,
                "parameters": request.parameters,
                "target_platform": request.target_platform,
                "stress_testing": request.stress_testing
            }
            
            # Run evaluation
            result = await self.raw_evaluation.evaluate_models(evaluation_config)
            
            # Track in MLflow
            await self.experiment_tracking.log_metrics(
                experiment_id=experiment_id,
                metrics=result.get("metrics", {}),
                step=0
            )
            
            # Calculate performance score
            performance_score = self._calculate_performance_score(result.get("metrics", {}))
            
            # Generate recommendations
            recommendations = self._generate_recommendations(result.get("metrics", {}))
            
            # Store evaluation
            evaluation_id = f"raw_{request.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.active_evaluations[evaluation_id] = {
                "type": "raw_evaluation",
                "model_name": request.model_name,
                "status": "completed",
                "result": result,
                "created_at": datetime.now()
            }
            
            return EvaluationResult(
                success=True,
                model_name=request.model_name,
                evaluation_type="raw",
                test_suite=request.test_suite,
                metrics=result.get("metrics", {}),
                performance_score=performance_score,
                recommendations=recommendations,
                created_at=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Raw model evaluation failed: {e}")
            return EvaluationResult(
                success=False,
                model_name=request.model_name,
                evaluation_type="raw",
                test_suite=request.test_suite,
                metrics={},
                performance_score=0.0,
                recommendations=[],
                created_at=datetime.now(),
                error_message=str(e)
            )
    
    async def test_custom_models(self, request: EvaluationRequest) -> EvaluationResult:
        """Test custom AI Architect models."""
        try:
            self.logger.info(f"Testing custom models: {request.model_name}")
            
            # Initialize MLflow experiment
            experiment_id = await self.experiment_tracking.start_experiment(
                name=f"custom_evaluation_{request.model_name}",
                tags={"type": "custom_evaluation", "model": request.model_name}
            )
            
            # Configure evaluation
            evaluation_config = {
                "model_name": request.model_name,
                "test_suite": request.test_suite,
                "parameters": request.parameters,
                "target_platform": request.target_platform,
                "stress_testing": request.stress_testing
            }
            
            # Run evaluation
            result = await self.custom_evaluation.evaluate_models(evaluation_config)
            
            # Track in MLflow
            await self.experiment_tracking.log_metrics(
                experiment_id=experiment_id,
                metrics=result.get("metrics", {}),
                step=0
            )
            
            # Calculate performance score
            performance_score = self._calculate_performance_score(result.get("metrics", {}))
            
            # Generate recommendations
            recommendations = self._generate_recommendations(result.get("metrics", {}))
            
            # Store evaluation
            evaluation_id = f"custom_{request.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.active_evaluations[evaluation_id] = {
                "type": "custom_evaluation",
                "model_name": request.model_name,
                "status": "completed",
                "result": result,
                "created_at": datetime.now()
            }
            
            return EvaluationResult(
                success=True,
                model_name=request.model_name,
                evaluation_type="custom",
                test_suite=request.test_suite,
                metrics=result.get("metrics", {}),
                performance_score=performance_score,
                recommendations=recommendations,
                created_at=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Custom model evaluation failed: {e}")
            return EvaluationResult(
                success=False,
                model_name=request.model_name,
                evaluation_type="custom",
                test_suite=request.test_suite,
                metrics={},
                performance_score=0.0,
                recommendations=[],
                created_at=datetime.now(),
                error_message=str(e)
            )
    
    async def test_agentic_workflows(self, request: EvaluationRequest) -> EvaluationResult:
        """Test agentic workflows (SmolAgent/LangGraph)."""
        try:
            self.logger.info(f"Testing agentic workflows: {request.model_name}")
            
            # Initialize MLflow experiment
            experiment_id = await self.experiment_tracking.start_experiment(
                name=f"agentic_evaluation_{request.model_name}",
                tags={"type": "agentic_evaluation", "model": request.model_name}
            )
            
            # Configure evaluation
            evaluation_config = {
                "model_name": request.model_name,
                "test_suite": request.test_suite,
                "parameters": request.parameters,
                "target_platform": request.target_platform,
                "stress_testing": request.stress_testing
            }
            
            # Run evaluation
            result = await self.agentic_evaluation.evaluate_workflows(evaluation_config)
            
            # Track in MLflow
            await self.agent_tracking.track_workflow_evaluation(
                experiment_id=experiment_id,
                workflow_name=request.model_name,
                metrics=result.get("metrics", {}),
                step=0
            )
            
            # Calculate performance score
            performance_score = self._calculate_performance_score(result.get("metrics", {}))
            
            # Generate recommendations
            recommendations = self._generate_recommendations(result.get("metrics", {}))
            
            # Store evaluation
            evaluation_id = f"agentic_{request.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.active_evaluations[evaluation_id] = {
                "type": "agentic_evaluation",
                "model_name": request.model_name,
                "status": "completed",
                "result": result,
                "created_at": datetime.now()
            }
            
            return EvaluationResult(
                success=True,
                model_name=request.model_name,
                evaluation_type="agentic",
                test_suite=request.test_suite,
                metrics=result.get("metrics", {}),
                performance_score=performance_score,
                recommendations=recommendations,
                created_at=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Agentic workflow evaluation failed: {e}")
            return EvaluationResult(
                success=False,
                model_name=request.model_name,
                evaluation_type="agentic",
                test_suite=request.test_suite,
                metrics={},
                performance_score=0.0,
                recommendations=[],
                created_at=datetime.now(),
                error_message=str(e)
            )
    
    async def test_retrieval_workflows(self, request: EvaluationRequest) -> EvaluationResult:
        """Test retrieval workflows (LangChain/LlamaIndex)."""
        try:
            self.logger.info(f"Testing retrieval workflows: {request.model_name}")
            
            # Initialize MLflow experiment
            experiment_id = await self.experiment_tracking.start_experiment(
                name=f"retrieval_evaluation_{request.model_name}",
                tags={"type": "retrieval_evaluation", "model": request.model_name}
            )
            
            # Configure evaluation
            evaluation_config = {
                "model_name": request.model_name,
                "test_suite": request.test_suite,
                "parameters": request.parameters,
                "target_platform": request.target_platform,
                "stress_testing": request.stress_testing
            }
            
            # Run evaluation
            result = await self.retrieval_evaluation.evaluate_workflows(evaluation_config)
            
            # Track in MLflow
            await self.retrieval_tracking.track_retrieval_evaluation(
                experiment_id=experiment_id,
                retrieval_name=request.model_name,
                metrics=result.get("metrics", {}),
                step=0
            )
            
            # Calculate performance score
            performance_score = self._calculate_performance_score(result.get("metrics", {}))
            
            # Generate recommendations
            recommendations = self._generate_recommendations(result.get("metrics", {}))
            
            # Store evaluation
            evaluation_id = f"retrieval_{request.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.active_evaluations[evaluation_id] = {
                "type": "retrieval_evaluation",
                "model_name": request.model_name,
                "status": "completed",
                "result": result,
                "created_at": datetime.now()
            }
            
            return EvaluationResult(
                success=True,
                model_name=request.model_name,
                evaluation_type="retrieval",
                test_suite=request.test_suite,
                metrics=result.get("metrics", {}),
                performance_score=performance_score,
                recommendations=recommendations,
                created_at=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Retrieval workflow evaluation failed: {e}")
            return EvaluationResult(
                success=False,
                model_name=request.model_name,
                evaluation_type="retrieval",
                test_suite=request.test_suite,
                metrics={},
                performance_score=0.0,
                recommendations=[],
                created_at=datetime.now(),
                error_message=str(e)
            )
    
    async def deploy_to_factory_roster(self, request: FactoryRosterRequest) -> FactoryRosterResult:
        """Deploy model to factory roster."""
        try:
            self.logger.info(f"Deploying {request.model_id} to factory roster")
            
            # Configure deployment
            deployment_config = {
                "model_id": request.model_id,
                "deployment_type": request.deployment_type,
                "target_environment": request.target_environment,
                "configuration": request.configuration,
                "monitoring_enabled": request.monitoring_enabled
            }
            
            # Deploy to factory roster
            result = await self.factory_roster.deploy_model(deployment_config)
            
            # Setup monitoring if enabled
            monitoring_url = None
            if request.monitoring_enabled:
                monitoring_result = await self.factory_roster.setup_monitoring(deployment_config)
                monitoring_url = monitoring_result.get("monitoring_url")
            
            # Store deployment
            deployment_id = f"deploy_{request.model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.factory_deployments[deployment_id] = {
                "model_id": request.model_id,
                "deployment_type": request.deployment_type,
                "target_environment": request.target_environment,
                "status": "deployed",
                "monitoring_url": monitoring_url,
                "created_at": datetime.now()
            }
            
            return FactoryRosterResult(
                success=True,
                model_id=request.model_id,
                deployment_type=request.deployment_type,
                deployment_status="deployed",
                monitoring_url=monitoring_url,
                created_at=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Factory roster deployment failed: {e}")
            return FactoryRosterResult(
                success=False,
                model_id=request.model_id,
                deployment_type=request.deployment_type,
                deployment_status="failed",
                created_at=datetime.now(),
                error_message=str(e)
            )
    
    async def get_evaluation_results(self) -> Dict[str, Any]:
        """Get all evaluation results."""
        try:
            results = {
                "total_evaluations": len(self.active_evaluations),
                "by_type": {},
                "recent_evaluations": [],
                "performance_summary": {}
            }
            
            # Group by evaluation type
            for evaluation_id, evaluation in self.active_evaluations.items():
                evaluation_type = evaluation["type"]
                if evaluation_type not in results["by_type"]:
                    results["by_type"][evaluation_type] = 0
                results["by_type"][evaluation_type] += 1
                
                # Add to recent evaluations (last 10)
                if len(results["recent_evaluations"]) < 10:
                    results["recent_evaluations"].append({
                        "id": evaluation_id,
                        "type": evaluation_type,
                        "model_name": evaluation["model_name"],
                        "status": evaluation["status"],
                        "created_at": evaluation["created_at"]
                    })
            
            # Calculate performance summary
            performance_metrics = []
            for evaluation in self.active_evaluations.values():
                if "result" in evaluation and "metrics" in evaluation["result"]:
                    performance_metrics.append(evaluation["result"]["metrics"])
            
            if performance_metrics:
                results["performance_summary"] = self._calculate_performance_summary(performance_metrics)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to get evaluation results: {e}")
            return {
                "total_evaluations": 0,
                "by_type": {},
                "recent_evaluations": [],
                "performance_summary": {},
                "error": str(e)
            }
    
    async def get_factory_roster_status(self) -> Dict[str, Any]:
        """Get factory roster deployment status."""
        try:
            status = {
                "total_deployments": len(self.factory_deployments),
                "by_type": {},
                "by_environment": {},
                "recent_deployments": [],
                "monitoring_status": {}
            }
            
            # Group by deployment type and environment
            for deployment_id, deployment in self.factory_deployments.items():
                deployment_type = deployment["deployment_type"]
                target_environment = deployment["target_environment"]
                
                if deployment_type not in status["by_type"]:
                    status["by_type"][deployment_type] = 0
                status["by_type"][deployment_type] += 1
                
                if target_environment not in status["by_environment"]:
                    status["by_environment"][target_environment] = 0
                status["by_environment"][target_environment] += 1
                
                # Add to recent deployments (last 10)
                if len(status["recent_deployments"]) < 10:
                    status["recent_deployments"].append({
                        "id": deployment_id,
                        "model_id": deployment["model_id"],
                        "deployment_type": deployment_type,
                        "target_environment": target_environment,
                        "status": deployment["status"],
                        "monitoring_url": deployment.get("monitoring_url"),
                        "created_at": deployment["created_at"]
                    })
            
            # Get monitoring status
            monitoring_status = await self.factory_roster.get_monitoring_status()
            status["monitoring_status"] = monitoring_status
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get factory roster status: {e}")
            return {
                "total_deployments": 0,
                "by_type": {},
                "by_environment": {},
                "recent_deployments": [],
                "monitoring_status": {},
                "error": str(e)
            }
    
    def _calculate_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall performance score from metrics."""
        try:
            # Weighted scoring based on key metrics
            weights = {
                "accuracy": 0.3,
                "latency": 0.2,
                "throughput": 0.2,
                "memory_usage": 0.15,
                "cpu_usage": 0.15
            }
            
            score = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric in metrics:
                    # Normalize metric value (assuming 0-1 scale)
                    value = float(metrics[metric])
                    if metric in ["latency", "memory_usage", "cpu_usage"]:
                        # Lower is better for these metrics
                        value = 1.0 - value
                    
                    score += value * weight
                    total_weight += weight
            
            return score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to calculate performance score: {e}")
            return 0.0
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on evaluation metrics."""
        recommendations = []
        
        try:
            # Check accuracy
            if "accuracy" in metrics and metrics["accuracy"] < 0.8:
                recommendations.append("Consider fine-tuning to improve accuracy")
            
            # Check latency
            if "latency" in metrics and metrics["latency"] > 1000:  # > 1 second
                recommendations.append("Optimize model for better latency performance")
            
            # Check memory usage
            if "memory_usage" in metrics and metrics["memory_usage"] > 0.8:
                recommendations.append("Consider model quantization to reduce memory usage")
            
            # Check throughput
            if "throughput" in metrics and metrics["throughput"] < 10:  # < 10 requests/second
                recommendations.append("Scale infrastructure to improve throughput")
            
            # Check CPU usage
            if "cpu_usage" in metrics and metrics["cpu_usage"] > 0.9:
                recommendations.append("Optimize CPU usage or scale horizontally")
            
            if not recommendations:
                recommendations.append("Model performance is within acceptable ranges")
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            recommendations.append("Unable to generate recommendations")
        
        return recommendations
    
    def _calculate_performance_summary(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance summary from multiple evaluation metrics."""
        try:
            summary = {}
            
            # Calculate averages for each metric
            metric_names = set()
            for metrics in metrics_list:
                metric_names.update(metrics.keys())
            
            for metric in metric_names:
                values = [metrics.get(metric, 0) for metrics in metrics_list if metric in metrics]
                if values:
                    summary[metric] = {
                        "average": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "count": len(values)
                    }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to calculate performance summary: {e}")
            return {}


# FastAPI Router for Model Evaluation Workspace
router = APIRouter(prefix="/model-evaluation", tags=["Model Evaluation Workspace"])

# Global workspace instance
workspace = ModelEvaluationWorkspace()


@router.get("/status")
async def get_workspace_status():
    """Get Model Evaluation workspace status."""
    try:
        evaluation_results = await workspace.get_evaluation_results()
        factory_status = await workspace.get_factory_roster_status()
        
        status = {
            "evaluation_results": evaluation_results,
            "factory_roster_status": factory_status,
            "timestamp": datetime.now()
        }
        
        return JSONResponse(content=status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate-model")
async def evaluate_model(request: EvaluationRequest):
    """Evaluate model based on request type."""
    try:
        if request.evaluation_type == "raw":
            result = await workspace.test_raw_models(request)
        elif request.evaluation_type == "custom":
            result = await workspace.test_custom_models(request)
        elif request.evaluation_type == "agentic":
            result = await workspace.test_agentic_workflows(request)
        elif request.evaluation_type == "retrieval":
            result = await workspace.test_retrieval_workflows(request)
        else:
            raise HTTPException(status_code=400, detail="Invalid evaluation type")
        
        return JSONResponse(content=asdict(result))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deploy-to-factory")
async def deploy_to_factory(request: FactoryRosterRequest):
    """Deploy model to factory roster."""
    try:
        result = await workspace.deploy_to_factory_roster(request)
        return JSONResponse(content=asdict(result))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluations")
async def get_evaluations():
    """Get all active evaluations."""
    try:
        return JSONResponse(content=workspace.active_evaluations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deployments")
async def get_deployments():
    """Get all factory roster deployments."""
    try:
        return JSONResponse(content=workspace.factory_deployments)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workspace")
async def get_workspace_interface():
    """Get Model Evaluation workspace interface."""
    try:
        with open("src/enterprise_llmops/frontend/enhanced_unified_platform.html", "r") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
