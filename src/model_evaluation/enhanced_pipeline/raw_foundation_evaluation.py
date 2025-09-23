"""
Raw Foundation Model Evaluation

This module provides comprehensive evaluation of raw foundation models
including performance testing, benchmarking, and stress testing.

Key Features:
- Raw foundation model testing
- Performance benchmarking
- Stress testing at business/consumer levels
- MLflow experiment tracking
- GitHub Models API integration
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

# GitHub Models API
from ...github_models_integration.inference_client import GitHubModelsInferenceClient

logger = logging.getLogger(__name__)


@dataclass
class FoundationModelTest:
    """Foundation model test configuration."""
    test_id: str
    model_name: str
    test_type: str  # "performance", "stress", "benchmark"
    parameters: Dict[str, Any]
    expected_metrics: Dict[str, float]
    test_prompts: List[str]
    success_criteria: Dict[str, Any]
    created_at: datetime
    status: str = "PENDING"  # PENDING, RUNNING, COMPLETED, FAILED


@dataclass
class FoundationModelResult:
    """Foundation model test result."""
    test_id: str
    model_name: str
    success: bool
    metrics: Dict[str, float]
    response_times: List[float]
    token_counts: List[int]
    error_rate: float
    throughput: float
    latency_p95: float
    latency_p99: float
    memory_usage: float
    cpu_usage: float
    error_message: Optional[str] = None
    completed_at: Optional[datetime] = None


class RawFoundationEvaluator:
    """
    Raw Foundation Model Evaluator
    
    This class provides comprehensive evaluation of raw foundation models
    including performance testing, benchmarking, and stress testing.
    """
    
    def __init__(self, 
                 mlflow_tracking_uri: str = "http://localhost:5000",
                 github_models_token: Optional[str] = None):
        """
        Initialize the Raw Foundation Evaluator.
        
        Args:
            mlflow_tracking_uri: MLflow tracking server URI
            github_models_token: GitHub Models API token
        """
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.github_models_token = github_models_token
        
        # Initialize MLflow
        if mlflow:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            self.client = MlflowClient(tracking_uri=mlflow_tracking_uri)
            self._ensure_experiment()
        else:
            self.client = None
            logger.warning("MLflow not available, experiment tracking disabled")
        
        # Initialize GitHub Models client
        if github_models_token:
            self.github_client = GitHubModelsInferenceClient(github_models_token)
        else:
            self.github_client = None
            logger.warning("GitHub Models token not provided, GitHub Models API disabled")
        
        # Test tracking
        self.active_tests: Dict[str, FoundationModelTest] = {}
        self.completed_tests: Dict[str, FoundationModelResult] = {}
        
        logger.info("Raw Foundation Evaluator initialized")
    
    def _ensure_experiment(self):
        """Ensure foundation model experiment exists."""
        if not self.client:
            return
            
        try:
            experiment_name = "foundation_model_evaluation"
            experiment = self.client.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = self.client.create_experiment(experiment_name)
                logger.info(f"Created experiment: {experiment_name}")
            else:
                logger.info(f"Using existing experiment: {experiment_name}")
        except Exception as e:
            logger.warning(f"Could not ensure experiment: {e}")
    
    def create_performance_test(self, 
                              model_name: str,
                              test_prompts: List[str],
                              parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a performance test for a foundation model.
        
        Args:
            model_name: Name of the model to test
            test_prompts: List of test prompts
            parameters: Test parameters
            
        Returns:
            Test ID
        """
        test_id = str(uuid.uuid4())
        
        if parameters is None:
            parameters = {
                "max_tokens": 100,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            }
        
        test = FoundationModelTest(
            test_id=test_id,
            model_name=model_name,
            test_type="performance",
            parameters=parameters,
            expected_metrics={
                "latency_p95": 2.0,  # seconds
                "throughput": 10.0,  # requests/second
                "error_rate": 0.01,  # 1%
                "memory_usage": 0.8   # 80% of available
            },
            test_prompts=test_prompts,
            success_criteria={
                "min_throughput": 5.0,
                "max_latency_p95": 5.0,
                "max_error_rate": 0.05
            },
            created_at=datetime.now()
        )
        
        self.active_tests[test_id] = test
        logger.info(f"Created performance test {test_id} for model {model_name}")
        
        return test_id
    
    def create_stress_test(self,
                          model_name: str,
                          test_prompts: List[str],
                          stress_level: str = "business",  # "business", "consumer"
                          parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a stress test for a foundation model.
        
        Args:
            model_name: Name of the model to test
            test_prompts: List of test prompts
            stress_level: Stress test level
            parameters: Test parameters
            
        Returns:
            Test ID
        """
        test_id = str(uuid.uuid4())
        
        if parameters is None:
            parameters = {
                "max_tokens": 100,
                "temperature": 0.7,
                "top_p": 0.9
            }
        
        # Define stress test criteria based on level
        if stress_level == "business":
            expected_metrics = {
                "latency_p95": 5.0,
                "throughput": 20.0,
                "error_rate": 0.02,
                "memory_usage": 0.9
            }
            success_criteria = {
                "min_throughput": 15.0,
                "max_latency_p95": 10.0,
                "max_error_rate": 0.05
            }
        else:  # consumer
            expected_metrics = {
                "latency_p95": 3.0,
                "throughput": 50.0,
                "error_rate": 0.01,
                "memory_usage": 0.7
            }
            success_criteria = {
                "min_throughput": 30.0,
                "max_latency_p95": 5.0,
                "max_error_rate": 0.03
            }
        
        test = FoundationModelTest(
            test_id=test_id,
            model_name=model_name,
            test_type="stress",
            parameters=parameters,
            expected_metrics=expected_metrics,
            test_prompts=test_prompts,
            success_criteria=success_criteria,
            created_at=datetime.now()
        )
        
        self.active_tests[test_id] = test
        logger.info(f"Created stress test {test_id} for model {model_name} at {stress_level} level")
        
        return test_id
    
    async def run_test(self, test_id: str) -> FoundationModelResult:
        """
        Run a foundation model test.
        
        Args:
            test_id: Test ID to run
            
        Returns:
            Test result
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self.active_tests[test_id]
        test.status = "RUNNING"
        
        logger.info(f"Starting test {test_id} for model {test.model_name}")
        
        # Start MLflow run
        run_id = None
        if self.client:
            try:
                with mlflow.start_run(experiment_id=self._get_experiment_id()):
                    run_id = mlflow.active_run().info.run_id
                    
                    # Log test parameters
                    mlflow.log_params({
                        "model_name": test.model_name,
                        "test_type": test.test_type,
                        **test.parameters
                    })
                    
                    # Run the actual test
                    result = await self._execute_test(test)
                    
                    # Log results
                    mlflow.log_metrics(result.metrics)
                    mlflow.log_artifacts(self._create_test_artifacts(result))
                    
            except Exception as e:
                logger.error(f"MLflow tracking failed: {e}")
                result = await self._execute_test(test)
        else:
            result = await self._execute_test(test)
        
        # Update test status
        test.status = "COMPLETED" if result.success else "FAILED"
        self.completed_tests[test_id] = result
        del self.active_tests[test_id]
        
        logger.info(f"Completed test {test_id}, success: {result.success}")
        
        return result
    
    async def _execute_test(self, test: FoundationModelTest) -> FoundationModelResult:
        """Execute the actual test."""
        start_time = time.time()
        response_times = []
        token_counts = []
        errors = 0
        total_requests = len(test.test_prompts)
        
        # Run test prompts
        for prompt in test.test_prompts:
            try:
                prompt_start = time.time()
                
                # Use GitHub Models API if available
                if self.github_client:
                    response = await self.github_client.generate(
                        model=test.model_name,
                        prompt=prompt,
                        **test.parameters
                    )
                    response_text = response.get("choices", [{}])[0].get("text", "")
                    tokens = response.get("usage", {}).get("completion_tokens", 0)
                else:
                    # Fallback to mock response
                    response_text = f"Mock response for: {prompt[:50]}..."
                    tokens = 50
                
                prompt_time = time.time() - prompt_start
                response_times.append(prompt_time)
                token_counts.append(tokens)
                
            except Exception as e:
                logger.error(f"Error in test prompt: {e}")
                errors += 1
                response_times.append(0)
                token_counts.append(0)
        
        # Calculate metrics
        total_time = time.time() - start_time
        throughput = total_requests / total_time if total_time > 0 else 0
        error_rate = errors / total_requests if total_requests > 0 else 0
        
        # Calculate percentiles
        sorted_times = sorted(response_times)
        latency_p95 = sorted_times[int(len(sorted_times) * 0.95)] if sorted_times else 0
        latency_p99 = sorted_times[int(len(sorted_times) * 0.99)] if sorted_times else 0
        
        # Check success criteria
        success = (
            throughput >= test.success_criteria.get("min_throughput", 0) and
            latency_p95 <= test.success_criteria.get("max_latency_p95", float('inf')) and
            error_rate <= test.success_criteria.get("max_error_rate", 1.0)
        )
        
        result = FoundationModelResult(
            test_id=test.test_id,
            model_name=test.model_name,
            success=success,
            metrics={
                "throughput": throughput,
                "latency_p95": latency_p95,
                "latency_p99": latency_p99,
                "error_rate": error_rate,
                "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
                "total_tokens": sum(token_counts),
                "avg_tokens_per_response": sum(token_counts) / len(token_counts) if token_counts else 0
            },
            response_times=response_times,
            token_counts=token_counts,
            error_rate=error_rate,
            throughput=throughput,
            latency_p95=latency_p95,
            latency_p99=latency_p99,
            memory_usage=0.8,  # Mock memory usage
            cpu_usage=0.6,     # Mock CPU usage
            completed_at=datetime.now()
        )
        
        return result
    
    def _get_experiment_id(self) -> str:
        """Get the foundation model experiment ID."""
        if not self.client:
            return None
            
        try:
            experiment = self.client.get_experiment_by_name("foundation_model_evaluation")
            return experiment.experiment_id if experiment else None
        except Exception:
            return None
    
    def _create_test_artifacts(self, result: FoundationModelResult) -> str:
        """Create test artifacts for MLflow."""
        artifacts = {
            "test_result": asdict(result),
            "summary": {
                "model_name": result.model_name,
                "success": result.success,
                "key_metrics": result.metrics
            }
        }
        
        # Save to temporary file
        import tempfile
        import os
        
        temp_dir = tempfile.mkdtemp()
        artifacts_file = os.path.join(temp_dir, "test_artifacts.json")
        
        with open(artifacts_file, 'w') as f:
            json.dump(artifacts, f, indent=2, default=str)
        
        return temp_dir
    
    def get_test_status(self, test_id: str) -> Dict[str, Any]:
        """Get the status of a test."""
        if test_id in self.active_tests:
            test = self.active_tests[test_id]
            return {
                "test_id": test_id,
                "status": test.status,
                "model_name": test.model_name,
                "test_type": test.test_type,
                "created_at": test.created_at
            }
        elif test_id in self.completed_tests:
            result = self.completed_tests[test_id]
            return {
                "test_id": test_id,
                "status": "COMPLETED",
                "model_name": result.model_name,
                "success": result.success,
                "metrics": result.metrics,
                "completed_at": result.completed_at
            }
        else:
            return {"error": "Test not found"}
    
    def list_tests(self) -> List[Dict[str, Any]]:
        """List all tests."""
        tests = []
        
        # Active tests
        for test_id, test in self.active_tests.items():
            tests.append({
                "test_id": test_id,
                "status": test.status,
                "model_name": test.model_name,
                "test_type": test.test_type,
                "created_at": test.created_at
            })
        
        # Completed tests
        for test_id, result in self.completed_tests.items():
            tests.append({
                "test_id": test_id,
                "status": "COMPLETED",
                "model_name": result.model_name,
                "success": result.success,
                "completed_at": result.completed_at
            })
        
        return tests


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Initialize evaluator
        evaluator = RawFoundationEvaluator()
        
        # Create performance test
        test_prompts = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "Write a short story about a robot."
        ]
        
        test_id = evaluator.create_performance_test(
            model_name="microsoft/phi-4-mini-instruct",
            test_prompts=test_prompts
        )
        
        print(f"Created test: {test_id}")
        
        # Run test
        result = await evaluator.run_test(test_id)
        print(f"Test completed: {result.success}")
        print(f"Metrics: {result.metrics}")
    
    asyncio.run(main())
