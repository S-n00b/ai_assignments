"""
Mobile Model Evaluation for Edge Deployment

This module provides comprehensive evaluation of mobile/edge models including
performance testing, optimization validation, and deployment readiness assessment.

Key Features:
- Mobile/edge specific evaluation
- Performance optimization testing
- Platform-specific validation
- Resource usage monitoring
- Deployment readiness assessment
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

# Import mobile optimization components
from ...enterprise_llmops.small_models.small_model_optimizer import SmallModelOptimizer
from ...enterprise_llmops.small_models.mobile_deployment_configs import MobileDeploymentConfigManager
from ...enterprise_llmops.small_models.model_performance_monitor import ModelPerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class MobileModelTest:
    """Mobile model test configuration."""
    test_id: str
    model_name: str
    platform: str  # "android", "ios", "edge", "embedded"
    optimization_level: str  # "light", "medium", "aggressive"
    test_scenarios: List[str]
    performance_requirements: Dict[str, Any]
    resource_limits: Dict[str, Any]
    created_at: datetime
    status: str = "PENDING"


@dataclass
class MobileModelResult:
    """Mobile model test result."""
    test_id: str
    model_name: str
    platform: str
    optimization_level: str
    success: bool
    performance_metrics: Dict[str, float]
    resource_usage: Dict[str, float]
    optimization_metrics: Dict[str, float]
    deployment_readiness: Dict[str, Any]
    platform_specific_metrics: Dict[str, Any]
    error_message: Optional[str] = None
    completed_at: Optional[datetime] = None


class MobileModelEvaluator:
    """
    Mobile Model Evaluator for Edge Deployment
    
    This class provides comprehensive evaluation of mobile/edge models including
    performance testing, optimization validation, and deployment readiness assessment.
    """
    
    def __init__(self, 
                 mlflow_tracking_uri: str = "http://localhost:5000"):
        """
        Initialize the Mobile Model Evaluator.
        
        Args:
            mlflow_tracking_uri: MLflow tracking server URI
        """
        self.mlflow_tracking_uri = mlflow_tracking_uri
        
        # Initialize MLflow
        if mlflow:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            self.client = MlflowClient(tracking_uri=mlflow_tracking_uri)
            self._ensure_experiment()
        else:
            self.client = None
            logger.warning("MLflow not available, experiment tracking disabled")
        
        # Initialize mobile optimization components
        self.optimizer = SmallModelOptimizer()
        self.deployment_manager = MobileDeploymentConfigManager()
        self.performance_monitor = ModelPerformanceMonitor()
        
        # Test tracking
        self.active_tests: Dict[str, MobileModelTest] = {}
        self.completed_tests: Dict[str, MobileModelResult] = {}
        
        logger.info("Mobile Model Evaluator initialized")
    
    def _ensure_experiment(self):
        """Ensure mobile model experiment exists."""
        if not self.client:
            return
            
        try:
            experiment_name = "mobile_model_evaluation"
            experiment = self.client.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = self.client.create_experiment(experiment_name)
                logger.info(f"Created experiment: {experiment_name}")
            else:
                logger.info(f"Using existing experiment: {experiment_name}")
        except Exception as e:
            logger.warning(f"Could not ensure experiment: {e}")
    
    def create_mobile_performance_test(self,
                                     model_name: str,
                                     platform: str,
                                     optimization_level: str = "medium",
                                     test_scenarios: Optional[List[str]] = None,
                                     performance_requirements: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a mobile performance test.
        
        Args:
            model_name: Name of the model to test
            platform: Target platform
            optimization_level: Optimization level
            test_scenarios: Test scenarios
            performance_requirements: Performance requirements
            
        Returns:
            Test ID
        """
        test_id = str(uuid.uuid4())
        
        if test_scenarios is None:
            test_scenarios = [
                "inference_speed",
                "memory_usage",
                "battery_consumption",
                "thermal_performance",
                "network_efficiency"
            ]
        
        if performance_requirements is None:
            performance_requirements = {
                "max_inference_time": 2.0,
                "max_memory_usage": 0.8,
                "max_battery_consumption": 0.1,
                "min_throughput": 5.0
            }
        
        # Platform-specific resource limits
        resource_limits = self._get_platform_resource_limits(platform)
        
        test = MobileModelTest(
            test_id=test_id,
            model_name=model_name,
            platform=platform,
            optimization_level=optimization_level,
            test_scenarios=test_scenarios,
            performance_requirements=performance_requirements,
            resource_limits=resource_limits,
            created_at=datetime.now()
        )
        
        self.active_tests[test_id] = test
        logger.info(f"Created mobile performance test {test_id} for {model_name} on {platform}")
        
        return test_id
    
    def create_optimization_validation_test(self,
                                          model_name: str,
                                          platform: str,
                                          optimization_level: str,
                                          base_model_name: str) -> str:
        """
        Create an optimization validation test.
        
        Args:
            model_name: Name of the optimized model
            platform: Target platform
            optimization_level: Optimization level
            base_model_name: Base model name for comparison
            
        Returns:
            Test ID
        """
        test_id = str(uuid.uuid4())
        
        test_scenarios = [
            "size_reduction",
            "speed_improvement",
            "accuracy_preservation",
            "resource_efficiency",
            "deployment_readiness"
        ]
        
        performance_requirements = {
            "min_size_reduction": 0.3,  # 30% size reduction
            "min_speed_improvement": 0.2,  # 20% speed improvement
            "min_accuracy_preservation": 0.95,  # 95% accuracy preservation
            "max_resource_usage": 0.7
        }
        
        resource_limits = self._get_platform_resource_limits(platform)
        
        test = MobileModelTest(
            test_id=test_id,
            model_name=model_name,
            platform=platform,
            optimization_level=optimization_level,
            test_scenarios=test_scenarios,
            performance_requirements=performance_requirements,
            resource_limits=resource_limits,
            created_at=datetime.now()
        )
        
        self.active_tests[test_id] = test
        logger.info(f"Created optimization validation test {test_id} for {model_name}")
        
        return test_id
    
    async def run_test(self, test_id: str) -> MobileModelResult:
        """
        Run a mobile model test.
        
        Args:
            test_id: Test ID to run
            
        Returns:
            Test result
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self.active_tests[test_id]
        test.status = "RUNNING"
        
        logger.info(f"Starting mobile model test {test_id} for {test.model_name}")
        
        # Start MLflow run
        run_id = None
        if self.client:
            try:
                with mlflow.start_run(experiment_id=self._get_experiment_id()):
                    run_id = mlflow.active_run().info.run_id
                    
                    # Log test parameters
                    mlflow.log_params({
                        "model_name": test.model_name,
                        "platform": test.platform,
                        "optimization_level": test.optimization_level,
                        **test.performance_requirements
                    })
                    
                    # Run the actual test
                    result = await self._execute_test(test)
                    
                    # Log results
                    mlflow.log_metrics(result.performance_metrics)
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
        
        logger.info(f"Completed mobile model test {test_id}, success: {result.success}")
        
        return result
    
    async def _execute_test(self, test: MobileModelTest) -> MobileModelResult:
        """Execute the actual mobile model test."""
        start_time = time.time()
        
        try:
            # Get optimization configuration
            optimization_config = self.optimizer.get_optimization_config(
                test.optimization_level
            )
            
            # Test performance metrics
            performance_metrics = await self._test_performance_metrics(test)
            
            # Test resource usage
            resource_usage = await self._test_resource_usage(test)
            
            # Test optimization metrics
            optimization_metrics = await self._test_optimization_metrics(test)
            
            # Test platform-specific metrics
            platform_metrics = await self._test_platform_specific_metrics(test)
            
            # Assess deployment readiness
            deployment_readiness = self._assess_deployment_readiness(
                test, performance_metrics, resource_usage, optimization_metrics
            )
            
            # Determine success
            success = self._evaluate_success(test, performance_metrics, resource_usage)
            
            result = MobileModelResult(
                test_id=test.test_id,
                model_name=test.model_name,
                platform=test.platform,
                optimization_level=test.optimization_level,
                success=success,
                performance_metrics=performance_metrics,
                resource_usage=resource_usage,
                optimization_metrics=optimization_metrics,
                deployment_readiness=deployment_readiness,
                platform_specific_metrics=platform_metrics,
                completed_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in mobile model test: {e}")
            result = MobileModelResult(
                test_id=test.test_id,
                model_name=test.model_name,
                platform=test.platform,
                optimization_level=test.optimization_level,
                success=False,
                performance_metrics={},
                resource_usage={},
                optimization_metrics={},
                deployment_readiness={},
                platform_specific_metrics={},
                error_message=str(e),
                completed_at=datetime.now()
            )
        
        return result
    
    async def _test_performance_metrics(self, test: MobileModelTest) -> Dict[str, float]:
        """Test performance metrics."""
        # Mock performance testing
        metrics = {
            "inference_time": 1.5,  # seconds
            "throughput": 12.0,     # requests/second
            "latency_p95": 2.1,     # seconds
            "latency_p99": 3.0,     # seconds
            "response_quality": 0.85
        }
        
        # Platform-specific adjustments
        if test.platform == "android":
            metrics["inference_time"] *= 1.1  # Android overhead
        elif test.platform == "ios":
            metrics["inference_time"] *= 0.9  # iOS optimization
        elif test.platform == "edge":
            metrics["inference_time"] *= 1.2  # Edge constraints
        elif test.platform == "embedded":
            metrics["inference_time"] *= 1.5  # Embedded limitations
        
        return metrics
    
    async def _test_resource_usage(self, test: MobileModelTest) -> Dict[str, float]:
        """Test resource usage."""
        # Mock resource usage testing
        usage = {
            "memory_usage": 0.6,      # 60% of available memory
            "cpu_usage": 0.4,         # 40% of CPU
            "battery_consumption": 0.08,  # 8% battery per hour
            "storage_usage": 0.3,    # 30% of storage
            "network_usage": 0.1      # 10% of network bandwidth
        }
        
        # Optimization level adjustments
        if test.optimization_level == "light":
            usage["memory_usage"] *= 1.2
            usage["cpu_usage"] *= 1.1
        elif test.optimization_level == "aggressive":
            usage["memory_usage"] *= 0.7
            usage["cpu_usage"] *= 0.8
        
        return usage
    
    async def _test_optimization_metrics(self, test: MobileModelTest) -> Dict[str, float]:
        """Test optimization metrics."""
        # Mock optimization testing
        metrics = {
            "size_reduction": 0.4,        # 40% size reduction
            "speed_improvement": 0.25,     # 25% speed improvement
            "accuracy_preservation": 0.95, # 95% accuracy preservation
            "efficiency_gain": 0.3,       # 30% efficiency gain
            "compression_ratio": 0.6      # 60% compression ratio
        }
        
        # Optimization level adjustments
        if test.optimization_level == "light":
            metrics["size_reduction"] = 0.2
            metrics["speed_improvement"] = 0.1
            metrics["accuracy_preservation"] = 0.98
        elif test.optimization_level == "aggressive":
            metrics["size_reduction"] = 0.6
            metrics["speed_improvement"] = 0.4
            metrics["accuracy_preservation"] = 0.90
        
        return metrics
    
    async def _test_platform_specific_metrics(self, test: MobileModelTest) -> Dict[str, float]:
        """Test platform-specific metrics."""
        metrics = {}
        
        if test.platform == "android":
            metrics = {
                "apk_size": 0.8,          # 80% of original size
                "install_time": 1.2,      # 1.2 seconds
                "startup_time": 0.8,     # 0.8 seconds
                "thermal_performance": 0.7
            }
        elif test.platform == "ios":
            metrics = {
                "app_size": 0.75,         # 75% of original size
                "install_time": 1.0,     # 1.0 seconds
                "startup_time": 0.6,     # 0.6 seconds
                "thermal_performance": 0.8
            }
        elif test.platform == "edge":
            metrics = {
                "container_size": 0.7,     # 70% of original size
                "deployment_time": 2.0,   # 2.0 seconds
                "startup_time": 1.5,     # 1.5 seconds
                "network_efficiency": 0.85
            }
        elif test.platform == "embedded":
            metrics = {
                "firmware_size": 0.5,     # 50% of original size
                "boot_time": 3.0,        # 3.0 seconds
                "startup_time": 2.0,     # 2.0 seconds
                "power_efficiency": 0.9
            }
        
        return metrics
    
    def _assess_deployment_readiness(self, 
                                   test: MobileModelTest,
                                   performance_metrics: Dict[str, float],
                                   resource_usage: Dict[str, float],
                                   optimization_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Assess deployment readiness."""
        readiness = {
            "overall_score": 0.0,
            "performance_ready": False,
            "resource_ready": False,
            "optimization_ready": False,
            "platform_ready": False,
            "recommendations": []
        }
        
        # Performance readiness
        if (performance_metrics.get("inference_time", 0) <= test.performance_requirements.get("max_inference_time", 2.0) and
            performance_metrics.get("throughput", 0) >= test.performance_requirements.get("min_throughput", 5.0)):
            readiness["performance_ready"] = True
        
        # Resource readiness
        if (resource_usage.get("memory_usage", 1.0) <= test.resource_limits.get("max_memory", 0.8) and
            resource_usage.get("cpu_usage", 1.0) <= test.resource_limits.get("max_cpu", 0.7)):
            readiness["resource_ready"] = True
        
        # Optimization readiness
        if (optimization_metrics.get("size_reduction", 0) >= test.performance_requirements.get("min_size_reduction", 0.3) and
            optimization_metrics.get("accuracy_preservation", 0) >= test.performance_requirements.get("min_accuracy_preservation", 0.95)):
            readiness["optimization_ready"] = True
        
        # Platform readiness (mock)
        readiness["platform_ready"] = True
        
        # Overall score
        ready_count = sum([readiness["performance_ready"], 
                          readiness["resource_ready"], 
                          readiness["optimization_ready"],
                          readiness["platform_ready"]])
        readiness["overall_score"] = ready_count / 4.0
        
        # Recommendations
        if not readiness["performance_ready"]:
            readiness["recommendations"].append("Optimize inference time and throughput")
        if not readiness["resource_ready"]:
            readiness["recommendations"].append("Reduce memory and CPU usage")
        if not readiness["optimization_ready"]:
            readiness["recommendations"].append("Improve size reduction and accuracy preservation")
        
        return readiness
    
    def _evaluate_success(self, 
                        test: MobileModelTest,
                        performance_metrics: Dict[str, float],
                        resource_usage: Dict[str, float]) -> bool:
        """Evaluate if the test meets success criteria."""
        success = True
        
        # Check performance requirements
        if performance_metrics.get("inference_time", 0) > test.performance_requirements.get("max_inference_time", 2.0):
            success = False
        
        if performance_metrics.get("throughput", 0) < test.performance_requirements.get("min_throughput", 5.0):
            success = False
        
        # Check resource limits
        if resource_usage.get("memory_usage", 1.0) > test.resource_limits.get("max_memory", 0.8):
            success = False
        
        if resource_usage.get("cpu_usage", 1.0) > test.resource_limits.get("max_cpu", 0.7):
            success = False
        
        return success
    
    def _get_platform_resource_limits(self, platform: str) -> Dict[str, Any]:
        """Get platform-specific resource limits."""
        limits = {
            "android": {
                "max_memory": 0.8,
                "max_cpu": 0.7,
                "max_battery": 0.1,
                "max_storage": 0.5
            },
            "ios": {
                "max_memory": 0.7,
                "max_cpu": 0.6,
                "max_battery": 0.08,
                "max_storage": 0.4
            },
            "edge": {
                "max_memory": 0.9,
                "max_cpu": 0.8,
                "max_network": 0.5,
                "max_storage": 0.6
            },
            "embedded": {
                "max_memory": 0.5,
                "max_cpu": 0.4,
                "max_power": 0.05,
                "max_storage": 0.3
            }
        }
        
        return limits.get(platform, limits["android"])
    
    def _get_experiment_id(self) -> str:
        """Get the mobile model experiment ID."""
        if not self.client:
            return None
            
        try:
            experiment = self.client.get_experiment_by_name("mobile_model_evaluation")
            return experiment.experiment_id if experiment else None
        except Exception:
            return None
    
    def _create_test_artifacts(self, result: MobileModelResult) -> str:
        """Create test artifacts for MLflow."""
        artifacts = {
            "test_result": asdict(result),
            "summary": {
                "model_name": result.model_name,
                "platform": result.platform,
                "success": result.success,
                "deployment_readiness": result.deployment_readiness
            }
        }
        
        # Save to temporary file
        import tempfile
        import os
        
        temp_dir = tempfile.mkdtemp()
        artifacts_file = os.path.join(temp_dir, "mobile_model_artifacts.json")
        
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
                "platform": test.platform,
                "optimization_level": test.optimization_level,
                "created_at": test.created_at
            }
        elif test_id in self.completed_tests:
            result = self.completed_tests[test_id]
            return {
                "test_id": test_id,
                "status": "COMPLETED",
                "model_name": result.model_name,
                "platform": result.platform,
                "success": result.success,
                "deployment_readiness": result.deployment_readiness,
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
                "platform": test.platform,
                "optimization_level": test.optimization_level,
                "created_at": test.created_at
            })
        
        # Completed tests
        for test_id, result in self.completed_tests.items():
            tests.append({
                "test_id": test_id,
                "status": "COMPLETED",
                "model_name": result.model_name,
                "platform": result.platform,
                "success": result.success,
                "completed_at": result.completed_at
            })
        
        return tests


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Initialize evaluator
        evaluator = MobileModelEvaluator()
        
        # Create mobile performance test
        test_id = evaluator.create_mobile_performance_test(
            model_name="phi-4-mini-optimized",
            platform="android",
            optimization_level="medium"
        )
        
        print(f"Created mobile performance test: {test_id}")
        
        # Run test
        result = await evaluator.run_test(test_id)
        print(f"Test completed: {result.success}")
        print(f"Deployment readiness: {result.deployment_readiness}")
    
    asyncio.run(main())
