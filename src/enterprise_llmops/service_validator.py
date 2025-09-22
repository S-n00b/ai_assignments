"""
Enterprise LLMOps Service Validator

This module provides comprehensive validation and connection testing for all
enterprise services including Ollama, MLflow, vector databases, and monitoring
stack.

Key Features:
- Service health checking and validation
- Connection testing for all enterprise components
- Integration testing for end-to-end workflows
- Performance benchmarking and monitoring
- Automated service startup and configuration
"""

import asyncio
import aiohttp
import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import requests
from pathlib import Path
import yaml
import subprocess
import psutil
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.enterprise_llmops.ollama_manager import OllamaManager
from src.enterprise_llmops.mlops.mlflow_manager import MLflowManager, ExperimentConfig
from src.enterprise_llmops.model_registry import EnterpriseModelRegistry


@dataclass
class ServiceConfig:
    """Configuration for a service."""
    name: str
    url: str
    health_endpoint: str
    timeout: int = 10
    required: bool = True
    start_command: Optional[str] = None
    docker_image: Optional[str] = None
    port: Optional[int] = None


@dataclass
class ValidationResult:
    """Result of service validation."""
    service_name: str
    status: str  # "healthy", "unhealthy", "unreachable"
    response_time: float
    error_message: Optional[str] = None
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class IntegrationTestResult:
    """Result of integration test."""
    test_name: str
    status: str  # "passed", "failed", "skipped"
    duration: float
    error_message: Optional[str] = None
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class ServiceValidator:
    """
    Comprehensive service validator for Enterprise LLMOps platform.
    
    This class provides validation, connection testing, and integration
    testing for all enterprise services.
    """
    
    def __init__(self, config_path: str = "config/enterprise-config.yaml"):
        """Initialize the service validator."""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.services = self._setup_services()
        self.session = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load enterprise configuration."""
        default_config = {
            "services": {
                "ollama": {
                    "url": "http://localhost:11434",
                    "health_endpoint": "/api/tags",
                    "required": True,
                    "timeout": 10
                },
                "mlflow": {
                    "url": "http://localhost:5000",
                    "health_endpoint": "/health",
                    "required": True,
                    "timeout": 10
                },
                "chroma": {
                    "url": "http://localhost:8081",
                    "health_endpoint": "/api/v1/heartbeat",
                    "required": True,
                    "timeout": 10
                },
                "weaviate": {
                    "url": "http://localhost:8080",
                    "health_endpoint": "/v1/meta",
                    "required": False,
                    "timeout": 10
                },
                "prometheus": {
                    "url": "http://localhost:9090",
                    "health_endpoint": "/-/healthy",
                    "required": False,
                    "timeout": 10
                },
                "grafana": {
                    "url": "http://localhost:3000",
                    "health_endpoint": "/api/health",
                    "required": False,
                    "timeout": 10
                },
                "gradio": {
                    "url": "http://localhost:7860",
                    "health_endpoint": "/",
                    "required": False,
                    "timeout": 10
                },
                "fastapi": {
                    "url": "http://localhost:8080",
                    "health_endpoint": "/docs",
                    "required": False,
                    "timeout": 10
                }
            },
            "integration_tests": {
                "ollama_model_listing": True,
                "mlflow_experiment_creation": True,
                "chroma_collection_creation": True,
                "model_evaluation_pipeline": True,
                "end_to_end_workflow": True
            }
        }
        
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                default_config.update(file_config)
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for service validator."""
        logger = logging.getLogger("service_validator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _setup_services(self) -> Dict[str, ServiceConfig]:
        """Setup service configurations."""
        services = {}
        for name, config in self.config["services"].items():
            services[name] = ServiceConfig(
                name=name,
                url=config["url"],
                health_endpoint=config["health_endpoint"],
                timeout=config.get("timeout", 10),
                required=config.get("required", True)
            )
        return services
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def validate_service(self, service_name: str) -> ValidationResult:
        """Validate a single service."""
        if service_name not in self.services:
            return ValidationResult(
                service_name=service_name,
                status="unreachable",
                response_time=0.0,
                error_message=f"Service {service_name} not configured"
            )
        
        service = self.services[service_name]
        start_time = time.time()
        
        try:
            session = await self._get_session()
            url = f"{service.url}{service.health_endpoint}"
            
            async with session.get(url, timeout=service.timeout) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    # Try to get response data
                    try:
                        data = await response.json()
                        details = {"status_code": response.status, "data": data}
                    except:
                        details = {"status_code": response.status, "data": None}
                    
                    return ValidationResult(
                        service_name=service_name,
                        status="healthy",
                        response_time=response_time,
                        details=details
                    )
                else:
                    return ValidationResult(
                        service_name=service_name,
                        status="unhealthy",
                        response_time=response_time,
                        error_message=f"HTTP {response.status}",
                        details={"status_code": response.status}
                    )
        
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            return ValidationResult(
                service_name=service_name,
                status="unreachable",
                response_time=response_time,
                error_message="Connection timeout"
            )
        except Exception as e:
            response_time = time.time() - start_time
            return ValidationResult(
                service_name=service_name,
                status="unreachable",
                response_time=response_time,
                error_message=str(e)
            )
    
    async def validate_all_services(self) -> List[ValidationResult]:
        """Validate all configured services."""
        self.logger.info("Validating all services...")
        
        tasks = []
        for service_name in self.services.keys():
            tasks.append(self.validate_service(service_name))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        validation_results = []
        for result in results:
            if isinstance(result, Exception):
                validation_results.append(ValidationResult(
                    service_name="unknown",
                    status="unreachable",
                    response_time=0.0,
                    error_message=str(result)
                ))
            else:
                validation_results.append(result)
        
        return validation_results
    
    def print_validation_results(self, results: List[ValidationResult]):
        """Print validation results in a formatted way."""
        print("\n" + "="*80)
        print("ENTERPRISE LLMOPS SERVICE VALIDATION RESULTS")
        print("="*80)
        
        healthy_services = []
        unhealthy_services = []
        unreachable_services = []
        
        for result in results:
            if result.status == "healthy":
                healthy_services.append(result)
            elif result.status == "unhealthy":
                unhealthy_services.append(result)
            else:
                unreachable_services.append(result)
        
        # Print healthy services
        if healthy_services:
            print(f"\n✅ HEALTHY SERVICES ({len(healthy_services)}):")
            for result in healthy_services:
                print(f"   {result.service_name:<15} - {result.response_time:.3f}s")
        
        # Print unhealthy services
        if unhealthy_services:
            print(f"\n⚠️  UNHEALTHY SERVICES ({len(unhealthy_services)}):")
            for result in unhealthy_services:
                print(f"   {result.service_name:<15} - {result.error_message}")
        
        # Print unreachable services
        if unreachable_services:
            print(f"\n❌ UNREACHABLE SERVICES ({len(unreachable_services)}):")
            for result in unreachable_services:
                print(f"   {result.service_name:<15} - {result.error_message}")
        
        print("\n" + "="*80)
        
        # Summary
        total_services = len(results)
        required_services = [s for s in self.services.values() if s.required]
        failed_required = [r for r in unreachable_services + unhealthy_services 
                          if self.services[r.service_name].required]
        
        print(f"SUMMARY: {len(healthy_services)}/{total_services} services healthy")
        if failed_required:
            print(f"CRITICAL: {len(failed_required)} required services failed")
        else:
            print("✅ All required services are operational")
        
        return len(failed_required) == 0
    
    async def test_ollama_integration(self) -> IntegrationTestResult:
        """Test Ollama integration."""
        start_time = time.time()
        
        try:
            # Test Ollama manager
            ollama_manager = OllamaManager()
            await ollama_manager.initialize()
            
            # List available models
            models = await ollama_manager.list_models()
            
            duration = time.time() - start_time
            
            return IntegrationTestResult(
                test_name="ollama_integration",
                status="passed",
                duration=duration,
                details={
                    "models_count": len(models),
                    "models": [model.name for model in models]
                }
            )
        
        except Exception as e:
            duration = time.time() - start_time
            return IntegrationTestResult(
                test_name="ollama_integration",
                status="failed",
                duration=duration,
                error_message=str(e)
            )
    
    async def test_mlflow_integration(self) -> IntegrationTestResult:
        """Test MLflow integration."""
        start_time = time.time()
        
        try:
            # Test MLflow manager
            mlflow_config = ExperimentConfig(
                experiment_name="integration_test",
                tracking_uri=self.config["services"]["mlflow"]["url"],
                description="Integration test experiment"
            )
            
            mlflow_manager = MLflowManager(mlflow_config)
            
            # Create a test experiment
            with mlflow_manager.start_run("integration_test") as run:
                mlflow_manager.log_metric("test_metric", 1.0)
                mlflow_manager.log_param("test_param", "integration_test")
            
            duration = time.time() - start_time
            
            return IntegrationTestResult(
                test_name="mlflow_integration",
                status="passed",
                duration=duration,
                details={"experiment_id": mlflow_manager.experiment_id}
            )
        
        except Exception as e:
            duration = time.time() - start_time
            return IntegrationTestResult(
                test_name="mlflow_integration",
                status="failed",
                duration=duration,
                error_message=str(e)
            )
    
    async def test_chroma_integration(self) -> IntegrationTestResult:
        """Test Chroma integration."""
        start_time = time.time()
        
        try:
            import chromadb
            
            # Connect to Chroma
            client = chromadb.HttpClient(host="localhost", port=8081)
            
            # Create a test collection
            collection_name = f"test_collection_{int(time.time())}"
            collection = client.create_collection(name=collection_name)
            
            # Add test documents
            collection.add(
                documents=["This is a test document"],
                metadatas=[{"source": "integration_test"}],
                ids=["test_doc_1"]
            )
            
            # Query the collection
            results = collection.query(
                query_texts=["test document"],
                n_results=1
            )
            
            # Clean up
            client.delete_collection(name=collection_name)
            
            duration = time.time() - start_time
            
            return IntegrationTestResult(
                test_name="chroma_integration",
                status="passed",
                duration=duration,
                details={"results_count": len(results["documents"][0])}
            )
        
        except Exception as e:
            duration = time.time() - start_time
            return IntegrationTestResult(
                test_name="chroma_integration",
                status="failed",
                duration=duration,
                error_message=str(e)
            )
    
    async def test_model_evaluation_pipeline(self) -> IntegrationTestResult:
        """Test model evaluation pipeline."""
        start_time = time.time()
        
        try:
            # Import model evaluation components
            from src.model_evaluation.factory import ModelFactory
            from src.model_evaluation.profiler import ModelProfiler
            
            # Test ModelFactory
            factory = ModelFactory()
            models = factory.get_available_models()
            
            # Test ModelProfiler
            profiler = ModelProfiler()
            
            duration = time.time() - start_time
            
            return IntegrationTestResult(
                test_name="model_evaluation_pipeline",
                status="passed",
                duration=duration,
                details={"available_models": len(models)}
            )
        
        except Exception as e:
            duration = time.time() - start_time
            return IntegrationTestResult(
                test_name="model_evaluation_pipeline",
                status="failed",
                duration=duration,
                error_message=str(e)
            )
    
    async def run_all_integration_tests(self) -> List[IntegrationTestResult]:
        """Run all integration tests."""
        self.logger.info("Running integration tests...")
        
        tests = []
        
        # Only run tests if the required services are available
        validation_results = await self.validate_all_services()
        healthy_services = [r.service_name for r in validation_results if r.status == "healthy"]
        
        if "ollama" in healthy_services:
            tests.append(self.test_ollama_integration())
        
        if "mlflow" in healthy_services:
            tests.append(self.test_mlflow_integration())
        
        if "chroma" in healthy_services:
            tests.append(self.test_chroma_integration())
        
        # Always test model evaluation pipeline (it's local)
        tests.append(self.test_model_evaluation_pipeline())
        
        results = await asyncio.gather(*tests, return_exceptions=True)
        
        # Process results
        integration_results = []
        for result in results:
            if isinstance(result, Exception):
                integration_results.append(IntegrationTestResult(
                    test_name="unknown",
                    status="failed",
                    duration=0.0,
                    error_message=str(result)
                ))
            else:
                integration_results.append(result)
        
        return integration_results
    
    def print_integration_results(self, results: List[IntegrationTestResult]):
        """Print integration test results."""
        print("\n" + "="*80)
        print("ENTERPRISE LLMOPS INTEGRATION TEST RESULTS")
        print("="*80)
        
        passed_tests = [r for r in results if r.status == "passed"]
        failed_tests = [r for r in results if r.status == "failed"]
        skipped_tests = [r for r in results if r.status == "skipped"]
        
        # Print passed tests
        if passed_tests:
            print(f"\n✅ PASSED TESTS ({len(passed_tests)}):")
            for result in passed_tests:
                print(f"   {result.test_name:<25} - {result.duration:.3f}s")
                if result.details:
                    for key, value in result.details.items():
                        print(f"      {key}: {value}")
        
        # Print failed tests
        if failed_tests:
            print(f"\n❌ FAILED TESTS ({len(failed_tests)}):")
            for result in failed_tests:
                print(f"   {result.test_name:<25} - {result.error_message}")
        
        # Print skipped tests
        if skipped_tests:
            print(f"\n⏭️  SKIPPED TESTS ({len(skipped_tests)}):")
            for result in skipped_tests:
                print(f"   {result.test_name:<25} - {result.error_message}")
        
        print("\n" + "="*80)
        print(f"INTEGRATION SUMMARY: {len(passed_tests)}/{len(results)} tests passed")
        
        return len(failed_tests) == 0
    
    async def run_full_validation(self) -> bool:
        """Run full validation including service checks and integration tests."""
        print("🚀 Starting Enterprise LLMOps Service Validation")
        print("="*80)
        
        # Validate services
        validation_results = await self.validate_all_services()
        services_healthy = self.print_validation_results(validation_results)
        
        if not services_healthy:
            print("\n⚠️  Some required services are not healthy. Integration tests may be limited.")
        
        # Run integration tests
        integration_results = await self.run_all_integration_tests()
        tests_passed = self.print_integration_results(integration_results)
        
        # Final summary
        print("\n" + "="*80)
        print("FINAL VALIDATION SUMMARY")
        print("="*80)
        
        if services_healthy and tests_passed:
            print("✅ ALL VALIDATIONS PASSED - Enterprise platform is ready!")
            return True
        else:
            print("❌ SOME VALIDATIONS FAILED - Check the details above")
            return False
    
    async def close(self):
        """Close the validator and cleanup resources."""
        if self.session:
            await self.session.close()


async def main():
    """Main function for running service validation."""
    validator = ServiceValidator()
    
    try:
        success = await validator.run_full_validation()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        sys.exit(1)
    finally:
        await validator.close()


if __name__ == "__main__":
    asyncio.run(main())
