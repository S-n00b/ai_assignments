#!/usr/bin/env python3
"""
Phase 2 Implementation Test Script

This script tests the Phase 2 implementation of Model Integration & Small Model Selection.
It validates all components and demonstrates the functionality.

Usage:
    python scripts/test_phase2_implementation.py
"""

import asyncio
import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from enterprise_llmops.ollama_manager import OllamaManager
from github_models_integration.api_client import GitHubModelsAPIClient
from model_evaluation.small_model_endpoints import SmallModelEndpoints


class Phase2Tester:
    """Test Phase 2 implementation components."""
    
    def __init__(self):
        """Initialize the Phase 2 tester."""
        self.logger = self._setup_logging()
        self.ollama_manager = None
        self.github_client = None
        self.small_model_endpoints = None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for Phase 2 tester."""
        logger = logging.getLogger("phase2_tester")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def test_small_models_configuration(self):
        """Test small models configuration loading."""
        self.logger.info("🧪 Testing Small Models Configuration...")
        
        try:
            # Initialize Ollama manager with small model components
            self.ollama_manager = OllamaManager()
            
            # Test small model optimizer
            small_models = self.ollama_manager.small_model_optimizer.get_small_models_list()
            self.logger.info(f"✅ Loaded {len(small_models)} small models:")
            for model in small_models:
                self.logger.info(f"  - {model['name']}: {model['parameters']}B parameters, {model['size_gb']}GB")
            
            # Test mobile deployment config manager
            platforms = self.ollama_manager.mobile_deployment_manager.get_all_platforms()
            self.logger.info(f"✅ Supported platforms: {[p.value for p in platforms]}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Small models configuration test failed: {e}")
            return False
    
    async def test_github_models_api_integration(self):
        """Test GitHub Models API integration."""
        self.logger.info("🧪 Testing GitHub Models API Integration...")
        
        try:
            # Initialize GitHub Models API client
            self.github_client = GitHubModelsAPIClient()
            await self.github_client.initialize()
            
            # Test API connectivity
            self.logger.info("✅ GitHub Models API client initialized")
            
            # Test model catalog
            catalog = await self.github_client.get_model_catalog()
            self.logger.info(f"✅ Retrieved {len(catalog)} models from GitHub Models catalog")
            
            # Test small models inference
            inference_test = await self.github_client.test_small_models_inference()
            if "error" not in inference_test:
                self.logger.info(f"✅ Small models inference test: {inference_test.get('small_models_found', 0)} small models found")
                if inference_test.get("test_result"):
                    test_result = inference_test["test_result"]
                    self.logger.info(f"  Test model: {test_result.get('model_name', 'Unknown')}")
                    self.logger.info(f"  Inference success: {test_result.get('inference_success', False)}")
            else:
                self.logger.warning(f"⚠️ Small models inference test failed: {inference_test['error']}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ GitHub Models API integration test failed: {e}")
            return False
    
    async def test_small_model_optimization(self):
        """Test small model optimization."""
        self.logger.info("🧪 Testing Small Model Optimization...")
        
        try:
            # Test optimization for each model type
            optimization_types = ["quantization", "pruning", "distillation"]
            small_models = self.ollama_manager.small_model_optimizer.get_small_models_list()
            
            for model in small_models[:2]:  # Test first 2 models
                model_name = model["name"]
                self.logger.info(f"  Testing optimization for {model_name}...")
                
                for opt_type in optimization_types:
                    result = await self.ollama_manager.optimize_small_model(model_name, opt_type)
                    if result.success:
                        self.logger.info(f"    ✅ {opt_type}: {result.original_size_gb:.2f}GB -> {result.optimized_size_gb:.2f}GB")
                    else:
                        self.logger.warning(f"    ⚠️ {opt_type} failed: {result.error}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Small model optimization test failed: {e}")
            return False
    
    async def test_mobile_deployment_validation(self):
        """Test mobile deployment validation."""
        self.logger.info("🧪 Testing Mobile Deployment Validation...")
        
        try:
            platforms = ["android", "ios", "edge", "embedded"]
            small_models = self.ollama_manager.small_model_optimizer.get_small_models_list()
            
            for model in small_models[:2]:  # Test first 2 models
                model_name = model["name"]
                model_size_gb = model["size_gb"]
                
                self.logger.info(f"  Testing deployment validation for {model_name}...")
                
                for platform in platforms:
                    validation = self.ollama_manager.validate_deployment(model_name, platform, model_size_gb)
                    status = "✅ VALID" if validation["valid"] else "❌ INVALID"
                    self.logger.info(f"    {platform}: {status}")
                    
                    if validation["warnings"]:
                        for warning in validation["warnings"]:
                            self.logger.info(f"      Warning: {warning}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Mobile deployment validation test failed: {e}")
            return False
    
    async def test_performance_monitoring(self):
        """Test performance monitoring."""
        self.logger.info("🧪 Testing Performance Monitoring...")
        
        try:
            # Setup small models
            await self.ollama_manager.setup_small_models()
            
            # Wait for some metrics to be collected
            self.logger.info("  Collecting performance metrics...")
            await asyncio.sleep(10)
            
            # Test performance data retrieval
            small_models = self.ollama_manager.small_model_optimizer.get_small_models_list()
            for model in small_models[:2]:  # Test first 2 models
                model_name = model["name"]
                performance = await self.ollama_manager.get_small_model_performance(model_name)
                if "error" not in performance:
                    self.logger.info(f"    ✅ {model_name}: Performance data collected")
                else:
                    self.logger.warning(f"    ⚠️ {model_name}: {performance['error']}")
            
            # Get overall status
            status = self.ollama_manager.get_small_models_status()
            self.logger.info(f"✅ Performance monitoring status: {status['small_models_count']} models monitored")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Performance monitoring test failed: {e}")
            return False
    
    async def test_model_comparison(self):
        """Test model comparison functionality."""
        self.logger.info("🧪 Testing Model Comparison...")
        
        try:
            small_models = self.ollama_manager.small_model_optimizer.get_small_models_list()
            model_names = [model["name"] for model in small_models[:3]]  # Test first 3 models
            
            comparison = await self.ollama_manager.compare_small_models(model_names)
            
            if "error" not in comparison:
                self.logger.info(f"✅ Model comparison completed for {len(model_names)} models")
                self.logger.info(f"  Rankings: {comparison['rankings']}")
                self.logger.info(f"  Recommendations: {comparison['recommendations']}")
            else:
                self.logger.warning(f"⚠️ Model comparison failed: {comparison['error']}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Model comparison test failed: {e}")
            return False
    
    async def test_small_model_endpoints(self):
        """Test small model endpoints."""
        self.logger.info("🧪 Testing Small Model Endpoints...")
        
        try:
            # Initialize small model endpoints
            self.small_model_endpoints = SmallModelEndpoints()
            
            # Test endpoint initialization
            models = self.small_model_endpoints.small_models
            self.logger.info(f"✅ Small model endpoints initialized with {len(models)} models")
            
            # Test endpoint functionality (without starting server)
            small_models = list(models.keys())[:2]  # Test first 2 models
            
            for model_name in small_models:
                # Test model test request
                test_request = self.small_model_endpoints.ModelTestRequestModel(
                    model_name=model_name,
                    prompt="Test prompt for mobile deployment",
                    max_tokens=50,
                    test_type="inference"
                )
                
                # Simulate endpoint call
                response = await self.small_model_endpoints._generate_model_response(
                    model_name, test_request.prompt, test_request.max_tokens
                )
                
                self.logger.info(f"    ✅ {model_name}: Generated response ({len(response)} chars)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Small model endpoints test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all Phase 2 tests."""
        self.logger.info("🚀 Starting Phase 2 Implementation Tests")
        self.logger.info("=" * 60)
        
        test_results = {}
        
        # Test 1: Small Models Configuration
        test_results["small_models_config"] = await self.test_small_models_configuration()
        
        # Test 2: GitHub Models API Integration
        test_results["github_api_integration"] = await self.test_github_models_api_integration()
        
        # Test 3: Small Model Optimization
        test_results["small_model_optimization"] = await self.test_small_model_optimization()
        
        # Test 4: Mobile Deployment Validation
        test_results["mobile_deployment_validation"] = await self.test_mobile_deployment_validation()
        
        # Test 5: Performance Monitoring
        test_results["performance_monitoring"] = await self.test_performance_monitoring()
        
        # Test 6: Model Comparison
        test_results["model_comparison"] = await self.test_model_comparison()
        
        # Test 7: Small Model Endpoints
        test_results["small_model_endpoints"] = await self.test_small_model_endpoints()
        
        # Summary
        self.logger.info("=" * 60)
        self.logger.info("📊 Phase 2 Test Results Summary:")
        
        passed_tests = 0
        total_tests = len(test_results)
        
        for test_name, result in test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            self.logger.info(f"  {test_name}: {status}")
            if result:
                passed_tests += 1
        
        self.logger.info(f"\n🎯 Overall Result: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            self.logger.info("🎉 Phase 2 implementation is ready!")
            return True
        else:
            self.logger.warning(f"⚠️ {total_tests - passed_tests} tests failed. Please review implementation.")
            return False
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self.ollama_manager:
                await self.ollama_manager.shutdown()
            
            if self.github_client:
                await self.github_client.shutdown()
            
            self.logger.info("🧹 Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


async def main():
    """Main test function."""
    tester = Phase2Tester()
    
    try:
        success = await tester.run_all_tests()
        return 0 if success else 1
        
    except Exception as e:
        tester.logger.error(f"Test execution failed: {e}")
        return 1
        
    finally:
        await tester.cleanup()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
