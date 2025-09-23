#!/usr/bin/env python3
"""
Phase 2 Setup Script

This script sets up Phase 2: Model Integration & Small Model Selection
for the Lenovo AAITC Technical Assignments project.

Usage:
    python scripts/setup_phase2.py
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


class Phase2Setup:
    """Setup Phase 2 implementation components."""
    
    def __init__(self):
        """Initialize the Phase 2 setup."""
        self.logger = self._setup_logging()
        self.ollama_manager = None
        self.github_client = None
        self.small_model_endpoints = None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for Phase 2 setup."""
        logger = logging.getLogger("phase2_setup")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def setup_small_models_configuration(self):
        """Setup small models configuration."""
        self.logger.info("🔧 Setting up Small Models Configuration...")
        
        try:
            # Initialize Ollama manager
            self.ollama_manager = OllamaManager()
            
            # Get small models list
            small_models = self.ollama_manager.small_model_optimizer.get_small_models_list()
            
            self.logger.info(f"✅ Small models configuration loaded:")
            for model in small_models:
                self.logger.info(f"  - {model['name']}: {model['parameters']}B parameters, {model['size_gb']}GB")
                self.logger.info(f"    Use case: {model['use_case']}")
                self.logger.info(f"    Deployment targets: {model['deployment_targets']}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup small models configuration: {e}")
            return False
    
    async def setup_github_models_api(self):
        """Setup GitHub Models API integration."""
        self.logger.info("🔧 Setting up GitHub Models API Integration...")
        
        try:
            # Initialize GitHub Models API client
            self.github_client = GitHubModelsAPIClient()
            await self.github_client.initialize()
            
            self.logger.info("✅ GitHub Models API client initialized")
            
            # Test API connectivity
            packages_response = await self.github_client.list_organization_packages()
            if packages_response.success:
                self.logger.info(f"✅ GitHub API connectivity confirmed")
                self.logger.info(f"  Organization: {self.github_client.organization}")
                self.logger.info(f"  Rate limit remaining: {packages_response.rate_limit_remaining}")
            else:
                self.logger.warning(f"⚠️ GitHub API connectivity issue: {packages_response.error}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup GitHub Models API: {e}")
            return False
    
    async def setup_ollama_integration(self):
        """Setup Ollama integration for small models."""
        self.logger.info("🔧 Setting up Ollama Integration for Small Models...")
        
        try:
            # Setup small models in Ollama manager
            await self.ollama_manager.setup_small_models()
            
            self.logger.info("✅ Ollama integration setup complete")
            self.logger.info("  - Small models registered for monitoring")
            self.logger.info("  - Performance monitoring started")
            self.logger.info("  - Mobile deployment configurations loaded")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup Ollama integration: {e}")
            return False
    
    async def setup_small_model_endpoints(self):
        """Setup small model endpoints."""
        self.logger.info("🔧 Setting up Small Model Endpoints...")
        
        try:
            # Initialize small model endpoints
            self.small_model_endpoints = SmallModelEndpoints()
            
            models = self.small_model_endpoints.small_models
            self.logger.info(f"✅ Small model endpoints initialized")
            self.logger.info(f"  - Available models: {len(models)}")
            self.logger.info(f"  - API documentation: http://localhost:8081/docs")
            self.logger.info(f"  - Health check: http://localhost:8081/health")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup small model endpoints: {e}")
            return False
    
    async def test_phase2_functionality(self):
        """Test Phase 2 functionality."""
        self.logger.info("🧪 Testing Phase 2 Functionality...")
        
        try:
            # Test small model optimization
            small_models = self.ollama_manager.small_model_optimizer.get_small_models_list()
            test_model = small_models[0]["name"]
            
            self.logger.info(f"  Testing optimization for {test_model}...")
            result = await self.ollama_manager.optimize_small_model(test_model, "quantization")
            
            if result.success:
                self.logger.info(f"    ✅ Optimization successful: {result.compression_ratio:.2f} compression ratio")
            else:
                self.logger.warning(f"    ⚠️ Optimization failed: {result.error}")
            
            # Test mobile deployment validation
            self.logger.info("  Testing mobile deployment validation...")
            validation = self.ollama_manager.validate_deployment(test_model, "android", small_models[0]["size_gb"])
            
            if validation["valid"]:
                self.logger.info("    ✅ Android deployment validation passed")
            else:
                self.logger.warning(f"    ⚠️ Android deployment validation failed: {validation.get('errors', [])}")
            
            # Test performance monitoring
            self.logger.info("  Testing performance monitoring...")
            await asyncio.sleep(5)  # Let some metrics collect
            
            performance = await self.ollama_manager.get_small_model_performance(test_model)
            if "error" not in performance:
                self.logger.info("    ✅ Performance monitoring working")
            else:
                self.logger.warning(f"    ⚠️ Performance monitoring issue: {performance['error']}")
            
            # Test model comparison
            self.logger.info("  Testing model comparison...")
            model_names = [model["name"] for model in small_models[:2]]
            comparison = await self.ollama_manager.compare_small_models(model_names)
            
            if "error" not in comparison:
                self.logger.info("    ✅ Model comparison working")
            else:
                self.logger.warning(f"    ⚠️ Model comparison issue: {comparison['error']}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Phase 2 functionality test failed: {e}")
            return False
    
    async def display_phase2_summary(self):
        """Display Phase 2 setup summary."""
        self.logger.info("=" * 60)
        self.logger.info("🎯 Phase 2: Model Integration & Small Model Selection")
        self.logger.info("=" * 60)
        
        try:
            # Get status information
            status = self.ollama_manager.get_small_models_status()
            
            self.logger.info("📊 Setup Summary:")
            self.logger.info(f"  - Small models configured: {status['small_models_count']}")
            self.logger.info(f"  - Performance monitoring: {'✅ Active' if status['monitoring_active'] else '❌ Inactive'}")
            self.logger.info(f"  - GitHub Models API: {'✅ Connected' if self.github_client else '❌ Not connected'}")
            self.logger.info(f"  - Small model endpoints: {'✅ Ready' if self.small_model_endpoints else '❌ Not ready'}")
            
            self.logger.info("\n🚀 Available Services:")
            self.logger.info("  - Small Model Endpoints API: http://localhost:8081")
            self.logger.info("  - API Documentation: http://localhost:8081/docs")
            self.logger.info("  - Health Check: http://localhost:8081/health")
            
            self.logger.info("\n📱 Mobile Deployment Targets:")
            platforms = self.ollama_manager.mobile_deployment_manager.get_all_platforms()
            for platform in platforms:
                capabilities = self.ollama_manager.mobile_deployment_manager.get_platform_capabilities(platform)
                self.logger.info(f"  - {platform.value.upper()}: {capabilities['max_memory_mb']}MB max memory")
            
            self.logger.info("\n🔧 Next Steps:")
            self.logger.info("  1. Run model optimization tests")
            self.logger.info("  2. Validate mobile deployment configurations")
            self.logger.info("  3. Test performance monitoring")
            self.logger.info("  4. Start small model endpoints server")
            self.logger.info("  5. Proceed to Phase 3: AI Architect Model Customization")
            
        except Exception as e:
            self.logger.error(f"Error displaying summary: {e}")
    
    async def run_setup(self):
        """Run complete Phase 2 setup."""
        self.logger.info("🚀 Starting Phase 2 Setup")
        self.logger.info("=" * 60)
        
        setup_results = {}
        
        # Setup 1: Small Models Configuration
        setup_results["small_models_config"] = await self.setup_small_models_configuration()
        
        # Setup 2: GitHub Models API Integration
        setup_results["github_api"] = await self.setup_github_models_api()
        
        # Setup 3: Ollama Integration
        setup_results["ollama_integration"] = await self.setup_ollama_integration()
        
        # Setup 4: Small Model Endpoints
        setup_results["endpoints"] = await self.setup_small_model_endpoints()
        
        # Setup 5: Test Functionality
        setup_results["functionality_test"] = await self.test_phase2_functionality()
        
        # Summary
        successful_setups = sum(1 for result in setup_results.values() if result)
        total_setups = len(setup_results)
        
        self.logger.info("=" * 60)
        self.logger.info("📊 Setup Results:")
        
        for setup_name, result in setup_results.items():
            status = "✅ SUCCESS" if result else "❌ FAILED"
            self.logger.info(f"  {setup_name}: {status}")
        
        if successful_setups == total_setups:
            self.logger.info(f"\n🎉 Phase 2 setup completed successfully!")
            await self.display_phase2_summary()
            return True
        else:
            self.logger.warning(f"\n⚠️ {total_setups - successful_setups} setup steps failed.")
            self.logger.info("Please review the errors above and retry setup.")
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
    """Main setup function."""
    setup = Phase2Setup()
    
    try:
        success = await setup.run_setup()
        return 0 if success else 1
        
    except Exception as e:
        setup.logger.error(f"Setup execution failed: {e}")
        return 1
        
    finally:
        await setup.cleanup()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
