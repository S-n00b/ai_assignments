#!/usr/bin/env python3
"""
GitHub Models Setup Script

This script sets up GitHub Models API integration for Phase 2
using the official GitHub Models inference API.

Usage:
    python scripts/setup_github_models.py
"""

import asyncio
import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from github_models_integration.api_client import GitHubModelsAPIClient
from github_models_integration.inference_client import GitHubModelsInferenceClient


class GitHubModelsSetup:
    """Setup GitHub Models API integration."""
    
    def __init__(self):
        """Initialize the GitHub Models setup."""
        self.logger = self._setup_logging()
        self.api_client = None
        self.inference_client = None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for GitHub Models setup."""
        logger = logging.getLogger("github_models_setup")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def check_environment(self):
        """Check environment setup for GitHub Models API."""
        self.logger.info("🔧 Checking Environment Setup...")
        
        # Check for GitHub token
        token = os.getenv("GITHUB_MODELS_TOKEN")
        if not token:
            self.logger.error("❌ GITHUB_MODELS_TOKEN environment variable not set")
            self.logger.info("Please set your GitHub Models token:")
            self.logger.info("  export GITHUB_MODELS_TOKEN='your_token_here'")
            return False
        
        self.logger.info("✅ GitHub Models token found")
        
        # Check token format
        if not token.startswith("ghp_") and not token.startswith("github_pat_"):
            self.logger.warning("⚠️ Token format may be incorrect (should start with 'ghp_' or 'github_pat_')")
        
        return True
    
    async def setup_api_client(self):
        """Setup GitHub Models API client."""
        self.logger.info("🔧 Setting up GitHub Models API Client...")
        
        try:
            # Initialize API client
            self.api_client = GitHubModelsAPIClient()
            await self.api_client.initialize()
            
            self.logger.info("✅ GitHub Models API client initialized")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup API client: {e}")
            return False
    
    async def setup_inference_client(self):
        """Setup GitHub Models inference client."""
        self.logger.info("🔧 Setting up GitHub Models Inference Client...")
        
        try:
            # Initialize inference client
            self.inference_client = GitHubModelsInferenceClient(organization="Brantwood")
            await self.inference_client.initialize()
            
            self.logger.info("✅ GitHub Models inference client initialized")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup inference client: {e}")
            return False
    
    async def test_model_catalog(self):
        """Test model catalog access."""
        self.logger.info("🧪 Testing Model Catalog Access...")
        
        try:
            # Get model catalog
            catalog = await self.inference_client.get_model_catalog()
            
            if catalog:
                self.logger.info(f"✅ Retrieved {len(catalog)} models from catalog")
                
                # Show some models
                for i, model in enumerate(catalog[:5]):
                    self.logger.info(f"  {i+1}. {model.name} ({model.publisher})")
                
                if len(catalog) > 5:
                    self.logger.info(f"  ... and {len(catalog) - 5} more models")
                
                return True
            else:
                self.logger.warning("⚠️ No models found in catalog")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to access model catalog: {e}")
            return False
    
    async def test_small_models_inference(self):
        """Test inference with small models."""
        self.logger.info("🧪 Testing Small Models Inference...")
        
        try:
            # Test small models
            test_results = await self.inference_client.test_small_models()
            
            if "error" not in test_results:
                small_models_count = test_results.get("small_models_found", 0)
                self.logger.info(f"✅ Found {small_models_count} small models")
                
                # Show available small models
                available_models = test_results.get("available_models", [])
                for model in available_models:
                    self.logger.info(f"  - {model['name']} ({model['publisher']})")
                
                # Test inference if models available
                if test_results.get("test_result"):
                    test_result = test_results["test_result"]
                    self.logger.info(f"✅ Inference test successful with {test_result['model_name']}")
                    
                    if test_result.get("response_preview"):
                        self.logger.info(f"  Response preview: {test_result['response_preview'][:100]}...")
                
                return True
            else:
                self.logger.warning(f"⚠️ Small models inference test failed: {test_results['error']}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to test small models inference: {e}")
            return False
    
    async def test_phase2_evaluation(self):
        """Test Phase 2 evaluation."""
        self.logger.info("🧪 Testing Phase 2 Evaluation...")
        
        try:
            # Get available models
            catalog = await self.inference_client.get_model_catalog()
            if not catalog:
                self.logger.warning("⚠️ No models available for testing")
                return False
            
            # Find a small model for testing
            test_model = None
            for model in catalog:
                model_name_lower = model.name.lower()
                if any(indicator in model_name_lower for indicator in ["3b", "mini", "small"]):
                    test_model = model
                    break
            
            if not test_model:
                # Use first available model
                test_model = catalog[0]
            
            self.logger.info(f"  Testing with model: {test_model.name} ({test_model.publisher})")
            
            # Run Phase 2 evaluation
            evaluation = await self.inference_client.run_phase2_evaluation(
                test_model.id,
                "Explain how this model can be optimized for mobile deployment in enterprise environments."
            )
            
            if evaluation.get("success"):
                self.logger.info("✅ Phase 2 evaluation successful")
                self.logger.info(f"  Inference time: {evaluation['inference_time_ms']:.1f}ms")
                self.logger.info(f"  Response length: {len(evaluation['response']) if evaluation.get('response') else 0} characters")
                
                if evaluation.get("usage"):
                    usage = evaluation["usage"]
                    self.logger.info(f"  Tokens used: {usage.get('total_tokens', 'Unknown')}")
                
                return True
            else:
                self.logger.warning(f"⚠️ Phase 2 evaluation failed: {evaluation.get('error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to test Phase 2 evaluation: {e}")
            return False
    
    async def display_setup_summary(self):
        """Display GitHub Models setup summary."""
        self.logger.info("=" * 60)
        self.logger.info("🎯 GitHub Models API Integration Setup")
        self.logger.info("=" * 60)
        
        try:
            # Get model catalog info
            catalog = await self.inference_client.get_model_catalog()
            small_models = await self.inference_client.test_small_models()
            
            self.logger.info("📊 Setup Summary:")
            self.logger.info(f"  - Total models available: {len(catalog)}")
            self.logger.info(f"  - Small models found: {small_models.get('small_models_found', 0)}")
            self.logger.info(f"  - API version: 2022-11-28")
            self.logger.info(f"  - Organization: Brantwood")
            
            self.logger.info("\n🚀 Available Features:")
            self.logger.info("  - Model catalog access")
            self.logger.info("  - Chat completions inference")
            self.logger.info("  - Organizational attribution")
            self.logger.info("  - Streaming and non-streaming responses")
            self.logger.info("  - Small model optimization testing")
            
            self.logger.info("\n📱 Phase 2 Integration:")
            self.logger.info("  - Small model inference testing")
            self.logger.info("  - Mobile deployment evaluation")
            self.logger.info("  - Performance benchmarking")
            self.logger.info("  - Ollama integration ready")
            
            self.logger.info("\n🔧 Next Steps:")
            self.logger.info("  1. Run Phase 2 implementation tests")
            self.logger.info("  2. Test model optimization workflows")
            self.logger.info("  3. Validate mobile deployment configurations")
            self.logger.info("  4. Proceed to Phase 3: AI Architect Model Customization")
            
        except Exception as e:
            self.logger.error(f"Error displaying summary: {e}")
    
    async def run_setup(self):
        """Run complete GitHub Models setup."""
        self.logger.info("🚀 Starting GitHub Models Setup")
        self.logger.info("=" * 60)
        
        setup_results = {}
        
        # Setup 1: Environment Check
        setup_results["environment"] = self.check_environment()
        if not setup_results["environment"]:
            self.logger.error("❌ Environment setup failed. Please fix issues and retry.")
            return False
        
        # Setup 2: API Client
        setup_results["api_client"] = await self.setup_api_client()
        
        # Setup 3: Inference Client
        setup_results["inference_client"] = await self.setup_inference_client()
        
        # Test 1: Model Catalog
        setup_results["model_catalog"] = await self.test_model_catalog()
        
        # Test 2: Small Models Inference
        setup_results["small_models_inference"] = await self.test_small_models_inference()
        
        # Test 3: Phase 2 Evaluation
        setup_results["phase2_evaluation"] = await self.test_phase2_evaluation()
        
        # Summary
        successful_setups = sum(1 for result in setup_results.values() if result)
        total_setups = len(setup_results)
        
        self.logger.info("=" * 60)
        self.logger.info("📊 Setup Results:")
        
        for setup_name, result in setup_results.items():
            status = "✅ SUCCESS" if result else "❌ FAILED"
            self.logger.info(f"  {setup_name}: {status}")
        
        if successful_setups == total_setups:
            self.logger.info(f"\n🎉 GitHub Models setup completed successfully!")
            await self.display_setup_summary()
            return True
        else:
            self.logger.warning(f"\n⚠️ {total_setups - successful_setups} setup steps failed.")
            self.logger.info("Please review the errors above and retry setup.")
            return False
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self.api_client:
                await self.api_client.shutdown()
            
            if self.inference_client:
                await self.inference_client.shutdown()
            
            self.logger.info("🧹 Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


async def main():
    """Main setup function."""
    setup = GitHubModelsSetup()
    
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
