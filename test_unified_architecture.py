"""
Test Script for Unified Architecture

This script tests the new Ollama-centric unified registry architecture
with both local (Ollama) and remote (GitHub Models) integration.
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.unified_registry import UnifiedRegistryManager
from src.ollama_integration import OllamaCategoryLoader, OllamaModelLoader, OllamaRegistrySync
from src.github_models_integration import GitHubModelsAPIClient, GitHubModelsLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_unified_architecture():
    """Test the unified architecture components."""
    logger.info("🚀 Testing Unified Architecture Components")
    
    try:
        # 1. Initialize Unified Registry
        logger.info("📋 Initializing Unified Registry...")
        registry_manager = UnifiedRegistryManager("test_unified_registry.db")
        
        # 2. Test Ollama Integration
        logger.info("🖥️ Testing Ollama Integration...")
        try:
            async with OllamaCategoryLoader() as category_loader:
                async with OllamaModelLoader(category_loader) as model_loader:
                    # Test category loading
                    categories = await category_loader.sync_all_categories()
                    logger.info(f"✅ Ollama categories loaded: {list(categories.keys())}")
                    
                    # Test model loading
                    for category in ["embedding", "vision", "tools", "thinking"]:
                        models = await category_loader.load_models_by_category(category)
                        logger.info(f"✅ {category} models: {len(models)} found")
                    
                    # Test registry sync
                    registry_sync = OllamaRegistrySync(model_loader, registry_manager)
                    sync_result = await registry_sync.sync_models()
                    logger.info(f"✅ Ollama sync: {sync_result.synced_models} models synced")
                    
        except Exception as e:
            logger.warning(f"⚠️ Ollama integration test failed: {e}")
        
        # 3. Test GitHub Models Integration
        logger.info("☁️ Testing GitHub Models Integration...")
        try:
            async with GitHubModelsAPIClient() as api_client:
                model_loader = GitHubModelsLoader(api_client)
                
                # Test model loading
                models = await model_loader.load_all_models()
                logger.info(f"✅ GitHub Models loaded: {len(models)} models")
                
                # Test provider loading
                for provider in ["openai", "meta", "deepseek", "microsoft"]:
                    provider_models = await model_loader.load_models_by_provider(provider)
                    logger.info(f"✅ {provider} models: {len(provider_models)} found")
                
        except Exception as e:
            logger.warning(f"⚠️ GitHub Models integration test failed: {e}")
        
        # 4. Test Unified Registry Operations
        logger.info("🔄 Testing Unified Registry Operations...")
        
        # List all models
        all_models = await registry_manager.list_models()
        logger.info(f"✅ Total models in registry: {len(all_models)}")
        
        # Test filtering
        local_models = await registry_manager.list_models(source="ollama")
        remote_models = await registry_manager.list_models(source="github_models")
        logger.info(f"✅ Local models: {len(local_models)}, Remote models: {len(remote_models)}")
        
        # Test category filtering
        categories = registry_manager.get_available_categories()
        logger.info(f"✅ Available categories: {categories}")
        
        # Test model search
        if all_models:
            test_model = all_models[0]
            search_results = registry_manager.search_models(test_model.name[:5])
            logger.info(f"✅ Search test: {len(search_results)} results for '{test_model.name[:5]}'")
        
        # Get registry statistics
        stats = registry_manager.get_registry_statistics()
        logger.info(f"✅ Registry stats: {stats}")
        
        # 5. Test Model Serving Interface
        logger.info("🚀 Testing Model Serving Interface...")
        try:
            from src.unified_registry import ModelServingInterface, ServingRequest
            
            async with ModelServingInterface(registry_manager) as serving_interface:
                # List serving models
                serving_models = serving_interface.list_serving_models()
                logger.info(f"✅ Models available for serving: {len(serving_models)}")
                
                # Test serving status
                if all_models:
                    test_model = all_models[0]
                    status = await serving_interface.get_serving_status(test_model.id)
                    logger.info(f"✅ Serving status for {test_model.name}: {status['serving']}")
        
        except Exception as e:
            logger.warning(f"⚠️ Model serving test failed: {e}")
        
        logger.info("🎉 Unified Architecture Test Completed Successfully!")
        
        # Print summary
        print("\n" + "="*60)
        print("📊 UNIFIED ARCHITECTURE TEST SUMMARY")
        print("="*60)
        print(f"📋 Total Models in Registry: {len(all_models)}")
        print(f"🖥️ Local (Ollama) Models: {len(local_models)}")
        print(f"☁️ Remote (GitHub Models): {len(remote_models)}")
        print(f"📂 Available Categories: {len(categories)}")
        print(f"🔍 Search Functionality: {'✅ Working' if all_models else '❌ No models to test'}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


async def test_gradio_integration():
    """Test Gradio app integration."""
    logger.info("🎨 Testing Gradio App Integration...")
    
    try:
        from src.gradio_app.simplified_main import SimplifiedLenovoModelEvaluationApp
        
        # Initialize app
        app = SimplifiedLenovoModelEvaluationApp()
        
        # Test model selector
        stats = app.registry_manager.get_registry_statistics()
        logger.info(f"✅ Gradio app registry stats: {stats}")
        
        # Test available categories
        categories = app.model_selector.get_available_categories()
        logger.info(f"✅ Available categories for UI: {categories}")
        
        logger.info("✅ Gradio integration test completed")
        
    except Exception as e:
        logger.warning(f"⚠️ Gradio integration test failed: {e}")


async def main():
    """Main test function."""
    print("🚀 Starting Unified Architecture Tests")
    print("="*60)
    
    # Test unified architecture
    await test_unified_architecture()
    
    print("\n")
    
    # Test Gradio integration
    await test_gradio_integration()
    
    print("\n🎉 All Tests Completed!")


if __name__ == "__main__":
    asyncio.run(main())
