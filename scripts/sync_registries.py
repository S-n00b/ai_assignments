#!/usr/bin/env python3
"""
Comprehensive Registry Sync Script
Synchronizes all registries with actual Ollama models
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.enterprise_llmops.ollama_manager import OllamaManager
from src.enterprise_llmops.model_registry import EnterpriseModelRegistry, ModelType, ModelStatus
from src.enterprise_llmops.mlops.mlflow_manager import MLflowManager, ExperimentConfig
from src.model_evaluation.prompt_registries import PromptRegistryManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("registry_sync")


class RegistrySyncManager:
    """Manages synchronization of all registries with Ollama models."""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.ollama_manager = None
        self.model_registry = None
        self.mlflow_manager = None
        self.prompt_manager = None
        
    async def initialize(self):
        """Initialize all managers."""
        try:
            # Initialize Ollama manager
            self.ollama_manager = OllamaManager()
            await self.ollama_manager.initialize()
            logger.info("Ollama manager initialized")
            
            # Initialize model registry
            self.model_registry = EnterpriseModelRegistry()
            logger.info("Model registry initialized")
            
            # Initialize MLflow manager
            mlflow_config = ExperimentConfig(
                experiment_name="llmops_enterprise",
                tracking_uri="http://localhost:5000",
                description="Enterprise LLMOps Experiment Tracking"
            )
            self.mlflow_manager = MLflowManager(mlflow_config)
            logger.info("MLflow manager initialized")
            
            # Initialize prompt manager
            self.prompt_manager = PromptRegistryManager()
            logger.info("Prompt manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize managers: {e}")
            raise
    
    async def get_ollama_models(self) -> List[Dict[str, Any]]:
        """Get list of available Ollama models."""
        try:
            models = await self.ollama_manager.list_models()
            logger.info(f"Found {len(models)} Ollama models")
            return models
        except Exception as e:
            logger.error(f"Failed to get Ollama models: {e}")
            return []
    
    async def sync_model_registry(self, ollama_models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Sync model registry with Ollama models."""
        logger.info("Syncing model registry...")
        
        results = {
            "registered": 0,
            "updated": 0,
            "errors": 0,
            "models": []
        }
        
        for model in ollama_models:
            try:
                model_name = model.get("name", "")
                model_size = model.get("size", 0)
                model_digest = model.get("digest", "")
                
                # Determine model type based on name
                model_type = ModelType.LLM
                if "code" in model_name.lower():
                    model_type = ModelType.CODE_GENERATION
                
                # Check if model already exists
                existing_models = await self.model_registry.list_models()
                model_exists = any(m.name == model_name for m in existing_models)
                
                if not model_exists:
                    # Register new model
                    model_version = await self.model_registry.register_model(
                        model_name=model_name,
                        model_type=model_type,
                        description=f"Ollama model: {model_name}",
                        created_by="system_sync",
                        size_bytes=model_size,
                        digest=model_digest,
                        source="ollama",
                        status=ModelStatus.AVAILABLE
                    )
                    results["registered"] += 1
                    results["models"].append({
                        "name": model_name,
                        "action": "registered",
                        "version": model_version.version
                    })
                    logger.info(f"Registered model: {model_name}")
                else:
                    # Update existing model
                    results["updated"] += 1
                    results["models"].append({
                        "name": model_name,
                        "action": "updated"
                    })
                    logger.info(f"Model already exists: {model_name}")
                    
            except Exception as e:
                results["errors"] += 1
                logger.error(f"Failed to sync model {model.get('name', 'unknown')}: {e}")
        
        logger.info(f"Model registry sync complete: {results['registered']} registered, {results['updated']} updated, {results['errors']} errors")
        return results
    
    async def sync_prompt_registry(self, ollama_models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Sync prompt registry with model capabilities."""
        logger.info("Syncing prompt registry...")
        
        try:
            # Load cached prompts
            await self.prompt_manager.load_cached_prompts()
            
            # Generate model-specific datasets for each model
            results = {
                "models_processed": 0,
                "datasets_generated": 0,
                "errors": 0
            }
            
            for model in ollama_models:
                try:
                    model_name = model.get("name", "")
                    
                    # Determine model capabilities
                    capabilities = {
                        "code_generation": "code" in model_name.lower(),
                        "text_generation": True,
                        "question_answering": True,
                        "reasoning": True
                    }
                    
                    # Generate dataset for this model
                    dataset = await self.prompt_manager.generate_model_specific_dataset(
                        model_capabilities=capabilities,
                        target_size=100  # Smaller size for individual models
                    )
                    
                    results["models_processed"] += 1
                    results["datasets_generated"] += 1
                    logger.info(f"Generated dataset for {model_name}: {len(dataset)} prompts")
                    
                except Exception as e:
                    results["errors"] += 1
                    logger.error(f"Failed to generate dataset for {model.get('name', 'unknown')}: {e}")
            
            logger.info(f"Prompt registry sync complete: {results['models_processed']} models processed")
            return results
            
        except Exception as e:
            logger.error(f"Failed to sync prompt registry: {e}")
            return {"error": str(e)}
    
    async def sync_experiments(self, ollama_models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Sync experiments with available models."""
        logger.info("Syncing experiments...")
        
        results = {
            "experiments_created": 0,
            "runs_created": 0,
            "errors": 0
        }
        
        for model in ollama_models:
            try:
                model_name = model.get("name", "")
                model_size = model.get("size", 0)
                
                # Create baseline experiment
                run_name = f"baseline_{model_name.replace(':', '_').replace('.', '_')}"
                
                run_id = self.mlflow_manager.start_run(
                    run_name=run_name,
                    tags={
                        "model_name": model_name,
                        "experiment_type": "baseline",
                        "source": "ollama",
                        "sync_timestamp": datetime.now().isoformat()
                    },
                    description=f"Baseline experiment for {model_name}"
                )
                
                # Log model metrics
                metrics = {
                    "model_size_gb": round(model_size / (1024**3), 2),
                    "model_available": 1.0,
                    "sync_timestamp": datetime.now().timestamp()
                }
                
                self.mlflow_manager.log_llm_metrics(metrics, run_id)
                
                # Log model parameters
                params = {
                    "model_name": model_name,
                    "model_type": "code_generation" if "code" in model_name.lower() else "general",
                    "source": "ollama"
                }
                
                self.mlflow_manager.log_llm_params(params, run_id)
                
                # End the run
                self.mlflow_manager.end_run(run_id, "FINISHED")
                
                results["experiments_created"] += 1
                results["runs_created"] += 1
                logger.info(f"Created baseline experiment for {model_name}")
                
            except Exception as e:
                results["errors"] += 1
                logger.error(f"Failed to create experiment for {model.get('name', 'unknown')}: {e}")
        
        logger.info(f"Experiment sync complete: {results['experiments_created']} experiments created")
        return results
    
    def update_gradio_config(self, ollama_models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update Gradio app configuration with available models."""
        logger.info("Updating Gradio configuration...")
        
        try:
            config = {
                "available_models": [],
                "model_capabilities": {},
                "last_updated": datetime.now().isoformat()
            }
            
            for model in ollama_models:
                model_name = model.get("name", "")
                model_size = model.get("size", 0)
                
                model_info = {
                    "name": model_name,
                    "size_gb": round(model_size / (1024**3), 2),
                    "type": "code_generation" if "code" in model_name.lower() else "general",
                    "status": "available",
                    "source": "ollama"
                }
                
                config["available_models"].append(model_info)
                
                # Set capabilities
                if "code" in model_name.lower():
                    config["model_capabilities"][model_name] = [
                        "code_generation", "debugging", "documentation", "refactoring"
                    ]
                else:
                    config["model_capabilities"][model_name] = [
                        "text_generation", "question_answering", "reasoning", "summarization"
                    ]
            
            # Save configuration
            config_path = project_root / "config" / "gradio_models.json"
            config_path.parent.mkdir(exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Updated Gradio configuration: {len(config['available_models'])} models")
            return {
                "config_path": str(config_path),
                "models_configured": len(config["available_models"]),
                "last_updated": config["last_updated"]
            }
            
        except Exception as e:
            logger.error(f"Failed to update Gradio configuration: {e}")
            return {"error": str(e)}
    
    async def comprehensive_sync(self) -> Dict[str, Any]:
        """Perform comprehensive synchronization of all registries."""
        logger.info("Starting comprehensive registry sync...")
        
        try:
            # Initialize managers
            await self.initialize()
            
            # Get Ollama models
            ollama_models = await self.get_ollama_models()
            if not ollama_models:
                logger.warning("No Ollama models found")
                return {"error": "No Ollama models found"}
            
            # Perform sync operations
            results = {
                "ollama_models": len(ollama_models),
                "model_registry": await self.sync_model_registry(ollama_models),
                "prompt_registry": await self.sync_prompt_registry(ollama_models),
                "experiments": await self.sync_experiments(ollama_models),
                "gradio_config": self.update_gradio_config(ollama_models),
                "sync_timestamp": datetime.now().isoformat()
            }
            
            logger.info("Comprehensive sync completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive sync failed: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.ollama_manager:
            await self.ollama_manager.shutdown()
        if self.model_registry:
            self.model_registry.close()


async def main():
    """Main entry point."""
    sync_manager = RegistrySyncManager()
    
    try:
        results = await sync_manager.comprehensive_sync()
        
        print("\n" + "="*50)
        print("🎉 COMPREHENSIVE SYNC COMPLETE")
        print("="*50)
        print(f"📊 Ollama Models: {results.get('ollama_models', 0)}")
        print(f"🤖 Model Registry: {results.get('model_registry', {}).get('registered', 0)} registered")
        print(f"📝 Prompt Registry: {results.get('prompt_registry', {}).get('models_processed', 0)} models processed")
        print(f"🧪 Experiments: {results.get('experiments', {}).get('experiments_created', 0)} created")
        print(f"🎯 Gradio Config: {results.get('gradio_config', {}).get('models_configured', 0)} models configured")
        print(f"⏰ Sync Time: {results.get('sync_timestamp', 'unknown')}")
        print("="*50)
        
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            sys.exit(1)
        else:
            print("✅ All registries synchronized successfully!")
            
    except Exception as e:
        logger.error(f"Sync failed: {e}")
        sys.exit(1)
    finally:
        await sync_manager.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
