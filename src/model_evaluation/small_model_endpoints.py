"""
Small Model Endpoints for Phase 2 Testing

This module provides API endpoints for testing small models (<4B parameters)
in the Phase 2 implementation.

Key Features:
- Small model inference endpoints
- Performance testing endpoints
- Model comparison endpoints
- Mobile deployment testing
- Integration with GitHub Models API
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import yaml
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn


@dataclass
class ModelTestRequest:
    """Request for testing a small model."""
    model_name: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    test_type: str = "inference"  # "inference", "performance", "comparison"


@dataclass
class ModelTestResponse:
    """Response from testing a small model."""
    model_name: str
    success: bool
    response_text: Optional[str] = None
    inference_time_ms: Optional[float] = None
    tokens_generated: Optional[int] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ModelTestRequestModel(BaseModel):
    """Pydantic model for model test requests."""
    model_name: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    test_type: str = "inference"


class SmallModelEndpoints:
    """
    API endpoints for small model testing and evaluation.
    
    This class provides comprehensive endpoints for testing small models
    including inference, performance testing, and mobile deployment validation.
    """
    
    def __init__(self, config_path: str = "config/small_models_config.yaml"):
        """Initialize the small model endpoints."""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.app = FastAPI(
            title="Small Model Testing API",
            description="API for testing small models (<4B parameters) for mobile/edge deployment",
            version="2.0.0"
        )
        self.small_models = self._load_small_models_config()
        
        # Initialize endpoints
        self._setup_endpoints()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                self.logger.warning(f"Config file not found: {config_path}")
                return {}
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {}
    
    def _load_small_models_config(self) -> Dict[str, Any]:
        """Load small models configuration."""
        return self.config.get("small_models", {})
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for small model endpoints."""
        logger = logging.getLogger("small_model_endpoints")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _setup_endpoints(self):
        """Setup FastAPI endpoints."""
        
        @self.app.get("/")
        async def root():
            """Root endpoint with API information."""
            return {
                "name": "Small Model Testing API",
                "version": "2.0.0",
                "description": "API for testing small models for mobile/edge deployment",
                "available_models": list(self.small_models.keys()),
                "endpoints": [
                    "/models",
                    "/models/{model_name}/test",
                    "/models/{model_name}/performance",
                    "/models/compare",
                    "/models/{model_name}/deployment/{platform}",
                    "/health"
                ]
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "available_models": len(self.small_models),
                "phase": "Phase 2: Model Integration & Small Model Selection"
            }
        
        @self.app.get("/models")
        async def list_models():
            """List all available small models."""
            models_list = []
            for name, config in self.small_models.items():
                models_list.append({
                    "name": name,
                    "provider": config["provider"],
                    "parameters": config["parameters"],
                    "size_gb": config["size_gb"],
                    "use_case": config["use_case"],
                    "deployment_targets": config["deployment_targets"],
                    "ollama_name": config["ollama_name"],
                    "github_models_id": config["github_models_id"]
                })
            
            return {
                "models": models_list,
                "total_count": len(models_list),
                "phase": "Phase 2"
            }
        
        @self.app.post("/models/{model_name}/test")
        async def test_model(model_name: str, request: ModelTestRequestModel):
            """Test a specific small model."""
            try:
                if model_name not in self.small_models:
                    raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
                
                # Simulate model inference
                start_time = time.time()
                
                # Generate response based on model characteristics
                response_text = await self._generate_model_response(
                    model_name, request.prompt, request.max_tokens
                )
                
                inference_time = (time.time() - start_time) * 1000  # Convert to ms
                
                return asdict(ModelTestResponse(
                    model_name=model_name,
                    success=True,
                    response_text=response_text,
                    inference_time_ms=inference_time,
                    tokens_generated=len(response_text.split()) if response_text else 0,
                    performance_metrics={
                        "model_size_gb": self.small_models[model_name]["size_gb"],
                        "parameters": self.small_models[model_name]["parameters"],
                        "use_case": self.small_models[model_name]["use_case"]
                    }
                ))
                
            except Exception as e:
                self.logger.error(f"Error testing model {model_name}: {e}")
                return asdict(ModelTestResponse(
                    model_name=model_name,
                    success=False,
                    error=str(e)
                ))
        
        @self.app.get("/models/{model_name}/performance")
        async def get_model_performance(model_name: str):
            """Get performance metrics for a model."""
            try:
                if model_name not in self.small_models:
                    raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
                
                model_config = self.small_models[model_name]
                
                # Simulate performance testing
                performance_metrics = await self._benchmark_model(model_name)
                
                return {
                    "model_name": model_name,
                    "performance_profile": model_config["performance_profile"],
                    "memory_requirements": model_config["memory_requirements"],
                    "deployment_targets": model_config["deployment_targets"],
                    "benchmark_results": performance_metrics,
                    "optimization_recommendations": self._get_optimization_recommendations(model_name)
                }
                
            except Exception as e:
                self.logger.error(f"Error getting performance for {model_name}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/models/compare")
        async def compare_models(request: Dict[str, Any]):
            """Compare multiple models."""
            try:
                model_names = request.get("models", [])
                if not model_names:
                    raise HTTPException(status_code=400, detail="No models specified for comparison")
                
                # Validate all models exist
                for model_name in model_names:
                    if model_name not in self.small_models:
                        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
                
                # Run comparison
                comparison_results = await self._compare_models(model_names)
                
                return {
                    "comparison_results": comparison_results,
                    "timestamp": datetime.now().isoformat(),
                    "models_compared": model_names
                }
                
            except Exception as e:
                self.logger.error(f"Error comparing models: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/models/{model_name}/deployment/{platform}")
        async def validate_deployment(model_name: str, platform: str):
            """Validate model deployment on a specific platform."""
            try:
                if model_name not in self.small_models:
                    raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
                
                # Validate platform
                valid_platforms = ["android", "ios", "edge", "embedded"]
                if platform not in valid_platforms:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Invalid platform. Must be one of: {valid_platforms}"
                    )
                
                # Run deployment validation
                validation_result = await self._validate_deployment(model_name, platform)
                
                return {
                    "model_name": model_name,
                    "platform": platform,
                    "validation_result": validation_result,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"Error validating deployment: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/models/{model_name}/github-integration")
        async def get_github_integration(model_name: str):
            """Get GitHub Models API integration status for a model."""
            try:
                if model_name not in self.small_models:
                    raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
                
                model_config = self.small_models[model_name]
                
                # Simulate GitHub Models API integration check
                github_status = await self._check_github_integration(model_name)
                
                return {
                    "model_name": model_name,
                    "github_models_id": model_config["github_models_id"],
                    "integration_status": github_status,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"Error checking GitHub integration: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _generate_model_response(self, model_name: str, prompt: str, max_tokens: int) -> str:
        """Generate a response from a model (simulated)."""
        # This would integrate with actual model serving infrastructure
        # For now, we'll simulate responses based on model characteristics
        
        model_config = self.small_models[model_name]
        
        # Simulate different response styles based on model
        if "phi-4-mini" in model_name.lower():
            response = f"[Phi-4-Mini Response] Processing your request: '{prompt[:50]}...' This is a simulated response from Microsoft's Phi-4-Mini model, optimized for mobile deployment."
        elif "llama-3.2-3b" in model_name.lower():
            response = f"[Llama-3.2-3B Response] I understand you're asking about: '{prompt[:50]}...' As Meta's Llama-3.2-3B model, I'm designed for efficient on-device inference."
        elif "qwen-2.5-3b" in model_name.lower():
            response = f"[Qwen-2.5-3B Response] 处理您的请求：'{prompt[:50]}...' 这是阿里巴巴Qwen-2.5-3B模型的模拟响应，专门为中文移动支持优化。"
        elif "mistral-nemo" in model_name.lower():
            response = f"[Mistral-Nemo Response] Efficiently processing: '{prompt[:50]}...' This is Mistral's Nemo model, designed for efficient mobile AI applications."
        else:
            response = f"[Generic Response] Processing: '{prompt[:50]}...' This is a simulated response from a small model optimized for mobile deployment."
        
        # Truncate to max_tokens
        words = response.split()
        if len(words) > max_tokens:
            response = " ".join(words[:max_tokens]) + "..."
        
        return response
    
    async def _benchmark_model(self, model_name: str) -> Dict[str, Any]:
        """Benchmark a model's performance."""
        model_config = self.small_models[model_name]
        
        # Simulate benchmarking
        benchmark_results = {
            "latency_ms": model_config["performance_profile"]["latency_ms"],
            "throughput_tokens_per_sec": model_config["performance_profile"]["throughput_tokens_per_sec"],
            "accuracy_score": model_config["performance_profile"]["accuracy_score"],
            "memory_usage_mb": model_config["size_gb"] * 1024,
            "cpu_usage_percent": 45.0,  # Simulated
            "energy_efficiency": "high" if model_config["size_gb"] < 2.0 else "medium"
        }
        
        return benchmark_results
    
    async def _compare_models(self, model_names: List[str]) -> Dict[str, Any]:
        """Compare multiple models."""
        comparison_results = {
            "models": {},
            "rankings": {},
            "recommendations": {}
        }
        
        for model_name in model_names:
            model_config = self.small_models[model_name]
            performance = await self._benchmark_model(model_name)
            
            comparison_results["models"][model_name] = {
                "config": model_config,
                "performance": performance
            }
        
        # Calculate rankings
        models_by_latency = sorted(model_names, 
                                 key=lambda m: comparison_results["models"][m]["performance"]["latency_ms"])
        models_by_throughput = sorted(model_names, 
                                    key=lambda m: comparison_results["models"][m]["performance"]["throughput_tokens_per_sec"],
                                    reverse=True)
        models_by_size = sorted(model_names, 
                              key=lambda m: comparison_results["models"][m]["config"]["size_gb"])
        
        comparison_results["rankings"] = {
            "fastest_latency": models_by_latency[0],
            "highest_throughput": models_by_throughput[0],
            "smallest_size": models_by_size[0]
        }
        
        # Generate recommendations
        comparison_results["recommendations"] = {
            "mobile_deployment": models_by_size[0] if models_by_size else None,
            "edge_deployment": models_by_throughput[0] if models_by_throughput else None,
            "embedded_deployment": models_by_size[0] if models_by_size else None
        }
        
        return comparison_results
    
    async def _validate_deployment(self, model_name: str, platform: str) -> Dict[str, Any]:
        """Validate model deployment on a platform."""
        model_config = self.small_models[model_name]
        
        # Platform-specific validation
        validation_result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        # Check if platform is supported
        if platform not in model_config["deployment_targets"]:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Platform {platform} not supported by {model_name}")
        
        # Check memory requirements
        memory_req = model_config["memory_requirements"]
        if platform == "android" and memory_req["minimum"] > 4:
            validation_result["warnings"].append("High memory requirements for Android deployment")
        
        if platform == "ios" and memory_req["minimum"] > 3:
            validation_result["warnings"].append("High memory requirements for iOS deployment")
        
        if platform == "embedded" and memory_req["minimum"] > 2:
            validation_result["warnings"].append("Very high memory requirements for embedded deployment")
        
        # Generate recommendations
        if platform == "mobile" and model_config["size_gb"] > 2.0:
            validation_result["recommendations"].append("Consider quantization for mobile deployment")
        
        if platform == "edge" and "quantization" in model_config["optimization_flags"]:
            validation_result["recommendations"].append("Quantization recommended for edge deployment")
        
        return validation_result
    
    async def _check_github_integration(self, model_name: str) -> Dict[str, Any]:
        """Check GitHub Models API integration status."""
        model_config = self.small_models[model_name]
        
        # Simulate GitHub Models API check
        return {
            "github_models_id": model_config["github_models_id"],
            "api_accessible": True,
            "package_available": True,
            "last_updated": "2025-01-15T10:30:00Z",
            "download_count": 1250,
            "integration_ready": True
        }
    
    def _get_optimization_recommendations(self, model_name: str) -> List[str]:
        """Get optimization recommendations for a model."""
        model_config = self.small_models[model_name]
        recommendations = []
        
        # Based on optimization flags
        if "quantization" in model_config["optimization_flags"]:
            recommendations.append("Apply quantization to reduce model size by ~70%")
        
        if "pruning" in model_config["optimization_flags"]:
            recommendations.append("Apply pruning to remove unnecessary parameters")
        
        if "distillation" in model_config["optimization_flags"]:
            recommendations.append("Use knowledge distillation for better performance")
        
        # Based on size
        if model_config["size_gb"] > 2.0:
            recommendations.append("Consider aggressive quantization for large models")
        
        return recommendations
    
    def run(self, host: str = "0.0.0.0", port: int = 8081):
        """Run the small model endpoints server."""
        self.logger.info(f"Starting Small Model Endpoints server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# Example usage and testing
def main():
    """Example usage of small model endpoints."""
    endpoints = SmallModelEndpoints()
    
    print("Small Model Endpoints initialized")
    print(f"Available models: {list(endpoints.small_models.keys())}")
    print(f"API Documentation: http://localhost:8081/docs")
    
    # Run the server
    endpoints.run(port=8081)


if __name__ == "__main__":
    main()
