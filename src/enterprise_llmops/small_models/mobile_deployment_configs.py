"""
Mobile Deployment Configurations for Small Models

This module provides deployment configurations and optimization settings
for small models targeting mobile and edge platforms.

Key Features:
- Android deployment configurations
- iOS deployment configurations
- Edge device deployment settings
- Platform-specific optimizations
- Memory and performance tuning
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class DeploymentPlatform(Enum):
    """Supported deployment platforms."""
    ANDROID = "android"
    IOS = "ios"
    EDGE = "edge"
    EMBEDDED = "embedded"


class OptimizationLevel(Enum):
    """Optimization levels for deployment."""
    O0 = "O0"  # No optimization
    O1 = "O1"  # Basic optimization
    O2 = "O2"  # Standard optimization
    O3 = "O3"  # Aggressive optimization


@dataclass
class MemoryConfiguration:
    """Memory configuration for deployment."""
    heap_size_mb: int
    stack_size_mb: int
    model_cache_mb: int
    total_available_mb: int
    gc_threshold_percent: float = 80.0


@dataclass
class PerformanceConfiguration:
    """Performance configuration for deployment."""
    max_threads: int
    cpu_affinity: Optional[List[int]] = None
    priority_level: int = 0  # 0=normal, -1=higher, 1=lower
    enable_gpu: bool = False
    gpu_memory_mb: int = 0


@dataclass
class DeploymentConfiguration:
    """Complete deployment configuration for a platform."""
    platform: DeploymentPlatform
    target_architecture: str
    optimization_level: OptimizationLevel
    memory_config: MemoryConfiguration
    performance_config: PerformanceConfiguration
    model_specific_settings: Dict[str, Any]
    security_settings: Dict[str, Any]


class MobileDeploymentConfigManager:
    """
    Manager for mobile and edge deployment configurations.
    
    This class provides platform-specific configurations and optimization
    settings for deploying small models on mobile and edge devices.
    """
    
    def __init__(self):
        """Initialize the mobile deployment config manager."""
        self.logger = self._setup_logging()
        self.configurations = self._load_default_configurations()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for mobile deployment config manager."""
        logger = logging.getLogger("mobile_deployment_config")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_default_configurations(self) -> Dict[DeploymentPlatform, DeploymentConfiguration]:
        """Load default deployment configurations for all platforms."""
        configurations = {}
        
        # Android configuration
        configurations[DeploymentPlatform.ANDROID] = DeploymentConfiguration(
            platform=DeploymentPlatform.ANDROID,
            target_architecture="arm64-v8a",
            optimization_level=OptimizationLevel.O3,
            memory_config=MemoryConfiguration(
                heap_size_mb=512,
                stack_size_mb=8,
                model_cache_mb=256,
                total_available_mb=1024,
                gc_threshold_percent=85.0
            ),
            performance_config=PerformanceConfiguration(
                max_threads=4,
                priority_level=0,
                enable_gpu=False,
                gpu_memory_mb=0
            ),
            model_specific_settings={
                "quantization": "int8",
                "precision": "fp16",
                "batch_size": 1,
                "sequence_length": 512,
                "enable_jit": True,
                "use_nnapi": False
            },
            security_settings={
                "enable_sandbox": True,
                "require_permissions": ["INTERNET", "ACCESS_NETWORK_STATE"],
                "encryption_enabled": True,
                "certificate_pinning": True
            }
        )
        
        # iOS configuration
        configurations[DeploymentPlatform.IOS] = DeploymentConfiguration(
            platform=DeploymentPlatform.IOS,
            target_architecture="arm64",
            optimization_level=OptimizationLevel.O3,
            memory_config=MemoryConfiguration(
                heap_size_mb=256,
                stack_size_mb=8,
                model_cache_mb=128,
                total_available_mb=512,
                gc_threshold_percent=90.0
            ),
            performance_config=PerformanceConfiguration(
                max_threads=2,
                priority_level=0,
                enable_gpu=False,
                gpu_memory_mb=0
            ),
            model_specific_settings={
                "quantization": "int8",
                "precision": "fp16",
                "batch_size": 1,
                "sequence_length": 256,
                "enable_jit": False,
                "use_coreml": True
            },
            security_settings={
                "enable_sandbox": True,
                "require_permissions": ["NSAppTransportSecurity"],
                "encryption_enabled": True,
                "keychain_storage": True
            }
        )
        
        # Edge configuration
        configurations[DeploymentPlatform.EDGE] = DeploymentConfiguration(
            platform=DeploymentPlatform.EDGE,
            target_architecture="x86_64",
            optimization_level=OptimizationLevel.O2,
            memory_config=MemoryConfiguration(
                heap_size_mb=1024,
                stack_size_mb=16,
                model_cache_mb=512,
                total_available_mb=2048,
                gc_threshold_percent=75.0
            ),
            performance_config=PerformanceConfiguration(
                max_threads=8,
                priority_level=0,
                enable_gpu=True,
                gpu_memory_mb=1024
            ),
            model_specific_settings={
                "quantization": "int4",
                "precision": "fp32",
                "batch_size": 4,
                "sequence_length": 1024,
                "enable_jit": True,
                "use_cuda": True
            },
            security_settings={
                "enable_sandbox": False,
                "require_permissions": [],
                "encryption_enabled": True,
                "tls_verification": True
            }
        )
        
        # Embedded configuration
        configurations[DeploymentPlatform.EMBEDDED] = DeploymentConfiguration(
            platform=DeploymentPlatform.EMBEDDED,
            target_architecture="armv7",
            optimization_level=OptimizationLevel.O3,
            memory_config=MemoryConfiguration(
                heap_size_mb=64,
                stack_size_mb=4,
                model_cache_mb=32,
                total_available_mb=128,
                gc_threshold_percent=95.0
            ),
            performance_config=PerformanceConfiguration(
                max_threads=1,
                priority_level=1,
                enable_gpu=False,
                gpu_memory_mb=0
            ),
            model_specific_settings={
                "quantization": "int8",
                "precision": "int8",
                "batch_size": 1,
                "sequence_length": 128,
                "enable_jit": False,
                "use_neon": True
            },
            security_settings={
                "enable_sandbox": True,
                "require_permissions": [],
                "encryption_enabled": False,
                "minimal_footprint": True
            }
        )
        
        return configurations
    
    def get_deployment_config(self, platform: DeploymentPlatform) -> Optional[DeploymentConfiguration]:
        """Get deployment configuration for a specific platform."""
        return self.configurations.get(platform)
    
    def get_platform_specific_settings(self, platform: DeploymentPlatform, model_name: str) -> Dict[str, Any]:
        """Get platform-specific settings for a model."""
        config = self.get_deployment_config(platform)
        if not config:
            return {}
        
        # Base settings from platform configuration
        settings = config.model_specific_settings.copy()
        
        # Add model-specific overrides
        model_overrides = self._get_model_specific_overrides(model_name, platform)
        settings.update(model_overrides)
        
        return settings
    
    def _get_model_specific_overrides(self, model_name: str, platform: DeploymentPlatform) -> Dict[str, Any]:
        """Get model-specific configuration overrides."""
        overrides = {}
        
        # Model-specific optimizations based on model characteristics
        if "phi-4-mini" in model_name.lower():
            if platform == DeploymentPlatform.ANDROID:
                overrides.update({
                    "batch_size": 2,
                    "sequence_length": 1024,
                    "enable_jit": True
                })
            elif platform == DeploymentPlatform.IOS:
                overrides.update({
                    "batch_size": 1,
                    "sequence_length": 512,
                    "use_coreml": True
                })
        
        elif "llama-3.2-3b" in model_name.lower():
            if platform == DeploymentPlatform.EDGE:
                overrides.update({
                    "batch_size": 8,
                    "sequence_length": 2048,
                    "use_cuda": True
                })
        
        elif "qwen-2.5-3b" in model_name.lower():
            # Optimize for Chinese language support
            overrides.update({
                "tokenizer_cache_size": 1024,
                "enable_chinese_optimization": True
            })
        
        elif "mistral-nemo" in model_name.lower():
            # Optimize for efficiency
            overrides.update({
                "attention_optimization": True,
                "memory_efficient_attention": True
            })
        
        return overrides
    
    def get_memory_requirements(self, platform: DeploymentPlatform, model_size_gb: float) -> Dict[str, Any]:
        """Get memory requirements for deploying a model on a platform."""
        config = self.get_deployment_config(platform)
        if not config:
            return {}
        
        memory_config = config.memory_config
        
        # Calculate memory requirements
        model_memory_mb = model_size_gb * 1024
        total_memory_required = (
            memory_config.heap_size_mb +
            memory_config.stack_size_mb +
            memory_config.model_cache_mb +
            model_memory_mb
        )
        
        # Add safety margin
        safety_margin = 1.2
        recommended_memory = int(total_memory_required * safety_margin)
        
        return {
            "model_memory_mb": model_memory_mb,
            "heap_memory_mb": memory_config.heap_size_mb,
            "stack_memory_mb": memory_config.stack_size_mb,
            "cache_memory_mb": memory_config.model_cache_mb,
            "total_required_mb": total_memory_required,
            "recommended_memory_mb": recommended_memory,
            "available_memory_mb": memory_config.total_available_mb,
            "memory_sufficient": recommended_memory <= memory_config.total_available_mb
        }
    
    def get_performance_estimates(self, platform: DeploymentPlatform, model_name: str) -> Dict[str, Any]:
        """Get performance estimates for a model on a platform."""
        config = self.get_deployment_config(platform)
        if not config:
            return {}
        
        # Base performance estimates
        base_latency_ms = 100  # Base latency in milliseconds
        base_throughput_tps = 50  # Base throughput in tokens per second
        
        # Platform-specific adjustments
        platform_multipliers = {
            DeploymentPlatform.ANDROID: {"latency": 1.0, "throughput": 1.0},
            DeploymentPlatform.IOS: {"latency": 0.8, "throughput": 1.2},
            DeploymentPlatform.EDGE: {"latency": 0.5, "throughput": 2.0},
            DeploymentPlatform.EMBEDDED: {"latency": 2.0, "throughput": 0.3}
        }
        
        multipliers = platform_multipliers.get(platform, {"latency": 1.0, "throughput": 1.0})
        
        # Model-specific adjustments
        model_adjustments = self._get_model_performance_adjustments(model_name)
        
        # Calculate final estimates
        estimated_latency = base_latency_ms * multipliers["latency"] * model_adjustments["latency"]
        estimated_throughput = base_throughput_tps * multipliers["throughput"] * model_adjustments["throughput"]
        
        return {
            "estimated_latency_ms": estimated_latency,
            "estimated_throughput_tps": estimated_throughput,
            "platform_multipliers": multipliers,
            "model_adjustments": model_adjustments,
            "optimization_level": config.optimization_level.value,
            "max_threads": config.performance_config.max_threads,
            "gpu_enabled": config.performance_config.enable_gpu
        }
    
    def _get_model_performance_adjustments(self, model_name: str) -> Dict[str, float]:
        """Get performance adjustment factors for a specific model."""
        adjustments = {"latency": 1.0, "throughput": 1.0}
        
        # Model-specific performance characteristics
        if "phi-4-mini" in model_name.lower():
            adjustments = {"latency": 0.9, "throughput": 1.1}
        elif "llama-3.2-3b" in model_name.lower():
            adjustments = {"latency": 1.0, "throughput": 1.0}
        elif "qwen-2.5-3b" in model_name.lower():
            adjustments = {"latency": 1.1, "throughput": 0.9}
        elif "mistral-nemo" in model_name.lower():
            adjustments = {"latency": 0.8, "throughput": 1.2}
        
        return adjustments
    
    def validate_deployment_config(self, platform: DeploymentPlatform, model_name: str, model_size_gb: float) -> Dict[str, Any]:
        """Validate deployment configuration for a model on a platform."""
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        config = self.get_deployment_config(platform)
        if not config:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Platform {platform.value} not supported")
            return validation_results
        
        # Check memory requirements
        memory_req = self.get_memory_requirements(platform, model_size_gb)
        if not memory_req["memory_sufficient"]:
            validation_results["valid"] = False
            validation_results["errors"].append(
                f"Insufficient memory: {memory_req['recommended_memory_mb']}MB required, "
                f"{memory_req['available_memory_mb']}MB available"
            )
        
        # Check model size constraints
        max_model_size_gb = memory_req["available_memory_mb"] / 1024 * 0.5  # Use max 50% of available memory
        if model_size_gb > max_model_size_gb:
            validation_results["warnings"].append(
                f"Model size ({model_size_gb:.2f}GB) is large for platform. "
                f"Consider quantization or model compression."
            )
        
        # Platform-specific validations
        if platform == DeploymentPlatform.IOS:
            if model_size_gb > 1.0:
                validation_results["warnings"].append(
                    "Large models may impact iOS app store approval. Consider optimization."
                )
        
        elif platform == DeploymentPlatform.EMBEDDED:
            if model_size_gb > 0.5:
                validation_results["warnings"].append(
                    "Model may be too large for embedded deployment. Consider aggressive optimization."
                )
        
        # Generate recommendations
        if validation_results["valid"]:
            performance_est = self.get_performance_estimates(platform, model_name)
            validation_results["recommendations"].extend([
                f"Expected latency: {performance_est['estimated_latency_ms']:.1f}ms",
                f"Expected throughput: {performance_est['estimated_throughput_tps']:.1f} tokens/sec",
                f"Optimization level: {performance_est['optimization_level']}"
            ])
        
        return validation_results
    
    def get_all_platforms(self) -> List[DeploymentPlatform]:
        """Get list of all supported platforms."""
        return list(self.configurations.keys())
    
    def get_platform_capabilities(self, platform: DeploymentPlatform) -> Dict[str, Any]:
        """Get capabilities and limitations of a platform."""
        config = self.get_deployment_config(platform)
        if not config:
            return {}
        
        return {
            "platform": platform.value,
            "architecture": config.target_architecture,
            "max_memory_mb": config.memory_config.total_available_mb,
            "max_threads": config.performance_config.max_threads,
            "gpu_support": config.performance_config.enable_gpu,
            "optimization_levels": [level.value for level in OptimizationLevel],
            "supported_quantization": config.model_specific_settings.get("quantization", "unknown"),
            "security_features": list(config.security_settings.keys()),
            "recommended_model_size_gb": config.memory_config.total_available_mb / 1024 * 0.3  # 30% of available memory
        }


# Example usage and testing
def main():
    """Example usage of mobile deployment config manager."""
    manager = MobileDeploymentConfigManager()
    
    # List all supported platforms
    platforms = manager.get_all_platforms()
    print(f"Supported platforms: {[p.value for p in platforms]}")
    
    # Get platform capabilities
    for platform in platforms:
        capabilities = manager.get_platform_capabilities(platform)
        print(f"\n{platform.value.upper()} capabilities:")
        for key, value in capabilities.items():
            print(f"  {key}: {value}")
    
    # Test deployment configuration for a model
    model_name = "phi-4-mini"
    model_size_gb = 2.3
    
    print(f"\nDeployment validation for {model_name} ({model_size_gb}GB):")
    for platform in platforms:
        validation = manager.validate_deployment_config(platform, model_name, model_size_gb)
        status = "✅ VALID" if validation["valid"] else "❌ INVALID"
        print(f"\n{platform.value.upper()}: {status}")
        
        if validation["errors"]:
            for error in validation["errors"]:
                print(f"  Error: {error}")
        
        if validation["warnings"]:
            for warning in validation["warnings"]:
                print(f"  Warning: {warning}")
        
        if validation["recommendations"]:
            for rec in validation["recommendations"]:
                print(f"  Recommendation: {rec}")


if __name__ == "__main__":
    main()
