"""
Mobile Optimization for Small Models

This module provides optimization techniques for deploying small models on mobile/edge devices
including quantization, pruning, and distillation.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class MobileOptimizer:
    """
    Mobile optimization for small models.
    
    Provides quantization, pruning, and distillation techniques for mobile deployment.
    """
    
    def __init__(self, model_name: str, target_device: str = "mobile"):
        """
        Initialize mobile optimizer.
        
        Args:
            model_name: Name of the model to optimize
            target_device: Target deployment device (mobile, edge, embedded)
        """
        self.model_name = model_name
        self.target_device = target_device
        self.optimization_configs = self._get_optimization_configs()
        
    def _get_optimization_configs(self) -> Dict[str, Dict]:
        """Get optimization configurations for different devices."""
        return {
            "mobile": {
                "quantization": "dynamic",
                "pruning_ratio": 0.2,
                "max_memory_mb": 512,
                "max_latency_ms": 100,
                "precision": "int8"
            },
            "edge": {
                "quantization": "static",
                "pruning_ratio": 0.3,
                "max_memory_mb": 1024,
                "max_latency_ms": 200,
                "precision": "int8"
            },
            "embedded": {
                "quantization": "static",
                "pruning_ratio": 0.4,
                "max_memory_mb": 256,
                "max_latency_ms": 50,
                "precision": "int8"
            }
        }
    
    def quantize_model(self, 
                      model: nn.Module, 
                      quantization_type: str = "dynamic") -> nn.Module:
        """
        Quantize model for mobile deployment.
        
        Args:
            model: Model to quantize
            quantization_type: Type of quantization (dynamic, static, qat)
            
        Returns:
            Quantized model
        """
        try:
            if quantization_type == "dynamic":
                # Dynamic quantization
                quantized_model = torch.quantization.quantize_dynamic(
                    model,
                    {nn.Linear, nn.LSTM, nn.GRU},
                    dtype=torch.qint8
                )
                logger.info("Applied dynamic quantization")
                
            elif quantization_type == "static":
                # Static quantization requires calibration
                model.eval()
                model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                model_prepared = torch.quantization.prepare(model)
                
                # Calibration would go here with representative data
                # For now, we'll use the prepared model
                quantized_model = torch.quantization.convert(model_prepared)
                logger.info("Applied static quantization")
                
            else:
                raise ValueError(f"Unsupported quantization type: {quantization_type}")
            
            return quantized_model
            
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            raise
    
    def prune_model(self, 
                   model: nn.Module, 
                   pruning_ratio: float = 0.2) -> nn.Module:
        """
        Prune model to reduce size.
        
        Args:
            model: Model to prune
            pruning_ratio: Ratio of parameters to prune (0.0 to 1.0)
            
        Returns:
            Pruned model
        """
        try:
            import torch.nn.utils.prune as prune
            
            # Get all linear layers
            linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
            
            # Apply structured pruning
            for layer in linear_layers:
                prune.ln_structured(
                    layer,
                    name='weight',
                    amount=pruning_ratio,
                    n=2,
                    dim=0
                )
            
            # Remove pruning reparameterization
            for layer in linear_layers:
                prune.remove(layer, 'weight')
            
            logger.info(f"Applied pruning with ratio {pruning_ratio}")
            return model
            
        except Exception as e:
            logger.error(f"Pruning failed: {e}")
            raise
    
    def distill_model(self, 
                     teacher_model: nn.Module,
                     student_model: nn.Module,
                     training_data: List[Dict],
                     epochs: int = 3) -> nn.Module:
        """
        Distill knowledge from teacher to student model.
        
        Args:
            teacher_model: Large teacher model
            student_model: Small student model
            training_data: Training data for distillation
            epochs: Number of training epochs
            
        Returns:
            Distilled student model
        """
        try:
            from torch.nn import KLDivLoss
            from torch.optim import Adam
            
            # Setup distillation
            teacher_model.eval()
            student_model.train()
            
            criterion = KLDivLoss(reduction='batchmean')
            optimizer = Adam(student_model.parameters(), lr=1e-4)
            
            # Temperature for distillation
            temperature = 3.0
            
            for epoch in range(epochs):
                total_loss = 0
                
                for batch in training_data:
                    optimizer.zero_grad()
                    
                    # Get teacher predictions
                    with torch.no_grad():
                        teacher_outputs = teacher_model(**batch)
                        teacher_logits = teacher_outputs.logits / temperature
                        teacher_probs = torch.softmax(teacher_logits, dim=-1)
                    
                    # Get student predictions
                    student_outputs = student_model(**batch)
                    student_logits = student_outputs.logits / temperature
                    student_log_probs = torch.log_softmax(student_logits, dim=-1)
                    
                    # Compute distillation loss
                    loss = criterion(student_log_probs, teacher_probs) * (temperature ** 2)
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                logger.info(f"Distillation epoch {epoch+1}/{epochs}, loss: {total_loss/len(training_data):.4f}")
            
            logger.info("Knowledge distillation completed")
            return student_model
            
        except Exception as e:
            logger.error(f"Distillation failed: {e}")
            raise
    
    def optimize_for_mobile(self, 
                          model: nn.Module,
                          optimization_level: str = "aggressive") -> nn.Module:
        """
        Apply comprehensive mobile optimization.
        
        Args:
            model: Model to optimize
            optimization_level: Level of optimization (conservative, moderate, aggressive)
            
        Returns:
            Optimized model
        """
        try:
            config = self.optimization_configs[self.target_device]
            
            if optimization_level == "conservative":
                # Light optimization
                model = self.quantize_model(model, "dynamic")
                
            elif optimization_level == "moderate":
                # Medium optimization
                model = self.quantize_model(model, "static")
                model = self.prune_model(model, config["pruning_ratio"] * 0.5)
                
            elif optimization_level == "aggressive":
                # Heavy optimization
                model = self.quantize_model(model, "static")
                model = self.prune_model(model, config["pruning_ratio"])
                
            else:
                raise ValueError(f"Unsupported optimization level: {optimization_level}")
            
            logger.info(f"Applied {optimization_level} mobile optimization")
            return model
            
        except Exception as e:
            logger.error(f"Mobile optimization failed: {e}")
            raise
    
    def create_mobile_config(self, 
                           model_path: str,
                           output_path: str) -> Dict:
        """
        Create mobile deployment configuration.
        
        Args:
            model_path: Path to optimized model
            output_path: Path to save configuration
            
        Returns:
            Mobile deployment configuration
        """
        try:
            config = {
                "model_info": {
                    "name": self.model_name,
                    "path": model_path,
                    "target_device": self.target_device,
                    "optimization_level": "mobile_optimized"
                },
                "deployment": {
                    "max_memory_mb": self.optimization_configs[self.target_device]["max_memory_mb"],
                    "max_latency_ms": self.optimization_configs[self.target_device]["max_latency_ms"],
                    "precision": self.optimization_configs[self.target_device]["precision"],
                    "batch_size": 1,
                    "max_sequence_length": 512
                },
                "platforms": {
                    "android": {
                        "architecture": "arm64-v8a",
                        "min_sdk": 21,
                        "target_sdk": 33
                    },
                    "ios": {
                        "architecture": "arm64",
                        "min_version": "12.0",
                        "target_version": "16.0"
                    },
                    "edge": {
                        "architecture": "x86_64",
                        "os": "linux",
                        "min_ram": "1GB"
                    }
                },
                "performance": {
                    "inference_time_ms": "< 100",
                    "memory_usage_mb": "< 512",
                    "model_size_mb": "< 100",
                    "throughput_tokens_per_second": "> 50"
                }
            }
            
            # Save configuration
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Mobile configuration saved to {output_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to create mobile config: {e}")
            raise
    
    def benchmark_mobile_performance(self, 
                                   model: nn.Module,
                                   test_data: List[Dict]) -> Dict[str, float]:
        """
        Benchmark model performance for mobile deployment.
        
        Args:
            model: Model to benchmark
            test_data: Test data for benchmarking
            
        Returns:
            Performance metrics
        """
        try:
            import time
            import psutil
            
            model.eval()
            
            # Memory usage
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Inference time
            times = []
            with torch.no_grad():
                for batch in test_data[:10]:  # Test on first 10 samples
                    start_time = time.time()
                    
                    # Run inference
                    outputs = model(**batch)
                    
                    end_time = time.time()
                    times.append((end_time - start_time) * 1000)  # Convert to ms
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = memory_after - memory_before
            
            # Calculate metrics
            avg_inference_time = sum(times) / len(times)
            max_inference_time = max(times)
            min_inference_time = min(times)
            
            # Model size
            model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024  # MB
            
            metrics = {
                "avg_inference_time_ms": avg_inference_time,
                "max_inference_time_ms": max_inference_time,
                "min_inference_time_ms": min_inference_time,
                "memory_usage_mb": memory_usage,
                "model_size_mb": model_size,
                "throughput_samples_per_second": 1000 / avg_inference_time
            }
            
            logger.info(f"Mobile performance metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
            raise
    
    def validate_mobile_requirements(self, 
                                   model: nn.Module,
                                   requirements: Dict[str, float]) -> Dict[str, bool]:
        """
        Validate model against mobile deployment requirements.
        
        Args:
            model: Model to validate
            requirements: Mobile deployment requirements
            
        Returns:
            Validation results
        """
        try:
            # Get performance metrics
            metrics = self.benchmark_mobile_performance(model, [])
            
            # Check requirements
            validation_results = {
                "memory_requirement": metrics["memory_usage_mb"] <= requirements.get("max_memory_mb", 512),
                "latency_requirement": metrics["avg_inference_time_ms"] <= requirements.get("max_latency_ms", 100),
                "size_requirement": metrics["model_size_mb"] <= requirements.get("max_model_size_mb", 100),
                "throughput_requirement": metrics["throughput_samples_per_second"] >= requirements.get("min_throughput", 10)
            }
            
            # Overall validation
            validation_results["mobile_ready"] = all(validation_results.values())
            
            logger.info(f"Mobile validation results: {validation_results}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Mobile validation failed: {e}")
            raise
