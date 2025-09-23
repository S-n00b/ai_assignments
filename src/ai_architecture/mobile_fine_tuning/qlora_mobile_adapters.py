"""
QLoRA Mobile Adapters for Small Models

This module provides QLoRA (Quantized Low-Rank Adaptation) adapters optimized
for mobile deployment with efficient parameter usage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class QLoRAMobileAdapter(nn.Module):
    """
    QLoRA adapter optimized for mobile deployment.
    
    Provides efficient parameter adaptation with minimal memory footprint
    and fast inference suitable for mobile devices.
    """
    
    def __init__(self, 
                 base_model_dim: int,
                 adapter_dim: int = 16,
                 rank: int = 8,
                 alpha: float = 16.0,
                 dropout: float = 0.1,
                 target_modules: List[str] = None):
        """
        Initialize QLoRA mobile adapter.
        
        Args:
            base_model_dim: Dimension of the base model
            adapter_dim: Dimension of the adapter
            rank: Rank of the low-rank decomposition
            alpha: Scaling factor for adapter
            dropout: Dropout rate
            target_modules: List of target module names
        """
        super().__init__()
        
        self.base_model_dim = base_model_dim
        self.adapter_dim = adapter_dim
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        # Initialize adapter layers
        self.adapters = nn.ModuleDict()
        self._initialize_adapters()
        
    def _initialize_adapters(self):
        """Initialize adapter layers for target modules."""
        for module_name in self.target_modules:
            self.adapters[module_name] = QLoRALayer(
                base_model_dim=self.base_model_dim,
                adapter_dim=self.adapter_dim,
                rank=self.rank,
                alpha=self.alpha,
                dropout=self.dropout
            )
    
    def forward(self, x: torch.Tensor, module_name: str) -> torch.Tensor:
        """
        Forward pass through QLoRA adapter.
        
        Args:
            x: Input tensor
            module_name: Name of the target module
            
        Returns:
            Adapted tensor
        """
        if module_name in self.adapters:
            return self.adapters[module_name](x)
        return x
    
    def get_adapter_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get adapter parameters for saving/loading.
        
        Returns:
            Dictionary of adapter parameters
        """
        return {name: adapter.state_dict() for name, adapter in self.adapters.items()}
    
    def load_adapter_parameters(self, parameters: Dict[str, Dict]):
        """
        Load adapter parameters.
        
        Args:
            parameters: Dictionary of adapter parameters
        """
        for name, params in parameters.items():
            if name in self.adapters:
                self.adapters[name].load_state_dict(params)
    
    def merge_adapters(self) -> Dict[str, torch.Tensor]:
        """
        Merge adapter weights into base model weights.
        
        Returns:
            Merged weights for each target module
        """
        merged_weights = {}
        
        for module_name, adapter in self.adapters.items():
            # Get adapter weights
            lora_A = adapter.lora_A.weight
            lora_B = adapter.lora_B.weight
            
            # Merge weights: W = W_base + (B @ A) * (alpha / rank)
            merged_weight = lora_B @ lora_A * (self.alpha / self.rank)
            merged_weights[module_name] = merged_weight
        
        return merged_weights
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage statistics for mobile deployment.
        
        Returns:
            Memory usage statistics
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Estimate memory usage (in MB)
        param_memory = total_params * 4 / 1024 / 1024  # 4 bytes per float32
        gradient_memory = trainable_params * 4 / 1024 / 1024
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "parameter_memory_mb": param_memory,
            "gradient_memory_mb": gradient_memory,
            "total_memory_mb": param_memory + gradient_memory
        }

class QLoRALayer(nn.Module):
    """
    Individual QLoRA layer for mobile deployment.
    
    Implements efficient low-rank adaptation with minimal parameters.
    """
    
    def __init__(self, 
                 base_model_dim: int,
                 adapter_dim: int,
                 rank: int,
                 alpha: float,
                 dropout: float):
        """
        Initialize QLoRA layer.
        
        Args:
            base_model_dim: Dimension of the base model
            adapter_dim: Dimension of the adapter
            rank: Rank of the low-rank decomposition
            alpha: Scaling factor
            dropout: Dropout rate
        """
        super().__init__()
        
        self.base_model_dim = base_model_dim
        self.adapter_dim = adapter_dim
        self.rank = rank
        self.alpha = alpha
        
        # Low-rank matrices
        self.lora_A = nn.Linear(base_model_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, adapter_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize adapter weights."""
        # Initialize A with normal distribution
        nn.init.normal_(self.lora_A.weight, std=0.01)
        # Initialize B with zeros
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through QLoRA layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Adapted tensor
        """
        # Apply low-rank adaptation
        h = self.lora_A(x)
        h = self.dropout(h)
        h = self.lora_B(h)
        
        # Scale by alpha/rank
        h = h * (self.alpha / self.rank)
        
        return h

class QLoRAMobileManager:
    """
    Manager for QLoRA mobile adapters.
    
    Provides functionality for creating, training, and deploying QLoRA adapters
    optimized for mobile devices.
    """
    
    def __init__(self, base_model_name: str, mobile_config: Dict):
        """
        Initialize QLoRA mobile manager.
        
        Args:
            base_model_name: Name of the base model
            mobile_config: Mobile deployment configuration
        """
        self.base_model_name = base_model_name
        self.mobile_config = mobile_config
        self.adapters = {}
        
    def create_adapter(self, 
                      target_modules: List[str],
                      rank: int = 8,
                      alpha: float = 16.0) -> QLoRAMobileAdapter:
        """
        Create QLoRA adapter for target modules.
        
        Args:
            target_modules: List of target module names
            rank: Rank of the low-rank decomposition
            alpha: Scaling factor
            
        Returns:
            Created QLoRA adapter
        """
        try:
            adapter = QLoRAMobileAdapter(
                base_model_dim=self.mobile_config.get("base_model_dim", 768),
                adapter_dim=self.mobile_config.get("adapter_dim", 16),
                rank=rank,
                alpha=alpha,
                dropout=self.mobile_config.get("dropout", 0.1),
                target_modules=target_modules
            )
            
            logger.info(f"Created QLoRA adapter with rank {rank}, alpha {alpha}")
            return adapter
            
        except Exception as e:
            logger.error(f"Failed to create QLoRA adapter: {e}")
            raise
    
    def train_adapter(self, 
                    adapter: QLoRAMobileAdapter,
                    training_data: List[Dict],
                    epochs: int = 3,
                    learning_rate: float = 1e-4) -> QLoRAMobileAdapter:
        """
        Train QLoRA adapter.
        
        Args:
            adapter: QLoRA adapter to train
            training_data: Training data
            epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            Trained adapter
        """
        try:
            from torch.optim import AdamW
            from torch.nn import CrossEntropyLoss
            
            # Setup training
            adapter.train()
            optimizer = AdamW(adapter.parameters(), lr=learning_rate)
            criterion = CrossEntropyLoss()
            
            for epoch in range(epochs):
                total_loss = 0
                
                for batch in training_data:
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = adapter(**batch)
                    loss = criterion(outputs.logits, batch["labels"])
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                logger.info(f"Epoch {epoch+1}/{epochs}, loss: {total_loss/len(training_data):.4f}")
            
            logger.info("QLoRA adapter training completed")
            return adapter
            
        except Exception as e:
            logger.error(f"QLoRA adapter training failed: {e}")
            raise
    
    def save_adapter(self, 
                    adapter: QLoRAMobileAdapter,
                    save_path: str) -> str:
        """
        Save QLoRA adapter.
        
        Args:
            adapter: Adapter to save
            save_path: Path to save adapter
            
        Returns:
            Path to saved adapter
        """
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save adapter parameters
            adapter_params = adapter.get_adapter_parameters()
            torch.save(adapter_params, save_path / "adapter_weights.pt")
            
            # Save adapter configuration
            config = {
                "base_model_dim": adapter.base_model_dim,
                "adapter_dim": adapter.adapter_dim,
                "rank": adapter.rank,
                "alpha": adapter.alpha,
                "dropout": adapter.dropout,
                "target_modules": adapter.target_modules
            }
            
            with open(save_path / "adapter_config.json", 'w') as f:
                json.dump(config, f, indent=2)
            
            # Save mobile-specific configuration
            mobile_config = {
                "memory_usage": adapter.get_memory_usage(),
                "mobile_optimizations": self.mobile_config,
                "deployment_ready": True
            }
            
            with open(save_path / "mobile_config.json", 'w') as f:
                json.dump(mobile_config, f, indent=2)
            
            logger.info(f"QLoRA adapter saved to {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Failed to save QLoRA adapter: {e}")
            raise
    
    def load_adapter(self, load_path: str) -> QLoRAMobileAdapter:
        """
        Load QLoRA adapter.
        
        Args:
            load_path: Path to load adapter from
            
        Returns:
            Loaded adapter
        """
        try:
            load_path = Path(load_path)
            
            # Load configuration
            with open(load_path / "adapter_config.json", 'r') as f:
                config = json.load(f)
            
            # Create adapter
            adapter = QLoRAMobileAdapter(
                base_model_dim=config["base_model_dim"],
                adapter_dim=config["adapter_dim"],
                rank=config["rank"],
                alpha=config["alpha"],
                dropout=config["dropout"],
                target_modules=config["target_modules"]
            )
            
            # Load weights
            adapter_params = torch.load(load_path / "adapter_weights.pt")
            adapter.load_adapter_parameters(adapter_params)
            
            logger.info(f"QLoRA adapter loaded from {load_path}")
            return adapter
            
        except Exception as e:
            logger.error(f"Failed to load QLoRA adapter: {e}")
            raise
    
    def deploy_adapter(self, 
                      adapter: QLoRAMobileAdapter,
                      deployment_config: Dict) -> Dict[str, str]:
        """
        Deploy QLoRA adapter for mobile devices.
        
        Args:
            adapter: Adapter to deploy
            deployment_config: Deployment configuration
            
        Returns:
            Deployment information
        """
        try:
            # Get merged weights
            merged_weights = adapter.merge_adapters()
            
            # Create deployment package
            deployment_info = {
                "adapter_path": deployment_config.get("adapter_path", "./deployed_adapters"),
                "model_name": self.base_model_name,
                "target_platforms": deployment_config.get("platforms", ["android", "ios", "edge"]),
                "optimization_level": deployment_config.get("optimization_level", "mobile"),
                "memory_usage": adapter.get_memory_usage()
            }
            
            # Save merged weights for each platform
            for platform in deployment_info["target_platforms"]:
                platform_path = Path(deployment_info["adapter_path"]) / platform
                platform_path.mkdir(parents=True, exist_ok=True)
                
                # Save platform-specific weights
                torch.save(merged_weights, platform_path / "merged_weights.pt")
                
                # Create platform-specific configuration
                platform_config = {
                    "platform": platform,
                    "optimization": deployment_config.get("platform_optimizations", {}).get(platform, {}),
                    "memory_requirements": adapter.get_memory_usage(),
                    "deployment_ready": True
                }
                
                with open(platform_path / "platform_config.json", 'w') as f:
                    json.dump(platform_config, f, indent=2)
            
            logger.info(f"QLoRA adapter deployed for platforms: {deployment_info['target_platforms']}")
            return deployment_info
            
        except Exception as e:
            logger.error(f"QLoRA adapter deployment failed: {e}")
            raise
    
    def benchmark_adapter(self, 
                         adapter: QLoRAMobileAdapter,
                         test_data: List[Dict]) -> Dict[str, float]:
        """
        Benchmark QLoRA adapter performance.
        
        Args:
            adapter: Adapter to benchmark
            test_data: Test data
            
        Returns:
            Performance metrics
        """
        try:
            import time
            
            adapter.eval()
            
            # Benchmark inference time
            times = []
            with torch.no_grad():
                for batch in test_data[:10]:  # Test on first 10 samples
                    start_time = time.time()
                    
                    # Run inference
                    outputs = adapter(**batch)
                    
                    end_time = time.time()
                    times.append((end_time - start_time) * 1000)  # Convert to ms
            
            # Calculate metrics
            avg_inference_time = sum(times) / len(times)
            max_inference_time = max(times)
            min_inference_time = min(times)
            
            # Memory usage
            memory_usage = adapter.get_memory_usage()
            
            metrics = {
                "avg_inference_time_ms": avg_inference_time,
                "max_inference_time_ms": max_inference_time,
                "min_inference_time_ms": min_inference_time,
                "throughput_samples_per_second": 1000 / avg_inference_time,
                "memory_usage_mb": memory_usage["total_memory_mb"],
                "parameter_count": memory_usage["total_parameters"],
                "trainable_parameters": memory_usage["trainable_parameters"]
            }
            
            logger.info(f"QLoRA adapter benchmark results: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"QLoRA adapter benchmarking failed: {e}")
            raise
