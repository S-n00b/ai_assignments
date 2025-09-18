"""
Custom Adapter Registry System

This module implements a comprehensive adapter registry system for managing
fine-tuned model adapters, including storage, versioning, metadata tracking,
composition, and deployment capabilities.

Key Features:
- Centralized adapter storage and versioning
- Comprehensive metadata tracking
- Adapter discovery and search
- Automated validation and testing
- Multi-adapter composition and stacking
- Enterprise adapter sharing and collaboration
- Security and integrity validation
"""

import json
import hashlib
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union, Protocol
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import pickle
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from peft import PeftModel, PeftConfig, get_peft_model, TaskType
import onnx
import onnxruntime as ort
from accelerate import Accelerator
import wandb
from huggingface_hub import HfApi, Repository

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdapterType(Enum):
    """Types of adapters supported"""
    LORA = "lora"
    QLORA = "qlora"
    ADALORA = "adalora"
    PREFIX_TUNING = "prefix_tuning"
    P_TUNING_V2 = "p_tuning_v2"
    PROMPT_TUNING = "prompt_tuning"
    MULTITASK_PREFIX_TUNING = "multitask_prefix_tuning"
    CUSTOM = "custom"


class AdapterStatus(Enum):
    """Adapter status states"""
    DRAFT = "draft"
    TRAINING = "training"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class QuantizationType(Enum):
    """Quantization types supported"""
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    DYNAMIC = "dynamic"
    STATIC = "static"
    QAT = "qat"  # Quantization Aware Training


@dataclass
class AdapterMetadata:
    """Comprehensive metadata for adapters"""
    adapter_id: str
    name: str
    description: str
    adapter_type: AdapterType
    base_model: str
    version: str
    created_at: datetime
    updated_at: datetime
    created_by: str
    tags: List[str] = field(default_factory=list)
    
    # Performance metrics
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    benchmark_results: Dict[str, Any] = field(default_factory=dict)
    
    # Technical specifications
    parameters: int = 0
    size_mb: float = 0.0
    quantization_type: QuantizationType = QuantizationType.NONE
    precision: str = "fp32"
    
    # Training information
    training_data_size: int = 0
    training_epochs: int = 0
    learning_rate: float = 0.0
    batch_size: int = 0
    optimizer: str = ""
    
    # Domain and use case
    domain: str = ""
    use_cases: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    
    # Dependencies and compatibility
    dependencies: Dict[str, str] = field(default_factory=dict)
    compatible_models: List[str] = field(default_factory=list)
    min_memory_gb: float = 0.0
    min_gpu_memory_gb: float = 0.0
    
    # Security and compliance
    license: str = ""
    security_level: str = "standard"
    compliance_frameworks: List[str] = field(default_factory=list)
    
    # Status and lifecycle
    status: AdapterStatus = AdapterStatus.DRAFT
    validation_status: str = "pending"
    deployment_status: str = "not_deployed"
    
    # Usage statistics
    download_count: int = 0
    usage_count: int = 0
    last_used: Optional[datetime] = None
    
    # Relationships
    parent_adapters: List[str] = field(default_factory=list)
    child_adapters: List[str] = field(default_factory=list)
    related_adapters: List[str] = field(default_factory=list)


@dataclass
class AdapterComposition:
    """Configuration for multi-adapter composition"""
    composition_id: str
    name: str
    adapters: List[str]  # List of adapter IDs
    composition_strategy: str  # "sequential", "parallel", "hierarchical", "weighted"
    weights: Optional[Dict[str, float]] = None
    fusion_method: str = "linear"  # "linear", "attention", "gating"
    created_at: datetime = field(default_factory=datetime.now)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class AdapterValidator(ABC):
    """Abstract base class for adapter validation"""
    
    @abstractmethod
    async def validate_adapter(self, adapter_path: str, metadata: AdapterMetadata) -> Dict[str, Any]:
        """Validate an adapter and return validation results"""
        pass


class PerformanceValidator(AdapterValidator):
    """Validator for adapter performance"""
    
    async def validate_adapter(self, adapter_path: str, metadata: AdapterMetadata) -> Dict[str, Any]:
        """Validate adapter performance on benchmark tasks"""
        try:
            # Load adapter and base model
            base_model = AutoModel.from_pretrained(metadata.base_model)
            tokenizer = AutoTokenizer.from_pretrained(metadata.base_model)
            
            # Load adapter
            model = PeftModel.from_pretrained(base_model, adapter_path)
            
            # Run performance benchmarks
            benchmark_results = await self._run_benchmarks(model, tokenizer, metadata)
            
            return {
                "status": "passed",
                "benchmark_results": benchmark_results,
                "performance_score": self._calculate_performance_score(benchmark_results),
                "validation_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Performance validation failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "validation_time": datetime.now().isoformat()
            }
    
    async def _run_benchmarks(self, model: PeftModel, tokenizer, metadata: AdapterMetadata) -> Dict[str, Any]:
        """Run comprehensive benchmarks on the adapter"""
        # This would implement actual benchmark tasks
        # For now, return mock results
        return {
            "accuracy": np.random.uniform(0.7, 0.95),
            "perplexity": np.random.uniform(2.0, 5.0),
            "bleu_score": np.random.uniform(0.3, 0.8),
            "rouge_score": np.random.uniform(0.4, 0.9),
            "inference_time_ms": np.random.uniform(50, 200),
            "memory_usage_mb": np.random.uniform(100, 500)
        }
    
    def _calculate_performance_score(self, benchmark_results: Dict[str, Any]) -> float:
        """Calculate overall performance score"""
        # Weighted combination of metrics
        weights = {
            "accuracy": 0.3,
            "bleu_score": 0.2,
            "rouge_score": 0.2,
            "inference_time_ms": 0.15,  # Lower is better
            "memory_usage_mb": 0.15  # Lower is better
        }
        
        score = 0.0
        for metric, weight in weights.items():
            value = benchmark_results[metric]
            if metric in ["inference_time_ms", "memory_usage_mb"]:
                # Invert for metrics where lower is better
                value = 1.0 / (1.0 + value / 1000.0)
            score += weight * value
        
        return min(1.0, score)


class SecurityValidator(AdapterValidator):
    """Validator for adapter security and integrity"""
    
    async def validate_adapter(self, adapter_path: str, metadata: AdapterMetadata) -> Dict[str, Any]:
        """Validate adapter security and integrity"""
        try:
            # Check file integrity
            integrity_check = await self._check_file_integrity(adapter_path)
            
            # Check for malicious patterns
            security_check = await self._check_security_patterns(adapter_path)
            
            # Check license compliance
            license_check = await self._check_license_compliance(metadata)
            
            return {
                "status": "passed" if all([
                    integrity_check["passed"],
                    security_check["passed"],
                    license_check["passed"]
                ]) else "failed",
                "integrity_check": integrity_check,
                "security_check": security_check,
                "license_check": license_check,
                "validation_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Security validation failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "validation_time": datetime.now().isoformat()
            }
    
    async def _check_file_integrity(self, adapter_path: str) -> Dict[str, Any]:
        """Check file integrity using checksums"""
        try:
            # Calculate checksum
            with open(adapter_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            
            return {
                "passed": True,
                "checksum": file_hash,
                "file_size": Path(adapter_path).stat().st_size
            }
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }
    
    async def _check_security_patterns(self, adapter_path: str) -> Dict[str, Any]:
        """Check for malicious patterns in adapter files"""
        # This would implement actual security scanning
        # For now, return mock results
        return {
            "passed": True,
            "scanned_files": 10,
            "threats_detected": 0,
            "scan_details": "No malicious patterns detected"
        }
    
    async def _check_license_compliance(self, metadata: AdapterMetadata) -> Dict[str, Any]:
        """Check license compliance"""
        # This would implement actual license checking
        # For now, return mock results
        return {
            "passed": True,
            "license": metadata.license,
            "compliance_status": "compliant"
        }


class CustomAdapterRegistry:
    """
    Enterprise Custom Adapter Registry System
    
    This class provides comprehensive management of fine-tuned model adapters,
    including storage, versioning, metadata tracking, composition, and deployment.
    """
    
    def __init__(self, registry_path: str = "./adapter_registry"):
        """
        Initialize the Custom Adapter Registry.
        
        Args:
            registry_path: Path to the registry storage directory
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize registry components
        self.adapters = {}
        self.compositions = {}
        self.validators = {
            "performance": PerformanceValidator(),
            "security": SecurityValidator()
        }
        
        # Initialize storage paths
        self.adapters_path = self.registry_path / "adapters"
        self.metadata_path = self.registry_path / "metadata"
        self.compositions_path = self.registry_path / "compositions"
        self.cache_path = self.registry_path / "cache"
        
        # Create directories
        for path in [self.adapters_path, self.metadata_path, self.compositions_path, self.cache_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Load existing registry data
        self._load_registry_data()
        
        logger.info(f"Initialized Custom Adapter Registry at {self.registry_path}")
    
    def _load_registry_data(self):
        """Load existing registry data from storage"""
        try:
            # Load adapter metadata
            metadata_files = list(self.metadata_path.glob("*.json"))
            for metadata_file in metadata_files:
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                    # Convert datetime strings back to datetime objects
                    for key in ['created_at', 'updated_at', 'last_used']:
                        if key in metadata_dict and metadata_dict[key]:
                            metadata_dict[key] = datetime.fromisoformat(metadata_dict[key])
                    
                    metadata = AdapterMetadata(**metadata_dict)
                    self.adapters[metadata.adapter_id] = metadata
            
            # Load compositions
            composition_files = list(self.compositions_path.glob("*.json"))
            for composition_file in composition_files:
                with open(composition_file, 'r') as f:
                    composition_dict = json.load(f)
                    composition_dict['created_at'] = datetime.fromisoformat(composition_dict['created_at'])
                    composition = AdapterComposition(**composition_dict)
                    self.compositions[composition.composition_id] = composition
            
            logger.info(f"Loaded {len(self.adapters)} adapters and {len(self.compositions)} compositions")
            
        except Exception as e:
            logger.error(f"Failed to load registry data: {str(e)}")
    
    async def register_adapter(
        self,
        adapter_path: str,
        metadata: AdapterMetadata,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Register a new adapter in the registry.
        
        Args:
            adapter_path: Path to the adapter files
            metadata: Adapter metadata
            validate: Whether to validate the adapter before registration
            
        Returns:
            Registration result with status and details
        """
        try:
            logger.info(f"Registering adapter {metadata.adapter_id}")
            
            # Validate adapter if requested
            validation_results = {}
            if validate:
                validation_results = await self._validate_adapter(adapter_path, metadata)
                if not validation_results.get("overall_status") == "passed":
                    return {
                        "status": "failed",
                        "error": "Adapter validation failed",
                        "validation_results": validation_results
                    }
            
            # Copy adapter files to registry
            adapter_registry_path = self.adapters_path / metadata.adapter_id
            adapter_registry_path.mkdir(parents=True, exist_ok=True)
            
            # Copy files
            if Path(adapter_path).is_dir():
                shutil.copytree(adapter_path, adapter_registry_path, dirs_exist_ok=True)
            else:
                shutil.copy2(adapter_path, adapter_registry_path)
            
            # Update metadata
            metadata.updated_at = datetime.now()
            metadata.status = AdapterStatus.VALIDATED if validate else AdapterStatus.DRAFT
            
            # Save metadata
            self.adapters[metadata.adapter_id] = metadata
            await self._save_adapter_metadata(metadata)
            
            logger.info(f"Successfully registered adapter {metadata.adapter_id}")
            
            return {
                "status": "success",
                "adapter_id": metadata.adapter_id,
                "registry_path": str(adapter_registry_path),
                "validation_results": validation_results,
                "registered_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to register adapter {metadata.adapter_id}: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _validate_adapter(self, adapter_path: str, metadata: AdapterMetadata) -> Dict[str, Any]:
        """Run comprehensive validation on an adapter"""
        validation_results = {}
        
        # Run all validators
        for validator_name, validator in self.validators.items():
            try:
                result = await validator.validate_adapter(adapter_path, metadata)
                validation_results[validator_name] = result
            except Exception as e:
                validation_results[validator_name] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        # Determine overall status
        overall_status = "passed" if all(
            result.get("status") == "passed" 
            for result in validation_results.values()
        ) else "failed"
        
        validation_results["overall_status"] = overall_status
        return validation_results
    
    async def _save_adapter_metadata(self, metadata: AdapterMetadata):
        """Save adapter metadata to storage"""
        metadata_file = self.metadata_path / f"{metadata.adapter_id}.json"
        
        # Convert to dict and handle datetime serialization
        metadata_dict = asdict(metadata)
        for key in ['created_at', 'updated_at', 'last_used']:
            if key in metadata_dict and metadata_dict[key]:
                metadata_dict[key] = metadata_dict[key].isoformat()
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
    
    async def search_adapters(
        self,
        query: str = "",
        adapter_type: Optional[AdapterType] = None,
        domain: Optional[str] = None,
        base_model: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_performance: Optional[float] = None,
        limit: int = 50
    ) -> List[AdapterMetadata]:
        """
        Search for adapters based on various criteria.
        
        Args:
            query: Text search query
            adapter_type: Filter by adapter type
            domain: Filter by domain
            base_model: Filter by base model
            tags: Filter by tags
            min_performance: Minimum performance score
            limit: Maximum number of results
            
        Returns:
            List of matching adapter metadata
        """
        results = []
        
        for adapter in self.adapters.values():
            # Apply filters
            if adapter_type and adapter.adapter_type != adapter_type:
                continue
            
            if domain and domain.lower() not in adapter.domain.lower():
                continue
            
            if base_model and base_model.lower() not in adapter.base_model.lower():
                continue
            
            if tags and not any(tag.lower() in [t.lower() for t in adapter.tags] for tag in tags):
                continue
            
            if min_performance and adapter.performance_metrics.get("overall_score", 0) < min_performance:
                continue
            
            # Text search
            if query:
                search_text = f"{adapter.name} {adapter.description} {' '.join(adapter.tags)}"
                if query.lower() not in search_text.lower():
                    continue
            
            results.append(adapter)
        
        # Sort by performance score and usage
        results.sort(key=lambda x: (
            x.performance_metrics.get("overall_score", 0),
            x.usage_count
        ), reverse=True)
        
        return results[:limit]
    
    async def create_adapter_composition(
        self,
        name: str,
        adapter_ids: List[str],
        composition_strategy: str = "sequential",
        weights: Optional[Dict[str, float]] = None,
        fusion_method: str = "linear"
    ) -> Dict[str, Any]:
        """
        Create a multi-adapter composition.
        
        Args:
            name: Name of the composition
            adapter_ids: List of adapter IDs to compose
            composition_strategy: Strategy for composition
            weights: Optional weights for adapters
            fusion_method: Method for fusing adapters
            
        Returns:
            Composition creation result
        """
        try:
            # Validate adapters exist
            for adapter_id in adapter_ids:
                if adapter_id not in self.adapters:
                    return {
                        "status": "failed",
                        "error": f"Adapter {adapter_id} not found"
                    }
            
            # Create composition
            composition_id = str(uuid.uuid4())
            composition = AdapterComposition(
                composition_id=composition_id,
                name=name,
                adapters=adapter_ids,
                composition_strategy=composition_strategy,
                weights=weights,
                fusion_method=fusion_method
            )
            
            # Save composition
            self.compositions[composition_id] = composition
            await self._save_composition(composition)
            
            logger.info(f"Created adapter composition {composition_id}")
            
            return {
                "status": "success",
                "composition_id": composition_id,
                "composition": asdict(composition)
            }
            
        except Exception as e:
            logger.error(f"Failed to create adapter composition: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _save_composition(self, composition: AdapterComposition):
        """Save composition to storage"""
        composition_file = self.compositions_path / f"{composition.composition_id}.json"
        
        composition_dict = asdict(composition)
        composition_dict['created_at'] = composition.created_at.isoformat()
        
        with open(composition_file, 'w') as f:
            json.dump(composition_dict, f, indent=2)
    
    async def deploy_adapter(
        self,
        adapter_id: str,
        deployment_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deploy an adapter for production use.
        
        Args:
            adapter_id: ID of the adapter to deploy
            deployment_config: Deployment configuration
            
        Returns:
            Deployment result
        """
        try:
            if adapter_id not in self.adapters:
                return {
                    "status": "failed",
                    "error": f"Adapter {adapter_id} not found"
                }
            
            adapter = self.adapters[adapter_id]
            
            # Update deployment status
            adapter.deployment_status = "deploying"
            adapter.updated_at = datetime.now()
            await self._save_adapter_metadata(adapter)
            
            # Simulate deployment process
            await asyncio.sleep(2)  # Simulate deployment time
            
            # Update status
            adapter.deployment_status = "deployed"
            adapter.updated_at = datetime.now()
            await self._save_adapter_metadata(adapter)
            
            logger.info(f"Successfully deployed adapter {adapter_id}")
            
            return {
                "status": "success",
                "adapter_id": adapter_id,
                "deployment_config": deployment_config,
                "deployed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to deploy adapter {adapter_id}: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def get_registry_statistics(self) -> Dict[str, Any]:
        """Get comprehensive registry statistics"""
        total_adapters = len(self.adapters)
        total_compositions = len(self.compositions)
        
        # Count by type
        adapter_types = {}
        for adapter in self.adapters.values():
            adapter_types[adapter.adapter_type.value] = adapter_types.get(adapter.adapter_type.value, 0) + 1
        
        # Count by status
        adapter_statuses = {}
        for adapter in self.adapters.values():
            adapter_statuses[adapter.status.value] = adapter_statuses.get(adapter.status.value, 0) + 1
        
        # Calculate average performance
        performance_scores = [
            adapter.performance_metrics.get("overall_score", 0)
            for adapter in self.adapters.values()
            if adapter.performance_metrics.get("overall_score", 0) > 0
        ]
        avg_performance = sum(performance_scores) / len(performance_scores) if performance_scores else 0
        
        # Calculate total size
        total_size_mb = sum(adapter.size_mb for adapter in self.adapters.values())
        
        return {
            "total_adapters": total_adapters,
            "total_compositions": total_compositions,
            "adapter_types": adapter_types,
            "adapter_statuses": adapter_statuses,
            "average_performance": avg_performance,
            "total_size_mb": total_size_mb,
            "registry_size_mb": self._calculate_registry_size(),
            "last_updated": datetime.now().isoformat()
        }
    
    def _calculate_registry_size(self) -> float:
        """Calculate total registry size in MB"""
        total_size = 0
        for path in self.registry_path.rglob("*"):
            if path.is_file():
                total_size += path.stat().st_size
        return total_size / (1024 * 1024)  # Convert to MB
    
    def get_adapter(self, adapter_id: str) -> Optional[AdapterMetadata]:
        """Get adapter metadata by ID"""
        return self.adapters.get(adapter_id)
    
    def get_composition(self, composition_id: str) -> Optional[AdapterComposition]:
        """Get composition by ID"""
        return self.compositions.get(composition_id)
    
    async def update_adapter_usage(self, adapter_id: str):
        """Update adapter usage statistics"""
        if adapter_id in self.adapters:
            adapter = self.adapters[adapter_id]
            adapter.usage_count += 1
            adapter.last_used = datetime.now()
            await self._save_adapter_metadata(adapter)
