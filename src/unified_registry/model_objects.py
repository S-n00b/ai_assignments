"""
Unified Model Objects

This module defines the unified model object structure for all model types,
providing a common interface for local (Ollama) and remote (GitHub Models) models.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class ModelSource(Enum):
    """Model source types."""
    OLLAMA = "ollama"
    GITHUB_MODELS = "github_models"
    MLFLOW = "mlflow"
    EXTERNAL = "external"


class ModelType(Enum):
    """Model type classification."""
    BASE = "base"
    EXPERIMENTAL = "experimental"
    VARIANT = "variant"


class ServingType(Enum):
    """Model serving types."""
    LOCAL = "local"
    REMOTE = "remote"
    HYBRID = "hybrid"


class ModelStatus(Enum):
    """Model status."""
    AVAILABLE = "available"
    BUSY = "busy"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


@dataclass
class ModelCapability:
    """Model capability specification."""
    name: str
    type: str  # text, vision, embedding, function_calling
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)


@dataclass
class ModelServingConfig:
    """Model serving configuration."""
    endpoint: str
    protocol: str = "http"  # http, grpc, websocket
    timeout: int = 30
    max_concurrent_requests: int = 10
    health_check_interval: int = 60
    retry_attempts: int = 3


@dataclass
class UnifiedModelObject:
    """Unified model object for all model types."""
    
    # Core identification
    id: str
    name: str
    version: str
    
    # Model source integration
    ollama_name: Optional[str] = None  # For local models
    github_models_id: Optional[str] = None  # For remote models
    category: str = "general"  # embedding, vision, tools, thinking, cloud_text, etc.
    
    # Model characteristics
    model_type: str = "base"  # base, experimental, variant
    source: str = "external"  # ollama, github_models, mlflow, external
    serving_type: str = "local"  # local, remote, hybrid
    
    # Capabilities
    capabilities: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Serving information
    local_endpoint: Optional[str] = None  # Ollama endpoint
    remote_endpoint: Optional[str] = None  # GitHub Models API endpoint
    status: str = "available"  # available, busy, error
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    description: str = ""
    
    # Performance metrics
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Additional metadata
    size_bytes: int = 0
    digest: str = ""
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "ollama_name": self.ollama_name,
            "github_models_id": self.github_models_id,
            "category": self.category,
            "model_type": self.model_type,
            "source": self.source,
            "serving_type": self.serving_type,
            "capabilities": self.capabilities,
            "parameters": self.parameters,
            "local_endpoint": self.local_endpoint,
            "remote_endpoint": self.remote_endpoint,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "description": self.description,
            "performance_metrics": self.performance_metrics,
            "size_bytes": self.size_bytes,
            "digest": self.digest,
            "tags": self.tags
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedModelObject':
        """Create from dictionary."""
        # Handle datetime fields
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()
        
        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        elif updated_at is None:
            updated_at = datetime.now()
        
        return cls(
            id=data["id"],
            name=data["name"],
            version=data["version"],
            ollama_name=data.get("ollama_name"),
            github_models_id=data.get("github_models_id"),
            category=data.get("category", "general"),
            model_type=data.get("model_type", "base"),
            source=data.get("source", "external"),
            serving_type=data.get("serving_type", "local"),
            capabilities=data.get("capabilities", []),
            parameters=data.get("parameters", {}),
            local_endpoint=data.get("local_endpoint"),
            remote_endpoint=data.get("remote_endpoint"),
            status=data.get("status", "available"),
            created_at=created_at,
            updated_at=updated_at,
            description=data.get("description", ""),
            performance_metrics=data.get("performance_metrics", {}),
            size_bytes=data.get("size_bytes", 0),
            digest=data.get("digest", ""),
            tags=data.get("tags", [])
        )
    
    def get_serving_endpoint(self) -> Optional[str]:
        """Get the serving endpoint for this model."""
        if self.serving_type == "local" and self.local_endpoint:
            return self.local_endpoint
        elif self.serving_type == "remote" and self.remote_endpoint:
            return self.remote_endpoint
        elif self.serving_type == "hybrid":
            # Prefer local if available, otherwise remote
            return self.local_endpoint or self.remote_endpoint
        return None
    
    def is_available(self) -> bool:
        """Check if model is available for serving."""
        return self.status == "available"
    
    def is_local(self) -> bool:
        """Check if model is served locally."""
        return self.source == "ollama" or self.serving_type in ["local", "hybrid"]
    
    def is_remote(self) -> bool:
        """Check if model is served remotely."""
        return self.source == "github_models" or self.serving_type in ["remote", "hybrid"]
    
    def update_status(self, status: str):
        """Update model status."""
        self.status = status
        self.updated_at = datetime.now()
    
    def add_capability(self, capability: str):
        """Add a capability to the model."""
        if capability not in self.capabilities:
            self.capabilities.append(capability)
            self.updated_at = datetime.now()
    
    def remove_capability(self, capability: str):
        """Remove a capability from the model."""
        if capability in self.capabilities:
            self.capabilities.remove(capability)
            self.updated_at = datetime.now()
    
    def update_parameters(self, parameters: Dict[str, Any]):
        """Update model parameters."""
        self.parameters.update(parameters)
        self.updated_at = datetime.now()
    
    def update_performance_metrics(self, metrics: Dict[str, float]):
        """Update performance metrics."""
        self.performance_metrics.update(metrics)
        self.updated_at = datetime.now()
    
    def add_tag(self, tag: str):
        """Add a tag to the model."""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.now()
    
    def remove_tag(self, tag: str):
        """Remove a tag from the model."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.now()
    
    def get_category_display_name(self) -> str:
        """Get human-readable category name."""
        category_mapping = {
            "embedding": "Text Embedding",
            "vision": "Vision & Multimodal",
            "tools": "Function Calling",
            "thinking": "Reasoning & Chat",
            "cloud_text": "Cloud Text Models",
            "cloud_code": "Cloud Code Models",
            "cloud_multimodal": "Cloud Multimodal",
            "cloud_general": "Cloud General",
            "general": "General Purpose"
        }
        return category_mapping.get(self.category, self.category.title())
    
    def get_source_display_name(self) -> str:
        """Get human-readable source name."""
        source_mapping = {
            "ollama": "Local (Ollama)",
            "github_models": "Remote (GitHub Models)",
            "mlflow": "Experimental (MLflow)",
            "external": "External"
        }
        return source_mapping.get(self.source, self.source.title())
    
    def get_serving_type_display_name(self) -> str:
        """Get human-readable serving type name."""
        serving_type_mapping = {
            "local": "Local Serving",
            "remote": "Remote Serving",
            "hybrid": "Hybrid Serving"
        }
        return serving_type_mapping.get(self.serving_type, self.serving_type.title())
    
    def __str__(self) -> str:
        """String representation of the model."""
        return f"UnifiedModelObject(id='{self.id}', name='{self.name}', source='{self.source}', category='{self.category}')"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"UnifiedModelObject("
                f"id='{self.id}', "
                f"name='{self.name}', "
                f"version='{self.version}', "
                f"source='{self.source}', "
                f"category='{self.category}', "
                f"serving_type='{self.serving_type}', "
                f"status='{self.status}')")


# Factory functions for creating unified model objects

def create_ollama_model(ollama_model_data: Dict[str, Any]) -> UnifiedModelObject:
    """Create unified model object from Ollama model data."""
    return UnifiedModelObject(
        id=f"ollama_{ollama_model_data['name'].replace(':', '_')}",
        name=ollama_model_data["name"],
        version="1.0.0",
        ollama_name=ollama_model_data["name"],
        category=ollama_model_data.get("category", "thinking"),
        model_type="base",
        source="ollama",
        serving_type="local",
        capabilities=ollama_model_data.get("capabilities", []),
        parameters=ollama_model_data.get("parameters", {}),
        local_endpoint="http://localhost:11434/api/generate",
        status="available",
        description=f"Ollama model: {ollama_model_data['name']}",
        size_bytes=ollama_model_data.get("size", 0),
        digest=ollama_model_data.get("digest", "")
    )


def create_github_models_model(github_model_data: Dict[str, Any]) -> UnifiedModelObject:
    """Create unified model object from GitHub Models data."""
    return UnifiedModelObject(
        id=f"github_{github_model_data['id'].replace('/', '_')}",
        name=github_model_data["name"],
        version="1.0.0",
        github_models_id=github_model_data["id"],
        category=github_model_data.get("category", "cloud_general"),
        model_type="base",
        source="github_models",
        serving_type="remote",
        capabilities=github_model_data.get("capabilities", []),
        parameters={},
        remote_endpoint="https://models.github.ai/inference/chat/completions",
        status="available",
        description=github_model_data.get("description", ""),
        tags=[github_model_data.get("provider", "unknown")]
    )


def create_experimental_model(
    base_model: UnifiedModelObject,
    experiment_config: Dict[str, Any]
) -> UnifiedModelObject:
    """Create experimental model from base model."""
    return UnifiedModelObject(
        id=f"exp_{base_model.id}_{int(datetime.now().timestamp())}",
        name=f"{base_model.name}_exp",
        version="1.0.0",
        category=base_model.category,
        model_type="experimental",
        source="mlflow",
        serving_type=base_model.serving_type,
        capabilities=base_model.capabilities.copy(),
        parameters=base_model.parameters.copy(),
        local_endpoint=base_model.local_endpoint,
        remote_endpoint=base_model.remote_endpoint,
        status="available",
        description=f"Experimental variant of {base_model.name}",
        tags=base_model.tags + ["experimental"]
    )
