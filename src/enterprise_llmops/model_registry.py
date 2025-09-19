"""
Enterprise Model Registry for LLMOps

This module provides a comprehensive model registry for enterprise AI model
lifecycle management, including versioning, metadata tracking, deployment
management, and compliance features.

Key Features:
- Model versioning and lineage tracking
- Metadata management and search
- Deployment pipeline integration
- Performance monitoring and drift detection
- Compliance and audit logging
- Integration with Ollama and other model serving platforms
"""

import asyncio
import json
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import yaml
import uuid
from enum import Enum
import aiofiles
import aiohttp
from dataclasses_json import dataclass_json


class ModelStatus(Enum):
    """Model lifecycle status."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class ModelType(Enum):
    """Model type classification."""
    LLM = "llm"
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"
    GENERATION = "generation"
    CODE = "code"
    MULTIMODAL = "multimodal"


@dataclass_json
@dataclass
class ModelMetadata:
    """Model metadata information."""
    name: str
    version: str
    model_type: ModelType
    description: str
    tags: List[str] = field(default_factory=list)
    framework: str = "ollama"
    size_bytes: int = 0
    parameters: int = 0
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: ModelStatus = ModelStatus.DEVELOPMENT
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    compliance_tags: List[str] = field(default_factory=list)
    custom_attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass_json
@dataclass
class ModelVersion:
    """Model version information."""
    model_name: str
    version: str
    metadata: ModelMetadata
    model_path: str
    checksum: str
    dependencies: List[str] = field(default_factory=list)
    deployment_config: Dict[str, Any] = field(default_factory=dict)
    performance_benchmarks: Dict[str, float] = field(default_factory=dict)
    validation_results: Dict[str, Any] = field(default_factory=dict)


@dataclass_json
@dataclass
class Deployment:
    """Model deployment information."""
    deployment_id: str
    model_name: str
    version: str
    environment: str  # "dev", "staging", "production"
    endpoint: str
    status: str  # "active", "inactive", "error"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    traffic_percentage: float = 100.0
    health_checks: List[Dict[str, Any]] = field(default_factory=list)
    monitoring_config: Dict[str, Any] = field(default_factory=dict)


class EnterpriseModelRegistry:
    """
    Enterprise model registry for comprehensive model lifecycle management.
    
    This class provides model registration, versioning, deployment tracking,
    and compliance features for enterprise AI model operations.
    """
    
    def __init__(self, registry_path: str = "data/model_registry"):
        """Initialize the enterprise model registry."""
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.registry_path / "registry.db"
        self.models_path = self.registry_path / "models"
        self.models_path.mkdir(exist_ok=True)
        
        self.logger = self._setup_logging()
        self.db_connection = None
        
        # Initialize database
        self._init_database()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for model registry."""
        logger = logging.getLogger("model_registry")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _init_database(self):
        """Initialize SQLite database for model registry."""
        try:
            self.db_connection = sqlite3.connect(str(self.db_path))
            cursor = self.db_connection.cursor()
            
            # Create models table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    description TEXT,
                    tags TEXT,  -- JSON array
                    framework TEXT,
                    size_bytes INTEGER,
                    parameters INTEGER,
                    created_by TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    status TEXT,
                    performance_metrics TEXT,  -- JSON object
                    compliance_tags TEXT,  -- JSON array
                    custom_attributes TEXT,  -- JSON object
                    model_path TEXT,
                    checksum TEXT,
                    dependencies TEXT,  -- JSON array
                    deployment_config TEXT,  -- JSON object
                    performance_benchmarks TEXT,  -- JSON object
                    validation_results TEXT,  -- JSON object
                    UNIQUE(name, version)
                )
            """)
            
            # Create deployments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS deployments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    deployment_id TEXT UNIQUE NOT NULL,
                    model_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    endpoint TEXT,
                    status TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    traffic_percentage REAL,
                    health_checks TEXT,  -- JSON array
                    monitoring_config TEXT  -- JSON object
                )
            """)
            
            # Create audit_log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    model_name TEXT,
                    version TEXT,
                    user_id TEXT,
                    timestamp TIMESTAMP,
                    details TEXT  -- JSON object
                )
            """)
            
            self.db_connection.commit()
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA-256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    async def register_model(
        self,
        model_name: str,
        model_type: ModelType,
        description: str,
        model_path: Optional[str] = None,
        version: str = "1.0.0",
        tags: Optional[List[str]] = None,
        created_by: str = "system",
        **kwargs
    ) -> ModelVersion:
        """Register a new model version in the registry."""
        try:
            # Generate version if not provided
            if version == "1.0.0":
                existing_versions = await self.list_model_versions(model_name)
                if existing_versions:
                    latest_version = max(existing_versions, key=lambda v: v["version"])
                    version = self._increment_version(latest_version["version"])
            
            # Create model metadata
            metadata = ModelMetadata(
                name=model_name,
                version=version,
                model_type=model_type,
                description=description,
                tags=tags or [],
                created_by=created_by,
                **kwargs
            )
            
            # Calculate checksum if model path provided
            checksum = ""
            if model_path and Path(model_path).exists():
                checksum = self._calculate_checksum(model_path)
                
                # Copy model to registry storage
                model_filename = f"{model_name}_{version}.bin"
                registry_model_path = self.models_path / model_filename
                
                # In a real implementation, you would copy the model file here
                # For now, we'll just store the path
                registry_model_path = model_path
            
            # Create model version
            model_version = ModelVersion(
                model_name=model_name,
                version=version,
                metadata=metadata,
                model_path=registry_model_path,
                checksum=checksum
            )
            
            # Store in database
            await self._store_model_version(model_version)
            
            # Log audit trail
            await self._log_audit("register_model", model_name, version, created_by, {
                "model_type": model_type.value,
                "description": description,
                "tags": tags
            })
            
            self.logger.info(f"Registered model {model_name} version {version}")
            return model_version
            
        except Exception as e:
            self.logger.error(f"Failed to register model {model_name}: {e}")
            raise
    
    def _increment_version(self, current_version: str) -> str:
        """Increment semantic version number."""
        try:
            parts = current_version.split('.')
            if len(parts) == 3:
                major, minor, patch = parts
                patch = str(int(patch) + 1)
                return f"{major}.{minor}.{patch}"
            else:
                return f"{current_version}.1"
        except Exception:
            return "1.0.0"
    
    async def _store_model_version(self, model_version: ModelVersion):
        """Store model version in database."""
        try:
            cursor = self.db_connection.cursor()
            
            cursor.execute("""
                INSERT INTO models (
                    name, version, model_type, description, tags, framework,
                    size_bytes, parameters, created_by, created_at, updated_at,
                    status, performance_metrics, compliance_tags, custom_attributes,
                    model_path, checksum, dependencies, deployment_config,
                    performance_benchmarks, validation_results
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_version.model_name,
                model_version.version,
                model_version.metadata.model_type.value,
                model_version.metadata.description,
                json.dumps(model_version.metadata.tags),
                model_version.metadata.framework,
                model_version.metadata.size_bytes,
                model_version.metadata.parameters,
                model_version.metadata.created_by,
                model_version.metadata.created_at.isoformat(),
                model_version.metadata.updated_at.isoformat(),
                model_version.metadata.status.value,
                json.dumps(model_version.metadata.performance_metrics),
                json.dumps(model_version.metadata.compliance_tags),
                json.dumps(model_version.metadata.custom_attributes),
                model_version.model_path,
                model_version.checksum,
                json.dumps(model_version.dependencies),
                json.dumps(model_version.deployment_config),
                json.dumps(model_version.performance_benchmarks),
                json.dumps(model_version.validation_results)
            ))
            
            self.db_connection.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to store model version: {e}")
            raise
    
    async def list_models(
        self,
        model_type: Optional[ModelType] = None,
        status: Optional[ModelStatus] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List models with optional filtering."""
        try:
            cursor = self.db_connection.cursor()
            
            # Build query
            query = "SELECT * FROM models WHERE 1=1"
            params = []
            
            if model_type:
                query += " AND model_type = ?"
                params.append(model_type.value)
            
            if status:
                query += " AND status = ?"
                params.append(status.value)
            
            if tags:
                for tag in tags:
                    query += " AND tags LIKE ?"
                    params.append(f'%"{tag}"%')
            
            query += " ORDER BY name, version DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to dictionaries
            models = []
            for row in rows:
                model_data = {
                    "id": row[0],
                    "name": row[1],
                    "version": row[2],
                    "model_type": row[3],
                    "description": row[4],
                    "tags": json.loads(row[5]) if row[5] else [],
                    "framework": row[6],
                    "size_bytes": row[7],
                    "parameters": row[8],
                    "created_by": row[9],
                    "created_at": row[10],
                    "updated_at": row[11],
                    "status": row[12],
                    "performance_metrics": json.loads(row[13]) if row[13] else {},
                    "compliance_tags": json.loads(row[14]) if row[14] else [],
                    "custom_attributes": json.loads(row[15]) if row[15] else {},
                    "model_path": row[16],
                    "checksum": row[17],
                    "dependencies": json.loads(row[18]) if row[18] else [],
                    "deployment_config": json.loads(row[19]) if row[19] else {},
                    "performance_benchmarks": json.loads(row[20]) if row[20] else {},
                    "validation_results": json.loads(row[21]) if row[21] else {}
                }
                models.append(model_data)
            
            return models
            
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            return []
    
    async def list_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """List all versions of a specific model."""
        try:
            cursor = self.db_connection.cursor()
            
            cursor.execute("""
                SELECT * FROM models WHERE name = ? ORDER BY version DESC
            """, (model_name,))
            
            rows = cursor.fetchall()
            
            versions = []
            for row in rows:
                version_data = {
                    "id": row[0],
                    "name": row[1],
                    "version": row[2],
                    "model_type": row[3],
                    "description": row[4],
                    "tags": json.loads(row[5]) if row[5] else [],
                    "framework": row[6],
                    "size_bytes": row[7],
                    "parameters": row[8],
                    "created_by": row[9],
                    "created_at": row[10],
                    "updated_at": row[11],
                    "status": row[12],
                    "performance_metrics": json.loads(row[13]) if row[13] else {},
                    "compliance_tags": json.loads(row[14]) if row[14] else [],
                    "custom_attributes": json.loads(row[15]) if row[15] else {},
                    "model_path": row[16],
                    "checksum": row[17],
                    "dependencies": json.loads(row[18]) if row[18] else [],
                    "deployment_config": json.loads(row[19]) if row[19] else {},
                    "performance_benchmarks": json.loads(row[20]) if row[20] else {},
                    "validation_results": json.loads(row[21]) if row[21] else {}
                }
                versions.append(version_data)
            
            return versions
            
        except Exception as e:
            self.logger.error(f"Failed to list versions for model {model_name}: {e}")
            return []
    
    async def get_model_version(self, model_name: str, version: str) -> Optional[Dict[str, Any]]:
        """Get specific model version."""
        try:
            cursor = self.db_connection.cursor()
            
            cursor.execute("""
                SELECT * FROM models WHERE name = ? AND version = ?
            """, (model_name, version))
            
            row = cursor.fetchone()
            
            if row:
                return {
                    "id": row[0],
                    "name": row[1],
                    "version": row[2],
                    "model_type": row[3],
                    "description": row[4],
                    "tags": json.loads(row[5]) if row[5] else [],
                    "framework": row[6],
                    "size_bytes": row[7],
                    "parameters": row[8],
                    "created_by": row[9],
                    "created_at": row[10],
                    "updated_at": row[11],
                    "status": row[12],
                    "performance_metrics": json.loads(row[13]) if row[13] else {},
                    "compliance_tags": json.loads(row[14]) if row[14] else [],
                    "custom_attributes": json.loads(row[15]) if row[15] else {},
                    "model_path": row[16],
                    "checksum": row[17],
                    "dependencies": json.loads(row[18]) if row[18] else [],
                    "deployment_config": json.loads(row[19]) if row[19] else {},
                    "performance_benchmarks": json.loads(row[20]) if row[20] else {},
                    "validation_results": json.loads(row[21]) if row[21] else {}
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get model version {model_name}:{version}: {e}")
            return None
    
    async def update_model_status(
        self,
        model_name: str,
        version: str,
        status: ModelStatus,
        user_id: str = "system"
    ) -> bool:
        """Update model status."""
        try:
            cursor = self.db_connection.cursor()
            
            cursor.execute("""
                UPDATE models SET status = ?, updated_at = ? 
                WHERE name = ? AND version = ?
            """, (status.value, datetime.now().isoformat(), model_name, version))
            
            if cursor.rowcount > 0:
                self.db_connection.commit()
                
                # Log audit trail
                await self._log_audit("update_status", model_name, version, user_id, {
                    "new_status": status.value
                })
                
                self.logger.info(f"Updated status for {model_name}:{version} to {status.value}")
                return True
            else:
                self.logger.warning(f"Model {model_name}:{version} not found")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to update model status: {e}")
            return False
    
    async def deploy_model(
        self,
        model_name: str,
        version: str,
        environment: str,
        endpoint: str,
        user_id: str = "system",
        **deployment_config
    ) -> Deployment:
        """Deploy a model version."""
        try:
            # Check if model exists
            model_version = await self.get_model_version(model_name, version)
            if not model_version:
                raise ValueError(f"Model {model_name}:{version} not found")
            
            # Create deployment
            deployment = Deployment(
                deployment_id=str(uuid.uuid4()),
                model_name=model_name,
                version=version,
                environment=environment,
                endpoint=endpoint,
                status="active",
                monitoring_config=deployment_config.get("monitoring", {})
            )
            
            # Store deployment
            cursor = self.db_connection.cursor()
            
            cursor.execute("""
                INSERT INTO deployments (
                    deployment_id, model_name, version, environment, endpoint,
                    status, created_at, updated_at, traffic_percentage,
                    health_checks, monitoring_config
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                deployment.deployment_id,
                deployment.model_name,
                deployment.version,
                deployment.environment,
                deployment.endpoint,
                deployment.status,
                deployment.created_at.isoformat(),
                deployment.updated_at.isoformat(),
                deployment.traffic_percentage,
                json.dumps(deployment.health_checks),
                json.dumps(deployment.monitoring_config)
            ))
            
            self.db_connection.commit()
            
            # Log audit trail
            await self._log_audit("deploy_model", model_name, version, user_id, {
                "environment": environment,
                "endpoint": endpoint,
                "deployment_id": deployment.deployment_id
            })
            
            self.logger.info(f"Deployed {model_name}:{version} to {environment}")
            return deployment
            
        except Exception as e:
            self.logger.error(f"Failed to deploy model: {e}")
            raise
    
    async def list_deployments(
        self,
        model_name: Optional[str] = None,
        environment: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List model deployments."""
        try:
            cursor = self.db_connection.cursor()
            
            # Build query
            query = "SELECT * FROM deployments WHERE 1=1"
            params = []
            
            if model_name:
                query += " AND model_name = ?"
                params.append(model_name)
            
            if environment:
                query += " AND environment = ?"
                params.append(environment)
            
            if status:
                query += " AND status = ?"
                params.append(status)
            
            query += " ORDER BY created_at DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            deployments = []
            for row in rows:
                deployment_data = {
                    "id": row[0],
                    "deployment_id": row[1],
                    "model_name": row[2],
                    "version": row[3],
                    "environment": row[4],
                    "endpoint": row[5],
                    "status": row[6],
                    "created_at": row[7],
                    "updated_at": row[8],
                    "traffic_percentage": row[9],
                    "health_checks": json.loads(row[10]) if row[10] else [],
                    "monitoring_config": json.loads(row[11]) if row[11] else {}
                }
                deployments.append(deployment_data)
            
            return deployments
            
        except Exception as e:
            self.logger.error(f"Failed to list deployments: {e}")
            return []
    
    async def _log_audit(
        self,
        action: str,
        model_name: Optional[str],
        version: Optional[str],
        user_id: str,
        details: Dict[str, Any]
    ):
        """Log audit trail."""
        try:
            cursor = self.db_connection.cursor()
            
            cursor.execute("""
                INSERT INTO audit_log (action, model_name, version, user_id, timestamp, details)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                action,
                model_name,
                version,
                user_id,
                datetime.now().isoformat(),
                json.dumps(details)
            ))
            
            self.db_connection.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to log audit trail: {e}")
    
    async def get_audit_log(
        self,
        model_name: Optional[str] = None,
        action: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get audit log entries."""
        try:
            cursor = self.db_connection.cursor()
            
            # Build query
            query = "SELECT * FROM audit_log WHERE 1=1"
            params = []
            
            if model_name:
                query += " AND model_name = ?"
                params.append(model_name)
            
            if action:
                query += " AND action = ?"
                params.append(action)
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            audit_entries = []
            for row in rows:
                entry = {
                    "id": row[0],
                    "action": row[1],
                    "model_name": row[2],
                    "version": row[3],
                    "user_id": row[4],
                    "timestamp": row[5],
                    "details": json.loads(row[6]) if row[6] else {}
                }
                audit_entries.append(entry)
            
            return audit_entries
            
        except Exception as e:
            self.logger.error(f"Failed to get audit log: {e}")
            return []
    
    async def search_models(
        self,
        query: str,
        model_type: Optional[ModelType] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search models by name, description, or tags."""
        try:
            cursor = self.db_connection.cursor()
            
            # Build search query
            search_query = """
                SELECT * FROM models WHERE (
                    name LIKE ? OR 
                    description LIKE ? OR 
                    tags LIKE ?
                )
            """
            params = [f"%{query}%", f"%{query}%", f"%{query}%"]
            
            if model_type:
                search_query += " AND model_type = ?"
                params.append(model_type.value)
            
            if tags:
                for tag in tags:
                    search_query += " AND tags LIKE ?"
                    params.append(f'%"{tag}"%')
            
            search_query += " ORDER BY name, version DESC"
            
            cursor.execute(search_query, params)
            rows = cursor.fetchall()
            
            # Convert to dictionaries
            models = []
            for row in rows:
                model_data = {
                    "id": row[0],
                    "name": row[1],
                    "version": row[2],
                    "model_type": row[3],
                    "description": row[4],
                    "tags": json.loads(row[5]) if row[5] else [],
                    "framework": row[6],
                    "size_bytes": row[7],
                    "parameters": row[8],
                    "created_by": row[9],
                    "created_at": row[10],
                    "updated_at": row[11],
                    "status": row[12],
                    "performance_metrics": json.loads(row[13]) if row[13] else {},
                    "compliance_tags": json.loads(row[14]) if row[14] else [],
                    "custom_attributes": json.loads(row[15]) if row[15] else {},
                    "model_path": row[16],
                    "checksum": row[17],
                    "dependencies": json.loads(row[18]) if row[18] else [],
                    "deployment_config": json.loads(row[19]) if row[19] else {},
                    "performance_benchmarks": json.loads(row[20]) if row[20] else {},
                    "validation_results": json.loads(row[21]) if row[21] else {}
                }
                models.append(model_data)
            
            return models
            
        except Exception as e:
            self.logger.error(f"Failed to search models: {e}")
            return []
    
    def close(self):
        """Close database connection."""
        if self.db_connection:
            self.db_connection.close()
            self.logger.info("Database connection closed")
