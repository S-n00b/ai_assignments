"""
Unified Registry Manager

This module provides unified model registry management with support for both
local (Ollama) and remote (GitHub Models) model sources.
"""

import asyncio
import logging
import sqlite3
import json
from typing import Dict, List, Optional, Any
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .model_objects import UnifiedModelObject

logger = logging.getLogger(__name__)


class UnifiedRegistryManager:
    """Manages the unified model registry."""
    
    def __init__(self, db_path: str = "data/unified_registry.db"):
        """Initialize the unified registry manager."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model_objects: Dict[str, UnifiedModelObject] = {}
        self.logger = logging.getLogger("unified_registry")
        
        # Initialize database
        self._init_database()
        
        # Load existing models
        asyncio.create_task(self._load_existing_models())
    
    def _init_database(self):
        """Initialize SQLite database for unified registry."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create unified_models table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS unified_models (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        version TEXT NOT NULL,
                        ollama_name TEXT,
                        github_models_id TEXT,
                        category TEXT NOT NULL,
                        model_type TEXT NOT NULL,
                        source TEXT NOT NULL,
                        serving_type TEXT NOT NULL,
                        capabilities TEXT,  -- JSON array
                        parameters TEXT,    -- JSON object
                        local_endpoint TEXT,
                        remote_endpoint TEXT,
                        status TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        description TEXT,
                        performance_metrics TEXT,  -- JSON object
                        size_bytes INTEGER DEFAULT 0,
                        digest TEXT,
                        tags TEXT,  -- JSON array
                        UNIQUE(name, version)
                    )
                """)
                
                # Create indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON unified_models(category)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_type ON unified_models(model_type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON unified_models(source)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON unified_models(status)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_ollama_name ON unified_models(ollama_name)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_github_models_id ON unified_models(github_models_id)")
                
                conn.commit()
                
            self.logger.info("Unified registry database initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def _load_existing_models(self):
        """Load existing models from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM unified_models")
                rows = cursor.fetchall()
                
                for row in rows:
                    model_data = self._row_to_dict(row, cursor.description)
                    model = UnifiedModelObject.from_dict(model_data)
                    self.model_objects[model.id] = model
                
                self.logger.info(f"Loaded {len(self.model_objects)} existing models")
                
        except Exception as e:
            self.logger.error(f"Failed to load existing models: {e}")
    
    def _row_to_dict(self, row, description) -> Dict[str, Any]:
        """Convert database row to dictionary."""
        return {
            desc[0]: row[i] 
            for i, desc in enumerate(description)
        }
    
    async def register_model(self, model: UnifiedModelObject) -> bool:
        """
        Register a model in the unified registry.
        
        Args:
            model: UnifiedModelObject to register
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            self.logger.info(f"Registering model: {model.id}")
            
            # Store in memory
            self.model_objects[model.id] = model
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Convert model to database format
                model_data = model.to_dict()
                
                # Convert JSON fields
                model_data["capabilities"] = json.dumps(model_data["capabilities"])
                model_data["parameters"] = json.dumps(model_data["parameters"])
                model_data["performance_metrics"] = json.dumps(model_data["performance_metrics"])
                model_data["tags"] = json.dumps(model_data["tags"])
                
                # Insert or update model
                cursor.execute("""
                    INSERT OR REPLACE INTO unified_models (
                        id, name, version, ollama_name, github_models_id,
                        category, model_type, source, serving_type,
                        capabilities, parameters, local_endpoint, remote_endpoint,
                        status, created_at, updated_at, description,
                        performance_metrics, size_bytes, digest, tags
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_data["id"],
                    model_data["name"],
                    model_data["version"],
                    model_data["ollama_name"],
                    model_data["github_models_id"],
                    model_data["category"],
                    model_data["model_type"],
                    model_data["source"],
                    model_data["serving_type"],
                    model_data["capabilities"],
                    model_data["parameters"],
                    model_data["local_endpoint"],
                    model_data["remote_endpoint"],
                    model_data["status"],
                    model_data["created_at"],
                    model_data["updated_at"],
                    model_data["description"],
                    model_data["performance_metrics"],
                    model_data["size_bytes"],
                    model_data["digest"],
                    model_data["tags"]
                ))
                
                conn.commit()
            
            self.logger.info(f"Successfully registered model: {model.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register model {model.id}: {e}")
            return False
    
    async def get_model(self, model_id: str) -> Optional[UnifiedModelObject]:
        """
        Get a model by ID.
        
        Args:
            model_id: Model ID to retrieve
            
        Returns:
            UnifiedModelObject or None if not found
        """
        return self.model_objects.get(model_id)
    
    async def list_models(
        self,
        category: Optional[str] = None,
        model_type: Optional[str] = None,
        source: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[UnifiedModelObject]:
        """
        List models with optional filtering.
        
        Args:
            category: Filter by category
            model_type: Filter by model type
            source: Filter by source
            status: Filter by status
            
        Returns:
            List of UnifiedModelObject objects
        """
        models = list(self.model_objects.values())
        
        # Apply filters
        if category:
            models = [m for m in models if m.category == category]
        
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        if source:
            models = [m for m in models if m.source == source]
        
        if status:
            models = [m for m in models if m.status == status]
        
        return models
    
    async def update_model(self, model_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update model information.
        
        Args:
            model_id: Model ID to update
            updates: Dictionary of updates
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            if model_id not in self.model_objects:
                self.logger.error(f"Model {model_id} not found")
                return False
            
            model = self.model_objects[model_id]
            
            # Update model fields
            for key, value in updates.items():
                if hasattr(model, key):
                    setattr(model, key, value)
            
            # Update timestamp
            model.updated_at = datetime.now()
            
            # Save to database
            await self.register_model(model)
            
            self.logger.info(f"Updated model: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update model {model_id}: {e}")
            return False
    
    async def delete_model(self, model_id: str) -> bool:
        """
        Delete a model from the registry.
        
        Args:
            model_id: Model ID to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            if model_id not in self.model_objects:
                self.logger.error(f"Model {model_id} not found")
                return False
            
            # Remove from memory
            del self.model_objects[model_id]
            
            # Remove from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM unified_models WHERE id = ?", (model_id,))
                conn.commit()
            
            self.logger.info(f"Deleted model: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete model {model_id}: {e}")
            return False
    
    async def sync_with_sources(self) -> Dict[str, Any]:
        """
        Sync with all model sources.
        
        Returns:
            Dictionary with sync results
        """
        try:
            self.logger.info("Starting sync with all model sources")
            
            sync_results = {
                "ollama_sync": {"success": False, "models_synced": 0},
                "github_models_sync": {"success": False, "models_synced": 0},
                "total_synced": 0,
                "errors": []
            }
            
            # Sync with Ollama (if available)
            try:
                from ..ollama_integration import OllamaRegistrySync, OllamaCategoryLoader, OllamaModelLoader
                
                async with OllamaCategoryLoader() as category_loader:
                    async with OllamaModelLoader(category_loader) as model_loader:
                        ollama_sync = OllamaRegistrySync(model_loader, self)
                        ollama_result = await ollama_sync.sync_models()
                        
                        sync_results["ollama_sync"]["success"] = ollama_result.success
                        sync_results["ollama_sync"]["models_synced"] = ollama_result.synced_models
                        sync_results["ollama_sync"]["errors"] = ollama_result.errors
                        
            except Exception as e:
                error_msg = f"Ollama sync failed: {str(e)}"
                self.logger.warning(error_msg)
                sync_results["errors"].append(error_msg)
            
            # Sync with GitHub Models (if available)
            try:
                from ..github_models_integration import GitHubModelsAPIClient, GitHubModelsLoader
                
                async with GitHubModelsAPIClient() as api_client:
                    model_loader = GitHubModelsLoader(api_client)
                    models = await model_loader.load_all_models()
                    
                    models_synced = 0
                    for github_model in models:
                        unified_model = UnifiedModelObject(
                            id=f"github_{github_model.id.replace('/', '_')}",
                            name=github_model.name,
                            version="1.0.0",
                            github_models_id=github_model.id,
                            category=github_model.category,
                            model_type="base",
                            source="github_models",
                            serving_type="remote",
                            capabilities=github_model.capabilities,
                            remote_endpoint="https://models.github.ai/inference/chat/completions",
                            status="available",
                            description=github_model.description,
                            tags=[github_model.provider]
                        )
                        
                        if await self.register_model(unified_model):
                            models_synced += 1
                    
                    sync_results["github_models_sync"]["success"] = True
                    sync_results["github_models_sync"]["models_synced"] = models_synced
                    
            except Exception as e:
                error_msg = f"GitHub Models sync failed: {str(e)}"
                self.logger.warning(error_msg)
                sync_results["errors"].append(error_msg)
            
            # Calculate total synced
            sync_results["total_synced"] = (
                sync_results["ollama_sync"]["models_synced"] +
                sync_results["github_models_sync"]["models_synced"]
            )
            
            self.logger.info(f"Sync completed: {sync_results['total_synced']} models synced")
            
            return sync_results
            
        except Exception as e:
            self.logger.error(f"Sync with sources failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "total_synced": 0
            }
    
    def get_registry_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        total_models = len(self.model_objects)
        
        # Count by category
        category_counts = {}
        for model in self.model_objects.values():
            category_counts[model.category] = category_counts.get(model.category, 0) + 1
        
        # Count by source
        source_counts = {}
        for model in self.model_objects.values():
            source_counts[model.source] = source_counts.get(model.source, 0) + 1
        
        # Count by status
        status_counts = {}
        for model in self.model_objects.values():
            status_counts[model.status] = status_counts.get(model.status, 0) + 1
        
        return {
            "total_models": total_models,
            "category_counts": category_counts,
            "source_counts": source_counts,
            "status_counts": status_counts,
            "local_models": sum(1 for m in self.model_objects.values() if m.source == "ollama"),
            "remote_models": sum(1 for m in self.model_objects.values() if m.source == "github_models"),
            "experimental_models": sum(1 for m in self.model_objects.values() if m.model_type == "experimental")
        }
    
    def get_available_categories(self) -> List[str]:
        """Get list of available categories."""
        categories = set()
        for model in self.model_objects.values():
            categories.add(model.category)
        return sorted(list(categories))
    
    def get_available_sources(self) -> List[str]:
        """Get list of available sources."""
        sources = set()
        for model in self.model_objects.values():
            sources.add(model.source)
        return sorted(list(sources))
    
    def search_models(self, query: str) -> List[UnifiedModelObject]:
        """
        Search models by name, description, or tags.
        
        Args:
            query: Search query
            
        Returns:
            List of matching UnifiedModelObject objects
        """
        query_lower = query.lower()
        matches = []
        
        for model in self.model_objects.values():
            # Search in name
            if query_lower in model.name.lower():
                matches.append(model)
                continue
            
            # Search in description
            if query_lower in model.description.lower():
                matches.append(model)
                continue
            
            # Search in tags
            for tag in model.tags:
                if query_lower in tag.lower():
                    matches.append(model)
                    break
            
            # Search in capabilities
            for capability in model.capabilities:
                if query_lower in capability.lower():
                    matches.append(model)
                    break
        
        return matches
    
    def clear_registry(self):
        """Clear all models from the registry."""
        try:
            # Clear memory
            self.model_objects.clear()
            
            # Clear database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM unified_models")
                conn.commit()
            
            self.logger.info("Registry cleared")
            
        except Exception as e:
            self.logger.error(f"Failed to clear registry: {e}")
    
    def export_registry(self, file_path: str) -> bool:
        """
        Export registry to JSON file.
        
        Args:
            file_path: Path to export file
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            export_data = {
                "models": [model.to_dict() for model in self.model_objects.values()],
                "exported_at": datetime.now().isoformat(),
                "total_models": len(self.model_objects)
            }
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Registry exported to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export registry: {e}")
            return False
    
    def import_registry(self, file_path: str) -> bool:
        """
        Import registry from JSON file.
        
        Args:
            file_path: Path to import file
            
        Returns:
            True if import successful, False otherwise
        """
        try:
            with open(file_path, 'r') as f:
                import_data = json.load(f)
            
            models_imported = 0
            for model_data in import_data.get("models", []):
                model = UnifiedModelObject.from_dict(model_data)
                self.model_objects[model.id] = model
                models_imported += 1
            
            self.logger.info(f"Imported {models_imported} models from {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import registry: {e}")
            return False
