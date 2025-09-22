"""
Simplified Model Selector for Gradio App

This module provides a streamlined model selection interface that uses the
unified registry system with category-based filtering and local/remote indicators.
"""

import gradio as gr
import logging
from typing import Dict, List, Optional, Any, Tuple
import asyncio

from ..unified_registry import UnifiedRegistryManager, UnifiedModelObject

logger = logging.getLogger(__name__)


class SimplifiedModelSelector:
    """Simplified model selector for Gradio app."""
    
    def __init__(self, registry_manager: UnifiedRegistryManager):
        """Initialize the simplified model selector."""
        self.registry_manager = registry_manager
        self.current_models = []
        self.current_category = "All"
        
    def create_model_selection_interface(self) -> Tuple[gr.Dropdown, gr.Dropdown, gr.Button, gr.HTML]:
        """Create the simplified model selection interface."""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 🎯 Model Selection")
                gr.Markdown("""
                **Streamlined Model Discovery:**
                - Category-based filtering
                - Local (Ollama) and Remote (GitHub Models) indicators
                - Unified model interface
                - Fast model switching
                """)
                
                # Category selection
                category_dropdown = gr.Dropdown(
                    choices=["All", "embedding", "vision", "tools", "thinking", "cloud_text", "cloud_code", "cloud_multimodal"],
                    value="All",
                    label="Model Category",
                    info="Filter models by category"
                )
                
                # Model selection
                model_dropdown = gr.Dropdown(
                    choices=[],
                    label="Select Model",
                    info="Choose a model for evaluation"
                )
                
                # Load button
                load_button = gr.Button("🔄 Load Model", variant="primary")
                
                # Model info display
                model_info = gr.HTML(
                    value="<div style='text-align: center; padding: 2rem; color: #666;'>Select a category and model to see details</div>"
                )
                
        return category_dropdown, model_dropdown, load_button, model_info
    
    async def update_model_list(self, category: str) -> gr.Dropdown:
        """Update model list based on selected category."""
        try:
            if category == "All":
                models = await self.registry_manager.list_models()
            else:
                models = await self.registry_manager.list_models(category=category)
            
            self.current_models = models
            self.current_category = category
            
            if not models:
                choices = []
                value = None
            else:
                choices = []
                for model in models:
                    # Create display name with source indicator
                    source_indicator = "🖥️" if model.is_local() else "☁️" if model.is_remote() else "🔄"
                    display_name = f"{source_indicator} {model.name} ({model.get_source_display_name()})"
                    choices.append((display_name, model.id))
                
                value = models[0].id if models else None
            
            return gr.Dropdown(
                choices=choices,
                value=value,
                label="Select Model"
            )
            
        except Exception as e:
            logger.error(f"Error updating model list: {e}")
            return gr.Dropdown(choices=[], value=None, label="Select Model")
    
    async def load_selected_model(self, model_id: str) -> str:
        """Load the selected model and display information."""
        try:
            if not model_id:
                return "<div style='text-align: center; padding: 2rem; color: #666;'>No model selected</div>"
            
            model = await self.registry_manager.get_model(model_id)
            if not model:
                return f"<div style='color: red; padding: 1rem;'>Model {model_id} not found</div>"
            
            # Create model info HTML
            info_html = self._create_model_info_html(model)
            
            return info_html
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            return f"<div style='color: red; padding: 1rem;'>Error loading model: {str(e)}</div>"
    
    def _create_model_info_html(self, model: UnifiedModelObject) -> str:
        """Create HTML display for model information."""
        # Status indicator
        status_color = {
            "available": "#4CAF50",
            "busy": "#FF9800", 
            "error": "#F44336",
            "unavailable": "#9E9E9E"
        }.get(model.status, "#9E9E9E")
        
        # Source indicator
        source_indicator = "🖥️ Local" if model.is_local() else "☁️ Remote" if model.is_remote() else "🔄 Hybrid"
        
        # Capabilities display
        capabilities_html = ""
        if model.capabilities:
            capabilities_html = f"""
            <div style='margin-top: 1rem;'>
                <strong>Capabilities:</strong><br>
                {', '.join(model.capabilities)}
            </div>
            """
        
        # Performance metrics
        metrics_html = ""
        if model.performance_metrics:
            metrics_html = f"""
            <div style='margin-top: 1rem;'>
                <strong>Performance:</strong><br>
                {', '.join([f"{k}: {v}" for k, v in model.performance_metrics.items()])}
            </div>
            """
        
        info_html = f"""
        <div style='padding: 1.5rem; border: 1px solid #e0e0e0; border-radius: 8px; background: #f9f9f9;'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;'>
                <h3 style='margin: 0; color: #333;'>{model.name}</h3>
                <span style='background: {status_color}; color: white; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.8rem;'>
                    {model.status.upper()}
                </span>
            </div>
            
            <div style='color: #666; margin-bottom: 1rem;'>
                <strong>Category:</strong> {model.get_category_display_name()}<br>
                <strong>Source:</strong> {source_indicator}<br>
                <strong>Type:</strong> {model.model_type.title()}<br>
                <strong>Serving:</strong> {model.get_serving_type_display_name()}
            </div>
            
            {f"<div style='margin-top: 1rem; color: #666;'><strong>Description:</strong> {model.description}</div>" if model.description else ""}
            
            {capabilities_html}
            {metrics_html}
            
            <div style='margin-top: 1rem; font-size: 0.8rem; color: #999;'>
                <strong>Model ID:</strong> {model.id}<br>
                <strong>Version:</strong> {model.version}<br>
                <strong>Last Updated:</strong> {model.updated_at.strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
        """
        
        return info_html
    
    async def get_model_for_evaluation(self, model_id: str) -> Optional[UnifiedModelObject]:
        """Get model object for evaluation."""
        try:
            if not model_id:
                return None
            
            model = await self.registry_manager.get_model(model_id)
            return model
            
        except Exception as e:
            logger.error(f"Error getting model for evaluation: {e}")
            return None
    
    def get_available_categories(self) -> List[str]:
        """Get list of available categories."""
        return self.registry_manager.get_available_categories()
    
    async def search_models(self, query: str) -> List[UnifiedModelObject]:
        """Search models by query."""
        try:
            return self.registry_manager.search_models(query)
        except Exception as e:
            logger.error(f"Error searching models: {e}")
            return []
    
    async def refresh_models(self) -> Tuple[gr.Dropdown, str]:
        """Refresh models from all sources."""
        try:
            logger.info("Refreshing models from all sources...")
            
            # Sync with all sources
            sync_result = await self.registry_manager.sync_with_sources()
            
            # Update model list for current category
            updated_dropdown = await self.update_model_list(self.current_category)
            
            # Create refresh status message
            total_synced = sync_result.get("total_synced", 0)
            ollama_synced = sync_result.get("ollama_sync", {}).get("models_synced", 0)
            github_synced = sync_result.get("github_models_sync", {}).get("models_synced", 0)
            
            status_message = f"""
            <div style='padding: 1rem; background: #e8f5e8; border: 1px solid #4CAF50; border-radius: 4px; margin: 1rem 0;'>
                <strong>✅ Models Refreshed Successfully</strong><br>
                Total models synced: {total_synced}<br>
                Local (Ollama): {ollama_synced} models<br>
                Remote (GitHub Models): {github_synced} models
            </div>
            """
            
            return updated_dropdown, status_message
            
        except Exception as e:
            logger.error(f"Error refreshing models: {e}")
            error_message = f"""
            <div style='padding: 1rem; background: #ffebee; border: 1px solid #F44336; border-radius: 4px; margin: 1rem 0;'>
                <strong>❌ Refresh Failed</strong><br>
                Error: {str(e)}
            </div>
            """
            
            return gr.Dropdown(choices=[], value=None, label="Select Model"), error_message
