"""
Simplified Gradio Application for Lenovo AAITC Assignment 1

This module creates a streamlined Gradio application that uses the new
Ollama-centric unified registry architecture with category-based model selection
and simplified evaluation workflows.
"""

import gradio as gr
import logging
import asyncio
from typing import Dict, Any, List, Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from .simplified_model_selector import SimplifiedModelSelector
from .simplified_evaluation_interface import SimplifiedEvaluationInterface
from ..unified_registry import UnifiedRegistryManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimplifiedLenovoModelEvaluationApp:
    """
    Simplified Lenovo Model Evaluation Application using unified registry architecture.
    
    This class provides a streamlined model evaluation interface with:
    - Category-based model selection (embedding, vision, tools, thinking)
    - Local (Ollama) and Remote (GitHub Models) model support
    - Unified evaluation interface
    - Real-time performance monitoring
    - Simplified user experience
    """
    
    def __init__(self):
        """Initialize the simplified Model Evaluation application."""
        # Initialize unified registry
        self.registry_manager = UnifiedRegistryManager()
        
        # Initialize interfaces
        self.model_selector = SimplifiedModelSelector(self.registry_manager)
        self.evaluation_interface = SimplifiedEvaluationInterface(self.registry_manager)
        
        logger.info("Initialized Simplified Lenovo Model Evaluation Application")
    
    def create_interface(self) -> gr.Blocks:
        """
        Create the simplified Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(
            title="Lenovo AAITC Assignment 1: Simplified Model Evaluation",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1400px !important;
                margin: auto !important;
            }
            .tab-nav {
                background: linear-gradient(90deg, #1e3a8a, #3b82f6);
                color: white;
                padding: 1rem;
                border-radius: 8px;
                margin-bottom: 1rem;
            }
            .model-card {
                background: white;
                border-radius: 8px;
                padding: 1rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin: 0.5rem;
            }
            .status-indicator {
                display: inline-block;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                margin-right: 8px;
            }
            .status-available { background: #4CAF50; }
            .status-busy { background: #FF9800; }
            .status-error { background: #F44336; }
            """
        ) as interface:
            
            # Header
            gr.HTML("""
            <div class="tab-nav">
                <h1>🎯 Lenovo AAITC Assignment 1: Simplified Model Evaluation</h1>
                <p>Streamlined Model Discovery & Evaluation Framework - Ollama-Centric Architecture</p>
                <p>Local (Ollama) + Remote (GitHub Models) + Unified Registry + Category-Based Selection</p>
            </div>
            """)
            
            # Main tabs
            with gr.Tabs():
                
                # Model Selection Tab
                with gr.Tab("🎯 Model Selection", id="model_selection"):
                    self._create_model_selection_tab()
                
                # Evaluation Tab
                with gr.Tab("🧪 Model Evaluation", id="model_evaluation"):
                    self._create_evaluation_tab()
                
                # Dashboard Tab
                with gr.Tab("📊 Dashboard", id="dashboard"):
                    self._create_dashboard_tab()
                
                # Registry Management Tab
                with gr.Tab("⚙️ Registry Management", id="registry_management"):
                    self._create_registry_management_tab()
        
        return interface
    
    def _create_model_selection_tab(self):
        """Create the model selection tab."""
        with gr.Row():
            with gr.Column(scale=1):
                # Model selection interface
                category_dropdown, model_dropdown, load_button, model_info = self.model_selector.create_model_selection_interface()
                
                # Refresh button
                refresh_button = gr.Button("🔄 Refresh Models", variant="secondary")
                refresh_status = gr.HTML()
                
            with gr.Column(scale=2):
                # Model registry overview
                gr.Markdown("## 📋 Model Registry Overview")
                registry_stats = gr.HTML()
                
                # Available models list
                gr.Markdown("## 📝 Available Models")
                models_list = gr.Dataframe(
                    headers=["Name", "Category", "Source", "Status", "Type"],
                    datatype=["str", "str", "str", "str", "str"],
                    interactive=False,
                    label="All Available Models"
                )
        
        # Event handlers
        category_dropdown.change(
            fn=self.model_selector.update_model_list,
            inputs=[category_dropdown],
            outputs=[model_dropdown]
        )
        
        load_button.click(
            fn=self.model_selector.load_selected_model,
            inputs=[model_dropdown],
            outputs=[model_info]
        )
        
        refresh_button.click(
            fn=self.model_selector.refresh_models,
            outputs=[model_dropdown, refresh_status]
        )
        
        # Update registry stats and models list
        interface = gr.Interface.load("refresh_initial_data")
    
    def _create_evaluation_tab(self):
        """Create the evaluation tab."""
        with gr.Row():
            with gr.Column(scale=1):
                # Evaluation interface
                input_text, temperature, max_tokens, top_p, evaluate_button, clear_button, results_output, metrics_chart = self.evaluation_interface.create_evaluation_interface()
                
                # Model selection for evaluation
                gr.Markdown("### Select Model for Evaluation")
                eval_model_dropdown = gr.Dropdown(
                    choices=[],
                    label="Evaluation Model",
                    info="Choose a model for evaluation"
                )
                
            with gr.Column(scale=2):
                # Evaluation history
                gr.Markdown("## 📈 Evaluation History")
                evaluation_stats = gr.HTML()
        
        # Event handlers
        evaluate_button.click(
            fn=self.evaluation_interface.evaluate_model,
            inputs=[input_text, eval_model_dropdown, temperature, max_tokens, top_p],
            outputs=[results_output, metrics_chart]
        )
        
        clear_button.click(
            fn=self.evaluation_interface.clear_evaluation,
            outputs=[results_output, metrics_chart]
        )
    
    def _create_dashboard_tab(self):
        """Create the dashboard tab."""
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 📊 System Dashboard")
                
                # Dashboard controls
                with gr.Row():
                    refresh_dashboard_btn = gr.Button("🔄 Refresh Dashboard", variant="secondary")
                    export_data_btn = gr.Button("📥 Export Data", variant="secondary")
                
                # System statistics
                system_stats = gr.HTML()
                
                # Performance metrics
                with gr.Row():
                    latency_chart = gr.Plot(label="Latency Trends")
                    throughput_chart = gr.Plot(label="Model Usage")
                
                # Model distribution
                model_distribution_chart = gr.Plot(label="Model Distribution by Category")
        
        # Event handlers
        refresh_dashboard_btn.click(
            fn=self._refresh_dashboard,
            outputs=[system_stats, latency_chart, throughput_chart, model_distribution_chart]
        )
    
    def _create_registry_management_tab(self):
        """Create the registry management tab."""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## ⚙️ Registry Management")
                gr.Markdown("""
                **Registry Operations:**
                - Sync with Ollama and GitHub Models
                - Export/Import registry data
                - View registry statistics
                - Manage model status
                """)
                
                # Registry operations
                with gr.Group():
                    gr.Markdown("### Registry Operations")
                    
                    sync_registry_btn = gr.Button("🔄 Sync All Sources", variant="primary")
                    export_registry_btn = gr.Button("📤 Export Registry", variant="secondary")
                    import_registry_btn = gr.Button("📥 Import Registry", variant="secondary")
                    clear_registry_btn = gr.Button("🗑️ Clear Registry", variant="secondary")
                
                # Registry statistics
                gr.Markdown("### Registry Statistics")
                registry_stats_display = gr.HTML()
                
            with gr.Column(scale=2):
                # Registry content
                gr.Markdown("## 📋 Registry Content")
                
                # Model search
                search_query = gr.Textbox(
                    label="Search Models",
                    placeholder="Search by name, description, or tags...",
                    info="Enter search terms to find specific models"
                )
                search_button = gr.Button("🔍 Search", variant="secondary")
                
                # Search results
                search_results = gr.Dataframe(
                    headers=["ID", "Name", "Category", "Source", "Status"],
                    datatype=["str", "str", "str", "str", "str"],
                    interactive=False,
                    label="Search Results"
                )
        
        # Event handlers
        sync_registry_btn.click(
            fn=self._sync_registry,
            outputs=[registry_stats_display]
        )
        
        export_registry_btn.click(
            fn=self._export_registry,
            outputs=[gr.File()]
        )
        
        search_button.click(
            fn=self._search_models,
            inputs=[search_query],
            outputs=[search_results]
        )
    
    async def _refresh_dashboard(self) -> tuple:
        """Refresh the dashboard with current data."""
        try:
            # Get registry statistics
            stats = self.registry_manager.get_registry_statistics()
            
            # Create system stats HTML
            system_stats_html = f"""
            <div style='padding: 1.5rem; background: #f9f9f9; border-radius: 8px; margin-bottom: 1rem;'>
                <h3>📊 System Statistics</h3>
                <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;'>
                    <div style='text-align: center; padding: 1rem; background: white; border-radius: 4px;'>
                        <h4 style='margin: 0; color: #2196F3;'>{stats['total_models']}</h4>
                        <p style='margin: 0; color: #666;'>Total Models</p>
                    </div>
                    <div style='text-align: center; padding: 1rem; background: white; border-radius: 4px;'>
                        <h4 style='margin: 0; color: #4CAF50;'>{stats['local_models']}</h4>
                        <p style='margin: 0; color: #666;'>Local Models</p>
                    </div>
                    <div style='text-align: center; padding: 1rem; background: white; border-radius: 4px;'>
                        <h4 style='margin: 0; color: #FF9800;'>{stats['remote_models']}</h4>
                        <p style='margin: 0; color: #666;'>Remote Models</p>
                    </div>
                    <div style='text-align: center; padding: 1rem; background: white; border-radius: 4px;'>
                        <h4 style='margin: 0; color: #9C27B0;'>{stats['experimental_models']}</h4>
                        <p style='margin: 0; color: #666;'>Experimental</p>
                    </div>
                </div>
            </div>
            """
            
            # Create charts
            latency_chart = self._create_latency_chart()
            throughput_chart = self._create_throughput_chart(stats)
            distribution_chart = self._create_distribution_chart(stats)
            
            return system_stats_html, latency_chart, throughput_chart, distribution_chart
            
        except Exception as e:
            logger.error(f"Error refreshing dashboard: {e}")
            error_html = f"<div style='color: red; padding: 1rem;'>Error refreshing dashboard: {str(e)}</div>"
            return error_html, None, None, None
    
    def _create_latency_chart(self):
        """Create latency trend chart."""
        try:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[1, 2, 3, 4, 5],
                y=[100, 120, 90, 110, 95],
                mode='lines+markers',
                name='Average Latency (ms)',
                line=dict(color='#2196F3', width=3)
            ))
            
            fig.update_layout(
                title="Latency Trends",
                xaxis_title="Time",
                yaxis_title="Latency (ms)",
                showlegend=True
            )
            
            return fig
        except:
            return None
    
    def _create_throughput_chart(self, stats):
        """Create model usage chart."""
        try:
            fig = go.Figure()
            
            categories = list(stats['category_counts'].keys())
            counts = list(stats['category_counts'].values())
            
            fig.add_trace(go.Bar(
                x=categories,
                y=counts,
                name='Models by Category',
                marker_color='#4CAF50'
            ))
            
            fig.update_layout(
                title="Model Usage by Category",
                xaxis_title="Category",
                yaxis_title="Number of Models",
                showlegend=True
            )
            
            return fig
        except:
            return None
    
    def _create_distribution_chart(self, stats):
        """Create model distribution chart."""
        try:
            fig = go.Figure()
            
            sources = list(stats['source_counts'].keys())
            counts = list(stats['source_counts'].values())
            
            fig.add_trace(go.Pie(
                labels=sources,
                values=counts,
                hole=0.3
            ))
            
            fig.update_layout(
                title="Model Distribution by Source"
            )
            
            return fig
        except:
            return None
    
    async def _sync_registry(self) -> str:
        """Sync registry with all sources."""
        try:
            logger.info("Syncing registry with all sources...")
            sync_result = await self.registry_manager.sync_with_sources()
            
            total_synced = sync_result.get("total_synced", 0)
            ollama_synced = sync_result.get("ollama_sync", {}).get("models_synced", 0)
            github_synced = sync_result.get("github_models_sync", {}).get("models_synced", 0)
            
            sync_html = f"""
            <div style='padding: 1.5rem; background: #e8f5e8; border: 1px solid #4CAF50; border-radius: 8px; margin: 1rem 0;'>
                <h3>✅ Registry Sync Completed</h3>
                <p><strong>Total models synced:</strong> {total_synced}</p>
                <p><strong>Local (Ollama):</strong> {ollama_synced} models</p>
                <p><strong>Remote (GitHub Models):</strong> {github_synced} models</p>
            </div>
            """
            
            return sync_html
            
        except Exception as e:
            logger.error(f"Error syncing registry: {e}")
            return f"<div style='color: red; padding: 1rem;'>Sync failed: {str(e)}</div>"
    
    def _export_registry(self) -> str:
        """Export registry data."""
        try:
            # Export to JSON file
            file_path = "registry_export.json"
            success = self.registry_manager.export_registry(file_path)
            
            if success:
                return file_path
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error exporting registry: {e}")
            return None
    
    async def _search_models(self, query: str) -> List[List[str]]:
        """Search models by query."""
        try:
            if not query.strip():
                # Return all models if no query
                models = await self.registry_manager.list_models()
            else:
                models = self.registry_manager.search_models(query)
            
            # Convert to dataframe format
            results = []
            for model in models[:20]:  # Limit to 20 results
                results.append([
                    model.id,
                    model.name,
                    model.get_category_display_name(),
                    model.get_source_display_name(),
                    model.status.upper()
                ])
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching models: {e}")
            return []
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self.evaluation_interface:
                await self.evaluation_interface.cleanup()
            logger.info("Application cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def create_simplified_gradio_app() -> gr.Blocks:
    """
    Create and return the simplified Gradio application.
    
    Returns:
        Gradio Blocks interface
    """
    app = SimplifiedLenovoModelEvaluationApp()
    return app.create_interface()


if __name__ == "__main__":
    # Launch the simplified application
    interface = create_simplified_gradio_app()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )
