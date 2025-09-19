"""
Main Gradio Application for Lenovo AAITC Solutions

This module creates the main Gradio application that provides an interactive
interface for both Assignment 1 (Model Evaluation) and Assignment 2 (AI Architecture).

Key Features:
- Multi-tab interface for different assignments
- Real-time model evaluation
- Interactive visualizations
- MCP server integration
- Export capabilities
"""

import gradio as gr
import logging
import asyncio
from typing import Dict, Any, List, Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from .components import (
    ModelEvaluationInterface,
    AIArchitectureInterface,
    VisualizationDashboard,
    ReportGenerator
)
from .modern_dashboard import create_modern_dashboard
from .agentic_flow_ui import create_agentic_flow_interface
from .copilot_integration import create_copilot_interface
from .knowledge_graph_ui import create_knowledge_graph_interface
# MCP server import removed - using Gradio's built-in MCP capabilities
from ..model_evaluation import (
    ComprehensiveEvaluationPipeline,
    ModelConfig,
    TaskType,
    LATEST_MODEL_CONFIGS
)
from ..ai_architecture import (
    HybridAIPlatform,
    ModelLifecycleManager,
    AgenticComputingFramework,
    RAGSystem
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LenovoAAITCApp:
    """
    Main application class for Lenovo AAITC Gradio interface.
    
    This class orchestrates the entire application, including the Gradio interface,
    MCP server, and integration with both assignment solutions.
    """
    
    def __init__(self):
        """Initialize the Lenovo AAITC application."""
        # MCP server removed - using Gradio's built-in MCP capabilities
        self.evaluation_interface = ModelEvaluationInterface()
        self.architecture_interface = AIArchitectureInterface()
        self.visualization_dashboard = VisualizationDashboard()
        self.report_generator = ReportGenerator()
        
        # Initialize evaluation pipeline
        self.evaluation_pipeline = None
        self.current_results = None
        
        logger.info("Initialized Lenovo AAITC Application with Gradio MCP support")
    
    def create_interface(self) -> gr.Blocks:
        """
        Create the main Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(
            title="Lenovo AAITC Solutions - Q3 2025",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
                margin: auto !important;
            }
            .tab-nav {
                background: linear-gradient(90deg, #1e3a8a, #3b82f6);
                color: white;
                padding: 1rem;
                border-radius: 8px;
                margin-bottom: 1rem;
            }
            .metric-card {
                background: white;
                border-radius: 8px;
                padding: 1rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin: 0.5rem;
            }
            """
        ) as interface:
            
            # Header
            gr.HTML("""
            <div class="tab-nav">
                <h1>🚀 Lenovo AAITC Technical Solutions</h1>
                <p>Advanced AI Model Evaluation & Architecture Framework - Q3 2025</p>
                <p>Featuring GPT-5, GPT-5-Codex, Claude 3.5 Sonnet, and Llama 3.3</p>
            </div>
            """)
            
            # Main tabs
            with gr.Tabs():
                
                # Assignment 1: Model Evaluation
                with gr.Tab("🎯 Model Evaluation", id="model_evaluation"):
                    self._create_model_evaluation_tab()
                
                # Assignment 2: AI Architecture
                with gr.Tab("🏗️ AI Architecture", id="ai_architecture"):
                    self._create_ai_architecture_tab()
                
                # Visualization Dashboard
                with gr.Tab("📊 Dashboard", id="dashboard"):
                    self._create_dashboard_tab()
                
                # MCP Server Interface
                with gr.Tab("🔧 MCP Server", id="mcp_server"):
                    self._create_mcp_server_tab()
                
                # Reports and Export
                with gr.Tab("📋 Reports", id="reports"):
                    self._create_reports_tab()
                
                # Modern UI Components
                with gr.Tab("🚀 Modern Dashboard", id="modern_dashboard"):
                    self._create_modern_dashboard_tab()
                
                with gr.Tab("🔄 Agentic Flow Builder", id="agentic_flow"):
                    self._create_agentic_flow_tab()
                
                with gr.Tab("🤖 AI Copilot", id="ai_copilot"):
                    self._create_copilot_tab()
                
                with gr.Tab("🕸️ Knowledge Graph", id="knowledge_graph"):
                    self._create_knowledge_graph_tab()
        
        return interface
    
    def _create_model_evaluation_tab(self):
        """Create the model evaluation tab interface."""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 🎯 Model Evaluation Framework")
                gr.Markdown("""
                **Latest Q3 2025 Models:**
                - GPT-5 (Advanced Reasoning)
                - GPT-5-Codex (74.5% Coding Success Rate)
                - Claude 3.5 Sonnet (Enhanced Analysis)
                - Llama 3.3 (Open Source)
                """)
                
                # Model selection
                model_selection = gr.CheckboxGroup(
                    choices=list(LATEST_MODEL_CONFIGS.keys()),
                    value=list(LATEST_MODEL_CONFIGS.keys())[:2],
                    label="Select Models to Evaluate",
                    info="Choose which models to include in the evaluation"
                )
                
                # Task selection
                task_selection = gr.CheckboxGroup(
                    choices=[task.value for task in TaskType],
                    value=[TaskType.TEXT_GENERATION.value, TaskType.CODE_GENERATION.value],
                    label="Select Evaluation Tasks",
                    info="Choose which tasks to evaluate"
                )
                
                # Evaluation options
                with gr.Accordion("Evaluation Options", open=False):
                    include_robustness = gr.Checkbox(
                        value=True,
                        label="Include Robustness Testing",
                        info="Test adversarial inputs and noise tolerance"
                    )
                    
                    include_bias_detection = gr.Checkbox(
                        value=True,
                        label="Include Bias Detection",
                        info="Analyze bias across multiple dimensions"
                    )
                    
                    enhanced_scale = gr.Checkbox(
                        value=True,
                        label="Enhanced Experimental Scale",
                        info="Use prompt registries for larger test datasets"
                    )
                
                # Start evaluation button
                start_evaluation_btn = gr.Button(
                    "🚀 Start Evaluation",
                    variant="primary",
                    size="lg"
                )
                
                # Progress bar
                progress_bar = gr.Progress()
            
            with gr.Column(scale=2):
                # Results display
                results_output = gr.HTML(
                    value="<div style='text-align: center; padding: 2rem; color: #666;'>Select models and tasks, then click 'Start Evaluation' to begin</div>"
                )
                
                # Model comparison table
                comparison_table = gr.Dataframe(
                    headers=["Model", "Overall Score", "Quality", "Performance", "Cost Efficiency", "Robustness"],
                    datatype=["str", "number", "number", "number", "number", "number"],
                    interactive=False
                )
        
        # Event handlers
        start_evaluation_btn.click(
            fn=self._run_evaluation,
            inputs=[model_selection, task_selection, include_robustness, include_bias_detection, enhanced_scale],
            outputs=[results_output, comparison_table],
            show_progress=True
        )
    
    def _create_ai_architecture_tab(self):
        """Create the AI architecture tab interface."""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 🏗️ AI Architecture Framework")
                gr.Markdown("""
                **Hybrid AI Platform Components:**
                - Model Lifecycle Management
                - Agentic Computing Framework
                - RAG System with Advanced Retrieval
                - Cross-Platform Orchestration
                """)
                
                # Architecture component selection
                component_selection = gr.CheckboxGroup(
                    choices=[
                        "Hybrid AI Platform",
                        "Model Lifecycle Manager",
                        "Agentic Computing Framework",
                        "RAG System"
                    ],
                    value=["Hybrid AI Platform"],
                    label="Select Architecture Components"
                )
                
                # Deployment scenario
                deployment_scenario = gr.Radio(
                    choices=["Cloud", "Edge", "Mobile", "Hybrid"],
                    value="Hybrid",
                    label="Deployment Scenario"
                )
                
                # Architecture visualization button
                visualize_btn = gr.Button(
                    "📐 Visualize Architecture",
                    variant="primary"
                )
            
            with gr.Column(scale=2):
                # Architecture diagram
                architecture_diagram = gr.Plot(
                    label="Architecture Diagram"
                )
                
                # Component details
                component_details = gr.HTML(
                    value="<div style='text-align: center; padding: 2rem; color: #666;'>Select components and click 'Visualize Architecture' to see the design</div>"
                )
        
        # Event handlers
        visualize_btn.click(
            fn=self._visualize_architecture,
            inputs=[component_selection, deployment_scenario],
            outputs=[architecture_diagram, component_details]
        )
    
    def _create_dashboard_tab(self):
        """Create the visualization dashboard tab."""
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 📊 Real-Time Performance Dashboard")
                
                # Dashboard controls
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh Dashboard", variant="secondary")
                    export_btn = gr.Button("📥 Export Data", variant="secondary")
                
                # Performance metrics
                with gr.Row():
                    with gr.Column():
                        latency_chart = gr.Plot(label="Latency Trends")
                        throughput_chart = gr.Plot(label="Throughput Trends")
                    
                    with gr.Column():
                        quality_chart = gr.Plot(label="Quality Metrics")
                        cost_chart = gr.Plot(label="Cost Analysis")
                
                # Model comparison radar chart
                radar_chart = gr.Plot(label="Model Comparison Radar")
        
        # Event handlers
        refresh_btn.click(
            fn=self._refresh_dashboard,
            outputs=[latency_chart, throughput_chart, quality_chart, cost_chart, radar_chart]
        )
        
        export_btn.click(
            fn=self._export_dashboard_data,
            outputs=[gr.File()]
        )
    
    def _create_mcp_server_tab(self):
        """Create the MCP server interface tab."""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 🔧 MCP Server Interface")
                gr.Markdown("""
                **Gradio Built-in MCP Server:**
                - ✅ Automatic MCP tool exposure
                - ✅ Function-based tool generation
                - ✅ Real-time model evaluation tools
                - ✅ Interactive API endpoints
                - ✅ No external dependencies required
                """)
                
                # Server status
                server_status = gr.Textbox(
                    value="✅ MCP Server: Active (Gradio Built-in)",
                    label="Server Status",
                    interactive=False
                )
                
                # MCP capabilities info
                gr.Markdown("""
                **Available MCP Tools:**
                - `run_evaluation` - Comprehensive model evaluation
                - `visualize_architecture` - AI architecture visualization
                - `refresh_dashboard` - Real-time performance monitoring
                - `generate_report` - Report generation and export
                """)
                
                # Server configuration
                with gr.Accordion("MCP Configuration", open=False):
                    gr.Markdown("""
                    **Gradio MCP Features:**
                    - Automatic tool discovery from function names
                    - Type hints for parameter validation
                    - Docstrings for tool descriptions
                    - Progress updates and file handling
                    - Authentication header support
                    """)
            
            with gr.Column(scale=2):
                # MCP tools documentation
                gr.Markdown("## 📚 MCP Tools Documentation")
                
                # Available tools
                api_endpoints = gr.Dataframe(
                    headers=["Tool Name", "Description", "Parameters", "Status"],
                    value=[
                        ["run_evaluation", "Run comprehensive model evaluation with selected models and tasks", "models, tasks, options", "✅ Active"],
                        ["visualize_architecture", "Generate AI architecture diagrams and component details", "components, deployment", "✅ Active"],
                        ["refresh_dashboard", "Update real-time performance dashboard with latest metrics", "time_range", "✅ Active"],
                        ["generate_report", "Generate comprehensive reports in multiple formats", "report_type, options", "✅ Active"],
                        ["export_dashboard_data", "Export dashboard data in various formats", "format", "✅ Active"]
                    ],
                    interactive=False
                )
                
                # MCP connection info
                gr.Markdown("""
                **MCP Client Connection:**
                ```
                Server: localhost:7860
                Protocol: Model Context Protocol
                Tools: Auto-discovered from Gradio functions
                Authentication: Optional headers support
                ```
                """)
    
    def _create_reports_tab(self):
        """Create the reports and export tab."""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📋 Reports & Export")
                gr.Markdown("""
                **Generate comprehensive reports:**
                - Executive summaries
                - Technical deep-dives
                - Performance analysis
                - Recommendations
                """)
                
                # Report type selection
                report_type = gr.Radio(
                    choices=[
                        "Executive Summary",
                        "Technical Report",
                        "Performance Analysis",
                        "Model Comparison",
                        "Architecture Review"
                    ],
                    value="Executive Summary",
                    label="Report Type"
                )
                
                # Report options
                with gr.Accordion("Report Options", open=False):
                    include_charts = gr.Checkbox(
                        value=True,
                        label="Include Charts and Visualizations"
                    )
                    
                    include_recommendations = gr.Checkbox(
                        value=True,
                        label="Include Recommendations"
                    )
                    
                    include_raw_data = gr.Checkbox(
                        value=False,
                        label="Include Raw Data"
                    )
                
                # Generate report button
                generate_report_btn = gr.Button(
                    "📄 Generate Report",
                    variant="primary"
                )
            
            with gr.Column(scale=2):
                # Report preview
                report_preview = gr.HTML(
                    value="<div style='text-align: center; padding: 2rem; color: #666;'>Select report type and click 'Generate Report' to create a comprehensive report</div>"
                )
                
                # Download buttons
                with gr.Row():
                    download_pdf_btn = gr.Button("📄 Download PDF", variant="secondary")
                    download_excel_btn = gr.Button("📊 Download Excel", variant="secondary")
                    download_json_btn = gr.Button("📋 Download JSON", variant="secondary")
        
        # Event handlers
        generate_report_btn.click(
            fn=self._generate_report,
            inputs=[report_type, include_charts, include_recommendations, include_raw_data],
            outputs=[report_preview]
        )
        
        download_pdf_btn.click(
            fn=self._download_report,
            inputs=[gr.State("pdf")],
            outputs=[gr.File()]
        )
        
        download_excel_btn.click(
            fn=self._download_report,
            inputs=[gr.State("excel")],
            outputs=[gr.File()]
        )
        
        download_json_btn.click(
            fn=self._download_report,
            inputs=[gr.State("json")],
            outputs=[gr.File()]
        )
    
    def _run_evaluation(
        self,
        selected_models: List[str],
        selected_tasks: List[str],
        include_robustness: bool,
        include_bias_detection: bool,
        enhanced_scale: bool
    ) -> tuple:
        """
        Run model evaluation with selected parameters.
        
        Args:
            selected_models: List of selected model names
            selected_tasks: List of selected task types
            include_robustness: Whether to include robustness testing
            include_bias_detection: Whether to include bias detection
            enhanced_scale: Whether to use enhanced experimental scale
            
        Returns:
            Tuple of (results_html, comparison_dataframe)
        """
        try:
            # Initialize evaluation pipeline
            models = [LATEST_MODEL_CONFIGS[model] for model in selected_models]
            self.evaluation_pipeline = ComprehensiveEvaluationPipeline(models)
            
            # Prepare test datasets
            test_datasets = self._prepare_test_datasets(selected_tasks, enhanced_scale)
            
            # Run evaluation
            results_df = self.evaluation_pipeline.run_multi_task_evaluation(
                test_datasets,
                include_robustness=include_robustness,
                include_bias_detection=include_bias_detection
            )
            
            # Store results
            self.current_results = results_df
            
            # Generate results HTML
            results_html = self._generate_results_html(results_df)
            
            # Prepare comparison table
            comparison_data = self._prepare_comparison_table(results_df)
            
            return results_html, comparison_data
            
        except Exception as e:
            logger.error(f"Error in evaluation: {str(e)}")
            error_html = f"<div style='color: red; padding: 1rem;'>Error: {str(e)}</div>"
            return error_html, []
    
    def _prepare_test_datasets(
        self,
        selected_tasks: List[str],
        enhanced_scale: bool
    ) -> Dict[TaskType, pd.DataFrame]:
        """Prepare test datasets for evaluation."""
        test_datasets = {}
        
        for task_str in selected_tasks:
            task_type = TaskType(task_str)
            
            if enhanced_scale:
                # Use enhanced dataset from prompt registries with AI Tool System Prompts
                from ..model_evaluation.prompt_registries import PromptRegistryManager
                registry_manager = PromptRegistryManager(cache_dir="cache/ai_tool_prompts")
                
                # Get enhanced dataset with AI tool prompts
                dataset = registry_manager.get_enhanced_evaluation_dataset(
                    target_size=1000,
                    categories=[task_type],
                    enhanced_scale=True
                )
            else:
                # Use basic dataset
                dataset = self._create_basic_dataset(task_type)
            
            test_datasets[task_type] = dataset
        
        return test_datasets
    
    def _create_basic_dataset(self, task_type: TaskType) -> pd.DataFrame:
        """Create basic test dataset for a task type."""
        # Mock implementation - in production, load from actual datasets
        sample_data = {
            'input': [
                f"Sample input for {task_type.value} task 1",
                f"Sample input for {task_type.value} task 2",
                f"Sample input for {task_type.value} task 3"
            ],
            'expected_output': [
                f"Expected output for {task_type.value} task 1",
                f"Expected output for {task_type.value} task 2",
                f"Expected output for {task_type.value} task 3"
            ]
        }
        
        return pd.DataFrame(sample_data)
    
    def _generate_results_html(self, results_df: pd.DataFrame) -> str:
        """Generate HTML display of evaluation results."""
        if results_df.empty:
            return "<div style='text-align: center; padding: 2rem; color: #666;'>No results available</div>"
        
        # Calculate summary statistics
        summary_stats = results_df.groupby('model').agg({
            'overall_score': 'mean',
            'latency_ms': 'mean',
            'rouge_l': 'mean',
            'cost_efficiency_score': 'mean'
        }).round(3)
        
        html = "<div style='padding: 1rem;'>"
        html += "<h3>📊 Evaluation Results Summary</h3>"
        
        # Summary table
        html += "<table style='width: 100%; border-collapse: collapse; margin: 1rem 0;'>"
        html += "<tr style='background: #f0f0f0;'>"
        html += "<th style='padding: 0.5rem; border: 1px solid #ddd;'>Model</th>"
        html += "<th style='padding: 0.5rem; border: 1px solid #ddd;'>Overall Score</th>"
        html += "<th style='padding: 0.5rem; border: 1px solid #ddd;'>Avg Latency (ms)</th>"
        html += "<th style='padding: 0.5rem; border: 1px solid #ddd;'>ROUGE-L</th>"
        html += "<th style='padding: 0.5rem; border: 1px solid #ddd;'>Cost Efficiency</th>"
        html += "</tr>"
        
        for model, stats in summary_stats.iterrows():
            html += "<tr>"
            html += f"<td style='padding: 0.5rem; border: 1px solid #ddd;'>{model}</td>"
            html += f"<td style='padding: 0.5rem; border: 1px solid #ddd;'>{stats['overall_score']:.3f}</td>"
            html += f"<td style='padding: 0.5rem; border: 1px solid #ddd;'>{stats['latency_ms']:.1f}</td>"
            html += f"<td style='padding: 0.5rem; border: 1px solid #ddd;'>{stats['rouge_l']:.3f}</td>"
            html += f"<td style='padding: 0.5rem; border: 1px solid #ddd;'>{stats['cost_efficiency_score']:.3f}</td>"
            html += "</tr>"
        
        html += "</table>"
        html += "</div>"
        
        return html
    
    def _prepare_comparison_table(self, results_df: pd.DataFrame) -> List[List]:
        """Prepare comparison table data for Gradio Dataframe."""
        if results_df.empty:
            return []
        
        # Group by model and calculate averages
        summary = results_df.groupby('model').agg({
            'overall_score': 'mean',
            'rouge_l': 'mean',
            'latency_ms': 'mean',
            'cost_efficiency_score': 'mean',
            'adversarial_robustness': 'mean'
        }).round(3)
        
        # Convert to list of lists for Gradio Dataframe
        table_data = []
        for model, stats in summary.iterrows():
            table_data.append([
                model,
                stats['overall_score'],
                stats['rouge_l'],
                stats['latency_ms'],
                stats['cost_efficiency_score'],
                stats['adversarial_robustness']
            ])
        
        return table_data
    
    def _visualize_architecture(
        self,
        selected_components: List[str],
        deployment_scenario: str
    ) -> tuple:
        """Visualize AI architecture based on selected components."""
        # Create architecture diagram
        fig = go.Figure()
        
        # Add components based on selection
        if "Hybrid AI Platform" in selected_components:
            fig.add_trace(go.Scatter(
                x=[1, 2, 3, 4],
                y=[1, 1, 1, 1],
                mode='markers+text',
                text=["Cloud", "Edge", "Mobile", "Hybrid"],
                textposition="top center",
                marker=dict(size=20, color='blue'),
                name="Hybrid AI Platform"
            ))
        
        fig.update_layout(
            title=f"AI Architecture - {deployment_scenario} Deployment",
            xaxis_title="Components",
            yaxis_title="Layers",
            showlegend=True
        )
        
        # Generate component details HTML
        details_html = f"""
        <div style='padding: 1rem;'>
            <h3>🏗️ Architecture Components</h3>
            <ul>
                {''.join([f'<li>{component}</li>' for component in selected_components])}
            </ul>
            <p><strong>Deployment Scenario:</strong> {deployment_scenario}</p>
        </div>
        """
        
        return fig, details_html
    
    def _refresh_dashboard(self) -> tuple:
        """Refresh the dashboard with current data."""
        # Create sample charts
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[100, 120, 90, 110], name="Latency"))
        fig1.update_layout(title="Latency Trends")
        
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=["Model A", "Model B", "Model C"], y=[85, 92, 78], name="Throughput"))
        fig2.update_layout(title="Throughput Comparison")
        
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=[1, 2, 3], y=[0.8, 0.9, 0.85], name="Quality"))
        fig3.update_layout(title="Quality Metrics")
        
        fig4 = go.Figure()
        fig4.add_trace(go.Pie(labels=["Compute", "Storage", "Network"], values=[40, 30, 30]))
        fig4.update_layout(title="Cost Breakdown")
        
        fig5 = go.Figure()
        fig5.add_trace(go.Scatterpolar(
            r=[0.9, 0.8, 0.7, 0.85, 0.9],
            theta=['Quality', 'Speed', 'Cost', 'Robustness', 'Safety'],
            fill='toself',
            name='Model A'
        ))
        fig5.update_layout(title="Model Comparison Radar")
        
        return fig1, fig2, fig3, fig4, fig5
    
    def _export_dashboard_data(self) -> str:
        """Export dashboard data."""
        # Mock implementation - in production, export actual data
        return "dashboard_data.csv"
    
    # MCP server methods removed - using Gradio's built-in MCP capabilities
    
    def _generate_report(
        self,
        report_type: str,
        include_charts: bool,
        include_recommendations: bool,
        include_raw_data: bool
    ) -> str:
        """Generate comprehensive report."""
        html = f"""
        <div style='padding: 2rem;'>
            <h2>📋 {report_type}</h2>
            <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h3>Executive Summary</h3>
            <p>This report provides a comprehensive analysis of the Lenovo AAITC model evaluation 
            and AI architecture framework, featuring the latest Q3 2025 model versions.</p>
            
            <h3>Key Findings</h3>
            <ul>
                <li>GPT-5 demonstrates superior reasoning capabilities</li>
                <li>GPT-5-Codex achieves 74.5% success rate on coding tasks</li>
                <li>Claude 3.5 Sonnet excels in analysis and conversation</li>
                <li>Llama 3.3 provides excellent open-source alternative</li>
            </ul>
            
            {f'<h3>Charts and Visualizations</h3><p>Performance charts and visualizations included.</p>' if include_charts else ''}
            {f'<h3>Recommendations</h3><p>Strategic recommendations for model selection and deployment.</p>' if include_recommendations else ''}
            {f'<h3>Raw Data</h3><p>Detailed evaluation data and metrics included.</p>' if include_raw_data else ''}
        </div>
        """
        return html
    
    def _download_report(self, format_type: str) -> str:
        """Download report in specified format."""
        # Mock implementation - in production, generate actual files
        if format_type == "pdf":
            return "report.pdf"
        elif format_type == "excel":
            return "report.xlsx"
        elif format_type == "json":
            return "report.json"
        return "report.txt"
    
    def _create_modern_dashboard_tab(self):
        """Create the modern dashboard tab interface."""
        return create_modern_dashboard()
    
    def _create_agentic_flow_tab(self):
        """Create the agentic flow builder tab interface."""
        return create_agentic_flow_interface()
    
    def _create_copilot_tab(self):
        """Create the AI copilot tab interface."""
        return create_copilot_interface()
    
    def _create_knowledge_graph_tab(self):
        """Create the knowledge graph tab interface."""
        return create_knowledge_graph_interface()


def create_gradio_app() -> gr.Blocks:
    """
    Create and return the main Gradio application.
    
    Returns:
        Gradio Blocks interface
    """
    app = LenovoAAITCApp()
    return app.create_interface()


if __name__ == "__main__":
    # Launch the application
    interface = create_gradio_app()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True,
        mcp_server=True  # Enable Gradio's built-in MCP capabilities
    )
