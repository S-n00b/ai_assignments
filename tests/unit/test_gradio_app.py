"""
Unit tests for Gradio application components.

Tests the core functionality of the Gradio frontend including
interfaces, components, and MCP server integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import gradio as gr

from src.gradio_app import (
    create_gradio_app,
    ModelEvaluationInterface,
    AIArchitectureInterface,
    VisualizationDashboard,
    ReportGenerator,
    MCPServer
)


class TestGradioApp:
    """Test cases for Gradio app creation and functionality."""
    
    @pytest.fixture
    def app(self):
        """Create Gradio app instance."""
        return create_gradio_app()
    
    def test_app_initialization(self, app):
        """Test app initialization."""
        assert app.model_evaluation_interface is not None
        assert app.ai_architecture_interface is not None
        assert app.visualization_dashboard is not None
        assert app.report_generator is not None
    
    def test_create_interface(self, app):
        """Test interface creation."""
        interface = app.create_interface()
        
        assert interface is not None
        assert isinstance(interface, gr.Blocks)
    
    def test_setup_tabs(self, app):
        """Test tab setup."""
        with patch.object(gr, 'Tabs') as mock_tabs:
            with patch.object(gr, 'Tab') as mock_tab:
                app._setup_tabs()
                
                # Verify tabs are created
                assert mock_tabs.called
                assert mock_tab.called
    
    @pytest.mark.asyncio
    async def test_handle_model_evaluation(self, app, sample_evaluation_data):
        """Test model evaluation handling."""
        with patch.object(app.model_evaluation_interface, 'evaluate_models', new_callable=AsyncMock) as mock_eval:
            mock_eval.return_value = {"results": "evaluation_complete"}
            
            result = await app._handle_model_evaluation(sample_evaluation_data)
            
            assert result["results"] == "evaluation_complete"
            mock_eval.assert_called_once_with(sample_evaluation_data)
    
    @pytest.mark.asyncio
    async def test_handle_architecture_design(self, app, mock_platform_config):
        """Test architecture design handling."""
        with patch.object(app.ai_architecture_interface, 'design_architecture', new_callable=AsyncMock) as mock_design:
            mock_design.return_value = {"architecture": "designed"}
            
            result = await app._handle_architecture_design(mock_platform_config)
            
            assert result["architecture"] == "designed"
            mock_design.assert_called_once_with(mock_platform_config)
    
    def test_generate_visualization(self, app, sample_metrics_data):
        """Test visualization generation."""
        with patch.object(app.visualization_dashboard, 'create_chart', return_value="chart_html"):
            chart = app._generate_visualization(sample_metrics_data, "accuracy")
            
            assert chart == "chart_html"
    
    def test_export_report(self, app, sample_metrics_data):
        """Test report export."""
        with patch.object(app.report_generator, 'generate_report', return_value="report.pdf"):
            report = app._export_report(sample_metrics_data, "pdf")
            
            assert report == "report.pdf"


class TestModelEvaluationInterface:
    """Test cases for ModelEvaluationInterface class."""
    
    @pytest.fixture
    def eval_interface(self):
        """Create model evaluation interface instance."""
        return ModelEvaluationInterface()
    
    def test_interface_initialization(self, eval_interface):
        """Test interface initialization."""
        assert eval_interface.evaluation_pipeline is not None
        assert eval_interface.available_models is not None
        assert len(eval_interface.available_models) > 0
    
    def test_create_model_selection_ui(self, eval_interface):
        """Test model selection UI creation."""
        with patch.object(gr, 'Dropdown') as mock_dropdown:
            with patch.object(gr, 'CheckboxGroup') as mock_checkbox:
                ui = eval_interface._create_model_selection_ui()
                
                assert mock_dropdown.called
                assert mock_checkbox.called
    
    def test_create_evaluation_parameters_ui(self, eval_interface):
        """Test evaluation parameters UI creation."""
        with patch.object(gr, 'Slider') as mock_slider:
            with patch.object(gr, 'Number') as mock_number:
                ui = eval_interface._create_evaluation_parameters_ui()
                
                assert mock_slider.called
                assert mock_number.called
    
    @pytest.mark.asyncio
    async def test_evaluate_models(self, eval_interface, sample_evaluation_data):
        """Test model evaluation execution."""
        selected_models = ["gpt-3.5-turbo", "claude-3-sonnet"]
        parameters = {"max_tokens": 1000, "temperature": 0.7}
        
        with patch.object(eval_interface.evaluation_pipeline, 'evaluate_all_models', new_callable=AsyncMock) as mock_eval:
            mock_eval.return_value = [{"model": "gpt-3.5-turbo", "score": 0.95}]
            
            results = await eval_interface.evaluate_models(
                selected_models, sample_evaluation_data, parameters
            )
            
            assert isinstance(results, list)
            assert len(results) > 0
            assert results[0]["model"] == "gpt-3.5-turbo"
            mock_eval.assert_called_once()
    
    def test_format_evaluation_results(self, eval_interface):
        """Test evaluation results formatting."""
        raw_results = [
            {"model": "gpt-3.5-turbo", "bleu_score": 0.85, "rouge_score": 0.88},
            {"model": "claude-3-sonnet", "bleu_score": 0.82, "rouge_score": 0.85}
        ]
        
        formatted = eval_interface._format_evaluation_results(raw_results)
        
        assert isinstance(formatted, str)
        assert "gpt-3.5-turbo" in formatted
        assert "claude-3-sonnet" in formatted
        assert "0.85" in formatted
    
    def test_validate_input_parameters(self, eval_interface):
        """Test input parameter validation."""
        valid_params = {"max_tokens": 1000, "temperature": 0.7}
        invalid_params = {"max_tokens": -1, "temperature": 2.0}
        
        assert eval_interface._validate_input_parameters(valid_params) == True
        assert eval_interface._validate_input_parameters(invalid_params) == False


class TestAIArchitectureInterface:
    """Test cases for AIArchitectureInterface class."""
    
    @pytest.fixture
    def arch_interface(self):
        """Create AI architecture interface instance."""
        return AIArchitectureInterface()
    
    def test_interface_initialization(self, arch_interface):
        """Test interface initialization."""
        assert arch_interface.platform is not None
        assert arch_interface.lifecycle_manager is not None
        assert arch_interface.agent_framework is not None
    
    def test_create_architecture_config_ui(self, arch_interface):
        """Test architecture configuration UI creation."""
        with patch.object(gr, 'Radio') as mock_radio:
            with patch.object(gr, 'Textbox') as mock_textbox:
                ui = arch_interface._create_architecture_config_ui()
                
                assert mock_radio.called
                assert mock_textbox.called
    
    def test_create_deployment_ui(self, arch_interface):
        """Test deployment UI creation."""
        with patch.object(gr, 'Button') as mock_button:
            with patch.object(gr, 'Progress') as mock_progress:
                ui = arch_interface._create_deployment_ui()
                
                assert mock_button.called
                assert mock_progress.called
    
    @pytest.mark.asyncio
    async def test_design_architecture(self, arch_interface, mock_platform_config):
        """Test architecture design."""
        with patch.object(arch_interface.platform, 'design_architecture', new_callable=AsyncMock) as mock_design:
            mock_design.return_value = {"architecture": "designed", "cost_estimate": 1000}
            
            result = await arch_interface.design_architecture(mock_platform_config)
            
            assert result["architecture"] == "designed"
            assert result["cost_estimate"] == 1000
            mock_design.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_deploy_architecture(self, arch_interface):
        """Test architecture deployment."""
        architecture_config = {"deployment_target": "aws", "instances": 3}
        
        with patch.object(arch_interface.platform, 'deploy_architecture', new_callable=AsyncMock) as mock_deploy:
            mock_deploy.return_value = {"deployment_id": "deploy_123", "status": "success"}
            
            result = await arch_interface.deploy_architecture(architecture_config)
            
            assert result["deployment_id"] == "deploy_123"
            assert result["status"] == "success"
            mock_deploy.assert_called_once()
    
    def test_validate_architecture_config(self, arch_interface):
        """Test architecture configuration validation."""
        valid_config = {
            "deployment_target": "hybrid",
            "infrastructure": {"cloud_provider": "aws", "region": "us-east-1"}
        }
        invalid_config = {"deployment_target": "invalid"}
        
        assert arch_interface._validate_architecture_config(valid_config) == True
        assert arch_interface._validate_architecture_config(invalid_config) == False


class TestVisualizationDashboard:
    """Test cases for VisualizationDashboard class."""
    
    @pytest.fixture
    def viz_dashboard(self):
        """Create visualization dashboard instance."""
        return VisualizationDashboard()
    
    def test_dashboard_initialization(self, viz_dashboard):
        """Test dashboard initialization."""
        assert viz_dashboard.chart_types is not None
        assert len(viz_dashboard.chart_types) > 0
        assert "bar" in viz_dashboard.chart_types
        assert "line" in viz_dashboard.chart_types
    
    def test_create_chart(self, viz_dashboard, sample_metrics_data):
        """Test chart creation."""
        chart_html = viz_dashboard.create_chart(
            data=sample_metrics_data,
            chart_type="bar",
            x_column="model",
            y_column="bleu_score"
        )
        
        assert isinstance(chart_html, str)
        assert len(chart_html) > 0
        assert "plotly" in chart_html.lower() or "chart" in chart_html.lower()
    
    def test_create_comparison_chart(self, viz_dashboard, sample_metrics_data):
        """Test comparison chart creation."""
        chart_html = viz_dashboard.create_comparison_chart(
            data=sample_metrics_data,
            metrics=["bleu_score", "rouge_score", "bert_score"]
        )
        
        assert isinstance(chart_html, str)
        assert len(chart_html) > 0
    
    def test_create_performance_dashboard(self, viz_dashboard, sample_performance_metrics):
        """Test performance dashboard creation."""
        dashboard_html = viz_dashboard.create_performance_dashboard(sample_performance_metrics)
        
        assert isinstance(dashboard_html, str)
        assert len(dashboard_html) > 0
    
    def test_export_chart(self, viz_dashboard, sample_metrics_data):
        """Test chart export."""
        with patch('builtins.open', mock_open()) as mock_file:
            result = viz_dashboard.export_chart(
                data=sample_metrics_data,
                chart_type="bar",
                filename="test_chart.png",
                format="png"
            )
            
            assert result == True
            mock_file.assert_called_once()


class TestReportGenerator:
    """Test cases for ReportGenerator class."""
    
    @pytest.fixture
    def report_generator(self):
        """Create report generator instance."""
        return ReportGenerator()
    
    def test_generator_initialization(self, report_generator):
        """Test report generator initialization."""
        assert report_generator.template_engine is not None
        assert report_generator.export_formats is not None
        assert "pdf" in report_generator.export_formats
        assert "html" in report_generator.export_formats
    
    def test_generate_evaluation_report(self, report_generator, sample_metrics_data):
        """Test evaluation report generation."""
        with patch.object(report_generator, '_render_template', return_value="rendered_html"):
            with patch.object(report_generator, '_export_to_pdf', return_value="report.pdf"):
                report = report_generator.generate_evaluation_report(sample_metrics_data)
                
                assert report == "report.pdf"
    
    def test_generate_architecture_report(self, report_generator, mock_platform_config):
        """Test architecture report generation."""
        with patch.object(report_generator, '_render_template', return_value="rendered_html"):
            with patch.object(report_generator, '_export_to_pdf', return_value="arch_report.pdf"):
                report = report_generator.generate_architecture_report(mock_platform_config)
                
                assert report == "arch_report.pdf"
    
    def test_render_template(self, report_generator):
        """Test template rendering."""
        template_data = {"title": "Test Report", "data": [1, 2, 3]}
        
        with patch.object(report_generator.template_engine, 'render', return_value="rendered_content"):
            result = report_generator._render_template("evaluation_template", template_data)
            
            assert result == "rendered_content"
    
    def test_export_to_pdf(self, report_generator):
        """Test PDF export."""
        html_content = "<html><body>Test content</body></html>"
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('subprocess.run', return_value=Mock(returncode=0)):
                result = report_generator._export_to_pdf(html_content, "test_report.pdf")
                
                assert result == "test_report.pdf"
                mock_file.assert_called_once()


class TestMCPServer:
    """Test cases for MCPServer class."""
    
    @pytest.fixture
    def mcp_server(self):
        """Create MCP server instance."""
        return MCPServer()
    
    def test_server_initialization(self, mcp_server):
        """Test MCP server initialization."""
        assert mcp_server.server is not None
        assert mcp_server.tools is not None
        assert len(mcp_server.tools) > 0
    
    def test_register_tool(self, mcp_server):
        """Test tool registration."""
        tool_config = {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {"param1": "string"}
        }
        
        result = mcp_server.register_tool(tool_config)
        
        assert result == True
        assert "test_tool" in mcp_server.tools
    
    @pytest.mark.asyncio
    async def test_handle_tool_request(self, mcp_server):
        """Test tool request handling."""
        request = {
            "tool": "test_tool",
            "parameters": {"param1": "test_value"}
        }
        
        with patch.object(mcp_server, '_execute_tool', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {"result": "success"}
            
            response = await mcp_server.handle_tool_request(request)
            
            assert response["result"] == "success"
            mock_execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_server(self, mcp_server):
        """Test server startup."""
        with patch.object(mcp_server.server, 'start', new_callable=AsyncMock):
            result = await mcp_server.start_server(port=8081)
            
            assert result == True
    
    @pytest.mark.asyncio
    async def test_stop_server(self, mcp_server):
        """Test server shutdown."""
        with patch.object(mcp_server.server, 'stop', new_callable=AsyncMock):
            result = await mcp_server.stop_server()
            
            assert result == True


# Helper function for mocking file operations
def mock_open():
    """Mock file open function for testing."""
    return Mock()
