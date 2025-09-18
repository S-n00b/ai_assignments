"""
Integration tests for Gradio application.

Tests the integration between Gradio frontend components and backend systems
including model evaluation, AI architecture, and MCP server.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import gradio as gr

from gradio_app import (
    LenovoAAITCApp,
    ModelEvaluationInterface,
    AIArchitectureInterface,
    MCPServer
)
from model_evaluation import ComprehensiveEvaluationPipeline, ModelConfig
from ai_architecture import HybridAIPlatform, ModelLifecycleManager


class TestGradioIntegration:
    """Integration tests for Gradio application components."""
    
    @pytest.fixture
    def gradio_setup(self, sample_model_config, mock_platform_config):
        """Set up Gradio components for integration testing."""
        app = LenovoAAITCApp()
        eval_interface = ModelEvaluationInterface()
        arch_interface = AIArchitectureInterface()
        mcp_server = MCPServer()
        
        return {
            "app": app,
            "eval_interface": eval_interface,
            "arch_interface": arch_interface,
            "mcp_server": mcp_server,
            "model_config": sample_model_config,
            "platform_config": mock_platform_config
        }
    
    @pytest.mark.asyncio
    async def test_app_model_evaluation_integration(self, gradio_setup, sample_evaluation_data, mock_api_client):
        """Test integration between app and model evaluation interface."""
        app = gradio_setup["app"]
        eval_interface = gradio_setup["eval_interface"]
        
        with patch.object(eval_interface, 'evaluate_models', new_callable=AsyncMock) as mock_eval:
            mock_eval.return_value = [
                {"model": "gpt-3.5-turbo", "bleu_score": 0.85, "rouge_score": 0.88}
            ]
            
            # Test app handling model evaluation
            result = await app._handle_model_evaluation(sample_evaluation_data)
            
            assert "results" in result
            assert result["results"] == [
                {"model": "gpt-3.5-turbo", "bleu_score": 0.85, "rouge_score": 0.88}
            ]
    
    @pytest.mark.asyncio
    async def test_app_architecture_integration(self, gradio_setup, mock_platform_config):
        """Test integration between app and AI architecture interface."""
        app = gradio_setup["app"]
        arch_interface = gradio_setup["arch_interface"]
        
        with patch.object(arch_interface, 'design_architecture', new_callable=AsyncMock) as mock_design:
            mock_design.return_value = {
                "architecture": "hybrid_cloud",
                "cost_estimate": 1500,
                "deployment_plan": "blue_green"
            }
            
            # Test app handling architecture design
            result = await app._handle_architecture_design(mock_platform_config)
            
            assert "architecture" in result
            assert result["architecture"] == "hybrid_cloud"
            assert result["cost_estimate"] == 1500
    
    @pytest.mark.asyncio
    async def test_mcp_server_integration(self, gradio_setup):
        """Test MCP server integration with Gradio app."""
        app = gradio_setup["app"]
        mcp_server = gradio_setup["mcp_server"]
        
        # Test MCP server tool registration
        tool_config = {
            "name": "model_evaluation_tool",
            "description": "Evaluate AI models",
            "parameters": {
                "model_name": "string",
                "test_data": "object"
            }
        }
        
        with patch.object(mcp_server, 'register_tool', return_value=True):
            registration_result = mcp_server.register_tool(tool_config)
            assert registration_result == True
        
        # Test MCP server handling tool requests
        tool_request = {
            "tool": "model_evaluation_tool",
            "parameters": {
                "model_name": "gpt-3.5-turbo",
                "test_data": {"prompts": ["Test prompt"]}
            }
        }
        
        with patch.object(mcp_server, 'handle_tool_request', new_callable=AsyncMock) as mock_handle:
            mock_handle.return_value = {"result": "evaluation_complete", "score": 0.95}
            
            response = await mcp_server.handle_tool_request(tool_request)
            
            assert response["result"] == "evaluation_complete"
            assert response["score"] == 0.95
    
    def test_interface_data_flow(self, gradio_setup, sample_metrics_data):
        """Test data flow between Gradio interfaces."""
        eval_interface = gradio_setup["eval_interface"]
        arch_interface = gradio_setup["arch_interface"]
        
        # Test evaluation results flowing to architecture interface
        evaluation_results = sample_metrics_data
        
        with patch.object(arch_interface, 'process_evaluation_results', return_value={
            "recommended_architecture": "hybrid",
            "resource_requirements": {"cpu": 8, "memory": 32}
        }):
            arch_recommendations = arch_interface.process_evaluation_results(evaluation_results)
            
            assert arch_recommendations["recommended_architecture"] == "hybrid"
            assert arch_recommendations["resource_requirements"]["cpu"] == 8
    
    @pytest.mark.asyncio
    async def test_concurrent_interface_operations(self, gradio_setup, sample_evaluation_data, mock_platform_config):
        """Test concurrent operations across multiple interfaces."""
        eval_interface = gradio_setup["eval_interface"]
        arch_interface = gradio_setup["arch_interface"]
        
        # Test concurrent evaluation and architecture design
        with patch.object(eval_interface, 'evaluate_models', new_callable=AsyncMock) as mock_eval:
            with patch.object(arch_interface, 'design_architecture', new_callable=AsyncMock) as mock_design:
                mock_eval.return_value = [{"model": "gpt-3.5-turbo", "score": 0.95}]
                mock_design.return_value = {"architecture": "hybrid", "cost": 1000}
                
                # Run concurrent operations
                eval_task = eval_interface.evaluate_models(
                    ["gpt-3.5-turbo"], sample_evaluation_data, {}
                )
                arch_task = arch_interface.design_architecture(mock_platform_config)
                
                eval_result, arch_result = await asyncio.gather(eval_task, arch_task)
                
                assert len(eval_result) == 1
                assert eval_result[0]["model"] == "gpt-3.5-turbo"
                assert arch_result["architecture"] == "hybrid"
    
    def test_ui_component_integration(self, gradio_setup):
        """Test integration between UI components."""
        eval_interface = gradio_setup["eval_interface"]
        arch_interface = gradio_setup["arch_interface"]
        
        # Test UI component creation
        with patch.object(gr, 'Dropdown') as mock_dropdown:
            with patch.object(gr, 'Button') as mock_button:
                with patch.object(gr, 'Textbox') as mock_textbox:
                    # Create UI components
                    eval_interface._create_model_selection_ui()
                    arch_interface._create_architecture_config_ui()
                    
                    # Verify components are created
                    assert mock_dropdown.called
                    assert mock_button.called
                    assert mock_textbox.called
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, gradio_setup):
        """Test error handling across integrated components."""
        app = gradio_setup["app"]
        eval_interface = gradio_setup["eval_interface"]
        
        # Test error in model evaluation
        with patch.object(eval_interface, 'evaluate_models', side_effect=Exception("Evaluation failed")):
            with pytest.raises(Exception, match="Evaluation failed"):
                await app._handle_model_evaluation({"invalid": "data"})
        
        # Test error in architecture design
        with patch.object(app.ai_architecture_interface, 'design_architecture', side_effect=Exception("Design failed")):
            with pytest.raises(Exception, match="Design failed"):
                await app._handle_architecture_design({"invalid": "config"})
    
    def test_state_management_integration(self, gradio_setup):
        """Test state management across integrated components."""
        app = gradio_setup["app"]
        
        # Test app state initialization
        assert app.model_evaluation_interface is not None
        assert app.ai_architecture_interface is not None
        assert app.visualization_dashboard is not None
        assert app.report_generator is not None
        
        # Test state updates
        with patch.object(app, '_update_evaluation_state', return_value=True):
            result = app._update_evaluation_state({"new_results": "data"})
            assert result == True
        
        with patch.object(app, '_update_architecture_state', return_value=True):
            result = app._update_architecture_state({"new_config": "data"})
            assert result == True
    
    @pytest.mark.asyncio
    async def test_export_integration(self, gradio_setup, sample_metrics_data):
        """Test export functionality integration."""
        app = gradio_setup["app"]
        
        # Test report export
        with patch.object(app.report_generator, 'generate_report', return_value="report.pdf"):
            report = app._export_report(sample_metrics_data, "pdf")
            assert report == "report.pdf"
        
        # Test visualization export
        with patch.object(app.visualization_dashboard, 'export_chart', return_value="chart.png"):
            chart = app._export_visualization(sample_metrics_data, "accuracy", "png")
            assert chart == "chart.png"
    
    def test_configuration_integration(self, gradio_setup):
        """Test configuration integration across components."""
        app = gradio_setup["app"]
        
        # Test app configuration
        app_config = app._get_app_configuration()
        assert "model_evaluation" in app_config
        assert "ai_architecture" in app_config
        assert "visualization" in app_config
        
        # Test configuration validation
        with patch.object(app, '_validate_app_configuration', return_value=True):
            is_valid = app._validate_app_configuration(app_config)
            assert is_valid == True
    
    @pytest.mark.asyncio
    async def test_real_time_updates_integration(self, gradio_setup):
        """Test real-time updates integration."""
        app = gradio_setup["app"]
        
        # Test real-time evaluation updates
        with patch.object(app, '_stream_evaluation_updates', new_callable=AsyncMock) as mock_stream:
            mock_stream.return_value = [
                {"progress": 25, "status": "evaluating_model_1"},
                {"progress": 50, "status": "evaluating_model_2"},
                {"progress": 100, "status": "evaluation_complete"}
            ]
            
            updates = await app._stream_evaluation_updates("evaluation_id")
            
            assert len(updates) == 3
            assert updates[0]["progress"] == 25
            assert updates[-1]["status"] == "evaluation_complete"
        
        # Test real-time architecture updates
        with patch.object(app, '_stream_architecture_updates', new_callable=AsyncMock) as mock_stream:
            mock_stream.return_value = [
                {"progress": 33, "status": "designing_architecture"},
                {"progress": 66, "status": "calculating_costs"},
                {"progress": 100, "status": "architecture_complete"}
            ]
            
            updates = await app._stream_architecture_updates("architecture_id")
            
            assert len(updates) == 3
            assert updates[0]["progress"] == 33
            assert updates[-1]["status"] == "architecture_complete"
