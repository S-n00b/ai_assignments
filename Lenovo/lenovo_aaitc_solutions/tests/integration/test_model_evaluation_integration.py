"""
Integration tests for model evaluation system.

Tests the integration between different components of the model evaluation
framework including pipeline, robustness testing, bias detection, and prompt registries.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd

from model_evaluation import (
    ComprehensiveEvaluationPipeline,
    RobustnessTestingSuite,
    BiasDetectionSystem,
    PromptRegistryManager,
    ModelConfig
)


class TestModelEvaluationIntegration:
    """Integration tests for model evaluation components."""
    
    @pytest.fixture
    def evaluation_setup(self, sample_model_config, sample_evaluation_data):
        """Set up evaluation components for integration testing."""
        pipeline = ComprehensiveEvaluationPipeline(
            model_configs=[sample_model_config],
            evaluation_metrics=["bleu", "rouge", "bert_score"]
        )
        
        robustness_suite = RobustnessTestingSuite(model_config=sample_model_config)
        bias_system = BiasDetectionSystem()
        prompt_manager = PromptRegistryManager()
        
        return {
            "pipeline": pipeline,
            "robustness_suite": robustness_suite,
            "bias_system": bias_system,
            "prompt_manager": prompt_manager,
            "test_data": sample_evaluation_data
        }
    
    @pytest.mark.asyncio
    async def test_complete_evaluation_workflow(self, evaluation_setup, mock_api_client):
        """Test complete evaluation workflow integration."""
        pipeline = evaluation_setup["pipeline"]
        test_data = evaluation_setup["test_data"]
        
        with patch.object(pipeline, '_get_api_client', return_value=mock_api_client):
            # Run evaluation
            results = await pipeline.evaluate_all_models(test_data)
            
            # Verify results structure
            assert isinstance(results, list)
            assert len(results) > 0
            
            for result in results:
                assert "model_name" in result
                assert "metrics" in result
                assert "performance" in result
                assert "robustness" in result
                assert "bias_analysis" in result
    
    @pytest.mark.asyncio
    async def test_robustness_integration_with_pipeline(self, evaluation_setup, mock_api_client):
        """Test robustness testing integration with evaluation pipeline."""
        pipeline = evaluation_setup["pipeline"]
        robustness_suite = evaluation_setup["robustness_suite"]
        test_data = evaluation_setup["test_data"]
        
        with patch.object(pipeline, '_get_api_client', return_value=mock_api_client):
            with patch.object(robustness_suite, '_get_api_client', return_value=mock_api_client):
                # Run evaluation with robustness testing
                results = await pipeline.evaluate_all_models(test_data)
                
                # Verify robustness results are included
                for result in results:
                    robustness_results = result["robustness"]
                    assert "adversarial_tests" in robustness_results
                    assert "noise_tolerance" in robustness_results
                    assert "edge_cases" in robustness_results
                    assert "safety_assessment" in robustness_results
    
    @pytest.mark.asyncio
    async def test_bias_detection_integration(self, evaluation_setup, sample_bias_test_data, mock_api_client):
        """Test bias detection integration with evaluation pipeline."""
        pipeline = evaluation_setup["pipeline"]
        bias_system = evaluation_setup["bias_system"]
        
        with patch.object(pipeline, '_get_api_client', return_value=mock_api_client):
            with patch.object(bias_system, '_get_api_client', return_value=mock_api_client):
                # Run evaluation with bias detection
                results = await pipeline.evaluate_all_models(sample_bias_test_data)
                
                # Verify bias analysis results
                for result in results:
                    bias_results = result["bias_analysis"]
                    assert "overall_bias_score" in bias_results
                    assert "category_scores" in bias_results
                    assert "recommendations" in bias_results
                    assert "gender_bias" in bias_results["category_scores"]
                    assert "racial_bias" in bias_results["category_scores"]
    
    @pytest.mark.asyncio
    async def test_prompt_registry_integration(self, evaluation_setup):
        """Test prompt registry integration with evaluation pipeline."""
        pipeline = evaluation_setup["pipeline"]
        prompt_manager = evaluation_setup["prompt_manager"]
        
        # Get prompts from registry
        registry_prompts = await prompt_manager.get_evaluation_prompts(
            registry="diffusiondb",
            category="creative",
            limit=5
        )
        
        # Use registry prompts in evaluation
        test_data = {
            "prompts": [prompt["text"] for prompt in registry_prompts],
            "expected_outputs": ["Expected output"] * len(registry_prompts),
            "task_types": ["creative"] * len(registry_prompts)
        }
        
        with patch.object(pipeline, '_get_api_client', return_value=Mock()):
            results = await pipeline.evaluate_all_models(test_data)
            
            assert len(results) > 0
            assert len(results[0]["metrics"]["predictions"]) == len(registry_prompts)
    
    def test_metrics_aggregation_across_components(self, evaluation_setup):
        """Test metrics aggregation across different evaluation components."""
        pipeline = evaluation_setup["pipeline"]
        
        # Simulate results from different components
        evaluation_results = {
            "bleu_score": 0.85,
            "rouge_score": 0.88,
            "bert_score": 0.92
        }
        
        robustness_results = {
            "adversarial_success_rate": 0.95,
            "noise_tolerance_score": 0.87,
            "edge_case_handling": 0.90
        }
        
        bias_results = {
            "overall_bias_score": 0.12,
            "gender_bias": 0.08,
            "racial_bias": 0.15
        }
        
        # Aggregate metrics
        aggregated = pipeline._aggregate_metrics(
            evaluation_results, robustness_results, bias_results
        )
        
        assert "overall_score" in aggregated
        assert "component_scores" in aggregated
        assert "recommendations" in aggregated
        assert aggregated["overall_score"] > 0
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, evaluation_setup):
        """Test error handling across integrated components."""
        pipeline = evaluation_setup["pipeline"]
        
        # Test with invalid model config
        invalid_config = ModelConfig(
            model_name="invalid-model",
            api_key="invalid-key"
        )
        
        pipeline.model_configs = [invalid_config]
        
        with patch.object(pipeline, '_get_api_client', side_effect=Exception("API Error")):
            with pytest.raises(Exception):
                await pipeline.evaluate_all_models(evaluation_setup["test_data"])
    
    def test_data_flow_between_components(self, evaluation_setup):
        """Test data flow between evaluation components."""
        pipeline = evaluation_setup["pipeline"]
        robustness_suite = evaluation_setup["robustness_suite"]
        bias_system = evaluation_setup["bias_system"]
        
        # Test data transformation through components
        original_prompts = ["What is AI?", "Explain machine learning"]
        
        # Pipeline processes prompts
        processed_prompts = pipeline._preprocess_prompts(original_prompts)
        assert len(processed_prompts) == len(original_prompts)
        
        # Robustness suite generates variations
        adversarial_prompts = robustness_suite._generate_adversarial_prompts(original_prompts[0])
        assert len(adversarial_prompts) > 0
        
        # Bias system analyzes prompts
        bias_analysis = bias_system._analyze_prompt_bias(original_prompts)
        assert "bias_indicators" in bias_analysis
    
    @pytest.mark.asyncio
    async def test_concurrent_evaluation_integration(self, evaluation_setup, mock_api_client):
        """Test concurrent evaluation across multiple models."""
        # Create multiple model configs
        model_configs = [
            ModelConfig(model_name="gpt-3.5-turbo", api_key="key1"),
            ModelConfig(model_name="claude-3-sonnet", api_key="key2"),
            ModelConfig(model_name="llama-2-70b", api_key="key3")
        ]
        
        pipeline = ComprehensiveEvaluationPipeline(
            model_configs=model_configs,
            evaluation_metrics=["bleu", "rouge"]
        )
        
        with patch.object(pipeline, '_get_api_client', return_value=mock_api_client):
            # Run concurrent evaluation
            results = await pipeline.evaluate_all_models(evaluation_setup["test_data"])
            
            assert len(results) == 3
            assert all("model_name" in result for result in results)
            assert all("metrics" in result for result in results)
    
    def test_configuration_consistency_across_components(self, evaluation_setup):
        """Test configuration consistency across evaluation components."""
        pipeline = evaluation_setup["pipeline"]
        robustness_suite = evaluation_setup["robustness_suite"]
        
        # Verify consistent model configuration
        assert pipeline.model_configs[0].model_name == robustness_suite.model_config.model_name
        assert pipeline.model_configs[0].api_key == robustness_suite.model_config.api_key
        
        # Test configuration validation
        assert pipeline._validate_model_configs() == True
        assert robustness_suite._validate_model_config() == True
    
    @pytest.mark.asyncio
    async def test_report_generation_integration(self, evaluation_setup, mock_api_client):
        """Test integrated report generation across all components."""
        pipeline = evaluation_setup["pipeline"]
        
        with patch.object(pipeline, '_get_api_client', return_value=mock_api_client):
            # Run complete evaluation
            results = await pipeline.evaluate_all_models(evaluation_setup["test_data"])
            
            # Generate comprehensive report
            report = pipeline.generate_comprehensive_report(results)
            
            assert "executive_summary" in report
            assert "detailed_results" in report
            assert "robustness_analysis" in report
            assert "bias_analysis" in report
            assert "recommendations" in report
            assert "visualizations" in report
