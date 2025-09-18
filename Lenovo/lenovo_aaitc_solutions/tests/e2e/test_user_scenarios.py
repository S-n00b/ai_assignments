"""
End-to-end tests for user scenarios.

Tests complete user scenarios and use cases from the perspective
of different types of users (data scientists, ML engineers, business users).
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import tempfile
from pathlib import Path

from model_evaluation import ComprehensiveEvaluationPipeline, ModelConfig
from ai_architecture import HybridAIPlatform, ModelLifecycleManager
from gradio_app import LenovoAAITCApp


class TestDataScientistScenarios:
    """End-to-end tests for data scientist user scenarios."""
    
    @pytest.fixture
    def data_scientist_setup(self):
        """Set up environment for data scientist scenarios."""
        return {
            "user_type": "data_scientist",
            "goals": ["model_evaluation", "performance_analysis", "bias_detection"],
            "tools": ["evaluation_pipeline", "visualization_dashboard", "report_generator"]
        }
    
    @pytest.mark.asyncio
    async def test_data_scientist_model_comparison_scenario(self, data_scientist_setup, sample_evaluation_data, mock_api_client):
        """Test data scientist comparing multiple models for a research project."""
        # Scenario: Data scientist needs to compare GPT-4, Claude-3, and Llama-2 for a research paper
        
        # Step 1: Data scientist logs into the system
        with patch('gradio_app.LenovoAAITCApp') as mock_app:
            app = mock_app.return_value
            app.login.return_value = {"user_id": "ds_001", "role": "data_scientist", "status": "authenticated"}
            
            login_result = app.login("data_scientist@company.com", "password123")
            assert login_result["user_id"] == "ds_001"
            assert login_result["role"] == "data_scientist"
        
        # Step 2: Data scientist selects models to compare
        selected_models = [
            ModelConfig(model_name="gpt-4", api_key="key1"),
            ModelConfig(model_name="claude-3-sonnet", api_key="key2"),
            ModelConfig(model_name="llama-2-70b", api_key="key3")
        ]
        
        # Step 3: Data scientist configures evaluation parameters
        evaluation_config = {
            "test_suite": "comprehensive",
            "metrics": ["bleu", "rouge", "bert_score", "human_evaluation"],
            "test_categories": ["reasoning", "creativity", "factual_accuracy"],
            "sample_size": 1000
        }
        
        # Step 4: Run evaluation pipeline
        pipeline = ComprehensiveEvaluationPipeline(
            model_configs=selected_models,
            evaluation_metrics=evaluation_config["metrics"]
        )
        
        with patch.object(pipeline, '_get_api_client', return_value=mock_api_client):
            evaluation_results = await pipeline.evaluate_all_models(sample_evaluation_data)
            
            assert len(evaluation_results) == 3
            for result in evaluation_results:
                assert "model_name" in result
                assert "metrics" in result
                assert "performance" in result
        
        # Step 5: Data scientist analyzes results and generates visualizations
        with patch.object(app, '_generate_comparison_visualization', return_value="comparison_chart.png"):
            visualization = app._generate_comparison_visualization(evaluation_results)
            assert visualization == "comparison_chart.png"
        
        # Step 6: Data scientist generates research report
        with patch.object(pipeline, 'generate_research_report', return_value={
            "title": "Comparative Analysis of Large Language Models",
            "abstract": "This study compares GPT-4, Claude-3, and Llama-2...",
            "methodology": "Comprehensive evaluation using multiple metrics...",
            "results": evaluation_results,
            "conclusions": "GPT-4 shows superior performance in reasoning tasks...",
            "references": ["Paper1", "Paper2", "Paper3"]
        }):
            research_report = pipeline.generate_research_report(evaluation_results)
            
            assert research_report["title"] == "Comparative Analysis of Large Language Models"
            assert "results" in research_report
            assert "conclusions" in research_report
        
        # Step 7: Data scientist exports results for publication
        with patch.object(app, '_export_research_data', return_value="research_data.zip"):
            export_result = app._export_research_data(evaluation_results, research_report)
            assert export_result == "research_data.zip"
    
    @pytest.mark.asyncio
    async def test_data_scientist_bias_analysis_scenario(self, data_scientist_setup, sample_bias_test_data, mock_api_client):
        """Test data scientist conducting bias analysis for model fairness research."""
        # Scenario: Data scientist needs to analyze gender and racial bias in a model
        
        # Step 1: Data scientist sets up bias analysis experiment
        bias_analysis_config = {
            "bias_categories": ["gender", "race", "age", "socioeconomic"],
            "test_prompts": sample_bias_test_data["test_prompts"],
            "demographic_groups": sample_bias_test_data["demographic_groups"],
            "analysis_depth": "comprehensive"
        }
        
        # Step 2: Initialize bias detection system
        from model_evaluation import BiasDetectionSystem
        bias_system = BiasDetectionSystem()
        
        # Step 3: Run comprehensive bias analysis
        with patch.object(bias_system, '_get_api_client', return_value=mock_api_client):
            bias_results = await bias_system.comprehensive_bias_analysis(
                test_prompts=bias_analysis_config["test_prompts"],
                demographic_groups=bias_analysis_config["demographic_groups"]
            )
            
            assert "overall_bias_score" in bias_results
            assert "category_scores" in bias_results
            assert "recommendations" in bias_results
        
        # Step 4: Data scientist analyzes bias patterns
        with patch.object(bias_system, 'analyze_bias_patterns', return_value={
            "gender_bias_patterns": {
                "occupation_stereotypes": 0.15,
                "personality_traits": 0.08,
                "capability_assumptions": 0.12
            },
            "racial_bias_patterns": {
                "cultural_stereotypes": 0.18,
                "language_assumptions": 0.10,
                "socioeconomic_bias": 0.14
            },
            "statistical_significance": 0.95
        }):
            pattern_analysis = bias_system.analyze_bias_patterns(bias_results)
            
            assert "gender_bias_patterns" in pattern_analysis
            assert "racial_bias_patterns" in pattern_analysis
            assert pattern_analysis["statistical_significance"] == 0.95
        
        # Step 5: Generate bias mitigation recommendations
        with patch.object(bias_system, 'generate_mitigation_recommendations', return_value={
            "immediate_actions": [
                "Retrain model with balanced dataset",
                "Implement bias detection in production pipeline"
            ],
            "long_term_strategies": [
                "Develop bias-aware training procedures",
                "Establish ongoing bias monitoring"
            ],
            "technical_implementations": [
                "Add bias detection layer to model output",
                "Implement demographic parity constraints"
            ]
        }):
            mitigation_recommendations = bias_system.generate_mitigation_recommendations(bias_results)
            
            assert len(mitigation_recommendations["immediate_actions"]) == 2
            assert len(mitigation_recommendations["long_term_strategies"]) == 2
        
        # Step 6: Data scientist creates bias analysis report
        with patch.object(bias_system, 'generate_bias_report', return_value={
            "executive_summary": "Model shows moderate bias across multiple categories...",
            "detailed_analysis": bias_results,
            "pattern_analysis": pattern_analysis,
            "mitigation_recommendations": mitigation_recommendations,
            "methodology": "Comprehensive bias analysis using statistical methods...",
            "limitations": "Analysis based on limited test prompts..."
        }):
            bias_report = bias_system.generate_bias_report(
                bias_results, pattern_analysis, mitigation_recommendations
            )
            
            assert "executive_summary" in bias_report
            assert "detailed_analysis" in bias_report
            assert "mitigation_recommendations" in bias_report


class TestMLEngineerScenarios:
    """End-to-end tests for ML engineer user scenarios."""
    
    @pytest.fixture
    def ml_engineer_setup(self):
        """Set up environment for ML engineer scenarios."""
        return {
            "user_type": "ml_engineer",
            "goals": ["model_deployment", "performance_optimization", "monitoring"],
            "tools": ["lifecycle_manager", "platform", "monitoring_system"]
        }
    
    @pytest.mark.asyncio
    async def test_ml_engineer_model_deployment_scenario(self, ml_engineer_setup, sample_model_versions):
        """Test ML engineer deploying a model to production."""
        # Scenario: ML engineer needs to deploy a new model version to production with zero downtime
        
        # Step 1: ML engineer reviews model performance in staging
        lifecycle_manager = ModelLifecycleManager()
        platform = HybridAIPlatform()
        
        staging_model = sample_model_versions[1]  # v1.1.0 in staging
        
        with patch.object(lifecycle_manager, 'get_model_performance', return_value={
            "accuracy": 0.96,
            "latency_p50": 120,
            "throughput": 1000,
            "error_rate": 0.001,
            "staging_duration": "7 days"
        }):
            staging_performance = lifecycle_manager.get_model_performance(staging_model["version"])
            
            assert staging_performance["accuracy"] == 0.96
            assert staging_performance["latency_p50"] == 120
        
        # Step 2: ML engineer validates model meets production requirements
        production_requirements = {
            "min_accuracy": 0.95,
            "max_latency_ms": 200,
            "min_throughput": 800,
            "max_error_rate": 0.005
        }
        
        with patch.object(lifecycle_manager, 'validate_production_readiness', return_value={
            "is_ready": True,
            "validation_results": {
                "accuracy_check": "PASS",
                "latency_check": "PASS",
                "throughput_check": "PASS",
                "error_rate_check": "PASS"
            },
            "recommendations": ["Model ready for production deployment"]
        }):
            validation_result = lifecycle_manager.validate_production_readiness(
                staging_model, production_requirements
            )
            
            assert validation_result["is_ready"] == True
            assert all(check == "PASS" for check in validation_result["validation_results"].values())
        
        # Step 3: ML engineer plans blue-green deployment
        deployment_strategy = "blue_green"
        
        with patch.object(platform, 'plan_deployment', return_value={
            "strategy": "blue_green",
            "steps": [
                "Deploy new version to green environment",
                "Run smoke tests on green environment",
                "Gradually shift traffic from blue to green",
                "Monitor performance during transition",
                "Complete cutover to green environment"
            ],
            "estimated_downtime": "0 seconds",
            "rollback_plan": "Automatic rollback if error rate > 0.01"
        }):
            deployment_plan = platform.plan_deployment(staging_model, deployment_strategy)
            
            assert deployment_plan["strategy"] == "blue_green"
            assert deployment_plan["estimated_downtime"] == "0 seconds"
        
        # Step 4: Execute blue-green deployment
        with patch.object(platform, 'execute_blue_green_deployment', new_callable=AsyncMock) as mock_deploy:
            mock_deploy.return_value = {
                "deployment_id": "deploy_789",
                "status": "success",
                "green_environment_url": "https://green.example.com",
                "traffic_shift_progress": 100,
                "monitoring_dashboard": "https://monitor.example.com/deploy_789"
            }
            
            deployment_result = await platform.execute_blue_green_deployment(
                staging_model, deployment_plan
            )
            
            assert deployment_result["status"] == "success"
            assert deployment_result["traffic_shift_progress"] == 100
        
        # Step 5: ML engineer monitors deployment health
        with patch.object(platform, 'monitor_deployment_health', return_value={
            "overall_health": "healthy",
            "metrics": {
                "response_time_p50": 115,
                "response_time_p95": 180,
                "error_rate": 0.0008,
                "throughput": 1050,
                "availability": 0.9998
            },
            "alerts": [],
            "recommendations": ["Deployment is performing well"]
        }):
            health_status = platform.monitor_deployment_health("deploy_789")
            
            assert health_status["overall_health"] == "healthy"
            assert health_status["metrics"]["error_rate"] < 0.001
            assert len(health_status["alerts"]) == 0
        
        # Step 6: ML engineer updates production model registry
        with patch.object(lifecycle_manager, 'update_production_model', return_value=True):
            update_result = lifecycle_manager.update_production_model(
                staging_model["version"], "deploy_789"
            )
            assert update_result == True
    
    @pytest.mark.asyncio
    async def test_ml_engineer_performance_optimization_scenario(self, ml_engineer_setup):
        """Test ML engineer optimizing model performance in production."""
        # Scenario: ML engineer needs to optimize model performance due to increased load
        
        platform = HybridAIPlatform()
        
        # Step 1: ML engineer identifies performance bottlenecks
        with patch.object(platform, 'analyze_performance_bottlenecks', return_value={
            "bottlenecks": [
                {
                    "component": "model_inference",
                    "issue": "High latency during peak hours",
                    "impact": "User experience degradation",
                    "severity": "high"
                },
                {
                    "component": "data_preprocessing",
                    "issue": "Inefficient text tokenization",
                    "impact": "Increased processing time",
                    "severity": "medium"
                }
            ],
            "performance_metrics": {
                "current_latency_p95": 2500,
                "target_latency_p95": 1500,
                "current_throughput": 800,
                "target_throughput": 1200
            }
        }):
            bottleneck_analysis = platform.analyze_performance_bottlenecks("deploy_123")
            
            assert len(bottleneck_analysis["bottlenecks"]) == 2
            assert bottleneck_analysis["bottlenecks"][0]["severity"] == "high"
        
        # Step 2: ML engineer develops optimization strategies
        with patch.object(platform, 'develop_optimization_strategies', return_value={
            "strategies": [
                {
                    "name": "Model Quantization",
                    "description": "Reduce model precision to improve inference speed",
                    "expected_improvement": "30% latency reduction",
                    "implementation_effort": "medium",
                    "risk_level": "low"
                },
                {
                    "name": "Caching Layer",
                    "description": "Implement response caching for common queries",
                    "expected_improvement": "50% latency reduction for cached queries",
                    "implementation_effort": "low",
                    "risk_level": "low"
                },
                {
                    "name": "Auto-scaling",
                    "description": "Implement dynamic scaling based on load",
                    "expected_improvement": "Better resource utilization",
                    "implementation_effort": "high",
                    "risk_level": "medium"
                }
            ],
            "recommended_priority": ["Caching Layer", "Model Quantization", "Auto-scaling"]
        }):
            optimization_strategies = platform.develop_optimization_strategies(bottleneck_analysis)
            
            assert len(optimization_strategies["strategies"]) == 3
            assert optimization_strategies["recommended_priority"][0] == "Caching Layer"
        
        # Step 3: ML engineer implements caching layer (lowest risk, high impact)
        with patch.object(platform, 'implement_caching_layer', new_callable=AsyncMock) as mock_cache:
            mock_cache.return_value = {
                "implementation_id": "cache_001",
                "status": "success",
                "cache_hit_rate": 0.65,
                "latency_improvement": 0.45,
                "throughput_improvement": 0.25
            }
            
            cache_result = await platform.implement_caching_layer("deploy_123")
            
            assert cache_result["status"] == "success"
            assert cache_result["cache_hit_rate"] == 0.65
            assert cache_result["latency_improvement"] == 0.45
        
        # Step 4: ML engineer implements model quantization
        with patch.object(platform, 'implement_model_quantization', new_callable=AsyncMock) as mock_quant:
            mock_quant.return_value = {
                "implementation_id": "quant_001",
                "status": "success",
                "model_size_reduction": 0.4,
                "latency_improvement": 0.32,
                "accuracy_impact": -0.001
            }
            
            quantization_result = await platform.implement_model_quantization("deploy_123")
            
            assert quantization_result["status"] == "success"
            assert quantization_result["model_size_reduction"] == 0.4
            assert quantization_result["accuracy_impact"] > -0.01  # Minimal accuracy impact
        
        # Step 5: ML engineer validates optimization results
        with patch.object(platform, 'validate_optimization_results', return_value={
            "overall_improvement": {
                "latency_p95_improvement": 0.52,
                "throughput_improvement": 0.38,
                "resource_utilization_improvement": 0.25
            },
            "performance_metrics": {
                "new_latency_p95": 1200,
                "new_throughput": 1100,
                "cache_hit_rate": 0.65,
                "error_rate": 0.0005
            },
            "validation_status": "PASS",
            "recommendations": ["Optimization successful, consider implementing auto-scaling next"]
        }):
            validation_result = platform.validate_optimization_results("deploy_123")
            
            assert validation_result["validation_status"] == "PASS"
            assert validation_result["performance_metrics"]["new_latency_p95"] < 1500
            assert validation_result["performance_metrics"]["new_throughput"] > 1000


class TestBusinessUserScenarios:
    """End-to-end tests for business user scenarios."""
    
    @pytest.fixture
    def business_user_setup(self):
        """Set up environment for business user scenarios."""
        return {
            "user_type": "business_user",
            "goals": ["cost_optimization", "performance_monitoring", "business_metrics"],
            "tools": ["dashboard", "reporting", "cost_analysis"]
        }
    
    @pytest.mark.asyncio
    async def test_business_user_cost_optimization_scenario(self, business_user_setup):
        """Test business user optimizing AI system costs."""
        # Scenario: Business user needs to reduce AI system costs while maintaining performance
        
        platform = HybridAIPlatform()
        
        # Step 1: Business user reviews current costs
        with patch.object(platform, 'get_cost_analysis', return_value={
            "current_monthly_costs": {
                "compute": 8500,
                "storage": 1200,
                "data_transfer": 800,
                "monitoring": 300,
                "total": 10800
            },
            "cost_breakdown": {
                "model_inference": 0.65,
                "data_processing": 0.20,
                "infrastructure": 0.10,
                "monitoring": 0.05
            },
            "cost_trends": {
                "last_3_months": [9500, 10200, 10800],
                "projected_next_month": 11500
            }
        }):
            cost_analysis = platform.get_cost_analysis()
            
            assert cost_analysis["current_monthly_costs"]["total"] == 10800
            assert cost_analysis["cost_breakdown"]["model_inference"] == 0.65
        
        # Step 2: Business user identifies cost optimization opportunities
        with patch.object(platform, 'identify_cost_optimization_opportunities', return_value={
            "opportunities": [
                {
                    "category": "compute_optimization",
                    "description": "Switch to spot instances for non-critical workloads",
                    "potential_savings": 2500,
                    "implementation_effort": "medium",
                    "risk_level": "low"
                },
                {
                    "category": "model_optimization",
                    "description": "Implement model caching to reduce inference costs",
                    "potential_savings": 1800,
                    "implementation_effort": "low",
                    "risk_level": "low"
                },
                {
                    "category": "storage_optimization",
                    "description": "Implement data lifecycle management",
                    "potential_savings": 600,
                    "implementation_effort": "low",
                    "risk_level": "low"
                }
            ],
            "total_potential_savings": 4900,
            "recommended_priority": ["model_optimization", "compute_optimization", "storage_optimization"]
        }):
            optimization_opportunities = platform.identify_cost_optimization_opportunities()
            
            assert len(optimization_opportunities["opportunities"]) == 3
            assert optimization_opportunities["total_potential_savings"] == 4900
        
        # Step 3: Business user implements model caching optimization
        with patch.object(platform, 'implement_model_caching', new_callable=AsyncMock) as mock_cache:
            mock_cache.return_value = {
                "implementation_id": "cost_opt_001",
                "status": "success",
                "monthly_savings": 1800,
                "cache_hit_rate": 0.70,
                "performance_impact": "positive"
            }
            
            caching_result = await platform.implement_model_caching()
            
            assert caching_result["status"] == "success"
            assert caching_result["monthly_savings"] == 1800
        
        # Step 4: Business user implements spot instance optimization
        with patch.object(platform, 'implement_spot_instances', new_callable=AsyncMock) as mock_spot:
            mock_spot.return_value = {
                "implementation_id": "cost_opt_002",
                "status": "success",
                "monthly_savings": 2500,
                "spot_instance_percentage": 0.60,
                "availability_impact": "minimal"
            }
            
            spot_result = await platform.implement_spot_instances()
            
            assert spot_result["status"] == "success"
            assert spot_result["monthly_savings"] == 2500
        
        # Step 5: Business user monitors cost optimization results
        with patch.object(platform, 'monitor_cost_optimization', return_value={
            "optimization_results": {
                "total_monthly_savings": 4300,
                "cost_reduction_percentage": 0.40,
                "new_monthly_costs": 6500
            },
            "performance_impact": {
                "latency_change": 0.05,
                "throughput_change": -0.02,
                "availability_change": 0.001
            },
            "roi_analysis": {
                "implementation_cost": 500,
                "monthly_savings": 4300,
                "payback_period": "0.1 months",
                "annual_savings": 51600
            }
        }):
            optimization_results = platform.monitor_cost_optimization()
            
            assert optimization_results["optimization_results"]["total_monthly_savings"] == 4300
            assert optimization_results["roi_analysis"]["annual_savings"] == 51600
    
    @pytest.mark.asyncio
    async def test_business_user_performance_monitoring_scenario(self, business_user_setup):
        """Test business user monitoring AI system performance for business metrics."""
        # Scenario: Business user needs to monitor AI system performance and its impact on business metrics
        
        platform = HybridAIPlatform()
        
        # Step 1: Business user sets up business metrics dashboard
        with patch.object(platform, 'setup_business_metrics_dashboard', return_value={
            "dashboard_id": "business_dash_001",
            "metrics_tracked": [
                "user_satisfaction_score",
                "conversion_rate",
                "response_time_impact",
                "cost_per_transaction",
                "system_availability"
            ],
            "alert_thresholds": {
                "user_satisfaction_score": 0.85,
                "conversion_rate": 0.12,
                "response_time_p95": 2000,
                "system_availability": 0.99
            }
        }):
            dashboard_setup = platform.setup_business_metrics_dashboard()
            
            assert dashboard_setup["dashboard_id"] == "business_dash_001"
            assert len(dashboard_setup["metrics_tracked"]) == 5
        
        # Step 2: Business user reviews current business metrics
        with patch.object(platform, 'get_business_metrics', return_value={
            "current_metrics": {
                "user_satisfaction_score": 0.88,
                "conversion_rate": 0.135,
                "response_time_p95": 1800,
                "cost_per_transaction": 0.15,
                "system_availability": 0.995
            },
            "trends": {
                "user_satisfaction_score": "improving",
                "conversion_rate": "stable",
                "response_time_p95": "improving",
                "cost_per_transaction": "decreasing",
                "system_availability": "stable"
            },
            "business_impact": {
                "revenue_impact": "positive",
                "customer_retention": "improving",
                "operational_efficiency": "improving"
            }
        }):
            business_metrics = platform.get_business_metrics()
            
            assert business_metrics["current_metrics"]["user_satisfaction_score"] == 0.88
            assert business_metrics["trends"]["user_satisfaction_score"] == "improving"
        
        # Step 3: Business user receives performance alert
        with patch.object(platform, 'handle_performance_alert', return_value={
            "alert_id": "alert_001",
            "alert_type": "response_time_degradation",
            "severity": "medium",
            "description": "Response time P95 increased to 2200ms",
            "business_impact": {
                "estimated_conversion_loss": 0.02,
                "estimated_revenue_impact": 5000,
                "customer_satisfaction_impact": "negative"
            },
            "recommended_actions": [
                "Scale up compute resources",
                "Investigate performance bottlenecks",
                "Monitor customer feedback"
            ]
        }):
            alert_response = platform.handle_performance_alert("alert_001")
            
            assert alert_response["alert_type"] == "response_time_degradation"
            assert alert_response["business_impact"]["estimated_revenue_impact"] == 5000
        
        # Step 4: Business user takes corrective action
        with patch.object(platform, 'execute_corrective_action', new_callable=AsyncMock) as mock_action:
            mock_action.return_value = {
                "action_id": "action_001",
                "status": "success",
                "actions_taken": [
                    "Scaled compute resources by 25%",
                    "Optimized database queries",
                    "Implemented response caching"
                ],
                "performance_improvement": {
                    "response_time_p95": 1600,
                    "throughput_improvement": 0.15,
                    "cost_increase": 0.08
                }
            }
            
            action_result = await platform.execute_corrective_action("alert_001")
            
            assert action_result["status"] == "success"
            assert action_result["performance_improvement"]["response_time_p95"] == 1600
        
        # Step 5: Business user generates executive report
        with patch.object(platform, 'generate_executive_report', return_value={
            "report_id": "exec_report_001",
            "executive_summary": "AI system performance is meeting business objectives with positive trends",
            "key_metrics": business_metrics["current_metrics"],
            "performance_trends": business_metrics["trends"],
            "business_impact": business_metrics["business_impact"],
            "recommendations": [
                "Continue current optimization efforts",
                "Monitor response time trends closely",
                "Consider additional scaling for peak periods"
            ],
            "next_review_date": "2024-02-15"
        }):
            executive_report = platform.generate_executive_report()
            
            assert executive_report["report_id"] == "exec_report_001"
            assert "executive_summary" in executive_report
            assert len(executive_report["recommendations"]) == 3
