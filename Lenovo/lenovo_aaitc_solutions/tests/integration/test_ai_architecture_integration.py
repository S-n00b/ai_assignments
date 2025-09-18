"""
Integration tests for AI architecture system.

Tests the integration between different components of the AI architecture
framework including platform, lifecycle management, agents, and RAG systems.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from ai_architecture import (
    HybridAIPlatform,
    ModelLifecycleManager,
    AgenticComputingFramework,
    RAGSystem,
    BaseAgent,
    AgentMessage,
    MessageType
)


class TestAIArchitectureIntegration:
    """Integration tests for AI architecture components."""
    
    @pytest.fixture
    def architecture_setup(self, mock_platform_config, sample_agents, sample_documents):
        """Set up AI architecture components for integration testing."""
        platform = HybridAIPlatform(config=mock_platform_config)
        lifecycle_manager = ModelLifecycleManager()
        agent_framework = AgenticComputingFramework()
        rag_system = RAGSystem()
        
        return {
            "platform": platform,
            "lifecycle_manager": lifecycle_manager,
            "agent_framework": agent_framework,
            "rag_system": rag_system,
            "agents": sample_agents,
            "documents": sample_documents
        }
    
    @pytest.mark.asyncio
    async def test_platform_lifecycle_integration(self, architecture_setup, sample_model_versions):
        """Test integration between platform and lifecycle management."""
        platform = architecture_setup["platform"]
        lifecycle_manager = architecture_setup["lifecycle_manager"]
        
        model_version = sample_model_versions[0]
        
        with patch.object(lifecycle_manager, 'register_model_version', return_value=True):
            with patch.object(platform, 'deploy_model', new_callable=AsyncMock) as mock_deploy:
                mock_deploy.return_value = {"deployment_id": "deploy_123", "status": "success"}
                
                # Register model version
                registration_result = lifecycle_manager.register_model_version(model_version)
                assert registration_result == True
                
                # Deploy through platform
                deployment_result = await platform.deploy_model(model_version)
                assert deployment_result["deployment_id"] == "deploy_123"
                assert deployment_result["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_agent_rag_integration(self, architecture_setup):
        """Test integration between agent framework and RAG system."""
        agent_framework = architecture_setup["agent_framework"]
        rag_system = architecture_setup["rag_system"]
        documents = architecture_setup["documents"]
        
        # Add documents to RAG system
        with patch.object(rag_system.vector_store, 'add_documents', new_callable=AsyncMock):
            await rag_system.add_documents(documents)
        
        # Create agent that uses RAG system
        research_agent = BaseAgent(
            agent_id="research_agent",
            name="Research Agent",
            role="researcher",
            capabilities=["research", "document_retrieval"]
        )
        
        # Register agent
        with patch.object(agent_framework, '_create_agent', return_value=research_agent):
            agent_framework.register_agent(architecture_setup["agents"][0])
        
        # Test agent using RAG system for research
        query = "What is artificial intelligence?"
        
        with patch.object(rag_system, 'retrieve_relevant_documents', new_callable=AsyncMock) as mock_retrieve:
            mock_retrieve.return_value = [
                {"content": "AI is the simulation of human intelligence", "score": 0.95}
            ]
            
            relevant_docs = await rag_system.retrieve_relevant_documents(query, top_k=3)
            
            assert len(relevant_docs) > 0
            assert relevant_docs[0]["content"] == "AI is the simulation of human intelligence"
    
    @pytest.mark.asyncio
    async def test_multi_agent_collaboration(self, architecture_setup):
        """Test multi-agent collaboration workflow."""
        agent_framework = architecture_setup["agent_framework"]
        
        # Create multiple agents
        research_agent = BaseAgent(
            agent_id="research_agent",
            name="Research Agent",
            role="researcher",
            capabilities=["research", "data_collection"]
        )
        
        writing_agent = BaseAgent(
            agent_id="writing_agent",
            name="Writing Agent",
            role="writer",
            capabilities=["content_generation", "editing"]
        )
        
        # Register agents
        with patch.object(agent_framework, '_create_agent', side_effect=[research_agent, writing_agent]):
            agent_framework.register_agent(architecture_setup["agents"][0])
            agent_framework.register_agent(architecture_setup["agents"][1])
        
        # Test agent collaboration
        task = {
            "type": "research_and_write",
            "description": "Research AI trends and write a summary",
            "required_agents": ["research_agent", "writing_agent"]
        }
        
        with patch.object(agent_framework, '_execute_coordination', new_callable=AsyncMock) as mock_coord:
            mock_coord.return_value = {
                "status": "completed",
                "result": "AI trends summary completed",
                "collaboration_log": ["research_agent: data collected", "writing_agent: summary written"]
            }
            
            result = await agent_framework.coordinate_agents(task)
            
            assert result["status"] == "completed"
            assert "collaboration_log" in result
            assert len(result["collaboration_log"]) == 2
    
    @pytest.mark.asyncio
    async def test_platform_agent_integration(self, architecture_setup):
        """Test integration between platform and agent framework."""
        platform = architecture_setup["platform"]
        agent_framework = architecture_setup["agent_framework"]
        
        # Create deployment task for agents
        deployment_task = {
            "type": "model_deployment",
            "description": "Deploy model to production",
            "required_agents": ["deployment_agent"],
            "platform_config": architecture_setup["platform"].config
        }
        
        with patch.object(agent_framework, '_execute_coordination', new_callable=AsyncMock) as mock_coord:
            with patch.object(platform, 'deploy_model', new_callable=AsyncMock) as mock_deploy:
                mock_coord.return_value = {"status": "coordinated", "deployment_ready": True}
                mock_deploy.return_value = {"deployment_id": "deploy_456", "status": "success"}
                
                # Coordinate agents for deployment
                coordination_result = await agent_framework.coordinate_agents(deployment_task)
                assert coordination_result["status"] == "coordinated"
                
                # Execute deployment through platform
                deployment_result = await platform.deploy_model({"model_name": "test_model"})
                assert deployment_result["deployment_id"] == "deploy_456"
    
    @pytest.mark.asyncio
    async def test_rag_lifecycle_integration(self, architecture_setup, sample_model_versions):
        """Test integration between RAG system and lifecycle management."""
        rag_system = architecture_setup["rag_system"]
        lifecycle_manager = architecture_setup["lifecycle_manager"]
        documents = architecture_setup["documents"]
        
        # Add documents to RAG system
        with patch.object(rag_system.vector_store, 'add_documents', new_callable=AsyncMock):
            await rag_system.add_documents(documents)
        
        # Create RAG model version
        rag_model_version = {
            "version": "v1.0.0",
            "model_path": "/models/rag_v1.0.0",
            "metrics": {"retrieval_accuracy": 0.95, "generation_quality": 0.92},
            "created_at": datetime.now(),
            "status": "staging",
            "rag_config": {
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "chunk_size": 512,
                "similarity_threshold": 0.7
            }
        }
        
        # Register RAG model version
        with patch.object(lifecycle_manager, 'register_model_version', return_value=True):
            registration_result = lifecycle_manager.register_model_version(rag_model_version)
            assert registration_result == True
        
        # Test RAG system with registered model
        query = "What is machine learning?"
        
        with patch.object(rag_system, 'retrieve_relevant_documents', new_callable=AsyncMock) as mock_retrieve:
            with patch.object(rag_system, 'generate_response', new_callable=AsyncMock) as mock_generate:
                mock_retrieve.return_value = [
                    {"content": "Machine learning is a subset of AI", "score": 0.95}
                ]
                mock_generate.return_value = "Machine learning is a subset of artificial intelligence..."
                
                # Test RAG pipeline
                relevant_docs = await rag_system.retrieve_relevant_documents(query, top_k=3)
                response = await rag_system.generate_response(query, relevant_docs)
                
                assert len(relevant_docs) > 0
                assert len(response) > 0
    
    def test_configuration_consistency_across_components(self, architecture_setup):
        """Test configuration consistency across AI architecture components."""
        platform = architecture_setup["platform"]
        lifecycle_manager = architecture_setup["lifecycle_manager"]
        agent_framework = architecture_setup["agent_framework"]
        rag_system = architecture_setup["rag_system"]
        
        # Test platform configuration
        assert platform.config is not None
        assert platform.deployment_target == "hybrid"
        
        # Test lifecycle manager configuration
        assert lifecycle_manager.model_registry is not None
        assert lifecycle_manager.deployment_tracker is not None
        
        # Test agent framework configuration
        assert agent_framework.agents == {}
        assert agent_framework.message_queue is not None
        
        # Test RAG system configuration
        assert rag_system.vector_store is not None
        assert rag_system.embedding_model is not None
    
    @pytest.mark.asyncio
    async def test_error_propagation_across_components(self, architecture_setup):
        """Test error propagation across integrated components."""
        platform = architecture_setup["platform"]
        lifecycle_manager = architecture_setup["lifecycle_manager"]
        
        # Test error in lifecycle manager affecting platform
        with patch.object(lifecycle_manager, 'register_model_version', side_effect=Exception("Registration failed")):
            with pytest.raises(Exception, match="Registration failed"):
                lifecycle_manager.register_model_version({"invalid": "config"})
        
        # Test error in platform affecting deployment
        with patch.object(platform, 'deploy_model', side_effect=Exception("Deployment failed")):
            with pytest.raises(Exception, match="Deployment failed"):
                await platform.deploy_model({"invalid": "model"})
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, architecture_setup, sample_performance_metrics):
        """Test performance monitoring across integrated components."""
        platform = architecture_setup["platform"]
        agent_framework = architecture_setup["agent_framework"]
        
        # Test platform performance monitoring
        with patch.object(platform, '_collect_metrics', return_value=sample_performance_metrics):
            platform_metrics = platform.monitor_performance("deploy_123")
            assert platform_metrics["throughput"] == 100
            assert platform_metrics["latency_p50"] == 500
        
        # Test agent performance monitoring
        with patch.object(agent_framework, '_collect_agent_metrics', return_value={
            "tasks_completed": 15,
            "success_rate": 0.93,
            "average_response_time": 1.1
        }):
            agent_metrics = agent_framework.monitor_agent_performance("agent1")
            assert agent_metrics["tasks_completed"] == 15
            assert agent_metrics["success_rate"] == 0.93
    
    @pytest.mark.asyncio
    async def test_scaling_integration(self, architecture_setup):
        """Test scaling integration across platform and lifecycle management."""
        platform = architecture_setup["platform"]
        lifecycle_manager = architecture_setup["lifecycle_manager"]
        
        deployment_id = "deploy_123"
        target_instances = 5
        
        # Test platform scaling
        with patch.object(platform, '_scale_cloud_resources', new_callable=AsyncMock) as mock_scale:
            mock_scale.return_value = {"status": "scaled", "instances": 5}
            
            scaling_result = await platform.scale_deployment(deployment_id, target_instances)
            assert scaling_result["status"] == "scaled"
            assert scaling_result["instances"] == 5
        
        # Test lifecycle manager tracking scaling events
        with patch.object(lifecycle_manager, 'track_deployment_event', return_value=True):
            tracking_result = lifecycle_manager.track_deployment_event(
                deployment_id, "scaling", {"target_instances": target_instances}
            )
            assert tracking_result == True
    
    @pytest.mark.asyncio
    async def test_complete_workflow_integration(self, architecture_setup, sample_model_versions):
        """Test complete workflow integration across all components."""
        platform = architecture_setup["platform"]
        lifecycle_manager = architecture_setup["lifecycle_manager"]
        agent_framework = architecture_setup["agent_framework"]
        rag_system = architecture_setup["rag_system"]
        
        model_version = sample_model_versions[0]
        
        # Complete workflow: Register -> Deploy -> Monitor -> Scale
        with patch.object(lifecycle_manager, 'register_model_version', return_value=True):
            with patch.object(platform, 'deploy_model', new_callable=AsyncMock) as mock_deploy:
                with patch.object(platform, 'monitor_performance', return_value=sample_performance_metrics):
                    with patch.object(platform, 'scale_deployment', new_callable=AsyncMock) as mock_scale:
                        mock_deploy.return_value = {"deployment_id": "deploy_789", "status": "success"}
                        mock_scale.return_value = {"status": "scaled", "instances": 3}
                        
                        # Step 1: Register model
                        registration_result = lifecycle_manager.register_model_version(model_version)
                        assert registration_result == True
                        
                        # Step 2: Deploy model
                        deployment_result = await platform.deploy_model(model_version)
                        assert deployment_result["deployment_id"] == "deploy_789"
                        
                        # Step 3: Monitor performance
                        performance_metrics = platform.monitor_performance("deploy_789")
                        assert performance_metrics["throughput"] == 100
                        
                        # Step 4: Scale deployment
                        scaling_result = await platform.scale_deployment("deploy_789", 3)
                        assert scaling_result["status"] == "scaled"
