"""
Unit tests for AI architecture components.

Tests the core functionality of AI architecture modules including
platform, lifecycle management, agents, and RAG systems.
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
    MessageType,
    DocumentChunk,
    DocumentMetadata
)


class TestHybridAIPlatform:
    """Test cases for HybridAIPlatform class."""
    
    @pytest.fixture
    def platform(self, mock_platform_config):
        """Create hybrid AI platform instance."""
        return HybridAIPlatform(config=mock_platform_config)
    
    def test_platform_initialization(self, platform):
        """Test platform initialization."""
        assert platform.config is not None
        assert platform.deployment_target == "hybrid"
        assert platform.infrastructure["cloud_provider"] == "aws"
    
    def test_validate_configuration(self, platform):
        """Test configuration validation."""
        valid_config = {
            "deployment_target": "hybrid",
            "infrastructure": {"cloud_provider": "aws", "region": "us-east-1"}
        }
        invalid_config = {"deployment_target": "invalid"}
        
        assert platform._validate_configuration(valid_config) == True
        assert platform._validate_configuration(invalid_config) == False
    
    def test_calculate_resource_requirements(self, platform):
        """Test resource requirements calculation."""
        workload = {
            "expected_requests_per_second": 100,
            "average_response_time_ms": 500,
            "model_size_gb": 2.5
        }
        
        requirements = platform._calculate_resource_requirements(workload)
        
        assert "cpu_cores" in requirements
        assert "memory_gb" in requirements
        assert "gpu_count" in requirements
        assert "storage_gb" in requirements
        assert all(value > 0 for value in requirements.values())
    
    @pytest.mark.asyncio
    async def test_deploy_model(self, platform, sample_model_config):
        """Test model deployment."""
        with patch.object(platform, '_deploy_to_cloud', new_callable=AsyncMock) as mock_deploy:
            mock_deploy.return_value = {"deployment_id": "deploy_123", "status": "success"}
            
            result = await platform.deploy_model(sample_model_config)
            
            assert result["deployment_id"] == "deploy_123"
            assert result["status"] == "success"
            mock_deploy.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_scale_deployment(self, platform):
        """Test deployment scaling."""
        deployment_id = "deploy_123"
        target_instances = 5
        
        with patch.object(platform, '_scale_cloud_resources', new_callable=AsyncMock) as mock_scale:
            mock_scale.return_value = {"status": "scaled", "instances": 5}
            
            result = await platform.scale_deployment(deployment_id, target_instances)
            
            assert result["status"] == "scaled"
            assert result["instances"] == 5
            mock_scale.assert_called_once_with(deployment_id, target_instances)
    
    def test_monitor_performance(self, platform, sample_performance_metrics):
        """Test performance monitoring."""
        with patch.object(platform, '_collect_metrics', return_value=sample_performance_metrics):
            metrics = platform.monitor_performance("deploy_123")
            
            assert metrics["throughput"] == 100
            assert metrics["latency_p50"] == 500
            assert metrics["error_rate"] == 0.01


class TestModelLifecycleManager:
    """Test cases for ModelLifecycleManager class."""
    
    @pytest.fixture
    def lifecycle_manager(self):
        """Create model lifecycle manager instance."""
        return ModelLifecycleManager()
    
    def test_lifecycle_manager_initialization(self, lifecycle_manager):
        """Test lifecycle manager initialization."""
        assert lifecycle_manager.model_registry is not None
        assert lifecycle_manager.deployment_tracker is not None
    
    def test_register_model_version(self, lifecycle_manager, sample_model_versions):
        """Test model version registration."""
        model_version = sample_model_versions[0]
        
        with patch.object(lifecycle_manager.model_registry, 'register', return_value=True):
            result = lifecycle_manager.register_model_version(model_version)
            
            assert result == True
            lifecycle_manager.model_registry.register.assert_called_once_with(model_version)
    
    def test_validate_model_quality(self, lifecycle_manager):
        """Test model quality validation."""
        model_metrics = {
            "accuracy": 0.95,
            "f1_score": 0.92,
            "precision": 0.94,
            "recall": 0.90
        }
        
        quality_score = lifecycle_manager._validate_model_quality(model_metrics)
        
        assert 0 <= quality_score <= 1
        assert quality_score > 0.9  # High quality model
    
    def test_determine_deployment_strategy(self, lifecycle_manager):
        """Test deployment strategy determination."""
        model_metrics = {"accuracy": 0.95, "latency_ms": 100}
        business_requirements = {"min_accuracy": 0.90, "max_latency_ms": 200}
        
        strategy = lifecycle_manager._determine_deployment_strategy(
            model_metrics, business_requirements
        )
        
        assert strategy in ["canary", "blue_green", "rolling", "immediate"]
    
    @pytest.mark.asyncio
    async def test_deploy_model_version(self, lifecycle_manager, sample_model_versions):
        """Test model version deployment."""
        model_version = sample_model_versions[0]
        deployment_strategy = "canary"
        
        with patch.object(lifecycle_manager, '_execute_deployment', new_callable=AsyncMock) as mock_deploy:
            mock_deploy.return_value = {"deployment_id": "deploy_456", "status": "success"}
            
            result = await lifecycle_manager.deploy_model_version(
                model_version, deployment_strategy
            )
            
            assert result["deployment_id"] == "deploy_456"
            assert result["status"] == "success"
            mock_deploy.assert_called_once()
    
    def test_rollback_deployment(self, lifecycle_manager):
        """Test deployment rollback."""
        deployment_id = "deploy_456"
        target_version = "v1.0.0"
        
        with patch.object(lifecycle_manager, '_execute_rollback', return_value=True):
            result = lifecycle_manager.rollback_deployment(deployment_id, target_version)
            
            assert result == True


class TestAgenticComputingFramework:
    """Test cases for AgenticComputingFramework class."""
    
    @pytest.fixture
    def agent_framework(self):
        """Create agentic computing framework instance."""
        return AgenticComputingFramework()
    
    def test_framework_initialization(self, agent_framework):
        """Test framework initialization."""
        assert agent_framework.agents == {}
        assert agent_framework.message_queue is not None
        assert agent_framework.coordinator is not None
    
    def test_register_agent(self, agent_framework, sample_agents):
        """Test agent registration."""
        agent_data = sample_agents[0]
        
        with patch.object(agent_framework, '_create_agent', return_value=Mock()) as mock_create:
            result = agent_framework.register_agent(agent_data)
            
            assert result == True
            mock_create.assert_called_once_with(agent_data)
    
    def test_create_agent_message(self, agent_framework):
        """Test agent message creation."""
        message = agent_framework._create_agent_message(
            sender_id="agent1",
            receiver_id="agent2",
            content="Hello from agent1",
            message_type=MessageType.COLLABORATION_REQUEST
        )
        
        assert isinstance(message, AgentMessage)
        assert message.sender_id == "agent1"
        assert message.receiver_id == "agent2"
        assert message.content == "Hello from agent1"
        assert message.message_type == MessageType.COLLABORATION_REQUEST
    
    @pytest.mark.asyncio
    async def test_send_message(self, agent_framework):
        """Test sending messages between agents."""
        message = AgentMessage(
            sender_id="agent1",
            receiver_id="agent2",
            content="Test message",
            message_type=MessageType.COLLABORATION_REQUEST
        )
        
        with patch.object(agent_framework.message_queue, 'put', new_callable=AsyncMock):
            result = await agent_framework.send_message(message)
            
            assert result == True
    
    @pytest.mark.asyncio
    async def test_coordinate_agents(self, agent_framework, sample_agents):
        """Test agent coordination."""
        task = {
            "type": "research_and_write",
            "description": "Research a topic and write a summary",
            "required_agents": ["research", "writing"]
        }
        
        with patch.object(agent_framework, '_execute_coordination', new_callable=AsyncMock) as mock_coord:
            mock_coord.return_value = {"status": "completed", "result": "Task completed successfully"}
            
            result = await agent_framework.coordinate_agents(task)
            
            assert result["status"] == "completed"
            assert result["result"] == "Task completed successfully"
            mock_coord.assert_called_once()
    
    def test_monitor_agent_performance(self, agent_framework):
        """Test agent performance monitoring."""
        agent_id = "agent1"
        
        with patch.object(agent_framework, '_collect_agent_metrics', return_value={
            "tasks_completed": 10,
            "success_rate": 0.95,
            "average_response_time": 1.2
        }):
            metrics = agent_framework.monitor_agent_performance(agent_id)
            
            assert metrics["tasks_completed"] == 10
            assert metrics["success_rate"] == 0.95
            assert metrics["average_response_time"] == 1.2


class TestRAGSystem:
    """Test cases for RAGSystem class."""
    
    @pytest.fixture
    def rag_system(self, mock_vector_store):
        """Create RAG system instance."""
        return RAGSystem(vector_store=mock_vector_store)
    
    def test_rag_system_initialization(self, rag_system):
        """Test RAG system initialization."""
        assert rag_system.vector_store is not None
        assert rag_system.embedding_model is not None
        assert rag_system.chunking_strategy is not None
    
    def test_chunk_document(self, rag_system, sample_documents):
        """Test document chunking."""
        document = sample_documents[0]
        
        chunks = rag_system._chunk_document(document)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(chunk.document_id == document["id"] for chunk in chunks)
    
    def test_create_document_metadata(self, rag_system, sample_documents):
        """Test document metadata creation."""
        document = sample_documents[0]
        
        metadata = rag_system._create_document_metadata(document)
        
        assert isinstance(metadata, DocumentMetadata)
        assert metadata.document_id == document["id"]
        assert metadata.source == document["metadata"]["source"]
        assert metadata.created_at is not None
    
    @pytest.mark.asyncio
    async def test_add_documents(self, rag_system, sample_documents):
        """Test adding documents to the RAG system."""
        with patch.object(rag_system.vector_store, 'add_documents', new_callable=AsyncMock):
            result = await rag_system.add_documents(sample_documents)
            
            assert result == True
            rag_system.vector_store.add_documents.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_retrieve_relevant_documents(self, rag_system):
        """Test retrieving relevant documents."""
        query = "What is artificial intelligence?"
        top_k = 3
        
        with patch.object(rag_system.vector_store, 'similarity_search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [
                {"content": "AI is the simulation of human intelligence", "score": 0.95},
                {"content": "Machine learning is a subset of AI", "score": 0.87}
            ]
            
            results = await rag_system.retrieve_relevant_documents(query, top_k)
            
            assert isinstance(results, list)
            assert len(results) <= top_k
            assert all("content" in result for result in results)
            mock_search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_response(self, rag_system, mock_api_client):
        """Test response generation with retrieved context."""
        query = "What is machine learning?"
        context_documents = [
            {"content": "Machine learning is a subset of AI", "score": 0.95},
            {"content": "ML algorithms learn from data", "score": 0.87}
        ]
        
        with patch.object(rag_system, '_get_api_client', return_value=mock_api_client):
            response = await rag_system.generate_response(query, context_documents)
            
            assert isinstance(response, str)
            assert len(response) > 0
    
    def test_calculate_relevance_score(self, rag_system):
        """Test relevance score calculation."""
        query = "artificial intelligence"
        document_content = "AI is the simulation of human intelligence in machines"
        
        score = rag_system._calculate_relevance_score(query, document_content)
        
        assert 0 <= score <= 1
        assert score > 0.5  # Should be relevant
    
    @pytest.mark.asyncio
    async def test_delete_documents(self, rag_system):
        """Test document deletion."""
        document_ids = ["doc1", "doc2"]
        
        with patch.object(rag_system.vector_store, 'delete_documents', new_callable=AsyncMock):
            result = await rag_system.delete_documents(document_ids)
            
            assert result == True
            rag_system.vector_store.delete_documents.assert_called_once_with(document_ids)


class TestBaseAgent:
    """Test cases for BaseAgent class."""
    
    @pytest.fixture
    def base_agent(self):
        """Create base agent instance."""
        return BaseAgent(
            agent_id="test_agent",
            name="Test Agent",
            role="tester",
            capabilities=["testing", "validation"]
        )
    
    def test_agent_initialization(self, base_agent):
        """Test agent initialization."""
        assert base_agent.agent_id == "test_agent"
        assert base_agent.name == "Test Agent"
        assert base_agent.role == "tester"
        assert "testing" in base_agent.capabilities
        assert base_agent.status == "idle"
    
    def test_agent_capability_check(self, base_agent):
        """Test agent capability checking."""
        assert base_agent.has_capability("testing") == True
        assert base_agent.has_capability("invalid_capability") == False
    
    @pytest.mark.asyncio
    async def test_process_message(self, base_agent):
        """Test message processing."""
        message = AgentMessage(
            sender_id="sender",
            receiver_id="test_agent",
            content="Test message",
            message_type=MessageType.COLLABORATION_REQUEST
        )
        
        with patch.object(base_agent, '_handle_message', new_callable=AsyncMock) as mock_handle:
            mock_handle.return_value = "Response message"
            
            response = await base_agent.process_message(message)
            
            assert response == "Response message"
            mock_handle.assert_called_once_with(message)
    
    def test_update_status(self, base_agent):
        """Test agent status updates."""
        base_agent.update_status("busy")
        assert base_agent.status == "busy"
        
        base_agent.update_status("idle")
        assert base_agent.status == "idle"
    
    def test_get_agent_info(self, base_agent):
        """Test getting agent information."""
        info = base_agent.get_agent_info()
        
        assert info["agent_id"] == "test_agent"
        assert info["name"] == "Test Agent"
        assert info["role"] == "tester"
        assert info["status"] == "idle"
        assert "capabilities" in info
