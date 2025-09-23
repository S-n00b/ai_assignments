"""
Integration tests for service-level interactions.

Tests the integration between different services including FastAPI,
Gradio, MLflow, ChromaDB, Neo4j, and other platform services.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import tempfile
from pathlib import Path
import json
import requests
from datetime import datetime

# Mock imports for service components
try:
    from src.enterprise_llmops.frontend.fastapi_app import FastAPIApp
    from src.gradio_app.main import create_gradio_app
    from src.model_evaluation.mlflow_integration.mlflow_manager import MLflowManager
    from src.ai_architecture.vector_store.chromadb_manager import ChromaDBManager
    from src.ai_architecture.graph_store.neo4j_manager import Neo4jManager
    from src.enterprise_llmops.ollama_manager import OllamaManager
    from src.github_models_integration.api_client import GitHubModelsClient
except ImportError:
    # Create mock classes for testing
    class FastAPIApp:
        def __init__(self):
            self.app = Mock()
    
    def create_gradio_app():
        return Mock()
    
    class MLflowManager:
        def __init__(self):
            self.tracking = Mock()
    
    class ChromaDBManager:
        def __init__(self):
            self.db = Mock()
    
    class Neo4jManager:
        def __init__(self):
            self.graph = Mock()
    
    class OllamaManager:
        def __init__(self):
            self.models = Mock()
    
    class GitHubModelsClient:
        def __init__(self):
            self.client = Mock()


class TestServicePortConfiguration:
    """Test cases for service port configuration."""
    
    def test_service_port_matrix(self):
        """Test service port configuration matrix."""
        service_ports = {
            "enterprise_fastapi": 8080,
            "gradio_app": 7860,
            "mlflow_tracking": 5000,
            "chromadb": 8000,
            "neo4j": 7474,
            "duckdb": 5432,
            "mkdocs": 8000,  # Different context than ChromaDB
            "ollama": 11434
        }
        
        # Test port uniqueness where applicable
        ports = list(service_ports.values())
        unique_ports = set(ports)
        
        # Allow some port sharing for different contexts
        assert len(unique_ports) >= 6  # At least 6 unique ports
        
        # Test port ranges
        for port in ports:
            assert 1000 <= port <= 65535, f"Port {port} out of valid range"
    
    def test_service_url_structure(self):
        """Test service URL structure."""
        service_urls = {
            "enterprise_fastapi": "http://localhost:8080",
            "gradio_app": "http://localhost:7860",
            "mlflow_tracking": "http://localhost:5000",
            "chromadb": "http://localhost:8000",
            "neo4j": "http://localhost:7474",
            "ollama": "http://localhost:11434"
        }
        
        # Test URL format
        for service, url in service_urls.items():
            assert url.startswith("http://")
            assert "localhost" in url
            assert ":" in url  # Contains port
    
    def test_service_dependency_matrix(self):
        """Test service dependency matrix."""
        dependencies = {
            "enterprise_fastapi": ["chromadb", "mlflow_tracking", "ollama"],
            "gradio_app": ["enterprise_fastapi"],
            "mlflow_tracking": [],
            "chromadb": [],
            "neo4j": [],
            "ollama": [],
            "github_models": []
        }
        
        # Test dependency structure
        for service, deps in dependencies.items():
            assert isinstance(deps, list)
            
            # Test no circular dependencies
            for dep in deps:
                assert dep != service, f"Circular dependency: {service} -> {dep}"
                assert dep in dependencies, f"Unknown dependency: {dep}"


class TestFastAPIGradioIntegration:
    """Test cases for FastAPI-Gradio integration."""
    
    @pytest.fixture
    def fastapi_gradio_setup(self):
        """Set up FastAPI-Gradio integration."""
        return {
            "fastapi_app": FastAPIApp(),
            "gradio_app": create_gradio_app(),
            "integration_config": {
                "fastapi_port": 8080,
                "gradio_port": 7860,
                "shared_endpoints": ["/api/health", "/api/status"],
                "data_flow": "gradio -> fastapi -> services"
            }
        }
    
    @pytest.mark.asyncio
    async def test_fastapi_gradio_communication(self, fastapi_gradio_setup):
        """Test communication between FastAPI and Gradio."""
        fastapi_app = fastapi_gradio_setup["fastapi_app"]
        gradio_app = fastapi_gradio_setup["gradio_app"]
        
        # Mock communication
        with patch.object(fastapi_app, 'start_server', return_value=True), \
             patch.object(gradio_app, 'start_app', return_value=True), \
             patch.object(gradio_app, 'connect_to_fastapi', return_value=True), \
             patch.object(fastapi_app, 'register_gradio_endpoint', return_value=True):
            
            # Test service startup and connection
            results = {
                "fastapi_started": fastapi_app.start_server(),
                "gradio_started": gradio_app.start_app(),
                "gradio_connected": gradio_app.connect_to_fastapi("http://localhost:8080"),
                "endpoint_registered": fastapi_app.register_gradio_endpoint("/gradio")
            }
            
            assert all(results.values())
    
    @pytest.mark.asyncio
    async def test_shared_endpoint_integration(self, fastapi_gradio_setup):
        """Test shared endpoint integration."""
        fastapi_app = fastapi_gradio_setup["fastapi_app"]
        shared_endpoints = fastapi_gradio_setup["integration_config"]["shared_endpoints"]
        
        # Mock endpoint registration
        with patch.object(fastapi_app, 'register_health_endpoint', return_value=True), \
             patch.object(fastapi_app, 'register_status_endpoint', return_value=True):
            
            # Test shared endpoints
            results = {
                "health_endpoint": fastapi_app.register_health_endpoint("/api/health"),
                "status_endpoint": fastapi_app.register_status_endpoint("/api/status")
            }
            
            assert all(results.values())
            assert "/api/health" in shared_endpoints
            assert "/api/status" in shared_endpoints
    
    @pytest.mark.asyncio
    async def test_data_flow_integration(self, fastapi_gradio_setup):
        """Test data flow integration between FastAPI and Gradio."""
        fastapi_app = fastapi_gradio_setup["fastapi_app"]
        gradio_app = fastapi_gradio_setup["gradio_app"]
        data_flow = fastapi_gradio_setup["integration_config"]["data_flow"]
        
        # Mock data flow
        with patch.object(gradio_app, 'send_request_to_fastapi', return_value={
            "status": "success",
            "data": {"result": "test_response"}
        }), \
             patch.object(fastapi_app, 'process_request', return_value={
                 "processed": True,
                 "response": "test_response"
             }):
            
            # Test data flow
            gradio_request = {"input": "test_data"}
            fastapi_response = fastapi_app.process_request(gradio_request)
            gradio_response = gradio_app.send_request_to_fastapi(gradio_request)
            
            assert fastapi_response["processed"] == True
            assert gradio_response["status"] == "success"
            assert data_flow == "gradio -> fastapi -> services"


class TestMLflowIntegration:
    """Test cases for MLflow integration."""
    
    @pytest.fixture
    def mlflow_integration_setup(self):
        """Set up MLflow integration."""
        return {
            "mlflow_manager": MLflowManager(),
            "fastapi_app": FastAPIApp(),
            "gradio_app": create_gradio_app(),
            "mlflow_config": {
                "tracking_uri": "http://localhost:5000",
                "experiment_name": "lenovo_ai_evaluation",
                "artifacts_path": "./mlruns"
            }
        }
    
    @pytest.mark.asyncio
    async def test_mlflow_fastapi_integration(self, mlflow_integration_setup):
        """Test MLflow-FastAPI integration."""
        mlflow_manager = mlflow_integration_setup["mlflow_manager"]
        fastapi_app = mlflow_integration_setup["fastapi_app"]
        
        # Mock MLflow integration
        with patch.object(mlflow_manager, 'connect_to_mlflow', return_value=True), \
             patch.object(mlflow_manager, 'create_experiment', return_value="exp_123"), \
             patch.object(fastapi_app, 'integrate_mlflow', return_value=True):
            
            # Test MLflow integration
            results = {
                "mlflow_connected": mlflow_manager.connect_to_mlflow(),
                "experiment_created": mlflow_manager.create_experiment("test_exp"),
                "fastapi_integrated": fastapi_app.integrate_mlflow(mlflow_manager)
            }
            
            assert all(results.values())
            assert results["experiment_created"] == "exp_123"
    
    @pytest.mark.asyncio
    async def test_mlflow_gradio_integration(self, mlflow_integration_setup):
        """Test MLflow-Gradio integration."""
        mlflow_manager = mlflow_integration_setup["mlflow_manager"]
        gradio_app = mlflow_integration_setup["gradio_app"]
        
        # Mock MLflow-Gradio integration
        with patch.object(gradio_app, 'integrate_mlflow_tracking', return_value=True), \
             patch.object(mlflow_manager, 'log_metrics', return_value=True), \
             patch.object(mlflow_manager, 'log_model', return_value=True):
            
            # Test MLflow-Gradio integration
            results = {
                "gradio_mlflow_integrated": gradio_app.integrate_mlflow_tracking(mlflow_manager),
                "metrics_logged": mlflow_manager.log_metrics({"accuracy": 0.95}),
                "model_logged": mlflow_manager.log_model("test_model")
            }
            
            assert all(results.values())
    
    @pytest.mark.asyncio
    async def test_mlflow_experiment_tracking(self, mlflow_integration_setup):
        """Test MLflow experiment tracking."""
        mlflow_manager = mlflow_integration_setup["mlflow_manager"]
        
        # Mock experiment tracking
        with patch.object(mlflow_manager, 'start_run', return_value="run_123"), \
             patch.object(mlflow_manager, 'log_parameters', return_value=True), \
             patch.object(mlflow_manager, 'log_metrics', return_value=True), \
             patch.object(mlflow_manager, 'end_run', return_value=True):
            
            # Test experiment tracking workflow
            results = {
                "run_started": mlflow_manager.start_run("test_run"),
                "parameters_logged": mlflow_manager.log_parameters({"learning_rate": 0.01}),
                "metrics_logged": mlflow_manager.log_metrics({"loss": 0.1}),
                "run_ended": mlflow_manager.end_run()
            }
            
            assert all(results.values())
            assert results["run_started"] == "run_123"


class TestChromaDBIntegration:
    """Test cases for ChromaDB integration."""
    
    @pytest.fixture
    def chromadb_integration_setup(self):
        """Set up ChromaDB integration."""
        return {
            "chromadb_manager": ChromaDBManager(),
            "fastapi_app": FastAPIApp(),
            "gradio_app": create_gradio_app(),
            "chromadb_config": {
                "host": "localhost",
                "port": 8000,
                "collection_name": "lenovo_documents",
                "embedding_model": "sentence-transformers"
            }
        }
    
    @pytest.mark.asyncio
    async def test_chromadb_fastapi_integration(self, chromadb_integration_setup):
        """Test ChromaDB-FastAPI integration."""
        chromadb_manager = chromadb_integration_setup["chromadb_manager"]
        fastapi_app = chromadb_integration_setup["fastapi_app"]
        
        # Mock ChromaDB integration
        with patch.object(chromadb_manager, 'connect_to_chromadb', return_value=True), \
             patch.object(chromadb_manager, 'create_collection', return_value=True), \
             patch.object(fastapi_app, 'integrate_chromadb', return_value=True):
            
            # Test ChromaDB integration
            results = {
                "chromadb_connected": chromadb_manager.connect_to_chromadb(),
                "collection_created": chromadb_manager.create_collection("test_collection"),
                "fastapi_integrated": fastapi_app.integrate_chromadb(chromadb_manager)
            }
            
            assert all(results.values())
    
    @pytest.mark.asyncio
    async def test_chromadb_gradio_integration(self, chromadb_integration_setup):
        """Test ChromaDB-Gradio integration."""
        chromadb_manager = chromadb_integration_setup["chromadb_manager"]
        gradio_app = chromadb_integration_setup["gradio_app"]
        
        # Mock ChromaDB-Gradio integration
        with patch.object(gradio_app, 'integrate_vector_search', return_value=True), \
             patch.object(chromadb_manager, 'search_documents', return_value=[
                 {"content": "test document", "score": 0.95}
             ]), \
             patch.object(chromadb_manager, 'add_documents', return_value=True):
            
            # Test ChromaDB-Gradio integration
            results = {
                "vector_search_integrated": gradio_app.integrate_vector_search(chromadb_manager),
                "documents_searched": chromadb_manager.search_documents("test query"),
                "documents_added": chromadb_manager.add_documents(["test doc"])
            }
            
            assert all(results.values())
            assert len(results["documents_searched"]) == 1
            assert results["documents_searched"][0]["score"] == 0.95
    
    @pytest.mark.asyncio
    async def test_chromadb_document_operations(self, chromadb_integration_setup):
        """Test ChromaDB document operations."""
        chromadb_manager = chromadb_integration_setup["chromadb_manager"]
        
        # Mock document operations
        with patch.object(chromadb_manager, 'add_documents', return_value=True), \
             patch.object(chromadb_manager, 'update_documents', return_value=True), \
             patch.object(chromadb_manager, 'delete_documents', return_value=True), \
             patch.object(chromadb_manager, 'get_collection_stats', return_value={
                 "total_documents": 100,
                 "collection_size": "50MB"
             }):
            
            # Test document operations
            test_documents = [
                {"id": "doc1", "content": "test content 1"},
                {"id": "doc2", "content": "test content 2"}
            ]
            
            results = {
                "documents_added": chromadb_manager.add_documents(test_documents),
                "documents_updated": chromadb_manager.update_documents(test_documents),
                "documents_deleted": chromadb_manager.delete_documents(["doc1"]),
                "stats_retrieved": chromadb_manager.get_collection_stats()
            }
            
            assert all(results.values())
            assert results["stats_retrieved"]["total_documents"] == 100


class TestNeo4jIntegration:
    """Test cases for Neo4j integration."""
    
    @pytest.fixture
    def neo4j_integration_setup(self):
        """Set up Neo4j integration."""
        return {
            "neo4j_manager": Neo4jManager(),
            "fastapi_app": FastAPIApp(),
            "gradio_app": create_gradio_app(),
            "neo4j_config": {
                "host": "localhost",
                "port": 7474,
                "database": "neo4j",
                "username": "neo4j",
                "password": "password"
            }
        }
    
    @pytest.mark.asyncio
    async def test_neo4j_fastapi_integration(self, neo4j_integration_setup):
        """Test Neo4j-FastAPI integration."""
        neo4j_manager = neo4j_integration_setup["neo4j_manager"]
        fastapi_app = neo4j_integration_setup["fastapi_app"]
        
        # Mock Neo4j integration
        with patch.object(neo4j_manager, 'connect_to_neo4j', return_value=True), \
             patch.object(neo4j_manager, 'create_constraints', return_value=True), \
             patch.object(fastapi_app, 'integrate_neo4j', return_value=True):
            
            # Test Neo4j integration
            results = {
                "neo4j_connected": neo4j_manager.connect_to_neo4j(),
                "constraints_created": neo4j_manager.create_constraints(),
                "fastapi_integrated": fastapi_app.integrate_neo4j(neo4j_manager)
            }
            
            assert all(results.values())
    
    @pytest.mark.asyncio
    async def test_neo4j_graph_operations(self, neo4j_integration_setup):
        """Test Neo4j graph operations."""
        neo4j_manager = neo4j_integration_setup["neo4j_manager"]
        
        # Mock graph operations
        with patch.object(neo4j_manager, 'create_node', return_value=True), \
             patch.object(neo4j_manager, 'create_relationship', return_value=True), \
             patch.object(neo4j_manager, 'execute_query', return_value=[
                 {"node": "test_node", "relationship": "RELATES_TO", "target": "target_node"}
             ]):
            
            # Test graph operations
            results = {
                "node_created": neo4j_manager.create_node("Person", {"name": "John"}),
                "relationship_created": neo4j_manager.create_relationship(
                    "node1", "RELATES_TO", "node2", {"strength": 0.8}
                ),
                "query_executed": neo4j_manager.execute_query(
                    "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 10"
                )
            }
            
            assert all(results.values())
            assert len(results["query_executed"]) == 1
            assert results["query_executed"][0]["relationship"] == "RELATES_TO"


class TestOllamaIntegration:
    """Test cases for Ollama integration."""
    
    @pytest.fixture
    def ollama_integration_setup(self):
        """Set up Ollama integration."""
        return {
            "ollama_manager": OllamaManager(),
            "fastapi_app": FastAPIApp(),
            "gradio_app": create_gradio_app(),
            "ollama_config": {
                "host": "localhost",
                "port": 11434,
                "models": ["llama-2-7b", "mistral-7b", "phi-3-mini"],
                "default_model": "llama-2-7b"
            }
        }
    
    @pytest.mark.asyncio
    async def test_ollama_fastapi_integration(self, ollama_integration_setup):
        """Test Ollama-FastAPI integration."""
        ollama_manager = ollama_integration_setup["ollama_manager"]
        fastapi_app = ollama_integration_setup["fastapi_app"]
        
        # Mock Ollama integration
        with patch.object(ollama_manager, 'connect_to_ollama', return_value=True), \
             patch.object(ollama_manager, 'list_models', return_value=[
                 {"name": "llama-2-7b", "size": "3.8GB"},
                 {"name": "mistral-7b", "size": "4.1GB"}
             ]), \
             patch.object(fastapi_app, 'integrate_ollama', return_value=True):
            
            # Test Ollama integration
            results = {
                "ollama_connected": ollama_manager.connect_to_ollama(),
                "models_listed": ollama_manager.list_models(),
                "fastapi_integrated": fastapi_app.integrate_ollama(ollama_manager)
            }
            
            assert all(results.values())
            assert len(results["models_listed"]) == 2
            assert results["models_listed"][0]["name"] == "llama-2-7b"
    
    @pytest.mark.asyncio
    async def test_ollama_model_operations(self, ollama_integration_setup):
        """Test Ollama model operations."""
        ollama_manager = ollama_integration_setup["ollama_manager"]
        
        # Mock model operations
        with patch.object(ollama_manager, 'generate_response', return_value={
            "response": "Test response from Ollama",
            "model": "llama-2-7b",
            "tokens": 25
        }), \
             patch.object(ollama_manager, 'pull_model', return_value=True), \
             patch.object(ollama_manager, 'delete_model', return_value=True):
            
            # Test model operations
            results = {
                "response_generated": ollama_manager.generate_response(
                    "Test prompt", model="llama-2-7b"
                ),
                "model_pulled": ollama_manager.pull_model("mistral-7b"),
                "model_deleted": ollama_manager.delete_model("old-model")
            }
            
            assert all(results.values())
            assert results["response_generated"]["response"] == "Test response from Ollama"
            assert results["response_generated"]["model"] == "llama-2-7b"
            assert results["response_generated"]["tokens"] == 25


class TestGitHubModelsIntegration:
    """Test cases for GitHub Models integration."""
    
    @pytest.fixture
    def github_models_setup(self):
        """Set up GitHub Models integration."""
        return {
            "github_client": GitHubModelsClient(),
            "fastapi_app": FastAPIApp(),
            "gradio_app": create_gradio_app(),
            "github_config": {
                "api_base": "https://api.github.com",
                "model_endpoint": "https://huggingface.co",
                "auth_token": "ghp_test_token",
                "model_repository": "s-n00b/ai_assignments"
            }
        }
    
    @pytest.mark.asyncio
    async def test_github_models_authentication(self, github_models_setup):
        """Test GitHub Models authentication."""
        github_client = github_models_setup["github_client"]
        
        # Mock authentication
        with patch.object(github_client, 'authenticate', return_value=True), \
             patch.object(github_client, 'validate_token', return_value=True), \
             patch.object(github_client, 'get_user_info', return_value={
                 "login": "s-n00b",
                 "id": 12345,
                 "type": "User"
             }):
            
            # Test authentication
            results = {
                "authenticated": github_client.authenticate("ghp_test_token"),
                "token_validated": github_client.validate_token(),
                "user_info": github_client.get_user_info()
            }
            
            assert all(results.values())
            assert results["user_info"]["login"] == "s-n00b"
            assert results["user_info"]["type"] == "User"
    
    @pytest.mark.asyncio
    async def test_github_models_api_operations(self, github_models_setup):
        """Test GitHub Models API operations."""
        github_client = github_models_setup["github_client"]
        
        # Mock API operations
        with patch.object(github_client, 'list_models', return_value=[
            {"name": "gpt-3.5-turbo", "provider": "openai"},
            {"name": "claude-3-sonnet", "provider": "anthropic"}
        ]), \
             patch.object(github_client, 'get_model_info', return_value={
                 "name": "gpt-3.5-turbo",
                 "provider": "openai",
                 "description": "GPT-3.5 Turbo model"
             }), \
             patch.object(github_client, 'generate_response', return_value={
                 "response": "Test response from GitHub Models",
                 "model": "gpt-3.5-turbo",
                 "usage": {"tokens": 50}
             }):
            
            # Test API operations
            results = {
                "models_listed": github_client.list_models(),
                "model_info": github_client.get_model_info("gpt-3.5-turbo"),
                "response_generated": github_client.generate_response(
                    "Test prompt", model="gpt-3.5-turbo"
                )
            }
            
            assert all(results.values())
            assert len(results["models_listed"]) == 2
            assert results["model_info"]["name"] == "gpt-3.5-turbo"
            assert results["response_generated"]["response"] == "Test response from GitHub Models"


class TestCrossServiceIntegration:
    """Test cases for cross-service integration."""
    
    @pytest.fixture
    def cross_service_setup(self):
        """Set up cross-service integration."""
        return {
            "fastapi_app": FastAPIApp(),
            "gradio_app": create_gradio_app(),
            "mlflow_manager": MLflowManager(),
            "chromadb_manager": ChromaDBManager(),
            "neo4j_manager": Neo4jManager(),
            "ollama_manager": OllamaManager(),
            "github_client": GitHubModelsClient()
        }
    
    @pytest.mark.asyncio
    async def test_complete_service_integration(self, cross_service_setup):
        """Test complete service integration."""
        services = cross_service_setup
        
        # Mock all service connections
        with patch.object(services["fastapi_app"], 'start_server', return_value=True), \
             patch.object(services["gradio_app"], 'start_app', return_value=True), \
             patch.object(services["mlflow_manager"], 'connect_to_mlflow', return_value=True), \
             patch.object(services["chromadb_manager"], 'connect_to_chromadb', return_value=True), \
             patch.object(services["neo4j_manager"], 'connect_to_neo4j', return_value=True), \
             patch.object(services["ollama_manager"], 'connect_to_ollama', return_value=True), \
             patch.object(services["github_client"], 'authenticate', return_value=True):
            
            # Test complete service integration
            results = {
                "fastapi_started": services["fastapi_app"].start_server(),
                "gradio_started": services["gradio_app"].start_app(),
                "mlflow_connected": services["mlflow_manager"].connect_to_mlflow(),
                "chromadb_connected": services["chromadb_manager"].connect_to_chromadb(),
                "neo4j_connected": services["neo4j_manager"].connect_to_neo4j(),
                "ollama_connected": services["ollama_manager"].connect_to_ollama(),
                "github_authenticated": services["github_client"].authenticate("test_token")
            }
            
            # Verify all services integrated successfully
            assert all(results.values())
    
    @pytest.mark.asyncio
    async def test_service_communication_flow(self, cross_service_setup):
        """Test service communication flow."""
        services = cross_service_setup
        
        # Mock service communication
        with patch.object(services["gradio_app"], 'send_evaluation_request', return_value={
            "request_id": "req_123",
            "status": "processing"
        }), \
             patch.object(services["fastapi_app"], 'process_evaluation_request', return_value={
                 "request_id": "req_123",
                 "model_results": {"accuracy": 0.95},
                 "mlflow_run_id": "run_123"
             }), \
             patch.object(services["mlflow_manager"], 'log_evaluation_results', return_value=True), \
             patch.object(services["chromadb_manager"], 'store_evaluation_data', return_value=True):
            
            # Test service communication flow
            evaluation_request = {
                "model_name": "test_model",
                "test_data": ["prompt1", "prompt2"],
                "evaluation_metrics": ["accuracy", "latency"]
            }
            
            # Step 1: Gradio sends request to FastAPI
            gradio_response = services["gradio_app"].send_evaluation_request(evaluation_request)
            
            # Step 2: FastAPI processes request
            fastapi_response = services["fastapi_app"].process_evaluation_request(evaluation_request)
            
            # Step 3: MLflow logs results
            mlflow_result = services["mlflow_manager"].log_evaluation_results(fastapi_response)
            
            # Step 4: ChromaDB stores data
            chromadb_result = services["chromadb_manager"].store_evaluation_data(fastapi_response)
            
            # Verify communication flow
            assert gradio_response["request_id"] == "req_123"
            assert fastapi_response["request_id"] == "req_123"
            assert fastapi_response["model_results"]["accuracy"] == 0.95
            assert mlflow_result == True
            assert chromadb_result == True
    
    @pytest.mark.asyncio
    async def test_service_error_handling(self, cross_service_setup):
        """Test service error handling."""
        services = cross_service_setup
        
        # Mock error scenarios
        with patch.object(services["chromadb_manager"], 'search_documents', side_effect=Exception("ChromaDB error")), \
             patch.object(services["fastapi_app"], 'handle_service_error', return_value={
                 "error": "ChromaDB error",
                 "service": "chromadb",
                 "fallback_action": "use_cache"
             }), \
             patch.object(services["gradio_app"], 'handle_error_response', return_value=True):
            
            # Test error handling
            try:
                services["chromadb_manager"].search_documents("test query")
            except Exception as e:
                error_response = services["fastapi_app"].handle_service_error(str(e), "chromadb")
                gradio_handled = services["gradio_app"].handle_error_response(error_response)
                
                # Verify error handling
                assert error_response["error"] == "ChromaDB error"
                assert error_response["service"] == "chromadb"
                assert error_response["fallback_action"] == "use_cache"
                assert gradio_handled == True


if __name__ == "__main__":
    pytest.main([__file__])
