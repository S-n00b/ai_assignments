"""
Phase 3 Implementation Test Script

This script tests all Phase 3 components including fine-tuning pipeline,
custom embeddings, hybrid RAG, and LangChain/LlamaIndex integration.
"""

import sys
import os
import logging
from pathlib import Path
import json
import numpy as np
from typing import Dict, List, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Import Phase 3 components
from mobile_fine_tuning import (
    LenovoDomainAdapter, MobileOptimizer, QLoRAMobileAdapter, 
    EdgeDeploymentConfig, MLflowFineTuningTracker
)
from custom_embeddings import (
    LenovoTechnicalEmbeddings, DeviceSupportEmbeddings, 
    CustomerServiceEmbeddings, BusinessProcessEmbeddings, ChromaDBVectorStore
)
from hybrid_rag import (
    MultiSourceRetrieval, LenovoKnowledgeGraph, DeviceContextRetrieval,
    CustomerJourneyRAG, UnifiedRetrievalOrchestrator
)
from retrieval_workflows import (
    LangChainFAISSIntegration, LlamaIndexRetrieval, HybridRetrievalSystem,
    RetrievalEvaluation, MLflowRetrievalTracking
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phase3Tester:
    """Comprehensive tester for Phase 3 implementation."""
    
    def __init__(self):
        self.test_results = {}
        self.test_data = self._create_test_data()
    
    def _create_test_data(self) -> Dict[str, Any]:
        """Create test data for Phase 3 components."""
        return {
            'lenovo_devices': [
                {
                    'id': 'device_001',
                    'name': 'ThinkPad X1 Carbon',
                    'type': 'laptop',
                    'category': 'business',
                    'specifications': {
                        'cpu': 'Intel Core i7',
                        'ram': '16GB',
                        'storage': '512GB SSD'
                    },
                    'support_info': 'Premium business laptop with advanced security features',
                    'common_issues': [
                        {'issue': 'Battery not charging', 'solution': 'Check power adapter connection'},
                        {'issue': 'Screen flickering', 'solution': 'Update graphics drivers'}
                    ]
                },
                {
                    'id': 'device_002',
                    'name': 'Moto Edge 40',
                    'type': 'smartphone',
                    'category': 'mobile',
                    'specifications': {
                        'cpu': 'MediaTek Dimensity 8020',
                        'ram': '8GB',
                        'storage': '256GB'
                    },
                    'support_info': '5G smartphone with advanced camera system',
                    'common_issues': [
                        {'issue': 'Camera not working', 'solution': 'Restart camera app'},
                        {'issue': 'Battery draining fast', 'solution': 'Check background apps'}
                    ]
                }
            ],
            'technical_docs': [
                {
                    'id': 'doc_001',
                    'title': 'ThinkPad X1 Carbon Setup Guide',
                    'content': 'Complete setup guide for ThinkPad X1 Carbon including initial configuration and security setup.',
                    'category': 'setup_guide',
                    'type': 'technical'
                },
                {
                    'id': 'doc_002',
                    'title': 'Moto Edge 40 Camera Features',
                    'content': 'Detailed guide to Moto Edge 40 camera features including night mode and portrait photography.',
                    'category': 'feature_guide',
                    'type': 'technical'
                }
            ],
            'customer_data': [
                {
                    'id': 'customer_001',
                    'name': 'Acme Corporation',
                    'type': 'enterprise',
                    'industry': 'technology',
                    'contact_info': {
                        'email': 'support@acme.com',
                        'phone': '+1-555-0123'
                    },
                    'interaction_history': [
                        {
                            'date': '2024-01-15',
                            'type': 'support',
                            'description': 'ThinkPad X1 Carbon setup assistance',
                            'satisfaction': 8
                        }
                    ]
                }
            ],
            'business_processes': [
                {
                    'id': 'process_001',
                    'name': 'Device Onboarding',
                    'type': 'operational',
                    'description': 'Process for onboarding new devices to enterprise environment',
                    'steps': [
                        'Device registration',
                        'Security configuration',
                        'User training',
                        'Deployment'
                    ]
                }
            ]
        }
    
    def test_mobile_fine_tuning(self) -> Dict[str, Any]:
        """Test mobile fine-tuning components."""
        logger.info("Testing mobile fine-tuning components...")
        
        results = {
            'lenovo_domain_adapter': False,
            'mobile_optimizer': False,
            'qlora_adapter': False,
            'edge_deployment': False,
            'mlflow_tracking': False
        }
        
        try:
            # Test Lenovo Domain Adapter
            adapter = LenovoDomainAdapter("microsoft/phi-4-mini-instruct")
            adapter.load_model()
            results['lenovo_domain_adapter'] = True
            
            # Test Mobile Optimizer
            optimizer = MobileOptimizer("phi-4-mini", "mobile")
            results['mobile_optimizer'] = True
            
            # Test QLoRA Adapter
            qlora_adapter = QLoRAMobileAdapter(768, 16, 8, 16.0)
            results['qlora_adapter'] = True
            
            # Test Edge Deployment Config
            edge_config = EdgeDeploymentConfig("phi-4-mini")
            android_config = edge_config.get_platform_config("android")
            results['edge_deployment'] = True
            
            # Test MLflow Tracking
            mlflow_tracker = MLflowFineTuningTracker("mobile_fine_tuning_test")
            results['mlflow_tracking'] = True
            
        except Exception as e:
            logger.error(f"Mobile fine-tuning test failed: {e}")
        
        return results
    
    def test_custom_embeddings(self) -> Dict[str, Any]:
        """Test custom embeddings components."""
        logger.info("Testing custom embeddings components...")
        
        results = {
            'technical_embeddings': False,
            'device_support_embeddings': False,
            'customer_service_embeddings': False,
            'business_process_embeddings': False,
            'chromadb_integration': False
        }
        
        try:
            # Test Technical Embeddings
            tech_embeddings = LenovoTechnicalEmbeddings()
            tech_embeddings.load_base_model()
            results['technical_embeddings'] = True
            
            # Test Device Support Embeddings
            device_embeddings = DeviceSupportEmbeddings()
            device_embeddings.load_base_model()
            results['device_support_embeddings'] = True
            
            # Test Customer Service Embeddings
            customer_embeddings = CustomerServiceEmbeddings()
            customer_embeddings.load_base_model()
            results['customer_service_embeddings'] = True
            
            # Test Business Process Embeddings
            business_embeddings = BusinessProcessEmbeddings()
            business_embeddings.load_base_model()
            results['business_process_embeddings'] = True
            
            # Test ChromaDB Integration
            chromadb_store = ChromaDBVectorStore()
            chromadb_store.initialize_client()
            results['chromadb_integration'] = True
            
        except Exception as e:
            logger.error(f"Custom embeddings test failed: {e}")
        
        return results
    
    def test_hybrid_rag(self) -> Dict[str, Any]:
        """Test hybrid RAG components."""
        logger.info("Testing hybrid RAG components...")
        
        results = {
            'multi_source_retrieval': False,
            'knowledge_graph': False,
            'device_context_retrieval': False,
            'customer_journey_rag': False,
            'unified_orchestrator': False
        }
        
        try:
            # Test Multi-Source Retrieval
            multi_source = MultiSourceRetrieval()
            results['multi_source_retrieval'] = True
            
            # Test Knowledge Graph
            knowledge_graph = LenovoKnowledgeGraph()
            results['knowledge_graph'] = True
            
            # Test Device Context Retrieval
            device_context = DeviceContextRetrieval()
            device_context.device_database = self.test_data
            results['device_context_retrieval'] = True
            
            # Test Customer Journey RAG
            customer_journey = CustomerJourneyRAG()
            customer_journey.customer_database = self.test_data
            results['customer_journey_rag'] = True
            
            # Test Unified Orchestrator
            orchestrator = UnifiedRetrievalOrchestrator(
                multi_source_retrieval=multi_source,
                knowledge_graph=knowledge_graph,
                device_context_retrieval=device_context,
                customer_journey_rag=customer_journey
            )
            results['unified_orchestrator'] = True
            
        except Exception as e:
            logger.error(f"Hybrid RAG test failed: {e}")
        
        return results
    
    def test_retrieval_workflows(self) -> Dict[str, Any]:
        """Test retrieval workflows components."""
        logger.info("Testing retrieval workflows components...")
        
        results = {
            'langchain_faiss': False,
            'llamaindex_retrieval': False,
            'hybrid_retrieval_system': False,
            'retrieval_evaluation': False,
            'mlflow_retrieval_tracking': False
        }
        
        try:
            # Test LangChain FAISS Integration
            langchain_faiss = LangChainFAISSIntegration()
            langchain_faiss.initialize_components()
            results['langchain_faiss'] = True
            
            # Test LlamaIndex Retrieval
            llamaindex_retrieval = LlamaIndexRetrieval()
            llamaindex_retrieval.initialize_components()
            results['llamaindex_retrieval'] = True
            
            # Test Hybrid Retrieval System
            hybrid_system = HybridRetrievalSystem(
                langchain_integration=langchain_faiss,
                llamaindex_integration=llamaindex_retrieval
            )
            results['hybrid_retrieval_system'] = True
            
            # Test Retrieval Evaluation
            retrieval_eval = RetrievalEvaluation()
            results['retrieval_evaluation'] = True
            
            # Test MLflow Retrieval Tracking
            mlflow_tracking = MLflowRetrievalTracking("retrieval_experiments_test")
            results['mlflow_retrieval_tracking'] = True
            
        except Exception as e:
            logger.error(f"Retrieval workflows test failed: {e}")
        
        return results
    
    def test_integration_workflows(self) -> Dict[str, Any]:
        """Test integration workflows between components."""
        logger.info("Testing integration workflows...")
        
        results = {
            'fine_tuning_workflow': False,
            'embedding_training_workflow': False,
            'hybrid_rag_workflow': False,
            'retrieval_evaluation_workflow': False,
            'end_to_end_workflow': False
        }
        
        try:
            # Test Fine-tuning Workflow
            adapter = LenovoDomainAdapter("microsoft/phi-4-mini-instruct")
            adapter.load_model()
            
            # Create test dataset
            test_data = [
                {
                    'instruction': 'How do I troubleshoot ThinkPad X1 Carbon battery issues?',
                    'context': 'ThinkPad X1 Carbon battery troubleshooting',
                    'response': 'Check power adapter connection and update power management drivers.',
                    'category': 'troubleshooting'
                }
            ]
            
            # Test workflow components
            dataset = adapter.prepare_lenovo_dataset_from_data(test_data)
            results['fine_tuning_workflow'] = True
            
            # Test Embedding Training Workflow
            tech_embeddings = LenovoTechnicalEmbeddings()
            tech_embeddings.load_base_model()
            
            # Create test documents
            test_docs = [
                'ThinkPad X1 Carbon setup guide for enterprise deployment',
                'Moto Edge 40 camera features and usage instructions'
            ]
            
            embeddings = tech_embeddings.create_technical_embeddings(test_docs)
            results['embedding_training_workflow'] = True
            
            # Test Hybrid RAG Workflow
            multi_source = MultiSourceRetrieval()
            device_context = DeviceContextRetrieval()
            device_context.device_database = self.test_data
            
            # Test device context retrieval
            device_specs = device_context.get_device_specifications('device_001')
            results['hybrid_rag_workflow'] = True
            
            # Test Retrieval Evaluation Workflow
            retrieval_eval = RetrievalEvaluation()
            test_queries = ['ThinkPad X1 Carbon battery issues']
            ground_truth = [['device_001']]
            
            # Mock retrieval system for testing
            class MockRetrievalSystem:
                def hybrid_retrieve(self, query, k=5):
                    return [{'id': 'device_001', 'content': 'ThinkPad X1 Carbon battery troubleshooting'}]
            
            mock_system = MockRetrievalSystem()
            evaluation_results = retrieval_eval.evaluate_retrieval_system(
                mock_system, test_queries, ground_truth
            )
            results['retrieval_evaluation_workflow'] = True
            
            # Test End-to-End Workflow
            orchestrator = UnifiedRetrievalOrchestrator(
                multi_source_retrieval=multi_source,
                device_context_retrieval=device_context
            )
            
            # Test orchestrated retrieval
            query_embedding = np.random.rand(384)  # Mock embedding
            orchestrated_results = orchestrator.orchestrate_retrieval(
                query="ThinkPad X1 Carbon battery issues",
                query_embedding=query_embedding
            )
            results['end_to_end_workflow'] = True
            
        except Exception as e:
            logger.error(f"Integration workflows test failed: {e}")
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all Phase 3 tests."""
        logger.info("Starting Phase 3 comprehensive testing...")
        
        # Run individual component tests
        mobile_fine_tuning_results = self.test_mobile_fine_tuning()
        custom_embeddings_results = self.test_custom_embeddings()
        hybrid_rag_results = self.test_hybrid_rag()
        retrieval_workflows_results = self.test_retrieval_workflows()
        integration_workflows_results = self.test_integration_workflows()
        
        # Compile results
        all_results = {
            'mobile_fine_tuning': mobile_fine_tuning_results,
            'custom_embeddings': custom_embeddings_results,
            'hybrid_rag': hybrid_rag_results,
            'retrieval_workflows': retrieval_workflows_results,
            'integration_workflows': integration_workflows_results
        }
        
        # Calculate overall success rate
        total_tests = 0
        passed_tests = 0
        
        for category, results in all_results.items():
            for test_name, passed in results.items():
                total_tests += 1
                if passed:
                    passed_tests += 1
        
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Create summary
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': success_rate,
            'phase3_status': 'COMPLETED' if success_rate >= 80 else 'PARTIAL',
            'detailed_results': all_results
        }
        
        logger.info(f"Phase 3 testing completed: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        
        return summary
    
    def save_test_results(self, results: Dict[str, Any], output_path: str = "phase3_test_results.json"):
        """Save test results to file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Test results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save test results: {e}")

def main():
    """Main test execution."""
    logger.info("🚀 Starting Phase 3 Implementation Testing")
    
    # Create tester
    tester = Phase3Tester()
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Save results
    tester.save_test_results(results)
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 PHASE 3 IMPLEMENTATION TEST RESULTS")
    print("="*60)
    print(f"Total Tests: {results['total_tests']}")
    print(f"Passed Tests: {results['passed_tests']}")
    print(f"Failed Tests: {results['failed_tests']}")
    print(f"Success Rate: {results['success_rate']:.1f}%")
    print(f"Phase 3 Status: {results['phase3_status']}")
    print("="*60)
    
    # Print detailed results
    for category, category_results in results['detailed_results'].items():
        print(f"\n📊 {category.upper().replace('_', ' ')}")
        print("-" * 40)
        for test_name, passed in category_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {test_name}: {status}")
    
    print("\n🎉 Phase 3 testing completed!")
    
    return results

if __name__ == "__main__":
    main()
