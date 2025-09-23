"""
Unified Retrieval Orchestrator for Hybrid RAG

This module provides unified orchestration of all retrieval components
including multi-source retrieval, knowledge graphs, and context-aware retrieval.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class UnifiedRetrievalOrchestrator:
    """
    Unified retrieval orchestrator for hybrid RAG.
    
    Orchestrates all retrieval components including multi-source retrieval,
    knowledge graphs, device context, and customer journey RAG.
    """
    
    def __init__(self, 
                 multi_source_retrieval=None,
                 knowledge_graph=None,
                 device_context_retrieval=None,
                 customer_journey_rag=None,
                 orchestration_config: Optional[Dict] = None):
        """
        Initialize unified retrieval orchestrator.
        
        Args:
            multi_source_retrieval: Multi-source retrieval component
            knowledge_graph: Knowledge graph component
            device_context_retrieval: Device context retrieval component
            customer_journey_rag: Customer journey RAG component
            orchestration_config: Orchestration configuration
        """
        self.multi_source_retrieval = multi_source_retrieval
        self.knowledge_graph = knowledge_graph
        self.device_context_retrieval = device_context_retrieval
        self.customer_journey_rag = customer_journey_rag
        
        # Default orchestration configuration
        self.orchestration_config = orchestration_config or {
            'enable_multi_source': True,
            'enable_knowledge_graph': True,
            'enable_device_context': True,
            'enable_customer_journey': True,
            'fusion_method': 'weighted',
            'max_results_per_source': 5,
            'total_max_results': 20,
            'context_weights': {
                'multi_source': 0.4,
                'knowledge_graph': 0.3,
                'device_context': 0.2,
                'customer_journey': 0.1
            }
        }
        
        # Component availability
        self.components_available = {
            'multi_source': multi_source_retrieval is not None,
            'knowledge_graph': knowledge_graph is not None,
            'device_context': device_context_retrieval is not None,
            'customer_journey': customer_journey_rag is not None
        }
    
    def orchestrate_retrieval(self, 
                            query: str,
                            query_embedding: np.ndarray,
                            context: Optional[Dict[str, Any]] = None,
                            retrieval_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Orchestrate retrieval from all available components.
        
        Args:
            query: Query text
            query_embedding: Query embedding
            context: Additional context
            retrieval_options: Retrieval options
            
        Returns:
            Orchestrated retrieval results
        """
        try:
            # Initialize results
            orchestrated_results = {
                'query': query,
                'context': context,
                'components_used': [],
                'results': [],
                'total_results': 0,
                'orchestration_metadata': {}
            }
            
            # Execute retrieval from all available components
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {}
                
                # Multi-source retrieval
                if self.components_available['multi_source'] and self.orchestration_config['enable_multi_source']:
                    futures['multi_source'] = executor.submit(
                        self._execute_multi_source_retrieval,
                        query, query_embedding, context, retrieval_options
                    )
                
                # Knowledge graph retrieval
                if self.components_available['knowledge_graph'] and self.orchestration_config['enable_knowledge_graph']:
                    futures['knowledge_graph'] = executor.submit(
                        self._execute_knowledge_graph_retrieval,
                        query, context, retrieval_options
                    )
                
                # Device context retrieval
                if self.components_available['device_context'] and self.orchestration_config['enable_device_context']:
                    futures['device_context'] = executor.submit(
                        self._execute_device_context_retrieval,
                        query, context, retrieval_options
                    )
                
                # Customer journey RAG
                if self.components_available['customer_journey'] and self.orchestration_config['enable_customer_journey']:
                    futures['customer_journey'] = executor.submit(
                        self._execute_customer_journey_retrieval,
                        query, context, retrieval_options
                    )
                
                # Collect results
                for component, future in futures.items():
                    try:
                        result = future.result(timeout=30)  # 30 second timeout
                        orchestrated_results['components_used'].append(component)
                        orchestrated_results['results'].extend(result.get('results', []))
                        orchestrated_results['orchestration_metadata'][component] = result.get('metadata', {})
                    except Exception as e:
                        logger.error(f"Component {component} failed: {e}")
                        orchestrated_results['orchestration_metadata'][component] = {'error': str(e)}
            
            # Fuse results
            fused_results = self._fuse_orchestrated_results(orchestrated_results['results'])
            
            # Limit total results
            max_results = self.orchestration_config.get('total_max_results', 20)
            fused_results = fused_results[:max_results]
            
            orchestrated_results['results'] = fused_results
            orchestrated_results['total_results'] = len(fused_results)
            
            logger.info(f"Orchestrated retrieval completed with {len(fused_results)} results")
            return orchestrated_results
            
        except Exception as e:
            logger.error(f"Orchestrated retrieval failed: {e}")
            raise
    
    def _execute_multi_source_retrieval(self, 
                                       query: str,
                                       query_embedding: np.ndarray,
                                       context: Optional[Dict[str, Any]],
                                       retrieval_options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute multi-source retrieval."""
        try:
            if not self.multi_source_retrieval:
                return {'results': [], 'metadata': {'error': 'Multi-source retrieval not available'}}
            
            # Get retrieval options
            n_results = retrieval_options.get('n_results', self.orchestration_config.get('max_results_per_source', 5))
            collection_name = retrieval_options.get('collection_name', 'lenovo_embeddings')
            
            # Execute multi-source retrieval
            results = self.multi_source_retrieval.retrieve_from_all_sources(
                query_embedding=query_embedding,
                query_text=query,
                collection_name=collection_name,
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            for source, source_results in results.items():
                if source in ['chromadb', 'neo4j', 'duckdb'] and source_results['available']:
                    for result in source_results['results']:
                        formatted_results.append({
                            'content': result.get('text', result.get('content', '')),
                            'title': result.get('title', ''),
                            'source': source,
                            'score': result.get('score', 0.0),
                            'metadata': result.get('metadata', {}),
                            'component': 'multi_source'
                        })
            
            return {
                'results': formatted_results,
                'metadata': {
                    'sources_used': list(results.keys()),
                    'total_sources': len(results),
                    'available_sources': sum(1 for r in results.values() if r['available'])
                }
            }
            
        except Exception as e:
            logger.error(f"Multi-source retrieval failed: {e}")
            return {'results': [], 'metadata': {'error': str(e)}}
    
    def _execute_knowledge_graph_retrieval(self, 
                                         query: str,
                                         context: Optional[Dict[str, Any]],
                                         retrieval_options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute knowledge graph retrieval."""
        try:
            if not self.knowledge_graph:
                return {'results': [], 'metadata': {'error': 'Knowledge graph not available'}}
            
            # Get retrieval options
            query_type = retrieval_options.get('query_type', 'text_search')
            limit = retrieval_options.get('limit', self.orchestration_config.get('max_results_per_source', 5))
            
            # Execute knowledge graph query
            results = self.knowledge_graph.query_knowledge_graph(
                query=query,
                query_type=query_type,
                limit=limit
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'content': result.get('name', ''),
                    'title': result.get('title', ''),
                    'source': 'knowledge_graph',
                    'score': 1.0,  # Default score for knowledge graph
                    'metadata': result.get('properties', {}),
                    'component': 'knowledge_graph'
                })
            
            return {
                'results': formatted_results,
                'metadata': {
                    'query_type': query_type,
                    'total_results': len(formatted_results)
                }
            }
            
        except Exception as e:
            logger.error(f"Knowledge graph retrieval failed: {e}")
            return {'results': [], 'metadata': {'error': str(e)}}
    
    def _execute_device_context_retrieval(self, 
                                        query: str,
                                        context: Optional[Dict[str, Any]],
                                        retrieval_options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute device context retrieval."""
        try:
            if not self.device_context_retrieval:
                return {'results': [], 'metadata': {'error': 'Device context retrieval not available'}}
            
            # Get retrieval options
            device_type = retrieval_options.get('device_type')
            category = retrieval_options.get('category')
            
            # Execute device context search
            results = self.device_context_retrieval.search_device_support(
                query=query,
                device_type=device_type,
                category=category
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'content': result.get('support_info', ''),
                    'title': result.get('device_name', ''),
                    'source': 'device_context',
                    'score': 0.8,  # Default score for device context
                    'metadata': {
                        'device_id': result.get('device_id'),
                        'device_type': result.get('device_type'),
                        'category': result.get('category'),
                        'common_issues': result.get('common_issues', [])
                    },
                    'component': 'device_context'
                })
            
            return {
                'results': formatted_results,
                'metadata': {
                    'device_type_filter': device_type,
                    'category_filter': category,
                    'total_results': len(formatted_results)
                }
            }
            
        except Exception as e:
            logger.error(f"Device context retrieval failed: {e}")
            return {'results': [], 'metadata': {'error': str(e)}}
    
    def _execute_customer_journey_retrieval(self, 
                                          query: str,
                                          context: Optional[Dict[str, Any]],
                                          retrieval_options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute customer journey retrieval."""
        try:
            if not self.customer_journey_rag:
                return {'results': [], 'metadata': {'error': 'Customer journey RAG not available'}}
            
            # Get retrieval options
            customer_id = retrieval_options.get('customer_id')
            context_type = retrieval_options.get('context_type', 'comprehensive')
            
            # Execute customer journey search
            results = self.customer_journey_rag.search_customer_context(
                query=query,
                customer_id=customer_id,
                context_type=context_type
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'content': result.get('content', ''),
                    'title': f"Customer {result.get('customer_id', '')}",
                    'source': 'customer_journey',
                    'score': result.get('relevance_score', 0.0),
                    'metadata': {
                        'customer_id': result.get('customer_id'),
                        'type': result.get('type'),
                        'relevance_score': result.get('relevance_score')
                    },
                    'component': 'customer_journey'
                })
            
            return {
                'results': formatted_results,
                'metadata': {
                    'customer_id_filter': customer_id,
                    'context_type': context_type,
                    'total_results': len(formatted_results)
                }
            }
            
        except Exception as e:
            logger.error(f"Customer journey retrieval failed: {e}")
            return {'results': [], 'metadata': {'error': str(e)}}
    
    def _fuse_orchestrated_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fuse results from all components."""
        try:
            # Apply component weights
            weighted_results = []
            for result in results:
                component = result.get('component', 'unknown')
                weight = self.orchestration_config['context_weights'].get(component, 0.1)
                
                # Calculate weighted score
                original_score = result.get('score', 0.0)
                weighted_score = original_score * weight
                
                result['weighted_score'] = weighted_score
                weighted_results.append(result)
            
            # Sort by weighted score
            weighted_results.sort(key=lambda x: x['weighted_score'], reverse=True)
            
            # Remove duplicates based on content similarity
            deduplicated_results = self._deduplicate_results(weighted_results)
            
            logger.info(f"Fused {len(results)} results into {len(deduplicated_results)} unique results")
            return deduplicated_results
            
        except Exception as e:
            logger.error(f"Result fusion failed: {e}")
            return results
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on content similarity."""
        try:
            deduplicated = []
            seen_content = set()
            
            for result in results:
                content = result.get('content', '')
                content_key = content[:100]  # Use first 100 characters as key
                
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    deduplicated.append(result)
            
            return deduplicated
            
        except Exception as e:
            logger.error(f"Result deduplication failed: {e}")
            return results
    
    def get_orchestration_statistics(self) -> Dict[str, Any]:
        """
        Get orchestration statistics.
        
        Returns:
            Orchestration statistics
        """
        try:
            stats = {
                'components_available': self.components_available,
                'orchestration_config': self.orchestration_config,
                'total_components': len(self.components_available),
                'available_components': sum(1 for available in self.components_available.values() if available)
            }
            
            logger.info(f"Orchestration statistics: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get orchestration statistics: {e}")
            raise
    
    def update_orchestration_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update orchestration configuration.
        
        Args:
            new_config: New configuration
        """
        try:
            self.orchestration_config.update(new_config)
            logger.info(f"Updated orchestration configuration: {new_config}")
            
        except Exception as e:
            logger.error(f"Failed to update orchestration configuration: {e}")
            raise
    
    def add_component(self, 
                     component_name: str,
                     component_instance: Any) -> None:
        """
        Add a new component to the orchestrator.
        
        Args:
            component_name: Name of the component
            component_instance: Component instance
        """
        try:
            if component_name == 'multi_source':
                self.multi_source_retrieval = component_instance
            elif component_name == 'knowledge_graph':
                self.knowledge_graph = component_instance
            elif component_name == 'device_context':
                self.device_context_retrieval = component_instance
            elif component_name == 'customer_journey':
                self.customer_journey_rag = component_instance
            else:
                raise ValueError(f"Unsupported component: {component_name}")
            
            self.components_available[component_name] = True
            logger.info(f"Added component: {component_name}")
            
        except Exception as e:
            logger.error(f"Failed to add component {component_name}: {e}")
            raise
    
    def remove_component(self, component_name: str) -> None:
        """
        Remove a component from the orchestrator.
        
        Args:
            component_name: Name of the component to remove
        """
        try:
            if component_name == 'multi_source':
                self.multi_source_retrieval = None
            elif component_name == 'knowledge_graph':
                self.knowledge_graph = None
            elif component_name == 'device_context':
                self.device_context_retrieval = None
            elif component_name == 'customer_journey':
                self.customer_journey_rag = None
            else:
                raise ValueError(f"Unsupported component: {component_name}")
            
            self.components_available[component_name] = False
            logger.info(f"Removed component: {component_name}")
            
        except Exception as e:
            logger.error(f"Failed to remove component {component_name}: {e}")
            raise
