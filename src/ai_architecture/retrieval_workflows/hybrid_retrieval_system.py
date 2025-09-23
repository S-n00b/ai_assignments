"""
Hybrid Retrieval System for LangChain and LlamaIndex

This module provides a unified hybrid retrieval system combining
LangChain and LlamaIndex for comprehensive retrieval workflows.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class HybridRetrievalSystem:
    """
    Hybrid retrieval system combining LangChain and LlamaIndex.
    
    Provides unified retrieval capabilities combining LangChain and LlamaIndex
    for comprehensive retrieval workflows.
    """
    
    def __init__(self, 
                 langchain_integration=None,
                 llamaindex_integration=None,
                 fusion_method: str = "weighted",
                 weights: Optional[Dict[str, float]] = None):
        """
        Initialize hybrid retrieval system.
        
        Args:
            langchain_integration: LangChain FAISS integration
            llamaindex_integration: LlamaIndex retrieval integration
            fusion_method: Method for fusing results (weighted, reciprocal_rank, comb_sum)
            weights: Weights for different components
        """
        self.langchain_integration = langchain_integration
        self.llamaindex_integration = llamaindex_integration
        self.fusion_method = fusion_method
        
        # Default weights
        self.weights = weights or {
            'langchain': 0.6,
            'llamaindex': 0.4
        }
        
        # Component availability
        self.components_available = {
            'langchain': langchain_integration is not None,
            'llamaindex': llamaindex_integration is not None
        }
    
    def retrieve_from_langchain(self, 
                               query: str,
                               k: int = 5,
                               search_type: str = "similarity") -> List[Dict[str, Any]]:
        """
        Retrieve from LangChain integration.
        
        Args:
            query: Query text
            k: Number of results
            search_type: Type of search
            
        Returns:
            List of retrieval results
        """
        try:
            if not self.components_available['langchain']:
                return []
            
            # Search using LangChain
            if search_type == "similarity":
                docs = self.langchain_integration.search_similar_documents(query, k=k)
            elif search_type == "mmr":
                docs = self.langchain_integration.search_mmr(query, k=k)
            else:
                docs = self.langchain_integration.search_similar_documents(query, k=k)
            
            # Format results
            results = []
            for doc in docs:
                result = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'source': 'langchain',
                    'score': 1.0,  # Default score
                    'component': 'langchain'
                }
                results.append(result)
            
            logger.info(f"Retrieved {len(results)} results from LangChain")
            return results
            
        except Exception as e:
            logger.error(f"LangChain retrieval failed: {e}")
            return []
    
    def retrieve_from_llamaindex(self, 
                                query: str,
                                similarity_top_k: int = 5,
                                similarity_cutoff: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve from LlamaIndex integration.
        
        Args:
            query: Query text
            similarity_top_k: Number of top similar documents
            similarity_cutoff: Similarity cutoff threshold
            
        Returns:
            List of retrieval results
        """
        try:
            if not self.components_available['llamaindex']:
                return []
            
            # Search using LlamaIndex
            results = self.llamaindex_integration.query_index(
                query=query,
                similarity_top_k=similarity_top_k,
                similarity_cutoff=similarity_cutoff
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_result = {
                    'content': result['text'],
                    'metadata': result['metadata'],
                    'source': 'llamaindex',
                    'score': result['score'],
                    'component': 'llamaindex'
                }
                formatted_results.append(formatted_result)
            
            logger.info(f"Retrieved {len(formatted_results)} results from LlamaIndex")
            return formatted_results
            
        except Exception as e:
            logger.error(f"LlamaIndex retrieval failed: {e}")
            return []
    
    def hybrid_retrieve(self, 
                       query: str,
                       k: int = 5,
                       langchain_k: int = 3,
                       llamaindex_k: int = 3) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval from both systems.
        
        Args:
            query: Query text
            k: Total number of results
            langchain_k: Number of results from LangChain
            llamaindex_k: Number of results from LlamaIndex
            
        Returns:
            List of fused retrieval results
        """
        try:
            # Retrieve from both systems in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit tasks
                langchain_future = executor.submit(
                    self.retrieve_from_langchain,
                    query, langchain_k
                )
                
                llamaindex_future = executor.submit(
                    self.retrieve_from_llamaindex,
                    query, llamaindex_k
                )
                
                # Get results
                langchain_results = langchain_future.result()
                llamaindex_results = llamaindex_future.result()
            
            # Combine results
            all_results = langchain_results + llamaindex_results
            
            # Fuse results
            fused_results = self._fuse_results(all_results, k)
            
            logger.info(f"Hybrid retrieval completed with {len(fused_results)} results")
            return fused_results
            
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            return []
    
    def _fuse_results(self, 
                     results: List[Dict[str, Any]], 
                     k: int) -> List[Dict[str, Any]]:
        """
        Fuse results from multiple systems.
        
        Args:
            results: List of all results
            k: Number of final results
            
        Returns:
            List of fused results
        """
        try:
            if self.fusion_method == "weighted":
                return self._weighted_fusion(results, k)
            elif self.fusion_method == "reciprocal_rank":
                return self._reciprocal_rank_fusion(results, k)
            elif self.fusion_method == "comb_sum":
                return self._comb_sum_fusion(results, k)
            else:
                raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
                
        except Exception as e:
            logger.error(f"Result fusion failed: {e}")
            return results[:k]
    
    def _weighted_fusion(self, 
                        results: List[Dict[str, Any]], 
                        k: int) -> List[Dict[str, Any]]:
        """Weighted fusion of results."""
        try:
            # Apply weights to scores
            weighted_results = []
            for result in results:
                component = result.get('component', 'unknown')
                weight = self.weights.get(component, 0.5)
                
                # Calculate weighted score
                original_score = result.get('score', 0.0)
                weighted_score = original_score * weight
                
                result['weighted_score'] = weighted_score
                weighted_results.append(result)
            
            # Sort by weighted score
            weighted_results.sort(key=lambda x: x['weighted_score'], reverse=True)
            
            # Remove duplicates based on content similarity
            deduplicated_results = self._deduplicate_results(weighted_results)
            
            return deduplicated_results[:k]
            
        except Exception as e:
            logger.error(f"Weighted fusion failed: {e}")
            return results[:k]
    
    def _reciprocal_rank_fusion(self, 
                               results: List[Dict[str, Any]], 
                               k: int) -> List[Dict[str, Any]]:
        """Reciprocal rank fusion of results."""
        try:
            # Group results by content similarity
            content_groups = {}
            for result in results:
                content_key = result['content'][:100]  # Use first 100 chars as key
                if content_key not in content_groups:
                    content_groups[content_key] = []
                content_groups[content_key].append(result)
            
            # Calculate reciprocal rank scores
            fused_results = []
            for content_key, group in content_groups.items():
                total_score = 0.0
                for i, result in enumerate(group):
                    rank = i + 1
                    total_score += 1.0 / rank
                
                # Use the best result from the group
                best_result = max(group, key=lambda x: 1.0 / (group.index(x) + 1))
                best_result['reciprocal_rank_score'] = total_score
                fused_results.append(best_result)
            
            # Sort by reciprocal rank score
            fused_results.sort(key=lambda x: x['reciprocal_rank_score'], reverse=True)
            
            return fused_results[:k]
            
        except Exception as e:
            logger.error(f"Reciprocal rank fusion failed: {e}")
            return results[:k]
    
    def _comb_sum_fusion(self, 
                        results: List[Dict[str, Any]], 
                        k: int) -> List[Dict[str, Any]]:
        """CombSUM fusion of results."""
        try:
            # Group results by content similarity
            content_groups = {}
            for result in results:
                content_key = result['content'][:100]
                if content_key not in content_groups:
                    content_groups[content_key] = []
                content_groups[content_key].append(result)
            
            # Calculate CombSUM scores
            fused_results = []
            for content_key, group in content_groups.items():
                total_score = sum(result.get('score', 0.0) for result in group)
                
                # Use the best result from the group
                best_result = max(group, key=lambda x: x.get('score', 0.0))
                best_result['comb_sum_score'] = total_score
                fused_results.append(best_result)
            
            # Sort by CombSUM score
            fused_results.sort(key=lambda x: x['comb_sum_score'], reverse=True)
            
            return fused_results[:k]
            
        except Exception as e:
            logger.error(f"CombSUM fusion failed: {e}")
            return results[:k]
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on content similarity."""
        try:
            deduplicated = []
            seen_content = set()
            
            for result in results:
                content = result['content']
                content_key = content[:100]  # Use first 100 characters as key
                
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    deduplicated.append(result)
            
            return deduplicated
            
        except Exception as e:
            logger.error(f"Result deduplication failed: {e}")
            return results
    
    def evaluate_hybrid_system(self, 
                             test_queries: List[str],
                             ground_truth: List[List[str]]) -> Dict[str, float]:
        """
        Evaluate hybrid retrieval system.
        
        Args:
            test_queries: List of test queries
            ground_truth: Ground truth results for each query
            
        Returns:
            Evaluation metrics
        """
        try:
            total_precision = 0.0
            total_recall = 0.0
            total_f1 = 0.0
            
            for i, query in enumerate(test_queries):
                # Get hybrid retrieval results
                results = self.hybrid_retrieve(query, k=5)
                retrieved_ids = [result['metadata'].get('id', '') for result in results]
                
                # Get ground truth
                gt_ids = ground_truth[i] if i < len(ground_truth) else []
                
                # Calculate metrics
                if retrieved_ids and gt_ids:
                    # Precision
                    precision = len(set(retrieved_ids).intersection(set(gt_ids))) / len(retrieved_ids)
                    total_precision += precision
                    
                    # Recall
                    recall = len(set(retrieved_ids).intersection(set(gt_ids))) / len(gt_ids)
                    total_recall += recall
                    
                    # F1 score
                    if precision + recall > 0:
                        f1 = 2 * (precision * recall) / (precision + recall)
                        total_f1 += f1
            
            # Calculate averages
            num_queries = len(test_queries)
            avg_precision = total_precision / num_queries if num_queries > 0 else 0.0
            avg_recall = total_recall / num_queries if num_queries > 0 else 0.0
            avg_f1 = total_f1 / num_queries if num_queries > 0 else 0.0
            
            metrics = {
                'precision': avg_precision,
                'recall': avg_recall,
                'f1_score': avg_f1,
                'fusion_method': self.fusion_method
            }
            
            logger.info(f"Hybrid system evaluation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to evaluate hybrid system: {e}")
            raise
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """
        Get hybrid system statistics.
        
        Returns:
            System statistics
        """
        try:
            stats = {
                'components_available': self.components_available,
                'fusion_method': self.fusion_method,
                'weights': self.weights,
                'total_components': len(self.components_available),
                'available_components': sum(1 for available in self.components_available.values() if available)
            }
            
            # Add component-specific statistics
            if self.components_available['langchain']:
                langchain_stats = self.langchain_integration.get_vectorstore_info()
                stats['langchain_stats'] = langchain_stats
            
            if self.components_available['llamaindex']:
                llamaindex_stats = self.llamaindex_integration.get_index_statistics()
                stats['llamaindex_stats'] = llamaindex_stats
            
            logger.info(f"Hybrid system statistics: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get system statistics: {e}")
            raise
    
    def update_fusion_method(self, method: str) -> None:
        """
        Update fusion method.
        
        Args:
            method: New fusion method
        """
        try:
            if method not in ['weighted', 'reciprocal_rank', 'comb_sum']:
                raise ValueError(f"Unsupported fusion method: {method}")
            
            self.fusion_method = method
            logger.info(f"Updated fusion method to: {method}")
            
        except Exception as e:
            logger.error(f"Failed to update fusion method: {e}")
            raise
    
    def update_weights(self, weights: Dict[str, float]) -> None:
        """
        Update component weights.
        
        Args:
            weights: New weights
        """
        try:
            self.weights.update(weights)
            logger.info(f"Updated weights: {weights}")
            
        except Exception as e:
            logger.error(f"Failed to update weights: {e}")
            raise
