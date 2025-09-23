"""
Retrieval Evaluation for Hybrid RAG

This module provides comprehensive evaluation capabilities for retrieval systems
including precision, recall, F1 score, and other retrieval metrics.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import ndcg_score, dcg_score
import time

logger = logging.getLogger(__name__)

class RetrievalEvaluation:
    """
    Retrieval evaluation for hybrid RAG systems.
    
    Provides comprehensive evaluation capabilities for retrieval systems
    including precision, recall, F1 score, and other retrieval metrics.
    """
    
    def __init__(self, 
                 evaluation_config: Optional[Dict] = None):
        """
        Initialize retrieval evaluation.
        
        Args:
            evaluation_config: Evaluation configuration
        """
        self.evaluation_config = evaluation_config or {
            'metrics': ['precision', 'recall', 'f1', 'ndcg', 'mrr'],
            'k_values': [1, 3, 5, 10],
            'similarity_threshold': 0.5,
            'timeout_seconds': 30
        }
    
    def evaluate_retrieval_system(self, 
                                retrieval_system,
                                test_queries: List[str],
                                ground_truth: List[List[str]],
                                evaluation_metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate retrieval system performance.
        
        Args:
            retrieval_system: Retrieval system to evaluate
            test_queries: List of test queries
            ground_truth: Ground truth results for each query
            evaluation_metrics: List of metrics to compute
            
        Returns:
            Evaluation results
        """
        try:
            if evaluation_metrics is None:
                evaluation_metrics = self.evaluation_config['metrics']
            
            # Initialize results
            evaluation_results = {
                'queries_evaluated': len(test_queries),
                'metrics': {},
                'per_query_results': [],
                'system_info': self._get_system_info(retrieval_system)
            }
            
            # Evaluate each query
            for i, query in enumerate(test_queries):
                query_start_time = time.time()
                
                try:
                    # Get retrieval results
                    if hasattr(retrieval_system, 'hybrid_retrieve'):
                        results = retrieval_system.hybrid_retrieve(query, k=10)
                    elif hasattr(retrieval_system, 'retrieve'):
                        results = retrieval_system.retrieve(query)
                    else:
                        results = retrieval_system(query)
                    
                    # Calculate query metrics
                    query_metrics = self._calculate_query_metrics(
                        results, ground_truth[i], evaluation_metrics
                    )
                    
                    query_metrics['query'] = query
                    query_metrics['query_index'] = i
                    query_metrics['execution_time'] = time.time() - query_start_time
                    
                    evaluation_results['per_query_results'].append(query_metrics)
                    
                except Exception as e:
                    logger.error(f"Query {i} evaluation failed: {e}")
                    evaluation_results['per_query_results'].append({
                        'query': query,
                        'query_index': i,
                        'error': str(e),
                        'execution_time': time.time() - query_start_time
                    })
            
            # Calculate aggregate metrics
            aggregate_metrics = self._calculate_aggregate_metrics(
                evaluation_results['per_query_results'], evaluation_metrics
            )
            
            evaluation_results['metrics'] = aggregate_metrics
            
            logger.info(f"Retrieval evaluation completed for {len(test_queries)} queries")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Retrieval evaluation failed: {e}")
            raise
    
    def _calculate_query_metrics(self, 
                                results: List[Dict[str, Any]], 
                                ground_truth: List[str],
                                metrics: List[str]) -> Dict[str, float]:
        """Calculate metrics for a single query."""
        try:
            query_metrics = {}
            
            # Extract retrieved IDs
            retrieved_ids = []
            for result in results:
                if isinstance(result, dict):
                    retrieved_ids.append(result.get('id', result.get('metadata', {}).get('id', '')))
                else:
                    retrieved_ids.append(str(result))
            
            # Calculate binary relevance
            binary_relevance = [1 if doc_id in ground_truth else 0 for doc_id in retrieved_ids]
            
            # Calculate metrics
            if 'precision' in metrics:
                query_metrics['precision'] = self._calculate_precision(binary_relevance)
            
            if 'recall' in metrics:
                query_metrics['recall'] = self._calculate_recall(binary_relevance, len(ground_truth))
            
            if 'f1' in metrics:
                precision = query_metrics.get('precision', 0.0)
                recall = query_metrics.get('recall', 0.0)
                query_metrics['f1'] = self._calculate_f1(precision, recall)
            
            if 'ndcg' in metrics:
                query_metrics['ndcg'] = self._calculate_ndcg(binary_relevance)
            
            if 'mrr' in metrics:
                query_metrics['mrr'] = self._calculate_mrr(binary_relevance)
            
            # Calculate metrics for different k values
            for k in self.evaluation_config['k_values']:
                if k <= len(binary_relevance):
                    k_relevance = binary_relevance[:k]
                    
                    if 'precision' in metrics:
                        query_metrics[f'precision@{k}'] = self._calculate_precision(k_relevance)
                    
                    if 'recall' in metrics:
                        query_metrics[f'recall@{k}'] = self._calculate_recall(k_relevance, len(ground_truth))
                    
                    if 'f1' in metrics:
                        precision_k = query_metrics.get(f'precision@{k}', 0.0)
                        recall_k = query_metrics.get(f'recall@{k}', 0.0)
                        query_metrics[f'f1@{k}'] = self._calculate_f1(precision_k, recall_k)
            
            return query_metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate query metrics: {e}")
            return {}
    
    def _calculate_precision(self, binary_relevance: List[int]) -> float:
        """Calculate precision."""
        if not binary_relevance:
            return 0.0
        
        return sum(binary_relevance) / len(binary_relevance)
    
    def _calculate_recall(self, binary_relevance: List[int], total_relevant: int) -> float:
        """Calculate recall."""
        if total_relevant == 0:
            return 0.0
        
        return sum(binary_relevance) / total_relevant
    
    def _calculate_f1(self, precision: float, recall: float) -> float:
        """Calculate F1 score."""
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def _calculate_ndcg(self, binary_relevance: List[int]) -> float:
        """Calculate NDCG."""
        if not binary_relevance:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for i, rel in enumerate(binary_relevance):
            dcg += rel / np.log2(i + 2)
        
        # Calculate IDCG (ideal DCG)
        ideal_relevance = sorted(binary_relevance, reverse=True)
        idcg = 0.0
        for i, rel in enumerate(ideal_relevance):
            idcg += rel / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def _calculate_mrr(self, binary_relevance: List[int]) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, rel in enumerate(binary_relevance):
            if rel == 1:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def _calculate_aggregate_metrics(self, 
                                   per_query_results: List[Dict[str, Any]], 
                                   metrics: List[str]) -> Dict[str, float]:
        """Calculate aggregate metrics across all queries."""
        try:
            aggregate_metrics = {}
            
            # Calculate averages for each metric
            for metric in metrics:
                values = []
                for result in per_query_results:
                    if metric in result and not isinstance(result[metric], str):
                        values.append(result[metric])
                
                if values:
                    aggregate_metrics[f'avg_{metric}'] = np.mean(values)
                    aggregate_metrics[f'std_{metric}'] = np.std(values)
                    aggregate_metrics[f'min_{metric}'] = np.min(values)
                    aggregate_metrics[f'max_{metric}'] = np.max(values)
            
            # Calculate metrics for different k values
            for k in self.evaluation_config['k_values']:
                for metric in metrics:
                    metric_k = f'{metric}@{k}'
                    values = []
                    for result in per_query_results:
                        if metric_k in result and not isinstance(result[metric_k], str):
                            values.append(result[metric_k])
                    
                    if values:
                        aggregate_metrics[f'avg_{metric_k}'] = np.mean(values)
                        aggregate_metrics[f'std_{metric_k}'] = np.std(values)
            
            # Calculate execution time statistics
            execution_times = [result.get('execution_time', 0) for result in per_query_results]
            if execution_times:
                aggregate_metrics['avg_execution_time'] = np.mean(execution_times)
                aggregate_metrics['std_execution_time'] = np.std(execution_times)
                aggregate_metrics['min_execution_time'] = np.min(execution_times)
                aggregate_metrics['max_execution_time'] = np.max(execution_times)
            
            return aggregate_metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate aggregate metrics: {e}")
            return {}
    
    def _get_system_info(self, retrieval_system) -> Dict[str, Any]:
        """Get system information."""
        try:
            system_info = {
                'system_type': type(retrieval_system).__name__,
                'available_methods': [method for method in dir(retrieval_system) if not method.startswith('_')]
            }
            
            # Get system-specific statistics
            if hasattr(retrieval_system, 'get_system_statistics'):
                system_info['statistics'] = retrieval_system.get_system_statistics()
            
            return system_info
            
        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            return {'error': str(e)}
    
    def compare_retrieval_systems(self, 
                                systems: List[Tuple[str, Any]],
                                test_queries: List[str],
                                ground_truth: List[List[str]]) -> Dict[str, Any]:
        """
        Compare multiple retrieval systems.
        
        Args:
            systems: List of (name, system) tuples
            test_queries: List of test queries
            ground_truth: Ground truth results for each query
            
        Returns:
            Comparison results
        """
        try:
            comparison_results = {
                'systems_compared': len(systems),
                'queries_evaluated': len(test_queries),
                'system_results': {},
                'comparison_metrics': {}
            }
            
            # Evaluate each system
            for system_name, system in systems:
                try:
                    system_results = self.evaluate_retrieval_system(
                        system, test_queries, ground_truth
                    )
                    comparison_results['system_results'][system_name] = system_results
                    
                except Exception as e:
                    logger.error(f"System {system_name} evaluation failed: {e}")
                    comparison_results['system_results'][system_name] = {'error': str(e)}
            
            # Calculate comparison metrics
            comparison_metrics = self._calculate_comparison_metrics(
                comparison_results['system_results']
            )
            comparison_results['comparison_metrics'] = comparison_metrics
            
            logger.info(f"Compared {len(systems)} retrieval systems")
            return comparison_results
            
        except Exception as e:
            logger.error(f"System comparison failed: {e}")
            raise
    
    def _calculate_comparison_metrics(self, system_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comparison metrics between systems."""
        try:
            comparison_metrics = {}
            
            # Get metrics for each system
            system_metrics = {}
            for system_name, results in system_results.items():
                if 'metrics' in results:
                    system_metrics[system_name] = results['metrics']
            
            # Compare metrics
            for metric_name in ['avg_precision', 'avg_recall', 'avg_f1', 'avg_ndcg', 'avg_mrr']:
                if metric_name in system_metrics.get(list(system_metrics.keys())[0], {}):
                    metric_values = {}
                    for system_name, metrics in system_metrics.items():
                        if metric_name in metrics:
                            metric_values[system_name] = metrics[metric_name]
                    
                    if metric_values:
                        # Find best system
                        best_system = max(metric_values, key=metric_values.get)
                        best_value = metric_values[best_system]
                        
                        comparison_metrics[metric_name] = {
                            'best_system': best_system,
                            'best_value': best_value,
                            'all_values': metric_values
                        }
            
            return comparison_metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate comparison metrics: {e}")
            return {}
    
    def generate_evaluation_report(self, 
                                 evaluation_results: Dict[str, Any],
                                 output_path: Optional[str] = None) -> str:
        """
        Generate evaluation report.
        
        Args:
            evaluation_results: Evaluation results
            output_path: Path to save report
            
        Returns:
            Report content
        """
        try:
            # Generate report content
            report_content = {
                'evaluation_summary': {
                    'queries_evaluated': evaluation_results.get('queries_evaluated', 0),
                    'system_info': evaluation_results.get('system_info', {}),
                    'evaluation_timestamp': time.time()
                },
                'aggregate_metrics': evaluation_results.get('metrics', {}),
                'per_query_results': evaluation_results.get('per_query_results', [])
            }
            
            # Save report if path provided
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(report_content, f, indent=2, ensure_ascii=False)
                logger.info(f"Evaluation report saved to {output_path}")
            
            return json.dumps(report_content, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Failed to generate evaluation report: {e}")
            raise
