"""
Multi-Source Retrieval for Hybrid RAG

This module provides multi-source retrieval capabilities combining
ChromaDB, Neo4j, and DuckDB for comprehensive knowledge retrieval.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class MultiSourceRetrieval:
    """
    Multi-source retrieval system for hybrid RAG.
    
    Combines ChromaDB, Neo4j, and DuckDB for comprehensive knowledge retrieval
    with intelligent source selection and result fusion.
    """
    
    def __init__(self, 
                 chromadb_client=None,
                 neo4j_client=None,
                 duckdb_client=None,
                 retrieval_weights: Optional[Dict[str, float]] = None):
        """
        Initialize multi-source retrieval system.
        
        Args:
            chromadb_client: ChromaDB client
            neo4j_client: Neo4j client
            duckdb_client: DuckDB client
            retrieval_weights: Weights for different sources
        """
        self.chromadb_client = chromadb_client
        self.neo4j_client = neo4j_client
        self.duckdb_client = duckdb_client
        
        # Default retrieval weights
        self.retrieval_weights = retrieval_weights or {
            'chromadb': 0.4,
            'neo4j': 0.3,
            'duckdb': 0.3
        }
        
        # Source availability
        self.sources_available = {
            'chromadb': chromadb_client is not None,
            'neo4j': neo4j_client is not None,
            'duckdb': duckdb_client is not None
        }
        
    def retrieve_from_chromadb(self, 
                              query_embedding: np.ndarray,
                              collection_name: str,
                              n_results: int = 5,
                              where: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Retrieve from ChromaDB vector store.
        
        Args:
            query_embedding: Query embedding
            collection_name: ChromaDB collection name
            n_results: Number of results
            where: Metadata filter
            
        Returns:
            ChromaDB retrieval results
        """
        try:
            if not self.sources_available['chromadb']:
                return {'results': [], 'source': 'chromadb', 'available': False}
            
            # Get collection
            collection = self.chromadb_client.get_collection(collection_name)
            
            # Search collection
            results = collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results,
                where=where,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = {
                'source': 'chromadb',
                'results': results,
                'count': len(results['ids'][0]) if results['ids'] else 0,
                'available': True
            }
            
            logger.info(f"Retrieved {formatted_results['count']} results from ChromaDB")
            return formatted_results
            
        except Exception as e:
            logger.error(f"ChromaDB retrieval failed: {e}")
            return {'results': [], 'source': 'chromadb', 'available': False, 'error': str(e)}
    
    def retrieve_from_neo4j(self, 
                           query_text: str,
                           cypher_query: Optional[str] = None,
                           limit: int = 5) -> Dict[str, Any]:
        """
        Retrieve from Neo4j graph database.
        
        Args:
            query_text: Query text
            cypher_query: Custom Cypher query
            limit: Maximum number of results
            
        Returns:
            Neo4j retrieval results
        """
        try:
            if not self.sources_available['neo4j']:
                return {'results': [], 'source': 'neo4j', 'available': False}
            
            # Default Cypher query for text search
            if cypher_query is None:
                cypher_query = """
                MATCH (n)
                WHERE toLower(n.text) CONTAINS toLower($query_text)
                   OR toLower(n.title) CONTAINS toLower($query_text)
                   OR toLower(n.description) CONTAINS toLower($query_text)
                RETURN n, n.text as text, n.title as title, n.description as description
                LIMIT $limit
                """
            
            # Execute query
            with self.neo4j_client.session() as session:
                results = session.run(
                    cypher_query,
                    query_text=query_text,
                    limit=limit
                )
                
                # Format results
                formatted_results = []
                for record in results:
                    formatted_results.append({
                        'text': record.get('text', ''),
                        'title': record.get('title', ''),
                        'description': record.get('description', ''),
                        'node': dict(record['n'])
                    })
                
                neo4j_results = {
                    'source': 'neo4j',
                    'results': formatted_results,
                    'count': len(formatted_results),
                    'available': True
                }
                
                logger.info(f"Retrieved {neo4j_results['count']} results from Neo4j")
                return neo4j_results
                
        except Exception as e:
            logger.error(f"Neo4j retrieval failed: {e}")
            return {'results': [], 'source': 'neo4j', 'available': False, 'error': str(e)}
    
    def retrieve_from_duckdb(self, 
                            query_text: str,
                            sql_query: Optional[str] = None,
                            limit: int = 5) -> Dict[str, Any]:
        """
        Retrieve from DuckDB analytics database.
        
        Args:
            query_text: Query text
            sql_query: Custom SQL query
            limit: Maximum number of results
            
        Returns:
            DuckDB retrieval results
        """
        try:
            if not self.sources_available['duckdb']:
                return {'results': [], 'source': 'duckdb', 'available': False}
            
            # Default SQL query for text search
            if sql_query is None:
                sql_query = """
                SELECT * FROM documents 
                WHERE LOWER(content) LIKE LOWER('%' || ? || '%')
                   OR LOWER(title) LIKE LOWER('%' || ? || '%')
                LIMIT ?
                """
            
            # Execute query
            cursor = self.duckdb_client.execute(sql_query, [query_text, query_text, limit])
            results = cursor.fetchall()
            
            # Get column names
            columns = [desc[0] for desc in cursor.description]
            
            # Format results
            formatted_results = []
            for row in results:
                formatted_results.append(dict(zip(columns, row)))
            
            duckdb_results = {
                'source': 'duckdb',
                'results': formatted_results,
                'count': len(formatted_results),
                'available': True
            }
            
            logger.info(f"Retrieved {duckdb_results['count']} results from DuckDB")
            return duckdb_results
            
        except Exception as e:
            logger.error(f"DuckDB retrieval failed: {e}")
            return {'results': [], 'source': 'duckdb', 'available': False, 'error': str(e)}
    
    def retrieve_from_all_sources(self, 
                                 query_embedding: np.ndarray,
                                 query_text: str,
                                 collection_name: str = "lenovo_embeddings",
                                 n_results: int = 5) -> Dict[str, Any]:
        """
        Retrieve from all available sources.
        
        Args:
            query_embedding: Query embedding for vector search
            query_text: Query text for text search
            collection_name: ChromaDB collection name
            n_results: Number of results per source
            
        Returns:
            Combined retrieval results
        """
        try:
            # Retrieve from all sources in parallel
            with ThreadPoolExecutor(max_workers=3) as executor:
                # Submit tasks
                chromadb_future = executor.submit(
                    self.retrieve_from_chromadb,
                    query_embedding,
                    collection_name,
                    n_results
                )
                
                neo4j_future = executor.submit(
                    self.retrieve_from_neo4j,
                    query_text,
                    None,
                    n_results
                )
                
                duckdb_future = executor.submit(
                    self.retrieve_from_duckdb,
                    query_text,
                    None,
                    n_results
                )
                
                # Get results
                chromadb_results = chromadb_future.result()
                neo4j_results = neo4j_future.result()
                duckdb_results = duckdb_future.result()
            
            # Combine results
            combined_results = {
                'chromadb': chromadb_results,
                'neo4j': neo4j_results,
                'duckdb': duckdb_results,
                'total_results': (
                    chromadb_results['count'] + 
                    neo4j_results['count'] + 
                    duckdb_results['count']
                )
            }
            
            logger.info(f"Retrieved {combined_results['total_results']} total results from all sources")
            return combined_results
            
        except Exception as e:
            logger.error(f"Multi-source retrieval failed: {e}")
            raise
    
    def fuse_results(self, 
                   retrieval_results: Dict[str, Any],
                   fusion_method: str = "weighted") -> List[Dict[str, Any]]:
        """
        Fuse results from multiple sources.
        
        Args:
            retrieval_results: Results from all sources
            fusion_method: Fusion method (weighted, reciprocal_rank, comb_sum)
            
        Returns:
            Fused results
        """
        try:
            if fusion_method == "weighted":
                return self._weighted_fusion(retrieval_results)
            elif fusion_method == "reciprocal_rank":
                return self._reciprocal_rank_fusion(retrieval_results)
            elif fusion_method == "comb_sum":
                return self._comb_sum_fusion(retrieval_results)
            else:
                raise ValueError(f"Unsupported fusion method: {fusion_method}")
                
        except Exception as e:
            logger.error(f"Result fusion failed: {e}")
            raise
    
    def _weighted_fusion(self, retrieval_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Weighted fusion of results."""
        try:
            fused_results = []
            
            # Process each source
            for source, results in retrieval_results.items():
                if source in ['chromadb', 'neo4j', 'duckdb'] and results['available']:
                    weight = self.retrieval_weights.get(source, 0.0)
                    
                    for i, result in enumerate(results['results']):
                        # Calculate weighted score
                        if source == 'chromadb':
                            # Use distance as score (lower is better)
                            score = 1.0 - (result['distances'][0] if 'distances' in result else 0.0)
                        else:
                            # Use rank as score (higher is better)
                            score = 1.0 / (i + 1)
                        
                        weighted_score = score * weight
                        
                        fused_results.append({
                            'content': result.get('text', result.get('content', '')),
                            'title': result.get('title', ''),
                            'source': source,
                            'score': weighted_score,
                            'metadata': result.get('metadata', {})
                        })
            
            # Sort by weighted score
            fused_results.sort(key=lambda x: x['score'], reverse=True)
            
            logger.info(f"Fused {len(fused_results)} results using weighted fusion")
            return fused_results
            
        except Exception as e:
            logger.error(f"Weighted fusion failed: {e}")
            raise
    
    def _reciprocal_rank_fusion(self, retrieval_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Reciprocal rank fusion of results."""
        try:
            # Collect all results with ranks
            all_results = []
            
            for source, results in retrieval_results.items():
                if source in ['chromadb', 'neo4j', 'duckdb'] and results['available']:
                    for i, result in enumerate(results['results']):
                        all_results.append({
                            'content': result.get('text', result.get('content', '')),
                            'title': result.get('title', ''),
                            'source': source,
                            'rank': i + 1,
                            'metadata': result.get('metadata', {})
                        })
            
            # Group by content similarity (simple approach)
            content_groups = {}
            for result in all_results:
                content_key = result['content'][:100]  # Use first 100 chars as key
                if content_key not in content_groups:
                    content_groups[content_key] = []
                content_groups[content_key].append(result)
            
            # Calculate reciprocal rank scores
            fused_results = []
            for content_key, group in content_groups.items():
                total_score = 0.0
                for result in group:
                    total_score += 1.0 / result['rank']
                
                # Use the best result from the group
                best_result = max(group, key=lambda x: 1.0 / x['rank'])
                fused_results.append({
                    'content': best_result['content'],
                    'title': best_result['title'],
                    'source': best_result['source'],
                    'score': total_score,
                    'metadata': best_result['metadata']
                })
            
            # Sort by score
            fused_results.sort(key=lambda x: x['score'], reverse=True)
            
            logger.info(f"Fused {len(fused_results)} results using reciprocal rank fusion")
            return fused_results
            
        except Exception as e:
            logger.error(f"Reciprocal rank fusion failed: {e}")
            raise
    
    def _comb_sum_fusion(self, retrieval_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """CombSUM fusion of results."""
        try:
            # Collect all results
            all_results = []
            
            for source, results in retrieval_results.items():
                if source in ['chromadb', 'neo4j', 'duckdb'] and results['available']:
                    for result in results['results']:
                        all_results.append({
                            'content': result.get('text', result.get('content', '')),
                            'title': result.get('title', ''),
                            'source': source,
                            'score': 1.0,  # Default score
                            'metadata': result.get('metadata', {})
                        })
            
            # Group by content similarity
            content_groups = {}
            for result in all_results:
                content_key = result['content'][:100]
                if content_key not in content_groups:
                    content_groups[content_key] = []
                content_groups[content_key].append(result)
            
            # Calculate CombSUM scores
            fused_results = []
            for content_key, group in content_groups.items():
                total_score = sum(result['score'] for result in group)
                
                # Use the best result from the group
                best_result = max(group, key=lambda x: x['score'])
                fused_results.append({
                    'content': best_result['content'],
                    'title': best_result['title'],
                    'source': best_result['source'],
                    'score': total_score,
                    'metadata': best_result['metadata']
                })
            
            # Sort by score
            fused_results.sort(key=lambda x: x['score'], reverse=True)
            
            logger.info(f"Fused {len(fused_results)} results using CombSUM fusion")
            return fused_results
            
        except Exception as e:
            logger.error(f"CombSUM fusion failed: {e}")
            raise
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """
        Get retrieval statistics.
        
        Returns:
            Retrieval statistics
        """
        try:
            stats = {
                'sources_available': self.sources_available,
                'retrieval_weights': self.retrieval_weights,
                'total_sources': len(self.sources_available),
                'available_sources': sum(1 for available in self.sources_available.values() if available)
            }
            
            logger.info(f"Retrieval statistics: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get retrieval statistics: {e}")
            raise
