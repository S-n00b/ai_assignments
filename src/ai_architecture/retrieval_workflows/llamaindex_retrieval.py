"""
LlamaIndex Retrieval for Hybrid RAG

This module provides LlamaIndex integration for advanced retrieval workflows
including document indexing, query engines, and retrieval evaluation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json
import numpy as np
from llama_index import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Document,
    StorageContext,
    load_index_from_storage
)
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SimilarityPostprocessor
from llama_index.evaluation import RetrieverEvaluator
from llama_index.evaluation import QueryResponseEvaluator

logger = logging.getLogger(__name__)

class LlamaIndexRetrieval:
    """
    LlamaIndex retrieval for hybrid RAG.
    
    Provides LlamaIndex integration for advanced retrieval workflows
    including document indexing, query engines, and retrieval evaluation.
    """
    
    def __init__(self, 
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 storage_path: Optional[str] = None,
                 chunk_size: int = 1024,
                 chunk_overlap: int = 200):
        """
        Initialize LlamaIndex retrieval.
        
        Args:
            embedding_model_name: Name of the embedding model
            storage_path: Path to storage directory
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.embedding_model_name = embedding_model_name
        self.storage_path = storage_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.embedding_model = None
        self.index = None
        self.retriever = None
        self.query_engine = None
        
    def initialize_components(self) -> None:
        """Initialize LlamaIndex components."""
        try:
            # Initialize embedding model
            self.embedding_model = HuggingFaceEmbedding(
                model_name=self.embedding_model_name
            )
            
            logger.info("Initialized LlamaIndex components")
            
        except Exception as e:
            logger.error(f"Failed to initialize LlamaIndex components: {e}")
            raise
    
    def create_documents_from_texts(self, 
                                  texts: List[str],
                                  metadatas: Optional[List[Dict]] = None) -> List[Document]:
        """
        Create LlamaIndex documents from texts.
        
        Args:
            texts: List of texts
            metadatas: List of metadata dictionaries
            
        Returns:
            List of LlamaIndex documents
        """
        try:
            documents = []
            
            for i, text in enumerate(texts):
                # Create metadata
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                metadata.update({
                    'source_index': i,
                    'text_length': len(text)
                })
                
                # Create document
                doc = Document(
                    text=text,
                    metadata=metadata
                )
                documents.append(doc)
            
            logger.info(f"Created {len(documents)} LlamaIndex documents from {len(texts)} texts")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to create LlamaIndex documents: {e}")
            raise
    
    def create_index_from_documents(self, 
                                  documents: List[Document],
                                  storage_path: Optional[str] = None) -> VectorStoreIndex:
        """
        Create LlamaIndex from documents.
        
        Args:
            documents: List of LlamaIndex documents
            storage_path: Path to save index
            
        Returns:
            LlamaIndex vector store index
        """
        try:
            if not self.embedding_model:
                self.initialize_components()
            
            # Create index
            index = VectorStoreIndex.from_documents(
                documents=documents,
                embed_model=self.embedding_model
            )
            
            # Save index if storage path provided
            if storage_path:
                index.storage_context.persist(persist_dir=storage_path)
                logger.info(f"Saved LlamaIndex to {storage_path}")
            
            self.index = index
            logger.info(f"Created LlamaIndex with {len(documents)} documents")
            return index
            
        except Exception as e:
            logger.error(f"Failed to create LlamaIndex: {e}")
            raise
    
    def load_index_from_storage(self, storage_path: str) -> VectorStoreIndex:
        """
        Load LlamaIndex from storage.
        
        Args:
            storage_path: Path to storage directory
            
        Returns:
            LlamaIndex vector store index
        """
        try:
            if not self.embedding_model:
                self.initialize_components()
            
            # Load index
            storage_context = StorageContext.from_defaults(persist_dir=storage_path)
            index = load_index_from_storage(storage_context)
            
            self.index = index
            logger.info(f"Loaded LlamaIndex from {storage_path}")
            return index
            
        except Exception as e:
            logger.error(f"Failed to load LlamaIndex: {e}")
            raise
    
    def create_retriever(self, 
                        similarity_top_k: int = 5,
                        similarity_cutoff: float = 0.0) -> VectorIndexRetriever:
        """
        Create retriever from index.
        
        Args:
            similarity_top_k: Number of top similar documents
            similarity_cutoff: Similarity cutoff threshold
            
        Returns:
            Vector index retriever
        """
        try:
            if not self.index:
                raise ValueError("Index not initialized")
            
            # Create retriever
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=similarity_top_k,
                similarity_cutoff=similarity_cutoff
            )
            
            self.retriever = retriever
            logger.info(f"Created retriever with top_k={similarity_top_k}, cutoff={similarity_cutoff}")
            return retriever
            
        except Exception as e:
            logger.error(f"Failed to create retriever: {e}")
            raise
    
    def create_query_engine(self, 
                          retriever: Optional[VectorIndexRetriever] = None,
                          response_mode: str = "compact",
                          similarity_cutoff: float = 0.0) -> RetrieverQueryEngine:
        """
        Create query engine from retriever.
        
        Args:
            retriever: Retriever to use
            response_mode: Response mode (compact, tree_summarize, simple_summarize)
            similarity_cutoff: Similarity cutoff threshold
            
        Returns:
            Retriever query engine
        """
        try:
            if not retriever:
                retriever = self.retriever
            
            if not retriever:
                raise ValueError("Retriever not initialized")
            
            # Create postprocessor
            postprocessor = SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
            
            # Create query engine
            query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                response_mode=response_mode,
                node_postprocessors=[postprocessor]
            )
            
            self.query_engine = query_engine
            logger.info(f"Created query engine with response_mode={response_mode}")
            return query_engine
            
        except Exception as e:
            logger.error(f"Failed to create query engine: {e}")
            raise
    
    def query_index(self, 
                   query: str,
                   similarity_top_k: int = 5,
                   similarity_cutoff: float = 0.0) -> List[Dict[str, Any]]:
        """
        Query the index directly.
        
        Args:
            query: Query text
            similarity_top_k: Number of top similar documents
            similarity_cutoff: Similarity cutoff threshold
            
        Returns:
            List of query results
        """
        try:
            if not self.index:
                raise ValueError("Index not initialized")
            
            # Create retriever
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=similarity_top_k,
                similarity_cutoff=similarity_cutoff
            )
            
            # Retrieve documents
            nodes = retriever.retrieve(query)
            
            # Format results
            results = []
            for node in nodes:
                result = {
                    'text': node.text,
                    'score': node.score,
                    'metadata': node.metadata,
                    'node_id': node.node_id
                }
                results.append(result)
            
            logger.info(f"Retrieved {len(results)} nodes for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to query index: {e}")
            raise
    
    def query_with_engine(self, 
                        query: str,
                        similarity_cutoff: float = 0.0) -> str:
        """
        Query using the query engine.
        
        Args:
            query: Query text
            similarity_cutoff: Similarity cutoff threshold
            
        Returns:
            Query response
        """
        try:
            if not self.query_engine:
                # Create query engine if not exists
                self.create_query_engine(similarity_cutoff=similarity_cutoff)
            
            # Query using engine
            response = self.query_engine.query(query)
            
            logger.info(f"Generated response for query: {query}")
            return str(response)
            
        except Exception as e:
            logger.error(f"Failed to query with engine: {e}")
            raise
    
    def add_documents_to_index(self, documents: List[Document]) -> None:
        """
        Add documents to existing index.
        
        Args:
            documents: List of documents to add
        """
        try:
            if not self.index:
                raise ValueError("Index not initialized")
            
            # Add documents to index
            for doc in documents:
                self.index.insert(doc)
            
            logger.info(f"Added {len(documents)} documents to index")
            
        except Exception as e:
            logger.error(f"Failed to add documents to index: {e}")
            raise
    
    def delete_documents_from_index(self, node_ids: List[str]) -> None:
        """
        Delete documents from index.
        
        Args:
            node_ids: List of node IDs to delete
        """
        try:
            if not self.index:
                raise ValueError("Index not initialized")
            
            # Delete documents from index
            for node_id in node_ids:
                self.index.delete_ref_doc(node_id)
            
            logger.info(f"Deleted {len(node_ids)} documents from index")
            
        except Exception as e:
            logger.error(f"Failed to delete documents from index: {e}")
            raise
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """
        Get index statistics.
        
        Returns:
            Index statistics
        """
        try:
            if not self.index:
                return {'error': 'Index not initialized'}
            
            # Get index information
            stats = {
                'total_documents': len(self.index.docstore.docs),
                'total_nodes': len(self.index.docstore.nodes),
                'embedding_model': self.embedding_model_name,
                'storage_path': self.storage_path
            }
            
            logger.info(f"Index statistics: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get index statistics: {e}")
            raise
    
    def evaluate_retriever(self, 
                         test_queries: List[str],
                         ground_truth: List[List[str]]) -> Dict[str, float]:
        """
        Evaluate retriever performance.
        
        Args:
            test_queries: List of test queries
            ground_truth: Ground truth results for each query
            
        Returns:
            Evaluation metrics
        """
        try:
            if not self.retriever:
                raise ValueError("Retriever not initialized")
            
            # Create evaluator
            evaluator = RetrieverEvaluator.from_metric_names(
                ["mrr", "hit_rate"]
            )
            
            # Evaluate retriever
            results = evaluator.evaluate(
                retriever=self.retriever,
                queries=test_queries,
                ground_truth=ground_truth
            )
            
            # Extract metrics
            metrics = {
                'mrr': results.get('mrr', 0.0),
                'hit_rate': results.get('hit_rate', 0.0)
            }
            
            logger.info(f"Retriever evaluation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to evaluate retriever: {e}")
            raise
    
    def evaluate_query_engine(self, 
                            test_queries: List[str],
                            ground_truth_responses: List[str]) -> Dict[str, float]:
        """
        Evaluate query engine performance.
        
        Args:
            test_queries: List of test queries
            ground_truth_responses: Ground truth responses
            
        Returns:
            Evaluation metrics
        """
        try:
            if not self.query_engine:
                raise ValueError("Query engine not initialized")
            
            # Create evaluator
            evaluator = QueryResponseEvaluator.from_metric_names(
                ["faithfulness", "relevance"]
            )
            
            # Evaluate query engine
            results = evaluator.evaluate(
                query_engine=self.query_engine,
                queries=test_queries,
                ground_truth=ground_truth_responses
            )
            
            # Extract metrics
            metrics = {
                'faithfulness': results.get('faithfulness', 0.0),
                'relevance': results.get('relevance', 0.0)
            }
            
            logger.info(f"Query engine evaluation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to evaluate query engine: {e}")
            raise
    
    def create_hybrid_retriever(self, 
                              keyword_retriever,
                              vector_retriever,
                              weights: List[float] = [0.5, 0.5]) -> 'HybridRetriever':
        """
        Create hybrid retriever combining keyword and vector search.
        
        Args:
            keyword_retriever: Keyword-based retriever
            vector_retriever: Vector-based retriever
            weights: Weights for each retriever
            
        Returns:
            Hybrid retriever
        """
        try:
            # Create hybrid retriever
            hybrid_retriever = HybridRetriever(
                retrievers=[keyword_retriever, vector_retriever],
                weights=weights
            )
            
            logger.info("Created hybrid retriever")
            return hybrid_retriever
            
        except Exception as e:
            logger.error(f"Failed to create hybrid retriever: {e}")
            raise
    
    def create_multi_query_retriever(self, 
                                   llm,
                                   similarity_top_k: int = 5) -> 'MultiQueryRetriever':
        """
        Create multi-query retriever.
        
        Args:
            llm: Language model for query generation
            similarity_top_k: Number of top similar documents
            
        Returns:
            Multi-query retriever
        """
        try:
            if not self.retriever:
                raise ValueError("Retriever not initialized")
            
            # Create multi-query retriever
            multi_query_retriever = MultiQueryRetriever.from_defaults(
                retriever=self.retriever,
                llm=llm,
                similarity_top_k=similarity_top_k
            )
            
            logger.info("Created multi-query retriever")
            return multi_query_retriever
            
        except Exception as e:
            logger.error(f"Failed to create multi-query retriever: {e}")
            raise
    
    def save_index(self, storage_path: str) -> None:
        """
        Save index to storage.
        
        Args:
            storage_path: Path to save index
        """
        try:
            if not self.index:
                raise ValueError("Index not initialized")
            
            # Save index
            self.index.storage_context.persist(persist_dir=storage_path)
            
            logger.info(f"Saved index to {storage_path}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise
    
    def load_index(self, storage_path: str) -> None:
        """
        Load index from storage.
        
        Args:
            storage_path: Path to load index from
        """
        try:
            # Load index
            self.index = self.load_index_from_storage(storage_path)
            
            logger.info(f"Loaded index from {storage_path}")
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise

class HybridRetriever:
    """Hybrid retriever combining multiple retrievers."""
    
    def __init__(self, retrievers: List, weights: List[float]):
        self.retrievers = retrievers
        self.weights = weights
    
    def retrieve(self, query: str) -> List:
        """Retrieve documents using hybrid approach."""
        # Implementation would combine results from multiple retrievers
        # This is a placeholder for the actual implementation
        pass

class MultiQueryRetriever:
    """Multi-query retriever for generating multiple queries."""
    
    def __init__(self, retriever, llm, similarity_top_k: int = 5):
        self.retriever = retriever
        self.llm = llm
        self.similarity_top_k = similarity_top_k
    
    def retrieve(self, query: str) -> List:
        """Retrieve documents using multiple queries."""
        # Implementation would generate multiple queries and combine results
        # This is a placeholder for the actual implementation
        pass
