"""
LangChain FAISS Integration for Hybrid RAG

This module provides LangChain integration with FAISS for efficient
vector similarity search and retrieval in hybrid RAG workflows.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json
import numpy as np
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.retrievers import VectorStoreRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.ensemble import EnsembleRetriever

logger = logging.getLogger(__name__)

class LangChainFAISSIntegration:
    """
    LangChain FAISS integration for hybrid RAG.
    
    Provides LangChain integration with FAISS for efficient vector similarity search
    and retrieval in hybrid RAG workflows.
    """
    
    def __init__(self, 
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 faiss_index_path: Optional[str] = None,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize LangChain FAISS integration.
        
        Args:
            embedding_model_name: Name of the embedding model
            faiss_index_path: Path to FAISS index
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.embedding_model_name = embedding_model_name
        self.faiss_index_path = faiss_index_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.embeddings = None
        self.vectorstore = None
        self.text_splitter = None
        
    def initialize_components(self) -> None:
        """Initialize LangChain components."""
        try:
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': 'cpu'}
            )
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            logger.info("Initialized LangChain components")
            
        except Exception as e:
            logger.error(f"Failed to initialize LangChain components: {e}")
            raise
    
    def create_documents_from_texts(self, 
                                  texts: List[str],
                                  metadatas: Optional[List[Dict]] = None) -> List[Document]:
        """
        Create LangChain documents from texts.
        
        Args:
            texts: List of texts
            metadatas: List of metadata dictionaries
            
        Returns:
            List of LangChain documents
        """
        try:
            documents = []
            
            for i, text in enumerate(texts):
                # Split text into chunks
                chunks = self.text_splitter.split_text(text)
                
                for j, chunk in enumerate(chunks):
                    # Create metadata
                    metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                    metadata.update({
                        'chunk_index': j,
                        'total_chunks': len(chunks),
                        'source_index': i
                    })
                    
                    # Create document
                    doc = Document(
                        page_content=chunk,
                        metadata=metadata
                    )
                    documents.append(doc)
            
            logger.info(f"Created {len(documents)} documents from {len(texts)} texts")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to create documents: {e}")
            raise
    
    def create_faiss_vectorstore(self, 
                               documents: List[Document],
                               index_path: Optional[str] = None) -> FAISS:
        """
        Create FAISS vectorstore from documents.
        
        Args:
            documents: List of LangChain documents
            index_path: Path to save FAISS index
            
        Returns:
            FAISS vectorstore
        """
        try:
            if not self.embeddings:
                self.initialize_components()
            
            # Create FAISS vectorstore
            vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            # Save index if path provided
            if index_path:
                vectorstore.save_local(index_path)
                logger.info(f"Saved FAISS index to {index_path}")
            
            self.vectorstore = vectorstore
            logger.info(f"Created FAISS vectorstore with {len(documents)} documents")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Failed to create FAISS vectorstore: {e}")
            raise
    
    def load_faiss_vectorstore(self, index_path: str) -> FAISS:
        """
        Load FAISS vectorstore from index.
        
        Args:
            index_path: Path to FAISS index
            
        Returns:
            FAISS vectorstore
        """
        try:
            if not self.embeddings:
                self.initialize_components()
            
            # Load FAISS vectorstore
            vectorstore = FAISS.load_local(
                index_path,
                self.embeddings
            )
            
            self.vectorstore = vectorstore
            logger.info(f"Loaded FAISS vectorstore from {index_path}")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Failed to load FAISS vectorstore: {e}")
            raise
    
    def create_retriever(self, 
                        search_type: str = "similarity",
                        search_kwargs: Optional[Dict] = None) -> VectorStoreRetriever:
        """
        Create retriever from vectorstore.
        
        Args:
            search_type: Type of search (similarity, mmr, similarity_score_threshold)
            search_kwargs: Search parameters
            
        Returns:
            Vector store retriever
        """
        try:
            if not self.vectorstore:
                raise ValueError("Vectorstore not initialized")
            
            # Default search parameters
            if search_kwargs is None:
                search_kwargs = {"k": 5}
            
            # Create retriever
            retriever = self.vectorstore.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs
            )
            
            logger.info(f"Created retriever with search type: {search_type}")
            return retriever
            
        except Exception as e:
            logger.error(f"Failed to create retriever: {e}")
            raise
    
    def create_multi_query_retriever(self, 
                                   llm,
                                   search_type: str = "similarity",
                                   search_kwargs: Optional[Dict] = None) -> MultiQueryRetriever:
        """
        Create multi-query retriever.
        
        Args:
            llm: Language model for query generation
            search_type: Type of search
            search_kwargs: Search parameters
            
        Returns:
            Multi-query retriever
        """
        try:
            if not self.vectorstore:
                raise ValueError("Vectorstore not initialized")
            
            # Create base retriever
            base_retriever = self.create_retriever(search_type, search_kwargs)
            
            # Create multi-query retriever
            multi_query_retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=llm
            )
            
            logger.info("Created multi-query retriever")
            return multi_query_retriever
            
        except Exception as e:
            logger.error(f"Failed to create multi-query retriever: {e}")
            raise
    
    def create_ensemble_retriever(self, 
                               retrievers: List[VectorStoreRetriever],
                               weights: Optional[List[float]] = None) -> EnsembleRetriever:
        """
        Create ensemble retriever.
        
        Args:
            retrievers: List of retrievers
            weights: Weights for each retriever
            
        Returns:
            Ensemble retriever
        """
        try:
            # Create ensemble retriever
            ensemble_retriever = EnsembleRetriever(
                retrievers=retrievers,
                weights=weights
            )
            
            logger.info(f"Created ensemble retriever with {len(retrievers)} retrievers")
            return ensemble_retriever
            
        except Exception as e:
            logger.error(f"Failed to create ensemble retriever: {e}")
            raise
    
    def search_similar_documents(self, 
                               query: str,
                               k: int = 5,
                               filter: Optional[Dict] = None) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results
            filter: Metadata filter
            
        Returns:
            List of similar documents
        """
        try:
            if not self.vectorstore:
                raise ValueError("Vectorstore not initialized")
            
            # Search for similar documents
            if filter:
                docs = self.vectorstore.similarity_search(
                    query=query,
                    k=k,
                    filter=filter
                )
            else:
                docs = self.vectorstore.similarity_search(
                    query=query,
                    k=k
                )
            
            logger.info(f"Found {len(docs)} similar documents for query: {query}")
            return docs
            
        except Exception as e:
            logger.error(f"Failed to search similar documents: {e}")
            raise
    
    def search_with_scores(self, 
                          query: str,
                          k: int = 5,
                          filter: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """
        Search for similar documents with scores.
        
        Args:
            query: Search query
            k: Number of results
            filter: Metadata filter
            
        Returns:
            List of (document, score) tuples
        """
        try:
            if not self.vectorstore:
                raise ValueError("Vectorstore not initialized")
            
            # Search with scores
            if filter:
                docs_with_scores = self.vectorstore.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter
                )
            else:
                docs_with_scores = self.vectorstore.similarity_search_with_score(
                    query=query,
                    k=k
                )
            
            logger.info(f"Found {len(docs_with_scores)} documents with scores for query: {query}")
            return docs_with_scores
            
        except Exception as e:
            logger.error(f"Failed to search with scores: {e}")
            raise
    
    def search_mmr(self, 
                  query: str,
                  k: int = 5,
                  fetch_k: int = 20,
                  lambda_mult: float = 0.5) -> List[Document]:
        """
        Search using Maximum Marginal Relevance (MMR).
        
        Args:
            query: Search query
            k: Number of results
            fetch_k: Number of documents to fetch
            lambda_mult: Lambda parameter for MMR
            
        Returns:
            List of documents
        """
        try:
            if not self.vectorstore:
                raise ValueError("Vectorstore not initialized")
            
            # Search using MMR
            docs = self.vectorstore.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult
            )
            
            logger.info(f"Found {len(docs)} documents using MMR for query: {query}")
            return docs
            
        except Exception as e:
            logger.error(f"Failed to search with MMR: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to vectorstore.
        
        Args:
            documents: List of documents to add
        """
        try:
            if not self.vectorstore:
                raise ValueError("Vectorstore not initialized")
            
            # Add documents
            self.vectorstore.add_documents(documents)
            
            logger.info(f"Added {len(documents)} documents to vectorstore")
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete documents from vectorstore.
        
        Args:
            ids: List of document IDs to delete
        """
        try:
            if not self.vectorstore:
                raise ValueError("Vectorstore not initialized")
            
            # Delete documents
            self.vectorstore.delete(ids)
            
            logger.info(f"Deleted {len(ids)} documents from vectorstore")
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise
    
    def get_vectorstore_info(self) -> Dict[str, Any]:
        """
        Get vectorstore information.
        
        Returns:
            Vectorstore information
        """
        try:
            if not self.vectorstore:
                return {'error': 'Vectorstore not initialized'}
            
            # Get index information
            index = self.vectorstore.index
            info = {
                'index_type': type(index).__name__,
                'index_size': index.ntotal if hasattr(index, 'ntotal') else 0,
                'embedding_dimension': index.d if hasattr(index, 'd') else 0,
                'is_trained': index.is_trained if hasattr(index, 'is_trained') else False
            }
            
            logger.info(f"Vectorstore info: {info}")
            return info
            
        except Exception as e:
            logger.error(f"Failed to get vectorstore info: {e}")
            raise
    
    def create_hybrid_retriever(self, 
                              keyword_retriever,
                              vector_retriever,
                              weights: List[float] = [0.5, 0.5]) -> EnsembleRetriever:
        """
        Create hybrid retriever combining keyword and vector search.
        
        Args:
            keyword_retriever: Keyword-based retriever
            vector_retriever: Vector-based retriever
            weights: Weights for each retriever
            
        Returns:
            Hybrid ensemble retriever
        """
        try:
            # Create ensemble retriever
            hybrid_retriever = EnsembleRetriever(
                retrievers=[keyword_retriever, vector_retriever],
                weights=weights
            )
            
            logger.info("Created hybrid retriever")
            return hybrid_retriever
            
        except Exception as e:
            logger.error(f"Failed to create hybrid retriever: {e}")
            raise
    
    def evaluate_retriever(self, 
                         retriever: VectorStoreRetriever,
                         test_queries: List[str],
                         ground_truth: List[List[str]]) -> Dict[str, float]:
        """
        Evaluate retriever performance.
        
        Args:
            retriever: Retriever to evaluate
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
                # Get retrieved documents
                retrieved_docs = retriever.get_relevant_documents(query)
                retrieved_ids = [doc.metadata.get('id', '') for doc in retrieved_docs]
                
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
                'f1_score': avg_f1
            }
            
            logger.info(f"Retriever evaluation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to evaluate retriever: {e}")
            raise
