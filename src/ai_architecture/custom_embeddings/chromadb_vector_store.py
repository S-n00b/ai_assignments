"""
ChromaDB Vector Store Integration

This module provides ChromaDB integration for storing and retrieving
custom embeddings with efficient vector operations.
"""

import chromadb
from chromadb.config import Settings
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json
import uuid

logger = logging.getLogger(__name__)

class ChromaDBVectorStore:
    """
    ChromaDB vector store for custom embeddings.
    
    Provides efficient storage and retrieval of custom embeddings
    with ChromaDB integration.
    """
    
    def __init__(self, 
                 persist_directory: str = "./chroma_data",
                 collection_name: str = "lenovo_embeddings",
                 host: str = "localhost",
                 port: int = 8000):
        """
        Initialize ChromaDB vector store.
        
        Args:
            persist_directory: Directory to persist data
            collection_name: Name of the collection
            host: ChromaDB host
            port: ChromaDB port
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.host = host
        self.port = port
        
        # Initialize ChromaDB client
        self.client = None
        self.collection = None
        
    def initialize_client(self) -> None:
        """Initialize ChromaDB client."""
        try:
            # Create settings
            settings = Settings(
                persist_directory=self.persist_directory,
                chroma_server_host=self.host,
                chroma_server_http_port=self.port
            )
            
            # Initialize client
            self.client = chromadb.PersistentClient(settings=settings)
            
            logger.info(f"Initialized ChromaDB client at {self.persist_directory}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise
    
    def create_collection(self, 
                         collection_name: Optional[str] = None,
                         metadata: Optional[Dict] = None) -> str:
        """
        Create or get collection.
        
        Args:
            collection_name: Name of the collection
            metadata: Collection metadata
            
        Returns:
            Collection name
        """
        try:
            if collection_name is None:
                collection_name = self.collection_name
            
            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata=metadata or {}
            )
            
            logger.info(f"Created/retrieved collection: {collection_name}")
            return collection_name
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    def add_embeddings(self, 
                      embeddings: np.ndarray,
                      documents: List[str],
                      metadatas: Optional[List[Dict]] = None,
                      ids: Optional[List[str]] = None) -> List[str]:
        """
        Add embeddings to collection.
        
        Args:
            embeddings: Array of embeddings
            documents: List of documents
            metadatas: List of metadata dictionaries
            ids: List of document IDs
            
        Returns:
            List of added IDs
        """
        try:
            if self.collection is None:
                raise ValueError("Collection not initialized")
            
            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(len(documents))]
            
            # Prepare metadatas
            if metadatas is None:
                metadatas = [{} for _ in range(len(documents))]
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} embeddings to collection")
            return ids
            
        except Exception as e:
            logger.error(f"Failed to add embeddings: {e}")
            raise
    
    def search_embeddings(self, 
                         query_embeddings: np.ndarray,
                         n_results: int = 5,
                         where: Optional[Dict] = None,
                         include: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Search embeddings in collection.
        
        Args:
            query_embeddings: Query embeddings
            n_results: Number of results to return
            where: Metadata filter
            include: Fields to include in results
            
        Returns:
            Search results
        """
        try:
            if self.collection is None:
                raise ValueError("Collection not initialized")
            
            # Set default include fields
            if include is None:
                include = ["documents", "metadatas", "distances"]
            
            # Search collection
            results = self.collection.query(
                query_embeddings=query_embeddings.tolist(),
                n_results=n_results,
                where=where,
                include=include
            )
            
            logger.info(f"Found {len(results['ids'][0])} results for query")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search embeddings: {e}")
            raise
    
    def update_embeddings(self, 
                         ids: List[str],
                         embeddings: Optional[np.ndarray] = None,
                         documents: Optional[List[str]] = None,
                         metadatas: Optional[List[Dict]] = None) -> None:
        """
        Update embeddings in collection.
        
        Args:
            ids: List of document IDs to update
            embeddings: New embeddings
            documents: New documents
            metadatas: New metadatas
        """
        try:
            if self.collection is None:
                raise ValueError("Collection not initialized")
            
            # Update collection
            self.collection.update(
                ids=ids,
                embeddings=embeddings.tolist() if embeddings is not None else None,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Updated {len(ids)} embeddings")
            
        except Exception as e:
            logger.error(f"Failed to update embeddings: {e}")
            raise
    
    def delete_embeddings(self, ids: List[str]) -> None:
        """
        Delete embeddings from collection.
        
        Args:
            ids: List of document IDs to delete
        """
        try:
            if self.collection is None:
                raise ValueError("Collection not initialized")
            
            # Delete from collection
            self.collection.delete(ids=ids)
            
            logger.info(f"Deleted {len(ids)} embeddings")
            
        except Exception as e:
            logger.error(f"Failed to delete embeddings: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get collection information.
        
        Returns:
            Collection information
        """
        try:
            if self.collection is None:
                raise ValueError("Collection not initialized")
            
            # Get collection count
            count = self.collection.count()
            
            # Get collection metadata
            metadata = self.collection.metadata
            
            info = {
                'collection_name': self.collection.name,
                'count': count,
                'metadata': metadata
            }
            
            logger.info(f"Collection info: {info}")
            return info
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            raise
    
    def create_index(self, index_type: str = "hnsw") -> None:
        """
        Create index for collection.
        
        Args:
            index_type: Type of index to create
        """
        try:
            if self.collection is None:
                raise ValueError("Collection not initialized")
            
            # Create index (ChromaDB handles this automatically)
            logger.info(f"Index created for collection: {index_type}")
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise
    
    def backup_collection(self, backup_path: str) -> str:
        """
        Backup collection to file.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            Path to backup file
        """
        try:
            if self.collection is None:
                raise ValueError("Collection not initialized")
            
            # Get all data from collection
            all_data = self.collection.get()
            
            # Create backup data
            backup_data = {
                'collection_name': self.collection.name,
                'metadata': self.collection.metadata,
                'data': all_data,
                'backup_timestamp': str(uuid.uuid4())
            }
            
            # Save backup
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Collection backed up to {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Failed to backup collection: {e}")
            raise
    
    def restore_collection(self, backup_path: str) -> None:
        """
        Restore collection from backup.
        
        Args:
            backup_path: Path to backup file
        """
        try:
            # Load backup data
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            # Create new collection
            collection_name = backup_data['collection_name']
            metadata = backup_data['metadata']
            
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata=metadata
            )
            
            # Restore data
            data = backup_data['data']
            if data['ids']:
                self.collection.add(
                    embeddings=data['embeddings'],
                    documents=data['documents'],
                    metadatas=data['metadatas'],
                    ids=data['ids']
                )
            
            logger.info(f"Collection restored from {backup_path}")
            
        except Exception as e:
            logger.error(f"Failed to restore collection: {e}")
            raise
    
    def create_hybrid_search(self, 
                            query_embeddings: np.ndarray,
                            query_text: str,
                            n_results: int = 5,
                            alpha: float = 0.7) -> Dict[str, Any]:
        """
        Create hybrid search combining vector and text search.
        
        Args:
            query_embeddings: Query embeddings
            query_text: Query text
            n_results: Number of results
            alpha: Weight for vector search (1-alpha for text search)
            
        Returns:
            Hybrid search results
        """
        try:
            if self.collection is None:
                raise ValueError("Collection not initialized")
            
            # Vector search
            vector_results = self.search_embeddings(
                query_embeddings=query_embeddings,
                n_results=n_results
            )
            
            # Text search (simple keyword matching)
            text_results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            
            # Combine results
            combined_results = {
                'vector_results': vector_results,
                'text_results': text_results,
                'alpha': alpha,
                'query_text': query_text
            }
            
            logger.info(f"Created hybrid search for query: {query_text}")
            return combined_results
            
        except Exception as e:
            logger.error(f"Failed to create hybrid search: {e}")
            raise
    
    def get_similarity_matrix(self, 
                             document_ids: List[str]) -> np.ndarray:
        """
        Get similarity matrix for documents.
        
        Args:
            document_ids: List of document IDs
            
        Returns:
            Similarity matrix
        """
        try:
            if self.collection is None:
                raise ValueError("Collection not initialized")
            
            # Get embeddings for documents
            results = self.collection.get(ids=document_ids, include=['embeddings'])
            embeddings = np.array(results['embeddings'])
            
            # Calculate similarity matrix
            similarity_matrix = np.dot(embeddings, embeddings.T)
            
            logger.info(f"Created similarity matrix for {len(document_ids)} documents")
            return similarity_matrix
            
        except Exception as e:
            logger.error(f"Failed to get similarity matrix: {e}")
            raise
    
    def cluster_embeddings(self, 
                          n_clusters: int = 5,
                          method: str = "kmeans") -> Dict[str, Any]:
        """
        Cluster embeddings in collection.
        
        Args:
            n_clusters: Number of clusters
            method: Clustering method
            
        Returns:
            Clustering results
        """
        try:
            if self.collection is None:
                raise ValueError("Collection not initialized")
            
            # Get all embeddings
            all_data = self.collection.get(include=['embeddings'])
            embeddings = np.array(all_data['embeddings'])
            
            if method == "kmeans":
                from sklearn.cluster import KMeans
                clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = clusterer.fit_predict(embeddings)
            else:
                raise ValueError(f"Unsupported clustering method: {method}")
            
            # Create clustering results
            clustering_results = {
                'method': method,
                'n_clusters': n_clusters,
                'cluster_labels': cluster_labels.tolist(),
                'document_ids': all_data['ids'],
                'cluster_centers': clusterer.cluster_centers_.tolist()
            }
            
            logger.info(f"Clustered embeddings into {n_clusters} clusters")
            return clustering_results
            
        except Exception as e:
            logger.error(f"Failed to cluster embeddings: {e}")
            raise
