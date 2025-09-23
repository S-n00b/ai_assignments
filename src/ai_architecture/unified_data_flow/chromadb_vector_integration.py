"""
ChromaDB Vector Integration

Provides ChromaDB integration for vector embeddings and RAG workflows:
- Document embedding and storage
- Vector similarity search
- RAG retrieval workflows
- Collection management
- Real-time synchronization
"""

import json
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import chromadb
from chromadb.config import Settings
import numpy as np
from sentence_transformers import SentenceTransformer

class ChromaDBVectorIntegration:
    """ChromaDB integration for vector operations"""
    
    def __init__(self, persist_directory: str = "chroma_data", host: str = "localhost", port: int = 8000):
        self.persist_directory = persist_directory
        self.host = host
        self.port = port
        self.client = None
        self.embedding_model = None
        self.collections = {}
        
    def initialize(self):
        """Initialize ChromaDB client and embedding model"""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    chroma_api_impl="chromadb.api.fastapi.FastAPI",
                    chroma_server_host=self.host,
                    chroma_server_http_port=self.port
                )
            )
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            print(f"ChromaDB initialized at {self.host}:{self.port}")
            return True
            
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            return False
    
    def create_collection(self, collection_name: str, metadata: Optional[Dict] = None) -> bool:
        """Create a new collection"""
        try:
            if metadata is None:
                metadata = {"description": f"Collection for {collection_name}"}
            
            collection = self.client.create_collection(
                name=collection_name,
                metadata=metadata
            )
            
            self.collections[collection_name] = collection
            print(f"Collection '{collection_name}' created successfully")
            return True
            
        except Exception as e:
            print(f"Error creating collection '{collection_name}': {e}")
            return False
    
    def get_collection(self, collection_name: str):
        """Get existing collection"""
        try:
            if collection_name in self.collections:
                return self.collections[collection_name]
            
            collection = self.client.get_collection(name=collection_name)
            self.collections[collection_name] = collection
            return collection
            
        except Exception as e:
            print(f"Error getting collection '{collection_name}': {e}")
            return None
    
    def add_documents(self, collection_name: str, documents: List[str], 
                     metadatas: List[Dict] = None, ids: List[str] = None) -> bool:
        """Add documents to collection"""
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                return False
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(documents).tolist()
            
            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in documents]
            
            # Generate metadata if not provided
            if metadatas is None:
                metadatas = [{"source": "generated", "timestamp": datetime.now().isoformat()} 
                           for _ in documents]
            
            # Add to collection
            collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"Added {len(documents)} documents to collection '{collection_name}'")
            return True
            
        except Exception as e:
            print(f"Error adding documents to collection '{collection_name}': {e}")
            return False
    
    def search_similar(self, collection_name: str, query: str, n_results: int = 5, 
                      where: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Search collection
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        'document': doc,
                        'distance': results['distances'][0][i] if results['distances'] else None,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'id': results['ids'][0][i] if results['ids'] else None
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching collection '{collection_name}': {e}")
            return []
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get collection information"""
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                return {}
            
            count = collection.count()
            
            return {
                'name': collection_name,
                'count': count,
                'metadata': collection.metadata,
                'created_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error getting collection info for '{collection_name}': {e}")
            return {}
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete collection"""
        try:
            self.client.delete_collection(name=collection_name)
            if collection_name in self.collections:
                del self.collections[collection_name]
            
            print(f"Collection '{collection_name}' deleted successfully")
            return True
            
        except Exception as e:
            print(f"Error deleting collection '{collection_name}': {e}")
            return False
    
    def update_document(self, collection_name: str, document_id: str, 
                       document: str, metadata: Optional[Dict] = None) -> bool:
        """Update document in collection"""
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                return False
            
            # Generate new embedding
            embedding = self.embedding_model.encode([document]).tolist()[0]
            
            # Update document
            collection.update(
                ids=[document_id],
                documents=[document],
                embeddings=[embedding],
                metadatas=[metadata] if metadata else None
            )
            
            print(f"Document '{document_id}' updated in collection '{collection_name}'")
            return True
            
        except Exception as e:
            print(f"Error updating document '{document_id}': {e}")
            return False
    
    def delete_document(self, collection_name: str, document_id: str) -> bool:
        """Delete document from collection"""
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                return False
            
            collection.delete(ids=[document_id])
            
            print(f"Document '{document_id}' deleted from collection '{collection_name}'")
            return True
            
        except Exception as e:
            print(f"Error deleting document '{document_id}': {e}")
            return False
    
    def list_collections(self) -> List[str]:
        """List all collections"""
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            print(f"Error listing collections: {e}")
            return []
    
    def populate_from_enterprise_data(self, enterprise_data: Dict[str, Any]) -> bool:
        """Populate ChromaDB with enterprise data"""
        try:
            # Create collections for different data types
            collections_to_create = [
                "device_specifications",
                "user_behavior", 
                "business_processes",
                "support_knowledge",
                "technical_documentation"
            ]
            
            for collection_name in collections_to_create:
                self.create_collection(collection_name)
            
            # Populate device specifications
            if "device_specifications" in enterprise_data:
                self._populate_device_specifications(enterprise_data["device_specifications"])
            
            # Populate support knowledge
            if "support_knowledge_base" in enterprise_data:
                self._populate_support_knowledge(enterprise_data["support_knowledge_base"])
            
            # Populate business processes
            if "business_processes" in enterprise_data:
                self._populate_business_processes(enterprise_data["business_processes"])
            
            print("Enterprise data populated in ChromaDB successfully")
            return True
            
        except Exception as e:
            print(f"Error populating enterprise data: {e}")
            return False
    
    def _populate_device_specifications(self, devices: List[Dict[str, Any]]):
        """Populate device specifications collection"""
        documents = []
        metadatas = []
        ids = []
        
        for device in devices:
            # Create document text
            doc_text = f"""
            Device: {device['model']}
            Series: {device['series']}
            Category: {device['category']}
            Processor: {device['processor']}
            Memory: {device['memory']}
            Storage: {device['storage']}
            Display: {device['display']}
            Connectivity: {', '.join(device['connectivity'])}
            OS: {device['os']}
            Price Range: {device['price_range']}
            Target Market: {device['target_market']}
            Support Level: {device['support_level']}
            """
            
            documents.append(doc_text.strip())
            metadatas.append({
                "type": "device_specification",
                "model": device['model'],
                "series": device['series'],
                "category": device['category'],
                "support_level": device['support_level']
            })
            ids.append(f"device_{device['model'].replace(' ', '_').lower()}")
        
        self.add_documents("device_specifications", documents, metadatas, ids)
    
    def _populate_support_knowledge(self, knowledge_entries: List[Dict[str, Any]]):
        """Populate support knowledge collection"""
        documents = []
        metadatas = []
        ids = []
        
        for entry in knowledge_entries:
            # Create document text
            doc_text = f"""
            Device: {entry['device_model']}
            Issue: {entry['issue_category']}
            Symptoms: {', '.join(entry['symptoms'])}
            Troubleshooting Steps: {', '.join(entry['troubleshooting_steps'])}
            Resolution: {entry['resolution']}
            Support Level: {entry['support_level']}
            """
            
            documents.append(doc_text.strip())
            metadatas.append({
                "type": "support_knowledge",
                "device_model": entry['device_model'],
                "issue_category": entry['issue_category'],
                "support_level": entry['support_level']
            })
            ids.append(f"support_{entry['device_model'].replace(' ', '_').lower()}_{entry['issue_category'].replace(' ', '_').lower()}")
        
        self.add_documents("support_knowledge", documents, metadatas, ids)
    
    def _populate_business_processes(self, processes: List[Dict[str, Any]]):
        """Populate business processes collection"""
        documents = []
        metadatas = []
        ids = []
        
        for process in processes:
            # Create document text
            steps_text = "\n".join([f"Step {i+1}: {step['step']}" for i, step in enumerate(process['steps'])])
            
            doc_text = f"""
            Process: {process['process_name']}
            Department: {process['department']}
            Type: {process['process_type']}
            Description: {process['description']}
            Steps:
            {steps_text}
            Frequency: {process['frequency']}
            Duration: {process['average_duration']} hours
            Success Rate: {process['success_rate']}
            """
            
            documents.append(doc_text.strip())
            metadatas.append({
                "type": "business_process",
                "process_name": process['process_name'],
                "department": process['department'],
                "process_type": process['process_type'],
                "frequency": process['frequency']
            })
            ids.append(f"process_{process['process_id']}")
        
        self.add_documents("business_processes", documents, metadatas, ids)
    
    def get_rag_context(self, query: str, collection_names: List[str] = None, 
                       max_results: int = 10) -> List[Dict[str, Any]]:
        """Get RAG context from multiple collections"""
        if collection_names is None:
            collection_names = ["device_specifications", "support_knowledge", "business_processes"]
        
        all_results = []
        
        for collection_name in collection_names:
            results = self.search_similar(collection_name, query, n_results=max_results//len(collection_names))
            for result in results:
                result['collection'] = collection_name
                all_results.append(result)
        
        # Sort by distance and return top results
        all_results.sort(key=lambda x: x.get('distance', 0))
        return all_results[:max_results]

if __name__ == "__main__":
    # Test ChromaDB integration
    chroma = ChromaDBVectorIntegration()
    if chroma.initialize():
        print("ChromaDB integration test successful")
    else:
        print("ChromaDB integration test failed")
