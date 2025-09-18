"""
Advanced RAG System Module

This module implements a sophisticated Retrieval-Augmented Generation (RAG) system
for enterprise knowledge management, including advanced document processing,
intelligent chunking strategies, multi-modal retrieval, and context engineering.

Key Features:
- Advanced document processing and chunking
- Multi-modal retrieval capabilities
- Intelligent context engineering
- Enterprise knowledge management
- Performance optimization and caching
- Security and access control
"""

import json
import asyncio
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union, Iterator
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import numpy as np
from pathlib import Path
import re
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Document chunking strategies"""
    FIXED_SIZE = "fixed_size"
    SLIDING_WINDOW = "sliding_window"
    HIERARCHICAL = "hierarchical"
    TOPIC_BASED = "topic_based"
    SEMANTIC = "semantic"
    ADAPTIVE = "adaptive"


class RetrievalMethod(Enum):
    """Retrieval methods"""
    DENSE_RETRIEVAL = "dense_retrieval"
    SPARSE_RETRIEVAL = "sparse_retrieval"
    HYBRID_RETRIEVAL = "hybrid_retrieval"
    MULTI_MODAL = "multi_modal"
    CONTEXTUAL = "contextual"


class DocumentType(Enum):
    """Document types"""
    TEXT = "text"
    PDF = "pdf"
    MARKDOWN = "markdown"
    HTML = "html"
    CODE = "code"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    STRUCTURED = "structured"


@dataclass
class DocumentMetadata:
    """Document metadata structure"""
    document_id: str
    title: str
    document_type: DocumentType
    source: str
    created_at: datetime
    modified_at: datetime
    author: str
    language: str
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    permissions: Dict[str, List[str]] = field(default_factory=dict)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    """Document chunk structure"""
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    embedding: Optional[np.ndarray] = None
    keywords: List[str] = field(default_factory=list)
    semantic_density: float = 0.0
    parent_chunk: Optional[str] = None
    child_chunks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Retrieval result structure"""
    chunk_id: str
    document_id: str
    content: str
    relevance_score: float
    retrieval_method: RetrievalMethod
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryContext:
    """Query context for retrieval"""
    query: str
    user_id: str
    session_id: str
    context_history: List[Dict[str, Any]] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)


class RAGSystem:
    """
    Advanced Retrieval-Augmented Generation System for Enterprise Knowledge Management.
    
    This class provides comprehensive RAG capabilities including:
    - Advanced document processing and chunking
    - Multi-modal retrieval and ranking
    - Intelligent context engineering
    - Enterprise security and access control
    - Performance optimization and caching
    - Real-time knowledge updates
    
    The system is designed for enterprise-scale knowledge management with
    sophisticated retrieval algorithms and context-aware generation.
    """
    
    def __init__(
        self,
        system_name: str = "Lenovo Enterprise RAG System",
        embedding_model: str = "sentence-transformers",
        vector_store: str = "faiss"
    ):
        """
        Initialize the RAG System.
        
        Args:
            system_name: Name of the RAG system
            embedding_model: Embedding model to use
            vector_store: Vector store backend
        """
        self.system_name = system_name
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        
        # Core components
        self.document_store = {}
        self.chunk_store = {}
        self.vector_index = None
        self.retrieval_cache = {}
        
        # Processing components
        self.text_splitter = None
        self.embedding_generator = None
        self.retrieval_engine = None
        
        # Configuration
        self.chunking_config = {
            "default_strategy": ChunkingStrategy.FIXED_SIZE,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "max_chunks_per_document": 100
        }
        
        self.retrieval_config = {
            "default_method": RetrievalMethod.HYBRID_RETRIEVAL,
            "max_results": 10,
            "similarity_threshold": 0.7,
            "rerank_results": True
        }
        
        # Initialize components
        self._initialize_components()
        
        logger.info(f"Initialized {system_name}")
    
    def _initialize_components(self):
        """Initialize RAG system components"""
        
        # Initialize text splitter (simulated)
        self.text_splitter = MockTextSplitter(
            chunk_size=self.chunking_config["chunk_size"],
            chunk_overlap=self.chunking_config["chunk_overlap"]
        )
        
        # Initialize embedding generator (simulated)
        self.embedding_generator = MockEmbeddingGenerator(
            model_name=self.embedding_model
        )
        
        # Initialize retrieval engine (simulated)
        self.retrieval_engine = MockRetrievalEngine(
            vector_store=self.vector_store
        )
        
        logger.info("RAG system components initialized")
    
    async def ingest_document(
        self,
        content: str,
        metadata: DocumentMetadata,
        chunking_strategy: ChunkingStrategy = None
    ) -> Dict[str, Any]:
        """
        Ingest a document into the RAG system.
        
        Args:
            content: Document content
            metadata: Document metadata
            chunking_strategy: Chunking strategy to use
            
        Returns:
            Ingestion result with chunk information
        """
        try:
            logger.info(f"Ingesting document {metadata.document_id}")
            
            # Store document metadata
            self.document_store[metadata.document_id] = {
                "metadata": asdict(metadata),
                "content": content,
                "ingested_at": datetime.now().isoformat()
            }
            
            # Chunk document
            strategy = chunking_strategy or self.chunking_config["default_strategy"]
            chunks = await self._chunk_document(content, metadata, strategy)
            
            # Generate embeddings
            chunks_with_embeddings = await self.generate_embeddings(chunks)
            
            # Store chunks
            for chunk in chunks_with_embeddings:
                self.chunk_store[chunk.chunk_id] = asdict(chunk)
            
            # Update vector index
            await self._update_vector_index(chunks_with_embeddings)
            
            logger.info(f"Successfully ingested document {metadata.document_id} with {len(chunks)} chunks")
            
            return {
                "status": "success",
                "document_id": metadata.document_id,
                "chunks_created": len(chunks),
                "chunking_strategy": strategy.value,
                "ingested_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to ingest document {metadata.document_id}: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _chunk_document(
        self,
        content: str,
        metadata: DocumentMetadata,
        strategy: ChunkingStrategy
    ) -> List[DocumentChunk]:
        """Chunk document using the specified strategy"""
        
        if strategy == ChunkingStrategy.FIXED_SIZE:
            return await self._fixed_size_chunking(content, metadata)
        elif strategy == ChunkingStrategy.SLIDING_WINDOW:
            return await self._sliding_window_chunking(content, metadata)
        elif strategy == ChunkingStrategy.HIERARCHICAL:
            return await self._hierarchical_chunking(content, metadata)
        elif strategy == ChunkingStrategy.TOPIC_BASED:
            return await self._topic_based_chunking(content, metadata)
        elif strategy == ChunkingStrategy.SEMANTIC:
            return await self._semantic_chunking(content, metadata)
        else:
            # Default to fixed size
            return await self._fixed_size_chunking(content, metadata)
    
    async def _fixed_size_chunking(
        self,
        content: str,
        metadata: DocumentMetadata
    ) -> List[DocumentChunk]:
        """Create fixed-size chunks with overlap"""
        
        text_chunks = self.text_splitter.split_text(content)
        
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk = DocumentChunk(
                chunk_id=f"{metadata.document_id}_fixed_{i}",
                document_id=metadata.document_id,
                content=chunk_text,
                chunk_index=i,
                metadata={
                    'chunking_strategy': 'fixed_size',
                    'chunk_size': len(chunk_text),
                    'total_chunks': len(text_chunks)
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _sliding_window_chunking(
        self,
        content: str,
        metadata: DocumentMetadata
    ) -> List[DocumentChunk]:
        """Create overlapping chunks with sliding window"""
        
        chunk_size = self.chunking_config["chunk_size"]
        overlap_size = self.chunking_config["chunk_overlap"]
        
        chunks = []
        chunk_index = 0
        start_pos = 0
        
        while start_pos < len(content):
            end_pos = min(start_pos + chunk_size, len(content))
            
            # Try to end at sentence boundary
            if end_pos < len(content):
                sentence_end = content.rfind('.', start_pos, end_pos)
                if sentence_end > start_pos + chunk_size // 2:  # At least half chunk
                    end_pos = sentence_end + 1
            
            chunk_content = content[start_pos:end_pos].strip()
            
            if chunk_content:
                chunk = DocumentChunk(
                    chunk_id=f"{metadata.document_id}_slide_{chunk_index}",
                    document_id=metadata.document_id,
                    content=chunk_content,
                    chunk_index=chunk_index,
                    metadata={
                        'chunking_strategy': 'sliding_window',
                        'start_pos': start_pos,
                        'end_pos': end_pos,
                        'overlap_size': overlap_size
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position with overlap
            start_pos = max(start_pos + chunk_size - overlap_size, end_pos)
        
        return chunks
    
    async def _hierarchical_chunking(
        self,
        content: str,
        metadata: DocumentMetadata
    ) -> List[DocumentChunk]:
        """Create hierarchical chunks based on document structure"""
        
        chunks = []
        chunk_index = 0
        
        # Split by headers/sections
        sections = self._split_by_headers(content)
        
        for section_title, section_content in sections:
            # Create parent chunk for section
            parent_id = f"{metadata.document_id}_parent_{chunk_index}"
            parent_chunk = DocumentChunk(
                chunk_id=parent_id,
                document_id=metadata.document_id,
                content=section_content,
                chunk_index=chunk_index,
                metadata={
                    'chunking_strategy': 'hierarchical',
                    'chunk_type': 'parent',
                    'section_title': section_title
                }
            )
            chunks.append(parent_chunk)
            
            # Create child chunks for section content
            child_texts = self.text_splitter.split_text(section_content)
            
            for i, child_text in enumerate(child_texts):
                child_chunk = DocumentChunk(
                    chunk_id=f"{metadata.document_id}_child_{chunk_index}_{i}",
                    document_id=metadata.document_id,
                    content=child_text,
                    chunk_index=chunk_index,
                    parent_chunk=parent_id,
                    metadata={
                        'chunking_strategy': 'hierarchical',
                        'chunk_type': 'child',
                        'parent_section': section_title,
                        'child_index': i
                    }
                )
                chunks.append(child_chunk)
                
                # Update parent with child reference
                parent_chunk.child_chunks.append(child_chunk.chunk_id)
            
            chunk_index += 1
        
        return chunks
    
    async def _topic_based_chunking(
        self,
        content: str,
        metadata: DocumentMetadata
    ) -> List[DocumentChunk]:
        """Chunk document based on topics"""
        
        # Simple topic-based chunking (could be enhanced with topic modeling)
        paragraphs = content.split('\n\n')
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if paragraph indicates topic change
            topic_indicators = ['## ', '### ', 'Chapter', 'Section', 'Part']
            is_topic_change = any(paragraph.startswith(indicator) for indicator in topic_indicators)
            
            if is_topic_change and current_chunk:
                # Create chunk from current paragraphs
                chunk_content = '\n\n'.join(current_chunk)
                chunk = DocumentChunk(
                    chunk_id=f"{metadata.document_id}_topic_{chunk_index}",
                    document_id=metadata.document_id,
                    content=chunk_content,
                    chunk_index=chunk_index,
                    metadata={
                        'chunking_strategy': 'topic_based',
                        'paragraph_count': len(current_chunk),
                        'topic_indicator': paragraph.split('\n')[0][:50]
                    }
                )
                chunks.append(chunk)
                
                current_chunk = [paragraph]
                current_length = len(paragraph)
                chunk_index += 1
            elif current_length + len(paragraph) > 1000 and current_chunk:
                # Create chunk due to size limit
                chunk_content = '\n\n'.join(current_chunk)
                chunk = DocumentChunk(
                    chunk_id=f"{metadata.document_id}_topic_{chunk_index}",
                    document_id=metadata.document_id,
                    content=chunk_content,
                    chunk_index=chunk_index,
                    metadata={
                        'chunking_strategy': 'topic_based',
                        'paragraph_count': len(current_chunk)
                    }
                )
                chunks.append(chunk)
                
                current_chunk = [paragraph]
                current_length = len(paragraph)
                chunk_index += 1
            else:
                current_chunk.append(paragraph)
                current_length += len(paragraph)
        
        # Handle remaining content
        if current_chunk:
            chunk_content = '\n\n'.join(current_chunk)
            chunk = DocumentChunk(
                chunk_id=f"{metadata.document_id}_topic_{chunk_index}",
                document_id=metadata.document_id,
                content=chunk_content,
                chunk_index=chunk_index,
                metadata={
                    'chunking_strategy': 'topic_based',
                    'paragraph_count': len(current_chunk)
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _semantic_chunking(
        self,
        content: str,
        metadata: DocumentMetadata
    ) -> List[DocumentChunk]:
        """Chunk document based on semantic similarity"""
        
        # Split into sentences
        sentences = self._split_into_sentences(content)
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # Check if adding this sentence would exceed chunk size
            if current_length + sentence_length > self.chunking_config["chunk_size"] and current_chunk:
                # Create chunk from current sentences
                chunk_content = ' '.join(current_chunk)
                chunk = DocumentChunk(
                    chunk_id=f"{metadata.document_id}_semantic_{chunk_index}",
                    document_id=metadata.document_id,
                    content=chunk_content,
                    chunk_index=chunk_index,
                    metadata={
                        'chunking_strategy': 'semantic',
                        'sentence_count': len(current_chunk),
                        'chunk_length': len(chunk_content)
                    }
                )
                chunks.append(chunk)
                
                current_chunk = [sentence]
                current_length = sentence_length
                chunk_index += 1
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Handle remaining content
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            chunk = DocumentChunk(
                chunk_id=f"{metadata.document_id}_semantic_{chunk_index}",
                document_id=metadata.document_id,
                content=chunk_content,
                chunk_index=chunk_index,
                metadata={
                    'chunking_strategy': 'semantic',
                    'sentence_count': len(current_chunk),
                    'chunk_length': len(chunk_content)
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        
        # Simple sentence splitting (could be enhanced with NLTK/spaCy)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter out very short sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return sentences
    
    def _split_by_headers(self, text: str) -> List[Tuple[str, str]]:
        """Split text by headers"""
        
        sections = []
        current_section = ""
        current_title = "Introduction"
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Check if line is a header (starts with # or is all caps)
            if (line.startswith('#') or 
                (len(line) > 0 and len(line) < 100 and line.isupper())):
                
                # Save previous section
                if current_section:
                    sections.append((current_title, current_section))
                
                # Start new section
                current_title = line.lstrip('# ')
                current_section = ""
            else:
                current_section += line + "\n"
        
        # Add final section
        if current_section:
            sections.append((current_title, current_section))
        
        return sections
    
    async def generate_embeddings(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Generate embeddings for document chunks"""
        
        for chunk in chunks:
            # Generate embedding (simulated)
            embedding = await self._generate_single_embedding(chunk.content)
            chunk.embedding = embedding
            
            # Extract keywords
            chunk.keywords = self._extract_keywords(chunk.content)
            
            # Calculate semantic density
            chunk.semantic_density = self._calculate_semantic_density(chunk.content)
        
        return chunks
    
    async def _generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        
        # Simulate embedding generation
        # In production, would use actual embedding model
        
        if self.embedding_model == 'sentence-transformers':
            # Simulate sentence-transformers embedding (384 dimensions)
            embedding = np.random.normal(0, 1, 384).astype(np.float32)
        else:
            # Simulate OpenAI embedding (1536 dimensions)
            embedding = np.random.normal(0, 1, 1536).astype(np.float32)
        
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    async def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for search query"""
        
        return await self._generate_single_embedding(query)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        
        # Simple keyword extraction (could use TF-IDF or more advanced methods)
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        # Extract words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter and count
        word_freq = {}
        for word in words:
            if len(word) > 3 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:10]]
    
    def _calculate_semantic_density(self, text: str) -> float:
        """Calculate semantic density of text"""
        
        # Simple semantic density calculation
        # In production, could use more sophisticated methods
        
        words = re.findall(r'\b\w+\b', text.lower())
        unique_words = set(words)
        
        if len(words) == 0:
            return 0.0
        
        return len(unique_words) / len(words)
    
    async def _update_vector_index(self, chunks: List[DocumentChunk]):
        """Update vector index with new chunks"""
        
        # Simulate vector index update
        # In production, would update actual vector store (FAISS, Pinecone, etc.)
        
        for chunk in chunks:
            if chunk.embedding is not None:
                # Store in simulated vector index
                if not hasattr(self, 'vector_index'):
                    self.vector_index = {}
                
                self.vector_index[chunk.chunk_id] = {
                    'embedding': chunk.embedding.tolist(),
                    'metadata': chunk.metadata
                }
    
    async def retrieve(
        self,
        query: str,
        context: QueryContext = None,
        retrieval_method: RetrievalMethod = None,
        max_results: int = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Search query
            context: Query context
            retrieval_method: Retrieval method to use
            max_results: Maximum number of results
            
        Returns:
            List of retrieval results
        """
        try:
            # Check cache first
            cache_key = f"{query}_{retrieval_method or self.retrieval_config['default_method']}"
            if cache_key in self.retrieval_cache:
                cached_result = self.retrieval_cache[cache_key]
                if datetime.now() - cached_result['timestamp'] < timedelta(hours=1):
                    logger.info(f"Retrieved from cache: {cache_key}")
                    return cached_result['results']
            
            # Generate query embedding
            query_embedding = await self.generate_query_embedding(query)
            
            # Use specified method or default
            method = retrieval_method or self.retrieval_config["default_method"]
            max_results = max_results or self.retrieval_config["max_results"]
            
            # Perform retrieval
            if method == RetrievalMethod.DENSE_RETRIEVAL:
                results = await self._dense_retrieval(query_embedding, max_results)
            elif method == RetrievalMethod.SPARSE_RETRIEVAL:
                results = await self._sparse_retrieval(query, max_results)
            elif method == RetrievalMethod.HYBRID_RETRIEVAL:
                results = await self._hybrid_retrieval(query, query_embedding, max_results)
            else:
                results = await self._dense_retrieval(query_embedding, max_results)
            
            # Apply filters if specified
            if context and context.filters:
                results = self._apply_filters(results, context.filters)
            
            # Cache results
            self.retrieval_cache[cache_key] = {
                'results': results,
                'timestamp': datetime.now()
            }
            
            logger.info(f"Retrieved {len(results)} results for query: {query[:50]}...")
            
            return results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            return []
    
    async def _dense_retrieval(
        self,
        query_embedding: np.ndarray,
        max_results: int
    ) -> List[RetrievalResult]:
        """Perform dense retrieval using embeddings"""
        
        results = []
        
        if not hasattr(self, 'vector_index') or not self.vector_index:
            return results
        
        # Calculate similarities
        similarities = []
        for chunk_id, vector_data in self.vector_index.items():
            chunk_embedding = np.array(vector_data['embedding'])
            similarity = np.dot(query_embedding, chunk_embedding)
            similarities.append((chunk_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Create results
        for chunk_id, similarity in similarities[:max_results]:
            if chunk_id in self.chunk_store:
                chunk_data = self.chunk_store[chunk_id]
                result = RetrievalResult(
                    chunk_id=chunk_id,
                    document_id=chunk_data['document_id'],
                    content=chunk_data['content'],
                    relevance_score=float(similarity),
                    retrieval_method=RetrievalMethod.DENSE_RETRIEVAL,
                    metadata=chunk_data['metadata']
                )
                results.append(result)
        
        return results
    
    async def _sparse_retrieval(
        self,
        query: str,
        max_results: int
    ) -> List[RetrievalResult]:
        """Perform sparse retrieval using keyword matching"""
        
        results = []
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        
        # Score chunks based on keyword overlap
        chunk_scores = []
        for chunk_id, chunk_data in self.chunk_store.items():
            chunk_words = set(chunk_data.get('keywords', []))
            overlap = len(query_words.intersection(chunk_words))
            if overlap > 0:
                score = overlap / len(query_words)
                chunk_scores.append((chunk_id, score))
        
        # Sort by score
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Create results
        for chunk_id, score in chunk_scores[:max_results]:
            chunk_data = self.chunk_store[chunk_id]
            result = RetrievalResult(
                chunk_id=chunk_id,
                document_id=chunk_data['document_id'],
                content=chunk_data['content'],
                relevance_score=score,
                retrieval_method=RetrievalMethod.SPARSE_RETRIEVAL,
                metadata=chunk_data['metadata']
            )
            results.append(result)
        
        return results
    
    async def _hybrid_retrieval(
        self,
        query: str,
        query_embedding: np.ndarray,
        max_results: int
    ) -> List[RetrievalResult]:
        """Perform hybrid retrieval combining dense and sparse methods"""
        
        # Get results from both methods
        dense_results = await self._dense_retrieval(query_embedding, max_results * 2)
        sparse_results = await self._sparse_retrieval(query, max_results * 2)
        
        # Combine and rerank
        combined_results = {}
        
        # Add dense results with weight
        for result in dense_results:
            chunk_id = result.chunk_id
            if chunk_id not in combined_results:
                combined_results[chunk_id] = result
                combined_results[chunk_id].relevance_score *= 0.7  # Dense weight
            else:
                combined_results[chunk_id].relevance_score += result.relevance_score * 0.7
        
        # Add sparse results with weight
        for result in sparse_results:
            chunk_id = result.chunk_id
            if chunk_id not in combined_results:
                combined_results[chunk_id] = result
                combined_results[chunk_id].relevance_score *= 0.3  # Sparse weight
            else:
                combined_results[chunk_id].relevance_score += result.relevance_score * 0.3
        
        # Sort by combined score
        final_results = list(combined_results.values())
        final_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Update retrieval method
        for result in final_results:
            result.retrieval_method = RetrievalMethod.HYBRID_RETRIEVAL
        
        return final_results[:max_results]
    
    def _apply_filters(
        self,
        results: List[RetrievalResult],
        filters: Dict[str, Any]
    ) -> List[RetrievalResult]:
        """Apply filters to retrieval results"""
        
        filtered_results = []
        
        for result in results:
            # Check document filters
            if 'document_types' in filters:
                doc_metadata = self.document_store.get(result.document_id, {}).get('metadata', {})
                doc_type = doc_metadata.get('document_type')
                if doc_type not in filters['document_types']:
                    continue
            
            # Check date filters
            if 'date_range' in filters:
                doc_metadata = self.document_store.get(result.document_id, {}).get('metadata', {})
                created_at = datetime.fromisoformat(doc_metadata.get('created_at', '2020-01-01'))
                start_date = filters['date_range'].get('start')
                end_date = filters['date_range'].get('end')
                
                if start_date and created_at < start_date:
                    continue
                if end_date and created_at > end_date:
                    continue
            
            # Check tag filters
            if 'tags' in filters:
                doc_metadata = self.document_store.get(result.document_id, {}).get('metadata', {})
                doc_tags = doc_metadata.get('tags', [])
                if not any(tag in doc_tags for tag in filters['tags']):
                    continue
            
            filtered_results.append(result)
        
        return filtered_results
    
    async def generate_response(
        self,
        query: str,
        retrieved_chunks: List[RetrievalResult],
        context: QueryContext = None
    ) -> Dict[str, Any]:
        """
        Generate a response using retrieved chunks.
        
        Args:
            query: Original query
            retrieved_chunks: Retrieved relevant chunks
            context: Query context
            
        Returns:
            Generated response with sources
        """
        try:
            # Combine retrieved content
            context_content = "\n\n".join([
                f"Source {i+1}: {chunk.content}"
                for i, chunk in enumerate(retrieved_chunks)
            ])
            
            # Generate response (simulated)
            # In production, would use actual LLM for generation
            response = f"Based on the retrieved information, here's what I found regarding your query '{query}':\n\n{context_content[:1000]}..."
            
            # Prepare sources
            sources = []
            for i, chunk in enumerate(retrieved_chunks):
                doc_metadata = self.document_store.get(chunk.document_id, {}).get('metadata', {})
                source = {
                    "source_id": i + 1,
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.chunk_id,
                    "title": doc_metadata.get('title', 'Unknown'),
                    "relevance_score": chunk.relevance_score,
                    "retrieval_method": chunk.retrieval_method.value
                }
                sources.append(source)
            
            return {
                "response": response,
                "sources": sources,
                "query": query,
                "retrieval_metadata": {
                    "total_chunks_retrieved": len(retrieved_chunks),
                    "avg_relevance_score": sum(c.relevance_score for c in retrieved_chunks) / len(retrieved_chunks) if retrieved_chunks else 0,
                    "generated_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            return {
                "response": f"Error generating response: {str(e)}",
                "sources": [],
                "query": query,
                "error": str(e)
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        
        total_documents = len(self.document_store)
        total_chunks = len(self.chunk_store)
        
        # Calculate chunking strategy distribution
        strategy_distribution = defaultdict(int)
        for chunk_data in self.chunk_store.values():
            strategy = chunk_data.get('metadata', {}).get('chunking_strategy', 'unknown')
            strategy_distribution[strategy] += 1
        
        # Calculate document type distribution
        doc_type_distribution = defaultdict(int)
        for doc_data in self.document_store.values():
            doc_type = doc_data.get('metadata', {}).get('document_type', 'unknown')
            doc_type_distribution[doc_type] += 1
        
        return {
            "system_name": self.system_name,
            "total_documents": total_documents,
            "total_chunks": total_chunks,
            "chunking_strategy_distribution": dict(strategy_distribution),
            "document_type_distribution": dict(doc_type_distribution),
            "embedding_model": self.embedding_model,
            "vector_store": self.vector_store,
            "cache_size": len(self.retrieval_cache),
            "last_updated": datetime.now().isoformat()
        }


# Mock classes for demonstration
class MockTextSplitter:
    """Mock text splitter for demonstration"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
        
        return chunks


class MockEmbeddingGenerator:
    """Mock embedding generator for demonstration"""
    
    def __init__(self, model_name: str = "sentence-transformers"):
        self.model_name = model_name
    
    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        # Simulate embedding generation
        if self.model_name == 'sentence-transformers':
            embedding = np.random.normal(0, 1, 384).astype(np.float32)
        else:
            embedding = np.random.normal(0, 1, 1536).astype(np.float32)
        
        return embedding / np.linalg.norm(embedding)


class MockRetrievalEngine:
    """Mock retrieval engine for demonstration"""
    
    def __init__(self, vector_store: str = "faiss"):
        self.vector_store = vector_store
    
    async def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        # Simulate vector search
        results = []
        for i in range(min(top_k, 5)):  # Simulate 5 results
            results.append({
                "id": f"chunk_{i}",
                "score": np.random.random(),
                "metadata": {"chunk_index": i}
            })
        
        return results
