"""
GraphRAG Neo4j Integration for Lenovo Enterprise Solutions

This module provides GraphRAG-optimized integration with Neo4j for Lenovo's
enterprise needs, including semantic search, knowledge graph queries, and
agentic workflow support.

Key Features:
- GraphRAG-optimized graph structure design
- Semantic search and retrieval
- Knowledge graph querying for agentic workflows
- Context-aware document retrieval
- Multi-hop reasoning capabilities
- Enterprise-specific graph patterns
- Integration with existing Lenovo systems
"""

import asyncio
import json
import uuid
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import hashlib

# Neo4j imports
try:
    from neo4j import GraphDatabase
    from py2neo import Graph, Node, Relationship
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logging.warning("Neo4j not available. Install with: pip install neo4j py2neo")

# Vector search imports
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    VECTOR_AVAILABLE = True
except ImportError:
    VECTOR_AVAILABLE = False
    logging.warning("Vector search not available. Install with: pip install sentence-transformers numpy")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphRAGNodeType(Enum):
    """Node types optimized for GraphRAG"""
    DOCUMENT = "document"
    CONCEPT = "concept"
    ENTITY = "entity"
    RELATIONSHIP = "relationship"
    CONTEXT = "context"
    QUERY = "query"
    ANSWER = "answer"
    EVIDENCE = "evidence"
    SOURCE = "source"
    METADATA = "metadata"


class GraphRAGRelationshipType(Enum):
    """Relationship types for GraphRAG"""
    CONTAINS = "contains"
    REFERENCES = "references"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    SIMILAR_TO = "similar_to"
    PART_OF = "part_of"
    CAUSES = "causes"
    ENABLES = "enables"
    PREVENTS = "prevents"
    LEADS_TO = "leads_to"


class QueryType(Enum):
    """Types of GraphRAG queries"""
    SEMANTIC_SEARCH = "semantic_search"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    CONTEXT_AGGREGATION = "context_aggregation"
    MULTI_HOP_REASONING = "multi_hop_reasoning"
    EVIDENCE_GATHERING = "evidence_gathering"
    RELATIONSHIP_EXPLORATION = "relationship_exploration"


@dataclass
class GraphRAGNode:
    """GraphRAG-optimized node"""
    id: str
    label: str
    node_type: GraphRAGNodeType
    properties: Dict[str, Any]
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class GraphRAGEdge:
    """GraphRAG-optimized edge"""
    id: str
    source: str
    target: str
    relationship_type: GraphRAGRelationshipType
    properties: Dict[str, Any]
    weight: float = 1.0
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class GraphRAGQuery:
    """GraphRAG query structure"""
    id: str
    query_text: str
    query_type: QueryType
    context: Dict[str, Any]
    results: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class GraphRAGResult:
    """GraphRAG query result"""
    query_id: str
    nodes: List[GraphRAGNode]
    edges: List[GraphRAGEdge]
    paths: List[List[str]]
    confidence: float
    evidence: List[Dict[str, Any]]
    created_at: datetime = field(default_factory=datetime.now)


class GraphRAGNeo4jIntegration:
    """GraphRAG integration with Neo4j for Lenovo enterprise solutions"""
    
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687", 
                 neo4j_user: str = "neo4j", neo4j_password: str = "password",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.graph: Optional[Graph] = None
        self.embedding_model = None
        
        # Initialize Neo4j connection
        if NEO4J_AVAILABLE:
            self._initialize_neo4j()
        
        # Initialize embedding model
        if VECTOR_AVAILABLE:
            self._initialize_embedding_model(embedding_model)
    
    def _initialize_neo4j(self):
        """Initialize Neo4j connection"""
        try:
            self.graph = Graph(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
            logger.info("Neo4j connection initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to connect to Neo4j: {e}")
            self.graph = None
    
    def _initialize_embedding_model(self, model_name: str):
        """Initialize sentence transformer model"""
        try:
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"Embedding model {model_name} initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize embedding model: {e}")
            self.embedding_model = None
    
    def create_graphrag_optimized_structure(self) -> bool:
        """Create GraphRAG-optimized graph structure in Neo4j"""
        if not NEO4J_AVAILABLE or not self.graph:
            logger.warning("Neo4j not available")
            return False
        
        try:
            # Create indexes for GraphRAG optimization
            indexes = [
                "CREATE INDEX document_id_index IF NOT EXISTS FOR (d:document) ON (d.id)",
                "CREATE INDEX concept_id_index IF NOT EXISTS FOR (c:concept) ON (c.id)",
                "CREATE INDEX entity_id_index IF NOT EXISTS FOR (e:entity) ON (e.id)",
                "CREATE INDEX embedding_index IF NOT EXISTS FOR (n) ON (n.embedding)",
                "CREATE INDEX text_content_index IF NOT EXISTS FOR (n) ON (n.text_content)",
                "CREATE INDEX created_at_index IF NOT EXISTS FOR (n) ON (n.created_at)"
            ]
            
            for index_query in indexes:
                try:
                    self.graph.run(index_query)
                except Exception as e:
                    logger.warning(f"Index creation warning: {e}")
            
            # Create constraints for data integrity
            constraints = [
                "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:document) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT concept_id_unique IF NOT EXISTS FOR (c:concept) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:entity) REQUIRE e.id IS UNIQUE"
            ]
            
            for constraint_query in constraints:
                try:
                    self.graph.run(constraint_query)
                except Exception as e:
                    logger.warning(f"Constraint creation warning: {e}")
            
            logger.info("GraphRAG-optimized structure created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create GraphRAG structure: {e}")
            return False
    
    def add_document_with_embedding(self, document_text: str, metadata: Dict[str, Any]) -> str:
        """Add document with embedding to the graph"""
        if not self.graph or not self.embedding_model:
            logger.warning("Neo4j or embedding model not available")
            return None
        
        try:
            # Generate document ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}"
            
            # Generate embedding
            embedding = self.embedding_model.encode(document_text).tolist()
            
            # Create document node
            doc_node = Node(
                "document",
                id=doc_id,
                text_content=document_text,
                embedding=embedding,
                metadata=json.dumps(metadata),
                created_at=datetime.now().isoformat()
            )
            self.graph.create(doc_node)
            
            # Extract entities and concepts
            entities = self._extract_entities(document_text)
            concepts = self._extract_concepts(document_text)
            
            # Create entity nodes and relationships
            for entity in entities:
                entity_id = f"ent_{hashlib.md5(entity.encode()).hexdigest()[:8]}"
                entity_node = Node(
                    "entity",
                    id=entity_id,
                    name=entity,
                    type="extracted_entity"
                )
                self.graph.create(entity_node)
                
                # Create relationship
                rel = Relationship(doc_node, "CONTAINS", entity_node)
                self.graph.create(rel)
            
            # Create concept nodes and relationships
            for concept in concepts:
                concept_id = f"concept_{hashlib.md5(concept.encode()).hexdigest()[:8]}"
                concept_node = Node(
                    "concept",
                    id=concept_id,
                    name=concept,
                    type="extracted_concept"
                )
                self.graph.create(concept_node)
                
                # Create relationship
                rel = Relationship(doc_node, "REFERENCES", concept_node)
                self.graph.create(rel)
            
            logger.info(f"Document {doc_id} added with {len(entities)} entities and {len(concepts)} concepts")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return None
    
    def semantic_search(self, query_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic search using embeddings"""
        if not self.graph or not self.embedding_model:
            logger.warning("Neo4j or embedding model not available")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query_text).tolist()
            
            # Cypher query for semantic search
            cypher_query = """
            MATCH (d:document)
            WHERE d.embedding IS NOT NULL
            WITH d, 
                 gds.similarity.cosine(d.embedding, $query_embedding) AS similarity
            WHERE similarity > 0.5
            RETURN d.id, d.text_content, d.metadata, similarity
            ORDER BY similarity DESC
            LIMIT $limit
            """
            
            results = self.graph.run(cypher_query, 
                                   query_embedding=query_embedding, 
                                   limit=limit).data()
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def knowledge_retrieval(self, query_text: str, max_hops: int = 3) -> GraphRAGResult:
        """Retrieve knowledge using multi-hop reasoning"""
        if not self.graph:
            logger.warning("Neo4j not available")
            return None
        
        try:
            # Find relevant documents
            relevant_docs = self.semantic_search(query_text, limit=5)
            
            if not relevant_docs:
                return GraphRAGResult(
                    query_id=f"query_{uuid.uuid4().hex[:8]}",
                    nodes=[],
                    edges=[],
                    paths=[],
                    confidence=0.0,
                    evidence=[]
                )
            
            # Build knowledge graph from relevant documents
            doc_ids = [doc['d.id'] for doc in relevant_docs]
            
            # Multi-hop traversal query
            cypher_query = f"""
            MATCH path = (d:document)-[*1..{max_hops}]-(related)
            WHERE d.id IN $doc_ids
            RETURN path, 
                   [node in nodes(path) | node] as nodes,
                   [rel in relationships(path) | rel] as edges
            """
            
            results = self.graph.run(cypher_query, doc_ids=doc_ids).data()
            
            # Process results
            all_nodes = set()
            all_edges = set()
            paths = []
            evidence = []
            
            for result in results:
                path = result['path']
                nodes = result['nodes']
                edges = result['edges']
                
                # Collect nodes and edges
                for node in nodes:
                    all_nodes.add(node['id'])
                
                for edge in edges:
                    all_edges.add((edge.start_node['id'], edge.end_node['id'], edge.type))
                
                # Extract paths
                path_ids = [node['id'] for node in nodes]
                paths.append(path_ids)
                
                # Collect evidence
                evidence.append({
                    'path': path_ids,
                    'confidence': 0.8,  # Placeholder confidence
                    'relevance': 0.9    # Placeholder relevance
                })
            
            return GraphRAGResult(
                query_id=f"query_{uuid.uuid4().hex[:8]}",
                nodes=[],  # Would need to fetch full node data
                edges=[],  # Would need to fetch full edge data
                paths=paths,
                confidence=0.8,
                evidence=evidence
            )
            
        except Exception as e:
            logger.error(f"Knowledge retrieval failed: {e}")
            return None
    
    def context_aggregation(self, query_text: str, context_size: int = 5) -> Dict[str, Any]:
        """Aggregate context from multiple sources"""
        if not self.graph:
            logger.warning("Neo4j not available")
            return {}
        
        try:
            # Find relevant documents
            relevant_docs = self.semantic_search(query_text, limit=context_size)
            
            # Aggregate context
            context = {
                'query': query_text,
                'relevant_documents': relevant_docs,
                'entities': set(),
                'concepts': set(),
                'relationships': set(),
                'metadata': {}
            }
            
            # Extract entities and concepts from relevant documents
            for doc in relevant_docs:
                doc_id = doc['d.id']
                
                # Get entities connected to this document
                entity_query = """
                MATCH (d:document)-[:CONTAINS]->(e:entity)
                WHERE d.id = $doc_id
                RETURN e.name, e.type
                """
                entities = self.graph.run(entity_query, doc_id=doc_id).data()
                for entity in entities:
                    context['entities'].add(entity['e.name'])
                
                # Get concepts connected to this document
                concept_query = """
                MATCH (d:document)-[:REFERENCES]->(c:concept)
                WHERE d.id = $doc_id
                RETURN c.name, c.type
                """
                concepts = self.graph.run(concept_query, doc_id=doc_id).data()
                for concept in concepts:
                    context['concepts'].add(concept['c.name'])
            
            # Convert sets to lists for JSON serialization
            context['entities'] = list(context['entities'])
            context['concepts'] = list(context['concepts'])
            
            return context
            
        except Exception as e:
            logger.error(f"Context aggregation failed: {e}")
            return {}
    
    def relationship_exploration(self, entity_name: str, max_depth: int = 2) -> Dict[str, Any]:
        """Explore relationships around a specific entity"""
        if not self.graph:
            logger.warning("Neo4j not available")
            return {}
        
        try:
            # Find entity and its relationships
            cypher_query = f"""
            MATCH path = (e:entity)-[*1..{max_depth}]-(related)
            WHERE e.name = $entity_name
            RETURN path,
                   [node in nodes(path) | {{id: node.id, label: node.name, type: labels(node)[0]}}] as nodes,
                   [rel in relationships(path) | {{type: rel.type, properties: properties(rel)}}] as edges
            """
            
            results = self.graph.run(cypher_query, entity_name=entity_name).data()
            
            exploration = {
                'entity': entity_name,
                'relationships': [],
                'connected_entities': set(),
                'connected_concepts': set(),
                'paths': []
            }
            
            for result in results:
                nodes = result['nodes']
                edges = result['edges']
                
                # Collect connected entities and concepts
                for node in nodes:
                    if node['type'] == 'entity':
                        exploration['connected_entities'].add(node['label'])
                    elif node['type'] == 'concept':
                        exploration['connected_concepts'].add(node['label'])
                
                # Store paths
                exploration['paths'].append({
                    'nodes': nodes,
                    'edges': edges
                })
            
            # Convert sets to lists
            exploration['connected_entities'] = list(exploration['connected_entities'])
            exploration['connected_concepts'] = list(exploration['connected_concepts'])
            
            return exploration
            
        except Exception as e:
            logger.error(f"Relationship exploration failed: {e}")
            return {}
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text (simplified implementation)"""
        # This is a simplified implementation
        # In production, you would use NER models like spaCy or transformers
        entities = []
        
        # Simple keyword-based entity extraction
        keywords = [
            "Lenovo", "ThinkPad", "ThinkCentre", "ThinkSystem", "ThinkAgile",
            "CEO", "CTO", "CFO", "VP", "Director", "Manager", "Engineer",
            "Project", "Department", "Team", "Customer", "Client", "Partner"
        ]
        
        for keyword in keywords:
            if keyword.lower() in text.lower():
                entities.append(keyword)
        
        return list(set(entities))
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract concepts from text (simplified implementation)"""
        # This is a simplified implementation
        # In production, you would use concept extraction models
        concepts = []
        
        # Simple concept extraction based on common business terms
        concept_keywords = [
            "strategy", "innovation", "technology", "business", "process",
            "management", "leadership", "collaboration", "efficiency",
            "quality", "performance", "growth", "development", "transformation"
        ]
        
        for concept in concept_keywords:
            if concept.lower() in text.lower():
                concepts.append(concept)
        
        return list(set(concepts))
    
    def create_lenovo_knowledge_base(self) -> bool:
        """Create Lenovo-specific knowledge base with GraphRAG optimization"""
        if not self.graph:
            logger.warning("Neo4j not available")
            return False
        
        try:
            # Create GraphRAG-optimized structure
            self.create_graphrag_optimized_structure()
            
            # Add Lenovo-specific documents
            lenovo_documents = [
                {
                    "text": "Lenovo is a global technology company that designs, develops, and manufactures innovative technology products and services. The company is headquartered in Beijing, China, with operations in over 60 countries worldwide.",
                    "metadata": {"type": "company_overview", "source": "lenovo_website"}
                },
                {
                    "text": "ThinkPad is Lenovo's flagship laptop series, known for its durability, security features, and business-focused design. ThinkPad laptops are widely used in enterprise environments for their reliability and performance.",
                    "metadata": {"type": "product_info", "source": "product_catalog"}
                },
                {
                    "text": "ThinkSystem servers provide enterprise-grade computing solutions for data centers and cloud environments. These servers are designed for high performance, scalability, and reliability in mission-critical applications.",
                    "metadata": {"type": "product_info", "source": "product_catalog"}
                },
                {
                    "text": "Lenovo's AI and machine learning solutions help enterprises transform their operations through intelligent automation, predictive analytics, and cognitive computing capabilities.",
                    "metadata": {"type": "technology_info", "source": "technology_whitepaper"}
                },
                {
                    "text": "The Lenovo enterprise sales team works with customers to understand their business requirements and recommend appropriate technology solutions that align with their strategic objectives.",
                    "metadata": {"type": "business_process", "source": "sales_playbook"}
                }
            ]
            
            # Add documents to the graph
            for doc in lenovo_documents:
                doc_id = self.add_document_with_embedding(
                    doc["text"], 
                    doc["metadata"]
                )
                if doc_id:
                    logger.info(f"Added document: {doc_id}")
            
            logger.info("Lenovo knowledge base created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create Lenovo knowledge base: {e}")
            return False
    
    def query_lenovo_knowledge(self, query: str) -> Dict[str, Any]:
        """Query Lenovo knowledge base using GraphRAG"""
        if not self.graph:
            logger.warning("Neo4j not available")
            return {}
        
        try:
            # Perform semantic search
            search_results = self.semantic_search(query, limit=5)
            
            # Aggregate context
            context = self.context_aggregation(query, context_size=3)
            
            # Explore relationships
            if context.get('entities'):
                entity_exploration = self.relationship_exploration(
                    context['entities'][0], max_depth=2
                )
            else:
                entity_exploration = {}
            
            # Combine results
            result = {
                'query': query,
                'search_results': search_results,
                'context': context,
                'entity_exploration': entity_exploration,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {}
    
    def export_graphrag_data(self, filepath: str) -> bool:
        """Export GraphRAG data for analysis"""
        if not self.graph:
            logger.warning("Neo4j not available")
            return False
        
        try:
            # Export documents
            docs_query = """
            MATCH (d:document)
            RETURN d.id, d.text_content, d.metadata, d.created_at
            """
            documents = self.graph.run(docs_query).data()
            
            # Export entities
            entities_query = """
            MATCH (e:entity)
            RETURN e.id, e.name, e.type
            """
            entities = self.graph.run(entities_query).data()
            
            # Export concepts
            concepts_query = """
            MATCH (c:concept)
            RETURN c.id, c.name, c.type
            """
            concepts = self.graph.run(concepts_query).data()
            
            # Export relationships
            relationships_query = """
            MATCH (a)-[r]->(b)
            RETURN a.id as source, b.id as target, r.type as relationship_type, properties(r) as properties
            """
            relationships = self.graph.run(relationships_query).data()
            
            # Combine data
            export_data = {
                'documents': documents,
                'entities': entities,
                'concepts': concepts,
                'relationships': relationships,
                'export_timestamp': datetime.now().isoformat()
            }
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"GraphRAG data exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False


async def main():
    """Main function to demonstrate GraphRAG integration"""
    
    # Initialize GraphRAG integration
    graphrag = GraphRAGNeo4jIntegration()
    
    # Create Lenovo knowledge base
    logger.info("Creating Lenovo knowledge base...")
    success = graphrag.create_lenovo_knowledge_base()
    
    if success:
        logger.info("Knowledge base created successfully")
        
        # Test queries
        test_queries = [
            "What is Lenovo's flagship laptop series?",
            "Tell me about Lenovo's enterprise solutions",
            "What are Lenovo's AI and machine learning capabilities?",
            "How does Lenovo's sales process work?"
        ]
        
        for query in test_queries:
            logger.info(f"Querying: {query}")
            result = graphrag.query_lenovo_knowledge(query)
            logger.info(f"Results: {len(result.get('search_results', []))} documents found")
        
        # Export data
        graphrag.export_graphrag_data("neo4j_data/graphrag_export.json")
        
    else:
        logger.error("Failed to create knowledge base")


if __name__ == "__main__":
    asyncio.run(main())
