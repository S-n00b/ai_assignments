"""
Lenovo Knowledge Graph for Hybrid RAG

This module provides Lenovo-specific knowledge graph construction and querying
for enhanced retrieval in hybrid RAG workflows.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json
import networkx as nx
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

class LenovoKnowledgeGraph:
    """
    Lenovo knowledge graph for hybrid RAG.
    
    Provides Lenovo-specific knowledge graph construction, querying,
    and integration with hybrid RAG workflows.
    """
    
    def __init__(self, 
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password"):
        """
        Initialize Lenovo knowledge graph.
        
        Args:
            neo4j_uri: Neo4j database URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.driver = None
        
    def connect(self) -> None:
        """Connect to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )
            logger.info(f"Connected to Neo4j at {self.neo4j_uri}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def create_lenovo_schema(self) -> None:
        """Create Lenovo-specific schema in Neo4j."""
        try:
            with self.driver.session() as session:
                # Create constraints and indexes
                constraints = [
                    "CREATE CONSTRAINT device_id IF NOT EXISTS FOR (d:Device) REQUIRE d.id IS UNIQUE",
                    "CREATE CONSTRAINT customer_id IF NOT EXISTS FOR (c:Customer) REQUIRE c.id IS UNIQUE",
                    "CREATE CONSTRAINT process_id IF NOT EXISTS FOR (p:Process) REQUIRE p.id IS UNIQUE",
                    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (doc:Document) REQUIRE doc.id IS UNIQUE"
                ]
                
                for constraint in constraints:
                    session.run(constraint)
                
                # Create indexes
                indexes = [
                    "CREATE INDEX device_name IF NOT EXISTS FOR (d:Device) ON (d.name)",
                    "CREATE INDEX customer_name IF NOT EXISTS FOR (c:Customer) ON (c.name)",
                    "CREATE INDEX process_name IF NOT EXISTS FOR (p:Process) ON (p.name)",
                    "CREATE INDEX document_title IF NOT EXISTS FOR (doc:Document) ON (doc.title)"
                ]
                
                for index in indexes:
                    session.run(index)
                
                logger.info("Created Lenovo schema in Neo4j")
                
        except Exception as e:
            logger.error(f"Failed to create Lenovo schema: {e}")
            raise
    
    def add_device_nodes(self, devices: List[Dict]) -> None:
        """
        Add device nodes to knowledge graph.
        
        Args:
            devices: List of device data
        """
        try:
            with self.driver.session() as session:
                for device in devices:
                    # Create device node
                    session.run("""
                        MERGE (d:Device {id: $id})
                        SET d.name = $name,
                            d.type = $type,
                            d.category = $category,
                            d.specifications = $specifications,
                            d.support_info = $support_info
                    """, 
                    id=device['id'],
                    name=device['name'],
                    type=device['type'],
                    category=device['category'],
                    specifications=device.get('specifications', {}),
                    support_info=device.get('support_info', '')
                    )
                    
                    # Create relationships with categories
                    if 'categories' in device:
                        for category in device['categories']:
                            session.run("""
                                MERGE (cat:Category {name: $category_name})
                                MERGE (d:Device {id: $device_id})
                                MERGE (d)-[:BELONGS_TO]->(cat)
                            """, 
                            category_name=category,
                            device_id=device['id']
                            )
                
                logger.info(f"Added {len(devices)} device nodes to knowledge graph")
                
        except Exception as e:
            logger.error(f"Failed to add device nodes: {e}")
            raise
    
    def add_customer_nodes(self, customers: List[Dict]) -> None:
        """
        Add customer nodes to knowledge graph.
        
        Args:
            customers: List of customer data
        """
        try:
            with self.driver.session() as session:
                for customer in customers:
                    # Create customer node
                    session.run("""
                        MERGE (c:Customer {id: $id})
                        SET c.name = $name,
                            c.type = $type,
                            c.industry = $industry,
                            c.contact_info = $contact_info,
                            c.preferences = $preferences
                    """, 
                    id=customer['id'],
                    name=customer['name'],
                    type=customer['type'],
                    industry=customer.get('industry', ''),
                    contact_info=customer.get('contact_info', {}),
                    preferences=customer.get('preferences', {})
                    )
                    
                    # Create relationships with industries
                    if 'industries' in customer:
                        for industry in customer['industries']:
                            session.run("""
                                MERGE (ind:Industry {name: $industry_name})
                                MERGE (c:Customer {id: $customer_id})
                                MERGE (c)-[:WORKS_IN]->(ind)
                            """, 
                            industry_name=industry,
                            customer_id=customer['id']
                            )
                
                logger.info(f"Added {len(customers)} customer nodes to knowledge graph")
                
        except Exception as e:
            logger.error(f"Failed to add customer nodes: {e}")
            raise
    
    def add_process_nodes(self, processes: List[Dict]) -> None:
        """
        Add business process nodes to knowledge graph.
        
        Args:
            processes: List of process data
        """
        try:
            with self.driver.session() as session:
                for process in processes:
                    # Create process node
                    session.run("""
                        MERGE (p:Process {id: $id})
                        SET p.name = $name,
                            p.type = $type,
                            p.description = $description,
                            p.steps = $steps,
                            p.department = $department
                    """, 
                    id=process['id'],
                    name=process['name'],
                    type=process['type'],
                    description=process.get('description', ''),
                    steps=process.get('steps', []),
                    department=process.get('department', '')
                    )
                    
                    # Create relationships with departments
                    if 'departments' in process:
                        for department in process['departments']:
                            session.run("""
                                MERGE (dept:Department {name: $dept_name})
                                MERGE (p:Process {id: $process_id})
                                MERGE (p)-[:BELONGS_TO]->(dept)
                            """, 
                            dept_name=department,
                            process_id=process['id']
                            )
                
                logger.info(f"Added {len(processes)} process nodes to knowledge graph")
                
        except Exception as e:
            logger.error(f"Failed to add process nodes: {e}")
            raise
    
    def add_document_nodes(self, documents: List[Dict]) -> None:
        """
        Add document nodes to knowledge graph.
        
        Args:
            documents: List of document data
        """
        try:
            with self.driver.session() as session:
                for document in documents:
                    # Create document node
                    session.run("""
                        MERGE (doc:Document {id: $id})
                        SET doc.title = $title,
                            doc.content = $content,
                            doc.type = $type,
                            doc.category = $category,
                            doc.creation_date = $creation_date
                    """, 
                    id=document['id'],
                    title=document['title'],
                    content=document['content'],
                    type=document['type'],
                    category=document.get('category', ''),
                    creation_date=document.get('creation_date', '')
                    )
                    
                    # Create relationships with categories
                    if 'categories' in document:
                        for category in document['categories']:
                            session.run("""
                                MERGE (cat:Category {name: $category_name})
                                MERGE (doc:Document {id: $doc_id})
                                MERGE (doc)-[:BELONGS_TO]->(cat)
                            """, 
                            category_name=category,
                            doc_id=document['id']
                            )
                
                logger.info(f"Added {len(documents)} document nodes to knowledge graph")
                
        except Exception as e:
            logger.error(f"Failed to add document nodes: {e}")
            raise
    
    def create_relationships(self, relationships: List[Dict]) -> None:
        """
        Create relationships between nodes.
        
        Args:
            relationships: List of relationship data
        """
        try:
            with self.driver.session() as session:
                for rel in relationships:
                    # Create relationship
                    session.run(f"""
                        MATCH (a:{rel['from_type']} {{id: $from_id}})
                        MATCH (b:{rel['to_type']} {{id: $to_id}})
                        MERGE (a)-[r:{rel['relationship_type']}]->(b)
                        SET r.properties = $properties
                    """, 
                    from_id=rel['from_id'],
                    to_id=rel['to_id'],
                    properties=rel.get('properties', {})
                    )
                
                logger.info(f"Created {len(relationships)} relationships in knowledge graph")
                
        except Exception as e:
            logger.error(f"Failed to create relationships: {e}")
            raise
    
    def query_knowledge_graph(self, 
                            query: str,
                            query_type: str = "text_search",
                            limit: int = 10) -> List[Dict[str, Any]]:
        """
        Query the knowledge graph.
        
        Args:
            query: Query text
            query_type: Type of query (text_search, device_search, customer_search, process_search)
            limit: Maximum number of results
            
        Returns:
            Query results
        """
        try:
            with self.driver.session() as session:
                if query_type == "text_search":
                    # General text search across all nodes
                    cypher_query = """
                        MATCH (n)
                        WHERE toLower(n.name) CONTAINS toLower($query)
                           OR toLower(n.title) CONTAINS toLower($query)
                           OR toLower(n.description) CONTAINS toLower($query)
                           OR toLower(n.content) CONTAINS toLower($query)
                        RETURN n, labels(n) as node_type
                        LIMIT $limit
                    """
                    
                elif query_type == "device_search":
                    # Device-specific search
                    cypher_query = """
                        MATCH (d:Device)
                        WHERE toLower(d.name) CONTAINS toLower($query)
                           OR toLower(d.type) CONTAINS toLower($query)
                           OR toLower(d.category) CONTAINS toLower($query)
                        RETURN d, labels(d) as node_type
                        LIMIT $limit
                    """
                    
                elif query_type == "customer_search":
                    # Customer-specific search
                    cypher_query = """
                        MATCH (c:Customer)
                        WHERE toLower(c.name) CONTAINS toLower($query)
                           OR toLower(c.type) CONTAINS toLower($query)
                           OR toLower(c.industry) CONTAINS toLower($query)
                        RETURN c, labels(c) as node_type
                        LIMIT $limit
                    """
                    
                elif query_type == "process_search":
                    # Process-specific search
                    cypher_query = """
                        MATCH (p:Process)
                        WHERE toLower(p.name) CONTAINS toLower($query)
                           OR toLower(p.type) CONTAINS toLower($query)
                           OR toLower(p.description) CONTAINS toLower($query)
                        RETURN p, labels(p) as node_type
                        LIMIT $limit
                    """
                    
                else:
                    raise ValueError(f"Unsupported query type: {query_type}")
                
                # Execute query
                results = session.run(cypher_query, query=query, limit=limit)
                
                # Format results
                formatted_results = []
                for record in results:
                    node = record['n']
                    node_type = record['node_type'][0] if record['node_type'] else 'Unknown'
                    
                    formatted_results.append({
                        'id': node.get('id', ''),
                        'name': node.get('name', ''),
                        'title': node.get('title', ''),
                        'type': node_type,
                        'properties': dict(node)
                    })
                
                logger.info(f"Found {len(formatted_results)} results for query: {query}")
                return formatted_results
                
        except Exception as e:
            logger.error(f"Knowledge graph query failed: {e}")
            raise
    
    def get_related_nodes(self, 
                         node_id: str,
                         node_type: str,
                         relationship_types: List[str] = None,
                         depth: int = 2) -> Dict[str, Any]:
        """
        Get related nodes for a given node.
        
        Args:
            node_id: ID of the node
            node_type: Type of the node
            relationship_types: Types of relationships to follow
            depth: Maximum depth to traverse
            
        Returns:
            Related nodes and relationships
        """
        try:
            with self.driver.session() as session:
                # Build relationship filter
                rel_filter = ""
                if relationship_types:
                    rel_filter = f"WHERE type(r) IN {relationship_types}"
                
                # Query for related nodes
                cypher_query = f"""
                    MATCH (n:{node_type} {{id: $node_id}})
                    MATCH path = (n)-[r*1..{depth}]-(related)
                    {rel_filter}
                    RETURN path, length(path) as path_length
                    ORDER BY path_length
                """
                
                results = session.run(cypher_query, node_id=node_id)
                
                # Format results
                related_nodes = []
                relationships = []
                
                for record in results:
                    path = record['path']
                    path_length = record['path_length']
                    
                    # Extract nodes and relationships from path
                    for i in range(len(path.nodes)):
                        node = path.nodes[i]
                        related_nodes.append({
                            'id': node.get('id', ''),
                            'labels': list(node.labels),
                            'properties': dict(node)
                        })
                    
                    for i in range(len(path.relationships)):
                        rel = path.relationships[i]
                        relationships.append({
                            'type': rel.type,
                            'properties': dict(rel)
                        })
                
                result = {
                    'node_id': node_id,
                    'node_type': node_type,
                    'related_nodes': related_nodes,
                    'relationships': relationships,
                    'max_depth': depth
                }
                
                logger.info(f"Found {len(related_nodes)} related nodes for {node_type}:{node_id}")
                return result
                
        except Exception as e:
            logger.error(f"Failed to get related nodes: {e}")
            raise
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get knowledge graph statistics.
        
        Returns:
            Graph statistics
        """
        try:
            with self.driver.session() as session:
                # Get node counts by type
                node_counts = {}
                node_types = ['Device', 'Customer', 'Process', 'Document', 'Category', 'Industry', 'Department']
                
                for node_type in node_types:
                    result = session.run(f"MATCH (n:{node_type}) RETURN count(n) as count")
                    count = result.single()['count']
                    node_counts[node_type] = count
                
                # Get relationship counts
                rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                relationship_count = rel_result.single()['count']
                
                # Get total node count
                total_nodes = sum(node_counts.values())
                
                stats = {
                    'total_nodes': total_nodes,
                    'total_relationships': relationship_count,
                    'node_counts': node_counts,
                    'node_types': node_types
                }
                
                logger.info(f"Knowledge graph statistics: {stats}")
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get graph statistics: {e}")
            raise
    
    def close(self) -> None:
        """Close Neo4j connection."""
        try:
            if self.driver:
                self.driver.close()
                logger.info("Closed Neo4j connection")
                
        except Exception as e:
            logger.error(f"Failed to close Neo4j connection: {e}")
            raise
