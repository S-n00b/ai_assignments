"""
Neo4j Service Integration for Enterprise LLMOps Platform

This module provides Neo4j database integration as a dedicated service
within the Lenovo enterprise platform, including CRUD operations,
GraphRAG queries, and real-time graph analytics.

Key Features:
- Neo4j database connection management
- CRUD operations for graph data
- GraphRAG query execution
- Real-time graph analytics
- Integration with existing Lenovo data models
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Neo4j imports
try:
    from neo4j import GraphDatabase, AsyncGraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logging.warning("Neo4j not available. Install with: pip install neo4j")

# FastAPI imports
try:
    from fastapi import HTTPException
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logging.warning("FastAPI not available")

# Configure logging
logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of Neo4j queries"""
    READ = "read"
    WRITE = "write"
    GRAPHRAG = "graphrag"
    ANALYTICS = "analytics"


@dataclass
class Neo4jConfig:
    """Neo4j configuration"""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"
    max_connection_lifetime: int = 3600
    max_connection_pool_size: int = 50
    connection_acquisition_timeout: int = 60


@dataclass
class GraphQuery:
    """Graph query structure"""
    query: str
    parameters: Dict[str, Any]
    query_type: QueryType
    timeout: int = 30


@dataclass
class GraphResult:
    """Graph query result"""
    data: List[Dict[str, Any]]
    summary: Dict[str, Any]
    execution_time: float
    query_type: QueryType


class Neo4jService:
    """Neo4j service for enterprise LLMOps platform"""
    
    def __init__(self, config: Neo4jConfig):
        self.config = config
        self.driver: Optional[AsyncGraphDatabase] = None
        self.connected = False
        self.logger = logging.getLogger("neo4j_service")
        
        if not NEO4J_AVAILABLE:
            self.logger.error("Neo4j driver not available")
            raise ImportError("Neo4j driver not available. Install with: pip install neo4j")
    
    async def connect(self) -> bool:
        """Connect to Neo4j database"""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                max_connection_lifetime=self.config.max_connection_lifetime,
                max_connection_pool_size=self.config.max_connection_pool_size,
                connection_acquisition_timeout=self.config.connection_acquisition_timeout
            )
            
            # Test connection
            async with self.driver.session(database=self.config.database) as session:
                result = await session.run("RETURN 1 as test")
                await result.single()
            
            self.connected = True
            self.logger.info(f"Connected to Neo4j at {self.config.uri}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from Neo4j database"""
        if self.driver:
            await self.driver.close()
            self.connected = False
            self.logger.info("Disconnected from Neo4j")
    
    async def execute_query(self, graph_query: GraphQuery) -> GraphResult:
        """Execute a graph query"""
        if not self.connected or not self.driver:
            raise HTTPException(status_code=503, detail="Neo4j not connected")
        
        start_time = datetime.now()
        
        try:
            async with self.driver.session(database=self.config.database) as session:
                result = await session.run(
                    graph_query.query,
                    graph_query.parameters,
                    timeout=graph_query.timeout
                )
                
                # Collect results
                data = []
                async for record in result:
                    # Convert Neo4j records to dictionaries
                    record_dict = {}
                    for key, value in record.items():
                        if hasattr(value, '__dict__'):
                            # Handle Neo4j node/relationship objects
                            record_dict[key] = dict(value)
                        else:
                            record_dict[key] = value
                    data.append(record_dict)
                
                # Get query summary
                summary = await result.consume()
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return GraphResult(
                    data=data,
                    summary={
                        "nodes_created": summary.counters.nodes_created,
                        "relationships_created": summary.counters.relationships_created,
                        "properties_set": summary.counters.properties_set,
                        "nodes_deleted": summary.counters.nodes_deleted,
                        "relationships_deleted": summary.counters.relationships_deleted,
                        "labels_added": summary.counters.labels_added,
                        "labels_removed": summary.counters.labels_removed,
                        "indexes_added": summary.counters.indexes_added,
                        "indexes_removed": summary.counters.indexes_removed,
                        "constraints_added": summary.counters.constraints_added,
                        "constraints_removed": summary.counters.constraints_removed
                    },
                    execution_time=execution_time,
                    query_type=graph_query.query_type
                )
                
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get Neo4j service health status"""
        if not self.connected:
            return {
                "status": "disconnected",
                "uri": self.config.uri,
                "database": self.config.database,
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Test query to check health
            graph_query = GraphQuery(
                query="RETURN 'healthy' as status, datetime() as timestamp",
                parameters={},
                query_type=QueryType.READ
            )
            
            result = await self.execute_query(graph_query)
            
            return {
                "status": "healthy",
                "uri": self.config.uri,
                "database": self.config.database,
                "timestamp": datetime.now().isoformat(),
                "query_response": result.data[0] if result.data else None,
                "execution_time": result.execution_time
            }
            
        except Exception as e:
            return {
                "status": "error",
                "uri": self.config.uri,
                "database": self.config.database,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    async def get_database_info(self) -> Dict[str, Any]:
        """Get database information"""
        if not self.connected:
            raise HTTPException(status_code=503, detail="Neo4j not connected")
        
        queries = {
            "node_count": "MATCH (n) RETURN count(n) as count",
            "relationship_count": "MATCH ()-[r]->() RETURN count(r) as count",
            "labels": "CALL db.labels() YIELD label RETURN collect(label) as labels",
            "relationship_types": "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types",
            "indexes": "CALL db.indexes() YIELD description, state, type RETURN collect({description: description, state: state, type: type}) as indexes"
        }
        
        results = {}
        for key, query in queries.items():
            try:
                graph_query = GraphQuery(
                    query=query,
                    parameters={},
                    query_type=QueryType.READ
                )
                result = await self.execute_query(graph_query)
                results[key] = result.data[0] if result.data else None
            except Exception as e:
                results[key] = {"error": str(e)}
        
        return results
    
    async def execute_graphrag_query(self, query_text: str, limit: int = 10) -> Dict[str, Any]:
        """Execute GraphRAG-style query"""
        if not self.connected:
            raise HTTPException(status_code=503, detail="Neo4j not connected")
        
        # Simple GraphRAG query - find documents and related entities
        graphrag_query = """
        MATCH (d:document)
        WHERE d.text_content CONTAINS $query_text
        OPTIONAL MATCH (d)-[:CONTAINS]->(e:entity)
        OPTIONAL MATCH (d)-[:REFERENCES]->(c:concept)
        RETURN d, collect(DISTINCT e) as entities, collect(DISTINCT c) as concepts
        LIMIT $limit
        """
        
        graph_query = GraphQuery(
            query=graphrag_query,
            parameters={"query_text": query_text, "limit": limit},
            query_type=QueryType.GRAPHRAG
        )
        
        result = await self.execute_query(graph_query)
        
        return {
            "query": query_text,
            "results": result.data,
            "execution_time": result.execution_time,
            "total_results": len(result.data)
        }
    
    async def get_organizational_structure(self) -> Dict[str, Any]:
        """Get Lenovo organizational structure"""
        if not self.connected:
            raise HTTPException(status_code=503, detail="Neo4j not connected")
        
        # Query for organizational structure
        org_query = """
        MATCH (p:person)-[:reports_to]->(m:person)
        OPTIONAL MATCH (p)-[:works_in]->(d:department)
        RETURN p, m, d
        ORDER BY p.level, p.name
        LIMIT 100
        """
        
        graph_query = GraphQuery(
            query=org_query,
            parameters={},
            query_type=QueryType.READ
        )
        
        result = await self.execute_query(graph_query)
        
        return {
            "organizational_data": result.data,
            "execution_time": result.execution_time,
            "total_relationships": len(result.data)
        }
    
    async def get_b2b_client_data(self) -> Dict[str, Any]:
        """Get B2B client data"""
        if not self.connected:
            raise HTTPException(status_code=503, detail="Neo4j not connected")
        
        # Query for B2B client relationships
        client_query = """
        MATCH (c:client)-[:serves]-(s:solution)
        OPTIONAL MATCH (c)-[:works_in]-(emp:person)
        RETURN c, s, collect(emp) as client_employees
        LIMIT 50
        """
        
        graph_query = GraphQuery(
            query=client_query,
            parameters={},
            query_type=QueryType.READ
        )
        
        result = await self.execute_query(graph_query)
        
        return {
            "client_data": result.data,
            "execution_time": result.execution_time,
            "total_clients": len(result.data)
        }
    
    async def get_project_dependencies(self) -> Dict[str, Any]:
        """Get project dependency network"""
        if not self.connected:
            raise HTTPException(status_code=503, detail="Neo4j not connected")
        
        # Query for project dependencies
        project_query = """
        MATCH (p1:project)-[:depends_on]->(p2:project)
        OPTIONAL MATCH (p1)-[:participates_in]-(emp:person)
        RETURN p1, p2, collect(emp) as team_members
        LIMIT 50
        """
        
        graph_query = GraphQuery(
            query=project_query,
            parameters={},
            query_type=QueryType.READ
        )
        
        result = await self.execute_query(graph_query)
        
        return {
            "project_dependencies": result.data,
            "execution_time": result.execution_time,
            "total_dependencies": len(result.data)
        }
    
    async def execute_custom_query(self, query: str, parameters: Dict[str, Any] = None) -> GraphResult:
        """Execute custom Cypher query"""
        if not self.connected:
            raise HTTPException(status_code=503, detail="Neo4j not connected")
        
        graph_query = GraphQuery(
            query=query,
            parameters=parameters or {},
            query_type=QueryType.READ
        )
        
        return await self.execute_query(graph_query)


# Global Neo4j service instance
neo4j_service: Optional[Neo4jService] = None


async def initialize_neo4j_service(config: Neo4jConfig = None) -> Neo4jService:
    """Initialize Neo4j service"""
    global neo4j_service
    
    if config is None:
        config = Neo4jConfig()
    
    neo4j_service = Neo4jService(config)
    await neo4j_service.connect()
    
    return neo4j_service


async def get_neo4j_service() -> Neo4jService:
    """Get Neo4j service instance"""
    global neo4j_service
    
    if neo4j_service is None:
        raise HTTPException(status_code=503, detail="Neo4j service not initialized")
    
    return neo4j_service


async def shutdown_neo4j_service():
    """Shutdown Neo4j service"""
    global neo4j_service
    
    if neo4j_service:
        await neo4j_service.disconnect()
        neo4j_service = None
