"""
Neo4j UI with Faker Data Integration for Realistic GraphRAG Demo

This module provides comprehensive Neo4j integration with Faker-generated data
for creating realistic GraphRAG demonstrations. It includes interactive knowledge
graph visualization, realistic data generation, and domain-specific data patterns.

Key Features:
- Interactive knowledge graph visualization with Neo4j UI
- Faker-generated realistic data for GraphRAG demos
- Domain-specific data generators (enterprise, healthcare, finance)
- Realistic user profiles, relationships, and interactions
- Temporal data generation for time-series analysis
- Graph-based model relationship mapping
- Natural language query interface
- Export capabilities for graph data and visualizations
"""

import asyncio
import json
import uuid
import os
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Callable, TypedDict, Annotated
from dataclasses import dataclass, field
from enum import Enum
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib

# Neo4j imports
try:
    from neo4j import GraphDatabase
    from py2neo import Graph, Node, Relationship, NodeMatcher, RelationshipMatcher
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logging.warning("Neo4j not available. Install with: pip install neo4j py2neo")

# Faker imports
try:
    from faker import Faker
    from faker.providers import internet, company, person, address, lorem, date_time
    FAKER_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False
    logging.warning("Faker not available. Install with: pip install faker")

# FastAPI imports for API integration
try:
    from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logging.warning("FastAPI not available for API integration")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DomainType(Enum):
    """Domain types for data generation"""
    ENTERPRISE = "enterprise"
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    EDUCATION = "education"
    RETAIL = "retail"
    MANUFACTURING = "manufacturing"
    TECHNOLOGY = "technology"


class NodeType(Enum):
    """Node types in the knowledge graph"""
    USER = "user"
    ORGANIZATION = "organization"
    PROJECT = "project"
    DOCUMENT = "document"
    MODEL = "model"
    TASK = "task"
    METRIC = "metric"
    INTERACTION = "interaction"
    RELATIONSHIP = "relationship"
    EVENT = "event"
    PRODUCT = "product"
    SERVICE = "service"


class RelationshipType(Enum):
    """Relationship types in the knowledge graph"""
    WORKS_FOR = "works_for"
    COLLABORATES_WITH = "collaborates_with"
    MANAGES = "manages"
    REPORTS_TO = "reports_to"
    PARTICIPATES_IN = "participates_in"
    CREATES = "creates"
    USES = "uses"
    EVALUATES = "evaluates"
    IMPROVES = "improves"
    DEPENDS_ON = "depends_on"
    INTERACTS_WITH = "interacts_with"
    BELONGS_TO = "belongs_to"
    CONTAINS = "contains"
    REFERENCES = "references"


@dataclass
class FakerConfig:
    """Configuration for Faker data generation"""
    domain: DomainType = DomainType.ENTERPRISE
    num_users: int = 100
    num_organizations: int = 20
    num_projects: int = 50
    num_documents: int = 200
    num_interactions: int = 500
    num_relationships: int = 300
    include_temporal_data: bool = True
    date_range_days: int = 365
    seed: Optional[int] = None
    locale: str = "en_US"


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph"""
    id: str
    label: str
    node_type: NodeType
    properties: Dict[str, Any]
    position: Optional[tuple] = None
    color: Optional[str] = None
    size: Optional[float] = None


@dataclass
class GraphEdge:
    """Represents an edge in the knowledge graph"""
    id: str
    source: str
    target: str
    relationship_type: RelationshipType
    properties: Dict[str, Any]
    weight: Optional[float] = None
    color: Optional[str] = None


@dataclass
class GraphData:
    """Complete graph data structure"""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    metadata: Dict[str, Any]
    created_at: datetime
    config: FakerConfig


class Neo4jFakerManager:
    """Main Neo4j Faker integration manager"""
    
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687", 
                 neo4j_user: str = "neo4j", neo4j_password: str = "password",
                 base_path: str = "neo4j_data"):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        self.graph: Optional[Graph] = None
        self.fake: Optional[Faker] = None
        self.logger = logging.getLogger("neo4j_faker_manager")
        
        # Initialize Faker
        if FAKER_AVAILABLE:
            self.fake = Faker()
            self.fake.add_provider(internet)
            self.fake.add_provider(company)
            self.fake.add_provider(person)
            self.fake.add_provider(address)
            self.fake.add_provider(lorem)
            self.fake.add_provider(date_time)
        
        # Initialize Neo4j connection
        if NEO4J_AVAILABLE:
            self._initialize_neo4j()
    
    def _initialize_neo4j(self):
        """Initialize Neo4j connection"""
        try:
            self.graph = Graph(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
            self.logger.info("Neo4j connection initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to connect to Neo4j: {e}")
            self.graph = None
    
    def generate_realistic_data(self, config: FakerConfig) -> GraphData:
        """Generate realistic data using Faker based on configuration"""
        if not FAKER_AVAILABLE:
            raise RuntimeError("Faker not available")
        
        # Set seed for reproducible data
        if config.seed:
            Faker.seed(config.seed)
        
        nodes = []
        edges = []
        
        # Generate organizations
        organizations = self._generate_organizations(config.num_organizations, config.domain)
        nodes.extend(organizations)
        
        # Generate users
        users = self._generate_users(config.num_users, config.domain, organizations)
        nodes.extend(users)
        
        # Generate projects
        projects = self._generate_projects(config.num_projects, config.domain, organizations, users)
        nodes.extend(projects)
        
        # Generate documents
        documents = self._generate_documents(config.num_documents, config.domain, projects, users)
        nodes.extend(documents)
        
        # Generate relationships
        relationships = self._generate_relationships(config.num_relationships, users, organizations, projects, documents)
        edges.extend(relationships)
        
        # Generate interactions
        interactions = self._generate_interactions(config.num_interactions, users, documents, config)
        edges.extend(interactions)
        
        metadata = {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "domain": config.domain.value,
            "generation_time": datetime.now().isoformat(),
            "faker_version": "latest"
        }
        
        return GraphData(
            nodes=nodes,
            edges=edges,
            metadata=metadata,
            created_at=datetime.now(),
            config=config
        )
    
    def _generate_organizations(self, count: int, domain: DomainType) -> List[GraphNode]:
        """Generate realistic organizations"""
        organizations = []
        
        for i in range(count):
            org_id = f"org_{uuid.uuid4().hex[:8]}"
            
            if domain == DomainType.ENTERPRISE:
                name = self.fake.company()
                industry = random.choice(["Technology", "Finance", "Healthcare", "Manufacturing", "Retail"])
            elif domain == DomainType.HEALTHCARE:
                name = self.fake.company() + " Medical Center"
                industry = "Healthcare"
            elif domain == DomainType.FINANCE:
                name = self.fake.company() + " Bank"
                industry = "Finance"
            else:
                name = self.fake.company()
                industry = domain.value.title()
            
            properties = {
                "name": name,
                "industry": industry,
                "size": random.choice(["Small", "Medium", "Large", "Enterprise"]),
                "founded_year": self.fake.year(),
                "revenue": self.fake.random_int(min=1000000, max=1000000000),
                "employees": self.fake.random_int(min=10, max=10000),
                "location": self.fake.city(),
                "website": self.fake.url(),
                "description": self.fake.text(max_nb_chars=200)
            }
            
            node = GraphNode(
                id=org_id,
                label=name,
                node_type=NodeType.ORGANIZATION,
                properties=properties,
                color="#FF6B6B",
                size=random.uniform(0.8, 1.2)
            )
            organizations.append(node)
        
        return organizations
    
    def _generate_users(self, count: int, domain: DomainType, organizations: List[GraphNode]) -> List[GraphNode]:
        """Generate realistic users"""
        users = []
        
        for i in range(count):
            user_id = f"user_{uuid.uuid4().hex[:8]}"
            
            # Assign to organization
            org = random.choice(organizations) if organizations else None
            
            if domain == DomainType.HEALTHCARE:
                roles = ["Doctor", "Nurse", "Administrator", "Technician", "Researcher"]
            elif domain == DomainType.FINANCE:
                roles = ["Analyst", "Manager", "Director", "Consultant", "Advisor"]
            elif domain == DomainType.TECHNOLOGY:
                roles = ["Developer", "Engineer", "Architect", "Manager", "Designer"]
            else:
                roles = ["Manager", "Analyst", "Specialist", "Coordinator", "Director"]
            
            properties = {
                "name": self.fake.name(),
                "email": self.fake.email(),
                "role": random.choice(roles),
                "department": self.fake.random_element(elements=("Engineering", "Marketing", "Sales", "HR", "Finance")),
                "experience_years": self.fake.random_int(min=1, max=20),
                "skills": [self.fake.word() for _ in range(random.randint(3, 8))],
                "location": self.fake.city(),
                "phone": self.fake.phone_number(),
                "hire_date": self.fake.date_between(start_date="-5y", end_date="today").isoformat(),
                "organization_id": org.id if org else None
            }
            
            node = GraphNode(
                id=user_id,
                label=properties["name"],
                node_type=NodeType.USER,
                properties=properties,
                color="#4ECDC4",
                size=random.uniform(0.6, 1.0)
            )
            users.append(node)
        
        return users
    
    def _generate_projects(self, count: int, domain: DomainType, organizations: List[GraphNode], users: List[GraphNode]) -> List[GraphNode]:
        """Generate realistic projects"""
        projects = []
        
        for i in range(count):
            project_id = f"project_{uuid.uuid4().hex[:8]}"
            
            # Assign to organization and users
            org = random.choice(organizations) if organizations else None
            project_lead = random.choice(users) if users else None
            
            if domain == DomainType.HEALTHCARE:
                project_types = ["Clinical Trial", "Research Study", "Quality Improvement", "Digital Health"]
            elif domain == DomainType.FINANCE:
                project_types = ["Risk Assessment", "Compliance", "Digital Transformation", "Analytics"]
            elif domain == DomainType.TECHNOLOGY:
                project_types = ["Software Development", "Infrastructure", "AI/ML", "Security"]
            else:
                project_types = ["Research", "Development", "Implementation", "Optimization"]
            
            start_date = self.fake.date_between(start_date="-2y", end_date="today")
            end_date = start_date + timedelta(days=random.randint(30, 365))
            
            properties = {
                "name": f"{random.choice(project_types)} - {self.fake.word().title()}",
                "description": self.fake.text(max_nb_chars=300),
                "status": random.choice(["Planning", "Active", "On Hold", "Completed"]),
                "priority": random.choice(["Low", "Medium", "High", "Critical"]),
                "budget": self.fake.random_int(min=10000, max=1000000),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "organization_id": org.id if org else None,
                "project_lead_id": project_lead.id if project_lead else None,
                "team_size": self.fake.random_int(min=2, max=20),
                "technologies": [self.fake.word() for _ in range(random.randint(2, 6))]
            }
            
            node = GraphNode(
                id=project_id,
                label=properties["name"],
                node_type=NodeType.PROJECT,
                properties=properties,
                color="#45B7D1",
                size=random.uniform(0.7, 1.1)
            )
            projects.append(node)
        
        return projects
    
    def _generate_documents(self, count: int, domain: DomainType, projects: List[GraphNode], users: List[GraphNode]) -> List[GraphNode]:
        """Generate realistic documents"""
        documents = []
        
        for i in range(count):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}"
            
            # Assign to project and author
            project = random.choice(projects) if projects else None
            author = random.choice(users) if users else None
            
            if domain == DomainType.HEALTHCARE:
                doc_types = ["Medical Report", "Research Paper", "Clinical Protocol", "Patient Record"]
            elif domain == DomainType.FINANCE:
                doc_types = ["Financial Report", "Risk Analysis", "Compliance Document", "Audit Report"]
            elif domain == DomainType.TECHNOLOGY:
                doc_types = ["Technical Specification", "Code Documentation", "Architecture Design", "Test Plan"]
            else:
                doc_types = ["Report", "Proposal", "Analysis", "Documentation"]
            
            properties = {
                "title": f"{random.choice(doc_types)} - {self.fake.sentence(nb_words=4)}",
                "content": self.fake.text(max_nb_chars=1000),
                "document_type": random.choice(doc_types),
                "version": f"v{random.randint(1, 5)}.{random.randint(0, 9)}",
                "created_date": self.fake.date_between(start_date="-1y", end_date="today").isoformat(),
                "modified_date": self.fake.date_between(start_date="-6m", end_date="today").isoformat(),
                "author_id": author.id if author else None,
                "project_id": project.id if project else None,
                "tags": [self.fake.word() for _ in range(random.randint(2, 5))],
                "file_size": self.fake.random_int(min=1024, max=10485760),  # 1KB to 10MB
                "access_level": random.choice(["Public", "Internal", "Confidential", "Restricted"])
            }
            
            node = GraphNode(
                id=doc_id,
                label=properties["title"][:50] + "..." if len(properties["title"]) > 50 else properties["title"],
                node_type=NodeType.DOCUMENT,
                properties=properties,
                color="#96CEB4",
                size=random.uniform(0.5, 0.9)
            )
            documents.append(node)
        
        return documents
    
    def _generate_relationships(self, count: int, users: List[GraphNode], organizations: List[GraphNode], 
                              projects: List[GraphNode], documents: List[GraphNode]) -> List[GraphEdge]:
        """Generate realistic relationships"""
        edges = []
        
        # User-Organization relationships
        for user in users:
            if user.properties.get("organization_id"):
                org = next((o for o in organizations if o.id == user.properties["organization_id"]), None)
                if org:
                    edge = GraphEdge(
                        id=f"rel_{uuid.uuid4().hex[:8]}",
                        source=user.id,
                        target=org.id,
                        relationship_type=RelationshipType.WORKS_FOR,
                        properties={
                            "start_date": user.properties.get("hire_date"),
                            "role": user.properties.get("role"),
                            "department": user.properties.get("department")
                        },
                        weight=1.0,
                        color="#FF6B6B"
                    )
                    edges.append(edge)
        
        # User-Project relationships
        for project in projects:
            if project.properties.get("project_lead_id"):
                lead = next((u for u in users if u.id == project.properties["project_lead_id"]), None)
                if lead:
                    edge = GraphEdge(
                        id=f"rel_{uuid.uuid4().hex[:8]}",
                        source=lead.id,
                        target=project.id,
                        relationship_type=RelationshipType.MANAGES,
                        properties={
                            "role": "Project Lead",
                            "start_date": project.properties.get("start_date")
                        },
                        weight=0.8,
                        color="#45B7D1"
                    )
                    edges.append(edge)
        
        # Document-Project relationships
        for document in documents:
            if document.properties.get("project_id"):
                project = next((p for p in projects if p.id == document.properties["project_id"]), None)
                if project:
                    edge = GraphEdge(
                        id=f"rel_{uuid.uuid4().hex[:8]}",
                        source=document.id,
                        target=project.id,
                        relationship_type=RelationshipType.BELONGS_TO,
                        properties={
                            "created_date": document.properties.get("created_date")
                        },
                        weight=0.6,
                        color="#96CEB4"
                    )
                    edges.append(edge)
        
        # User-User collaboration relationships
        for i in range(min(count // 4, len(users) // 2)):
            user1, user2 = random.sample(users, 2)
            edge = GraphEdge(
                id=f"rel_{uuid.uuid4().hex[:8]}",
                source=user1.id,
                target=user2.id,
                relationship_type=RelationshipType.COLLABORATES_WITH,
                properties={
                    "collaboration_type": random.choice(["Project", "Research", "Development"]),
                    "frequency": random.choice(["Daily", "Weekly", "Monthly"])
                },
                weight=random.uniform(0.3, 0.8),
                color="#4ECDC4"
            )
            edges.append(edge)
        
        return edges
    
    def _generate_interactions(self, count: int, users: List[GraphNode], documents: List[GraphNode], config: FakerConfig) -> List[GraphEdge]:
        """Generate realistic interactions"""
        edges = []
        
        for i in range(count):
            user = random.choice(users)
            document = random.choice(documents)
            
            interaction_types = ["viewed", "edited", "commented", "shared", "downloaded"]
            interaction_type = random.choice(interaction_types)
            
            # Generate temporal data if enabled
            if config.include_temporal_data:
                start_date = datetime.now() - timedelta(days=config.date_range_days)
                interaction_date = self.fake.date_time_between(start_date=start_date, end_date="now")
            else:
                interaction_date = self.fake.date_time_between(start_date="-1y", end_date="now")
            
            edge = GraphEdge(
                id=f"interaction_{uuid.uuid4().hex[:8]}",
                source=user.id,
                target=document.id,
                relationship_type=RelationshipType.INTERACTS_WITH,
                properties={
                    "interaction_type": interaction_type,
                    "timestamp": interaction_date.isoformat(),
                    "duration_seconds": self.fake.random_int(min=10, max=3600),
                    "device": random.choice(["Desktop", "Mobile", "Tablet"]),
                    "location": self.fake.city()
                },
                weight=random.uniform(0.2, 0.7),
                color="#FFEAA7"
            )
            edges.append(edge)
        
        return edges
    
    async def save_to_neo4j(self, graph_data: GraphData) -> bool:
        """Save graph data to Neo4j database"""
        if not NEO4J_AVAILABLE or not self.graph:
            self.logger.warning("Neo4j not available, skipping save")
            return False
        
        try:
            # Clear existing data
            self.graph.delete_all()
            
            # Create nodes
            for node in graph_data.nodes:
                neo4j_node = Node(
                    node.node_type.value,
                    **node.properties
                )
                neo4j_node["id"] = node.id
                neo4j_node["label"] = node.label
                self.graph.create(neo4j_node)
            
            # Create relationships
            for edge in graph_data.edges:
                source_node = self.graph.nodes.match(id=edge.source).first()
                target_node = self.graph.nodes.match(id=edge.target).first()
                
                if source_node and target_node:
                    relationship = Relationship(
                        source_node,
                        edge.relationship_type.value,
                        target_node,
                        **edge.properties
                    )
                    relationship["id"] = edge.id
                    relationship["weight"] = edge.weight
                    self.graph.create(relationship)
            
            self.logger.info(f"Successfully saved {len(graph_data.nodes)} nodes and {len(graph_data.edges)} edges to Neo4j")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save to Neo4j: {e}")
            return False
    
    async def query_neo4j(self, cypher_query: str) -> List[Dict[str, Any]]:
        """Execute Cypher query on Neo4j database"""
        if not NEO4J_AVAILABLE or not self.graph:
            return []
        
        try:
            result = self.graph.run(cypher_query)
            return [dict(record) for record in result]
        except Exception as e:
            self.logger.error(f"Failed to execute Cypher query: {e}")
            return []
    
    def export_graph_data(self, graph_data: GraphData, format: str = "json") -> str:
        """Export graph data in various formats"""
        if format == "json":
            return json.dumps({
                "nodes": [asdict(node) for node in graph_data.nodes],
                "edges": [asdict(edge) for edge in graph_data.edges],
                "metadata": graph_data.metadata
            }, indent=2, default=str)
        elif format == "cypher":
            # Generate Cypher CREATE statements
            cypher_statements = []
            
            # Create nodes
            for node in graph_data.nodes:
                properties = {k: v for k, v in node.properties.items()}
                properties["id"] = node.id
                properties["label"] = node.label
                
                props_str = ", ".join([f"{k}: {json.dumps(v)}" for k, v in properties.items()])
                cypher_statements.append(f"CREATE (n:{node.node_type.value} {{{props_str}}})")
            
            # Create relationships
            for edge in graph_data.edges:
                properties = {k: v for k, v in edge.properties.items()}
                properties["id"] = edge.id
                properties["weight"] = edge.weight
                
                props_str = ", ".join([f"{k}: {json.dumps(v)}" for k, v in properties.items()])
                cypher_statements.append(
                    f"MATCH (a), (b) WHERE a.id = '{edge.source}' AND b.id = '{edge.target}' "
                    f"CREATE (a)-[r:{edge.relationship_type.value} {{{props_str}}}]->(b)"
                )
            
            return "\n".join(cypher_statements)
        
        return ""


# Global Neo4j Faker manager instance
neo4j_faker_manager: Optional[Neo4jFakerManager] = None


def initialize_neo4j_faker_manager(neo4j_uri: str = "bolt://localhost:7687",
                                  neo4j_user: str = "neo4j", 
                                  neo4j_password: str = "password",
                                  base_path: str = "neo4j_data") -> Neo4jFakerManager:
    """Initialize the global Neo4j Faker manager"""
    global neo4j_faker_manager
    neo4j_faker_manager = Neo4jFakerManager(neo4j_uri, neo4j_user, neo4j_password, base_path)
    return neo4j_faker_manager


def get_neo4j_faker_manager() -> Neo4jFakerManager:
    """Get the global Neo4j Faker manager"""
    if neo4j_faker_manager is None:
        raise RuntimeError("Neo4j Faker manager not initialized")
    return neo4j_faker_manager


# FastAPI integration functions
if FASTAPI_AVAILABLE:
    class FakerConfigRequest(BaseModel):
        """Request model for Faker configuration"""
        domain: str = "enterprise"
        num_users: int = 100
        num_organizations: int = 20
        num_projects: int = 50
        num_documents: int = 200
        num_interactions: int = 500
        num_relationships: int = 300
        include_temporal_data: bool = True
        date_range_days: int = 365
        seed: Optional[int] = None
        locale: str = "en_US"
    
    class GraphDataResponse(BaseModel):
        """Response model for graph data"""
        nodes: List[Dict[str, Any]]
        edges: List[Dict[str, Any]]
        metadata: Dict[str, Any]
        created_at: str
    
    class CypherQueryRequest(BaseModel):
        """Request model for Cypher queries"""
        query: str
    
    def create_neo4j_faker_endpoints(app: FastAPI):
        """Create Neo4j Faker API endpoints for FastAPI app"""
        
        @app.get("/api/neo4j-faker/status")
        async def get_neo4j_faker_status():
            """Get Neo4j Faker integration status"""
            manager = get_neo4j_faker_manager()
            return {
                "neo4j_available": NEO4J_AVAILABLE and manager.graph is not None,
                "faker_available": FAKER_AVAILABLE,
                "neo4j_uri": manager.neo4j_uri,
                "status": "ready" if (NEO4J_AVAILABLE and FAKER_AVAILABLE) else "missing_dependencies"
            }
        
        @app.post("/api/neo4j-faker/generate", response_model=GraphDataResponse)
        async def generate_faker_data(request: FakerConfigRequest):
            """Generate realistic data using Faker"""
            manager = get_neo4j_faker_manager()
            
            config = FakerConfig(
                domain=DomainType(request.domain),
                num_users=request.num_users,
                num_organizations=request.num_organizations,
                num_projects=request.num_projects,
                num_documents=request.num_documents,
                num_interactions=request.num_interactions,
                num_relationships=request.num_relationships,
                include_temporal_data=request.include_temporal_data,
                date_range_days=request.date_range_days,
                seed=request.seed,
                locale=request.locale
            )
            
            graph_data = manager.generate_realistic_data(config)
            
            return GraphDataResponse(
                nodes=[asdict(node) for node in graph_data.nodes],
                edges=[asdict(edge) for edge in graph_data.edges],
                metadata=graph_data.metadata,
                created_at=graph_data.created_at.isoformat()
            )
        
        @app.post("/api/neo4j-faker/save")
        async def save_to_neo4j(request: FakerConfigRequest):
            """Generate and save data to Neo4j"""
            manager = get_neo4j_faker_manager()
            
            config = FakerConfig(
                domain=DomainType(request.domain),
                num_users=request.num_users,
                num_organizations=request.num_organizations,
                num_projects=request.num_projects,
                num_documents=request.num_documents,
                num_interactions=request.num_interactions,
                num_relationships=request.num_relationships,
                include_temporal_data=request.include_temporal_data,
                date_range_days=request.date_range_days,
                seed=request.seed,
                locale=request.locale
            )
            
            graph_data = manager.generate_realistic_data(config)
            success = await manager.save_to_neo4j(graph_data)
            
            return {
                "success": success,
                "nodes_created": len(graph_data.nodes),
                "edges_created": len(graph_data.edges),
                "message": "Data saved to Neo4j successfully" if success else "Failed to save to Neo4j"
            }
        
        @app.post("/api/neo4j-faker/query")
        async def query_neo4j(request: CypherQueryRequest):
            """Execute Cypher query on Neo4j"""
            manager = get_neo4j_faker_manager()
            results = await manager.query_neo4j(request.query)
            return {"results": results, "count": len(results)}
        
        @app.get("/api/neo4j-faker/dashboard", response_class=HTMLResponse)
        async def get_neo4j_faker_dashboard():
            """Get Neo4j Faker dashboard UI"""
            return HTMLResponse("""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Neo4j Faker Dashboard - GraphRAG Demo</title>
                <script src="https://cdn.tailwindcss.com"></script>
                <script src="https://d3js.org/d3.v7.min.js"></script>
                <style>
                    .lenovo-gradient {
                        background: linear-gradient(135deg, #E2231A, #C01E17);
                    }
                    .lenovo-text-gradient {
                        background: linear-gradient(90deg, #E2231A, #0066CC);
                        background-clip: text;
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                    }
                </style>
            </head>
            <body class="bg-gray-900 text-white">
                <div class="container mx-auto px-4 py-8">
                    <div class="lenovo-gradient text-white p-6 rounded-lg mb-8">
                        <h1 class="text-3xl font-bold mb-2">🎯 Neo4j Faker Dashboard</h1>
                        <p class="text-lg opacity-90">Realistic GraphRAG Demo with Faker-Generated Data</p>
                    </div>
                    
                    <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
                        <!-- Configuration Panel -->
                        <div class="bg-gray-800 p-6 rounded-lg">
                            <h2 class="text-xl font-semibold mb-4">🔧 Data Generation Configuration</h2>
                            
                            <div class="space-y-4">
                                <div>
                                    <label class="block text-sm font-medium mb-2">Domain</label>
                                    <select id="domain" class="w-full p-2 bg-gray-700 border border-gray-600 rounded text-white">
                                        <option value="enterprise">Enterprise</option>
                                        <option value="healthcare">Healthcare</option>
                                        <option value="finance">Finance</option>
                                        <option value="technology">Technology</option>
                                    </select>
                                </div>
                                
                                <div class="grid grid-cols-2 gap-4">
                                    <div>
                                        <label class="block text-sm font-medium mb-2">Users</label>
                                        <input type="number" id="num_users" value="100" class="w-full p-2 bg-gray-700 border border-gray-600 rounded text-white">
                                    </div>
                                    <div>
                                        <label class="block text-sm font-medium mb-2">Organizations</label>
                                        <input type="number" id="num_organizations" value="20" class="w-full p-2 bg-gray-700 border border-gray-600 rounded text-white">
                                    </div>
                                </div>
                                
                                <div class="grid grid-cols-2 gap-4">
                                    <div>
                                        <label class="block text-sm font-medium mb-2">Projects</label>
                                        <input type="number" id="num_projects" value="50" class="w-full p-2 bg-gray-700 border border-gray-600 rounded text-white">
                                    </div>
                                    <div>
                                        <label class="block text-sm font-medium mb-2">Documents</label>
                                        <input type="number" id="num_documents" value="200" class="w-full p-2 bg-gray-700 border border-gray-600 rounded text-white">
                                    </div>
                                </div>
                                
                                <div class="flex items-center space-x-4">
                                    <label class="flex items-center">
                                        <input type="checkbox" id="include_temporal" checked class="mr-2">
                                        <span>Include Temporal Data</span>
                                    </label>
                                    <label class="flex items-center">
                                        <input type="checkbox" id="save_to_neo4j" class="mr-2">
                                        <span>Save to Neo4j</span>
                                    </label>
                                </div>
                                
                                <button onclick="generateData()" class="w-full lenovo-gradient text-white py-2 px-4 rounded hover:opacity-90 transition">
                                    🚀 Generate Realistic Data
                                </button>
                            </div>
                        </div>
                        
                        <!-- Graph Visualization -->
                        <div class="bg-gray-800 p-6 rounded-lg">
                            <h2 class="text-xl font-semibold mb-4">📊 Graph Visualization</h2>
                            <div id="graph-container" class="w-full h-96 border border-gray-600 rounded bg-gray-900">
                                <div class="flex items-center justify-center h-full text-gray-400">
                                    Generate data to see the graph visualization
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Status and Results -->
                    <div class="mt-8 bg-gray-800 p-6 rounded-lg">
                        <h2 class="text-xl font-semibold mb-4">📈 Generation Status</h2>
                        <div id="status" class="text-gray-300">
                            Ready to generate realistic GraphRAG demo data
                        </div>
                    </div>
                </div>
                
                <script>
                    async function generateData() {
                        const statusDiv = document.getElementById('status');
                        statusDiv.innerHTML = '🔄 Generating realistic data...';
                        
                        const config = {
                            domain: document.getElementById('domain').value,
                            num_users: parseInt(document.getElementById('num_users').value),
                            num_organizations: parseInt(document.getElementById('num_organizations').value),
                            num_projects: parseInt(document.getElementById('num_projects').value),
                            num_documents: parseInt(document.getElementById('num_documents').value),
                            num_interactions: 500,
                            num_relationships: 300,
                            include_temporal_data: document.getElementById('include_temporal').checked,
                            date_range_days: 365,
                            seed: Math.floor(Math.random() * 10000)
                        };
                        
                        try {
                            const endpoint = document.getElementById('save_to_neo4j').checked ? 
                                '/api/neo4j-faker/save' : '/api/neo4j-faker/generate';
                            
                            const response = await fetch(endpoint, {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json',
                                },
                                body: JSON.stringify(config)
                            });
                            
                            const result = await response.json();
                            
                            if (response.ok) {
                                statusDiv.innerHTML = `✅ Successfully generated ${result.nodes?.length || result.nodes_created} nodes and ${result.edges?.length || result.edges_created} edges`;
                                
                                if (result.nodes) {
                                    visualizeGraph(result.nodes, result.edges);
                                }
                            } else {
                                statusDiv.innerHTML = `❌ Error: ${result.detail || 'Failed to generate data'}`;
                            }
                        } catch (error) {
                            statusDiv.innerHTML = `❌ Error: ${error.message}`;
                        }
                    }
                    
                    function visualizeGraph(nodes, edges) {
                        const container = document.getElementById('graph-container');
                        container.innerHTML = '';
                        
                        const width = container.clientWidth;
                        const height = container.clientHeight;
                        
                        const svg = d3.select(container)
                            .append('svg')
                            .attr('width', width)
                            .attr('height', height);
                        
                        // Simple force-directed graph visualization
                        const simulation = d3.forceSimulation(nodes)
                            .force('link', d3.forceLink(edges).id(d => d.id).distance(100))
                            .force('charge', d3.forceManyBody().strength(-300))
                            .force('center', d3.forceCenter(width / 2, height / 2));
                        
                        const link = svg.append('g')
                            .selectAll('line')
                            .data(edges)
                            .enter().append('line')
                            .attr('stroke', '#666')
                            .attr('stroke-opacity', 0.6)
                            .attr('stroke-width', d => Math.sqrt(d.weight || 1));
                        
                        const node = svg.append('g')
                            .selectAll('circle')
                            .data(nodes)
                            .enter().append('circle')
                            .attr('r', d => (d.size || 1) * 5)
                            .attr('fill', d => d.color || '#4ECDC4')
                            .attr('stroke', '#fff')
                            .attr('stroke-width', 2);
                        
                        simulation.on('tick', () => {
                            link
                                .attr('x1', d => d.source.x)
                                .attr('y1', d => d.source.y)
                                .attr('x2', d => d.target.x)
                                .attr('y2', d => d.target.y);
                            
                            node
                                .attr('cx', d => d.x)
                                .attr('cy', d => d.y);
                        });
                    }
                </script>
            </body>
            </html>
            """)

