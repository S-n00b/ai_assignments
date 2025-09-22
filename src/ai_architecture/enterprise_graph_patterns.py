"""
Enterprise Graph Patterns for Lenovo GraphRAG Solutions

This module implements common enterprise graph patterns used in GraphRAG solutions,
including organizational hierarchies, project dependencies, knowledge graphs,
and business process flows optimized for Lenovo's enterprise needs.

Key Patterns:
- Organizational Hierarchy Patterns
- Project Dependency Networks
- Knowledge Graph Patterns
- Business Process Flows
- Customer Journey Mapping
- Technology Stack Relationships
- Compliance and Risk Networks
"""

import asyncio
import json
import uuid
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

# Neo4j imports
try:
    from neo4j import GraphDatabase
    from py2neo import Graph, Node, Relationship
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logging.warning("Neo4j not available. Install with: pip install neo4j py2neo")

# Faker imports
try:
    from faker import Faker
    from faker.providers import company, person, address, lorem, date_time, internet
    FAKER_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False
    logging.warning("Faker not available. Install with: pip install faker")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnterprisePatternType(Enum):
    """Types of enterprise graph patterns"""
    ORG_HIERARCHY = "org_hierarchy"
    PROJECT_NETWORK = "project_network"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    BUSINESS_PROCESS = "business_process"
    CUSTOMER_JOURNEY = "customer_journey"
    TECHNOLOGY_STACK = "technology_stack"
    COMPLIANCE_NETWORK = "compliance_network"
    RISK_NETWORK = "risk_network"
    SUPPLY_CHAIN = "supply_chain"
    PARTNER_ECOSYSTEM = "partner_ecosystem"


class BusinessProcessStage(Enum):
    """Stages in business processes"""
    INITIATION = "initiation"
    PLANNING = "planning"
    EXECUTION = "execution"
    MONITORING = "monitoring"
    CLOSURE = "closure"
    MAINTENANCE = "maintenance"


class CustomerJourneyStage(Enum):
    """Stages in customer journey"""
    AWARENESS = "awareness"
    INTEREST = "interest"
    CONSIDERATION = "consideration"
    PURCHASE = "purchase"
    ONBOARDING = "onboarding"
    ADOPTION = "adoption"
    RETENTION = "retention"
    ADVOCACY = "advocacy"


@dataclass
class GraphNode:
    """Represents a node in the enterprise graph"""
    id: str
    label: str
    node_type: str
    properties: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class GraphEdge:
    """Represents an edge in the enterprise graph"""
    id: str
    source: str
    target: str
    relationship_type: str
    properties: Dict[str, Any]
    weight: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class EnterpriseGraphPattern:
    """Complete enterprise graph pattern"""
    pattern_type: EnterprisePatternType
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    metadata: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)


class EnterpriseGraphPatternGenerator:
    """Generator for enterprise graph patterns"""
    
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687", 
                 neo4j_user: str = "neo4j", neo4j_password: str = "password"):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.graph: Optional[Graph] = None
        
        # Initialize Faker
        if FAKER_AVAILABLE:
            self.fake = Faker()
            self.fake.add_provider(company)
            self.fake.add_provider(person)
            self.fake.add_provider(address)
            self.fake.add_provider(lorem)
            self.fake.add_provider(date_time)
            self.fake.add_provider(internet)
        
        # Initialize Neo4j connection
        if NEO4J_AVAILABLE:
            self._initialize_neo4j()
    
    def _initialize_neo4j(self):
        """Initialize Neo4j connection"""
        try:
            self.graph = Graph(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
            logger.info("Neo4j connection initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to connect to Neo4j: {e}")
            self.graph = None
    
    def generate_org_hierarchy_pattern(self, num_levels: int = 5, 
                                     employees_per_level: int = 10) -> EnterpriseGraphPattern:
        """Generate organizational hierarchy pattern"""
        nodes = []
        edges = []
        
        # Generate CEO
        ceo = GraphNode(
            id="ceo_1",
            label="Chief Executive Officer",
            node_type="executive",
            properties={
                "name": "CEO",
                "level": 0,
                "department": "Executive",
                "reports_to": None,
                "direct_reports": []
            }
        )
        nodes.append(ceo)
        
        current_level = [ceo]
        node_id_counter = 2
        
        for level in range(1, num_levels):
            next_level = []
            
            for manager in current_level:
                # Generate direct reports for this manager
                num_reports = random.randint(3, employees_per_level)
                
                for i in range(num_reports):
                    role = self._generate_role_by_level(level)
                    employee = GraphNode(
                        id=f"emp_{node_id_counter}",
                        label=role,
                        node_type="employee",
                        properties={
                            "name": role,
                            "level": level,
                            "department": self._generate_department_by_level(level),
                            "reports_to": manager.id,
                            "direct_reports": []
                        }
                    )
                    nodes.append(employee)
                    next_level.append(employee)
                    
                    # Create reporting relationship
                    edge = GraphEdge(
                        id=f"reports_to_{employee.id}_{manager.id}",
                        source=employee.id,
                        target=manager.id,
                        relationship_type="reports_to",
                        properties={
                            "relationship_type": "direct_report",
                            "level_difference": 1
                        }
                    )
                    edges.append(edge)
                    
                    node_id_counter += 1
            
            current_level = next_level
        
        metadata = {
            "pattern_type": "org_hierarchy",
            "total_levels": num_levels,
            "total_employees": len(nodes),
            "avg_reports_per_manager": employees_per_level
        }
        
        return EnterpriseGraphPattern(
            pattern_type=EnterprisePatternType.ORG_HIERARCHY,
            nodes=nodes,
            edges=edges,
            metadata=metadata
        )
    
    def generate_project_network_pattern(self, num_projects: int = 20) -> EnterpriseGraphPattern:
        """Generate project dependency network pattern"""
        nodes = []
        edges = []
        
        # Generate projects
        projects = []
        for i in range(num_projects):
            project = GraphNode(
                id=f"proj_{i+1}",
                label=f"Project {i+1}",
                node_type="project",
                properties={
                    "name": f"Project {i+1}",
                    "status": random.choice(["planning", "active", "on_hold", "completed"]),
                    "priority": random.choice(["low", "medium", "high", "critical"]),
                    "budget": random.randint(100000, 5000000),
                    "duration_weeks": random.randint(4, 52),
                    "team_size": random.randint(3, 20),
                    "complexity": random.choice(["low", "medium", "high"]),
                    "start_date": self.fake.date_between(start_date='-1y', end_date='today'),
                    "end_date": self.fake.date_between(start_date='today', end_date='+1y')
                }
            )
            projects.append(project)
            nodes.append(project)
        
        # Generate project dependencies
        for i, project in enumerate(projects):
            # Each project can depend on 0-3 other projects
            num_dependencies = random.randint(0, min(3, i))
            if num_dependencies > 0:
                dependencies = random.sample(projects[:i], num_dependencies)
                
                for dep_project in dependencies:
                    edge = GraphEdge(
                        id=f"depends_on_{project.id}_{dep_project.id}",
                        source=project.id,
                        target=dep_project.id,
                        relationship_type="depends_on",
                        properties={
                            "dependency_type": random.choice(["finish_to_start", "start_to_start", "finish_to_finish"]),
                            "lag_days": random.randint(0, 30),
                            "critical": random.choice([True, False])
                        }
                    )
                    edges.append(edge)
        
        # Generate project teams
        team_members = []
        for i in range(num_projects * 3):  # 3 team members per project on average
            member = GraphNode(
                id=f"member_{i+1}",
                label=f"Team Member {i+1}",
                node_type="team_member",
                properties={
                    "name": f"{self.fake.first_name()} {self.fake.last_name()}",
                    "role": random.choice(["developer", "analyst", "designer", "tester", "manager"]),
                    "skills": random.sample(["Python", "Java", "React", "AWS", "Docker"], random.randint(2, 4))
                }
            )
            team_members.append(member)
            nodes.append(member)
        
        # Assign team members to projects
        for project in projects:
            num_members = random.randint(2, 8)
            project_members = random.sample(team_members, min(num_members, len(team_members)))
            
            for member in project_members:
                edge = GraphEdge(
                    id=f"works_on_{member.id}_{project.id}",
                    source=member.id,
                    target=project.id,
                    relationship_type="works_on",
                    properties={
                        "allocation_percentage": random.randint(25, 100),
                        "role": random.choice(["lead", "contributor", "reviewer"]),
                        "start_date": self.fake.date_between(start_date='-6m', end_date='today')
                    }
                )
                edges.append(edge)
        
        metadata = {
            "pattern_type": "project_network",
            "total_projects": num_projects,
            "total_team_members": len(team_members),
            "avg_dependencies_per_project": len(edges) / num_projects
        }
        
        return EnterpriseGraphPattern(
            pattern_type=EnterprisePatternType.PROJECT_NETWORK,
            nodes=nodes,
            edges=edges,
            metadata=metadata
        )
    
    def generate_knowledge_graph_pattern(self, num_concepts: int = 50) -> EnterpriseGraphPattern:
        """Generate knowledge graph pattern for enterprise knowledge"""
        nodes = []
        edges = []
        
        # Generate knowledge concepts
        concept_categories = [
            "Technology", "Business Process", "Domain Knowledge", "Best Practice",
            "Compliance", "Security", "Performance", "Quality", "Innovation"
        ]
        
        concepts = []
        for i in range(num_concepts):
            category = random.choice(concept_categories)
            concept = GraphNode(
                id=f"concept_{i+1}",
                label=f"{category} Concept {i+1}",
                node_type="knowledge_concept",
                properties={
                    "name": f"{category} Concept {i+1}",
                    "category": category,
                    "description": self.fake.text(max_nb_chars=200),
                    "complexity": random.choice(["basic", "intermediate", "advanced"]),
                    "usage_frequency": random.randint(1, 100),
                    "last_updated": self.fake.date_between(start_date='-1y', end_date='today'),
                    "expertise_level": random.choice(["beginner", "intermediate", "expert"])
                }
            )
            concepts.append(concept)
            nodes.append(concept)
        
        # Generate concept relationships
        for i, concept in enumerate(concepts):
            # Each concept can relate to 2-5 other concepts
            remaining_concepts = len(concepts) - i - 1
            if remaining_concepts > 0:
                num_relationships = random.randint(1, min(5, remaining_concepts))
                related_concepts = random.sample(concepts[i+1:], num_relationships)
                
                for related_concept in related_concepts:
                    relationship_type = random.choice([
                        "prerequisite", "related_to", "part_of", "implements", "conflicts_with"
                    ])
                    
                    edge = GraphEdge(
                        id=f"relates_to_{concept.id}_{related_concept.id}",
                        source=concept.id,
                        target=related_concept.id,
                        relationship_type=relationship_type,
                        properties={
                            "strength": random.uniform(0.1, 1.0),
                            "confidence": random.uniform(0.5, 1.0),
                            "last_verified": self.fake.date_between(start_date='-6m', end_date='today')
                        }
                    )
                    edges.append(edge)
        
        # Generate knowledge sources
        sources = []
        for i in range(20):
            source = GraphNode(
                id=f"source_{i+1}",
                label=f"Knowledge Source {i+1}",
                node_type="knowledge_source",
                properties={
                    "name": f"Source {i+1}",
                    "type": random.choice(["document", "video", "training", "expert", "database"]),
                    "reliability": random.uniform(0.5, 1.0),
                    "last_updated": self.fake.date_between(start_date='-2y', end_date='today'),
                    "access_level": random.choice(["public", "internal", "confidential"])
                }
            )
            sources.append(source)
            nodes.append(source)
        
        # Link concepts to sources
        for concept in concepts:
            num_sources = random.randint(1, 5)
            concept_sources = random.sample(sources, min(num_sources, len(sources)))
            
            for source in concept_sources:
                edge = GraphEdge(
                    id=f"documented_in_{concept.id}_{source.id}",
                    source=concept.id,
                    target=source.id,
                    relationship_type="documented_in",
                    properties={
                        "relevance": random.uniform(0.3, 1.0),
                        "last_referenced": self.fake.date_between(start_date='-1y', end_date='today')
                    }
                )
                edges.append(edge)
        
        metadata = {
            "pattern_type": "knowledge_graph",
            "total_concepts": num_concepts,
            "total_sources": len(sources),
            "avg_relationships_per_concept": len(edges) / num_concepts
        }
        
        return EnterpriseGraphPattern(
            pattern_type=EnterprisePatternType.KNOWLEDGE_GRAPH,
            nodes=nodes,
            edges=edges,
            metadata=metadata
        )
    
    def generate_business_process_pattern(self, num_processes: int = 15) -> EnterpriseGraphPattern:
        """Generate business process flow pattern"""
        nodes = []
        edges = []
        
        # Generate business processes
        processes = []
        process_types = [
            "Customer Onboarding", "Product Development", "Sales Process", "Support Process",
            "Compliance Review", "Risk Assessment", "Quality Assurance", "Vendor Management",
            "Employee Onboarding", "Budget Planning", "Performance Review", "Incident Response"
        ]
        
        for i in range(num_processes):
            process_type = random.choice(process_types)
            process = GraphNode(
                id=f"process_{i+1}",
                label=process_type,
                node_type="business_process",
                properties={
                    "name": process_type,
                    "description": self.fake.text(max_nb_chars=300),
                    "owner": f"{self.fake.first_name()} {self.fake.last_name()}",
                    "frequency": random.choice(["daily", "weekly", "monthly", "quarterly", "annually"]),
                    "complexity": random.choice(["low", "medium", "high"]),
                    "automation_level": random.choice(["manual", "semi_automated", "fully_automated"]),
                    "compliance_required": random.choice([True, False]),
                    "sla_hours": random.randint(1, 168)  # 1 hour to 1 week
                }
            )
            processes.append(process)
            nodes.append(process)
        
        # Generate process stages
        stages = []
        for process in processes:
            num_stages = random.randint(3, 8)
            process_stages = []
            
            for j in range(num_stages):
                stage = GraphNode(
                    id=f"stage_{process.id}_{j+1}",
                    label=f"{process.properties['name']} - Stage {j+1}",
                    node_type="process_stage",
                    properties={
                        "name": f"Stage {j+1}",
                        "process_id": process.id,
                        "stage_type": random.choice(list(BusinessProcessStage)).value,
                        "duration_hours": random.randint(1, 48),
                        "responsible_role": random.choice(["manager", "analyst", "specialist", "coordinator"]),
                        "automated": random.choice([True, False]),
                        "requires_approval": random.choice([True, False])
                    }
                )
                process_stages.append(stage)
                stages.append(stage)
                nodes.append(stage)
            
            # Create stage sequence
            for j in range(len(process_stages) - 1):
                edge = GraphEdge(
                    id=f"follows_{process_stages[j+1].id}_{process_stages[j].id}",
                    source=process_stages[j].id,
                    target=process_stages[j+1].id,
                    relationship_type="follows",
                    properties={
                        "sequence_order": j + 1,
                        "transition_condition": random.choice(["automatic", "approval_required", "manual_trigger"]),
                        "avg_duration_hours": random.randint(1, 24)
                    }
                )
                edges.append(edge)
        
        # Generate process dependencies
        for i, process in enumerate(processes):
            num_dependencies = random.randint(0, min(3, i))
            if num_dependencies > 0:
                dependencies = random.sample(processes[:i], num_dependencies)
                
                for dep_process in dependencies:
                    edge = GraphEdge(
                        id=f"depends_on_{process.id}_{dep_process.id}",
                        source=process.id,
                        target=dep_process.id,
                        relationship_type="depends_on",
                        properties={
                            "dependency_type": random.choice(["prerequisite", "input", "approval"]),
                            "critical": random.choice([True, False])
                        }
                    )
                    edges.append(edge)
        
        metadata = {
            "pattern_type": "business_process",
            "total_processes": num_processes,
            "total_stages": len(stages),
            "avg_stages_per_process": len(stages) / num_processes
        }
        
        return EnterpriseGraphPattern(
            pattern_type=EnterprisePatternType.BUSINESS_PROCESS,
            nodes=nodes,
            edges=edges,
            metadata=metadata
        )
    
    def generate_customer_journey_pattern(self, num_customers: int = 30) -> EnterpriseGraphPattern:
        """Generate customer journey mapping pattern"""
        nodes = []
        edges = []
        
        # Generate customer personas
        personas = [
            "Enterprise IT Manager", "Small Business Owner", "Individual Consumer",
            "Government Procurement Officer", "Healthcare IT Director", "Educational Administrator"
        ]
        
        customers = []
        for i in range(num_customers):
            persona = random.choice(personas)
            customer = GraphNode(
                id=f"customer_{i+1}",
                label=f"Customer {i+1}",
                node_type="customer",
                properties={
                    "name": f"{self.fake.first_name()} {self.fake.last_name()}",
                    "persona": persona,
                    "company": self.fake.company(),
                    "industry": random.choice(["Technology", "Healthcare", "Finance", "Education", "Government"]),
                    "company_size": random.choice(["startup", "small", "medium", "large", "enterprise"]),
                    "budget_range": random.choice(["<10k", "10k-100k", "100k-1M", ">1M"]),
                    "decision_making_authority": random.choice(["individual", "team", "committee", "board"])
                }
            )
            customers.append(customer)
            nodes.append(customer)
        
        # Generate journey stages
        journey_stages = []
        for customer in customers:
            stages = list(CustomerJourneyStage)
            customer_stages = []
            
            for j, stage in enumerate(stages):
                journey_stage = GraphNode(
                    id=f"journey_{customer.id}_{stage.value}",
                    label=f"{customer.properties['name']} - {stage.value.title()}",
                    node_type="journey_stage",
                    properties={
                        "customer_id": customer.id,
                        "stage": stage.value,
                        "status": random.choice(["not_started", "in_progress", "completed", "abandoned"]),
                        "duration_days": random.randint(1, 90),
                        "touchpoints": random.randint(1, 10),
                        "satisfaction_score": random.randint(1, 10),
                        "conversion_probability": random.uniform(0.0, 1.0)
                    }
                )
                customer_stages.append(journey_stage)
                journey_stages.append(journey_stage)
                nodes.append(journey_stage)
            
            # Create stage progression
            for j in range(len(customer_stages) - 1):
                edge = GraphEdge(
                    id=f"progresses_to_{customer_stages[j+1].id}_{customer_stages[j].id}",
                    source=customer_stages[j].id,
                    target=customer_stages[j+1].id,
                    relationship_type="progresses_to",
                    properties={
                        "transition_probability": random.uniform(0.3, 1.0),
                        "avg_duration_days": random.randint(1, 30),
                        "key_factors": random.sample([
                            "price", "features", "support", "reputation", "ease_of_use"
                        ], random.randint(1, 3))
                    }
                )
                edges.append(edge)
        
        # Generate touchpoints
        touchpoints = []
        touchpoint_types = [
            "Website Visit", "Email Campaign", "Sales Call", "Demo", "Proposal",
            "Contract Review", "Implementation", "Training", "Support Call", "Renewal"
        ]
        
        for i in range(50):
            touchpoint = GraphNode(
                id=f"touchpoint_{i+1}",
                label=f"Touchpoint {i+1}",
                node_type="touchpoint",
                properties={
                    "name": random.choice(touchpoint_types),
                    "channel": random.choice(["digital", "phone", "email", "in_person", "social"]),
                    "duration_minutes": random.randint(5, 120),
                    "effectiveness_score": random.uniform(0.0, 1.0),
                    "cost": random.randint(10, 1000)
                }
            )
            touchpoints.append(touchpoint)
            nodes.append(touchpoint)
        
        # Link touchpoints to journey stages
        for stage in journey_stages:
            num_touchpoints = random.randint(1, 5)
            stage_touchpoints = random.sample(touchpoints, min(num_touchpoints, len(touchpoints)))
            
            for touchpoint in stage_touchpoints:
                edge = GraphEdge(
                    id=f"includes_{stage.id}_{touchpoint.id}",
                    source=stage.id,
                    target=touchpoint.id,
                    relationship_type="includes",
                    properties={
                        "sequence_order": random.randint(1, 10),
                        "effectiveness": random.uniform(0.0, 1.0)
                    }
                )
                edges.append(edge)
        
        metadata = {
            "pattern_type": "customer_journey",
            "total_customers": num_customers,
            "total_stages": len(journey_stages),
            "total_touchpoints": len(touchpoints)
        }
        
        return EnterpriseGraphPattern(
            pattern_type=EnterprisePatternType.CUSTOMER_JOURNEY,
            nodes=nodes,
            edges=edges,
            metadata=metadata
        )
    
    def _generate_role_by_level(self, level: int) -> str:
        """Generate role based on organizational level"""
        if level == 1:
            return random.choice(["VP", "Senior Director", "Chief Officer"])
        elif level == 2:
            return random.choice(["Director", "Senior Manager", "Principal"])
        elif level == 3:
            return random.choice(["Manager", "Lead", "Senior Specialist"])
        elif level == 4:
            return random.choice(["Specialist", "Analyst", "Coordinator"])
        else:
            return random.choice(["Associate", "Assistant", "Junior"])
    
    def _generate_department_by_level(self, level: int) -> str:
        """Generate department based on organizational level"""
        departments = [
            "Engineering", "Sales", "Marketing", "HR", "Finance", "IT",
            "Operations", "Customer Service", "R&D", "Legal", "Compliance"
        ]
        return random.choice(departments)
    
    async def save_pattern_to_neo4j(self, pattern: EnterpriseGraphPattern) -> bool:
        """Save enterprise graph pattern to Neo4j"""
        if not NEO4J_AVAILABLE or not self.graph:
            logger.warning("Neo4j not available, skipping save")
            return False
        
        try:
            # Create nodes
            for node in pattern.nodes:
                neo4j_node = Node(
                    node.node_type,
                    **node.properties
                )
                neo4j_node["id"] = node.id
                neo4j_node["label"] = node.label
                self.graph.create(neo4j_node)
            
            # Create relationships
            for edge in pattern.edges:
                source_node = self.graph.nodes.match(id=edge.source).first()
                target_node = self.graph.nodes.match(id=edge.target).first()
                
                if source_node and target_node:
                    relationship = Relationship(
                        source_node,
                        edge.relationship_type,
                        target_node,
                        **edge.properties
                    )
                    relationship["id"] = edge.id
                    relationship["weight"] = edge.weight
                    self.graph.create(relationship)
            
            logger.info(f"Successfully saved {pattern.pattern_type.value} pattern with {len(pattern.nodes)} nodes and {len(pattern.edges)} edges to Neo4j")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save pattern to Neo4j: {e}")
            return False
    
    def export_pattern_to_json(self, pattern: EnterpriseGraphPattern, filepath: str) -> bool:
        """Export enterprise graph pattern to JSON"""
        try:
            # Helper function to convert datetime objects to strings
            def serialize_properties(props):
                serialized = {}
                for key, value in props.items():
                    if hasattr(value, 'isoformat'):  # datetime object
                        serialized[key] = value.isoformat()
                    else:
                        serialized[key] = value
                return serialized
            
            export_data = {
                "pattern_type": pattern.pattern_type.value,
                "metadata": pattern.metadata,
                "created_at": pattern.created_at.isoformat(),
                "nodes": [
                    {
                        "id": node.id,
                        "label": node.label,
                        "node_type": node.node_type,
                        "properties": serialize_properties(node.properties),
                        "created_at": node.created_at.isoformat()
                    }
                    for node in pattern.nodes
                ],
                "edges": [
                    {
                        "id": edge.id,
                        "source": edge.source,
                        "target": edge.target,
                        "relationship_type": edge.relationship_type,
                        "properties": serialize_properties(edge.properties),
                        "weight": edge.weight,
                        "created_at": edge.created_at.isoformat()
                    }
                    for edge in pattern.edges
                ]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Pattern exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export pattern to JSON: {e}")
            return False


async def main():
    """Main function to generate enterprise graph patterns"""
    
    # Initialize generator
    generator = EnterpriseGraphPatternGenerator()
    
    # Create output directory
    output_dir = Path("neo4j_data/enterprise_patterns")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate different enterprise patterns
    patterns = [
        ("org_hierarchy", generator.generate_org_hierarchy_pattern(5, 8)),
        ("project_network", generator.generate_project_network_pattern(15)),
        ("knowledge_graph", generator.generate_knowledge_graph_pattern(40)),
        ("business_process", generator.generate_business_process_pattern(12)),
        ("customer_journey", generator.generate_customer_journey_pattern(25))
    ]
    
    for pattern_name, pattern in patterns:
        logger.info(f"Generating {pattern_name} pattern...")
        
        # Save to Neo4j
        await generator.save_pattern_to_neo4j(pattern)
        
        # Export to JSON
        json_file = output_dir / f"{pattern_name}_pattern.json"
        generator.export_pattern_to_json(pattern, str(json_file))
    
    logger.info("Enterprise graph patterns generation completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
