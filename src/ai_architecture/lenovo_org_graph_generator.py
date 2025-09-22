"""
Lenovo Organizational Graph Generator with Neo4j Integration

This module generates comprehensive organizational graphs for Lenovo and B2B clients
using Faker data, following Neo4j best practices for enterprise GraphRAG solutions.

Key Features:
- Lenovo organizational hierarchy with realistic departments and roles
- B2B client organizational structures for enterprise solutions
- Enterprise graph patterns (org charts, project hierarchies, knowledge graphs)
- GraphRAG-optimized structure for agentic workflows
- Neo4j best practices for entity and semantic mapping
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


class LenovoDepartment(Enum):
    """Lenovo organizational departments"""
    EXECUTIVE = "executive"
    ENGINEERING = "engineering"
    SALES = "sales"
    MARKETING = "marketing"
    HR = "human_resources"
    FINANCE = "finance"
    IT = "information_technology"
    OPERATIONS = "operations"
    CUSTOMER_SERVICE = "customer_service"
    RESEARCH_DEVELOPMENT = "research_development"
    SUPPLY_CHAIN = "supply_chain"
    LEGAL = "legal"
    COMPLIANCE = "compliance"


class B2BClientType(Enum):
    """B2B client organization types"""
    ENTERPRISE_CORPORATION = "enterprise_corporation"
    GOVERNMENT_AGENCY = "government_agency"
    HEALTHCARE_SYSTEM = "healthcare_system"
    FINANCIAL_INSTITUTION = "financial_institution"
    EDUCATIONAL_INSTITUTION = "educational_institution"
    MANUFACTURING_COMPANY = "manufacturing_company"
    RETAIL_CHAIN = "retail_chain"
    TECHNOLOGY_STARTUP = "technology_startup"


class GraphNodeType(Enum):
    """Node types for the organizational graph"""
    PERSON = "person"
    DEPARTMENT = "department"
    ROLE = "role"
    PROJECT = "project"
    CLIENT = "client"
    SOLUTION = "solution"
    SKILL = "skill"
    CERTIFICATION = "certification"
    LOCATION = "location"
    TEAM = "team"


class RelationshipType(Enum):
    """Relationship types for organizational graph"""
    REPORTS_TO = "reports_to"
    WORKS_IN = "works_in"
    MANAGES = "manages"
    COLLABORATES_WITH = "collaborates_with"
    LEADS = "leads"
    PARTICIPATES_IN = "participates_in"
    HAS_SKILL = "has_skill"
    CERTIFIED_IN = "certified_in"
    LOCATED_AT = "located_at"
    SERVES = "serves"
    USES = "uses"
    DEPENDS_ON = "depends_on"


@dataclass
class GraphNode:
    """Represents a node in the organizational graph"""
    id: str
    label: str
    node_type: GraphNodeType
    properties: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class GraphEdge:
    """Represents an edge in the organizational graph"""
    id: str
    source: str
    target: str
    relationship_type: RelationshipType
    properties: Dict[str, Any]
    weight: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class LenovoOrgGraph:
    """Complete Lenovo organizational graph structure"""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    metadata: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)


class LenovoOrgGraphGenerator:
    """Generator for Lenovo organizational graphs with B2B client scenarios"""
    
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
    
    def generate_lenovo_org_structure(self, 
                                    num_employees: int = 500,
                                    num_departments: int = 15,
                                    num_projects: int = 50) -> LenovoOrgGraph:
        """Generate comprehensive Lenovo organizational structure"""
        
        nodes = []
        edges = []
        
        # Generate Lenovo departments
        departments = self._generate_lenovo_departments(num_departments)
        nodes.extend(departments)
        
        # Generate Lenovo employees
        employees = self._generate_lenovo_employees(num_employees, departments)
        nodes.extend(employees)
        
        # Generate organizational relationships
        org_edges = self._generate_organizational_relationships(employees, departments)
        edges.extend(org_edges)
        
        # Generate projects
        projects = self._generate_lenovo_projects(num_projects, departments)
        nodes.extend(projects)
        
        # Generate project relationships
        project_edges = self._generate_project_relationships(employees, projects)
        edges.extend(project_edges)
        
        # Generate skills and certifications
        skills = self._generate_skills_and_certifications(employees)
        nodes.extend(skills['nodes'])
        edges.extend(skills['edges'])
        
        # Generate locations
        locations = self._generate_locations()
        nodes.extend(locations)
        
        # Generate location relationships
        location_edges = self._generate_location_relationships(employees, locations)
        edges.extend(location_edges)
        
        metadata = {
            "total_employees": len(employees),
            "total_departments": len(departments),
            "total_projects": len(projects),
            "total_skills": len([n for n in skills['nodes'] if n.node_type == GraphNodeType.SKILL]),
            "generation_timestamp": datetime.now().isoformat(),
            "graph_type": "lenovo_organizational"
        }
        
        return LenovoOrgGraph(nodes, edges, metadata)
    
    def generate_b2b_client_scenarios(self, num_clients: int = 20) -> LenovoOrgGraph:
        """Generate B2B client organizational structures"""
        
        nodes = []
        edges = []
        
        # Generate B2B clients
        clients = self._generate_b2b_clients(num_clients)
        nodes.extend(clients)
        
        # Generate client organizational structures
        for client in clients:
            client_structure = self._generate_client_org_structure(client)
            nodes.extend(client_structure['nodes'])
            edges.extend(client_structure['edges'])
        
        # Generate Lenovo solutions for clients
        solutions = self._generate_lenovo_solutions(clients)
        nodes.extend(solutions)
        
        # Generate solution relationships
        solution_edges = self._generate_solution_relationships(clients, solutions)
        edges.extend(solution_edges)
        
        metadata = {
            "total_clients": len(clients),
            "total_solutions": len(solutions),
            "generation_timestamp": datetime.now().isoformat(),
            "graph_type": "b2b_client_scenarios"
        }
        
        return LenovoOrgGraph(nodes, edges, metadata)
    
    def _generate_lenovo_departments(self, num_departments: int) -> List[GraphNode]:
        """Generate Lenovo departments with realistic structure"""
        departments = []
        
        # Core Lenovo departments
        lenovo_departments = [
            ("Executive Office", "C-Level executives and board members"),
            ("Engineering", "Product development and engineering teams"),
            ("Sales", "Global sales and business development"),
            ("Marketing", "Brand marketing and product marketing"),
            ("Human Resources", "Talent acquisition and employee relations"),
            ("Finance", "Financial planning and accounting"),
            ("Information Technology", "IT infrastructure and digital transformation"),
            ("Operations", "Supply chain and manufacturing operations"),
            ("Customer Service", "Customer support and success"),
            ("Research & Development", "Innovation and advanced research"),
            ("Supply Chain", "Procurement and logistics"),
            ("Legal", "Legal affairs and compliance"),
            ("Compliance", "Regulatory compliance and risk management"),
            ("Quality Assurance", "Product quality and testing"),
            ("Business Development", "Strategic partnerships and alliances")
        ]
        
        for i, (dept_name, description) in enumerate(lenovo_departments[:num_departments]):
            department = GraphNode(
                id=f"dept_{i+1}",
                label=dept_name,
                node_type=GraphNodeType.DEPARTMENT,
                properties={
                    "name": dept_name,
                    "description": description,
                    "budget": random.randint(1000000, 50000000),
                    "headcount": random.randint(10, 200),
                    "location": self.fake.city(),
                    "established": self.fake.date_between(start_date='-10y', end_date='today'),
                    "department_type": "core" if i < 8 else "support"
                }
            )
            departments.append(department)
        
        return departments
    
    def _generate_lenovo_employees(self, num_employees: int, departments: List[GraphNode]) -> List[GraphNode]:
        """Generate Lenovo employees with realistic profiles"""
        employees = []
        
        # Executive roles
        executive_roles = [
            "Chief Executive Officer", "Chief Technology Officer", "Chief Financial Officer",
            "Chief Marketing Officer", "Chief Operating Officer", "Chief Human Resources Officer",
            "Chief Information Officer", "Chief Legal Officer", "Chief Strategy Officer"
        ]
        
        # Management roles
        management_roles = [
            "Vice President", "Senior Director", "Director", "Senior Manager", "Manager",
            "Team Lead", "Principal Engineer", "Senior Engineer", "Engineer", "Analyst"
        ]
        
        # Individual contributor roles
        ic_roles = [
            "Senior Software Engineer", "Software Engineer", "Data Scientist", "Product Manager",
            "Sales Representative", "Marketing Specialist", "HR Specialist", "Financial Analyst",
            "Customer Success Manager", "Research Scientist", "Quality Engineer"
        ]
        
        all_roles = executive_roles + management_roles + ic_roles
        
        for i in range(num_employees):
            # Determine role level
            if i < 10:  # Top executives
                role = random.choice(executive_roles)
                level = "executive"
            elif i < 50:  # Management
                role = random.choice(management_roles)
                level = "management"
            else:  # Individual contributors
                role = random.choice(ic_roles)
                level = "individual_contributor"
            
            # Generate employee profile
            employee = GraphNode(
                id=f"emp_{i+1}",
                label=f"{self.fake.first_name()} {self.fake.last_name()}",
                node_type=GraphNodeType.PERSON,
                properties={
                    "first_name": self.fake.first_name(),
                    "last_name": self.fake.last_name(),
                    "email": self.fake.email(),
                    "phone": self.fake.phone_number(),
                    "role": role,
                    "level": level,
                    "salary": self._generate_salary_by_level(level),
                    "hire_date": self.fake.date_between(start_date='-5y', end_date='today'),
                    "performance_rating": random.choice(["exceeds", "meets", "needs_improvement"]),
                    "location": self.fake.city(),
                    "timezone": self.fake.timezone(),
                    "languages": random.sample(["English", "Chinese", "Spanish", "French", "German"], 
                                            random.randint(1, 3))
                }
            )
            employees.append(employee)
        
        return employees
    
    def _generate_salary_by_level(self, level: str) -> int:
        """Generate realistic salary based on role level"""
        if level == "executive":
            return random.randint(200000, 800000)
        elif level == "management":
            return random.randint(80000, 200000)
        else:
            return random.randint(40000, 120000)
    
    def _generate_organizational_relationships(self, employees: List[GraphNode], 
                                            departments: List[GraphNode]) -> List[GraphEdge]:
        """Generate organizational relationships"""
        edges = []
        
        # Assign employees to departments
        for employee in employees:
            # Assign to random department
            dept = random.choice(departments)
            edge = GraphEdge(
                id=f"works_in_{employee.id}_{dept.id}",
                source=employee.id,
                target=dept.id,
                relationship_type=RelationshipType.WORKS_IN,
                properties={
                    "start_date": self.fake.date_between(start_date='-3y', end_date='today'),
                    "is_primary": True
                }
            )
            edges.append(edge)
        
        # Create reporting relationships
        executives = [emp for emp in employees if emp.properties.get('level') == 'executive']
        managers = [emp for emp in employees if emp.properties.get('level') == 'management']
        contributors = [emp for emp in employees if emp.properties.get('level') == 'individual_contributor']
        
        # Executives report to CEO
        ceo = next((emp for emp in executives if 'CEO' in emp.properties.get('role', '')), executives[0])
        for executive in executives:
            if executive.id != ceo.id:
                edge = GraphEdge(
                    id=f"reports_to_{executive.id}_{ceo.id}",
                    source=executive.id,
                    target=ceo.id,
                    relationship_type=RelationshipType.REPORTS_TO,
                    properties={"relationship_type": "direct_report"}
                )
                edges.append(edge)
        
        # Managers report to executives
        for manager in managers:
            executive = random.choice(executives)
            edge = GraphEdge(
                id=f"reports_to_{manager.id}_{executive.id}",
                source=manager.id,
                target=executive.id,
                relationship_type=RelationshipType.REPORTS_TO,
                properties={"relationship_type": "direct_report"}
            )
            edges.append(edge)
        
        # Contributors report to managers
        for contributor in contributors:
            manager = random.choice(managers)
            edge = GraphEdge(
                id=f"reports_to_{contributor.id}_{manager.id}",
                source=contributor.id,
                target=manager.id,
                relationship_type=RelationshipType.REPORTS_TO,
                properties={"relationship_type": "direct_report"}
            )
            edges.append(edge)
        
        return edges
    
    def _generate_lenovo_projects(self, num_projects: int, departments: List[GraphNode]) -> List[GraphNode]:
        """Generate Lenovo projects"""
        projects = []
        
        project_types = [
            "Product Development", "Digital Transformation", "Market Expansion",
            "Process Improvement", "Technology Innovation", "Customer Experience",
            "Supply Chain Optimization", "Sustainability Initiative", "AI/ML Implementation",
            "Cloud Migration", "Security Enhancement", "Compliance Project"
        ]
        
        for i in range(num_projects):
            project_type = random.choice(project_types)
            project = GraphNode(
                id=f"proj_{i+1}",
                label=f"{project_type} Project {i+1}",
                node_type=GraphNodeType.PROJECT,
                properties={
                    "name": f"{project_type} Project {i+1}",
                    "description": self.fake.text(max_nb_chars=200),
                    "project_type": project_type,
                    "status": random.choice(["planning", "active", "on_hold", "completed"]),
                    "priority": random.choice(["low", "medium", "high", "critical"]),
                    "budget": random.randint(100000, 5000000),
                    "start_date": self.fake.date_between(start_date='-2y', end_date='today'),
                    "end_date": self.fake.date_between(start_date='today', end_date='+1y'),
                    "client_impact": random.choice(["internal", "external", "both"]),
                    "technology_stack": random.sample([
                        "Python", "Java", "React", "Node.js", "AWS", "Azure", "Docker", "Kubernetes"
                    ], random.randint(2, 5))
                }
            )
            projects.append(project)
        
        return projects
    
    def _generate_project_relationships(self, employees: List[GraphNode], 
                                      projects: List[GraphNode]) -> List[GraphEdge]:
        """Generate project participation relationships"""
        edges = []
        
        for project in projects:
            # Assign project manager
            project_manager = random.choice(employees)
            edge = GraphEdge(
                id=f"leads_{project_manager.id}_{project.id}",
                source=project_manager.id,
                target=project.id,
                relationship_type=RelationshipType.LEADS,
                properties={"role": "project_manager"}
            )
            edges.append(edge)
            
            # Assign team members
            team_size = random.randint(3, 15)
            team_members = random.sample(employees, min(team_size, len(employees)))
            
            for member in team_members:
                if member.id != project_manager.id:
                    edge = GraphEdge(
                        id=f"participates_{member.id}_{project.id}",
                        source=member.id,
                        target=project.id,
                        relationship_type=RelationshipType.PARTICIPATES_IN,
                        properties={
                            "role": random.choice(["developer", "analyst", "designer", "tester"]),
                            "allocation": random.randint(10, 100)
                        }
                    )
                    edges.append(edge)
        
        return edges
    
    def _generate_skills_and_certifications(self, employees: List[GraphNode]) -> Dict[str, List]:
        """Generate skills and certifications for employees"""
        nodes = []
        edges = []
        
        # Technical skills
        technical_skills = [
            "Python", "Java", "JavaScript", "React", "Node.js", "AWS", "Azure", "Docker",
            "Kubernetes", "Machine Learning", "Data Science", "DevOps", "Cybersecurity",
            "Cloud Architecture", "Database Design", "API Development", "Mobile Development"
        ]
        
        # Business skills
        business_skills = [
            "Project Management", "Agile", "Scrum", "Leadership", "Communication",
            "Strategic Planning", "Financial Analysis", "Marketing", "Sales",
            "Customer Relations", "Negotiation", "Presentation", "Team Building"
        ]
        
        # Certifications
        certifications = [
            "AWS Certified Solutions Architect", "Google Cloud Professional",
            "Microsoft Azure Expert", "PMP Certification", "Scrum Master",
            "ITIL Foundation", "CISSP", "CISA", "Six Sigma", "Lean Management"
        ]
        
        all_skills = technical_skills + business_skills
        
        # Create skill nodes
        for skill in all_skills:
            skill_node = GraphNode(
                id=f"skill_{skill.lower().replace(' ', '_')}",
                label=skill,
                node_type=GraphNodeType.SKILL,
                properties={
                    "name": skill,
                    "category": "technical" if skill in technical_skills else "business",
                    "demand_level": random.choice(["low", "medium", "high"])
                }
            )
            nodes.append(skill_node)
        
        # Create certification nodes
        for cert in certifications:
            cert_node = GraphNode(
                id=f"cert_{cert.lower().replace(' ', '_')}",
                label=cert,
                node_type=GraphNodeType.CERTIFICATION,
                properties={
                    "name": cert,
                    "issuer": random.choice(["AWS", "Google", "Microsoft", "PMI", "Scrum.org"]),
                    "validity_period": random.randint(1, 3)
                }
            )
            nodes.append(cert_node)
        
        # Assign skills to employees
        for employee in employees:
            # Assign 3-8 skills per employee
            num_skills = random.randint(3, 8)
            employee_skills = random.sample(all_skills, min(num_skills, len(all_skills)))
            
            for skill in employee_skills:
                skill_id = f"skill_{skill.lower().replace(' ', '_')}"
                edge = GraphEdge(
                    id=f"has_skill_{employee.id}_{skill_id}",
                    source=employee.id,
                    target=skill_id,
                    relationship_type=RelationshipType.HAS_SKILL,
                    properties={
                        "proficiency": random.choice(["beginner", "intermediate", "advanced", "expert"]),
                        "years_experience": random.randint(1, 10)
                    }
                )
                edges.append(edge)
            
            # Assign 0-3 certifications per employee
            num_certs = random.randint(0, 3)
            if num_certs > 0:
                employee_certs = random.sample(certifications, min(num_certs, len(certifications)))
                for cert in employee_certs:
                    cert_id = f"cert_{cert.lower().replace(' ', '_')}"
                    edge = GraphEdge(
                        id=f"certified_{employee.id}_{cert_id}",
                        source=employee.id,
                        target=cert_id,
                        relationship_type=RelationshipType.CERTIFIED_IN,
                        properties={
                            "certification_date": self.fake.date_between(start_date='-3y', end_date='today'),
                            "expiry_date": self.fake.date_between(start_date='today', end_date='+3y')
                        }
                    )
                    edges.append(edge)
        
        return {"nodes": nodes, "edges": edges}
    
    def _generate_locations(self) -> List[GraphNode]:
        """Generate Lenovo office locations"""
        locations = []
        
        # Major Lenovo locations
        lenovo_locations = [
            ("Beijing, China", "Global Headquarters"),
            ("Morrisville, NC, USA", "North America Headquarters"),
            ("Singapore", "Asia Pacific Headquarters"),
            ("London, UK", "Europe Headquarters"),
            ("São Paulo, Brazil", "Latin America Headquarters"),
            ("Tokyo, Japan", "Japan Office"),
            ("Bangalore, India", "India R&D Center"),
            ("Berlin, Germany", "Germany Office"),
            ("Sydney, Australia", "Australia Office"),
            ("Toronto, Canada", "Canada Office")
        ]
        
        for i, (location, description) in enumerate(lenovo_locations):
            location_node = GraphNode(
                id=f"loc_{i+1}",
                label=location,
                node_type=GraphNodeType.LOCATION,
                properties={
                    "name": location,
                    "description": description,
                    "office_type": "headquarters" if "Headquarters" in description else "office",
                    "employee_count": random.randint(50, 1000),
                    "timezone": self.fake.timezone(),
                    "established": self.fake.date_between(start_date='-20y', end_date='-5y')
                }
            )
            locations.append(location_node)
        
        return locations
    
    def _generate_location_relationships(self, employees: List[GraphNode], 
                                     locations: List[GraphNode]) -> List[GraphEdge]:
        """Generate location relationships for employees"""
        edges = []
        
        for employee in employees:
            location = random.choice(locations)
            edge = GraphEdge(
                id=f"located_{employee.id}_{location.id}",
                source=employee.id,
                target=location.id,
                relationship_type=RelationshipType.LOCATED_AT,
                properties={
                    "work_type": random.choice(["office", "remote", "hybrid"]),
                    "start_date": self.fake.date_between(start_date='-3y', end_date='today')
                }
            )
            edges.append(edge)
        
        return edges
    
    def _generate_b2b_clients(self, num_clients: int) -> List[GraphNode]:
        """Generate B2B client organizations"""
        clients = []
        
        client_types = list(B2BClientType)
        
        for i in range(num_clients):
            client_type = random.choice(client_types)
            company_name = self.fake.company()
            
            client = GraphNode(
                id=f"client_{i+1}",
                label=company_name,
                node_type=GraphNodeType.CLIENT,
                properties={
                    "name": company_name,
                    "client_type": client_type.value,
                    "industry": self._get_industry_by_type(client_type),
                    "revenue": random.randint(10000000, 1000000000),
                    "employee_count": random.randint(100, 50000),
                    "location": self.fake.city(),
                    "country": self.fake.country(),
                    "contract_value": random.randint(100000, 10000000),
                    "contract_start": self.fake.date_between(start_date='-2y', end_date='today'),
                    "contract_end": self.fake.date_between(start_date='today', end_date='+3y'),
                    "satisfaction_score": random.randint(1, 10),
                    "priority": random.choice(["low", "medium", "high", "strategic"])
                }
            )
            clients.append(client)
        
        return clients
    
    def _get_industry_by_type(self, client_type: B2BClientType) -> str:
        """Get industry based on client type"""
        industry_mapping = {
            B2BClientType.ENTERPRISE_CORPORATION: "Technology",
            B2BClientType.GOVERNMENT_AGENCY: "Government",
            B2BClientType.HEALTHCARE_SYSTEM: "Healthcare",
            B2BClientType.FINANCIAL_INSTITUTION: "Financial Services",
            B2BClientType.EDUCATIONAL_INSTITUTION: "Education",
            B2BClientType.MANUFACTURING_COMPANY: "Manufacturing",
            B2BClientType.RETAIL_CHAIN: "Retail",
            B2BClientType.TECHNOLOGY_STARTUP: "Technology"
        }
        return industry_mapping.get(client_type, "Technology")
    
    def _generate_client_org_structure(self, client: GraphNode) -> Dict[str, List]:
        """Generate organizational structure for a B2B client"""
        nodes = []
        edges = []
        
        # Generate client departments
        client_departments = [
            "IT Department", "Finance", "Operations", "HR", "Legal", "Procurement",
            "Customer Service", "Marketing", "Sales", "Research & Development"
        ]
        
        dept_nodes = []
        for i, dept_name in enumerate(client_departments):
            dept_node = GraphNode(
                id=f"{client.id}_dept_{i+1}",
                label=f"{client.properties['name']} - {dept_name}",
                node_type=GraphNodeType.DEPARTMENT,
                properties={
                    "name": dept_name,
                    "client_id": client.id,
                    "headcount": random.randint(5, 100),
                    "budget": random.randint(100000, 5000000)
                }
            )
            dept_nodes.append(dept_node)
            nodes.append(dept_node)
        
        # Generate client employees
        num_employees = random.randint(20, 200)
        emp_nodes = []
        for i in range(num_employees):
            emp_node = GraphNode(
                id=f"{client.id}_emp_{i+1}",
                label=f"{self.fake.first_name()} {self.fake.last_name()}",
                node_type=GraphNodeType.PERSON,
                properties={
                    "first_name": self.fake.first_name(),
                    "last_name": self.fake.last_name(),
                    "email": self.fake.email(),
                    "role": self.fake.job(),
                    "department": random.choice(client_departments),
                    "client_id": client.id
                }
            )
            emp_nodes.append(emp_node)
            nodes.append(emp_node)
        
        # Create department relationships
        for emp in emp_nodes:
            dept = random.choice(dept_nodes)
            edge = GraphEdge(
                id=f"works_in_{emp.id}_{dept.id}",
                source=emp.id,
                target=dept.id,
                relationship_type=RelationshipType.WORKS_IN,
                properties={}
            )
            edges.append(edge)
        
        return {"nodes": nodes, "edges": edges}
    
    def _generate_lenovo_solutions(self, clients: List[GraphNode]) -> List[GraphNode]:
        """Generate Lenovo solutions for B2B clients"""
        solutions = []
        
        solution_types = [
            "ThinkPad Laptop Fleet", "ThinkCentre Desktop Solutions", "ThinkStation Workstations",
            "ThinkSystem Servers", "ThinkAgile Infrastructure", "ThinkShield Security",
            "ThinkSmart Collaboration", "ThinkEdge AI Solutions", "ThinkReality VR/AR",
            "ThinkPad for Business", "ThinkCentre for Education", "ThinkSystem for Healthcare"
        ]
        
        for i, client in enumerate(clients):
            num_solutions = random.randint(1, 5)
            client_solutions = random.sample(solution_types, min(num_solutions, len(solution_types)))
            
            for j, solution_type in enumerate(client_solutions):
                solution = GraphNode(
                    id=f"sol_{client.id}_{j+1}",
                    label=f"{solution_type} for {client.properties['name']}",
                    node_type=GraphNodeType.SOLUTION,
                    properties={
                        "name": solution_type,
                        "client_id": client.id,
                        "solution_type": solution_type,
                        "implementation_status": random.choice(["planned", "in_progress", "completed"]),
                        "value": random.randint(50000, 2000000),
                        "start_date": self.fake.date_between(start_date='-1y', end_date='today'),
                        "completion_date": self.fake.date_between(start_date='today', end_date='+1y'),
                        "success_metrics": {
                            "performance_improvement": random.randint(10, 50),
                            "cost_reduction": random.randint(5, 30),
                            "user_satisfaction": random.randint(7, 10)
                        }
                    }
                )
                solutions.append(solution)
        
        return solutions
    
    def _generate_solution_relationships(self, clients: List[GraphNode], 
                                       solutions: List[GraphNode]) -> List[GraphEdge]:
        """Generate relationships between clients and solutions"""
        edges = []
        
        for solution in solutions:
            client_id = solution.properties.get('client_id')
            if client_id:
                client = next((c for c in clients if c.id == client_id), None)
                if client:
                    edge = GraphEdge(
                        id=f"serves_{solution.id}_{client.id}",
                        source=solution.id,
                        target=client.id,
                        relationship_type=RelationshipType.SERVES,
                        properties={
                            "contract_value": solution.properties.get('value', 0),
                            "implementation_phase": solution.properties.get('implementation_status', 'planned')
                        }
                    )
                    edges.append(edge)
        
        return edges
    
    async def save_to_neo4j(self, graph_data: LenovoOrgGraph) -> bool:
        """Save graph data to Neo4j database"""
        if not NEO4J_AVAILABLE or not self.graph:
            logger.warning("Neo4j not available, skipping save")
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
            
            logger.info(f"Successfully saved {len(graph_data.nodes)} nodes and {len(graph_data.edges)} edges to Neo4j")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save to Neo4j: {e}")
            return False
    
    def export_to_json(self, graph_data: LenovoOrgGraph, filepath: str) -> bool:
        """Export graph data to JSON file"""
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
                "metadata": graph_data.metadata,
                "created_at": graph_data.created_at.isoformat(),
                "nodes": [
                    {
                        "id": node.id,
                        "label": node.label,
                        "node_type": node.node_type.value,
                        "properties": serialize_properties(node.properties),
                        "created_at": node.created_at.isoformat()
                    }
                    for node in graph_data.nodes
                ],
                "edges": [
                    {
                        "id": edge.id,
                        "source": edge.source,
                        "target": edge.target,
                        "relationship_type": edge.relationship_type.value,
                        "properties": serialize_properties(edge.properties),
                        "weight": edge.weight,
                        "created_at": edge.created_at.isoformat()
                    }
                    for edge in graph_data.edges
                ]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Graph data exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export to JSON: {e}")
            return False


async def main():
    """Main function to generate and save Lenovo organizational graphs"""
    
    # Initialize generator
    generator = LenovoOrgGraphGenerator()
    
    # Generate Lenovo organizational structure
    logger.info("Generating Lenovo organizational structure...")
    lenovo_graph = generator.generate_lenovo_org_structure(
        num_employees=500,
        num_departments=15,
        num_projects=50
    )
    
    # Generate B2B client scenarios
    logger.info("Generating B2B client scenarios...")
    b2b_graph = generator.generate_b2b_client_scenarios(num_clients=20)
    
    # Save to Neo4j
    logger.info("Saving to Neo4j...")
    await generator.save_to_neo4j(lenovo_graph)
    await generator.save_to_neo4j(b2b_graph)
    
    # Export to JSON
    logger.info("Exporting to JSON...")
    generator.export_to_json(lenovo_graph, "neo4j_data/lenovo_org_graph.json")
    generator.export_to_json(b2b_graph, "neo4j_data/b2b_client_graph.json")
    
    logger.info("Graph generation completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
