# Neo4j Service API Documentation

## 🎯 Overview

The Neo4j Service provides comprehensive graph database integration for the Lenovo Enterprise LLMOps Platform, featuring GraphRAG capabilities, organizational structure analysis, and real-time graph analytics.

## 🚀 Key Features

### Core Capabilities

- **Graph Database Operations**: Full CRUD operations for graph data
- **GraphRAG Integration**: Semantic search and knowledge retrieval
- **Lenovo Org Structure**: Realistic organizational data with Faker
- **B2B Client Scenarios**: Enterprise client relationship mapping
- **Enterprise Patterns**: Org charts, project networks, knowledge graphs
- **Real-time Analytics**: Live insights and relationship analysis

### Integration Features

- **FastAPI Integration**: RESTful API endpoints for all operations
- **Async Operations**: High-performance asynchronous query execution
- **Connection Pooling**: Optimized database connection management
- **Health Monitoring**: Service status and performance tracking
- **Error Handling**: Comprehensive error handling and logging

## 🌐 Service Integration

### Port Configuration

| Service            | Port | URL                             | Description                  |
| ------------------ | ---- | ------------------------------- | ---------------------------- |
| **Neo4j Browser**  | 7474 | http://localhost:7474           | Neo4j graph database browser |
| **Neo4j API**      | 8080 | http://localhost:8080/api/neo4j | Neo4j service endpoints      |
| **Neo4j Database** | 7687 | bolt://localhost:7687           | Neo4j database connection    |

### Service Dependencies

- **Neo4j Database**: Core graph database engine
- **Enterprise Platform**: FastAPI integration
- **GraphRAG**: Semantic search capabilities
- **Faker**: Realistic data generation

## 📚 API Endpoints

### Health & Status

#### GET /api/neo4j/health

Get Neo4j service health status.

**Response:**

```json
{
  "status": "healthy",
  "uri": "bolt://localhost:7687",
  "database": "neo4j",
  "timestamp": "2024-12-19T10:30:00Z",
  "query_response": {
    "status": "healthy",
    "timestamp": "2024-12-19T10:30:00Z"
  },
  "execution_time": 0.05
}
```

#### GET /api/neo4j/info

Get Neo4j database information and statistics.

**Response:**

```json
{
  "node_count": 5000,
  "relationship_count": 15000,
  "labels": ["person", "department", "project", "client"],
  "relationship_types": ["reports_to", "works_in", "participates_in"],
  "indexes": [
    {
      "description": "INDEX ON :person(id)",
      "state": "ONLINE",
      "type": "BTREE"
    }
  ]
}
```

### Graph Operations

#### POST /api/neo4j/query

Execute custom Cypher queries.

**Request:**

```json
{
  "query": "MATCH (n) RETURN count(n) as total_nodes",
  "parameters": {},
  "timeout": 30
}
```

**Response:**

```json
{
  "data": [{ "total_nodes": 5000 }],
  "summary": {
    "nodes_created": 0,
    "relationships_created": 0,
    "properties_set": 0
  },
  "execution_time": 0.1,
  "query_type": "read"
}
```

#### POST /api/neo4j/graphrag

Execute GraphRAG semantic search queries.

**Request:**

```json
{
  "query": "What is Lenovo's flagship laptop series?",
  "limit": 10
}
```

**Response:**

```json
{
  "query": "What is Lenovo's flagship laptop series?",
  "results": [
    {
      "d": {
        "id": "doc_1",
        "text_content": "ThinkPad is Lenovo's flagship laptop series...",
        "metadata": { "type": "product_info" }
      },
      "entities": [{ "name": "ThinkPad", "type": "product" }],
      "concepts": [{ "name": "laptop", "type": "product_category" }]
    }
  ],
  "execution_time": 0.2,
  "total_results": 1
}
```

### Lenovo Data

#### GET /api/neo4j/org-structure

Get Lenovo organizational structure data.

**Response:**

```json
{
  "organizational_data": [
    {
      "p": {
        "id": "emp_1",
        "name": "John Smith",
        "role": "Senior Engineer",
        "level": "individual_contributor"
      },
      "m": {
        "id": "emp_2",
        "name": "Jane Doe",
        "role": "Engineering Manager",
        "level": "management"
      },
      "d": {
        "id": "dept_1",
        "name": "Engineering",
        "budget": 5000000
      }
    }
  ],
  "execution_time": 0.15,
  "total_relationships": 500
}
```

#### GET /api/neo4j/b2b-clients

Get B2B client data and relationships.

**Response:**

```json
{
  "client_data": [
    {
      "c": {
        "id": "client_1",
        "name": "Acme Corporation",
        "industry": "Technology",
        "contract_value": 1000000
      },
      "s": {
        "id": "sol_1",
        "name": "ThinkPad Laptop Fleet",
        "implementation_status": "completed"
      },
      "client_employees": [
        {
          "id": "client_emp_1",
          "name": "Client Manager",
          "role": "IT Manager"
        }
      ]
    }
  ],
  "execution_time": 0.12,
  "total_clients": 20
}
```

#### GET /api/neo4j/project-dependencies

Get project dependency network.

**Response:**

```json
{
  "project_dependencies": [
    {
      "p1": {
        "id": "proj_1",
        "name": "AI Platform Development",
        "status": "active",
        "priority": "high"
      },
      "p2": {
        "id": "proj_2",
        "name": "Infrastructure Setup",
        "status": "completed",
        "priority": "critical"
      },
      "team_members": [
        {
          "id": "emp_1",
          "name": "Project Lead",
          "role": "Technical Lead"
        }
      ]
    }
  ],
  "execution_time": 0.18,
  "total_dependencies": 50
}
```

### Employee & Organization Data

#### GET /api/neo4j/employees

Get employee information with optional filtering.

**Parameters:**

- `limit` (int): Maximum number of employees (default: 50)
- `department` (string): Filter by department name

**Response:**

```json
{
  "employees": [
    {
      "id": "emp_1",
      "name": "John Smith",
      "role": "Senior Engineer",
      "department": "Engineering",
      "level": "individual_contributor",
      "skills": ["Python", "Machine Learning", "DevOps"]
    }
  ],
  "total_returned": 50,
  "execution_time": 0.1
}
```

#### GET /api/neo4j/departments

Get department information with employee counts.

**Response:**

```json
{
  "departments": [
    {
      "d": {
        "id": "dept_1",
        "name": "Engineering",
        "budget": 5000000,
        "headcount": 150
      },
      "employee_count": 150
    }
  ],
  "total_departments": 15,
  "execution_time": 0.08
}
```

#### GET /api/neo4j/projects

Get project information with team assignments.

**Parameters:**

- `status` (string): Filter by project status
- `limit` (int): Maximum number of projects (default: 50)

**Response:**

```json
{
  "projects": [
    {
      "p": {
        "id": "proj_1",
        "name": "AI Platform Development",
        "status": "active",
        "priority": "high",
        "budget": 2000000
      },
      "team_members": [
        {
          "id": "emp_1",
          "name": "Project Manager",
          "role": "lead"
        }
      ]
    }
  ],
  "total_returned": 50,
  "execution_time": 0.12
}
```

#### GET /api/neo4j/skills

Get skills and certifications information.

**Parameters:**

- `category` (string): Filter by skill category

**Response:**

```json
{
  "skills": [
    {
      "s": {
        "id": "skill_1",
        "name": "Python",
        "category": "technical",
        "demand_level": "high"
      },
      "employee_count": 120
    }
  ],
  "total_skills": 50,
  "execution_time": 0.09
}
```

### Analytics

#### GET /api/neo4j/analytics/org-chart

Get organizational chart analytics.

**Response:**

```json
{
  "management_hierarchy": [
    {
      "m": {
        "id": "emp_1",
        "name": "CEO",
        "role": "Chief Executive Officer"
      },
      "report_count": 8,
      "direct_reports": [
        {
          "id": "emp_2",
          "name": "CTO",
          "role": "Chief Technology Officer"
        }
      ]
    }
  ],
  "execution_time": 0.2
}
```

#### GET /api/neo4j/analytics/project-metrics

Get project metrics and analytics.

**Response:**

```json
{
  "project_metrics": [
    {
      "status": "active",
      "count": 25,
      "avg_budget": 1500000,
      "sample_projects": [
        "AI Platform Development",
        "Infrastructure Modernization"
      ]
    }
  ],
  "execution_time": 0.15
}
```

#### GET /api/neo4j/analytics/skill-gaps

Get skill gap analysis.

**Response:**

```json
{
  "skill_gaps": [
    {
      "s": {
        "id": "skill_1",
        "name": "Quantum Computing",
        "category": "emerging"
      },
      "employee_count": 2
    }
  ],
  "total_gaps": 15,
  "execution_time": 0.1
}
```

## 🔧 Configuration

### Environment Variables

```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j

# Connection Pool Settings
NEO4J_MAX_CONNECTION_LIFETIME=3600
NEO4J_MAX_CONNECTION_POOL_SIZE=50
NEO4J_CONNECTION_ACQUISITION_TIMEOUT=60
```

### Service Configuration

```python
from src.enterprise_llmops.neo4j_service import Neo4jConfig

config = Neo4jConfig(
    uri="bolt://localhost:7687",
    username="neo4j",
    password="password",
    database="neo4j",
    max_connection_lifetime=3600,
    max_connection_pool_size=50,
    connection_acquisition_timeout=60
)
```

## 📊 Usage Examples

### Basic Query Execution

```python
import requests

# Health check
response = requests.get("http://localhost:8080/api/neo4j/health")
print(response.json())

# Custom query
query_data = {
    "query": "MATCH (p:person) WHERE p.level = 'executive' RETURN p",
    "parameters": {},
    "timeout": 30
}
response = requests.post("http://localhost:8080/api/neo4j/query", json=query_data)
print(response.json())
```

### GraphRAG Query

```python
# GraphRAG semantic search
graphrag_data = {
    "query": "Tell me about Lenovo's AI capabilities",
    "limit": 5
}
response = requests.post("http://localhost:8080/api/neo4j/graphrag", json=graphrag_data)
print(response.json())
```

### Organizational Analysis

```python
# Get organizational structure
response = requests.get("http://localhost:8080/api/neo4j/org-structure")
org_data = response.json()

# Get project dependencies
response = requests.get("http://localhost:8080/api/neo4j/project-dependencies")
project_data = response.json()

# Get skill gap analysis
response = requests.get("http://localhost:8080/api/neo4j/analytics/skill-gaps")
skill_gaps = response.json()
```

## 🚨 Troubleshooting

### Common Issues

#### Connection Failed

```bash
# Check Neo4j status
curl http://localhost:8080/api/neo4j/health

# Verify Neo4j is running
neo4j status

# Check connection settings
# Default: bolt://localhost:7687, username: neo4j, password: password
```

#### Query Timeout

```bash
# Increase timeout in query request
{
  "query": "MATCH (n) RETURN n",
  "timeout": 60
}
```

#### Authentication Error

```bash
# Verify credentials
curl -u neo4j:password http://localhost:8080/api/neo4j/health
```

### Debug Procedures

#### Check Service Status

```bash
# Health check
curl http://localhost:8080/api/neo4j/health

# Database info
curl http://localhost:8080/api/neo4j/info
```

#### Test Query Execution

```bash
# Simple test query
curl -X POST http://localhost:8080/api/neo4j/query \
  -H "Content-Type: application/json" \
  -d '{"query": "RETURN 1 as test"}'
```

## 📞 Support

### Resources

- **Neo4j Documentation**: [Neo4j Graph Database](https://neo4j.com/docs/)
- **GraphRAG Research**: [Microsoft GraphRAG](https://github.com/microsoft/graphrag)
- **FastAPI Documentation**: [FastAPI Docs](https://fastapi.tiangolo.com/)

### Getting Help

- Check the health endpoint for service status
- Review Neo4j logs for database issues
- Verify connection settings and credentials
- Test with simple queries first

---

**Last Updated**: 2024-12-19  
**Version**: 1.0.0  
**Status**: Production Ready  
**Integration**: Full Neo4j Service Integration
