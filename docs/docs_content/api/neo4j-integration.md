# Neo4j Integration Documentation

## 🎯 Overview

The Neo4j Integration provides comprehensive knowledge graph capabilities for the Lenovo AAITC Enterprise Platform, enabling advanced semantic search, relationship analysis, and intelligent data retrieval through GraphRAG (Graph Retrieval-Augmented Generation).

## 🚀 Key Features

### Core Capabilities

- **Knowledge Graph Management**: Complete graph database operations with Cypher queries
- **GraphRAG Implementation**: Advanced retrieval-augmented generation using graph structures
- **Faker Data Integration**: Realistic test data generation for development and testing
- **Lenovo Organization Modeling**: Enterprise structure representation and analysis
- **B2B Relationship Mapping**: Client and project relationship analysis
- **Skills & Certification Tracking**: Employee competency management and tracking

### Integration Features

- **FastAPI Backend Integration**: Seamless API integration with enterprise platform
- **Real-time Graph Visualization**: Interactive graph exploration and analysis
- **Advanced Cypher Queries**: Custom graph query capabilities with performance optimization
- **Automated Data Generation**: Faker-based test data creation for realistic scenarios

## 🌐 API Endpoints

### Health & Status

- `GET /api/neo4j/health` - Neo4j service health status and connectivity
- `GET /api/neo4j/info` - Database information, statistics, and configuration

### Graph Operations

- `POST /api/neo4j/query` - Execute custom Cypher queries with parameter binding
- `POST /api/neo4j/graphrag` - GraphRAG semantic search and knowledge retrieval
- `GET /api/neo4j/org-structure` - Lenovo organizational data and hierarchy
- `GET /api/neo4j/b2b-clients` - B2B client relationships and project mappings
- `GET /api/neo4j/project-dependencies` - Project network analysis and dependencies

### Data Management

- `GET /api/neo4j/employees` - Employee information and skill profiles
- `GET /api/neo4j/departments` - Department data and organizational structure
- `GET /api/neo4j/projects` - Project information and status tracking
- `GET /api/neo4j/skills` - Skills and certifications database

### Analytics & Insights

- `GET /api/neo4j/analytics/*` - Advanced analytics endpoints for business intelligence
- `GET /api/neo4j/network-analysis` - Network analysis and relationship mapping
- `GET /api/neo4j/competency-gaps` - Skills gap analysis and recommendations

## 🔧 Configuration

### Service Setup

```bash
# Start Neo4j service with Docker
docker run -d \
  --name neo4j \
  -p 7474:7474 \
  -p 7687:7687 \
  -v neo4j_data:/data \
  -v neo4j_logs:/logs \
  -e NEO4J_AUTH=neo4j/password \
  -e NEO4J_PLUGINS=["apoc"] \
  neo4j:latest
```

### API Integration

```python
# Neo4j service client integration
from src.enterprise_llmops.neo4j_service import Neo4jService

neo4j = Neo4jService()
result = await neo4j.execute_query("MATCH (n) RETURN n LIMIT 10")
```

## 📊 Graph Schema

### Core Entities

```cypher
// Employee Entity with Skills
CREATE (e:Employee {
  id: $id,
  name: $name,
  email: $email,
  role: $role,
  department: $department,
  skills: $skills,
  certifications: $certifications
})

// Department Entity
CREATE (d:Department {
  id: $id,
  name: $name,
  manager: $manager,
  budget: $budget,
  location: $location
})

// Project Entity
CREATE (p:Project {
  id: $id,
  name: $name,
  status: $status,
  budget: $budget,
  start_date: $start_date,
  end_date: $end_date
})
```

### Relationships

```cypher
// Employee-Department Relationship
MATCH (e:Employee), (d:Department)
WHERE e.department = d.name
CREATE (e)-[:WORKS_IN]->(d)

// Employee-Project Relationship
MATCH (e:Employee), (p:Project)
WHERE e.id IN p.team_members
CREATE (e)-[:WORKING_ON]->(p)

// Project Dependencies
MATCH (p1:Project), (p2:Project)
WHERE p1.depends_on = p2.id
CREATE (p1)-[:DEPENDS_ON]->(p2)
```

## 🚀 Quick Start

### 1. Access Neo4j Browser

- **URL**: http://localhost:7474
- **Username**: neo4j
- **Password**: password

### 2. Test API Endpoints

```bash
# Health check
curl http://localhost:8080/api/neo4j/health

# Database info
curl http://localhost:8080/api/neo4j/info

# Custom query
curl -X POST http://localhost:8080/api/neo4j/query \
  -H "Content-Type: application/json" \
  -d '{"query": "MATCH (n) RETURN count(n) as total_nodes"}'
```

### 3. GraphRAG Search

```bash
# Semantic search
curl -X POST http://localhost:8080/api/neo4j/graphrag \
  -H "Content-Type: application/json" \
  -d '{"query": "Find employees working on AI projects"}'
```

## 📈 Performance & Scaling

### Optimization Features

- **Index Management**: Automatic index creation for common query patterns
- **Query Caching**: Result caching for frequently accessed data
- **Connection Pooling**: Efficient database connection management
- **Batch Operations**: Bulk data processing capabilities

### Monitoring

- **Query Performance**: Execution time tracking and optimization
- **Memory Usage**: Graph database memory monitoring
- **Connection Status**: Active connection tracking and management
- **Error Rates**: Query failure monitoring and alerting

## 🔗 Integration Examples

### FastAPI Integration

```python
@app.get("/api/neo4j/org-structure")
async def get_org_structure():
    query = """
    MATCH (d:Department)-[:MANAGES]->(e:Employee)
    RETURN d.name as department,
           collect(e.name) as employees
    """
    return await neo4j.execute_query(query)
```

### Gradio Integration

```python
def visualize_org_chart():
    query = """
    MATCH (d:Department)-[:MANAGES]->(e:Employee)
    RETURN d, e
    """
    results = neo4j.execute_query(query)
    return create_graph_visualization(results)
```

## 🛠️ Development

### Code Structure

```
src/enterprise_llmops/
├── neo4j_service.py          # Neo4j service implementation
├── graph_operations.py       # Graph query operations
├── faker_integration.py      # Test data generation
└── analytics/
    ├── org_analysis.py       # Organization analytics
    ├── project_analysis.py   # Project analytics
    └── skills_analysis.py    # Skills analytics
```

### Adding New Queries

1. Define query in `graph_operations.py`
2. Add endpoint in FastAPI application
3. Update documentation
4. Add comprehensive tests

### Testing

```bash
# Test Neo4j connectivity
python -m pytest tests/test_neo4j.py -v

# Test specific endpoints
curl http://localhost:8080/api/neo4j/health
```

## 🚨 Troubleshooting

### Common Issues

1. **Connection Refused**: Ensure Neo4j is running on port 7474
2. **Authentication Failed**: Check username/password configuration
3. **Query Timeout**: Optimize complex queries or increase timeout
4. **Memory Issues**: Monitor Neo4j memory usage and adjust settings

### Debug Mode

```bash
# Enable debug logging
export NEO4J_DEBUG=true
python -m src.enterprise_llmops.main --log-level debug
```

## 📞 Support

For issues and questions:

1. Check the [FastAPI documentation](fastapi-enterprise.md)
2. Review the [troubleshooting guide](../resources/troubleshooting.md)
3. Check the [progress bulletin](../progress-bulletin.md)
4. Access Neo4j browser at http://localhost:7474

---

**Last Updated**: January 19, 2025  
**Version**: 2.1.0  
**Status**: Production Ready  
**Integration**: Full Enterprise Platform Integration
