# Neo4j Graph Data Directory

This directory contains generated Neo4j graph data for Lenovo's organizational structure and B2B client scenarios.

## 🚀 Quick Start

### Generate Graphs
```powershell
# Activate virtual environment
& C:\Users\samne\PycharmProjects\ai_assignments\venv\Scripts\Activate.ps1

# Generate all graphs
python scripts\generate_lenovo_graphs_simple.py
```

### What Gets Generated

1. **Lenovo Organizational Structure** (`lenovo_org_structure.json`)
   - 500+ employees with roles, skills, and hierarchies
   - 15 departments with budgets and headcount
   - 50+ projects with dependencies and teams
   - Global office locations and assignments

2. **B2B Client Scenarios** (`b2b_client_scenarios.json`)
   - 20+ fictional client organizations across industries
   - Lenovo solutions tailored to each client
   - Client organizational structures

3. **Enterprise Graph Patterns**
   - `org_hierarchy_pattern.json` - Organizational hierarchies
   - `project_network_pattern.json` - Project dependencies
   - `knowledge_graph_pattern.json` - Domain knowledge
   - `business_process_pattern.json` - Business processes
   - `customer_journey_pattern.json` - Customer journeys

4. **GraphRAG Integration** (`graphrag_export.json`)
   - Lenovo knowledge base with embeddings
   - Semantic search capabilities
   - Multi-hop reasoning support

## 🔍 Neo4j Queries

### View Organizational Structure
```cypher
// Lenovo organizational hierarchy
MATCH (p:person)-[:reports_to]->(m:person)
RETURN p, m LIMIT 20
```

### Find Project Dependencies
```cypher
// Project dependency network
MATCH (p1:project)-[:depends_on]->(p2:project)
RETURN p1, p2 LIMIT 20
```

### Explore B2B Client Relationships
```cypher
// Client-solution relationships
MATCH (c:client)-[:serves]-(s:solution)
RETURN c, s LIMIT 20
```

### Employee Skills
```cypher
// Employee skills and certifications
MATCH (p:person)-[:has_skill]->(s:skill)
RETURN p, s LIMIT 20
```

## 📊 Graph Statistics

After generation, check `generation_report.json` for:
- Total nodes and edges
- Department and employee counts
- Project and client statistics
- Pattern generation results

## 🔗 Access Points

- **Neo4j Browser**: http://localhost:7474
- **GraphRAG Queries**: Use the Python integration scripts
- **Documentation**: `docs/docs_content/resources/lenovo-graph-structure.md`

## 🛠️ Troubleshooting

### Neo4j Not Running
```bash
# Install and start Neo4j
# Download from: https://neo4j.com/download/
# Set password to 'password' for default connection
```

### Missing Dependencies
```bash
pip install neo4j py2neo faker sentence-transformers numpy
```

### Connection Issues
- Default URI: `bolt://localhost:7687`
- Username: `neo4j`
- Password: `password`

## 📁 File Structure

```
neo4j_data/
├── README.md                           # This file
├── generation_report.json              # Generation summary
├── lenovo_org_structure.json           # Main Lenovo org data
├── b2b_client_scenarios.json           # B2B client data
├── org_hierarchy_pattern.json          # Org hierarchy pattern
├── project_network_pattern.json        # Project network pattern
├── knowledge_graph_pattern.json        # Knowledge graph pattern
├── business_process_pattern.json       # Business process pattern
├── customer_journey_pattern.json       # Customer journey pattern
└── graphrag_export.json                # GraphRAG knowledge base
```

---

**Last Updated**: 2024-12-19  
**Version**: 1.0.0  
**Status**: Production Ready
