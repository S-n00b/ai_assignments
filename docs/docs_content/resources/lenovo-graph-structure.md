# Lenovo Graph Structure Documentation

## 🎯 Overview

This documentation describes the comprehensive Neo4j graph structure generated for Lenovo's organizational data and B2B client scenarios. The graph is designed following Neo4j best practices for enterprise GraphRAG solutions and agentic workflows.

## 🚀 Key Features

### Core Capabilities

- **Lenovo Organizational Structure**: Complete hierarchy with departments, employees, roles, and relationships
- **B2B Client Scenarios**: Fictional client organizations with their structures and Lenovo solutions
- **Enterprise Graph Patterns**: Common enterprise patterns (org charts, project networks, knowledge graphs)
- **GraphRAG Integration**: Optimized structure for semantic search and knowledge retrieval
- **Faker-Generated Data**: Realistic data using Faker library for comprehensive testing

### Integration Features

- **Neo4j Best Practices**: Entity and semantic mapping following Neo4j guidelines
- **Multi-Hop Reasoning**: Support for complex relationship traversal
- **Semantic Search**: Vector embeddings for similarity search
- **Context Aggregation**: Multi-source context gathering for agentic workflows

## 📊 Graph Structure

### Node Types

#### Lenovo Organizational Structure

- **Person**: Employees with roles, skills, and personal information
- **Department**: Organizational departments with budgets and headcount
- **Project**: Lenovo projects with status, budget, and team assignments
- **Skill**: Technical and business skills with proficiency levels
- **Certification**: Professional certifications with validity periods
- **Location**: Lenovo office locations worldwide

#### B2B Client Scenarios

- **Client**: B2B client organizations with industry and company information
- **Solution**: Lenovo solutions provided to clients
- **Department**: Client organizational departments
- **Person**: Client employees and stakeholders

#### Enterprise Graph Patterns

- **Business Process**: Business processes with stages and dependencies
- **Journey Stage**: Customer journey stages with touchpoints
- **Knowledge Concept**: Domain knowledge concepts and relationships
- **Touchpoint**: Customer interaction touchpoints

#### GraphRAG Integration

- **Document**: Text documents with embeddings
- **Entity**: Extracted entities from documents
- **Concept**: Extracted concepts from documents
- **Context**: Contextual information for queries

### Relationship Types

#### Organizational Relationships

- **reports_to**: Hierarchical reporting relationships
- **works_in**: Employee-department assignments
- **manages**: Management relationships
- **collaborates_with**: Peer collaboration relationships
- **leads**: Project leadership relationships
- **participates_in**: Project participation

#### Knowledge Relationships

- **contains**: Document-entity relationships
- **references**: Document-concept relationships
- **supports**: Evidence-supporting relationships
- **contradicts**: Conflicting information relationships
- **similar_to**: Similarity relationships
- **part_of**: Composition relationships

#### Business Relationships

- **serves**: Solution-client relationships
- **depends_on**: Process and project dependencies
- **follows**: Sequential process stages
- **progresses_to**: Customer journey progression
- **includes**: Stage-touchpoint relationships

## 🌐 Service Integration

### Neo4j Configuration

- **URI**: bolt://localhost:7687
- **Authentication**: Username/password based
- **Database**: Default database
- **Indexes**: Optimized for GraphRAG queries

### GraphRAG Integration

- **Embedding Model**: all-MiniLM-L6-v2
- **Vector Search**: Cosine similarity for semantic search
- **Multi-Hop Reasoning**: Up to 3 hops for relationship traversal
- **Context Aggregation**: Multi-source context gathering

## 🔧 Configuration

### Required Dependencies

```bash
# Core Neo4j dependencies
neo4j>=5.15.0
py2neo>=2021.2.3

# Data generation
faker>=19.0.0

# GraphRAG integration
sentence-transformers>=2.2.0
numpy>=1.24.0
```

### Environment Setup

```powershell
# Activate virtual environment
& C:\Users\samne\PycharmProjects\ai_assignments\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r config/requirements.txt
```

## 📚 Usage Examples

### Generate Lenovo Graphs

```powershell
# Run the PowerShell script
.\scripts\generate-lenovo-graphs.ps1

# Or run Python script directly
python scripts\generate_lenovo_graphs.py
```

### Neo4j Queries

#### View Organizational Structure

```cypher
// Lenovo organizational hierarchy
MATCH (p:person)-[:reports_to]->(m:person)
RETURN p, m LIMIT 20
```

#### Find Project Dependencies

```cypher
// Project dependency network
MATCH (p1:project)-[:depends_on]->(p2:project)
RETURN p1, p2 LIMIT 20
```

#### Explore B2B Client Relationships

```cypher
// Client-solution relationships
MATCH (c:client)-[:serves]-(s:solution)
RETURN c, s LIMIT 20
```

#### Semantic Search

```cypher
// Find documents similar to query
MATCH (d:document)
WHERE d.embedding IS NOT NULL
WITH d, gds.similarity.cosine(d.embedding, $query_embedding) AS similarity
WHERE similarity > 0.5
RETURN d.text_content, similarity
ORDER BY similarity DESC
LIMIT 10
```

### GraphRAG Queries

#### Knowledge Retrieval

```python
from src.ai_architecture.graphrag_neo4j_integration import GraphRAGNeo4jIntegration

# Initialize GraphRAG integration
graphrag = GraphRAGNeo4jIntegration()

# Query knowledge base
result = graphrag.query_lenovo_knowledge("What is Lenovo's flagship laptop series?")
print(f"Found {len(result['search_results'])} relevant documents")
```

#### Context Aggregation

```python
# Aggregate context from multiple sources
context = graphrag.context_aggregation("Lenovo enterprise solutions", context_size=5)
print(f"Entities: {context['entities']}")
print(f"Concepts: {context['concepts']}")
```

## 🛠️ Development

### Adding New Node Types

```python
# Define new node type
class NewNodeType(Enum):
    CUSTOM_NODE = "custom_node"

# Create node
node = GraphNode(
    id="custom_1",
    label="Custom Node",
    node_type=NewNodeType.CUSTOM_NODE,
    properties={"custom_property": "value"}
)
```

### Adding New Relationships

```python
# Define new relationship type
class NewRelationshipType(Enum):
    CUSTOM_REL = "custom_relationship"

# Create relationship
edge = GraphEdge(
    id="custom_rel_1",
    source="node1",
    target="node2",
    relationship_type=NewRelationshipType.CUSTOM_REL,
    properties={"custom_prop": "value"}
)
```

### Custom Graph Patterns

```python
# Create custom enterprise pattern
def generate_custom_pattern(self, config):
    nodes = []
    edges = []

    # Generate nodes based on configuration
    for i in range(config['num_nodes']):
        node = GraphNode(
            id=f"custom_{i+1}",
            label=f"Custom Node {i+1}",
            node_type=GraphNodeType.CUSTOM,
            properties={"index": i+1}
        )
        nodes.append(node)

    # Generate relationships
    for i in range(len(nodes) - 1):
        edge = GraphEdge(
            id=f"rel_{i+1}",
            source=nodes[i].id,
            target=nodes[i+1].id,
            relationship_type=RelationshipType.CONNECTS,
            properties={"sequence": i+1}
        )
        edges.append(edge)

    return EnterpriseGraphPattern(
        pattern_type=EnterprisePatternType.CUSTOM,
        nodes=nodes,
        edges=edges,
        metadata={"custom": True}
    )
```

## 🚨 Troubleshooting

### Common Issues

#### Neo4j Connection Failed

```bash
# Check Neo4j status
neo4j status

# Start Neo4j
neo4j start

# Check connection
cypher-shell -u neo4j -p password "RETURN 1"
```

#### Missing Dependencies

```bash
# Install missing packages
pip install neo4j py2neo faker sentence-transformers numpy

# Verify installation
python -c "import neo4j, py2neo, faker, sentence_transformers, numpy; print('All packages available')"
```

#### Graph Generation Fails

```bash
# Check logs
type logs\llmops.log

# Run with verbose output
python scripts\generate_lenovo_graphs.py --verbose

# Check Neo4j logs
neo4j logs
```

### Debug Procedures

#### Test Neo4j Connection

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
with driver.session() as session:
    result = session.run("RETURN 1 as test")
    print(result.single())
driver.close()
```

#### Verify Graph Structure

```cypher
// Check node counts by type
MATCH (n)
RETURN labels(n)[0] as node_type, count(n) as count
ORDER BY count DESC

// Check relationship counts by type
MATCH ()-[r]->()
RETURN type(r) as relationship_type, count(r) as count
ORDER BY count DESC
```

## 📞 Support

### Resources

- **Neo4j Documentation**: [Neo4j Graph Database](https://neo4j.com/docs/)
- **GraphRAG Research**: [Microsoft GraphRAG](https://github.com/microsoft/graphrag)
- **Faker Documentation**: [Faker Library](https://faker.readthedocs.io/)

### Getting Help

- Check the troubleshooting section above
- Review Neo4j logs for connection issues
- Verify all dependencies are installed correctly
- Test with smaller datasets first

---

**Last Updated**: 2024-12-19  
**Version**: 1.0.0  
**Status**: Production Ready  
**Integration**: Full Neo4j and GraphRAG Integration
