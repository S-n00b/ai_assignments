#!/usr/bin/env python3
"""
Simple Lenovo Graph Generation Script

This script generates comprehensive Neo4j graphs for Lenovo organizational structure
and B2B client scenarios using Faker data, following Neo4j best practices.

Usage:
    python scripts/generate_lenovo_graphs_simple.py
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

def check_dependencies():
    """Check if required dependencies are installed"""
    missing_packages = []
    
    try:
        import neo4j
        print("✅ neo4j package available")
    except ImportError:
        missing_packages.append("neo4j>=5.15.0")
    
    try:
        import py2neo
        print("✅ py2neo package available")
    except ImportError:
        missing_packages.append("py2neo>=2021.2.3")
    
    try:
        from faker import Faker
        print("✅ faker package available")
    except ImportError:
        missing_packages.append("faker>=19.0.0")
    
    try:
        import sentence_transformers
        print("✅ sentence-transformers package available")
    except ImportError:
        missing_packages.append("sentence-transformers>=2.2.0")
    
    try:
        import numpy
        print("✅ numpy package available")
    except ImportError:
        missing_packages.append("numpy>=1.24.0")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def test_neo4j_connection(uri="bolt://localhost:7687", user="neo4j", password="password"):
    """Test Neo4j connection"""
    try:
        from neo4j import GraphDatabase
        
        print(f"🔍 Testing Neo4j connection to {uri}...")
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            test_value = result.single()["test"]
            
        driver.close()
        
        if test_value == 1:
            print("✅ Neo4j connection successful")
            return True
        else:
            print("❌ Neo4j connection test failed")
            return False
            
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        print("💡 Make sure Neo4j is running and accessible")
        print("   Default connection: bolt://localhost:7687")
        print("   Username: neo4j, Password: password")
        return False

async def generate_graphs():
    """Generate all Lenovo graphs"""
    print("🚀 Starting Lenovo graph generation...")
    
    try:
        # Import generators
        from ai_architecture.lenovo_org_graph_generator import LenovoOrgGraphGenerator
        from ai_architecture.enterprise_graph_patterns import EnterpriseGraphPatternGenerator
        from ai_architecture.graphrag_neo4j_integration import GraphRAGNeo4jIntegration
        
        # Initialize generators
        org_generator = LenovoOrgGraphGenerator()
        pattern_generator = EnterpriseGraphPatternGenerator()
        graphrag_integration = GraphRAGNeo4jIntegration()
        
        # Create output directory
        output_dir = Path("neo4j_data")
        output_dir.mkdir(exist_ok=True)
        
        print("\n📊 Generating Lenovo organizational structure...")
        lenovo_graph = org_generator.generate_lenovo_org_structure(
            num_employees=500,
            num_departments=15,
            num_projects=50
        )
        
        # Save to Neo4j
        await org_generator.save_to_neo4j(lenovo_graph)
        
        # Export to JSON
        org_generator.export_to_json(
            lenovo_graph, 
            str(output_dir / "lenovo_org_structure.json")
        )
        
        print(f"   ✅ Lenovo org structure: {len(lenovo_graph.nodes)} nodes, {len(lenovo_graph.edges)} edges")
        
        print("\n🏢 Generating B2B client scenarios...")
        b2b_graph = org_generator.generate_b2b_client_scenarios(num_clients=20)
        
        # Save to Neo4j
        await org_generator.save_to_neo4j(b2b_graph)
        
        # Export to JSON
        org_generator.export_to_json(
            b2b_graph, 
            str(output_dir / "b2b_client_scenarios.json")
        )
        
        print(f"   ✅ B2B client scenarios: {len(b2b_graph.nodes)} nodes, {len(b2b_graph.edges)} edges")
        
        print("\n🔧 Generating enterprise graph patterns...")
        patterns = [
            ("org_hierarchy", pattern_generator.generate_org_hierarchy_pattern(5, 8)),
            ("project_network", pattern_generator.generate_project_network_pattern(15)),
            ("knowledge_graph", pattern_generator.generate_knowledge_graph_pattern(40)),
            ("business_process", pattern_generator.generate_business_process_pattern(12)),
            ("customer_journey", pattern_generator.generate_customer_journey_pattern(25))
        ]
        
        for pattern_name, pattern in patterns:
            print(f"   📋 Generating {pattern_name} pattern...")
            await pattern_generator.save_pattern_to_neo4j(pattern)
            pattern_generator.export_pattern_to_json(
                pattern, 
                str(output_dir / f"{pattern_name}_pattern.json")
            )
            print(f"      ✅ {pattern_name}: {len(pattern.nodes)} nodes, {len(pattern.edges)} edges")
        
        print("\n🧠 Creating GraphRAG knowledge base...")
        graphrag_success = graphrag_integration.create_lenovo_knowledge_base()
        
        if graphrag_success:
            print("   ✅ GraphRAG knowledge base created")
            
            # Test GraphRAG queries
            test_queries = [
                "What is Lenovo's flagship laptop series?",
                "Tell me about Lenovo's enterprise solutions",
                "What are Lenovo's AI capabilities?",
                "How does Lenovo's sales process work?"
            ]
            
            print("   🔍 Testing GraphRAG queries...")
            for query in test_queries:
                result = graphrag_integration.query_lenovo_knowledge(query)
                search_results = result.get('search_results', [])
                print(f"      Query: '{query}' → {len(search_results)} documents found")
            
            # Export GraphRAG data
            graphrag_integration.export_graphrag_data(str(output_dir / "graphrag_export.json"))
        else:
            print("   ⚠️  GraphRAG knowledge base creation failed")
        
        # Generate summary
        print("\n📋 Generating summary report...")
        summary = {
            "generation_timestamp": str(asyncio.get_event_loop().time()),
            "lenovo_organizational_structure": {
                "total_nodes": len(lenovo_graph.nodes),
                "total_edges": len(lenovo_graph.edges),
                "departments": len([n for n in lenovo_graph.nodes if n.node_type.value == "department"]),
                "employees": len([n for n in lenovo_graph.nodes if n.node_type.value == "person"]),
                "projects": len([n for n in lenovo_graph.nodes if n.node_type.value == "project"])
            },
            "b2b_client_scenarios": {
                "total_nodes": len(b2b_graph.nodes),
                "total_edges": len(b2b_graph.edges),
                "clients": len([n for n in b2b_graph.nodes if n.node_type.value == "client"]),
                "solutions": len([n for n in b2b_graph.nodes if n.node_type.value == "solution"])
            },
            "enterprise_patterns": {
                "total_patterns": len(patterns),
                "patterns": [{"name": name, "nodes": len(pattern.nodes), "edges": len(pattern.edges)} for name, pattern in patterns]
            },
            "graphrag_integration": {
                "status": "enabled" if graphrag_success else "disabled"
            }
        }
        
        # Save summary
        import json
        with open(output_dir / "generation_report.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("✅ Summary report saved")
        
        # Display results
        print("\n🎉 GRAPH GENERATION COMPLETED!")
        print("=" * 50)
        print(f"📊 Lenovo Org Structure: {summary['lenovo_organizational_structure']['total_nodes']} nodes, {summary['lenovo_organizational_structure']['total_edges']} edges")
        print(f"🏢 B2B Client Scenarios: {summary['b2b_client_scenarios']['total_nodes']} nodes, {summary['b2b_client_scenarios']['total_edges']} edges")
        print(f"🔧 Enterprise Patterns: {summary['enterprise_patterns']['total_patterns']} patterns")
        print(f"🧠 GraphRAG Integration: {summary['graphrag_integration']['status']}")
        print("\n📁 Generated files in 'neo4j_data' directory:")
        
        for file in sorted(output_dir.glob("*.json")):
            size_kb = file.stat().st_size / 1024
            print(f"   📄 {file.name} ({size_kb:.1f} KB)")
        
        print(f"\n🔗 Access Neo4j Browser at: http://localhost:7474")
        print(f"📚 Check documentation: docs/docs_content/resources/lenovo-graph-structure.md")
        
        return True
        
    except Exception as e:
        print(f"❌ Graph generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🎯 Lenovo Graph Generation Script")
    print("=" * 40)
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        return 1
    
    # Test Neo4j connection
    print("\n🔍 Testing Neo4j connection...")
    neo4j_available = test_neo4j_connection()
    
    if not neo4j_available:
        print("\n⚠️  Neo4j not available. Continuing with JSON export only...")
        print("💡 To use Neo4j features:")
        print("   1. Install Neo4j: https://neo4j.com/download/")
        print("   2. Start Neo4j service")
        print("   3. Set password to 'password' (or update connection settings)")
    
    # Generate graphs
    print(f"\n🚀 Starting graph generation...")
    success = asyncio.run(generate_graphs())
    
    if success:
        print("\n✅ All done! Check the 'neo4j_data' directory for generated files.")
        return 0
    else:
        print("\n❌ Generation failed. Check the error messages above.")
        return 1

if __name__ == "__main__":
    exit(main())
