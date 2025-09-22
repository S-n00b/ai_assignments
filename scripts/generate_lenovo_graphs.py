#!/usr/bin/env python3
"""
Lenovo Graph Generation Script

This script generates comprehensive Neo4j graphs for Lenovo organizational structure
and B2B client scenarios using Faker data, following Neo4j best practices for
enterprise GraphRAG solutions.

Usage:
    python scripts/generate_lenovo_graphs.py [--neo4j-uri URI] [--neo4j-user USER] [--neo4j-password PASSWORD]
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ai_architecture.lenovo_org_graph_generator import LenovoOrgGraphGenerator
from ai_architecture.enterprise_graph_patterns import EnterpriseGraphPatternGenerator
from ai_architecture.graphrag_neo4j_integration import GraphRAGNeo4jIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LenovoGraphGenerator:
    """Main generator for Lenovo graphs"""
    
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687", 
                 neo4j_user: str = "neo4j", neo4j_password: str = "password"):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        
        # Initialize generators
        self.org_generator = LenovoOrgGraphGenerator(neo4j_uri, neo4j_user, neo4j_password)
        self.pattern_generator = EnterpriseGraphPatternGenerator(neo4j_uri, neo4j_user, neo4j_password)
        self.graphrag_integration = GraphRAGNeo4jIntegration(neo4j_uri, neo4j_user, neo4j_password)
        
        # Create output directory
        self.output_dir = Path("neo4j_data")
        self.output_dir.mkdir(exist_ok=True)
    
    async def generate_all_graphs(self):
        """Generate all Lenovo graphs"""
        logger.info("Starting Lenovo graph generation...")
        
        try:
            # 1. Generate Lenovo organizational structure
            logger.info("Generating Lenovo organizational structure...")
            lenovo_graph = self.org_generator.generate_lenovo_org_structure(
                num_employees=500,
                num_departments=15,
                num_projects=50
            )
            
            # Save to Neo4j
            await self.org_generator.save_to_neo4j(lenovo_graph)
            
            # Export to JSON
            self.org_generator.export_to_json(
                lenovo_graph, 
                str(self.output_dir / "lenovo_org_structure.json")
            )
            
            logger.info(f"Lenovo org structure: {len(lenovo_graph.nodes)} nodes, {len(lenovo_graph.edges)} edges")
            
            # 2. Generate B2B client scenarios
            logger.info("Generating B2B client scenarios...")
            b2b_graph = self.org_generator.generate_b2b_client_scenarios(num_clients=20)
            
            # Save to Neo4j
            await self.org_generator.save_to_neo4j(b2b_graph)
            
            # Export to JSON
            self.org_generator.export_to_json(
                b2b_graph, 
                str(self.output_dir / "b2b_client_scenarios.json")
            )
            
            logger.info(f"B2B client scenarios: {len(b2b_graph.nodes)} nodes, {len(b2b_graph.edges)} edges")
            
            # 3. Generate enterprise graph patterns
            logger.info("Generating enterprise graph patterns...")
            patterns = await self._generate_enterprise_patterns()
            
            # 4. Create GraphRAG-optimized knowledge base
            logger.info("Creating GraphRAG-optimized knowledge base...")
            graphrag_success = self.graphrag_integration.create_lenovo_knowledge_base()
            
            if graphrag_success:
                logger.info("GraphRAG knowledge base created successfully")
                
                # Test GraphRAG queries
                await self._test_graphrag_queries()
                
                # Export GraphRAG data
                self.graphrag_integration.export_graphrag_data(
                    str(self.output_dir / "graphrag_export.json")
                )
            else:
                logger.warning("GraphRAG knowledge base creation failed")
            
            # 5. Generate summary report
            await self._generate_summary_report(lenovo_graph, b2b_graph, patterns)
            
            logger.info("Lenovo graph generation completed successfully!")
            
        except Exception as e:
            logger.error(f"Graph generation failed: {e}")
            raise
    
    async def _generate_enterprise_patterns(self):
        """Generate enterprise graph patterns"""
        patterns = []
        
        # Organizational hierarchy pattern
        logger.info("Generating organizational hierarchy pattern...")
        org_pattern = self.pattern_generator.generate_org_hierarchy_pattern(5, 8)
        await self.pattern_generator.save_pattern_to_neo4j(org_pattern)
        self.pattern_generator.export_pattern_to_json(
            org_pattern, 
            str(self.output_dir / "org_hierarchy_pattern.json")
        )
        patterns.append(("org_hierarchy", org_pattern))
        
        # Project network pattern
        logger.info("Generating project network pattern...")
        project_pattern = self.pattern_generator.generate_project_network_pattern(15)
        await self.pattern_generator.save_pattern_to_neo4j(project_pattern)
        self.pattern_generator.export_pattern_to_json(
            project_pattern, 
            str(self.output_dir / "project_network_pattern.json")
        )
        patterns.append(("project_network", project_pattern))
        
        # Knowledge graph pattern
        logger.info("Generating knowledge graph pattern...")
        knowledge_pattern = self.pattern_generator.generate_knowledge_graph_pattern(40)
        await self.pattern_generator.save_pattern_to_neo4j(knowledge_pattern)
        self.pattern_generator.export_pattern_to_json(
            knowledge_pattern, 
            str(self.output_dir / "knowledge_graph_pattern.json")
        )
        patterns.append(("knowledge_graph", knowledge_pattern))
        
        # Business process pattern
        logger.info("Generating business process pattern...")
        process_pattern = self.pattern_generator.generate_business_process_pattern(12)
        await self.pattern_generator.save_pattern_to_neo4j(process_pattern)
        self.pattern_generator.export_pattern_to_json(
            process_pattern, 
            str(self.output_dir / "business_process_pattern.json")
        )
        patterns.append(("business_process", process_pattern))
        
        # Customer journey pattern
        logger.info("Generating customer journey pattern...")
        journey_pattern = self.pattern_generator.generate_customer_journey_pattern(25)
        await self.pattern_generator.save_pattern_to_neo4j(journey_pattern)
        self.pattern_generator.export_pattern_to_json(
            journey_pattern, 
            str(self.output_dir / "customer_journey_pattern.json")
        )
        patterns.append(("customer_journey", journey_pattern))
        
        return patterns
    
    async def _test_graphrag_queries(self):
        """Test GraphRAG queries"""
        test_queries = [
            "What is Lenovo's flagship laptop series?",
            "Tell me about Lenovo's enterprise solutions",
            "What are Lenovo's AI and machine learning capabilities?",
            "How does Lenovo's sales process work?",
            "What are the key features of ThinkPad laptops?",
            "How does Lenovo support enterprise customers?",
            "What is Lenovo's approach to digital transformation?",
            "Tell me about Lenovo's ThinkSystem servers"
        ]
        
        logger.info("Testing GraphRAG queries...")
        for query in test_queries:
            logger.info(f"Querying: {query}")
            result = self.graphrag_integration.query_lenovo_knowledge(query)
            
            search_results = result.get('search_results', [])
            entities = result.get('context', {}).get('entities', [])
            concepts = result.get('context', {}).get('concepts', [])
            
            logger.info(f"  Found {len(search_results)} documents, {len(entities)} entities, {len(concepts)} concepts")
    
    async def _generate_summary_report(self, lenovo_graph, b2b_graph, patterns):
        """Generate summary report"""
        report = {
            "generation_timestamp": datetime.now().isoformat(),
            "lenovo_organizational_structure": {
                "total_nodes": len(lenovo_graph.nodes),
                "total_edges": len(lenovo_graph.edges),
                "departments": len([n for n in lenovo_graph.nodes if n.node_type.value == "department"]),
                "employees": len([n for n in lenovo_graph.nodes if n.node_type.value == "person"]),
                "projects": len([n for n in lenovo_graph.nodes if n.node_type.value == "project"]),
                "skills": len([n for n in lenovo_graph.nodes if n.node_type.value == "skill"]),
                "locations": len([n for n in lenovo_graph.nodes if n.node_type.value == "location"])
            },
            "b2b_client_scenarios": {
                "total_nodes": len(b2b_graph.nodes),
                "total_edges": len(b2b_graph.edges),
                "clients": len([n for n in b2b_graph.nodes if n.node_type.value == "client"]),
                "solutions": len([n for n in b2b_graph.nodes if n.node_type.value == "solution"])
            },
            "enterprise_patterns": {
                "total_patterns": len(patterns),
                "patterns": [
                    {
                        "name": name,
                        "nodes": len(pattern.nodes),
                        "edges": len(pattern.edges),
                        "type": pattern.pattern_type.value
                    }
                    for name, pattern in patterns
                ]
            },
            "graphrag_integration": {
                "status": "enabled",
                "embedding_model": "all-MiniLM-L6-v2",
                "optimization": "GraphRAG-optimized structure created"
            },
            "neo4j_integration": {
                "uri": self.neo4j_uri,
                "user": self.neo4j_user,
                "status": "connected"
            }
        }
        
        # Save report
        report_file = self.output_dir / "generation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Summary report saved to {report_file}")
        
        # Print summary
        logger.info("=== GENERATION SUMMARY ===")
        logger.info(f"Lenovo Org Structure: {report['lenovo_organizational_structure']['total_nodes']} nodes, {report['lenovo_organizational_structure']['total_edges']} edges")
        logger.info(f"B2B Client Scenarios: {report['b2b_client_scenarios']['total_nodes']} nodes, {report['b2b_client_scenarios']['total_edges']} edges")
        logger.info(f"Enterprise Patterns: {report['enterprise_patterns']['total_patterns']} patterns generated")
        logger.info(f"GraphRAG Integration: {report['graphrag_integration']['status']}")
        logger.info(f"Neo4j Status: {report['neo4j_integration']['status']}")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate Lenovo Neo4j graphs")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j username")
    parser.add_argument("--neo4j-password", default="password", help="Neo4j password")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = LenovoGraphGenerator(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password
    )
    
    # Generate all graphs
    await generator.generate_all_graphs()


if __name__ == "__main__":
    asyncio.run(main())
