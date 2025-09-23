"""
Generate All Enterprise Data

Comprehensive data generation script that creates all enterprise data types:
- Lenovo device data
- Enterprise user behavior
- Business processes
- Customer journeys
- Technical documentation
- Support knowledge
"""

import json
import os
from datetime import datetime
from typing import Dict, Any

from .lenovo_device_data_generator import LenovoDeviceDataGenerator
from .enterprise_user_behavior_generator import EnterpriseUserBehaviorGenerator
from .business_process_data_generator import BusinessProcessDataGenerator
from .customer_journey_generator import CustomerJourneyGenerator
from .synthetic_enterprise_documents import SyntheticEnterpriseDocuments
from .device_support_knowledge_generator import DeviceSupportKnowledgeGenerator
from .lenovo_technical_documentation import LenovoTechnicalDocumentation

class EnterpriseDataGenerator:
    """Comprehensive enterprise data generator"""
    
    def __init__(self):
        self.generators = {
            "device_data": LenovoDeviceDataGenerator(),
            "user_behavior": EnterpriseUserBehaviorGenerator(),
            "business_processes": BusinessProcessDataGenerator(),
            "customer_journeys": CustomerJourneyGenerator(),
            "enterprise_documents": SyntheticEnterpriseDocuments(),
            "support_knowledge": DeviceSupportKnowledgeGenerator(),
            "technical_docs": LenovoTechnicalDocumentation()
        }
        
        self.generated_data = {}
        self.statistics = {}
    
    def generate_all_data(self) -> Dict[str, Any]:
        """Generate all enterprise data types"""
        print("🚀 Starting comprehensive enterprise data generation...")
        print("=" * 60)
        
        # Generate device data
        print("📱 Generating Lenovo device data...")
        device_data = self.generators["device_data"].generate_all_device_data()
        self.generated_data["device_data"] = device_data
        self.statistics["device_data"] = device_data["statistics"]
        print(f"✅ Generated {device_data['statistics']['total_devices']} devices")
        
        # Generate user behavior data
        print("\n👥 Generating enterprise user behavior data...")
        behavior_data = self.generators["user_behavior"].generate_all_behavior_data()
        self.generated_data["user_behavior"] = behavior_data
        self.statistics["user_behavior"] = behavior_data["statistics"]
        print(f"✅ Generated {behavior_data['statistics']['total_users']} users with {behavior_data['statistics']['total_interactions']} interactions")
        
        # Generate business process data
        print("\n🏢 Generating business process data...")
        process_data = self.generators["business_processes"].generate_all_process_data()
        self.generated_data["business_processes"] = process_data
        self.statistics["business_processes"] = process_data["statistics"]
        print(f"✅ Generated {process_data['statistics']['total_processes']} processes with {process_data['statistics']['total_metrics']} metrics")
        
        # Generate customer journey data
        print("\n🛣️ Generating customer journey data...")
        journey_data = self.generators["customer_journeys"].generate_all_journey_data()
        self.generated_data["customer_journeys"] = journey_data
        self.statistics["customer_journeys"] = journey_data["statistics"]
        print(f"✅ Generated {journey_data['statistics']['total_journeys']} customer journeys")
        
        # Generate enterprise documents
        print("\n📄 Generating synthetic enterprise documents...")
        doc_data = self.generators["enterprise_documents"].generate_all_documents()
        self.generated_data["enterprise_documents"] = doc_data
        self.statistics["enterprise_documents"] = doc_data["statistics"]
        print(f"✅ Generated {doc_data['statistics']['total_documents']} documents")
        
        # Generate support knowledge
        print("\n🔧 Generating device support knowledge...")
        support_data = self.generators["support_knowledge"].generate_all_knowledge()
        self.generated_data["support_knowledge"] = support_data
        self.statistics["support_knowledge"] = support_data["statistics"]
        print(f"✅ Generated {support_data['statistics']['total_knowledge_entries']} support entries")
        
        # Generate technical documentation
        print("\n📚 Generating Lenovo technical documentation...")
        tech_doc_data = self.generators["technical_docs"].generate_all_technical_docs()
        self.generated_data["technical_docs"] = tech_doc_data
        self.statistics["technical_docs"] = tech_doc_data["statistics"]
        print(f"✅ Generated {tech_doc_data['statistics']['total_documents']} technical documents")
        
        # Compile comprehensive results
        comprehensive_data = {
            "generation_timestamp": datetime.now().isoformat(),
            "data_types": list(self.generated_data.keys()),
            "statistics": self.statistics,
            "data": self.generated_data,
            "summary": self._generate_summary()
        }
        
        print("\n" + "=" * 60)
        print("🎉 Enterprise data generation completed successfully!")
        print(f"📊 Total data types generated: {len(self.generated_data)}")
        print(f"📈 Total records generated: {sum(stats.get('total_devices', 0) + stats.get('total_users', 0) + stats.get('total_processes', 0) + stats.get('total_journeys', 0) + stats.get('total_documents', 0) + stats.get('total_knowledge_entries', 0) for stats in self.statistics.values())}")
        
        return comprehensive_data
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive data summary"""
        total_records = 0
        data_types = []
        
        for data_type, stats in self.statistics.items():
            data_types.append({
                "type": data_type,
                "records": sum(v for k, v in stats.items() if isinstance(v, int)),
                "categories": len([k for k in stats.keys() if not k.startswith('total_')])
            })
            total_records += sum(v for k, v in stats.items() if isinstance(v, int))
        
        return {
            "total_data_types": len(self.statistics),
            "total_records": total_records,
            "data_types": data_types,
            "generation_time": datetime.now().isoformat()
        }
    
    def save_all_data(self, base_filename: str = "comprehensive_enterprise_data") -> Dict[str, str]:
        """Save all generated data to files"""
        print("\n💾 Saving all generated data...")
        
        saved_files = {}
        
        # Save individual data types
        for data_type, data in self.generated_data.items():
            filename = f"{base_filename}_{data_type}.json"
            filepath = self._save_data(data, filename)
            saved_files[data_type] = filepath
            print(f"✅ Saved {data_type} to {filepath}")
        
        # Save comprehensive data
        comprehensive_filename = f"{base_filename}_comprehensive.json"
        comprehensive_filepath = self._save_data({
            "generation_timestamp": datetime.now().isoformat(),
            "data_types": list(self.generated_data.keys()),
            "statistics": self.statistics,
            "data": self.generated_data,
            "summary": self._generate_summary()
        }, comprehensive_filename)
        saved_files["comprehensive"] = comprehensive_filepath
        
        print(f"✅ Saved comprehensive data to {comprehensive_filepath}")
        
        return saved_files
    
    def _save_data(self, data: Dict[str, Any], filename: str) -> str:
        """Save data to JSON file"""
        os.makedirs("data/enterprise_data", exist_ok=True)
        filepath = f"data/enterprise_data/{filename}"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def generate_data_flow_diagram(self) -> str:
        """Generate data flow diagram"""
        diagram = """
        Enterprise Data Generation Flow:
        
        ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
        │   Device Data   │    │  User Behavior  │    │ Business Procs  │
        │   Generator     │    │   Generator    │    │   Generator     │
        └─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
                  │                      │                      │
                  └──────────────────────┼──────────────────────┘
                                        │
                                        ▼
        ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
        │ Customer Journeys│    │Enterprise Docs │    │ Support Knowledge│
        │   Generator     │    │   Generator    │    │   Generator     │
        └─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
                  │                      │                      │
                  └──────────────────────┼──────────────────────┘
                                        │
                                        ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │                Unified Data Synchronization                    │
        │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
        │  │  ChromaDB   │ │    Neo4j    │ │   DuckDB    │ │   MLflow    ││
        │  │   Vector    │ │    Graph    │ │  Analytics  │ │Experiment  ││
        │  │  Database   │ │  Database   │ │  Database   │ │  Tracking   ││
        │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
        └─────────────────────────────────────────────────────────────────┘
        """
        return diagram
    
    def export_generation_report(self, filename: str = "enterprise_data_generation_report.json") -> str:
        """Export comprehensive generation report"""
        report = {
            "generation_timestamp": datetime.now().isoformat(),
            "generation_summary": self._generate_summary(),
            "data_statistics": self.statistics,
            "data_flow_diagram": self.generate_data_flow_diagram(),
            "file_locations": {},
            "next_steps": [
                "Populate ChromaDB with vector embeddings",
                "Create Neo4j knowledge graphs",
                "Setup DuckDB analytics database",
                "Initialize MLflow experiment tracking",
                "Start real-time data synchronization"
            ]
        }
        
        os.makedirs("data/reports", exist_ok=True)
        filepath = f"data/reports/{filename}"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Generation report saved to {filepath}")
        return filepath

def main():
    """Main execution function"""
    print("🚀 Lenovo AAITC Enterprise Data Generation")
    print("=" * 50)
    
    # Initialize generator
    generator = EnterpriseDataGenerator()
    
    # Generate all data
    comprehensive_data = generator.generate_all_data()
    
    # Save all data
    saved_files = generator.save_all_data()
    
    # Export generation report
    report_path = generator.export_generation_report()
    
    # Print data flow diagram
    print("\n📊 Data Flow Diagram:")
    print(generator.generate_data_flow_diagram())
    
    print(f"\n🎉 All enterprise data generated successfully!")
    print(f"📁 Files saved: {len(saved_files)}")
    print(f"📊 Report: {report_path}")

if __name__ == "__main__":
    main()
