#!/usr/bin/env python3
"""
Enterprise Data Generation Script

This script generates comprehensive enterprise data for the Lenovo AAITC Technical Architecture Implementation Plan.
It creates all the data types needed for Phase 1: Enhanced Data Generation & Multi-Database Integration.
"""

import sys
import os
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ai_architecture.enterprise_data_generators.generate_all_data import EnterpriseDataGenerator

def main():
    """Main execution function"""
    print("🚀 Lenovo AAITC Enterprise Data Generation")
    print("=" * 60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # Initialize generator
        print("🔧 Initializing enterprise data generator...")
        generator = EnterpriseDataGenerator()
        
        # Generate all data
        print("\n📊 Starting comprehensive data generation...")
        comprehensive_data = generator.generate_all_data()
        
        # Save all data
        print("\n💾 Saving all generated data...")
        saved_files = generator.save_all_data()
        
        # Export generation report
        print("\n📋 Generating comprehensive report...")
        report_path = generator.export_generation_report()
        
        # Print summary
        print("\n" + "=" * 60)
        print("🎉 Enterprise data generation completed successfully!")
        print("=" * 60)
        
        # Print data flow diagram
        print("\n📊 Data Flow Architecture:")
        print(generator.generate_data_flow_diagram())
        
        # Print file locations
        print(f"\n📁 Generated Files:")
        for data_type, filepath in saved_files.items():
            print(f"  • {data_type}: {filepath}")
        
        print(f"\n📊 Comprehensive Report: {report_path}")
        
        # Print statistics
        print(f"\n📈 Generation Statistics:")
        for data_type, stats in comprehensive_data['statistics'].items():
            print(f"  • {data_type}: {sum(v for k, v in stats.items() if isinstance(v, int))} records")
        
        print(f"\n✅ Phase 1: Enhanced Data Generation & Multi-Database Integration - COMPLETED")
        print(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during data generation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
