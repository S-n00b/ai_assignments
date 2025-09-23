"""
Business Process Data Generator

Generates realistic Lenovo business process data including:
- Business workflows and processes
- Process documentation
- Process metrics and KPIs
- Process optimization opportunities
- Compliance and audit trails
"""

import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import faker

@dataclass
class BusinessProcess:
    """Business process data structure"""
    process_id: str
    process_name: str
    department: str
    process_type: str
    description: str
    steps: List[Dict[str, Any]]
    stakeholders: List[str]
    frequency: str
    average_duration: int
    success_rate: float
    compliance_requirements: List[str]
    automation_level: str
    cost_per_execution: float

@dataclass
class ProcessMetric:
    """Process metric data structure"""
    metric_id: str
    process_id: str
    metric_name: str
    metric_type: str
    value: float
    unit: str
    target_value: float
    measurement_date: str
    trend: str
    status: str

@dataclass
class ProcessOptimization:
    """Process optimization opportunity data structure"""
    optimization_id: str
    process_id: str
    opportunity_name: str
    description: str
    potential_savings: float
    implementation_effort: str
    risk_level: str
    priority: str
    estimated_roi: float
    automation_potential: str

class BusinessProcessDataGenerator:
    """Generates comprehensive business process data"""
    
    def __init__(self):
        self.fake = faker.Faker()
        
        self.departments = [
            "Engineering", "Product Management", "Sales", "Marketing",
            "Customer Support", "IT Operations", "Human Resources",
            "Finance", "Legal", "Research & Development", "Manufacturing",
            "Quality Assurance", "Supply Chain", "Procurement"
        ]
        
        self.process_types = [
            "Operational", "Strategic", "Compliance", "Customer-facing",
            "Internal", "Cross-functional", "Automated", "Manual"
        ]
        
        self.compliance_requirements = [
            "GDPR", "SOX", "ISO 27001", "SOC 2", "HIPAA", "PCI DSS",
            "FDA", "CE Marking", "FCC", "RoHS", "Energy Star"
        ]
        
        self.automation_levels = ["Manual", "Semi-automated", "Automated", "AI-powered"]
        
        self.metric_types = [
            "Efficiency", "Quality", "Cost", "Time", "Customer Satisfaction",
            "Compliance", "Risk", "Performance", "Resource Utilization"
        ]

    def generate_business_processes(self, count: int = 100) -> List[BusinessProcess]:
        """Generate business process definitions"""
        processes = []
        
        process_templates = {
            "Engineering": [
                {
                    "name": "Software Development Lifecycle",
                    "type": "Operational",
                    "steps": [
                        {"step": "Requirements Gathering", "duration": 5, "role": "Product Manager"},
                        {"step": "Technical Design", "duration": 3, "role": "Senior Engineer"},
                        {"step": "Implementation", "duration": 10, "role": "Software Engineer"},
                        {"step": "Code Review", "duration": 2, "role": "Senior Engineer"},
                        {"step": "Testing", "duration": 3, "role": "QA Engineer"},
                        {"step": "Deployment", "duration": 1, "role": "DevOps Engineer"}
                    ],
                    "frequency": "Continuous",
                    "average_duration": 24,
                    "success_rate": 0.85
                },
                {
                    "name": "Bug Triage Process",
                    "type": "Operational",
                    "steps": [
                        {"step": "Bug Report", "duration": 0.5, "role": "User/Support"},
                        {"step": "Initial Assessment", "duration": 1, "role": "QA Engineer"},
                        {"step": "Severity Classification", "duration": 0.5, "role": "Engineering Manager"},
                        {"step": "Assignment", "duration": 0.5, "role": "Engineering Manager"},
                        {"step": "Resolution", "duration": 4, "role": "Software Engineer"},
                        {"step": "Verification", "duration": 1, "role": "QA Engineer"}
                    ],
                    "frequency": "Daily",
                    "average_duration": 7.5,
                    "success_rate": 0.90
                }
            ],
            "Sales": [
                {
                    "name": "Lead to Opportunity Process",
                    "type": "Customer-facing",
                    "steps": [
                        {"step": "Lead Generation", "duration": 2, "role": "Marketing"},
                        {"step": "Lead Qualification", "duration": 1, "role": "Sales Rep"},
                        {"step": "Initial Contact", "duration": 1, "role": "Sales Rep"},
                        {"step": "Needs Assessment", "duration": 2, "role": "Sales Rep"},
                        {"step": "Proposal Creation", "duration": 3, "role": "Sales Rep"},
                        {"step": "Negotiation", "duration": 2, "role": "Sales Manager"},
                        {"step": "Contract Signing", "duration": 1, "role": "Legal"}
                    ],
                    "frequency": "Weekly",
                    "average_duration": 12,
                    "success_rate": 0.25
                }
            ],
            "Customer Support": [
                {
                    "name": "Customer Issue Resolution",
                    "type": "Customer-facing",
                    "steps": [
                        {"step": "Issue Intake", "duration": 0.5, "role": "Support Agent"},
                        {"step": "Initial Triage", "duration": 1, "role": "Support Agent"},
                        {"step": "Investigation", "duration": 2, "role": "Support Engineer"},
                        {"step": "Solution Development", "duration": 3, "role": "Support Engineer"},
                        {"step": "Solution Testing", "duration": 1, "role": "QA Engineer"},
                        {"step": "Customer Communication", "duration": 0.5, "role": "Support Agent"},
                        {"step": "Resolution Verification", "duration": 1, "role": "Support Manager"}
                    ],
                    "frequency": "Daily",
                    "average_duration": 9,
                    "success_rate": 0.80
                }
            ],
            "Manufacturing": [
                {
                    "name": "Product Manufacturing Process",
                    "type": "Operational",
                    "steps": [
                        {"step": "Material Procurement", "duration": 5, "role": "Procurement"},
                        {"step": "Quality Inspection", "duration": 1, "role": "Quality Engineer"},
                        {"step": "Assembly", "duration": 8, "role": "Manufacturing Technician"},
                        {"step": "Testing", "duration": 2, "role": "Test Engineer"},
                        {"step": "Packaging", "duration": 1, "role": "Packaging Specialist"},
                        {"step": "Shipping", "duration": 1, "role": "Logistics Coordinator"}
                    ],
                    "frequency": "Daily",
                    "average_duration": 18,
                    "success_rate": 0.95
                }
            ],
            "Finance": [
                {
                    "name": "Financial Close Process",
                    "type": "Compliance",
                    "steps": [
                        {"step": "Data Collection", "duration": 3, "role": "Financial Analyst"},
                        {"step": "Reconciliation", "duration": 2, "role": "Senior Analyst"},
                        {"step": "Adjustment Entries", "duration": 1, "role": "Senior Analyst"},
                        {"step": "Review", "duration": 1, "role": "Finance Manager"},
                        {"step": "Approval", "duration": 0.5, "role": "Controller"},
                        {"step": "Reporting", "duration": 1, "role": "Financial Analyst"}
                    ],
                    "frequency": "Monthly",
                    "average_duration": 8.5,
                    "success_rate": 0.98
                }
            ]
        }
        
        for i in range(count):
            department = random.choice(self.departments)
            
            # Select process template or create custom
            if department in process_templates and random.random() < 0.7:
                template = random.choice(process_templates[department])
                process_name = template["name"]
                steps = template["steps"]
                frequency = template["frequency"]
                average_duration = template["average_duration"]
                success_rate = template["success_rate"]
            else:
                process_name = f"{department} Process {i+1}"
                steps = self._generate_random_steps()
                frequency = random.choice(["Daily", "Weekly", "Monthly", "Quarterly", "As needed"])
                average_duration = random.randint(2, 48)
                success_rate = random.uniform(0.70, 0.95)
            
            processes.append(BusinessProcess(
                process_id=f"PROC_{i+1:04d}",
                process_name=process_name,
                department=department,
                process_type=random.choice(self.process_types),
                description=self.fake.text(max_nb_chars=200),
                steps=steps,
                stakeholders=self._generate_stakeholders(department),
                frequency=frequency,
                average_duration=average_duration,
                success_rate=success_rate,
                compliance_requirements=random.sample(self.compliance_requirements, random.randint(0, 3)),
                automation_level=random.choice(self.automation_levels),
                cost_per_execution=random.uniform(100, 5000)
            ))
        
        return processes

    def _generate_random_steps(self) -> List[Dict[str, Any]]:
        """Generate random process steps"""
        step_count = random.randint(3, 8)
        steps = []
        
        roles = ["Manager", "Analyst", "Engineer", "Specialist", "Coordinator", "Director"]
        
        for i in range(step_count):
            steps.append({
                "step": f"Step {i+1}: {self.fake.catch_phrase()}",
                "duration": random.randint(1, 10),
                "role": random.choice(roles)
            })
        
        return steps

    def _generate_stakeholders(self, department: str) -> List[str]:
        """Generate stakeholders for a department"""
        stakeholder_templates = {
            "Engineering": ["Software Engineers", "QA Engineers", "DevOps Engineers", "Product Managers"],
            "Sales": ["Sales Representatives", "Account Managers", "Sales Directors", "Marketing Team"],
            "Customer Support": ["Support Agents", "Support Engineers", "Support Managers", "Product Team"],
            "Manufacturing": ["Manufacturing Technicians", "Quality Engineers", "Production Managers", "Supply Chain"],
            "Finance": ["Financial Analysts", "Controllers", "CFO", "Auditors"]
        }
        
        if department in stakeholder_templates:
            return random.sample(stakeholder_templates[department], random.randint(2, 4))
        else:
            return [f"{department} Team", "Management", "External Partners"]

    def generate_process_metrics(self, processes: List[BusinessProcess], days: int = 30) -> List[ProcessMetric]:
        """Generate process metrics over time"""
        metrics = []
        
        for process in processes:
            # Generate metrics for each day
            for day in range(days):
                date = datetime.now() - timedelta(days=day)
                
                # Generate different types of metrics
                metric_types = random.sample(self.metric_types, random.randint(2, 5))
                
                for metric_type in metric_types:
                    # Generate realistic metric values
                    if metric_type == "Efficiency":
                        value = random.uniform(0.6, 0.95)
                        target = 0.85
                    elif metric_type == "Quality":
                        value = random.uniform(0.7, 0.98)
                        target = 0.90
                    elif metric_type == "Cost":
                        value = random.uniform(100, 2000)
                        target = value * random.uniform(0.8, 1.2)
                    elif metric_type == "Time":
                        value = random.uniform(process.average_duration * 0.7, process.average_duration * 1.3)
                        target = process.average_duration
                    elif metric_type == "Customer Satisfaction":
                        value = random.uniform(3.0, 5.0)
                        target = 4.0
                    else:
                        value = random.uniform(0.5, 1.0)
                        target = 0.8
                    
                    # Determine trend
                    trend = random.choice(["Improving", "Stable", "Declining"])
                    status = "Good" if value >= target * 0.9 else "Needs Attention"
                    
                    metrics.append(ProcessMetric(
                        metric_id=f"METRIC_{len(metrics)+1:06d}",
                        process_id=process.process_id,
                        metric_name=f"{metric_type} Metric",
                        metric_type=metric_type,
                        value=round(value, 2),
                        unit=self._get_metric_unit(metric_type),
                        target_value=round(target, 2),
                        measurement_date=date.isoformat(),
                        trend=trend,
                        status=status
                    ))
        
        return metrics

    def _get_metric_unit(self, metric_type: str) -> str:
        """Get appropriate unit for metric type"""
        units = {
            "Efficiency": "%",
            "Quality": "%",
            "Cost": "$",
            "Time": "hours",
            "Customer Satisfaction": "rating",
            "Compliance": "%",
            "Risk": "score",
            "Performance": "score",
            "Resource Utilization": "%"
        }
        return units.get(metric_type, "units")

    def generate_optimization_opportunities(self, processes: List[BusinessProcess]) -> List[ProcessOptimization]:
        """Generate process optimization opportunities"""
        optimizations = []
        
        for process in processes:
            # Generate 1-3 optimization opportunities per process
            opportunity_count = random.randint(1, 3)
            
            for _ in range(opportunity_count):
                opportunity_types = [
                    "Automation", "Streamlining", "Digital Transformation",
                    "Resource Optimization", "Quality Improvement", "Cost Reduction"
                ]
                
                opportunity_type = random.choice(opportunity_types)
                
                optimizations.append(ProcessOptimization(
                    optimization_id=f"OPT_{len(optimizations)+1:04d}",
                    process_id=process.process_id,
                    opportunity_name=f"{opportunity_type} for {process.process_name}",
                    description=self.fake.text(max_nb_chars=150),
                    potential_savings=random.uniform(1000, 50000),
                    implementation_effort=random.choice(["Low", "Medium", "High"]),
                    risk_level=random.choice(["Low", "Medium", "High"]),
                    priority=random.choice(["Low", "Medium", "High", "Critical"]),
                    estimated_roi=random.uniform(1.2, 5.0),
                    automation_potential=random.choice(["Low", "Medium", "High", "Full Automation"])
                ))
        
        return optimizations

    def generate_compliance_audit_trail(self, processes: List[BusinessProcess]) -> List[Dict[str, Any]]:
        """Generate compliance audit trail data"""
        audit_entries = []
        
        for process in processes:
            if process.compliance_requirements:
                # Generate audit entries for compliance requirements
                for requirement in process.compliance_requirements:
                    # Generate 5-15 audit entries per requirement
                    entry_count = random.randint(5, 15)
                    
                    for _ in range(entry_count):
                        audit_date = self.fake.date_between(start_date='-1y', end_date='today')
                        
                        audit_entries.append({
                            "audit_id": f"AUDIT_{len(audit_entries)+1:06d}",
                            "process_id": process.process_id,
                            "compliance_requirement": requirement,
                            "audit_date": audit_date.isoformat(),
                            "auditor": self.fake.name(),
                            "audit_type": random.choice(["Internal", "External", "Self-Assessment"]),
                            "status": random.choice(["Pass", "Fail", "Warning", "Pending"]),
                            "findings": self.fake.text(max_nb_chars=100),
                            "recommendations": self.fake.text(max_nb_chars=100),
                            "next_audit_date": (audit_date + timedelta(days=random.randint(30, 365))).isoformat()
                        })
        
        return audit_entries

    def generate_all_process_data(self) -> Dict[str, Any]:
        """Generate comprehensive business process data"""
        print("Generating business process data...")
        
        # Generate processes
        processes = self.generate_business_processes(100)
        
        # Generate metrics
        metrics = self.generate_process_metrics(processes, 30)
        
        # Generate optimizations
        optimizations = self.generate_optimization_opportunities(processes)
        
        # Generate audit trail
        audit_trail = self.generate_compliance_audit_trail(processes)
        
        # Compile results
        result = {
            "generation_timestamp": datetime.now().isoformat(),
            "business_processes": [asdict(process) for process in processes],
            "process_metrics": [asdict(metric) for metric in metrics],
            "optimization_opportunities": [asdict(opt) for opt in optimizations],
            "compliance_audit_trail": audit_trail,
            "statistics": {
                "total_processes": len(processes),
                "total_metrics": len(metrics),
                "total_optimizations": len(optimizations),
                "total_audit_entries": len(audit_trail),
                "departments": len(set(p.department for p in processes)),
                "compliance_requirements": len(set(req for p in processes for req in p.compliance_requirements))
            }
        }
        
        return result

    def save_to_json(self, data: Dict[str, Any], filename: str = "business_process_data.json"):
        """Save generated data to JSON file"""
        import os
        os.makedirs("data/business_processes", exist_ok=True)
        filepath = f"data/business_processes/{filename}"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Business process data saved to {filepath}")
        return filepath

if __name__ == "__main__":
    generator = BusinessProcessDataGenerator()
    data = generator.generate_all_process_data()
    filepath = generator.save_to_json(data)
    print(f"Generated {data['statistics']['total_processes']} processes with {data['statistics']['total_metrics']} metrics")
