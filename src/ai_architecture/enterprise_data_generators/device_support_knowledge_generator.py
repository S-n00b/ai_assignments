"""
Device Support Knowledge Generator

Generates comprehensive device support knowledge including:
- Device-specific support scenarios
- Troubleshooting guides
- FAQ databases
- Support workflows
- Knowledge base entries
"""

import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import faker

@dataclass
class SupportKnowledgeEntry:
    """Support knowledge entry data structure"""
    entry_id: str
    device_model: str
    issue_category: str
    issue_title: str
    description: str
    symptoms: List[str]
    troubleshooting_steps: List[str]
    resolution: str
    prevention_tips: List[str]
    related_issues: List[str]
    support_level: str
    difficulty: str
    estimated_resolution_time: str

@dataclass
class SupportWorkflow:
    """Support workflow data structure"""
    workflow_id: str
    workflow_name: str
    trigger_conditions: List[str]
    steps: List[Dict[str, Any]]
    escalation_criteria: List[str]
    success_metrics: List[str]
    average_resolution_time: int
    success_rate: float

class DeviceSupportKnowledgeGenerator:
    """Generates comprehensive device support knowledge"""
    
    def __init__(self):
        self.fake = faker.Faker()
        
        self.device_models = [
            "ThinkPad X1 Carbon", "ThinkPad T14", "ThinkPad P16", "ThinkPad L14",
            "ThinkSystem SR650", "ThinkSystem SR850", "ThinkSystem SR950",
            "Moto Edge 50 Pro", "Moto Edge 50", "Moto Edge 40 Pro"
        ]
        
        self.issue_categories = [
            "Hardware Issues", "Software Problems", "Performance Issues",
            "Connectivity Problems", "Battery Issues", "Display Problems",
            "Audio Issues", "Security Concerns", "Compatibility Problems",
            "User Interface Issues", "Data Loss", "System Crashes"
        ]
        
        self.support_levels = ["Basic", "Intermediate", "Advanced", "Expert"]
        self.difficulty_levels = ["Easy", "Medium", "Hard", "Expert"]

    def generate_support_knowledge(self, count: int = 500) -> List[SupportKnowledgeEntry]:
        """Generate support knowledge entries"""
        entries = []
        
        for i in range(count):
            device_model = random.choice(self.device_models)
            issue_category = random.choice(self.issue_categories)
            
            # Generate issue details
            issue_title = self._generate_issue_title(issue_category, device_model)
            description = self._generate_issue_description(issue_category, device_model)
            symptoms = self._generate_symptoms(issue_category)
            troubleshooting_steps = self._generate_troubleshooting_steps(issue_category)
            resolution = self._generate_resolution(issue_category)
            prevention_tips = self._generate_prevention_tips(issue_category)
            related_issues = self._generate_related_issues(issue_category)
            
            entries.append(SupportKnowledgeEntry(
                entry_id=f"SUPPORT_{i+1:04d}",
                device_model=device_model,
                issue_category=issue_category,
                issue_title=issue_title,
                description=description,
                symptoms=symptoms,
                troubleshooting_steps=troubleshooting_steps,
                resolution=resolution,
                prevention_tips=prevention_tips,
                related_issues=related_issues,
                support_level=random.choice(self.support_levels),
                difficulty=random.choice(self.difficulty_levels),
                estimated_resolution_time=random.choice(["5 minutes", "15 minutes", "30 minutes", "1 hour", "2+ hours"])
            ))
        
        return entries

    def _generate_issue_title(self, category: str, device: str) -> str:
        """Generate issue title"""
        titles = {
            "Hardware Issues": [
                f"{device} not powering on",
                f"{device} overheating",
                f"{device} keyboard not responding",
                f"{device} trackpad issues",
                f"{device} USB ports not working"
            ],
            "Software Problems": [
                f"{device} operating system errors",
                f"{device} application crashes",
                f"{device} driver installation issues",
                f"{device} software compatibility problems",
                f"{device} system update failures"
            ],
            "Performance Issues": [
                f"{device} running slowly",
                f"{device} memory issues",
                f"{device} storage problems",
                f"{device} CPU performance issues",
                f"{device} battery draining quickly"
            ],
            "Connectivity Problems": [
                f"{device} WiFi connection issues",
                f"{device} Bluetooth problems",
                f"{device} network adapter errors",
                f"{device} internet connectivity issues",
                f"{device} VPN connection problems"
            ],
            "Display Problems": [
                f"{device} screen flickering",
                f"{device} display resolution issues",
                f"{device} external monitor problems",
                f"{device} screen brightness issues",
                f"{device} graphics driver problems"
            ]
        }
        
        if category in titles:
            return random.choice(titles[category])
        else:
            return f"{device} {category.lower()}"

    def _generate_issue_description(self, category: str, device: str) -> str:
        """Generate issue description"""
        descriptions = {
            "Hardware Issues": f"The {device} is experiencing hardware-related problems that may require physical inspection or component replacement.",
            "Software Problems": f"The {device} is encountering software-related issues that may be resolved through system updates or reinstallation.",
            "Performance Issues": f"The {device} is not performing at expected levels, which may be due to resource constraints or optimization needs.",
            "Connectivity Problems": f"The {device} is having difficulty establishing or maintaining network connections.",
            "Display Problems": f"The {device} display is not functioning correctly, which may affect user experience and productivity."
        }
        
        return descriptions.get(category, f"The {device} is experiencing {category.lower()} that requires troubleshooting and resolution.")

    def _generate_symptoms(self, category: str) -> List[str]:
        """Generate issue symptoms"""
        symptoms_by_category = {
            "Hardware Issues": [
                "Device won't turn on",
                "Unusual noises from device",
                "Physical damage visible",
                "Components not responding",
                "Overheating or excessive heat"
            ],
            "Software Problems": [
                "Error messages appearing",
                "Applications crashing",
                "System freezing or hanging",
                "Blue screen of death",
                "Software not launching"
            ],
            "Performance Issues": [
                "Slow system response",
                "High CPU usage",
                "Memory warnings",
                "Long boot times",
                "Applications running slowly"
            ],
            "Connectivity Problems": [
                "No internet connection",
                "WiFi not detected",
                "Bluetooth pairing failures",
                "Network adapter errors",
                "VPN connection issues"
            ],
            "Display Problems": [
                "Screen flickering",
                "Display resolution issues",
                "External monitor not detected",
                "Screen brightness problems",
                "Graphics artifacts or distortion"
            ]
        }
        
        base_symptoms = symptoms_by_category.get(category, ["General system issues"])
        return random.sample(base_symptoms, random.randint(2, 4))

    def _generate_troubleshooting_steps(self, category: str) -> List[str]:
        """Generate troubleshooting steps"""
        steps_by_category = {
            "Hardware Issues": [
                "Check power connections and cables",
                "Inspect device for physical damage",
                "Test with different power adapter",
                "Remove and reseat components",
                "Contact hardware support if needed"
            ],
            "Software Problems": [
                "Restart the device",
                "Update system software",
                "Check for driver updates",
                "Run system diagnostics",
                "Reinstall problematic software"
            ],
            "Performance Issues": [
                "Close unnecessary applications",
                "Check system resource usage",
                "Update system software",
                "Clear temporary files",
                "Optimize system settings"
            ],
            "Connectivity Problems": [
                "Check network settings",
                "Restart network adapter",
                "Update network drivers",
                "Test with different network",
                "Reset network configuration"
            ],
            "Display Problems": [
                "Check display connections",
                "Update graphics drivers",
                "Adjust display settings",
                "Test with external monitor",
                "Reset display configuration"
            ]
        }
        
        base_steps = steps_by_category.get(category, ["Basic troubleshooting steps"])
        return random.sample(base_steps, random.randint(3, 5))

    def _generate_resolution(self, category: str) -> str:
        """Generate resolution description"""
        resolutions = {
            "Hardware Issues": "Hardware component replacement or repair may be required. Contact Lenovo support for warranty service.",
            "Software Problems": "Software update or reinstallation typically resolves the issue. Follow troubleshooting steps for best results.",
            "Performance Issues": "System optimization and resource management improvements usually resolve performance problems.",
            "Connectivity Problems": "Network configuration updates and driver installations typically resolve connectivity issues.",
            "Display Problems": "Display driver updates and configuration adjustments usually resolve display-related problems."
        }
        
        return resolutions.get(category, "Follow the troubleshooting steps to resolve the issue. Contact support if problems persist.")

    def _generate_prevention_tips(self, category: str) -> List[str]:
        """Generate prevention tips"""
        tips_by_category = {
            "Hardware Issues": [
                "Handle device with care",
                "Use proper power adapters",
                "Keep device clean and dust-free",
                "Avoid extreme temperatures",
                "Regular hardware maintenance"
            ],
            "Software Problems": [
                "Keep system updated",
                "Install software from trusted sources",
                "Regular system maintenance",
                "Backup important data",
                "Use antivirus software"
            ],
            "Performance Issues": [
                "Regular system cleanup",
                "Monitor resource usage",
                "Close unused applications",
                "Regular software updates",
                "Optimize system settings"
            ],
            "Connectivity Problems": [
                "Keep network drivers updated",
                "Use secure networks",
                "Regular network maintenance",
                "Monitor connection quality",
                "Update network settings"
            ],
            "Display Problems": [
                "Update graphics drivers",
                "Adjust display settings properly",
                "Use appropriate resolution",
                "Regular display maintenance",
                "Protect screen from damage"
            ]
        }
        
        base_tips = tips_by_category.get(category, ["General maintenance tips"])
        return random.sample(base_tips, random.randint(2, 4))

    def _generate_related_issues(self, category: str) -> List[str]:
        """Generate related issues"""
        related_issues = {
            "Hardware Issues": ["Power Issues", "Component Failures", "Physical Damage"],
            "Software Problems": ["System Errors", "Application Issues", "Driver Problems"],
            "Performance Issues": ["Resource Constraints", "System Optimization", "Memory Issues"],
            "Connectivity Problems": ["Network Issues", "Driver Problems", "Configuration Issues"],
            "Display Problems": ["Graphics Issues", "Driver Problems", "Hardware Failures"]
        }
        
        return related_issues.get(category, ["Related technical issues"])

    def generate_support_workflows(self, count: int = 50) -> List[SupportWorkflow]:
        """Generate support workflows"""
        workflows = []
        
        workflow_templates = [
            {
                "name": "Hardware Issue Resolution",
                "trigger_conditions": ["Hardware failure", "Physical damage", "Component malfunction"],
                "steps": [
                    {"step": "Initial Assessment", "duration": 15, "role": "Support Agent"},
                    {"step": "Hardware Diagnostics", "duration": 30, "role": "Technical Specialist"},
                    {"step": "Component Testing", "duration": 45, "role": "Hardware Engineer"},
                    {"step": "Replacement Planning", "duration": 20, "role": "Support Manager"},
                    {"step": "Resolution Implementation", "duration": 60, "role": "Service Technician"}
                ],
                "escalation_criteria": ["Warranty issues", "Complex hardware problems", "Data recovery needs"],
                "success_metrics": ["Resolution time", "Customer satisfaction", "First-call resolution"],
                "average_resolution_time": 170,
                "success_rate": 0.85
            },
            {
                "name": "Software Problem Resolution",
                "trigger_conditions": ["Software errors", "Application crashes", "System instability"],
                "steps": [
                    {"step": "Issue Identification", "duration": 10, "role": "Support Agent"},
                    {"step": "Software Diagnostics", "duration": 20, "role": "Technical Specialist"},
                    {"step": "Solution Implementation", "duration": 30, "role": "Software Engineer"},
                    {"step": "Testing and Validation", "duration": 15, "role": "QA Specialist"},
                    {"step": "Customer Communication", "duration": 10, "role": "Support Agent"}
                ],
                "escalation_criteria": ["Complex software issues", "Data corruption", "System reinstallation"],
                "success_metrics": ["Resolution time", "System stability", "User satisfaction"],
                "average_resolution_time": 85,
                "success_rate": 0.90
            },
            {
                "name": "Performance Optimization",
                "trigger_conditions": ["Slow performance", "Resource constraints", "System optimization"],
                "steps": [
                    {"step": "Performance Analysis", "duration": 25, "role": "Performance Specialist"},
                    {"step": "Resource Assessment", "duration": 20, "role": "System Administrator"},
                    {"step": "Optimization Implementation", "duration": 40, "role": "Technical Specialist"},
                    {"step": "Performance Testing", "duration": 20, "role": "QA Specialist"},
                    {"step": "Results Documentation", "duration": 10, "role": "Support Agent"}
                ],
                "escalation_criteria": ["Hardware limitations", "Complex optimization", "System architecture changes"],
                "success_metrics": ["Performance improvement", "Resource utilization", "User satisfaction"],
                "average_resolution_time": 115,
                "success_rate": 0.80
            }
        ]
        
        for i in range(count):
            template = random.choice(workflow_templates)
            
            workflows.append(SupportWorkflow(
                workflow_id=f"WORKFLOW_{i+1:03d}",
                workflow_name=f"{template['name']} {i+1}",
                trigger_conditions=template["trigger_conditions"],
                steps=template["steps"],
                escalation_criteria=template["escalation_criteria"],
                success_metrics=template["success_metrics"],
                average_resolution_time=template["average_resolution_time"],
                success_rate=template["success_rate"]
            ))
        
        return workflows

    def generate_faq_database(self, count: int = 200) -> List[Dict[str, Any]]:
        """Generate FAQ database"""
        faqs = []
        
        faq_categories = [
            "General Questions", "Hardware Issues", "Software Problems",
            "Performance Issues", "Connectivity Problems", "Display Issues",
            "Battery Problems", "Audio Issues", "Security Concerns",
            "Warranty Questions", "Support Options", "Technical Specifications"
        ]
        
        for i in range(count):
            category = random.choice(faq_categories)
            device = random.choice(self.device_models)
            
            faqs.append({
                "faq_id": f"FAQ_{i+1:04d}",
                "category": category,
                "device_model": device,
                "question": self._generate_faq_question(category, device),
                "answer": self._generate_faq_answer(category, device),
                "keywords": self._generate_faq_keywords(category),
                "helpfulness_rating": random.uniform(3.0, 5.0),
                "view_count": random.randint(10, 1000)
            })
        
        return faqs

    def _generate_faq_question(self, category: str, device: str) -> str:
        """Generate FAQ question"""
        questions = {
            "General Questions": [
                f"How do I set up my {device}?",
                f"What accessories are compatible with {device}?",
                f"How do I contact Lenovo support for {device}?",
                f"What warranty coverage does {device} have?",
                f"How do I register my {device}?"
            ],
            "Hardware Issues": [
                f"Why won't my {device} turn on?",
                f"How do I replace the battery in {device}?",
                f"What should I do if {device} overheats?",
                f"How do I clean my {device}?",
                f"Why is my {device} making strange noises?"
            ],
            "Software Problems": [
                f"How do I update the software on {device}?",
                f"Why is my {device} running slowly?",
                f"How do I reset {device} to factory settings?",
                f"What should I do if {device} freezes?",
                f"How do I install drivers for {device}?"
            ],
            "Performance Issues": [
                f"How can I improve {device} performance?",
                f"Why is my {device} using so much memory?",
                f"How do I optimize {device} battery life?",
                f"What causes {device} to run slowly?",
                f"How do I monitor {device} performance?"
            ]
        }
        
        if category in questions:
            return random.choice(questions[category])
        else:
            return f"How do I resolve {category.lower()} with {device}?"

    def _generate_faq_answer(self, category: str, device: str) -> str:
        """Generate FAQ answer"""
        answers = {
            "General Questions": f"For {device} setup and general questions, please refer to the user manual or contact Lenovo support for assistance.",
            "Hardware Issues": f"For {device} hardware issues, please check the troubleshooting guide or contact Lenovo support for warranty service.",
            "Software Problems": f"For {device} software issues, try updating the system software or contact Lenovo support for technical assistance.",
            "Performance Issues": f"For {device} performance optimization, please follow the performance tuning guide or contact Lenovo support for advanced assistance."
        }
        
        return answers.get(category, f"For {category.lower()} with {device}, please contact Lenovo support for assistance.")

    def _generate_faq_keywords(self, category: str) -> List[str]:
        """Generate FAQ keywords"""
        keywords_by_category = {
            "General Questions": ["setup", "configuration", "getting started", "basics"],
            "Hardware Issues": ["hardware", "components", "physical", "repair"],
            "Software Problems": ["software", "applications", "system", "programs"],
            "Performance Issues": ["performance", "speed", "optimization", "efficiency"]
        }
        
        base_keywords = keywords_by_category.get(category, ["support", "help", "assistance"])
        return base_keywords + random.sample(["lenovo", "device", "troubleshooting", "support"], random.randint(1, 2))

    def generate_all_knowledge(self) -> Dict[str, Any]:
        """Generate comprehensive support knowledge"""
        print("Generating device support knowledge...")
        
        # Generate knowledge entries
        knowledge_entries = self.generate_support_knowledge(500)
        
        # Generate workflows
        workflows = self.generate_support_workflows(50)
        
        # Generate FAQ database
        faqs = self.generate_faq_database(200)
        
        # Generate analytics
        analytics = self._generate_knowledge_analytics(knowledge_entries, workflows, faqs)
        
        # Compile results
        result = {
            "generation_timestamp": datetime.now().isoformat(),
            "support_knowledge_entries": [asdict(entry) for entry in knowledge_entries],
            "support_workflows": [asdict(workflow) for workflow in workflows],
            "faq_database": faqs,
            "knowledge_analytics": analytics,
            "statistics": {
                "total_knowledge_entries": len(knowledge_entries),
                "total_workflows": len(workflows),
                "total_faqs": len(faqs),
                "device_models": len(set(entry.device_model for entry in knowledge_entries)),
                "issue_categories": len(set(entry.issue_category for entry in knowledge_entries)),
                "support_levels": len(set(entry.support_level for entry in knowledge_entries))
            }
        }
        
        return result

    def _generate_knowledge_analytics(self, entries: List[SupportKnowledgeEntry], 
                                    workflows: List[SupportWorkflow], faqs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate knowledge analytics"""
        analytics = {
            "total_entries": len(entries),
            "device_distribution": {},
            "category_distribution": {},
            "difficulty_distribution": {},
            "support_level_distribution": {},
            "workflow_effectiveness": {},
            "faq_popularity": {}
        }
        
        # Calculate distributions
        for entry in entries:
            # Device distribution
            if entry.device_model not in analytics["device_distribution"]:
                analytics["device_distribution"][entry.device_model] = 0
            analytics["device_distribution"][entry.device_model] += 1
            
            # Category distribution
            if entry.issue_category not in analytics["category_distribution"]:
                analytics["category_distribution"][entry.issue_category] = 0
            analytics["category_distribution"][entry.issue_category] += 1
            
            # Difficulty distribution
            if entry.difficulty not in analytics["difficulty_distribution"]:
                analytics["difficulty_distribution"][entry.difficulty] = 0
            analytics["difficulty_distribution"][entry.difficulty] += 1
            
            # Support level distribution
            if entry.support_level not in analytics["support_level_distribution"]:
                analytics["support_level_distribution"][entry.support_level] = 0
            analytics["support_level_distribution"][entry.support_level] += 1
        
        # Workflow effectiveness
        for workflow in workflows:
            analytics["workflow_effectiveness"][workflow.workflow_name] = {
                "success_rate": workflow.success_rate,
                "average_time": workflow.average_resolution_time
            }
        
        # FAQ popularity
        for faq in faqs:
            analytics["faq_popularity"][faq["faq_id"]] = {
                "view_count": faq["view_count"],
                "helpfulness": faq["helpfulness_rating"]
            }
        
        return analytics

    def save_to_json(self, data: Dict[str, Any], filename: str = "device_support_knowledge.json"):
        """Save generated data to JSON file"""
        import os
        os.makedirs("data/support_knowledge", exist_ok=True)
        filepath = f"data/support_knowledge/{filename}"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Support knowledge saved to {filepath}")
        return filepath

if __name__ == "__main__":
    generator = DeviceSupportKnowledgeGenerator()
    data = generator.generate_all_knowledge()
    filepath = generator.save_to_json(data)
    print(f"Generated {data['statistics']['total_knowledge_entries']} support knowledge entries")
