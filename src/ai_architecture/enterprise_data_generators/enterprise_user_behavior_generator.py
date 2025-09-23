"""
Enterprise User Behavior Generator

Generates realistic enterprise user behavior patterns including:
- User interaction patterns and workflows
- Role-based access patterns
- Department-specific usage patterns
- Device usage patterns
- Application usage analytics
"""

import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import faker

@dataclass
class UserProfile:
    """User profile data structure"""
    user_id: str
    username: str
    email: str
    department: str
    role: str
    seniority_level: str
    device_preference: str
    primary_use_case: str
    work_schedule: str
    location: str
    team_size: int
    manager_id: Optional[str] = None

@dataclass
class UserInteraction:
    """User interaction data structure"""
    interaction_id: str
    user_id: str
    timestamp: str
    application: str
    action: str
    duration_minutes: int
    device_type: str
    location: str
    success: bool
    error_message: Optional[str] = None

@dataclass
class WorkflowPattern:
    """Workflow pattern data structure"""
    workflow_id: str
    workflow_name: str
    department: str
    steps: List[Dict[str, Any]]
    average_duration: int
    success_rate: float
    frequency: str
    participants: List[str]

class EnterpriseUserBehaviorGenerator:
    """Generates comprehensive enterprise user behavior data"""
    
    def __init__(self):
        self.fake = faker.Faker()
        
        self.departments = [
            "Engineering", "Product Management", "Sales", "Marketing",
            "Customer Support", "IT Operations", "Human Resources",
            "Finance", "Legal", "Research & Development"
        ]
        
        self.roles = {
            "Engineering": ["Software Engineer", "Senior Engineer", "Principal Engineer", "Engineering Manager"],
            "Product Management": ["Product Manager", "Senior Product Manager", "Product Director", "VP Product"],
            "Sales": ["Sales Representative", "Account Manager", "Sales Director", "VP Sales"],
            "Marketing": ["Marketing Specialist", "Marketing Manager", "Marketing Director", "CMO"],
            "Customer Support": ["Support Agent", "Senior Support Agent", "Support Manager", "VP Support"],
            "IT Operations": ["System Administrator", "DevOps Engineer", "IT Manager", "CTO"],
            "Human Resources": ["HR Specialist", "HR Manager", "HR Director", "CHRO"],
            "Finance": ["Financial Analyst", "Finance Manager", "CFO", "Controller"],
            "Legal": ["Legal Counsel", "Senior Legal Counsel", "Legal Director", "General Counsel"],
            "Research & Development": ["Research Scientist", "Senior Scientist", "Research Director", "VP R&D"]
        }
        
        self.seniority_levels = ["Junior", "Mid-level", "Senior", "Principal", "Director", "VP", "C-Level"]
        
        self.device_preferences = {
            "Engineering": ["ThinkPad P16", "ThinkPad T14", "ThinkPad X1 Carbon"],
            "Product Management": ["ThinkPad X1 Yoga", "ThinkPad T14", "ThinkPad L14"],
            "Sales": ["ThinkPad T14", "ThinkPad L14", "Moto Edge 50 Pro"],
            "Marketing": ["ThinkPad X1 Yoga", "ThinkPad T14", "Moto Edge 50 Pro"],
            "Customer Support": ["ThinkPad T14", "ThinkPad L14", "Moto Edge 50 Pro"],
            "IT Operations": ["ThinkSystem SR650", "ThinkPad P16", "ThinkPad T14"],
            "Human Resources": ["ThinkPad T14", "ThinkPad L14", "Moto Edge 50 Pro"],
            "Finance": ["ThinkPad T14", "ThinkPad L14", "ThinkPad X1 Carbon"],
            "Legal": ["ThinkPad T14", "ThinkPad X1 Carbon", "ThinkPad L14"],
            "Research & Development": ["ThinkSystem SR850", "ThinkPad P16", "ThinkPad T14"]
        }
        
        self.applications = [
            "Microsoft Office 365", "Slack", "Microsoft Teams", "Zoom", "Salesforce",
            "Jira", "Confluence", "GitHub", "Docker", "Kubernetes", "AWS Console",
            "Azure Portal", "Google Workspace", "Adobe Creative Suite", "Tableau",
            "Power BI", "ServiceNow", "Workday", "SAP", "Oracle"
        ]
        
        self.actions = [
            "login", "logout", "create_document", "edit_document", "share_document",
            "schedule_meeting", "join_meeting", "send_message", "upload_file",
            "download_file", "search", "view_report", "create_report", "approve_request",
            "reject_request", "submit_form", "review_document", "comment", "like"
        ]

    def generate_user_profiles(self, count: int = 200) -> List[UserProfile]:
        """Generate enterprise user profiles"""
        profiles = []
        
        for i in range(count):
            department = random.choice(self.departments)
            role = random.choice(self.roles[department])
            seniority = random.choice(self.seniority_levels)
            
            # Generate manager relationships
            manager_id = None
            if seniority not in ["C-Level", "VP"]:
                manager_id = f"USER_{random.randint(1, count)}"
            
            profiles.append(UserProfile(
                user_id=f"USER_{i+1:04d}",
                username=self.fake.user_name(),
                email=self.fake.email(),
                department=department,
                role=role,
                seniority_level=seniority,
                device_preference=random.choice(self.device_preferences[department]),
                primary_use_case=random.choice([
                    "Software Development", "Data Analysis", "Project Management",
                    "Customer Relations", "Content Creation", "System Administration",
                    "Financial Analysis", "Legal Research", "Scientific Research"
                ]),
                work_schedule=random.choice([
                    "9-5 Standard", "Flexible Hours", "Shift Work", "Remote", "Hybrid"
                ]),
                location=random.choice([
                    "New York", "San Francisco", "London", "Singapore", "Tokyo",
                    "Remote", "Hybrid", "On-site"
                ]),
                team_size=random.randint(3, 15),
                manager_id=manager_id
            ))
        
        return profiles

    def generate_user_interactions(self, profiles: List[UserProfile], days: int = 30) -> List[UserInteraction]:
        """Generate user interaction patterns"""
        interactions = []
        
        # Generate interactions for each user over the specified period
        for profile in profiles:
            # Determine interaction frequency based on role and department
            daily_interactions = self._get_daily_interaction_count(profile)
            
            for day in range(days):
                date = datetime.now() - timedelta(days=day)
                
                # Generate interactions for this day
                day_interactions = random.randint(
                    max(1, daily_interactions - 10),
                    daily_interactions + 10
                )
                
                for _ in range(day_interactions):
                    # Generate random time during work hours
                    hour = random.randint(8, 18)
                    minute = random.randint(0, 59)
                    timestamp = date.replace(hour=hour, minute=minute)
                    
                    application = random.choice(self.applications)
                    action = random.choice(self.actions)
                    duration = random.randint(1, 60)
                    
                    # Determine success rate based on user seniority
                    success_rate = self._get_success_rate(profile.seniority_level)
                    success = random.random() < success_rate
                    
                    error_message = None
                    if not success:
                        error_message = random.choice([
                            "Connection timeout", "Authentication failed", "Permission denied",
                            "Resource not found", "Server error", "Network error"
                        ])
                    
                    interactions.append(UserInteraction(
                        interaction_id=f"INT_{len(interactions)+1:06d}",
                        user_id=profile.user_id,
                        timestamp=timestamp.isoformat(),
                        application=application,
                        action=action,
                        duration_minutes=duration,
                        device_type=profile.device_preference,
                        location=profile.location,
                        success=success,
                        error_message=error_message
                    ))
        
        return interactions

    def generate_workflow_patterns(self, profiles: List[UserProfile]) -> List[WorkflowPattern]:
        """Generate enterprise workflow patterns"""
        workflows = []
        
        workflow_templates = {
            "Engineering": [
                {
                    "name": "Code Review Process",
                    "steps": [
                        {"action": "create_pull_request", "duration": 30},
                        {"action": "assign_reviewers", "duration": 5},
                        {"action": "code_review", "duration": 60},
                        {"action": "address_feedback", "duration": 45},
                        {"action": "merge_code", "duration": 10}
                    ],
                    "average_duration": 150,
                    "success_rate": 0.85
                },
                {
                    "name": "Bug Fix Workflow",
                    "steps": [
                        {"action": "report_bug", "duration": 15},
                        {"action": "investigate_issue", "duration": 120},
                        {"action": "implement_fix", "duration": 90},
                        {"action": "test_fix", "duration": 60},
                        {"action": "deploy_fix", "duration": 30}
                    ],
                    "average_duration": 315,
                    "success_rate": 0.75
                }
            ],
            "Sales": [
                {
                    "name": "Lead Qualification Process",
                    "steps": [
                        {"action": "receive_lead", "duration": 5},
                        {"action": "qualify_lead", "duration": 30},
                        {"action": "schedule_demo", "duration": 15},
                        {"action": "conduct_demo", "duration": 60},
                        {"action": "follow_up", "duration": 20}
                    ],
                    "average_duration": 130,
                    "success_rate": 0.70
                }
            ],
            "Customer Support": [
                {
                    "name": "Ticket Resolution Process",
                    "steps": [
                        {"action": "receive_ticket", "duration": 5},
                        {"action": "categorize_issue", "duration": 10},
                        {"action": "investigate_issue", "duration": 45},
                        {"action": "provide_solution", "duration": 30},
                        {"action": "close_ticket", "duration": 10}
                    ],
                    "average_duration": 100,
                    "success_rate": 0.80
                }
            ]
        }
        
        for department in self.departments:
            if department in workflow_templates:
                for template in workflow_templates[department]:
                    # Get department users
                    dept_users = [p for p in profiles if p.department == department]
                    
                    if dept_users:
                        workflows.append(WorkflowPattern(
                            workflow_id=f"WF_{len(workflows)+1:03d}",
                            workflow_name=template["name"],
                            department=department,
                            steps=template["steps"],
                            average_duration=template["average_duration"],
                            success_rate=template["success_rate"],
                            frequency=random.choice(["Daily", "Weekly", "Monthly", "As needed"]),
                            participants=[user.user_id for user in random.sample(dept_users, min(5, len(dept_users)))]
                        ))
        
        return workflows

    def _get_daily_interaction_count(self, profile: UserProfile) -> int:
        """Calculate daily interaction count based on user profile"""
        base_count = 50
        
        # Adjust based on seniority
        seniority_multiplier = {
            "Junior": 0.8,
            "Mid-level": 1.0,
            "Senior": 1.2,
            "Principal": 1.1,
            "Director": 0.9,
            "VP": 0.8,
            "C-Level": 0.7
        }
        
        # Adjust based on department
        department_multiplier = {
            "Engineering": 1.3,
            "Product Management": 1.1,
            "Sales": 1.2,
            "Marketing": 1.0,
            "Customer Support": 1.4,
            "IT Operations": 1.1,
            "Human Resources": 0.9,
            "Finance": 0.8,
            "Legal": 0.7,
            "Research & Development": 1.0
        }
        
        multiplier = seniority_multiplier.get(profile.seniority_level, 1.0) * \
                   department_multiplier.get(profile.department, 1.0)
        
        return int(base_count * multiplier)

    def _get_success_rate(self, seniority_level: str) -> float:
        """Calculate success rate based on seniority level"""
        success_rates = {
            "Junior": 0.75,
            "Mid-level": 0.85,
            "Senior": 0.90,
            "Principal": 0.92,
            "Director": 0.88,
            "VP": 0.85,
            "C-Level": 0.80
        }
        
        return success_rates.get(seniority_level, 0.80)

    def generate_usage_analytics(self, interactions: List[UserInteraction]) -> Dict[str, Any]:
        """Generate usage analytics from interactions"""
        analytics = {
            "total_interactions": len(interactions),
            "successful_interactions": len([i for i in interactions if i.success]),
            "failed_interactions": len([i for i in interactions if not i.success]),
            "success_rate": len([i for i in interactions if i.success]) / len(interactions) if interactions else 0,
            "application_usage": {},
            "device_usage": {},
            "hourly_patterns": {},
            "department_analytics": {}
        }
        
        # Application usage
        for interaction in interactions:
            app = interaction.application
            if app not in analytics["application_usage"]:
                analytics["application_usage"][app] = {"count": 0, "total_duration": 0}
            analytics["application_usage"][app]["count"] += 1
            analytics["application_usage"][app]["total_duration"] += interaction.duration_minutes
        
        # Device usage
        for interaction in interactions:
            device = interaction.device_type
            if device not in analytics["device_usage"]:
                analytics["device_usage"][device] = 0
            analytics["device_usage"][device] += 1
        
        # Hourly patterns
        for interaction in interactions:
            hour = datetime.fromisoformat(interaction.timestamp).hour
            if hour not in analytics["hourly_patterns"]:
                analytics["hourly_patterns"][hour] = 0
            analytics["hourly_patterns"][hour] += 1
        
        return analytics

    def generate_all_behavior_data(self) -> Dict[str, Any]:
        """Generate comprehensive user behavior data"""
        print("Generating enterprise user behavior data...")
        
        # Generate user profiles
        profiles = self.generate_user_profiles(200)
        
        # Generate interactions
        interactions = self.generate_user_interactions(profiles, 30)
        
        # Generate workflows
        workflows = self.generate_workflow_patterns(profiles)
        
        # Generate analytics
        analytics = self.generate_usage_analytics(interactions)
        
        # Compile results
        result = {
            "generation_timestamp": datetime.now().isoformat(),
            "user_profiles": [asdict(profile) for profile in profiles],
            "user_interactions": [asdict(interaction) for interaction in interactions],
            "workflow_patterns": [asdict(workflow) for workflow in workflows],
            "usage_analytics": analytics,
            "statistics": {
                "total_users": len(profiles),
                "total_interactions": len(interactions),
                "total_workflows": len(workflows),
                "departments": len(set(p.department for p in profiles)),
                "success_rate": analytics["success_rate"]
            }
        }
        
        return result

    def save_to_json(self, data: Dict[str, Any], filename: str = "enterprise_user_behavior.json"):
        """Save generated data to JSON file"""
        import os
        os.makedirs("data/enterprise_behavior", exist_ok=True)
        filepath = f"data/enterprise_behavior/{filename}"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"User behavior data saved to {filepath}")
        return filepath

if __name__ == "__main__":
    generator = EnterpriseUserBehaviorGenerator()
    data = generator.generate_all_behavior_data()
    filepath = generator.save_to_json(data)
    print(f"Generated {data['statistics']['total_users']} users with {data['statistics']['total_interactions']} interactions")
