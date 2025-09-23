"""
Customer Journey Generator

Generates realistic customer journey data including:
- B2B client scenarios and interactions
- Customer touchpoints and experiences
- Journey mapping and optimization
- Customer satisfaction metrics
"""

import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import faker

@dataclass
class CustomerJourney:
    """Customer journey data structure"""
    journey_id: str
    customer_type: str
    journey_stage: str
    touchpoints: List[Dict[str, Any]]
    duration_days: int
    satisfaction_score: float
    conversion_rate: float
    pain_points: List[str]
    success_factors: List[str]

@dataclass
class Touchpoint:
    """Customer touchpoint data structure"""
    touchpoint_id: str
    touchpoint_type: str
    channel: str
    interaction: str
    duration_minutes: int
    outcome: str
    satisfaction: float
    next_action: str

class CustomerJourneyGenerator:
    """Generates comprehensive customer journey data"""
    
    def __init__(self):
        self.fake = faker.Faker()
        
        self.customer_types = [
            "Enterprise B2B", "SMB B2B", "Consumer", "Government", "Education"
        ]
        
        self.journey_stages = [
            "Awareness", "Consideration", "Evaluation", "Purchase", "Onboarding",
            "Usage", "Support", "Renewal", "Advocacy"
        ]
        
        self.touchpoint_types = [
            "Website Visit", "Product Demo", "Sales Call", "Email Campaign",
            "Social Media", "Support Call", "Training Session", "Account Review"
        ]
        
        self.channels = [
            "Website", "Phone", "Email", "In-Person", "Video Call",
            "Social Media", "Mobile App", "Chat", "Support Portal"
        ]

    def generate_customer_journeys(self, count: int = 100) -> List[CustomerJourney]:
        """Generate customer journey data"""
        journeys = []
        
        for i in range(count):
            customer_type = random.choice(self.customer_types)
            journey_stage = random.choice(self.journey_stages)
            
            # Generate touchpoints for this journey
            touchpoints = self._generate_touchpoints(customer_type, journey_stage)
            
            # Calculate journey metrics
            duration_days = random.randint(7, 180)
            satisfaction_score = random.uniform(3.0, 5.0)
            conversion_rate = random.uniform(0.1, 0.8)
            
            # Generate pain points and success factors
            pain_points = self._generate_pain_points(customer_type)
            success_factors = self._generate_success_factors(customer_type)
            
            journeys.append(CustomerJourney(
                journey_id=f"JOURNEY_{i+1:04d}",
                customer_type=customer_type,
                journey_stage=journey_stage,
                touchpoints=[asdict(tp) for tp in touchpoints],
                duration_days=duration_days,
                satisfaction_score=round(satisfaction_score, 2),
                conversion_rate=round(conversion_rate, 2),
                pain_points=pain_points,
                success_factors=success_factors
            ))
        
        return journeys

    def _generate_touchpoints(self, customer_type: str, journey_stage: str) -> List[Touchpoint]:
        """Generate touchpoints for a customer journey"""
        touchpoints = []
        touchpoint_count = random.randint(3, 8)
        
        for i in range(touchpoint_count):
            touchpoint_type = random.choice(self.touchpoint_types)
            channel = random.choice(self.channels)
            
            # Generate interaction based on touchpoint type
            interaction = self._generate_interaction(touchpoint_type, customer_type)
            
            touchpoints.append(Touchpoint(
                touchpoint_id=f"TP_{len(touchpoints)+1:03d}",
                touchpoint_type=touchpoint_type,
                channel=channel,
                interaction=interaction,
                duration_minutes=random.randint(5, 120),
                outcome=random.choice(["Positive", "Neutral", "Negative"]),
                satisfaction=random.uniform(2.0, 5.0),
                next_action=random.choice([
                    "Schedule Follow-up", "Request Information", "Make Purchase",
                    "Contact Support", "Provide Feedback", "Continue Journey"
                ])
            ))
        
        return touchpoints

    def _generate_interaction(self, touchpoint_type: str, customer_type: str) -> str:
        """Generate interaction description"""
        interactions = {
            "Website Visit": [
                "Browsed product specifications",
                "Downloaded whitepaper",
                "Watched product video",
                "Requested quote",
                "Compared products"
            ],
            "Product Demo": [
                "Attended live demonstration",
                "Scheduled personalized demo",
                "Participated in group demo",
                "Requested technical demo",
                "Viewed recorded demo"
            ],
            "Sales Call": [
                "Initial discovery call",
                "Technical requirements discussion",
                "Pricing and proposal review",
                "Decision maker meeting",
                "Contract negotiation"
            ],
            "Email Campaign": [
                "Received product newsletter",
                "Opened promotional email",
                "Clicked on product link",
                "Downloaded resource",
                "Responded to survey"
            ],
            "Support Call": [
                "Technical issue resolution",
                "Product training session",
                "Account management review",
                "Feature request discussion",
                "Billing inquiry"
            ]
        }
        
        if touchpoint_type in interactions:
            return random.choice(interactions[touchpoint_type])
        else:
            return f"General {touchpoint_type.lower()} interaction"

    def _generate_pain_points(self, customer_type: str) -> List[str]:
        """Generate pain points based on customer type"""
        pain_points_by_type = {
            "Enterprise B2B": [
                "Complex procurement process",
                "Long decision-making cycles",
                "Integration challenges",
                "Security compliance requirements",
                "Budget constraints"
            ],
            "SMB B2B": [
                "Limited IT resources",
                "Budget limitations",
                "Technical complexity",
                "Support availability",
                "Scalability concerns"
            ],
            "Consumer": [
                "Price sensitivity",
                "Product complexity",
                "Support accessibility",
                "Warranty concerns",
                "Upgrade decisions"
            ],
            "Government": [
                "Compliance requirements",
                "Security protocols",
                "Budget approval process",
                "Vendor certification",
                "Long procurement cycles"
            ],
            "Education": [
                "Budget constraints",
                "Student accessibility",
                "Technical support",
                "Training requirements",
                "Scalability needs"
            ]
        }
        
        return random.sample(
            pain_points_by_type.get(customer_type, ["General concerns"]),
            random.randint(2, 4)
        )

    def _generate_success_factors(self, customer_type: str) -> List[str]:
        """Generate success factors based on customer type"""
        success_factors_by_type = {
            "Enterprise B2B": [
                "Strong technical support",
                "Proven ROI",
                "Security compliance",
                "Integration capabilities",
                "Executive sponsorship"
            ],
            "SMB B2B": [
                "Easy implementation",
                "Cost effectiveness",
                "Quick support response",
                "Training availability",
                "Scalable solutions"
            ],
            "Consumer": [
                "User-friendly interface",
                "Reliable performance",
                "Good value for money",
                "Easy setup process",
                "Strong warranty"
            ],
            "Government": [
                "Security certifications",
                "Compliance documentation",
                "Proven track record",
                "Local support",
                "Cost transparency"
            ],
            "Education": [
                "Educational pricing",
                "Student-friendly features",
                "Training resources",
                "Technical support",
                "Scalable deployment"
            ]
        }
        
        return random.sample(
            success_factors_by_type.get(customer_type, ["Quality service"]),
            random.randint(2, 4)
        )

    def generate_journey_analytics(self, journeys: List[CustomerJourney]) -> Dict[str, Any]:
        """Generate journey analytics"""
        analytics = {
            "total_journeys": len(journeys),
            "customer_type_distribution": {},
            "journey_stage_distribution": {},
            "average_satisfaction": 0,
            "average_conversion_rate": 0,
            "common_pain_points": {},
            "common_success_factors": {},
            "touchpoint_effectiveness": {}
        }
        
        # Calculate distributions
        for journey in journeys:
            # Customer type distribution
            if journey.customer_type not in analytics["customer_type_distribution"]:
                analytics["customer_type_distribution"][journey.customer_type] = 0
            analytics["customer_type_distribution"][journey.customer_type] += 1
            
            # Journey stage distribution
            if journey.journey_stage not in analytics["journey_stage_distribution"]:
                analytics["journey_stage_distribution"][journey.journey_stage] = 0
            analytics["journey_stage_distribution"][journey.journey_stage] += 1
            
            # Pain points analysis
            for pain_point in journey.pain_points:
                if pain_point not in analytics["common_pain_points"]:
                    analytics["common_pain_points"][pain_point] = 0
                analytics["common_pain_points"][pain_point] += 1
            
            # Success factors analysis
            for success_factor in journey.success_factors:
                if success_factor not in analytics["common_success_factors"]:
                    analytics["common_success_factors"][success_factor] = 0
                analytics["common_success_factors"][success_factor] += 1
        
        # Calculate averages
        analytics["average_satisfaction"] = round(
            sum(j.satisfaction_score for j in journeys) / len(journeys), 2
        )
        analytics["average_conversion_rate"] = round(
            sum(j.conversion_rate for j in journeys) / len(journeys), 2
        )
        
        return analytics

    def generate_all_journey_data(self) -> Dict[str, Any]:
        """Generate comprehensive customer journey data"""
        print("Generating customer journey data...")
        
        # Generate journeys
        journeys = self.generate_customer_journeys(100)
        
        # Generate analytics
        analytics = self.generate_journey_analytics(journeys)
        
        # Compile results
        result = {
            "generation_timestamp": datetime.now().isoformat(),
            "customer_journeys": [asdict(journey) for journey in journeys],
            "journey_analytics": analytics,
            "statistics": {
                "total_journeys": len(journeys),
                "customer_types": len(set(j.customer_type for j in journeys)),
                "journey_stages": len(set(j.journey_stage for j in journeys)),
                "average_satisfaction": analytics["average_satisfaction"],
                "average_conversion_rate": analytics["average_conversion_rate"]
            }
        }
        
        return result

    def save_to_json(self, data: Dict[str, Any], filename: str = "customer_journey_data.json"):
        """Save generated data to JSON file"""
        import os
        os.makedirs("data/customer_journeys", exist_ok=True)
        filepath = f"data/customer_journeys/{filename}"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Customer journey data saved to {filepath}")
        return filepath

if __name__ == "__main__":
    generator = CustomerJourneyGenerator()
    data = generator.generate_all_journey_data()
    filepath = generator.save_to_json(data)
    print(f"Generated {data['statistics']['total_journeys']} customer journeys")
