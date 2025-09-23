"""
Lenovo Business Prompts

Enterprise business scenario prompts for Lenovo use cases:
- B2B client interactions
- Enterprise sales scenarios
- Business process optimization
- Executive decision making
- Strategic planning
"""

from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class BusinessPrompt:
    """Business prompt data structure"""
    prompt_id: str
    category: str
    scenario: str
    prompt: str
    expected_response: str
    difficulty: str
    business_context: str
    success_criteria: List[str]

class LenovoBusinessPrompts:
    """Lenovo business scenario prompts"""
    
    def __init__(self):
        self.prompts = self._initialize_prompts()
    
    def _initialize_prompts(self) -> List[BusinessPrompt]:
        """Initialize business prompts"""
        prompts = []
        
        # B2B Sales Scenarios
        prompts.extend([
            BusinessPrompt(
                prompt_id="B2B_SALES_001",
                category="B2B Sales",
                scenario="Enterprise Server Sale",
                prompt="""
                You are a Lenovo sales representative meeting with a Fortune 500 company's IT Director. 
                The company is looking to upgrade their data center infrastructure and is considering 
                Lenovo ThinkSystem servers. The client has expressed concerns about:
                1. Total cost of ownership
                2. Performance scalability
                3. Support and maintenance
                4. Integration with existing systems
                
                How would you address these concerns and present the value proposition of Lenovo ThinkSystem servers?
                """,
                expected_response="Comprehensive response addressing TCO, scalability, support, and integration",
                difficulty="High",
                business_context="Enterprise data center upgrade",
                success_criteria=[
                    "Addresses all four client concerns",
                    "Provides specific technical details",
                    "Includes ROI calculations",
                    "References customer success stories"
                ]
            ),
            
            BusinessPrompt(
                prompt_id="B2B_SALES_002",
                category="B2B Sales",
                scenario="ThinkPad Enterprise Deployment",
                prompt="""
                A large consulting firm is evaluating ThinkPad laptops for their 500+ employee workforce.
                They need devices that can handle:
                - Heavy data analysis workloads
                - Video conferencing and collaboration
                - Remote work capabilities
                - Security compliance requirements
                
                Create a comprehensive proposal for ThinkPad deployment that addresses their specific needs.
                """,
                expected_response="Detailed proposal covering technical specs, security, support, and deployment",
                difficulty="Medium",
                business_context="Large-scale laptop deployment",
                success_criteria=[
                    "Matches technical requirements",
                    "Addresses security concerns",
                    "Includes deployment timeline",
                    "Provides cost breakdown"
                ]
            )
        ])
        
        # Executive Decision Making
        prompts.extend([
            BusinessPrompt(
                prompt_id="EXEC_DECISION_001",
                category="Executive Decision Making",
                scenario="AI Strategy Planning",
                prompt="""
                As Lenovo's Chief Technology Officer, you need to present an AI strategy to the board 
                that includes:
                1. Current AI capabilities and market position
                2. Investment priorities for the next 3 years
                3. Competitive advantages and differentiation
                4. Risk mitigation strategies
                5. Expected ROI and business impact
                
                Develop a comprehensive AI strategy presentation.
                """,
                expected_response="Strategic AI roadmap with clear priorities and business justification",
                difficulty="High",
                business_context="C-level strategic planning",
                success_criteria=[
                    "Clear strategic vision",
                    "Specific investment recommendations",
                    "Competitive analysis",
                    "Risk assessment",
                    "Financial projections"
                ]
            ),
            
            BusinessPrompt(
                prompt_id="EXEC_DECISION_002",
                category="Executive Decision Making",
                scenario="Market Expansion",
                prompt="""
                Lenovo is considering expanding into the edge computing market with new ThinkEdge devices.
                Analyze the market opportunity and provide recommendations for:
                1. Market size and growth potential
                2. Competitive landscape
                3. Product positioning strategy
                4. Go-to-market approach
                5. Resource requirements and timeline
                """,
                expected_response="Comprehensive market analysis with actionable recommendations",
                difficulty="High",
                business_context="Market expansion strategy",
                success_criteria=[
                    "Market size analysis",
                    "Competitive positioning",
                    "Clear go-to-market strategy",
                    "Resource planning",
                    "Timeline and milestones"
                ]
            )
        ])
        
        # Business Process Optimization
        prompts.extend([
            BusinessPrompt(
                prompt_id="PROCESS_OPT_001",
                category="Process Optimization",
                scenario="Supply Chain Optimization",
                prompt="""
                Lenovo's supply chain team needs to optimize the manufacturing process for ThinkPad laptops.
                Current challenges include:
                - Long lead times for custom configurations
                - Inventory management across multiple regions
                - Quality control bottlenecks
                - Cost optimization opportunities
                
                Propose a comprehensive supply chain optimization strategy.
                """,
                expected_response="Detailed optimization plan with specific improvements and metrics",
                difficulty="Medium",
                business_context="Supply chain management",
                success_criteria=[
                    "Addresses all four challenges",
                    "Provides specific solutions",
                    "Includes implementation timeline",
                    "Defines success metrics"
                ]
            ),
            
            BusinessPrompt(
                prompt_id="PROCESS_OPT_002",
                category="Process Optimization",
                scenario="Customer Support Efficiency",
                prompt="""
                Lenovo's customer support team wants to improve response times and resolution rates.
                Current metrics show:
                - Average response time: 4 hours
                - First-call resolution rate: 65%
                - Customer satisfaction: 78%
                - Support ticket volume: 10,000/month
                
                Design a customer support optimization plan to improve these metrics.
                """,
                expected_response="Comprehensive support improvement strategy",
                difficulty="Medium",
                business_context="Customer service optimization",
                success_criteria=[
                    "Specific improvement targets",
                    "Implementation roadmap",
                    "Technology recommendations",
                    "Training requirements",
                    "Success metrics"
                ]
            )
        ])
        
        # Strategic Planning
        prompts.extend([
            BusinessPrompt(
                prompt_id="STRATEGIC_001",
                category="Strategic Planning",
                scenario="Product Portfolio Strategy",
                prompt="""
                Lenovo's product management team needs to develop a 5-year product portfolio strategy.
                Consider:
                1. Current product lines (ThinkPad, ThinkSystem, Moto Edge)
                2. Emerging technologies (AI, edge computing, 5G)
                3. Market trends and customer needs
                4. Competitive positioning
                5. Resource allocation and investment priorities
                
                Create a comprehensive product portfolio strategy.
                """,
                expected_response="Strategic product roadmap with clear priorities and investments",
                difficulty="High",
                business_context="Product strategy planning",
                success_criteria=[
                    "Clear product roadmap",
                    "Technology integration plan",
                    "Market analysis",
                    "Resource allocation",
                    "Success metrics"
                ]
            ),
            
            BusinessPrompt(
                prompt_id="STRATEGIC_002",
                category="Strategic Planning",
                scenario="Digital Transformation",
                prompt="""
                Lenovo is undergoing a digital transformation initiative. As the transformation lead,
                develop a comprehensive digital transformation strategy that includes:
                1. Current state assessment
                2. Technology modernization roadmap
                3. Process digitization priorities
                4. Change management approach
                5. Success metrics and KPIs
                """,
                expected_response="Comprehensive digital transformation strategy",
                difficulty="High",
                business_context="Enterprise transformation",
                success_criteria=[
                    "Current state analysis",
                    "Technology roadmap",
                    "Process improvements",
                    "Change management plan",
                    "Success metrics"
                ]
            )
        ])
        
        return prompts
    
    def get_prompts_by_category(self, category: str) -> List[BusinessPrompt]:
        """Get prompts by category"""
        return [p for p in self.prompts if p.category == category]
    
    def get_prompts_by_difficulty(self, difficulty: str) -> List[BusinessPrompt]:
        """Get prompts by difficulty level"""
        return [p for p in self.prompts if p.difficulty == difficulty]
    
    def get_random_prompt(self, category: str = None, difficulty: str = None) -> BusinessPrompt:
        """Get random prompt with optional filters"""
        import random
        
        filtered_prompts = self.prompts
        
        if category:
            filtered_prompts = [p for p in filtered_prompts if p.category == category]
        
        if difficulty:
            filtered_prompts = [p for p in filtered_prompts if p.difficulty == difficulty]
        
        return random.choice(filtered_prompts) if filtered_prompts else None
    
    def get_prompt_by_id(self, prompt_id: str) -> BusinessPrompt:
        """Get specific prompt by ID"""
        for prompt in self.prompts:
            if prompt.prompt_id == prompt_id:
                return prompt
        return None
    
    def get_all_categories(self) -> List[str]:
        """Get all available categories"""
        return list(set(p.category for p in self.prompts))
    
    def get_all_difficulties(self) -> List[str]:
        """Get all available difficulty levels"""
        return list(set(p.difficulty for p in self.prompts))
    
    def get_prompts_for_evaluation(self, count: int = 10) -> List[BusinessPrompt]:
        """Get prompts for model evaluation"""
        import random
        return random.sample(self.prompts, min(count, len(self.prompts)))
    
    def export_prompts(self, filename: str = "lenovo_business_prompts.json") -> str:
        """Export prompts to JSON file"""
        import json
        import os
        
        os.makedirs("data/prompts", exist_ok=True)
        filepath = f"data/prompts/{filename}"
        
        prompts_data = []
        for prompt in self.prompts:
            prompts_data.append({
                "prompt_id": prompt.prompt_id,
                "category": prompt.category,
                "scenario": prompt.scenario,
                "prompt": prompt.prompt,
                "expected_response": prompt.expected_response,
                "difficulty": prompt.difficulty,
                "business_context": prompt.business_context,
                "success_criteria": prompt.success_criteria
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(prompts_data, f, indent=2, ensure_ascii=False)
        
        print(f"Business prompts exported to {filepath}")
        return filepath

if __name__ == "__main__":
    # Test Lenovo business prompts
    prompts = LenovoBusinessPrompts()
    
    print(f"Total prompts: {len(prompts.prompts)}")
    print(f"Categories: {prompts.get_all_categories()}")
    print(f"Difficulties: {prompts.get_all_difficulties()}")
    
    # Test random prompt
    random_prompt = prompts.get_random_prompt()
    if random_prompt:
        print(f"\nRandom prompt: {random_prompt.prompt_id} - {random_prompt.scenario}")
    
    # Export prompts
    export_path = prompts.export_prompts()
    print(f"Prompts exported to {export_path}")
