"""
Model Factory for Lenovo AAITC Assignment 1

This module provides automated model selection framework for Lenovo's
internal operations and B2B processes, implementing the Model Factory concept.
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class UseCaseCategory(Enum):
    """Use case categories for Lenovo operations."""
    INTERNAL_OPERATIONS = "internal_operations"
    B2B_PROCESSES = "b2b_processes"
    CUSTOMER_SERVICE = "customer_service"
    TECHNICAL_SUPPORT = "technical_support"
    DOCUMENTATION = "documentation"
    CODE_GENERATION = "code_generation"


class DeploymentScenario(Enum):
    """Deployment scenarios for models."""
    CLOUD = "cloud"
    EDGE = "edge"
    MOBILE = "mobile"
    HYBRID = "hybrid"


class PerformanceRequirement(Enum):
    """Performance requirement levels."""
    HIGH_PERFORMANCE = "high_performance"
    BALANCED = "balanced"
    COST_OPTIMIZED = "cost_optimized"


@dataclass
class UseCaseProfile:
    """Profile for a specific use case."""
    description: str
    category: UseCaseCategory
    deployment_scenario: DeploymentScenario
    performance_requirement: PerformanceRequirement
    latency_requirement: float  # seconds
    cost_budget: float  # per request
    expected_volume: int  # requests per day
    criticality: str  # high, medium, low


@dataclass
class ModelRecommendation:
    """Model recommendation with rationale."""
    model_name: str
    confidence_score: float
    estimated_performance: float
    estimated_cost: float
    rationale: str
    pros: List[str]
    cons: List[str]
    alternatives: List[str]


@dataclass
class ModelCapability:
    """Model capability profile."""
    model_name: str
    performance_scores: Dict[str, float]
    cost_per_request: float
    latency_performance: float
    deployment_compatibility: Dict[DeploymentScenario, bool]
    use_case_suitability: Dict[UseCaseCategory, float]


class ModelFactory:
    """
    Automated model selection framework for Lenovo's Model Factory concept.
    
    Implements intelligent routing logic that matches use case requirements
    to model capabilities with performance vs. cost trade-offs.
    """
    
    def __init__(self):
        """Initialize the Model Factory."""
        self.model_capabilities = self._initialize_model_capabilities()
        self.selection_history = []
        logger.info("ModelFactory initialized")
    
    def _initialize_model_capabilities(self) -> Dict[str, ModelCapability]:
        """Initialize model capability profiles."""
        capabilities = {}
        
        # GPT-5 capabilities
        capabilities["GPT-5"] = ModelCapability(
            model_name="GPT-5",
            performance_scores={
                "reasoning": 0.95,
                "code_generation": 0.90,
                "documentation": 0.92,
                "customer_support": 0.88,
                "translation": 0.85
            },
            cost_per_request=0.05,
            latency_performance=0.8,  # 0-1 scale
            deployment_compatibility={
                DeploymentScenario.CLOUD: True,
                DeploymentScenario.EDGE: False,
                DeploymentScenario.MOBILE: False,
                DeploymentScenario.HYBRID: True
            },
            use_case_suitability={
                UseCaseCategory.INTERNAL_OPERATIONS: 0.92,
                UseCaseCategory.B2B_PROCESSES: 0.88,
                UseCaseCategory.CUSTOMER_SERVICE: 0.90,
                UseCaseCategory.TECHNICAL_SUPPORT: 0.95,
                UseCaseCategory.DOCUMENTATION: 0.93,
                UseCaseCategory.CODE_GENERATION: 0.90
            }
        )
        
        # GPT-5-Codex capabilities
        capabilities["GPT-5-Codex"] = ModelCapability(
            model_name="GPT-5-Codex",
            performance_scores={
                "reasoning": 0.88,
                "code_generation": 0.95,
                "documentation": 0.85,
                "customer_support": 0.75,
                "translation": 0.70
            },
            cost_per_request=0.04,
            latency_performance=0.75,
            deployment_compatibility={
                DeploymentScenario.CLOUD: True,
                DeploymentScenario.EDGE: False,
                DeploymentScenario.MOBILE: False,
                DeploymentScenario.HYBRID: True
            },
            use_case_suitability={
                UseCaseCategory.INTERNAL_OPERATIONS: 0.85,
                UseCaseCategory.B2B_PROCESSES: 0.80,
                UseCaseCategory.CUSTOMER_SERVICE: 0.70,
                UseCaseCategory.TECHNICAL_SUPPORT: 0.95,
                UseCaseCategory.DOCUMENTATION: 0.90,
                UseCaseCategory.CODE_GENERATION: 0.98
            }
        )
        
        # Claude 3.5 Sonnet capabilities
        capabilities["Claude 3.5 Sonnet"] = ModelCapability(
            model_name="Claude 3.5 Sonnet",
            performance_scores={
                "reasoning": 0.93,
                "code_generation": 0.88,
                "documentation": 0.95,
                "customer_support": 0.92,
                "translation": 0.88
            },
            cost_per_request=0.06,
            latency_performance=0.85,
            deployment_compatibility={
                DeploymentScenario.CLOUD: True,
                DeploymentScenario.EDGE: True,
                DeploymentScenario.MOBILE: False,
                DeploymentScenario.HYBRID: True
            },
            use_case_suitability={
                UseCaseCategory.INTERNAL_OPERATIONS: 0.90,
                UseCaseCategory.B2B_PROCESSES: 0.92,
                UseCaseCategory.CUSTOMER_SERVICE: 0.95,
                UseCaseCategory.TECHNICAL_SUPPORT: 0.90,
                UseCaseCategory.DOCUMENTATION: 0.96,
                UseCaseCategory.CODE_GENERATION: 0.85
            }
        )
        
        # Llama 3.3 capabilities
        capabilities["Llama 3.3"] = ModelCapability(
            model_name="Llama 3.3",
            performance_scores={
                "reasoning": 0.85,
                "code_generation": 0.80,
                "documentation": 0.88,
                "customer_support": 0.85,
                "translation": 0.90
            },
            cost_per_request=0.02,
            latency_performance=0.70,
            deployment_compatibility={
                DeploymentScenario.CLOUD: True,
                DeploymentScenario.EDGE: True,
                DeploymentScenario.MOBILE: True,
                DeploymentScenario.HYBRID: True
            },
            use_case_suitability={
                UseCaseCategory.INTERNAL_OPERATIONS: 0.85,
                UseCaseCategory.B2B_PROCESSES: 0.82,
                UseCaseCategory.CUSTOMER_SERVICE: 0.88,
                UseCaseCategory.TECHNICAL_SUPPORT: 0.80,
                UseCaseCategory.DOCUMENTATION: 0.85,
                UseCaseCategory.CODE_GENERATION: 0.75
            }
        )
        
        return capabilities
    
    def select_optimal_model(self, use_case_profile: UseCaseProfile) -> ModelRecommendation:
        """
        Select optimal model for a given use case profile.
        
        Args:
            use_case_profile: Profile of the use case requirements
            
        Returns:
            ModelRecommendation with optimal model selection
        """
        logger.info(f"Selecting optimal model for use case: {use_case_profile.description}")
        
        best_model = None
        best_score = -1
        all_scores = {}
        
        # Calculate suitability scores for each model
        for model_name, capability in self.model_capabilities.items():
            score = self._calculate_model_score(use_case_profile, capability)
            all_scores[model_name] = score
            
            if score > best_score:
                best_score = score
                best_model = model_name
        
        if best_model is None:
            raise ValueError("No suitable model found for the use case")
        
        # Create recommendation
        capability = self.model_capabilities[best_model]
        recommendation = ModelRecommendation(
            model_name=best_model,
            confidence_score=best_score,
            estimated_performance=capability.use_case_suitability[use_case_profile.category],
            estimated_cost=capability.cost_per_request,
            rationale=self._generate_rationale(use_case_profile, capability, all_scores),
            pros=self._generate_pros(capability),
            cons=self._generate_cons(capability),
            alternatives=self._get_alternatives(all_scores, best_model)
        )
        
        # Record selection
        self.selection_history.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "use_case": use_case_profile.description,
            "selected_model": best_model,
            "confidence_score": best_score
        })
        
        logger.info(f"Selected {best_model} with confidence score {best_score:.2f}")
        
        return recommendation
    
    def _calculate_model_score(self, use_case: UseCaseProfile, capability: ModelCapability) -> float:
        """Calculate suitability score for a model given use case requirements."""
        score = 0.0
        
        # Base suitability score for the use case category
        category_score = capability.use_case_suitability.get(use_case.category, 0.0)
        score += category_score * 0.4
        
        # Deployment compatibility
        if capability.deployment_compatibility.get(use_case.deployment_scenario, False):
            score += 0.3
        else:
            score -= 0.2  # Penalty for incompatible deployment
        
        # Cost consideration
        if capability.cost_per_request <= use_case.cost_budget:
            cost_score = 1.0 - (capability.cost_per_request / use_case.cost_budget)
            score += cost_score * 0.15
        else:
            score -= 0.3  # Penalty for exceeding budget
        
        # Latency consideration
        if use_case.performance_requirement == PerformanceRequirement.HIGH_PERFORMANCE:
            score += capability.latency_performance * 0.15
        elif use_case.performance_requirement == PerformanceRequirement.BALANCED:
            score += capability.latency_performance * 0.1
        # Cost optimized doesn't prioritize latency
        
        return max(0.0, min(1.0, score))  # Clamp between 0 and 1
    
    def _generate_rationale(self, use_case: UseCaseProfile, capability: ModelCapability, all_scores: Dict[str, float]) -> str:
        """Generate selection rationale."""
        rationale_parts = []
        
        rationale_parts.append(f"Selected based on {use_case.category.value} suitability score of {capability.use_case_suitability[use_case.category]:.2f}")
        
        if capability.deployment_compatibility.get(use_case.deployment_scenario, False):
            rationale_parts.append(f"Compatible with {use_case.deployment_scenario.value} deployment")
        
        if capability.cost_per_request <= use_case.cost_budget:
            rationale_parts.append(f"Cost-effective at ${capability.cost_per_request:.3f} per request")
        
        if use_case.performance_requirement == PerformanceRequirement.HIGH_PERFORMANCE:
            rationale_parts.append(f"High latency performance score of {capability.latency_performance:.2f}")
        
        return ". ".join(rationale_parts) + "."
    
    def _generate_pros(self, capability: ModelCapability) -> List[str]:
        """Generate pros list for the model."""
        pros = []
        
        # Find strongest capabilities
        best_tasks = sorted(capability.performance_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        for task, score in best_tasks:
            pros.append(f"Excellent {task} performance ({score:.2f})")
        
        # Deployment advantages
        compatible_deployments = [d.value for d, compatible in capability.deployment_compatibility.items() if compatible]
        if len(compatible_deployments) > 2:
            pros.append(f"Supports multiple deployment scenarios: {', '.join(compatible_deployments)}")
        
        # Cost advantage
        if capability.cost_per_request < 0.03:
            pros.append("Cost-effective option")
        
        return pros
    
    def _generate_cons(self, capability: ModelCapability) -> List[str]:
        """Generate cons list for the model."""
        cons = []
        
        # Find weakest capabilities
        worst_tasks = sorted(capability.performance_scores.items(), key=lambda x: x[1])[:2]
        for task, score in worst_tasks:
            if score < 0.8:
                cons.append(f"Limited {task} performance ({score:.2f})")
        
        # Deployment limitations
        incompatible_deployments = [d.value for d, compatible in capability.deployment_compatibility.items() if not compatible]
        if incompatible_deployments:
            cons.append(f"Not compatible with: {', '.join(incompatible_deployments)}")
        
        # Cost concerns
        if capability.cost_per_request > 0.05:
            cons.append("Higher cost per request")
        
        return cons
    
    def _get_alternatives(self, all_scores: Dict[str, float], best_model: str) -> List[str]:
        """Get alternative model recommendations."""
        sorted_models = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        alternatives = [model for model, score in sorted_models[1:3] if score > 0.6]
        return alternatives
    
    def get_selection_history(self) -> List[Dict[str, Any]]:
        """Get model selection history."""
        return self.selection_history
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return list(self.model_capabilities.keys())
    
    def analyze_use_case_taxonomy(self, use_case_description: str) -> UseCaseProfile:
        """
        Analyze use case description and create profile.
        
        Args:
            use_case_description: Description of the use case
            
        Returns:
            UseCaseProfile object
        """
        description_lower = use_case_description.lower()
        
        # Determine category based on keywords
        if any(keyword in description_lower for keyword in ["internal", "hr", "it support", "documentation"]):
            category = UseCaseCategory.INTERNAL_OPERATIONS
        elif any(keyword in description_lower for keyword in ["customer", "service", "support"]):
            category = UseCaseCategory.CUSTOMER_SERVICE
        elif any(keyword in description_lower for keyword in ["technical", "code", "development"]):
            category = UseCaseCategory.TECHNICAL_SUPPORT
        elif any(keyword in description_lower for keyword in ["document", "manual", "guide"]):
            category = UseCaseCategory.DOCUMENTATION
        else:
            category = UseCaseCategory.INTERNAL_OPERATIONS  # Default
        
        # Create profile with reasonable defaults
        profile = UseCaseProfile(
            description=use_case_description,
            category=category,
            deployment_scenario=DeploymentScenario.CLOUD,  # Default
            performance_requirement=PerformanceRequirement.BALANCED,  # Default
            latency_requirement=2.0,  # Default 2 seconds
            cost_budget=0.1,  # Default $0.1 per request
            expected_volume=1000,  # Default 1000 requests per day
            criticality="medium"  # Default
        )
        
        return profile
