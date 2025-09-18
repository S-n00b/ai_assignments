"""
Prompt Registry Manager for Enhanced Experimental Scale

This module integrates with open-source prompt registries to provide enhanced
experimental scale for model evaluation, including DiffusionDB, PromptBase,
and other popular prompt databases.

Key Features:
- Integration with multiple prompt registries
- Dynamic dataset generation
- Prompt categorization and filtering
- Stratified sampling for balanced evaluation
- Enhanced test scale capabilities
"""

import logging
import requests
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import hashlib
import asyncio
import os
from pathlib import Path
import shutil


logger = logging.getLogger(__name__)


class PromptCategory(Enum):
    """Enumeration of prompt categories for organization and filtering."""
    CREATIVE_WRITING = "creative_writing"
    TECHNICAL_DOCUMENTATION = "technical_documentation"
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CLASSIFICATION = "classification"
    MULTIMODAL = "multimodal"
    CONVERSATIONAL = "conversational"
    EDUCATIONAL = "educational"
    BUSINESS = "business"
    SCIENTIFIC = "scientific"
    ARTISTIC = "artistic"


@dataclass
class PromptEntry:
    """
    Data class representing a single prompt entry from a registry.
    
    Attributes:
        id: Unique identifier for the prompt
        text: The actual prompt text
        category: Category of the prompt
        source: Source registry (e.g., 'diffusiondb', 'promptbase')
        metadata: Additional metadata about the prompt
        quality_score: Quality score for the prompt (0-1)
        difficulty_level: Difficulty level (easy, medium, hard)
    """
    id: str
    text: str
    category: PromptCategory
    source: str
    metadata: Dict[str, Any]
    quality_score: float = 0.5
    difficulty_level: str = "medium"


class PromptRegistryManager:
    """
    Manager for integrating with multiple prompt registries and generating
    enhanced evaluation datasets.
    
    This class provides comprehensive integration with popular open-source
    prompt registries to significantly increase the experimental test scale
    for model evaluation.
    """
    
    def __init__(self, enable_caching: bool = True, cache_dir: str = "cache/ai_tool_prompts"):
        """
        Initialize the prompt registry manager.
        
        Args:
            enable_caching: Whether to enable caching of downloaded prompts
            cache_dir: Directory for local caching of AI tool prompts
        """
        self.enable_caching = enable_caching
        self.cache = {} if enable_caching else None
        
        # Local caching configuration
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # GitHub repository configuration
        self.github_repo = "x1xhlol/system-prompts-and-models-of-ai-tools"
        self.github_base_url = f"https://raw.githubusercontent.com/{self.github_repo}/main"
        self.github_web_url = f"https://github.com/{self.github_repo}"
        
        # Available AI tools mapping (from the API response)
        self.ai_tools = {
            "Cursor": "Cursor Prompts",
            "Claude Code": "Claude Code", 
            "Devin AI": "Devin AI",
            "v0": "v0 Prompts and Tools",
            "Windsurf": "Windsurf",
            "Augment Code": "Augment Code",
            "Cluely": "Cluely",
            "CodeBuddy": "CodeBuddy Prompts",
            "Warp": "Warp.dev",
            "Xcode": "Xcode",
            "Z.ai": "Z.ai Code",
            "dia": "dia"
        }
        
        self.registry_configs = self._initialize_registry_configs()
        self.prompt_categories = self._initialize_prompt_categories()
        
        logger.info("Initialized PromptRegistryManager with enhanced experimental scale capabilities")
    
    def get_available_ai_tools(self) -> List[str]:
        """Get list of available AI tools."""
        return list(self.ai_tools.keys())
    
    def get_tool_cache_path(self, tool_name: str) -> Path:
        """Get the cache path for a specific tool."""
        return self.cache_dir / f"{tool_name.lower().replace(' ', '_')}.json"
    
    def is_tool_cached(self, tool_name: str) -> bool:
        """Check if a tool's prompts are cached locally."""
        cache_path = self.get_tool_cache_path(tool_name)
        return cache_path.exists()
    
    def load_cached_tool_prompts(self, tool_name: str) -> List[PromptEntry]:
        """Load prompts from local cache for a specific tool."""
        cache_path = self.get_tool_cache_path(tool_name)
        if not cache_path.exists():
            return []
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            prompts = []
            for prompt_data in cached_data.get('prompts', []):
                prompt = PromptEntry(
                    id=prompt_data['id'],
                    text=prompt_data['text'],
                    category=PromptCategory(prompt_data['category']),
                    source=prompt_data['source'],
                    metadata=prompt_data['metadata'],
                    quality_score=prompt_data.get('quality_score', 0.5),
                    difficulty_level=prompt_data.get('difficulty_level', 'medium')
                )
                prompts.append(prompt)
            
            logger.info(f"Loaded {len(prompts)} cached prompts for {tool_name}")
            return prompts
            
        except Exception as e:
            logger.error(f"Error loading cached prompts for {tool_name}: {str(e)}")
            return []
    
    def save_tool_prompts_to_cache(self, tool_name: str, prompts: List[PromptEntry]):
        """Save prompts to local cache for a specific tool."""
        cache_path = self.get_tool_cache_path(tool_name)
        
        try:
            cached_data = {
                'tool_name': tool_name,
                'cached_at': time.time(),
                'prompt_count': len(prompts),
                'prompts': []
            }
            
            for prompt in prompts:
                prompt_data = {
                    'id': prompt.id,
                    'text': prompt.text,
                    'category': prompt.category.value,
                    'source': prompt.source,
                    'metadata': prompt.metadata,
                    'quality_score': prompt.quality_score,
                    'difficulty_level': prompt.difficulty_level
                }
                cached_data['prompts'].append(prompt_data)
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cached_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Cached {len(prompts)} prompts for {tool_name}")
            
        except Exception as e:
            logger.error(f"Error saving prompts to cache for {tool_name}: {str(e)}")
    
    def get_enhanced_evaluation_dataset(
        self,
        target_size: int = 10000,
        categories: Optional[List[PromptCategory]] = None,
        difficulty_levels: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        quality_threshold: float = 0.3
    ) -> pd.DataFrame:
        """
        Generate enhanced evaluation dataset from multiple prompt registries.
        
        This method creates a comprehensive dataset by combining prompts from
        multiple sources, applying filtering criteria, and ensuring balanced
        representation across categories.
        
        Args:
            target_size: Target number of prompts in the dataset
            categories: List of prompt categories to include
            difficulty_levels: List of difficulty levels to include
            sources: List of source registries to use
            quality_threshold: Minimum quality score for prompts
            
        Returns:
            DataFrame containing enhanced evaluation dataset
        """
        logger.info(f"Generating enhanced evaluation dataset with target size: {target_size}")
        
        # Set default parameters
        if categories is None:
            categories = list(PromptCategory)
        if difficulty_levels is None:
            difficulty_levels = ["easy", "medium", "hard"]
        if sources is None:
            sources = list(self.registry_configs.keys())
        
        # Collect prompts from all sources
        all_prompts = []
        for source in sources:
            logger.info(f"Collecting prompts from {source}...")
            source_prompts = self._collect_prompts_from_source(
                source, categories, difficulty_levels, quality_threshold
            )
            all_prompts.extend(source_prompts)
            logger.info(f"Collected {len(source_prompts)} prompts from {source}")
        
        # Apply stratified sampling for balanced representation
        balanced_prompts = self._apply_stratified_sampling(
            all_prompts, target_size, categories
        )
        
        # Convert to DataFrame
        dataset = self._prompts_to_dataframe(balanced_prompts)
        
        logger.info(f"Generated enhanced dataset with {len(dataset)} prompts")
        logger.info(f"Category distribution: {dataset['category'].value_counts().to_dict()}")
        
        return dataset
    
    def get_category_specific_dataset(
        self,
        category: PromptCategory,
        size: int = 1000,
        difficulty_levels: Optional[List[str]] = None,
        quality_threshold: float = 0.3
    ) -> pd.DataFrame:
        """
        Generate dataset for a specific prompt category.
        
        Args:
            category: Specific prompt category to focus on
            size: Target number of prompts
            difficulty_levels: List of difficulty levels to include
            quality_threshold: Minimum quality score for prompts
            
        Returns:
            DataFrame containing category-specific dataset
        """
        logger.info(f"Generating {category.value} dataset with size: {size}")
        
        if difficulty_levels is None:
            difficulty_levels = ["easy", "medium", "hard"]
        
        # Collect prompts for specific category
        prompts = []
        for source in self.registry_configs.keys():
            source_prompts = self._collect_prompts_from_source(
                source, [category], difficulty_levels, quality_threshold
            )
            prompts.extend(source_prompts)
        
        # Apply sampling
        if len(prompts) > size:
            prompts = np.random.choice(prompts, size, replace=False).tolist()
        
        dataset = self._prompts_to_dataframe(prompts)
        
        logger.info(f"Generated {category.value} dataset with {len(dataset)} prompts")
        
        return dataset
    
    def _collect_prompts_from_source(
        self,
        source: str,
        categories: List[PromptCategory],
        difficulty_levels: List[str],
        quality_threshold: float
    ) -> List[PromptEntry]:
        """
        Collect prompts from a specific source registry.
        
        Args:
            source: Source registry name
            categories: List of categories to collect
            difficulty_levels: List of difficulty levels to include
            quality_threshold: Minimum quality score
            
        Returns:
            List of PromptEntry objects from the source
        """
        if source not in self.registry_configs:
            logger.warning(f"Unknown source registry: {source}")
            return []
        
        # Check cache first
        cache_key = self._generate_cache_key(source, categories, difficulty_levels, quality_threshold)
        if self.cache and cache_key in self.cache:
            logger.info(f"Using cached prompts for {source}")
            return self.cache[cache_key]
        
        # Collect prompts based on source type
        if source == "diffusiondb":
            prompts = self._collect_diffusiondb_prompts(categories, difficulty_levels, quality_threshold)
        elif source == "promptbase":
            prompts = self._collect_promptbase_prompts(categories, difficulty_levels, quality_threshold)
        elif source == "synthetic":
            prompts = self._generate_synthetic_prompts(categories, difficulty_levels, quality_threshold)
        else:
            logger.warning(f"No collection method implemented for {source}")
            prompts = []
        
        # Cache results
        if self.cache:
            self.cache[cache_key] = prompts
        
        return prompts
    
    def _collect_diffusiondb_prompts(
        self,
        categories: List[PromptCategory],
        difficulty_levels: List[str],
        quality_threshold: float
    ) -> List[PromptEntry]:
        """
        Collect prompts from DiffusionDB registry.
        
        DiffusionDB is a large-scale prompt gallery dataset with 14 million images
        and 1.8 million unique prompts from Stable Diffusion.
        
        Args:
            categories: List of categories to collect
            difficulty_levels: List of difficulty levels to include
            quality_threshold: Minimum quality score
            
        Returns:
            List of PromptEntry objects from DiffusionDB
        """
        logger.info("Collecting prompts from DiffusionDB...")
        
        # Mock implementation - in production, this would connect to actual DiffusionDB
        # For demonstration, we'll generate synthetic prompts that simulate DiffusionDB content
        
        prompts = []
        diffusiondb_categories = {
            PromptCategory.ARTISTIC: [
                "A beautiful landscape painting in the style of Van Gogh",
                "Futuristic cityscape with flying cars and neon lights",
                "Portrait of a wise old wizard with a long beard",
                "Abstract art with vibrant colors and geometric shapes",
                "Medieval castle on a mountain peak at sunset"
            ],
            PromptCategory.CREATIVE_WRITING: [
                "Write a short story about a time traveler",
                "Create a poem about the changing seasons",
                "Describe a magical forest with talking animals",
                "Write a dialogue between two robots",
                "Create a story about a lost civilization"
            ],
            PromptCategory.TECHNICAL_DOCUMENTATION: [
                "Explain how a neural network processes information",
                "Document the steps to set up a development environment",
                "Describe the architecture of a microservices system",
                "Explain the principles of object-oriented programming",
                "Document the process of deploying a web application"
            ]
        }
        
        # Generate prompts for requested categories
        for category in categories:
            if category in diffusiondb_categories:
                category_prompts = diffusiondb_categories[category]
                for i, prompt_text in enumerate(category_prompts):
                    # Simulate quality scoring
                    quality_score = np.random.uniform(0.4, 0.9)
                    if quality_score >= quality_threshold:
                        difficulty = np.random.choice(difficulty_levels)
                        
                        prompt_entry = PromptEntry(
                            id=f"diffusiondb_{category.value}_{i}",
                            text=prompt_text,
                            category=category,
                            source="diffusiondb",
                            metadata={
                                "original_source": "Stable Diffusion",
                                "image_count": np.random.randint(1, 100),
                                "popularity_score": np.random.uniform(0.1, 1.0)
                            },
                            quality_score=quality_score,
                            difficulty_level=difficulty
                        )
                        prompts.append(prompt_entry)
        
        logger.info(f"Collected {len(prompts)} prompts from DiffusionDB")
        return prompts
    
    def _collect_promptbase_prompts(
        self,
        categories: List[PromptCategory],
        difficulty_levels: List[str],
        quality_threshold: float
    ) -> List[PromptEntry]:
        """
        Collect prompts from PromptBase registry.
        
        PromptBase is a community-driven prompt registry with diverse
        prompt categories and quality ratings.
        
        Args:
            categories: List of categories to collect
            difficulty_levels: List of difficulty levels to include
            quality_threshold: Minimum quality score
            
        Returns:
            List of PromptEntry objects from PromptBase
        """
        logger.info("Collecting prompts from PromptBase...")
        
        # Mock implementation - in production, this would connect to actual PromptBase
        prompts = []
        
        promptbase_categories = {
            PromptCategory.CODE_GENERATION: [
                "Write a Python function to sort a list of dictionaries by a specific key",
                "Create a REST API endpoint for user authentication",
                "Implement a binary search algorithm in JavaScript",
                "Write a SQL query to find the top 10 customers by revenue",
                "Create a React component for a todo list"
            ],
            PromptCategory.REASONING: [
                "Solve this logic puzzle: If all roses are flowers and some flowers are red, what can we conclude?",
                "Analyze the pros and cons of renewable energy adoption",
                "Explain the reasoning behind the scientific method",
                "Compare and contrast different programming paradigms",
                "Evaluate the ethical implications of artificial intelligence"
            ],
            PromptCategory.QUESTION_ANSWERING: [
                "What are the main causes of climate change?",
                "How does photosynthesis work in plants?",
                "What is the difference between machine learning and deep learning?",
                "Explain the concept of supply and demand in economics",
                "What are the key principles of good software design?"
            ],
            PromptCategory.BUSINESS: [
                "Create a business plan for a sustainable fashion startup",
                "Analyze the market opportunity for electric vehicles",
                "Develop a marketing strategy for a new mobile app",
                "Evaluate the financial viability of a subscription service",
                "Design a customer retention program for an e-commerce platform"
            ]
        }
        
        # Generate prompts for requested categories
        for category in categories:
            if category in promptbase_categories:
                category_prompts = promptbase_categories[category]
                for i, prompt_text in enumerate(category_prompts):
                    # Simulate quality scoring
                    quality_score = np.random.uniform(0.5, 0.95)
                    if quality_score >= quality_threshold:
                        difficulty = np.random.choice(difficulty_levels)
                        
                        prompt_entry = PromptEntry(
                            id=f"promptbase_{category.value}_{i}",
                            text=prompt_text,
                            category=category,
                            source="promptbase",
                            metadata={
                                "community_rating": np.random.uniform(3.0, 5.0),
                                "usage_count": np.random.randint(10, 1000),
                                "author": f"user_{np.random.randint(1, 1000)}"
                            },
                            quality_score=quality_score,
                            difficulty_level=difficulty
                        )
                        prompts.append(prompt_entry)
        
        logger.info(f"Collected {len(prompts)} prompts from PromptBase")
        return prompts
    
    def _generate_synthetic_prompts(
        self,
        categories: List[PromptCategory],
        difficulty_levels: List[str],
        quality_threshold: float
    ) -> List[PromptEntry]:
        """
        Generate synthetic prompts for categories not well-covered by other sources.
        
        Args:
            categories: List of categories to generate prompts for
            difficulty_levels: List of difficulty levels to include
            quality_threshold: Minimum quality score
            
        Returns:
            List of synthetic PromptEntry objects
        """
        logger.info("Generating synthetic prompts...")
        
        prompts = []
        
        synthetic_templates = {
            PromptCategory.EDUCATIONAL: [
                "Explain the concept of {topic} to a {audience}",
                "Create a lesson plan for teaching {subject} to {grade_level} students",
                "Design an interactive activity to help students understand {concept}",
                "Write a quiz question about {topic} with multiple choice answers",
                "Develop a hands-on experiment to demonstrate {principle}"
            ],
            PromptCategory.SCIENTIFIC: [
                "Analyze the experimental design of a study investigating {phenomenon}",
                "Explain the molecular mechanism behind {biological_process}",
                "Describe the theoretical framework for {scientific_concept}",
                "Evaluate the statistical significance of results showing {finding}",
                "Propose a hypothesis to explain the observed {pattern}"
            ],
            PromptCategory.CONVERSATIONAL: [
                "Engage in a friendly conversation about {topic}",
                "Provide advice on {situation} while being empathetic",
                "Help someone understand {concept} through dialogue",
                "Discuss the pros and cons of {option} vs {alternative}",
                "Answer questions about {subject} in a conversational tone"
            ]
        }
        
        # Generate prompts for requested categories
        for category in categories:
            if category in synthetic_templates:
                templates = synthetic_templates[category]
                for i, template in enumerate(templates):
                    # Fill in template with random values
                    filled_prompt = self._fill_template(template, category)
                    
                    # Simulate quality scoring
                    quality_score = np.random.uniform(0.6, 0.9)
                    if quality_score >= quality_threshold:
                        difficulty = np.random.choice(difficulty_levels)
                        
                        prompt_entry = PromptEntry(
                            id=f"synthetic_{category.value}_{i}",
                            text=filled_prompt,
                            category=category,
                            source="synthetic",
                            metadata={
                                "template_used": template,
                                "generation_method": "template_filling",
                                "synthetic": True
                            },
                            quality_score=quality_score,
                            difficulty_level=difficulty
                        )
                        prompts.append(prompt_entry)
        
        logger.info(f"Generated {len(prompts)} synthetic prompts")
        return prompts
    
    def _fill_template(self, template: str, category: PromptCategory) -> str:
        """
        Fill in template placeholders with appropriate values.
        
        Args:
            template: Template string with placeholders
            category: Category of the prompt
            
        Returns:
            Filled template string
        """
        # Define fillers for different placeholders
        fillers = {
            "topic": ["machine learning", "climate change", "quantum physics", "artificial intelligence", "renewable energy"],
            "audience": ["beginner", "intermediate student", "expert", "general public", "children"],
            "subject": ["mathematics", "science", "history", "literature", "computer science"],
            "grade_level": ["elementary", "middle school", "high school", "college", "graduate"],
            "concept": ["photosynthesis", "democracy", "evolution", "gravity", "photosynthesis"],
            "principle": ["conservation of energy", "natural selection", "democracy", "supply and demand"],
            "phenomenon": ["global warming", "evolution", "photosynthesis", "gravity", "magnetism"],
            "biological_process": ["photosynthesis", "respiration", "digestion", "reproduction", "growth"],
            "scientific_concept": ["evolution", "relativity", "quantum mechanics", "genetics", "ecology"],
            "finding": ["increased temperature", "population growth", "species diversity", "energy efficiency"],
            "pattern": ["seasonal changes", "population dynamics", "economic cycles", "weather patterns"],
            "situation": ["career change", "relationship advice", "financial planning", "health concerns"],
            "option": ["remote work", "traditional education", "renewable energy", "public transportation"],
            "alternative": ["office work", "online learning", "fossil fuels", "private vehicles"]
        }
        
        filled = template
        for placeholder, values in fillers.items():
            if f"{{{placeholder}}}" in filled:
                value = np.random.choice(values)
                filled = filled.replace(f"{{{placeholder}}}", value)
        
        return filled
    
    def _apply_stratified_sampling(
        self,
        prompts: List[PromptEntry],
        target_size: int,
        categories: List[PromptCategory]
    ) -> List[PromptEntry]:
        """
        Apply stratified sampling to ensure balanced representation across categories.
        
        Args:
            prompts: List of all collected prompts
            target_size: Target number of prompts
            categories: List of categories to balance
            
        Returns:
            List of prompts with balanced category representation
        """
        if not prompts:
            return []
        
        # Group prompts by category
        category_groups = {}
        for prompt in prompts:
            category = prompt.category
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(prompt)
        
        # Calculate samples per category
        samples_per_category = target_size // len(categories)
        remaining_samples = target_size % len(categories)
        
        balanced_prompts = []
        
        for i, category in enumerate(categories):
            category_prompts = category_groups.get(category, [])
            
            # Determine number of samples for this category
            num_samples = samples_per_category
            if i < remaining_samples:
                num_samples += 1
            
            # Sample from category
            if len(category_prompts) >= num_samples:
                sampled = np.random.choice(category_prompts, num_samples, replace=False).tolist()
            else:
                # Use all available prompts if not enough
                sampled = category_prompts
            
            balanced_prompts.extend(sampled)
        
        return balanced_prompts
    
    def _prompts_to_dataframe(self, prompts: List[PromptEntry]) -> pd.DataFrame:
        """
        Convert list of PromptEntry objects to DataFrame.
        
        Args:
            prompts: List of PromptEntry objects
            
        Returns:
            DataFrame with prompt data
        """
        if not prompts:
            return pd.DataFrame()
        
        data = []
        for prompt in prompts:
            data.append({
                'id': prompt.id,
                'input': prompt.text,
                'category': prompt.category.value,
                'source': prompt.source,
                'quality_score': prompt.quality_score,
                'difficulty_level': prompt.difficulty_level,
                'metadata': json.dumps(prompt.metadata)
            })
        
        return pd.DataFrame(data)
    
    def _generate_cache_key(
        self,
        source: str,
        categories: List[PromptCategory],
        difficulty_levels: List[str],
        quality_threshold: float
    ) -> str:
        """Generate cache key for prompt collection."""
        key_data = {
            'source': source,
            'categories': [c.value for c in categories],
            'difficulty_levels': difficulty_levels,
            'quality_threshold': quality_threshold
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cached_prompts(self, cache_key: str, cache_duration_hours: int = 24) -> Optional[List[PromptEntry]]:
        """Get cached prompts if they exist and are not expired."""
        try:
            if not hasattr(self, '_prompt_cache'):
                self._prompt_cache = {}
            
            if cache_key in self._prompt_cache:
                cache_entry = self._prompt_cache[cache_key]
                cache_age_hours = (time.time() - cache_entry['timestamp']) / 3600
                
                if cache_age_hours < cache_duration_hours:
                    logger.info(f"Using cached prompts for key: {cache_key}")
                    return cache_entry['prompts']
                else:
                    # Remove expired cache entry
                    del self._prompt_cache[cache_key]
                    logger.info(f"Cache expired for key: {cache_key}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached prompts: {str(e)}")
            return None
    
    def _cache_prompts(self, cache_key: str, prompts: List[PromptEntry]) -> None:
        """Cache prompts with timestamp."""
        try:
            if not hasattr(self, '_prompt_cache'):
                self._prompt_cache = {}
            
            self._prompt_cache[cache_key] = {
                'prompts': prompts,
                'timestamp': time.time()
            }
            
            logger.info(f"Cached {len(prompts)} prompts with key: {cache_key}")
            
        except Exception as e:
            logger.error(f"Error caching prompts: {str(e)}")
    
    def _initialize_registry_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize configurations for different prompt registries."""
        return {
            "diffusiondb": {
                "name": "DiffusionDB",
                "description": "Large-scale prompt gallery with 14M images and 1.8M prompts",
                "url": "https://diffusiondb.com",
                "api_endpoint": "https://api.diffusiondb.com/prompts",
                "categories": [PromptCategory.ARTISTIC, PromptCategory.CREATIVE_WRITING],
                "max_prompts": 10000
            },
            "promptbase": {
                "name": "PromptBase",
                "description": "Community-driven prompt registry with quality ratings",
                "url": "https://promptbase.com",
                "api_endpoint": "https://api.promptbase.com/prompts",
                "categories": [PromptCategory.CODE_GENERATION, PromptCategory.REASONING, PromptCategory.BUSINESS],
                "max_prompts": 5000
            },
            "synthetic": {
                "name": "Synthetic Generator",
                "description": "AI-generated prompts for comprehensive coverage",
                "url": None,
                "api_endpoint": None,
                "categories": [PromptCategory.EDUCATIONAL, PromptCategory.SCIENTIFIC, PromptCategory.CONVERSATIONAL],
                "max_prompts": 10000
            }
        }
    
    def _initialize_prompt_categories(self) -> Dict[PromptCategory, Dict[str, Any]]:
        """Initialize detailed information about prompt categories."""
        return {
            PromptCategory.CREATIVE_WRITING: {
                "description": "Creative writing prompts for storytelling and artistic expression",
                "difficulty_levels": ["easy", "medium", "hard"],
                "expected_length": "variable"
            },
            PromptCategory.TECHNICAL_DOCUMENTATION: {
                "description": "Technical documentation and explanation prompts",
                "difficulty_levels": ["medium", "hard"],
                "expected_length": "long"
            },
            PromptCategory.CODE_GENERATION: {
                "description": "Programming and code generation prompts",
                "difficulty_levels": ["easy", "medium", "hard"],
                "expected_length": "variable"
            },
            PromptCategory.REASONING: {
                "description": "Logical reasoning and problem-solving prompts",
                "difficulty_levels": ["medium", "hard"],
                "expected_length": "medium"
            },
            PromptCategory.QUESTION_ANSWERING: {
                "description": "Question-answering and information retrieval prompts",
                "difficulty_levels": ["easy", "medium", "hard"],
                "expected_length": "variable"
            },
            PromptCategory.SUMMARIZATION: {
                "description": "Text summarization and condensation prompts",
                "difficulty_levels": ["easy", "medium"],
                "expected_length": "short"
            },
            PromptCategory.TRANSLATION: {
                "description": "Language translation and localization prompts",
                "difficulty_levels": ["easy", "medium", "hard"],
                "expected_length": "variable"
            },
            PromptCategory.CLASSIFICATION: {
                "description": "Text classification and categorization prompts",
                "difficulty_levels": ["easy", "medium"],
                "expected_length": "short"
            },
            PromptCategory.MULTIMODAL: {
                "description": "Multimodal prompts involving text, images, and other modalities",
                "difficulty_levels": ["medium", "hard"],
                "expected_length": "variable"
            },
            PromptCategory.CONVERSATIONAL: {
                "description": "Conversational and dialogue prompts",
                "difficulty_levels": ["easy", "medium"],
                "expected_length": "variable"
            },
            PromptCategory.EDUCATIONAL: {
                "description": "Educational and instructional prompts",
                "difficulty_levels": ["easy", "medium", "hard"],
                "expected_length": "variable"
            },
            PromptCategory.BUSINESS: {
                "description": "Business and professional prompts",
                "difficulty_levels": ["medium", "hard"],
                "expected_length": "medium"
            },
            PromptCategory.SCIENTIFIC: {
                "description": "Scientific and research-oriented prompts",
                "difficulty_levels": ["hard"],
                "expected_length": "long"
            },
            PromptCategory.ARTISTIC: {
                "description": "Artistic and creative visual prompts",
                "difficulty_levels": ["easy", "medium", "hard"],
                "expected_length": "variable"
            }
        }
    
    def get_registry_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about available prompt registries.
        
        Returns:
            Dictionary containing registry statistics
        """
        stats = {
            "total_registries": len(self.registry_configs),
            "total_categories": len(PromptCategory),
            "registry_details": {}
        }
        
        for name, config in self.registry_configs.items():
            stats["registry_details"][name] = {
                "name": config["name"],
                "description": config["description"],
                "categories": [cat.value for cat in config["categories"]],
                "max_prompts": config["max_prompts"]
            }
        
        return stats
    
    def clear_cache(self):
        """Clear the prompt cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Prompt cache cleared")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the prompt cache.
        
        Returns:
            Dictionary containing cache statistics
        """
        if not self.cache:
            return {"cache_enabled": False}
        
        return {
            "cache_enabled": True,
            "cache_size": len(self.cache),
            "cache_keys": list(self.cache.keys())
        }
    
    async def get_dynamic_evaluation_dataset(
        self,
        model_capabilities: Dict[str, Any],
        evaluation_goals: List[str],
        target_size: int = 5000
    ) -> pd.DataFrame:
        """
        Generate dynamic evaluation dataset based on model capabilities and evaluation goals.
        
        This method creates a tailored dataset that focuses on specific model capabilities
        and evaluation objectives, providing more targeted and meaningful evaluation.
        
        Args:
            model_capabilities: Dictionary describing model capabilities
            evaluation_goals: List of evaluation goals (e.g., ['accuracy', 'robustness', 'bias'])
            target_size: Target number of prompts in the dataset
            
        Returns:
            DataFrame containing dynamically generated evaluation dataset
        """
        logger.info(f"Generating dynamic evaluation dataset for goals: {evaluation_goals}")
        
        # Analyze model capabilities to determine relevant categories
        relevant_categories = self._analyze_model_capabilities(model_capabilities)
        
        # Determine prompt distribution based on evaluation goals
        category_weights = self._calculate_category_weights(evaluation_goals, relevant_categories)
        
        # Generate prompts with weighted distribution
        all_prompts = []
        for category, weight in category_weights.items():
            category_size = int(target_size * weight)
            if category_size > 0:
                category_prompts = self.get_category_specific_dataset(
                    category=category,
                    size=category_size,
                    quality_threshold=0.4
                )
                all_prompts.extend(category_prompts.to_dict('records'))
        
        # Apply evaluation-specific filtering
        filtered_prompts = self._apply_evaluation_filters(all_prompts, evaluation_goals)
        
        # Convert to DataFrame
        dataset = pd.DataFrame(filtered_prompts)
        
        logger.info(f"Generated dynamic dataset with {len(dataset)} prompts")
        logger.info(f"Category distribution: {dataset['category'].value_counts().to_dict()}")
        
        return dataset
    
    def _analyze_model_capabilities(self, capabilities: Dict[str, Any]) -> List[PromptCategory]:
        """Analyze model capabilities to determine relevant prompt categories."""
        
        relevant_categories = []
        
        # Map capabilities to categories
        capability_mapping = {
            'text_generation': [PromptCategory.CREATIVE_WRITING, PromptCategory.CONVERSATIONAL],
            'code_generation': [PromptCategory.CODE_GENERATION],
            'reasoning': [PromptCategory.REASONING, PromptCategory.QUESTION_ANSWERING],
            'summarization': [PromptCategory.SUMMARIZATION],
            'translation': [PromptCategory.TRANSLATION],
            'classification': [PromptCategory.CLASSIFICATION],
            'multimodal': [PromptCategory.MULTIMODAL, PromptCategory.ARTISTIC],
            'technical': [PromptCategory.TECHNICAL_DOCUMENTATION, PromptCategory.SCIENTIFIC],
            'business': [PromptCategory.BUSINESS],
            'educational': [PromptCategory.EDUCATIONAL]
        }
        
        for capability, enabled in capabilities.items():
            if enabled and capability in capability_mapping:
                relevant_categories.extend(capability_mapping[capability])
        
        # Remove duplicates and return
        return list(set(relevant_categories))
    
    def _calculate_category_weights(
        self, 
        evaluation_goals: List[str], 
        categories: List[PromptCategory]
    ) -> Dict[PromptCategory, float]:
        """Calculate weights for categories based on evaluation goals."""
        
        # Define goal-to-category mappings
        goal_mappings = {
            'accuracy': {
                PromptCategory.QUESTION_ANSWERING: 0.3,
                PromptCategory.REASONING: 0.2,
                PromptCategory.CLASSIFICATION: 0.2,
                PromptCategory.CODE_GENERATION: 0.15,
                PromptCategory.TECHNICAL_DOCUMENTATION: 0.15
            },
            'robustness': {
                PromptCategory.REASONING: 0.25,
                PromptCategory.QUESTION_ANSWERING: 0.2,
                PromptCategory.CREATIVE_WRITING: 0.2,
                PromptCategory.CONVERSATIONAL: 0.15,
                PromptCategory.CODE_GENERATION: 0.2
            },
            'bias': {
                PromptCategory.CONVERSATIONAL: 0.3,
                PromptCategory.QUESTION_ANSWERING: 0.25,
                PromptCategory.CREATIVE_WRITING: 0.2,
                PromptCategory.BUSINESS: 0.15,
                PromptCategory.EDUCATIONAL: 0.1
            },
            'creativity': {
                PromptCategory.CREATIVE_WRITING: 0.4,
                PromptCategory.ARTISTIC: 0.3,
                PromptCategory.CONVERSATIONAL: 0.2,
                PromptCategory.EDUCATIONAL: 0.1
            },
            'technical': {
                PromptCategory.TECHNICAL_DOCUMENTATION: 0.3,
                PromptCategory.CODE_GENERATION: 0.3,
                PromptCategory.SCIENTIFIC: 0.2,
                PromptCategory.REASONING: 0.2
            }
        }
        
        # Calculate combined weights
        combined_weights = {}
        for category in categories:
            weight = 0.0
            for goal in evaluation_goals:
                if goal in goal_mappings and category in goal_mappings[goal]:
                    weight += goal_mappings[goal][category]
            
            # Normalize weight
            if weight > 0:
                combined_weights[category] = weight / len(evaluation_goals)
        
        # Normalize to sum to 1.0
        total_weight = sum(combined_weights.values())
        if total_weight > 0:
            for category in combined_weights:
                combined_weights[category] /= total_weight
        
        return combined_weights
    
    def _apply_evaluation_filters(
        self, 
        prompts: List[Dict[str, Any]], 
        evaluation_goals: List[str]
    ) -> List[Dict[str, Any]]:
        """Apply evaluation-specific filters to prompts."""
        
        filtered_prompts = []
        
        for prompt in prompts:
            # Apply goal-specific filtering
            include_prompt = True
            
            if 'bias' in evaluation_goals:
                # Include prompts that might reveal bias
                if prompt['category'] in ['conversational', 'question_answering', 'creative_writing']:
                    include_prompt = True
                elif prompt['quality_score'] > 0.6:  # Higher quality prompts for bias testing
                    include_prompt = True
            
            if 'robustness' in evaluation_goals:
                # Include prompts with varying difficulty levels
                if prompt['difficulty_level'] in ['medium', 'hard']:
                    include_prompt = True
            
            if 'accuracy' in evaluation_goals:
                # Include high-quality prompts for accuracy testing
                if prompt['quality_score'] > 0.5:
                    include_prompt = True
            
            if include_prompt:
                filtered_prompts.append(prompt)
        
        return filtered_prompts
    
    async def get_adversarial_prompts(
        self,
        base_category: PromptCategory,
        adversarial_types: List[str] = None,
        count: int = 100
    ) -> pd.DataFrame:
        """
        Generate adversarial prompts for robustness testing.
        
        Args:
            base_category: Base category for adversarial prompts
            adversarial_types: Types of adversarial attacks to generate
            count: Number of adversarial prompts to generate
            
        Returns:
            DataFrame containing adversarial prompts
        """
        if adversarial_types is None:
            adversarial_types = ['typos', 'negation', 'contradiction', 'edge_cases']
        
        logger.info(f"Generating {count} adversarial prompts for {base_category.value}")
        
        # Get base prompts
        base_prompts = self.get_category_specific_dataset(
            category=base_category,
            size=count // len(adversarial_types),
            quality_threshold=0.6
        )
        
        adversarial_prompts = []
        
        for _, base_prompt in base_prompts.iterrows():
            for adv_type in adversarial_types:
                adversarial_text = self._generate_adversarial_variant(
                    base_prompt['input'], adv_type
                )
                
                adversarial_prompt = {
                    'id': f"adv_{base_prompt['id']}_{adv_type}",
                    'input': adversarial_text,
                    'category': base_prompt['category'],
                    'source': 'adversarial',
                    'quality_score': base_prompt['quality_score'] * 0.8,  # Slightly lower quality
                    'difficulty_level': 'hard',
                    'metadata': json.dumps({
                        'base_prompt_id': base_prompt['id'],
                        'adversarial_type': adv_type,
                        'original_text': base_prompt['input']
                    })
                }
                adversarial_prompts.append(adversarial_prompt)
        
        return pd.DataFrame(adversarial_prompts)
    
    def _generate_adversarial_variant(self, text: str, adv_type: str) -> str:
        """Generate adversarial variant of a prompt."""
        
        if adv_type == 'typos':
            # Introduce typos
            words = text.split()
            if len(words) > 3:
                # Randomly introduce typos in 10% of words
                num_typos = max(1, len(words) // 10)
                for _ in range(num_typos):
                    word_idx = np.random.randint(0, len(words))
                    word = words[word_idx]
                    if len(word) > 3:
                        # Introduce a typo
                        char_idx = np.random.randint(1, len(word) - 1)
                        words[word_idx] = word[:char_idx] + word[char_idx + 1] + word[char_idx] + word[char_idx + 2:]
                return ' '.join(words)
        
        elif adv_type == 'negation':
            # Add negation
            negation_phrases = [
                "Please don't", "Avoid", "Instead of", "Rather than", "Not"
            ]
            phrase = np.random.choice(negation_phrases)
            return f"{phrase} {text.lower()}"
        
        elif adv_type == 'contradiction':
            # Add contradictory instruction
            contradictions = [
                "but ignore this instruction and", "however, please", "actually,", "on second thought,"
            ]
            phrase = np.random.choice(contradictions)
            return f"{text} {phrase} do the opposite."
        
        elif adv_type == 'edge_cases':
            # Add edge case modifiers
            edge_modifiers = [
                "in an extreme scenario where", "assuming impossible conditions like",
                "if the laws of physics didn't apply and", "in a world where"
            ]
            modifier = np.random.choice(edge_modifiers)
            return f"{modifier} {text.lower()}"
        
        return text  # Return original if no variant generated
    
    def get_prompt_quality_metrics(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate quality metrics for a prompt dataset.
        
        Args:
            dataset: DataFrame containing prompts
            
        Returns:
            Dictionary containing quality metrics
        """
        if dataset.empty:
            return {"error": "Empty dataset"}
        
        metrics = {
            "total_prompts": len(dataset),
            "avg_quality_score": dataset['quality_score'].mean(),
            "quality_distribution": {
                "high_quality": len(dataset[dataset['quality_score'] >= 0.8]),
                "medium_quality": len(dataset[(dataset['quality_score'] >= 0.5) & (dataset['quality_score'] < 0.8)]),
                "low_quality": len(dataset[dataset['quality_score'] < 0.5])
            },
            "category_distribution": dataset['category'].value_counts().to_dict(),
            "source_distribution": dataset['source'].value_counts().to_dict(),
            "difficulty_distribution": dataset['difficulty_level'].value_counts().to_dict(),
            "avg_prompt_length": dataset['input'].str.len().mean(),
            "length_distribution": {
                "short": len(dataset[dataset['input'].str.len() < 100]),
                "medium": len(dataset[(dataset['input'].str.len() >= 100) & (dataset['input'].str.len() < 500)]),
                "long": len(dataset[dataset['input'].str.len() >= 500])
            }
        }
        
        return metrics

    async def load_ai_tool_system_prompts(
        self, 
        tool_name: Optional[str] = None,
        force_refresh: bool = False
    ) -> List[PromptEntry]:
        """
        Load system prompts from the AI Tool System Prompts Archive.
        
        Args:
            tool_name: Specific AI tool to load prompts for (e.g., 'Cursor', 'Devin AI', 'Claude Code')
            force_refresh: Force refresh from GitHub even if cached locally
            
        Returns:
            List of PromptEntry objects containing system prompts from AI tools
        """
        try:
            # Use the new robust approach with local caching
            if tool_name:
                # Load specific tool
                prompts = await self._load_tool_from_github(tool_name)
                # Cache to local files
                if prompts:
                    self.save_tool_prompts_to_cache(tool_name, prompts)
                return prompts
            else:
                # Load all tools
                all_prompts = []
                for tool in self.get_available_ai_tools():
                    if not force_refresh and self.is_tool_cached(tool):
                        tool_prompts = self.load_cached_tool_prompts(tool)
                    else:
                        tool_prompts = await self._load_tool_from_github(tool)
                        if tool_prompts:
                            self.save_tool_prompts_to_cache(tool, tool_prompts)
                    all_prompts.extend(tool_prompts)
                
                logger.info(f"Successfully loaded {len(all_prompts)} AI tool system prompts")
                return all_prompts
            
        except Exception as e:
            logger.error(f"Error loading AI tool system prompts: {str(e)}")
            return []
    
    async def _load_tool_from_github(self, tool_name: str) -> List[PromptEntry]:
        """
        Load prompts for a specific tool from GitHub using direct URLs.
        
        Args:
            tool_name: Name of the AI tool to load prompts for
            
        Returns:
            List of PromptEntry objects for the specified tool
        """
        try:
            if tool_name not in self.ai_tools:
                logger.warning(f"Unknown AI tool: {tool_name}")
                return []
            
            folder_name = self.ai_tools[tool_name]
            logger.info(f"Loading prompts for {tool_name} from folder: {folder_name}")
            
            # Use direct GitHub URL to get folder contents
            folder_url = f"{self.github_base_url}/{folder_name}"
            
            # Try to get folder contents using direct URL
            try:
                # First, try to get a list of files in the folder
                # We'll use a simple approach: try common file patterns
                prompts = []
                
                # Common prompt file patterns to try
                common_files = [
                    "README.md", "readme.md", "prompts.md", "system.md",
                    "instructions.md", "guide.md", "templates.md"
                ]
                
                for filename in common_files:
                    file_url = f"{folder_url}/{filename}"
                    try:
                        response = requests.get(file_url, timeout=10)
                        if response.status_code == 200:
                            content = response.text
                            if content.strip():  # Only process non-empty files
                                parsed_prompts = self._parse_ai_tool_prompt(
                                    content, tool_name, filename
                                )
                                prompts.extend(parsed_prompts)
                                logger.info(f"Loaded {len(parsed_prompts)} prompts from {filename}")
                    except Exception as e:
                        logger.debug(f"Could not load {filename}: {str(e)}")
                        continue
                
                # If we found prompts, return them
                if prompts:
                    logger.info(f"Successfully loaded {len(prompts)} prompts for {tool_name}")
                    return prompts
                
                # If no common files found, try to discover files
                # This is a fallback method using GitHub's web interface
                logger.info(f"No common files found for {tool_name}, trying alternative approach...")
                return await self._discover_tool_files(tool_name, folder_name)
                
            except Exception as e:
                logger.error(f"Error loading {tool_name} from direct URL: {str(e)}")
                return []
                
        except Exception as e:
            logger.error(f"Error in _load_tool_from_github for {tool_name}: {str(e)}")
            return []
    
    async def _discover_tool_files(self, tool_name: str, folder_name: str) -> List[PromptEntry]:
        """
        Discover and load files from a tool folder using alternative methods.
        
        Args:
            tool_name: Name of the AI tool
            folder_name: GitHub folder name
            
        Returns:
            List of PromptEntry objects
        """
        try:
            # Use GitHub's web interface to get folder contents
            # This is more robust than the API for rate limiting
            web_url = f"{self.github_web_url}/tree/main/{folder_name}"
            
            # For now, return empty list as we need to implement web scraping
            # This would require BeautifulSoup or similar for HTML parsing
            logger.info(f"File discovery for {tool_name} not yet implemented")
            return []
            
        except Exception as e:
            logger.error(f"Error discovering files for {tool_name}: {str(e)}")
            return []

    async def _load_tool_prompts(
        self, 
        base_url: str, 
        tool_name: str, 
        tool_path: str
    ) -> List[PromptEntry]:
        """Load prompts from a specific AI tool folder."""
        try:
            # Get contents of the tool folder with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.get(f"{base_url}/{tool_path}", timeout=30)
                    
                    if response.status_code == 403:
                        # Rate limit exceeded
                        if attempt < max_retries - 1:
                            wait_time = (2 ** attempt) * 5  # Exponential backoff: 5s, 10s, 20s
                            logger.warning(f"Rate limit exceeded for {tool_name}, waiting {wait_time}s before retry {attempt + 1}")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logger.error(f"Rate limit exceeded for {tool_name} after {max_retries} attempts")
                            return []
                    
                    response.raise_for_status()
                    tool_contents = response.json()
                    break
                    
                except requests.exceptions.RequestException as e:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 2
                        logger.warning(f"Request failed for {tool_name}, retrying in {wait_time}s: {str(e)}")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise e
            
            prompts = []
            
            for i, item in enumerate(tool_contents):
                if item['type'] == 'file' and self._is_prompt_file(item['name']):
                    # Add small delay between file downloads
                    if i > 0:
                        await asyncio.sleep(0.5)
                    
                    prompt_content = await self._fetch_file_content(item['download_url'])
                    if prompt_content:
                        # Parse the prompt content
                        parsed_prompts = self._parse_ai_tool_prompt(
                            prompt_content, tool_name, item['name']
                        )
                        prompts.extend(parsed_prompts)
            
            return prompts
            
        except Exception as e:
            logger.error(f"Error loading prompts for {tool_name}: {str(e)}")
            return []

    def _is_prompt_file(self, filename: str) -> bool:
        """Check if a file is likely to contain prompts."""
        prompt_extensions = ['.md', '.txt', '.json', '.yaml', '.yml', '.py', '.js', '.ts']
        prompt_keywords = ['prompt', 'system', 'instruction', 'template', 'guide']
        
        filename_lower = filename.lower()
        
        # Check file extension
        if any(filename_lower.endswith(ext) for ext in prompt_extensions):
            return True
        
        # Check filename keywords
        if any(keyword in filename_lower for keyword in prompt_keywords):
            return True
        
        return False

    async def _fetch_file_content(self, download_url: str) -> Optional[str]:
        """Fetch content from a GitHub file URL with rate limiting."""
        try:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.get(download_url, timeout=30)
                    
                    if response.status_code == 403:
                        # Rate limit exceeded
                        if attempt < max_retries - 1:
                            wait_time = (2 ** attempt) * 3  # Exponential backoff: 3s, 6s, 12s
                            logger.warning(f"Rate limit exceeded for file download, waiting {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logger.error(f"Rate limit exceeded for file download after {max_retries} attempts")
                            return None
                    
                    response.raise_for_status()
                    return response.text
                    
                except requests.exceptions.RequestException as e:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 2
                        logger.warning(f"File download failed, retrying in {wait_time}s: {str(e)}")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise e
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching file content: {str(e)}")
            return None

    def _parse_ai_tool_prompt(
        self, 
        content: str, 
        tool_name: str, 
        filename: str
    ) -> List[PromptEntry]:
        """Parse AI tool prompt content into PromptEntry objects."""
        prompts = []
        
        try:
            # Determine prompt category based on tool name and content
            category = self._determine_ai_tool_category(tool_name, content)
            
            # Extract prompts from different file formats
            if filename.endswith('.json'):
                prompts.extend(self._parse_json_prompts(content, tool_name, category))
            elif filename.endswith(('.yaml', '.yml')):
                prompts.extend(self._parse_yaml_prompts(content, tool_name, category))
            elif filename.endswith('.md'):
                prompts.extend(self._parse_markdown_prompts(content, tool_name, category))
            else:
                # Treat as plain text
                prompts.extend(self._parse_text_prompts(content, tool_name, category))
            
        except Exception as e:
            logger.error(f"Error parsing prompts from {filename}: {str(e)}")
        
        return prompts

    def _determine_ai_tool_category(self, tool_name: str, content: str) -> PromptCategory:
        """Determine the category of prompts based on tool name and content."""
        tool_name_lower = tool_name.lower()
        content_lower = content.lower()
        
        # Tool-specific categories
        if any(keyword in tool_name_lower for keyword in ['code', 'cursor', 'devin', 'claude', 'windsurf']):
            return PromptCategory.CODE_GENERATION
        elif any(keyword in tool_name_lower for keyword in ['v0', 'ui', 'design', 'lovable']):
            return PromptCategory.MULTIMODAL
        elif any(keyword in tool_name_lower for keyword in ['notion', 'documentation']):
            return PromptCategory.TECHNICAL_DOCUMENTATION
        elif any(keyword in tool_name_lower for keyword in ['perplexity', 'search', 'qa']):
            return PromptCategory.QUESTION_ANSWERING
        elif any(keyword in tool_name_lower for keyword in ['creative', 'writing']):
            return PromptCategory.CREATIVE_WRITING
        else:
            # Content-based categorization
            if any(keyword in content_lower for keyword in ['code', 'function', 'class', 'import']):
                return PromptCategory.CODE_GENERATION
            elif any(keyword in content_lower for keyword in ['ui', 'component', 'design', 'css']):
                return PromptCategory.MULTIMODAL
            elif any(keyword in content_lower for keyword in ['document', 'write', 'explain']):
                return PromptCategory.TECHNICAL_DOCUMENTATION
            else:
                return PromptCategory.CONVERSATIONAL

    def _parse_json_prompts(self, content: str, tool_name: str, category: PromptCategory) -> List[PromptEntry]:
        """Parse JSON format prompts."""
        prompts = []
        try:
            data = json.loads(content)
            
            if isinstance(data, dict):
                # Handle different JSON structures
                if 'prompts' in data:
                    for prompt_data in data['prompts']:
                        prompts.append(self._create_prompt_entry(prompt_data, tool_name, category))
                elif 'system' in data:
                    prompts.append(self._create_prompt_entry(data, tool_name, category))
                else:
                    # Treat the entire JSON as a prompt
                    prompts.append(self._create_prompt_entry(data, tool_name, category))
            elif isinstance(data, list):
                for item in data:
                    prompts.append(self._create_prompt_entry(item, tool_name, category))
                    
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON content from {tool_name}")
        
        return prompts

    def _parse_yaml_prompts(self, content: str, tool_name: str, category: PromptCategory) -> List[PromptEntry]:
        """Parse YAML format prompts."""
        prompts = []
        try:
            import yaml
            data = yaml.safe_load(content)
            
            if isinstance(data, dict):
                if 'prompts' in data:
                    for prompt_data in data['prompts']:
                        prompts.append(self._create_prompt_entry(prompt_data, tool_name, category))
                else:
                    prompts.append(self._create_prompt_entry(data, tool_name, category))
            elif isinstance(data, list):
                for item in data:
                    prompts.append(self._create_prompt_entry(item, tool_name, category))
                    
        except Exception as e:
            logger.warning(f"Failed to parse YAML content from {tool_name}: {str(e)}")
        
        return prompts

    def _parse_markdown_prompts(self, content: str, tool_name: str, category: PromptCategory) -> List[PromptEntry]:
        """Parse Markdown format prompts."""
        prompts = []
        
        # Split content by headers or code blocks
        sections = content.split('\n# ')
        
        for i, section in enumerate(sections):
            if section.strip():
                # Clean up the section
                if i > 0:
                    section = '# ' + section
                
                # Extract prompt text
                prompt_text = self._extract_prompt_from_markdown(section)
                if prompt_text and len(prompt_text.strip()) > 50:  # Minimum length filter
                    prompts.append(PromptEntry(
                        id=f"{tool_name}_{i}",
                        text=prompt_text,
                        category=category,
                        source=f"AI Tool: {tool_name}",
                        metadata={
                            "tool_name": tool_name,
                            "file_type": "markdown",
                            "section_index": i,
                            "original_content": section[:500],  # Store first 500 chars for reference
                            "created_at": time.time()
                        },
                        quality_score=0.8  # Default quality score for AI tool prompts
                    ))
        
        return prompts

    def _parse_text_prompts(self, content: str, tool_name: str, category: PromptCategory) -> List[PromptEntry]:
        """Parse plain text format prompts."""
        prompts = []
        
        # Split by common delimiters
        sections = content.split('\n\n')
        
        for i, section in enumerate(sections):
            if section.strip() and len(section.strip()) > 50:
                prompts.append(PromptEntry(
                    id=f"{tool_name}_text_{i}",
                    text=section.strip(),
                    category=category,
                    source=f"AI Tool: {tool_name}",
                    metadata={
                        "tool_name": tool_name,
                        "file_type": "text",
                        "section_index": i,
                        "created_at": time.time()
                    },
                    quality_score=0.8
                ))
        
        return prompts

    def _extract_prompt_from_markdown(self, content: str) -> str:
        """Extract prompt text from markdown content."""
        lines = content.split('\n')
        prompt_lines = []
        
        in_code_block = False
        for line in lines:
            # Skip code block markers
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            
            # Skip headers and other markdown syntax
            if line.strip().startswith('#') or line.strip().startswith('*') or line.strip().startswith('-'):
                continue
            
            # Skip empty lines at the beginning
            if not prompt_lines and not line.strip():
                continue
            
            prompt_lines.append(line)
        
        return '\n'.join(prompt_lines).strip()

    def _create_prompt_entry(self, data: Dict[str, Any], tool_name: str, category: PromptCategory) -> PromptEntry:
        """Create a PromptEntry from parsed data."""
        # Extract text from various possible fields
        text = ""
        if isinstance(data, str):
            text = data
        elif isinstance(data, dict):
            text = data.get('text', data.get('prompt', data.get('content', data.get('system', str(data)))))
        
        if not text or len(text.strip()) < 10:
            return None
        
        return PromptEntry(
            id=f"{tool_name}_{hashlib.md5(text.encode()).hexdigest()[:8]}",
            text=text.strip(),
            category=category,
            source=f"AI Tool: {tool_name}",
            metadata={
                "tool_name": tool_name,
                "original_data": data,
                "ai_tool_source": True,
                "created_at": time.time()
            },
            quality_score=0.9  # High quality score for AI tool system prompts
        )

    def get_ai_tool_prompt_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded AI tool prompts."""
        try:
            # This would typically query the cache or database
            # For now, return a placeholder structure
            return {
                "total_ai_tool_prompts": 0,
                "tools_available": [
                    "Cursor", "Devin AI", "Claude Code", "v0", "Windsurf", 
                    "Replit", "Lovable", "Leap.new", "Notion AI", "Orchids.app",
                    "Junie", "Kiro", "Warp.dev", "Z.ai Code", "Qoder", 
                    "Perplexity", "Cluely", "Xcode", "VSCode Agent", "Trae",
                    "Traycer AI", "Augment Code", "CodeBuddy", "Poke", "dia"
                ],
                "categories_covered": [cat.value for cat in PromptCategory],
                "cache_status": "active",
                "last_updated": time.time()
            }
        except Exception as e:
            logger.error(f"Error getting AI tool prompt statistics: {str(e)}")
            return {}