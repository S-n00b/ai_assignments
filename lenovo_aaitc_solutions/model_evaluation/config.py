"""
Model Configuration Module for Lenovo AAITC Assignment

This module defines configuration classes for the latest foundation models (Q3 2025),
including international models (GPT-5, GPT-5-Codex, Claude 3.5 Sonnet, Llama 3.3, Grok-2, Mixtral),
Chinese models (Qwen 2.5, Qwen-VL Plus), specialized code generation models
(DeepSeek Coder V3, CodeLlama 3.1, StarCoder2, WizardCoder, Phind CodeLlama, Magicoder, Qwen Coder),
and voice-to-voice models (ElevenLabs, Azure Speech, AWS Polly, Google TTS, OpenAI Voice, Replica, Descript, Baidu Deep Voice, Google Moshi)
for Lenovo's Hong Kong operations.

Key Features:
- Latest model versions with updated capabilities
- Chinese language support for Hong Kong market
- Enhanced multimodal capabilities (image, video, audio, document processing)
- Voice-to-voice models with real-time conversation capabilities
- Comprehensive evaluation metrics
- Task type definitions including Chinese-specific and voice-specific tasks
- Cost and performance specifications
- License information and compliance tracking
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
import json


class TaskType(Enum):
    """Enumeration of evaluation task types for comprehensive model assessment"""
    TEXT_GENERATION = "text_generation"
    SUMMARIZATION = "summarization"
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    QUESTION_ANSWERING = "qa"
    TRANSLATION = "translation"
    CLASSIFICATION = "classification"
    MULTIMODAL = "multimodal"
    CREATIVE_WRITING = "creative_writing"
    TECHNICAL_DOCUMENTATION = "technical_documentation"
    # Enhanced multimodal tasks for Q3 2025
    IMAGE_ANALYSIS = "image_analysis"
    VIDEO_UNDERSTANDING = "video_understanding"
    AUDIO_PROCESSING = "audio_processing"
    DOCUMENT_OCR = "document_ocr"
    VISUAL_QUESTION_ANSWERING = "visual_qa"
    MULTIMODAL_REASONING = "multimodal_reasoning"
    # Chinese language specific tasks
    CHINESE_NLP = "chinese_nlp"
    CHINESE_CODE_GENERATION = "chinese_code_generation"
    CHINESE_MULTIMODAL = "chinese_multimodal"
    # Voice-to-voice tasks for 2025
    VOICE_TO_VOICE = "voice_to_voice"
    SPEECH_SYNTHESIS = "speech_synthesis"
    VOICE_CLONING = "voice_cloning"
    REAL_TIME_CONVERSATION = "real_time_conversation"
    MULTILINGUAL_VOICE = "multilingual_voice"
    EMOTION_SYNTHESIS = "emotion_synthesis"


@dataclass
class ModelConfig:
    """
    Configuration class for foundation models with latest Q3 2025 specifications.
    
    This class encapsulates all necessary configuration parameters for evaluating
    state-of-the-art foundation models, including international models (GPT-5, GPT-5-Codex, 
    Claude 3.5 Sonnet, Llama 3.3, Grok-2, Mixtral), Chinese models (Qwen 2.5, Qwen-VL Plus),
    specialized code generation models (DeepSeek Coder V3, CodeLlama 3.1, StarCoder2, 
    WizardCoder, Phind CodeLlama, Magicoder, Qwen Coder), and voice-to-voice models
    (ElevenLabs, Azure Speech, AWS Polly, Google TTS, OpenAI Voice, Replica, Descript, Baidu Deep Voice, Google Moshi)
    for Lenovo's Hong Kong operations.
    
    Attributes:
        name: Human-readable model name
        provider: API provider (openai, anthropic, meta, alibaba, xai, mistral, deepseek, huggingface, wizardlm, phind, ise-uiuc, elevenlabs, microsoft, aws, google, replica, descript, baidu, local)
        model_id: Specific model identifier for API calls
        api_key: API key for authentication (optional for local models)
        max_tokens: Maximum tokens to generate per request
        temperature: Sampling temperature for response generation
        cost_per_1k_tokens: Cost in USD per 1000 tokens (Q3 2025 pricing)
        context_window: Maximum context window size
        capabilities: List of model capabilities including multimodal and Chinese language support
        deployment_scenarios: Supported deployment environments
        performance_profile: Expected performance characteristics including multimodal metrics
        license_type: Type of license (commercial, open_source, research_only, api_only)
        license_details: Specific license text or reference
        commercial_use_allowed: Whether commercial use is permitted
        redistribution_allowed: Whether redistribution is permitted
        modification_allowed: Whether modification is permitted
    """
    
    # Core identification
    name: str
    provider: str  # 'openai', 'anthropic', 'meta', 'alibaba', 'xai', 'mistral', 'deepseek', 'huggingface', 'wizardlm', 'phind', 'ise-uiuc', 'elevenlabs', 'microsoft', 'aws', 'google', 'replica', 'descript', 'baidu', 'local'
    model_id: str
    
    # API configuration
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    
    # Generation parameters
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # Cost and performance (Q3 2025 pricing)
    cost_per_1k_tokens_input: float = 0.0
    cost_per_1k_tokens_output: float = 0.0
    cost_per_1k_tokens: float = 0.0  # Legacy field for backward compatibility
    
    # Model specifications
    context_window: int = 4096
    parameters: int = 0  # Model parameter count in billions
    
    # Capabilities and strengths
    capabilities: List[str] = field(default_factory=list)
    deployment_scenarios: List[str] = field(default_factory=list)
    
    # Performance characteristics
    performance_profile: Dict[str, Any] = field(default_factory=dict)
    
    # License information
    license_type: str = "unknown"  # e.g., "commercial", "open_source", "research_only", "api_only"
    license_details: str = ""  # Specific license text or reference
    commercial_use_allowed: bool = False
    redistribution_allowed: bool = False
    modification_allowed: bool = False
    
    def __post_init__(self):
        """Post-initialization processing to set derived fields"""
        # Calculate average cost if not provided
        if self.cost_per_1k_tokens == 0.0 and (self.cost_per_1k_tokens_input > 0 or self.cost_per_1k_tokens_output > 0):
            self.cost_per_1k_tokens = (self.cost_per_1k_tokens_input + self.cost_per_1k_tokens_output) / 2
        
        # Set default capabilities based on model name
        if not self.capabilities:
            self.capabilities = self._get_default_capabilities()
        
        # Set default deployment scenarios
        if not self.deployment_scenarios:
            self.deployment_scenarios = self._get_default_deployment_scenarios()
    
    def _get_default_capabilities(self) -> List[str]:
        """Get default capabilities based on model name and provider"""
        capabilities = []
        
        if "gpt-5" in self.name.lower():
            capabilities.extend([
                "advanced_reasoning",
                "multimodal_processing", 
                "code_generation",
                "mathematical_reasoning",
                "creative_writing",
                "technical_documentation",
                "image_analysis",
                "video_understanding",
                "visual_qa"
            ])
        elif "gpt-5-codex" in self.name.lower():
            capabilities.extend([
                "code_generation",
                "software_engineering",
                "debugging",
                "code_review",
                "documentation_generation",
                "technical_analysis",
                "multimodal_code_understanding"
            ])
        elif "claude" in self.name.lower():
            capabilities.extend([
                "conversational_ai",
                "reasoning",
                "analysis",
                "creative_writing",
                "code_generation",
                "multimodal_processing",
                "image_analysis",
                "document_ocr"
            ])
        elif "llama" in self.name.lower():
            capabilities.extend([
                "text_generation",
                "conversation",
                "reasoning",
                "code_generation",
                "multilingual_support",
                "multimodal_processing"
            ])
        elif "qwen" in self.name.lower():
            capabilities.extend([
                "chinese_nlp",
                "multilingual_support",
                "code_generation",
                "multimodal_processing",
                "chinese_code_generation",
                "chinese_multimodal",
                "image_analysis",
                "document_ocr",
                "visual_qa"
            ])
        elif "grok" in self.name.lower():
            capabilities.extend([
                "advanced_reasoning",
                "multimodal_processing",
                "real_time_processing",
                "humor_generation",
                "creative_writing",
                "code_generation",
                "image_analysis",
                "video_understanding"
            ])
        elif "mixtral" in self.name.lower():
            capabilities.extend([
                "multilingual_support",
                "code_generation",
                "reasoning",
                "multimodal_processing",
                "efficient_inference",
                "image_analysis",
                "document_processing"
            ])
        elif "deepseek" in self.name.lower():
            capabilities.extend([
                "code_generation",
                "software_engineering",
                "debugging",
                "code_review",
                "mathematical_reasoning",
                "technical_analysis",
                "chinese_code_generation"
            ])
        elif "codellama" in self.name.lower():
            capabilities.extend([
                "code_generation",
                "software_engineering",
                "debugging",
                "code_completion",
                "documentation_generation",
                "multilingual_code_support"
            ])
        elif "starcoder" in self.name.lower():
            capabilities.extend([
                "code_generation",
                "code_completion",
                "software_engineering",
                "debugging",
                "code_explanation",
                "multilingual_programming"
            ])
        elif "wizardcoder" in self.name.lower():
            capabilities.extend([
                "code_generation",
                "software_engineering",
                "debugging",
                "code_optimization",
                "algorithm_implementation",
                "technical_documentation"
            ])
        elif "phind" in self.name.lower():
            capabilities.extend([
                "code_generation",
                "software_engineering",
                "debugging",
                "code_explanation",
                "technical_analysis",
                "research_assistance"
            ])
        elif "elevenlabs" in self.name.lower():
            capabilities.extend([
                "voice_to_voice",
                "speech_synthesis",
                "voice_cloning",
                "real_time_conversation",
                "multilingual_voice",
                "emotion_synthesis"
            ])
        elif "azure" in self.name.lower() and "speech" in self.name.lower():
            capabilities.extend([
                "voice_to_voice",
                "speech_synthesis",
                "real_time_conversation",
                "multilingual_voice",
                "emotion_synthesis"
            ])
        elif "aws" in self.name.lower() and "polly" in self.name.lower():
            capabilities.extend([
                "voice_to_voice",
                "speech_synthesis",
                "multilingual_voice",
                "emotion_synthesis"
            ])
        elif "google" in self.name.lower() and "text-to-speech" in self.name.lower():
            capabilities.extend([
                "voice_to_voice",
                "speech_synthesis",
                "multilingual_voice",
                "emotion_synthesis"
            ])
        elif "openai" in self.name.lower() and "voice" in self.name.lower():
            capabilities.extend([
                "voice_to_voice",
                "speech_synthesis",
                "real_time_conversation",
                "multilingual_voice"
            ])
        elif "replica" in self.name.lower():
            capabilities.extend([
                "voice_to_voice",
                "speech_synthesis",
                "voice_cloning",
                "real_time_conversation",
                "emotion_synthesis"
            ])
        elif "descript" in self.name.lower():
            capabilities.extend([
                "voice_to_voice",
                "speech_synthesis",
                "voice_cloning",
                "real_time_conversation"
            ])
        elif "moshi" in self.name.lower():
            capabilities.extend([
                "voice_to_voice",
                "speech_synthesis",
                "real_time_conversation",
                "multilingual_voice",
                "emotion_synthesis",
                "facial_expression_sync"
            ])
        
        return capabilities
    
    def _get_default_deployment_scenarios(self) -> List[str]:
        """Get default deployment scenarios based on model characteristics"""
        scenarios = ["cloud"]
        
        # Add edge deployment for smaller models
        if self.parameters < 50:  # Less than 50B parameters
            scenarios.extend(["edge", "mobile"])
        
        # Add local deployment for open-source models
        if self.provider in ["meta", "alibaba", "mistral", "deepseek", "huggingface", "wizardlm", "ise-uiuc", "local"]:
            scenarios.append("on_premises")
        
        return scenarios
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization"""
        return {
            "name": self.name,
            "provider": self.provider,
            "model_id": self.model_id,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "cost_per_1k_tokens": self.cost_per_1k_tokens,
            "context_window": self.context_window,
            "parameters": self.parameters,
            "capabilities": self.capabilities,
            "deployment_scenarios": self.deployment_scenarios,
            "license_type": self.license_type,
            "license_details": self.license_details,
            "commercial_use_allowed": self.commercial_use_allowed,
            "redistribution_allowed": self.redistribution_allowed,
            "modification_allowed": self.modification_allowed
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create ModelConfig from dictionary"""
        return cls(**config_dict)


@dataclass
class EvaluationMetrics:
    """
    Comprehensive container for evaluation metrics across multiple dimensions.
    
    This class aggregates all relevant metrics for model evaluation, including
    quality metrics (BLEU, ROUGE, BERT Score), performance metrics (latency,
    throughput), cost efficiency, and robustness metrics.
    
    Attributes:
        Quality Metrics: BLEU, ROUGE variants, BERT Score, F1, semantic similarity
        Performance Metrics: Latency, token generation speed, memory usage, throughput
        Cost & Efficiency: Cost per token, cost efficiency score
        Robustness Metrics: Adversarial robustness, noise tolerance, bias score, safety
    """
    
    # Quality Metrics
    bleu: float = 0.0
    rouge_1: float = 0.0
    rouge_2: float = 0.0
    rouge_l: float = 0.0
    bert_score: float = 0.0
    perplexity: float = 0.0
    f1: float = 0.0
    semantic_similarity: float = 0.0
    
    # Performance Metrics
    latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    memory_mb: float = 0.0
    throughput_qps: float = 0.0
    
    # Cost & Efficiency
    cost_per_1k_tokens: float = 0.0
    cost_efficiency_score: float = 0.0
    
    # Robustness Metrics
    adversarial_robustness: float = 0.0
    noise_tolerance: float = 0.0
    bias_score: float = 0.0
    safety_score: float = 0.0
    
    # Additional Q3 2025 Metrics
    reasoning_accuracy: float = 0.0
    code_success_rate: float = 0.0
    multimodal_accuracy: float = 0.0
    context_utilization: float = 0.0
    
    # Enhanced multimodal metrics
    image_analysis_accuracy: float = 0.0
    video_understanding_accuracy: float = 0.0
    audio_processing_accuracy: float = 0.0
    document_ocr_accuracy: float = 0.0
    visual_qa_accuracy: float = 0.0
    multimodal_reasoning_accuracy: float = 0.0
    
    # Chinese language specific metrics
    chinese_nlp_accuracy: float = 0.0
    chinese_code_accuracy: float = 0.0
    chinese_multimodal_accuracy: float = 0.0
    
    # Voice-to-voice metrics
    voice_to_voice_accuracy: float = 0.0
    speech_synthesis_quality: float = 0.0
    voice_cloning_accuracy: float = 0.0
    real_time_latency_ms: float = 0.0
    multilingual_voice_accuracy: float = 0.0
    emotion_synthesis_accuracy: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary for serialization and analysis"""
        return {
            # Quality metrics
            "bleu": self.bleu,
            "rouge_1": self.rouge_1,
            "rouge_2": self.rouge_2,
            "rouge_l": self.rouge_l,
            "bert_score": self.bert_score,
            "perplexity": self.perplexity,
            "f1": self.f1,
            "semantic_similarity": self.semantic_similarity,
            
            # Performance metrics
            "latency_ms": self.latency_ms,
            "tokens_per_second": self.tokens_per_second,
            "memory_mb": self.memory_mb,
            "throughput_qps": self.throughput_qps,
            
            # Cost & efficiency
            "cost_per_1k_tokens": self.cost_per_1k_tokens,
            "cost_efficiency_score": self.cost_efficiency_score,
            
            # Robustness metrics
            "adversarial_robustness": self.adversarial_robustness,
            "noise_tolerance": self.noise_tolerance,
            "bias_score": self.bias_score,
            "safety_score": self.safety_score,
            
            # Q3 2025 specific metrics
            "reasoning_accuracy": self.reasoning_accuracy,
            "code_success_rate": self.code_success_rate,
            "multimodal_accuracy": self.multimodal_accuracy,
            "context_utilization": self.context_utilization,
            
            # Enhanced multimodal metrics
            "image_analysis_accuracy": self.image_analysis_accuracy,
            "video_understanding_accuracy": self.video_understanding_accuracy,
            "audio_processing_accuracy": self.audio_processing_accuracy,
            "document_ocr_accuracy": self.document_ocr_accuracy,
            "visual_qa_accuracy": self.visual_qa_accuracy,
            "multimodal_reasoning_accuracy": self.multimodal_reasoning_accuracy,
            
            # Chinese language specific metrics
            "chinese_nlp_accuracy": self.chinese_nlp_accuracy,
            "chinese_code_accuracy": self.chinese_code_accuracy,
            "chinese_multimodal_accuracy": self.chinese_multimodal_accuracy,
            
            # Voice-to-voice metrics
            "voice_to_voice_accuracy": self.voice_to_voice_accuracy,
            "speech_synthesis_quality": self.speech_synthesis_quality,
            "voice_cloning_accuracy": self.voice_cloning_accuracy,
            "real_time_latency_ms": self.real_time_latency_ms,
            "multilingual_voice_accuracy": self.multilingual_voice_accuracy,
            "emotion_synthesis_accuracy": self.emotion_synthesis_accuracy
        }
    
    def calculate_overall_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate weighted overall score from all metrics.
        
        Args:
            weights: Optional custom weights for different metric categories.
                    If None, uses default balanced weights.
        
        Returns:
            Overall score between 0 and 1
        """
        if weights is None:
            weights = {
                "quality": 0.4,
                "performance": 0.2,
                "cost_efficiency": 0.2,
                "robustness": 0.2
            }
        
        # Calculate category scores
        quality_score = (
            self.rouge_l + self.bert_score + self.f1 + self.semantic_similarity
        ) / 4
        
        performance_score = min(1.0, 1000 / max(self.latency_ms, 1)) * min(1.0, self.throughput_qps / 10)
        
        cost_efficiency_score = self.cost_efficiency_score
        
        robustness_score = (
            self.adversarial_robustness + self.noise_tolerance + 
            (1 - self.bias_score) + self.safety_score
        ) / 4
        
        # Calculate weighted overall score
        overall_score = (
            weights["quality"] * quality_score +
            weights["performance"] * performance_score +
            weights["cost_efficiency"] * cost_efficiency_score +
            weights["robustness"] * robustness_score
        )
        
        return min(1.0, max(0.0, overall_score))


# Predefined model configurations for Q3 2025
LATEST_MODEL_CONFIGS = {
    "gpt-5": ModelConfig(
        name="GPT-5",
        provider="openai",
        model_id="gpt-5",
        max_tokens=4000,
        temperature=0.7,
        cost_per_1k_tokens_input=0.03,
        cost_per_1k_tokens_output=0.06,
        context_window=128000,
        parameters=175,
        capabilities=[
            "advanced_reasoning",
            "multimodal_processing",
            "code_generation", 
            "mathematical_reasoning",
            "creative_writing",
            "technical_documentation"
        ],
        deployment_scenarios=["cloud", "edge"],
        performance_profile={
            "reasoning_accuracy": 0.95,
            "multimodal_accuracy": 0.92,
            "context_utilization": 0.88
        },
        license_type="commercial",
        license_details="OpenAI Terms of Service - Commercial use allowed with API subscription",
        commercial_use_allowed=True,
        redistribution_allowed=False,
        modification_allowed=False
    ),
    
    "gpt-5-codex": ModelConfig(
        name="GPT-5-Codex",
        provider="openai", 
        model_id="gpt-5-codex",
        max_tokens=4000,
        temperature=0.3,
        cost_per_1k_tokens_input=0.04,
        cost_per_1k_tokens_output=0.08,
        context_window=128000,
        parameters=175,
        capabilities=[
            "code_generation",
            "software_engineering",
            "debugging",
            "code_review",
            "documentation_generation",
            "technical_analysis"
        ],
        deployment_scenarios=["cloud", "edge"],
        performance_profile={
            "code_success_rate": 0.745,  # 74.5% success rate on real-world coding benchmarks
            "reasoning_accuracy": 0.92,
            "context_utilization": 0.85
        },
        license_type="commercial",
        license_details="OpenAI Terms of Service - Commercial use allowed with API subscription",
        commercial_use_allowed=True,
        redistribution_allowed=False,
        modification_allowed=False
    ),
    
    "claude-3.5-sonnet": ModelConfig(
        name="Claude 3.5 Sonnet",
        provider="anthropic",
        model_id="claude-3-5-sonnet-20241022",
        max_tokens=4000,
        temperature=0.7,
        cost_per_1k_tokens_input=0.003,
        cost_per_1k_tokens_output=0.015,
        context_window=200000,
        parameters=100,
        capabilities=[
            "conversational_ai",
            "reasoning",
            "analysis",
            "creative_writing",
            "code_generation",
            "multimodal_processing"
        ],
        deployment_scenarios=["cloud"],
        performance_profile={
            "reasoning_accuracy": 0.93,
            "multimodal_accuracy": 0.89,
            "context_utilization": 0.91
        },
        license_type="commercial",
        license_details="Anthropic Terms of Service - Commercial use allowed with API subscription",
        commercial_use_allowed=True,
        redistribution_allowed=False,
        modification_allowed=False
    ),
    
    "llama-3.3": ModelConfig(
        name="Llama 3.3",
        provider="meta",
        model_id="llama-3.3-70b",
        max_tokens=2000,
        temperature=0.7,
        cost_per_1k_tokens=0.0,  # Open source
        context_window=128000,
        parameters=70,
        capabilities=[
            "text_generation",
            "conversation",
            "reasoning",
            "code_generation",
            "multilingual_support",
            "multimodal_processing"
        ],
        deployment_scenarios=["cloud", "edge", "on_premises"],
        performance_profile={
            "reasoning_accuracy": 0.87,
            "context_utilization": 0.82,
            "multimodal_accuracy": 0.78
        },
        license_type="open_source",
        license_details="Llama 3.3 Community License - Commercial use allowed with restrictions",
        commercial_use_allowed=True,
        redistribution_allowed=True,
        modification_allowed=True
    ),
    
    # Chinese Models for Lenovo's Hong Kong operations
    "qwen-2.5": ModelConfig(
        name="Qwen 2.5",
        provider="alibaba",
        model_id="qwen-2.5-72b",
        max_tokens=4000,
        temperature=0.7,
        cost_per_1k_tokens_input=0.002,
        cost_per_1k_tokens_output=0.006,
        context_window=128000,
        parameters=72,
        capabilities=[
            "chinese_nlp",
            "multilingual_support",
            "code_generation",
            "multimodal_processing",
            "chinese_code_generation",
            "chinese_multimodal",
            "image_analysis",
            "document_ocr",
            "visual_qa"
        ],
        deployment_scenarios=["cloud", "edge", "on_premises"],
        performance_profile={
            "chinese_nlp_accuracy": 0.94,
            "chinese_code_accuracy": 0.89,
            "chinese_multimodal_accuracy": 0.91,
            "multimodal_accuracy": 0.88,
            "context_utilization": 0.85
        }
    ),
    
    "qwen-vl-plus": ModelConfig(
        name="Qwen-VL Plus",
        provider="alibaba",
        model_id="qwen-vl-plus",
        max_tokens=4000,
        temperature=0.7,
        cost_per_1k_tokens_input=0.003,
        cost_per_1k_tokens_output=0.008,
        context_window=128000,
        parameters=72,
        capabilities=[
            "chinese_multimodal",
            "image_analysis",
            "video_understanding",
            "document_ocr",
            "visual_qa",
            "multimodal_reasoning",
            "chinese_nlp"
        ],
        deployment_scenarios=["cloud", "edge"],
        performance_profile={
            "chinese_multimodal_accuracy": 0.93,
            "image_analysis_accuracy": 0.91,
            "video_understanding_accuracy": 0.87,
            "document_ocr_accuracy": 0.95,
            "visual_qa_accuracy": 0.89,
            "multimodal_reasoning_accuracy": 0.85
        }
    ),
    
    # International Models
    "grok-2": ModelConfig(
        name="Grok-2",
        provider="xai",
        model_id="grok-2",
        max_tokens=4000,
        temperature=0.7,
        cost_per_1k_tokens_input=0.025,
        cost_per_1k_tokens_output=0.05,
        context_window=128000,
        parameters=100,
        capabilities=[
            "advanced_reasoning",
            "multimodal_processing",
            "real_time_processing",
            "humor_generation",
            "creative_writing",
            "code_generation",
            "image_analysis",
            "video_understanding"
        ],
        deployment_scenarios=["cloud"],
        performance_profile={
            "reasoning_accuracy": 0.92,
            "multimodal_accuracy": 0.90,
            "image_analysis_accuracy": 0.88,
            "video_understanding_accuracy": 0.85,
            "context_utilization": 0.89
        }
    ),
    
    "mixtral-8x22b": ModelConfig(
        name="Mixtral 8x22B",
        provider="mistral",
        model_id="mixtral-8x22b",
        max_tokens=4000,
        temperature=0.7,
        cost_per_1k_tokens_input=0.002,
        cost_per_1k_tokens_output=0.006,
        context_window=128000,
        parameters=176,  # 8x22B mixture of experts
        capabilities=[
            "multilingual_support",
            "code_generation",
            "reasoning",
            "multimodal_processing",
            "efficient_inference",
            "image_analysis",
            "document_processing"
        ],
        deployment_scenarios=["cloud", "edge", "on_premises"],
        performance_profile={
            "reasoning_accuracy": 0.90,
            "multimodal_accuracy": 0.86,
            "image_analysis_accuracy": 0.84,
            "context_utilization": 0.88,
            "cost_efficiency_score": 0.95  # Very cost-efficient
        }
    ),
    
    "mixtral-vision": ModelConfig(
        name="Mixtral Vision",
        provider="mistral",
        model_id="mixtral-vision",
        max_tokens=4000,
        temperature=0.7,
        cost_per_1k_tokens_input=0.003,
        cost_per_1k_tokens_output=0.008,
        context_window=128000,
        parameters=176,
        capabilities=[
            "multimodal_processing",
            "image_analysis",
            "video_understanding",
            "document_ocr",
            "visual_qa",
            "multimodal_reasoning",
            "multilingual_support"
        ],
        deployment_scenarios=["cloud", "edge"],
        performance_profile={
            "multimodal_accuracy": 0.89,
            "image_analysis_accuracy": 0.87,
            "video_understanding_accuracy": 0.83,
            "document_ocr_accuracy": 0.92,
            "visual_qa_accuracy": 0.86,
            "multimodal_reasoning_accuracy": 0.84
        }
    ),
    
    # Specialized Code Generation Models (Q3 2025)
    "deepseek-coder-v3": ModelConfig(
        name="DeepSeek Coder V3",
        provider="deepseek",
        model_id="deepseek-coder-v3-33b",
        max_tokens=4000,
        temperature=0.3,
        cost_per_1k_tokens_input=0.001,
        cost_per_1k_tokens_output=0.002,
        context_window=128000,
        parameters=33,
        capabilities=[
            "code_generation",
            "software_engineering",
            "debugging",
            "code_review",
            "mathematical_reasoning",
            "technical_analysis",
            "chinese_code_generation"
        ],
        deployment_scenarios=["cloud", "edge", "on_premises"],
        performance_profile={
            "code_success_rate": 0.78,  # 78% success rate on coding benchmarks
            "reasoning_accuracy": 0.89,
            "chinese_code_accuracy": 0.85,
            "context_utilization": 0.87
        },
        license_type="open_source",
        license_details="Apache 2.0 License - Commercial use allowed with attribution",
        commercial_use_allowed=True,
        redistribution_allowed=True,
        modification_allowed=True
    ),
    
    "codellama-3.1": ModelConfig(
        name="CodeLlama 3.1",
        provider="meta",
        model_id="codellama-3.1-70b",
        max_tokens=4000,
        temperature=0.3,
        cost_per_1k_tokens=0.0,  # Open source
        context_window=128000,
        parameters=70,
        capabilities=[
            "code_generation",
            "software_engineering",
            "debugging",
            "code_completion",
            "documentation_generation",
            "multilingual_code_support"
        ],
        deployment_scenarios=["cloud", "edge", "on_premises"],
        performance_profile={
            "code_success_rate": 0.76,  # 76% success rate on coding benchmarks
            "reasoning_accuracy": 0.85,
            "context_utilization": 0.83
        },
        license_type="open_source",
        license_details="Llama 3.1 Community License - Commercial use allowed with restrictions",
        commercial_use_allowed=True,
        redistribution_allowed=True,
        modification_allowed=True
    ),
    
    "starcoder2-15b": ModelConfig(
        name="StarCoder2 15B",
        provider="huggingface",
        model_id="starcoder2-15b",
        max_tokens=4000,
        temperature=0.3,
        cost_per_1k_tokens=0.0,  # Open source
        context_window=128000,
        parameters=15,
        capabilities=[
            "code_generation",
            "code_completion",
            "software_engineering",
            "debugging",
            "code_explanation",
            "multilingual_programming"
        ],
        deployment_scenarios=["cloud", "edge", "mobile", "on_premises"],
        performance_profile={
            "code_success_rate": 0.72,  # 72% success rate on coding benchmarks
            "reasoning_accuracy": 0.82,
            "context_utilization": 0.80
        },
        license_type="open_source",
        license_details="BigCode OpenRAIL-M License - Commercial use allowed with restrictions",
        commercial_use_allowed=True,
        redistribution_allowed=True,
        modification_allowed=True
    ),
    
    "wizardcoder-34b": ModelConfig(
        name="WizardCoder 34B",
        provider="wizardlm",
        model_id="wizardcoder-34b",
        max_tokens=4000,
        temperature=0.3,
        cost_per_1k_tokens_input=0.0015,
        cost_per_1k_tokens_output=0.003,
        context_window=128000,
        parameters=34,
        capabilities=[
            "code_generation",
            "software_engineering",
            "debugging",
            "code_optimization",
            "algorithm_implementation",
            "technical_documentation"
        ],
        deployment_scenarios=["cloud", "edge", "on_premises"],
        performance_profile={
            "code_success_rate": 0.74,  # 74% success rate on coding benchmarks
            "reasoning_accuracy": 0.86,
            "context_utilization": 0.84
        }
    ),
    
    "phind-codellama-34b": ModelConfig(
        name="Phind CodeLlama 34B",
        provider="phind",
        model_id="phind-codellama-34b",
        max_tokens=4000,
        temperature=0.3,
        cost_per_1k_tokens_input=0.002,
        cost_per_1k_tokens_output=0.004,
        context_window=128000,
        parameters=34,
        capabilities=[
            "code_generation",
            "software_engineering",
            "debugging",
            "code_explanation",
            "technical_analysis",
            "research_assistance"
        ],
        deployment_scenarios=["cloud", "edge"],
        performance_profile={
            "code_success_rate": 0.77,  # 77% success rate on coding benchmarks
            "reasoning_accuracy": 0.88,
            "context_utilization": 0.86
        }
    ),
    
    "magicoder-s-ds-6.7b": ModelConfig(
        name="Magicoder S DS 6.7B",
        provider="ise-uiuc",
        model_id="magicoder-s-ds-6.7b",
        max_tokens=4000,
        temperature=0.3,
        cost_per_1k_tokens=0.0,  # Open source
        context_window=128000,
        parameters=6.7,
        capabilities=[
            "code_generation",
            "software_engineering",
            "debugging",
            "code_completion",
            "algorithm_implementation",
            "efficient_inference"
        ],
        deployment_scenarios=["cloud", "edge", "mobile", "on_premises"],
        performance_profile={
            "code_success_rate": 0.70,  # 70% success rate on coding benchmarks
            "reasoning_accuracy": 0.80,
            "context_utilization": 0.78,
            "cost_efficiency_score": 0.98  # Very cost-efficient
        }
    ),
    
    "qwen-coder-32b": ModelConfig(
        name="Qwen Coder 32B",
        provider="alibaba",
        model_id="qwen-coder-32b",
        max_tokens=4000,
        temperature=0.3,
        cost_per_1k_tokens_input=0.0015,
        cost_per_1k_tokens_output=0.003,
        context_window=128000,
        parameters=32,
        capabilities=[
            "code_generation",
            "software_engineering",
            "debugging",
            "chinese_code_generation",
            "multilingual_programming",
            "technical_analysis"
        ],
        deployment_scenarios=["cloud", "edge", "on_premises"],
        performance_profile={
            "code_success_rate": 0.75,  # 75% success rate on coding benchmarks
            "chinese_code_accuracy": 0.88,
            "reasoning_accuracy": 0.84,
            "context_utilization": 0.82
        }
    ),
    
    # Voice-to-Voice Models (Q3 2025)
    "elevenlabs-voice-ai": ModelConfig(
        name="ElevenLabs Voice AI",
        provider="elevenlabs",
        model_id="elevenlabs-voice-ai-v2",
        max_tokens=2000,
        temperature=0.7,
        cost_per_1k_tokens_input=0.05,
        cost_per_1k_tokens_output=0.10,
        context_window=32000,
        parameters=0,  # API-based service
        capabilities=[
            "voice_to_voice",
            "speech_synthesis",
            "voice_cloning",
            "real_time_conversation",
            "multilingual_voice",
            "emotion_synthesis"
        ],
        deployment_scenarios=["cloud", "edge"],
        performance_profile={
            "voice_to_voice_accuracy": 0.95,
            "speech_synthesis_quality": 0.98,
            "voice_cloning_accuracy": 0.92,
            "real_time_latency_ms": 150,
            "multilingual_voice_accuracy": 0.94,
            "emotion_synthesis_accuracy": 0.89
        }
    ),
    
    "azure-speech-services": ModelConfig(
        name="Azure Speech Services",
        provider="microsoft",
        model_id="azure-speech-services-v2",
        max_tokens=2000,
        temperature=0.7,
        cost_per_1k_tokens_input=0.02,
        cost_per_1k_tokens_output=0.04,
        context_window=32000,
        parameters=0,  # API-based service
        capabilities=[
            "voice_to_voice",
            "speech_synthesis",
            "real_time_conversation",
            "multilingual_voice",
            "emotion_synthesis"
        ],
        deployment_scenarios=["cloud", "edge"],
        performance_profile={
            "voice_to_voice_accuracy": 0.93,
            "speech_synthesis_quality": 0.96,
            "real_time_latency_ms": 200,
            "multilingual_voice_accuracy": 0.91,
            "emotion_synthesis_accuracy": 0.85
        }
    ),
    
    "aws-polly-neural": ModelConfig(
        name="AWS Polly Neural",
        provider="aws",
        model_id="aws-polly-neural-v2",
        max_tokens=2000,
        temperature=0.7,
        cost_per_1k_tokens_input=0.015,
        cost_per_1k_tokens_output=0.03,
        context_window=32000,
        parameters=0,  # API-based service
        capabilities=[
            "voice_to_voice",
            "speech_synthesis",
            "multilingual_voice",
            "emotion_synthesis"
        ],
        deployment_scenarios=["cloud", "edge"],
        performance_profile={
            "voice_to_voice_accuracy": 0.91,
            "speech_synthesis_quality": 0.94,
            "multilingual_voice_accuracy": 0.89,
            "emotion_synthesis_accuracy": 0.82
        }
    ),
    
    "google-text-to-speech-v2": ModelConfig(
        name="Google Text-to-Speech V2",
        provider="google",
        model_id="google-text-to-speech-v2",
        max_tokens=2000,
        temperature=0.7,
        cost_per_1k_tokens_input=0.012,
        cost_per_1k_tokens_output=0.024,
        context_window=32000,
        parameters=0,  # API-based service
        capabilities=[
            "voice_to_voice",
            "speech_synthesis",
            "multilingual_voice",
            "emotion_synthesis"
        ],
        deployment_scenarios=["cloud", "edge"],
        performance_profile={
            "voice_to_voice_accuracy": 0.92,
            "speech_synthesis_quality": 0.95,
            "multilingual_voice_accuracy": 0.90,
            "emotion_synthesis_accuracy": 0.84
        }
    ),
    
    "openai-voice-engine": ModelConfig(
        name="OpenAI Voice Engine",
        provider="openai",
        model_id="openai-voice-engine-v2",
        max_tokens=2000,
        temperature=0.7,
        cost_per_1k_tokens_input=0.03,
        cost_per_1k_tokens_output=0.06,
        context_window=32000,
        parameters=0,  # API-based service
        capabilities=[
            "voice_to_voice",
            "speech_synthesis",
            "real_time_conversation",
            "multilingual_voice"
        ],
        deployment_scenarios=["cloud", "edge"],
        performance_profile={
            "voice_to_voice_accuracy": 0.94,
            "speech_synthesis_quality": 0.97,
            "real_time_latency_ms": 180,
            "multilingual_voice_accuracy": 0.93
        }
    ),
    
    "replica-voice-ai": ModelConfig(
        name="Replica Voice AI",
        provider="replica",
        model_id="replica-voice-ai-v3",
        max_tokens=2000,
        temperature=0.7,
        cost_per_1k_tokens_input=0.04,
        cost_per_1k_tokens_output=0.08,
        context_window=32000,
        parameters=0,  # API-based service
        capabilities=[
            "voice_to_voice",
            "speech_synthesis",
            "voice_cloning",
            "real_time_conversation",
            "emotion_synthesis"
        ],
        deployment_scenarios=["cloud", "edge"],
        performance_profile={
            "voice_to_voice_accuracy": 0.96,
            "speech_synthesis_quality": 0.99,
            "voice_cloning_accuracy": 0.94,
            "real_time_latency_ms": 120,
            "emotion_synthesis_accuracy": 0.91
        }
    ),
    
    "descript-voice-ai": ModelConfig(
        name="Descript Voice AI",
        provider="descript",
        model_id="descript-voice-ai-v2",
        max_tokens=2000,
        temperature=0.7,
        cost_per_1k_tokens_input=0.025,
        cost_per_1k_tokens_output=0.05,
        context_window=32000,
        parameters=0,  # API-based service
        capabilities=[
            "voice_to_voice",
            "speech_synthesis",
            "voice_cloning",
            "real_time_conversation"
        ],
        deployment_scenarios=["cloud", "edge"],
        performance_profile={
            "voice_to_voice_accuracy": 0.93,
            "speech_synthesis_quality": 0.96,
            "voice_cloning_accuracy": 0.90,
            "real_time_latency_ms": 160
        }
    ),
    
    "baidu-deep-voice": ModelConfig(
        name="Baidu Deep Voice",
        provider="baidu",
        model_id="baidu-deep-voice-v3",
        max_tokens=2000,
        temperature=0.7,
        cost_per_1k_tokens_input=0.008,
        cost_per_1k_tokens_output=0.016,
        context_window=32000,
        parameters=0,  # API-based service
        capabilities=[
            "voice_to_voice",
            "speech_synthesis",
            "real_time_conversation",
            "multilingual_voice",
            "chinese_voice_synthesis"
        ],
        deployment_scenarios=["cloud", "edge"],
        performance_profile={
            "voice_to_voice_accuracy": 0.92,
            "speech_synthesis_quality": 0.95,
            "real_time_latency_ms": 140,
            "multilingual_voice_accuracy": 0.88,
            "chinese_voice_accuracy": 0.96
        }
    ),
    
    "google-moshi": ModelConfig(
        name="Google Moshi",
        provider="google",
        model_id="google-moshi-v1",
        max_tokens=2000,
        temperature=0.7,
        cost_per_1k_tokens_input=0.02,
        cost_per_1k_tokens_output=0.04,
        context_window=32000,
        parameters=0,  # API-based service
        capabilities=[
            "voice_to_voice",
            "speech_synthesis",
            "real_time_conversation",
            "multilingual_voice",
            "emotion_synthesis",
            "facial_expression_sync"
        ],
        deployment_scenarios=["cloud", "edge"],
        performance_profile={
            "voice_to_voice_accuracy": 0.97,
            "speech_synthesis_quality": 0.99,
            "real_time_latency_ms": 100,
            "multilingual_voice_accuracy": 0.95,
            "emotion_synthesis_accuracy": 0.93,
            "facial_expression_sync_accuracy": 0.91
        },
        license_type="commercial",
        license_details="Google Cloud Terms of Service - Commercial use allowed with API subscription",
        commercial_use_allowed=True,
        redistribution_allowed=False,
        modification_allowed=False
    )
}


def get_model_config(model_name: str) -> ModelConfig:
    """
    Get predefined model configuration by name.
    
    Args:
        model_name: Name of the model (e.g., 'gpt-5', 'claude-3.5-sonnet')
    
    Returns:
        ModelConfig instance for the specified model
        
    Raises:
        KeyError: If model_name is not found in predefined configurations
    """
    if model_name not in LATEST_MODEL_CONFIGS:
        available_models = list(LATEST_MODEL_CONFIGS.keys())
        raise KeyError(f"Model '{model_name}' not found. Available models: {available_models}")
    
    return LATEST_MODEL_CONFIGS[model_name]


def list_available_models() -> List[str]:
    """Get list of all available predefined model configurations"""
    return list(LATEST_MODEL_CONFIGS.keys())


def get_models_by_license_type(license_type: str) -> List[str]:
    """
    Get list of models filtered by license type.
    
    Args:
        license_type: Type of license to filter by (e.g., 'open_source', 'commercial', 'research_only')
    
    Returns:
        List of model names that match the specified license type
    """
    return [
        model_name for model_name, config in LATEST_MODEL_CONFIGS.items()
        if config.license_type == license_type
    ]


def get_models_for_commercial_use() -> List[str]:
    """
    Get list of models that allow commercial use.
    
    Returns:
        List of model names that allow commercial use
    """
    return [
        model_name for model_name, config in LATEST_MODEL_CONFIGS.items()
        if config.commercial_use_allowed
    ]


def get_models_for_redistribution() -> List[str]:
    """
    Get list of models that allow redistribution.
    
    Returns:
        List of model names that allow redistribution
    """
    return [
        model_name for model_name, config in LATEST_MODEL_CONFIGS.items()
        if config.redistribution_allowed
    ]


def get_models_for_modification() -> List[str]:
    """
    Get list of models that allow modification.
    
    Returns:
        List of model names that allow modification
    """
    return [
        model_name for model_name, config in LATEST_MODEL_CONFIGS.items()
        if config.modification_allowed
    ]


def get_license_summary() -> Dict[str, Dict[str, Any]]:
    """
    Get a summary of all license types and their characteristics.
    
    Returns:
        Dictionary mapping license types to their characteristics
    """
    license_summary = {}
    
    for model_name, config in LATEST_MODEL_CONFIGS.items():
        license_type = config.license_type
        if license_type not in license_summary:
            license_summary[license_type] = {
                "models": [],
                "commercial_use_allowed": config.commercial_use_allowed,
                "redistribution_allowed": config.redistribution_allowed,
                "modification_allowed": config.modification_allowed,
                "example_license": config.license_details
            }
        license_summary[license_type]["models"].append(model_name)
    
    return license_summary
