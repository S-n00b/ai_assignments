"""
Model Evaluator for GitHub Models Backend

Provides comprehensive model evaluation capabilities using GitHub Models API.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json

from .github_models_client import GitHubModelsClient, EvaluationRequest, EvaluationResult, ModelInfo
from .cache_manager import cache_manager
from .rate_limiter import rate_limiter

logger = logging.getLogger(__name__)


class EvaluationMetric(Enum):
    """Available evaluation metrics."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    QUALITY = "quality"
    COST_EFFICIENCY = "cost_efficiency"
    ROBUSTNESS = "robustness"
    BIAS_DETECTION = "bias_detection"


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    models: List[str]
    tasks: List[str]
    metrics: List[EvaluationMetric]
    parameters: Dict[str, Any] = field(default_factory=dict)
    use_cache: bool = True
    max_concurrent: int = 3
    timeout_seconds: int = 300


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report."""
    config: EvaluationConfig
    results: List[EvaluationResult]
    summary: Dict[str, Any]
    timestamp: float
    duration_seconds: float


class ModelEvaluator:
    """
    Comprehensive model evaluator using GitHub Models API.
    
    Features:
    - Multi-model evaluation
    - Multiple task types
    - Comprehensive metrics
    - Caching and rate limiting
    - Async evaluation
    """
    
    def __init__(self):
        self.client = GitHubModelsClient()
        self.evaluation_tasks = self._initialize_evaluation_tasks()
        self.evaluation_metrics = self._initialize_evaluation_metrics()
    
    def _initialize_evaluation_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Initialize predefined evaluation tasks."""
        return {
            "text_generation": {
                "description": "Generate text based on prompts",
                "test_cases": [
                    "Write a short story about a robot learning to paint.",
                    "Explain the concept of machine learning in simple terms.",
                    "Create a poem about the future of AI."
                ],
                "evaluation_criteria": ["creativity", "coherence", "relevance"]
            },
            "question_answering": {
                "description": "Answer questions based on context",
                "test_cases": [
                    {
                        "context": "The capital of France is Paris, a city known for its art, culture, and the Eiffel Tower.",
                        "question": "What is the capital of France?"
                    },
                    {
                        "context": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
                        "question": "What is machine learning?"
                    }
                ],
                "evaluation_criteria": ["accuracy", "completeness", "clarity"]
            },
            "summarization": {
                "description": "Summarize long texts",
                "test_cases": [
                    "Summarize this article about climate change and its impact on global weather patterns...",
                    "Provide a brief summary of the key points in this research paper about neural networks..."
                ],
                "evaluation_criteria": ["conciseness", "accuracy", "completeness"]
            },
            "code_generation": {
                "description": "Generate code based on specifications",
                "test_cases": [
                    "Write a Python function to calculate the factorial of a number.",
                    "Create a JavaScript function to sort an array of numbers.",
                    "Write a SQL query to find the top 10 customers by total purchases."
                ],
                "evaluation_criteria": ["correctness", "efficiency", "readability"]
            },
            "reasoning": {
                "description": "Solve logical reasoning problems",
                "test_cases": [
                    "If all birds can fly and penguins are birds, can penguins fly? Explain your reasoning.",
                    "A farmer has 17 sheep. All but 9 die. How many sheep are left?",
                    "What comes next in the sequence: 2, 4, 8, 16, ?"
                ],
                "evaluation_criteria": ["logical_consistency", "accuracy", "explanation_quality"]
            }
        }
    
    def _initialize_evaluation_metrics(self) -> Dict[EvaluationMetric, Dict[str, Any]]:
        """Initialize evaluation metrics and their calculations."""
        return {
            EvaluationMetric.LATENCY: {
                "description": "Response time in milliseconds",
                "calculation": "latency_ms",
                "weight": 0.2
            },
            EvaluationMetric.THROUGHPUT: {
                "description": "Tokens generated per second",
                "calculation": "tokens_per_second",
                "weight": 0.15
            },
            EvaluationMetric.QUALITY: {
                "description": "Overall response quality score",
                "calculation": "quality_score",
                "weight": 0.4
            },
            EvaluationMetric.COST_EFFICIENCY: {
                "description": "Cost per token ratio",
                "calculation": "cost_per_token",
                "weight": 0.15
            },
            EvaluationMetric.ROBUSTNESS: {
                "description": "Consistency across multiple runs",
                "calculation": "robustness_score",
                "weight": 0.1
            }
        }
    
    async def evaluate_single_model(self, model_id: str, task_type: str, 
                                  test_cases: List[Any], parameters: Optional[Dict[str, Any]] = None) -> List[EvaluationResult]:
        """
        Evaluate a single model on a specific task.
        
        Args:
            model_id: Model to evaluate
            task_type: Type of task to evaluate
            test_cases: List of test cases
            parameters: Optional model parameters
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for i, test_case in enumerate(test_cases):
            try:
                # Prepare input data based on task type
                if task_type == "question_answering" and isinstance(test_case, dict):
                    input_data = [
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                        {"role": "user", "content": f"Context: {test_case['context']}\n\nQuestion: {test_case['question']}"}
                    ]
                else:
                    input_data = str(test_case)
                
                # Create evaluation request
                request = EvaluationRequest(
                    model_id=model_id,
                    task_type=task_type,
                    input_data=input_data,
                    parameters=parameters or {}
                )
                
                # Check cache first
                if cache_manager.get(model_id, input_data if isinstance(input_data, list) else [input_data], parameters or {}):
                    logger.debug(f"Using cached result for {model_id} - {task_type} - test case {i+1}")
                
                # Evaluate model
                result = await self.client.evaluate_model(request)
                results.append(result)
                
                # Cache the result
                cache_manager.set(model_id, input_data if isinstance(input_data, list) else [input_data], parameters or {}, result)
                
                logger.info(f"Completed evaluation: {model_id} - {task_type} - test case {i+1}/{len(test_cases)}")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_id} on {task_type} test case {i+1}: {e}")
                # Create error result
                error_result = EvaluationResult(
                    model_id=model_id,
                    task_type=task_type,
                    input_data=test_case,
                    output={"error": str(e)},
                    metrics={"error": True},
                    latency_ms=0,
                    timestamp=time.time(),
                    provider=self.client.get_model_info(model_id).provider if self.client.get_model_info(model_id) else None
                )
                results.append(error_result)
        
        return results
    
    async def evaluate_models(self, config: EvaluationConfig) -> EvaluationReport:
        """
        Evaluate multiple models with comprehensive metrics.
        
        Args:
            config: Evaluation configuration
            
        Returns:
            Comprehensive evaluation report
        """
        start_time = time.time()
        all_results = []
        
        logger.info(f"Starting evaluation of {len(config.models)} models on {len(config.tasks)} tasks")
        
        # Create semaphore for concurrent requests
        semaphore = asyncio.Semaphore(config.max_concurrent)
        
        async def evaluate_model_task(model_id: str, task_type: str):
            async with semaphore:
                if task_type in self.evaluation_tasks:
                    test_cases = self.evaluation_tasks[task_type]["test_cases"]
                    return await self.evaluate_single_model(model_id, task_type, test_cases, config.parameters)
                else:
                    logger.warning(f"Unknown task type: {task_type}")
                    return []
        
        # Create all evaluation tasks
        tasks = []
        for model_id in config.models:
            for task_type in config.tasks:
                tasks.append(evaluate_model_task(model_id, task_type))
        
        # Execute all evaluations concurrently
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=config.timeout_seconds
            )
            
            # Flatten results
            for result in results:
                if isinstance(result, list):
                    all_results.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Evaluation task failed: {result}")
            
        except asyncio.TimeoutError:
            logger.error(f"Evaluation timed out after {config.timeout_seconds} seconds")
        
        # Generate summary
        summary = self._generate_summary(all_results, config)
        
        duration = time.time() - start_time
        
        return EvaluationReport(
            config=config,
            results=all_results,
            summary=summary,
            timestamp=start_time,
            duration_seconds=duration
        )
    
    def _generate_summary(self, results: List[EvaluationResult], config: EvaluationConfig) -> Dict[str, Any]:
        """Generate evaluation summary."""
        if not results:
            return {"error": "No results to summarize"}
        
        # Group results by model and task
        model_scores = {}
        task_scores = {}
        
        for result in results:
            if result.model_id not in model_scores:
                model_scores[result.model_id] = []
            if result.task_type not in task_scores:
                task_scores[result.task_type] = []
            
            # Calculate overall score (simplified)
            if "error" not in result.metrics:
                score = self._calculate_overall_score(result)
                model_scores[result.model_id].append(score)
                task_scores[result.task_type].append(score)
        
        # Calculate averages
        model_averages = {
            model_id: sum(scores) / len(scores) if scores else 0
            for model_id, scores in model_scores.items()
        }
        
        task_averages = {
            task_type: sum(scores) / len(scores) if scores else 0
            for task_type, scores in task_scores.items()
        }
        
        # Find best model
        best_model = max(model_averages.items(), key=lambda x: x[1]) if model_averages else None
        
        return {
            "total_evaluations": len(results),
            "successful_evaluations": len([r for r in results if "error" not in r.metrics]),
            "failed_evaluations": len([r for r in results if "error" in r.metrics]),
            "model_scores": model_averages,
            "task_scores": task_averages,
            "best_model": best_model,
            "average_latency": sum(r.latency_ms for r in results if "error" not in r.metrics) / len([r for r in results if "error" not in r.metrics]) if results else 0,
            "cache_stats": cache_manager.get_stats(),
            "rate_limit_stats": rate_limiter.get_status("github_models")
        }
    
    def _calculate_overall_score(self, result: EvaluationResult) -> float:
        """Calculate overall score for a result."""
        # Simplified scoring based on available metrics
        score = 0.0
        
        # Latency score (lower is better)
        if "latency_ms" in result.metrics:
            latency_score = max(0, 1 - (result.latency_ms / 10000))  # Normalize to 0-1
            score += latency_score * 0.3
        
        # Throughput score (higher is better)
        if "tokens_per_second" in result.metrics:
            throughput_score = min(1, result.metrics["tokens_per_second"] / 100)  # Normalize to 0-1
            score += throughput_score * 0.2
        
        # Output quality (simplified - based on length and content)
        if isinstance(result.output, str) and len(result.output) > 10:
            quality_score = min(1, len(result.output) / 1000)  # Normalize to 0-1
            score += quality_score * 0.5
        
        return min(1.0, score)
    
    def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models."""
        return self.client.get_available_models()
    
    def get_available_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available evaluation tasks."""
        return self.evaluation_tasks
    
    def get_evaluation_metrics(self) -> Dict[EvaluationMetric, Dict[str, Any]]:
        """Get available evaluation metrics."""
        return self.evaluation_metrics


# Global evaluator instance
model_evaluator = ModelEvaluator()

