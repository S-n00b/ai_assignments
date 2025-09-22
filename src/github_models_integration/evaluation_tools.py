"""
GitHub Models Evaluation Tools

This module provides evaluation tools using GitHub Models API for
remote model evaluation and comparison.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime

from .api_client import GitHubModelsAPIClient, GitHubModel

logger = logging.getLogger(__name__)


@dataclass
class EvaluationRequest:
    """Request for model evaluation."""
    model_id: str
    task_type: str
    input_data: Union[str, List[str], Dict[str, Any]]
    parameters: Optional[Dict[str, Any]] = None
    evaluation_metrics: Optional[List[str]] = None


@dataclass
class EvaluationResult:
    """Result of model evaluation."""
    model_id: str
    task_type: str
    input_data: Union[str, List[str], Dict[str, Any]]
    output: Union[str, List[str], Dict[str, Any]]
    metrics: Dict[str, float]
    latency_ms: float
    timestamp: float
    provider: str
    success: bool = True
    error: Optional[str] = None


@dataclass
class ComparisonResult:
    """Result of model comparison."""
    model_results: List[EvaluationResult]
    best_model: Optional[str] = None
    comparison_metrics: Dict[str, Any] = None
    summary: str = ""


class GitHubModelsEvaluator:
    """Evaluation tools using GitHub Models API."""
    
    def __init__(self, api_client: GitHubModelsAPIClient):
        """Initialize the GitHub Models evaluator."""
        self.api_client = api_client
        self.evaluation_history = []
        
    async def run_evaluation(
        self,
        model_id: str,
        evaluation_config: Dict[str, Any]
    ) -> EvaluationResult:
        """
        Run evaluation using GitHub Models API.
        
        Args:
            model_id: Model ID to evaluate
            evaluation_config: Evaluation configuration
            
        Returns:
            EvaluationResult with model output and metrics
        """
        start_time = time.time()
        
        try:
            logger.info(f"Running evaluation for model: {model_id}")
            
            # Get model information
            model_info = await self.api_client.get_model_details(model_id)
            if not model_info:
                raise ValueError(f"Model {model_id} not found")
            
            # Prepare input data for GitHub Models API format
            messages = self._prepare_messages(evaluation_config)
            
            # Prepare parameters
            parameters = evaluation_config.get("parameters", {})
            
            # Call GitHub Models API
            request_data = {
                "messages": messages,
                "parameters": parameters
            }
            
            response = await self.api_client.evaluate_model(model_id, request_data)
            
            # Process response
            if response.get("success"):
                output = self._extract_output(response["response"])
                success = True
                error = None
            else:
                output = {"error": response.get("error", "Unknown error")}
                success = False
                error = response.get("error", "Unknown error")
            
            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000
            input_text = self._extract_input_text(messages)
            metrics = self._calculate_metrics(input_text, output, latency_ms, model_info)
            
            result = EvaluationResult(
                model_id=model_id,
                task_type=evaluation_config.get("task_type", "general"),
                input_data=evaluation_config.get("input_data", ""),
                output=output,
                metrics=metrics,
                latency_ms=latency_ms,
                timestamp=time.time(),
                provider=model_info.provider,
                success=success,
                error=error
            )
            
            # Record evaluation history
            self.evaluation_history.append(result)
            
            logger.info(f"Evaluation completed for {model_id} in {latency_ms:.2f}ms")
            
            return result
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error_msg = f"Evaluation failed for {model_id}: {str(e)}"
            logger.error(error_msg)
            
            return EvaluationResult(
                model_id=model_id,
                task_type=evaluation_config.get("task_type", "general"),
                input_data=evaluation_config.get("input_data", ""),
                output={"error": error_msg},
                metrics={"latency_ms": latency_ms},
                latency_ms=latency_ms,
                timestamp=time.time(),
                provider="unknown",
                success=False,
                error=error_msg
            )
    
    async def batch_evaluate(
        self,
        model_ids: List[str],
        evaluation_config: Dict[str, Any]
    ) -> List[EvaluationResult]:
        """
        Run batch evaluation for multiple models.
        
        Args:
            model_ids: List of model IDs to evaluate
            evaluation_config: Evaluation configuration
            
        Returns:
            List of EvaluationResult objects
        """
        logger.info(f"Running batch evaluation for {len(model_ids)} models")
        
        # Create tasks for concurrent evaluation
        tasks = [self.run_evaluation(model_id, evaluation_config) for model_id in model_ids]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(EvaluationResult(
                    model_id=model_ids[i],
                    task_type=evaluation_config.get("task_type", "general"),
                    input_data=evaluation_config.get("input_data", ""),
                    output={"error": str(result)},
                    metrics={"latency_ms": 0.0},
                    latency_ms=0.0,
                    timestamp=time.time(),
                    provider="unknown",
                    success=False,
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        successful_evaluations = sum(1 for r in processed_results if r.success)
        logger.info(f"Batch evaluation completed: {successful_evaluations}/{len(model_ids)} successful")
        
        return processed_results
    
    async def compare_models(
        self,
        model_ids: List[str],
        test_data: List[Dict[str, Any]]
    ) -> ComparisonResult:
        """
        Compare multiple models using GitHub Models API.
        
        Args:
            model_ids: List of model IDs to compare
            test_data: List of test cases
            
        Returns:
            ComparisonResult with comparison analysis
        """
        logger.info(f"Comparing {len(model_ids)} models with {len(test_data)} test cases")
        
        all_results = []
        
        # Run evaluation for each model and test case
        for test_case in test_data:
            evaluation_config = {
                "task_type": test_case.get("task_type", "general"),
                "input_data": test_case.get("input_data", ""),
                "parameters": test_case.get("parameters", {})
            }
            
            batch_results = await self.batch_evaluate(model_ids, evaluation_config)
            all_results.extend(batch_results)
        
        # Analyze results
        comparison_metrics = self._analyze_comparison_results(all_results, model_ids)
        
        # Determine best model
        best_model = self._determine_best_model(all_results, model_ids)
        
        # Generate summary
        summary = self._generate_comparison_summary(all_results, model_ids, best_model)
        
        result = ComparisonResult(
            model_results=all_results,
            best_model=best_model,
            comparison_metrics=comparison_metrics,
            summary=summary
        )
        
        logger.info(f"Model comparison completed. Best model: {best_model}")
        
        return result
    
    def _prepare_messages(self, evaluation_config: Dict[str, Any]) -> List[Dict[str, str]]:
        """Prepare messages for GitHub Models API."""
        messages = []
        
        # Add system message for task context
        task_type = evaluation_config.get("task_type", "general")
        system_message = {
            "role": "system",
            "content": f"You are an AI assistant helping with {task_type}. Please provide a helpful response."
        }
        messages.append(system_message)
        
        # Add user input
        input_data = evaluation_config.get("input_data", "")
        if isinstance(input_data, dict):
            if "role" in input_data and "content" in input_data:
                messages.append(input_data)
            else:
                messages.append({"role": "user", "content": str(input_data)})
        elif isinstance(input_data, list):
            if len(input_data) > 0 and isinstance(input_data[0], dict):
                messages.extend(input_data)
            else:
                content = "\n".join(str(item) for item in input_data)
                messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": str(input_data)})
        
        return messages
    
    def _extract_output(self, response: Dict[str, Any]) -> Union[str, List[str], Dict[str, Any]]:
        """Extract output from GitHub Models API response."""
        # GitHub Models API returns responses in OpenAI-compatible format
        if "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
            elif "text" in choice:
                return choice["text"]
        
        # Fallback to full response if structure is unexpected
        return response
    
    def _extract_input_text(self, messages: List[Dict[str, str]]) -> str:
        """Extract input text from messages."""
        return " ".join(msg.get("content", "") for msg in messages)
    
    def _calculate_metrics(
        self,
        inputs: str,
        output: Union[str, List[str], Dict[str, Any]],
        latency_ms: float,
        model_info: GitHubModel
    ) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {
            "latency_ms": latency_ms,
            "input_length": len(inputs),
            "output_length": len(str(output)),
            "tokens_per_second": len(str(output)) / (latency_ms / 1000) if latency_ms > 0 else 0
        }
        
        # Add provider-specific metrics
        metrics["provider"] = hash(model_info.provider) % 1000  # Simple hash for numeric value
        
        # Add model-specific metrics
        if model_info.max_tokens:
            metrics["max_tokens"] = model_info.max_tokens
        
        if model_info.cost_per_token:
            metrics["estimated_cost"] = len(str(output)) * model_info.cost_per_token
        
        return metrics
    
    def _analyze_comparison_results(
        self,
        results: List[EvaluationResult],
        model_ids: List[str]
    ) -> Dict[str, Any]:
        """Analyze comparison results."""
        metrics = {
            "total_evaluations": len(results),
            "successful_evaluations": sum(1 for r in results if r.success),
            "model_performance": {},
            "average_metrics": {}
        }
        
        # Group results by model
        model_results = {model_id: [] for model_id in model_ids}
        for result in results:
            if result.model_id in model_results:
                model_results[result.model_id].append(result)
        
        # Calculate performance metrics for each model
        for model_id, model_result_list in model_results.items():
            if not model_result_list:
                continue
                
            successful_results = [r for r in model_result_list if r.success]
            
            if successful_results:
                avg_latency = sum(r.latency_ms for r in successful_results) / len(successful_results)
                avg_output_length = sum(r.metrics.get("output_length", 0) for r in successful_results) / len(successful_results)
                success_rate = len(successful_results) / len(model_result_list)
                
                metrics["model_performance"][model_id] = {
                    "success_rate": success_rate,
                    "average_latency_ms": avg_latency,
                    "average_output_length": avg_output_length,
                    "total_evaluations": len(model_result_list),
                    "successful_evaluations": len(successful_results)
                }
        
        return metrics
    
    def _determine_best_model(
        self,
        results: List[EvaluationResult],
        model_ids: List[str]
    ) -> Optional[str]:
        """Determine the best model based on evaluation results."""
        if not results:
            return None
        
        # Group results by model
        model_results = {model_id: [] for model_id in model_ids}
        for result in results:
            if result.model_id in model_results:
                model_results[result.model_id].append(result)
        
        # Calculate scores for each model
        model_scores = {}
        for model_id, model_result_list in model_results.items():
            if not model_result_list:
                continue
                
            successful_results = [r for r in model_result_list if r.success]
            
            if successful_results:
                success_rate = len(successful_results) / len(model_result_list)
                avg_latency = sum(r.latency_ms for r in successful_results) / len(successful_results)
                
                # Score based on success rate and latency (lower is better for latency)
                score = success_rate * (1.0 / (1.0 + avg_latency / 1000.0))  # Normalize latency
                model_scores[model_id] = score
        
        # Return model with highest score
        if model_scores:
            return max(model_scores, key=model_scores.get)
        
        return None
    
    def _generate_comparison_summary(
        self,
        results: List[EvaluationResult],
        model_ids: List[str],
        best_model: Optional[str]
    ) -> str:
        """Generate comparison summary."""
        if not results:
            return "No evaluation results available."
        
        successful_results = [r for r in results if r.success]
        success_rate = len(successful_results) / len(results)
        
        summary = f"Model Comparison Summary:\n"
        summary += f"Total evaluations: {len(results)}\n"
        summary += f"Success rate: {success_rate:.2%}\n"
        
        if best_model:
            summary += f"Best performing model: {best_model}\n"
        
        summary += f"Models compared: {', '.join(model_ids)}\n"
        
        return summary
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get evaluation statistics."""
        if not self.evaluation_history:
            return {
                "total_evaluations": 0,
                "successful_evaluations": 0,
                "failed_evaluations": 0,
                "average_latency_ms": 0.0,
                "models_evaluated": set()
            }
        
        successful_evaluations = [r for r in self.evaluation_history if r.success]
        failed_evaluations = [r for r in self.evaluation_history if not r.success]
        
        avg_latency = 0.0
        if successful_evaluations:
            avg_latency = sum(r.latency_ms for r in successful_evaluations) / len(successful_evaluations)
        
        models_evaluated = set(r.model_id for r in self.evaluation_history)
        
        return {
            "total_evaluations": len(self.evaluation_history),
            "successful_evaluations": len(successful_evaluations),
            "failed_evaluations": len(failed_evaluations),
            "average_latency_ms": avg_latency,
            "models_evaluated": list(models_evaluated),
            "success_rate": len(successful_evaluations) / len(self.evaluation_history) if self.evaluation_history else 0.0
        }
    
    def clear_evaluation_history(self):
        """Clear evaluation history."""
        self.evaluation_history.clear()
        logger.info("Evaluation history cleared")
