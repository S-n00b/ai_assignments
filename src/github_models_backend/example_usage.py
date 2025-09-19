"""
Example usage of GitHub Models Backend

This example demonstrates how to use the GitHub Models backend
for model evaluation and inference in your applications.
"""

import asyncio
import logging
from typing import List

from .github_models_client import GitHubModelsClient, EvaluationRequest
from .model_evaluator import ModelEvaluator, EvaluationConfig, EvaluationMetric
from .rate_limiter import rate_limiter
from .cache_manager import cache_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_single_model_evaluation():
    """Example: Evaluate a single model on a specific task."""
    print("=== Single Model Evaluation Example ===")
    
    async with GitHubModelsClient() as client:
        # Get available models
        models = client.get_available_models()
        print(f"Available models: {[model.name for model in models]}")
        
        # Create evaluation request
        request = EvaluationRequest(
            model_id="openai/gpt-4o-mini",
            task_type="text_generation",
            input_data="Write a short poem about artificial intelligence.",
            parameters={
                "temperature": 0.7,
                "max_tokens": 200
            }
        )
        
        # Evaluate model
        result = await client.evaluate_model(request)
        
        print(f"Model: {result.model_id}")
        print(f"Task: {result.task_type}")
        print(f"Output: {result.output}")
        print(f"Latency: {result.latency_ms:.2f}ms")
        print(f"Metrics: {result.metrics}")


async def example_batch_evaluation():
    """Example: Evaluate multiple models on multiple tasks."""
    print("\n=== Batch Evaluation Example ===")
    
    evaluator = ModelEvaluator()
    
    # Create evaluation configuration
    config = EvaluationConfig(
        models=["openai/gpt-4o-mini", "meta/llama-3.1-8b"],
        tasks=["text_generation", "question_answering"],
        metrics=[EvaluationMetric.LATENCY, EvaluationMetric.QUALITY],
        parameters={
            "temperature": 0.5,
            "max_tokens": 150
        },
        use_cache=True,
        max_concurrent=2
    )
    
    # Run evaluation
    report = await evaluator.evaluate_models(config)
    
    print(f"Evaluation completed in {report.duration_seconds:.2f} seconds")
    print(f"Total evaluations: {report.summary['total_evaluations']}")
    print(f"Successful: {report.summary['successful_evaluations']}")
    print(f"Failed: {report.summary['failed_evaluations']}")
    
    if report.summary.get('best_model'):
        best_model, score = report.summary['best_model']
        print(f"Best model: {best_model} (score: {score:.3f})")
    
    print(f"Average latency: {report.summary['average_latency']:.2f}ms")


async def example_rate_limiting():
    """Example: Demonstrate rate limiting functionality."""
    print("\n=== Rate Limiting Example ===")
    
    # Check current rate limit status
    status = rate_limiter.get_status("github_models")
    print(f"Rate limit status: {status}")
    
    # Simulate multiple requests
    for i in range(5):
        wait_time = rate_limiter.wait_if_needed("github_models")
        if wait_time > 0:
            print(f"Request {i+1}: Waited {wait_time:.2f} seconds")
        else:
            print(f"Request {i+1}: No wait needed")
        
        rate_limiter.record_request("github_models")
    
    # Check final status
    final_status = rate_limiter.get_status("github_models")
    print(f"Final rate limit status: {final_status}")


async def example_caching():
    """Example: Demonstrate caching functionality."""
    print("\n=== Caching Example ===")
    
    # Check cache stats
    stats = cache_manager.get_stats()
    print(f"Initial cache stats: {stats}")
    
    # Simulate cache operations
    test_data = {"test": "data", "value": 123}
    cache_manager.set("test_model", [{"role": "user", "content": "test"}], {}, test_data, ttl_seconds=60)
    
    # Retrieve from cache
    cached_data = cache_manager.get("test_model", [{"role": "user", "content": "test"}], {})
    print(f"Cached data retrieved: {cached_data}")
    
    # Check final cache stats
    final_stats = cache_manager.get_stats()
    print(f"Final cache stats: {final_stats}")


async def example_github_models_integration():
    """Example: Complete GitHub Models integration workflow."""
    print("\n=== Complete GitHub Models Integration Example ===")
    
    try:
        # Initialize components
        client = GitHubModelsClient()
        evaluator = ModelEvaluator()
        
        # Check if GitHub token is available
        token = client._get_github_token()
        if token == "demo_token":
            print("⚠️  No GitHub token found. Using demo mode with limited functionality.")
            print("   To use full GitHub Models API:")
            print("   1. Set GITHUB_TOKEN environment variable, or")
            print("   2. Run 'gh auth login' to authenticate with GitHub CLI")
        else:
            print("✅ GitHub token found. Full API access available.")
        
        # Get available models and tasks
        models = evaluator.get_available_models()
        tasks = evaluator.get_available_tasks()
        
        print(f"Available models: {len(models)}")
        print(f"Available tasks: {list(tasks.keys())}")
        
        # Example: Quick model comparison
        if models and token != "demo_token":
            print("\nRunning quick model comparison...")
            
            config = EvaluationConfig(
                models=[models[0].id, models[1].id] if len(models) >= 2 else [models[0].id],
                tasks=["text_generation"],
                metrics=[EvaluationMetric.LATENCY, EvaluationMetric.QUALITY],
                parameters={"max_tokens": 100},
                max_concurrent=1
            )
            
            report = await evaluator.evaluate_models(config)
            
            print(f"Comparison completed:")
            print(f"- Models tested: {len(config.models)}")
            print(f"- Tasks: {len(config.tasks)}")
            print(f"- Total time: {report.duration_seconds:.2f}s")
            
            if report.summary.get('model_scores'):
                print("Model scores:")
                for model, score in report.summary['model_scores'].items():
                    print(f"  - {model}: {score:.3f}")
        
        # Show rate limiting and caching stats
        print(f"\nRate limiting stats: {rate_limiter.get_status('github_models')}")
        print(f"Cache stats: {cache_manager.get_stats()}")
        
    except Exception as e:
        logger.error(f"Error in GitHub Models integration: {e}")
        print(f"❌ Error: {e}")


async def main():
    """Run all examples."""
    print("GitHub Models Backend - Usage Examples")
    print("=" * 50)
    
    # Run examples
    await example_single_model_evaluation()
    await example_batch_evaluation()
    await example_rate_limiting()
    await example_caching()
    await example_github_models_integration()
    
    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())

