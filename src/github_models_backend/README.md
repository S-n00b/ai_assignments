# GitHub Models Backend Integration

This module provides a comprehensive backend integration with the official [GitHub Models API](https://docs.github.com/en/rest/models/inference?apiVersion=2022-11-28) for model evaluation, prototyping, and monitoring.

## Features

- **Real GitHub Models API Integration**: Uses the official GitHub Models API endpoints
- **Rate Limiting**: Built-in rate limiting for public showcase applications
- **Caching**: Intelligent caching to reduce API calls and costs
- **Multi-Model Evaluation**: Evaluate multiple models across different tasks
- **Comprehensive Metrics**: Latency, throughput, quality, and cost efficiency metrics
- **No API Keys Required**: Uses GitHub credentials (token or CLI authentication)

## Quick Start

### 1. Authentication Setup

**⚠️ Important**: GitHub Models API requires a Personal Access Token (PAT) with the `models` scope.

#### Option 1: Automated Setup (Recommended)

Run the setup script to guide you through the process:

```bash
# Python setup script
python scripts/setup-github-models.py

# PowerShell setup script (Windows)
.\scripts\setup-github-models.ps1
```

#### Option 2: Manual Setup

**Create a Personal Access Token:**

1. Go to [GitHub Settings > Personal Access Tokens](https://github.com/settings/tokens)
2. Click "Generate new token (classic)"
3. Name: "Lenovo AAITC Models API"
4. **Select scopes**: ✅ `models` (required for GitHub Models API)
5. Set expiration: 90 days (recommended)
6. Click "Generate token"
7. **Copy the token immediately** (it won't be shown again!)

**Set the token:**

```bash
# Linux/Mac
export GITHUB_TOKEN="your_token_here"

# Windows PowerShell
$env:GITHUB_TOKEN = "your_token_here"

# Windows Command Prompt
set GITHUB_TOKEN=your_token_here
```

#### Option 3: GitHub CLI (Easiest)

```bash
# Install GitHub CLI if needed
# Then authenticate
gh auth login

# This automatically sets up the token with correct scopes
```

#### Option 4: Demo Mode (Limited Functionality)

For testing without real API calls:

```bash
export GITHUB_TOKEN="demo_token"
```

### 2. Basic Usage

```python
import asyncio
from src.github_models_backend import GitHubModelsClient, EvaluationRequest

async def main():
    async with GitHubModelsClient() as client:
        # Get available models
        models = client.get_available_models()
        print(f"Available models: {[model.name for model in models]}")

        # Evaluate a model
        request = EvaluationRequest(
            model_id="openai/gpt-4o-mini",
            task_type="text_generation",
            input_data="Write a short poem about AI.",
            parameters={"temperature": 0.7, "max_tokens": 200}
        )

        result = await client.evaluate_model(request)
        print(f"Output: {result.output}")
        print(f"Latency: {result.latency_ms:.2f}ms")

asyncio.run(main())
```

### 3. Batch Evaluation

```python
from src.github_models_backend import ModelEvaluator, EvaluationConfig, EvaluationMetric

async def batch_evaluation():
    evaluator = ModelEvaluator()

    config = EvaluationConfig(
        models=["openai/gpt-4o-mini", "meta/llama-3.1-8b"],
        tasks=["text_generation", "question_answering"],
        metrics=[EvaluationMetric.LATENCY, EvaluationMetric.QUALITY],
        parameters={"temperature": 0.5, "max_tokens": 150}
    )

    report = await evaluator.evaluate_models(config)

    print(f"Best model: {report.summary['best_model']}")
    print(f"Average latency: {report.summary['average_latency']:.2f}ms")

asyncio.run(batch_evaluation())
```

## Available Models

The backend supports models from multiple providers via GitHub Models:

### OpenAI Models

- `openai/gpt-4.1` - Latest GPT-4 with enhanced capabilities
- `openai/gpt-4o` - Multimodal GPT-4 with vision
- `openai/gpt-4o-mini` - Faster, cost-effective variant
- `openai/gpt-3.5-turbo` - Fast and efficient

### Meta Models

- `meta/llama-3.1-8b` - Open-source 8B parameter model
- `meta/llama-3.1-70b` - Large-scale 70B parameter model

### Other Providers

- `deepseek/deepseek-chat` - Advanced reasoning and coding
- `microsoft/phi-3-medium` - Efficient small language model

## Evaluation Tasks

The backend includes predefined evaluation tasks:

- **Text Generation**: Creative writing, explanations, poetry
- **Question Answering**: Context-based Q&A
- **Summarization**: Text summarization
- **Code Generation**: Programming tasks
- **Reasoning**: Logical reasoning problems

## Rate Limiting

Built-in rate limiting ensures fair usage:

- **GitHub Models**: 30 requests/minute, 300/hour, 3000/day
- **Evaluation**: 30 requests/minute, 300/hour, 3000/day
- **Burst Handling**: Allows short bursts of requests

## Caching

Intelligent caching reduces API calls:

- **TTL-based**: Configurable time-to-live
- **Request Deduplication**: Identical requests use cache
- **Statistics**: Hit rates and performance metrics
- **Automatic Cleanup**: Expired entries removed automatically

## Integration with Gradio App

To integrate with your Gradio application:

```python
# In your Gradio app
from src.github_models_backend import ModelEvaluator, EvaluationConfig

class LenovoAAITCApp:
    def __init__(self):
        self.evaluator = ModelEvaluator()

    async def run_evaluation(self, models, tasks, parameters):
        config = EvaluationConfig(
            models=models,
            tasks=tasks,
            parameters=parameters
        )

        report = await self.evaluator.evaluate_models(config)
        return report.summary
```

## Error Handling

The backend includes comprehensive error handling:

- **API Errors**: Graceful handling of GitHub Models API errors
- **Rate Limit Exceeded**: Automatic waiting and retry
- **Network Issues**: Timeout and connection error handling
- **Invalid Models**: Clear error messages for unsupported models

## Monitoring and Statistics

Access detailed statistics:

```python
# Rate limiting stats
status = rate_limiter.get_status("github_models")

# Cache statistics
stats = cache_manager.get_stats()

# Evaluation metrics
metrics = evaluator.get_evaluation_metrics()
```

## Example Usage

See `example_usage.py` for comprehensive examples of:

- Single model evaluation
- Batch evaluation
- Rate limiting demonstration
- Caching functionality
- Complete integration workflow

## API Reference

### GitHubModelsClient

Main client for GitHub Models API integration.

**Methods:**

- `get_available_models()` - Get list of available models
- `evaluate_model(request)` - Evaluate a single model
- `batch_evaluate(requests)` - Evaluate multiple models
- `get_rate_limit_status()` - Get rate limiting status

### ModelEvaluator

Comprehensive model evaluation framework.

**Methods:**

- `evaluate_models(config)` - Run comprehensive evaluation
- `get_available_tasks()` - Get available evaluation tasks
- `get_evaluation_metrics()` - Get available metrics

### RateLimiter

Rate limiting for API requests.

**Methods:**

- `wait_if_needed(endpoint)` - Wait if rate limit exceeded
- `get_status(endpoint)` - Get current rate limit status
- `reset_endpoint(endpoint)` - Reset rate limit counters

### CacheManager

Caching for API responses.

**Methods:**

- `get(model_id, messages, parameters)` - Get cached response
- `set(model_id, messages, parameters, data, ttl)` - Cache response
- `get_stats()` - Get cache statistics
- `clear()` - Clear all cache entries

## Contributing

To extend the backend:

1. Add new models to `_initialize_models()`
2. Add new tasks to `_initialize_evaluation_tasks()`
3. Add new metrics to `_initialize_evaluation_metrics()`
4. Update rate limits in `RateLimiter` class

## License

This module is part of the Lenovo AAITC Solutions project and follows the same licensing terms.

