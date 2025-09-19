# Model Evaluation API

## Overview

This document provides comprehensive API documentation for the Model Evaluation components of the Lenovo AAITC Solutions framework, covering comprehensive evaluation pipelines, robustness testing, bias detection, and prompt registry integration.

## Core Classes

### `ModelConfig`

Configuration class for foundation models with latest Q3 2025 specifications.

```python
@dataclass
class ModelConfig:
    name: str
    provider: str  # 'openai', 'anthropic', 'meta', 'local'
    model_id: str
    api_key: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.7
    cost_per_1k_tokens: float = 0.0
    context_window: int = 4096
    parameters: int = 0  # Model parameter count in billions
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Methods:**

- `from_dict(config_dict: Dict[str, Any]) -> ModelConfig`: Create from dictionary
- `to_dict() -> Dict[str, Any]`: Convert to dictionary
- `validate() -> bool`: Validate configuration

### `ComprehensiveEvaluationPipeline`

Main evaluation pipeline for comparing foundation models across multiple dimensions.

```python
class ComprehensiveEvaluationPipeline:
    def __init__(self, models: List[ModelConfig], enable_logging: bool = True)

    async def evaluate_model_comprehensive(
        self,
        model_config: ModelConfig,
        test_data: pd.DataFrame,
        task_type: TaskType,
        include_robustness: bool = True,
        include_bias_detection: bool = True
    ) -> Dict[str, Any]

    async def run_multi_task_evaluation(
        self,
        test_datasets: Dict[TaskType, pd.DataFrame],
        include_robustness: bool = True,
        include_bias_detection: bool = True
    ) -> pd.DataFrame

    def generate_evaluation_report(
        self,
        results: pd.DataFrame,
        output_format: str = "html"
    ) -> str
```

### `RobustnessTestingSuite`

Comprehensive robustness testing for model evaluation.

```python
class RobustnessTestingSuite:
    def __init__(self)

    async def test_adversarial_robustness(
        self,
        model_config: ModelConfig,
        test_prompts: List[str]
    ) -> Dict[str, Any]

    async def test_noise_tolerance(
        self,
        model_config: ModelConfig,
        test_prompts: List[str]
    ) -> Dict[str, Any]

    async def test_edge_cases(
        self,
        model_config: ModelConfig,
        edge_case_prompts: List[str]
    ) -> Dict[str, Any]
```

### `BiasDetectionSystem`

Multi-dimensional bias detection and analysis system.

```python
class BiasDetectionSystem:
    def __init__(self)

    async def detect_bias(
        self,
        model_config: ModelConfig,
        test_prompts: List[str],
        protected_characteristics: List[str]
    ) -> Dict[str, Any]

    def calculate_fairness_metrics(
        self,
        predictions: List[str],
        ground_truth: List[str],
        protected_attributes: List[str]
    ) -> Dict[str, float]

    def generate_bias_report(
        self,
        bias_results: Dict[str, Any]
    ) -> str
```

### `PromptRegistryManager`

Manager for integrating with multiple prompt registries and generating enhanced evaluation datasets.

```python
class PromptRegistryManager:
    def __init__(self, enable_caching: bool = True, cache_dir: str = "cache/ai_tool_prompts")

    def get_enhanced_evaluation_dataset(
        self,
        target_size: int = 10000,
        categories: Optional[List[PromptCategory]] = None,
        difficulty_levels: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        quality_threshold: float = 0.3
    ) -> pd.DataFrame

    async def get_dynamic_evaluation_dataset(
        self,
        model_capabilities: Dict[str, Any],
        evaluation_goals: List[str],
        target_size: int = 5000
    ) -> pd.DataFrame

    async def get_adversarial_prompts(
        self,
        base_category: PromptCategory,
        adversarial_types: List[str] = None,
        count: int = 100
    ) -> pd.DataFrame
```

## AI Tool System Prompts Archive Integration

The `PromptRegistryManager` includes comprehensive integration with the AI Tool System Prompts Archive, providing access to system prompts from 25+ popular AI tools including Cursor, Claude Code, Devin AI, v0, Windsurf, and more.

### Key Features

- **Local Caching**: Intelligent caching system to manage repository size and improve performance
- **Direct GitHub Integration**: Robust loading using direct URLs to avoid API rate limits
- **Dynamic Tool Discovery**: Automatic discovery and loading of available AI tools
- **Force Refresh**: Ability to bypass cache and load fresh prompts when needed

### Usage Examples

```python
# Initialize with local caching
registry = PromptRegistryManager(cache_dir="cache/ai_tool_prompts")

# Get available AI tools
tools = registry.get_available_ai_tools()
print(f"Available tools: {tools}")

# Load prompts for a specific tool
cursor_prompts = await registry.load_ai_tool_system_prompts("Cursor")

# Load all available prompts
all_prompts = await registry.load_ai_tool_system_prompts()

# Force refresh from GitHub
fresh_prompts = await registry.load_ai_tool_system_prompts("Cursor", force_refresh=True)

# Check cache status
if registry.is_tool_cached("Cursor"):
    cached_prompts = registry.load_cached_tool_prompts("Cursor")
```

### Supported AI Tools

- Cursor, Claude Code, Devin AI, v0, Windsurf
- Augment Code, Cluely, CodeBuddy, Warp, Xcode
- Z.ai Code, dia, and more

## Error Handling

All APIs use consistent error handling patterns:

```python
try:
    result = await api_method(parameters)
    return {"status": "success", "data": result}
except ValidationError as e:
    return {"status": "error", "error": f"Validation error: {str(e)}"}
except APIError as e:
    return {"status": "error", "error": f"API error: {str(e)}"}
except Exception as e:
    return {"status": "error", "error": f"Unexpected error: {str(e)}"}
```

## Response Formats

All API responses follow a consistent format:

```python
{
    "status": "success|error",
    "data": Any,  # Response data (only for success)
    "error": str,  # Error message (only for error)
    "metadata": {
        "timestamp": "2025-01-XX",
        "request_id": "uuid",
        "execution_time_ms": 1234
    }
}
```

## Rate Limiting

APIs implement rate limiting to ensure system stability:

- **Model Evaluation**: 100 requests per minute per user
- **Robustness Testing**: 50 requests per minute per user
- **Bias Detection**: 75 requests per minute per user
- **Prompt Registry**: 200 requests per minute per user

## Authentication

APIs support multiple authentication methods:

- **API Keys**: For programmatic access
- **OAuth 2.0**: For web application integration
- **JWT Tokens**: For session-based authentication
- **Enterprise SSO**: For corporate integration

---

For more detailed information, please refer to the individual module documentation and examples in the codebase.
