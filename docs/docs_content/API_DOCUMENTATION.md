# API Documentation - Lenovo AAITC Solutions

## Overview

This document provides comprehensive API documentation for the Lenovo AAITC Solutions framework, covering both Assignment 1 (Model Evaluation) and Assignment 2 (AI Architecture) components.

## Table of Contents

1. [Model Evaluation API](#model-evaluation-api)
2. [AI Architecture API](#ai-architecture-api)
3. [Chat Playground API](#chat-playground-api)
4. [Gradio Application API](#gradio-application-api)
5. [Utilities API](#utilities-api)
6. [MCP Server API](#mcp-server-api)

---

## Model Evaluation API

### Core Classes

#### `ModelConfig`

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

#### `ComprehensiveEvaluationPipeline`

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

#### `RobustnessTestingSuite`

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

#### `BiasDetectionSystem`

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

#### `PromptRegistryManager`

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

    # AI Tool System Prompts Archive Integration
    def get_available_ai_tools(self) -> List[str]

    def is_tool_cached(self, tool_name: str) -> bool

    def load_cached_tool_prompts(self, tool_name: str) -> List[PromptEntry]

    def save_tool_prompts_to_cache(self, tool_name: str, prompts: List[PromptEntry])

    async def load_ai_tool_system_prompts(
        self,
        tool_name: Optional[str] = None,
        force_refresh: bool = False
    ) -> List[PromptEntry]

    def get_ai_tool_prompt_statistics(self) -> Dict[str, Any]
```

**AI Tool System Prompts Archive Integration:**

The `PromptRegistryManager` now includes comprehensive integration with the AI Tool System Prompts Archive, providing access to system prompts from 25+ popular AI tools including Cursor, Claude Code, Devin AI, v0, Windsurf, and more.

**Key Features:**

- **Local Caching**: Intelligent caching system to manage repository size and improve performance
- **Direct GitHub Integration**: Robust loading using direct URLs to avoid API rate limits
- **Dynamic Tool Discovery**: Automatic discovery and loading of available AI tools
- **Force Refresh**: Ability to bypass cache and load fresh prompts when needed

**Usage Examples:**

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

**Supported AI Tools:**

- Cursor, Claude Code, Devin AI, v0, Windsurf
- Augment Code, Cluely, CodeBuddy, Warp, Xcode
- Z.ai Code, dia, and more

---

## Chat Playground API

The Chat Playground provides a comprehensive UX studio for comparing Ollama and GitHub model services side-by-side, similar to Google AI Studio's user experience.

### API Endpoints

#### Ollama Integration

**GET** `/api/ollama/models`

- **Description**: List available Ollama models
- **Response**: `{"models": [{"name": "llama3.1:8b", "size": "4.7GB"}]}`

**POST** `/api/ollama/generate`

- **Description**: Generate response using Ollama
- **Request Body**:
  ```json
  {
    "model_name": "llama3.1:8b",
    "prompt": "Explain quantum computing",
    "parameters": {
      "temperature": 0.7,
      "max_tokens": 1000
    }
  }
  ```
- **Response**:
  ```json
  {
    "response": "Quantum computing is a revolutionary approach...",
    "model": "llama3.1:8b",
    "usage": {
      "prompt_tokens": 10,
      "completion_tokens": 150,
      "total_tokens": 160
    },
    "timestamp": "2024-01-15T10:30:00Z"
  }
  ```

#### GitHub Models Integration

**GET** `/api/github-models/available`

- **Description**: List available GitHub Models
- **Response**: `{"models": [{"id": "openai/gpt-4o", "name": "GPT-4o", "provider": "openai"}]}`

**POST** `/api/github-models/generate`

- **Description**: Generate response using GitHub Models
- **Request Body**:
  ```json
  {
    "model_id": "openai/gpt-4o",
    "prompt": "Explain quantum computing",
    "parameters": {
      "temperature": 0.7,
      "max_tokens": 1000
    }
  }
  ```

### Features

- **Side-by-Side Comparison**: Real-time comparison of local and cloud models
- **Performance Metrics**: Live tracking of response times and token usage
- **Export Functionality**: Export chat conversations as JSON
- **Model Management**: Dynamic model loading and selection
- **Error Handling**: Graceful fallbacks and user feedback

### Integration

The Chat Playground is fully integrated with the Enterprise LLMOps platform:

- **Navigation**: Accessible via sidebar after "About & Pitch"
- **Authentication**: Uses the same authentication system
- **Monitoring**: Integrated with platform monitoring and logging

---

## AI Architecture API

### Core Classes

#### `HybridAIPlatform`

Enterprise Hybrid AI Platform for comprehensive AI system orchestration.

```python
class HybridAIPlatform:
    def __init__(self, platform_name: str = "Lenovo Hybrid AI Platform")

    async def deploy_model(
        self,
        model_config: ModelDeploymentConfig,
        target_environment: DeploymentTarget
    ) -> Dict[str, Any]

    async def get_platform_metrics(
        self,
        time_window_hours: int = 24
    ) -> Dict[str, Any]

    async def scale_deployment(
        self,
        deployment_id: str,
        scaling_config: Dict[str, Any]
    ) -> Dict[str, Any]
```

#### `ModelLifecycleManager`

Comprehensive Model Lifecycle Management System.

```python
class ModelLifecycleManager:
    def __init__(self, registry_path: str = "./model_registry")

    async def register_model(
        self,
        model_id: str,
        version: str,
        stage: ModelStage,
        created_by: str,
        description: str,
        metadata: Dict[str, Any] = None,
        performance_metrics: Dict[str, float] = None,
        dependencies: List[str] = None,
        tags: List[str] = None
    ) -> ModelVersion

    async def promote_model(
        self,
        model_id: str,
        version: str,
        target_stage: ModelStage,
        deployment_config: DeploymentConfig = None
    ) -> Dict[str, Any]

    async def deploy_model(
        self,
        model_id: str,
        version: str,
        deployment_strategy: DeploymentStrategy = DeploymentStrategy.BLUE_GREEN,
        target_environment: str = "production"
    ) -> Dict[str, Any]
```

#### `AgenticComputingFramework`

Enterprise Agentic Computing Framework for multi-agent system orchestration.

```python
class AgenticComputingFramework:
    def __init__(self, framework_name: str = "Lenovo Agentic Computing Framework")

    async def register_agent(self, agent: BaseAgent) -> bool

    async def submit_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        target_agent_type: str = None,
        target_agent_id: str = None,
        deadline: datetime = None,
        dependencies: List[str] = None
    ) -> str

    async def get_agent_metrics(
        self,
        agent_id: str,
        time_window_hours: int = 24
    ) -> Dict[str, Any]
```

#### `RAGSystem`

Advanced Retrieval-Augmented Generation System for Enterprise Knowledge Management.

```python
class RAGSystem:
    def __init__(
        self,
        system_name: str = "Lenovo Enterprise RAG System",
        embedding_model: str = "sentence-transformers",
        vector_store: str = "faiss"
    )

    async def ingest_document(
        self,
        content: str,
        metadata: DocumentMetadata,
        chunking_strategy: ChunkingStrategy = None
    ) -> Dict[str, Any]

    async def retrieve(
        self,
        query: str,
        context: QueryContext = None,
        retrieval_method: RetrievalMethod = None,
        max_results: int = None
    ) -> List[RetrievalResult]

    async def generate_response(
        self,
        query: str,
        retrieved_chunks: List[RetrievalResult],
        context: QueryContext = None
    ) -> Dict[str, Any]
```

---

## Gradio Application API

### Core Classes

#### `LenovoAAITCApp`

Main application class for Lenovo AAITC Gradio interface.

```python
class LenovoAAITCApp:
    def __init__(self)

    def create_interface(self) -> gr.Blocks

    def launch(
        self,
        server_name: str = "0.0.0.0",
        server_port: int = 7860,
        share: bool = False,
        mcp_server: bool = True
    ) -> None
```

#### `ModelEvaluationInterface`

Interface for model evaluation functionality.

```python
class ModelEvaluationInterface:
    def __init__(self)

    def create_interface(self) -> gr.Blocks

    def _run_evaluation(
        self,
        selected_models: List[str],
        selected_tasks: List[str],
        include_robustness: bool,
        include_bias_detection: bool,
        enhanced_scale: bool
    ) -> tuple
```

#### `AIArchitectureInterface`

Interface for AI architecture functionality.

```python
class AIArchitectureInterface:
    def __init__(self)

    def create_interface(self) -> gr.Blocks

    def _deploy_architecture(
        self,
        architecture_type: str,
        deployment_target: str,
        configuration: Dict[str, Any]
    ) -> Dict[str, Any]
```

---

## Utilities API

### Logging System

#### `LoggingSystem`

Enterprise-grade logging system with multi-layer architecture.

```python
class LoggingSystem:
    def __init__(
        self,
        system_name: str = "Lenovo AAITC System",
        log_directory: str = "./logs",
        enable_console: bool = True,
        enable_file: bool = True,
        enable_remote: bool = False,
        max_file_size: int = 100 * 1024 * 1024,
        backup_count: int = 5,
        enable_performance_tracking: bool = True,
        enable_security_monitoring: bool = True
    )

    def info(self, message: str, category: LogCategory = LogCategory.APPLICATION, **kwargs)
    def warning(self, message: str, category: LogCategory = LogCategory.APPLICATION, **kwargs)
    def error(self, message: str, category: LogCategory = LogCategory.ERROR, **kwargs)
    def critical(self, message: str, category: LogCategory = LogCategory.ERROR, **kwargs)
    def audit(self, message: str, user_id: str = None, action: str = None, **kwargs)
    def security(self, message: str, event_type: str = None, **kwargs)
    def performance(self, message: str, metrics: Dict[str, float] = None, **kwargs)
```

### Visualization Utilities

#### `VisualizationUtils`

Comprehensive visualization utilities for AI applications.

```python
class VisualizationUtils:
    def __init__(self, default_theme: str = "plotly_white")

    def create_model_performance_chart(
        self,
        data: pd.DataFrame,
        metrics: List[str],
        models: List[str],
        config: ChartConfig = None
    ) -> go.Figure

    def create_performance_trend_chart(
        self,
        data: pd.DataFrame,
        metric: str,
        time_column: str = "timestamp",
        config: ChartConfig = None
    ) -> go.Figure

    def create_architecture_diagram(
        self,
        components: List[Dict[str, Any]],
        connections: List[Dict[str, Any]],
        config: ChartConfig = None
    ) -> go.Figure

    def export_chart(
        self,
        fig: go.Figure,
        filename: str,
        format: ExportFormat = ExportFormat.HTML,
        width: int = 800,
        height: int = 600
    ) -> str
```

### Data Utilities

#### `DataUtils`

Comprehensive data processing and manipulation utilities.

```python
class DataUtils:
    def __init__(self)

    def validate_data(
        self,
        data: pd.DataFrame,
        schema: Dict[str, Any] = None,
        strict: bool = False
    ) -> Tuple[bool, List[str]]

    def clean_data(
        self,
        data: pd.DataFrame,
        cleaning_rules: Dict[str, Any] = None
    ) -> pd.DataFrame

    def assess_data_quality(self, data: pd.DataFrame) -> DataQualityReport

    def transform_data(
        self,
        data: pd.DataFrame,
        transformations: Dict[str, Any]
    ) -> pd.DataFrame

    def calculate_statistics(self, data: pd.DataFrame) -> Dict[str, Any]
```

### Configuration Utilities

#### `ConfigUtils`

Comprehensive configuration management utilities.

```python
class ConfigUtils:
    def __init__(self, config_directory: str = "./config")

    def load_config(
        self,
        config_name: str,
        format: ConfigFormat = ConfigFormat.JSON,
        validate: bool = True
    ) -> Dict[str, Any]

    def save_config(
        self,
        config: Dict[str, Any],
        config_name: str,
        format: ConfigFormat = ConfigFormat.JSON,
        backup: bool = True
    ) -> str

    def get_config_value(
        self,
        config_name: str,
        key: str,
        default: Any = None,
        use_env: bool = True
    ) -> Any

    def set_config_value(
        self,
        config_name: str,
        key: str,
        value: Any,
        save: bool = True
    ) -> None
```

---

## MCP Server API

### Core Classes

#### `EnterpriseAIMCP`

Enterprise-grade MCP server for AI architecture and model evaluation.

```python
class EnterpriseAIMCP:
    def __init__(self, server_name: str = "Lenovo Enterprise AI MCP")

    async def start_server(
        self,
        host: str = "0.0.0.0",
        port: int = 8081,
        max_connections: int = 100
    ) -> None

    async def stop_server(self) -> None

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> ToolResult

    def register_tool(self, tool: MCPTool) -> None

    def get_available_tools(self) -> List[Dict[str, Any]]
```

### Tool Categories

#### Model Evaluation Tools

- `comprehensive_model_evaluation`: Complete model evaluation pipeline
- `robustness_testing`: Adversarial and edge case testing
- `bias_detection`: Multi-dimensional bias analysis
- `performance_analysis`: Performance metrics and benchmarking
- `prompt_registry_integration`: Enhanced experimental scale

#### AI Architecture Tools

- `deploy_model_factory`: Dynamic model deployment
- `create_global_alert_system`: Enterprise-wide monitoring
- `register_tenant`: Multi-tenant architecture management
- `create_deployment_pipeline`: CI/CD pipeline creation
- `setup_enterprise_metrics`: Comprehensive metrics collection

#### Monitoring and Analytics Tools

- `get_system_metrics`: Real-time system monitoring
- `generate_performance_report`: Performance analysis reports
- `create_visualization`: Interactive charts and dashboards
- `export_data`: Data export in multiple formats
- `configure_alerting`: Alert system configuration

---

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
- **AI Architecture**: 50 requests per minute per user
- **MCP Server**: 200 requests per minute per connection
- **Utilities**: 500 requests per minute per user

## Authentication

APIs support multiple authentication methods:

- **API Keys**: For programmatic access
- **OAuth 2.0**: For web application integration
- **JWT Tokens**: For session-based authentication
- **Enterprise SSO**: For corporate integration

## Versioning

APIs use semantic versioning:

- **v1.0**: Initial release with core functionality
- **v1.1**: Enhanced experimental scale features
- **v1.2**: Enterprise architecture capabilities
- **v2.0**: Major architectural improvements (planned)

---

For more detailed information, please refer to the individual module documentation and examples in the codebase.
