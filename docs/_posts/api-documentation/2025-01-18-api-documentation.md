---
layout: post
title: "API Documentation"
date: 2025-09-18 10:00:00 -0400
categories: [Documentation, API]
tags: [API, Documentation, Model Evaluation, AI Architecture]
author: Lenovo AAITC Team
---

# API Documentation - Lenovo AAITC Solutions

## Overview

This document provides comprehensive API documentation for the Lenovo AAITC Solutions framework, covering both Assignment 1 (Model Evaluation) and Assignment 2 (AI Architecture) components.

## Table of Contents

1. [Model Evaluation API](#model-evaluation-api)
2. [AI Architecture API](#ai-architecture-api)
3. [Gradio Application API](#gradio-application-api)
4. [Utilities API](#utilities-api)
5. [MCP Server API](#mcp-server-api)

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
        port: int = 8000,
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

---

For more detailed information, please refer to the individual module documentation and examples in the codebase.
