# Gradio Model Evaluation App Documentation

## 🎯 Overview

The Gradio Model Evaluation App provides an interactive interface for comprehensive model evaluation, specifically designed for Lenovo AAITC Assignment 1: Model Evaluation Engineer role. This application integrates seamlessly with the Enterprise LLMOps Platform to provide a complete model evaluation workflow.

## 🚀 Key Features

### Core Capabilities

- **Comprehensive Evaluation Pipeline**: Multi-task evaluation framework for foundation models
- **Model Profiling & Characterization**: Performance analysis and capability assessment
- **Model Factory Architecture**: Automated model selection with use case taxonomy
- **Practical Evaluation Exercise**: Hands-on testing for Lenovo's internal operations
- **Real-time Dashboard**: Live performance monitoring and visualization
- **Report Generation**: Export capabilities for stakeholders

### Integration Features

- **FastAPI Backend Integration**: Seamless connection to Enterprise LLMOps Platform
- **MCP Server Capabilities**: Built-in Model Context Protocol support
- **MLflow Integration**: Automatic experiment tracking and logging
- **Vector Database Support**: ChromaDB integration for document embeddings

## 📊 Application Structure

### Main Tabs

1. **📊 Evaluation Pipeline** - Comprehensive model evaluation framework
2. **🔍 Model Profiling** - Performance profiling and characterization
3. **🏭 Model Factory** - Automated model selection framework
4. **🧪 Practical Evaluation** - Hands-on evaluation exercise
5. **📊 Dashboard** - Real-time visualization and monitoring
6. **📋 Reports** - Export and reporting functionality

### Supported Models

- **GPT-5** - Advanced reasoning capabilities
- **GPT-5-Codex** - 74.5% coding success rate
- **Claude 3.5 Sonnet** - Enhanced analysis and conversation
- **Llama 3.3** - Open-source alternative

## 🌐 Service Integration

### FastAPI Backend Integration

The Gradio app integrates with the Enterprise FastAPI platform:

| Service              | Port | URL                   | Integration Type    |
| -------------------- | ---- | --------------------- | ------------------- |
| **Gradio App**       | 7860 | http://localhost:7860 | Frontend Interface  |
| **FastAPI Platform** | 8080 | http://localhost:8080 | Backend Services    |
| **MLflow**           | 5000 | http://localhost:5000 | Experiment Tracking |
| **ChromaDB**         | 8081 | http://localhost:8081 | Vector Database     |

### API Endpoints Used

- `GET /api/models` - Model registry information
- `POST /api/experiments/start` - Start evaluation experiments
- `POST /api/experiments/{run_id}/log-metrics` - Log evaluation metrics
- `GET /api/chromadb/collections` - Vector database collections
- `WS /ws` - Real-time updates via WebSocket

## 🚀 Quick Start

### 1. Start the Application

```bash
# Activate virtual environment
& C:\Users\samne\PycharmProjects\ai_assignments\venv\Scripts\Activate.ps1

# Start the Gradio app
python -m src.gradio_app.main --host 0.0.0.0 --port 7860
```

### 2. Access the Interface

- **Main Interface**: http://localhost:7860
- **MCP Server**: Built-in MCP capabilities enabled
- **API Integration**: Automatic connection to FastAPI backend

### 3. Start Evaluation

1. Select models to evaluate (GPT-5, Claude 3.5 Sonnet, etc.)
2. Choose evaluation tasks (text generation, code generation, etc.)
3. Configure evaluation options (robustness, bias detection, enhanced scale)
4. Click "🚀 Start Evaluation" to begin

## 🔧 Configuration

### Model Configuration

The app uses model configurations defined in:

```python
LATEST_MODEL_CONFIGS = {
    "gpt-5": ModelConfig(...),
    "gpt-5-codex": ModelConfig(...),
    "claude-3.5-sonnet": ModelConfig(...),
    "llama-3.3": ModelConfig(...)
}
```

### Evaluation Options

- **Robustness Testing**: Adversarial inputs and noise tolerance
- **Bias Detection**: Multi-dimensional bias analysis
- **Enhanced Scale**: Prompt registries for larger datasets

## 📊 Evaluation Framework

### Task Types

- **Text Generation** - Natural language generation tasks
- **Code Generation** - Programming and coding tasks
- **Question Answering** - Information retrieval and QA
- **Summarization** - Text summarization tasks
- **Translation** - Language translation tasks

### Metrics

- **BLEU Score** - Translation quality metric
- **ROUGE Score** - Summarization quality metric
- **BERT Score** - Semantic similarity metric
- **Custom Technical Accuracy** - Domain-specific accuracy
- **Readability Score** - Output readability assessment

## 🏭 Model Factory Architecture

### Use Case Taxonomy

The Model Factory classifies use cases into categories:

- **Documentation Generation** - Technical documentation creation
- **Code Assistance** - Programming help and code review
- **Data Analysis** - Data processing and analysis tasks
- **Customer Support** - Customer service and support tasks

### Selection Criteria

- **Performance Requirements** - Speed, accuracy, quality
- **Cost Optimization** - Performance vs. cost trade-offs
- **Deployment Scenario** - Cloud, edge, mobile, hybrid
- **Use Case Specificity** - Domain-specific requirements

## 🧪 Practical Evaluation Exercise

### Lenovo Use Case

**Internal Technical Documentation Generation**

- Evaluate models for creating technical documentation
- Test with Lenovo-specific content and terminology
- Assess output quality and consistency
- Measure performance metrics and costs

### Evaluation Process

1. **Dataset Upload** - Upload evaluation datasets (JSON, CSV, TXT)
2. **Model Selection** - Choose models to evaluate
3. **Metric Selection** - Select evaluation metrics
4. **Run Evaluation** - Execute comprehensive evaluation
5. **Analysis** - Review detailed results and recommendations

## 📊 Dashboard & Visualization

### Real-time Monitoring

- **Latency Trends** - Response time monitoring
- **Throughput Comparison** - Model performance comparison
- **Quality Metrics** - Output quality assessment
- **Cost Analysis** - Resource usage and cost tracking
- **Model Comparison Radar** - Multi-dimensional comparison

### Export Capabilities

- **PDF Reports** - Executive summaries and technical reports
- **Excel Data** - Raw data and metrics export
- **JSON Format** - Structured data export
- **Dashboard Data** - Visualization data export

## 🔗 MCP Server Integration

### Built-in MCP Capabilities

The Gradio app includes built-in Model Context Protocol support:

- **Automatic Tool Discovery** - Function-based tool generation
- **Type Validation** - Parameter type checking
- **Documentation** - Automatic tool documentation
- **Progress Updates** - Real-time progress reporting

### Available MCP Tools

- `run_evaluation` - Comprehensive model evaluation
- `visualize_architecture` - AI architecture visualization
- `refresh_dashboard` - Real-time performance monitoring
- `generate_report` - Report generation and export
- `export_dashboard_data` - Data export functionality

## 📚 Integration with Enterprise Platform

### Data Flow

1. **User Interface** (Gradio) → User interactions and evaluation requests
2. **Backend Processing** (FastAPI) → Model management and experiment tracking
3. **Data Storage** (MLflow/ChromaDB) → Results storage and retrieval
4. **Real-time Updates** (WebSocket) → Live monitoring and notifications

### Shared Resources

- **Model Registry** - Centralized model management
- **Experiment Tracking** - MLflow integration for reproducibility
- **Vector Database** - ChromaDB for document embeddings
- **Monitoring** - Shared monitoring and alerting

## 🛠️ Development

### Code Structure

```
src/gradio_app/
├── main.py                    # Main application entry point
├── components.py              # UI components and interfaces
├── agentic_flow_ui.py         # Agent workflow visualization
├── knowledge_graph_ui.py      # Knowledge graph interface
├── modern_dashboard.py        # Dashboard components
├── mcp_server.py              # MCP server implementation
└── copilot_integration.py     # CopilotKit integration
```

### Adding New Features

1. Define new components in `components.py`
2. Add UI elements to the main interface
3. Implement backend integration in `main.py`
4. Update documentation and tests

### Testing

```bash
# Test the application
python -m src.gradio_app.main --debug

# Test MCP integration
# MCP tools are automatically available when the app is running
```

## 📈 Performance & Optimization

### Optimization Features

- **Caching** - Result caching for repeated evaluations
- **Async Processing** - Non-blocking evaluation execution
- **Progress Updates** - Real-time progress reporting
- **Error Handling** - Comprehensive error management

### Resource Management

- **Memory Optimization** - Efficient model loading and unloading
- **CPU Usage** - Optimized processing for evaluation tasks
- **Network Efficiency** - Minimized API calls and data transfer

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors** - Ensure all dependencies are installed
2. **Port Conflicts** - Check if port 7860 is available
3. **Backend Connection** - Verify FastAPI platform is running
4. **Model Loading** - Check model availability and permissions

### Debug Mode

```bash
# Enable debug mode
python -m src.gradio_app.main --debug
```

### Logs

Check application logs for debugging information:

- Console output - Real-time debugging
- Browser console - Frontend debugging
- Network tab - API call debugging

## 📞 Support

For issues and questions:

1. Check the [FastAPI documentation](fastapi-enterprise.md)
2. Review the [troubleshooting guide](../resources/troubleshooting.md)
3. Check the [progress bulletin](../progress-bulletin.md)
4. Examine the [live applications](../live-applications/index.md)

---

**Last Updated**: January 19, 2025  
**Version**: 2.1.0  
**Status**: Production Ready  
**Integration**: Full FastAPI Backend Integration
