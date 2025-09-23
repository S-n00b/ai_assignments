# MLflow Integration Documentation

## 🎯 Overview

The MLflow Integration provides comprehensive experiment tracking, model registry, and MLOps capabilities for the Lenovo AAITC Enterprise Platform. This integration enables reproducible machine learning workflows, model versioning, and automated deployment pipelines.

## 🚀 Key Features

### Core Capabilities

- **Experiment Tracking**: Comprehensive logging of parameters, metrics, and artifacts
- **Model Registry**: Centralized model lifecycle management with versioning
- **Model Deployment**: Automated model serving and deployment pipelines
- **Artifact Storage**: Secure storage and versioning of models and datasets
- **Reproducibility**: Complete experiment reproducibility with environment tracking

### Integration Features

- **FastAPI Backend Integration**: Seamless API integration with enterprise platform
- **Real-time Monitoring**: Live experiment tracking and model performance monitoring
- **Automated Workflows**: CI/CD integration for model training and deployment
- **Multi-User Support**: Collaborative experiment tracking and model sharing

## 🌐 API Endpoints

### Health & Status

- `GET /api/mlflow/health` - MLflow service health status and connectivity
- `GET /api/mlflow/info` - MLflow server information and configuration
- `GET /api/mlflow/version` - MLflow version and feature information

### Experiment Management

- `GET /api/mlflow/experiments` - List all experiments and their metadata
- `POST /api/mlflow/experiments/create` - Create new experiment
- `GET /api/mlflow/experiments/{experiment_id}` - Get experiment details
- `POST /api/mlflow/experiments/{experiment_id}/runs/start` - Start new run
- `POST /api/mlflow/experiments/{experiment_id}/runs/{run_id}/end` - End run

### Run Management

- `GET /api/mlflow/runs/{run_id}` - Get run details and metadata
- `POST /api/mlflow/runs/{run_id}/log-params` - Log parameters to run
- `POST /api/mlflow/runs/{run_id}/log-metrics` - Log metrics to run
- `POST /api/mlflow/runs/{run_id}/log-artifacts` - Log artifacts to run
- `GET /api/mlflow/runs/{run_id}/artifacts` - Get run artifacts

### Model Registry

- `GET /api/mlflow/models` - List registered models
- `POST /api/mlflow/models/register` - Register new model version
- `GET /api/mlflow/models/{model_name}` - Get model details
- `GET /api/mlflow/models/{model_name}/versions` - Get model versions
- `POST /api/mlflow/models/{model_name}/versions/{version}/stage` - Update model stage

### Model Serving

- `GET /api/mlflow/serving/models` - List served models
- `POST /api/mlflow/serving/models/deploy` - Deploy model for serving
- `POST /api/mlflow/serving/models/{model_name}/predict` - Make predictions
- `DELETE /api/mlflow/serving/models/{model_name}` - Stop model serving

## 🔧 Configuration

### Service Setup

```bash
# Start MLflow server
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000 \
  --workers 4
```

### API Integration

```python
# MLflow client integration
from src.enterprise_llmops.mlops.mlflow_manager import MLflowManager

mlflow = MLflowManager()
experiment_id = await mlflow.create_experiment("Model Evaluation")
```

## 📊 Experiment Tracking

### Basic Usage

```python
import mlflow
import mlflow.sklearn

# Start experiment
mlflow.set_experiment("Lenovo Model Evaluation")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("n_estimators", 100)

    # Train model
    model = RandomForestClassifier(max_depth=10, n_estimators=100)
    model.fit(X_train, y_train)

    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.sklearn.log_model(model, "model")
```

### Advanced Tracking

```python
# Custom metrics and artifacts
with mlflow.start_run():
    # Log multiple metrics
    mlflow.log_metrics({
        "accuracy": 0.95,
        "precision": 0.94,
        "recall": 0.93,
        "f1_score": 0.935
    })

    # Log artifacts
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact("feature_importance.csv")

    # Log model with custom signature
    signature = mlflow.models.infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(
        model,
        "model",
        signature=signature,
        input_example=X_test[:5]
    )
```

## 🚀 Quick Start

### 1. Access MLflow UI

- **URL**: http://localhost:5000
- **Features**: Experiment tracking, model registry, artifact browser

### 2. Test API Endpoints

```bash
# Health check
curl http://localhost:8080/api/mlflow/health

# List experiments
curl http://localhost:8080/api/mlflow/experiments

# Create experiment
curl -X POST http://localhost:8080/api/mlflow/experiments/create \
  -H "Content-Type: application/json" \
  -d '{"name": "Test Experiment", "description": "Test MLflow integration"}'
```

### 3. Start Experiment

```bash
# Start new run
curl -X POST http://localhost:8080/api/mlflow/experiments/1/runs/start \
  -H "Content-Type: application/json" \
  -d '{"run_name": "Test Run", "tags": {"model_type": "test"}}'
```

## 📈 Model Registry

### Model Registration

```python
# Register model
model_name = "lenovo-model-evaluation"
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name=model_name
)

# Update model stage
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Production"
)
```

### Model Serving

```python
# Deploy model for serving
mlflow models serve \
  --model-uri models:/lenovo-model-evaluation/1 \
  --host 0.0.0.0 \
  --port 5001

# Make predictions
import requests
response = requests.post(
    "http://localhost:5001/invocations",
    json={"inputs": [[1, 2, 3, 4]]}
)
```

## 🔗 Integration Examples

### FastAPI Integration

```python
@app.post("/api/mlflow/experiments/start")
async def start_experiment(experiment_data: ExperimentData):
    experiment_id = await mlflow.create_experiment(
        name=experiment_data.name,
        description=experiment_data.description
    )
    return {"experiment_id": experiment_id}
```

### Gradio Integration

```python
def track_model_evaluation(model_name, metrics):
    with mlflow.start_run():
        mlflow.log_param("model_name", model_name)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact("evaluation_report.pdf")
```

## 🛠️ Development

### Code Structure

```
src/enterprise_llmops/
├── mlops/
│   ├── mlflow_manager.py      # MLflow service implementation
│   ├── experiment_tracker.py  # Experiment tracking utilities
│   ├── model_registry.py      # Model registry operations
│   └── deployment.py          # Model deployment utilities
└── integrations/
    ├── mlflow_client.py       # MLflow client wrapper
    └── artifact_manager.py    # Artifact management
```

### Adding New Features

1. Define new functionality in appropriate module
2. Add API endpoints in FastAPI application
3. Update documentation and examples
4. Add comprehensive tests

### Testing

```bash
# Test MLflow connectivity
python -m pytest tests/test_mlflow.py -v

# Test specific endpoints
curl http://localhost:8080/api/mlflow/health
```

## 📊 Monitoring & Analytics

### Experiment Analytics

- **Run Comparison**: Compare multiple runs and their performance
- **Parameter Tuning**: Track hyperparameter optimization results
- **Model Performance**: Monitor model performance over time
- **Artifact Management**: Track and version model artifacts

### Automated Workflows

- **CI/CD Integration**: Automated model training and deployment
- **Model Validation**: Automated model validation and testing
- **Performance Monitoring**: Real-time model performance tracking
- **Alerting**: Automated alerts for model performance degradation

## 🚨 Troubleshooting

### Common Issues

1. **Connection Refused**: Ensure MLflow server is running on port 5000
2. **Database Errors**: Check backend store URI configuration
3. **Artifact Storage**: Verify artifact root directory permissions
4. **Memory Issues**: Monitor MLflow server memory usage

### Debug Mode

```bash
# Enable debug logging
export MLFLOW_DEBUG=true
mlflow server --log-level DEBUG
```

### Logs

Check MLflow server logs for debugging information:

- Console output - Real-time debugging
- Log files - Persistent logging
- Database logs - Backend store operations

## 📞 Support

For issues and questions:

1. Check the [FastAPI documentation](fastapi-enterprise.md)
2. Review the [troubleshooting guide](../resources/troubleshooting.md)
3. Check the [progress bulletin](../progress-bulletin.md)
4. Access MLflow UI at http://localhost:5000

---

**Last Updated**: January 19, 2025  
**Version**: 2.1.0  
**Status**: Production Ready  
**Integration**: Full Enterprise Platform Integration
