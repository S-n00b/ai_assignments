# Utilities API

## Overview

The Utilities API provides helper functions and services for common operations including data processing, configuration management, logging, and visualization.

## Base URL
```
https://utils.ai-system.com/v1
```

## Authentication
```bash
Authorization: Bearer <your-token>
```

## Data Processing Utilities

### Data Validation
```http
POST /utils/data/validate
```

**Request Body:**
```json
{
  "data": [
    {"id": 1, "name": "John", "email": "john@example.com"},
    {"id": 2, "name": "Jane", "email": "invalid-email"}
  ],
  "schema": {
    "id": {"type": "integer", "required": true},
    "name": {"type": "string", "required": true},
    "email": {"type": "email", "required": true}
  }
}
```

**Response:**
```json
{
  "valid": false,
  "errors": [
    {
      "row": 1,
      "field": "email",
      "message": "Invalid email format"
    }
  ],
  "valid_rows": 1,
  "total_rows": 2
}
```

### Data Transformation
```http
POST /utils/data/transform
```

**Request Body:**
```json
{
  "data": [1, 2, 3, 4, 5],
  "operations": [
    {"type": "normalize", "method": "minmax"},
    {"type": "filter", "condition": "> 0.5"}
  ]
}
```

### Data Aggregation
```http
POST /utils/data/aggregate
```

**Request Body:**
```json
{
  "data": [
    {"category": "A", "value": 10},
    {"category": "A", "value": 20},
    {"category": "B", "value": 15}
  ],
  "group_by": ["category"],
  "aggregations": {
    "value": ["sum", "avg", "count"]
  }
}
```

## Configuration Management

### Get Configuration
```http
GET /utils/config/{config_id}
```

**Response:**
```json
{
  "config_id": "model-config-v1",
  "config": {
    "model": {
      "architecture": "transformer",
      "layers": 12,
      "hidden_size": 768
    },
    "training": {
      "batch_size": 32,
      "learning_rate": 0.001
    }
  },
  "version": "1.0.0",
  "created_at": "2025-01-01T00:00:00Z"
}
```

### Update Configuration
```http
PUT /utils/config/{config_id}
```

**Request Body:**
```json
{
  "config": {
    "model": {
      "architecture": "transformer",
      "layers": 12,
      "hidden_size": 768
    },
    "training": {
      "batch_size": 64,
      "learning_rate": 0.0005
    }
  },
  "version": "1.1.0"
}
```

## Logging Utilities

### Create Log Entry
```http
POST /utils/logs
```

**Request Body:**
```json
{
  "level": "INFO",
  "message": "Model training started",
  "context": {
    "model_id": "model-123",
    "user_id": "user-456"
  },
  "metadata": {
    "training_data_size": 10000,
    "epochs": 10
  }
}
```

### Query Logs
```http
GET /utils/logs
```

**Query Parameters:**
- `level`: Log level (DEBUG, INFO, WARN, ERROR)
- `start_date`: Start date (ISO format)
- `end_date`: End date (ISO format)
- `context`: Context filter (JSON)

**Response:**
```json
{
  "logs": [
    {
      "id": "log-123",
      "timestamp": "2025-01-01T00:00:00Z",
      "level": "INFO",
      "message": "Model training started",
      "context": {
        "model_id": "model-123"
      }
    }
  ],
  "total": 1
}
```

## Visualization Utilities

### Generate Chart
```http
POST /utils/visualization/chart
```

**Request Body:**
```json
{
  "chart_type": "line",
  "data": {
    "labels": ["Jan", "Feb", "Mar"],
    "datasets": [
      {
        "label": "Accuracy",
        "data": [0.85, 0.87, 0.90]
      }
    ]
  },
  "options": {
    "title": "Model Accuracy Over Time",
    "y_axis": {
      "min": 0,
      "max": 1
    }
  }
}
```

**Response:**
```json
{
  "chart_id": "chart-123",
  "image_url": "https://utils.ai-system.com/charts/chart-123.png",
  "svg_url": "https://utils.ai-system.com/charts/chart-123.svg"
}
```

### Generate Dashboard
```http
POST /utils/visualization/dashboard
```

**Request Body:**
```json
{
  "title": "Model Performance Dashboard",
  "widgets": [
    {
      "type": "metric",
      "title": "Accuracy",
      "value": 0.94,
      "format": "percentage"
    },
    {
      "type": "chart",
      "title": "Training Loss",
      "chart_config": {
        "type": "line",
        "data": {...}
      }
    }
  ]
}
```

## File Operations

### Upload File
```http
POST /utils/files/upload
```

**Request Body:**
```multipart/form-data
Content-Type: multipart/form-data

file: <binary_file_data>
metadata: {"description": "Training dataset"}
```

### Process File
```http
POST /utils/files/{file_id}/process
```

**Request Body:**
```json
{
  "operation": "validate_schema",
  "parameters": {
    "schema_file": "schema.json"
  }
}
```

## Cache Management

### Set Cache
```http
POST /utils/cache
```

**Request Body:**
```json
{
  "key": "model-predictions-user-123",
  "value": {
    "predictions": [...],
    "timestamp": "2025-01-01T00:00:00Z"
  },
  "ttl": 3600
}
```

### Get Cache
```http
GET /utils/cache/{key}
```

### Clear Cache
```http
DELETE /utils/cache/{key}
```

## Notification Services

### Send Notification
```http
POST /utils/notifications
```

**Request Body:**
```json
{
  "type": "email",
  "recipient": "user@example.com",
  "subject": "Model Training Complete",
  "message": "Your model training has completed successfully.",
  "template": "training_complete"
}
```

### Notification Templates
```http
GET /utils/notifications/templates
```

## Health Check
```http
GET /utils/health
```

**Response:**
```json
{
  "status": "healthy",
  "services": {
    "database": "healthy",
    "cache": "healthy",
    "file_storage": "healthy"
  },
  "timestamp": "2025-01-01T00:00:00Z"
}
```

## Error Handling

### Error Response Format
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "data",
      "issue": "Required field is missing"
    }
  }
}
```

## Rate Limiting
- **Standard**: 1000 requests per hour
- **Premium**: 5000 requests per hour
- **Enterprise**: 20000 requests per hour
