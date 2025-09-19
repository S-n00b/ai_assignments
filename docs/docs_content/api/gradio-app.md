# Gradio Application API

## Overview

The Gradio Application API provides endpoints for interacting with the web-based AI interface, including model inference, file uploads, and real-time interactions.

## Base URL
```
https://gradio.ai-system.com
```

## Authentication
```bash
Authorization: Bearer <your-token>
```

## Core Endpoints

### Model Inference

#### Text Classification
```http
POST /api/predict/text-classification
```

**Request Body:**
```json
{
  "text": "This is a sample text for classification",
  "model_id": "sentiment-classifier-v1"
}
```

**Response:**
```json
{
  "prediction": "positive",
  "confidence": 0.94,
  "probabilities": {
    "positive": 0.94,
    "negative": 0.06
  }
}
```

#### Image Classification
```http
POST /api/predict/image-classification
```

**Request Body:**
```json
{
  "image": "base64_encoded_image_data",
  "model_id": "image-classifier-v1"
}
```

### File Operations

#### Upload File
```http
POST /api/files/upload
```

**Request Body:**
```multipart/form-data
Content-Type: multipart/form-data

file: <binary_file_data>
model_id: "document-processor-v1"
```

**Response:**
```json
{
  "file_id": "file-123",
  "filename": "document.pdf",
  "size": 1024000,
  "status": "uploaded"
}
```

#### Process File
```http
POST /api/files/{file_id}/process
```

**Response:**
```json
{
  "file_id": "file-123",
  "processing_status": "completed",
  "results": {
    "text": "Extracted text content...",
    "metadata": {
      "pages": 10,
      "language": "en"
    }
  }
}
```

### Chat Interface

#### Start Chat Session
```http
POST /api/chat/sessions
```

**Response:**
```json
{
  "session_id": "session-123",
  "created_at": "2025-01-01T00:00:00Z"
}
```

#### Send Message
```http
POST /api/chat/sessions/{session_id}/messages
```

**Request Body:**
```json
{
  "message": "Hello, how can you help me?",
  "context": {
    "user_id": "user-123",
    "session_type": "general"
  }
}
```

**Response:**
```json
{
  "message_id": "msg-123",
  "response": "Hello! I can help you with various AI tasks...",
  "timestamp": "2025-01-01T00:00:00Z",
  "metadata": {
    "model_used": "gpt-4",
    "response_time": 1.2
  }
}
```

### Real-time Features

#### WebSocket Connection
```javascript
const ws = new WebSocket('wss://gradio.ai-system.com/ws');
ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

#### Streaming Response
```http
POST /api/predict/stream
```

**Response:**
```json
{
  "type": "stream_start",
  "session_id": "session-123"
}

{
  "type": "token",
  "content": "Hello"
}

{
  "type": "token", 
  "content": " there"
}

{
  "type": "stream_end",
  "final_response": "Hello there"
}
```

## Error Handling

### Error Response Format
```json
{
  "error": {
    "code": "MODEL_NOT_FOUND",
    "message": "The requested model is not available",
    "details": {
      "model_id": "invalid-model-id"
    }
  }
}
```

## Rate Limiting
- **Standard**: 100 requests per minute
- **Premium**: 500 requests per minute
- **Enterprise**: 2000 requests per minute

## WebSocket Events

### Connection Events
```json
{
  "type": "connection_established",
  "session_id": "session-123"
}
```

### Error Events
```json
{
  "type": "error",
  "code": "RATE_LIMIT_EXCEEDED",
  "message": "Rate limit exceeded"
}
```

## SDK Examples

### Python SDK
```python
from gradio_client import Client

client = Client("https://gradio.ai-system.com")

# Text classification
result = client.predict(
    "This is a positive text",
    api_name="/predict"
)

# File upload and processing
file_result = client.upload_file("document.pdf")
processed = client.process_file(file_result["file_id"])
```

### JavaScript SDK
```javascript
import { GradioClient } from '@gradio/client';

const client = new GradioClient('https://gradio.ai-system.com');

// Text classification
const result = await client.predict({
  text: "This is a positive text",
  model_id: "sentiment-classifier-v1"
});

// File upload
const file = document.getElementById('fileInput').files[0];
const uploadResult = await client.uploadFile(file);
```
