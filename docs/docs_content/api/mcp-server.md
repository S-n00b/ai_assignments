# MCP Server API

## Overview

The Model Context Protocol (MCP) Server API provides standardized endpoints for model communication, context management, and protocol compliance across different AI systems.

## Base URL
```
https://mcp.ai-system.com/v1
```

## Authentication
```bash
Authorization: Bearer <your-token>
X-MCP-Version: 1.0
```

## Protocol Endpoints

### Initialize Connection
```http
POST /mcp/initialize
```

**Request Body:**
```json
{
  "protocol_version": "1.0",
  "capabilities": {
    "tools": true,
    "resources": true,
    "prompts": true
  },
  "client_info": {
    "name": "ai-client",
    "version": "1.0.0"
  }
}
```

**Response:**
```json
{
  "protocol_version": "1.0",
  "capabilities": {
    "tools": {
      "list_changed": true,
      "call_tool": true
    },
    "resources": {
      "subscribe": true,
      "unsubscribe": true
    },
    "prompts": {
      "list": true,
      "get": true
    }
  },
  "server_info": {
    "name": "ai-system-mcp",
    "version": "1.0.0"
  }
}
```

### List Tools
```http
GET /mcp/tools
```

**Response:**
```json
{
  "tools": [
    {
      "name": "predict",
      "description": "Make predictions using AI models",
      "input_schema": {
        "type": "object",
        "properties": {
          "model_id": {"type": "string"},
          "input_data": {"type": "object"}
        },
        "required": ["model_id", "input_data"]
      }
    },
    {
      "name": "analyze_data",
      "description": "Analyze data using statistical methods",
      "input_schema": {
        "type": "object",
        "properties": {
          "data": {"type": "array"},
          "analysis_type": {"type": "string", "enum": ["statistical", "ml"]}
        }
      }
    }
  ]
}
```

### Call Tool
```http
POST /mcp/tools/call
```

**Request Body:**
```json
{
  "name": "predict",
  "arguments": {
    "model_id": "sentiment-classifier-v1",
    "input_data": {
      "text": "This is a positive message"
    }
  }
}
```

**Response:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "Prediction: positive (confidence: 0.94)"
    }
  ],
  "is_error": false
}
```

### List Resources
```http
GET /mcp/resources
```

**Response:**
```json
{
  "resources": [
    {
      "uri": "file://models/model-config.json",
      "name": "Model Configuration",
      "description": "Configuration for AI models",
      "mimeType": "application/json"
    },
    {
      "uri": "memory://conversation/123",
      "name": "Conversation Context",
      "description": "Current conversation context",
      "mimeType": "application/json"
    }
  ]
}
```

### Read Resource
```http
GET /mcp/resources/read
```

**Query Parameters:**
- `uri`: Resource URI

**Response:**
```json
{
  "contents": [
    {
      "uri": "file://models/model-config.json",
      "mimeType": "application/json",
      "text": "{\n  \"model\": {\n    \"architecture\": \"transformer\"\n  }\n}"
    }
  ]
}
```

### Subscribe to Resource
```http
POST /mcp/resources/subscribe
```

**Request Body:**
```json
{
  "uri": "memory://conversation/123"
}
```

**Response:**
```json
{
  "uri": "memory://conversation/123",
  "subscription_id": "sub-123"
}
```

### List Prompts
```http
GET /mcp/prompts
```

**Response:**
```json
{
  "prompts": [
    {
      "name": "sentiment_analysis",
      "description": "Analyze sentiment of text input",
      "arguments": [
        {
          "name": "text",
          "description": "Text to analyze",
          "required": true
        }
      ]
    }
  ]
}
```

### Get Prompt
```http
POST /mcp/prompts/get
```

**Request Body:**
```json
{
  "name": "sentiment_analysis",
  "arguments": {
    "text": "I love this product!"
  }
}
```

**Response:**
```json
{
  "description": "Analyze the sentiment of the provided text",
  "messages": [
    {
      "role": "user",
      "content": {
        "type": "text",
        "text": "Analyze the sentiment of: 'I love this product!'"
      }
    }
  ]
}
```

## Context Management

### Update Context
```http
POST /mcp/context/update
```

**Request Body:**
```json
{
  "context_id": "session-123",
  "context_data": {
    "user_id": "user-456",
    "conversation_history": [
      {
        "role": "user",
        "content": "Hello"
      },
      {
        "role": "assistant", 
        "content": "Hi! How can I help you?"
      }
    ]
  }
}
```

### Get Context
```http
GET /mcp/context/{context_id}
```

**Response:**
```json
{
  "context_id": "session-123",
  "context_data": {
    "user_id": "user-456",
    "conversation_history": [...],
    "metadata": {
      "created_at": "2025-01-01T00:00:00Z",
      "last_updated": "2025-01-01T00:05:00Z"
    }
  }
}
```

## Model Integration

### Model Discovery
```http
GET /mcp/models
```

**Response:**
```json
{
  "models": [
    {
      "model_id": "gpt-4",
      "name": "GPT-4",
      "capabilities": ["text_generation", "conversation"],
      "max_tokens": 8192,
      "supported_formats": ["text", "json"]
    },
    {
      "model_id": "claude-3",
      "name": "Claude 3",
      "capabilities": ["text_generation", "analysis"],
      "max_tokens": 100000,
      "supported_formats": ["text", "json", "markdown"]
    }
  ]
}
```

### Model Invocation
```http
POST /mcp/models/invoke
```

**Request Body:**
```json
{
  "model_id": "gpt-4",
  "messages": [
    {
      "role": "user",
      "content": "Explain quantum computing"
    }
  ],
  "parameters": {
    "max_tokens": 1000,
    "temperature": 0.7
  }
}
```

## Error Handling

### MCP Error Response
```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The request format is invalid",
    "data": {
      "field": "arguments",
      "issue": "Missing required field 'model_id'"
    }
  }
}
```

### Error Codes
- `INVALID_REQUEST`: Request format is invalid
- `TOOL_NOT_FOUND`: Requested tool does not exist
- `RESOURCE_NOT_FOUND`: Requested resource does not exist
- `CONTEXT_ERROR`: Context management error
- `MODEL_ERROR`: Model invocation error

## WebSocket Support

### WebSocket Connection
```javascript
const ws = new WebSocket('wss://mcp.ai-system.com/ws');
ws.onmessage = function(event) {
  const message = JSON.parse(event.data);
  console.log('MCP Message:', message);
};
```

### WebSocket Message Format
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "predict",
    "arguments": {...}
  },
  "id": "request-123"
}
```

## Protocol Compliance

### Version Negotiation
The MCP server supports version negotiation:
- Client declares supported versions
- Server responds with compatible version
- Fallback to lowest common version

### Capability Discovery
Clients can discover server capabilities:
- Tools available for invocation
- Resources available for access
- Prompts available for use

### Streaming Support
Long-running operations support streaming:
- Real-time progress updates
- Partial results delivery
- Cancellation support

## Best Practices

### Connection Management
- Implement proper connection pooling
- Handle connection failures gracefully
- Use keep-alive for long connections

### Error Handling
- Implement retry logic with exponential backoff
- Handle rate limiting appropriately
- Log errors for debugging

### Performance Optimization
- Use connection multiplexing
- Implement request batching
- Cache frequently accessed resources

This MCP Server API provides a standardized interface for AI system communication, ensuring compatibility and interoperability across different AI platforms and tools.
