# Chat Playground API Documentation

## 🎯 Overview

The Chat Playground is a comprehensive UX studio that showcases Ollama and GitHub model services with side-by-side comparison capabilities, similar to Google AI Studio's user experience. This feature provides real-time model comparison, performance metrics, and seamless integration with both local and cloud-based AI models.

## 🚀 Key Features

### Core Capabilities

- **Side-by-Side Comparison**: Compare Ollama (local) and GitHub Models (cloud) responses in real-time
- **Model Selection**: Dynamic model loading and selection for both Ollama and GitHub Models
- **Real-time Chat Interface**: Modern chat UI with typing indicators and message history
- **Performance Metrics**: Live tracking of response times, token usage, and model performance
- **Export Functionality**: Export chat conversations and metrics for analysis

### Integration Features

- **Ollama Integration**: Direct API integration with local Ollama models
- **GitHub Models Integration**: Cloud-based model access via GitHub Models API
- **Unified Platform**: Seamless integration with the Enterprise LLMOps platform
- **Authentication**: Demo mode with optional production authentication

## 📊 Structure/Architecture

### Frontend Components

- **Chat Interface**: Dual-pane chat interface with Ollama (left) and GitHub Models (right)
- **Model Selection**: Dropdown selectors with refresh capabilities for both model types
- **Performance Dashboard**: Real-time metrics display at the bottom
- **Export Tools**: Clear chat and export conversation functionality

### Backend Integration

- **FastAPI Endpoints**: RESTful API for model management and inference
- **Ollama Manager**: Integration with local Ollama instance
- **GitHub Models Client**: Cloud model access and management
- **Real-time Updates**: WebSocket support for live monitoring

## 🌐 Service Integration

### API Endpoints

| Endpoint                       | Method | Description                           |
| ------------------------------ | ------ | ------------------------------------- |
| `/api/ollama/models`           | GET    | List available Ollama models          |
| `/api/ollama/generate`         | POST   | Generate response using Ollama        |
| `/api/github-models/available` | GET    | List available GitHub Models          |
| `/api/github-models/generate`  | POST   | Generate response using GitHub Models |

### Request/Response Examples

#### Ollama Model Generation

```http
POST /api/ollama/generate
Content-Type: application/json

{
  "model_name": "llama3.1:8b",
  "prompt": "Explain quantum computing",
  "parameters": {
    "temperature": 0.7,
    "max_tokens": 1000
  }
}
```

#### GitHub Models Generation

```http
POST /api/github-models/generate
Content-Type: application/json

{
  "model_id": "openai/gpt-4o",
  "prompt": "Explain quantum computing",
  "parameters": {
    "temperature": 0.7,
    "max_tokens": 1000
  }
}
```

### Response Format

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

## 🔧 Configuration

### Model Configuration

- **Ollama Models**: Automatically loaded from local Ollama instance
- **GitHub Models**: Pre-configured with popular models (GPT-4o, Llama 3.1, etc.)
- **Fallback Support**: Simulation mode when APIs are unavailable

### Performance Settings

- **Response Timeout**: 30 seconds maximum
- **Token Limits**: Configurable per model type
- **Rate Limiting**: Built-in protection for API calls

## 📚 Documentation

### Quick Start Guide

1. **Access the Chat Playground**:

   - Navigate to the unified platform
   - Click "Chat Playground" in the sidebar (after About & Pitch)

2. **Select Models**:

   - Choose an Ollama model from the dropdown (left side)
   - Select a GitHub model from the dropdown (right side)

3. **Start Chatting**:

   - Type messages in either chat interface
   - Compare responses side-by-side
   - Monitor performance metrics in real-time

4. **Export Results**:
   - Use "Export Chat" to save conversation history
   - Use "Clear Chat" to reset the interface

### Integration with Unified Platform

The Chat Playground is fully integrated with the Enterprise LLMOps platform:

- **Navigation**: Accessible via sidebar after "About & Pitch"
- **Authentication**: Uses the same authentication system as other services
- **Monitoring**: Integrated with platform monitoring and logging
- **Documentation**: Part of the unified documentation system

## 🛠️ Development

### Frontend Implementation

- **HTML/CSS**: Modern responsive design with Tailwind CSS
- **JavaScript**: Vanilla JavaScript with ES6+ features
- **Real-time Updates**: Dynamic UI updates without page refresh
- **Error Handling**: Comprehensive error handling and user feedback

### Backend Implementation

- **FastAPI**: RESTful API with automatic documentation
- **Async Support**: Full async/await support for high performance
- **Error Handling**: Graceful error handling with detailed error messages
- **Logging**: Comprehensive logging for debugging and monitoring

## 🚨 Troubleshooting

### Common Issues

1. **Ollama Models Not Loading**:

   - Ensure Ollama is running on localhost:11434
   - Check that models are pulled and available
   - Verify network connectivity

2. **GitHub Models API Errors**:

   - Check GitHub token configuration
   - Verify API rate limits
   - Ensure proper authentication

3. **Chat Interface Not Responding**:
   - Check browser console for JavaScript errors
   - Verify API endpoints are accessible
   - Clear browser cache and reload

### Debug Commands

```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Test GitHub Models API
curl -H "Authorization: Bearer $GITHUB_TOKEN" \
     https://api.github.com/models

# Check platform health
curl http://localhost:8080/health
```

## 📞 Support

### Resources

- **API Documentation**: http://localhost:8080/docs
- **Platform Status**: http://localhost:8080/api/status
- **GitHub Repository**: https://github.com/s-n00b/ai_assignments
- **Documentation Site**: https://s-n00b.github.io/ai_assignments

### Getting Help

- Check the troubleshooting section above
- Review the platform logs for detailed error information
- Consult the unified platform documentation
- Check the progress bulletin for current status

---

**Last Updated**: January 15, 2025  
**Version**: 2.1.0  
**Status**: Production Ready  
**Integration**: Full FastAPI Backend Integration
