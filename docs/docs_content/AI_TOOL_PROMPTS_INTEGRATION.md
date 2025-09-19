# AI Tool System Prompts Archive Integration

## Overview

This document describes the integration of the AI Tool System Prompts Archive into the Lenovo AAITC Solutions framework. This integration provides access to system prompts from 25+ popular AI tools, significantly enhancing the experimental scale for model evaluation.

## Features

### 🚀 Core Capabilities

- **25+ AI Tools Supported**: Cursor, Claude Code, Devin AI, v0, Windsurf, and more
- **Local Caching System**: Intelligent caching to manage repository size and improve performance
- **Direct GitHub Integration**: Robust loading using direct URLs to avoid API rate limits
- **Dynamic Tool Discovery**: Automatic discovery and loading of available AI tools
- **Force Refresh**: Ability to bypass cache and load fresh prompts when needed

### 📊 Repository Statistics

- **Total AI Tools**: 25+ supported tools
- **Estimated Prompts**: 20,000+ system prompts
- **Repository Size**: Managed through intelligent caching
- **Update Frequency**: Dynamic loading on each run

## Supported AI Tools

| Tool Name    | GitHub Folder        | Description                           |
| ------------ | -------------------- | ------------------------------------- |
| Cursor       | Cursor Prompts       | AI-powered code editor system prompts |
| Claude Code  | Claude Code          | Anthropic's coding assistant prompts  |
| Devin AI     | Devin AI             | AI software engineer system prompts   |
| v0           | v0 Prompts and Tools | Vercel's AI UI generation prompts     |
| Windsurf     | Windsurf             | AI-powered development environment    |
| Augment Code | Augment Code         | Code augmentation AI prompts          |
| Cluely       | Cluely               | AI assistant system prompts           |
| CodeBuddy    | CodeBuddy Prompts    | Code assistance AI prompts            |
| Warp         | Warp.dev             | Terminal AI assistant prompts         |
| Xcode        | Xcode                | Apple development environment AI      |
| Z.ai Code    | Z.ai Code            | AI coding assistant prompts           |
| dia          | dia                  | AI development assistant              |

## Installation & Setup

### Prerequisites

- Python 3.8+
- Internet connection for initial prompt loading
- Local storage for caching (configurable)

### Basic Setup

```python
from src.model_evaluation.prompt_registries import PromptRegistryManager

# Initialize with default cache directory
registry = PromptRegistryManager()

# Or specify custom cache directory
registry = PromptRegistryManager(cache_dir="custom/cache/path")
```

## Usage Examples

### Basic Usage

```python
import asyncio
from src.model_evaluation.prompt_registries import PromptRegistryManager

async def main():
    # Initialize registry
    registry = PromptRegistryManager(cache_dir="cache/ai_tool_prompts")

    # Get available AI tools
    tools = registry.get_available_ai_tools()
    print(f"Available tools: {tools}")

    # Load prompts for a specific tool
    cursor_prompts = await registry.load_ai_tool_system_prompts("Cursor")
    print(f"Loaded {len(cursor_prompts)} Cursor prompts")

    # Load all available prompts
    all_prompts = await registry.load_ai_tool_system_prompts()
    print(f"Total prompts loaded: {len(all_prompts)}")

# Run the example
asyncio.run(main())
```

### Advanced Usage

```python
import asyncio
from src.model_evaluation.prompt_registries import PromptRegistryManager, PromptCategory

async def advanced_example():
    registry = PromptRegistryManager(cache_dir="cache/ai_tool_prompts")

    # Check cache status
    for tool in ["Cursor", "Claude Code", "Devin AI"]:
        if registry.is_tool_cached(tool):
            print(f"{tool} is cached")
            cached_prompts = registry.load_cached_tool_prompts(tool)
            print(f"  Cached prompts: {len(cached_prompts)}")
        else:
            print(f"{tool} is not cached")

    # Force refresh from GitHub
    fresh_prompts = await registry.load_ai_tool_system_prompts("Cursor", force_refresh=True)
    print(f"Fresh prompts loaded: {len(fresh_prompts)}")

    # Get statistics
    stats = registry.get_ai_tool_prompt_statistics()
    print(f"Statistics: {stats}")

asyncio.run(advanced_example())
```

### Integration with Model Evaluation

```python
from src.model_evaluation.prompt_registries import PromptRegistryManager, PromptCategory

async def evaluation_integration():
    registry = PromptRegistryManager()

    # Load AI tool prompts for evaluation
    ai_tool_prompts = await registry.load_ai_tool_system_prompts()

    # Get enhanced evaluation dataset
    dataset = registry.get_enhanced_evaluation_dataset(
        target_size=10000,
        categories=[PromptCategory.CODE_GENERATION, PromptCategory.REASONING],
        enhanced_scale=True
    )

    print(f"Enhanced dataset size: {len(dataset)}")
    print(f"AI tool prompts included: {len(ai_tool_prompts)}")

asyncio.run(evaluation_integration())
```

## API Reference

### PromptRegistryManager

#### Constructor

```python
def __init__(self, enable_caching: bool = True, cache_dir: str = "cache/ai_tool_prompts")
```

**Parameters:**

- `enable_caching`: Whether to enable caching (default: True)
- `cache_dir`: Directory for local caching (default: "cache/ai_tool_prompts")

#### Core Methods

##### `get_available_ai_tools() -> List[str]`

Returns a list of available AI tool names.

```python
tools = registry.get_available_ai_tools()
# Returns: ['Cursor', 'Claude Code', 'Devin AI', 'v0', 'Windsurf', ...]
```

##### `is_tool_cached(tool_name: str) -> bool`

Checks if a tool's prompts are cached locally.

```python
if registry.is_tool_cached("Cursor"):
    print("Cursor prompts are cached")
```

##### `load_cached_tool_prompts(tool_name: str) -> List[PromptEntry]`

Loads prompts from local cache for a specific tool.

```python
cached_prompts = registry.load_cached_tool_prompts("Cursor")
```

##### `save_tool_prompts_to_cache(tool_name: str, prompts: List[PromptEntry])`

Saves prompts to local cache for a specific tool.

```python
registry.save_tool_prompts_to_cache("Cursor", prompts)
```

##### `load_ai_tool_system_prompts(tool_name: Optional[str] = None, force_refresh: bool = False) -> List[PromptEntry]`

Loads AI tool system prompts from GitHub or cache.

**Parameters:**

- `tool_name`: Specific tool to load (None for all tools)
- `force_refresh`: Force refresh from GitHub (default: False)

**Returns:**

- List of PromptEntry objects

```python
# Load specific tool
cursor_prompts = await registry.load_ai_tool_system_prompts("Cursor")

# Load all tools
all_prompts = await registry.load_ai_tool_system_prompts()

# Force refresh
fresh_prompts = await registry.load_ai_tool_system_prompts("Cursor", force_refresh=True)
```

##### `get_ai_tool_prompt_statistics() -> Dict[str, Any]`

Returns statistics about available AI tools and prompts.

```python
stats = registry.get_ai_tool_prompt_statistics()
# Returns: {'tools_available': [...], 'total_prompts': 1234, ...}
```

## Caching System

### Local File Caching

The system implements intelligent local file caching to:

- **Reduce GitHub API calls**: Avoid rate limiting
- **Improve performance**: Faster loading on subsequent runs
- **Manage repository size**: Only cache what's needed
- **Enable offline usage**: Work with cached prompts when offline

### Cache Directory Structure

```
cache/ai_tool_prompts/
├── cursor.json
├── claude_code.json
├── devin_ai.json
├── v0.json
└── ...
```

### Cache File Format

Each cache file contains:

```json
{
  "tool_name": "Cursor",
  "cached_at": 1695123456.789,
  "prompt_count": 201,
  "prompts": [
    {
      "id": "Cursor Prompts_text_0",
      "text": "You are an AI coding assistant...",
      "category": "code_generation",
      "source": "AI Tool: Cursor Prompts",
      "metadata": {...},
      "quality_score": 0.85,
      "difficulty_level": "medium"
    }
  ]
}
```

## Configuration

### Environment Variables

```bash
# Optional: Custom cache directory
AI_TOOL_PROMPTS_CACHE_DIR=custom/cache/path

# Optional: GitHub repository URL
AI_TOOL_PROMPTS_REPO_URL=https://github.com/x1xhlol/system-prompts-and-models-of-ai-tools
```

### Configuration Options

```python
# Custom configuration
registry = PromptRegistryManager(
    enable_caching=True,
    cache_dir="custom/cache/path"
)

# Disable caching (always fetch from GitHub)
registry = PromptRegistryManager(enable_caching=False)
```

## Performance Considerations

### Loading Performance

- **First Run**: Downloads from GitHub (slower, ~5-10 seconds per tool)
- **Cached Runs**: Loads from local files (faster, ~100ms per tool)
- **Force Refresh**: Bypasses cache and downloads fresh (slower)

### Memory Usage

- **In-Memory Cache**: Stores recently loaded prompts
- **Local File Cache**: Persistent storage on disk
- **Memory Management**: Automatic cleanup of old cache entries

### Network Usage

- **Direct GitHub URLs**: Uses `raw.githubusercontent.com` for reliability
- **Rate Limiting**: Built-in delays to avoid GitHub rate limits
- **Retry Logic**: Automatic retry with exponential backoff

## Troubleshooting

### Common Issues

#### 1. Rate Limiting

**Problem**: GitHub API rate limit exceeded

**Solution**: The system automatically handles rate limiting with retry logic and delays.

#### 2. Cache Not Working

**Problem**: Prompts not being cached locally

**Solution**: Check cache directory permissions and disk space.

```python
# Check cache directory
import os
cache_dir = "cache/ai_tool_prompts"
print(f"Cache directory exists: {os.path.exists(cache_dir)}")
print(f"Cache directory writable: {os.access(cache_dir, os.W_OK)}")
```

#### 3. Network Issues

**Problem**: Cannot connect to GitHub

**Solution**: Check internet connection and GitHub availability.

```python
# Test GitHub connectivity
import requests
try:
    response = requests.get("https://raw.githubusercontent.com", timeout=10)
    print(f"GitHub accessible: {response.status_code == 200}")
except Exception as e:
    print(f"GitHub not accessible: {e}")
```

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your code here
registry = PromptRegistryManager()
```

## Best Practices

### 1. Cache Management

- **Regular Cleanup**: Periodically clean old cache files
- **Size Monitoring**: Monitor cache directory size
- **Backup**: Backup important cached prompts

### 2. Performance Optimization

- **Batch Loading**: Load multiple tools in sequence
- **Selective Loading**: Only load tools you need
- **Cache Warming**: Pre-load frequently used tools

### 3. Error Handling

- **Graceful Degradation**: Handle network failures gracefully
- **Fallback Options**: Use cached prompts when GitHub is unavailable
- **User Feedback**: Provide clear error messages

## Future Enhancements

### Planned Features

- **Incremental Updates**: Only download changed prompts
- **Compression**: Compress cache files to save space
- **Parallel Loading**: Load multiple tools simultaneously
- **Web Interface**: GUI for managing cached prompts
- **Analytics**: Usage statistics and performance metrics

### Integration Opportunities

- **CI/CD Integration**: Automated prompt updates in pipelines
- **Monitoring**: Integration with monitoring systems
- **Backup**: Automated backup of cached prompts
- **Distribution**: Share cached prompts across team members

## Contributing

### Adding New AI Tools

To add support for new AI tools:

1. **Update Tool Mapping**: Add to `ai_tools` dictionary in `PromptRegistryManager`
2. **Test Integration**: Verify the tool's prompts can be loaded
3. **Update Documentation**: Add the tool to supported tools list
4. **Submit PR**: Create pull request with changes

### Reporting Issues

When reporting issues, please include:

- **Tool Name**: Which AI tool is affected
- **Error Message**: Complete error message
- **Cache Status**: Whether caching is working
- **Network Status**: GitHub connectivity status
- **Logs**: Relevant log output

## License

This integration uses the AI Tool System Prompts Archive repository, which is available under its respective license. Please refer to the original repository for licensing information.

## Acknowledgments

- **AI Tool System Prompts Archive**: [x1xhlol/system-prompts-and-models-of-ai-tools](https://github.com/x1xhlol/system-prompts-and-models-of-ai-tools)
- **GitHub**: For hosting the prompt repository
- **AI Tool Developers**: For creating the system prompts

---

**Last Updated**: September 2025  
**Version**: 1.0.0  
**Status**: Production Ready ✅
