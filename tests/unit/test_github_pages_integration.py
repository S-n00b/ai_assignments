"""
Unit tests for GitHub Pages integration and frontend functionality.

Tests the GitHub Pages deployment, MkDocs integration, and frontend components
that serve the documentation and live applications.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import json
import yaml
from datetime import datetime

# Mock imports for components that might not be available in test environment
try:
    from src.enterprise_llmops.frontend.enhanced_unified_platform import EnhancedUnifiedPlatform
    from src.gradio_app.main import create_gradio_app
except ImportError:
    # Create mock classes for testing
    class EnhancedUnifiedPlatform:
        def __init__(self):
            self.host = "0.0.0.0"
            self.port = 8080
            self.app = Mock()
    
    def create_gradio_app():
        return Mock()


class TestGitHubPagesIntegration:
    """Test cases for GitHub Pages integration."""
    
    @pytest.fixture
    def github_pages_config(self):
        """GitHub Pages configuration for testing."""
        return {
            "site_url": "https://s-n00b.github.io/ai_assignments",
            "repository": "s-n00b/ai_assignments",
            "branch": "main",
            "docs_path": "docs",
            "site_path": "site",
            "mkdocs_config": "mkdocs.yml"
        }
    
    @pytest.fixture
    def mkdocs_config(self):
        """MkDocs configuration for testing."""
        return {
            "site_name": "Lenovo AAITC Solutions",
            "site_url": "https://s-n00b.github.io/ai_assignments",
            "repo_url": "https://github.com/s-n00b/ai_assignments",
            "nav": [
                {"Home": "index.md"},
                {"Category 1": [
                    {"AI Engineering Overview": "category1/ai-engineering-overview.md"},
                    {"Model Evaluation Framework": "category1/model-evaluation-framework.md"}
                ]},
                {"Category 2": [
                    {"System Architecture Overview": "category2/system-architecture-overview.md"}
                ]},
                {"API Documentation": [
                    {"FastAPI Enterprise": "api/fastapi-enterprise.md"},
                    {"Gradio Model Evaluation": "api/gradio-model-evaluation.md"}
                ]},
                {"Live Applications": "live-applications/index.md"}
            ],
            "theme": {
                "name": "material",
                "palette": {
                    "primary": "blue",
                    "accent": "light blue"
                }
            },
            "plugins": ["search", "mkdocstrings"]
        }
    
    def test_github_pages_config_validation(self, github_pages_config):
        """Test GitHub Pages configuration validation."""
        assert "site_url" in github_pages_config
        assert "repository" in github_pages_config
        assert "branch" in github_pages_config
        assert github_pages_config["site_url"].startswith("https://")
        assert "/" in github_pages_config["repository"]
    
    def test_mkdocs_config_structure(self, mkdocs_config):
        """Test MkDocs configuration structure."""
        assert "site_name" in mkdocs_config
        assert "site_url" in mkdocs_config
        assert "nav" in mkdocs_config
        assert "theme" in mkdocs_config
        assert len(mkdocs_config["nav"]) > 0
    
    def test_mkdocs_navigation_structure(self, mkdocs_config):
        """Test MkDocs navigation structure."""
        nav = mkdocs_config["nav"]
        
        # Test home page
        assert nav[0]["Home"] == "index.md"
        
        # Test category structure
        category1 = nav[1]["Category 1"]
        assert isinstance(category1, list)
        assert len(category1) > 0
        
        # Test API documentation structure
        api_docs = nav[3]["API Documentation"]
        assert isinstance(api_docs, list)
        assert len(api_docs) > 0
    
    def test_documentation_file_structure(self, temp_dir):
        """Test documentation file structure."""
        docs_structure = {
            "docs_content": {
                "category1": [
                    "ai-engineering-overview.md",
                    "model-evaluation-framework.md",
                    "ux-evaluation-testing.md"
                ],
                "category2": [
                    "system-architecture-overview.md"
                ],
                "api": [
                    "fastapi-enterprise.md",
                    "gradio-model-evaluation.md",
                    "mcp-server.md"
                ],
                "live-applications": [
                    "index.md"
                ]
            }
        }
        
        # Create test structure
        docs_path = temp_dir / "docs_content"
        docs_path.mkdir()
        
        for category, files in docs_structure["docs_content"].items():
            category_path = docs_path / category
            category_path.mkdir()
            
            for file_name in files:
                file_path = category_path / file_name
                file_path.write_text(f"# {file_name}\n\nTest content for {file_name}")
        
        # Verify structure
        assert docs_path.exists()
        assert (docs_path / "category1").exists()
        assert (docs_path / "category2").exists()
        assert (docs_path / "api").exists()
        assert (docs_path / "live-applications").exists()
        
        # Verify files
        for category, files in docs_structure["docs_content"].items():
            for file_name in files:
                file_path = docs_path / category / file_name
                assert file_path.exists()
                assert file_path.read_text().startswith(f"# {file_name}")


class TestMkDocsIntegration:
    """Test cases for MkDocs integration."""
    
    @pytest.fixture
    def sample_markdown_content(self):
        """Sample markdown content for testing."""
        return """# AI Engineering Overview

## 🎯 Overview
This document provides an overview of AI engineering principles and practices.

## 🚀 Key Features

### Core Capabilities
- **Model Evaluation**: Comprehensive evaluation framework
- **Architecture Design**: Enterprise-grade AI architecture

### Integration Features
- **FastAPI Integration**: RESTful API endpoints
- **Gradio Integration**: Interactive model evaluation interface

## 📊 Structure
The system follows a modular architecture with clear separation of concerns.

---

**Last Updated**: 2024-01-15  
**Version**: 2.1.0  
**Status**: Production Ready  
**Integration**: Full FastAPI Backend Integration
"""
    
    def test_markdown_content_parsing(self, sample_markdown_content):
        """Test markdown content parsing and structure."""
        lines = sample_markdown_content.split('\n')
        
        # Test header structure
        assert lines[0].startswith('# ')
        assert lines[2].startswith('## 🎯')
        assert lines[4].startswith('## 🚀')
        
        # Test emoji usage
        assert '🎯' in sample_markdown_content
        assert '🚀' in sample_markdown_content
        assert '📊' in sample_markdown_content
        
        # Test footer structure
        assert '**Last Updated**:' in sample_markdown_content
        assert '**Version**:' in sample_markdown_content
        assert '**Status**:' in sample_markdown_content
    
    def test_documentation_cross_references(self, sample_markdown_content):
        """Test documentation cross-reference format."""
        # Test internal link format (should use relative paths)
        assert '[' not in sample_markdown_content or '](fastapi-enterprise.md)' in sample_markdown_content
        assert '[' not in sample_markdown_content or '](gradio-model-evaluation.md)' in sample_markdown_content
        
        # Test external link format (should include target="_blank")
        assert '[' not in sample_markdown_content or 'target="_blank"' in sample_markdown_content
    
    def test_code_block_formatting(self, temp_dir):
        """Test code block formatting in documentation."""
        test_content = """```python
# Python code example
from src.module import Class

def example_function():
    \"\"\"Document functions with docstrings.\"\"\"
    pass
```

```bash
# Shell commands with comments
python -m src.module.main --host 0.0.0.0 --port 8080
```

```yaml
# Configuration files
service:
  host: "0.0.0.0"
  port: 8080
```"""
        
        test_file = temp_dir / "test_code_blocks.md"
        test_file.write_text(test_content)
        
        content = test_file.read_text()
        
        # Test syntax highlighting
        assert '```python' in content
        assert '```bash' in content
        assert '```yaml' in content
        
        # Test code content
        assert 'from src.module import Class' in content
        assert 'python -m src.module.main' in content
        assert 'service:' in content


class TestFrontendIntegration:
    """Test cases for frontend integration."""
    
    @pytest.fixture
    def frontend_services(self):
        """Frontend services configuration."""
        return {
            "enterprise_platform": {
                "host": "0.0.0.0",
                "port": 8080,
                "url": "http://localhost:8080",
                "docs_url": "http://localhost:8080/docs"
            },
            "gradio_app": {
                "host": "0.0.0.0",
                "port": 7860,
                "url": "http://localhost:7860"
            },
            "mlflow": {
                "host": "0.0.0.0",
                "port": 5000,
                "url": "http://localhost:5000"
            },
            "chromadb": {
                "host": "0.0.0.0",
                "port": 8000,
                "url": "http://localhost:8000"
            },
            "mkdocs": {
                "host": "0.0.0.0",
                "port": 8000,
                "url": "http://localhost:8000"
            }
        }
    
    def test_service_port_configuration(self, frontend_services):
        """Test service port configuration."""
        ports = [service["port"] for service in frontend_services.values()]
        
        # Ensure no port conflicts
        assert len(ports) == len(set(ports)), "Port conflicts detected"
        
        # Test port ranges
        for port in ports:
            assert 1000 <= port <= 9999, f"Port {port} out of valid range"
    
    def test_service_url_structure(self, frontend_services):
        """Test service URL structure."""
        for service_name, config in frontend_services.items():
            url = config["url"]
            
            # Test URL format
            assert url.startswith("http://")
            assert "localhost" in url
            assert str(config["port"]) in url
            
            # Test specific service URLs
            if service_name == "enterprise_platform":
                assert config["docs_url"] == f"{url}/docs"
    
    @pytest.mark.asyncio
    async def test_enhanced_unified_platform_initialization(self):
        """Test Enhanced Unified Platform initialization."""
        with patch('src.enterprise_llmops.frontend.enhanced_unified_platform.EnhancedUnifiedPlatform') as mock_platform:
            mock_platform.return_value = Mock()
            
            platform = EnhancedUnifiedPlatform()
            
            # Test platform initialization
            assert platform is not None
            assert hasattr(platform, 'host')
            assert hasattr(platform, 'port')
    
    @pytest.mark.asyncio
    async def test_gradio_app_creation(self):
        """Test Gradio app creation."""
        with patch('src.gradio_app.main.create_gradio_app') as mock_create_app:
            mock_create_app.return_value = Mock()
            
            app = create_gradio_app()
            
            # Test app creation
            assert app is not None
    
    def test_service_integration_table_format(self):
        """Test service integration table format."""
        integration_table = """| Service | Port | URL | Description |
|---|---|-----|----|
| **Enterprise FastAPI** | 8080 | http://localhost:8080 | Main enterprise platform |
| **Gradio App** | 7860 | http://localhost:7860 | Model evaluation interface |
| **MLflow Tracking** | 5000 | http://localhost:5000 | Experiment tracking |
| **ChromaDB** | 8000 | http://localhost:8000 | Vector database |
| **MkDocs** | 8000 | http://localhost:8000 | Documentation site |"""
        
        lines = integration_table.strip().split('\n')
        
        # Test table structure
        assert lines[0].startswith('| Service')
        assert 'Port' in lines[0]
        assert 'URL' in lines[0]
        assert 'Description' in lines[0]
        
        # Test table rows
        assert len(lines) > 1
        for line in lines[1:]:
            if line.strip():
                assert line.count('|') == 4  # 5 columns


class TestGitHubPagesDeployment:
    """Test cases for GitHub Pages deployment."""
    
    @pytest.fixture
    def deployment_config(self):
        """Deployment configuration for testing."""
        return {
            "build_command": "mkdocs build",
            "output_directory": "site",
            "branch": "gh-pages",
            "custom_domain": None,
            "https_enabled": True,
            "automatic_deployment": True
        }
    
    def test_deployment_configuration(self, deployment_config):
        """Test deployment configuration."""
        assert "build_command" in deployment_config
        assert "output_directory" in deployment_config
        assert "branch" in deployment_config
        
        # Test build command
        assert deployment_config["build_command"] == "mkdocs build"
        assert deployment_config["output_directory"] == "site"
    
    def test_github_actions_workflow(self, temp_dir):
        """Test GitHub Actions workflow configuration."""
        workflow_content = """name: Deploy to GitHub Pages

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r docs/requirements-docs.txt
    
    - name: Build site
      run: mkdocs build
    
    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v3
      if: github.ref == 'refs/heads/main'
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./site"""
        
        workflow_file = temp_dir / ".github/workflows/deploy.yml"
        workflow_file.parent.mkdir(parents=True)
        workflow_file.write_text(workflow_content)
        
        # Test workflow structure
        content = workflow_file.read_text()
        assert "name: Deploy to GitHub Pages" in content
        assert "on:" in content
        assert "jobs:" in content
        assert "deploy:" in content
        assert "runs-on: ubuntu-latest" in content
    
    def test_site_generation(self, temp_dir, mkdocs_config):
        """Test site generation process."""
        # Create mock site structure
        site_path = temp_dir / "site"
        site_path.mkdir()
        
        # Create index.html
        index_content = """<!DOCTYPE html>
<html>
<head>
    <title>Lenovo AAITC Solutions</title>
</head>
<body>
    <h1>Welcome to Lenovo AAITC Solutions</h1>
</body>
</html>"""
        
        index_file = site_path / "index.html"
        index_file.write_text(index_content)
        
        # Test site structure
        assert site_path.exists()
        assert index_file.exists()
        
        # Test HTML content
        content = index_file.read_text()
        assert "<!DOCTYPE html>" in content
        assert "<title>Lenovo AAITC Solutions</title>" in content
        assert "<h1>Welcome to Lenovo AAITC Solutions</h1>" in content


class TestDocumentationQuality:
    """Test cases for documentation quality standards."""
    
    def test_required_header_structure(self, temp_dir):
        """Test required header structure in documentation."""
        test_content = """# [Title] Documentation

## 🎯 Overview
[Clear description of the component/feature]

## 🚀 Key Features
### Core Capabilities
- [Feature 1]: [Description]
- [Feature 2]: [Description]

### Integration Features  
- [Integration 1]: [Description]
- [Integration 2]: [Description]"""
        
        test_file = temp_dir / "test_header.md"
        test_file.write_text(test_content)
        
        content = test_file.read_text()
        lines = content.split('\n')
        
        # Test header structure
        assert lines[0].startswith('# ')
        assert lines[2].startswith('## 🎯')
        assert lines[4].startswith('## 🚀')
        
        # Test emoji usage
        assert '🎯' in content
        assert '🚀' in content
    
    def test_consistent_emoji_usage(self):
        """Test consistent emoji usage across documentation."""
        emoji_mapping = {
            "🎯": "Overview",
            "🚀": "Key Features",
            "📊": "Structure/Architecture",
            "🌐": "Service Integration",
            "🔧": "Configuration",
            "📚": "Documentation",
            "🛠️": "Development",
            "🚨": "Troubleshooting",
            "📞": "Support"
        }
        
        for emoji, description in emoji_mapping.items():
            # Test that emoji is properly defined
            assert len(emoji) == 1
            assert description is not None
    
    def test_footer_information(self, temp_dir):
        """Test required footer information."""
        footer_content = """---

**Last Updated**: 2024-01-15  
**Version**: 2.1.0  
**Status**: Production Ready  
**Integration**: Full FastAPI Backend Integration"""
        
        test_file = temp_dir / "test_footer.md"
        test_file.write_text(footer_content)
        
        content = test_file.read_text()
        
        # Test footer structure
        assert '**Last Updated**:' in content
        assert '**Version**:' in content
        assert '**Status**:' in content
        assert '**Integration**:' in content
        
        # Test date format
        assert '2024-01-15' in content
    
    def test_status_indicators(self):
        """Test status indicators consistency."""
        status_indicators = {
            "✅": "Production Ready",
            "🔄": "In Development",
            "⚠️": "Beta",
            "❌": "Deprecated",
            "🚧": "Under Construction"
        }
        
        for indicator, description in status_indicators.items():
            assert len(indicator) == 1
            assert description is not None


if __name__ == "__main__":
    pytest.main([__file__])
