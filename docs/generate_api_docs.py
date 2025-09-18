#!/usr/bin/env python3
"""
API Documentation Generator for Lenovo AAITC Solutions

This script automatically generates API documentation from Python docstrings
and integrates it with the Jekyll documentation site.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import json
import yaml
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def run_command(command, cwd=None):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            cwd=cwd, 
            capture_output=True, 
            text=True, 
            check=True
        )
        return result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Return code: {e.returncode}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        raise

def setup_sphinx_environment():
    """Set up the Sphinx documentation environment."""
    print("Setting up Sphinx environment...")
    
    # Install required packages
    requirements_file = os.path.join(os.path.dirname(__file__), 'requirements-docs.txt')
    if os.path.exists(requirements_file):
        run_command(f"pip install -r {requirements_file}")
    
    # Create necessary directories
    sphinx_dir = os.path.join(os.path.dirname(__file__), 'sphinx')
    os.makedirs(os.path.join(sphinx_dir, '_static'), exist_ok=True)
    os.makedirs(os.path.join(sphinx_dir, '_templates'), exist_ok=True)
    os.makedirs(os.path.join(sphinx_dir, '_build'), exist_ok=True)

def generate_sphinx_docs():
    """Generate Sphinx documentation."""
    print("Generating Sphinx documentation...")
    
    sphinx_dir = os.path.join(os.path.dirname(__file__), 'sphinx')
    
    # Clean previous build
    build_dir = os.path.join(sphinx_dir, '_build')
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    
    # Generate documentation
    run_command("sphinx-build -b html . _build", cwd=sphinx_dir)
    
    return build_dir

def convert_sphinx_to_jekyll(sphinx_build_dir):
    """Convert Sphinx HTML to Jekyll-compatible format."""
    print("Converting Sphinx documentation to Jekyll format...")
    
    # Create Jekyll posts directory for API docs
    jekyll_posts_dir = os.path.join(os.path.dirname(__file__), '_posts', 'api-docs')
    os.makedirs(jekyll_posts_dir, exist_ok=True)
    
    # Read the Sphinx HTML files and convert them
    html_files = []
    for root, dirs, files in os.walk(sphinx_build_dir):
        for file in files:
            if file.endswith('.html') and file != 'index.html':
                html_files.append(os.path.join(root, file))
    
    # Convert each HTML file to a Jekyll post
    for html_file in html_files:
        convert_html_to_jekyll_post(html_file, jekyll_posts_dir)

def convert_html_to_jekyll_post(html_file, jekyll_posts_dir):
    """Convert a single HTML file to a Jekyll post."""
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Extract title from HTML
        title = extract_title_from_html(html_content)
        if not title:
            return
        
        # Create Jekyll post filename
        safe_title = title.lower().replace(' ', '-').replace('_', '-')
        post_filename = f"2025-09-18-{safe_title}.md"
        post_path = os.path.join(jekyll_posts_dir, post_filename)
        
        # Convert HTML to Markdown (simplified)
        markdown_content = html_to_markdown(html_content)
        
        # Create Jekyll post content
        jekyll_post = f"""---
layout: post
title: "{title}"
date: 2025-09-18 10:00:00 -0400
categories: [Documentation, API]
tags: [API, Documentation, {title.replace(' ', ', ')}]
author: Lenovo AAITC Team
---

# {title}

{markdown_content}
"""
        
        # Write the Jekyll post
        with open(post_path, 'w', encoding='utf-8') as f:
            f.write(jekyll_post)
        
        print(f"Created Jekyll post: {post_filename}")
        
    except Exception as e:
        print(f"Error converting {html_file}: {e}")

def extract_title_from_html(html_content):
    """Extract title from HTML content."""
    import re
    
    # Try to find title in various places
    title_patterns = [
        r'<title>(.*?)</title>',
        r'<h1[^>]*>(.*?)</h1>',
        r'<h2[^>]*>(.*?)</h2>',
    ]
    
    for pattern in title_patterns:
        match = re.search(pattern, html_content, re.IGNORECASE | re.DOTALL)
        if match:
            title = match.group(1).strip()
            # Clean up the title
            title = re.sub(r'<[^>]+>', '', title)  # Remove HTML tags
            title = title.replace('&nbsp;', ' ').replace('&amp;', '&')
            if title and len(title) > 3:
                return title
    
    return None

def html_to_markdown(html_content):
    """Convert HTML content to Markdown (simplified)."""
    import re
    
    # Remove HTML tags and convert to basic Markdown
    # This is a simplified conversion - in production, you'd want to use a proper HTML to Markdown converter
    
    # Remove script and style tags
    html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    
    # Convert headers
    html_content = re.sub(r'<h1[^>]*>(.*?)</h1>', r'# \1', html_content, flags=re.IGNORECASE | re.DOTALL)
    html_content = re.sub(r'<h2[^>]*>(.*?)</h2>', r'## \1', html_content, flags=re.IGNORECASE | re.DOTALL)
    html_content = re.sub(r'<h3[^>]*>(.*?)</h3>', r'### \1', html_content, flags=re.IGNORECASE | re.DOTALL)
    html_content = re.sub(r'<h4[^>]*>(.*?)</h4>', r'#### \1', html_content, flags=re.IGNORECASE | re.DOTALL)
    
    # Convert paragraphs
    html_content = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n\n', html_content, flags=re.IGNORECASE | re.DOTALL)
    
    # Convert lists
    html_content = re.sub(r'<li[^>]*>(.*?)</li>', r'- \1', html_content, flags=re.IGNORECASE | re.DOTALL)
    
    # Convert code blocks
    html_content = re.sub(r'<pre[^>]*><code[^>]*>(.*?)</code></pre>', r'```\n\1\n```', html_content, flags=re.DOTALL | re.IGNORECASE)
    html_content = re.sub(r'<code[^>]*>(.*?)</code>', r'`\1`', html_content, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove remaining HTML tags
    html_content = re.sub(r'<[^>]+>', '', html_content)
    
    # Clean up whitespace
    html_content = re.sub(r'\n\s*\n\s*\n', '\n\n', html_content)
    html_content = html_content.strip()
    
    return html_content

def create_api_index_post():
    """Create an index post for the API documentation."""
    print("Creating API documentation index post...")
    
    jekyll_posts_dir = os.path.join(os.path.dirname(__file__), '_posts', 'api-docs')
    os.makedirs(jekyll_posts_dir, exist_ok=True)
    
    index_content = """---
layout: post
title: "API Documentation Index"
date: 2025-09-18 10:00:00 -0400
categories: [Documentation, API]
tags: [API, Documentation, Index, Reference]
author: Lenovo AAITC Team
---

# API Documentation Index

Welcome to the comprehensive API documentation for the Lenovo AAITC Solutions framework.

## AI Architecture Module

The AI Architecture module provides comprehensive AI architecture capabilities for enterprise-scale AI systems.

### Core Components

- **HybridAIPlatform**: Enterprise hybrid AI platform architecture
- **ModelLifecycleManager**: Complete MLOps pipeline and model lifecycle management
- **AgenticComputingFramework**: Multi-agent systems and intelligent orchestration
- **RAGSystem**: Advanced retrieval-augmented generation with enterprise features

### Key Features

- Multi-cloud deployment strategies
- Intelligent workload distribution
- Enterprise integration capabilities
- Comprehensive monitoring and alerting
- Security and compliance frameworks

## Model Evaluation Module

The Model Evaluation module provides comprehensive model evaluation capabilities for AI systems.

### Core Components

- **EvaluationPipeline**: Complete model evaluation pipeline
- **BiasDetection**: Advanced bias detection and mitigation
- **RobustnessTesting**: Comprehensive robustness testing framework
- **PromptRegistries**: Centralized prompt management

## Utilities Module

The Utilities module provides shared utilities and common functionality.

### Core Components

- **LoggingSystem**: Comprehensive multi-layer logging architecture
- **VisualizationUtils**: Plotting and chart generation utilities
- **DataUtils**: Data processing and manipulation utilities
- **ConfigUtils**: Configuration management utilities

## Gradio Application Module

The Gradio Application module provides a web-based interface for the framework.

### Core Components

- **Main Application**: Primary web interface
- **Components**: Reusable UI components
- **MCP Server**: Model Context Protocol server

## Getting Started

1. **Installation**: Follow the installation guide
2. **Quick Start**: Use the quick start guide
3. **Examples**: Explore the example notebooks
4. **API Reference**: Browse the detailed API documentation

## Support

For technical support or questions:
- **GitHub Issues**: Create an issue in the repository
- **Documentation**: Check the setup guide posts
- **Team Contact**: aaitc-support@lenovo.com

---

*This documentation is automatically generated from the source code and updated with each release.*
"""
    
    index_path = os.path.join(jekyll_posts_dir, '2025-09-18-api-documentation-index.md')
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(index_content)
    
    print(f"Created API documentation index: {index_path}")

def main():
    """Main function to generate API documentation."""
    print("Starting API documentation generation...")
    
    try:
        # Set up Sphinx environment
        setup_sphinx_environment()
        
        # Generate Sphinx documentation
        sphinx_build_dir = generate_sphinx_docs()
        
        # Convert to Jekyll format
        convert_sphinx_to_jekyll(sphinx_build_dir)
        
        # Create API index post
        create_api_index_post()
        
        print("API documentation generation completed successfully!")
        
    except Exception as e:
        print(f"Error generating API documentation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
