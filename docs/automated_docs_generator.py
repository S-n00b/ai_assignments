#!/usr/bin/env python3
"""
Automated Documentation Generator for Lenovo AAITC Solutions

This script automatically extracts documentation from code comments, docstrings,
and other sources to generate comprehensive documentation for the Jekyll site.
"""

import os
import sys
import ast
import inspect
import subprocess
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import re
import logging

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentationExtractor:
    """Extract documentation from various sources."""
    
    def __init__(self, src_path: str):
        self.src_path = Path(src_path)
        self.documentation = {}
        
    def extract_all_documentation(self) -> Dict[str, Any]:
        """Extract documentation from all sources."""
        logger.info("Starting documentation extraction...")
        
        # Extract from Python modules
        self.extract_python_documentation()
        
        # Extract from configuration files
        self.extract_config_documentation()
        
        # Extract from README files
        self.extract_readme_documentation()
        
        # Extract from test files
        self.extract_test_documentation()
        
        # Extract from scripts
        self.extract_script_documentation()
        
        logger.info("Documentation extraction completed.")
        return self.documentation
    
    def extract_python_documentation(self):
        """Extract documentation from Python modules."""
        logger.info("Extracting Python documentation...")
        
        python_files = list(self.src_path.rglob("*.py"))
        
        for py_file in python_files:
            try:
                module_doc = self.extract_module_documentation(py_file)
                if module_doc:
                    self.documentation[f"python_{py_file.stem}"] = module_doc
            except Exception as e:
                logger.warning(f"Error extracting documentation from {py_file}: {e}")
    
    def extract_module_documentation(self, py_file: Path) -> Optional[Dict[str, Any]]:
        """Extract documentation from a single Python module."""
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            module_doc = {
                'file_path': str(py_file.relative_to(self.src_path.parent)),
                'module_name': py_file.stem,
                'docstring': ast.get_docstring(tree),
                'classes': [],
                'functions': [],
                'constants': [],
                'imports': [],
                'last_modified': datetime.fromtimestamp(py_file.stat().st_mtime).isoformat()
            }
            
            # Extract classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_doc = self.extract_class_documentation(node)
                    module_doc['classes'].append(class_doc)
                
                elif isinstance(node, ast.FunctionDef) and not self._is_method(node, tree):
                    func_doc = self.extract_function_documentation(node)
                    module_doc['functions'].append(func_doc)
                
                elif isinstance(node, ast.Assign):
                    const_doc = self.extract_constant_documentation(node)
                    if const_doc:
                        module_doc['constants'].append(const_doc)
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_doc = self.extract_import_documentation(node)
                    module_doc['imports'].append(import_doc)
            
            return module_doc if module_doc['docstring'] or module_doc['classes'] or module_doc['functions'] else None
            
        except Exception as e:
            logger.warning(f"Error parsing {py_file}: {e}")
            return None
    
    def extract_class_documentation(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Extract documentation from a class definition."""
        class_doc = {
            'name': node.name,
            'docstring': ast.get_docstring(node),
            'bases': [self._get_name(base) for base in node.bases],
            'methods': [],
            'properties': [],
            'decorators': [self._get_name(dec) for dec in node.decorator_list]
        }
        
        # Extract methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_doc = self.extract_function_documentation(item)
                class_doc['methods'].append(method_doc)
            elif isinstance(item, ast.Assign):
                prop_doc = self.extract_property_documentation(item)
                if prop_doc:
                    class_doc['properties'].append(prop_doc)
        
        return class_doc
    
    def extract_function_documentation(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Extract documentation from a function definition."""
        func_doc = {
            'name': node.name,
            'docstring': ast.get_docstring(node),
            'args': [],
            'returns': None,
            'decorators': [self._get_name(dec) for dec in node.decorator_list],
            'is_async': isinstance(node, ast.AsyncFunctionDef)
        }
        
        # Extract arguments
        for arg in node.args.args:
            arg_doc = {
                'name': arg.arg,
                'annotation': self._get_annotation(arg.annotation) if arg.annotation else None,
                'default': None
            }
            func_doc['args'].append(arg_doc)
        
        # Extract return annotation
        if node.returns:
            func_doc['returns'] = self._get_annotation(node.returns)
        
        return func_doc
    
    def extract_constant_documentation(self, node: ast.Assign) -> Optional[Dict[str, Any]]:
        """Extract documentation from constant assignments."""
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            const_name = node.targets[0].id
            
            # Only document constants that are all uppercase
            if const_name.isupper():
                return {
                    'name': const_name,
                    'value': self._get_constant_value(node.value),
                    'annotation': None
                }
        
        return None
    
    def extract_property_documentation(self, node: ast.Assign) -> Optional[Dict[str, Any]]:
        """Extract documentation from property assignments."""
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            return {
                'name': node.targets[0].id,
                'value': self._get_constant_value(node.value),
                'annotation': None
            }
        
        return None
    
    def extract_import_documentation(self, node: ast.Import) -> Dict[str, Any]:
        """Extract documentation from import statements."""
        if isinstance(node, ast.Import):
            return {
                'type': 'import',
                'module': node.names[0].name,
                'alias': node.names[0].asname
            }
        elif isinstance(node, ast.ImportFrom):
            return {
                'type': 'from_import',
                'module': node.module,
                'names': [name.name for name in node.names],
                'level': node.level
            }
    
    def extract_config_documentation(self):
        """Extract documentation from configuration files."""
        logger.info("Extracting configuration documentation...")
        
        config_files = [
            'config/requirements.txt',
            'config/requirements-testing.txt',
            'config/pytest.ini',
            'config/Makefile'
        ]
        
        for config_file in config_files:
            config_path = self.src_path.parent / config_file
            if config_path.exists():
                try:
                    config_doc = self.extract_config_file_documentation(config_path)
                    if config_doc:
                        self.documentation[f"config_{config_path.stem}"] = config_doc
                except Exception as e:
                    logger.warning(f"Error extracting config from {config_path}: {e}")
    
    def extract_config_file_documentation(self, config_path: Path) -> Dict[str, Any]:
        """Extract documentation from a configuration file."""
        config_doc = {
            'file_path': str(config_path.relative_to(self.src_path.parent)),
            'file_type': config_path.suffix,
            'content': [],
            'last_modified': datetime.fromtimestamp(config_path.stat().st_mtime).isoformat()
        }
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if config_path.suffix == '.txt':
                # Requirements file
                config_doc['content'] = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')]
            elif config_path.suffix == '.ini':
                # INI file
                config_doc['content'] = self._parse_ini_content(content)
            elif config_path.name == 'Makefile':
                # Makefile
                config_doc['content'] = self._parse_makefile_content(content)
            else:
                config_doc['content'] = content.split('\n')
            
            return config_doc
            
        except Exception as e:
            logger.warning(f"Error reading config file {config_path}: {e}")
            return None
    
    def extract_readme_documentation(self):
        """Extract documentation from README files."""
        logger.info("Extracting README documentation...")
        
        readme_files = list(self.src_path.parent.rglob("README*.md"))
        
        for readme_file in readme_files:
            try:
                readme_doc = self.extract_readme_file_documentation(readme_file)
                if readme_doc:
                    self.documentation[f"readme_{readme_file.stem}"] = readme_doc
            except Exception as e:
                logger.warning(f"Error extracting README from {readme_file}: {e}")
    
    def extract_readme_file_documentation(self, readme_path: Path) -> Dict[str, Any]:
        """Extract documentation from a README file."""
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            readme_doc = {
                'file_path': str(readme_path.relative_to(self.src_path.parent)),
                'title': self._extract_title_from_markdown(content),
                'sections': self._extract_sections_from_markdown(content),
                'last_modified': datetime.fromtimestamp(readme_path.stat().st_mtime).isoformat()
            }
            
            return readme_doc
            
        except Exception as e:
            logger.warning(f"Error reading README file {readme_path}: {e}")
            return None
    
    def extract_test_documentation(self):
        """Extract documentation from test files."""
        logger.info("Extracting test documentation...")
        
        test_files = list(self.src_path.parent.rglob("test_*.py"))
        test_files.extend(list(self.src_path.parent.rglob("*_test.py")))
        
        for test_file in test_files:
            try:
                test_doc = self.extract_test_file_documentation(test_file)
                if test_doc:
                    self.documentation[f"test_{test_file.stem}"] = test_doc
            except Exception as e:
                logger.warning(f"Error extracting test from {test_file}: {e}")
    
    def extract_test_file_documentation(self, test_path: Path) -> Dict[str, Any]:
        """Extract documentation from a test file."""
        try:
            with open(test_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            test_doc = {
                'file_path': str(test_path.relative_to(self.src_path.parent)),
                'module_name': test_path.stem,
                'docstring': ast.get_docstring(tree),
                'test_functions': [],
                'test_classes': [],
                'last_modified': datetime.fromtimestamp(test_path.stat().st_mtime).isoformat()
            }
            
            # Extract test functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    test_func_doc = self.extract_function_documentation(node)
                    test_doc['test_functions'].append(test_func_doc)
                elif isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
                    test_class_doc = self.extract_class_documentation(node)
                    test_doc['test_classes'].append(test_class_doc)
            
            return test_doc if test_doc['test_functions'] or test_doc['test_classes'] else None
            
        except Exception as e:
            logger.warning(f"Error parsing test file {test_path}: {e}")
            return None
    
    def extract_script_documentation(self):
        """Extract documentation from script files."""
        logger.info("Extracting script documentation...")
        
        script_files = list(self.src_path.parent.rglob("*.py"))
        script_files = [f for f in script_files if f.name not in ['__init__.py'] and 'test' not in f.name]
        
        for script_file in script_files:
            try:
                script_doc = self.extract_script_file_documentation(script_file)
                if script_doc:
                    self.documentation[f"script_{script_file.stem}"] = script_doc
            except Exception as e:
                logger.warning(f"Error extracting script from {script_file}: {e}")
    
    def extract_script_file_documentation(self, script_path: Path) -> Dict[str, Any]:
        """Extract documentation from a script file."""
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            script_doc = {
                'file_path': str(script_path.relative_to(self.src_path.parent)),
                'script_name': script_path.stem,
                'docstring': ast.get_docstring(tree),
                'main_function': None,
                'functions': [],
                'classes': [],
                'last_modified': datetime.fromtimestamp(script_path.stat().st_mtime).isoformat()
            }
            
            # Extract main function and other functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.name == 'main':
                        script_doc['main_function'] = self.extract_function_documentation(node)
                    else:
                        func_doc = self.extract_function_documentation(node)
                        script_doc['functions'].append(func_doc)
                elif isinstance(node, ast.ClassDef):
                    class_doc = self.extract_class_documentation(node)
                    script_doc['classes'].append(class_doc)
            
            return script_doc if script_doc['docstring'] or script_doc['main_function'] else None
            
        except Exception as e:
            logger.warning(f"Error parsing script file {script_path}: {e}")
            return None
    
    # Helper methods
    def _is_method(self, node: ast.FunctionDef, tree: ast.AST) -> bool:
        """Check if a function is a method (defined inside a class)."""
        for parent in ast.walk(tree):
            if isinstance(parent, ast.ClassDef):
                for item in parent.body:
                    if item == node:
                        return True
        return False
    
    def _get_name(self, node: ast.AST) -> str:
        """Get the name of an AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        else:
            return str(node)
    
    def _get_annotation(self, node: ast.AST) -> str:
        """Get the type annotation of an AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return str(node.value)
        else:
            return str(node)
    
    def _get_constant_value(self, node: ast.AST) -> Any:
        """Get the value of a constant AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.NameConstant):
            return node.value
        else:
            return str(node)
    
    def _parse_ini_content(self, content: str) -> List[Dict[str, str]]:
        """Parse INI file content."""
        sections = []
        current_section = None
        
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1]
                sections.append({'section': current_section, 'options': []})
            elif line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                if sections:
                    sections[-1]['options'].append({'key': key.strip(), 'value': value.strip()})
        
        return sections
    
    def _parse_makefile_content(self, content: str) -> List[Dict[str, str]]:
        """Parse Makefile content."""
        targets = []
        
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and ':' in line:
                target, dependencies = line.split(':', 1)
                targets.append({
                    'target': target.strip(),
                    'dependencies': dependencies.strip()
                })
        
        return targets
    
    def _extract_title_from_markdown(self, content: str) -> str:
        """Extract title from markdown content."""
        lines = content.split('\n')
        for line in lines:
            if line.startswith('# '):
                return line[2:].strip()
        return "Untitled"
    
    def _extract_sections_from_markdown(self, content: str) -> List[Dict[str, str]]:
        """Extract sections from markdown content."""
        sections = []
        current_section = None
        
        for line in content.split('\n'):
            if line.startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('# ').strip()
                current_section = {
                    'level': level,
                    'title': title,
                    'content': []
                }
                sections.append(current_section)
            elif current_section and line.strip():
                current_section['content'].append(line)
        
        return sections


class JekyllPostGenerator:
    """Generate Jekyll posts from extracted documentation."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_posts(self, documentation: Dict[str, Any]):
        """Generate Jekyll posts from documentation."""
        logger.info("Generating Jekyll posts...")
        
        for doc_key, doc_data in documentation.items():
            try:
                post_content = self.generate_post_content(doc_key, doc_data)
                if post_content:
                    self.save_post(doc_key, post_content)
            except Exception as e:
                logger.warning(f"Error generating post for {doc_key}: {e}")
    
    def generate_post_content(self, doc_key: str, doc_data: Dict[str, Any]) -> Optional[str]:
        """Generate Jekyll post content from documentation data."""
        if doc_key.startswith('python_'):
            return self.generate_python_post(doc_data)
        elif doc_key.startswith('config_'):
            return self.generate_config_post(doc_data)
        elif doc_key.startswith('readme_'):
            return self.generate_readme_post(doc_data)
        elif doc_key.startswith('test_'):
            return self.generate_test_post(doc_data)
        elif doc_key.startswith('script_'):
            return self.generate_script_post(doc_data)
        
        return None
    
    def generate_python_post(self, doc_data: Dict[str, Any]) -> str:
        """Generate Jekyll post for Python module documentation."""
        title = f"Python Module: {doc_data['module_name']}"
        categories = ["Documentation", "Python", "API"]
        tags = ["Python", "Module", doc_data['module_name']]
        
        content = f"""# {title}

## Overview

{doc_data['docstring'] or 'No module docstring available.'}

## File Information

- **File Path**: `{doc_data['file_path']}`
- **Last Modified**: {doc_data['last_modified']}

## Classes

"""
        
        for class_doc in doc_data['classes']:
            content += f"""### {class_doc['name']}

{class_doc['docstring'] or 'No class docstring available.'}

**Bases**: {', '.join(class_doc['bases']) if class_doc['bases'] else 'None'}

**Methods**:
"""
            for method in class_doc['methods']:
                content += f"- `{method['name']}()`: {method['docstring'] or 'No docstring available.'}\n"
            
            content += "\n"
        
        content += """## Functions

"""
        
        for func_doc in doc_data['functions']:
            content += f"""### {func_doc['name']}

{func_doc['docstring'] or 'No function docstring available.'}

**Arguments**:
"""
            for arg in func_doc['args']:
                content += f"- `{arg['name']}`: {arg['annotation'] or 'No type annotation'}\n"
            
            if func_doc['returns']:
                content += f"**Returns**: {func_doc['returns']}\n"
            
            content += "\n"
        
        return self.create_jekyll_post(title, categories, tags, content)
    
    def generate_config_post(self, doc_data: Dict[str, Any]) -> str:
        """Generate Jekyll post for configuration file documentation."""
        title = f"Configuration: {doc_data['file_path']}"
        categories = ["Documentation", "Configuration"]
        tags = ["Configuration", "Setup", doc_data['file_type']]
        
        content = f"""# {title}

## Overview

Configuration file: `{doc_data['file_path']}`

**File Type**: {doc_data['file_type']}
**Last Modified**: {doc_data['last_modified']}

## Content

"""
        
        if isinstance(doc_data['content'], list):
            for item in doc_data['content']:
                if isinstance(item, dict):
                    content += f"- **{item.get('key', 'Item')}**: {item.get('value', '')}\n"
                else:
                    content += f"- {item}\n"
        else:
            content += f"```\n{doc_data['content']}\n```\n"
        
        return self.create_jekyll_post(title, categories, tags, content)
    
    def generate_readme_post(self, doc_data: Dict[str, Any]) -> str:
        """Generate Jekyll post for README file documentation."""
        title = f"README: {doc_data['title']}"
        categories = ["Documentation", "README"]
        tags = ["README", "Documentation", "Guide"]
        
        content = f"""# {title}

## Overview

{doc_data['title']}

**File Path**: `{doc_data['file_path']}`
**Last Modified**: {doc_data['last_modified']}

## Sections

"""
        
        for section in doc_data['sections']:
            content += f"""### {section['title']}

{chr(10).join(section['content'])}

"""
        
        return self.create_jekyll_post(title, categories, tags, content)
    
    def generate_test_post(self, doc_data: Dict[str, Any]) -> str:
        """Generate Jekyll post for test file documentation."""
        title = f"Test Module: {doc_data['module_name']}"
        categories = ["Documentation", "Testing"]
        tags = ["Testing", "Unit Tests", doc_data['module_name']]
        
        content = f"""# {title}

## Overview

{doc_data['docstring'] or 'No module docstring available.'}

**File Path**: `{doc_data['file_path']}`
**Last Modified**: {doc_data['last_modified']}

## Test Functions

"""
        
        for test_func in doc_data['test_functions']:
            content += f"""### {test_func['name']}

{test_func['docstring'] or 'No test docstring available.'}

"""
        
        content += """## Test Classes

"""
        
        for test_class in doc_data['test_classes']:
            content += f"""### {test_class['name']}

{test_class['docstring'] or 'No class docstring available.'}

"""
        
        return self.create_jekyll_post(title, categories, tags, content)
    
    def generate_script_post(self, doc_data: Dict[str, Any]) -> str:
        """Generate Jekyll post for script file documentation."""
        title = f"Script: {doc_data['script_name']}"
        categories = ["Documentation", "Scripts"]
        tags = ["Script", "Python", doc_data['script_name']]
        
        content = f"""# {title}

## Overview

{doc_data['docstring'] or 'No script docstring available.'}

**File Path**: `{doc_data['file_path']}`
**Last Modified**: {doc_data['last_modified']}

## Main Function

"""
        
        if doc_data['main_function']:
            content += f"""### {doc_data['main_function']['name']}

{doc_data['main_function']['docstring'] or 'No main function docstring available.'}

"""
        
        content += """## Functions

"""
        
        for func_doc in doc_data['functions']:
            content += f"""### {func_doc['name']}

{func_doc['docstring'] or 'No function docstring available.'}

"""
        
        return self.create_jekyll_post(title, categories, tags, content)
    
    def create_jekyll_post(self, title: str, categories: List[str], tags: List[str], content: str) -> str:
        """Create a Jekyll post with proper front matter."""
        front_matter = f"""---
layout: post
title: "{title}"
date: 2025-09-18 10:00:00 -0400
categories: {categories}
tags: {tags}
author: Lenovo AAITC Team
---

{content}
"""
        return front_matter
    
    def save_post(self, doc_key: str, content: str):
        """Save Jekyll post to file."""
        safe_title = doc_key.replace('_', '-').lower()
        filename = f"2025-09-18-{safe_title}.md"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Generated post: {filename}")


def main():
    """Main function to generate automated documentation."""
    logger.info("Starting automated documentation generation...")
    
    try:
        # Set up paths
        src_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
        output_dir = os.path.join(os.path.dirname(__file__), '_posts', 'auto-generated')
        
        # Extract documentation
        extractor = DocumentationExtractor(src_path)
        documentation = extractor.extract_all_documentation()
        
        # Generate Jekyll posts
        generator = JekyllPostGenerator(output_dir)
        generator.generate_posts(documentation)
        
        # Save documentation metadata
        metadata_file = os.path.join(output_dir, 'documentation_metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(documentation, f, indent=2, default=str)
        
        logger.info(f"Generated {len(documentation)} documentation entries")
        logger.info(f"Documentation saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error generating automated documentation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
