"""
Configuration Utilities for AI Model Evaluation and Architecture

This module provides comprehensive configuration management utilities
for both Assignment 1 (Model Evaluation) and Assignment 2 (AI Architecture) solutions.

Key Features:
- Configuration file management
- Environment variable handling
- Configuration validation
- Dynamic configuration updates
- Configuration templates
"""

import json
import yaml
import os
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import logging
from datetime import datetime
import copy

logger = logging.getLogger(__name__)


class ConfigFormat(Enum):
    """Configuration file formats"""
    JSON = "json"
    YAML = "yaml"
    ENV = "env"
    INI = "ini"


@dataclass
class ConfigValidationRule:
    """Configuration validation rule"""
    key: str
    required: bool = True
    data_type: type = str
    default_value: Any = None
    allowed_values: List[Any] = field(default_factory=list)
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    custom_validator: Optional[callable] = None


class ConfigUtils:
    """
    Comprehensive configuration management utilities.
    
    This class provides extensive configuration management capabilities including:
    - Configuration file loading and saving
    - Environment variable integration
    - Configuration validation and schema enforcement
    - Dynamic configuration updates
    - Configuration templates and defaults
    """
    
    def __init__(self, config_directory: str = "./config"):
        """
        Initialize configuration utilities.
        
        Args:
            config_directory: Directory for configuration files
        """
        self.config_directory = Path(config_directory)
        self.config_directory.mkdir(exist_ok=True)
        
        self.config_cache = {}
        self.validation_rules = {}
        self.config_templates = {}
        
        logger.info("Configuration utilities initialized")
    
    def load_config(
        self,
        config_name: str,
        format: ConfigFormat = ConfigFormat.JSON,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_name: Name of the configuration file
            format: Configuration file format
            validate: Whether to validate the configuration
            
        Returns:
            Configuration dictionary
        """
        config_path = self.config_directory / f"{config_name}.{format.value}"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                if format == ConfigFormat.JSON:
                    config = json.load(f)
                elif format == ConfigFormat.YAML:
                    config = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            # Cache the configuration
            self.config_cache[config_name] = config
            
            # Validate if requested
            if validate and config_name in self.validation_rules:
                self.validate_config(config, config_name)
            
            logger.info(f"Configuration loaded: {config_name}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration {config_name}: {str(e)}")
            raise
    
    def save_config(
        self,
        config: Dict[str, Any],
        config_name: str,
        format: ConfigFormat = ConfigFormat.JSON,
        backup: bool = True
    ) -> str:
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary
            config_name: Name of the configuration file
            format: Configuration file format
            backup: Whether to create a backup of existing file
            
        Returns:
            Path to saved configuration file
        """
        config_path = self.config_directory / f"{config_name}.{format.value}"
        
        # Create backup if requested and file exists
        if backup and config_path.exists():
            backup_path = config_path.with_suffix(f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.backup")
            config_path.rename(backup_path)
            logger.info(f"Backup created: {backup_path}")
        
        try:
            with open(config_path, 'w') as f:
                if format == ConfigFormat.JSON:
                    json.dump(config, f, indent=2, default=str)
                elif format == ConfigFormat.YAML:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            # Update cache
            self.config_cache[config_name] = config
            
            logger.info(f"Configuration saved: {config_path}")
            return str(config_path)
            
        except Exception as e:
            logger.error(f"Failed to save configuration {config_name}: {str(e)}")
            raise
    
    def get_config_value(
        self,
        config_name: str,
        key: str,
        default: Any = None,
        use_env: bool = True
    ) -> Any:
        """
        Get a configuration value with fallback to environment variables.
        
        Args:
            config_name: Name of the configuration
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            use_env: Whether to check environment variables
            
        Returns:
            Configuration value
        """
        # Check cache first
        if config_name in self.config_cache:
            config = self.config_cache[config_name]
        else:
            # Try to load from file
            try:
                config = self.load_config(config_name)
            except FileNotFoundError:
                config = {}
        
        # Navigate through nested keys
        value = self._get_nested_value(config, key)
        
        # Check environment variable if value not found and use_env is True
        if value is None and use_env:
            env_key = f"{config_name.upper()}_{key.upper().replace('.', '_')}"
            value = os.getenv(env_key)
        
        # Return default if still None
        if value is None:
            value = default
        
        return value
    
    def set_config_value(
        self,
        config_name: str,
        key: str,
        value: Any,
        save: bool = True
    ) -> None:
        """
        Set a configuration value.
        
        Args:
            config_name: Name of the configuration
            key: Configuration key (supports dot notation)
            value: Value to set
            save: Whether to save the configuration to file
        """
        # Get or create configuration
        if config_name in self.config_cache:
            config = self.config_cache[config_name]
        else:
            config = {}
        
        # Set nested value
        self._set_nested_value(config, key, value)
        
        # Update cache
        self.config_cache[config_name] = config
        
        # Save if requested
        if save:
            self.save_config(config, config_name)
    
    def _get_nested_value(self, config: Dict[str, Any], key: str) -> Any:
        """Get value from nested dictionary using dot notation."""
        keys = key.split('.')
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any) -> None:
        """Set value in nested dictionary using dot notation."""
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def validate_config(
        self,
        config: Dict[str, Any],
        config_name: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate configuration against rules.
        
        Args:
            config: Configuration dictionary
            config_name: Name of the configuration
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        if config_name not in self.validation_rules:
            logger.warning(f"No validation rules found for {config_name}")
            return True, []
        
        rules = self.validation_rules[config_name]
        issues = []
        
        for rule in rules:
            value = self._get_nested_value(config, rule.key)
            
            # Check if required field is present
            if rule.required and value is None:
                issues.append(f"Required field '{rule.key}' is missing")
                continue
            
            # Skip validation if value is None and not required
            if value is None:
                continue
            
            # Type validation
            if rule.data_type != type(value):
                try:
                    # Try to convert to expected type
                    value = rule.data_type(value)
                except (ValueError, TypeError):
                    issues.append(f"Field '{rule.key}' has wrong type. Expected {rule.data_type.__name__}, got {type(value).__name__}")
                    continue
            
            # Allowed values validation
            if rule.allowed_values and value not in rule.allowed_values:
                issues.append(f"Field '{rule.key}' has invalid value '{value}'. Allowed values: {rule.allowed_values}")
            
            # Range validation for numeric types
            if isinstance(value, (int, float)):
                if rule.min_value is not None and value < rule.min_value:
                    issues.append(f"Field '{rule.key}' value {value} is below minimum {rule.min_value}")
                
                if rule.max_value is not None and value > rule.max_value:
                    issues.append(f"Field '{rule.key}' value {value} is above maximum {rule.max_value}")
            
            # Custom validation
            if rule.custom_validator and not rule.custom_validator(value):
                issues.append(f"Field '{rule.key}' failed custom validation")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def add_validation_rule(
        self,
        config_name: str,
        rule: ConfigValidationRule
    ) -> None:
        """
        Add validation rule for a configuration.
        
        Args:
            config_name: Name of the configuration
            rule: Validation rule
        """
        if config_name not in self.validation_rules:
            self.validation_rules[config_name] = []
        
        self.validation_rules[config_name].append(rule)
        logger.info(f"Validation rule added for {config_name}: {rule.key}")
    
    def create_config_template(
        self,
        config_name: str,
        template: Dict[str, Any],
        description: str = ""
    ) -> None:
        """
        Create a configuration template.
        
        Args:
            config_name: Name of the configuration template
            template: Template dictionary
            description: Template description
        """
        self.config_templates[config_name] = {
            'template': template,
            'description': description,
            'created_at': datetime.now().isoformat()
        }
        
        logger.info(f"Configuration template created: {config_name}")
    
    def generate_config_from_template(
        self,
        template_name: str,
        config_name: str,
        overrides: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate configuration from template.
        
        Args:
            template_name: Name of the template
            config_name: Name for the new configuration
            overrides: Values to override in the template
            
        Returns:
            Generated configuration
        """
        if template_name not in self.config_templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = copy.deepcopy(self.config_templates[template_name]['template'])
        
        # Apply overrides
        if overrides:
            for key, value in overrides.items():
                self._set_nested_value(template, key, value)
        
        # Save generated configuration
        self.save_config(template, config_name)
        
        logger.info(f"Configuration generated from template: {config_name}")
        return template
    
    def merge_configs(
        self,
        base_config: Dict[str, Any],
        override_config: Dict[str, Any],
        deep_merge: bool = True
    ) -> Dict[str, Any]:
        """
        Merge two configurations.
        
        Args:
            base_config: Base configuration
            override_config: Configuration to merge in
            deep_merge: Whether to perform deep merge
            
        Returns:
            Merged configuration
        """
        if deep_merge:
            merged = copy.deepcopy(base_config)
            self._deep_merge(merged, override_config)
        else:
            merged = {**base_config, **override_config}
        
        return merged
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Perform deep merge of dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def load_env_config(
        self,
        config_name: str,
        env_prefix: str = None
    ) -> Dict[str, Any]:
        """
        Load configuration from environment variables.
        
        Args:
            config_name: Name of the configuration
            env_prefix: Environment variable prefix
            
        Returns:
            Configuration dictionary
        """
        if env_prefix is None:
            env_prefix = config_name.upper()
        
        config = {}
        
        # Get all environment variables with the prefix
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                # Remove prefix and convert to nested key
                config_key = key[len(env_prefix) + 1:].lower().replace('_', '.')
                
                # Try to convert to appropriate type
                converted_value = self._convert_env_value(value)
                
                # Set nested value
                self._set_nested_value(config, config_key, converted_value)
        
        # Cache the configuration
        self.config_cache[config_name] = config
        
        logger.info(f"Environment configuration loaded: {config_name}")
        return config
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Try boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def export_config(
        self,
        config_name: str,
        format: ConfigFormat = ConfigFormat.JSON,
        include_metadata: bool = True
    ) -> str:
        """
        Export configuration to file.
        
        Args:
            config_name: Name of the configuration
            format: Export format
            include_metadata: Whether to include metadata
            
        Returns:
            Path to exported file
        """
        if config_name not in self.config_cache:
            raise ValueError(f"Configuration '{config_name}' not found in cache")
        
        config = self.config_cache[config_name]
        
        if include_metadata:
            config = {
                'metadata': {
                    'exported_at': datetime.now().isoformat(),
                    'config_name': config_name,
                    'format': format.value
                },
                'config': config
            }
        
        export_path = self.config_directory / f"{config_name}_export.{format.value}"
        
        with open(export_path, 'w') as f:
            if format == ConfigFormat.JSON:
                json.dump(config, f, indent=2, default=str)
            elif format == ConfigFormat.YAML:
                yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration exported: {export_path}")
        return str(export_path)
    
    def list_configs(self) -> List[str]:
        """List all available configurations."""
        configs = []
        
        # From cache
        configs.extend(self.config_cache.keys())
        
        # From files
        for file_path in self.config_directory.glob("*"):
            if file_path.is_file() and file_path.suffix in ['.json', '.yaml', '.yml']:
                config_name = file_path.stem
                if config_name not in configs:
                    configs.append(config_name)
        
        return sorted(configs)
    
    def delete_config(self, config_name: str, delete_file: bool = True) -> None:
        """
        Delete configuration.
        
        Args:
            config_name: Name of the configuration
            delete_file: Whether to delete the configuration file
        """
        # Remove from cache
        if config_name in self.config_cache:
            del self.config_cache[config_name]
        
        # Delete file if requested
        if delete_file:
            for format in ConfigFormat:
                config_path = self.config_directory / f"{config_name}.{format.value}"
                if config_path.exists():
                    config_path.unlink()
                    logger.info(f"Configuration file deleted: {config_path}")
        
        logger.info(f"Configuration deleted: {config_name}")
    
    def get_config_info(self, config_name: str) -> Dict[str, Any]:
        """
        Get information about a configuration.
        
        Args:
            config_name: Name of the configuration
            
        Returns:
            Configuration information
        """
        info = {
            'name': config_name,
            'in_cache': config_name in self.config_cache,
            'files': [],
            'validation_rules': len(self.validation_rules.get(config_name, [])),
            'template_available': config_name in self.config_templates
        }
        
        # Check for files
        for format in ConfigFormat:
            config_path = self.config_directory / f"{config_name}.{format.value}"
            if config_path.exists():
                info['files'].append({
                    'format': format.value,
                    'path': str(config_path),
                    'size': config_path.stat().st_size,
                    'modified': datetime.fromtimestamp(config_path.stat().st_mtime).isoformat()
                })
        
        return info
