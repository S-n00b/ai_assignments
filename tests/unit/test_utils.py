"""
Unit tests for utility modules.

Tests the core functionality of utility modules including
logging, visualization, data processing, and configuration management.
"""

import pytest
import tempfile
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from datetime import datetime

from src.utils import (
    logging_system,
    visualization,
    data_utils,
    config_utils
)


class TestLoggingSystem:
    """Test cases for logging system utilities."""
    
    @pytest.fixture
    def logger(self):
        """Create logger instance."""
        return logging_system.setup_logger("test_logger")
    
    def test_logger_setup(self, logger):
        """Test logger setup."""
        assert logger is not None
        assert logger.name == "test_logger"
    
    def test_log_level_configuration(self):
        """Test log level configuration."""
        with patch('logging.basicConfig') as mock_config:
            logging_system.setup_logger("test", level="DEBUG")
            mock_config.assert_called_once()
    
    def test_file_logging(self, temp_dir):
        """Test file logging functionality."""
        log_file = temp_dir / "test.log"
        
        with patch('logging.FileHandler') as mock_handler:
            logging_system.setup_file_logging(str(log_file))
            mock_handler.assert_called_once()
    
    def test_structured_logging(self, logger):
        """Test structured logging."""
        with patch.object(logger, 'info') as mock_info:
            logging_system.log_structured(
                logger, "test_event", {"key": "value", "level": "info"}
            )
            mock_info.assert_called_once()
    
    def test_performance_logging(self, logger):
        """Test performance logging."""
        with patch.object(logger, 'info') as mock_info:
            logging_system.log_performance(
                logger, "test_operation", 1.5, {"memory_mb": 100}
            )
            mock_info.assert_called_once()
    
    def test_error_logging(self, logger):
        """Test error logging with context."""
        with patch.object(logger, 'error') as mock_error:
            try:
                raise ValueError("Test error")
            except Exception as e:
                logging_system.log_error(logger, e, {"context": "test"})
                mock_error.assert_called_once()


class TestVisualization:
    """Test cases for visualization utilities."""
    
    @pytest.fixture
    def viz_utils(self):
        """Create visualization utilities instance."""
        return visualization.VisualizationUtils()
    
    def test_create_bar_chart(self, viz_utils, sample_metrics_data):
        """Test bar chart creation."""
        chart = viz_utils.create_bar_chart(
            data=sample_metrics_data,
            x_column="model",
            y_column="bleu_score",
            title="Model Performance Comparison"
        )
        
        assert chart is not None
        assert hasattr(chart, 'data')
        assert hasattr(chart, 'layout')
    
    def test_create_line_chart(self, viz_utils):
        """Test line chart creation."""
        data = {
            "x": [1, 2, 3, 4, 5],
            "y": [10, 20, 15, 25, 30]
        }
        
        chart = viz_utils.create_line_chart(
            data=data,
            title="Performance Over Time"
        )
        
        assert chart is not None
        assert hasattr(chart, 'data')
    
    def test_create_heatmap(self, viz_utils):
        """Test heatmap creation."""
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        labels = ["Model A", "Model B", "Model C"]
        
        chart = viz_utils.create_heatmap(
            data=data,
            labels=labels,
            title="Model Comparison Matrix"
        )
        
        assert chart is not None
        assert hasattr(chart, 'data')
    
    def test_create_dashboard(self, viz_utils, sample_metrics_data):
        """Test dashboard creation."""
        charts = [
            viz_utils.create_bar_chart(sample_metrics_data, "model", "bleu_score"),
            viz_utils.create_bar_chart(sample_metrics_data, "model", "rouge_score")
        ]
        
        dashboard = viz_utils.create_dashboard(
            charts=charts,
            title="Model Evaluation Dashboard"
        )
        
        assert dashboard is not None
        assert hasattr(dashboard, 'data')
    
    def test_export_chart(self, viz_utils, temp_dir):
        """Test chart export functionality."""
        chart = viz_utils.create_bar_chart(
            data=sample_metrics_data,
            x_column="model",
            y_column="bleu_score"
        )
        
        output_file = temp_dir / "test_chart.png"
        
        with patch('plotly.offline.plot') as mock_plot:
            result = viz_utils.export_chart(chart, str(output_file), "png")
            assert result == True
    
    def test_customize_theme(self, viz_utils):
        """Test theme customization."""
        theme_config = {
            "colors": ["#FF6B6B", "#4ECDC4", "#45B7D1"],
            "font": {"family": "Arial", "size": 12},
            "background": "#FFFFFF"
        }
        
        viz_utils.set_theme(theme_config)
        
        # Verify theme is applied
        assert viz_utils.current_theme == theme_config


class TestDataUtils:
    """Test cases for data processing utilities."""
    
    @pytest.fixture
    def data_processor(self):
        """Create data processor instance."""
        return data_utils.DataProcessor()
    
    def test_load_csv_data(self, data_processor, temp_dir):
        """Test CSV data loading."""
        csv_file = temp_dir / "test_data.csv"
        test_data = "name,age,city\nJohn,25,NYC\nJane,30,LA"
        
        with open(csv_file, 'w') as f:
            f.write(test_data)
        
        df = data_processor.load_csv(str(csv_file))
        
        assert df is not None
        assert len(df) == 2
        assert list(df.columns) == ["name", "age", "city"]
    
    def test_load_json_data(self, data_processor, temp_dir):
        """Test JSON data loading."""
        json_file = temp_dir / "test_data.json"
        test_data = {"users": [{"name": "John", "age": 25}, {"name": "Jane", "age": 30}]}
        
        with open(json_file, 'w') as f:
            json.dump(test_data, f)
        
        data = data_processor.load_json(str(json_file))
        
        assert data is not None
        assert "users" in data
        assert len(data["users"]) == 2
    
    def test_clean_data(self, data_processor):
        """Test data cleaning functionality."""
        dirty_data = pd.DataFrame({
            "name": ["John", "Jane", "", "Bob"],
            "age": [25, 30, None, 35],
            "email": ["john@email.com", "invalid-email", "jane@email.com", "bob@email.com"]
        })
        
        clean_data = data_processor.clean_data(dirty_data)
        
        assert len(clean_data) == 3  # One row with empty name removed
        assert clean_data["age"].isna().sum() == 0  # No null ages
        assert clean_data["email"].str.contains("@").all()  # Valid emails
    
    def test_normalize_data(self, data_processor):
        """Test data normalization."""
        data = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [100, 200, 300, 400, 500]
        })
        
        normalized = data_processor.normalize_data(data)
        
        assert normalized["feature1"].min() >= 0
        assert normalized["feature1"].max() <= 1
        assert normalized["feature2"].min() >= 0
        assert normalized["feature2"].max() <= 1
    
    def test_validate_data_schema(self, data_processor):
        """Test data schema validation."""
        data = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["John", "Jane", "Bob"],
            "age": [25, 30, 35]
        })
        
        schema = {
            "id": "int64",
            "name": "object",
            "age": "int64"
        }
        
        is_valid = data_processor.validate_schema(data, schema)
        assert is_valid == True
        
        # Test invalid schema
        invalid_schema = {
            "id": "float64",  # Wrong type
            "name": "object",
            "age": "int64"
        }
        
        is_valid = data_processor.validate_schema(data, invalid_schema)
        assert is_valid == False
    
    def test_aggregate_data(self, data_processor):
        """Test data aggregation."""
        data = pd.DataFrame({
            "category": ["A", "A", "B", "B", "C"],
            "value": [10, 20, 30, 40, 50]
        })
        
        aggregated = data_processor.aggregate_data(data, "category", "value", "sum")
        
        assert len(aggregated) == 3
        assert aggregated.loc[aggregated["category"] == "A", "value"].iloc[0] == 30
        assert aggregated.loc[aggregated["category"] == "B", "value"].iloc[0] == 70


class TestConfigUtils:
    """Test cases for configuration utilities."""
    
    @pytest.fixture
    def config_manager(self):
        """Create configuration manager instance."""
        return config_utils.ConfigManager()
    
    def test_load_yaml_config(self, config_manager, temp_dir):
        """Test YAML configuration loading."""
        config_file = temp_dir / "config.yaml"
        config_data = {
            "database": {"host": "localhost", "port": 5432},
            "api": {"timeout": 30, "retries": 3}
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config = config_manager.load_yaml_config(str(config_file))
        
        assert config is not None
        assert config["database"]["host"] == "localhost"
        assert config["api"]["timeout"] == 30
    
    def test_load_json_config(self, config_manager, temp_dir):
        """Test JSON configuration loading."""
        config_file = temp_dir / "config.json"
        config_data = {
            "model": {"name": "gpt-3.5-turbo", "temperature": 0.7},
            "evaluation": {"metrics": ["bleu", "rouge"]}
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        config = config_manager.load_json_config(str(config_file))
        
        assert config is not None
        assert config["model"]["name"] == "gpt-3.5-turbo"
        assert config["evaluation"]["metrics"] == ["bleu", "rouge"]
    
    def test_validate_config(self, config_manager):
        """Test configuration validation."""
        config = {
            "required_field": "value",
            "optional_field": "optional_value"
        }
        
        schema = {
            "required_field": {"type": "string", "required": True},
            "optional_field": {"type": "string", "required": False}
        }
        
        is_valid = config_manager.validate_config(config, schema)
        assert is_valid == True
        
        # Test missing required field
        invalid_config = {"optional_field": "value"}
        is_valid = config_manager.validate_config(invalid_config, schema)
        assert is_valid == False
    
    def test_merge_configs(self, config_manager):
        """Test configuration merging."""
        base_config = {
            "database": {"host": "localhost", "port": 5432},
            "api": {"timeout": 30}
        }
        
        override_config = {
            "database": {"port": 3306},
            "api": {"retries": 3}
        }
        
        merged = config_manager.merge_configs(base_config, override_config)
        
        assert merged["database"]["host"] == "localhost"
        assert merged["database"]["port"] == 3306  # Overridden
        assert merged["api"]["timeout"] == 30
        assert merged["api"]["retries"] == 3  # Added
    
    def test_get_config_value(self, config_manager):
        """Test getting nested configuration values."""
        config = {
            "database": {
                "connection": {
                    "host": "localhost",
                    "port": 5432
                }
            }
        }
        
        host = config_manager.get_config_value(config, "database.connection.host")
        port = config_manager.get_config_value(config, "database.connection.port")
        missing = config_manager.get_config_value(config, "database.connection.user", "default")
        
        assert host == "localhost"
        assert port == 5432
        assert missing == "default"
    
    def test_save_config(self, config_manager, temp_dir):
        """Test configuration saving."""
        config = {"test": "value", "nested": {"key": "value"}}
        config_file = temp_dir / "output_config.yaml"
        
        result = config_manager.save_config(config, str(config_file))
        
        assert result == True
        assert config_file.exists()
        
        # Verify saved content
        with open(config_file, 'r') as f:
            saved_config = yaml.safe_load(f)
            assert saved_config == config
    
    def test_environment_variable_substitution(self, config_manager):
        """Test environment variable substitution in config."""
        import os
        os.environ["TEST_VAR"] = "test_value"
        
        config = {
            "database": {"host": "${TEST_VAR}"},
            "api": {"key": "${MISSING_VAR:default_value}"}
        }
        
        processed = config_manager.substitute_env_vars(config)
        
        assert processed["database"]["host"] == "test_value"
        assert processed["api"]["key"] == "default_value"
        
        # Clean up
        del os.environ["TEST_VAR"]
