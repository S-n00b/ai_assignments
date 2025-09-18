"""
Basic functionality tests for core components.
These tests focus on basic functionality without requiring heavy dependencies.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

from src.model_evaluation.config import ModelConfig
from src.utils.config_utils import ConfigUtils
from src.utils.data_utils import DataValidator


class TestModelConfig:
    """Test cases for ModelConfig class."""
    
    def test_model_config_creation(self):
        """Test basic model configuration creation."""
        config = ModelConfig(
            model_name="gpt-3.5-turbo",
            model_version="2024-01-01",
            api_key="test-key"
        )
        
        assert config.model_name == "gpt-3.5-turbo"
        assert config.model_version == "2024-01-01"
        assert config.api_key == "test-key"
        assert config.max_tokens == 1000  # default value
        assert config.temperature == 0.7  # default value
    
    def test_model_config_validation(self):
        """Test model configuration validation."""
        with pytest.raises(ValueError, match="Model name is required"):
            ModelConfig(model_name="", api_key="test-key")
        
        with pytest.raises(ValueError, match="API key is required"):
            ModelConfig(model_name="gpt-3.5-turbo", api_key="")
    
    def test_model_config_serialization(self):
        """Test model configuration serialization."""
        config = ModelConfig(
            model_name="gpt-3.5-turbo",
            model_version="2024-01-01",
            api_key="test-key"
        )
        
        config_dict = config.to_dict()
        assert config_dict["model_name"] == "gpt-3.5-turbo"
        assert config_dict["model_version"] == "2024-01-01"
        assert "api_key" not in config_dict  # API key should be excluded for security
    
    def test_model_config_from_dict(self):
        """Test creating model config from dictionary."""
        config_dict = {
            "model_name": "claude-3-sonnet",
            "model_version": "2024-01-01",
            "max_tokens": 2000,
            "temperature": 0.5
        }
        
        config = ModelConfig.from_dict(config_dict)
        assert config.model_name == "claude-3-sonnet"
        assert config.max_tokens == 2000
        assert config.temperature == 0.5


class TestConfigUtils:
    """Test cases for ConfigUtils class."""
    
    @pytest.fixture
    def config_utils(self):
        """Create config utils instance."""
        return ConfigUtils()
    
    def test_config_utils_initialization(self, config_utils):
        """Test config utils initialization."""
        assert config_utils is not None
        assert hasattr(config_utils, 'load_config')
        assert hasattr(config_utils, 'save_config')
    
    def test_load_config(self, config_utils, temp_dir):
        """Test loading configuration from file."""
        # Create a test config file
        config_data = {
            "model_name": "test-model",
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        config_file = temp_dir / "test_config.json"
        import json
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Test loading config
        loaded_config = config_utils.load_config(str(config_file))
        assert loaded_config["model_name"] == "test-model"
        assert loaded_config["temperature"] == 0.7
        assert loaded_config["max_tokens"] == 1000
    
    def test_save_config(self, config_utils, temp_dir):
        """Test saving configuration to file."""
        config_data = {
            "model_name": "test-model",
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        config_file = temp_dir / "test_save_config.json"
        config_utils.save_config(config_data, str(config_file))
        
        # Verify file was created and contains correct data
        assert config_file.exists()
        import json
        with open(config_file, 'r') as f:
            saved_data = json.load(f)
        assert saved_data == config_data


class TestDataValidator:
    """Test cases for DataValidator class."""
    
    @pytest.fixture
    def data_validator(self):
        """Create data validator instance."""
        return DataValidator()
    
    def test_data_validator_initialization(self, data_validator):
        """Test data validator initialization."""
        assert data_validator is not None
        assert hasattr(data_validator, 'validate_data')
        assert hasattr(data_validator, 'validate_schema')
    
    def test_validate_data(self, data_validator):
        """Test data validation."""
        # Test valid data
        valid_data = pd.DataFrame({
            "text": ["Hello world", "Test message"],
            "label": [1, 0]
        })
        
        result = data_validator.validate_data(valid_data)
        assert result["is_valid"] == True
        assert result["quality_score"] > 0.8
    
    def test_validate_schema(self, data_validator):
        """Test schema validation."""
        schema = {
            "text": "string",
            "label": "integer"
        }
        
        valid_data = pd.DataFrame({
            "text": ["Hello world", "Test message"],
            "label": [1, 0]
        })
        
        result = data_validator.validate_schema(valid_data, schema)
        assert result["is_valid"] == True
        assert len(result["errors"]) == 0
    
    def test_validate_schema_with_errors(self, data_validator):
        """Test schema validation with errors."""
        schema = {
            "text": "string",
            "label": "integer"
        }
        
        invalid_data = pd.DataFrame({
            "text": ["Hello world", "Test message"],
            "label": ["invalid", "also_invalid"]  # Should be integers
        })
        
        result = data_validator.validate_schema(invalid_data, schema)
        assert result["is_valid"] == False
        assert len(result["errors"]) > 0


class TestBasicUtilities:
    """Test cases for basic utility functions."""
    
    def test_datetime_handling(self):
        """Test datetime handling utilities."""
        now = datetime.now()
        assert now is not None
        assert isinstance(now, datetime)
    
    def test_pandas_operations(self):
        """Test basic pandas operations."""
        df = pd.DataFrame({
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50]
        })
        
        assert len(df) == 5
        assert df["A"].sum() == 15
        assert df["B"].mean() == 30
    
    def test_numpy_operations(self):
        """Test basic numpy operations."""
        arr = np.array([1, 2, 3, 4, 5])
        
        assert len(arr) == 5
        assert arr.sum() == 15
        assert arr.mean() == 3.0
        assert arr.std() > 0


class TestMockFunctionality:
    """Test cases for mock functionality."""
    
    def test_mock_creation(self):
        """Test creating mock objects."""
        mock_obj = Mock()
        mock_obj.test_method.return_value = "test_result"
        
        assert mock_obj.test_method() == "test_result"
        mock_obj.test_method.assert_called_once()
    
    def test_async_mock(self):
        """Test async mock functionality."""
        async_mock = Mock()
        async_mock.async_method = Mock(return_value="async_result")
        
        # Test that we can call the async method
        result = async_mock.async_method()
        assert result == "async_result"
    
    def test_patch_functionality(self):
        """Test patch functionality."""
        with patch('builtins.len') as mock_len:
            mock_len.return_value = 42
            
            # Test that patch works
            result = len([1, 2, 3])
            assert result == 42
            mock_len.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
