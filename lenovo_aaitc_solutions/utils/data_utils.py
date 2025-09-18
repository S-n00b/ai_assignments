"""
Data Utilities for AI Model Evaluation and Architecture

This module provides comprehensive data processing and manipulation utilities
for both Assignment 1 (Model Evaluation) and Assignment 2 (AI Architecture) solutions.

Key Features:
- Data validation and cleaning
- Statistical analysis utilities
- Data transformation and preprocessing
- Export and import capabilities
- Data quality assessment
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json
import logging
from pathlib import Path
import hashlib
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)


class DataType(Enum):
    """Data types for validation"""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEXT = "text"
    DATETIME = "datetime"
    BOOLEAN = "boolean"


class QualityLevel(Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


@dataclass
class DataQualityReport:
    """Data quality assessment report"""
    total_rows: int
    total_columns: int
    missing_values: Dict[str, int]
    duplicate_rows: int
    data_types: Dict[str, str]
    quality_score: float
    quality_level: QualityLevel
    issues: List[str]
    recommendations: List[str]


class DataUtils:
    """
    Comprehensive data processing and manipulation utilities.
    
    This class provides extensive data processing capabilities including:
    - Data validation and cleaning
    - Statistical analysis and profiling
    - Data transformation and preprocessing
    - Quality assessment and reporting
    - Export and import utilities
    """
    
    def __init__(self):
        """Initialize data utilities."""
        self.supported_formats = ['csv', 'json', 'parquet', 'excel', 'pickle']
        logger.info("Data utilities initialized")
    
    def validate_data(
        self,
        data: pd.DataFrame,
        schema: Dict[str, Any] = None,
        strict: bool = False
    ) -> Tuple[bool, List[str]]:
        """
        Validate data against schema or basic rules.
        
        Args:
            data: DataFrame to validate
            schema: Validation schema
            strict: Whether to use strict validation
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if data.empty:
            issues.append("DataFrame is empty")
            return False, issues
        
        # Basic validation
        if data.isnull().all().any():
            issues.append("Some columns contain only null values")
        
        if data.duplicated().any():
            issues.append("DataFrame contains duplicate rows")
        
        # Schema validation
        if schema:
            for column, rules in schema.items():
                if column not in data.columns:
                    issues.append(f"Required column '{column}' is missing")
                    continue
                
                # Data type validation
                if 'type' in rules:
                    expected_type = rules['type']
                    actual_type = str(data[column].dtype)
                    
                    if not self._is_type_compatible(actual_type, expected_type):
                        issues.append(f"Column '{column}' has type {actual_type}, expected {expected_type}")
                
                # Value range validation
                if 'min' in rules and data[column].dtype in ['int64', 'float64']:
                    if data[column].min() < rules['min']:
                        issues.append(f"Column '{column}' has values below minimum {rules['min']}")
                
                if 'max' in rules and data[column].dtype in ['int64', 'float64']:
                    if data[column].max() > rules['max']:
                        issues.append(f"Column '{column}' has values above maximum {rules['max']}")
                
                # Categorical validation
                if 'categories' in rules:
                    unique_values = set(data[column].dropna().unique())
                    expected_categories = set(rules['categories'])
                    
                    if strict and not unique_values.issubset(expected_categories):
                        issues.append(f"Column '{column}' contains unexpected values")
                    elif not unique_values.intersection(expected_categories):
                        issues.append(f"Column '{column}' has no valid categorical values")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def _is_type_compatible(self, actual_type: str, expected_type: str) -> bool:
        """Check if actual data type is compatible with expected type."""
        type_mapping = {
            'int64': ['int', 'integer', 'numerical'],
            'float64': ['float', 'numerical', 'decimal'],
            'object': ['string', 'text', 'categorical'],
            'bool': ['boolean', 'bool'],
            'datetime64[ns]': ['datetime', 'date', 'timestamp']
        }
        
        for actual, expected_list in type_mapping.items():
            if actual_type.startswith(actual):
                return expected_type.lower() in expected_list
        
        return False
    
    def clean_data(
        self,
        data: pd.DataFrame,
        cleaning_rules: Dict[str, Any] = None
    ) -> pd.DataFrame:
        """
        Clean data based on specified rules.
        
        Args:
            data: DataFrame to clean
            cleaning_rules: Dictionary of cleaning rules
            
        Returns:
            Cleaned DataFrame
        """
        cleaned_data = data.copy()
        
        if cleaning_rules is None:
            cleaning_rules = {
                'remove_duplicates': True,
                'handle_missing': 'drop',
                'normalize_text': True,
                'remove_outliers': False
            }
        
        # Remove duplicates
        if cleaning_rules.get('remove_duplicates', True):
            initial_rows = len(cleaned_data)
            cleaned_data = cleaned_data.drop_duplicates()
            removed_duplicates = initial_rows - len(cleaned_data)
            if removed_duplicates > 0:
                logger.info(f"Removed {removed_duplicates} duplicate rows")
        
        # Handle missing values
        missing_strategy = cleaning_rules.get('handle_missing', 'drop')
        if missing_strategy == 'drop':
            cleaned_data = cleaned_data.dropna()
        elif missing_strategy == 'fill':
            fill_method = cleaning_rules.get('fill_method', 'mean')
            if fill_method == 'mean':
                cleaned_data = cleaned_data.fillna(cleaned_data.mean(numeric_only=True))
            elif fill_method == 'median':
                cleaned_data = cleaned_data.fillna(cleaned_data.median(numeric_only=True))
            elif fill_method == 'mode':
                cleaned_data = cleaned_data.fillna(cleaned_data.mode().iloc[0])
            elif fill_method == 'forward':
                cleaned_data = cleaned_data.fillna(method='ffill')
            elif fill_method == 'backward':
                cleaned_data = cleaned_data.fillna(method='bfill')
        
        # Normalize text columns
        if cleaning_rules.get('normalize_text', True):
            text_columns = cleaned_data.select_dtypes(include=['object']).columns
            for col in text_columns:
                cleaned_data[col] = cleaned_data[col].astype(str).str.strip().str.lower()
        
        # Remove outliers
        if cleaning_rules.get('remove_outliers', False):
            numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                Q1 = cleaned_data[col].quantile(0.25)
                Q3 = cleaned_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                cleaned_data = cleaned_data[
                    (cleaned_data[col] >= lower_bound) & 
                    (cleaned_data[col] <= upper_bound)
                ]
        
        logger.info(f"Data cleaning completed. Shape: {cleaned_data.shape}")
        return cleaned_data
    
    def assess_data_quality(self, data: pd.DataFrame) -> DataQualityReport:
        """
        Assess data quality and generate report.
        
        Args:
            data: DataFrame to assess
            
        Returns:
            DataQualityReport object
        """
        total_rows = len(data)
        total_columns = len(data.columns)
        
        # Missing values analysis
        missing_values = data.isnull().sum().to_dict()
        total_missing = sum(missing_values.values())
        missing_percentage = (total_missing / (total_rows * total_columns)) * 100
        
        # Duplicate rows
        duplicate_rows = data.duplicated().sum()
        
        # Data types
        data_types = data.dtypes.astype(str).to_dict()
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(
            total_rows, total_columns, missing_percentage, duplicate_rows
        )
        
        # Determine quality level
        if quality_score >= 0.9:
            quality_level = QualityLevel.EXCELLENT
        elif quality_score >= 0.7:
            quality_level = QualityLevel.GOOD
        elif quality_score >= 0.5:
            quality_level = QualityLevel.FAIR
        else:
            quality_level = QualityLevel.POOR
        
        # Identify issues
        issues = []
        if missing_percentage > 10:
            issues.append(f"High missing value percentage: {missing_percentage:.1f}%")
        
        if duplicate_rows > total_rows * 0.05:
            issues.append(f"High duplicate rate: {duplicate_rows} rows ({duplicate_rows/total_rows*100:.1f}%)")
        
        # Check for constant columns
        constant_columns = [col for col in data.columns if data[col].nunique() <= 1]
        if constant_columns:
            issues.append(f"Constant columns found: {constant_columns}")
        
        # Generate recommendations
        recommendations = []
        if missing_percentage > 5:
            recommendations.append("Consider imputation strategies for missing values")
        
        if duplicate_rows > 0:
            recommendations.append("Remove duplicate rows")
        
        if constant_columns:
            recommendations.append("Remove or investigate constant columns")
        
        if quality_score < 0.7:
            recommendations.append("Overall data quality needs improvement")
        
        return DataQualityReport(
            total_rows=total_rows,
            total_columns=total_columns,
            missing_values=missing_values,
            duplicate_rows=duplicate_rows,
            data_types=data_types,
            quality_score=quality_score,
            quality_level=quality_level,
            issues=issues,
            recommendations=recommendations
        )
    
    def _calculate_quality_score(
        self,
        total_rows: int,
        total_columns: int,
        missing_percentage: float,
        duplicate_rows: int
    ) -> float:
        """Calculate overall data quality score."""
        
        # Base score
        score = 1.0
        
        # Penalize missing values
        missing_penalty = min(missing_percentage / 100, 0.5)
        score -= missing_penalty
        
        # Penalize duplicates
        duplicate_penalty = min(duplicate_rows / total_rows, 0.3)
        score -= duplicate_penalty
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))
    
    def transform_data(
        self,
        data: pd.DataFrame,
        transformations: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Apply data transformations.
        
        Args:
            data: DataFrame to transform
            transformations: Dictionary of transformation rules
            
        Returns:
            Transformed DataFrame
        """
        transformed_data = data.copy()
        
        for column, rules in transformations.items():
            if column not in transformed_data.columns:
                logger.warning(f"Column '{column}' not found in data")
                continue
            
            # Scaling
            if 'scale' in rules:
                scale_method = rules['scale']
                if scale_method == 'minmax':
                    min_val = transformed_data[column].min()
                    max_val = transformed_data[column].max()
                    transformed_data[column] = (transformed_data[column] - min_val) / (max_val - min_val)
                elif scale_method == 'standard':
                    mean_val = transformed_data[column].mean()
                    std_val = transformed_data[column].std()
                    transformed_data[column] = (transformed_data[column] - mean_val) / std_val
                elif scale_method == 'robust':
                    median_val = transformed_data[column].median()
                    q75 = transformed_data[column].quantile(0.75)
                    q25 = transformed_data[column].quantile(0.25)
                    transformed_data[column] = (transformed_data[column] - median_val) / (q75 - q25)
            
            # Encoding
            if 'encode' in rules:
                encode_method = rules['encode']
                if encode_method == 'onehot':
                    # One-hot encoding
                    dummies = pd.get_dummies(transformed_data[column], prefix=column)
                    transformed_data = pd.concat([transformed_data, dummies], axis=1)
                    transformed_data = transformed_data.drop(columns=[column])
                elif encode_method == 'label':
                    # Label encoding
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    transformed_data[column] = le.fit_transform(transformed_data[column].astype(str))
            
            # Log transformation
            if rules.get('log_transform', False):
                if transformed_data[column].min() > 0:
                    transformed_data[column] = np.log(transformed_data[column])
                else:
                    transformed_data[column] = np.log(transformed_data[column] - transformed_data[column].min() + 1)
            
            # Square root transformation
            if rules.get('sqrt_transform', False):
                if transformed_data[column].min() >= 0:
                    transformed_data[column] = np.sqrt(transformed_data[column])
                else:
                    transformed_data[column] = np.sqrt(transformed_data[column] - transformed_data[column].min())
        
        logger.info(f"Data transformation completed. Shape: {transformed_data.shape}")
        return transformed_data
    
    def calculate_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics for the dataset.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dictionary containing statistical information
        """
        stats = {
            'basic_info': {
                'shape': data.shape,
                'memory_usage': data.memory_usage(deep=True).sum(),
                'dtypes': data.dtypes.value_counts().to_dict()
            },
            'numerical_stats': {},
            'categorical_stats': {},
            'correlations': {}
        }
        
        # Numerical statistics
        numerical_columns = data.select_dtypes(include=[np.number]).columns
        if len(numerical_columns) > 0:
            stats['numerical_stats'] = data[numerical_columns].describe().to_dict()
            
            # Correlation matrix
            if len(numerical_columns) > 1:
                stats['correlations'] = data[numerical_columns].corr().to_dict()
        
        # Categorical statistics
        categorical_columns = data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            stats['categorical_stats'][col] = {
                'unique_count': data[col].nunique(),
                'most_frequent': data[col].mode().iloc[0] if not data[col].mode().empty else None,
                'frequency_distribution': data[col].value_counts().head(10).to_dict()
            }
        
        return stats
    
    def export_data(
        self,
        data: pd.DataFrame,
        filepath: str,
        format: str = 'csv',
        **kwargs
    ) -> str:
        """
        Export data to file.
        
        Args:
            data: DataFrame to export
            filepath: Output file path
            format: Export format
            **kwargs: Additional arguments for export function
            
        Returns:
            Path to exported file
        """
        filepath = Path(filepath)
        
        if format.lower() == 'csv':
            data.to_csv(filepath, index=kwargs.get('index', False), **kwargs)
        elif format.lower() == 'json':
            data.to_json(filepath, orient=kwargs.get('orient', 'records'), **kwargs)
        elif format.lower() == 'parquet':
            data.to_parquet(filepath, **kwargs)
        elif format.lower() == 'excel':
            data.to_excel(filepath, index=kwargs.get('index', False), **kwargs)
        elif format.lower() == 'pickle':
            data.to_pickle(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Data exported to {filepath}")
        return str(filepath)
    
    def import_data(
        self,
        filepath: str,
        format: str = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Import data from file.
        
        Args:
            filepath: Input file path
            format: File format (auto-detected if None)
            **kwargs: Additional arguments for import function
            
        Returns:
            Imported DataFrame
        """
        filepath = Path(filepath)
        
        if format is None:
            format = filepath.suffix.lower().lstrip('.')
        
        if format == 'csv':
            data = pd.read_csv(filepath, **kwargs)
        elif format == 'json':
            data = pd.read_json(filepath, **kwargs)
        elif format == 'parquet':
            data = pd.read_parquet(filepath, **kwargs)
        elif format == 'excel':
            data = pd.read_excel(filepath, **kwargs)
        elif format == 'pickle':
            data = pd.read_pickle(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Data imported from {filepath}. Shape: {data.shape}")
        return data
    
    def create_sample_data(
        self,
        n_rows: int = 1000,
        n_columns: int = 10,
        data_types: Dict[str, str] = None
    ) -> pd.DataFrame:
        """
        Create sample data for testing and demonstration.
        
        Args:
            n_rows: Number of rows
            n_columns: Number of columns
            data_types: Dictionary mapping column names to data types
            
        Returns:
            Generated DataFrame
        """
        if data_types is None:
            data_types = {
                'numerical': ['float', 'int'],
                'categorical': ['category'],
                'text': ['string'],
                'datetime': ['datetime']
            }
        
        data = {}
        
        for i in range(n_columns):
            col_name = f"column_{i}"
            
            if i < n_columns * 0.4:  # 40% numerical
                data[col_name] = np.random.normal(0, 1, n_rows)
            elif i < n_columns * 0.7:  # 30% categorical
                categories = ['A', 'B', 'C', 'D', 'E']
                data[col_name] = np.random.choice(categories, n_rows)
            elif i < n_columns * 0.9:  # 20% text
                data[col_name] = [f"text_{j}" for j in range(n_rows)]
            else:  # 10% datetime
                data[col_name] = pd.date_range('2020-01-01', periods=n_rows, freq='D')
        
        df = pd.DataFrame(data)
        
        # Add some missing values
        missing_indices = np.random.choice(df.index, size=int(n_rows * 0.05), replace=False)
        df.loc[missing_indices, df.columns[0]] = np.nan
        
        logger.info(f"Sample data created. Shape: {df.shape}")
        return df
    
    def compare_datasets(
        self,
        data1: pd.DataFrame,
        data2: pd.DataFrame,
        tolerance: float = 1e-6
    ) -> Dict[str, Any]:
        """
        Compare two datasets and return differences.
        
        Args:
            data1: First DataFrame
            data2: Second DataFrame
            tolerance: Numerical tolerance for comparison
            
        Returns:
            Dictionary containing comparison results
        """
        comparison = {
            'shape_match': data1.shape == data2.shape,
            'columns_match': list(data1.columns) == list(data2.columns),
            'dtypes_match': data1.dtypes.equals(data2.dtypes),
            'values_match': True,
            'differences': []
        }
        
        if not comparison['shape_match']:
            comparison['differences'].append(f"Shape mismatch: {data1.shape} vs {data2.shape}")
        
        if not comparison['columns_match']:
            comparison['differences'].append("Column names don't match")
        
        if not comparison['dtypes_match']:
            comparison['differences'].append("Data types don't match")
        
        # Compare values if shapes match
        if comparison['shape_match'] and comparison['columns_match']:
            try:
                # For numerical columns
                numerical_cols = data1.select_dtypes(include=[np.number]).columns
                for col in numerical_cols:
                    if not np.allclose(data1[col], data2[col], rtol=tolerance, equal_nan=True):
                        comparison['values_match'] = False
                        comparison['differences'].append(f"Values differ in column '{col}'")
                
                # For non-numerical columns
                non_numerical_cols = data1.select_dtypes(exclude=[np.number]).columns
                for col in non_numerical_cols:
                    if not data1[col].equals(data2[col]):
                        comparison['values_match'] = False
                        comparison['differences'].append(f"Values differ in column '{col}'")
                        
            except Exception as e:
                comparison['values_match'] = False
                comparison['differences'].append(f"Error comparing values: {str(e)}")
        
        return comparison
