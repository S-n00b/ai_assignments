"""
Visualization Utilities for AI Model Evaluation and Architecture

This module provides comprehensive visualization capabilities for both
Assignment 1 (Model Evaluation) and Assignment 2 (AI Architecture) solutions,
including performance charts, architecture diagrams, and interactive dashboards.

Key Features:
- Model performance visualization
- Architecture diagram generation
- Interactive dashboards
- Real-time monitoring charts
- Export capabilities
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json
import base64
from io import BytesIO
import logging

logger = logging.getLogger(__name__)


class ChartType(Enum):
    """Types of charts available"""
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    PIE = "pie"
    HISTOGRAM = "histogram"
    BOX = "box"
    VIOLIN = "violin"
    RADAR = "radar"
    SANKEY = "sankey"


class ExportFormat(Enum):
    """Export formats for visualizations"""
    PNG = "png"
    JPEG = "jpeg"
    SVG = "svg"
    PDF = "pdf"
    HTML = "html"
    JSON = "json"


@dataclass
class ChartConfig:
    """Configuration for chart generation"""
    title: str
    x_label: str
    y_label: str
    width: int = 800
    height: int = 600
    theme: str = "plotly_white"
    color_scheme: str = "viridis"
    show_legend: bool = True
    show_grid: bool = True
    annotations: List[Dict[str, Any]] = None


class VisualizationUtils:
    """
    Comprehensive visualization utilities for AI applications.
    
    This class provides extensive visualization capabilities including:
    - Model performance charts and metrics
    - Architecture diagrams and flowcharts
    - Interactive dashboards
    - Real-time monitoring visualizations
    - Export and sharing capabilities
    """
    
    def __init__(self, default_theme: str = "plotly_white"):
        """
        Initialize visualization utilities.
        
        Args:
            default_theme: Default theme for visualizations
        """
        self.default_theme = default_theme
        self.color_palettes = {
            'viridis': px.colors.sequential.Viridis,
            'plasma': px.colors.sequential.Plasma,
            'inferno': px.colors.sequential.Inferno,
            'magma': px.colors.sequential.Magma,
            'cividis': px.colors.sequential.Cividis,
            'rainbow': px.colors.sequential.Rainbow,
            'turbo': px.colors.sequential.Turbo
        }
        
        # Set default styling
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        logger.info("Visualization utilities initialized")
    
    def create_model_performance_chart(
        self,
        data: pd.DataFrame,
        metrics: List[str],
        models: List[str],
        config: ChartConfig = None
    ) -> go.Figure:
        """
        Create a comprehensive model performance comparison chart.
        
        Args:
            data: DataFrame containing model performance data
            metrics: List of metrics to visualize
            models: List of model names
            config: Chart configuration
            
        Returns:
            Plotly figure object
        """
        if config is None:
            config = ChartConfig(
                title="Model Performance Comparison",
                x_label="Models",
                y_label="Performance Score"
            )
        
        # Create subplots
        fig = make_subplots(
            rows=len(metrics),
            cols=1,
            subplot_titles=metrics,
            vertical_spacing=0.1
        )
        
        colors = self.color_palettes.get(config.color_scheme, px.colors.qualitative.Set3)
        
        for i, metric in enumerate(metrics):
            for j, model in enumerate(models):
                model_data = data[data['model'] == model]
                if not model_data.empty:
                    fig.add_trace(
                        go.Bar(
                            x=[model],
                            y=[model_data[metric].iloc[0]],
                            name=f"{model} - {metric}",
                            marker_color=colors[j % len(colors)],
                            showlegend=(i == 0)  # Only show legend for first subplot
                        ),
                        row=i + 1,
                        col=1
                    )
        
        # Update layout
        fig.update_layout(
            title=config.title,
            height=config.height * len(metrics),
            width=config.width,
            template=config.theme,
            showlegend=config.show_legend
        )
        
        # Update axes
        for i in range(len(metrics)):
            fig.update_xaxes(title_text=config.x_label, row=i + 1, col=1)
            fig.update_yaxes(title_text=config.y_label, row=i + 1, col=1)
        
        return fig
    
    def create_performance_trend_chart(
        self,
        data: pd.DataFrame,
        metric: str,
        time_column: str = "timestamp",
        config: ChartConfig = None
    ) -> go.Figure:
        """
        Create a performance trend chart over time.
        
        Args:
            data: DataFrame containing time series performance data
            metric: Metric to visualize
            time_column: Name of the time column
            config: Chart configuration
            
        Returns:
            Plotly figure object
        """
        if config is None:
            config = ChartConfig(
                title=f"{metric} Performance Trend",
                x_label="Time",
                y_label=metric
            )
        
        # Create line chart
        fig = go.Figure()
        
        # Group by model if model column exists
        if 'model' in data.columns:
            colors = self.color_palettes.get(config.color_scheme, px.colors.qualitative.Set3)
            for i, model in enumerate(data['model'].unique()):
                model_data = data[data['model'] == model]
                fig.add_trace(
                    go.Scatter(
                        x=model_data[time_column],
                        y=model_data[metric],
                        mode='lines+markers',
                        name=model,
                        line=dict(color=colors[i % len(colors)], width=2),
                        marker=dict(size=6)
                    )
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=data[time_column],
                    y=data[metric],
                    mode='lines+markers',
                    name=metric,
                    line=dict(width=2),
                    marker=dict(size=6)
                )
            )
        
        # Update layout
        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_label,
            yaxis_title=config.y_label,
            template=config.theme,
            showlegend=config.show_legend,
            width=config.width,
            height=config.height
        )
        
        if config.show_grid:
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    def create_architecture_diagram(
        self,
        components: List[Dict[str, Any]],
        connections: List[Dict[str, Any]],
        config: ChartConfig = None
    ) -> go.Figure:
        """
        Create an architecture diagram using Sankey diagram.
        
        Args:
            components: List of component definitions
            connections: List of connections between components
            config: Chart configuration
            
        Returns:
            Plotly figure object
        """
        if config is None:
            config = ChartConfig(
                title="System Architecture",
                x_label="",
                y_label=""
            )
        
        # Prepare data for Sankey diagram
        source = []
        target = []
        value = []
        label = []
        
        # Create component mapping
        component_map = {}
        for i, component in enumerate(components):
            component_map[component['id']] = i
            label.append(component['name'])
        
        # Add connections
        for connection in connections:
            source_idx = component_map.get(connection['source'])
            target_idx = component_map.get(connection['target'])
            
            if source_idx is not None and target_idx is not None:
                source.append(source_idx)
                target.append(target_idx)
                value.append(connection.get('weight', 1))
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=label,
                color="blue"
            ),
            link=dict(
                source=source,
                target=target,
                value=value
            )
        )])
        
        # Update layout
        fig.update_layout(
            title=config.title,
            font_size=10,
            width=config.width,
            height=config.height,
            template=config.theme
        )
        
        return fig
    
    def create_heatmap(
        self,
        data: pd.DataFrame,
        x_column: str,
        y_column: str,
        value_column: str,
        config: ChartConfig = None
    ) -> go.Figure:
        """
        Create a heatmap visualization.
        
        Args:
            data: DataFrame containing data
            x_column: Column for x-axis
            y_column: Column for y-axis
            value_column: Column for values
            config: Chart configuration
            
        Returns:
            Plotly figure object
        """
        if config is None:
            config = ChartConfig(
                title="Performance Heatmap",
                x_label=x_column,
                y_label=y_column
            )
        
        # Create pivot table
        pivot_data = data.pivot_table(
            values=value_column,
            index=y_column,
            columns=x_column,
            aggfunc='mean'
        )
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale=config.color_scheme,
            showscale=True
        ))
        
        # Update layout
        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_label,
            yaxis_title=config.y_label,
            template=config.theme,
            width=config.width,
            height=config.height
        )
        
        return fig
    
    def create_radar_chart(
        self,
        data: Dict[str, float],
        categories: List[str],
        config: ChartConfig = None
    ) -> go.Figure:
        """
        Create a radar chart for multi-dimensional comparison.
        
        Args:
            data: Dictionary with model names as keys and metric values as values
            categories: List of metric categories
            config: Chart configuration
            
        Returns:
            Plotly figure object
        """
        if config is None:
            config = ChartConfig(
                title="Multi-Dimensional Performance Comparison",
                x_label="",
                y_label=""
            )
        
        fig = go.Figure()
        
        colors = self.color_palettes.get(config.color_scheme, px.colors.qualitative.Set3)
        
        for i, (model, values) in enumerate(data.items()):
            # Ensure values are in the same order as categories
            model_values = [values.get(cat, 0) for cat in categories]
            
            fig.add_trace(go.Scatterpolar(
                r=model_values,
                theta=categories,
                fill='toself',
                name=model,
                line_color=colors[i % len(colors)]
            ))
        
        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=config.show_legend,
            title=config.title,
            template=config.theme,
            width=config.width,
            height=config.height
        )
        
        return fig
    
    def create_distribution_chart(
        self,
        data: pd.DataFrame,
        column: str,
        chart_type: ChartType = ChartType.HISTOGRAM,
        config: ChartConfig = None
    ) -> go.Figure:
        """
        Create a distribution chart.
        
        Args:
            data: DataFrame containing data
            column: Column to visualize
            chart_type: Type of distribution chart
            config: Chart configuration
            
        Returns:
            Plotly figure object
        """
        if config is None:
            config = ChartConfig(
                title=f"{column} Distribution",
                x_label=column,
                y_label="Frequency"
            )
        
        if chart_type == ChartType.HISTOGRAM:
            fig = px.histogram(
                data,
                x=column,
                title=config.title,
                template=config.theme,
                color_discrete_sequence=self.color_palettes.get(config.color_scheme, px.colors.qualitative.Set3)
            )
        elif chart_type == ChartType.BOX:
            fig = px.box(
                data,
                y=column,
                title=config.title,
                template=config.theme,
                color_discrete_sequence=self.color_palettes.get(config.color_scheme, px.colors.qualitative.Set3)
            )
        elif chart_type == ChartType.VIOLIN:
            fig = px.violin(
                data,
                y=column,
                title=config.title,
                template=config.theme,
                color_discrete_sequence=self.color_palettes.get(config.color_scheme, px.colors.qualitative.Set3)
            )
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        
        # Update layout
        fig.update_layout(
            width=config.width,
            height=config.height,
            showlegend=config.show_legend
        )
        
        return fig
    
    def create_correlation_matrix(
        self,
        data: pd.DataFrame,
        columns: List[str] = None,
        config: ChartConfig = None
    ) -> go.Figure:
        """
        Create a correlation matrix heatmap.
        
        Args:
            data: DataFrame containing numerical data
            columns: List of columns to include (if None, uses all numerical columns)
            config: Chart configuration
            
        Returns:
            Plotly figure object
        """
        if config is None:
            config = ChartConfig(
                title="Correlation Matrix",
                x_label="",
                y_label=""
            )
        
        # Select columns
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate correlation matrix
        corr_matrix = data[columns].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            showscale=True
        ))
        
        # Update layout
        fig.update_layout(
            title=config.title,
            template=config.theme,
            width=config.width,
            height=config.height
        )
        
        return fig
    
    def create_dashboard(
        self,
        charts: List[go.Figure],
        layout_config: Dict[str, Any] = None
    ) -> go.Figure:
        """
        Create a dashboard with multiple charts.
        
        Args:
            charts: List of Plotly figures
            layout_config: Layout configuration
            
        Returns:
            Combined Plotly figure
        """
        if layout_config is None:
            layout_config = {
                'rows': 2,
                'cols': 2,
                'subplot_titles': [f"Chart {i+1}" for i in range(len(charts))]
            }
        
        # Create subplots
        fig = make_subplots(
            rows=layout_config['rows'],
            cols=layout_config['cols'],
            subplot_titles=layout_config['subplot_titles']
        )
        
        # Add charts to subplots
        for i, chart in enumerate(charts):
            row = (i // layout_config['cols']) + 1
            col = (i % layout_config['cols']) + 1
            
            for trace in chart.data:
                fig.add_trace(trace, row=row, col=col)
        
        # Update layout
        fig.update_layout(
            title="AI System Dashboard",
            template=self.default_theme,
            showlegend=True,
            height=800,
            width=1200
        )
        
        return fig
    
    def export_chart(
        self,
        fig: go.Figure,
        filename: str,
        format: ExportFormat = ExportFormat.HTML,
        width: int = 800,
        height: int = 600
    ) -> str:
        """
        Export chart to file.
        
        Args:
            fig: Plotly figure to export
            filename: Output filename
            format: Export format
            width: Chart width
            height: Chart height
            
        Returns:
            Path to exported file
        """
        if format == ExportFormat.HTML:
            fig.write_html(filename)
        elif format == ExportFormat.PNG:
            fig.write_image(filename, width=width, height=height)
        elif format == ExportFormat.JPEG:
            fig.write_image(filename, width=width, height=height)
        elif format == ExportFormat.SVG:
            fig.write_image(filename, width=width, height=height)
        elif format == ExportFormat.PDF:
            fig.write_image(filename, width=width, height=height)
        elif format == ExportFormat.JSON:
            fig.write_json(filename)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Chart exported to {filename}")
        return filename
    
    def get_chart_as_base64(
        self,
        fig: go.Figure,
        format: ExportFormat = ExportFormat.PNG,
        width: int = 800,
        height: int = 600
    ) -> str:
        """
        Get chart as base64 encoded string.
        
        Args:
            fig: Plotly figure
            format: Export format
            width: Chart width
            height: Chart height
            
        Returns:
            Base64 encoded chart
        """
        if format == ExportFormat.PNG:
            img_bytes = fig.to_image(format="png", width=width, height=height)
        elif format == ExportFormat.JPEG:
            img_bytes = fig.to_image(format="jpeg", width=width, height=height)
        elif format == ExportFormat.SVG:
            img_bytes = fig.to_image(format="svg", width=width, height=height)
        elif format == ExportFormat.PDF:
            img_bytes = fig.to_image(format="pdf", width=width, height=height)
        else:
            raise ValueError(f"Unsupported format for base64: {format}")
        
        return base64.b64encode(img_bytes).decode()
    
    def create_interactive_table(
        self,
        data: pd.DataFrame,
        title: str = "Data Table",
        page_size: int = 10
    ) -> go.Figure:
        """
        Create an interactive data table.
        
        Args:
            data: DataFrame to display
            title: Table title
            page_size: Number of rows per page
            
        Returns:
            Plotly figure with table
        """
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(data.columns),
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=[data[col] for col in data.columns],
                fill_color='lavender',
                align='left'
            )
        )])
        
        fig.update_layout(
            title=title,
            template=self.default_theme
        )
        
        return fig
    
    def create_metric_summary_cards(
        self,
        metrics: Dict[str, float],
        title: str = "Performance Metrics"
    ) -> go.Figure:
        """
        Create metric summary cards.
        
        Args:
            metrics: Dictionary of metric names and values
            title: Chart title
            
        Returns:
            Plotly figure with metric cards
        """
        # Create subplots for each metric
        num_metrics = len(metrics)
        cols = min(4, num_metrics)
        rows = (num_metrics + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows,
            cols=cols,
            specs=[[{"type": "indicator"}] * cols for _ in range(rows)]
        )
        
        for i, (metric_name, value) in enumerate(metrics.items()):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=value,
                    title={"text": metric_name},
                    number={'font': {'size': 20}}
                ),
                row=row,
                col=col
            )
        
        fig.update_layout(
            title=title,
            template=self.default_theme,
            height=200 * rows,
            width=300 * cols
        )
        
        return fig
