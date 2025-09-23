"""
Agent Visualization

This module provides visualization capabilities for LangGraph workflows,
including workflow diagrams, execution traces, and performance analytics.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid
from pathlib import Path

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
    import networkx as nx
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    matplotlib = None
    plt = None
    patches = None
    FancyBboxPatch = None
    Circle = None
    Rectangle = None
    nx = None
    go = None
    px = None
    make_subplots = None

logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    style: str = "modern"  # modern, classic, minimal
    color_scheme: str = "lenovo"  # lenovo, default, custom
    node_size: int = 1000
    edge_width: float = 2.0
    font_size: int = 12
    show_labels: bool = True
    show_metrics: bool = True
    interactive: bool = True


@dataclass
class NodeVisualization:
    """Node visualization data."""
    node_id: str
    node_type: str
    name: str
    position: Tuple[float, float]
    size: float
    color: str
    shape: str
    metrics: Dict[str, Any]
    status: str


@dataclass
class EdgeVisualization:
    """Edge visualization data."""
    source: str
    target: str
    weight: float
    color: str
    style: str
    label: Optional[str] = None


class AgentVisualization:
    """
    Agent Visualization for LangGraph workflows.
    
    This class provides comprehensive visualization capabilities
    including workflow diagrams, execution traces, and analytics.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize the Agent Visualization.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        self.color_schemes = {
            "lenovo": {
                "primary": "#E2231A",  # Lenovo red
                "secondary": "#000000",  # Black
                "accent": "#FFD700",  # Gold
                "background": "#F5F5F5",  # Light gray
                "text": "#333333"  # Dark gray
            },
            "default": {
                "primary": "#1f77b4",
                "secondary": "#ff7f0e",
                "accent": "#2ca02c",
                "background": "#ffffff",
                "text": "#000000"
            }
        }
        
        # Set up color scheme
        self.colors = self.color_schemes.get(self.config.color_scheme, self.color_schemes["default"])
        
        logger.info("Agent Visualization initialized")
    
    def create_workflow_diagram(self, workflow_config: Dict[str, Any], 
                              output_path: Optional[str] = None) -> str:
        """
        Create a workflow diagram.
        
        Args:
            workflow_config: Workflow configuration
            output_path: Path to save the diagram
            
        Returns:
            Path to the saved diagram
        """
        try:
            if not plt:
                raise ImportError("Matplotlib is not installed. Please install it with: pip install matplotlib")
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.set_aspect('equal')
            
            # Create network graph
            G = nx.DiGraph()
            
            # Add nodes
            for node in workflow_config.get('nodes', []):
                G.add_node(node['node_id'], **node)
            
            # Add edges
            for edge in workflow_config.get('edges', []):
                G.add_edge(edge['source_node'], edge['target_node'], **edge)
            
            # Calculate layout
            pos = nx.spring_layout(G, k=3, iterations=50)
            
            # Draw nodes
            self._draw_nodes(ax, G, pos)
            
            # Draw edges
            self._draw_edges(ax, G, pos)
            
            # Draw labels
            if self.config.show_labels:
                self._draw_labels(ax, G, pos)
            
            # Set title and styling
            ax.set_title(f"Workflow: {workflow_config.get('name', 'Unknown')}", 
                        fontsize=16, fontweight='bold')
            ax.axis('off')
            
            # Apply styling
            self._apply_styling(ax)
            
            # Save or return
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Workflow diagram saved to {output_path}")
                return output_path
            else:
                # Return as base64 string for web display
                import io
                import base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            logger.error(f"Failed to create workflow diagram: {e}")
            raise
    
    def _draw_nodes(self, ax, G, pos):
        """Draw workflow nodes."""
        for node_id, data in G.nodes(data=True):
            x, y = pos[node_id]
            
            # Get node type and set properties
            node_type = data.get('node_type', 'agent')
            color = self._get_node_color(node_type)
            shape = self._get_node_shape(node_type)
            size = self._get_node_size(node_type)
            
            # Draw node
            if shape == 'circle':
                circle = Circle((x, y), size/1000, color=color, alpha=0.8)
                ax.add_patch(circle)
            elif shape == 'rectangle':
                rect = Rectangle((x-size/2000, y-size/2000), size/1000, size/1000, 
                               color=color, alpha=0.8)
                ax.add_patch(rect)
            else:  # rounded rectangle
                bbox = FancyBboxPatch((x-size/2000, y-size/2000), size/1000, size/1000,
                                    boxstyle="round,pad=0.01", color=color, alpha=0.8)
                ax.add_patch(bbox)
            
            # Add node label
            if self.config.show_labels:
                ax.text(x, y, data.get('name', node_id), ha='center', va='center',
                       fontsize=self.config.font_size, fontweight='bold')
    
    def _draw_edges(self, ax, G, pos):
        """Draw workflow edges."""
        for edge in G.edges():
            source, target = edge
            x1, y1 = pos[source]
            x2, y2 = pos[target]
            
            # Draw edge
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', 
                                     color=self.colors['text'],
                                     lw=self.config.edge_width,
                                     alpha=0.7))
    
    def _draw_labels(self, ax, G, pos):
        """Draw edge labels."""
        for edge in G.edges():
            source, target = edge
            x1, y1 = pos[source]
            x2, y2 = pos[target]
            
            # Calculate midpoint
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            # Add edge label if available
            edge_data = G[source][target]
            if 'label' in edge_data:
                ax.text(mid_x, mid_y, edge_data['label'], ha='center', va='center',
                       fontsize=self.config.font_size-2, alpha=0.8)
    
    def _get_node_color(self, node_type: str) -> str:
        """Get color for node type."""
        color_map = {
            'start': self.colors['accent'],
            'end': self.colors['secondary'],
            'agent': self.colors['primary'],
            'tool': '#4CAF50',  # Green
            'conditional': '#FF9800',  # Orange
            'parallel': '#9C27B0',  # Purple
            'sequential': '#2196F3'  # Blue
        }
        return color_map.get(node_type, self.colors['primary'])
    
    def _get_node_shape(self, node_type: str) -> str:
        """Get shape for node type."""
        shape_map = {
            'start': 'circle',
            'end': 'circle',
            'agent': 'rounded_rectangle',
            'tool': 'rectangle',
            'conditional': 'diamond',
            'parallel': 'rounded_rectangle',
            'sequential': 'rounded_rectangle'
        }
        return shape_map.get(node_type, 'rounded_rectangle')
    
    def _get_node_size(self, node_type: str) -> float:
        """Get size for node type."""
        size_map = {
            'start': self.config.node_size * 0.8,
            'end': self.config.node_size * 0.8,
            'agent': self.config.node_size,
            'tool': self.config.node_size * 0.9,
            'conditional': self.config.node_size * 0.9,
            'parallel': self.config.node_size,
            'sequential': self.config.node_size
        }
        return size_map.get(node_type, self.config.node_size)
    
    def _apply_styling(self, ax):
        """Apply styling to the diagram."""
        ax.set_facecolor(self.colors['background'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    def create_execution_trace(self, execution_data: Dict[str, Any], 
                             output_path: Optional[str] = None) -> str:
        """
        Create an execution trace visualization.
        
        Args:
            execution_data: Execution data
            output_path: Path to save the trace
            
        Returns:
            Path to the saved trace
        """
        try:
            if not go:
                raise ImportError("Plotly is not installed. Please install it with: pip install plotly")
            
            # Extract execution steps
            steps = execution_data.get('execution_steps', [])
            if not steps:
                raise ValueError("No execution steps found")
            
            # Create timeline data
            timestamps = []
            node_names = []
            execution_times = []
            statuses = []
            
            for i, step in enumerate(steps):
                timestamps.append(i)
                node_names.append(step.get('node_name', f'Step {i}'))
                execution_times.append(step.get('execution_time_ms', 0))
                statuses.append(step.get('status', 'completed'))
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Execution Timeline', 'Performance Metrics'),
                vertical_spacing=0.1
            )
            
            # Timeline plot
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=node_names,
                    mode='markers+lines',
                    marker=dict(size=10, color=self.colors['primary']),
                    line=dict(color=self.colors['primary'], width=2),
                    name='Execution Flow'
                ),
                row=1, col=1
            )
            
            # Performance plot
            fig.add_trace(
                go.Bar(
                    x=node_names,
                    y=execution_times,
                    marker_color=self.colors['accent'],
                    name='Execution Time (ms)'
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                title=f"Execution Trace: {execution_data.get('workflow_id', 'Unknown')}",
                showlegend=True,
                height=800,
                plot_bgcolor=self.colors['background']
            )
            
            # Update axes
            fig.update_xaxes(title_text="Step", row=1, col=1)
            fig.update_yaxes(title_text="Node", row=1, col=1)
            fig.update_xaxes(title_text="Node", row=2, col=1)
            fig.update_yaxes(title_text="Execution Time (ms)", row=2, col=1)
            
            # Save or return
            if output_path:
                fig.write_html(output_path)
                logger.info(f"Execution trace saved to {output_path}")
                return output_path
            else:
                return fig.to_html()
            
        except Exception as e:
            logger.error(f"Failed to create execution trace: {e}")
            raise
    
    def create_performance_analytics(self, performance_data: List[Dict[str, Any]], 
                                   output_path: Optional[str] = None) -> str:
        """
        Create performance analytics visualization.
        
        Args:
            performance_data: Performance data
            output_path: Path to save the analytics
            
        Returns:
            Path to the saved analytics
        """
        try:
            if not go:
                raise ImportError("Plotly is not installed. Please install it with: pip install plotly")
            
            # Extract metrics
            timestamps = []
            execution_times = []
            memory_usage = []
            cpu_usage = []
            success_rates = []
            
            for data in performance_data:
                timestamps.append(data.get('timestamp', ''))
                execution_times.append(data.get('execution_time_ms', 0))
                memory_usage.append(data.get('memory_usage_mb', 0))
                cpu_usage.append(data.get('cpu_usage_percent', 0))
                success_rates.append(data.get('success_rate', 0))
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Execution Time', 'Memory Usage', 'CPU Usage', 'Success Rate'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Execution time plot
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=execution_times,
                    mode='lines+markers',
                    name='Execution Time',
                    line=dict(color=self.colors['primary'])
                ),
                row=1, col=1
            )
            
            # Memory usage plot
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=memory_usage,
                    mode='lines+markers',
                    name='Memory Usage',
                    line=dict(color=self.colors['secondary'])
                ),
                row=1, col=2
            )
            
            # CPU usage plot
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=cpu_usage,
                    mode='lines+markers',
                    name='CPU Usage',
                    line=dict(color=self.colors['accent'])
                ),
                row=2, col=1
            )
            
            # Success rate plot
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=success_rates,
                    mode='lines+markers',
                    name='Success Rate',
                    line=dict(color='#4CAF50')
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title="Performance Analytics Dashboard",
                showlegend=True,
                height=800,
                plot_bgcolor=self.colors['background']
            )
            
            # Update axes
            fig.update_xaxes(title_text="Time", row=1, col=1)
            fig.update_yaxes(title_text="Execution Time (ms)", row=1, col=1)
            fig.update_xaxes(title_text="Time", row=1, col=2)
            fig.update_yaxes(title_text="Memory Usage (MB)", row=1, col=2)
            fig.update_xaxes(title_text="Time", row=2, col=1)
            fig.update_yaxes(title_text="CPU Usage (%)", row=2, col=1)
            fig.update_xaxes(title_text="Time", row=2, col=2)
            fig.update_yaxes(title_text="Success Rate", row=2, col=2)
            
            # Save or return
            if output_path:
                fig.write_html(output_path)
                logger.info(f"Performance analytics saved to {output_path}")
                return output_path
            else:
                return fig.to_html()
            
        except Exception as e:
            logger.error(f"Failed to create performance analytics: {e}")
            raise
    
    def create_interactive_dashboard(self, workflow_data: Dict[str, Any], 
                                   execution_data: Dict[str, Any],
                                   performance_data: List[Dict[str, Any]]) -> str:
        """
        Create an interactive dashboard.
        
        Args:
            workflow_data: Workflow configuration
            execution_data: Execution data
            performance_data: Performance data
            
        Returns:
            HTML dashboard
        """
        try:
            if not go:
                raise ImportError("Plotly is not installed. Please install it with: pip install plotly")
            
            # Create dashboard layout
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Workflow Overview', 'Execution Timeline', 
                              'Performance Metrics', 'Resource Usage',
                              'Success Rate', 'Error Analysis'),
                specs=[[{"type": "scatter"}, {"type": "scatter"}],
                       [{"type": "bar"}, {"type": "scatter"}],
                       [{"type": "scatter"}, {"type": "bar"}]]
            )
            
            # Workflow overview (placeholder)
            fig.add_trace(
                go.Scatter(
                    x=[1, 2, 3, 4],
                    y=[1, 2, 3, 4],
                    mode='markers+lines',
                    name='Workflow Flow',
                    marker=dict(size=15, color=self.colors['primary'])
                ),
                row=1, col=1
            )
            
            # Execution timeline
            steps = execution_data.get('execution_steps', [])
            if steps:
                timestamps = list(range(len(steps)))
                execution_times = [step.get('execution_time_ms', 0) for step in steps]
                
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=execution_times,
                        mode='lines+markers',
                        name='Execution Time',
                        line=dict(color=self.colors['secondary'])
                    ),
                    row=1, col=2
                )
            
            # Performance metrics
            if performance_data:
                timestamps = [data.get('timestamp', '') for data in performance_data]
                execution_times = [data.get('execution_time_ms', 0) for data in performance_data]
                
                fig.add_trace(
                    go.Bar(
                        x=timestamps,
                        y=execution_times,
                        name='Performance',
                        marker_color=self.colors['accent']
                    ),
                    row=2, col=1
                )
            
            # Resource usage
            if performance_data:
                memory_usage = [data.get('memory_usage_mb', 0) for data in performance_data]
                cpu_usage = [data.get('cpu_usage_percent', 0) for data in performance_data]
                
                fig.add_trace(
                    go.Scatter(
                        x=memory_usage,
                        y=cpu_usage,
                        mode='markers',
                        name='Resource Usage',
                        marker=dict(size=10, color=self.colors['primary'])
                    ),
                    row=2, col=2
                )
            
            # Success rate
            if performance_data:
                success_rates = [data.get('success_rate', 0) for data in performance_data]
                
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=success_rates,
                        mode='lines+markers',
                        name='Success Rate',
                        line=dict(color='#4CAF50')
                    ),
                    row=3, col=1
                )
            
            # Error analysis
            error_counts = [data.get('error_count', 0) for data in performance_data]
            
            fig.add_trace(
                go.Bar(
                    x=timestamps,
                    y=error_counts,
                    name='Errors',
                    marker_color='#F44336'
                ),
                row=3, col=2
            )
            
            # Update layout
            fig.update_layout(
                title="LangGraph Workflow Dashboard",
                showlegend=True,
                height=1200,
                plot_bgcolor=self.colors['background']
            )
            
            return fig.to_html()
            
        except Exception as e:
            logger.error(f"Failed to create interactive dashboard: {e}")
            raise
    
    def export_visualization(self, visualization_data: Dict[str, Any], 
                           export_path: str, format: str = "html") -> bool:
        """
        Export visualization to file.
        
        Args:
            visualization_data: Visualization data
            export_path: Path to export file
            format: Export format (html, png, svg)
            
        Returns:
            True if successful
        """
        try:
            if format == "html":
                with open(export_path, 'w') as f:
                    f.write(visualization_data)
            elif format == "png":
                # This would require additional setup for static image export
                logger.warning("PNG export not implemented yet")
                return False
            elif format == "svg":
                # This would require additional setup for SVG export
                logger.warning("SVG export not implemented yet")
                return False
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Visualization exported to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export visualization: {e}")
            return False


# Factory functions for common visualization scenarios
def create_lenovo_visualization() -> AgentVisualization:
    """Create a Lenovo-themed visualization."""
    config = VisualizationConfig(
        style="modern",
        color_scheme="lenovo",
        node_size=1200,
        font_size=14,
        show_labels=True,
        show_metrics=True,
        interactive=True
    )
    return AgentVisualization(config)


def create_mobile_visualization() -> AgentVisualization:
    """Create a mobile-optimized visualization."""
    config = VisualizationConfig(
        style="minimal",
        color_scheme="default",
        node_size=800,
        font_size=10,
        show_labels=True,
        show_metrics=False,
        interactive=False
    )
    return AgentVisualization(config)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create visualization
    viz = create_lenovo_visualization()
    
    # Example workflow data
    workflow_data = {
        "name": "Lenovo Device Support",
        "nodes": [
            {"node_id": "start", "node_type": "start", "name": "Start"},
            {"node_id": "analyzer", "node_type": "agent", "name": "Device Analyzer"},
            {"node_id": "solution", "node_type": "agent", "name": "Solution Provider"},
            {"node_id": "end", "node_type": "end", "name": "End"}
        ],
        "edges": [
            {"source_node": "start", "target_node": "analyzer"},
            {"source_node": "analyzer", "target_node": "solution"},
            {"source_node": "solution", "target_node": "end"}
        ]
    }
    
    # Create workflow diagram
    try:
        diagram_path = viz.create_workflow_diagram(workflow_data, "workflow_diagram.png")
        print(f"Workflow diagram created: {diagram_path}")
    except Exception as e:
        print(f"Failed to create workflow diagram: {e}")
    
    # Example execution data
    execution_data = {
        "workflow_id": "test_workflow",
        "execution_steps": [
            {"node_name": "Start", "execution_time_ms": 10, "status": "completed"},
            {"node_name": "Device Analyzer", "execution_time_ms": 150, "status": "completed"},
            {"node_name": "Solution Provider", "execution_time_ms": 200, "status": "completed"},
            {"node_name": "End", "execution_time_ms": 5, "status": "completed"}
        ]
    }
    
    # Create execution trace
    try:
        trace_html = viz.create_execution_trace(execution_data)
        print("Execution trace created")
    except Exception as e:
        print(f"Failed to create execution trace: {e}")
