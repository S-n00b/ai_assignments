"""
Modern Dashboard for Enterprise LLMOps Platform

This module provides a modern, responsive dashboard with real-time metrics,
customizable widgets, and advanced analytics capabilities inspired by
LangChain Studio, CopilotKit, and Neo4j interfaces.

Key Features:
- Real-time metrics and monitoring
- Customizable widget system
- Interactive data visualizations
- Agent workflow visualization
- Knowledge graph integration
- Modern responsive design
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st
from streamlit_plotly_events import plotly_events
import aiohttp
import websockets
from pathlib import Path
import base64


@dataclass
class DashboardWidget:
    """Configuration for a dashboard widget."""
    widget_id: str
    title: str
    widget_type: str  # "metric", "chart", "table", "agent_flow", "knowledge_graph"
    data_source: str
    position: Dict[str, int]  # x, y, width, height
    config: Dict[str, Any] = None
    refresh_interval: int = 30
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}


@dataclass
class AgentNode:
    """Node in an agent workflow."""
    node_id: str
    name: str
    agent_type: str
    status: str  # "running", "idle", "error", "completed"
    position: Dict[str, float]  # x, y coordinates
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AgentConnection:
    """Connection between agent nodes."""
    source_id: str
    target_id: str
    connection_type: str  # "data", "control", "feedback"
    status: str  # "active", "inactive", "error"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ModernDashboard:
    """
    Modern dashboard for Enterprise LLMOps platform.
    
    This class provides a comprehensive dashboard with real-time metrics,
    interactive visualizations, and modern UI components.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the modern dashboard."""
        self.config = config
        self.logger = self._setup_logging()
        self.widgets = {}
        self.agent_nodes = {}
        self.agent_connections = {}
        self.websocket_connections = {}
        
        # Initialize dashboard widgets
        self._setup_default_widgets()
        
        # Initialize agent workflow
        self._setup_agent_workflow()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for dashboard."""
        logger = logging.getLogger("modern_dashboard")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _setup_default_widgets(self):
        """Setup default dashboard widgets."""
        default_widgets = [
            DashboardWidget(
                widget_id="system_metrics",
                title="System Metrics",
                widget_type="metric",
                data_source="system_metrics",
                position={"x": 0, "y": 0, "width": 4, "height": 2}
            ),
            DashboardWidget(
                widget_id="model_performance",
                title="Model Performance",
                widget_type="chart",
                data_source="model_metrics",
                position={"x": 4, "y": 0, "width": 8, "height": 4}
            ),
            DashboardWidget(
                widget_id="agent_workflow",
                title="Agent Workflow",
                widget_type="agent_flow",
                data_source="agent_status",
                position={"x": 0, "y": 2, "width": 6, "height": 6}
            ),
            DashboardWidget(
                widget_id="knowledge_graph",
                title="Knowledge Graph",
                widget_type="knowledge_graph",
                data_source="knowledge_base",
                position={"x": 6, "y": 2, "width": 6, "height": 6}
            ),
            DashboardWidget(
                widget_id="recent_activity",
                title="Recent Activity",
                widget_type="table",
                data_source="activity_log",
                position={"x": 0, "y": 8, "width": 12, "height": 4}
            )
        ]
        
        for widget in default_widgets:
            self.widgets[widget.widget_id] = widget
    
    def _setup_agent_workflow(self):
        """Setup default agent workflow."""
        # Model Evaluation Agent
        self.agent_nodes["model_evaluator"] = AgentNode(
            node_id="model_evaluator",
            name="Model Evaluator",
            agent_type="evaluation",
            status="running",
            position={"x": 100, "y": 100},
            metadata={"description": "Evaluates model performance", "color": "#4CAF50"}
        )
        
        # Model Selector Agent
        self.agent_nodes["model_selector"] = AgentNode(
            node_id="model_selector",
            name="Model Selector",
            agent_type="selection",
            status="running",
            position={"x": 300, "y": 100},
            metadata={"description": "Selects optimal models", "color": "#2196F3"}
        )
        
        # Performance Monitor Agent
        self.agent_nodes["performance_monitor"] = AgentNode(
            node_id="performance_monitor",
            name="Performance Monitor",
            agent_type="monitoring",
            status="running",
            position={"x": 500, "y": 100},
            metadata={"description": "Monitors system performance", "color": "#FF9800"}
        )
        
        # Knowledge Manager Agent
        self.agent_nodes["knowledge_manager"] = AgentNode(
            node_id="knowledge_manager",
            name="Knowledge Manager",
            agent_type="knowledge",
            status="idle",
            position={"x": 300, "y": 300},
            metadata={"description": "Manages knowledge base", "color": "#9C27B0"}
        )
        
        # Agent connections
        self.agent_connections["eval_to_selector"] = AgentConnection(
            source_id="model_evaluator",
            target_id="model_selector",
            connection_type="data",
            status="active"
        )
        
        self.agent_connections["selector_to_monitor"] = AgentConnection(
            source_id="model_selector",
            target_id="performance_monitor",
            connection_type="control",
            status="active"
        )
        
        self.agent_connections["monitor_to_knowledge"] = AgentConnection(
            source_id="performance_monitor",
            target_id="knowledge_manager",
            connection_type="feedback",
            status="active"
        )
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics data."""
        try:
            # Simulate system metrics
            import psutil
            
            return {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "active_services": 8,
                "total_requests": 1250,
                "error_rate": 0.02,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to get system metrics: {e}")
            return {
                "cpu_usage": 45.2,
                "memory_usage": 67.8,
                "disk_usage": 23.4,
                "active_services": 8,
                "total_requests": 1250,
                "error_rate": 0.02,
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_model_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics."""
        try:
            # Simulate model metrics
            models = ["GPT-5", "Claude 3.5 Sonnet", "Llama 3.3", "GPT-5-Codex"]
            data = []
            
            for model in models:
                data.append({
                    "model": model,
                    "accuracy": 0.85 + (hash(model) % 100) / 1000,
                    "latency": 1.2 + (hash(model) % 50) / 100,
                    "cost": 0.03 + (hash(model) % 30) / 1000,
                    "requests": 100 + (hash(model) % 500),
                    "timestamp": datetime.now().isoformat()
                })
            
            return {"models": data}
        except Exception as e:
            self.logger.error(f"Failed to get model metrics: {e}")
            return {"models": []}
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get agent workflow status."""
        return {
            "nodes": [asdict(node) for node in self.agent_nodes.values()],
            "connections": [asdict(conn) for conn in self.agent_connections.values()],
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_knowledge_graph_data(self) -> Dict[str, Any]:
        """Get knowledge graph data."""
        try:
            # Simulate knowledge graph data
            nodes = [
                {"id": "gpt5", "label": "GPT-5", "type": "model", "size": 50, "color": "#4CAF50"},
                {"id": "claude", "label": "Claude 3.5", "type": "model", "size": 45, "color": "#2196F3"},
                {"id": "llama", "label": "Llama 3.3", "type": "model", "size": 40, "color": "#FF9800"},
                {"id": "evaluation", "label": "Evaluation", "type": "process", "size": 30, "color": "#9C27B0"},
                {"id": "selection", "label": "Selection", "type": "process", "size": 25, "color": "#F44336"},
                {"id": "deployment", "label": "Deployment", "type": "process", "size": 35, "color": "#607D8B"}
            ]
            
            edges = [
                {"source": "gpt5", "target": "evaluation", "weight": 0.9},
                {"source": "claude", "target": "evaluation", "weight": 0.85},
                {"source": "llama", "target": "evaluation", "weight": 0.8},
                {"source": "evaluation", "target": "selection", "weight": 1.0},
                {"source": "selection", "target": "deployment", "weight": 0.95}
            ]
            
            return {"nodes": nodes, "edges": edges}
        except Exception as e:
            self.logger.error(f"Failed to get knowledge graph data: {e}")
            return {"nodes": [], "edges": []}
    
    async def get_activity_log(self) -> List[Dict[str, Any]]:
        """Get recent activity log."""
        activities = [
            {
                "timestamp": (datetime.now() - timedelta(minutes=5)).isoformat(),
                "type": "model_evaluation",
                "message": "GPT-5 evaluation completed",
                "status": "success",
                "details": {"model": "GPT-5", "score": 0.92}
            },
            {
                "timestamp": (datetime.now() - timedelta(minutes=10)).isoformat(),
                "type": "model_selection",
                "message": "Optimal model selected for use case",
                "status": "success",
                "details": {"selected_model": "Claude 3.5 Sonnet", "confidence": 0.87}
            },
            {
                "timestamp": (datetime.now() - timedelta(minutes=15)).isoformat(),
                "type": "deployment",
                "message": "Model deployed to production",
                "status": "success",
                "details": {"environment": "production", "replicas": 3}
            },
            {
                "timestamp": (datetime.now() - timedelta(minutes=20)).isoformat(),
                "type": "monitoring",
                "message": "Performance alert triggered",
                "status": "warning",
                "details": {"metric": "response_time", "value": "2.5s"}
            }
        ]
        
        return activities
    
    def create_metric_widget(self, data: Dict[str, Any]) -> str:
        """Create a metric widget."""
        html = f"""
        <div class="metric-widget">
            <div class="metric-grid">
                <div class="metric-item">
                    <div class="metric-value">{data.get('cpu_usage', 0):.1f}%</div>
                    <div class="metric-label">CPU Usage</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{data.get('memory_usage', 0):.1f}%</div>
                    <div class="metric-label">Memory Usage</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{data.get('active_services', 0)}</div>
                    <div class="metric-label">Active Services</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{data.get('total_requests', 0)}</div>
                    <div class="metric-label">Total Requests</div>
                </div>
            </div>
        </div>
        """
        return html
    
    def create_chart_widget(self, data: Dict[str, Any]) -> str:
        """Create a chart widget."""
        if not data.get('models'):
            return "<div>No data available</div>"
        
        models_data = data['models']
        df = pd.DataFrame(models_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy', 'Latency', 'Cost', 'Requests'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Add traces
        fig.add_trace(
            go.Bar(x=df['model'], y=df['accuracy'], name='Accuracy'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=df['model'], y=df['latency'], name='Latency'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=df['model'], y=df['cost'], name='Cost'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=df['model'], y=df['requests'], name='Requests'),
            row=2, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        
        return fig.to_html(include_plotlyjs='cdn', div_id="model-performance-chart")
    
    def create_agent_flow_widget(self, data: Dict[str, Any]) -> str:
        """Create an agent workflow widget."""
        nodes = data.get('nodes', [])
        connections = data.get('connections', [])
        
        # Create HTML for agent flow visualization
        html = f"""
        <div class="agent-flow-widget">
            <svg width="600" height="400" class="agent-flow-svg">
                <!-- Agent Nodes -->
                {self._render_agent_nodes(nodes)}
                
                <!-- Connections -->
                {self._render_agent_connections(connections)}
            </svg>
        </div>
        """
        
        return html
    
    def _render_agent_nodes(self, nodes: List[Dict[str, Any]]) -> str:
        """Render agent nodes in SVG."""
        svg_nodes = ""
        for node in nodes:
            x = node['position']['x']
            y = node['position']['y']
            status_color = {"running": "#4CAF50", "idle": "#FFC107", "error": "#F44336"}.get(node['status'], "#9E9E9E")
            
            svg_nodes += f"""
            <g class="agent-node" data-node-id="{node['node_id']}">
                <circle cx="{x}" cy="{y}" r="30" fill="{status_color}" stroke="#333" stroke-width="2"/>
                <text x="{x}" y="{y}" text-anchor="middle" dy=".3em" font-size="10" fill="white">
                    {node['name'][:8]}
                </text>
            </g>
            """
        
        return svg_nodes
    
    def _render_agent_connections(self, connections: List[Dict[str, Any]]) -> str:
        """Render agent connections in SVG."""
        svg_connections = ""
        for conn in connections:
            source_node = next((n for n in self.agent_nodes.values() if n.node_id == conn['source_id']), None)
            target_node = next((n for n in self.agent_nodes.values() if n.node_id == conn['target_id']), None)
            
            if source_node and target_node:
                x1 = source_node.position['x']
                y1 = source_node.position['y']
                x2 = target_node.position['x']
                y2 = target_node.position['y']
                
                stroke_color = {"active": "#4CAF50", "inactive": "#9E9E9E", "error": "#F44336"}.get(conn['status'], "#9E9E9E")
                
                svg_connections += f"""
                <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" 
                      stroke="{stroke_color}" stroke-width="2" opacity="0.7"/>
                """
        
        return svg_connections
    
    def create_knowledge_graph_widget(self, data: Dict[str, Any]) -> str:
        """Create a knowledge graph widget."""
        nodes = data.get('nodes', [])
        edges = data.get('edges', [])
        
        # Create HTML for knowledge graph visualization
        html = f"""
        <div class="knowledge-graph-widget">
            <svg width="600" height="400" class="knowledge-graph-svg">
                <!-- Knowledge Nodes -->
                {self._render_knowledge_nodes(nodes)}
                
                <!-- Knowledge Edges -->
                {self._render_knowledge_edges(edges)}
            </svg>
        </div>
        """
        
        return html
    
    def _render_knowledge_nodes(self, nodes: List[Dict[str, Any]]) -> str:
        """Render knowledge graph nodes."""
        svg_nodes = ""
        for node in nodes:
            # Calculate position (simplified layout)
            x = 100 + (hash(node['id']) % 400)
            y = 100 + (hash(node['id'] + 'y') % 200)
            
            svg_nodes += f"""
            <g class="knowledge-node" data-node-id="{node['id']}">
                <circle cx="{x}" cy="{y}" r="{node.get('size', 20)}" 
                        fill="{node.get('color', '#4CAF50')}" 
                        stroke="#333" stroke-width="1"/>
                <text x="{x}" y="{y}" text-anchor="middle" dy=".3em" 
                      font-size="10" fill="white">
                    {node['label']}
                </text>
            </g>
            """
        
        return svg_nodes
    
    def _render_knowledge_edges(self, edges: List[Dict[str, Any]]) -> str:
        """Render knowledge graph edges."""
        svg_edges = ""
        for edge in edges:
            # Find source and target nodes (simplified)
            source_x = 100 + (hash(edge['source']) % 400)
            source_y = 100 + (hash(edge['source'] + 'y') % 200)
            target_x = 100 + (hash(edge['target']) % 400)
            target_y = 100 + (hash(edge['target'] + 'y') % 200)
            
            stroke_width = max(1, int(edge.get('weight', 0.5) * 5))
            
            svg_edges += f"""
            <line x1="{source_x}" y1="{source_y}" x2="{target_x}" y2="{target_y}" 
                  stroke="#666" stroke-width="{stroke_width}" opacity="0.6"/>
            """
        
        return svg_edges
    
    def create_table_widget(self, data: List[Dict[str, Any]]) -> str:
        """Create a table widget."""
        if not data:
            return "<div>No data available</div>"
        
        html = """
        <div class="table-widget">
            <table class="activity-table">
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Type</th>
                        <th>Message</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for item in data[:10]:  # Show last 10 items
            status_class = {"success": "status-success", "warning": "status-warning", "error": "status-error"}.get(item.get('status'), '')
            html += f"""
                    <tr>
                        <td>{item.get('timestamp', '')[:19]}</td>
                        <td>{item.get('type', '')}</td>
                        <td>{item.get('message', '')}</td>
                        <td class="{status_class}">{item.get('status', '')}</td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
        """
        
        return html
    
    async def render_dashboard(self) -> str:
        """Render the complete dashboard."""
        # Get data for all widgets
        system_metrics = await self.get_system_metrics()
        model_metrics = await self.get_model_metrics()
        agent_status = await self.get_agent_status()
        knowledge_data = await self.get_knowledge_graph_data()
        activity_log = await self.get_activity_log()
        
        # Create widget HTML
        widgets_html = ""
        
        for widget_id, widget in self.widgets.items():
            if widget.widget_type == "metric":
                widget_html = self.create_metric_widget(system_metrics)
            elif widget.widget_type == "chart":
                widget_html = self.create_chart_widget(model_metrics)
            elif widget.widget_type == "agent_flow":
                widget_html = self.create_agent_flow_widget(agent_status)
            elif widget.widget_type == "knowledge_graph":
                widget_html = self.create_knowledge_graph_widget(knowledge_data)
            elif widget.widget_type == "table":
                widget_html = self.create_table_widget(activity_log)
            else:
                widget_html = f"<div>Unknown widget type: {widget.widget_type}</div>"
            
            widgets_html += f"""
            <div class="dashboard-widget" 
                 style="grid-column: {widget.position['x'] + 1} / span {widget.position['width']};
                        grid-row: {widget.position['y'] + 1} / span {widget.position['height']};">
                <div class="widget-header">
                    <h3>{widget.title}</h3>
                </div>
                <div class="widget-content">
                    {widget_html}
                </div>
            </div>
            """
        
        # Create complete dashboard HTML
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Enterprise LLMOps Dashboard</title>
            <style>
                {self._get_dashboard_css()}
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <div class="dashboard-container">
                <div class="dashboard-header">
                    <h1>Enterprise LLMOps Dashboard</h1>
                    <div class="dashboard-controls">
                        <button onclick="refreshDashboard()">Refresh</button>
                        <button onclick="toggleTheme()">Toggle Theme</button>
                    </div>
                </div>
                <div class="dashboard-grid">
                    {widgets_html}
                </div>
            </div>
            <script>
                {self._get_dashboard_js()}
            </script>
        </body>
        </html>
        """
        
        return dashboard_html
    
    def _get_dashboard_css(self) -> str:
        """Get dashboard CSS styles."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }
        
        .dashboard-container {
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .dashboard-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            color: white;
        }
        
        .dashboard-header h1 {
            font-size: 2.5em;
            font-weight: 300;
        }
        
        .dashboard-controls button {
            background: rgba(255, 255, 255, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            margin-left: 10px;
            transition: all 0.3s ease;
        }
        
        .dashboard-controls button:hover {
            background: rgba(255, 255, 255, 0.3);
            transform: translateY(-2px);
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(12, 1fr);
            grid-template-rows: repeat(12, 100px);
            gap: 20px;
        }
        
        .dashboard-widget {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
        }
        
        .dashboard-widget:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
        }
        
        .widget-header h3 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.2em;
            font-weight: 600;
        }
        
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }
        
        .metric-item {
            text-align: center;
            padding: 15px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 10px;
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }
        
        .metric-label {
            font-size: 0.9em;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .agent-flow-widget svg {
            width: 100%;
            height: 100%;
            background: #f8f9fa;
            border-radius: 10px;
        }
        
        .knowledge-graph-widget svg {
            width: 100%;
            height: 100%;
            background: #f8f9fa;
            border-radius: 10px;
        }
        
        .activity-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }
        
        .activity-table th,
        .activity-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        .activity-table th {
            background: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }
        
        .status-success {
            color: #28a745;
            font-weight: bold;
        }
        
        .status-warning {
            color: #ffc107;
            font-weight: bold;
        }
        
        .status-error {
            color: #dc3545;
            font-weight: bold;
        }
        
        .dark-theme {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        }
        
        .dark-theme .dashboard-widget {
            background: rgba(44, 62, 80, 0.9);
            color: #ecf0f1;
        }
        
        .dark-theme .widget-header h3 {
            color: #ecf0f1;
        }
        
        .dark-theme .metric-item {
            background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
        }
        
        .dark-theme .metric-value {
            color: #ecf0f1;
        }
        
        .dark-theme .metric-label {
            color: #bdc3c7;
        }
        
        @media (max-width: 768px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
                grid-template-rows: auto;
            }
            
            .dashboard-widget {
                grid-column: 1 !important;
                grid-row: auto !important;
            }
            
            .metric-grid {
                grid-template-columns: 1fr;
            }
        }
        """
    
    def _get_dashboard_js(self) -> str:
        """Get dashboard JavaScript."""
        return """
        let isDarkTheme = false;
        
        function refreshDashboard() {
            location.reload();
        }
        
        function toggleTheme() {
            isDarkTheme = !isDarkTheme;
            document.body.classList.toggle('dark-theme', isDarkTheme);
        }
        
        // Auto-refresh every 30 seconds
        setInterval(refreshDashboard, 30000);
        
        // Add interactivity to agent nodes
        document.addEventListener('DOMContentLoaded', function() {
            const agentNodes = document.querySelectorAll('.agent-node');
            agentNodes.forEach(node => {
                node.addEventListener('click', function() {
                    const nodeId = this.dataset.nodeId;
                    alert('Clicked on agent: ' + nodeId);
                });
            });
            
            const knowledgeNodes = document.querySelectorAll('.knowledge-node');
            knowledgeNodes.forEach(node => {
                node.addEventListener('click', function() {
                    const nodeId = this.dataset.nodeId;
                    alert('Clicked on knowledge node: ' + nodeId);
                });
            });
        });
        """
    
    async def start_websocket_server(self, port: int = 8765):
        """Start WebSocket server for real-time updates."""
        async def handle_client(websocket, path):
            self.logger.info(f"Client connected: {websocket.remote_address}")
            self.websocket_connections[websocket] = True
            
            try:
                while True:
                    # Send real-time updates
                    update_data = {
                        "system_metrics": await self.get_system_metrics(),
                        "model_metrics": await self.get_model_metrics(),
                        "agent_status": await self.get_agent_status(),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    await websocket.send(json.dumps(update_data))
                    await asyncio.sleep(5)  # Update every 5 seconds
                    
            except websockets.exceptions.ConnectionClosed:
                self.logger.info(f"Client disconnected: {websocket.remote_address}")
            finally:
                if websocket in self.websocket_connections:
                    del self.websocket_connections[websocket]
        
        self.logger.info(f"Starting WebSocket server on port {port}")
        await websockets.serve(handle_client, "localhost", port)
    
    async def save_dashboard_config(self, filepath: str):
        """Save dashboard configuration to file."""
        config_data = {
            "widgets": {widget_id: asdict(widget) for widget_id, widget in self.widgets.items()},
            "agent_nodes": {node_id: asdict(node) for node_id, node in self.agent_nodes.items()},
            "agent_connections": {conn_id: asdict(conn) for conn_id, conn in self.agent_connections.items()},
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        self.logger.info(f"Dashboard configuration saved to {filepath}")
    
    async def load_dashboard_config(self, filepath: str):
        """Load dashboard configuration from file."""
        if not Path(filepath).exists():
            self.logger.warning(f"Dashboard config file not found: {filepath}")
            return
        
        with open(filepath, 'r') as f:
            config_data = json.load(f)
        
        # Load widgets
        self.widgets = {}
        for widget_id, widget_data in config_data.get("widgets", {}).items():
            self.widgets[widget_id] = DashboardWidget(**widget_data)
        
        # Load agent nodes
        self.agent_nodes = {}
        for node_id, node_data in config_data.get("agent_nodes", {}).items():
            self.agent_nodes[node_id] = AgentNode(**node_data)
        
        # Load agent connections
        self.agent_connections = {}
        for conn_id, conn_data in config_data.get("agent_connections", {}).items():
            self.agent_connections[conn_id] = AgentConnection(**conn_data)
        
        self.logger.info(f"Dashboard configuration loaded from {filepath}")


async def main():
    """Main function for testing the modern dashboard."""
    config = {
        "refresh_interval": 30,
        "websocket_port": 8765
    }
    
    dashboard = ModernDashboard(config)
    
    # Render dashboard
    dashboard_html = await dashboard.render_dashboard()
    
    # Save to file
    with open("modern_dashboard.html", "w") as f:
        f.write(dashboard_html)
    
    print("Modern dashboard rendered and saved to modern_dashboard.html")
    
    # Start WebSocket server for real-time updates
    await dashboard.start_websocket_server()


if __name__ == "__main__":
    asyncio.run(main())
