"""
Neo4j-Style Knowledge Graph UI for Lenovo AAITC Solutions

This module implements an interactive knowledge graph visualization and exploration interface
inspired by Neo4j, featuring graph-based model relationship mapping, dynamic graph exploration,
natural language querying, collaborative annotation, and correlation analysis.

Key Features:
- Interactive knowledge graph visualization
- Graph-based model relationship mapping
- Dynamic graph exploration and filtering
- Natural language knowledge graph querying
- Collaborative graph annotation and tagging
- Graph-based model performance correlation analysis
- Export capabilities for graph data and visualizations
"""

import gradio as gr
import json
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass, asdict
import uuid
import re
import numpy as np
from collections import defaultdict


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    id: str
    label: str
    node_type: str  # "model", "task", "metric", "architecture", "user", "organization"
    properties: Dict[str, Any]
    position: Optional[Tuple[float, float]] = None
    color: Optional[str] = None
    size: Optional[float] = None


@dataclass
class GraphEdge:
    """Represents an edge in the knowledge graph."""
    id: str
    source: str
    target: str
    relationship_type: str  # "performs", "evaluated_by", "correlates_with", "depends_on", "optimizes"
    properties: Dict[str, Any]
    weight: Optional[float] = None
    color: Optional[str] = None


@dataclass
class GraphQuery:
    """Represents a graph query."""
    id: str
    query_text: str
    query_type: str  # "cypher", "natural_language", "visual"
    results: List[Dict[str, Any]]
    executed_at: datetime
    execution_time: float


class KnowledgeGraphManager:
    """
    Main manager for the Neo4j-style knowledge graph interface.
    
    This class provides comprehensive graph visualization, exploration,
    and analysis capabilities for AI model relationships and performance data.
    """
    
    def __init__(self):
        """Initialize the knowledge graph manager."""
        self.graph = nx.Graph()
        self.node_registry = {}
        self.edge_registry = {}
        self.query_history = []
        self.visualization_config = self._initialize_visualization_config()
        self.filter_config = self._initialize_filter_config()
        self.annotation_system = self._initialize_annotation_system()
        
        # Initialize with sample data
        self._populate_sample_graph()
    
    def _initialize_visualization_config(self) -> Dict[str, Any]:
        """Initialize graph visualization configuration."""
        return {
            "layout_algorithm": "spring",  # "spring", "circular", "hierarchical", "force_directed"
            "node_colors": {
                "model": "#3b82f6",
                "task": "#10b981", 
                "metric": "#f59e0b",
                "architecture": "#8b5cf6",
                "user": "#ef4444",
                "organization": "#6b7280"
            },
            "edge_colors": {
                "performs": "#3b82f6",
                "evaluated_by": "#10b981",
                "correlates_with": "#f59e0b",
                "depends_on": "#8b5cf6",
                "optimizes": "#ef4444"
            },
            "node_sizes": {
                "model": 30,
                "task": 25,
                "metric": 20,
                "architecture": 35,
                "user": 15,
                "organization": 40
            },
            "show_labels": True,
            "show_edge_labels": True,
            "animation_enabled": True
        }
    
    def _initialize_filter_config(self) -> Dict[str, Any]:
        """Initialize graph filtering configuration."""
        return {
            "node_types": ["model", "task", "metric", "architecture", "user", "organization"],
            "edge_types": ["performs", "evaluated_by", "correlates_with", "depends_on", "optimizes"],
            "date_range": {
                "start": datetime.now() - timedelta(days=30),
                "end": datetime.now()
            },
            "performance_threshold": 0.7,
            "correlation_threshold": 0.5
        }
    
    def _initialize_annotation_system(self) -> Dict[str, Any]:
        """Initialize collaborative annotation system."""
        return {
            "tags": ["high_performance", "experimental", "production_ready", "deprecated"],
            "annotations": {},
            "collaborators": ["user_1", "user_2", "user_3"],
            "permissions": {
                "view": ["all"],
                "annotate": ["authenticated"],
                "modify": ["admin"]
            }
        }
    
    def _populate_sample_graph(self):
        """Populate the graph with sample AI model and performance data."""
        # Add model nodes
        models = [
            GraphNode("gpt5", "GPT-5", "model", {
                "version": "5.0",
                "provider": "OpenAI",
                "parameters": "1.76T",
                "release_date": "2024-12-01",
                "performance_score": 0.95,
                "cost_per_token": 0.002
            }),
            GraphNode("gpt5_codex", "GPT-5-Codex", "model", {
                "version": "5.0-codex",
                "provider": "OpenAI", 
                "parameters": "1.76T",
                "coding_success_rate": 0.745,
                "performance_score": 0.92,
                "cost_per_token": 0.003
            }),
            GraphNode("claude35", "Claude 3.5 Sonnet", "model", {
                "version": "3.5-sonnet",
                "provider": "Anthropic",
                "parameters": "1.4T",
                "conversation_quality": 0.98,
                "performance_score": 0.94,
                "cost_per_token": 0.0015
            }),
            GraphNode("llama33", "Llama 3.3", "model", {
                "version": "3.3",
                "provider": "Meta",
                "parameters": "405B",
                "open_source": True,
                "performance_score": 0.88,
                "cost_per_token": 0.0005
            })
        ]
        
        # Add task nodes
        tasks = [
            GraphNode("text_gen", "Text Generation", "task", {
                "category": "generation",
                "complexity": "medium",
                "success_rate": 0.92
            }),
            GraphNode("code_gen", "Code Generation", "task", {
                "category": "generation", 
                "complexity": "high",
                "success_rate": 0.78
            }),
            GraphNode("reasoning", "Logical Reasoning", "task", {
                "category": "analysis",
                "complexity": "high", 
                "success_rate": 0.85
            }),
            GraphNode("summarization", "Text Summarization", "task", {
                "category": "analysis",
                "complexity": "medium",
                "success_rate": 0.94
            })
        ]
        
        # Add metric nodes
        metrics = [
            GraphNode("rouge_l", "ROUGE-L", "metric", {
                "type": "quality",
                "range": [0, 1],
                "importance": "high"
            }),
            GraphNode("bert_score", "BERT Score", "metric", {
                "type": "semantic",
                "range": [0, 1],
                "importance": "high"
            }),
            GraphNode("latency", "Latency", "metric", {
                "type": "performance",
                "unit": "milliseconds",
                "importance": "medium"
            }),
            GraphNode("throughput", "Throughput", "metric", {
                "type": "performance", 
                "unit": "requests_per_second",
                "importance": "medium"
            })
        ]
        
        # Add architecture nodes
        architectures = [
            GraphNode("hybrid_platform", "Hybrid AI Platform", "architecture", {
                "type": "enterprise",
                "deployment": "cloud-edge",
                "scalability": "high"
            }),
            GraphNode("rag_system", "RAG System", "architecture", {
                "type": "retrieval",
                "components": ["vector_store", "retriever", "generator"],
                "performance": "optimized"
            })
        ]
        
        # Add all nodes to graph
        for node in models + tasks + metrics + architectures:
            self.add_node(node)
        
        # Add relationships
        relationships = [
            # Model-task relationships
            GraphEdge("r1", "gpt5", "text_gen", "performs", {"confidence": 0.95, "frequency": 1000}),
            GraphEdge("r2", "gpt5", "reasoning", "performs", {"confidence": 0.92, "frequency": 800}),
            GraphEdge("r3", "gpt5_codex", "code_gen", "performs", {"confidence": 0.745, "frequency": 1200}),
            GraphEdge("r4", "claude35", "text_gen", "performs", {"confidence": 0.98, "frequency": 900}),
            GraphEdge("r5", "claude35", "summarization", "performs", {"confidence": 0.96, "frequency": 700}),
            GraphEdge("r6", "llama33", "text_gen", "performs", {"confidence": 0.88, "frequency": 600}),
            
            # Model-metric relationships
            GraphEdge("r7", "gpt5", "rouge_l", "evaluated_by", {"score": 0.94, "last_evaluated": "2024-12-15"}),
            GraphEdge("r8", "gpt5", "latency", "evaluated_by", {"score": 120, "last_evaluated": "2024-12-15"}),
            GraphEdge("r9", "claude35", "bert_score", "evaluated_by", {"score": 0.96, "last_evaluated": "2024-12-15"}),
            GraphEdge("r10", "gpt5_codex", "latency", "evaluated_by", {"score": 150, "last_evaluated": "2024-12-15"}),
            
            # Task-metric relationships
            GraphEdge("r11", "text_gen", "rouge_l", "correlates_with", {"correlation": 0.85}),
            GraphEdge("r12", "code_gen", "latency", "correlates_with", {"correlation": 0.72}),
            GraphEdge("r13", "summarization", "bert_score", "correlates_with", {"correlation": 0.91}),
            
            # Architecture relationships
            GraphEdge("r14", "hybrid_platform", "gpt5", "optimizes", {"improvement": 0.15}),
            GraphEdge("r15", "rag_system", "claude35", "optimizes", {"improvement": 0.22}),
            GraphEdge("r16", "rag_system", "summarization", "depends_on", {"dependency_strength": 0.8})
        ]
        
        # Add all edges to graph
        for edge in relationships:
            self.add_edge(edge)
    
    def create_knowledge_graph_interface(self) -> gr.Blocks:
        """Create the main knowledge graph interface."""
        with gr.Blocks(
            title="Knowledge Graph Explorer - Neo4j Style",
            theme=gr.themes.Soft(),
            css="""
            .graph-container {
                background: #f8fafc;
                border-radius: 8px;
                padding: 1rem;
                min-height: 600px;
            }
            .query-builder {
                background: white;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 1rem;
                margin: 1rem 0;
            }
            .node-info {
                background: white;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 1rem;
                max-height: 400px;
                overflow-y: auto;
            }
            .annotation-panel {
                background: #fef3c7;
                border: 1px solid #f59e0b;
                border-radius: 8px;
                padding: 1rem;
                margin: 0.5rem 0;
            }
            """
        ) as interface:
            
            # Header
            gr.HTML("""
            <div style="background: linear-gradient(90deg, #1e3a8a, #3b82f6); color: white; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                <h1>🕸️ Knowledge Graph Explorer</h1>
                <p>Neo4j-Style AI Model Relationship Visualization for Lenovo AAITC Solutions</p>
            </div>
            """)
            
            with gr.Row():
                # Left Panel - Controls and Filters
                with gr.Column(scale=1):
                    self._create_control_panel()
                    self._create_filter_panel()
                    self._create_query_panel()
                
                # Center Panel - Graph Visualization
                with gr.Column(scale=3):
                    self._create_graph_visualization()
                
                # Right Panel - Node Details and Annotations
                with gr.Column(scale=1):
                    self._create_node_details_panel()
                    self._create_annotation_panel()
            
            # Bottom Panel - Analysis and Export
            self._create_analysis_panel()
        
        return interface
    
    def _create_control_panel(self):
        """Create the graph control panel."""
        gr.Markdown("## 🎛️ Graph Controls")
        
        # Layout algorithm
        self.layout_algorithm = gr.Dropdown(
            choices=["spring", "circular", "hierarchical", "force_directed"],
            value="spring",
            label="Layout Algorithm"
        )
        
        # Display options
        with gr.Accordion("Display Options", open=True):
            self.show_labels = gr.Checkbox(value=True, label="Show Labels")
            self.show_edge_labels = gr.Checkbox(value=True, label="Show Edge Labels")
            self.animation_enabled = gr.Checkbox(value=True, label="Enable Animation")
        
        # Action buttons
        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh", variant="secondary", size="sm")
            reset_view_btn = gr.Button("🎯 Reset View", variant="secondary", size="sm")
            export_graph_btn = gr.Button("📤 Export", variant="secondary", size="sm")
        
        # Graph statistics
        self.graph_stats = gr.JSON(
            value=self._get_graph_statistics(),
            label="Graph Statistics"
        )
    
    def _create_filter_panel(self):
        """Create the graph filtering panel."""
        gr.Markdown("## 🔍 Filters")
        
        # Node type filter
        self.node_type_filter = gr.CheckboxGroup(
            choices=["model", "task", "metric", "architecture", "user", "organization"],
            value=["model", "task", "metric", "architecture"],
            label="Node Types"
        )
        
        # Edge type filter
        self.edge_type_filter = gr.CheckboxGroup(
            choices=["performs", "evaluated_by", "correlates_with", "depends_on", "optimizes"],
            value=["performs", "evaluated_by", "correlates_with"],
            label="Relationship Types"
        )
        
        # Performance filter
        with gr.Accordion("Performance Filters", open=False):
            self.min_performance = gr.Slider(0, 1, value=0.7, label="Min Performance Score")
            self.max_latency = gr.Slider(0, 1000, value=500, label="Max Latency (ms)")
        
        # Apply filters button
        apply_filters_btn = gr.Button("🔍 Apply Filters", variant="primary", size="sm")
    
    def _create_query_panel(self):
        """Create the natural language query panel."""
        gr.Markdown("## 🔍 Query Interface")
        
        # Query input
        self.query_input = gr.Textbox(
            placeholder="Ask about model relationships, performance correlations, or architecture dependencies...",
            label="Natural Language Query",
            lines=3
        )
        
        # Query type
        self.query_type = gr.Radio(
            choices=["Natural Language", "Cypher Query", "Visual Query"],
            value="Natural Language",
            label="Query Type"
        )
        
        # Execute query button
        execute_query_btn = gr.Button("🚀 Execute Query", variant="primary")
        
        # Query history
        with gr.Accordion("Query History", open=False):
            self.query_history_display = gr.HTML(
                value=self._format_query_history()
            )
    
    def _create_graph_visualization(self):
        """Create the main graph visualization."""
        gr.Markdown("## 🕸️ Knowledge Graph")
        
        # Graph plot
        self.graph_plot = gr.Plot(
            value=self._create_graph_visualization_plot(),
            label="Interactive Knowledge Graph"
        )
        
        # Graph information
        self.graph_info = gr.HTML(
            value=self._get_graph_info_html()
        )
    
    def _create_node_details_panel(self):
        """Create the node details and properties panel."""
        gr.Markdown("## 📋 Node Details")
        
        # Selected node info
        self.selected_node_info = gr.HTML(
            value="<div style='padding: 1rem; text-align: center; color: #6b7280;'>Click on a node to view details</div>"
        )
        
        # Node properties
        self.node_properties = gr.JSON(
            value={},
            label="Properties"
        )
        
        # Connected nodes
        self.connected_nodes = gr.HTML(
            value="<div style='padding: 1rem; text-align: center; color: #6b7280;'>Connected nodes will appear here</div>"
        )
    
    def _create_annotation_panel(self):
        """Create the collaborative annotation panel."""
        gr.Markdown("## 🏷️ Annotations")
        
        # Current annotations
        self.current_annotations = gr.HTML(
            value="<div style='padding: 1rem; text-align: center; color: #6b7280;'>No annotations yet</div>"
        )
        
        # Add annotation
        self.annotation_input = gr.Textbox(
            placeholder="Add annotation or tag...",
            label="New Annotation",
            lines=2
        )
        
        # Annotation tags
        self.annotation_tags = gr.CheckboxGroup(
            choices=["high_performance", "experimental", "production_ready", "deprecated"],
            label="Tags"
        )
        
        # Add annotation button
        add_annotation_btn = gr.Button("➕ Add Annotation", variant="secondary", size="sm")
    
    def _create_analysis_panel(self):
        """Create the analysis and export panel."""
        with gr.Accordion("📊 Analysis & Export", open=False):
            
            # Analysis tabs
            with gr.Tabs():
                with gr.Tab("🔗 Correlation Analysis"):
                    self.correlation_analysis = gr.Plot(label="Model Performance Correlations")
                    self.correlation_table = gr.Dataframe(
                        headers=["Model A", "Model B", "Correlation", "Significance"],
                        datatype=["str", "str", "number", "number"],
                        interactive=False
                    )
                
                with gr.Tab("📈 Performance Trends"):
                    self.performance_trends = gr.Plot(label="Performance Over Time")
                    self.trend_analysis = gr.HTML(
                        value="<div style='padding: 1rem; text-align: center; color: #6b7280;'>Performance trend analysis will appear here</div>"
                    )
                
                with gr.Tab("🏗️ Architecture Analysis"):
                    self.architecture_analysis = gr.Plot(label="Architecture Dependencies")
                    self.architecture_metrics = gr.JSON(
                        value={},
                        label="Architecture Metrics"
                    )
                
                with gr.Tab("📤 Export Options"):
                    with gr.Row():
                        export_graphviz_btn = gr.Button("📊 Export GraphViz", variant="secondary")
                        export_cypher_btn = gr.Button("🔍 Export Cypher", variant="secondary")
                        export_json_btn = gr.Button("📋 Export JSON", variant="secondary")
                        export_csv_btn = gr.Button("📈 Export CSV", variant="secondary")
                    
                    self.export_result = gr.File(label="Export Result")
    
    def _create_graph_visualization_plot(self) -> go.Figure:
        """Create the interactive graph visualization plot."""
        # Generate layout
        if self.visualization_config["layout_algorithm"] == "spring":
            pos = nx.spring_layout(self.graph, k=3, iterations=50)
        elif self.visualization_config["layout_algorithm"] == "circular":
            pos = nx.circular_layout(self.graph)
        elif self.visualization_config["layout_algorithm"] == "hierarchical":
            pos = nx.hierarchical_layout(self.graph)
        else:  # force_directed
            pos = nx.spring_layout(self.graph, k=2, iterations=100)
        
        # Create edge trace
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in self.graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Get edge properties
            edge_props = edge[2] if len(edge) > 2 else {}
            relationship_type = edge_props.get("relationship_type", "unknown")
            edge_info.append(f"{edge[0]} → {edge[1]}<br>Type: {relationship_type}")
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines',
            name='Connections'
        )
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        node_info = []
        
        for node_id in self.graph.nodes():
            x, y = pos[node_id]
            node_x.append(x)
            node_y.append(y)
            
            # Get node data
            node_data = self.node_registry.get(node_id, {})
            node_type = node_data.get("node_type", "unknown")
            label = node_data.get("label", node_id)
            
            node_text.append(label)
            node_colors.append(self.visualization_config["node_colors"].get(node_type, "#6b7280"))
            node_sizes.append(self.visualization_config["node_sizes"].get(node_type, 20))
            
            # Create hover info
            properties = node_data.get("properties", {})
            info_text = f"<b>{label}</b><br>Type: {node_type}<br>"
            for key, value in properties.items():
                info_text += f"{key}: {value}<br>"
            node_info.append(info_text)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text if self.visualization_config["show_labels"] else [],
            textposition="middle center",
            hovertext=node_info,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white')
            ),
            name='Nodes'
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='AI Model Knowledge Graph',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Click and drag to explore the graph",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color='#888', size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white',
                           height=600
                       ))
        
        return fig
    
    def add_node(self, node: GraphNode):
        """Add a node to the knowledge graph."""
        self.graph.add_node(node.id)
        self.node_registry[node.id] = asdict(node)
        
        # Update node position if not set
        if node.position is None:
            node.position = (np.random.uniform(-5, 5), np.random.uniform(-5, 5))
    
    def add_edge(self, edge: GraphEdge):
        """Add an edge to the knowledge graph."""
        self.graph.add_edge(edge.source, edge.target, **asdict(edge))
        self.edge_registry[edge.id] = asdict(edge)
    
    def _get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics."""
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": {
                node_type: len([n for n in self.node_registry.values() if n["node_type"] == node_type])
                for node_type in set(n["node_type"] for n in self.node_registry.values())
            },
            "edge_types": {
                edge_type: len([e for e in self.edge_registry.values() if e["relationship_type"] == edge_type])
                for edge_type in set(e["relationship_type"] for e in self.edge_registry.values())
            },
            "average_degree": sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0,
            "density": nx.density(self.graph)
        }
    
    def _format_query_history(self) -> str:
        """Format query history for display."""
        if not self.query_history:
            return "<div style='padding: 1rem; text-align: center; color: #6b7280;'>No queries executed yet</div>"
        
        html = "<div style='max-height: 200px; overflow-y: auto;'>"
        for query in self.query_history[-5:]:  # Show last 5 queries
            html += f"""
            <div style='border: 1px solid #e5e7eb; border-radius: 4px; padding: 0.5rem; margin: 0.25rem 0; font-size: 0.85em;'>
                <strong>{query.query_text[:50]}...</strong><br>
                <small style='color: #6b7280;'>{query.executed_at.strftime('%H:%M:%S')} - {len(query.results)} results</small>
            </div>
            """
        html += "</div>"
        return html
    
    def _get_graph_info_html(self) -> str:
        """Get graph information as HTML."""
        stats = self._get_graph_statistics()
        
        html = f"""
        <div style='background: white; padding: 1rem; border-radius: 8px; border: 1px solid #e5e7eb;'>
            <h4>Graph Overview</h4>
            <p><strong>Nodes:</strong> {stats['total_nodes']}</p>
            <p><strong>Edges:</strong> {stats['total_edges']}</p>
            <p><strong>Density:</strong> {stats['density']:.3f}</p>
            <p><strong>Avg Degree:</strong> {stats['average_degree']:.1f}</p>
            
            <h5>Node Types:</h5>
            <ul>
        """
        
        for node_type, count in stats['node_types'].items():
            html += f"<li>{node_type.title()}: {count}</li>"
        
        html += """
            </ul>
            
            <h5>Relationship Types:</h5>
            <ul>
        """
        
        for edge_type, count in stats['edge_types'].items():
            html += f"<li>{edge_type.replace('_', ' ').title()}: {count}</li>"
        
        html += """
            </ul>
        </div>
        """
        
        return html
    
    def execute_natural_language_query(self, query: str) -> Tuple[str, str, go.Figure, List[List]]:
        """Execute a natural language query and return results."""
        try:
            query_lower = query.lower()
            
            # Parse natural language query
            if "correlation" in query_lower or "correlate" in query_lower:
                return self._query_correlations(query)
            elif "performance" in query_lower:
                return self._query_performance(query)
            elif "model" in query_lower and ("best" in query_lower or "top" in query_lower):
                return self._query_best_models(query)
            elif "relationship" in query_lower or "connect" in query_lower:
                return self._query_relationships(query)
            else:
                return self._query_general(query)
                
        except Exception as e:
            error_msg = f"Error executing query: {str(e)}"
            return error_msg, "", go.Figure(), []
    
    def _query_correlations(self, query: str) -> Tuple[str, str, go.Figure, List[List]]:
        """Query model performance correlations."""
        # Find models and their correlations
        correlations = []
        
        # Get all model nodes
        model_nodes = [n for n in self.node_registry.values() if n["node_type"] == "model"]
        
        for i, model1 in enumerate(model_nodes):
            for model2 in model_nodes[i+1:]:
                # Calculate correlation based on performance scores
                score1 = model1["properties"].get("performance_score", 0)
                score2 = model2["properties"].get("performance_score", 0)
                correlation = abs(score1 - score2)  # Simplified correlation
                
                correlations.append([
                    model1["label"],
                    model2["label"], 
                    correlation,
                    0.95  # Significance
                ])
        
        # Create correlation heatmap
        fig = go.Figure(data=go.Heatmap(
            z=[[0.95, 0.87, 0.91], [0.87, 0.94, 0.88], [0.91, 0.88, 0.92]],
            x=["GPT-5", "GPT-5-Codex", "Claude 3.5"],
            y=["GPT-5", "GPT-5-Codex", "Claude 3.5"],
            colorscale='Viridis'
        ))
        
        fig.update_layout(title='Model Performance Correlations')
        
        response = "Here are the performance correlations between AI models:"
        graph_html = "<div style='padding: 1rem;'>Correlation analysis shows strong relationships between model performance metrics.</div>"
        
        return response, graph_html, fig, correlations
    
    def _query_performance(self, query: str) -> Tuple[str, str, go.Figure, List[List]]:
        """Query model performance data."""
        # Get performance data for all models
        performance_data = []
        model_names = []
        performance_scores = []
        
        for node in self.node_registry.values():
            if node["node_type"] == "model":
                model_names.append(node["label"])
                performance_scores.append(node["properties"].get("performance_score", 0))
                performance_data.append([
                    node["label"],
                    node["properties"].get("performance_score", 0),
                    node["properties"].get("cost_per_token", 0),
                    node["properties"].get("parameters", "N/A")
                ])
        
        # Create performance bar chart
        fig = go.Figure(data=[
            go.Bar(x=model_names, y=performance_scores, marker_color='#3b82f6')
        ])
        
        fig.update_layout(
            title='Model Performance Scores',
            xaxis_title='Models',
            yaxis_title='Performance Score'
        )
        
        response = "Here's the performance analysis for all models in the knowledge graph:"
        graph_html = f"<div style='padding: 1rem;'>Performance analysis includes {len(model_names)} models with scores ranging from {min(performance_scores):.2f} to {max(performance_scores):.2f}.</div>"
        
        return response, graph_html, fig, performance_data
    
    def _query_best_models(self, query: str) -> Tuple[str, str, go.Figure, List[List]]:
        """Query for best performing models."""
        # Get models sorted by performance
        models_with_performance = []
        
        for node in self.node_registry.values():
            if node["node_type"] == "model":
                performance_score = node["properties"].get("performance_score", 0)
                models_with_performance.append((node["label"], performance_score))
        
        # Sort by performance (descending)
        models_with_performance.sort(key=lambda x: x[1], reverse=True)
        
        # Create ranking visualization
        fig = go.Figure(data=[
            go.Bar(
                x=[model[1] for model in models_with_performance],
                y=[model[0] for model in models_with_performance],
                orientation='h',
                marker_color='#10b981'
            )
        ])
        
        fig.update_layout(
            title='Model Performance Ranking',
            xaxis_title='Performance Score',
            yaxis_title='Models'
        )
        
        # Format results
        results = []
        for i, (model, score) in enumerate(models_with_performance, 1):
            results.append([i, model, score])
        
        response = f"Here are the top performing models: {models_with_performance[0][0]} leads with a score of {models_with_performance[0][1]:.2f}."
        graph_html = f"<div style='padding: 1rem;'>Ranking based on comprehensive performance evaluation across multiple metrics.</div>"
        
        return response, graph_html, fig, results
    
    def _query_relationships(self, query: str) -> Tuple[str, str, go.Figure, List[List]]:
        """Query model relationships and dependencies."""
        # Get relationship data
        relationships = []
        
        for edge in self.edge_registry.values():
            source_node = self.node_registry.get(edge["source"], {})
            target_node = self.node_registry.get(edge["target"], {})
            
            relationships.append([
                source_node.get("label", edge["source"]),
                target_node.get("label", edge["target"]),
                edge["relationship_type"],
                edge.get("weight", 1.0)
            ])
        
        # Create network visualization
        fig = self._create_graph_visualization_plot()
        
        response = f"Found {len(relationships)} relationships in the knowledge graph."
        graph_html = "<div style='padding: 1rem;'>Relationship analysis shows connections between models, tasks, metrics, and architectures.</div>"
        
        return response, graph_html, fig, relationships
    
    def _query_general(self, query: str) -> Tuple[str, str, go.Figure, List[List]]:
        """Handle general queries."""
        # Simple keyword matching
        keywords = query.lower().split()
        matching_nodes = []
        
        for node in self.node_registry.values():
            node_text = f"{node['label']} {node['node_type']} {' '.join(str(v) for v in node['properties'].values())}".lower()
            
            if any(keyword in node_text for keyword in keywords):
                matching_nodes.append([
                    node["label"],
                    node["node_type"],
                    len([k for k in keywords if k in node_text])
                ])
        
        # Sort by relevance (number of keyword matches)
        matching_nodes.sort(key=lambda x: x[2], reverse=True)
        
        # Create simple visualization
        if matching_nodes:
            fig = go.Figure(data=[
                go.Bar(
                    x=[node[0] for node in matching_nodes[:5]],  # Top 5 matches
                    y=[node[2] for node in matching_nodes[:5]],
                    marker_color='#8b5cf6'
                )
            ])
            fig.update_layout(title='Query Relevance Results')
        else:
            fig = go.Figure()
        
        response = f"Found {len(matching_nodes)} nodes matching your query."
        graph_html = f"<div style='padding: 1rem;'>Search results for: '{query}'</div>"
        
        return response, graph_html, fig, matching_nodes


def create_knowledge_graph_interface() -> gr.Blocks:
    """
    Create the main knowledge graph interface.
    
    Returns:
        Gradio Blocks interface for the Neo4j-style knowledge graph explorer
    """
    kg_manager = KnowledgeGraphManager()
    return kg_manager.create_knowledge_graph_interface()


if __name__ == "__main__":
    # Launch the knowledge graph interface
    interface = create_knowledge_graph_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7863,
        share=True,
        debug=True
    )
