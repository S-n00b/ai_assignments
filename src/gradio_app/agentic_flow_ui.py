"""
LangChain Studio-Style Agentic Flow UI for Lenovo AAITC Solutions

This module implements a modern, interactive agent workflow visualization and builder
inspired by LangChain Studio, featuring drag-and-drop workflow creation, real-time
agent communication monitoring, and collaborative workflow editing capabilities.

Key Features:
- Drag-and-drop workflow builder
- Real-time agent communication visualization
- Interactive workflow debugging and monitoring
- Agent performance metrics dashboard
- Workflow template library
- Collaborative workflow editing
"""

import gradio as gr
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from dataclasses import dataclass, asdict
import uuid


@dataclass
class AgentNode:
    """Represents an agent node in the workflow."""
    id: str
    name: str
    type: str  # "llm", "tool", "memory", "decision"
    position: Tuple[float, float]
    config: Dict[str, Any]
    status: str = "idle"  # "idle", "running", "completed", "error"
    metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


@dataclass
class WorkflowEdge:
    """Represents a connection between agents in the workflow."""
    id: str
    source: str
    target: str
    label: str = ""
    type: str = "data"  # "data", "control", "feedback"
    status: str = "inactive"  # "inactive", "active", "completed"


@dataclass
class WorkflowTemplate:
    """Represents a workflow template."""
    id: str
    name: str
    description: str
    category: str
    nodes: List[AgentNode]
    edges: List[WorkflowEdge]
    tags: List[str]
    created_by: str
    created_at: datetime


class AgenticFlowBuilder:
    """
    Main class for the LangChain Studio-style agentic flow builder.
    
    This class provides the core functionality for creating, editing, and monitoring
    agent workflows with an intuitive drag-and-drop interface.
    """
    
    def __init__(self):
        """Initialize the agentic flow builder."""
        self.current_workflow = {
            "nodes": [],
            "edges": [],
            "metadata": {
                "name": "Untitled Workflow",
                "description": "",
                "created_at": datetime.now().isoformat(),
                "modified_at": datetime.now().isoformat()
            }
        }
        
        self.workflow_templates = self._initialize_templates()
        self.agent_types = self._initialize_agent_types()
        self.real_time_metrics = {}
        self.workflow_history = []
        
    def _initialize_templates(self) -> List[WorkflowTemplate]:
        """Initialize predefined workflow templates."""
        templates = []
        
        # Model Evaluation Workflow
        eval_nodes = [
            AgentNode(
                id="input_processor",
                name="Input Processor",
                type="tool",
                position=(100, 100),
                config={"function": "process_inputs", "timeout": 30}
            ),
            AgentNode(
                id="model_evaluator",
                name="Model Evaluator",
                type="llm",
                position=(300, 100),
                config={"model": "gpt-5", "temperature": 0.1}
            ),
            AgentNode(
                id="quality_analyzer",
                name="Quality Analyzer",
                type="tool",
                position=(500, 100),
                config={"function": "analyze_quality", "metrics": ["rouge", "bert"]}
            ),
            AgentNode(
                id="report_generator",
                name="Report Generator",
                type="llm",
                position=(700, 100),
                config={"model": "claude-3.5-sonnet", "format": "html"}
            )
        ]
        
        eval_edges = [
            WorkflowEdge("e1", "input_processor", "model_evaluator", "processed_data"),
            WorkflowEdge("e2", "model_evaluator", "quality_analyzer", "evaluation_results"),
            WorkflowEdge("e3", "quality_analyzer", "report_generator", "analysis_data")
        ]
        
        templates.append(WorkflowTemplate(
            id="model_evaluation",
            name="Model Evaluation Pipeline",
            description="Complete model evaluation workflow with quality analysis and reporting",
            category="evaluation",
            nodes=eval_nodes,
            edges=eval_edges,
            tags=["evaluation", "quality", "reporting"],
            created_by="system",
            created_at=datetime.now()
        ))
        
        # RAG System Workflow
        rag_nodes = [
            AgentNode(
                id="document_ingestor",
                name="Document Ingestor",
                type="tool",
                position=(100, 200),
                config={"function": "ingest_documents", "formats": ["pdf", "txt", "md"]}
            ),
            AgentNode(
                id="embedding_generator",
                name="Embedding Generator",
                type="llm",
                position=(300, 200),
                config={"model": "text-embedding-3-large", "chunk_size": 512}
            ),
            AgentNode(
                id="vector_store",
                name="Vector Store",
                type="memory",
                position=(500, 200),
                config={"store_type": "chroma", "collection": "documents"}
            ),
            AgentNode(
                id="retrieval_agent",
                name="Retrieval Agent",
                type="llm",
                position=(700, 200),
                config={"model": "gpt-5", "retrieval_count": 5}
            ),
            AgentNode(
                id="response_generator",
                name="Response Generator",
                type="llm",
                position=(900, 200),
                config={"model": "claude-3.5-sonnet", "temperature": 0.7}
            )
        ]
        
        rag_edges = [
            WorkflowEdge("r1", "document_ingestor", "embedding_generator", "documents"),
            WorkflowEdge("r2", "embedding_generator", "vector_store", "embeddings"),
            WorkflowEdge("r3", "vector_store", "retrieval_agent", "similar_documents"),
            WorkflowEdge("r4", "retrieval_agent", "response_generator", "context")
        ]
        
        templates.append(WorkflowTemplate(
            id="rag_system",
            name="RAG System Pipeline",
            description="Complete RAG system with document ingestion, embedding, and response generation",
            category="rag",
            nodes=rag_nodes,
            edges=rag_edges,
            tags=["rag", "retrieval", "embedding", "generation"],
            created_by="system",
            created_at=datetime.now()
        ))
        
        # Multi-Agent Collaboration Workflow
        collab_nodes = [
            AgentNode(
                id="coordinator",
                name="Workflow Coordinator",
                type="llm",
                position=(200, 300),
                config={"model": "gpt-5", "role": "coordinator"}
            ),
            AgentNode(
                id="researcher",
                name="Research Agent",
                type="llm",
                position=(400, 200),
                config={"model": "claude-3.5-sonnet", "specialty": "research"}
            ),
            AgentNode(
                id="analyst",
                name="Analysis Agent",
                type="llm",
                position=(400, 400),
                config={"model": "gpt-5", "specialty": "analysis"}
            ),
            AgentNode(
                id="synthesizer",
                name="Synthesis Agent",
                type="llm",
                position=(600, 300),
                config={"model": "claude-3.5-sonnet", "role": "synthesizer"}
            )
        ]
        
        collab_edges = [
            WorkflowEdge("c1", "coordinator", "researcher", "research_task"),
            WorkflowEdge("c2", "coordinator", "analyst", "analysis_task"),
            WorkflowEdge("c3", "researcher", "synthesizer", "research_results"),
            WorkflowEdge("c4", "analyst", "synthesizer", "analysis_results")
        ]
        
        templates.append(WorkflowTemplate(
            id="multi_agent_collab",
            name="Multi-Agent Collaboration",
            description="Collaborative workflow with specialized agents working together",
            category="collaboration",
            nodes=collab_nodes,
            edges=collab_edges,
            tags=["collaboration", "multi-agent", "specialization"],
            created_by="system",
            created_at=datetime.now()
        ))
        
        return templates
    
    def _initialize_agent_types(self) -> Dict[str, Dict[str, Any]]:
        """Initialize available agent types and their configurations."""
        return {
            "llm": {
                "name": "LLM Agent",
                "icon": "🤖",
                "description": "Large Language Model agent for text processing and generation",
                "config_schema": {
                    "model": {"type": "string", "required": True},
                    "temperature": {"type": "number", "default": 0.7},
                    "max_tokens": {"type": "number", "default": 2048},
                    "system_prompt": {"type": "string", "default": ""}
                }
            },
            "tool": {
                "name": "Tool Agent",
                "icon": "🔧",
                "description": "Tool-based agent for executing specific functions",
                "config_schema": {
                    "function": {"type": "string", "required": True},
                    "timeout": {"type": "number", "default": 30},
                    "retries": {"type": "number", "default": 3}
                }
            },
            "memory": {
                "name": "Memory Agent",
                "icon": "🧠",
                "description": "Memory-based agent for storing and retrieving information",
                "config_schema": {
                    "store_type": {"type": "string", "default": "vector"},
                    "capacity": {"type": "number", "default": 1000},
                    "persistence": {"type": "boolean", "default": True}
                }
            },
            "decision": {
                "name": "Decision Agent",
                "icon": "🎯",
                "description": "Decision-making agent for routing and control flow",
                "config_schema": {
                    "decision_type": {"type": "string", "required": True},
                    "criteria": {"type": "array", "default": []},
                    "default_path": {"type": "string", "default": ""}
                }
            }
        }
    
    def create_workflow_builder_interface(self) -> gr.Blocks:
        """Create the main workflow builder interface."""
        with gr.Blocks(
            title="Agentic Flow Builder - LangChain Studio Style",
            theme=gr.themes.Soft(),
            css="""
            .workflow-canvas {
                background: #f8fafc;
                border: 2px dashed #e2e8f0;
                border-radius: 8px;
                min-height: 600px;
                position: relative;
            }
            .agent-node {
                background: white;
                border: 2px solid #3b82f6;
                border-radius: 8px;
                padding: 8px;
                cursor: move;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                position: absolute;
                min-width: 120px;
                text-align: center;
            }
            .agent-node.running {
                border-color: #10b981;
                background: #ecfdf5;
            }
            .agent-node.error {
                border-color: #ef4444;
                background: #fef2f2;
            }
            .agent-node.completed {
                border-color: #8b5cf6;
                background: #f3e8ff;
            }
            .workflow-template {
                border: 1px solid #e5e7eb;
                border-radius: 6px;
                padding: 12px;
                margin: 8px 0;
                cursor: pointer;
                transition: all 0.2s;
            }
            .workflow-template:hover {
                border-color: #3b82f6;
                background: #f8fafc;
            }
            .metrics-dashboard {
                background: white;
                border-radius: 8px;
                padding: 16px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            """
        ) as interface:
            
            # Header
            gr.HTML("""
            <div style="background: linear-gradient(90deg, #1e3a8a, #3b82f6); color: white; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                <h1>🚀 Agentic Flow Builder</h1>
                <p>LangChain Studio-Style Workflow Designer for Lenovo AAITC Solutions</p>
            </div>
            """)
            
            with gr.Row():
                # Left Panel - Agent Palette and Templates
                with gr.Column(scale=1):
                    self._create_agent_palette()
                    self._create_template_library()
                
                # Center Panel - Workflow Canvas
                with gr.Column(scale=3):
                    self._create_workflow_canvas()
                
                # Right Panel - Properties and Monitoring
                with gr.Column(scale=1):
                    self._create_properties_panel()
                    self._create_monitoring_dashboard()
        
        return interface
    
    def _create_agent_palette(self):
        """Create the agent palette for drag-and-drop."""
        gr.Markdown("## 🎨 Agent Palette")
        
        with gr.Accordion("Available Agents", open=True):
            for agent_type, agent_info in self.agent_types.items():
                with gr.Row():
                    gr.HTML(f"""
                    <div class="agent-palette-item" data-agent-type="{agent_type}">
                        <div style="display: flex; align-items: center; padding: 8px; border: 1px solid #e5e7eb; border-radius: 4px; margin: 4px 0; cursor: grab;">
                            <span style="font-size: 1.5em; margin-right: 8px;">{agent_info['icon']}</span>
                            <div>
                                <strong>{agent_info['name']}</strong><br>
                                <small style="color: #6b7280;">{agent_info['description']}</small>
                            </div>
                        </div>
                    </div>
                    """)
        
        # Add Agent Button
        add_agent_btn = gr.Button("➕ Add Agent to Canvas", variant="primary", size="sm")
        
        # Agent Configuration
        with gr.Accordion("Agent Configuration", open=False):
            self.agent_name_input = gr.Textbox(label="Agent Name", placeholder="Enter agent name")
            self.agent_type_dropdown = gr.Dropdown(
                choices=list(self.agent_types.keys()),
                label="Agent Type"
            )
            self.agent_config_json = gr.JSON(label="Configuration", value={})
    
    def _create_template_library(self):
        """Create the workflow template library."""
        gr.Markdown("## 📚 Template Library")
        
        template_selector = gr.Dropdown(
            choices=[f"{t.name} - {t.category}" for t in self.workflow_templates],
            label="Select Template",
            allow_custom_value=False
        )
        
        load_template_btn = gr.Button("📥 Load Template", variant="secondary", size="sm")
        
        # Template Preview
        template_preview = gr.HTML(
            value="<div style='padding: 1rem; text-align: center; color: #6b7280;'>Select a template to preview</div>"
        )
        
        # Template Actions
        with gr.Row():
            save_template_btn = gr.Button("💾 Save as Template", variant="secondary", size="sm")
            export_workflow_btn = gr.Button("📤 Export Workflow", variant="secondary", size="sm")
    
    def _create_workflow_canvas(self):
        """Create the main workflow canvas."""
        gr.Markdown("## 🎯 Workflow Canvas")
        
        # Canvas Controls
        with gr.Row():
            run_workflow_btn = gr.Button("▶️ Run Workflow", variant="primary")
            pause_workflow_btn = gr.Button("⏸️ Pause", variant="secondary")
            stop_workflow_btn = gr.Button("⏹️ Stop", variant="stop")
            clear_canvas_btn = gr.Button("🗑️ Clear", variant="secondary")
        
        # Workflow Canvas
        self.workflow_canvas = gr.HTML(
            value=self._render_workflow_canvas(),
            elem_classes=["workflow-canvas"]
        )
        
        # Workflow Status
        self.workflow_status = gr.Textbox(
            value="Ready to build workflow",
            label="Status",
            interactive=False
        )
        
        # Execution Log
        with gr.Accordion("Execution Log", open=False):
            self.execution_log = gr.Textbox(
                value="",
                label="Log",
                lines=10,
                interactive=False,
                max_lines=20
            )
    
    def _create_properties_panel(self):
        """Create the properties panel for selected agents."""
        gr.Markdown("## ⚙️ Properties")
        
        # Selected Agent Info
        self.selected_agent_info = gr.HTML(
            value="<div style='padding: 1rem; text-align: center; color: #6b7280;'>Select an agent to view properties</div>"
        )
        
        # Agent Properties Form
        with gr.Accordion("Agent Configuration", open=True):
            self.property_name = gr.Textbox(label="Name")
            self.property_type = gr.Dropdown(
                choices=list(self.agent_types.keys()),
                label="Type",
                interactive=False
            )
            self.property_config = gr.JSON(label="Configuration")
            
            update_agent_btn = gr.Button("💾 Update Agent", variant="primary", size="sm")
        
        # Connection Properties
        with gr.Accordion("Connections", open=False):
            self.connection_source = gr.Dropdown(label="Source Agent")
            self.connection_target = gr.Dropdown(label="Target Agent")
            self.connection_label = gr.Textbox(label="Connection Label")
            
            create_connection_btn = gr.Button("🔗 Create Connection", variant="secondary", size="sm")
    
    def _create_monitoring_dashboard(self):
        """Create the real-time monitoring dashboard."""
        gr.Markdown("## 📊 Real-Time Monitoring")
        
        # Performance Metrics
        with gr.Row():
            self.avg_latency = gr.Number(label="Avg Latency (ms)", value=0, precision=1)
            self.success_rate = gr.Number(label="Success Rate (%)", value=0, precision=1)
        
        # Agent Status Overview
        self.agent_status_chart = gr.Plot(label="Agent Status")
        
        # Communication Flow
        self.communication_flow = gr.Plot(label="Communication Flow")
        
        # Metrics History
        self.metrics_history = gr.Plot(label="Metrics History")
        
        # Refresh Metrics Button
        refresh_metrics_btn = gr.Button("🔄 Refresh Metrics", variant="secondary", size="sm")
    
    def _render_workflow_canvas(self) -> str:
        """Render the workflow canvas with current nodes and edges."""
        canvas_html = """
        <div class="workflow-canvas" id="workflow-canvas">
            <svg width="100%" height="600" style="position: absolute; top: 0; left: 0; z-index: 1;">
                <!-- Edges will be rendered here -->
        """
        
        # Render edges
        for edge in self.current_workflow["edges"]:
            # Find source and target positions
            source_node = next((n for n in self.current_workflow["nodes"] if n["id"] == edge["source"]), None)
            target_node = next((n for n in self.current_workflow["nodes"] if n["id"] == edge["target"]), None)
            
            if source_node and target_node:
                x1, y1 = source_node["position"]
                x2, y2 = target_node["position"]
                
                canvas_html += f"""
                <line x1="{x1 + 60}" y1="{y1 + 30}" x2="{x2 + 60}" y2="{y2 + 30}" 
                      stroke="#3b82f6" stroke-width="2" marker-end="url(#arrowhead)" />
                <text x="{(x1 + x2) / 2 + 60}" y="{(y1 + y2) / 2 + 25}" 
                      text-anchor="middle" font-size="12" fill="#6b7280">{edge["label"]}</text>
                """
        
        canvas_html += """
            </svg>
            
            <!-- Arrow marker definition -->
            <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                        refX="10" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#3b82f6" />
                </marker>
            </defs>
        """
        
        # Render nodes
        for node in self.current_workflow["nodes"]:
            x, y = node["position"]
            agent_type_info = self.agent_types.get(node["type"], {})
            icon = agent_type_info.get("icon", "🤖")
            
            canvas_html += f"""
            <div class="agent-node {node.get('status', 'idle')}" 
                 style="left: {x}px; top: {y}px; z-index: 2;"
                 data-node-id="{node['id']}">
                <div style="font-size: 1.2em; margin-bottom: 4px;">{icon}</div>
                <div style="font-weight: bold; margin-bottom: 2px;">{node['name']}</div>
                <div style="font-size: 0.8em; color: #6b7280;">{node['type']}</div>
            </div>
            """
        
        canvas_html += "</div>"
        
        return canvas_html
    
    def add_agent_to_workflow(
        self,
        agent_name: str,
        agent_type: str,
        config: Dict[str, Any]
    ) -> Tuple[str, str]:
        """Add a new agent to the workflow."""
        try:
            # Create new agent node
            agent_id = f"agent_{len(self.current_workflow['nodes']) + 1}"
            new_node = {
                "id": agent_id,
                "name": agent_name or f"{agent_type}_agent",
                "type": agent_type,
                "position": (100 + len(self.current_workflow['nodes']) * 150, 100),
                "config": config,
                "status": "idle",
                "metrics": {}
            }
            
            self.current_workflow["nodes"].append(new_node)
            self.current_workflow["metadata"]["modified_at"] = datetime.now().isoformat()
            
            # Update canvas
            canvas_html = self._render_workflow_canvas()
            status = f"Added {agent_name} ({agent_type}) to workflow"
            
            return canvas_html, status
            
        except Exception as e:
            return self._render_workflow_canvas(), f"Error adding agent: {str(e)}"
    
    def load_workflow_template(self, template_name: str) -> Tuple[str, str, str]:
        """Load a workflow template."""
        try:
            # Find template by name
            template = None
            for t in self.workflow_templates:
                if t.name in template_name:
                    template = t
                    break
            
            if not template:
                return (
                    self._render_workflow_canvas(),
                    "<div style='color: red;'>Template not found</div>",
                    "Error: Template not found"
                )
            
            # Convert template to workflow format
            self.current_workflow = {
                "nodes": [asdict(node) for node in template.nodes],
                "edges": [asdict(edge) for edge in template.edges],
                "metadata": {
                    "name": template.name,
                    "description": template.description,
                    "created_at": datetime.now().isoformat(),
                    "modified_at": datetime.now().isoformat()
                }
            }
            
            # Generate preview
            preview_html = f"""
            <div style="padding: 1rem;">
                <h3>{template.name}</h3>
                <p>{template.description}</p>
                <p><strong>Category:</strong> {template.category}</p>
                <p><strong>Nodes:</strong> {len(template.nodes)}</p>
                <p><strong>Connections:</strong> {len(template.edges)}</p>
                <p><strong>Tags:</strong> {', '.join(template.tags)}</p>
            </div>
            """
            
            canvas_html = self._render_workflow_canvas()
            status = f"Loaded template: {template.name}"
            
            return canvas_html, preview_html, status
            
        except Exception as e:
            return (
                self._render_workflow_canvas(),
                "<div style='color: red;'>Error loading template</div>",
                f"Error: {str(e)}"
            )
    
    def run_workflow(self) -> Tuple[str, str, go.Figure, go.Figure, go.Figure]:
        """Run the current workflow and update monitoring."""
        try:
            # Update workflow status
            self.current_workflow["metadata"]["status"] = "running"
            self.current_workflow["metadata"]["started_at"] = datetime.now().isoformat()
            
            # Simulate workflow execution
            execution_log = "Starting workflow execution...\n"
            
            for i, node in enumerate(self.current_workflow["nodes"]):
                # Update node status
                node["status"] = "running"
                
                # Simulate processing time
                import time
                time.sleep(0.5)
                
                # Update metrics
                node["metrics"] = {
                    "latency": 100 + i * 50,
                    "success_rate": 95 - i * 2,
                    "throughput": 10 - i * 0.5
                }
                
                execution_log += f"Executing {node['name']}... ✅\n"
                node["status"] = "completed"
            
            # Update workflow status
            self.current_workflow["metadata"]["status"] = "completed"
            self.current_workflow["metadata"]["completed_at"] = datetime.now().isoformat()
            
            # Generate monitoring charts
            status_chart = self._create_agent_status_chart()
            flow_chart = self._create_communication_flow_chart()
            metrics_chart = self._create_metrics_history_chart()
            
            # Update canvas
            canvas_html = self._render_workflow_canvas()
            status = "Workflow execution completed successfully"
            
            return canvas_html, execution_log, status_chart, flow_chart, metrics_chart
            
        except Exception as e:
            return (
                self._render_workflow_canvas(),
                f"Error during execution: {str(e)}",
                "Error: Workflow execution failed",
                go.Figure(),
                go.Figure(),
                go.Figure()
            )
    
    def _create_agent_status_chart(self) -> go.Figure:
        """Create agent status overview chart."""
        if not self.current_workflow["nodes"]:
            return go.Figure()
        
        status_counts = {}
        for node in self.current_workflow["nodes"]:
            status = node.get("status", "idle")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        fig = go.Figure(data=[go.Pie(
            labels=list(status_counts.keys()),
            values=list(status_counts.values()),
            hole=0.3
        )])
        
        fig.update_layout(
            title="Agent Status Overview",
            height=300
        )
        
        return fig
    
    def _create_communication_flow_chart(self) -> go.Figure:
        """Create communication flow visualization."""
        if not self.current_workflow["edges"]:
            return go.Figure()
        
        # Create network graph
        G = nx.DiGraph()
        
        # Add nodes
        for node in self.current_workflow["nodes"]:
            G.add_node(node["id"], name=node["name"], type=node["type"])
        
        # Add edges
        for edge in self.current_workflow["edges"]:
            G.add_edge(edge["source"], edge["target"], label=edge["label"])
        
        # Generate layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Create edge trace
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(G.nodes[node]['name'])
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=50,
                color='lightblue',
                line=dict(width=2, color='blue')
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Communication Flow',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=400
                       ))
        
        return fig
    
    def _create_metrics_history_chart(self) -> go.Figure:
        """Create metrics history chart."""
        # Generate sample metrics over time
        timestamps = pd.date_range(start=datetime.now() - timedelta(hours=1), 
                                 end=datetime.now(), freq='5T')
        
        fig = go.Figure()
        
        # Add latency trend
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=[100 + i * 5 + (i % 10) * 10 for i in range(len(timestamps))],
            mode='lines',
            name='Latency (ms)',
            line=dict(color='blue')
        ))
        
        # Add throughput trend
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=[8 + i * 0.1 + (i % 5) * 0.5 for i in range(len(timestamps))],
            mode='lines',
            name='Throughput (QPS)',
            yaxis='y2',
            line=dict(color='green')
        ))
        
        fig.update_layout(
            title='Performance Metrics History',
            xaxis_title='Time',
            yaxis=dict(title='Latency (ms)', side='left'),
            yaxis2=dict(title='Throughput (QPS)', side='right', overlaying='y'),
            height=300
        )
        
        return fig


def create_agentic_flow_interface() -> gr.Blocks:
    """
    Create the main agentic flow interface.
    
    Returns:
        Gradio Blocks interface for the agentic flow builder
    """
    builder = AgenticFlowBuilder()
    return builder.create_workflow_builder_interface()


if __name__ == "__main__":
    # Launch the agentic flow builder
    interface = create_agentic_flow_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
        debug=True
    )
