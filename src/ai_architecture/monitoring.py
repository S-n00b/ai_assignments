"""
Comprehensive Monitoring and Observability Module for Lenovo AAITC Solutions

This module provides enterprise-grade monitoring capabilities including:
- Grafana dashboard configuration and management
- Prometheus metrics collection and alerting
- AI-specific monitoring (model performance, drift, bias)
- Infrastructure monitoring (Kubernetes, containers, resources)
- Business metrics and KPI tracking
- Real-time alerting and notification systems

Designed for production AI systems with comprehensive observability.
"""

import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics for monitoring"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class DashboardCategory(Enum):
    """Dashboard categories"""
    AI_MODELS = "ai_models"
    INFRASTRUCTURE = "infrastructure"
    BUSINESS_METRICS = "business_metrics"
    SECURITY = "security"
    PERFORMANCE = "performance"

@dataclass
class MetricConfig:
    """Configuration for a Prometheus metric"""
    name: str
    description: str
    metric_type: MetricType
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histogram
    objectives: Optional[Dict[float, float]] = None  # For summary

@dataclass
class AlertRule:
    """Configuration for an alert rule"""
    name: str
    expression: str
    severity: AlertSeverity
    description: str
    runbook_url: Optional[str] = None
    notification_channels: List[str] = field(default_factory=list)

@dataclass
class DashboardPanel:
    """Configuration for a Grafana dashboard panel"""
    title: str
    type: str  # graph, stat, table, etc.
    targets: List[Dict[str, Any]]
    y_axes: Optional[Dict[str, Any]] = None
    thresholds: Optional[List[Dict[str, Any]]] = None
    grid_pos: Optional[Dict[str, int]] = None

class PrometheusMetricsCollector:
    """
    Collects and manages Prometheus metrics for AI systems.
    Provides custom metrics for model performance, agent collaboration, and infrastructure.
    """
    
    def __init__(self, metrics_port: int = 8000):
        self.metrics_port = metrics_port
        self.metrics_registry = {}
        self.custom_metrics = {}
        logger.info(f"PrometheusMetricsCollector initialized on port {metrics_port}")
    
    def register_metric(self, config: MetricConfig) -> bool:
        """Register a new metric configuration."""
        try:
            self.metrics_registry[config.name] = config
            logger.info(f"Metric registered: {config.name}")
            return True
        except Exception as e:
            logger.error(f"Error registering metric {config.name}: {e}")
            return False
    
    def create_ai_model_metrics(self) -> Dict[str, MetricConfig]:
        """Create AI model-specific metrics."""
        metrics = {
            "model_inference_duration": MetricConfig(
                name="model_inference_duration_seconds",
                description="Time taken for model inference",
                metric_type=MetricType.HISTOGRAM,
                labels=["model_name", "model_version", "task_type"],
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
            ),
            "model_accuracy": MetricConfig(
                name="model_accuracy_score",
                description="Model accuracy score",
                metric_type=MetricType.GAUGE,
                labels=["model_name", "model_version", "dataset"]
            ),
            "model_throughput": MetricConfig(
                name="model_requests_per_second",
                description="Model requests per second",
                metric_type=MetricType.GAUGE,
                labels=["model_name", "model_version"]
            ),
            "model_error_rate": MetricConfig(
                name="model_error_rate",
                description="Model error rate percentage",
                metric_type=MetricType.GAUGE,
                labels=["model_name", "model_version", "error_type"]
            ),
            "model_drift_score": MetricConfig(
                name="model_drift_score",
                description="Model drift detection score",
                metric_type=MetricType.GAUGE,
                labels=["model_name", "model_version", "drift_type"]
            ),
            "model_bias_score": MetricConfig(
                name="model_bias_score",
                description="Model bias detection score",
                metric_type=MetricType.GAUGE,
                labels=["model_name", "model_version", "bias_type"]
            )
        }
        
        for metric in metrics.values():
            self.register_metric(metric)
        
        return metrics
    
    def create_agent_metrics(self) -> Dict[str, MetricConfig]:
        """Create agent system-specific metrics."""
        metrics = {
            "agent_task_duration": MetricConfig(
                name="agent_task_duration_seconds",
                description="Time taken for agent task completion",
                metric_type=MetricType.HISTOGRAM,
                labels=["agent_id", "task_type", "crew_id"],
                buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0]
            ),
            "agent_collaboration_count": MetricConfig(
                name="agent_collaboration_total",
                description="Total number of agent collaborations",
                metric_type=MetricType.COUNTER,
                labels=["agent_id", "collaboration_type"]
            ),
            "agent_queue_length": MetricConfig(
                name="agent_queue_length",
                description="Number of tasks in agent queue",
                metric_type=MetricType.GAUGE,
                labels=["agent_id"]
            ),
            "agent_success_rate": MetricConfig(
                name="agent_success_rate",
                description="Agent task success rate",
                metric_type=MetricType.GAUGE,
                labels=["agent_id", "task_type"]
            ),
            "crew_workflow_duration": MetricConfig(
                name="crew_workflow_duration_seconds",
                description="Time taken for crew workflow completion",
                metric_type=MetricType.HISTOGRAM,
                labels=["crew_id", "workflow_type"],
                buckets=[10.0, 30.0, 60.0, 300.0, 600.0, 1800.0, 3600.0]
            )
        }
        
        for metric in metrics.values():
            self.register_metric(metric)
        
        return metrics
    
    def create_infrastructure_metrics(self) -> Dict[str, MetricConfig]:
        """Create infrastructure-specific metrics."""
        metrics = {
            "kubernetes_pod_cpu_usage": MetricConfig(
                name="kubernetes_pod_cpu_usage_percent",
                description="Kubernetes pod CPU usage percentage",
                metric_type=MetricType.GAUGE,
                labels=["namespace", "pod", "container"]
            ),
            "kubernetes_pod_memory_usage": MetricConfig(
                name="kubernetes_pod_memory_usage_bytes",
                description="Kubernetes pod memory usage in bytes",
                metric_type=MetricType.GAUGE,
                labels=["namespace", "pod", "container"]
            ),
            "kubernetes_deployment_replicas": MetricConfig(
                name="kubernetes_deployment_replicas",
                description="Kubernetes deployment replica count",
                metric_type=MetricType.GAUGE,
                labels=["namespace", "deployment"]
            ),
            "terraform_deployment_duration": MetricConfig(
                name="terraform_deployment_duration_seconds",
                description="Terraform deployment duration",
                metric_type=MetricType.HISTOGRAM,
                labels=["environment", "resource_type"],
                buckets=[60.0, 300.0, 600.0, 1800.0, 3600.0, 7200.0]
            ),
            "helm_chart_deployment_status": MetricConfig(
                name="helm_chart_deployment_status",
                description="Helm chart deployment status (1=success, 0=failed)",
                metric_type=MetricType.GAUGE,
                labels=["namespace", "release", "chart"]
            )
        }
        
        for metric in metrics.values():
            self.register_metric(metric)
        
        return metrics
    
    def create_business_metrics(self) -> Dict[str, MetricConfig]:
        """Create business-specific metrics."""
        metrics = {
            "user_satisfaction_score": MetricConfig(
                name="user_satisfaction_score",
                description="User satisfaction score",
                metric_type=MetricType.GAUGE,
                labels=["service", "user_segment"]
            ),
            "api_response_time": MetricConfig(
                name="api_response_time_seconds",
                description="API response time",
                metric_type=MetricType.HISTOGRAM,
                labels=["endpoint", "method", "status_code"],
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            ),
            "cost_per_inference": MetricConfig(
                name="cost_per_inference_usd",
                description="Cost per model inference in USD",
                metric_type=MetricType.GAUGE,
                labels=["model_name", "provider"]
            ),
            "model_usage_count": MetricConfig(
                name="model_usage_total",
                description="Total model usage count",
                metric_type=MetricType.COUNTER,
                labels=["model_name", "user_type"]
            )
        }
        
        for metric in metrics.values():
            self.register_metric(metric)
        
        return metrics
    
    def generate_prometheus_config(self) -> str:
        """Generate Prometheus configuration for metrics collection."""
        config = {
            "global": {
                "scrape_interval": "15s",
                "evaluation_interval": "15s"
            },
            "rule_files": [
                "alert_rules.yml"
            ],
            "scrape_configs": [
                {
                    "job_name": "lenovo-aaitc-metrics",
                    "static_configs": [
                        {
                            "targets": [f"localhost:{self.metrics_port}"]
                        }
                    ],
                    "scrape_interval": "5s",
                    "metrics_path": "/metrics"
                },
                {
                    "job_name": "kubernetes-pods",
                    "kubernetes_sd_configs": [
                        {
                            "role": "pod"
                        }
                    ],
                    "relabel_configs": [
                        {
                            "source_labels": ["__meta_kubernetes_pod_annotation_prometheus_io_scrape"],
                            "action": "keep",
                            "regex": "true"
                        }
                    ]
                }
            ],
            "alerting": {
                "alertmanagers": [
                    {
                        "static_configs": [
                            {
                                "targets": ["alertmanager:9093"]
                            }
                        ]
                    }
                ]
            }
        }
        
        return yaml.dump(config, default_flow_style=False)

class GrafanaDashboardManager:
    """
    Manages Grafana dashboards for comprehensive AI system monitoring.
    Creates and configures dashboards for different monitoring categories.
    """
    
    def __init__(self, grafana_url: str = "http://localhost:3000", api_key: str = None):
        self.grafana_url = grafana_url
        self.api_key = api_key
        self.dashboards = {}
        logger.info(f"GrafanaDashboardManager initialized: {grafana_url}")
    
    def create_ai_models_dashboard(self) -> Dict[str, Any]:
        """Create comprehensive AI models monitoring dashboard."""
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "Lenovo AAITC - AI Models Monitoring",
                "tags": ["ai", "models", "lenovo", "aaitc"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "Model Inference Duration",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.95, rate(model_inference_duration_seconds_bucket[5m]))",
                                "legendFormat": "95th percentile"
                            },
                            {
                                "expr": "histogram_quantile(0.50, rate(model_inference_duration_seconds_bucket[5m]))",
                                "legendFormat": "50th percentile"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Duration (seconds)",
                                "min": 0
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "Model Accuracy Scores",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "model_accuracy_score",
                                "legendFormat": "{{model_name}} v{{model_version}}"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Accuracy Score",
                                "min": 0,
                                "max": 1
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    },
                    {
                        "id": 3,
                        "title": "Model Throughput",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(model_requests_per_second[5m])",
                                "legendFormat": "{{model_name}}"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Requests/Second",
                                "min": 0
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
                    },
                    {
                        "id": 4,
                        "title": "Model Error Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "model_error_rate",
                                "legendFormat": "{{model_name}} - {{error_type}}"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Error Rate (%)",
                                "min": 0,
                                "max": 100
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
                    },
                    {
                        "id": 5,
                        "title": "Model Drift Detection",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "model_drift_score",
                                "legendFormat": "{{model_name}} - {{drift_type}}"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Drift Score",
                                "min": 0,
                                "max": 1
                            }
                        ],
                        "thresholds": [
                            {
                                "value": 0.1,
                                "colorMode": "critical",
                                "op": "gt"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16}
                    },
                    {
                        "id": 6,
                        "title": "Model Bias Detection",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "model_bias_score",
                                "legendFormat": "{{model_name}} - {{bias_type}}"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Bias Score",
                                "min": 0,
                                "max": 1
                            }
                        ],
                        "thresholds": [
                            {
                                "value": 0.2,
                                "colorMode": "critical",
                                "op": "gt"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16}
                    }
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "5s"
            }
        }
        
        self.dashboards["ai_models"] = dashboard
        return dashboard
    
    def create_agent_system_dashboard(self) -> Dict[str, Any]:
        """Create agent system monitoring dashboard."""
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "Lenovo AAITC - Agent System Monitoring",
                "tags": ["agents", "crewai", "lenovo", "aaitc"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "Agent Task Duration",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.95, rate(agent_task_duration_seconds_bucket[5m]))",
                                "legendFormat": "95th percentile - {{agent_id}}"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Duration (seconds)",
                                "min": 0
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "Agent Collaboration Count",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(agent_collaboration_total[5m])",
                                "legendFormat": "{{agent_id}} - {{collaboration_type}}"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Collaborations/Second",
                                "min": 0
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    },
                    {
                        "id": 3,
                        "title": "Agent Queue Length",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "agent_queue_length",
                                "legendFormat": "{{agent_id}}"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Queue Length",
                                "min": 0
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
                    },
                    {
                        "id": 4,
                        "title": "Agent Success Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "agent_success_rate",
                                "legendFormat": "{{agent_id}} - {{task_type}}"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Success Rate (%)",
                                "min": 0,
                                "max": 100
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
                    },
                    {
                        "id": 5,
                        "title": "Crew Workflow Duration",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.95, rate(crew_workflow_duration_seconds_bucket[5m]))",
                                "legendFormat": "95th percentile - {{crew_id}}"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Duration (seconds)",
                                "min": 0
                            }
                        ],
                        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 16}
                    }
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "5s"
            }
        }
        
        self.dashboards["agent_system"] = dashboard
        return dashboard
    
    def create_infrastructure_dashboard(self) -> Dict[str, Any]:
        """Create infrastructure monitoring dashboard."""
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "Lenovo AAITC - Infrastructure Monitoring",
                "tags": ["infrastructure", "kubernetes", "terraform", "lenovo", "aaitc"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "Kubernetes Pod CPU Usage",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "kubernetes_pod_cpu_usage_percent",
                                "legendFormat": "{{namespace}}/{{pod}}"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "CPU Usage (%)",
                                "min": 0,
                                "max": 100
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "Kubernetes Pod Memory Usage",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "kubernetes_pod_memory_usage_bytes / 1024 / 1024",
                                "legendFormat": "{{namespace}}/{{pod}} (MB)"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Memory Usage (MB)",
                                "min": 0
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    },
                    {
                        "id": 3,
                        "title": "Kubernetes Deployment Replicas",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "kubernetes_deployment_replicas",
                                "legendFormat": "{{namespace}}/{{deployment}}"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Replica Count",
                                "min": 0
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
                    },
                    {
                        "id": 4,
                        "title": "Terraform Deployment Duration",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.95, rate(terraform_deployment_duration_seconds_bucket[5m]))",
                                "legendFormat": "95th percentile - {{environment}}"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Duration (seconds)",
                                "min": 0
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
                    },
                    {
                        "id": 5,
                        "title": "Helm Chart Deployment Status",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "helm_chart_deployment_status",
                                "legendFormat": "{{namespace}}/{{release}}"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 16}
                    }
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "5s"
            }
        }
        
        self.dashboards["infrastructure"] = dashboard
        return dashboard
    
    def create_business_metrics_dashboard(self) -> Dict[str, Any]:
        """Create business metrics monitoring dashboard."""
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "Lenovo AAITC - Business Metrics",
                "tags": ["business", "kpi", "lenovo", "aaitc"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "User Satisfaction Score",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "user_satisfaction_score",
                                "legendFormat": "{{service}} - {{user_segment}}"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Satisfaction Score",
                                "min": 0,
                                "max": 5
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "API Response Time",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.95, rate(api_response_time_seconds_bucket[5m]))",
                                "legendFormat": "95th percentile - {{endpoint}}"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Response Time (seconds)",
                                "min": 0
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    },
                    {
                        "id": 3,
                        "title": "Cost per Inference",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "cost_per_inference_usd",
                                "legendFormat": "{{model_name}} - {{provider}}"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Cost (USD)",
                                "min": 0
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
                    },
                    {
                        "id": 4,
                        "title": "Model Usage Count",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(model_usage_total[5m])",
                                "legendFormat": "{{model_name}} - {{user_type}}"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Usage/Second",
                                "min": 0
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
                    }
                ],
                "time": {
                    "from": "now-24h",
                    "to": "now"
                },
                "refresh": "30s"
            }
        }
        
        self.dashboards["business_metrics"] = dashboard
        return dashboard
    
    def create_executive_dashboard(self) -> Dict[str, Any]:
        """Create executive summary dashboard."""
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "Lenovo AAITC - Executive Dashboard",
                "tags": ["executive", "summary", "lenovo", "aaitc"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "System Health Overview",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "up",
                                "legendFormat": "System Status"
                            }
                        ],
                        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "Active Models",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "count(model_accuracy_score)",
                                "legendFormat": "Active Models"
                            }
                        ],
                        "gridPos": {"h": 4, "w": 6, "x": 6, "y": 0}
                    },
                    {
                        "id": 3,
                        "title": "Total Requests Today",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "sum(increase(model_usage_total[24h]))",
                                "legendFormat": "Requests"
                            }
                        ],
                        "gridPos": {"h": 4, "w": 6, "x": 12, "y": 0}
                    },
                    {
                        "id": 4,
                        "title": "Average Response Time",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "avg(histogram_quantile(0.50, rate(model_inference_duration_seconds_bucket[5m])))",
                                "legendFormat": "Response Time"
                            }
                        ],
                        "gridPos": {"h": 4, "w": 6, "x": 18, "y": 0}
                    },
                    {
                        "id": 5,
                        "title": "Model Performance Trend",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "avg(model_accuracy_score)",
                                "legendFormat": "Average Accuracy"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Accuracy Score",
                                "min": 0,
                                "max": 1
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 4}
                    },
                    {
                        "id": 6,
                        "title": "Cost Analysis",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "sum(rate(cost_per_inference_usd[1h]))",
                                "legendFormat": "Hourly Cost"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Cost (USD/hour)",
                                "min": 0
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 4}
                    }
                ],
                "time": {
                    "from": "now-7d",
                    "to": "now"
                },
                "refresh": "1m"
            }
        }
        
        self.dashboards["executive"] = dashboard
        return dashboard
    
    def generate_all_dashboards(self) -> Dict[str, Any]:
        """Generate all monitoring dashboards."""
        dashboards = {
            "ai_models": self.create_ai_models_dashboard(),
            "agent_system": self.create_agent_system_dashboard(),
            "infrastructure": self.create_infrastructure_dashboard(),
            "business_metrics": self.create_business_metrics_dashboard(),
            "executive": self.create_executive_dashboard()
        }
        
        logger.info(f"Generated {len(dashboards)} monitoring dashboards")
        return dashboards

class AlertManager:
    """
    Manages alerting rules and notifications for AI system monitoring.
    Provides comprehensive alerting for model performance, infrastructure, and business metrics.
    """
    
    def __init__(self):
        self.alert_rules = {}
        self.notification_channels = {}
        logger.info("AlertManager initialized")
    
    def create_ai_model_alerts(self) -> List[AlertRule]:
        """Create alert rules for AI model monitoring."""
        alerts = [
            AlertRule(
                name="HighModelErrorRate",
                expression="model_error_rate > 5",
                severity=AlertSeverity.WARNING,
                description="Model error rate is above 5%",
                runbook_url="https://docs.lenovo.com/aaitc/model-error-troubleshooting",
                notification_channels=["slack", "email"]
            ),
            AlertRule(
                name="ModelDriftDetected",
                expression="model_drift_score > 0.1",
                severity=AlertSeverity.CRITICAL,
                description="Model drift detected - accuracy may be degrading",
                runbook_url="https://docs.lenovo.com/aaitc/model-drift-mitigation",
                notification_channels=["slack", "email", "pagerduty"]
            ),
            AlertRule(
                name="ModelBiasDetected",
                expression="model_bias_score > 0.2",
                severity=AlertSeverity.CRITICAL,
                description="Model bias detected - fairness concerns",
                runbook_url="https://docs.lenovo.com/aaitc/model-bias-mitigation",
                notification_channels=["slack", "email", "pagerduty"]
            ),
            AlertRule(
                name="ModelInferenceSlow",
                expression="histogram_quantile(0.95, rate(model_inference_duration_seconds_bucket[5m])) > 10",
                severity=AlertSeverity.WARNING,
                description="Model inference is slower than expected",
                runbook_url="https://docs.lenovo.com/aaitc/performance-optimization",
                notification_channels=["slack"]
            ),
            AlertRule(
                name="ModelAccuracyDegraded",
                expression="model_accuracy_score < 0.8",
                severity=AlertSeverity.WARNING,
                description="Model accuracy has degraded below threshold",
                runbook_url="https://docs.lenovo.com/aaitc/model-retraining",
                notification_channels=["slack", "email"]
            )
        ]
        
        for alert in alerts:
            self.alert_rules[alert.name] = alert
        
        return alerts
    
    def create_infrastructure_alerts(self) -> List[AlertRule]:
        """Create alert rules for infrastructure monitoring."""
        alerts = [
            AlertRule(
                name="HighCPUUsage",
                expression="kubernetes_pod_cpu_usage_percent > 80",
                severity=AlertSeverity.WARNING,
                description="Kubernetes pod CPU usage is above 80%",
                runbook_url="https://docs.lenovo.com/aaitc/kubernetes-scaling",
                notification_channels=["slack"]
            ),
            AlertRule(
                name="HighMemoryUsage",
                expression="kubernetes_pod_memory_usage_bytes / kubernetes_pod_memory_limit_bytes > 0.8",
                severity=AlertSeverity.WARNING,
                description="Kubernetes pod memory usage is above 80%",
                runbook_url="https://docs.lenovo.com/aaitc/memory-optimization",
                notification_channels=["slack"]
            ),
            AlertRule(
                name="PodCrashLooping",
                expression="rate(kube_pod_container_status_restarts_total[5m]) > 0",
                severity=AlertSeverity.CRITICAL,
                description="Pod is crash looping",
                runbook_url="https://docs.lenovo.com/aaitc/pod-troubleshooting",
                notification_channels=["slack", "email", "pagerduty"]
            ),
            AlertRule(
                name="TerraformDeploymentFailed",
                expression="terraform_deployment_duration_seconds == 0",
                severity=AlertSeverity.CRITICAL,
                description="Terraform deployment has failed",
                runbook_url="https://docs.lenovo.com/aaitc/terraform-troubleshooting",
                notification_channels=["slack", "email", "pagerduty"]
            ),
            AlertRule(
                name="HelmChartDeploymentFailed",
                expression="helm_chart_deployment_status == 0",
                severity=AlertSeverity.CRITICAL,
                description="Helm chart deployment has failed",
                runbook_url="https://docs.lenovo.com/aaitc/helm-troubleshooting",
                notification_channels=["slack", "email", "pagerduty"]
            )
        ]
        
        for alert in alerts:
            self.alert_rules[alert.name] = alert
        
        return alerts
    
    def create_business_alerts(self) -> List[AlertRule]:
        """Create alert rules for business metrics monitoring."""
        alerts = [
            AlertRule(
                name="LowUserSatisfaction",
                expression="user_satisfaction_score < 3.0",
                severity=AlertSeverity.WARNING,
                description="User satisfaction score is below 3.0",
                runbook_url="https://docs.lenovo.com/aaitc/user-experience-improvement",
                notification_channels=["slack", "email"]
            ),
            AlertRule(
                name="HighAPILatency",
                expression="histogram_quantile(0.95, rate(api_response_time_seconds_bucket[5m])) > 2",
                severity=AlertSeverity.WARNING,
                description="API response time is above 2 seconds",
                runbook_url="https://docs.lenovo.com/aaitc/api-optimization",
                notification_channels=["slack"]
            ),
            AlertRule(
                name="HighCostPerInference",
                expression="cost_per_inference_usd > 0.01",
                severity=AlertSeverity.WARNING,
                description="Cost per inference is above $0.01",
                runbook_url="https://docs.lenovo.com/aaitc/cost-optimization",
                notification_channels=["slack", "email"]
            ),
            AlertRule(
                name="LowModelUsage",
                expression="rate(model_usage_total[1h]) < 10",
                severity=AlertSeverity.INFO,
                description="Model usage is below 10 requests per hour",
                runbook_url="https://docs.lenovo.com/aaitc/model-promotion",
                notification_channels=["slack"]
            )
        ]
        
        for alert in alerts:
            self.alert_rules[alert.name] = alert
        
        return alerts
    
    def generate_alert_rules_config(self) -> str:
        """Generate Prometheus alert rules configuration."""
        groups = [
            {
                "name": "ai_models",
                "rules": [
                    {
                        "alert": alert.name,
                        "expr": alert.expression,
                        "for": "5m",
                        "labels": {
                            "severity": alert.severity.value,
                            "service": "ai_models"
                        },
                        "annotations": {
                            "summary": alert.description,
                            "runbook_url": alert.runbook_url or ""
                        }
                    }
                    for alert in self.create_ai_model_alerts()
                ]
            },
            {
                "name": "infrastructure",
                "rules": [
                    {
                        "alert": alert.name,
                        "expr": alert.expression,
                        "for": "5m",
                        "labels": {
                            "severity": alert.severity.value,
                            "service": "infrastructure"
                        },
                        "annotations": {
                            "summary": alert.description,
                            "runbook_url": alert.runbook_url or ""
                        }
                    }
                    for alert in self.create_infrastructure_alerts()
                ]
            },
            {
                "name": "business_metrics",
                "rules": [
                    {
                        "alert": alert.name,
                        "expr": alert.expression,
                        "for": "5m",
                        "labels": {
                            "severity": alert.severity.value,
                            "service": "business"
                        },
                        "annotations": {
                            "summary": alert.description,
                            "runbook_url": alert.runbook_url or ""
                        }
                    }
                    for alert in self.create_business_alerts()
                ]
            }
        ]
        
        return yaml.dump({"groups": groups}, default_flow_style=False)

class MonitoringOrchestrator:
    """
    Main orchestrator for comprehensive monitoring and observability.
    Coordinates Prometheus metrics, Grafana dashboards, and alerting.
    """
    
    def __init__(self, grafana_url: str = "http://localhost:3000", 
                 prometheus_port: int = 8000):
        self.metrics_collector = PrometheusMetricsCollector(prometheus_port)
        self.dashboard_manager = GrafanaDashboardManager(grafana_url)
        self.alert_manager = AlertManager()
        logger.info("MonitoringOrchestrator initialized")
    
    async def setup_comprehensive_monitoring(self) -> bool:
        """Setup comprehensive monitoring system."""
        try:
            logger.info("Setting up comprehensive monitoring system")
            
            # 1. Register all metrics
            logger.info("Step 1: Registering metrics")
            self.metrics_collector.create_ai_model_metrics()
            self.metrics_collector.create_agent_metrics()
            self.metrics_collector.create_infrastructure_metrics()
            self.metrics_collector.create_business_metrics()
            
            # 2. Create all dashboards
            logger.info("Step 2: Creating Grafana dashboards")
            self.dashboard_manager.generate_all_dashboards()
            
            # 3. Setup alerting rules
            logger.info("Step 3: Setting up alerting rules")
            self.alert_manager.create_ai_model_alerts()
            self.alert_manager.create_infrastructure_alerts()
            self.alert_manager.create_business_alerts()
            
            logger.info("Comprehensive monitoring system setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up monitoring system: {e}")
            return False
    
    def export_configurations(self, output_dir: str = "./monitoring_configs") -> bool:
        """Export all monitoring configurations to files."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Export Prometheus configuration
            prometheus_config = self.metrics_collector.generate_prometheus_config()
            (output_path / "prometheus.yml").write_text(prometheus_config)
            
            # Export alert rules
            alert_rules = self.alert_manager.generate_alert_rules_config()
            (output_path / "alert_rules.yml").write_text(alert_rules)
            
            # Export Grafana dashboards
            dashboards = self.dashboard_manager.generate_all_dashboards()
            for name, dashboard in dashboards.items():
                dashboard_file = output_path / f"grafana_dashboard_{name}.json"
                dashboard_file.write_text(json.dumps(dashboard, indent=2))
            
            logger.info(f"Monitoring configurations exported to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting configurations: {e}")
            return False
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring system summary."""
        return {
            "metrics_registered": len(self.metrics_collector.metrics_registry),
            "dashboards_created": len(self.dashboard_manager.dashboards),
            "alert_rules_configured": len(self.alert_manager.alert_rules),
            "categories": {
                "ai_models": {
                    "metrics": 6,
                    "dashboards": 1,
                    "alerts": 5
                },
                "agent_system": {
                    "metrics": 5,
                    "dashboards": 1,
                    "alerts": 0
                },
                "infrastructure": {
                    "metrics": 5,
                    "dashboards": 1,
                    "alerts": 5
                },
                "business_metrics": {
                    "metrics": 4,
                    "dashboards": 1,
                    "alerts": 4
                },
                "executive": {
                    "metrics": 0,
                    "dashboards": 1,
                    "alerts": 0
                }
            },
            "last_updated": datetime.now().isoformat()
        }

# Example usage and testing
async def main():
    """Example usage of the monitoring module."""
    
    # Initialize monitoring orchestrator
    orchestrator = MonitoringOrchestrator()
    
    # Setup comprehensive monitoring
    success = await orchestrator.setup_comprehensive_monitoring()
    if success:
        print("✅ Monitoring system setup successful")
        
        # Export configurations
        export_success = orchestrator.export_configurations()
        if export_success:
            print("✅ Monitoring configurations exported")
        
        # Get summary
        summary = orchestrator.get_monitoring_summary()
        print(f"📊 Monitoring Summary: {summary['metrics_registered']} metrics, {summary['dashboards_created']} dashboards, {summary['alert_rules_configured']} alerts")
        
    else:
        print("❌ Monitoring system setup failed")

if __name__ == "__main__":
    asyncio.run(main())
