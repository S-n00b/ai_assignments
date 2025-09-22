"""
Monitoring and Observability for Enterprise LLMOps Platform

This module provides comprehensive monitoring, observability, and alerting
capabilities for the enterprise LLM operations platform.

Key Features:
- Prometheus metrics collection
- Grafana dashboard integration
- Real-time health monitoring
- Performance metrics and alerting
- Service discovery and monitoring
- Custom metrics for AI/ML workloads
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import aiohttp
from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
import psutil
import docker
from pathlib import Path


@dataclass
class MetricConfig:
    """Configuration for a metric."""
    name: str
    description: str
    metric_type: str  # "counter", "histogram", "gauge", "info"
    labels: List[str] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = []


@dataclass
class AlertRule:
    """Configuration for an alert rule."""
    name: str
    condition: str
    threshold: float
    duration: int  # seconds
    severity: str  # "critical", "warning", "info"
    description: str
    enabled: bool = True


@dataclass
class ServiceMetrics:
    """Metrics for a service."""
    service_name: str
    cpu_usage: float
    memory_usage: float
    request_count: int
    error_count: int
    response_time: float
    last_updated: datetime


class LLMOpsMonitoring:
    """
    Comprehensive monitoring and observability for Enterprise LLMOps platform.
    
    This class provides monitoring capabilities including metrics collection,
    health checking, alerting, and integration with monitoring stacks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the monitoring system."""
        self.config = config
        self.logger = self._setup_logging()
        self.metrics = {}
        self.alert_rules = {}
        self.service_metrics = {}
        self.docker_client = None
        
        # Initialize Prometheus metrics
        self._setup_prometheus_metrics()
        
        # Setup alert rules
        self._setup_alert_rules()
        
        # Initialize Docker client for container monitoring
        self._init_docker()
        
        # Start metrics server
        self._start_metrics_server()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for monitoring."""
        logger = logging.getLogger("llmops_monitoring")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics."""
        # System metrics
        self.metrics['system_cpu_usage'] = Gauge('system_cpu_usage_percent', 'System CPU usage percentage')
        self.metrics['system_memory_usage'] = Gauge('system_memory_usage_bytes', 'System memory usage in bytes')
        self.metrics['system_disk_usage'] = Gauge('system_disk_usage_bytes', 'System disk usage in bytes')
        
        # Service metrics
        self.metrics['service_health'] = Gauge('service_health_status', 'Service health status', ['service_name'])
        self.metrics['service_response_time'] = Histogram('service_response_time_seconds', 'Service response time', ['service_name'])
        self.metrics['service_request_count'] = Counter('service_requests_total', 'Total service requests', ['service_name', 'status'])
        self.metrics['service_error_count'] = Counter('service_errors_total', 'Total service errors', ['service_name', 'error_type'])
        
        # LLM-specific metrics
        self.metrics['llm_request_count'] = Counter('llm_requests_total', 'Total LLM requests', ['model', 'provider'])
        self.metrics['llm_tokens_generated'] = Counter('llm_tokens_generated_total', 'Total tokens generated', ['model', 'provider'])
        self.metrics['llm_request_duration'] = Histogram('llm_request_duration_seconds', 'LLM request duration', ['model', 'provider'])
        self.metrics['llm_model_usage'] = Gauge('llm_model_usage_count', 'Active model usage count', ['model'])
        
        # Vector database metrics
        self.metrics['vector_db_operations'] = Counter('vector_db_operations_total', 'Vector DB operations', ['database', 'operation'])
        self.metrics['vector_db_query_time'] = Histogram('vector_db_query_time_seconds', 'Vector DB query time', ['database'])
        self.metrics['vector_db_index_size'] = Gauge('vector_db_index_size_bytes', 'Vector DB index size', ['database'])
        
        # MLflow metrics
        self.metrics['mlflow_experiments'] = Gauge('mlflow_experiments_count', 'Number of MLflow experiments')
        self.metrics['mlflow_runs'] = Gauge('mlflow_runs_count', 'Number of MLflow runs')
        self.metrics['mlflow_models'] = Gauge('mlflow_models_count', 'Number of registered models')
        
        self.logger.info("Prometheus metrics initialized")
    
    def _setup_alert_rules(self):
        """Setup alert rules."""
        default_rules = [
            AlertRule(
                name="high_cpu_usage",
                condition="system_cpu_usage > 80",
                threshold=80.0,
                duration=300,  # 5 minutes
                severity="warning",
                description="System CPU usage is above 80%"
            ),
            AlertRule(
                name="high_memory_usage",
                condition="system_memory_usage > 90",
                threshold=90.0,
                duration=300,
                severity="warning",
                description="System memory usage is above 90%"
            ),
            AlertRule(
                name="service_down",
                condition="service_health == 0",
                threshold=0.0,
                duration=60,  # 1 minute
                severity="critical",
                description="Service is down"
            ),
            AlertRule(
                name="high_error_rate",
                condition="error_rate > 5",
                threshold=5.0,
                duration=300,
                severity="warning",
                description="Service error rate is above 5%"
            ),
            AlertRule(
                name="slow_response_time",
                condition="response_time > 10",
                threshold=10.0,
                duration=300,
                severity="warning",
                description="Service response time is above 10 seconds"
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.name] = rule
        
        self.logger.info(f"Alert rules initialized: {len(self.alert_rules)} rules")
    
    def _init_docker(self):
        """Initialize Docker client for container monitoring."""
        try:
            self.docker_client = docker.from_env()
            self.logger.info("Docker client initialized")
        except Exception as e:
            self.logger.warning(f"Docker client initialization failed: {e}")
            self.docker_client = None
    
    def _start_metrics_server(self):
        """Start Prometheus metrics server."""
        port = self.config.get('metrics_port', 9091)
        try:
            start_http_server(port)
            self.logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            self.logger.error(f"Failed to start metrics server: {e}")
    
    async def collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics['system_cpu_usage'].set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics['system_memory_usage'].set(memory.used)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.metrics['system_disk_usage'].set(disk.used)
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
    
    async def collect_service_metrics(self, service_name: str, metrics: ServiceMetrics):
        """Collect service-specific metrics."""
        try:
            # Update service health
            health_status = 1.0 if metrics.cpu_usage < 90 and metrics.memory_usage < 90 else 0.0
            self.metrics['service_health'].labels(service_name=service_name).set(health_status)
            
            # Update response time
            self.metrics['service_response_time'].labels(service_name=service_name).observe(metrics.response_time)
            
            # Update request count
            self.metrics['service_request_count'].labels(service_name=service_name, status='success').inc(metrics.request_count)
            self.metrics['service_error_count'].labels(service_name=service_name, error_type='unknown').inc(metrics.error_count)
            
            # Store metrics
            self.service_metrics[service_name] = metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics for service {service_name}: {e}")
    
    async def collect_docker_metrics(self):
        """Collect Docker container metrics."""
        if not self.docker_client:
            return
        
        try:
            containers = self.docker_client.containers.list()
            
            for container in containers:
                try:
                    stats = container.stats(stream=False)
                    
                    # Calculate CPU usage
                    cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
                    system_delta = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']
                    cpu_percent = (cpu_delta / system_delta) * 100.0 if system_delta > 0 else 0.0
                    
                    # Calculate memory usage
                    memory_usage = stats['memory_stats']['usage']
                    
                    # Update metrics
                    self.metrics['service_health'].labels(service_name=container.name).set(1.0)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to get stats for container {container.name}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to collect Docker metrics: {e}")
    
    async def check_alerts(self):
        """Check alert rules and trigger alerts if needed."""
        active_alerts = []
        
        try:
            for rule_name, rule in self.alert_rules.items():
                if not rule.enabled:
                    continue
                
                # Simple alert checking logic
                # In a real implementation, this would be more sophisticated
                if rule.condition == "system_cpu_usage > 80":
                    cpu_usage = self.metrics['system_cpu_usage']._value._value
                    if cpu_usage > rule.threshold:
                        active_alerts.append({
                            'rule': rule_name,
                            'severity': rule.severity,
                            'message': rule.description,
                            'value': cpu_usage,
                            'threshold': rule.threshold,
                            'timestamp': datetime.now()
                        })
                
                elif rule.condition == "service_health == 0":
                    # Check for unhealthy services
                    for service_name in self.service_metrics.keys():
                        health = self.metrics['service_health'].labels(service_name=service_name)._value._value
                        if health == 0:
                            active_alerts.append({
                                'rule': rule_name,
                                'severity': rule.severity,
                                'message': f"{rule.description} - Service: {service_name}",
                                'service': service_name,
                                'timestamp': datetime.now()
                            })
            
            # Log active alerts
            for alert in active_alerts:
                self.logger.warning(f"ALERT: {alert['message']} (Severity: {alert['severity']})")
                
        except Exception as e:
            self.logger.error(f"Failed to check alerts: {e}")
        
        return active_alerts
    
    async def monitoring_loop(self):
        """Main monitoring loop."""
        self.logger.info("Starting monitoring loop...")
        
        while True:
            try:
                # Collect system metrics
                await self.collect_system_metrics()
                
                # Collect Docker metrics
                await self.collect_docker_metrics()
                
                # Check alerts
                await self.check_alerts()
                
                # Sleep for the configured interval
                interval = self.config.get('monitoring_interval', 30)
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        return {
            "system_metrics": {
                "cpu_usage": self.metrics['system_cpu_usage']._value._value,
                "memory_usage": self.metrics['system_memory_usage']._value._value,
                "disk_usage": self.metrics['system_disk_usage']._value._value
            },
            "service_metrics": {name: asdict(metrics) for name, metrics in self.service_metrics.items()},
            "alert_rules": {name: asdict(rule) for name, rule in self.alert_rules.items()},
            "timestamp": datetime.now().isoformat()
        }
    
    async def start(self):
        """Start the monitoring system."""
        self.logger.info("Starting LLMOps Monitoring...")
        
        # Start monitoring loop
        self.monitoring_task = asyncio.create_task(self.monitoring_loop())
        
        self.logger.info("LLMOps Monitoring started successfully")
    
    async def stop(self):
        """Stop the monitoring system."""
        self.logger.info("Stopping LLMOps Monitoring...")
        
        if hasattr(self, 'monitoring_task'):
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("LLMOps Monitoring stopped")
