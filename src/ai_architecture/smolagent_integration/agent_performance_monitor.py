"""
Agent Performance Monitor

This module provides comprehensive performance monitoring for SmolAgent workflows,
including real-time metrics, alerting, and analytics.
"""

import logging
import time
import json
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for agent workflows."""
    workflow_name: str
    agent_name: str
    timestamp: datetime
    execution_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: float
    tokens_processed: int
    tokens_per_second: float
    accuracy_score: float
    error_count: int
    success_rate: float
    queue_length: int
    throughput: float


@dataclass
class AlertThreshold:
    """Alert threshold configuration."""
    metric_name: str
    threshold_value: float
    comparison_operator: str  # "gt", "lt", "eq", "gte", "lte"
    severity: str  # "info", "warning", "critical"
    enabled: bool = True


@dataclass
class Alert:
    """Alert instance."""
    alert_id: str
    workflow_name: str
    agent_name: str
    metric_name: str
    current_value: float
    threshold_value: float
    severity: str
    timestamp: datetime
    message: str
    resolved: bool = False


class AgentPerformanceMonitor:
    """
    Agent Performance Monitor for SmolAgent workflows.
    
    This class provides comprehensive performance monitoring capabilities
    including real-time metrics collection, alerting, and analytics.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Agent Performance Monitor.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "config/agent_monitoring.json"
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_thresholds: Dict[str, AlertThreshold] = {}
        self.monitoring_active: bool = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Performance tracking
        self.workflow_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.agent_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Load configuration
        self._load_config()
        
        logger.info("Agent Performance Monitor initialized")
    
    def _load_config(self):
        """Load monitoring configuration."""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    
                    # Load alert thresholds
                    for threshold_data in config.get('alert_thresholds', []):
                        threshold = AlertThreshold(**threshold_data)
                        self.alert_thresholds[threshold.metric_name] = threshold
                    
                    logger.info(f"Loaded {len(self.alert_thresholds)} alert thresholds")
        except Exception as e:
            logger.warning(f"Could not load monitoring config: {e}")
    
    def _save_config(self):
        """Save monitoring configuration."""
        try:
            config = {
                'alert_thresholds': [asdict(threshold) for threshold in self.alert_thresholds.values()]
            }
            config_file = Path(self.config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info("Monitoring configuration saved")
        except Exception as e:
            logger.error(f"Could not save monitoring config: {e}")
    
    def start_monitoring(self, interval_seconds: int = 5):
        """
        Start performance monitoring.
        
        Args:
            interval_seconds: Monitoring interval in seconds
        """
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"Performance monitoring started with {interval_seconds}s interval")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Check for alerts
                self._check_alerts()
                
                # Update statistics
                self._update_statistics()
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)
    
    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            import psutil
            
            # System metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            memory_usage_mb = memory.used / 1024 / 1024
            
            # Update system stats
            self.workflow_stats['system'] = {
                'cpu_usage_percent': cpu_percent,
                'memory_usage_mb': memory_usage_mb,
                'timestamp': datetime.now()
            }
            
        except ImportError:
            # Fallback metrics without psutil
            self.workflow_stats['system'] = {
                'cpu_usage_percent': 0,
                'memory_usage_mb': 0,
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _check_alerts(self):
        """Check for alert conditions."""
        for metric_name, threshold in self.alert_thresholds.items():
            if not threshold.enabled:
                continue
            
            # Get current metric value
            current_value = self._get_metric_value(metric_name)
            if current_value is None:
                continue
            
            # Check threshold condition
            alert_triggered = self._check_threshold_condition(
                current_value, threshold.threshold_value, threshold.comparison_operator
            )
            
            if alert_triggered:
                self._create_alert(metric_name, current_value, threshold)
    
    def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current value for a metric."""
        # System metrics
        if metric_name == "cpu_usage_percent":
            return self.workflow_stats.get('system', {}).get('cpu_usage_percent')
        elif metric_name == "memory_usage_mb":
            return self.workflow_stats.get('system', {}).get('memory_usage_mb')
        
        # Agent-specific metrics
        for agent_name, stats in self.agent_stats.items():
            if metric_name in stats:
                return stats[metric_name]
        
        return None
    
    def _check_threshold_condition(self, current_value: float, 
                                 threshold_value: float, 
                                 operator: str) -> bool:
        """Check if threshold condition is met."""
        if operator == "gt":
            return current_value > threshold_value
        elif operator == "lt":
            return current_value < threshold_value
        elif operator == "eq":
            return current_value == threshold_value
        elif operator == "gte":
            return current_value >= threshold_value
        elif operator == "lte":
            return current_value <= threshold_value
        else:
            return False
    
    def _create_alert(self, metric_name: str, current_value: float, 
                     threshold: AlertThreshold):
        """Create a new alert."""
        alert_id = f"{metric_name}_{int(time.time())}"
        
        alert = Alert(
            alert_id=alert_id,
            workflow_name="system",
            agent_name="system",
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold.threshold_value,
            severity=threshold.severity,
            timestamp=datetime.now(),
            message=f"{metric_name} is {current_value} (threshold: {threshold.threshold_value})"
        )
        
        self.active_alerts[alert_id] = alert
        
        # Trigger alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        logger.warning(f"Alert created: {alert.message}")
    
    def _update_statistics(self):
        """Update performance statistics."""
        # Update workflow statistics
        for workflow_name, stats in self.workflow_stats.items():
            if 'execution_count' not in stats:
                stats['execution_count'] = 0
            if 'total_execution_time' not in stats:
                stats['total_execution_time'] = 0
            
            # Calculate averages
            if stats['execution_count'] > 0:
                stats['average_execution_time'] = stats['total_execution_time'] / stats['execution_count']
    
    def record_metrics(self, metrics: PerformanceMetrics):
        """
        Record performance metrics for an agent.
        
        Args:
            metrics: Performance metrics to record
        """
        try:
            # Store metrics in history
            key = f"{metrics.workflow_name}_{metrics.agent_name}"
            self.metrics_history[key].append(metrics)
            
            # Update agent statistics
            agent_key = f"{metrics.workflow_name}_{metrics.agent_name}"
            if agent_key not in self.agent_stats:
                self.agent_stats[agent_key] = {
                    'execution_count': 0,
                    'total_execution_time': 0,
                    'total_memory_usage': 0,
                    'total_tokens_processed': 0,
                    'error_count': 0,
                    'success_count': 0
                }
            
            stats = self.agent_stats[agent_key]
            stats['execution_count'] += 1
            stats['total_execution_time'] += metrics.execution_time_ms
            stats['total_memory_usage'] += metrics.memory_usage_mb
            stats['total_tokens_processed'] += metrics.tokens_processed
            stats['error_count'] += metrics.error_count
            
            if metrics.success_rate > 0.8:
                stats['success_count'] += 1
            
            # Update workflow statistics
            workflow_key = metrics.workflow_name
            if workflow_key not in self.workflow_stats:
                self.workflow_stats[workflow_key] = {
                    'execution_count': 0,
                    'total_execution_time': 0,
                    'agent_count': 0
                }
            
            workflow_stats = self.workflow_stats[workflow_key]
            workflow_stats['execution_count'] += 1
            workflow_stats['total_execution_time'] += metrics.execution_time_ms
            
            logger.debug(f"Recorded metrics for {metrics.workflow_name}.{metrics.agent_name}")
            
        except Exception as e:
            logger.error(f"Error recording metrics: {e}")
    
    def get_metrics_history(self, workflow_name: str, 
                          agent_name: Optional[str] = None,
                          limit: int = 100) -> List[PerformanceMetrics]:
        """
        Get metrics history for a workflow or agent.
        
        Args:
            workflow_name: Name of the workflow
            agent_name: Name of the agent (optional)
            limit: Maximum number of records to return
            
        Returns:
            List of performance metrics
        """
        if agent_name:
            key = f"{workflow_name}_{agent_name}"
            history = list(self.metrics_history[key])
        else:
            # Get all agents for the workflow
            history = []
            for key, metrics_list in self.metrics_history.items():
                if key.startswith(f"{workflow_name}_"):
                    history.extend(list(metrics_list))
        
        # Sort by timestamp and limit
        history.sort(key=lambda x: x.timestamp, reverse=True)
        return history[:limit]
    
    def get_performance_summary(self, workflow_name: str) -> Dict[str, Any]:
        """
        Get performance summary for a workflow.
        
        Args:
            workflow_name: Name of the workflow
            
        Returns:
            Performance summary
        """
        try:
            # Get workflow statistics
            workflow_stats = self.workflow_stats.get(workflow_name, {})
            
            # Get agent statistics
            agent_stats = {}
            for key, stats in self.agent_stats.items():
                if key.startswith(f"{workflow_name}_"):
                    agent_name = key.split('_', 1)[1]
                    agent_stats[agent_name] = stats
            
            # Calculate summary metrics
            total_executions = workflow_stats.get('execution_count', 0)
            total_time = workflow_stats.get('total_execution_time', 0)
            average_time = total_time / total_executions if total_executions > 0 else 0
            
            # Calculate success rate
            total_success = sum(stats.get('success_count', 0) for stats in agent_stats.values())
            total_errors = sum(stats.get('error_count', 0) for stats in agent_stats.values())
            success_rate = total_success / (total_success + total_errors) if (total_success + total_errors) > 0 else 0
            
            summary = {
                'workflow_name': workflow_name,
                'total_executions': total_executions,
                'average_execution_time_ms': average_time,
                'success_rate': success_rate,
                'agent_count': len(agent_stats),
                'agent_performance': agent_stats,
                'active_alerts': len([a for a in self.active_alerts.values() if not a.resolved]),
                'monitoring_active': self.monitoring_active
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}
    
    def add_alert_threshold(self, threshold: AlertThreshold):
        """
        Add an alert threshold.
        
        Args:
            threshold: Alert threshold configuration
        """
        self.alert_thresholds[threshold.metric_name] = threshold
        self._save_config()
        logger.info(f"Added alert threshold for {threshold.metric_name}")
    
    def remove_alert_threshold(self, metric_name: str):
        """
        Remove an alert threshold.
        
        Args:
            metric_name: Name of the metric
        """
        if metric_name in self.alert_thresholds:
            del self.alert_thresholds[metric_name]
            self._save_config()
            logger.info(f"Removed alert threshold for {metric_name}")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """
        Add an alert callback function.
        
        Args:
            callback: Function to call when alerts are triggered
        """
        self.alert_callbacks.append(callback)
        logger.info("Added alert callback")
    
    def get_active_alerts(self, severity: Optional[str] = None) -> List[Alert]:
        """
        Get active alerts.
        
        Args:
            severity: Filter by severity (optional)
            
        Returns:
            List of active alerts
        """
        alerts = [alert for alert in self.active_alerts.values() if not alert.resolved]
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return alerts
    
    def resolve_alert(self, alert_id: str):
        """
        Resolve an alert.
        
        Args:
            alert_id: ID of the alert to resolve
        """
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolved = True
            logger.info(f"Resolved alert {alert_id}")
    
    def export_metrics(self, workflow_name: str, 
                      export_path: str,
                      format: str = "json") -> bool:
        """
        Export metrics to file.
        
        Args:
            workflow_name: Name of the workflow
            export_path: Path to export file
            format: Export format (json, csv)
            
        Returns:
            True if exported successfully
        """
        try:
            # Get metrics history
            metrics_history = self.get_metrics_history(workflow_name)
            
            if format == "json":
                # Export as JSON
                export_data = {
                    'workflow_name': workflow_name,
                    'export_timestamp': datetime.now().isoformat(),
                    'metrics_count': len(metrics_history),
                    'metrics': [asdict(metric) for metric in metrics_history]
                }
                
                with open(export_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            
            elif format == "csv":
                # Export as CSV
                import csv
                with open(export_path, 'w', newline='') as f:
                    if metrics_history:
                        writer = csv.DictWriter(f, fieldnames=asdict(metrics_history[0]).keys())
                        writer.writeheader()
                        for metric in metrics_history:
                            writer.writerow(asdict(metric))
            
            logger.info(f"Exported metrics for {workflow_name} to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get overall health status of the monitoring system.
        
        Returns:
            Health status information
        """
        try:
            # Count active alerts by severity
            alert_counts = defaultdict(int)
            for alert in self.active_alerts.values():
                if not alert.resolved:
                    alert_counts[alert.severity] += 1
            
            # Calculate system health score
            total_alerts = sum(alert_counts.values())
            critical_alerts = alert_counts.get('critical', 0)
            warning_alerts = alert_counts.get('warning', 0)
            
            if critical_alerts > 0:
                health_score = 0
                health_status = "critical"
            elif warning_alerts > 3:
                health_score = 50
                health_status = "warning"
            elif total_alerts > 5:
                health_score = 75
                health_status = "degraded"
            else:
                health_score = 100
                health_status = "healthy"
            
            return {
                'health_score': health_score,
                'health_status': health_status,
                'monitoring_active': self.monitoring_active,
                'total_workflows': len(self.workflow_stats),
                'total_agents': len(self.agent_stats),
                'active_alerts': total_alerts,
                'alert_breakdown': dict(alert_counts),
                'monitoring_uptime': self._get_monitoring_uptime()
            }
            
        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {'health_status': 'error', 'error': str(e)}
    
    def _get_monitoring_uptime(self) -> str:
        """Get monitoring uptime."""
        # This would track actual uptime in a real implementation
        return "24h 15m 30s"


# Example usage and factory functions
def create_default_monitor() -> AgentPerformanceMonitor:
    """Create a default performance monitor with standard thresholds."""
    monitor = AgentPerformanceMonitor()
    
    # Add default alert thresholds
    default_thresholds = [
        AlertThreshold("cpu_usage_percent", 80.0, "gte", "warning"),
        AlertThreshold("cpu_usage_percent", 95.0, "gte", "critical"),
        AlertThreshold("memory_usage_mb", 1024, "gte", "warning"),
        AlertThreshold("memory_usage_mb", 2048, "gte", "critical"),
        AlertThreshold("execution_time_ms", 5000, "gte", "warning"),
        AlertThreshold("execution_time_ms", 10000, "gte", "critical"),
        AlertThreshold("error_count", 5, "gte", "warning"),
        AlertThreshold("error_count", 10, "gte", "critical")
    ]
    
    for threshold in default_thresholds:
        monitor.add_alert_threshold(threshold)
    
    return monitor


def create_mobile_monitor() -> AgentPerformanceMonitor:
    """Create a mobile-optimized performance monitor."""
    monitor = AgentPerformanceMonitor()
    
    # Mobile-specific thresholds
    mobile_thresholds = [
        AlertThreshold("memory_usage_mb", 256, "gte", "warning"),
        AlertThreshold("memory_usage_mb", 512, "gte", "critical"),
        AlertThreshold("execution_time_ms", 2000, "gte", "warning"),
        AlertThreshold("execution_time_ms", 5000, "gte", "critical"),
        AlertThreshold("cpu_usage_percent", 70, "gte", "warning"),
        AlertThreshold("cpu_usage_percent", 90, "gte", "critical")
    ]
    
    for threshold in mobile_thresholds:
        monitor.add_alert_threshold(threshold)
    
    return monitor


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create performance monitor
    monitor = create_default_monitor()
    
    # Start monitoring
    monitor.start_monitoring(interval_seconds=5)
    
    # Simulate some metrics
    for i in range(10):
        metrics = PerformanceMetrics(
            workflow_name="test_workflow",
            agent_name="test_agent",
            timestamp=datetime.now(),
            execution_time_ms=100 + i * 10,
            memory_usage_mb=100 + i * 5,
            cpu_usage_percent=20 + i * 2,
            gpu_usage_percent=0,
            tokens_processed=100 + i * 10,
            tokens_per_second=10.0,
            accuracy_score=0.95,
            error_count=0,
            success_rate=0.95,
            queue_length=0,
            throughput=10.0
        )
        monitor.record_metrics(metrics)
        time.sleep(1)
    
    # Get performance summary
    summary = monitor.get_performance_summary("test_workflow")
    print(f"Performance summary: {summary}")
    
    # Get health status
    health = monitor.get_health_status()
    print(f"Health status: {health}")
    
    # Stop monitoring
    monitor.stop_monitoring()
