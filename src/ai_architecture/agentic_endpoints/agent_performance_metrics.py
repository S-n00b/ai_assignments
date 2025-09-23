"""
Agent Performance Metrics

This module provides performance metrics collection and analysis
for agentic workflows including SmolAgent and LangGraph.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric data."""
    metric_name: str
    value: float
    timestamp: datetime
    workflow_id: str
    agent_id: str
    metric_type: str  # latency, throughput, accuracy, memory, cpu
    unit: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PerformanceSummary:
    """Performance summary for an agent or workflow."""
    workflow_id: str
    agent_id: str
    metric_type: str
    count: int
    min_value: float
    max_value: float
    mean_value: float
    median_value: float
    std_dev: float
    p95_value: float
    p99_value: float
    time_range: Dict[str, datetime]
    trend: str  # increasing, decreasing, stable


class AgentPerformanceMetrics:
    """
    Agent Performance Metrics collector and analyzer.
    
    This class provides comprehensive performance metrics collection
    and analysis for agentic workflows.
    """
    
    def __init__(self, max_metrics_per_agent: int = 1000):
        """
        Initialize the Agent Performance Metrics.
        
        Args:
            max_metrics_per_agent: Maximum number of metrics to store per agent
        """
        self.max_metrics_per_agent = max_metrics_per_agent
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_metrics_per_agent))
        self.agent_summaries: Dict[str, Dict[str, PerformanceSummary]] = {}
        self.workflow_summaries: Dict[str, Dict[str, PerformanceSummary]] = {}
        
        logger.info("Agent Performance Metrics initialized")
    
    def record_metric(self, metric: PerformanceMetric):
        """
        Record a performance metric.
        
        Args:
            metric: Performance metric to record
        """
        try:
            # Create metric key
            metric_key = f"{metric.workflow_id}_{metric.agent_id}_{metric.metric_type}"
            
            # Store metric
            self.metrics[metric_key].append(metric)
            
            # Update summaries
            self._update_agent_summary(metric)
            self._update_workflow_summary(metric)
            
            logger.debug(f"Recorded metric: {metric.metric_name} = {metric.value} {metric.unit}")
            
        except Exception as e:
            logger.error(f"Failed to record metric: {e}")
    
    def _update_agent_summary(self, metric: PerformanceMetric):
        """Update agent performance summary."""
        try:
            agent_key = f"{metric.workflow_id}_{metric.agent_id}"
            metric_type = metric.metric_type
            
            if agent_key not in self.agent_summaries:
                self.agent_summaries[agent_key] = {}
            
            # Get existing metrics for this agent and metric type
            metric_key = f"{metric.workflow_id}_{metric.agent_id}_{metric_type}"
            agent_metrics = list(self.metrics[metric_key])
            
            if not agent_metrics:
                return
            
            # Calculate summary statistics
            values = [m.value for m in agent_metrics]
            timestamps = [m.timestamp for m in agent_metrics]
            
            summary = PerformanceSummary(
                workflow_id=metric.workflow_id,
                agent_id=metric.agent_id,
                metric_type=metric_type,
                count=len(values),
                min_value=min(values),
                max_value=max(values),
                mean_value=statistics.mean(values),
                median_value=statistics.median(values),
                std_dev=statistics.stdev(values) if len(values) > 1 else 0,
                p95_value=self._calculate_percentile(values, 95),
                p99_value=self._calculate_percentile(values, 99),
                time_range={
                    "start": min(timestamps),
                    "end": max(timestamps)
                },
                trend=self._calculate_trend(values)
            )
            
            self.agent_summaries[agent_key][metric_type] = summary
            
        except Exception as e:
            logger.error(f"Failed to update agent summary: {e}")
    
    def _update_workflow_summary(self, metric: PerformanceMetric):
        """Update workflow performance summary."""
        try:
            workflow_id = metric.workflow_id
            metric_type = metric.metric_type
            
            if workflow_id not in self.workflow_summaries:
                self.workflow_summaries[workflow_id] = {}
            
            # Get all metrics for this workflow and metric type
            workflow_metrics = []
            for key, metrics_deque in self.metrics.items():
                if key.startswith(f"{workflow_id}_") and key.endswith(f"_{metric_type}"):
                    workflow_metrics.extend(list(metrics_deque))
            
            if not workflow_metrics:
                return
            
            # Calculate summary statistics
            values = [m.value for m in workflow_metrics]
            timestamps = [m.timestamp for m in workflow_metrics]
            
            summary = PerformanceSummary(
                workflow_id=workflow_id,
                agent_id="workflow",
                metric_type=metric_type,
                count=len(values),
                min_value=min(values),
                max_value=max(values),
                mean_value=statistics.mean(values),
                median_value=statistics.median(values),
                std_dev=statistics.stdev(values) if len(values) > 1 else 0,
                p95_value=self._calculate_percentile(values, 95),
                p99_value=self._calculate_percentile(values, 99),
                time_range={
                    "start": min(timestamps),
                    "end": max(timestamps)
                },
                trend=self._calculate_trend(values)
            )
            
            self.workflow_summaries[workflow_id][metric_type] = summary
            
        except Exception as e:
            logger.error(f"Failed to update workflow summary: {e}")
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction."""
        if len(values) < 2:
            return "stable"
        
        # Simple trend calculation using linear regression
        n = len(values)
        x = list(range(n))
        y = values
        
        # Calculate slope
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "stable"
        
        slope = numerator / denominator
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    def get_agent_summary(self, workflow_id: str, agent_id: str) -> Dict[str, Any]:
        """
        Get performance summary for an agent.
        
        Args:
            workflow_id: Workflow ID
            agent_id: Agent ID
            
        Returns:
            Agent performance summary
        """
        try:
            agent_key = f"{workflow_id}_{agent_id}"
            
            if agent_key not in self.agent_summaries:
                return {"message": "No metrics found for agent"}
            
            summaries = self.agent_summaries[agent_key]
            
            return {
                "workflow_id": workflow_id,
                "agent_id": agent_id,
                "metrics": {metric_type: asdict(summary) for metric_type, summary in summaries.items()},
                "total_metrics": sum(summary.count for summary in summaries.values()),
                "time_range": self._get_agent_time_range(workflow_id, agent_id)
            }
            
        except Exception as e:
            logger.error(f"Failed to get agent summary: {e}")
            return {"error": str(e)}
    
    def get_workflow_summary(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get performance summary for a workflow.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            Workflow performance summary
        """
        try:
            if workflow_id not in self.workflow_summaries:
                return {"message": "No metrics found for workflow"}
            
            summaries = self.workflow_summaries[workflow_id]
            
            return {
                "workflow_id": workflow_id,
                "metrics": {metric_type: asdict(summary) for metric_type, summary in summaries.items()},
                "total_metrics": sum(summary.count for summary in summaries.values()),
                "time_range": self._get_workflow_time_range(workflow_id)
            }
            
        except Exception as e:
            logger.error(f"Failed to get workflow summary: {e}")
            return {"error": str(e)}
    
    def _get_agent_time_range(self, workflow_id: str, agent_id: str) -> Dict[str, Any]:
        """Get time range for an agent."""
        try:
            agent_metrics = []
            for key, metrics_deque in self.metrics.items():
                if key.startswith(f"{workflow_id}_{agent_id}_"):
                    agent_metrics.extend(list(metrics_deque))
            
            if not agent_metrics:
                return {"start": None, "end": None}
            
            timestamps = [m.timestamp for m in agent_metrics]
            return {
                "start": min(timestamps).isoformat(),
                "end": max(timestamps).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get agent time range: {e}")
            return {"start": None, "end": None}
    
    def _get_workflow_time_range(self, workflow_id: str) -> Dict[str, Any]:
        """Get time range for a workflow."""
        try:
            workflow_metrics = []
            for key, metrics_deque in self.metrics.items():
                if key.startswith(f"{workflow_id}_"):
                    workflow_metrics.extend(list(metrics_deque))
            
            if not workflow_metrics:
                return {"start": None, "end": None}
            
            timestamps = [m.timestamp for m in workflow_metrics]
            return {
                "start": min(timestamps).isoformat(),
                "end": max(timestamps).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get workflow time range: {e}")
            return {"start": None, "end": None}
    
    def get_metrics_by_type(self, metric_type: str, 
                          workflow_id: Optional[str] = None,
                          agent_id: Optional[str] = None,
                          limit: int = 100) -> List[PerformanceMetric]:
        """
        Get metrics by type.
        
        Args:
            metric_type: Type of metric to retrieve
            workflow_id: Filter by workflow ID (optional)
            agent_id: Filter by agent ID (optional)
            limit: Maximum number of metrics to return
            
        Returns:
            List of performance metrics
        """
        try:
            metrics = []
            
            for key, metrics_deque in self.metrics.items():
                if not key.endswith(f"_{metric_type}"):
                    continue
                
                # Parse key to extract workflow_id and agent_id
                parts = key.split("_")
                if len(parts) < 3:
                    continue
                
                key_workflow_id = parts[0]
                key_agent_id = "_".join(parts[1:-1])  # Agent ID might contain underscores
                
                # Apply filters
                if workflow_id and key_workflow_id != workflow_id:
                    continue
                if agent_id and key_agent_id != agent_id:
                    continue
                
                metrics.extend(list(metrics_deque))
            
            # Sort by timestamp and limit
            metrics.sort(key=lambda x: x.timestamp, reverse=True)
            return metrics[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get metrics by type: {e}")
            return []
    
    def get_performance_alerts(self, threshold_config: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Get performance alerts based on thresholds.
        
        Args:
            threshold_config: Threshold configuration
            
        Returns:
            List of performance alerts
        """
        try:
            alerts = []
            
            for workflow_id, summaries in self.workflow_summaries.items():
                for metric_type, summary in summaries.items():
                    threshold_key = f"{metric_type}_threshold"
                    if threshold_key not in threshold_config:
                        continue
                    
                    threshold = threshold_config[threshold_key]
                    
                    # Check if metric exceeds threshold
                    if summary.mean_value > threshold:
                        alert = {
                            "workflow_id": workflow_id,
                            "metric_type": metric_type,
                            "current_value": summary.mean_value,
                            "threshold": threshold,
                            "severity": "high" if summary.mean_value > threshold * 1.5 else "medium",
                            "timestamp": datetime.now().isoformat()
                        }
                        alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to get performance alerts: {e}")
            return []
    
    def export_metrics(self, workflow_id: str, export_path: str, 
                      format: str = "json") -> bool:
        """
        Export metrics to file.
        
        Args:
            workflow_id: Workflow ID
            export_path: Path to export file
            format: Export format (json, csv)
            
        Returns:
            True if successful
        """
        try:
            # Get all metrics for workflow
            workflow_metrics = []
            for key, metrics_deque in self.metrics.items():
                if key.startswith(f"{workflow_id}_"):
                    workflow_metrics.extend(list(metrics_deque))
            
            if not workflow_metrics:
                logger.warning(f"No metrics found for workflow {workflow_id}")
                return False
            
            if format == "json":
                # Export as JSON
                export_data = {
                    "workflow_id": workflow_id,
                    "export_timestamp": datetime.now().isoformat(),
                    "metrics_count": len(workflow_metrics),
                    "metrics": [asdict(metric) for metric in workflow_metrics]
                }
                
                with open(export_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            
            elif format == "csv":
                # Export as CSV
                import csv
                with open(export_path, 'w', newline='') as f:
                    if workflow_metrics:
                        writer = csv.DictWriter(f, fieldnames=asdict(workflow_metrics[0]).keys())
                        writer.writeheader()
                        for metric in workflow_metrics:
                            writer.writerow(asdict(metric))
            
            logger.info(f"Exported {len(workflow_metrics)} metrics for workflow {workflow_id} to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return False
    
    def get_overall_summary(self) -> Dict[str, Any]:
        """
        Get overall performance summary.
        
        Returns:
            Overall performance summary
        """
        try:
            total_metrics = sum(len(metrics_deque) for metrics_deque in self.metrics.values())
            total_workflows = len(self.workflow_summaries)
            total_agents = len(self.agent_summaries)
            
            # Calculate overall statistics
            all_metrics = []
            for metrics_deque in self.metrics.values():
                all_metrics.extend(list(metrics_deque))
            
            if not all_metrics:
                return {
                    "total_metrics": 0,
                    "total_workflows": 0,
                    "total_agents": 0,
                    "time_range": {"start": None, "end": None}
                }
            
            # Calculate time range
            timestamps = [m.timestamp for m in all_metrics]
            time_range = {
                "start": min(timestamps).isoformat(),
                "end": max(timestamps).isoformat()
            }
            
            # Calculate metric type distribution
            metric_types = defaultdict(int)
            for metric in all_metrics:
                metric_types[metric.metric_type] += 1
            
            return {
                "total_metrics": total_metrics,
                "total_workflows": total_workflows,
                "total_agents": total_agents,
                "time_range": time_range,
                "metric_type_distribution": dict(metric_types),
                "average_metrics_per_workflow": total_metrics / total_workflows if total_workflows > 0 else 0,
                "average_metrics_per_agent": total_metrics / total_agents if total_agents > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get overall summary: {e}")
            return {"error": str(e)}


# Factory functions for common performance monitoring scenarios
def create_basic_metrics_collector() -> AgentPerformanceMetrics:
    """Create a basic performance metrics collector."""
    return AgentPerformanceMetrics(max_metrics_per_agent=500)


def create_high_volume_metrics_collector() -> AgentPerformanceMetrics:
    """Create a high-volume performance metrics collector."""
    return AgentPerformanceMetrics(max_metrics_per_agent=5000)


def create_mobile_metrics_collector() -> AgentPerformanceMetrics:
    """Create a mobile-optimized performance metrics collector."""
    return AgentPerformanceMetrics(max_metrics_per_agent=100)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create metrics collector
    metrics = create_basic_metrics_collector()
    
    # Record some sample metrics
    for i in range(10):
        metric = PerformanceMetric(
            metric_name=f"latency_{i}",
            value=100 + i * 10,
            timestamp=datetime.now(),
            workflow_id="test_workflow",
            agent_id="test_agent",
            metric_type="latency",
            unit="ms"
        )
        metrics.record_metric(metric)
    
    # Get agent summary
    agent_summary = metrics.get_agent_summary("test_workflow", "test_agent")
    print(f"Agent summary: {agent_summary}")
    
    # Get workflow summary
    workflow_summary = metrics.get_workflow_summary("test_workflow")
    print(f"Workflow summary: {workflow_summary}")
    
    # Get overall summary
    overall_summary = metrics.get_overall_summary()
    print(f"Overall summary: {overall_summary}")
    
    # Get performance alerts
    threshold_config = {"latency_threshold": 150}
    alerts = metrics.get_performance_alerts(threshold_config)
    print(f"Performance alerts: {alerts}")
