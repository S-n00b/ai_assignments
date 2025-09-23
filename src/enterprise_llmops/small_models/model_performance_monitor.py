"""
Model Performance Monitor for Small Models

This module provides real-time performance monitoring and analytics
for small models deployed on mobile and edge platforms.

Key Features:
- Real-time performance metrics collection
- Latency and throughput monitoring
- Memory usage tracking
- GPU utilization monitoring
- Performance alerts and notifications
- Historical performance analytics
"""

import asyncio
import logging
import time
import psutil
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque
import threading
from pathlib import Path


@dataclass
class PerformanceMetrics:
    """Performance metrics for a model."""
    model_name: str
    timestamp: datetime
    latency_ms: float
    throughput_tokens_per_sec: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    request_count: int = 1
    error_count: int = 0
    success_rate: float = 1.0


@dataclass
class PerformanceAlert:
    """Performance alert configuration."""
    metric_name: str
    threshold_value: float
    comparison_operator: str  # ">", "<", ">=", "<=", "=="
    severity: str  # "low", "medium", "high", "critical"
    enabled: bool = True


@dataclass
class PerformanceSummary:
    """Performance summary for a time period."""
    model_name: str
    time_period: str
    start_time: datetime
    end_time: datetime
    total_requests: int
    total_errors: int
    average_latency_ms: float
    average_throughput_tps: float
    peak_memory_usage_mb: float
    average_cpu_usage_percent: float
    success_rate: float
    p95_latency_ms: float
    p99_latency_ms: float


class ModelPerformanceMonitor:
    """
    Real-time performance monitor for small models.
    
    This class provides comprehensive performance monitoring including
    metrics collection, alerting, and historical analytics.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the performance monitor."""
        self.config = config or self._get_default_config()
        self.logger = self._setup_logging()
        
        # Performance data storage
        self.metrics_history = {}  # model_name -> deque of PerformanceMetrics
        self.active_models = set()
        self.performance_alerts = []
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task = None
        self.alert_callbacks = []
        
        # Performance thresholds
        self.thresholds = self.config.get("performance_monitoring", {}).get("alert_thresholds", {})
        
        # Initialize alert configurations
        self._setup_default_alerts()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "performance_monitoring": {
                "metrics_collection_interval": 5,
                "latency_tracking": True,
                "memory_usage_tracking": True,
                "gpu_utilization_tracking": False,
                "history_retention_hours": 24,
                "alert_thresholds": {
                    "latency_ms": 500,
                    "memory_usage_percent": 80,
                    "cpu_usage_percent": 85,
                    "error_rate_percent": 10
                }
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for performance monitor."""
        logger = logging.getLogger("model_performance_monitor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _setup_default_alerts(self):
        """Setup default performance alerts."""
        thresholds = self.thresholds
        
        self.performance_alerts = [
            PerformanceAlert(
                metric_name="latency_ms",
                threshold_value=thresholds.get("latency_ms", 500),
                comparison_operator=">",
                severity="medium"
            ),
            PerformanceAlert(
                metric_name="memory_usage_percent",
                threshold_value=thresholds.get("memory_usage_percent", 80),
                comparison_operator=">",
                severity="high"
            ),
            PerformanceAlert(
                metric_name="cpu_usage_percent",
                threshold_value=thresholds.get("cpu_usage_percent", 85),
                comparison_operator=">",
                severity="medium"
            ),
            PerformanceAlert(
                metric_name="error_rate_percent",
                threshold_value=thresholds.get("error_rate_percent", 10),
                comparison_operator=">",
                severity="high"
            )
        ]
    
    async def start_monitoring(self):
        """Start performance monitoring."""
        if self.monitoring_active:
            self.logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        collection_interval = self.config["performance_monitoring"]["metrics_collection_interval"]
        
        while self.monitoring_active:
            try:
                # Collect metrics for all active models
                for model_name in self.active_models:
                    await self._collect_metrics(model_name)
                
                # Check alerts
                await self._check_alerts()
                
                # Clean up old metrics
                await self._cleanup_old_metrics()
                
                await asyncio.sleep(collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(collection_interval)
    
    async def _collect_metrics(self, model_name: str):
        """Collect performance metrics for a model."""
        try:
            # Get system resource usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Get model-specific metrics (simulated for now)
            model_metrics = await self._get_model_specific_metrics(model_name)
            
            # Create performance metrics
            metrics = PerformanceMetrics(
                model_name=model_name,
                timestamp=datetime.now(),
                latency_ms=model_metrics.get("latency_ms", 100.0),
                throughput_tokens_per_sec=model_metrics.get("throughput_tps", 50.0),
                memory_usage_mb=memory.used / (1024 * 1024),
                cpu_usage_percent=cpu_percent,
                gpu_usage_percent=model_metrics.get("gpu_usage_percent"),
                gpu_memory_mb=model_metrics.get("gpu_memory_mb"),
                request_count=model_metrics.get("request_count", 1),
                error_count=model_metrics.get("error_count", 0),
                success_rate=model_metrics.get("success_rate", 1.0)
            )
            
            # Store metrics
            if model_name not in self.metrics_history:
                self.metrics_history[model_name] = deque(maxlen=1000)
            
            self.metrics_history[model_name].append(metrics)
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics for {model_name}: {e}")
    
    async def _get_model_specific_metrics(self, model_name: str) -> Dict[str, Any]:
        """Get model-specific performance metrics."""
        # This would integrate with actual model serving infrastructure
        # For now, we'll simulate metrics based on model characteristics
        
        base_metrics = {
            "latency_ms": 100.0,
            "throughput_tps": 50.0,
            "request_count": 1,
            "error_count": 0,
            "success_rate": 1.0
        }
        
        # Model-specific adjustments
        if "phi-4-mini" in model_name.lower():
            base_metrics.update({
                "latency_ms": 85.0,
                "throughput_tps": 55.0
            })
        elif "llama-3.2-3b" in model_name.lower():
            base_metrics.update({
                "latency_ms": 95.0,
                "throughput_tps": 52.0
            })
        elif "mistral-nemo" in model_name.lower():
            base_metrics.update({
                "latency_ms": 75.0,
                "throughput_tps": 60.0
            })
        
        # Add some realistic variance
        import random
        variance_factor = random.uniform(0.9, 1.1)
        base_metrics["latency_ms"] *= variance_factor
        base_metrics["throughput_tps"] *= variance_factor
        
        return base_metrics
    
    async def _check_alerts(self):
        """Check performance alerts."""
        for alert in self.performance_alerts:
            if not alert.enabled:
                continue
            
            try:
                # Check if any model exceeds the threshold
                for model_name, metrics_history in self.metrics_history.items():
                    if not metrics_history:
                        continue
                    
                    latest_metrics = metrics_history[-1]
                    metric_value = getattr(latest_metrics, alert.metric_name, None)
                    
                    if metric_value is not None:
                        if self._evaluate_alert_condition(metric_value, alert):
                            await self._trigger_alert(alert, model_name, metric_value, latest_metrics)
                            
            except Exception as e:
                self.logger.error(f"Error checking alert {alert.metric_name}: {e}")
    
    def _evaluate_alert_condition(self, metric_value: float, alert: PerformanceAlert) -> bool:
        """Evaluate if an alert condition is met."""
        if alert.comparison_operator == ">":
            return metric_value > alert.threshold_value
        elif alert.comparison_operator == "<":
            return metric_value < alert.threshold_value
        elif alert.comparison_operator == ">=":
            return metric_value >= alert.threshold_value
        elif alert.comparison_operator == "<=":
            return metric_value <= alert.threshold_value
        elif alert.comparison_operator == "==":
            return abs(metric_value - alert.threshold_value) < 0.001
        else:
            return False
    
    async def _trigger_alert(self, alert: PerformanceAlert, model_name: str, metric_value: float, metrics: PerformanceMetrics):
        """Trigger a performance alert."""
        alert_message = (
            f"Performance Alert [{alert.severity.upper()}]: "
            f"Model {model_name} {alert.metric_name} = {metric_value:.2f} "
            f"{alert.comparison_operator} {alert.threshold_value}"
        )
        
        self.logger.warning(alert_message)
        
        # Call registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert, model_name, metric_value, metrics)
                else:
                    callback(alert, model_name, metric_value, metrics)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    async def _cleanup_old_metrics(self):
        """Clean up old metrics based on retention policy."""
        retention_hours = self.config["performance_monitoring"]["history_retention_hours"]
        cutoff_time = datetime.now() - timedelta(hours=retention_hours)
        
        for model_name, metrics_history in self.metrics_history.items():
            # Remove old metrics
            while metrics_history and metrics_history[0].timestamp < cutoff_time:
                metrics_history.popleft()
    
    def register_model(self, model_name: str):
        """Register a model for monitoring."""
        self.active_models.add(model_name)
        if model_name not in self.metrics_history:
            self.metrics_history[model_name] = deque(maxlen=1000)
        self.logger.info(f"Registered model for monitoring: {model_name}")
    
    def unregister_model(self, model_name: str):
        """Unregister a model from monitoring."""
        self.active_models.discard(model_name)
        self.logger.info(f"Unregistered model from monitoring: {model_name}")
    
    def add_alert_callback(self, callback: Callable):
        """Add an alert callback function."""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self, model_name: str) -> Optional[PerformanceMetrics]:
        """Get current performance metrics for a model."""
        if model_name not in self.metrics_history or not self.metrics_history[model_name]:
            return None
        
        return self.metrics_history[model_name][-1]
    
    def get_performance_summary(self, model_name: str, hours: int = 1) -> Optional[PerformanceSummary]:
        """Get performance summary for a model over a time period."""
        if model_name not in self.metrics_history:
            return None
        
        metrics_history = self.metrics_history[model_name]
        if not metrics_history:
            return None
        
        # Filter metrics for the specified time period
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return None
        
        # Calculate summary statistics
        total_requests = sum(m.request_count for m in recent_metrics)
        total_errors = sum(m.error_count for m in recent_metrics)
        
        latencies = [m.latency_ms for m in recent_metrics]
        throughputs = [m.throughput_tokens_per_sec for m in recent_metrics]
        memory_usage = [m.memory_usage_mb for m in recent_metrics]
        cpu_usage = [m.cpu_usage_percent for m in recent_metrics]
        
        # Calculate percentiles
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)
        p95_latency = latencies_sorted[int(n * 0.95)] if n > 0 else 0
        p99_latency = latencies_sorted[int(n * 0.99)] if n > 0 else 0
        
        return PerformanceSummary(
            model_name=model_name,
            time_period=f"last_{hours}_hours",
            start_time=recent_metrics[0].timestamp,
            end_time=recent_metrics[-1].timestamp,
            total_requests=total_requests,
            total_errors=total_errors,
            average_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            average_throughput_tps=sum(throughputs) / len(throughputs) if throughputs else 0,
            peak_memory_usage_mb=max(memory_usage) if memory_usage else 0,
            average_cpu_usage_percent=sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0,
            success_rate=(total_requests - total_errors) / total_requests if total_requests > 0 else 0,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency
        )
    
    def get_all_models_status(self) -> Dict[str, Any]:
        """Get status of all monitored models."""
        status = {
            "monitoring_active": self.monitoring_active,
            "total_models": len(self.active_models),
            "models": {}
        }
        
        for model_name in self.active_models:
            current_metrics = self.get_current_metrics(model_name)
            summary = self.get_performance_summary(model_name, hours=1)
            
            status["models"][model_name] = {
                "active": True,
                "current_metrics": asdict(current_metrics) if current_metrics else None,
                "summary_1h": asdict(summary) if summary else None,
                "metrics_count": len(self.metrics_history.get(model_name, []))
            }
        
        return status
    
    def export_metrics(self, model_name: str, output_file: str):
        """Export performance metrics to a file."""
        if model_name not in self.metrics_history:
            return False
        
        try:
            metrics_data = []
            for metrics in self.metrics_history[model_name]:
                metrics_dict = asdict(metrics)
                metrics_dict["timestamp"] = metrics.timestamp.isoformat()
                metrics_data.append(metrics_dict)
            
            with open(output_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            self.logger.info(f"Exported {len(metrics_data)} metrics for {model_name} to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
            return False


# Example usage and testing
async def main():
    """Example usage of model performance monitor."""
    monitor = ModelPerformanceMonitor()
    
    # Add alert callback
    async def alert_callback(alert, model_name, metric_value, metrics):
        print(f"ALERT: {alert.metric_name} = {metric_value:.2f} for {model_name}")
    
    monitor.add_alert_callback(alert_callback)
    
    try:
        # Start monitoring
        await monitor.start_monitoring()
        
        # Register models
        models = ["phi-4-mini", "llama-3.2-3b", "mistral-nemo"]
        for model in models:
            monitor.register_model(model)
        
        # Monitor for a short period
        print("Monitoring models for 30 seconds...")
        await asyncio.sleep(30)
        
        # Get performance summaries
        for model in models:
            summary = monitor.get_performance_summary(model, hours=1)
            if summary:
                print(f"\n{model} Performance Summary:")
                print(f"  Average Latency: {summary.average_latency_ms:.1f}ms")
                print(f"  Average Throughput: {summary.average_throughput_tps:.1f} tokens/sec")
                print(f"  Success Rate: {summary.success_rate:.2%}")
                print(f"  P95 Latency: {summary.p95_latency_ms:.1f}ms")
        
        # Get overall status
        status = monitor.get_all_models_status()
        print(f"\nOverall Status: {status['total_models']} models monitored")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await monitor.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())
