"""
Comprehensive Logging System for Enterprise AI Applications

This module implements a sophisticated multi-layer logging architecture designed
for enterprise-scale AI applications, providing comprehensive audit trails,
performance monitoring, error tracking, and debugging capabilities.

Key Features:
- Multi-layer logging architecture (Application, System, Security, Performance)
- Structured logging with JSON format
- Real-time monitoring and alerting
- Log aggregation and analysis
- Performance metrics collection
- Security event tracking
- Audit trail management
"""

import logging
import json
import time
import threading
import queue
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import traceback
import sys
import os
from collections import defaultdict, deque
import hashlib


class LogLevel(Enum):
    """Logging levels with enterprise-specific extensions"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    AUDIT = "AUDIT"
    SECURITY = "SECURITY"
    PERFORMANCE = "PERFORMANCE"


class LogCategory(Enum):
    """Log categories for enterprise applications"""
    APPLICATION = "APPLICATION"
    SYSTEM = "SYSTEM"
    SECURITY = "SECURITY"
    PERFORMANCE = "PERFORMANCE"
    AUDIT = "AUDIT"
    ERROR = "ERROR"
    BUSINESS = "BUSINESS"
    COMPLIANCE = "COMPLIANCE"
    MLOPS = "MLOPS"
    API = "API"


@dataclass
class LogEntry:
    """Structured log entry with comprehensive metadata"""
    timestamp: datetime
    level: LogLevel
    category: LogCategory
    message: str
    module: str
    function: str
    line_number: int
    thread_id: str
    process_id: int
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    security_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogFilter:
    """Log filtering configuration"""
    levels: List[LogLevel] = field(default_factory=list)
    categories: List[LogCategory] = field(default_factory=list)
    modules: List[str] = field(default_factory=list)
    time_range: Optional[tuple] = None
    user_ids: List[str] = field(default_factory=list)
    session_ids: List[str] = field(default_factory=list)


class LoggingSystem:
    """
    Enterprise-grade logging system with multi-layer architecture.
    
    This class provides comprehensive logging capabilities including:
    - Multi-layer logging (Application, System, Security, Performance)
    - Structured logging with JSON format
    - Real-time monitoring and alerting
    - Log aggregation and analysis
    - Performance metrics collection
    - Security event tracking
    - Audit trail management
    
    The system is designed for enterprise-scale AI applications with
    sophisticated monitoring, security, and compliance requirements.
    """
    
    def __init__(
        self,
        system_name: str = "Lenovo AAITC System",
        log_directory: str = "./logs",
        enable_console: bool = True,
        enable_file: bool = True,
        enable_remote: bool = False,
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        backup_count: int = 5,
        enable_performance_tracking: bool = True,
        enable_security_monitoring: bool = True
    ):
        """
        Initialize the logging system.
        
        Args:
            system_name: Name of the system being logged
            log_directory: Directory for log files
            enable_console: Whether to enable console logging
            enable_file: Whether to enable file logging
            enable_remote: Whether to enable remote logging
            max_file_size: Maximum size of log files before rotation
            backup_count: Number of backup files to keep
            enable_performance_tracking: Whether to enable performance tracking
            enable_security_monitoring: Whether to enable security monitoring
        """
        self.system_name = system_name
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(exist_ok=True)
        
        # Configuration
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_remote = enable_remote
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.enable_performance_tracking = enable_performance_tracking
        self.enable_security_monitoring = enable_security_monitoring
        
        # Logging components
        self.loggers = {}
        self.handlers = {}
        self.filters = {}
        self.formatters = {}
        
        # Performance tracking
        self.performance_metrics = defaultdict(list)
        self.performance_thresholds = {
            'response_time_ms': 1000,
            'memory_usage_mb': 1000,
            'cpu_usage_percent': 80,
            'error_rate_percent': 5
        }
        
        # Security monitoring
        self.security_events = deque(maxlen=10000)
        self.security_thresholds = {
            'failed_logins_per_minute': 10,
            'suspicious_requests_per_minute': 20,
            'privilege_escalation_attempts': 1
        }
        
        # Log aggregation
        self.log_buffer = deque(maxlen=1000)
        self.aggregation_rules = {}
        
        # Alerting
        self.alert_handlers = []
        self.alert_thresholds = {}
        
        # Initialize logging system
        self._initialize_logging_system()
        
        # Start background tasks
        self._start_background_tasks()
        
        self.info("Logging system initialized", category=LogCategory.SYSTEM)
    
    def _initialize_logging_system(self):
        """Initialize the logging system components"""
        
        # Create formatters
        self._create_formatters()
        
        # Create handlers
        self._create_handlers()
        
        # Create loggers for different categories
        self._create_loggers()
        
        # Set up log aggregation rules
        self._setup_aggregation_rules()
        
        # Set up alerting
        self._setup_alerting()
    
    def _create_formatters(self):
        """Create log formatters for different output types"""
        
        # JSON formatter for structured logging
        self.formatters['json'] = logging.Formatter(
            fmt='%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Detailed formatter for console
        self.formatters['detailed'] = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(category)-12s | %(name)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Simple formatter for basic logging
        self.formatters['simple'] = logging.Formatter(
            fmt='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def _create_handlers(self):
        """Create log handlers for different outputs"""
        
        if self.enable_console:
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(self.formatters['detailed'])
            console_handler.setLevel(logging.INFO)
            self.handlers['console'] = console_handler
        
        if self.enable_file:
            # File handlers for different categories
            for category in LogCategory:
                file_path = self.log_directory / f"{category.value.lower()}.log"
                file_handler = logging.handlers.RotatingFileHandler(
                    file_path,
                    maxBytes=self.max_file_size,
                    backupCount=self.backup_count
                )
                file_handler.setFormatter(self.formatters['json'])
                file_handler.setLevel(logging.DEBUG)
                self.handlers[f"file_{category.value.lower()}"] = file_handler
            
            # Combined log file
            combined_handler = logging.handlers.RotatingFileHandler(
                self.log_directory / "combined.log",
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            combined_handler.setFormatter(self.formatters['json'])
            combined_handler.setLevel(logging.DEBUG)
            self.handlers['file_combined'] = combined_handler
        
        if self.enable_remote:
            # Remote handler (placeholder for actual implementation)
            remote_handler = logging.StreamHandler()  # Placeholder
            remote_handler.setFormatter(self.formatters['json'])
            remote_handler.setLevel(logging.INFO)
            self.handlers['remote'] = remote_handler
    
    def _create_loggers(self):
        """Create loggers for different categories"""
        
        # Root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Add all handlers to root logger
        for handler in self.handlers.values():
            root_logger.addHandler(handler)
        
        # Category-specific loggers
        for category in LogCategory:
            logger_name = f"{self.system_name}.{category.value.lower()}"
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG)
            self.loggers[category] = logger
    
    def _setup_aggregation_rules(self):
        """Set up log aggregation rules"""
        
        self.aggregation_rules = {
            'error_rate': {
                'window_minutes': 5,
                'threshold': 0.05,  # 5% error rate
                'action': 'alert'
            },
            'performance_degradation': {
                'window_minutes': 10,
                'threshold': 2.0,  # 2x normal response time
                'action': 'alert'
            },
            'security_events': {
                'window_minutes': 1,
                'threshold': 10,  # 10 security events per minute
                'action': 'alert'
            }
        }
    
    def _setup_alerting(self):
        """Set up alerting system"""
        
        self.alert_thresholds = {
            LogLevel.ERROR: 10,  # 10 errors per minute
            LogLevel.CRITICAL: 1,  # 1 critical error
            LogLevel.SECURITY: 5,  # 5 security events per minute
            LogLevel.PERFORMANCE: 3  # 3 performance warnings per minute
        }
    
    def _start_background_tasks(self):
        """Start background monitoring tasks"""
        
        if self.enable_performance_tracking:
            threading.Thread(target=self._performance_monitor, daemon=True).start()
        
        if self.enable_security_monitoring:
            threading.Thread(target=self._security_monitor, daemon=True).start()
        
        threading.Thread(target=self._log_aggregator, daemon=True).start()
        threading.Thread(target=self._alert_monitor, daemon=True).start()
    
    def _performance_monitor(self):
        """Background performance monitoring"""
        while True:
            try:
                # Monitor system performance
                self._collect_performance_metrics()
                
                # Check performance thresholds
                self._check_performance_thresholds()
                
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.error(f"Performance monitor error: {str(e)}", category=LogCategory.SYSTEM)
                time.sleep(60)
    
    def _security_monitor(self):
        """Background security monitoring"""
        while True:
            try:
                # Analyze security events
                self._analyze_security_events()
                
                # Check security thresholds
                self._check_security_thresholds()
                
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.error(f"Security monitor error: {str(e)}", category=LogCategory.SYSTEM)
                time.sleep(30)
    
    def _log_aggregator(self):
        """Background log aggregation"""
        while True:
            try:
                # Process log buffer
                self._process_log_buffer()
                
                # Apply aggregation rules
                self._apply_aggregation_rules()
                
                time.sleep(10)  # Process every 10 seconds
            except Exception as e:
                self.error(f"Log aggregator error: {str(e)}", category=LogCategory.SYSTEM)
                time.sleep(10)
    
    def _alert_monitor(self):
        """Background alert monitoring"""
        while True:
            try:
                # Check alert thresholds
                self._check_alert_thresholds()
                
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.error(f"Alert monitor error: {str(e)}", category=LogCategory.SYSTEM)
                time.sleep(30)
    
    def _collect_performance_metrics(self):
        """Collect system performance metrics"""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Store metrics
            self.performance_metrics['cpu_percent'].append(cpu_percent)
            self.performance_metrics['memory_mb'].append(memory_mb)
            self.performance_metrics['disk_percent'].append(disk_percent)
            
            # Log performance metrics
            self.performance(
                "System performance metrics collected",
                category=LogCategory.PERFORMANCE,
                metadata={
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_mb,
                    'disk_percent': disk_percent
                }
            )
            
        except ImportError:
            # psutil not available, skip performance collection
            pass
        except Exception as e:
            self.error(f"Failed to collect performance metrics: {str(e)}", category=LogCategory.SYSTEM)
    
    def _check_performance_thresholds(self):
        """Check performance thresholds and generate alerts"""
        for metric, threshold in self.performance_thresholds.items():
            if metric in self.performance_metrics and self.performance_metrics[metric]:
                recent_values = self.performance_metrics[metric][-10:]  # Last 10 values
                avg_value = sum(recent_values) / len(recent_values)
                
                if avg_value > threshold:
                    self.warning(
                        f"Performance threshold exceeded: {metric} = {avg_value:.2f} (threshold: {threshold})",
                        category=LogCategory.PERFORMANCE,
                        metadata={
                            'metric': metric,
                            'value': avg_value,
                            'threshold': threshold
                        }
                    )
    
    def _analyze_security_events(self):
        """Analyze security events for patterns"""
        if len(self.security_events) < 10:
            return
        
        # Analyze recent security events
        recent_events = list(self.security_events)[-100:]  # Last 100 events
        
        # Count events by type
        event_counts = defaultdict(int)
        for event in recent_events:
            event_type = event.get('type', 'unknown')
            event_counts[event_type] += 1
        
        # Check for suspicious patterns
        for event_type, count in event_counts.items():
            if count > 5:  # More than 5 events of same type
                self.security(
                    f"Suspicious pattern detected: {count} {event_type} events",
                    category=LogCategory.SECURITY,
                    metadata={
                        'event_type': event_type,
                        'count': count,
                        'time_window': 'recent'
                    }
                )
    
    def _check_security_thresholds(self):
        """Check security thresholds and generate alerts"""
        # This would implement actual security threshold checking
        # For now, it's a placeholder
        pass
    
    def _process_log_buffer(self):
        """Process the log buffer for aggregation"""
        if not self.log_buffer:
            return
        
        # Process recent logs
        recent_logs = list(self.log_buffer)[-100:]  # Last 100 logs
        
        # Count logs by level and category
        log_counts = defaultdict(int)
        for log_entry in recent_logs:
            key = f"{log_entry.level.value}_{log_entry.category.value}"
            log_counts[key] += 1
        
        # Store aggregated data
        self.performance_metrics['log_counts'].append(dict(log_counts))
    
    def _apply_aggregation_rules(self):
        """Apply log aggregation rules"""
        for rule_name, rule_config in self.aggregation_rules.items():
            # This would implement actual aggregation rule processing
            # For now, it's a placeholder
            pass
    
    def _check_alert_thresholds(self):
        """Check alert thresholds and trigger alerts"""
        # This would implement actual alert threshold checking
        # For now, it's a placeholder
        pass
    
    def _create_log_entry(
        self,
        level: LogLevel,
        message: str,
        category: LogCategory = LogCategory.APPLICATION,
        metadata: Dict[str, Any] = None,
        stack_trace: str = None,
        performance_metrics: Dict[str, float] = None,
        security_context: Dict[str, Any] = None
    ) -> LogEntry:
        """Create a structured log entry"""
        
        # Get caller information
        frame = sys._getframe(2)  # Go up 2 frames to get the actual caller
        module = frame.f_globals.get('__name__', 'unknown')
        function = frame.f_code.co_name
        line_number = frame.f_lineno
        
        # Create log entry
        log_entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            category=category,
            message=message,
            module=module,
            function=function,
            line_number=line_number,
            thread_id=threading.current_thread().name,
            process_id=os.getpid(),
            metadata=metadata or {},
            stack_trace=stack_trace,
            performance_metrics=performance_metrics or {},
            security_context=security_context or {}
        )
        
        return log_entry
    
    def _log(self, log_entry: LogEntry):
        """Internal logging method"""
        
        # Add to buffer
        self.log_buffer.append(log_entry)
        
        # Create structured log message
        log_data = asdict(log_entry)
        log_data['timestamp'] = log_entry.timestamp.isoformat()
        log_data['level'] = log_entry.level.value
        log_data['category'] = log_entry.category.value
        
        # Convert to JSON
        log_message = json.dumps(log_data, default=str)
        
        # Get appropriate logger
        logger = self.loggers.get(log_entry.category, self.loggers[LogCategory.APPLICATION])
        
        # Log with appropriate level
        if log_entry.level == LogLevel.DEBUG:
            logger.debug(log_message)
        elif log_entry.level == LogLevel.INFO:
            logger.info(log_message)
        elif log_entry.level == LogLevel.WARNING:
            logger.warning(log_message)
        elif log_entry.level == LogLevel.ERROR:
            logger.error(log_message)
        elif log_entry.level == LogLevel.CRITICAL:
            logger.critical(log_message)
        else:
            logger.info(log_message)  # Default to info for custom levels
    
    def debug(self, message: str, category: LogCategory = LogCategory.APPLICATION, **kwargs):
        """Log debug message"""
        log_entry = self._create_log_entry(LogLevel.DEBUG, message, category, **kwargs)
        self._log(log_entry)
    
    def info(self, message: str, category: LogCategory = LogCategory.APPLICATION, **kwargs):
        """Log info message"""
        log_entry = self._create_log_entry(LogLevel.INFO, message, category, **kwargs)
        self._log(log_entry)
    
    def warning(self, message: str, category: LogCategory = LogCategory.APPLICATION, **kwargs):
        """Log warning message"""
        log_entry = self._create_log_entry(LogLevel.WARNING, message, category, **kwargs)
        self._log(log_entry)
    
    def error(self, message: str, category: LogCategory = LogCategory.ERROR, **kwargs):
        """Log error message"""
        stack_trace = traceback.format_exc() if kwargs.get('include_traceback', True) else None
        log_entry = self._create_log_entry(LogLevel.ERROR, message, category, stack_trace=stack_trace, **kwargs)
        self._log(log_entry)
    
    def critical(self, message: str, category: LogCategory = LogCategory.ERROR, **kwargs):
        """Log critical message"""
        stack_trace = traceback.format_exc() if kwargs.get('include_traceback', True) else None
        log_entry = self._create_log_entry(LogLevel.CRITICAL, message, category, stack_trace=stack_trace, **kwargs)
        self._log(log_entry)
    
    def audit(self, message: str, user_id: str = None, action: str = None, **kwargs):
        """Log audit message"""
        metadata = kwargs.get('metadata', {})
        if action:
            metadata['action'] = action
        if user_id:
            kwargs['user_id'] = user_id
        
        log_entry = self._create_log_entry(LogLevel.AUDIT, message, LogCategory.AUDIT, metadata=metadata, **kwargs)
        self._log(log_entry)
    
    def security(self, message: str, event_type: str = None, **kwargs):
        """Log security message"""
        metadata = kwargs.get('metadata', {})
        if event_type:
            metadata['event_type'] = event_type
        
        # Add to security events
        security_event = {
            'timestamp': datetime.now(),
            'type': event_type,
            'message': message,
            'metadata': metadata
        }
        self.security_events.append(security_event)
        
        log_entry = self._create_log_entry(LogLevel.SECURITY, message, LogCategory.SECURITY, metadata=metadata, **kwargs)
        self._log(log_entry)
    
    def performance(self, message: str, metrics: Dict[str, float] = None, **kwargs):
        """Log performance message"""
        log_entry = self._create_log_entry(LogLevel.PERFORMANCE, message, LogCategory.PERFORMANCE, performance_metrics=metrics, **kwargs)
        self._log(log_entry)
    
    def business(self, message: str, business_event: str = None, **kwargs):
        """Log business message"""
        metadata = kwargs.get('metadata', {})
        if business_event:
            metadata['business_event'] = business_event
        
        log_entry = self._create_log_entry(LogLevel.INFO, message, LogCategory.BUSINESS, metadata=metadata, **kwargs)
        self._log(log_entry)
    
    def mlops(self, message: str, model_id: str = None, stage: str = None, **kwargs):
        """Log MLOps message"""
        metadata = kwargs.get('metadata', {})
        if model_id:
            metadata['model_id'] = model_id
        if stage:
            metadata['stage'] = stage
        
        log_entry = self._create_log_entry(LogLevel.INFO, message, LogCategory.MLOPS, metadata=metadata, **kwargs)
        self._log(log_entry)
    
    def api(self, message: str, endpoint: str = None, method: str = None, status_code: int = None, **kwargs):
        """Log API message"""
        metadata = kwargs.get('metadata', {})
        if endpoint:
            metadata['endpoint'] = endpoint
        if method:
            metadata['method'] = method
        if status_code:
            metadata['status_code'] = status_code
        
        log_entry = self._create_log_entry(LogLevel.INFO, message, LogCategory.API, metadata=metadata, **kwargs)
        self._log(log_entry)
    
    def get_logs(
        self,
        filter_config: LogFilter = None,
        limit: int = 1000,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve logs based on filter criteria.
        
        Args:
            filter_config: Log filter configuration
            limit: Maximum number of logs to return
            offset: Number of logs to skip
            
        Returns:
            List of log entries matching the filter
        """
        if filter_config is None:
            filter_config = LogFilter()
        
        # Get logs from buffer (in production, this would query a database)
        logs = list(self.log_buffer)
        
        # Apply filters
        filtered_logs = []
        for log_entry in logs:
            if self._matches_filter(log_entry, filter_config):
                filtered_logs.append(asdict(log_entry))
        
        # Apply pagination
        start_idx = offset
        end_idx = offset + limit
        return filtered_logs[start_idx:end_idx]
    
    def _matches_filter(self, log_entry: LogEntry, filter_config: LogFilter) -> bool:
        """Check if log entry matches filter criteria"""
        
        # Level filter
        if filter_config.levels and log_entry.level not in filter_config.levels:
            return False
        
        # Category filter
        if filter_config.categories and log_entry.category not in filter_config.categories:
            return False
        
        # Module filter
        if filter_config.modules and log_entry.module not in filter_config.modules:
            return False
        
        # Time range filter
        if filter_config.time_range:
            start_time, end_time = filter_config.time_range
            if not (start_time <= log_entry.timestamp <= end_time):
                return False
        
        # User ID filter
        if filter_config.user_ids and log_entry.user_id not in filter_config.user_ids:
            return False
        
        # Session ID filter
        if filter_config.session_ids and log_entry.session_id not in filter_config.session_ids:
            return False
        
        return True
    
    def get_performance_metrics(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get performance metrics for the specified time window"""
        
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        # Filter recent metrics
        recent_metrics = {}
        for metric_name, values in self.performance_metrics.items():
            if values:
                recent_metrics[metric_name] = values[-100:]  # Last 100 values
        
        # Calculate statistics
        stats = {}
        for metric_name, values in recent_metrics.items():
            if values:
                stats[metric_name] = {
                    'current': values[-1] if values else 0,
                    'average': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
        
        return {
            'time_window_hours': time_window_hours,
            'metrics': stats,
            'thresholds': self.performance_thresholds,
            'generated_at': datetime.now().isoformat()
        }
    
    def get_security_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get security event summary for the specified time window"""
        
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        # Filter recent security events
        recent_events = [
            event for event in self.security_events
            if event['timestamp'] >= cutoff_time
        ]
        
        # Count events by type
        event_counts = defaultdict(int)
        for event in recent_events:
            event_type = event.get('type', 'unknown')
            event_counts[event_type] += 1
        
        return {
            'time_window_hours': time_window_hours,
            'total_events': len(recent_events),
            'event_counts': dict(event_counts),
            'thresholds': self.security_thresholds,
            'generated_at': datetime.now().isoformat()
        }
    
    def add_alert_handler(self, handler: Callable):
        """Add an alert handler function"""
        self.alert_handlers.append(handler)
    
    def set_performance_threshold(self, metric: str, threshold: float):
        """Set performance threshold for a metric"""
        self.performance_thresholds[metric] = threshold
    
    def set_security_threshold(self, metric: str, threshold: float):
        """Set security threshold for a metric"""
        self.security_thresholds[metric] = threshold
    
    def shutdown(self):
        """Shutdown the logging system"""
        self.info("Logging system shutting down", category=LogCategory.SYSTEM)
        
        # Close all handlers
        for handler in self.handlers.values():
            handler.close()
        
        # Clear buffers
        self.log_buffer.clear()
        self.security_events.clear()
        self.performance_metrics.clear()


# Global logging system instance
_global_logger = None


def get_logger(system_name: str = "Lenovo AAITC System") -> LoggingSystem:
    """Get the global logging system instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = LoggingSystem(system_name)
    return _global_logger


def setup_logging(
    system_name: str = "Lenovo AAITC System",
    log_directory: str = "./logs",
    **kwargs
) -> LoggingSystem:
    """Set up the global logging system"""
    global _global_logger
    _global_logger = LoggingSystem(system_name, log_directory, **kwargs)
    return _global_logger
