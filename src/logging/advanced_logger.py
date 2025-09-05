"""
Advanced Logging and Debugging System
Professional-grade logging with structured output, correlation IDs, and debugging tools
"""

import os
import sys
import json
import time
import threading
import traceback
import logging
import logging.handlers
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from contextlib import contextmanager
import uuid
import warnings
warnings.filterwarnings('ignore')

try:
    import colorama
    from colorama import Fore, Back, Style
    colorama.init()
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

class LogLevel(Enum):
    """Extended log levels"""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    AUDIT = 60

class LogFormat(Enum):
    """Log output formats"""
    SIMPLE = "simple"
    DETAILED = "detailed"
    JSON = "json"
    COLORED = "colored"

@dataclass
class LogContext:
    """Log context information"""
    correlation_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: datetime
    level: str
    logger_name: str
    message: str
    context: Optional[LogContext] = None
    exception: Optional[Dict[str, Any]] = None
    performance: Optional[Dict[str, Any]] = None
    extra_fields: Dict[str, Any] = field(default_factory=dict)

class AdvancedLogger:
    """Advanced logging system with structured output and debugging features"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.loggers: Dict[str, logging.Logger] = {}
        self.context_stack = threading.local()
        
        # Performance tracking
        self.performance_metrics: Dict[str, List[float]] = {}
        
        # Log aggregation and analysis
        self.log_buffer: List[LogEntry] = []
        self.log_stats = {
            'total_logs': 0,
            'logs_by_level': {},
            'error_count': 0,
            'warning_count': 0
        }
        
        # Setup main logger
        self.setup_logging()
        
        # Create main logger instance
        self.logger = self.get_logger("stock_ai")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default logging configuration"""
        return {
            'level': 'INFO',
            'format': LogFormat.DETAILED,
            'output_file': 'logs/stock_ai.log',
            'max_file_size': 100 * 1024 * 1024,  # 100MB
            'backup_count': 10,
            'enable_console': True,
            'enable_file': True,
            'enable_json_file': True,
            'json_file': 'logs/stock_ai.jsonl',
            'enable_performance_logging': True,
            'enable_audit_logging': True,
            'audit_file': 'logs/audit.log',
            'log_retention_days': 30,
            'buffer_size': 1000,
            'enable_correlation_ids': True,
            'enable_colored_output': True
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        
        # Create logs directory
        log_dir = os.path.dirname(self.config.get('output_file', 'logs/stock_ai.log'))
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config['level']))
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler
        if self.config['enable_console']:
            console_handler = self._create_console_handler()
            root_logger.addHandler(console_handler)
        
        # File handler
        if self.config['enable_file']:
            file_handler = self._create_file_handler()
            root_logger.addHandler(file_handler)
        
        # JSON file handler
        if self.config['enable_json_file']:
            json_handler = self._create_json_handler()
            root_logger.addHandler(json_handler)
        
        # Audit handler
        if self.config['enable_audit_logging']:
            audit_handler = self._create_audit_handler()
            # Audit handler only processes AUDIT level logs
            audit_handler.addFilter(lambda record: record.levelno >= LogLevel.AUDIT.value)
            root_logger.addHandler(audit_handler)
    
    def _create_console_handler(self) -> logging.Handler:
        """Create console log handler"""
        
        handler = logging.StreamHandler(sys.stdout)
        
        if self.config['format'] == LogFormat.COLORED and COLORAMA_AVAILABLE:
            formatter = ColoredFormatter()
        elif self.config['format'] == LogFormat.JSON:
            formatter = JSONFormatter()
        else:
            formatter = DetailedFormatter()
        
        handler.setFormatter(formatter)
        return handler
    
    def _create_file_handler(self) -> logging.Handler:
        """Create rotating file handler"""
        
        handler = logging.handlers.RotatingFileHandler(
            filename=self.config['output_file'],
            maxBytes=self.config['max_file_size'],
            backupCount=self.config['backup_count'],
            encoding='utf-8'
        )
        
        formatter = DetailedFormatter()
        handler.setFormatter(formatter)
        
        return handler
    
    def _create_json_handler(self) -> logging.Handler:
        """Create JSON file handler"""
        
        handler = logging.handlers.RotatingFileHandler(
            filename=self.config['json_file'],
            maxBytes=self.config['max_file_size'],
            backupCount=self.config['backup_count'],
            encoding='utf-8'
        )
        
        formatter = JSONFormatter()
        handler.setFormatter(formatter)
        
        return handler
    
    def _create_audit_handler(self) -> logging.Handler:
        """Create audit log handler"""
        
        audit_file = self.config.get('audit_file', 'logs/audit.log')
        
        handler = logging.handlers.RotatingFileHandler(
            filename=audit_file,
            maxBytes=self.config['max_file_size'],
            backupCount=self.config['backup_count'],
            encoding='utf-8'
        )
        
        formatter = AuditFormatter()
        handler.setFormatter(formatter)
        
        return handler
    
    def get_logger(self, name: str) -> 'ContextLogger':
        """Get or create a context-aware logger"""
        
        if name not in self.loggers:
            logger = logging.getLogger(name)
            self.loggers[name] = logger
        
        return ContextLogger(self.loggers[name], self)
    
    def set_context(self, **context_data):
        """Set logging context for current thread"""
        
        if not hasattr(self.context_stack, 'context'):
            self.context_stack.context = LogContext(
                correlation_id=str(uuid.uuid4())
            )
        
        # Update context with provided data
        for key, value in context_data.items():
            if hasattr(self.context_stack.context, key):
                setattr(self.context_stack.context, key, value)
            else:
                self.context_stack.context.metadata[key] = value
    
    def get_context(self) -> Optional[LogContext]:
        """Get current logging context"""
        return getattr(self.context_stack, 'context', None)
    
    def clear_context(self):
        """Clear current logging context"""
        if hasattr(self.context_stack, 'context'):
            delattr(self.context_stack, 'context')
    
    @contextmanager
    def context(self, **context_data):
        """Context manager for temporary logging context"""
        
        # Save current context
        old_context = getattr(self.context_stack, 'context', None)
        
        try:
            # Set new context
            if old_context:
                # Merge with existing context
                new_context = LogContext(
                    correlation_id=old_context.correlation_id,
                    user_id=context_data.get('user_id', old_context.user_id),
                    session_id=context_data.get('session_id', old_context.session_id),
                    component=context_data.get('component', old_context.component),
                    operation=context_data.get('operation', old_context.operation),
                    request_id=context_data.get('request_id', old_context.request_id),
                    metadata={**old_context.metadata, **context_data.get('metadata', {})}
                )
                
                # Add other context data to metadata
                for key, value in context_data.items():
                    if key not in ['user_id', 'session_id', 'component', 'operation', 'request_id', 'metadata']:
                        new_context.metadata[key] = value
            else:
                new_context = LogContext(
                    correlation_id=str(uuid.uuid4()),
                    **{k: v for k, v in context_data.items() if k != 'metadata'},
                    metadata=context_data.get('metadata', {})
                )
                
                # Add other context data to metadata
                for key, value in context_data.items():
                    if key not in ['user_id', 'session_id', 'component', 'operation', 'request_id', 'metadata']:
                        new_context.metadata[key] = value
            
            self.context_stack.context = new_context
            
            yield new_context
            
        finally:
            # Restore old context
            if old_context:
                self.context_stack.context = old_context
            else:
                self.clear_context()
    
    @contextmanager
    def performance_timer(self, operation: str, component: str = None):
        """Context manager for performance timing"""
        
        start_time = time.time()
        
        with self.context(operation=operation, component=component):
            try:
                yield
                
            finally:
                duration = time.time() - start_time
                
                # Record performance metric
                if component:
                    key = f"{component}.{operation}"
                else:
                    key = operation
                
                if key not in self.performance_metrics:
                    self.performance_metrics[key] = []
                
                self.performance_metrics[key].append(duration)
                
                # Log performance if enabled
                if self.config['enable_performance_logging']:
                    self.logger.info(
                        f"Performance: {operation} completed",
                        extra={
                            'performance': {
                                'operation': operation,
                                'duration_seconds': duration,
                                'duration_ms': duration * 1000
                            }
                        }
                    )
    
    def audit_log(self, action: str, details: Dict[str, Any], user_id: str = None):
        """Log audit event"""
        
        audit_logger = logging.getLogger('audit')
        
        audit_data = {
            'action': action,
            'details': details,
            'user_id': user_id or (self.get_context().user_id if self.get_context() else None),
            'timestamp': datetime.now().isoformat(),
            'correlation_id': self.get_context().correlation_id if self.get_context() else str(uuid.uuid4())
        }
        
        audit_logger.log(LogLevel.AUDIT.value, json.dumps(audit_data))
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics"""
        
        stats = {}
        
        for operation, timings in self.performance_metrics.items():
            if timings:
                stats[operation] = {
                    'count': len(timings),
                    'total_time': sum(timings),
                    'avg_time': sum(timings) / len(timings),
                    'min_time': min(timings),
                    'max_time': max(timings),
                    'p95_time': sorted(timings)[int(len(timings) * 0.95)] if len(timings) > 20 else max(timings)
                }
        
        return stats
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        return self.log_stats.copy()

class ContextLogger:
    """Logger wrapper with context awareness"""
    
    def __init__(self, logger: logging.Logger, advanced_logger: AdvancedLogger):
        self.logger = logger
        self.advanced_logger = advanced_logger
    
    def _log_with_context(self, level: int, message: str, *args, **kwargs):
        """Log with context information"""
        
        # Get current context
        context = self.advanced_logger.get_context()
        
        # Prepare extra fields
        extra = kwargs.get('extra', {})
        
        if context:
            extra['correlation_id'] = context.correlation_id
            if context.user_id:
                extra['user_id'] = context.user_id
            if context.session_id:
                extra['session_id'] = context.session_id
            if context.component:
                extra['component'] = context.component
            if context.operation:
                extra['operation'] = context.operation
            if context.request_id:
                extra['request_id'] = context.request_id
            if context.metadata:
                extra['metadata'] = context.metadata
        
        kwargs['extra'] = extra
        
        # Update statistics
        self.advanced_logger.log_stats['total_logs'] += 1
        level_name = logging.getLevelName(level)
        self.advanced_logger.log_stats['logs_by_level'][level_name] = \
            self.advanced_logger.log_stats['logs_by_level'].get(level_name, 0) + 1
        
        if level >= logging.ERROR:
            self.advanced_logger.log_stats['error_count'] += 1
        elif level >= logging.WARNING:
            self.advanced_logger.log_stats['warning_count'] += 1
        
        # Log the message
        self.logger.log(level, message, *args, **kwargs)
    
    def trace(self, message: str, *args, **kwargs):
        """Log trace message"""
        self._log_with_context(LogLevel.TRACE.value, message, *args, **kwargs)
    
    def debug(self, message: str, *args, **kwargs):
        """Log debug message"""
        self._log_with_context(logging.DEBUG, message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Log info message"""
        self._log_with_context(logging.INFO, message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log warning message"""
        self._log_with_context(logging.WARNING, message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log error message"""
        self._log_with_context(logging.ERROR, message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Log critical message"""
        self._log_with_context(logging.CRITICAL, message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs):
        """Log exception with traceback"""
        kwargs['exc_info'] = True
        self._log_with_context(logging.ERROR, message, *args, **kwargs)

class DetailedFormatter(logging.Formatter):
    """Detailed log formatter"""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(correlation_id)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def format(self, record):
        # Ensure correlation_id exists
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = 'N/A'
        
        # Add context fields to record
        for field in ['user_id', 'session_id', 'component', 'operation', 'request_id']:
            if hasattr(record, field):
                setattr(record, field, getattr(record, field) or 'N/A')
        
        return super().format(record)

class JSONFormatter(logging.Formatter):
    """JSON log formatter"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add context information
        for field in ['correlation_id', 'user_id', 'session_id', 'component', 'operation', 'request_id']:
            if hasattr(record, field) and getattr(record, field):
                log_entry[field] = getattr(record, field)
        
        # Add metadata
        if hasattr(record, 'metadata') and record.metadata:
            log_entry['metadata'] = record.metadata
        
        # Add performance data
        if hasattr(record, 'performance') and record.performance:
            log_entry['performance'] = record.performance
        
        # Add exception information
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_entry)

class ColoredFormatter(logging.Formatter):
    """Colored console log formatter"""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    } if COLORAMA_AVAILABLE else {}
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
    
    def format(self, record):
        if COLORAMA_AVAILABLE:
            # Color the level name
            level_color = self.COLORS.get(record.levelname, '')
            if level_color:
                record.levelname = f"{level_color}{record.levelname}{Style.RESET_ALL}"
            
            # Color the message based on level
            if record.levelno >= logging.ERROR:
                record.msg = f"{Fore.RED}{record.msg}{Style.RESET_ALL}"
            elif record.levelno >= logging.WARNING:
                record.msg = f"{Fore.YELLOW}{record.msg}{Style.RESET_ALL}"
        
        return super().format(record)

class AuditFormatter(logging.Formatter):
    """Audit log formatter"""
    
    def format(self, record):
        # Audit logs are already JSON formatted
        return record.getMessage()

class DebugHelper:
    """Debugging utilities"""
    
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
    
    def dump_variables(self, variables: Dict[str, Any], context: str = "Variable dump"):
        """Dump variables for debugging"""
        
        self.logger.logger.debug(f"{context}:")
        for name, value in variables.items():
            self.logger.logger.debug(f"  {name}: {repr(value)}")
    
    def trace_function_calls(self, func):
        """Decorator to trace function calls"""
        
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            
            with self.logger.context(operation=func_name):
                self.logger.logger.debug(f"Entering {func_name}")
                self.logger.logger.trace(f"Args: {args}, Kwargs: {kwargs}")
                
                try:
                    with self.logger.performance_timer(func_name):
                        result = func(*args, **kwargs)
                    
                    self.logger.logger.trace(f"Result: {result}")
                    self.logger.logger.debug(f"Exiting {func_name}")
                    
                    return result
                    
                except Exception as e:
                    self.logger.logger.exception(f"Exception in {func_name}: {e}")
                    raise
        
        return wrapper
    
    def memory_usage(self):
        """Log current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            self.logger.logger.info(
                "Memory usage",
                extra={
                    'memory': {
                        'rss_mb': memory_info.rss / 1024 / 1024,
                        'vms_mb': memory_info.vms / 1024 / 1024
                    }
                }
            )
        except ImportError:
            self.logger.logger.warning("psutil not available for memory monitoring")

# Global logger instance
_global_logger = None

def get_logger(name: str = "stock_ai") -> ContextLogger:
    """Get the global logger instance"""
    global _global_logger
    
    if _global_logger is None:
        _global_logger = AdvancedLogger()
    
    return _global_logger.get_logger(name)

def setup_logging(config: Dict[str, Any] = None):
    """Setup global logging configuration"""
    global _global_logger
    
    _global_logger = AdvancedLogger(config)
    return _global_logger

# Example usage and testing
if __name__ == "__main__":
    print("📝 Advanced Logging System")
    print("=" * 32)
    
    # Initialize logging system
    config = {
        'level': 'DEBUG',
        'format': LogFormat.COLORED,
        'enable_console': True,
        'enable_file': False,  # Disable for testing
        'enable_json_file': False,  # Disable for testing
        'enable_performance_logging': True
    }
    
    advanced_logger = AdvancedLogger(config)
    logger = advanced_logger.get_logger("test")
    debug_helper = DebugHelper(advanced_logger)
    
    print("✅ Logging system initialized")
    
    # Test basic logging
    print("\n📋 Testing basic logging...")
    logger.info("System startup")
    logger.debug("Debug information")
    logger.warning("This is a warning")
    logger.error("This is an error")
    
    # Test context logging
    print("\n🔗 Testing context logging...")
    with advanced_logger.context(user_id="user123", component="data_collector"):
        logger.info("Processing user data")
        
        with advanced_logger.context(operation="fetch_data"):
            logger.info("Fetching data from external API")
            logger.warning("API rate limit approaching")
    
    # Test performance timing
    print("\n⚡ Testing performance timing...")
    with advanced_logger.performance_timer("data_processing", "predictor"):
        time.sleep(0.1)  # Simulate work
        logger.info("Data processing completed")
    
    # Test audit logging
    print("\n📊 Testing audit logging...")
    advanced_logger.audit_log("user_login", {
        "user_id": "user123",
        "ip_address": "192.168.1.100",
        "user_agent": "Mozilla/5.0..."
    })
    
    # Test debug helper
    print("\n🔍 Testing debug helper...")
    test_vars = {
        'symbol': 'AAPL',
        'price': 150.25,
        'volume': 1000000
    }
    debug_helper.dump_variables(test_vars, "Stock data")
    
    # Test function tracing
    @debug_helper.trace_function_calls
    def sample_function(x, y):
        return x + y
    
    result = sample_function(10, 20)
    
    # Test exception logging
    print("\n❌ Testing exception logging...")
    try:
        raise ValueError("Test exception for logging")
    except Exception as e:
        logger.exception("An error occurred during testing")
    
    # Get performance statistics
    print("\n📈 Performance Statistics:")
    perf_stats = advanced_logger.get_performance_stats()
    for operation, stats in perf_stats.items():
        print(f"   {operation}: {stats['avg_time']*1000:.1f}ms avg ({stats['count']} calls)")
    
    # Get log statistics
    print("\n📊 Log Statistics:")
    log_stats = advanced_logger.get_log_stats()
    print(f"   Total logs: {log_stats['total_logs']}")
    print(f"   Errors: {log_stats['error_count']}")
    print(f"   Warnings: {log_stats['warning_count']}")
    
    for level, count in log_stats['logs_by_level'].items():
        print(f"   {level}: {count}")
    
    print(f"\n🎯 Advanced logging system ready!")
    print(f"📋 Features:")
    print(f"   • Structured logging with context")
    print(f"   • Performance timing")
    print(f"   • Audit logging")
    print(f"   • Multiple output formats")
    print(f"   • Correlation IDs")
    print(f"   • Debug utilities")
    print(f"   • Statistics tracking")