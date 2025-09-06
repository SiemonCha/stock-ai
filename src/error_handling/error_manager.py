"""
Robust Error Handling and Recovery System
Professional-grade error management with automatic recovery mechanisms
"""

import asyncio
import json
import time
import traceback
import logging
import threading
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from contextlib import contextmanager
import warnings
warnings.filterwarnings('ignore')

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for classification"""
    DATA_SOURCE = "data_source"
    NETWORK = "network"
    COMPUTATION = "computation"
    STORAGE = "storage"
    VALIDATION = "validation"
    EXTERNAL_API = "external_api"
    SYSTEM = "system"
    USER_INPUT = "user_input"

class RecoveryStrategy(Enum):
    """Recovery strategies"""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    RESTART = "restart"
    MANUAL = "manual"

@dataclass
class ErrorContext:
    """Error context information"""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    component: str
    function_name: str
    error_message: str
    stack_trace: str
    recovery_strategy: RecoveryStrategy
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class RecoveryAction:
    """Recovery action definition"""
    action_type: RecoveryStrategy
    handler: Callable
    max_attempts: int = 3
    delay_seconds: float = 1.0
    exponential_backoff: bool = True
    conditions: Dict[str, Any] = field(default_factory=dict)

class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        self.lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self.lock:
            if self.state == "open":
                if self._should_attempt_reset():
                    self.state = "half_open"
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except Exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if self.last_failure_time is None:
            return True
        
        return time.time() - self.last_failure_time > self.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution"""
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self):
        """Handle failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"

class RetryManager:
    """Advanced retry mechanism with exponential backoff"""
    
    @staticmethod
    async def retry_async(
        func: Callable,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        exponential_backoff: bool = True,
        max_delay: float = 60.0,
        retry_exceptions: Tuple = (Exception,),
        *args, **kwargs
    ):
        """Async retry with exponential backoff"""
        
        last_exception = None
        
        for attempt in range(max_attempts):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except retry_exceptions as e:
                last_exception = e
                
                if attempt < max_attempts - 1:  # Not the last attempt
                    if exponential_backoff:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                    else:
                        delay = base_delay
                    
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {str(e)}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {max_attempts} attempts failed")
        
        raise last_exception
    
    @staticmethod
    def retry_sync(
        func: Callable,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        exponential_backoff: bool = True,
        max_delay: float = 60.0,
        retry_exceptions: Tuple = (Exception,),
        *args, **kwargs
    ):
        """Synchronous retry with exponential backoff"""
        
        last_exception = None
        
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
                
            except retry_exceptions as e:
                last_exception = e
                
                if attempt < max_attempts - 1:
                    if exponential_backoff:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                    else:
                        delay = base_delay
                    
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {str(e)}")
                    time.sleep(delay)
                else:
                    logger.error(f"All {max_attempts} attempts failed")
        
        raise last_exception

class ErrorRecoveryManager:
    """Comprehensive error recovery management system"""
    
    def __init__(self):
        self.error_history: List[ErrorContext] = []
        self.recovery_handlers: Dict[str, RecoveryAction] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.fallback_handlers: Dict[str, Callable] = {}
        
        # Error patterns and frequencies
        self.error_patterns: Dict[str, Dict] = {}
        self.component_health: Dict[str, Dict] = {}
        
        # Performance metrics
        self.recovery_stats = {
            'total_errors': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'average_recovery_time': 0.0
        }
        
        # Background monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
    
    def register_recovery_handler(self, error_pattern: str, recovery_action: RecoveryAction):
        """Register recovery handler for specific error patterns"""
        self.recovery_handlers[error_pattern] = recovery_action
        logger.info(f"Registered recovery handler for pattern: {error_pattern}")
    
    def register_fallback_handler(self, component: str, handler: Callable):
        """Register fallback handler for component"""
        self.fallback_handlers[component] = handler
        logger.info(f"Registered fallback handler for component: {component}")
    
    def get_circuit_breaker(self, component: str) -> CircuitBreaker:
        """Get or create circuit breaker for component"""
        if component not in self.circuit_breakers:
            self.circuit_breakers[component] = CircuitBreaker()
        return self.circuit_breakers[component]
    
    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Handle error with automatic recovery"""
        
        # Create error context
        error_context = self._create_error_context(error, context)
        self.error_history.append(error_context)
        self.recovery_stats['total_errors'] += 1
        
        # Log error
        logger.error(f"Error in {error_context.component}.{error_context.function_name}: {error_context.error_message}")
        
        # Update component health
        self._update_component_health(error_context)
        
        # Attempt recovery
        recovery_start = time.time()
        recovery_successful = await self._attempt_recovery(error_context)
        recovery_time = time.time() - recovery_start
        
        # Update statistics
        if recovery_successful:
            self.recovery_stats['successful_recoveries'] += 1
            error_context.resolved = True
            error_context.resolution_time = datetime.now()
            logger.info(f"Successfully recovered from error {error_context.error_id} in {recovery_time:.2f}s")
        else:
            self.recovery_stats['failed_recoveries'] += 1
            logger.error(f"Failed to recover from error {error_context.error_id}")
        
        # Update average recovery time
        total_recoveries = self.recovery_stats['successful_recoveries'] + self.recovery_stats['failed_recoveries']
        current_avg = self.recovery_stats['average_recovery_time']
        self.recovery_stats['average_recovery_time'] = (
            (current_avg * (total_recoveries - 1) + recovery_time) / total_recoveries
        )
        
        return recovery_successful
    
    def _create_error_context(self, error: Exception, context: Dict[str, Any]) -> ErrorContext:
        """Create error context from exception and context"""
        
        error_id = f"err_{int(time.time())}_{hash(str(error)) % 10000:04d}"
        
        # Classify error
        severity = self._classify_severity(error)
        category = self._classify_category(error, context)
        recovery_strategy = self._determine_recovery_strategy(error, category)
        
        return ErrorContext(
            error_id=error_id,
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            component=context.get('component', 'unknown'),
            function_name=context.get('function', 'unknown'),
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            recovery_strategy=recovery_strategy,
            metadata=context
        )
    
    def _classify_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity"""
        
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Critical errors
        critical_patterns = [
            'system', 'memory', 'disk', 'database corruption',
            'authentication failed', 'security'
        ]
        
        if any(pattern in error_message for pattern in critical_patterns):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        high_patterns = [
            'timeout', 'connection', 'network', 'service unavailable',
            'permission denied', 'file not found'
        ]
        
        if any(pattern in error_message for pattern in high_patterns):
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        medium_patterns = [
            'validation', 'parsing', 'format', 'invalid'
        ]
        
        if any(pattern in error_message for pattern in medium_patterns):
            return ErrorSeverity.MEDIUM
        
        return ErrorSeverity.LOW
    
    def _classify_category(self, error: Exception, context: Dict[str, Any]) -> ErrorCategory:
        """Classify error category"""
        
        error_message = str(error).lower()
        component = context.get('component', '').lower()
        
        # Network errors
        if any(keyword in error_message for keyword in ['connection', 'timeout', 'network', 'socket']):
            return ErrorCategory.NETWORK
        
        # Data source errors
        if any(keyword in error_message for keyword in ['data', 'source', 'api', 'fetch']):
            return ErrorCategory.DATA_SOURCE
        
        # Storage errors
        if any(keyword in error_message for keyword in ['database', 'storage', 'disk', 'file']):
            return ErrorCategory.STORAGE
        
        # Validation errors
        if any(keyword in error_message for keyword in ['validation', 'invalid', 'format']):
            return ErrorCategory.VALIDATION
        
        # External API errors
        if any(keyword in error_message for keyword in ['api', 'service', 'external']):
            return ErrorCategory.EXTERNAL_API
        
        # Computation errors
        if any(keyword in error_message for keyword in ['calculation', 'computation', 'math']):
            return ErrorCategory.COMPUTATION
        
        return ErrorCategory.SYSTEM
    
    def _determine_recovery_strategy(self, error: Exception, category: ErrorCategory) -> RecoveryStrategy:
        """Determine appropriate recovery strategy"""
        
        strategy_map = {
            ErrorCategory.NETWORK: RecoveryStrategy.RETRY,
            ErrorCategory.DATA_SOURCE: RecoveryStrategy.FALLBACK,
            ErrorCategory.EXTERNAL_API: RecoveryStrategy.RETRY,
            ErrorCategory.VALIDATION: RecoveryStrategy.SKIP,
            ErrorCategory.COMPUTATION: RecoveryStrategy.RETRY,
            ErrorCategory.STORAGE: RecoveryStrategy.FALLBACK,
            ErrorCategory.SYSTEM: RecoveryStrategy.RESTART,
            ErrorCategory.USER_INPUT: RecoveryStrategy.MANUAL
        }
        
        return strategy_map.get(category, RecoveryStrategy.MANUAL)
    
    async def _attempt_recovery(self, error_context: ErrorContext) -> bool:
        """Attempt to recover from error"""
        
        strategy = error_context.recovery_strategy
        
        try:
            if strategy == RecoveryStrategy.RETRY:
                return await self._retry_recovery(error_context)
            
            elif strategy == RecoveryStrategy.FALLBACK:
                return await self._fallback_recovery(error_context)
            
            elif strategy == RecoveryStrategy.SKIP:
                return await self._skip_recovery(error_context)
            
            elif strategy == RecoveryStrategy.RESTART:
                return await self._restart_recovery(error_context)
            
            else:  # MANUAL
                return await self._manual_recovery(error_context)
                
        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
            return False
    
    async def _retry_recovery(self, error_context: ErrorContext) -> bool:
        """Retry-based recovery"""
        
        # Check if we've exceeded max retries
        if error_context.retry_count >= error_context.max_retries:
            logger.warning(f"Max retries exceeded for {error_context.error_id}")
            return False
        
        # Find matching recovery handler
        handler = self._find_recovery_handler(error_context)
        if handler:
            try:
                error_context.retry_count += 1
                await asyncio.sleep(handler.delay_seconds * (2 ** (error_context.retry_count - 1)))
                
                result = await handler.handler(error_context)
                return result is not False
                
            except Exception as e:
                logger.error(f"Retry recovery failed: {e}")
                return False
        
        return False
    
    async def _fallback_recovery(self, error_context: ErrorContext) -> bool:
        """Fallback-based recovery"""
        
        component = error_context.component
        
        if component in self.fallback_handlers:
            try:
                fallback_handler = self.fallback_handlers[component]
                
                if asyncio.iscoroutinefunction(fallback_handler):
                    result = await fallback_handler(error_context)
                else:
                    result = fallback_handler(error_context)
                
                return result is not False
                
            except Exception as e:
                logger.error(f"Fallback recovery failed: {e}")
                return False
        
        logger.warning(f"No fallback handler registered for component: {component}")
        return False
    
    async def _skip_recovery(self, error_context: ErrorContext) -> bool:
        """Skip-based recovery (mark as handled)"""
        logger.info(f"Skipping error {error_context.error_id} as per recovery strategy")
        return True
    
    async def _restart_recovery(self, error_context: ErrorContext) -> bool:
        """Restart-based recovery"""
        
        component = error_context.component
        
        # Log restart attempt
        logger.warning(f"Attempting to restart component: {component}")
        
        # In a real implementation, this would restart the specific component
        # For now, we'll simulate a restart delay
        await asyncio.sleep(2)
        
        # Reset component health
        if component in self.component_health:
            self.component_health[component]['last_restart'] = datetime.now()
            self.component_health[component]['restart_count'] += 1
        
        return True
    
    async def _manual_recovery(self, error_context: ErrorContext) -> bool:
        """Manual recovery (requires human intervention)"""
        logger.error(f"Manual intervention required for error {error_context.error_id}")
        
        # In a production system, this would trigger alerts/notifications
        # For now, we'll just log and return False
        return False
    
    def _find_recovery_handler(self, error_context: ErrorContext) -> Optional[RecoveryAction]:
        """Find matching recovery handler for error"""
        
        error_message = error_context.error_message.lower()
        
        for pattern, handler in self.recovery_handlers.items():
            if pattern.lower() in error_message:
                return handler
        
        return None
    
    def _update_component_health(self, error_context: ErrorContext):
        """Update component health metrics"""
        
        component = error_context.component
        
        if component not in self.component_health:
            self.component_health[component] = {
                'error_count': 0,
                'last_error': None,
                'error_rate': 0.0,
                'health_score': 1.0,
                'restart_count': 0,
                'last_restart': None
            }
        
        health = self.component_health[component]
        health['error_count'] += 1
        health['last_error'] = datetime.now()
        
        # Calculate error rate (errors per hour)
        if health['error_count'] > 1:
            time_window = (datetime.now() - health['last_restart']).total_seconds() / 3600 if health['last_restart'] else 1
            health['error_rate'] = health['error_count'] / max(time_window, 0.1)
        
        # Calculate health score (0.0 to 1.0)
        base_score = max(0.0, 1.0 - (health['error_rate'] / 10.0))  # Penalize high error rates
        severity_penalty = {
            ErrorSeverity.LOW: 0.05,
            ErrorSeverity.MEDIUM: 0.1,
            ErrorSeverity.HIGH: 0.2,
            ErrorSeverity.CRITICAL: 0.4
        }.get(error_context.severity, 0.1)
        
        health['health_score'] = max(0.0, base_score - severity_penalty)
    
    @contextmanager
    def error_boundary(self, component: str, function: str = "unknown"):
        """Context manager for error boundaries"""
        
        context = {
            'component': component,
            'function': function,
            'start_time': time.time()
        }
        
        try:
            yield context
            
        except Exception as e:
            # Handle error asynchronously in background
            error_future = asyncio.ensure_future(
                self.handle_error(e, context)
            )
            
            # Don't wait for recovery, re-raise the error
            raise e
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report"""
        
        total_errors = self.recovery_stats['total_errors']
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_health': self._calculate_overall_health(),
            'error_statistics': self.recovery_stats.copy(),
            'component_health': {
                comp: {
                    'health_score': health['health_score'],
                    'error_count': health['error_count'],
                    'error_rate': health['error_rate'],
                    'last_error': health['last_error'].isoformat() if health['last_error'] else None
                }
                for comp, health in self.component_health.items()
            },
            'recent_errors': [
                {
                    'error_id': err.error_id,
                    'component': err.component,
                    'severity': err.severity.value,
                    'category': err.category.value,
                    'resolved': err.resolved,
                    'timestamp': err.timestamp.isoformat()
                }
                for err in self.error_history[-10:]  # Last 10 errors
            ],
            'circuit_breaker_status': {
                comp: cb.state for comp, cb in self.circuit_breakers.items()
            }
        }
    
    def _calculate_overall_health(self) -> float:
        """Calculate overall system health score"""
        
        if not self.component_health:
            return 1.0
        
        # Weight components equally for now
        health_scores = [health['health_score'] for health in self.component_health.values()]
        return sum(health_scores) / len(health_scores)
    
    def start_monitoring(self):
        """Start background error monitoring"""
        
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Error recovery monitoring started")
    
    def stop_monitoring(self):
        """Stop background error monitoring"""
        
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Error recovery monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        
        while self.monitoring_active:
            try:
                # Check component health
                self._check_component_health()
                
                # Clean up old error history
                self._cleanup_error_history()
                
                # Reset circuit breakers if appropriate
                self._check_circuit_breakers()
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def _check_component_health(self):
        """Check and alert on component health issues"""
        
        for component, health in self.component_health.items():
            if health['health_score'] < 0.3:  # Health critical
                logger.warning(f"Component {component} health critical: {health['health_score']:.2f}")
            
            elif health['health_score'] < 0.7:  # Health degraded
                logger.info(f"Component {component} health degraded: {health['health_score']:.2f}")
    
    def _cleanup_error_history(self):
        """Clean up old error history"""
        
        # Keep only last 1000 errors and errors from last 24 hours
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        self.error_history = [
            error for error in self.error_history[-1000:]
            if error.timestamp >= cutoff_time
        ]
    
    def _check_circuit_breakers(self):
        """Check and potentially reset circuit breakers"""
        
        for component, cb in self.circuit_breakers.items():
            if cb.state == "open" and cb._should_attempt_reset():
                logger.info(f"Attempting to reset circuit breaker for {component}")
                cb.state = "half_open"

# Convenience decorators and utilities
def with_error_recovery(component: str, recovery_manager: ErrorRecoveryManager = None):
    """Decorator for automatic error recovery"""
    
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            manager = recovery_manager or _get_default_recovery_manager()
            
            with manager.error_boundary(component, func.__name__):
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            manager = recovery_manager or _get_default_recovery_manager()
            
            with manager.error_boundary(component, func.__name__):
                return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Global recovery manager instance
_default_recovery_manager = None

def _get_default_recovery_manager() -> ErrorRecoveryManager:
    """Get or create default recovery manager"""
    global _default_recovery_manager
    
    if _default_recovery_manager is None:
        _default_recovery_manager = ErrorRecoveryManager()
    
    return _default_recovery_manager

def get_recovery_manager() -> ErrorRecoveryManager:
    """Get the default recovery manager instance"""
    return _get_default_recovery_manager()

# Example usage and testing
if __name__ == "__main__":
    print("🔧 Error Handling and Recovery System")
    print("=" * 45)
    
    async def test_error_recovery():
        """Test error recovery system"""
        
        # Initialize recovery manager
        recovery_manager = ErrorRecoveryManager()
        
        # Register some recovery handlers
        async def retry_handler(error_context):
            print(f"Attempting retry for {error_context.error_id}")
            # Simulate successful retry
            return True
        
        def fallback_handler(error_context):
            print(f"Using fallback for {error_context.component}")
            return True
        
        recovery_manager.register_recovery_handler(
            "connection",
            RecoveryAction(RecoveryStrategy.RETRY, retry_handler, max_attempts=3)
        )
        
        recovery_manager.register_fallback_handler("data_source", fallback_handler)
        
        print("✅ Recovery handlers registered")
        
        # Test error handling
        test_errors = [
            (ConnectionError("Connection timeout"), {'component': 'data_collector', 'function': 'fetch_data'}),
            (ValueError("Invalid data format"), {'component': 'validator', 'function': 'validate_data'}),
            (Exception("Unknown system error"), {'component': 'predictor', 'function': 'predict'})
        ]
        
        for error, context in test_errors:
            print(f"\n🧪 Testing error: {type(error).__name__}: {error}")
            success = await recovery_manager.handle_error(error, context)
            print(f"   Recovery {'successful' if success else 'failed'}")
        
        # Generate health report
        print(f"\n📊 System Health Report:")
        health_report = recovery_manager.get_system_health_report()
        
        print(f"   Overall Health: {health_report['overall_health']:.2f}")
        print(f"   Total Errors: {health_report['error_statistics']['total_errors']}")
        print(f"   Successful Recoveries: {health_report['error_statistics']['successful_recoveries']}")
        print(f"   Failed Recoveries: {health_report['error_statistics']['failed_recoveries']}")
        
        print(f"\n📋 Component Health:")
        for component, health in health_report['component_health'].items():
            print(f"   {component}: {health['health_score']:.2f} (errors: {health['error_count']})")
        
        # Test circuit breaker
        print(f"\n🔌 Testing Circuit Breaker:")
        cb = recovery_manager.get_circuit_breaker("test_component")
        
        def failing_function():
            raise Exception("Simulated failure")
        
        # Trigger circuit breaker
        for i in range(7):  # More than failure threshold
            try:
                cb.call(failing_function)
            except Exception:
                pass
        
        print(f"   Circuit breaker state: {cb.state}")
        
        # Test retry manager
        print(f"\n🔄 Testing Retry Manager:")
        
        attempt_count = [0]
        
        def intermittent_failure():
            attempt_count[0] += 1
            if attempt_count[0] <= 2:
                raise Exception("Temporary failure")
            return "Success!"
        
        try:
            result = await RetryManager.retry_async(intermittent_failure, max_attempts=5)
            print(f"   Retry result: {result} (after {attempt_count[0]} attempts)")
        except Exception as e:
            print(f"   Retry failed: {e}")
    
    # Run test
    asyncio.run(test_error_recovery())
    
    print(f"\n🎯 Error handling system ready!")
    print(f"📋 Features:")
    print(f"   • Automatic error classification")
    print(f"   • Multiple recovery strategies")
    print(f"   • Circuit breaker pattern")
    print(f"   • Exponential backoff retry")
    print(f"   • Component health monitoring")
    print(f"   • System health reporting")