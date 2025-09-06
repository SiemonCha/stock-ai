"""
Production Health Checks and Monitoring System
Real-time system health monitoring with alerting and diagnostics
"""

import asyncio
import json
import time
import threading
import socket
import psutil
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import warnings
warnings.filterwarnings('ignore')

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health check status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"  
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"

class ComponentType(Enum):
    """Types of components to monitor"""
    API_ENDPOINT = "api_endpoint"
    DATABASE = "database"
    EXTERNAL_SERVICE = "external_service"
    SYSTEM_RESOURCE = "system_resource"
    PROCESS = "process"
    NETWORK = "network"
    STORAGE = "storage"

@dataclass
class HealthCheck:
    """Health check definition"""
    name: str
    component_type: ComponentType
    check_function: Callable
    interval_seconds: int = 60
    timeout_seconds: int = 30
    retry_count: int = 3
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HealthResult:
    """Health check result"""
    check_name: str
    status: HealthStatus
    response_time_ms: float
    timestamp: datetime
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    process_count: int
    load_average: List[float]
    timestamp: datetime
    uptime_seconds: float

class HealthMonitor:
    """Comprehensive system health monitoring"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_results: Dict[str, List[HealthResult]] = {}
        self.system_metrics: List[SystemMetrics] = []
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_threads = []
        
        # Performance counters
        self.metrics = {
            'checks_performed': 0,
            'checks_failed': 0,
            'avg_response_time': 0.0,
            'system_uptime': 0.0
        }
        
        # Alert thresholds
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'cpu_percent': 80,
            'memory_percent': 85,
            'disk_percent': 90,
            'response_time_ms': 5000,
            'consecutive_failures': 3
        })
        
        # Initialize Prometheus metrics if available
        self.prometheus_metrics = None
        if PROMETHEUS_AVAILABLE and self.config.get('enable_prometheus', False):
            self._setup_prometheus_metrics()
        
        logger.info("Health monitor initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default monitoring configuration"""
        return {
            'metrics_retention_hours': 24,
            'system_metrics_interval': 30,
            'enable_prometheus': False,
            'prometheus_port': 8000,
            'alert_thresholds': {
                'cpu_percent': 80,
                'memory_percent': 85,
                'disk_percent': 90,
                'response_time_ms': 5000,
                'consecutive_failures': 3
            },
            'notification_channels': []
        }
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        self.prometheus_metrics = {
            'health_check_duration': Histogram(
                'health_check_duration_seconds',
                'Time spent on health checks',
                ['check_name', 'status']
            ),
            'health_check_status': Gauge(
                'health_check_status',
                'Health check status (1=healthy, 0=unhealthy)',
                ['check_name', 'component_type']
            ),
            'system_cpu_percent': Gauge(
                'system_cpu_percent',
                'CPU usage percentage'
            ),
            'system_memory_percent': Gauge(
                'system_memory_percent',
                'Memory usage percentage'
            ),
            'system_disk_percent': Gauge(
                'system_disk_percent', 
                'Disk usage percentage'
            )
        }
        
        # Start Prometheus HTTP server
        start_http_server(self.config['prometheus_port'])
        logger.info(f"Prometheus metrics server started on port {self.config['prometheus_port']}")
    
    def register_health_check(self, health_check: HealthCheck):
        """Register a new health check"""
        self.health_checks[health_check.name] = health_check
        self.health_results[health_check.name] = []
        logger.info(f"Registered health check: {health_check.name}")
    
    def unregister_health_check(self, check_name: str):
        """Unregister a health check"""
        if check_name in self.health_checks:
            del self.health_checks[check_name]
            del self.health_results[check_name]
            logger.info(f"Unregistered health check: {check_name}")
    
    async def run_health_check(self, check_name: str) -> HealthResult:
        """Run a single health check"""
        
        if check_name not in self.health_checks:
            raise ValueError(f"Health check not found: {check_name}")
        
        health_check = self.health_checks[check_name]
        
        if not health_check.enabled:
            return HealthResult(
                check_name=check_name,
                status=HealthStatus.HEALTHY,
                response_time_ms=0.0,
                timestamp=datetime.now(),
                message="Check disabled"
            )
        
        start_time = time.time()
        result = None
        
        # Retry logic
        for attempt in range(health_check.retry_count):
            try:
                # Run the health check with timeout
                if asyncio.iscoroutinefunction(health_check.check_function):
                    check_result = await asyncio.wait_for(
                        health_check.check_function(),
                        timeout=health_check.timeout_seconds
                    )
                else:
                    check_result = health_check.check_function()
                
                response_time = (time.time() - start_time) * 1000  # Convert to ms
                
                # Process check result
                if isinstance(check_result, dict):
                    status = HealthStatus(check_result.get('status', 'healthy'))
                    message = check_result.get('message', 'OK')
                    details = check_result.get('details', {})
                    error = check_result.get('error')
                elif isinstance(check_result, bool):
                    status = HealthStatus.HEALTHY if check_result else HealthStatus.UNHEALTHY
                    message = "OK" if check_result else "Check failed"
                    details = {}
                    error = None
                else:
                    status = HealthStatus.HEALTHY
                    message = str(check_result)
                    details = {}
                    error = None
                
                result = HealthResult(
                    check_name=check_name,
                    status=status,
                    response_time_ms=response_time,
                    timestamp=datetime.now(),
                    message=message,
                    details=details,
                    error=error
                )
                
                break  # Success, no need to retry
                
            except asyncio.TimeoutError:
                error_msg = f"Health check timed out after {health_check.timeout_seconds}s"
                logger.warning(f"{check_name}: {error_msg}")
                
                if attempt == health_check.retry_count - 1:  # Last attempt
                    result = HealthResult(
                        check_name=check_name,
                        status=HealthStatus.CRITICAL,
                        response_time_ms=(time.time() - start_time) * 1000,
                        timestamp=datetime.now(),
                        message="Timeout",
                        error=error_msg
                    )
                
            except Exception as e:
                error_msg = f"Health check failed: {str(e)}"
                logger.error(f"{check_name}: {error_msg}")
                
                if attempt == health_check.retry_count - 1:  # Last attempt
                    result = HealthResult(
                        check_name=check_name,
                        status=HealthStatus.CRITICAL,
                        response_time_ms=(time.time() - start_time) * 1000,
                        timestamp=datetime.now(),
                        message="Error",
                        error=error_msg
                    )
                else:
                    await asyncio.sleep(1)  # Brief delay before retry
        
        # Store result
        self.health_results[check_name].append(result)
        self._trim_health_results(check_name)
        
        # Update metrics
        self.metrics['checks_performed'] += 1
        if result.status != HealthStatus.HEALTHY:
            self.metrics['checks_failed'] += 1
        
        # Update Prometheus metrics
        if self.prometheus_metrics:
            self.prometheus_metrics['health_check_duration'].labels(
                check_name=check_name,
                status=result.status.value
            ).observe(result.response_time_ms / 1000)
            
            status_value = 1 if result.status == HealthStatus.HEALTHY else 0
            self.prometheus_metrics['health_check_status'].labels(
                check_name=check_name,
                component_type=health_check.component_type.value
            ).set(status_value)
        
        return result
    
    async def run_all_health_checks(self) -> Dict[str, HealthResult]:
        """Run all registered health checks"""
        
        tasks = []
        for check_name, health_check in self.health_checks.items():
            if health_check.enabled:
                task = asyncio.create_task(self.run_health_check(check_name))
                tasks.append((check_name, task))
        
        results = {}
        for check_name, task in tasks:
            try:
                result = await task
                results[check_name] = result
            except Exception as e:
                logger.error(f"Failed to run health check {check_name}: {e}")
                results[check_name] = HealthResult(
                    check_name=check_name,
                    status=HealthStatus.CRITICAL,
                    response_time_ms=0.0,
                    timestamp=datetime.now(),
                    message="Internal error",
                    error=str(e)
                )
        
        return results
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Network metrics
            network_io = psutil.net_io_counters()._asdict()
            
            # Process metrics
            process_count = len(psutil.pids())
            
            # Load average (Unix systems)
            try:
                load_average = list(psutil.getloadavg())
            except AttributeError:
                load_average = [0.0, 0.0, 0.0]  # Windows doesn't have load average
            
            # System uptime
            boot_time = psutil.boot_time()
            uptime_seconds = time.time() - boot_time
            
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                network_io=network_io,
                process_count=process_count,
                load_average=load_average,
                timestamp=datetime.now(),
                uptime_seconds=uptime_seconds
            )
            
            # Store metrics
            self.system_metrics.append(metrics)
            self._trim_system_metrics()
            
            # Update Prometheus metrics
            if self.prometheus_metrics:
                self.prometheus_metrics['system_cpu_percent'].set(cpu_percent)
                self.prometheus_metrics['system_memory_percent'].set(memory_percent)
                self.prometheus_metrics['system_disk_percent'].set(disk_percent)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_percent=0.0,
                network_io={},
                process_count=0,
                load_average=[0.0, 0.0, 0.0],
                timestamp=datetime.now(),
                uptime_seconds=0.0
            )
    
    def _trim_health_results(self, check_name: str):
        """Trim old health results to maintain retention policy"""
        
        retention_hours = self.config['metrics_retention_hours']
        cutoff_time = datetime.now() - timedelta(hours=retention_hours)
        
        self.health_results[check_name] = [
            result for result in self.health_results[check_name]
            if result.timestamp >= cutoff_time
        ]
    
    def _trim_system_metrics(self):
        """Trim old system metrics"""
        
        retention_hours = self.config['metrics_retention_hours']
        cutoff_time = datetime.now() - timedelta(hours=retention_hours)
        
        self.system_metrics = [
            metrics for metrics in self.system_metrics
            if metrics.timestamp >= cutoff_time
        ]
    
    def start_monitoring(self):
        """Start continuous health monitoring"""
        
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        # Start health check monitoring thread
        health_thread = threading.Thread(
            target=self._health_monitoring_loop,
            daemon=True
        )
        health_thread.start()
        self.monitoring_threads.append(health_thread)
        
        # Start system metrics monitoring thread
        metrics_thread = threading.Thread(
            target=self._system_metrics_loop,
            daemon=True
        )
        metrics_thread.start()
        self.monitoring_threads.append(metrics_thread)
        
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        
        self.monitoring_active = False
        
        # Wait for threads to finish
        for thread in self.monitoring_threads:
            thread.join(timeout=5)
        
        self.monitoring_threads.clear()
        logger.info("Health monitoring stopped")
    
    def _health_monitoring_loop(self):
        """Health check monitoring loop"""
        
        while self.monitoring_active:
            try:
                # Run all health checks
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                results = loop.run_until_complete(self.run_all_health_checks())
                
                # Check for alerts
                self._check_health_alerts(results)
                
                loop.close()
                
                # Wait for next interval (use minimum interval from all checks)
                min_interval = min(
                    check.interval_seconds for check in self.health_checks.values()
                    if check.enabled
                ) if self.health_checks else 60
                
                time.sleep(min_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring loop error: {e}")
                time.sleep(60)
    
    def _system_metrics_loop(self):
        """System metrics monitoring loop"""
        
        interval = self.config['system_metrics_interval']
        
        while self.monitoring_active:
            try:
                metrics = self.collect_system_metrics()
                self._check_system_alerts(metrics)
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"System metrics loop error: {e}")
                time.sleep(interval)
    
    def _check_health_alerts(self, results: Dict[str, HealthResult]):
        """Check health results for alert conditions"""
        
        for check_name, result in results.items():
            # Check response time threshold
            if result.response_time_ms > self.alert_thresholds['response_time_ms']:
                self._trigger_alert(
                    'high_response_time',
                    f"Health check {check_name} response time is high: {result.response_time_ms:.1f}ms",
                    {'check_name': check_name, 'response_time': result.response_time_ms}
                )
            
            # Check for consecutive failures
            if result.status != HealthStatus.HEALTHY:
                recent_results = self.health_results[check_name][-self.alert_thresholds['consecutive_failures']:]
                
                if len(recent_results) >= self.alert_thresholds['consecutive_failures']:
                    if all(r.status != HealthStatus.HEALTHY for r in recent_results):
                        self._trigger_alert(
                            'consecutive_failures',
                            f"Health check {check_name} has {len(recent_results)} consecutive failures",
                            {'check_name': check_name, 'failure_count': len(recent_results)}
                        )
    
    def _check_system_alerts(self, metrics: SystemMetrics):
        """Check system metrics for alert conditions"""
        
        # CPU usage alert
        if metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
            self._trigger_alert(
                'high_cpu_usage',
                f"High CPU usage: {metrics.cpu_percent:.1f}%",
                {'cpu_percent': metrics.cpu_percent}
            )
        
        # Memory usage alert
        if metrics.memory_percent > self.alert_thresholds['memory_percent']:
            self._trigger_alert(
                'high_memory_usage',
                f"High memory usage: {metrics.memory_percent:.1f}%",
                {'memory_percent': metrics.memory_percent}
            )
        
        # Disk usage alert
        if metrics.disk_percent > self.alert_thresholds['disk_percent']:
            self._trigger_alert(
                'high_disk_usage',
                f"High disk usage: {metrics.disk_percent:.1f}%",
                {'disk_percent': metrics.disk_percent}
            )
    
    def _trigger_alert(self, alert_type: str, message: str, details: Dict[str, Any]):
        """Trigger an alert"""
        
        alert = {
            'type': alert_type,
            'message': message,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.warning(f"ALERT: {message}")
        
        # Here you would integrate with notification systems
        # (email, Slack, PagerDuty, etc.)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current overall health status"""
        
        # Get latest results for each check
        latest_results = {}
        overall_status = HealthStatus.HEALTHY
        
        for check_name, results in self.health_results.items():
            if results:
                latest_result = results[-1]
                latest_results[check_name] = {
                    'status': latest_result.status.value,
                    'message': latest_result.message,
                    'response_time_ms': latest_result.response_time_ms,
                    'timestamp': latest_result.timestamp.isoformat()
                }
                
                # Determine worst status
                if latest_result.status == HealthStatus.CRITICAL:
                    overall_status = HealthStatus.CRITICAL
                elif latest_result.status == HealthStatus.UNHEALTHY and overall_status != HealthStatus.CRITICAL:
                    overall_status = HealthStatus.UNHEALTHY
                elif latest_result.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
        
        # Get latest system metrics
        latest_system_metrics = {}
        if self.system_metrics:
            latest_metrics = self.system_metrics[-1]
            latest_system_metrics = {
                'cpu_percent': latest_metrics.cpu_percent,
                'memory_percent': latest_metrics.memory_percent,
                'disk_percent': latest_metrics.disk_percent,
                'process_count': latest_metrics.process_count,
                'uptime_seconds': latest_metrics.uptime_seconds,
                'timestamp': latest_metrics.timestamp.isoformat()
            }
        
        return {
            'overall_status': overall_status.value,
            'timestamp': datetime.now().isoformat(),
            'health_checks': latest_results,
            'system_metrics': latest_system_metrics,
            'monitoring_stats': {
                'checks_performed': self.metrics['checks_performed'],
                'checks_failed': self.metrics['checks_failed'],
                'failure_rate': self.metrics['checks_failed'] / max(self.metrics['checks_performed'], 1)
            }
        }
    
    def generate_health_report(self) -> str:
        """Generate comprehensive health report"""
        
        status = self.get_health_status()
        
        report_lines = [
            "🏥 SYSTEM HEALTH REPORT",
            "=" * 30,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Overall Status: {self._get_status_emoji(status['overall_status'])} {status['overall_status'].upper()}",
            ""
        ]
        
        # Health checks section
        if status['health_checks']:
            report_lines.extend([
                "🔍 HEALTH CHECKS",
                "-" * 15
            ])
            
            for check_name, check_data in status['health_checks'].items():
                emoji = self._get_status_emoji(check_data['status'])
                report_lines.append(
                    f"{emoji} {check_name}: {check_data['status']} "
                    f"({check_data['response_time_ms']:.1f}ms)"
                )
        
        # System metrics section
        if status['system_metrics']:
            metrics = status['system_metrics']
            report_lines.extend([
                "",
                "📊 SYSTEM METRICS",
                "-" * 16,
                f"CPU Usage: {metrics['cpu_percent']:.1f}%",
                f"Memory Usage: {metrics['memory_percent']:.1f}%",
                f"Disk Usage: {metrics['disk_percent']:.1f}%",
                f"Process Count: {metrics['process_count']}",
                f"Uptime: {metrics['uptime_seconds']:.0f}s"
            ])
        
        # Monitoring statistics
        stats = status['monitoring_stats']
        report_lines.extend([
            "",
            "📈 MONITORING STATS",
            "-" * 18,
            f"Checks Performed: {stats['checks_performed']}",
            f"Checks Failed: {stats['checks_failed']}",
            f"Failure Rate: {stats['failure_rate']:.1%}"
        ])
        
        return "\n".join(report_lines)
    
    def _get_status_emoji(self, status: str) -> str:
        """Get emoji for status"""
        emoji_map = {
            'healthy': '✅',
            'degraded': '⚠️',
            'unhealthy': '❌',
            'critical': '🚨'
        }
        return emoji_map.get(status.lower(), '❓')

# Built-in health checks
class BuiltInHealthChecks:
    """Collection of built-in health checks"""
    
    @staticmethod
    def http_endpoint_check(url: str, timeout: int = 10):
        """HTTP endpoint health check"""
        async def check():
            if not REQUESTS_AVAILABLE:
                return {
                    'status': 'unhealthy',
                    'message': 'Requests library not available'
                }
            
            try:
                response = requests.get(url, timeout=timeout)
                if response.status_code == 200:
                    return {
                        'status': 'healthy',
                        'message': f'HTTP {response.status_code}',
                        'details': {'status_code': response.status_code}
                    }
                else:
                    return {
                        'status': 'unhealthy',
                        'message': f'HTTP {response.status_code}',
                        'details': {'status_code': response.status_code}
                    }
            except Exception as e:
                return {
                    'status': 'critical',
                    'message': 'Connection failed',
                    'error': str(e)
                }
        
        return check
    
    @staticmethod
    def database_check(connection_string: str):
        """Database connectivity check"""
        def check():
            # This is a placeholder - implement based on your database
            try:
                # Simulate database check
                time.sleep(0.1)  # Simulate connection time
                return {
                    'status': 'healthy',
                    'message': 'Database connection OK'
                }
            except Exception as e:
                return {
                    'status': 'critical',
                    'message': 'Database connection failed',
                    'error': str(e)
                }
        
        return check
    
    @staticmethod
    def redis_check(host: str = 'localhost', port: int = 6379):
        """Redis connectivity check"""
        def check():
            if not REDIS_AVAILABLE:
                return {
                    'status': 'unhealthy',
                    'message': 'Redis library not available'
                }
            
            try:
                r = redis.Redis(host=host, port=port, socket_timeout=5)
                r.ping()
                return {
                    'status': 'healthy',
                    'message': 'Redis connection OK'
                }
            except Exception as e:
                return {
                    'status': 'critical',
                    'message': 'Redis connection failed',
                    'error': str(e)
                }
        
        return check
    
    @staticmethod
    def disk_space_check(path: str = '/', threshold_percent: float = 90):
        """Disk space health check"""
        def check():
            try:
                disk_usage = psutil.disk_usage(path)
                percent_used = (disk_usage.used / disk_usage.total) * 100
                
                if percent_used < threshold_percent:
                    return {
                        'status': 'healthy',
                        'message': f'Disk usage: {percent_used:.1f}%',
                        'details': {'percent_used': percent_used}
                    }
                else:
                    return {
                        'status': 'critical',
                        'message': f'Disk usage critical: {percent_used:.1f}%',
                        'details': {'percent_used': percent_used}
                    }
            except Exception as e:
                return {
                    'status': 'critical',
                    'message': 'Disk check failed',
                    'error': str(e)
                }
        
        return check

# Example usage and testing
if __name__ == "__main__":
    print("🏥 Health Monitoring System")
    print("=" * 32)
    
    async def test_health_monitoring():
        """Test health monitoring system"""
        
        # Initialize monitor
        config = {
            'metrics_retention_hours': 1,
            'system_metrics_interval': 5,
            'enable_prometheus': False
        }
        
        monitor = HealthMonitor(config)
        
        # Register built-in health checks
        monitor.register_health_check(HealthCheck(
            name="disk_space",
            component_type=ComponentType.STORAGE,
            check_function=BuiltInHealthChecks.disk_space_check('/', 95),
            interval_seconds=30
        ))
        
        monitor.register_health_check(HealthCheck(
            name="http_example",
            component_type=ComponentType.API_ENDPOINT,
            check_function=BuiltInHealthChecks.http_endpoint_check("https://httpbin.org/status/200"),
            interval_seconds=60
        ))
        
        # Custom health check
        def custom_check():
            return {
                'status': 'healthy',
                'message': 'Custom check passed',
                'details': {'test_value': 42}
            }
        
        monitor.register_health_check(HealthCheck(
            name="custom_check",
            component_type=ComponentType.PROCESS,
            check_function=custom_check,
            interval_seconds=30
        ))
        
        print(f"✅ Registered {len(monitor.health_checks)} health checks")
        
        # Run all health checks once
        print(f"\n🔍 Running health checks...")
        results = await monitor.run_all_health_checks()
        
        for check_name, result in results.items():
            status_emoji = monitor._get_status_emoji(result.status.value)
            print(f"   {status_emoji} {check_name}: {result.status.value} ({result.response_time_ms:.1f}ms)")
        
        # Collect system metrics
        print(f"\n📊 Collecting system metrics...")
        system_metrics = monitor.collect_system_metrics()
        print(f"   CPU: {system_metrics.cpu_percent:.1f}%")
        print(f"   Memory: {system_metrics.memory_percent:.1f}%")
        print(f"   Disk: {system_metrics.disk_percent:.1f}%")
        print(f"   Processes: {system_metrics.process_count}")
        
        # Get health status
        print(f"\n💡 Health Status:")
        health_status = monitor.get_health_status()
        print(f"   Overall: {health_status['overall_status']}")
        
        # Generate health report
        print(f"\n📋 Health Report:")
        print("-" * 20)
        report = monitor.generate_health_report()
        print(report)
        
        # Test monitoring for a short period
        print(f"\n⏰ Starting monitoring (5 seconds)...")
        monitor.start_monitoring()
        
        await asyncio.sleep(5)
        
        monitor.stop_monitoring()
        print("   Monitoring stopped")
    
    # Run test
    asyncio.run(test_health_monitoring())
    
    print(f"\n🎯 Health monitoring system ready!")
    print(f"📋 Features:")
    print(f"   • Real-time health checks")
    print(f"   • System metrics collection")
    print(f"   • Built-in health checks")
    print(f"   • Prometheus integration")
    print(f"   • Alert management")
    print(f"   • Comprehensive reporting")