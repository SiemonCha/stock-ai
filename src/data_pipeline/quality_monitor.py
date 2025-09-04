"""
Data Quality Monitoring and Alerting System
Professional-grade monitoring with real-time alerts and comprehensive dashboards
"""

import asyncio
import smtplib
import json
import time
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import threading
from collections import deque, defaultdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

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
    from prometheus_client import Counter, Histogram, Gauge, start_http_server, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    """Types of quality metrics"""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    AVAILABILITY = "availability"
    PERFORMANCE = "performance"

class AlertChannel(Enum):
    """Alert delivery channels"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    LOG = "log"

@dataclass
class QualityMetric:
    """Quality metric definition"""
    name: str
    metric_type: MetricType
    value: float
    threshold: float
    symbol: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QualityAlert:
    """Quality alert definition"""
    alert_id: str
    level: AlertLevel
    message: str
    metric: QualityMetric
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AlertRule:
    """Alert rule definition"""
    name: str
    metric_type: MetricType
    condition: str  # 'less_than', 'greater_than', 'equals', etc.
    threshold: float
    severity: AlertLevel
    cooldown_minutes: int = 15
    enabled: bool = True
    symbols: Optional[List[str]] = None  # None means all symbols
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    check_interval_seconds: int = 60
    metric_retention_hours: int = 168  # 1 week
    alert_retention_hours: int = 720   # 30 days
    enable_email_alerts: bool = True
    enable_slack_alerts: bool = False
    enable_prometheus: bool = False
    email_config: Dict[str, str] = field(default_factory=dict)
    slack_config: Dict[str, str] = field(default_factory=dict)

class EmailAlerter:
    """Email alert handler"""
    
    def __init__(self, config: Dict[str, str]):
        self.smtp_server = config.get('smtp_server', 'localhost')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username')
        self.password = config.get('password')
        self.from_address = config.get('from_address')
        self.to_addresses = config.get('to_addresses', '').split(',')
        self.use_tls = config.get('use_tls', True)
        
    async def send_alert(self, alert: QualityAlert):
        """Send email alert"""
        
        try:
            # Create message
            msg = MimeMultipart()
            msg['From'] = self.from_address
            msg['To'] = ', '.join(self.to_addresses)
            msg['Subject'] = f"[{alert.level.value.upper()}] Data Quality Alert: {alert.metric.name}"
            
            # Email body
            body = f"""
Data Quality Alert

Alert ID: {alert.alert_id}
Level: {alert.level.value.upper()}
Timestamp: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Metric Details:
- Name: {alert.metric.name}
- Type: {alert.metric.metric_type.value}
- Current Value: {alert.metric.value:.4f}
- Threshold: {alert.metric.threshold:.4f}
- Symbol: {alert.metric.symbol or 'All'}

Message: {alert.message}

This is an automated alert from the Stock AI Data Quality Monitoring System.
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            
            if self.use_tls:
                server.starttls()
            
            if self.username and self.password:
                server.login(self.username, self.password)
            
            text = msg.as_string()
            server.sendmail(self.from_address, self.to_addresses, text)
            server.quit()
            
            logger.info(f"Email alert sent: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

class SlackAlerter:
    """Slack alert handler"""
    
    def __init__(self, config: Dict[str, str]):
        self.webhook_url = config.get('webhook_url')
        self.channel = config.get('channel', '#alerts')
        self.username = config.get('username', 'Stock AI Monitor')
        
    async def send_alert(self, alert: QualityAlert):
        """Send Slack alert"""
        
        if not REQUESTS_AVAILABLE or not self.webhook_url:
            logger.warning("Requests library or webhook URL not available for Slack alerts")
            return
        
        try:
            # Determine color based on alert level
            color_map = {
                AlertLevel.INFO: '#36a64f',      # Green
                AlertLevel.WARNING: '#ff9500',   # Orange
                AlertLevel.ERROR: '#ff4444',     # Red
                AlertLevel.CRITICAL: '#8B0000'   # Dark Red
            }
            
            color = color_map.get(alert.level, '#36a64f')
            
            # Create Slack message
            payload = {
                'channel': self.channel,
                'username': self.username,
                'icon_emoji': ':warning:',
                'attachments': [
                    {
                        'color': color,
                        'title': f'{alert.level.value.upper()}: {alert.metric.name}',
                        'text': alert.message,
                        'fields': [
                            {
                                'title': 'Metric Type',
                                'value': alert.metric.metric_type.value,
                                'short': True
                            },
                            {
                                'title': 'Current Value',
                                'value': f'{alert.metric.value:.4f}',
                                'short': True
                            },
                            {
                                'title': 'Threshold',
                                'value': f'{alert.metric.threshold:.4f}',
                                'short': True
                            },
                            {
                                'title': 'Symbol',
                                'value': alert.metric.symbol or 'All',
                                'short': True
                            }
                        ],
                        'timestamp': int(alert.timestamp.timestamp())
                    }
                ]
            }
            
            # Send to Slack
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Slack alert sent: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

class PrometheusExporter:
    """Prometheus metrics exporter"""
    
    def __init__(self, port: int = 8000):
        self.port = port
        self.registry = CollectorRegistry()
        
        # Create metrics
        self.quality_metrics = Gauge(
            'data_quality_score',
            'Data quality score',
            ['metric_type', 'symbol'],
            registry=self.registry
        )
        
        self.alert_counter = Counter(
            'data_quality_alerts_total',
            'Total data quality alerts',
            ['level', 'metric_type'],
            registry=self.registry
        )
        
        self.processing_time = Histogram(
            'data_quality_check_duration_seconds',
            'Time spent on data quality checks',
            registry=self.registry
        )
        
        # Start HTTP server
        if PROMETHEUS_AVAILABLE:
            start_http_server(port, registry=self.registry)
            logger.info(f"Prometheus metrics server started on port {port}")
    
    def record_metric(self, metric: QualityMetric):
        """Record quality metric"""
        if PROMETHEUS_AVAILABLE:
            self.quality_metrics.labels(
                metric_type=metric.metric_type.value,
                symbol=metric.symbol or 'all'
            ).set(metric.value)
    
    def record_alert(self, alert: QualityAlert):
        """Record alert"""
        if PROMETHEUS_AVAILABLE:
            self.alert_counter.labels(
                level=alert.level.value,
                metric_type=alert.metric.metric_type.value
            ).inc()

class DataQualityMonitor:
    """
    Comprehensive data quality monitoring system
    """
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alerters: Dict[AlertChannel, Any] = {}
        
        # Metrics storage (in-memory with configurable retention)
        self.metrics_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=int(config.metric_retention_hours * 3600 / config.check_interval_seconds))
        )
        
        # Alert storage
        self.active_alerts: Dict[str, QualityAlert] = {}
        self.alert_history: deque = deque(
            maxlen=int(config.alert_retention_hours * 3600 / config.check_interval_seconds)
        )
        
        # Alert cooldowns (to prevent spam)
        self.alert_cooldowns: Dict[str, datetime] = {}
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Performance metrics
        self.monitoring_stats = {
            'checks_performed': 0,
            'metrics_collected': 0,
            'alerts_triggered': 0,
            'avg_check_time': 0.0
        }
        
        # Initialize alerters
        self._initialize_alerters()
        
        # Initialize Prometheus exporter if enabled
        self.prometheus_exporter = None
        if config.enable_prometheus and PROMETHEUS_AVAILABLE:
            self.prometheus_exporter = PrometheusExporter()
        
        logger.info("DataQualityMonitor initialized")
    
    def _initialize_alerters(self):
        """Initialize alert handlers"""
        
        if self.config.enable_email_alerts and self.config.email_config:
            self.alerters[AlertChannel.EMAIL] = EmailAlerter(self.config.email_config)
            
        if self.config.enable_slack_alerts and self.config.slack_config:
            self.alerters[AlertChannel.SLACK] = SlackAlerter(self.config.slack_config)
    
    def add_alert_rule(self, rule: AlertRule):
        """Add alert rule"""
        self.alert_rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """Remove alert rule"""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
    
    async def record_metric(self, metric: QualityMetric):
        """Record a quality metric"""
        
        # Store in history
        metric_key = f"{metric.metric_type.value}_{metric.symbol or 'all'}"
        self.metrics_history[metric_key].append(metric)
        
        self.monitoring_stats['metrics_collected'] += 1
        
        # Export to Prometheus if enabled
        if self.prometheus_exporter:
            self.prometheus_exporter.record_metric(metric)
        
        # Check alert rules
        await self._check_alert_rules(metric)
        
        logger.debug(f"Recorded metric: {metric.name} = {metric.value}")
    
    async def _check_alert_rules(self, metric: QualityMetric):
        """Check metric against alert rules"""
        
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            # Check if rule applies to this metric
            if rule.metric_type != metric.metric_type:
                continue
            
            # Check symbol filter
            if rule.symbols and metric.symbol and metric.symbol not in rule.symbols:
                continue
            
            # Check cooldown
            cooldown_key = f"{rule_name}_{metric.symbol or 'all'}"
            if cooldown_key in self.alert_cooldowns:
                cooldown_end = self.alert_cooldowns[cooldown_key] + timedelta(minutes=rule.cooldown_minutes)
                if datetime.now() < cooldown_end:
                    continue
            
            # Evaluate condition
            should_alert = False
            
            if rule.condition == 'less_than' and metric.value < rule.threshold:
                should_alert = True
            elif rule.condition == 'greater_than' and metric.value > rule.threshold:
                should_alert = True
            elif rule.condition == 'equals' and abs(metric.value - rule.threshold) < 0.001:
                should_alert = True
            elif rule.condition == 'not_equals' and abs(metric.value - rule.threshold) >= 0.001:
                should_alert = True
            
            if should_alert:
                await self._trigger_alert(rule, metric)
    
    async def _trigger_alert(self, rule: AlertRule, metric: QualityMetric):
        """Trigger an alert"""
        
        # Generate alert ID
        alert_id = f"{rule.name}_{metric.symbol or 'all'}_{int(time.time())}"
        
        # Create alert message
        condition_text = {
            'less_than': 'below',
            'greater_than': 'above',
            'equals': 'equal to',
            'not_equals': 'not equal to'
        }.get(rule.condition, rule.condition)
        
        message = (f"{metric.name} is {condition_text} threshold: "
                  f"{metric.value:.4f} vs {rule.threshold:.4f}")
        
        # Create alert
        alert = QualityAlert(
            alert_id=alert_id,
            level=rule.severity,
            message=message,
            metric=metric,
            metadata={
                'rule_name': rule.name,
                'condition': rule.condition
            }
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Set cooldown
        cooldown_key = f"{rule.name}_{metric.symbol or 'all'}"
        self.alert_cooldowns[cooldown_key] = datetime.now()
        
        # Update stats
        self.monitoring_stats['alerts_triggered'] += 1
        
        # Export to Prometheus
        if self.prometheus_exporter:
            self.prometheus_exporter.record_alert(alert)
        
        # Send alerts
        await self._send_alert(alert)
        
        logger.warning(f"Alert triggered: {alert_id} - {message}")
    
    async def _send_alert(self, alert: QualityAlert):
        """Send alert through configured channels"""
        
        send_tasks = []
        
        for channel, alerter in self.alerters.items():
            if alerter:
                task = asyncio.create_task(alerter.send_alert(alert))
                send_tasks.append((channel, task))
        
        # Execute alert sending concurrently
        for channel, task in send_tasks:
            try:
                await task
                logger.info(f"Alert sent via {channel.value}: {alert.alert_id}")
            except Exception as e:
                logger.error(f"Failed to send alert via {channel.value}: {e}")
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Data quality monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Data quality monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        
        while self.monitoring_active:
            try:
                start_time = time.time()
                
                # Perform system health checks
                self._perform_system_checks()
                
                # Clean up old alerts
                self._cleanup_old_data()
                
                check_time = time.time() - start_time
                
                # Update performance stats
                self.monitoring_stats['checks_performed'] += 1
                total_checks = self.monitoring_stats['checks_performed']
                current_avg = self.monitoring_stats['avg_check_time']
                self.monitoring_stats['avg_check_time'] = (
                    (current_avg * (total_checks - 1) + check_time) / total_checks
                )
                
                # Sleep until next check
                sleep_time = max(0, self.config.check_interval_seconds - check_time)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.config.check_interval_seconds)
    
    def _perform_system_checks(self):
        """Perform system health checks"""
        
        # Check system resources
        if PSUTIL_AVAILABLE:
            # CPU usage
            cpu_percent = psutil.cpu_percent()
            if cpu_percent > 90:
                logger.warning(f"High CPU usage: {cpu_percent}%")
            
            # Memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                logger.warning(f"High memory usage: {memory.percent}%")
            
            # Disk usage
            disk = psutil.disk_usage('/')
            if disk.percent > 90:
                logger.warning(f"High disk usage: {disk.percent}%")
    
    def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        
        # Remove resolved alerts older than retention period
        retention_cutoff = datetime.now() - timedelta(hours=self.config.alert_retention_hours)
        
        # Clean up active alerts
        resolved_alerts = []
        for alert_id, alert in self.active_alerts.items():
            if alert.resolved and alert.timestamp < retention_cutoff:
                resolved_alerts.append(alert_id)
        
        for alert_id in resolved_alerts:
            del self.active_alerts[alert_id]
        
        # Clean up cooldowns
        expired_cooldowns = []
        for key, timestamp in self.alert_cooldowns.items():
            if datetime.now() - timestamp > timedelta(hours=24):  # 24 hour max cooldown
                expired_cooldowns.append(key)
        
        for key in expired_cooldowns:
            del self.alert_cooldowns[key]
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system"):
        """Acknowledge an alert"""
        
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            self.active_alerts[alert_id].metadata['acknowledged_by'] = acknowledged_by
            self.active_alerts[alert_id].metadata['acknowledged_at'] = datetime.now().isoformat()
            
            logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "system"):
        """Resolve an alert"""
        
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolved = True
            self.active_alerts[alert_id].metadata['resolved_by'] = resolved_by
            self.active_alerts[alert_id].metadata['resolved_at'] = datetime.now().isoformat()
            
            logger.info(f"Alert resolved: {alert_id} by {resolved_by}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status"""
        
        # Calculate alert statistics
        active_alerts_by_level = defaultdict(int)
        for alert in self.active_alerts.values():
            if not alert.resolved:
                active_alerts_by_level[alert.level.value] += 1
        
        # Recent metrics summary
        recent_metrics = {}
        for metric_key, metric_history in self.metrics_history.items():
            if metric_history:
                latest_metric = metric_history[-1]
                recent_metrics[metric_key] = {
                    'latest_value': latest_metric.value,
                    'timestamp': latest_metric.timestamp.isoformat(),
                    'history_length': len(metric_history)
                }
        
        return {
            'monitoring_active': self.monitoring_active,
            'configuration': {
                'check_interval': self.config.check_interval_seconds,
                'metric_retention_hours': self.config.metric_retention_hours,
                'alert_retention_hours': self.config.alert_retention_hours
            },
            'statistics': self.monitoring_stats.copy(),
            'alert_rules': {
                name: {
                    'enabled': rule.enabled,
                    'metric_type': rule.metric_type.value,
                    'condition': rule.condition,
                    'threshold': rule.threshold,
                    'severity': rule.severity.value
                }
                for name, rule in self.alert_rules.items()
            },
            'active_alerts': dict(active_alerts_by_level),
            'total_active_alerts': len([a for a in self.active_alerts.values() if not a.resolved]),
            'recent_metrics': recent_metrics,
            'alerter_channels': list(self.alerters.keys())
        }
    
    def get_metrics_history(self, metric_type: MetricType = None, symbol: str = None, 
                          hours: int = 24) -> Dict[str, List[Tuple[datetime, float]]]:
        """Get metrics history"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        history = {}
        
        for metric_key, metric_history in self.metrics_history.items():
            # Filter by criteria
            if metric_type:
                if not metric_key.startswith(metric_type.value):
                    continue
            
            if symbol:
                if not metric_key.endswith(symbol):
                    continue
            
            # Extract recent history
            recent_metrics = [
                (m.timestamp, m.value) for m in metric_history 
                if m.timestamp >= cutoff_time
            ]
            
            if recent_metrics:
                history[metric_key] = recent_metrics
        
        return history
    
    def generate_quality_report(self) -> str:
        """Generate comprehensive quality report"""
        
        status = self.get_monitoring_status()
        
        report_lines = [
            "📊 DATA QUALITY MONITORING REPORT",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Monitoring Status: {'🟢 Active' if status['monitoring_active'] else '🔴 Inactive'}",
            "",
            "📈 MONITORING STATISTICS",
            "-" * 30,
            f"Checks Performed: {status['statistics']['checks_performed']:,}",
            f"Metrics Collected: {status['statistics']['metrics_collected']:,}",
            f"Alerts Triggered: {status['statistics']['alerts_triggered']:,}",
            f"Average Check Time: {status['statistics']['avg_check_time']:.4f}s",
            "",
            "🚨 ACTIVE ALERTS",
            "-" * 20,
            f"Total Active: {status['total_active_alerts']}"
        ]
        
        # Alert breakdown by level
        for level, count in status['active_alerts'].items():
            if count > 0:
                emoji = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "critical": "🔴"}.get(level, "•")
                report_lines.append(f"  {emoji} {level.upper()}: {count}")
        
        # Alert rules summary
        report_lines.extend([
            "",
            "📋 ALERT RULES",
            "-" * 15,
            f"Total Rules: {len(status['alert_rules'])}"
        ])
        
        enabled_rules = sum(1 for rule in status['alert_rules'].values() if rule['enabled'])
        report_lines.append(f"Enabled Rules: {enabled_rules}")
        
        # Recent metrics
        if status['recent_metrics']:
            report_lines.extend([
                "",
                "📊 RECENT METRICS",
                "-" * 20
            ])
            
            for metric_key, metric_info in list(status['recent_metrics'].items())[:10]:
                report_lines.append(f"{metric_key}: {metric_info['latest_value']:.4f}")
        
        # Configuration
        report_lines.extend([
            "",
            "⚙️ CONFIGURATION",
            "-" * 18,
            f"Check Interval: {status['configuration']['check_interval']}s",
            f"Metric Retention: {status['configuration']['metric_retention_hours']}h",
            f"Alert Retention: {status['configuration']['alert_retention_hours']}h",
            f"Alert Channels: {', '.join([c.name for c in status['alerter_channels']])}"
        ])
        
        return "\n".join(report_lines)

# Pre-configured alert rules for common scenarios
def create_default_alert_rules() -> List[AlertRule]:
    """Create default alert rules for stock data quality"""
    
    return [
        # Data completeness rules
        AlertRule(
            name="low_data_completeness",
            metric_type=MetricType.COMPLETENESS,
            condition="less_than",
            threshold=0.95,
            severity=AlertLevel.WARNING,
            cooldown_minutes=30
        ),
        
        AlertRule(
            name="critical_data_completeness",
            metric_type=MetricType.COMPLETENESS,
            condition="less_than",
            threshold=0.80,
            severity=AlertLevel.CRITICAL,
            cooldown_minutes=15
        ),
        
        # Data accuracy rules
        AlertRule(
            name="low_data_accuracy",
            metric_type=MetricType.ACCURACY,
            condition="less_than",
            threshold=0.90,
            severity=AlertLevel.WARNING,
            cooldown_minutes=20
        ),
        
        # Data timeliness rules
        AlertRule(
            name="stale_data",
            metric_type=MetricType.TIMELINESS,
            condition="less_than",
            threshold=0.95,
            severity=AlertLevel.ERROR,
            cooldown_minutes=10
        ),
        
        # System performance rules
        AlertRule(
            name="slow_processing",
            metric_type=MetricType.PERFORMANCE,
            condition="greater_than",
            threshold=10.0,  # 10 seconds
            severity=AlertLevel.WARNING,
            cooldown_minutes=30
        )
    ]

# Example usage and testing
if __name__ == "__main__":
    print("📊 Data Quality Monitoring System")
    print("=" * 50)
    
    async def test_monitoring_system():
        # Create monitoring configuration
        config = MonitoringConfig(
            check_interval_seconds=5,  # Fast for testing
            metric_retention_hours=1,
            alert_retention_hours=24,
            enable_email_alerts=False,  # Disable for testing
            enable_slack_alerts=False,
            enable_prometheus=False
        )
        
        # Initialize monitor
        monitor = DataQualityMonitor(config)
        
        # Add default alert rules
        default_rules = create_default_alert_rules()
        for rule in default_rules:
            monitor.add_alert_rule(rule)
        
        print(f"✅ Monitor initialized with {len(default_rules)} alert rules")
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Simulate some quality metrics
        print(f"\n📊 Simulating quality metrics...")
        
        # Good metrics (should not trigger alerts)
        good_metrics = [
            QualityMetric("data_completeness", MetricType.COMPLETENESS, 0.98, 0.95, "AAPL"),
            QualityMetric("data_accuracy", MetricType.ACCURACY, 0.95, 0.90, "AAPL"),
            QualityMetric("data_timeliness", MetricType.TIMELINESS, 0.99, 0.95, "AAPL")
        ]
        
        for metric in good_metrics:
            await monitor.record_metric(metric)
            print(f"   📈 {metric.name}: {metric.value:.3f} (Good)")
        
        # Wait a bit
        await asyncio.sleep(2)
        
        # Bad metrics (should trigger alerts)
        print(f"\n🚨 Simulating problematic metrics...")
        
        bad_metrics = [
            QualityMetric("data_completeness", MetricType.COMPLETENESS, 0.75, 0.95, "MSFT"),  # Below warning
            QualityMetric("data_accuracy", MetricType.ACCURACY, 0.85, 0.90, "MSFT"),        # Below warning
            QualityMetric("processing_time", MetricType.PERFORMANCE, 15.0, 10.0, "GOOGL")   # Above warning
        ]
        
        for metric in bad_metrics:
            await monitor.record_metric(metric)
            print(f"   ⚠️ {metric.name}: {metric.value:.3f} (Alert expected)")
        
        # Wait for alerts to be processed
        await asyncio.sleep(3)
        
        # Check monitoring status
        print(f"\n📊 Monitoring Status:")
        status = monitor.get_monitoring_status()
        
        print(f"   Active Monitoring: {status['monitoring_active']}")
        print(f"   Checks Performed: {status['statistics']['checks_performed']}")
        print(f"   Metrics Collected: {status['statistics']['metrics_collected']}")
        print(f"   Alerts Triggered: {status['statistics']['alerts_triggered']}")
        print(f"   Total Active Alerts: {status['total_active_alerts']}")
        
        if status['active_alerts']:
            print(f"   Alert Breakdown:")
            for level, count in status['active_alerts'].items():
                print(f"      {level.upper()}: {count}")
        
        # Test alert acknowledgment
        if monitor.active_alerts:
            alert_id = list(monitor.active_alerts.keys())[0]
            monitor.acknowledge_alert(alert_id, "test_user")
            print(f"   ✅ Acknowledged alert: {alert_id}")
        
        # Generate quality report
        print(f"\n📋 Quality Report:")
        print("-" * 30)
        report = monitor.generate_quality_report()
        print(report)
        
        # Test metrics history
        print(f"\n📈 Metrics History:")
        history = monitor.get_metrics_history(hours=1)
        
        for metric_key, data_points in history.items():
            print(f"   {metric_key}: {len(data_points)} data points")
            if data_points:
                latest = data_points[-1]
                print(f"      Latest: {latest[1]:.4f} at {latest[0].strftime('%H:%M:%S')}")
        
        # Stop monitoring
        monitor.stop_monitoring()
        print(f"\n✅ Monitoring stopped")
    
    # Run test
    asyncio.run(test_monitoring_system())
    
    print(f"\n🎯 Data quality monitoring system ready!")
    print(f"📋 Features:")
    print(f"   • Real-time quality monitoring")
    print(f"   • Configurable alert rules")
    print(f"   • Multiple alert channels")
    print(f"   • Prometheus metrics export")
    print(f"   • Historical data tracking")
    print(f"   • System health monitoring")
    print(f"   • Comprehensive reporting")