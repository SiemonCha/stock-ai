"""
Comprehensive Monitoring Dashboards
Real-time system visualization and analytics
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import threading
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from collections import deque, defaultdict

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.utils
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import dash
    from dash import dcc, html, Input, Output, callback
    import dash_bootstrap_components as dbc
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

try:
    from flask import Flask, render_template, jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

class DashboardType(Enum):
    """Types of monitoring dashboards"""
    SYSTEM_HEALTH = "system_health"
    TRADING_PERFORMANCE = "trading_performance"
    MODEL_METRICS = "model_metrics"
    DATA_QUALITY = "data_quality"
    PORTFOLIO_ANALYTICS = "portfolio_analytics"
    RISK_MONITORING = "risk_monitoring"
    REAL_TIME_TRADING = "real_time_trading"

@dataclass
class MetricData:
    """Time series metric data point"""
    timestamp: datetime
    value: float
    metric_name: str
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    name: str
    dashboard_type: DashboardType
    refresh_interval: int = 5  # seconds
    max_data_points: int = 1000
    auto_refresh: bool = True
    theme: str = "dark"
    enable_alerts: bool = True

class MetricsCollector:
    """Real-time metrics collection system"""
    
    def __init__(self):
        self.metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.metric_metadata: Dict[str, Dict[str, Any]] = {}
        self.collection_active = False
        self.collection_thread = None
        
    def add_metric(self, metric: MetricData):
        """Add metric data point"""
        self.metrics_buffer[metric.metric_name].append(metric)
        
        # Store metadata
        if metric.metric_name not in self.metric_metadata:
            self.metric_metadata[metric.metric_name] = {
                'first_timestamp': metric.timestamp,
                'data_type': 'numeric',
                'tags': metric.tags
            }
    
    def get_metric_series(self, metric_name: str, 
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[MetricData]:
        """Get time series data for metric"""
        
        if metric_name not in self.metrics_buffer:
            return []
        
        metrics = list(self.metrics_buffer[metric_name])
        
        # Filter by time range
        if start_time:
            metrics = [m for m in metrics if m.timestamp >= start_time]
        
        if end_time:
            metrics = [m for m in metrics if m.timestamp <= end_time]
        
        return sorted(metrics, key=lambda x: x.timestamp)
    
    def get_latest_metrics(self) -> Dict[str, MetricData]:
        """Get latest value for each metric"""
        
        latest_metrics = {}
        
        for metric_name, metric_buffer in self.metrics_buffer.items():
            if metric_buffer:
                latest_metrics[metric_name] = metric_buffer[-1]
        
        return latest_metrics
    
    def start_collection(self):
        """Start background metrics collection"""
        
        if self.collection_active:
            return
        
        self.collection_active = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True
        )
        self.collection_thread.start()
        logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection"""
        self.collection_active = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("Metrics collection stopped")
    
    def _collection_loop(self):
        """Background metrics collection loop"""
        
        while self.collection_active:
            try:
                # Collect system metrics
                if PSUTIL_AVAILABLE:
                    self._collect_system_metrics()
                
                # Collect application metrics
                self._collect_application_metrics()
                
                time.sleep(1)  # Collect every second
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                time.sleep(5)
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        
        timestamp = datetime.now()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent()
        self.add_metric(MetricData(timestamp, cpu_percent, "system.cpu.percent"))
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.add_metric(MetricData(timestamp, memory.percent, "system.memory.percent"))
        self.add_metric(MetricData(timestamp, memory.available / (1024**3), "system.memory.available_gb"))
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        self.add_metric(MetricData(timestamp, disk.percent, "system.disk.percent"))
        
        # Network metrics
        net_io = psutil.net_io_counters()
        self.add_metric(MetricData(timestamp, net_io.bytes_sent / (1024**2), "system.network.sent_mb"))
        self.add_metric(MetricData(timestamp, net_io.bytes_recv / (1024**2), "system.network.recv_mb"))
    
    def _collect_application_metrics(self):
        """Collect application-specific metrics"""
        
        timestamp = datetime.now()
        
        # Simulated trading metrics
        self.add_metric(MetricData(timestamp, np.random.uniform(0.7, 0.95), "trading.accuracy"))
        self.add_metric(MetricData(timestamp, np.random.uniform(1.2, 2.5), "trading.sharpe_ratio"))
        self.add_metric(MetricData(timestamp, np.random.uniform(-0.05, -0.02), "trading.max_drawdown"))
        self.add_metric(MetricData(timestamp, np.random.uniform(50, 200), "trading.positions_count"))
        
        # Model performance metrics
        self.add_metric(MetricData(timestamp, np.random.uniform(1, 50), "model.prediction_latency_ms"))
        self.add_metric(MetricData(timestamp, np.random.uniform(0.85, 0.99), "model.confidence_score"))
        self.add_metric(MetricData(timestamp, np.random.uniform(500, 2000), "model.predictions_per_minute"))
        
        # Data quality metrics
        self.add_metric(MetricData(timestamp, np.random.uniform(0.95, 1.0), "data.completeness"))
        self.add_metric(MetricData(timestamp, np.random.uniform(0.90, 0.98), "data.accuracy"))
        self.add_metric(MetricData(timestamp, np.random.uniform(0, 5), "data.anomalies_detected"))

class ChartGenerator:
    """Generate interactive charts for dashboards"""
    
    def __init__(self, theme: str = "dark"):
        self.theme = theme
        self.color_palette = self._get_color_palette()
    
    def _get_color_palette(self) -> List[str]:
        """Get color palette based on theme"""
        if self.theme == "dark":
            return [
                '#00d4aa', '#ff6b6b', '#4ecdc4', '#45b7d1',
                '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff'
            ]
        else:
            return [
                '#2ecc71', '#e74c3c', '#3498db', '#f39c12',
                '#9b59b6', '#1abc9c', '#e67e22', '#34495e'
            ]
    
    def create_time_series_chart(self, metrics_data: List[MetricData], 
                               title: str, y_title: str = "Value") -> Dict[str, Any]:
        """Create time series chart"""
        
        if not PLOTLY_AVAILABLE or not metrics_data:
            return {}
        
        # Prepare data
        timestamps = [m.timestamp for m in metrics_data]
        values = [m.value for m in metrics_data]
        
        # Create figure
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            mode='lines+markers',
            name=title,
            line=dict(color=self.color_palette[0], width=2),
            marker=dict(size=4)
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Time",
            yaxis_title=y_title,
            template="plotly_dark" if self.theme == "dark" else "plotly_white",
            height=400,
            showlegend=False
        )
        
        return fig.to_dict()
    
    def create_multi_series_chart(self, series_data: Dict[str, List[MetricData]], 
                                title: str) -> Dict[str, Any]:
        """Create multi-series time series chart"""
        
        if not PLOTLY_AVAILABLE:
            return {}
        
        fig = go.Figure()
        
        for i, (series_name, metrics) in enumerate(series_data.items()):
            if not metrics:
                continue
            
            timestamps = [m.timestamp for m in metrics]
            values = [m.value for m in metrics]
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=values,
                mode='lines',
                name=series_name,
                line=dict(color=self.color_palette[i % len(self.color_palette)], width=2)
            ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Time",
            yaxis_title="Value",
            template="plotly_dark" if self.theme == "dark" else "plotly_white",
            height=400
        )
        
        return fig.to_dict()
    
    def create_gauge_chart(self, value: float, title: str, 
                          min_val: float = 0, max_val: float = 100,
                          thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Create gauge chart"""
        
        if not PLOTLY_AVAILABLE:
            return {}
        
        # Default thresholds
        if thresholds is None:
            thresholds = {"red": max_val * 0.8, "yellow": max_val * 0.6}
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            delta={'reference': thresholds.get("yellow", max_val * 0.6)},
            gauge={
                'axis': {'range': [None, max_val]},
                'bar': {'color': self.color_palette[0]},
                'steps': [
                    {'range': [min_val, thresholds.get("yellow", max_val * 0.6)], 'color': "lightgreen"},
                    {'range': [thresholds.get("yellow", max_val * 0.6), thresholds.get("red", max_val * 0.8)], 'color': "yellow"},
                    {'range': [thresholds.get("red", max_val * 0.8), max_val], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': thresholds.get("red", max_val * 0.8)
                }
            }
        ))
        
        fig.update_layout(
            template="plotly_dark" if self.theme == "dark" else "plotly_white",
            height=300
        )
        
        return fig.to_dict()
    
    def create_heatmap(self, data: pd.DataFrame, title: str) -> Dict[str, Any]:
        """Create correlation heatmap"""
        
        if not PLOTLY_AVAILABLE:
            return {}
        
        fig = go.Figure(data=go.Heatmap(
            z=data.values,
            x=data.columns,
            y=data.index,
            colorscale='RdYlBu_r',
            zmid=0
        ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            template="plotly_dark" if self.theme == "dark" else "plotly_white",
            height=400
        )
        
        return fig.to_dict()

class DashboardGenerator:
    """Generate complete monitoring dashboards"""
    
    def __init__(self, metrics_collector: MetricsCollector, theme: str = "dark"):
        self.metrics_collector = metrics_collector
        self.chart_generator = ChartGenerator(theme)
        self.theme = theme
    
    def generate_system_health_dashboard(self) -> Dict[str, Any]:
        """Generate system health dashboard"""
        
        # Get recent data (last hour)
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        
        # System metrics
        cpu_metrics = self.metrics_collector.get_metric_series("system.cpu.percent", start_time, end_time)
        memory_metrics = self.metrics_collector.get_metric_series("system.memory.percent", start_time, end_time)
        disk_metrics = self.metrics_collector.get_metric_series("system.disk.percent", start_time, end_time)
        
        # Latest values
        latest_metrics = self.metrics_collector.get_latest_metrics()
        
        dashboard_data = {
            'title': 'System Health Dashboard',
            'timestamp': datetime.now().isoformat(),
            'charts': {},
            'gauges': {},
            'metrics_summary': {}
        }
        
        # Time series charts
        dashboard_data['charts']['cpu_usage'] = self.chart_generator.create_time_series_chart(
            cpu_metrics, "CPU Usage (%)", "Percent"
        )
        
        dashboard_data['charts']['memory_usage'] = self.chart_generator.create_time_series_chart(
            memory_metrics, "Memory Usage (%)", "Percent"
        )
        
        dashboard_data['charts']['system_overview'] = self.chart_generator.create_multi_series_chart(
            {
                'CPU': cpu_metrics,
                'Memory': memory_metrics,
                'Disk': disk_metrics
            },
            "System Resources Overview"
        )
        
        # Gauge charts
        if 'system.cpu.percent' in latest_metrics:
            dashboard_data['gauges']['cpu_gauge'] = self.chart_generator.create_gauge_chart(
                latest_metrics['system.cpu.percent'].value,
                "Current CPU Usage",
                thresholds={"yellow": 60, "red": 80}
            )
        
        if 'system.memory.percent' in latest_metrics:
            dashboard_data['gauges']['memory_gauge'] = self.chart_generator.create_gauge_chart(
                latest_metrics['system.memory.percent'].value,
                "Current Memory Usage",
                thresholds={"yellow": 70, "red": 90}
            )
        
        # Metrics summary
        dashboard_data['metrics_summary'] = {
            metric_name: {
                'current_value': metric.value,
                'timestamp': metric.timestamp.isoformat(),
                'status': self._get_metric_status(metric_name, metric.value)
            }
            for metric_name, metric in latest_metrics.items()
            if metric_name.startswith('system.')
        }
        
        return dashboard_data
    
    def generate_trading_performance_dashboard(self) -> Dict[str, Any]:
        """Generate trading performance dashboard"""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)  # Last 24 hours
        
        # Trading metrics
        accuracy_metrics = self.metrics_collector.get_metric_series("trading.accuracy", start_time, end_time)
        sharpe_metrics = self.metrics_collector.get_metric_series("trading.sharpe_ratio", start_time, end_time)
        drawdown_metrics = self.metrics_collector.get_metric_series("trading.max_drawdown", start_time, end_time)
        positions_metrics = self.metrics_collector.get_metric_series("trading.positions_count", start_time, end_time)
        
        latest_metrics = self.metrics_collector.get_latest_metrics()
        
        dashboard_data = {
            'title': 'Trading Performance Dashboard',
            'timestamp': datetime.now().isoformat(),
            'charts': {},
            'gauges': {},
            'kpis': {}
        }
        
        # Performance charts
        dashboard_data['charts']['accuracy_trend'] = self.chart_generator.create_time_series_chart(
            accuracy_metrics, "Prediction Accuracy", "Accuracy"
        )
        
        dashboard_data['charts']['sharpe_trend'] = self.chart_generator.create_time_series_chart(
            sharpe_metrics, "Sharpe Ratio", "Ratio"
        )
        
        dashboard_data['charts']['trading_overview'] = self.chart_generator.create_multi_series_chart(
            {
                'Accuracy': [MetricData(m.timestamp, m.value * 100, m.metric_name) for m in accuracy_metrics],
                'Positions': positions_metrics
            },
            "Trading Overview"
        )
        
        # Performance gauges
        if 'trading.accuracy' in latest_metrics:
            dashboard_data['gauges']['accuracy_gauge'] = self.chart_generator.create_gauge_chart(
                latest_metrics['trading.accuracy'].value * 100,
                "Current Accuracy (%)",
                max_val=100,
                thresholds={"yellow": 70, "red": 60}
            )
        
        # KPI calculations
        if accuracy_metrics:
            avg_accuracy = sum(m.value for m in accuracy_metrics) / len(accuracy_metrics)
            dashboard_data['kpis']['average_accuracy'] = f"{avg_accuracy:.1%}"
        
        if sharpe_metrics:
            current_sharpe = sharpe_metrics[-1].value if sharpe_metrics else 0
            dashboard_data['kpis']['current_sharpe'] = f"{current_sharpe:.2f}"
        
        return dashboard_data
    
    def generate_model_metrics_dashboard(self) -> Dict[str, Any]:
        """Generate ML model metrics dashboard"""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=6)  # Last 6 hours
        
        # Model metrics
        latency_metrics = self.metrics_collector.get_metric_series("model.prediction_latency_ms", start_time, end_time)
        confidence_metrics = self.metrics_collector.get_metric_series("model.confidence_score", start_time, end_time)
        throughput_metrics = self.metrics_collector.get_metric_series("model.predictions_per_minute", start_time, end_time)
        
        latest_metrics = self.metrics_collector.get_latest_metrics()
        
        dashboard_data = {
            'title': 'Model Performance Dashboard',
            'timestamp': datetime.now().isoformat(),
            'charts': {},
            'gauges': {},
            'model_stats': {}
        }
        
        # Model performance charts
        dashboard_data['charts']['latency_trend'] = self.chart_generator.create_time_series_chart(
            latency_metrics, "Prediction Latency (ms)", "Milliseconds"
        )
        
        dashboard_data['charts']['confidence_trend'] = self.chart_generator.create_time_series_chart(
            confidence_metrics, "Model Confidence", "Score"
        )
        
        dashboard_data['charts']['throughput_trend'] = self.chart_generator.create_time_series_chart(
            throughput_metrics, "Predictions per Minute", "Count"
        )
        
        # Model gauges
        if 'model.prediction_latency_ms' in latest_metrics:
            dashboard_data['gauges']['latency_gauge'] = self.chart_generator.create_gauge_chart(
                latest_metrics['model.prediction_latency_ms'].value,
                "Current Latency (ms)",
                max_val=100,
                thresholds={"yellow": 30, "red": 50}
            )
        
        if 'model.confidence_score' in latest_metrics:
            dashboard_data['gauges']['confidence_gauge'] = self.chart_generator.create_gauge_chart(
                latest_metrics['model.confidence_score'].value * 100,
                "Model Confidence (%)",
                max_val=100,
                thresholds={"yellow": 80, "red": 70}
            )
        
        # Model statistics
        if latency_metrics:
            latencies = [m.value for m in latency_metrics]
            dashboard_data['model_stats'] = {
                'avg_latency_ms': f"{np.mean(latencies):.1f}",
                'p95_latency_ms': f"{np.percentile(latencies, 95):.1f}",
                'min_latency_ms': f"{np.min(latencies):.1f}",
                'max_latency_ms': f"{np.max(latencies):.1f}"
            }
        
        return dashboard_data
    
    def _get_metric_status(self, metric_name: str, value: float) -> str:
        """Get status for metric value"""
        
        # Define thresholds for different metrics
        thresholds = {
            'system.cpu.percent': {'warning': 70, 'critical': 90},
            'system.memory.percent': {'warning': 80, 'critical': 95},
            'system.disk.percent': {'warning': 85, 'critical': 95},
            'trading.accuracy': {'warning': 0.7, 'critical': 0.6},
            'model.prediction_latency_ms': {'warning': 30, 'critical': 50}
        }
        
        if metric_name not in thresholds:
            return 'ok'
        
        threshold = thresholds[metric_name]
        
        # For accuracy, lower is worse
        if 'accuracy' in metric_name:
            if value < threshold['critical']:
                return 'critical'
            elif value < threshold['warning']:
                return 'warning'
            else:
                return 'ok'
        else:
            # For other metrics, higher is worse
            if value > threshold['critical']:
                return 'critical'
            elif value > threshold['warning']:
                return 'warning'
            else:
                return 'ok'

class WebDashboard:
    """Web-based dashboard interface"""
    
    def __init__(self, metrics_collector: MetricsCollector, port: int = 8050):
        self.metrics_collector = metrics_collector
        self.dashboard_generator = DashboardGenerator(metrics_collector)
        self.port = port
        self.app = None
        
        if FLASK_AVAILABLE:
            self._setup_flask_app()
        elif DASH_AVAILABLE:
            self._setup_dash_app()
    
    def _setup_flask_app(self):
        """Setup Flask application"""
        
        self.app = Flask(__name__)
        
        @self.app.route('/')
        def index():
            return '''
            <html>
                <head><title>Stock AI Monitoring</title></head>
                <body>
                    <h1>Stock AI Monitoring Dashboards</h1>
                    <ul>
                        <li><a href="/api/dashboard/system_health">System Health API</a></li>
                        <li><a href="/api/dashboard/trading_performance">Trading Performance API</a></li>
                        <li><a href="/api/dashboard/model_metrics">Model Metrics API</a></li>
                    </ul>
                </body>
            </html>
            '''
        
        @self.app.route('/api/dashboard/<dashboard_type>')
        def get_dashboard_data(dashboard_type):
            if dashboard_type == 'system_health':
                data = self.dashboard_generator.generate_system_health_dashboard()
            elif dashboard_type == 'trading_performance':
                data = self.dashboard_generator.generate_trading_performance_dashboard()
            elif dashboard_type == 'model_metrics':
                data = self.dashboard_generator.generate_model_metrics_dashboard()
            else:
                return jsonify({'error': 'Unknown dashboard type'}), 400
            
            return jsonify(data)
        
        @self.app.route('/api/metrics/latest')
        def get_latest_metrics():
            latest = self.metrics_collector.get_latest_metrics()
            
            return jsonify({
                metric_name: {
                    'value': metric.value,
                    'timestamp': metric.timestamp.isoformat()
                }
                for metric_name, metric in latest.items()
            })
    
    def _setup_dash_app(self):
        """Setup Dash application"""
        
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Stock AI Monitoring", className="text-center mb-4"),
                    dcc.Tabs(id="dashboard-tabs", value="system-health", children=[
                        dcc.Tab(label="System Health", value="system-health"),
                        dcc.Tab(label="Trading Performance", value="trading-performance"),
                        dcc.Tab(label="Model Metrics", value="model-metrics")
                    ])
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div(id="dashboard-content")
                ])
            ]),
            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # Update every 5 seconds
                n_intervals=0
            )
        ], fluid=True)
        
        @self.app.callback(
            Output('dashboard-content', 'children'),
            [Input('dashboard-tabs', 'value'),
             Input('interval-component', 'n_intervals')]
        )
        def update_dashboard_content(active_tab, n_intervals):
            if active_tab == 'system-health':
                data = self.dashboard_generator.generate_system_health_dashboard()
                return self._create_system_health_layout(data)
            elif active_tab == 'trading-performance':
                data = self.dashboard_generator.generate_trading_performance_dashboard()
                return self._create_trading_performance_layout(data)
            elif active_tab == 'model-metrics':
                data = self.dashboard_generator.generate_model_metrics_dashboard()
                return self._create_model_metrics_layout(data)
            
            return html.Div("Select a dashboard")
    
    def _create_system_health_layout(self, data: Dict[str, Any]):
        """Create system health dashboard layout"""
        
        if not DASH_AVAILABLE:
            return []
        
        return [
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=data['charts'].get('cpu_usage', {}))
                ], width=6),
                dbc.Col([
                    dcc.Graph(figure=data['charts'].get('memory_usage', {}))
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=data['gauges'].get('cpu_gauge', {}))
                ], width=6),
                dbc.Col([
                    dcc.Graph(figure=data['gauges'].get('memory_gauge', {}))
                ], width=6)
            ])
        ]
    
    def _create_trading_performance_layout(self, data: Dict[str, Any]):
        """Create trading performance dashboard layout"""
        
        if not DASH_AVAILABLE:
            return []
        
        return [
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=data['charts'].get('accuracy_trend', {}))
                ], width=6),
                dbc.Col([
                    dcc.Graph(figure=data['charts'].get('sharpe_trend', {}))
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=data['gauges'].get('accuracy_gauge', {}))
                ], width=12)
            ])
        ]
    
    def _create_model_metrics_layout(self, data: Dict[str, Any]):
        """Create model metrics dashboard layout"""
        
        if not DASH_AVAILABLE:
            return []
        
        return [
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=data['charts'].get('latency_trend', {}))
                ], width=6),
                dbc.Col([
                    dcc.Graph(figure=data['charts'].get('confidence_trend', {}))
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=data['charts'].get('throughput_trend', {}))
                ], width=12)
            ])
        ]
    
    def run(self, debug: bool = False, host: str = '0.0.0.0'):
        """Run the web dashboard"""
        
        if not self.app:
            logger.error("No web framework available")
            return
        
        logger.info(f"Starting web dashboard on http://{host}:{self.port}")
        
        if hasattr(self.app, 'run_server'):  # Dash app
            self.app.run_server(debug=debug, host=host, port=self.port)
        else:  # Flask app
            self.app.run(debug=debug, host=host, port=self.port)

# Example usage and testing
if __name__ == "__main__":
    print("📊 Monitoring Dashboard System")
    print("=" * 35)
    
    # Initialize components
    metrics_collector = MetricsCollector()
    dashboard_generator = DashboardGenerator(metrics_collector)
    
    print("✅ Metrics collector initialized")
    
    # Start metrics collection
    metrics_collector.start_collection()
    print("✅ Metrics collection started")
    
    # Let some data collect
    print("📈 Collecting sample metrics...")
    time.sleep(5)
    
    # Generate dashboards
    print("\n📊 Generating dashboards...")
    
    # System health dashboard
    system_health = dashboard_generator.generate_system_health_dashboard()
    print(f"   ✅ System Health Dashboard: {len(system_health.get('charts', {}))} charts")
    
    # Trading performance dashboard
    trading_performance = dashboard_generator.generate_trading_performance_dashboard()
    print(f"   ✅ Trading Performance Dashboard: {len(trading_performance.get('charts', {}))} charts")
    
    # Model metrics dashboard
    model_metrics = dashboard_generator.generate_model_metrics_dashboard()
    print(f"   ✅ Model Metrics Dashboard: {len(model_metrics.get('charts', {}))} charts")
    
    # Show latest metrics
    print("\n📈 Latest Metrics:")
    latest_metrics = metrics_collector.get_latest_metrics()
    
    for metric_name, metric in list(latest_metrics.items())[:10]:  # Show first 10
        status_emoji = {"ok": "✅", "warning": "⚠️", "critical": "❌"}.get(
            dashboard_generator._get_metric_status(metric_name, metric.value), "📊"
        )
        print(f"   {status_emoji} {metric_name}: {metric.value:.2f}")
    
    # Test web dashboard setup
    print(f"\n🌐 Web Dashboard:")
    
    try:
        web_dashboard = WebDashboard(metrics_collector, port=8051)  # Different port for testing
        
        if web_dashboard.app:
            print(f"   ✅ Web dashboard initialized")
            print(f"   🌐 Available at: http://localhost:8051")
            
            if FLASK_AVAILABLE:
                print(f"   📋 API Endpoints:")
                print(f"      • /api/dashboard/system_health")
                print(f"      • /api/dashboard/trading_performance")
                print(f"      • /api/dashboard/model_metrics")
                print(f"      • /api/metrics/latest")
            
            # Don't actually start the server in testing
            print(f"   (Server not started for testing)")
            
        else:
            print(f"   ⚠️  Web dashboard not available (missing dependencies)")
    
    except Exception as e:
        print(f"   ❌ Web dashboard error: {e}")
    
    # Stop metrics collection
    metrics_collector.stop_collection()
    print(f"\n🛑 Metrics collection stopped")
    
    print(f"\n🎯 Monitoring dashboard system ready!")
    print(f"📋 Features:")
    print(f"   • Real-time metrics collection")
    print(f"   • Interactive chart generation")
    print(f"   • Multiple dashboard types")
    print(f"   • Web-based interface")
    print(f"   • REST API endpoints")
    print(f"   • Customizable themes")
    print(f"   • Auto-refresh capabilities")
    
    print(f"\n📊 Dashboard types supported:")
    for dashboard_type in DashboardType:
        print(f"   • {dashboard_type.value}")
    
    print(f"\n📈 Metrics tracked:")
    print(f"   • System: CPU, Memory, Disk, Network")
    print(f"   • Trading: Accuracy, Sharpe, Drawdown, Positions")
    print(f"   • Models: Latency, Confidence, Throughput")
    print(f"   • Data: Completeness, Accuracy, Anomalies")