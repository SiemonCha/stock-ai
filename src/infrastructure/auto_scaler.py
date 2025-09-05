"""
Advanced Auto-Scaling System for Stock AI Platform

This module provides intelligent scaling mechanisms that automatically adjust system resources
based on real-time metrics including trading volume, computational load, market volatility,
and prediction accuracy requirements.

Features:
- Horizontal Pod Autoscaling (HPA) for Kubernetes
- Custom metrics-based scaling (trading volume, model inference latency)
- Predictive scaling based on market patterns
- Multi-dimensional scaling strategies
- Cost-optimized resource allocation
- Emergency scaling for market events
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import numpy as np
from datetime import datetime, timedelta
import kubernetes as k8s
from kubernetes.client.rest import ApiException
import docker
import psutil
import redis
from prometheus_client import CollectorRegistry, Gauge, Counter, push_to_gateway


class ScalingTrigger(Enum):
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    TRADING_VOLUME = "trading_volume"
    MODEL_LATENCY = "model_latency"
    QUEUE_DEPTH = "queue_depth"
    MARKET_VOLATILITY = "market_volatility"
    PREDICTION_LOAD = "prediction_load"
    CUSTOM_METRIC = "custom_metric"


class ScalingAction(Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"
    EMERGENCY_SCALE = "emergency_scale"


@dataclass
class ScalingRule:
    name: str
    trigger: ScalingTrigger
    threshold_up: float
    threshold_down: float
    scale_up_factor: float = 1.5
    scale_down_factor: float = 0.7
    cooldown_period: int = 300  # seconds
    min_replicas: int = 1
    max_replicas: int = 100
    enabled: bool = True
    priority: int = 1  # Higher number = higher priority


@dataclass
class ScalingMetrics:
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    trading_volume: float
    model_latency: float
    queue_depth: int
    market_volatility: float
    prediction_load: float
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScalingDecision:
    action: ScalingAction
    target_replicas: int
    current_replicas: int
    triggered_by: List[ScalingTrigger]
    confidence: float
    estimated_cost: float
    reasoning: str


class PredictiveScaler:
    """
    Predictive scaling based on historical patterns and market events
    """
    
    def __init__(self):
        self.historical_data: List[ScalingMetrics] = []
        self.pattern_cache = {}
        
    def predict_scaling_needs(self, 
                            current_metrics: ScalingMetrics,
                            horizon_minutes: int = 15) -> Dict[str, Any]:
        """
        Predict scaling needs based on historical patterns
        """
        prediction = {
            'predicted_cpu': current_metrics.cpu_usage,
            'predicted_memory': current_metrics.memory_usage,
            'predicted_volume': current_metrics.trading_volume,
            'confidence': 0.5,
            'pattern_match': None
        }
        
        if len(self.historical_data) < 100:
            return prediction
        
        # Find similar market conditions
        similar_periods = self._find_similar_periods(current_metrics)
        
        if similar_periods:
            # Calculate weighted predictions
            weights = np.exp(-np.arange(len(similar_periods)) * 0.1)
            weights /= weights.sum()
            
            cpu_predictions = []
            memory_predictions = []
            volume_predictions = []
            
            for period in similar_periods:
                future_idx = min(len(self.historical_data) - 1,
                               period + horizon_minutes)
                future_metrics = self.historical_data[future_idx]
                
                cpu_predictions.append(future_metrics.cpu_usage)
                memory_predictions.append(future_metrics.memory_usage)
                volume_predictions.append(future_metrics.trading_volume)
            
            prediction.update({
                'predicted_cpu': np.average(cpu_predictions, weights=weights),
                'predicted_memory': np.average(memory_predictions, weights=weights),
                'predicted_volume': np.average(volume_predictions, weights=weights),
                'confidence': min(0.9, len(similar_periods) / 20),
                'pattern_match': f"Found {len(similar_periods)} similar periods"
            })
        
        return prediction
    
    def _find_similar_periods(self, current_metrics: ScalingMetrics) -> List[int]:
        """Find historical periods with similar characteristics"""
        similar = []
        current_hour = current_metrics.timestamp.hour
        current_day = current_metrics.timestamp.weekday()
        
        for i, historical in enumerate(self.historical_data[:-60]):  # Exclude recent data
            if (abs(historical.timestamp.hour - current_hour) <= 1 and
                historical.timestamp.weekday() == current_day and
                abs(historical.trading_volume - current_metrics.trading_volume) / 
                max(current_metrics.trading_volume, 1) < 0.3):
                similar.append(i)
        
        return similar[-50:]  # Return up to 50 most recent similar periods


class KubernetesScaler:
    """
    Kubernetes-based horizontal pod autoscaler
    """
    
    def __init__(self, namespace: str = "stock-ai"):
        self.namespace = namespace
        self.k8s_apps_v1 = k8s.client.AppsV1Api()
        self.k8s_core_v1 = k8s.client.CoreV1Api()
        self.k8s_autoscaling_v2 = k8s.client.AutoscalingV2Api()
        self.logger = logging.getLogger(__name__)
        
    async def scale_deployment(self, 
                             deployment_name: str, 
                             target_replicas: int) -> bool:
        """Scale a Kubernetes deployment to target replica count"""
        try:
            # Get current deployment
            deployment = self.k8s_apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
            )
            
            current_replicas = deployment.spec.replicas
            
            if current_replicas == target_replicas:
                return True
            
            # Update replica count
            deployment.spec.replicas = target_replicas
            
            self.k8s_apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace,
                body=deployment
            )
            
            self.logger.info(
                f"Scaled deployment {deployment_name} from {current_replicas} "
                f"to {target_replicas} replicas"
            )
            
            return True
            
        except ApiException as e:
            self.logger.error(f"Failed to scale deployment {deployment_name}: {e}")
            return False
    
    async def create_hpa(self, 
                        deployment_name: str, 
                        min_replicas: int,
                        max_replicas: int,
                        target_cpu: int = 70,
                        target_memory: int = 80) -> bool:
        """Create Horizontal Pod Autoscaler for deployment"""
        try:
            hpa_spec = k8s.client.V2HorizontalPodAutoscalerSpec(
                scale_target_ref=k8s.client.V2CrossVersionObjectReference(
                    api_version="apps/v1",
                    kind="Deployment",
                    name=deployment_name
                ),
                min_replicas=min_replicas,
                max_replicas=max_replicas,
                metrics=[
                    k8s.client.V2MetricSpec(
                        type="Resource",
                        resource=k8s.client.V2ResourceMetricSource(
                            name="cpu",
                            target=k8s.client.V2MetricTarget(
                                type="Utilization",
                                average_utilization=target_cpu
                            )
                        )
                    ),
                    k8s.client.V2MetricSpec(
                        type="Resource",
                        resource=k8s.client.V2ResourceMetricSource(
                            name="memory",
                            target=k8s.client.V2MetricTarget(
                                type="Utilization",
                                average_utilization=target_memory
                            )
                        )
                    )
                ]
            )
            
            hpa = k8s.client.V2HorizontalPodAutoscaler(
                metadata=k8s.client.V1ObjectMeta(name=f"{deployment_name}-hpa"),
                spec=hpa_spec
            )
            
            self.k8s_autoscaling_v2.create_namespaced_horizontal_pod_autoscaler(
                namespace=self.namespace,
                body=hpa
            )
            
            self.logger.info(f"Created HPA for deployment {deployment_name}")
            return True
            
        except ApiException as e:
            self.logger.error(f"Failed to create HPA for {deployment_name}: {e}")
            return False


class AutoScaler:
    """
    Main auto-scaling orchestrator with intelligent scaling decisions
    """
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 prometheus_gateway: str = "localhost:9091"):
        self.redis_client = redis.from_url(redis_url)
        self.prometheus_gateway = prometheus_gateway
        self.k8s_scaler = KubernetesScaler()
        self.predictive_scaler = PredictiveScaler()
        
        # Scaling rules
        self.scaling_rules = [
            ScalingRule(
                name="CPU High Load",
                trigger=ScalingTrigger.CPU_USAGE,
                threshold_up=75.0,
                threshold_down=45.0,
                priority=3
            ),
            ScalingRule(
                name="Memory Pressure",
                trigger=ScalingTrigger.MEMORY_USAGE,
                threshold_up=80.0,
                threshold_down=50.0,
                priority=3
            ),
            ScalingRule(
                name="Trading Volume Spike",
                trigger=ScalingTrigger.TRADING_VOLUME,
                threshold_up=1000.0,  # trades per minute
                threshold_down=200.0,
                scale_up_factor=2.0,
                priority=5
            ),
            ScalingRule(
                name="Model Latency",
                trigger=ScalingTrigger.MODEL_LATENCY,
                threshold_up=500.0,  # milliseconds
                threshold_down=100.0,
                priority=4
            ),
            ScalingRule(
                name="Market Volatility",
                trigger=ScalingTrigger.MARKET_VOLATILITY,
                threshold_up=0.05,  # 5% volatility
                threshold_down=0.02,
                scale_up_factor=1.8,
                priority=4
            )
        ]
        
        # Cooldown tracking
        self.last_scaling_actions = {}
        
        # Metrics
        self.registry = CollectorRegistry()
        self.scaling_actions_counter = Counter(
            'autoscaler_scaling_actions_total',
            'Total number of scaling actions performed',
            ['deployment', 'action', 'trigger'],
            registry=self.registry
        )
        self.current_replicas_gauge = Gauge(
            'autoscaler_current_replicas',
            'Current number of replicas for deployment',
            ['deployment'],
            registry=self.registry
        )
        
        self.logger = logging.getLogger(__name__)
        self.running = False
    
    async def start_autoscaler(self, interval: int = 30):
        """Start the auto-scaling loop"""
        self.running = True
        self.logger.info("Starting auto-scaler")
        
        while self.running:
            try:
                await self._scaling_cycle()
                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error in scaling cycle: {e}")
                await asyncio.sleep(interval)
    
    def stop_autoscaler(self):
        """Stop the auto-scaling loop"""
        self.running = False
        self.logger.info("Stopping auto-scaler")
    
    async def _scaling_cycle(self):
        """Execute one complete scaling cycle"""
        # Collect current metrics
        current_metrics = await self._collect_metrics()
        
        # Get predictive insights
        predictions = self.predictive_scaler.predict_scaling_needs(current_metrics)
        
        # Evaluate scaling rules for each deployment
        deployments = await self._get_deployments()
        
        for deployment in deployments:
            decision = await self._make_scaling_decision(
                deployment, current_metrics, predictions
            )
            
            if decision.action != ScalingAction.MAINTAIN:
                await self._execute_scaling_decision(deployment, decision)
        
        # Store metrics for predictive scaling
        self.predictive_scaler.historical_data.append(current_metrics)
        if len(self.predictive_scaler.historical_data) > 10000:
            self.predictive_scaler.historical_data = \
                self.predictive_scaler.historical_data[-5000:]
        
        # Push metrics to Prometheus
        await self._push_metrics()
    
    async def _collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics"""
        try:
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            
            # Trading metrics from Redis
            trading_volume = float(
                self.redis_client.get('trading_volume_per_minute') or 0
            )
            model_latency = float(
                self.redis_client.get('avg_model_inference_latency') or 0
            )
            queue_depth = int(
                self.redis_client.llen('prediction_queue') or 0
            )
            market_volatility = float(
                self.redis_client.get('market_volatility_5min') or 0
            )
            prediction_load = float(
                self.redis_client.get('predictions_per_minute') or 0
            )
            
            return ScalingMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                trading_volume=trading_volume,
                model_latency=model_latency,
                queue_depth=queue_depth,
                market_volatility=market_volatility,
                prediction_load=prediction_load
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
            return ScalingMetrics(
                timestamp=datetime.now(),
                cpu_usage=0, memory_usage=0, trading_volume=0,
                model_latency=0, queue_depth=0, market_volatility=0,
                prediction_load=0
            )
    
    async def _get_deployments(self) -> List[str]:
        """Get list of deployments to manage"""
        try:
            deployments = self.k8s_scaler.k8s_apps_v1.list_namespaced_deployment(
                namespace=self.k8s_scaler.namespace,
                label_selector="autoscaling=enabled"
            )
            return [dep.metadata.name for dep in deployments.items]
        except Exception as e:
            self.logger.error(f"Error getting deployments: {e}")
            return []
    
    async def _make_scaling_decision(self, 
                                   deployment: str,
                                   metrics: ScalingMetrics,
                                   predictions: Dict[str, Any]) -> ScalingDecision:
        """Make intelligent scaling decision based on rules and predictions"""
        
        # Get current replica count
        try:
            dep = self.k8s_scaler.k8s_apps_v1.read_namespaced_deployment(
                name=deployment,
                namespace=self.k8s_scaler.namespace
            )
            current_replicas = dep.spec.replicas
        except:
            current_replicas = 1
        
        triggered_rules = []
        scale_factors = []
        reasoning_parts = []
        
        # Evaluate each scaling rule
        for rule in self.scaling_rules:
            if not rule.enabled:
                continue
            
            # Check cooldown
            last_action_key = f"{deployment}_{rule.name}"
            if last_action_key in self.last_scaling_actions:
                time_since_last = time.time() - self.last_scaling_actions[last_action_key]
                if time_since_last < rule.cooldown_period:
                    continue
            
            # Get metric value
            metric_value = self._get_metric_value(metrics, rule.trigger)
            
            if metric_value > rule.threshold_up:
                triggered_rules.append(rule.trigger)
                scale_factors.append(rule.scale_up_factor * rule.priority)
                reasoning_parts.append(
                    f"{rule.trigger.value} ({metric_value:.2f}) > {rule.threshold_up}"
                )
            elif metric_value < rule.threshold_down and current_replicas > rule.min_replicas:
                triggered_rules.append(rule.trigger)
                scale_factors.append(rule.scale_down_factor)
                reasoning_parts.append(
                    f"{rule.trigger.value} ({metric_value:.2f}) < {rule.threshold_down}"
                )
        
        # Determine scaling action
        if not triggered_rules:
            return ScalingDecision(
                action=ScalingAction.MAINTAIN,
                target_replicas=current_replicas,
                current_replicas=current_replicas,
                triggered_by=[],
                confidence=1.0,
                estimated_cost=0.0,
                reasoning="No scaling triggers activated"
            )
        
        # Calculate target replicas
        if any(factor > 1.0 for factor in scale_factors):
            action = ScalingAction.SCALE_UP
            avg_factor = np.mean([f for f in scale_factors if f > 1.0])
            target_replicas = min(100, int(current_replicas * avg_factor))
        else:
            action = ScalingAction.SCALE_DOWN
            avg_factor = np.mean(scale_factors)
            target_replicas = max(1, int(current_replicas * avg_factor))
        
        # Consider predictive insights
        confidence = 0.8
        if predictions['confidence'] > 0.7:
            confidence = min(0.95, confidence + predictions['confidence'] * 0.2)
            reasoning_parts.append(f"Predictive confidence: {predictions['confidence']:.2f}")
        
        # Emergency scaling for extreme conditions
        if (metrics.trading_volume > 5000 or 
            metrics.market_volatility > 0.1 or
            metrics.model_latency > 1000):
            action = ScalingAction.EMERGENCY_SCALE
            target_replicas = min(100, current_replicas * 3)
            reasoning_parts.append("Emergency scaling triggered")
        
        # Estimate cost impact
        cost_per_replica_hour = 0.1  # $0.10 per replica per hour
        replica_change = target_replicas - current_replicas
        estimated_cost = replica_change * cost_per_replica_hour
        
        return ScalingDecision(
            action=action,
            target_replicas=target_replicas,
            current_replicas=current_replicas,
            triggered_by=triggered_rules,
            confidence=confidence,
            estimated_cost=estimated_cost,
            reasoning=" | ".join(reasoning_parts)
        )
    
    def _get_metric_value(self, metrics: ScalingMetrics, trigger: ScalingTrigger) -> float:
        """Get metric value for specific trigger"""
        metric_map = {
            ScalingTrigger.CPU_USAGE: metrics.cpu_usage,
            ScalingTrigger.MEMORY_USAGE: metrics.memory_usage,
            ScalingTrigger.TRADING_VOLUME: metrics.trading_volume,
            ScalingTrigger.MODEL_LATENCY: metrics.model_latency,
            ScalingTrigger.QUEUE_DEPTH: metrics.queue_depth,
            ScalingTrigger.MARKET_VOLATILITY: metrics.market_volatility,
            ScalingTrigger.PREDICTION_LOAD: metrics.prediction_load
        }
        return metric_map.get(trigger, 0.0)
    
    async def _execute_scaling_decision(self, 
                                      deployment: str,
                                      decision: ScalingDecision):
        """Execute the scaling decision"""
        if decision.action == ScalingAction.MAINTAIN:
            return
        
        self.logger.info(
            f"Scaling {deployment}: {decision.action.value} "
            f"({decision.current_replicas} -> {decision.target_replicas}) "
            f"Confidence: {decision.confidence:.2f} "
            f"Reason: {decision.reasoning}"
        )
        
        # Execute scaling
        success = await self.k8s_scaler.scale_deployment(
            deployment, decision.target_replicas
        )
        
        if success:
            # Update metrics
            self.scaling_actions_counter.labels(
                deployment=deployment,
                action=decision.action.value,
                trigger=",".join([t.value for t in decision.triggered_by])
            ).inc()
            
            self.current_replicas_gauge.labels(
                deployment=deployment
            ).set(decision.target_replicas)
            
            # Record scaling action for cooldown
            for trigger in decision.triggered_by:
                rule_name = f"{deployment}_{trigger.value}"
                self.last_scaling_actions[rule_name] = time.time()
            
            # Store scaling decision in Redis for monitoring
            scaling_event = {
                'timestamp': decision.target_replicas,
                'deployment': deployment,
                'action': decision.action.value,
                'current_replicas': decision.current_replicas,
                'target_replicas': decision.target_replicas,
                'triggered_by': [t.value for t in decision.triggered_by],
                'confidence': decision.confidence,
                'estimated_cost': decision.estimated_cost,
                'reasoning': decision.reasoning
            }
            
            self.redis_client.lpush(
                'scaling_events',
                json.dumps(scaling_event, default=str)
            )
            self.redis_client.ltrim('scaling_events', 0, 1000)  # Keep last 1000 events
        
        else:
            self.logger.error(f"Failed to execute scaling decision for {deployment}")
    
    async def _push_metrics(self):
        """Push metrics to Prometheus gateway"""
        try:
            push_to_gateway(
                self.prometheus_gateway,
                job='autoscaler',
                registry=self.registry
            )
        except Exception as e:
            self.logger.error(f"Failed to push metrics to Prometheus: {e}")
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add a new scaling rule"""
        self.scaling_rules.append(rule)
        self.logger.info(f"Added scaling rule: {rule.name}")
    
    def remove_scaling_rule(self, rule_name: str):
        """Remove a scaling rule by name"""
        self.scaling_rules = [r for r in self.scaling_rules if r.name != rule_name]
        self.logger.info(f"Removed scaling rule: {rule_name}")
    
    async def manual_scale(self, 
                          deployment: str, 
                          target_replicas: int,
                          reason: str = "Manual scaling") -> bool:
        """Manually scale a deployment"""
        success = await self.k8s_scaler.scale_deployment(deployment, target_replicas)
        
        if success:
            # Record manual scaling
            scaling_event = {
                'timestamp': datetime.now().isoformat(),
                'deployment': deployment,
                'action': 'manual_scale',
                'target_replicas': target_replicas,
                'reason': reason
            }
            
            self.redis_client.lpush(
                'scaling_events',
                json.dumps(scaling_event)
            )
            
            self.logger.info(f"Manual scaling: {deployment} to {target_replicas} replicas")
        
        return success


# Example usage and configuration
async def main():
    """Example usage of the AutoScaler"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize auto-scaler
    autoscaler = AutoScaler(
        redis_url="redis://localhost:6379",
        prometheus_gateway="localhost:9091"
    )
    
    # Add custom scaling rules
    volatility_rule = ScalingRule(
        name="Extreme Market Volatility",
        trigger=ScalingTrigger.MARKET_VOLATILITY,
        threshold_up=0.15,  # 15% volatility
        threshold_down=0.03,
        scale_up_factor=3.0,
        priority=10,
        cooldown_period=60  # React quickly to volatility
    )
    autoscaler.add_scaling_rule(volatility_rule)
    
    # Start auto-scaling (runs indefinitely)
    try:
        await autoscaler.start_autoscaler(interval=15)  # Check every 15 seconds
    except KeyboardInterrupt:
        autoscaler.stop_autoscaler()
        print("Auto-scaler stopped")


if __name__ == "__main__":
    asyncio.run(main())