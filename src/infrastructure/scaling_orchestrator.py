"""
Scaling Orchestrator - Unified Auto-Scaling Management

This module provides a unified interface for managing all scaling systems including
Kubernetes HPA, Docker Compose scaling, and predictive scaling algorithms.
It coordinates between different scaling mechanisms and provides intelligent
decision making based on market conditions and system state.
"""

import asyncio
import logging
import json
import time
import yaml
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from enum import Enum

from .auto_scaler import AutoScaler, ScalingRule, ScalingTrigger
from .docker_scaler import DockerAutoScaler, ServiceScalingConfig


class OrchestrationMode(Enum):
    KUBERNETES = "kubernetes"
    DOCKER_COMPOSE = "docker_compose"
    DOCKER_SWARM = "docker_swarm"
    HYBRID = "hybrid"


@dataclass
class GlobalScalingPolicy:
    """Global scaling policy configuration"""
    mode: OrchestrationMode
    max_total_replicas: int = 200
    cost_optimization: bool = True
    emergency_scaling: bool = True
    predictive_scaling: bool = True
    market_hours_only: bool = False
    scaling_aggressiveness: float = 1.0  # 0.5 = conservative, 2.0 = aggressive


@dataclass
class ServiceDefinition:
    """Service definition for scaling orchestration"""
    name: str
    type: str  # prediction-api, data-pipeline, model-server, etc.
    priority: int  # 1-10, higher = more important
    dependencies: List[str] = field(default_factory=list)
    scaling_config: Dict[str, Any] = field(default_factory=dict)
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    health_check: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalingEvent:
    """Record of scaling events for analysis"""
    timestamp: datetime
    service: str
    action: str
    from_replicas: int
    to_replicas: int
    trigger: str
    confidence: float
    cost_impact: float
    success: bool
    duration_ms: float


class MarketConditionAnalyzer:
    """
    Analyzes market conditions to inform scaling decisions
    """
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
    
    def get_market_condition(self) -> Dict[str, Any]:
        """Get current market condition analysis"""
        try:
            # Get market metrics from Redis
            volatility = float(self.redis_client.get('market_volatility_1h') or 0)
            volume = float(self.redis_client.get('trading_volume_1h') or 0)
            price_change = float(self.redis_client.get('price_change_1h') or 0)
            
            # Classify market condition
            if volatility > 0.05 or abs(price_change) > 0.03:
                condition = "HIGH_VOLATILITY"
                scaling_multiplier = 2.0
            elif volume > 10000:  # High volume threshold
                condition = "HIGH_VOLUME"
                scaling_multiplier = 1.5
            elif self._is_market_hours():
                condition = "NORMAL_TRADING"
                scaling_multiplier = 1.0
            else:
                condition = "LOW_ACTIVITY"
                scaling_multiplier = 0.5
            
            return {
                'condition': condition,
                'volatility': volatility,
                'volume': volume,
                'price_change': price_change,
                'scaling_multiplier': scaling_multiplier,
                'is_market_hours': self._is_market_hours(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market condition: {e}")
            return {
                'condition': 'UNKNOWN',
                'scaling_multiplier': 1.0,
                'is_market_hours': False
            }
    
    def _is_market_hours(self) -> bool:
        """Check if it's currently market trading hours (9:30 AM - 4:00 PM EST)"""
        from datetime import datetime
        import pytz
        
        try:
            est = pytz.timezone('US/Eastern')
            now = datetime.now(est)
            
            # Weekend check
            if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False
            
            # Market hours check (9:30 AM - 4:00 PM EST)
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            
            return market_open <= now <= market_close
            
        except Exception:
            # Default to market hours if timezone detection fails
            now = datetime.now()
            return 9 <= now.hour <= 16 and now.weekday() < 5


class CostOptimizer:
    """
    Optimizes scaling decisions based on cost considerations
    """
    
    def __init__(self):
        # Cost per replica per hour (in USD)
        self.cost_per_replica = {
            'prediction-api': 0.15,
            'data-pipeline': 0.10,
            'model-server': 0.25,
            'monitoring': 0.05,
            'default': 0.12
        }
        
        # Spot instance availability (mock data)
        self.spot_availability = {
            'high': 0.3,    # 30% of on-demand price
            'medium': 0.5,  # 50% of on-demand price
            'low': 0.8      # 80% of on-demand price
        }
    
    def calculate_scaling_cost(self, 
                             service: str,
                             current_replicas: int,
                             target_replicas: int,
                             duration_hours: float = 1.0) -> Dict[str, float]:
        """Calculate cost impact of scaling decision"""
        
        cost_per_hour = self.cost_per_replica.get(service, self.cost_per_replica['default'])
        replica_change = target_replicas - current_replicas
        
        hourly_cost_change = replica_change * cost_per_hour
        total_cost_change = hourly_cost_change * duration_hours
        
        # Consider spot instance savings
        spot_savings = 0.0
        if replica_change > 0:  # Scaling up
            spot_savings = total_cost_change * self.spot_availability['medium']
        
        return {
            'hourly_cost_change': hourly_cost_change,
            'total_cost_change': total_cost_change,
            'potential_spot_savings': spot_savings,
            'cost_per_replica': cost_per_hour,
            'net_cost_change': total_cost_change - spot_savings
        }
    
    def should_delay_scaling(self, 
                           cost_analysis: Dict[str, float],
                           urgency: float) -> bool:
        """Determine if scaling should be delayed for cost optimization"""
        
        # Never delay emergency scaling
        if urgency > 0.8:
            return False
        
        # Delay expensive scale-up during low urgency
        if (cost_analysis['hourly_cost_change'] > 5.0 and 
            urgency < 0.3):
            return True
        
        return False


class ScalingOrchestrator:
    """
    Main orchestrator that coordinates all scaling systems
    """
    
    def __init__(self, config_file: str = "scaling_config.yaml"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
        
        # Initialize components based on configuration
        self.k8s_scaler = None
        self.docker_scaler = None
        
        if self.config.mode in [OrchestrationMode.KUBERNETES, OrchestrationMode.HYBRID]:
            self.k8s_scaler = AutoScaler()
        
        if self.config.mode in [OrchestrationMode.DOCKER_COMPOSE, 
                               OrchestrationMode.DOCKER_SWARM, 
                               OrchestrationMode.HYBRID]:
            self.docker_scaler = DockerAutoScaler(
                use_swarm=(self.config.mode == OrchestrationMode.DOCKER_SWARM)
            )
        
        # Analytics and optimization
        import redis
        self.redis_client = redis.from_url("redis://localhost:6379")
        self.market_analyzer = MarketConditionAnalyzer(self.redis_client)
        self.cost_optimizer = CostOptimizer()
        
        # State tracking
        self.services = {}
        self.scaling_history = []
        self.active_scaling_operations = set()
        
        self.logger = logging.getLogger(__name__)
        self.running = False
    
    def _load_config(self) -> GlobalScalingPolicy:
        """Load configuration from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                    return GlobalScalingPolicy(**config_data.get('global_policy', {}))
        except Exception as e:
            logging.warning(f"Failed to load config: {e}. Using defaults.")
        
        return GlobalScalingPolicy(mode=OrchestrationMode.KUBERNETES)
    
    def _save_config(self):
        """Save current configuration to file"""
        try:
            config_data = {
                'global_policy': {
                    'mode': self.config.mode.value,
                    'max_total_replicas': self.config.max_total_replicas,
                    'cost_optimization': self.config.cost_optimization,
                    'emergency_scaling': self.config.emergency_scaling,
                    'predictive_scaling': self.config.predictive_scaling,
                    'market_hours_only': self.config.market_hours_only,
                    'scaling_aggressiveness': self.config.scaling_aggressiveness
                },
                'services': {name: service.__dict__ for name, service in self.services.items()}
            }
            
            with open(self.config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
                
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
    
    def register_service(self, service: ServiceDefinition):
        """Register a service for orchestrated scaling"""
        self.services[service.name] = service
        self.logger.info(f"Registered service: {service.name}")
        
        # Configure underlying scalers
        if self.k8s_scaler and service.scaling_config:
            # Add Kubernetes scaling rules
            for trigger, config in service.scaling_config.items():
                if trigger in ['cpu', 'memory', 'custom']:
                    rule = ScalingRule(
                        name=f"{service.name}_{trigger}",
                        trigger=ScalingTrigger(config.get('trigger', 'cpu_usage')),
                        threshold_up=config.get('threshold_up', 75),
                        threshold_down=config.get('threshold_down', 30),
                        priority=service.priority
                    )
                    self.k8s_scaler.add_scaling_rule(rule)
        
        if self.docker_scaler and service.scaling_config:
            # Add Docker scaling config
            docker_config = ServiceScalingConfig(
                service_name=service.name,
                min_replicas=service.scaling_config.get('min_replicas', 1),
                max_replicas=service.scaling_config.get('max_replicas', 10),
                target_cpu=service.scaling_config.get('target_cpu', 70),
                scale_up_threshold=service.scaling_config.get('scale_up_threshold', 80)
            )
            self.docker_scaler.add_service_config(docker_config)
    
    async def start_orchestrator(self, interval: int = 15):
        """Start the orchestration loop"""
        self.running = True
        self.logger.info("Starting scaling orchestrator")
        
        # Start underlying scalers
        tasks = []
        
        if self.k8s_scaler:
            tasks.append(asyncio.create_task(
                self.k8s_scaler.start_autoscaler(interval)
            ))
        
        if self.docker_scaler:
            tasks.append(asyncio.create_task(
                self.docker_scaler.start_autoscaler(interval)
            ))
        
        # Start orchestration loop
        tasks.append(asyncio.create_task(
            self._orchestration_loop(interval)
        ))
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Error in orchestrator: {e}")
        finally:
            self.running = False
    
    def stop_orchestrator(self):
        """Stop all scaling operations"""
        self.running = False
        
        if self.k8s_scaler:
            self.k8s_scaler.stop_autoscaler()
        
        if self.docker_scaler:
            self.docker_scaler.stop_autoscaler()
        
        self.logger.info("Scaling orchestrator stopped")
    
    async def _orchestration_loop(self, interval: int):
        """Main orchestration coordination loop"""
        while self.running:
            try:
                await self._orchestration_cycle()
                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error in orchestration cycle: {e}")
                await asyncio.sleep(interval)
    
    async def _orchestration_cycle(self):
        """Execute one orchestration cycle"""
        
        # Analyze market conditions
        market_condition = self.market_analyzer.get_market_condition()
        
        # Skip scaling during off-market hours if configured
        if (self.config.market_hours_only and 
            not market_condition.get('is_market_hours', True)):
            return
        
        # Global scaling budget management
        total_replicas = await self._get_total_replicas()
        available_scaling_budget = self.config.max_total_replicas - total_replicas
        
        if available_scaling_budget <= 0:
            self.logger.warning("Global replica limit reached, scaling constrained")
            return
        
        # Priority-based scaling decisions
        priority_services = sorted(
            self.services.values(),
            key=lambda s: s.priority,
            reverse=True
        )
        
        scaling_decisions = []
        
        for service in priority_services:
            if available_scaling_budget <= 0:
                break
            
            decision = await self._make_service_scaling_decision(
                service, market_condition, available_scaling_budget
            )
            
            if decision and decision['action'] != 'maintain':
                scaling_decisions.append(decision)
                
                # Reserve scaling budget
                replica_change = decision['target_replicas'] - decision['current_replicas']
                available_scaling_budget -= max(0, replica_change)
        
        # Execute scaling decisions
        for decision in scaling_decisions:
            await self._execute_orchestrated_scaling(decision)
        
        # Cleanup completed scaling operations
        self._cleanup_scaling_operations()
    
    async def _make_service_scaling_decision(self, 
                                           service: ServiceDefinition,
                                           market_condition: Dict[str, Any],
                                           available_budget: int) -> Optional[Dict[str, Any]]:
        """Make scaling decision for a specific service"""
        
        # Get current service state
        current_replicas = await self._get_service_replicas(service.name)
        if current_replicas == 0:
            return None
        
        # Calculate base scaling need
        base_decision = await self._calculate_base_scaling_need(service)
        if not base_decision:
            return None
        
        # Apply market condition adjustments
        market_multiplier = market_condition.get('scaling_multiplier', 1.0)
        adjusted_target = int(base_decision['target_replicas'] * 
                            market_multiplier * 
                            self.config.scaling_aggressiveness)
        
        # Enforce service limits
        min_replicas = service.scaling_config.get('min_replicas', 1)
        max_replicas = service.scaling_config.get('max_replicas', 20)
        adjusted_target = max(min_replicas, min(max_replicas, adjusted_target))
        
        # Enforce global budget
        replica_change = adjusted_target - current_replicas
        if replica_change > available_budget:
            adjusted_target = current_replicas + available_budget
        
        # Cost analysis
        cost_analysis = self.cost_optimizer.calculate_scaling_cost(
            service.name, current_replicas, adjusted_target
        )
        
        # Determine urgency
        urgency = self._calculate_scaling_urgency(service, base_decision)
        
        # Check if scaling should be delayed for cost optimization
        if (self.config.cost_optimization and 
            self.cost_optimizer.should_delay_scaling(cost_analysis, urgency)):
            return None
        
        return {
            'service': service.name,
            'action': 'scale_up' if adjusted_target > current_replicas else 'scale_down',
            'current_replicas': current_replicas,
            'target_replicas': adjusted_target,
            'urgency': urgency,
            'cost_analysis': cost_analysis,
            'market_condition': market_condition['condition'],
            'reasoning': base_decision.get('reasoning', 'Orchestrated scaling decision')
        }
    
    async def _calculate_base_scaling_need(self, 
                                         service: ServiceDefinition) -> Optional[Dict[str, Any]]:
        """Calculate base scaling need without market adjustments"""
        
        # This is a simplified version - in practice, you'd integrate with
        # the actual metrics from your scalers
        try:
            # Get service metrics from Redis
            cpu_usage = float(
                self.redis_client.get(f'{service.name}_avg_cpu') or 0
            )
            memory_usage = float(
                self.redis_client.get(f'{service.name}_avg_memory') or 0
            )
            request_rate = float(
                self.redis_client.get(f'{service.name}_request_rate') or 0
            )
            
            current_replicas = await self._get_service_replicas(service.name)
            
            # Simple scaling logic
            target_replicas = current_replicas
            reasoning = []
            
            if cpu_usage > 80 or memory_usage > 80:
                target_replicas = int(current_replicas * 1.5)
                reasoning.append(f"High resource usage (CPU: {cpu_usage}%, Memory: {memory_usage}%)")
            elif cpu_usage < 30 and memory_usage < 30 and current_replicas > 1:
                target_replicas = max(1, int(current_replicas * 0.7))
                reasoning.append(f"Low resource usage (CPU: {cpu_usage}%, Memory: {memory_usage}%)")
            
            if request_rate > 1000:  # High request rate
                target_replicas = max(target_replicas, int(current_replicas * 1.3))
                reasoning.append(f"High request rate: {request_rate} req/min")
            
            if target_replicas != current_replicas:
                return {
                    'target_replicas': target_replicas,
                    'reasoning': '; '.join(reasoning)
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error calculating scaling need for {service.name}: {e}")
            return None
    
    def _calculate_scaling_urgency(self, 
                                 service: ServiceDefinition,
                                 decision: Dict[str, Any]) -> float:
        """Calculate urgency score (0.0 - 1.0) for scaling decision"""
        
        urgency = 0.5  # Base urgency
        
        # Service priority influence
        urgency += (service.priority / 10) * 0.2
        
        # Replica change magnitude influence
        replica_change_ratio = abs(
            decision['target_replicas'] - decision.get('current_replicas', 1)
        ) / max(decision.get('current_replicas', 1), 1)
        
        urgency += min(0.3, replica_change_ratio)
        
        # Critical service types get higher urgency
        if service.type in ['prediction-api', 'model-server']:
            urgency += 0.2
        
        return min(1.0, urgency)
    
    async def _get_service_replicas(self, service_name: str) -> int:
        """Get current replica count for a service"""
        try:
            if self.k8s_scaler:
                # Try Kubernetes first
                dep = self.k8s_scaler.k8s_scaler.k8s_apps_v1.read_namespaced_deployment(
                    name=service_name,
                    namespace=self.k8s_scaler.k8s_scaler.namespace
                )
                return dep.spec.replicas
            elif self.docker_scaler:
                # Fall back to Docker
                if self.docker_scaler.use_swarm:
                    service_info = await self.docker_scaler.swarm_manager.get_swarm_service_info(service_name)
                    return service_info.get('replicas', 0) if service_info else 0
                else:
                    containers = await self.docker_scaler.compose_manager.get_service_containers(service_name)
                    return len(containers)
            
            return 1  # Default
            
        except Exception:
            return 1  # Default fallback
    
    async def _get_total_replicas(self) -> int:
        """Get total replica count across all services"""
        total = 0
        for service_name in self.services.keys():
            total += await self._get_service_replicas(service_name)
        return total
    
    async def _execute_orchestrated_scaling(self, decision: Dict[str, Any]):
        """Execute a scaling decision"""
        service_name = decision['service']
        
        if service_name in self.active_scaling_operations:
            self.logger.info(f"Scaling already in progress for {service_name}")
            return
        
        self.active_scaling_operations.add(service_name)
        start_time = time.time()
        
        try:
            self.logger.info(
                f"Orchestrated scaling: {service_name} "
                f"({decision['current_replicas']} -> {decision['target_replicas']}) "
                f"Reason: {decision['reasoning']}"
            )
            
            success = False
            
            # Execute scaling through appropriate backend
            if self.k8s_scaler:
                success = await self.k8s_scaler.manual_scale(
                    service_name, 
                    decision['target_replicas'],
                    f"Orchestrated: {decision['reasoning']}"
                )
            elif self.docker_scaler:
                success = await self.docker_scaler.manual_scale_service(
                    service_name,
                    decision['target_replicas']
                )
            
            # Record scaling event
            duration_ms = (time.time() - start_time) * 1000
            event = ScalingEvent(
                timestamp=datetime.now(),
                service=service_name,
                action=decision['action'],
                from_replicas=decision['current_replicas'],
                to_replicas=decision['target_replicas'],
                trigger='orchestrator',
                confidence=decision.get('urgency', 0.5),
                cost_impact=decision['cost_analysis']['net_cost_change'],
                success=success,
                duration_ms=duration_ms
            )
            
            self.scaling_history.append(event)
            
            # Keep history limited
            if len(self.scaling_history) > 1000:
                self.scaling_history = self.scaling_history[-500:]
            
            # Store in Redis for monitoring
            self.redis_client.lpush(
                'orchestrator_scaling_events',
                json.dumps({
                    'timestamp': event.timestamp.isoformat(),
                    'service': event.service,
                    'action': event.action,
                    'from_replicas': event.from_replicas,
                    'to_replicas': event.to_replicas,
                    'success': event.success,
                    'cost_impact': event.cost_impact
                })
            )
            self.redis_client.ltrim('orchestrator_scaling_events', 0, 500)
            
        except Exception as e:
            self.logger.error(f"Failed to execute scaling for {service_name}: {e}")
        finally:
            self.active_scaling_operations.discard(service_name)
    
    def _cleanup_scaling_operations(self):
        """Clean up completed scaling operations"""
        # This is handled by removing from active_scaling_operations in _execute_orchestrated_scaling
        pass
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get current orchestrator status"""
        return {
            'running': self.running,
            'mode': self.config.mode.value,
            'services': len(self.services),
            'active_operations': len(self.active_scaling_operations),
            'total_scaling_events': len(self.scaling_history),
            'config': {
                'max_total_replicas': self.config.max_total_replicas,
                'cost_optimization': self.config.cost_optimization,
                'emergency_scaling': self.config.emergency_scaling,
                'predictive_scaling': self.config.predictive_scaling,
                'scaling_aggressiveness': self.config.scaling_aggressiveness
            }
        }
    
    async def emergency_scale_all(self, multiplier: float = 2.0):
        """Emergency scaling for all critical services"""
        if not self.config.emergency_scaling:
            self.logger.warning("Emergency scaling is disabled")
            return
        
        self.logger.warning(f"Initiating emergency scaling (multiplier: {multiplier})")
        
        critical_services = [s for s in self.services.values() if s.priority >= 7]
        
        for service in critical_services:
            current_replicas = await self._get_service_replicas(service.name)
            target_replicas = min(
                service.scaling_config.get('max_replicas', 20),
                int(current_replicas * multiplier)
            )
            
            decision = {
                'service': service.name,
                'action': 'emergency_scale',
                'current_replicas': current_replicas,
                'target_replicas': target_replicas,
                'urgency': 1.0,
                'cost_analysis': {'net_cost_change': 0},
                'reasoning': f'Emergency scaling (x{multiplier})'
            }
            
            await self._execute_orchestrated_scaling(decision)


# Example configuration and usage
def create_example_config():
    """Create example scaling configuration"""
    
    orchestrator = ScalingOrchestrator("scaling_config.yaml")
    
    # Configure global policy
    orchestrator.config = GlobalScalingPolicy(
        mode=OrchestrationMode.KUBERNETES,
        max_total_replicas=500,
        cost_optimization=True,
        emergency_scaling=True,
        predictive_scaling=True,
        scaling_aggressiveness=1.2
    )
    
    # Register services
    services = [
        ServiceDefinition(
            name="prediction-api",
            type="api",
            priority=9,
            dependencies=["model-server", "data-pipeline"],
            scaling_config={
                'min_replicas': 3,
                'max_replicas': 50,
                'target_cpu': 70,
                'scale_up_threshold': 80,
                'scale_down_threshold': 30
            },
            resource_limits={
                'cpu': '2000m',
                'memory': '4Gi'
            },
            health_check={
                'path': '/health',
                'port': 8080,
                'timeout': 5
            }
        ),
        ServiceDefinition(
            name="model-server",
            type="inference",
            priority=10,  # Highest priority
            scaling_config={
                'min_replicas': 5,
                'max_replicas': 100,
                'target_cpu': 65,
                'scale_up_threshold': 75,
                'scale_down_threshold': 25
            }
        ),
        ServiceDefinition(
            name="data-pipeline",
            type="processing",
            priority=7,
            scaling_config={
                'min_replicas': 2,
                'max_replicas': 20,
                'target_cpu': 75,
                'scale_up_threshold': 85,
                'scale_down_threshold': 35
            }
        )
    ]
    
    for service in services:
        orchestrator.register_service(service)
    
    return orchestrator


# Main execution example
async def main():
    """Example usage of the Scaling Orchestrator"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and configure orchestrator
    orchestrator = create_example_config()
    
    # Start orchestration
    try:
        await orchestrator.start_orchestrator(interval=10)
    except KeyboardInterrupt:
        orchestrator.stop_orchestrator()
        print("Scaling orchestrator stopped")


if __name__ == "__main__":
    asyncio.run(main())