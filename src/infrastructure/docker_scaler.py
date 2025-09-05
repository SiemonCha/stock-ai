"""
Docker-based Auto-Scaling for Non-Kubernetes Environments

This module provides intelligent container scaling using Docker Compose and Docker Swarm
for environments that don't have Kubernetes available. It includes container health monitoring,
service discovery, and load balancing capabilities.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import docker
import yaml
import subprocess
from pathlib import Path
import requests
import psutil
from concurrent.futures import ThreadPoolExecutor


@dataclass
class ContainerMetrics:
    container_id: str
    name: str
    cpu_usage: float
    memory_usage: float
    memory_limit: int
    network_io: Dict[str, int]
    running: bool
    healthy: bool
    last_updated: float


@dataclass
class ServiceScalingConfig:
    service_name: str
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu: float = 70.0
    target_memory: float = 80.0
    scale_up_threshold: float = 85.0
    scale_down_threshold: float = 30.0
    cooldown_seconds: int = 300


class DockerComposeManager:
    """
    Manages Docker Compose services with dynamic scaling
    """
    
    def __init__(self, compose_file: str = "docker-compose.yml"):
        self.compose_file = Path(compose_file)
        self.docker_client = docker.from_env()
        self.logger = logging.getLogger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def load_compose_config(self) -> Dict[str, Any]:
        """Load Docker Compose configuration"""
        try:
            with open(self.compose_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load compose file: {e}")
            return {}
    
    def save_compose_config(self, config: Dict[str, Any]):
        """Save Docker Compose configuration"""
        try:
            with open(self.compose_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        except Exception as e:
            self.logger.error(f"Failed to save compose file: {e}")
    
    async def scale_service(self, 
                           service_name: str, 
                           replica_count: int) -> bool:
        """Scale a Docker Compose service"""
        try:
            # Use docker-compose scale command
            cmd = [
                'docker-compose',
                '-f', str(self.compose_file),
                'scale',
                f'{service_name}={replica_count}'
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.logger.info(f"Scaled {service_name} to {replica_count} replicas")
                return True
            else:
                self.logger.error(f"Failed to scale {service_name}: {stderr.decode()}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error scaling service {service_name}: {e}")
            return False
    
    async def get_service_containers(self, service_name: str) -> List[str]:
        """Get container IDs for a specific service"""
        try:
            containers = []
            for container in self.docker_client.containers.list():
                labels = container.labels
                if labels.get('com.docker.compose.service') == service_name:
                    containers.append(container.id)
            return containers
        except Exception as e:
            self.logger.error(f"Error getting containers for {service_name}: {e}")
            return []
    
    async def get_container_metrics(self, container_id: str) -> Optional[ContainerMetrics]:
        """Get metrics for a specific container"""
        try:
            container = self.docker_client.containers.get(container_id)
            stats = container.stats(stream=False)
            
            # Calculate CPU usage
            cpu_delta = (stats['cpu_stats']['cpu_usage']['total_usage'] - 
                        stats['precpu_stats']['cpu_usage']['total_usage'])
            system_cpu_delta = (stats['cpu_stats']['system_cpu_usage'] - 
                              stats['precpu_stats']['system_cpu_usage'])
            
            cpu_usage = 0.0
            if system_cpu_delta > 0:
                cpu_usage = (cpu_delta / system_cpu_delta) * \
                          len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100.0
            
            # Calculate memory usage
            memory_usage = stats['memory_stats']['usage']
            memory_limit = stats['memory_stats']['limit']
            memory_percent = (memory_usage / memory_limit) * 100.0
            
            # Network I/O
            networks = stats.get('networks', {})
            network_io = {}
            for interface, data in networks.items():
                network_io[interface] = {
                    'rx_bytes': data['rx_bytes'],
                    'tx_bytes': data['tx_bytes']
                }
            
            return ContainerMetrics(
                container_id=container_id,
                name=container.name,
                cpu_usage=cpu_usage,
                memory_usage=memory_percent,
                memory_limit=memory_limit,
                network_io=network_io,
                running=container.status == 'running',
                healthy=self._check_container_health(container),
                last_updated=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Error getting metrics for container {container_id}: {e}")
            return None
    
    def _check_container_health(self, container) -> bool:
        """Check if container is healthy"""
        try:
            # Check health status if available
            health = container.attrs.get('State', {}).get('Health', {})
            if health:
                return health.get('Status') == 'healthy'
            
            # Fallback: check if container is running
            return container.status == 'running'
            
        except Exception:
            return False


class DockerSwarmManager:
    """
    Manages Docker Swarm services with advanced scaling
    """
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.logger = logging.getLogger(__name__)
        
        # Check if Swarm is initialized
        try:
            self.docker_client.swarm.attrs
        except Exception:
            self.logger.warning("Docker Swarm not initialized. Some features may not work.")
    
    async def scale_swarm_service(self, 
                                 service_name: str, 
                                 replica_count: int) -> bool:
        """Scale a Docker Swarm service"""
        try:
            services = self.docker_client.services.list(
                filters={'name': service_name}
            )
            
            if not services:
                self.logger.error(f"Service {service_name} not found")
                return False
            
            service = services[0]
            service.update(mode=docker.types.ServiceMode('replicated', replica_count))
            
            self.logger.info(f"Scaled Swarm service {service_name} to {replica_count} replicas")
            return True
            
        except Exception as e:
            self.logger.error(f"Error scaling Swarm service {service_name}: {e}")
            return False
    
    async def get_swarm_service_info(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a Swarm service"""
        try:
            services = self.docker_client.services.list(
                filters={'name': service_name}
            )
            
            if not services:
                return None
            
            service = services[0]
            spec = service.attrs['Spec']
            status = service.attrs.get('ServiceStatus', {})
            
            return {
                'name': service.name,
                'replicas': spec.get('Mode', {}).get('Replicated', {}).get('Replicas', 0),
                'running_tasks': status.get('RunningTasks', 0),
                'desired_tasks': status.get('DesiredTasks', 0),
                'completed_tasks': status.get('CompletedTasks', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting Swarm service info: {e}")
            return None


class ContainerLoadBalancer:
    """
    Simple load balancer for Docker containers
    """
    
    def __init__(self, service_port_map: Dict[str, int]):
        self.service_port_map = service_port_map
        self.container_health = {}
        self.logger = logging.getLogger(__name__)
    
    async def health_check_containers(self, containers: List[str]):
        """Perform health checks on containers"""
        for container_id in containers:
            try:
                container = docker.from_env().containers.get(container_id)
                
                # Get container IP
                networks = container.attrs['NetworkSettings']['Networks']
                container_ip = None
                for network_name, network_info in networks.items():
                    if network_info.get('IPAddress'):
                        container_ip = network_info['IPAddress']
                        break
                
                if container_ip:
                    # Simple HTTP health check
                    service_name = container.labels.get('com.docker.compose.service')
                    port = self.service_port_map.get(service_name, 8080)
                    
                    try:
                        response = requests.get(
                            f'http://{container_ip}:{port}/health',
                            timeout=2
                        )
                        healthy = response.status_code == 200
                    except:
                        healthy = False
                    
                    self.container_health[container_id] = {
                        'healthy': healthy,
                        'ip': container_ip,
                        'port': port,
                        'last_check': time.time()
                    }
                    
            except Exception as e:
                self.logger.error(f"Health check failed for {container_id}: {e}")
                self.container_health[container_id] = {
                    'healthy': False,
                    'last_check': time.time()
                }
    
    def get_healthy_containers(self, service_name: str) -> List[str]:
        """Get list of healthy containers for a service"""
        healthy = []
        for container_id, health_info in self.container_health.items():
            if health_info.get('healthy', False):
                try:
                    container = docker.from_env().containers.get(container_id)
                    if container.labels.get('com.docker.compose.service') == service_name:
                        healthy.append(container_id)
                except:
                    continue
        return healthy


class DockerAutoScaler:
    """
    Main Docker-based auto-scaler
    """
    
    def __init__(self, 
                 compose_file: str = "docker-compose.yml",
                 use_swarm: bool = False):
        self.compose_manager = DockerComposeManager(compose_file)
        self.swarm_manager = DockerSwarmManager() if use_swarm else None
        self.load_balancer = ContainerLoadBalancer({
            'prediction-api': 8080,
            'data-pipeline': 8081,
            'model-server': 8082
        })
        
        self.use_swarm = use_swarm
        self.scaling_configs = {}
        self.metrics_history = {}
        self.last_scaling_actions = {}
        
        self.logger = logging.getLogger(__name__)
        self.running = False
    
    def add_service_config(self, config: ServiceScalingConfig):
        """Add scaling configuration for a service"""
        self.scaling_configs[config.service_name] = config
        self.logger.info(f"Added scaling config for {config.service_name}")
    
    async def start_autoscaler(self, interval: int = 30):
        """Start the Docker auto-scaling loop"""
        self.running = True
        self.logger.info("Starting Docker auto-scaler")
        
        while self.running:
            try:
                await self._scaling_cycle()
                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error in Docker scaling cycle: {e}")
                await asyncio.sleep(interval)
    
    def stop_autoscaler(self):
        """Stop the auto-scaling loop"""
        self.running = False
        self.logger.info("Stopping Docker auto-scaler")
    
    async def _scaling_cycle(self):
        """Execute one complete Docker scaling cycle"""
        for service_name, config in self.scaling_configs.items():
            try:
                await self._scale_service_if_needed(service_name, config)
            except Exception as e:
                self.logger.error(f"Error scaling service {service_name}: {e}")
    
    async def _scale_service_if_needed(self, 
                                     service_name: str,
                                     config: ServiceScalingConfig):
        """Scale a service based on its metrics"""
        
        # Get current containers
        if self.use_swarm:
            service_info = await self.swarm_manager.get_swarm_service_info(service_name)
            if not service_info:
                return
            current_replicas = service_info['replicas']
            containers = []  # Swarm manages containers differently
        else:
            containers = await self.compose_manager.get_service_containers(service_name)
            current_replicas = len(containers)
        
        if current_replicas == 0:
            return
        
        # Collect metrics
        total_cpu = 0.0
        total_memory = 0.0
        healthy_containers = 0
        
        if not self.use_swarm:
            for container_id in containers:
                metrics = await self.compose_manager.get_container_metrics(container_id)
                if metrics and metrics.healthy:
                    total_cpu += metrics.cpu_usage
                    total_memory += metrics.memory_usage
                    healthy_containers += 1
        
        if healthy_containers == 0 and not self.use_swarm:
            self.logger.warning(f"No healthy containers found for {service_name}")
            return
        
        # Calculate average metrics
        if not self.use_swarm:
            avg_cpu = total_cpu / healthy_containers if healthy_containers > 0 else 0
            avg_memory = total_memory / healthy_containers if healthy_containers > 0 else 0
        else:
            # For Swarm, use system metrics as approximation
            avg_cpu = psutil.cpu_percent(interval=1)
            avg_memory = psutil.virtual_memory().percent
        
        # Store metrics history
        if service_name not in self.metrics_history:
            self.metrics_history[service_name] = []
        
        self.metrics_history[service_name].append({
            'timestamp': time.time(),
            'cpu': avg_cpu,
            'memory': avg_memory,
            'replicas': current_replicas
        })
        
        # Keep only last 100 entries
        if len(self.metrics_history[service_name]) > 100:
            self.metrics_history[service_name] = self.metrics_history[service_name][-50:]
        
        # Check cooldown
        last_action_time = self.last_scaling_actions.get(service_name, 0)
        if time.time() - last_action_time < config.cooldown_seconds:
            return
        
        # Determine scaling action
        target_replicas = current_replicas
        scale_reason = ""
        
        if (avg_cpu > config.scale_up_threshold or 
            avg_memory > config.scale_up_threshold):
            target_replicas = min(config.max_replicas, 
                                int(current_replicas * 1.5))
            scale_reason = f"High resource usage (CPU: {avg_cpu:.1f}%, Memory: {avg_memory:.1f}%)"
            
        elif (avg_cpu < config.scale_down_threshold and 
              avg_memory < config.scale_down_threshold and
              current_replicas > config.min_replicas):
            target_replicas = max(config.min_replicas, 
                                int(current_replicas * 0.7))
            scale_reason = f"Low resource usage (CPU: {avg_cpu:.1f}%, Memory: {avg_memory:.1f}%)"
        
        # Execute scaling if needed
        if target_replicas != current_replicas:
            self.logger.info(
                f"Scaling {service_name}: {current_replicas} -> {target_replicas} "
                f"({scale_reason})"
            )
            
            success = False
            if self.use_swarm:
                success = await self.swarm_manager.scale_swarm_service(
                    service_name, target_replicas
                )
            else:
                success = await self.compose_manager.scale_service(
                    service_name, target_replicas
                )
            
            if success:
                self.last_scaling_actions[service_name] = time.time()
                
                # Wait for containers to start/stop
                await asyncio.sleep(10)
                
                # Update load balancer health checks
                if not self.use_swarm:
                    new_containers = await self.compose_manager.get_service_containers(service_name)
                    await self.load_balancer.health_check_containers(new_containers)
    
    async def manual_scale_service(self, 
                                  service_name: str, 
                                  replica_count: int) -> bool:
        """Manually scale a service"""
        if self.use_swarm:
            return await self.swarm_manager.scale_swarm_service(
                service_name, replica_count
            )
        else:
            return await self.compose_manager.scale_service(
                service_name, replica_count
            )
    
    def get_service_metrics(self, service_name: str) -> Dict[str, Any]:
        """Get current metrics for a service"""
        history = self.metrics_history.get(service_name, [])
        if not history:
            return {}
        
        latest = history[-1]
        return {
            'current_cpu': latest['cpu'],
            'current_memory': latest['memory'],
            'current_replicas': latest['replicas'],
            'avg_cpu_5min': np.mean([m['cpu'] for m in history[-10:]]) if len(history) >= 10 else latest['cpu'],
            'avg_memory_5min': np.mean([m['memory'] for m in history[-10:]]) if len(history) >= 10 else latest['memory'],
            'last_updated': latest['timestamp']
        }


# Example usage and testing
async def main():
    """Example usage of Docker AutoScaler"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize Docker auto-scaler
    autoscaler = DockerAutoScaler(
        compose_file="docker-compose.yml",
        use_swarm=False  # Set to True if using Docker Swarm
    )
    
    # Configure services for auto-scaling
    services_config = [
        ServiceScalingConfig(
            service_name="prediction-api",
            min_replicas=2,
            max_replicas=20,
            target_cpu=70.0,
            scale_up_threshold=80.0,
            scale_down_threshold=30.0
        ),
        ServiceScalingConfig(
            service_name="data-pipeline",
            min_replicas=1,
            max_replicas=10,
            target_cpu=60.0,
            scale_up_threshold=75.0,
            scale_down_threshold=25.0
        ),
        ServiceScalingConfig(
            service_name="model-server",
            min_replicas=3,
            max_replicas=50,
            target_cpu=65.0,
            scale_up_threshold=85.0,
            scale_down_threshold=35.0,
            cooldown_seconds=180  # Faster scaling for model servers
        )
    ]
    
    for config in services_config:
        autoscaler.add_service_config(config)
    
    # Start auto-scaling
    try:
        await autoscaler.start_autoscaler(interval=20)
    except KeyboardInterrupt:
        autoscaler.stop_autoscaler()
        print("Docker auto-scaler stopped")


if __name__ == "__main__":
    asyncio.run(main())