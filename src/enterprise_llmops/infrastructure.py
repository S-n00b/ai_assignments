"""
Infrastructure Management for Enterprise LLMOps Platform

This module provides comprehensive infrastructure management capabilities
including Kubernetes orchestration, Docker container management, and
cloud resource provisioning.

Key Features:
- Kubernetes deployment and management
- Docker container orchestration
- Terraform infrastructure as code
- Auto-scaling and load balancing
- Resource monitoring and optimization
- Disaster recovery and backup
"""

import asyncio
import logging
import time
import json
import yaml
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import docker
import subprocess
import psutil
import aiohttp


@dataclass
class KubernetesConfig:
    """Configuration for Kubernetes deployment."""
    namespace: str
    replicas: int
    image: str
    port: int
    env_vars: Dict[str, str] = None
    resource_limits: Dict[str, str] = None
    resource_requests: Dict[str, str] = None
    
    def __post_init__(self):
        if self.env_vars is None:
            self.env_vars = {}
        if self.resource_limits is None:
            self.resource_limits = {"cpu": "2", "memory": "4Gi"}
        if self.resource_requests is None:
            self.resource_requests = {"cpu": "1", "memory": "2Gi"}


@dataclass
class DockerConfig:
    """Configuration for Docker deployment."""
    image_name: str
    tag: str
    ports: Dict[int, int]
    volumes: Dict[str, str] = None
    env_vars: Dict[str, str] = None
    restart_policy: str = "unless-stopped"
    
    def __post_init__(self):
        if self.volumes is None:
            self.volumes = {}
        if self.env_vars is None:
            self.env_vars = {}


@dataclass
class ResourceUsage:
    """Resource usage information."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_in: float
    network_out: float
    timestamp: datetime


class InfrastructureManager:
    """
    Comprehensive infrastructure manager for Enterprise LLMOps platform.
    
    This class provides infrastructure management capabilities including
    Kubernetes orchestration, Docker container management, and resource monitoring.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the infrastructure manager."""
        self.config = config
        self.logger = self._setup_logging()
        self.docker_client = None
        self.k8s_config_path = Path("src/enterprise_llmops/infrastructure/kubernetes")
        
        # Initialize Docker client
        self._init_docker()
        
        # Resource monitoring
        self.resource_usage = {}
        self.monitoring_active = False
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for infrastructure management."""
        logger = logging.getLogger("infrastructure")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _init_docker(self):
        """Initialize Docker client."""
        try:
            self.docker_client = docker.from_env()
            self.logger.info("Docker client initialized")
        except Exception as e:
            self.logger.warning(f"Docker client initialization failed: {e}")
            self.docker_client = None
    
    def build_docker_image(self, dockerfile_path: str, image_name: str, tag: str = "latest") -> bool:
        """Build a Docker image."""
        try:
            if not self.docker_client:
                self.logger.error("Docker client not available")
                return False
            
            self.logger.info(f"Building Docker image: {image_name}:{tag}")
            
            image, build_logs = self.docker_client.images.build(
                path=dockerfile_path,
                tag=f"{image_name}:{tag}",
                rm=True
            )
            
            self.logger.info(f"Docker image built successfully: {image.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to build Docker image: {e}")
            return False
    
    def run_docker_container(self, config: DockerConfig) -> Optional[str]:
        """Run a Docker container."""
        try:
            if not self.docker_client:
                self.logger.error("Docker client not available")
                return None
            
            self.logger.info(f"Starting Docker container: {config.image_name}:{config.tag}")
            
            # Prepare port mapping
            ports = {}
            for container_port, host_port in config.ports.items():
                ports[f"{container_port}/tcp"] = host_port
            
            # Run container
            container = self.docker_client.containers.run(
                f"{config.image_name}:{config.tag}",
                ports=ports,
                volumes=config.volumes,
                environment=config.env_vars,
                restart_policy=config.restart_policy,
                detach=True
            )
            
            self.logger.info(f"Docker container started: {container.id}")
            return container.id
            
        except Exception as e:
            self.logger.error(f"Failed to start Docker container: {e}")
            return None
    
    def stop_docker_container(self, container_id: str) -> bool:
        """Stop a Docker container."""
        try:
            if not self.docker_client:
                self.logger.error("Docker client not available")
                return False
            
            container = self.docker_client.containers.get(container_id)
            container.stop()
            
            self.logger.info(f"Docker container stopped: {container_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop Docker container: {e}")
            return False
    
    def list_docker_containers(self) -> List[Dict[str, Any]]:
        """List all Docker containers."""
        try:
            if not self.docker_client:
                return []
            
            containers = self.docker_client.containers.list(all=True)
            
            container_info = []
            for container in containers:
                info = {
                    "id": container.id,
                    "name": container.name,
                    "image": container.image.tags[0] if container.image.tags else container.image.id,
                    "status": container.status,
                    "created": container.attrs["Created"],
                    "ports": container.attrs["NetworkSettings"]["Ports"]
                }
                container_info.append(info)
            
            return container_info
            
        except Exception as e:
            self.logger.error(f"Failed to list Docker containers: {e}")
            return []
    
    def generate_kubernetes_deployment(self, config: KubernetesConfig) -> str:
        """Generate Kubernetes deployment YAML."""
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": config.namespace,
                "namespace": config.namespace,
                "labels": {
                    "app": config.namespace,
                    "component": "llmops"
                }
            },
            "spec": {
                "replicas": config.replicas,
                "selector": {
                    "matchLabels": {
                        "app": config.namespace
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": config.namespace
                        }
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": config.namespace,
                                "image": config.image,
                                "ports": [
                                    {
                                        "containerPort": config.port,
                                        "name": "http"
                                    }
                                ],
                                "env": [
                                    {"name": k, "value": v} for k, v in config.env_vars.items()
                                ],
                                "resources": {
                                    "limits": config.resource_limits,
                                    "requests": config.resource_requests
                                },
                                "livenessProbe": {
                                    "httpGet": {
                                        "path": "/health",
                                        "port": config.port
                                    },
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 30
                                },
                                "readinessProbe": {
                                    "httpGet": {
                                        "path": "/health",
                                        "port": config.port
                                    },
                                    "initialDelaySeconds": 10,
                                    "periodSeconds": 10
                                }
                            }
                        ]
                    }
                }
            }
        }
        
        return yaml.dump(deployment, default_flow_style=False)
    
    def deploy_to_kubernetes(self, deployment_yaml: str) -> bool:
        """Deploy to Kubernetes using kubectl."""
        try:
            # Write deployment to temporary file
            temp_file = Path("temp_deployment.yaml")
            temp_file.write_text(deployment_yaml)
            
            # Apply deployment
            result = subprocess.run(
                ["kubectl", "apply", "-f", str(temp_file)],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Clean up
            temp_file.unlink()
            
            self.logger.info(f"Kubernetes deployment applied: {result.stdout}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to deploy to Kubernetes: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"Kubernetes deployment error: {e}")
            return False
    
    def scale_kubernetes_deployment(self, namespace: str, deployment_name: str, replicas: int) -> bool:
        """Scale a Kubernetes deployment."""
        try:
            result = subprocess.run(
                ["kubectl", "scale", f"deployment/{deployment_name}", f"--replicas={replicas}", "-n", namespace],
                capture_output=True,
                text=True,
                check=True
            )
            
            self.logger.info(f"Kubernetes deployment scaled: {result.stdout}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to scale Kubernetes deployment: {e.stderr}")
            return False
    
    def get_kubernetes_status(self, namespace: str) -> Dict[str, Any]:
        """Get Kubernetes deployment status."""
        try:
            # Get pods
            result = subprocess.run(
                ["kubectl", "get", "pods", "-n", namespace, "-o", "json"],
                capture_output=True,
                text=True,
                check=True
            )
            
            pods_data = json.loads(result.stdout)
            
            # Get services
            result = subprocess.run(
                ["kubectl", "get", "services", "-n", namespace, "-o", "json"],
                capture_output=True,
                text=True,
                check=True
            )
            
            services_data = json.loads(result.stdout)
            
            return {
                "pods": pods_data,
                "services": services_data,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get Kubernetes status: {e}")
            return {}
    
    async def collect_resource_usage(self) -> ResourceUsage:
        """Collect system resource usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Network usage (simplified)
            network = psutil.net_io_counters()
            network_in = network.bytes_recv
            network_out = network.bytes_sent
            
            usage = ResourceUsage(
                cpu_usage=cpu_percent,
                memory_usage=memory_percent,
                disk_usage=disk_percent,
                network_in=network_in,
                network_out=network_out,
                timestamp=datetime.now()
            )
            
            return usage
            
        except Exception as e:
            self.logger.error(f"Failed to collect resource usage: {e}")
            return ResourceUsage(
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_in=0.0,
                network_out=0.0,
                timestamp=datetime.now()
            )
    
    async def resource_monitoring_loop(self):
        """Background resource monitoring loop."""
        self.logger.info("Starting resource monitoring...")
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                usage = await self.collect_resource_usage()
                self.resource_usage[usage.timestamp] = usage
                
                # Keep only last 100 measurements
                if len(self.resource_usage) > 100:
                    oldest_key = min(self.resource_usage.keys())
                    del self.resource_usage[oldest_key]
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(60)
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get resource usage summary."""
        if not self.resource_usage:
            return {}
        
        latest_usage = max(self.resource_usage.values(), key=lambda x: x.timestamp)
        
        return {
            "current": asdict(latest_usage),
            "history_count": len(self.resource_usage),
            "timestamp": datetime.now().isoformat()
        }
    
    def deploy_enterprise_stack(self) -> bool:
        """Deploy the complete enterprise stack."""
        self.logger.info("Deploying enterprise stack...")
        
        try:
            # Deploy Ollama
            ollama_config = KubernetesConfig(
                namespace="llmops-enterprise",
                replicas=1,
                image="ollama/ollama:latest",
                port=11434,
                env_vars={"OLLAMA_HOST": "0.0.0.0"}
            )
            
            ollama_yaml = self.generate_kubernetes_deployment(ollama_config)
            if not self.deploy_to_kubernetes(ollama_yaml):
                self.logger.error("Failed to deploy Ollama")
                return False
            
            # Deploy MLflow
            mlflow_config = KubernetesConfig(
                namespace="llmops-enterprise",
                replicas=2,
                image="python:3.9-slim",
                port=5000,
                env_vars={
                    "MLFLOW_TRACKING_URI": "http://localhost:5000"
                }
            )
            
            mlflow_yaml = self.generate_kubernetes_deployment(mlflow_config)
            if not self.deploy_to_kubernetes(mlflow_yaml):
                self.logger.error("Failed to deploy MLflow")
                return False
            
            # Deploy Chroma
            chroma_config = KubernetesConfig(
                namespace="llmops-enterprise",
                replicas=2,
                image="chromadb/chroma:latest",
                port=8081,
                env_vars={
                    "CHROMA_SERVER_HOST": "0.0.0.0",
                    "CHROMA_SERVER_HTTP_PORT": "8081"
                }
            )
            
            chroma_yaml = self.generate_kubernetes_deployment(chroma_config)
            if not self.deploy_to_kubernetes(chroma_yaml):
                self.logger.error("Failed to deploy Chroma")
                return False
            
            self.logger.info("Enterprise stack deployed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to deploy enterprise stack: {e}")
            return False
    
    async def start(self):
        """Start the infrastructure manager."""
        self.logger.info("Starting Infrastructure Manager...")
        
        # Start resource monitoring
        self.monitoring_task = asyncio.create_task(self.resource_monitoring_loop())
        
        self.logger.info("Infrastructure Manager started successfully")
    
    async def stop(self):
        """Stop the infrastructure manager."""
        self.logger.info("Stopping Infrastructure Manager...")
        
        # Stop resource monitoring
        self.monitoring_active = False
        if hasattr(self, 'monitoring_task'):
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Infrastructure Manager stopped")
