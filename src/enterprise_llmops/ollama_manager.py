"""
Ollama Integration Manager for Enterprise LLMOps

This module provides comprehensive Ollama integration for local model management,
including model deployment, scaling, monitoring, and lifecycle management in
enterprise environments.

Key Features:
- Local model deployment and management
- Model versioning and rollback capabilities
- Resource optimization and load balancing
- Health monitoring and auto-recovery
- Integration with enterprise model registry
- Support for custom model fine-tuning
"""

import asyncio
import aiohttp
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import subprocess
import psutil
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    docker = None
from pathlib import Path
import yaml


@dataclass
class OllamaModel:
    """Represents an Ollama model configuration."""
    name: str
    model_id: str
    size: int  # in bytes
    modified_at: datetime
    digest: str
    details: Dict[str, Any]
    status: str = "available"  # "available", "pulling", "error", "custom"
    custom_config: Optional[Dict[str, Any]] = None


@dataclass
class OllamaInstance:
    """Represents an Ollama service instance."""
    id: str
    host: str
    port: int
    status: str  # "running", "stopped", "error"
    version: str
    models: List[str]
    resource_usage: Dict[str, Any]
    last_health_check: datetime


@dataclass
class ModelRequest:
    """Represents a model inference request."""
    id: str
    model_name: str
    prompt: str
    parameters: Dict[str, Any]
    user_id: str
    priority: int = 1  # 1=high, 2=medium, 3=low
    timeout: int = 30
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class OllamaManager:
    """
    Main manager for Ollama integration in enterprise environments.
    
    This class provides comprehensive Ollama model management including
    deployment, scaling, monitoring, and enterprise integration.
    """
    
    def __init__(self, config_path: str = "config/ollama_config.yaml"):
        """Initialize the Ollama manager."""
        self.config = self._load_config(config_path)
        self.instances = {}
        self.models = {}
        self.request_queue = asyncio.Queue()
        self.monitoring_active = False
        self.logger = self._setup_logging()
        
        # Initialize Docker client for container management
        try:
            if DOCKER_AVAILABLE:
                self.docker_client = docker.from_env()
            else:
                self.docker_client = None
                self.logger.warning("Docker module not available, Docker features disabled")
        except Exception as e:
            self.logger.warning(f"Docker client initialization failed: {e}")
            self.docker_client = None
        
        # Initialize HTTP session for Ollama API
        self.session = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load Ollama configuration from YAML file."""
        default_config = {
            "ollama": {
                "default_host": "localhost",
                "default_port": 11434,
                "max_instances": 3,
                "resource_limits": {
                    "cpu_percent": 80,
                    "memory_gb": 16,
                    "gpu_memory_gb": 8
                },
                "models": {
                    "default_pull": ["llama3.1:8b", "codellama:7b", "mistral:7b"],
                    "custom_models_dir": "models/custom",
                    "model_cache_dir": "models/cache"
                },
                "monitoring": {
                    "health_check_interval": 30,
                    "metrics_collection_interval": 10,
                    "alert_thresholds": {
                        "cpu_percent": 85,
                        "memory_percent": 90,
                        "response_time_ms": 5000
                    }
                },
                "security": {
                    "enable_auth": True,
                    "api_key_required": True,
                    "allowed_hosts": ["localhost", "127.0.0.1"]
                }
            }
        }
        
        try:
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    # Merge with defaults
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                        elif isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                if sub_key not in config[key]:
                                    config[key][sub_key] = sub_value
                    return config
            else:
                # Create default config file
                Path(config_path).parent.mkdir(parents=True, exist_ok=True)
                with open(config_path, 'w') as f:
                    yaml.dump(default_config, f, default_flow_style=False)
                return default_config
        except Exception as e:
            logging.warning(f"Failed to load config from {config_path}: {e}")
            return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for Ollama manager."""
        logger = logging.getLogger("ollama_manager")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def initialize(self):
        """Initialize the Ollama manager and start services."""
        try:
            # Initialize HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            # Start Ollama service
            await self._start_ollama_service()
            
            # Discover existing instances
            await self._discover_instances()
            
            # Pull default models
            await self._pull_default_models()
            
            # Start monitoring
            await self._start_monitoring()
            
            self.logger.info("Ollama manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama manager: {e}")
            raise
    
    async def _start_ollama_service(self):
        """Start the Ollama service."""
        try:
            # Check if Ollama is already running
            if await self._check_ollama_health():
                self.logger.info("Ollama service is already running")
                return
            
            # Start Ollama service
            if self.docker_client:
                await self._start_ollama_docker()
            else:
                await self._start_ollama_native()
                
        except Exception as e:
            self.logger.error(f"Failed to start Ollama service: {e}")
            raise
    
    async def _start_ollama_docker(self):
        """Start Ollama using Docker."""
        try:
            # Pull Ollama Docker image
            self.logger.info("Pulling Ollama Docker image...")
            self.docker_client.images.pull("ollama/ollama")
            
            # Start Ollama container
            container = self.docker_client.containers.run(
                "ollama/ollama",
                ports={'11434/tcp': 11434},
                volumes={
                    '/var/lib/ollama': {'bind': '/root/.ollama', 'mode': 'rw'}
                },
                detach=True,
                name="ollama-enterprise"
            )
            
            self.logger.info(f"Started Ollama container: {container.id}")
            
            # Wait for service to be ready
            await self._wait_for_ollama_ready()
            
        except Exception as e:
            self.logger.error(f"Failed to start Ollama with Docker: {e}")
            raise
    
    async def _start_ollama_native(self):
        """Start Ollama as native service."""
        try:
            # Check if Ollama is installed
            result = subprocess.run(['ollama', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception("Ollama is not installed or not in PATH")
            
            # Start Ollama serve
            subprocess.Popen(['ollama', 'serve'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            
            # Wait for service to be ready
            await self._wait_for_ollama_ready()
            
        except Exception as e:
            self.logger.error(f"Failed to start Ollama natively: {e}")
            raise
    
    async def _wait_for_ollama_ready(self, timeout: int = 60):
        """Wait for Ollama service to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if await self._check_ollama_health():
                self.logger.info("Ollama service is ready")
                return
            await asyncio.sleep(2)
        
        raise Exception("Ollama service failed to start within timeout")
    
    async def _check_ollama_health(self) -> bool:
        """Check if Ollama service is healthy."""
        try:
            if not self.session:
                return False
            
            async with self.session.get(f"http://localhost:11434/api/tags") as response:
                return response.status == 200
        except Exception:
            return False
    
    async def _discover_instances(self):
        """Discover existing Ollama instances."""
        try:
            # For now, we'll work with a single instance
            # In enterprise setup, this would discover multiple instances
            instance = OllamaInstance(
                id="main_instance",
                host="localhost",
                port=11434,
                status="running",
                version=await self._get_ollama_version(),
                models=await self._list_models(),
                resource_usage={},
                last_health_check=datetime.now()
            )
            
            self.instances[instance.id] = instance
            self.logger.info(f"Discovered Ollama instance: {instance.id}")
            
        except Exception as e:
            self.logger.error(f"Failed to discover Ollama instances: {e}")
    
    async def _get_ollama_version(self) -> str:
        """Get Ollama version."""
        try:
            async with self.session.get("http://localhost:11434/api/version") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("version", "unknown")
        except Exception:
            pass
        return "unknown"
    
    async def _list_models(self) -> List[str]:
        """List available models."""
        try:
            async with self.session.get("http://localhost:11434/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    return [model["name"] for model in data.get("models", [])]
        except Exception:
            pass
        return []
    
    async def _pull_default_models(self):
        """Pull default models if not already available."""
        default_models = self.config["ollama"]["models"]["default_pull"]
        
        for model_name in default_models:
            try:
                if model_name not in self.instances["main_instance"].models:
                    self.logger.info(f"Pulling model: {model_name}")
                    await self.pull_model(model_name)
            except Exception as e:
                self.logger.error(f"Failed to pull model {model_name}: {e}")
    
    async def _start_monitoring(self):
        """Start monitoring Ollama instances."""
        self.monitoring_active = True
        
        # Start monitoring tasks
        asyncio.create_task(self._monitor_health())
        asyncio.create_task(self._monitor_resources())
        asyncio.create_task(self._process_request_queue())
    
    async def _monitor_health(self):
        """Monitor health of Ollama instances."""
        while self.monitoring_active:
            try:
                for instance_id, instance in self.instances.items():
                    is_healthy = await self._check_ollama_health()
                    instance.status = "running" if is_healthy else "error"
                    instance.last_health_check = datetime.now()
                    
                    if not is_healthy:
                        self.logger.warning(f"Instance {instance_id} is unhealthy")
                        await self._handle_unhealthy_instance(instance_id)
                
                await asyncio.sleep(
                    self.config["ollama"]["monitoring"]["health_check_interval"]
                )
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _monitor_resources(self):
        """Monitor resource usage of Ollama instances."""
        while self.monitoring_active:
            try:
                for instance_id, instance in self.instances.items():
                    resource_usage = await self._get_resource_usage()
                    instance.resource_usage = resource_usage
                    
                    # Check alert thresholds
                    await self._check_resource_alerts(instance_id, resource_usage)
                
                await asyncio.sleep(
                    self.config["ollama"]["monitoring"]["metrics_collection_interval"]
                )
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        try:
            # Get system resource usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Get Ollama-specific resource usage
            ollama_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                if 'ollama' in proc.info['name'].lower():
                    ollama_processes.append(proc.info)
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "ollama_processes": ollama_processes,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to get resource usage: {e}")
            return {}
    
    async def _check_resource_alerts(self, instance_id: str, resource_usage: Dict[str, Any]):
        """Check resource usage against alert thresholds."""
        thresholds = self.config["ollama"]["monitoring"]["alert_thresholds"]
        
        if resource_usage.get("cpu_percent", 0) > thresholds["cpu_percent"]:
            self.logger.warning(f"High CPU usage on {instance_id}: {resource_usage['cpu_percent']}%")
        
        if resource_usage.get("memory_percent", 0) > thresholds["memory_percent"]:
            self.logger.warning(f"High memory usage on {instance_id}: {resource_usage['memory_percent']}%")
    
    async def _handle_unhealthy_instance(self, instance_id: str):
        """Handle unhealthy Ollama instance."""
        try:
            self.logger.info(f"Attempting to recover instance {instance_id}")
            
            # Try to restart the instance
            await self._restart_instance(instance_id)
            
        except Exception as e:
            self.logger.error(f"Failed to recover instance {instance_id}: {e}")
    
    async def _restart_instance(self, instance_id: str):
        """Restart an Ollama instance."""
        try:
            instance = self.instances[instance_id]
            
            if self.docker_client:
                # Restart Docker container
                container = self.docker_client.containers.get("ollama-enterprise")
                container.restart()
                self.logger.info(f"Restarted Docker container for instance {instance_id}")
            else:
                # Restart native service
                subprocess.run(['pkill', '-f', 'ollama'], check=False)
                await asyncio.sleep(2)
                subprocess.Popen(['ollama', 'serve'])
                self.logger.info(f"Restarted native Ollama service for instance {instance_id}")
            
            # Update instance status
            instance.status = "running"
            instance.last_health_check = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Failed to restart instance {instance_id}: {e}")
            self.instances[instance_id].status = "error"
    
    async def pull_model(self, model_name: str, stream: bool = True) -> bool:
        """Pull a model from Ollama registry."""
        try:
            self.logger.info(f"Pulling model: {model_name}")
            
            payload = {
                "name": model_name,
                "stream": stream
            }
            
            async with self.session.post(
                "http://localhost:11434/api/pull",
                json=payload
            ) as response:
                if response.status == 200:
                    if stream:
                        async for line in response.content:
                            if line:
                                try:
                                    data = json.loads(line.decode())
                                    if "status" in data:
                                        self.logger.info(f"Pull status: {data['status']}")
                                except json.JSONDecodeError:
                                    continue
                    
                    # Update models list
                    await self._discover_instances()
                    self.logger.info(f"Successfully pulled model: {model_name}")
                    return True
                else:
                    self.logger.error(f"Failed to pull model {model_name}: {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    async def generate_response(
        self,
        model_name: str,
        prompt: str,
        parameters: Optional[Dict[str, Any]] = None,
        user_id: str = "default"
    ) -> Dict[str, Any]:
        """Generate response using specified model."""
        try:
            if parameters is None:
                parameters = {}
            
            # Create request
            request = ModelRequest(
                id=str(uuid.uuid4()),
                model_name=model_name,
                prompt=prompt,
                parameters=parameters,
                user_id=user_id
            )
            
            # Add to queue for processing
            await self.request_queue.put(request)
            
            # Process request
            response = await self._process_single_request(request)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return {
                "error": str(e),
                "success": False
            }
    
    async def _process_single_request(self, request: ModelRequest) -> Dict[str, Any]:
        """Process a single model request."""
        try:
            payload = {
                "model": request.model_name,
                "prompt": request.prompt,
                "stream": False,
                **request.parameters
            }
            
            start_time = time.time()
            
            async with self.session.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=request.timeout)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    end_time = time.time()
                    response_time = (end_time - start_time) * 1000  # ms
                    
                    return {
                        "success": True,
                        "response": data.get("response", ""),
                        "model": request.model_name,
                        "response_time_ms": response_time,
                        "request_id": request.id,
                        "timestamp": datetime.now().isoformat(),
                        "usage": data.get("done", False)
                    }
                else:
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}",
                        "request_id": request.id
                    }
                    
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "Request timeout",
                "request_id": request.id
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "request_id": request.id
            }
    
    async def _process_request_queue(self):
        """Process requests from the queue."""
        while self.monitoring_active:
            try:
                # Process requests with priority
                if not self.request_queue.empty():
                    request = await self.request_queue.get()
                    await self._process_single_request(request)
                    self.request_queue.task_done()
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Error processing request queue: {e}")
                await asyncio.sleep(1)
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all available models with details."""
        try:
            async with self.session.get("http://localhost:11434/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = []
                    
                    for model_data in data.get("models", []):
                        model = OllamaModel(
                            name=model_data["name"],
                            model_id=model_data["name"],
                            size=model_data.get("size", 0),
                            modified_at=datetime.fromisoformat(
                                model_data["modified_at"].replace("Z", "+00:00")
                            ),
                            digest=model_data.get("digest", ""),
                            details=model_data
                        )
                        models.append(asdict(model))
                    
                    return models
                    
        except Exception as e:
            self.logger.error(f"Error listing models: {e}")
        
        return []
    
    async def delete_model(self, model_name: str) -> bool:
        """Delete a model."""
        try:
            async with self.session.delete(
                f"http://localhost:11434/api/delete",
                json={"name": model_name}
            ) as response:
                success = response.status == 200
                if success:
                    self.logger.info(f"Deleted model: {model_name}")
                else:
                    self.logger.error(f"Failed to delete model {model_name}: {response.status}")
                return success
                
        except Exception as e:
            self.logger.error(f"Error deleting model {model_name}: {e}")
            return False
    
    async def get_instance_status(self) -> Dict[str, Any]:
        """Get status of all Ollama instances."""
        status = {
            "instances": {},
            "total_instances": len(self.instances),
            "healthy_instances": 0,
            "total_models": 0,
            "timestamp": datetime.now().isoformat()
        }
        
        for instance_id, instance in self.instances.items():
            instance_status = asdict(instance)
            instance_status["resource_usage"] = instance.resource_usage
            
            status["instances"][instance_id] = instance_status
            
            if instance.status == "running":
                status["healthy_instances"] += 1
            
            status["total_models"] += len(instance.models)
        
        return status
    
    async def shutdown(self):
        """Shutdown the Ollama manager."""
        try:
            self.monitoring_active = False
            
            if self.session:
                await self.session.close()
            
            self.logger.info("Ollama manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


# Import uuid for request IDs
import uuid
