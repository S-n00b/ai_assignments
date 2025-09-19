"""
SmolAgents Integration Module for Lightweight Agent Deployment

This module provides SmolAgents integration for deploying lightweight, resource-efficient
AI agents optimized for edge computing, IoT devices, and resource-constrained environments.
It enables micro-agent architectures and efficient resource utilization.

Key Features:
- Lightweight agent deployment and management
- Edge computing optimization
- Micro-agent architectures
- Resource-efficient agent patterns
- IoT device integration
- Distributed agent coordination
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Callable, Protocol
from dataclasses import dataclass, field
from enum import Enum
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
import psutil
import threading
import time

# SmolAgents imports
try:
    from smolagents import SmolAgent, SmolTool, SmolMemory
    from smolagents.tools import BaseTool as SmolBaseTool
    SMOLAGENTS_AVAILABLE = True
except ImportError:
    SMOLAGENTS_AVAILABLE = False
    logging.warning("SmolAgents not available. Install with: pip install smolagents")

# Additional imports for edge computing
try:
    import numpy as np
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentSize(Enum):
    """Agent size categories for resource optimization"""
    NANO = "nano"      # < 10MB memory, < 100ms response
    MICRO = "micro"    # < 50MB memory, < 500ms response
    MINI = "mini"      # < 100MB memory, < 1s response
    SMALL = "small"    # < 200MB memory, < 2s response


class DeploymentEnvironment(Enum):
    """Deployment environments for agents"""
    EDGE_DEVICE = "edge_device"
    IOT_SENSOR = "iot_sensor"
    MOBILE_DEVICE = "mobile_device"
    EMBEDDED_SYSTEM = "embedded_system"
    CLOUD_INSTANCE = "cloud_instance"
    HYBRID_CLOUD = "hybrid_cloud"


@dataclass
class ResourceConstraints:
    """Resource constraints for lightweight agents"""
    max_memory_mb: int
    max_cpu_percent: float
    max_response_time_ms: int
    max_concurrent_tasks: int
    battery_powered: bool = False
    network_limited: bool = False
    storage_limited: bool = False


@dataclass
class EdgeDeviceProfile:
    """Profile for edge device deployment"""
    device_type: str
    cpu_cores: int
    memory_mb: int
    storage_mb: int
    network_bandwidth_mbps: float
    battery_capacity_mah: Optional[int] = None
    power_consumption_w: Optional[float] = None
    operating_system: str = "linux"
    architecture: str = "x86_64"


class LenovoSmolAgent:
    """
    Lenovo-optimized SmolAgent for lightweight deployment.
    Provides resource-efficient agent capabilities with edge computing optimization.
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_size: AgentSize,
        deployment_env: DeploymentEnvironment,
        resource_constraints: ResourceConstraints,
        device_profile: EdgeDeviceProfile = None
    ):
        """
        Initialize a Lenovo SmolAgent.
        
        Args:
            agent_id: Unique agent identifier
            agent_size: Size category for the agent
            deployment_env: Deployment environment
            resource_constraints: Resource constraints
            device_profile: Edge device profile
        """
        if not SMOLAGENTS_AVAILABLE:
            raise ImportError("SmolAgents not available. Install with: pip install smolagents")
        
        self.agent_id = agent_id
        self.agent_size = agent_size
        self.deployment_env = deployment_env
        self.resource_constraints = resource_constraints
        self.device_profile = device_profile
        
        # Agent state
        self.status = "idle"
        self.current_tasks = []
        self.task_history = []
        self.performance_metrics = {}
        self.resource_usage = {}
        
        # Initialize SmolAgent with size-optimized configuration
        self.smol_agent = self._create_smol_agent()
        
        # Resource monitoring
        self._resource_monitor_thread = None
        self._monitoring_active = False
        
        logger.info(f"Lenovo SmolAgent initialized: {agent_id} ({agent_size.value})")
    
    def _create_smol_agent(self) -> Optional[SmolAgent]:
        """Create a size-optimized SmolAgent instance."""
        try:
            # Configure agent based on size and environment
            config = self._get_agent_configuration()
            
            # Create SmolAgent with optimized settings
            agent = SmolAgent(
                name=self.agent_id,
                system_prompt=config["system_prompt"],
                tools=config["tools"],
                memory=config["memory"],
                max_tokens=config["max_tokens"],
                temperature=config["temperature"],
                model=config["model"]
            )
            
            logger.info(f"SmolAgent created for {self.agent_id}")
            return agent
            
        except Exception as e:
            logger.error(f"Error creating SmolAgent: {e}")
            return None
    
    def _get_agent_configuration(self) -> Dict[str, Any]:
        """Get agent configuration based on size and environment."""
        
        # Base configurations for different agent sizes
        size_configs = {
            AgentSize.NANO: {
                "max_tokens": 100,
                "temperature": 0.1,
                "model": "gpt-3.5-turbo",
                "tools": [],
                "memory": None,
                "system_prompt": "You are a lightweight AI agent optimized for minimal resource usage."
            },
            AgentSize.MICRO: {
                "max_tokens": 200,
                "temperature": 0.2,
                "model": "gpt-3.5-turbo",
                "tools": [],
                "memory": SmolMemory(max_size=100),
                "system_prompt": "You are a micro AI agent designed for edge computing environments."
            },
            AgentSize.MINI: {
                "max_tokens": 500,
                "temperature": 0.3,
                "model": "gpt-4",
                "tools": [],
                "memory": SmolMemory(max_size=500),
                "system_prompt": "You are a mini AI agent with balanced capabilities and efficiency."
            },
            AgentSize.SMALL: {
                "max_tokens": 1000,
                "temperature": 0.4,
                "model": "gpt-4",
                "tools": [],
                "memory": SmolMemory(max_size=1000),
                "system_prompt": "You are a small AI agent with comprehensive capabilities."
            }
        }
        
        config = size_configs[self.agent_size].copy()
        
        # Adjust configuration based on deployment environment
        if self.deployment_env == DeploymentEnvironment.IOT_SENSOR:
            config["max_tokens"] = max(50, config["max_tokens"] // 2)
            config["temperature"] = 0.1
        elif self.deployment_env == DeploymentEnvironment.MOBILE_DEVICE:
            config["max_tokens"] = max(100, config["max_tokens"] // 2)
            config["temperature"] = 0.2
        
        # Adjust based on resource constraints
        if self.resource_constraints.max_memory_mb < 50:
            config["memory"] = None
            config["max_tokens"] = min(config["max_tokens"], 100)
        
        return config
    
    async def execute_task(self, task_description: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a task using the lightweight agent."""
        try:
            if not self.smol_agent:
                raise Exception("SmolAgent not initialized")
            
            # Check resource constraints before execution
            if not self._check_resource_constraints():
                return {
                    "success": False,
                    "error": "Resource constraints exceeded",
                    "agent_id": self.agent_id
                }
            
            # Record task start
            task_id = str(uuid.uuid4())
            task_start = datetime.now()
            
            # Execute task with timeout
            timeout_seconds = self.resource_constraints.max_response_time_ms / 1000
            
            result = await asyncio.wait_for(
                self._execute_with_monitoring(task_description, context),
                timeout=timeout_seconds
            )
            
            # Record task completion
            task_duration = (datetime.now() - task_start).total_seconds()
            self._record_task_completion(task_id, task_description, result, task_duration)
            
            return {
                "success": True,
                "task_id": task_id,
                "result": result,
                "agent_id": self.agent_id,
                "agent_size": self.agent_size.value,
                "execution_time": task_duration,
                "resource_usage": self.resource_usage,
                "timestamp": datetime.now().isoformat()
            }
            
        except asyncio.TimeoutError:
            logger.warning(f"Task timeout for agent {self.agent_id}")
            return {
                "success": False,
                "error": "Task execution timeout",
                "agent_id": self.agent_id,
                "execution_time": self.resource_constraints.max_response_time_ms / 1000
            }
        except Exception as e:
            logger.error(f"Error executing task for agent {self.agent_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent_id": self.agent_id,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _execute_with_monitoring(self, task_description: str, context: Dict[str, Any]) -> Any:
        """Execute task with resource monitoring."""
        # Start resource monitoring
        self._start_resource_monitoring()
        
        try:
            # Execute task
            if context:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.smol_agent.run(f"{task_description}\n\nContext: {json.dumps(context)}")
                )
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.smol_agent.run(task_description)
                )
            
            return result
            
        finally:
            # Stop resource monitoring
            self._stop_resource_monitoring()
    
    def _check_resource_constraints(self) -> bool:
        """Check if current resource usage is within constraints."""
        try:
            # Check memory usage
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            if memory_usage > self.resource_constraints.max_memory_mb:
                logger.warning(f"Memory constraint exceeded: {memory_usage:.2f}MB > {self.resource_constraints.max_memory_mb}MB")
                return False
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > self.resource_constraints.max_cpu_percent:
                logger.warning(f"CPU constraint exceeded: {cpu_percent:.2f}% > {self.resource_constraints.max_cpu_percent}%")
                return False
            
            # Check concurrent tasks
            if len(self.current_tasks) >= self.resource_constraints.max_concurrent_tasks:
                logger.warning(f"Concurrent task limit exceeded: {len(self.current_tasks)} >= {self.resource_constraints.max_concurrent_tasks}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking resource constraints: {e}")
            return False
    
    def _start_resource_monitoring(self):
        """Start resource monitoring thread."""
        if not self._monitoring_active:
            self._monitoring_active = True
            self._resource_monitor_thread = threading.Thread(target=self._monitor_resources)
            self._resource_monitor_thread.daemon = True
            self._resource_monitor_thread.start()
    
    def _stop_resource_monitoring(self):
        """Stop resource monitoring thread."""
        self._monitoring_active = False
        if self._resource_monitor_thread:
            self._resource_monitor_thread.join(timeout=1.0)
    
    def _monitor_resources(self):
        """Monitor resource usage in background thread."""
        while self._monitoring_active:
            try:
                # Get current resource usage
                memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                cpu_percent = psutil.cpu_percent()
                
                self.resource_usage = {
                    "memory_mb": memory_usage,
                    "cpu_percent": cpu_percent,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Check for constraint violations
                if memory_usage > self.resource_constraints.max_memory_mb:
                    logger.warning(f"Memory usage high: {memory_usage:.2f}MB")
                
                if cpu_percent > self.resource_constraints.max_cpu_percent:
                    logger.warning(f"CPU usage high: {cpu_percent:.2f}%")
                
                time.sleep(0.5)  # Monitor every 500ms
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                break
    
    def _record_task_completion(self, task_id: str, description: str, result: Any, duration: float):
        """Record task completion for performance tracking."""
        task_record = {
            "task_id": task_id,
            "description": description,
            "result": result,
            "duration": duration,
            "completed_at": datetime.now(),
            "agent_id": self.agent_id,
            "agent_size": self.agent_size.value,
            "resource_usage": self.resource_usage.copy()
        }
        
        self.task_history.append(task_record)
        
        # Update performance metrics
        if "task_count" not in self.performance_metrics:
            self.performance_metrics["task_count"] = 0
            self.performance_metrics["total_duration"] = 0.0
            self.performance_metrics["avg_duration"] = 0.0
            self.performance_metrics["avg_memory_usage"] = 0.0
            self.performance_metrics["avg_cpu_usage"] = 0.0
        
        self.performance_metrics["task_count"] += 1
        self.performance_metrics["total_duration"] += duration
        self.performance_metrics["avg_duration"] = (
            self.performance_metrics["total_duration"] / self.performance_metrics["task_count"]
        )
        
        if self.resource_usage:
            memory_usage = self.resource_usage.get("memory_mb", 0)
            cpu_usage = self.resource_usage.get("cpu_percent", 0)
            
            self.performance_metrics["avg_memory_usage"] = (
                (self.performance_metrics["avg_memory_usage"] * (self.performance_metrics["task_count"] - 1) + memory_usage) /
                self.performance_metrics["task_count"]
            )
            self.performance_metrics["avg_cpu_usage"] = (
                (self.performance_metrics["avg_cpu_usage"] * (self.performance_metrics["task_count"] - 1) + cpu_usage) /
                self.performance_metrics["task_count"]
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for the agent."""
        return {
            "agent_id": self.agent_id,
            "agent_size": self.agent_size.value,
            "deployment_env": self.deployment_env.value,
            "performance": self.performance_metrics,
            "resource_constraints": {
                "max_memory_mb": self.resource_constraints.max_memory_mb,
                "max_cpu_percent": self.resource_constraints.max_cpu_percent,
                "max_response_time_ms": self.resource_constraints.max_response_time_ms,
                "max_concurrent_tasks": self.resource_constraints.max_concurrent_tasks
            },
            "current_resource_usage": self.resource_usage,
            "task_history_count": len(self.task_history),
            "recent_tasks": self.task_history[-5:] if self.task_history else []
        }


class LenovoSmolAgentManager:
    """
    Manager for deploying and coordinating multiple lightweight SmolAgents.
    Provides distributed agent coordination and resource optimization.
    """
    
    def __init__(self):
        """Initialize the SmolAgent manager."""
        self.agents = {}
        self.agent_pools = {}
        self.deployment_templates = {}
        self.coordination_network = {}
        
        logger.info("Lenovo SmolAgent Manager initialized")
    
    async def deploy_agent(self, agent: LenovoSmolAgent) -> bool:
        """Deploy a lightweight agent."""
        try:
            self.agents[agent.agent_id] = agent
            
            # Add to appropriate agent pool based on size
            size_pool = f"{agent.agent_size.value}_agents"
            if size_pool not in self.agent_pools:
                self.agent_pools[size_pool] = []
            self.agent_pools[size_pool].append(agent.agent_id)
            
            logger.info(f"SmolAgent deployed: {agent.agent_id} ({agent.agent_size.value})")
            return True
            
        except Exception as e:
            logger.error(f"Error deploying agent: {e}")
            return False
    
    async def create_agent_pool(self, pool_id: str, agent_size: AgentSize, 
                              count: int, deployment_env: DeploymentEnvironment) -> List[str]:
        """Create a pool of lightweight agents."""
        try:
            agent_ids = []
            
            for i in range(count):
                agent_id = f"{pool_id}_agent_{i}"
                
                # Create resource constraints based on agent size
                constraints = self._get_default_constraints(agent_size)
                
                # Create edge device profile
                device_profile = self._get_device_profile(deployment_env)
                
                # Create agent
                agent = LenovoSmolAgent(
                    agent_id=agent_id,
                    agent_size=agent_size,
                    deployment_env=deployment_env,
                    resource_constraints=constraints,
                    device_profile=device_profile
                )
                
                # Deploy agent
                await self.deploy_agent(agent)
                agent_ids.append(agent_id)
            
            self.agent_pools[pool_id] = agent_ids
            logger.info(f"Created agent pool: {pool_id} with {count} {agent_size.value} agents")
            
            return agent_ids
            
        except Exception as e:
            logger.error(f"Error creating agent pool: {e}")
            return []
    
    def _get_default_constraints(self, agent_size: AgentSize) -> ResourceConstraints:
        """Get default resource constraints for agent size."""
        constraints_map = {
            AgentSize.NANO: ResourceConstraints(
                max_memory_mb=10,
                max_cpu_percent=20.0,
                max_response_time_ms=100,
                max_concurrent_tasks=1,
                battery_powered=True,
                network_limited=True
            ),
            AgentSize.MICRO: ResourceConstraints(
                max_memory_mb=50,
                max_cpu_percent=40.0,
                max_response_time_ms=500,
                max_concurrent_tasks=2,
                battery_powered=True,
                network_limited=True
            ),
            AgentSize.MINI: ResourceConstraints(
                max_memory_mb=100,
                max_cpu_percent=60.0,
                max_response_time_ms=1000,
                max_concurrent_tasks=3,
                battery_powered=False,
                network_limited=False
            ),
            AgentSize.SMALL: ResourceConstraints(
                max_memory_mb=200,
                max_cpu_percent=80.0,
                max_response_time_ms=2000,
                max_concurrent_tasks=5,
                battery_powered=False,
                network_limited=False
            )
        }
        
        return constraints_map[agent_size]
    
    def _get_device_profile(self, deployment_env: DeploymentEnvironment) -> EdgeDeviceProfile:
        """Get device profile for deployment environment."""
        profiles = {
            DeploymentEnvironment.IOT_SENSOR: EdgeDeviceProfile(
                device_type="iot_sensor",
                cpu_cores=1,
                memory_mb=32,
                storage_mb=128,
                network_bandwidth_mbps=1.0,
                battery_capacity_mah=1000,
                power_consumption_w=0.5
            ),
            DeploymentEnvironment.EDGE_DEVICE: EdgeDeviceProfile(
                device_type="edge_device",
                cpu_cores=4,
                memory_mb=512,
                storage_mb=8192,
                network_bandwidth_mbps=100.0,
                power_consumption_w=10.0
            ),
            DeploymentEnvironment.MOBILE_DEVICE: EdgeDeviceProfile(
                device_type="mobile_device",
                cpu_cores=8,
                memory_mb=4096,
                storage_mb=65536,
                network_bandwidth_mbps=1000.0,
                battery_capacity_mah=4000,
                power_consumption_w=5.0
            ),
            DeploymentEnvironment.CLOUD_INSTANCE: EdgeDeviceProfile(
                device_type="cloud_instance",
                cpu_cores=16,
                memory_mb=16384,
                storage_mb=1048576,
                network_bandwidth_mbps=10000.0
            )
        }
        
        return profiles.get(deployment_env, profiles[DeploymentEnvironment.CLOUD_INSTANCE])
    
    async def execute_distributed_task(self, task_description: str, 
                                     agent_pool: str = None,
                                     parallel_execution: bool = True) -> Dict[str, Any]:
        """Execute a task across multiple agents in parallel or sequence."""
        try:
            # Select agents for task execution
            if agent_pool and agent_pool in self.agent_pools:
                agent_ids = self.agent_pools[agent_pool]
            else:
                # Use all available agents
                agent_ids = list(self.agents.keys())
            
            if not agent_ids:
                return {
                    "success": False,
                    "error": "No agents available for task execution"
                }
            
            # Execute task
            if parallel_execution:
                # Execute in parallel
                tasks = [
                    self.agents[agent_id].execute_task(task_description)
                    for agent_id in agent_ids
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                # Execute sequentially
                results = []
                for agent_id in agent_ids:
                    result = await self.agents[agent_id].execute_task(task_description)
                    results.append(result)
            
            # Analyze results
            successful_results = [r for r in results if isinstance(r, dict) and r.get("success", False)]
            failed_results = [r for r in results if isinstance(r, dict) and not r.get("success", False)]
            
            return {
                "success": len(successful_results) > 0,
                "total_agents": len(agent_ids),
                "successful_agents": len(successful_results),
                "failed_agents": len(failed_results),
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error executing distributed task: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_manager_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics for the agent manager."""
        try:
            total_agents = len(self.agents)
            
            # Agent pool statistics
            pool_stats = {}
            for pool_id, agent_ids in self.agent_pools.items():
                pool_stats[pool_id] = {
                    "agent_count": len(agent_ids),
                    "agents": agent_ids
                }
            
            # Agent performance summary
            agent_performance = {}
            for agent_id, agent in self.agents.items():
                agent_performance[agent_id] = agent.get_performance_metrics()
            
            # Resource usage summary
            total_memory_usage = 0
            total_cpu_usage = 0
            for agent in self.agents.values():
                if agent.resource_usage:
                    total_memory_usage += agent.resource_usage.get("memory_mb", 0)
                    total_cpu_usage += agent.resource_usage.get("cpu_percent", 0)
            
            return {
                "manager_summary": {
                    "total_agents": total_agents,
                    "total_pools": len(self.agent_pools),
                    "total_memory_usage_mb": total_memory_usage,
                    "avg_cpu_usage": total_cpu_usage / total_agents if total_agents > 0 else 0,
                    "timestamp": datetime.now().isoformat()
                },
                "pool_statistics": pool_stats,
                "agent_performance": agent_performance
            }
            
        except Exception as e:
            logger.error(f"Error getting manager analytics: {e}")
            return {"error": str(e)}


class LenovoEdgeAgentCoordinator:
    """
    Coordinator for managing edge agents across distributed environments.
    Provides network-aware coordination and resource optimization.
    """
    
    def __init__(self):
        """Initialize the edge agent coordinator."""
        self.edge_nodes = {}
        self.agent_networks = {}
        self.coordination_protocols = {}
        
        logger.info("Lenovo Edge Agent Coordinator initialized")
    
    async def register_edge_node(self, node_id: str, node_info: Dict[str, Any]) -> bool:
        """Register an edge node with the coordinator."""
        try:
            self.edge_nodes[node_id] = {
                "node_info": node_info,
                "agents": {},
                "last_heartbeat": datetime.now(),
                "status": "active"
            }
            
            logger.info(f"Edge node registered: {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering edge node: {e}")
            return False
    
    async def deploy_agents_to_edge(self, edge_node_id: str, agent_configs: List[Dict[str, Any]]) -> List[str]:
        """Deploy agents to a specific edge node."""
        try:
            if edge_node_id not in self.edge_nodes:
                raise ValueError(f"Edge node {edge_node_id} not registered")
            
            deployed_agents = []
            
            for config in agent_configs:
                agent_id = config["agent_id"]
                agent_size = AgentSize(config["agent_size"])
                deployment_env = DeploymentEnvironment(config["deployment_env"])
                constraints = ResourceConstraints(**config["resource_constraints"])
                
                # Create device profile
                device_profile = EdgeDeviceProfile(**config.get("device_profile", {}))
                
                # Create agent
                agent = LenovoSmolAgent(
                    agent_id=agent_id,
                    agent_size=agent_size,
                    deployment_env=deployment_env,
                    resource_constraints=constraints,
                    device_profile=device_profile
                )
                
                # Register agent with edge node
                self.edge_nodes[edge_node_id]["agents"][agent_id] = agent
                deployed_agents.append(agent_id)
            
            logger.info(f"Deployed {len(deployed_agents)} agents to edge node {edge_node_id}")
            return deployed_agents
            
        except Exception as e:
            logger.error(f"Error deploying agents to edge node: {e}")
            return []
    
    async def coordinate_distributed_workflow(self, workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate a workflow across multiple edge nodes."""
        try:
            workflow_id = workflow_config.get("workflow_id", str(uuid.uuid4()))
            stages = workflow_config.get("stages", [])
            
            results = {}
            
            for stage in stages:
                stage_id = stage["id"]
                target_nodes = stage.get("target_nodes", list(self.edge_nodes.keys()))
                task_description = stage["task_description"]
                
                # Execute stage across target nodes
                stage_results = []
                for node_id in target_nodes:
                    if node_id in self.edge_nodes:
                        node_agents = self.edge_nodes[node_id]["agents"]
                        
                        # Execute task on all agents in the node
                        node_task_results = []
                        for agent_id, agent in node_agents.items():
                            result = await agent.execute_task(task_description)
                            node_task_results.append(result)
                        
                        stage_results.append({
                            "node_id": node_id,
                            "results": node_task_results
                        })
                
                results[stage_id] = stage_results
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error coordinating distributed workflow: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# Factory functions for creating lightweight agents
def create_nano_agent(agent_id: str, deployment_env: DeploymentEnvironment = DeploymentEnvironment.IOT_SENSOR) -> LenovoSmolAgent:
    """Create a nano-sized agent for minimal resource usage."""
    constraints = ResourceConstraints(
        max_memory_mb=10,
        max_cpu_percent=20.0,
        max_response_time_ms=100,
        max_concurrent_tasks=1,
        battery_powered=True,
        network_limited=True
    )
    
    return LenovoSmolAgent(
        agent_id=agent_id,
        agent_size=AgentSize.NANO,
        deployment_env=deployment_env,
        resource_constraints=constraints
    )


def create_micro_agent(agent_id: str, deployment_env: DeploymentEnvironment = DeploymentEnvironment.EDGE_DEVICE) -> LenovoSmolAgent:
    """Create a micro-sized agent for edge computing."""
    constraints = ResourceConstraints(
        max_memory_mb=50,
        max_cpu_percent=40.0,
        max_response_time_ms=500,
        max_concurrent_tasks=2,
        battery_powered=True,
        network_limited=True
    )
    
    return LenovoSmolAgent(
        agent_id=agent_id,
        agent_size=AgentSize.MICRO,
        deployment_env=deployment_env,
        resource_constraints=constraints
    )


def create_mini_agent(agent_id: str, deployment_env: DeploymentEnvironment = DeploymentEnvironment.MOBILE_DEVICE) -> LenovoSmolAgent:
    """Create a mini-sized agent for mobile devices."""
    constraints = ResourceConstraints(
        max_memory_mb=100,
        max_cpu_percent=60.0,
        max_response_time_ms=1000,
        max_concurrent_tasks=3,
        battery_powered=False,
        network_limited=False
    )
    
    return LenovoSmolAgent(
        agent_id=agent_id,
        agent_size=AgentSize.MINI,
        deployment_env=deployment_env,
        resource_constraints=constraints
    )


def create_small_agent(agent_id: str, deployment_env: DeploymentEnvironment = DeploymentEnvironment.CLOUD_INSTANCE) -> LenovoSmolAgent:
    """Create a small-sized agent for cloud instances."""
    constraints = ResourceConstraints(
        max_memory_mb=200,
        max_cpu_percent=80.0,
        max_response_time_ms=2000,
        max_concurrent_tasks=5,
        battery_powered=False,
        network_limited=False
    )
    
    return LenovoSmolAgent(
        agent_id=agent_id,
        agent_size=AgentSize.SMALL,
        deployment_env=deployment_env,
        resource_constraints=constraints
    )
