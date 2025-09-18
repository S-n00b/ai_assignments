"""
Agentic Computing Framework Module

This module implements a sophisticated multi-agent system framework for enterprise
AI applications, including intelligent agent orchestration, communication protocols,
workflow management, and collaborative AI capabilities.

Key Features:
- Multi-agent system architecture
- Intelligent agent orchestration
- Message passing and communication protocols
- Workflow management and task distribution
- Collaborative AI capabilities
- Performance monitoring and metrics
"""

import json
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable, Protocol, Union, AsyncIterator
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import logging
from concurrent.futures import ThreadPoolExecutor
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages in the agent system"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    WORKFLOW_EVENT = "workflow_event"
    STATUS_UPDATE = "status_update"
    ERROR_REPORT = "error_report"
    HEARTBEAT = "heartbeat"
    COLLABORATION_REQUEST = "collaboration_request"
    COLLABORATION_RESPONSE = "collaboration_response"


class AgentStatus(Enum):
    """Agent status states"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AgentMessage:
    """Message structure for agent communication"""
    id: str
    sender: str
    recipient: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    priority: TaskPriority = TaskPriority.NORMAL
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """Task structure for agent processing"""
    id: str
    agent_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentCapabilities:
    """Agent capabilities definition"""
    task_types: List[str]
    max_concurrent_tasks: int
    supported_languages: List[str]
    specializations: List[str]
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    
    This class defines the interface and common functionality for all agents,
    including task processing, communication, and lifecycle management.
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: AgentCapabilities,
        name: str = None
    ):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent (e.g., "nlp_agent", "vision_agent")
            capabilities: Agent capabilities definition
            name: Human-readable agent name
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.name = name or f"{agent_type}_{agent_id}"
        
        # Agent state
        self.status = AgentStatus.IDLE
        self.current_tasks = []
        self.task_queue = asyncio.Queue()
        self.message_queue = asyncio.Queue()
        
        # Performance tracking
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "avg_processing_time": 0.0,
            "last_activity": datetime.now()
        }
        
        # Communication
        self.message_handlers = {}
        self.collaboration_partners = set()
        
        logger.info(f"Initialized agent {self.name} ({self.agent_id})")
    
    @abstractmethod
    async def process_task(self, task: Task) -> Dict[str, Any]:
        """
        Process a task assigned to this agent.
        
        Args:
            task: Task to process
            
        Returns:
            Task processing result
        """
        pass
    
    async def send_message(self, message: AgentMessage) -> bool:
        """
        Send a message to another agent.
        
        Args:
            message: Message to send
            
        Returns:
            True if message was sent successfully
        """
        try:
            await self.message_queue.put(message)
            logger.debug(f"Agent {self.agent_id} sent message {message.id} to {message.recipient}")
            return True
        except Exception as e:
            logger.error(f"Failed to send message from {self.agent_id}: {str(e)}")
            return False
    
    async def receive_message(self) -> Optional[AgentMessage]:
        """
        Receive a message from the message queue.
        
        Returns:
            Received message or None if queue is empty
        """
        try:
            message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
            return message
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Failed to receive message for {self.agent_id}: {str(e)}")
            return None
    
    async def handle_message(self, message: AgentMessage) -> Dict[str, Any]:
        """
        Handle an incoming message.
        
        Args:
            message: Message to handle
            
        Returns:
            Message handling result
        """
        try:
            handler = self.message_handlers.get(message.message_type)
            if handler:
                return await handler(message)
            else:
                logger.warning(f"No handler for message type {message.message_type} in agent {self.agent_id}")
                return {"status": "unhandled", "message_type": message.message_type.value}
        except Exception as e:
            logger.error(f"Error handling message in agent {self.agent_id}: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """
        Register a message handler for a specific message type.
        
        Args:
            message_type: Type of message to handle
            handler: Handler function
        """
        self.message_handlers[message_type] = handler
        logger.debug(f"Registered handler for {message_type.value} in agent {self.agent_id}")
    
    async def start(self):
        """Start the agent's main processing loop"""
        self.status = AgentStatus.IDLE
        logger.info(f"Agent {self.agent_id} started")
        
        # Start message processing loop
        asyncio.create_task(self._message_processing_loop())
        
        # Start task processing loop
        asyncio.create_task(self._task_processing_loop())
    
    async def stop(self):
        """Stop the agent"""
        self.status = AgentStatus.OFFLINE
        logger.info(f"Agent {self.agent_id} stopped")
    
    async def _message_processing_loop(self):
        """Main message processing loop"""
        while self.status != AgentStatus.OFFLINE:
            try:
                message = await self.receive_message()
                if message:
                    await self.handle_message(message)
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in message processing loop for {self.agent_id}: {str(e)}")
                await asyncio.sleep(1)
    
    async def _task_processing_loop(self):
        """Main task processing loop"""
        while self.status != AgentStatus.OFFLINE:
            try:
                if self.status == AgentStatus.IDLE and len(self.current_tasks) < self.capabilities.max_concurrent_tasks:
                    task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                    asyncio.create_task(self._process_task_async(task))
                await asyncio.sleep(0.1)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in task processing loop for {self.agent_id}: {str(e)}")
                await asyncio.sleep(1)
    
    async def _process_task_async(self, task: Task):
        """Process a task asynchronously"""
        try:
            self.status = AgentStatus.BUSY
            self.current_tasks.append(task)
            
            start_time = datetime.now()
            result = await self.process_task(task)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update metrics
            self.metrics["tasks_completed"] += 1
            self.metrics["avg_processing_time"] = (
                (self.metrics["avg_processing_time"] * (self.metrics["tasks_completed"] - 1) + processing_time) /
                self.metrics["tasks_completed"]
            )
            self.metrics["last_activity"] = datetime.now()
            
            # Remove task from current tasks
            self.current_tasks.remove(task)
            
            # Send task response
            response_message = AgentMessage(
                id=str(uuid.uuid4()),
                sender=self.agent_id,
                recipient=task.agent_id,
                message_type=MessageType.TASK_RESPONSE,
                content={
                    "task_id": task.id,
                    "result": result,
                    "processing_time": processing_time,
                    "status": "completed"
                },
                correlation_id=task.id
            )
            await self.send_message(response_message)
            
            if len(self.current_tasks) == 0:
                self.status = AgentStatus.IDLE
            
            logger.info(f"Agent {self.agent_id} completed task {task.id} in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed to process task {task.id}: {str(e)}")
            
            # Update metrics
            self.metrics["tasks_failed"] += 1
            self.current_tasks.remove(task)
            
            # Send error response
            error_message = AgentMessage(
                id=str(uuid.uuid4()),
                sender=self.agent_id,
                recipient=task.agent_id,
                message_type=MessageType.ERROR_REPORT,
                content={
                    "task_id": task.id,
                    "error": str(e),
                    "status": "failed"
                },
                correlation_id=task.id
            )
            await self.send_message(error_message)
            
            if len(self.current_tasks) == 0:
                self.status = AgentStatus.IDLE


class AgenticComputingFramework:
    """
    Enterprise Agentic Computing Framework for multi-agent system orchestration.
    
    This class provides comprehensive management of multi-agent systems including:
    - Agent registration and lifecycle management
    - Task distribution and load balancing
    - Inter-agent communication and collaboration
    - Workflow orchestration and coordination
    - Performance monitoring and optimization
    - Fault tolerance and error recovery
    """
    
    def __init__(self, framework_name: str = "Lenovo Agentic Computing Framework"):
        """
        Initialize the Agentic Computing Framework.
        
        Args:
            framework_name: Name of the framework instance
        """
        self.framework_name = framework_name
        self.agents = {}
        self.agent_types = {}
        self.task_queue = asyncio.Queue()
        self.message_bus = asyncio.Queue()
        self.workflows = {}
        self.metrics_storage = []
        
        # Framework components
        self.task_scheduler = None
        self.load_balancer = None
        self.monitoring_system = None
        
        # Initialize framework components
        self._initialize_framework_components()
        
        logger.info(f"Initialized {framework_name}")
    
    def _initialize_framework_components(self):
        """Initialize framework components"""
        
        # Initialize task scheduler
        self.task_scheduler = TaskScheduler(self)
        
        # Initialize load balancer
        self.load_balancer = LoadBalancer(self)
        
        # Initialize monitoring system
        self.monitoring_system = MonitoringSystem(self)
        
        logger.info("Framework components initialized")
    
    async def register_agent(self, agent: BaseAgent) -> bool:
        """
        Register an agent with the framework.
        
        Args:
            agent: Agent to register
            
        Returns:
            True if registration was successful
        """
        try:
            if agent.agent_id in self.agents:
                raise ValueError(f"Agent {agent.agent_id} already registered")
            
            # Register agent
            self.agents[agent.agent_id] = agent
            
            # Register agent type
            if agent.agent_type not in self.agent_types:
                self.agent_types[agent.agent_type] = []
            self.agent_types[agent.agent_type].append(agent.agent_id)
            
            # Start agent
            await agent.start()
            
            logger.info(f"Registered agent {agent.agent_id} of type {agent.agent_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent.agent_id}: {str(e)}")
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the framework.
        
        Args:
            agent_id: ID of agent to unregister
            
        Returns:
            True if unregistration was successful
        """
        try:
            if agent_id not in self.agents:
                raise ValueError(f"Agent {agent_id} not registered")
            
            agent = self.agents[agent_id]
            
            # Stop agent
            await agent.stop()
            
            # Remove from registries
            del self.agents[agent_id]
            if agent.agent_type in self.agent_types:
                self.agent_types[agent.agent_type].remove(agent_id)
                if not self.agent_types[agent.agent_type]:
                    del self.agent_types[agent.agent_type]
            
            logger.info(f"Unregistered agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister agent {agent_id}: {str(e)}")
            return False
    
    async def submit_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        target_agent_type: str = None,
        target_agent_id: str = None,
        deadline: datetime = None,
        dependencies: List[str] = None
    ) -> str:
        """
        Submit a task to the framework for processing.
        
        Args:
            task_type: Type of task to process
            payload: Task payload/data
            priority: Task priority
            target_agent_type: Specific agent type to handle the task
            target_agent_id: Specific agent ID to handle the task
            deadline: Task deadline
            dependencies: Task dependencies
            
        Returns:
            Task ID
        """
        try:
            # Create task
            task_id = str(uuid.uuid4())
            task = Task(
                id=task_id,
                agent_id=target_agent_id or "framework",
                task_type=task_type,
                payload=payload,
                priority=priority,
                deadline=deadline,
                dependencies=dependencies or []
            )
            
            # Submit to task scheduler
            await self.task_scheduler.schedule_task(task, target_agent_type)
            
            logger.info(f"Submitted task {task_id} of type {task_type}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to submit task: {str(e)}")
            raise
    
    async def send_message(self, message: AgentMessage) -> bool:
        """
        Send a message through the framework's message bus.
        
        Args:
            message: Message to send
            
        Returns:
            True if message was sent successfully
        """
        try:
            await self.message_bus.put(message)
            logger.debug(f"Message queued: {message.id} from {message.sender} to {message.recipient}")
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {str(e)}")
            return False
    
    async def _agent_message_processor(self, agent: BaseAgent):
        """Process messages for a specific agent"""
        while True:
            try:
                # Check if agent still registered
                if agent.agent_id not in self.agents:
                    break
                
                # Process tasks from agent queue
                try:
                    task = await asyncio.wait_for(agent.task_queue.get(), timeout=1.0)
                    result = await agent.process_task(task)
                    
                    # Update metrics
                    self.metrics_storage.append({
                        "agent_id": agent.agent_id,
                        "task_id": task.id,
                        "result": result,
                        "timestamp": datetime.now(),
                        "processing_time": (datetime.now() - task.created_at).total_seconds()
                    })
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error processing task for agent {agent.agent_id}: {str(e)}")
                    
                    # Record error metrics
                    self.metrics_storage.append({
                        "agent_id": agent.agent_id,
                        "error": str(e),
                        "timestamp": datetime.now(),
                        "error_info": traceback.format_exc()
                    })
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in agent message processor for {agent.agent_id}: {str(e)}")
                await asyncio.sleep(1)
    
    async def get_agent_metrics(self, agent_id: str, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Get metrics for a specific agent.
        
        Args:
            agent_id: Agent ID
            time_window_hours: Time window for metrics
            
        Returns:
            Agent metrics
        """
        if agent_id not in self.agents:
            return {"error": "Agent not found"}
        
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        agent_metrics = [
            m for m in self.metrics_storage 
            if m.get("agent_id") == agent_id and m["timestamp"] >= cutoff_time
        ]
        
        if not agent_metrics:
            return {"error": "No metrics found for agent"}
        
        # Calculate metrics
        successful_tasks = len([m for m in agent_metrics if "result" in m])
        failed_tasks = len([m for m in agent_metrics if "error" in m])
        total_tasks = successful_tasks + failed_tasks
        
        processing_times = [m.get("processing_time", 0) for m in agent_metrics if "processing_time" in m]
        avg_duration = sum(processing_times) / len(processing_times) if processing_times else 0
        
        task_types = {}
        for m in agent_metrics:
            if "result" in m:
                task_type = m.get("task_type", "unknown")
                task_types[task_type] = task_types.get(task_type, 0) + 1
        
        return {
            "agent_id": agent_id,
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
            "error_rate": failed_tasks / total_tasks if total_tasks > 0 else 0,
            "avg_duration": avg_duration,
            "task_type_distribution": task_types,
            "recent_errors": [m['error_info'] for m in agent_metrics if m.get('error_info')],
            "performance_trend": self._calculate_performance_trend(agent_metrics)
        }
    
    async def get_system_metrics(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get system-wide metrics"""
        
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        recent_metrics = [
            m for m in self.metrics_storage 
            if m['timestamp'] >= cutoff_time
        ]
        
        if not recent_metrics:
            return {'error': 'No system metrics found'}
        
        # System-wide calculations
        total_tasks = len([m for m in recent_metrics if "result" in m])
        total_errors = len([m for m in recent_metrics if "error" in m])
        
        # Agent performance
        agent_performance = {}
        for agent_id in self.agents.keys():
            agent_metrics = await self.get_agent_metrics(agent_id, time_window_hours)
            if "error" not in agent_metrics:
                agent_performance[agent_id] = agent_metrics
        
        return {
            "framework_name": self.framework_name,
            "total_agents": len(self.agents),
            "agent_types": list(self.agent_types.keys()),
            "total_tasks": total_tasks,
            "total_errors": total_errors,
            "system_health": "healthy" if total_errors / (total_tasks + total_errors) < 0.1 else "degraded",
            "agent_performance": agent_performance,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_performance_trend(self, metrics: List[Dict[str, Any]]) -> List[float]:
        """Calculate performance trend from metrics"""
        # Simple trend calculation - could be enhanced with more sophisticated analysis
        if len(metrics) < 2:
            return [0.0]
        
        processing_times = [m.get("processing_time", 0) for m in metrics if "processing_time" in m]
        if len(processing_times) < 2:
            return [0.0]
        
        # Calculate moving average trend
        window_size = min(5, len(processing_times))
        trend = []
        for i in range(window_size, len(processing_times)):
            window_avg = sum(processing_times[i-window_size:i]) / window_size
            trend.append(window_avg)
        
        return trend[-10:] if trend else [0.0]
    
    def get_registered_agents(self) -> Dict[str, Any]:
        """Get information about all registered agents"""
        return {
            agent_id: {
                "agent_type": agent.agent_type,
                "name": agent.name,
                "status": agent.status.value,
                "capabilities": asdict(agent.capabilities),
                "current_tasks": len(agent.current_tasks),
                "metrics": agent.metrics
            }
            for agent_id, agent in self.agents.items()
        }


class TaskScheduler:
    """Task scheduler for the agentic computing framework"""
    
    def __init__(self, framework: 'AgenticComputingFramework'):
        self.framework = framework
        self.scheduled_tasks = {}
    
    async def schedule_task(self, task: Task, target_agent_type: str = None):
        """Schedule a task for execution"""
        try:
            # Find suitable agent
            agent = await self._find_suitable_agent(task, target_agent_type)
            if not agent:
                raise ValueError(f"No suitable agent found for task type {task.task_type}")
            
            # Assign task to agent
            await agent.task_queue.put(task)
            self.scheduled_tasks[task.id] = {
                "task": task,
                "assigned_agent": agent.agent_id,
                "scheduled_at": datetime.now()
            }
            
            logger.info(f"Scheduled task {task.id} to agent {agent.agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to schedule task {task.id}: {str(e)}")
            raise
    
    async def _find_suitable_agent(self, task: Task, target_agent_type: str = None) -> Optional[BaseAgent]:
        """Find a suitable agent for the task"""
        
        # Filter agents by type if specified
        candidate_agents = []
        if target_agent_type:
            agent_ids = self.framework.agent_types.get(target_agent_type, [])
            candidate_agents = [self.framework.agents[aid] for aid in agent_ids if aid in self.framework.agents]
        else:
            candidate_agents = list(self.framework.agents.values())
        
        # Filter by capabilities
        suitable_agents = [
            agent for agent in candidate_agents
            if task.task_type in agent.capabilities.task_types
            and len(agent.current_tasks) < agent.capabilities.max_concurrent_tasks
            and agent.status == AgentStatus.IDLE
        ]
        
        if not suitable_agents:
            return None
        
        # Use load balancer to select best agent
        return await self.framework.load_balancer.select_agent(suitable_agents, task)


class LoadBalancer:
    """Load balancer for agent selection"""
    
    def __init__(self, framework: 'AgenticComputingFramework'):
        self.framework = framework
    
    async def select_agent(self, agents: List[BaseAgent], task: Task) -> BaseAgent:
        """Select the best agent for a task based on load balancing strategy"""
        
        if not agents:
            return None
        
        # Simple round-robin with load consideration
        # In production, this could be more sophisticated (weighted, least connections, etc.)
        
        # Sort by current load (number of active tasks)
        agents.sort(key=lambda a: len(a.current_tasks))
        
        # Select agent with lowest load
        return agents[0]


class MonitoringSystem:
    """Monitoring system for the agentic computing framework"""
    
    def __init__(self, framework: 'AgenticComputingFramework'):
        self.framework = framework
        self.alerts = []
    
    async def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health"""
        
        health_status = {
            "overall": "healthy",
            "agents": {},
            "alerts": []
        }
        
        # Check agent health
        for agent_id, agent in self.framework.agents.items():
            agent_health = {
                "status": agent.status.value,
                "current_tasks": len(agent.current_tasks),
                "max_tasks": agent.capabilities.max_concurrent_tasks,
                "last_activity": agent.metrics["last_activity"].isoformat()
            }
            
            # Check for issues
            if agent.status == AgentStatus.ERROR:
                health_status["alerts"].append(f"Agent {agent_id} is in error state")
                health_status["overall"] = "degraded"
            elif len(agent.current_tasks) >= agent.capabilities.max_concurrent_tasks:
                health_status["alerts"].append(f"Agent {agent_id} is at capacity")
            
            health_status["agents"][agent_id] = agent_health
        
        return health_status
