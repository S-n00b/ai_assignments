"""
SmolAgent Workflow Designer

This module provides SmolAgent workflow design capabilities for creating
agentic workflows optimized for small models and mobile deployment.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from pathlib import Path

# SmolAgent imports
try:
    from smolagent import SmolAgent, Workflow, Task, Agent
    from smolagent.workflows import WorkflowBuilder
    from smolagent.agents import AgentBuilder
except ImportError:
    # Fallback for development
    SmolAgent = None
    Workflow = None
    Task = None
    Agent = None
    WorkflowBuilder = None
    AgentBuilder = None

logger = logging.getLogger(__name__)


class WorkflowType(Enum):
    """Types of agentic workflows supported."""
    SIMPLE_CHAT = "simple_chat"
    TASK_ORCHESTRATION = "task_orchestration"
    MULTI_AGENT_COLLABORATION = "multi_agent_collaboration"
    MOBILE_OPTIMIZED = "mobile_optimized"
    EDGE_DEPLOYMENT = "edge_deployment"


class AgentRole(Enum):
    """Agent roles in workflows."""
    COORDINATOR = "coordinator"
    EXECUTOR = "executor"
    ANALYZER = "analyzer"
    COMMUNICATOR = "communicator"
    OPTIMIZER = "optimizer"


@dataclass
class AgentConfig:
    """Configuration for an agent in a workflow."""
    name: str
    role: AgentRole
    model_name: str
    system_prompt: str
    max_tokens: int = 2048
    temperature: float = 0.7
    mobile_optimized: bool = False
    memory_limit: int = 1000
    capabilities: List[str] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []


@dataclass
class WorkflowConfig:
    """Configuration for a SmolAgent workflow."""
    name: str
    description: str
    workflow_type: WorkflowType
    agents: List[AgentConfig]
    tasks: List[Dict[str, Any]]
    mobile_optimized: bool = False
    max_concurrent_agents: int = 3
    timeout_seconds: int = 300
    retry_attempts: int = 3
    memory_management: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class SmolAgentWorkflowDesigner:
    """
    SmolAgent Workflow Designer for creating and managing agentic workflows.
    
    This class provides comprehensive workflow design capabilities optimized
    for small models and mobile deployment scenarios.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the SmolAgent Workflow Designer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "config/smolagent_config.json"
        self.workflows: Dict[str, WorkflowConfig] = {}
        self.active_workflows: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Load configuration
        self._load_config()
        
        logger.info("SmolAgent Workflow Designer initialized")
    
    def _load_config(self):
        """Load configuration from file."""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    # Load existing workflows
                    for workflow_data in config.get('workflows', []):
                        workflow = WorkflowConfig(**workflow_data)
                        self.workflows[workflow.name] = workflow
                logger.info(f"Loaded {len(self.workflows)} workflows from config")
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
    
    def _save_config(self):
        """Save configuration to file."""
        try:
            config = {
                'workflows': [workflow.to_dict() for workflow in self.workflows.values()]
            }
            config_file = Path(self.config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info("Configuration saved successfully")
        except Exception as e:
            logger.error(f"Could not save config: {e}")
    
    def create_workflow(self, config: WorkflowConfig) -> str:
        """
        Create a new SmolAgent workflow.
        
        Args:
            config: Workflow configuration
            
        Returns:
            Workflow ID
        """
        try:
            # Validate configuration
            self._validate_workflow_config(config)
            
            # Store workflow configuration
            self.workflows[config.name] = config
            
            # Create SmolAgent workflow if available
            if SmolAgent is not None:
                workflow = self._build_smolagent_workflow(config)
                self.active_workflows[config.name] = workflow
                logger.info(f"Created workflow: {config.name}")
            else:
                logger.warning("SmolAgent not available, workflow created in config only")
            
            # Save configuration
            self._save_config()
            
            return config.name
            
        except Exception as e:
            logger.error(f"Failed to create workflow: {e}")
            raise
    
    def _validate_workflow_config(self, config: WorkflowConfig):
        """Validate workflow configuration."""
        if not config.name:
            raise ValueError("Workflow name is required")
        
        if not config.agents:
            raise ValueError("At least one agent is required")
        
        if not config.tasks:
            raise ValueError("At least one task is required")
        
        # Validate agent configurations
        for agent in config.agents:
            if not agent.name:
                raise ValueError("Agent name is required")
            if not agent.model_name:
                raise ValueError("Agent model name is required")
            if not agent.system_prompt:
                raise ValueError("Agent system prompt is required")
    
    def _build_smolagent_workflow(self, config: WorkflowConfig):
        """Build SmolAgent workflow from configuration."""
        try:
            # Create workflow builder
            builder = WorkflowBuilder()
            
            # Add agents
            for agent_config in config.agents:
                agent = AgentBuilder() \
                    .with_name(agent_config.name) \
                    .with_model(agent_config.model_name) \
                    .with_system_prompt(agent_config.system_prompt) \
                    .with_max_tokens(agent_config.max_tokens) \
                    .with_temperature(agent_config.temperature)
                
                if agent_config.mobile_optimized:
                    agent = agent.with_mobile_optimization()
                
                builder = builder.add_agent(agent.build())
            
            # Add tasks
            for task_data in config.tasks:
                task = Task(
                    name=task_data.get('name', ''),
                    description=task_data.get('description', ''),
                    agent=task_data.get('agent', ''),
                    inputs=task_data.get('inputs', {}),
                    outputs=task_data.get('outputs', {})
                )
                builder = builder.add_task(task)
            
            # Build workflow
            workflow = builder.build()
            
            # Configure workflow settings
            if config.mobile_optimized:
                workflow.enable_mobile_optimization()
            
            workflow.set_timeout(config.timeout_seconds)
            workflow.set_retry_attempts(config.retry_attempts)
            
            if config.memory_management:
                workflow.enable_memory_management()
            
            return workflow
            
        except Exception as e:
            logger.error(f"Failed to build SmolAgent workflow: {e}")
            raise
    
    def get_workflow(self, name: str) -> Optional[WorkflowConfig]:
        """Get workflow configuration by name."""
        return self.workflows.get(name)
    
    def list_workflows(self) -> List[str]:
        """List all workflow names."""
        return list(self.workflows.keys())
    
    def delete_workflow(self, name: str) -> bool:
        """
        Delete a workflow.
        
        Args:
            name: Workflow name
            
        Returns:
            True if deleted successfully
        """
        try:
            if name in self.workflows:
                del self.workflows[name]
            
            if name in self.active_workflows:
                del self.active_workflows[name]
            
            self._save_config()
            logger.info(f"Deleted workflow: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete workflow: {e}")
            return False
    
    def execute_workflow(self, name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a workflow with given inputs.
        
        Args:
            name: Workflow name
            inputs: Input data for the workflow
            
        Returns:
            Execution results
        """
        try:
            if name not in self.workflows:
                raise ValueError(f"Workflow '{name}' not found")
            
            if name not in self.active_workflows:
                raise ValueError(f"Workflow '{name}' is not active")
            
            workflow = self.active_workflows[name]
            
            # Execute workflow
            if SmolAgent is not None:
                results = workflow.execute(inputs)
            else:
                # Fallback execution without SmolAgent
                results = self._fallback_execution(name, inputs)
            
            # Track performance metrics
            self._track_execution_metrics(name, results)
            
            logger.info(f"Executed workflow: {name}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to execute workflow: {e}")
            raise
    
    def _fallback_execution(self, name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback execution when SmolAgent is not available."""
        config = self.workflows[name]
        
        # Simulate workflow execution
        results = {
            'workflow_name': name,
            'status': 'completed',
            'inputs': inputs,
            'outputs': {},
            'execution_time': 0.1,
            'agents_used': [agent.name for agent in config.agents]
        }
        
        # Simulate task execution
        for task_data in config.tasks:
            task_name = task_data.get('name', 'unknown_task')
            results['outputs'][task_name] = {
                'status': 'completed',
                'result': f"Simulated result for {task_name}"
            }
        
        return results
    
    def _track_execution_metrics(self, name: str, results: Dict[str, Any]):
        """Track execution metrics for performance monitoring."""
        if name not in self.performance_metrics:
            self.performance_metrics[name] = {
                'total_executions': 0,
                'successful_executions': 0,
                'failed_executions': 0,
                'average_execution_time': 0,
                'execution_times': []
            }
        
        metrics = self.performance_metrics[name]
        metrics['total_executions'] += 1
        
        if results.get('status') == 'completed':
            metrics['successful_executions'] += 1
        else:
            metrics['failed_executions'] += 1
        
        execution_time = results.get('execution_time', 0)
        metrics['execution_times'].append(execution_time)
        
        # Update average execution time
        if metrics['execution_times']:
            metrics['average_execution_time'] = sum(metrics['execution_times']) / len(metrics['execution_times'])
    
    def get_performance_metrics(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics for workflows."""
        if name:
            return self.performance_metrics.get(name, {})
        return self.performance_metrics
    
    def create_mobile_optimized_workflow(self, name: str, description: str, 
                                       agents: List[AgentConfig], 
                                       tasks: List[Dict[str, Any]]) -> str:
        """
        Create a mobile-optimized workflow.
        
        Args:
            name: Workflow name
            description: Workflow description
            agents: List of agent configurations
            tasks: List of task configurations
            
        Returns:
            Workflow ID
        """
        config = WorkflowConfig(
            name=name,
            description=description,
            workflow_type=WorkflowType.MOBILE_OPTIMIZED,
            agents=agents,
            tasks=tasks,
            mobile_optimized=True,
            max_concurrent_agents=2,  # Limit for mobile
            timeout_seconds=120,      # Shorter timeout for mobile
            memory_management=True
        )
        
        return self.create_workflow(config)
    
    def create_edge_deployment_workflow(self, name: str, description: str,
                                      agents: List[AgentConfig],
                                      tasks: List[Dict[str, Any]]) -> str:
        """
        Create an edge deployment workflow.
        
        Args:
            name: Workflow name
            description: Workflow description
            agents: List of agent configurations
            tasks: List of task configurations
            
        Returns:
            Workflow ID
        """
        config = WorkflowConfig(
            name=name,
            description=description,
            workflow_type=WorkflowType.EDGE_DEPLOYMENT,
            agents=agents,
            tasks=tasks,
            mobile_optimized=True,
            max_concurrent_agents=1,  # Single agent for edge
            timeout_seconds=60,      # Very short timeout for edge
            memory_management=True
        )
        
        return self.create_workflow(config)
    
    def export_workflow(self, name: str, export_path: str) -> bool:
        """
        Export workflow configuration to file.
        
        Args:
            name: Workflow name
            export_path: Path to export file
            
        Returns:
            True if exported successfully
        """
        try:
            if name not in self.workflows:
                raise ValueError(f"Workflow '{name}' not found")
            
            workflow_data = self.workflows[name].to_dict()
            
            with open(export_path, 'w') as f:
                json.dump(workflow_data, f, indent=2)
            
            logger.info(f"Exported workflow '{name}' to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export workflow: {e}")
            return False
    
    def import_workflow(self, import_path: str) -> str:
        """
        Import workflow configuration from file.
        
        Args:
            import_path: Path to import file
            
        Returns:
            Workflow name
        """
        try:
            with open(import_path, 'r') as f:
                workflow_data = json.load(f)
            
            config = WorkflowConfig(**workflow_data)
            self.workflows[config.name] = config
            
            # Create active workflow if SmolAgent is available
            if SmolAgent is not None:
                workflow = self._build_smolagent_workflow(config)
                self.active_workflows[config.name] = workflow
            
            self._save_config()
            logger.info(f"Imported workflow: {config.name}")
            return config.name
            
        except Exception as e:
            logger.error(f"Failed to import workflow: {e}")
            raise


# Example usage and factory functions
def create_lenovo_device_support_workflow() -> SmolAgentWorkflowDesigner:
    """Create a Lenovo device support workflow."""
    designer = SmolAgentWorkflowDesigner()
    
    # Define agents for device support
    agents = [
        AgentConfig(
            name="device_analyzer",
            role=AgentRole.ANALYZER,
            model_name="phi-4-mini",
            system_prompt="You are a Lenovo device support specialist. Analyze device issues and provide technical solutions.",
            mobile_optimized=True,
            capabilities=["device_diagnosis", "troubleshooting", "technical_analysis"]
        ),
        AgentConfig(
            name="solution_provider",
            role=AgentRole.EXECUTOR,
            model_name="llama-3.2-3b",
            system_prompt="You provide step-by-step solutions for Lenovo device issues.",
            mobile_optimized=True,
            capabilities=["solution_generation", "step_by_step_guidance"]
        )
    ]
    
    # Define tasks
    tasks = [
        {
            "name": "analyze_device_issue",
            "description": "Analyze the reported device issue",
            "agent": "device_analyzer",
            "inputs": {"issue_description": "string", "device_model": "string"},
            "outputs": {"analysis": "string", "severity": "string"}
        },
        {
            "name": "provide_solution",
            "description": "Provide solution based on analysis",
            "agent": "solution_provider",
            "inputs": {"analysis": "string", "device_model": "string"},
            "outputs": {"solution": "string", "steps": "list"}
        }
    ]
    
    # Create workflow
    workflow_id = designer.create_mobile_optimized_workflow(
        name="lenovo_device_support",
        description="Lenovo device support workflow for mobile deployment",
        agents=agents,
        tasks=tasks
    )
    
    return designer


def create_factory_roster_workflow() -> SmolAgentWorkflowDesigner:
    """Create a factory roster management workflow."""
    designer = SmolAgentWorkflowDesigner()
    
    # Define agents for factory roster
    agents = [
        AgentConfig(
            name="roster_coordinator",
            role=AgentRole.COORDINATOR,
            model_name="qwen-2.5-3b",
            system_prompt="You coordinate factory roster assignments and manage production schedules.",
            mobile_optimized=True,
            capabilities=["roster_management", "scheduling", "coordination"]
        ),
        AgentConfig(
            name="quality_monitor",
            role=AgentRole.ANALYZER,
            model_name="mistral-nemo",
            system_prompt="You monitor production quality and identify issues.",
            mobile_optimized=True,
            capabilities=["quality_analysis", "issue_detection", "monitoring"]
        )
    ]
    
    # Define tasks
    tasks = [
        {
            "name": "assign_roster",
            "description": "Assign workers to production roster",
            "agent": "roster_coordinator",
            "inputs": {"workers": "list", "production_plan": "dict"},
            "outputs": {"roster_assignment": "dict", "schedule": "dict"}
        },
        {
            "name": "monitor_quality",
            "description": "Monitor production quality",
            "agent": "quality_monitor",
            "inputs": {"production_data": "dict", "quality_metrics": "dict"},
            "outputs": {"quality_report": "dict", "issues": "list"}
        }
    ]
    
    # Create workflow
    workflow_id = designer.create_edge_deployment_workflow(
        name="factory_roster_management",
        description="Factory roster management workflow for edge deployment",
        agents=agents,
        tasks=tasks
    )
    
    return designer


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create Lenovo device support workflow
    designer = create_lenovo_device_support_workflow()
    
    # Test workflow execution
    test_inputs = {
        "issue_description": "Laptop not booting",
        "device_model": "ThinkPad X1 Carbon"
    }
    
    try:
        results = designer.execute_workflow("lenovo_device_support", test_inputs)
        print(f"Workflow execution results: {results}")
    except Exception as e:
        print(f"Workflow execution failed: {e}")
    
    # Get performance metrics
    metrics = designer.get_performance_metrics("lenovo_device_support")
    print(f"Performance metrics: {metrics}")
