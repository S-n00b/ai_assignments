"""
CrewAI Integration Module for Enhanced Multi-Agent Orchestration

This module provides sophisticated CrewAI integration for the Lenovo AAITC AI Architecture
framework, enabling advanced multi-agent collaboration, task decomposition, and workflow
orchestration capabilities.

Key Features:
- Specialized AI agents for different enterprise tasks
- Advanced task decomposition and orchestration
- Multi-agent collaboration patterns
- Performance optimization and monitoring
- Enterprise-grade workflow management
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor

# CrewAI imports
try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import BaseTool
    from crewai.process import Process
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    logging.warning("CrewAI not available. Install with: pip install crewai")

# LangChain imports for enhanced tool integration
try:
    from langchain.tools import BaseTool as LangChainTool
    from langchain.agents import Tool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available. Install with: pip install langchain")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentSpecialization(Enum):
    """Specialized agent types for enterprise AI tasks"""
    DATA_ANALYST = "data_analyst"
    MODEL_EVALUATOR = "model_evaluator"
    SYSTEM_ARCHITECT = "system_architect"
    SECURITY_EXPERT = "security_expert"
    PERFORMANCE_OPTIMIZER = "performance_optimizer"
    COMPLIANCE_OFFICER = "compliance_officer"
    CUSTOMER_SUCCESS = "customer_success"
    DEVOPS_ENGINEER = "devops_engineer"
    RESEARCH_SCIENTIST = "research_scientist"
    PRODUCT_MANAGER = "product_manager"


class TaskComplexity(Enum):
    """Task complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ENTERPRISE = "enterprise"


@dataclass
class AgentProfile:
    """Comprehensive agent profile for specialized roles"""
    specialization: AgentSpecialization
    expertise_level: str  # "junior", "senior", "expert"
    domain_knowledge: List[str]
    technical_skills: List[str]
    soft_skills: List[str]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    collaboration_style: str = "collaborative"
    communication_preference: str = "detailed"


@dataclass
class TaskDecomposition:
    """Task decomposition structure for complex workflows"""
    parent_task_id: str
    subtasks: List[Dict[str, Any]]
    dependencies: Dict[str, List[str]]
    estimated_complexity: TaskComplexity
    required_agents: List[AgentSpecialization]
    parallel_execution: bool = False
    deadline: Optional[datetime] = None


class LenovoEnterpriseAgent:
    """
    Specialized Lenovo enterprise agent with domain-specific capabilities.
    Each agent is tailored for specific business functions and enterprise requirements.
    """
    
    def __init__(
        self,
        agent_id: str,
        specialization: AgentSpecialization,
        profile: AgentProfile,
        llm_model: str = "gpt-4",
        tools: List[BaseTool] = None
    ):
        """
        Initialize a Lenovo enterprise agent.
        
        Args:
            agent_id: Unique agent identifier
            specialization: Agent specialization type
            profile: Comprehensive agent profile
            llm_model: LLM model to use
            tools: Available tools for the agent
        """
        self.agent_id = agent_id
        self.specialization = specialization
        self.profile = profile
        self.llm_model = llm_model
        self.tools = tools or []
        
        # Agent state
        self.current_tasks = []
        self.task_history = []
        self.collaboration_sessions = []
        self.performance_data = {}
        
        # Initialize CrewAI agent
        self.crewai_agent = self._create_crewai_agent()
        
        logger.info(f"Lenovo Enterprise Agent initialized: {agent_id} ({specialization.value})")
    
    def _create_crewai_agent(self) -> Optional[Agent]:
        """Create the underlying CrewAI agent with specialized configuration."""
        if not CREWAI_AVAILABLE:
            logger.warning("CrewAI not available")
            return None
        
        try:
            # Generate specialized role and goal based on specialization
            role, goal, backstory = self._generate_agent_identity()
            
            agent = Agent(
                role=role,
                goal=goal,
                backstory=backstory,
                tools=self.tools,
                verbose=True,
                allow_delegation=True,
                max_iter=5,
                memory=True,
                max_execution_time=300  # 5 minutes max execution time
            )
            
            logger.info(f"CrewAI agent created for {self.agent_id}")
            return agent
            
        except Exception as e:
            logger.error(f"Error creating CrewAI agent: {e}")
            return None
    
    def _generate_agent_identity(self) -> tuple:
        """Generate role, goal, and backstory based on specialization and profile."""
        
        identity_templates = {
            AgentSpecialization.DATA_ANALYST: {
                "role": "Senior Data Analyst",
                "goal": "Provide comprehensive data analysis and insights for Lenovo's AI initiatives",
                "backstory": f"You are a {self.profile.expertise_level} data analyst with expertise in {', '.join(self.profile.domain_knowledge)}. You excel at turning complex data into actionable business insights."
            },
            AgentSpecialization.MODEL_EVALUATOR: {
                "role": "AI Model Evaluation Specialist",
                "goal": "Ensure AI models meet Lenovo's quality and performance standards",
                "backstory": f"You are a {self.profile.expertise_level} AI researcher specializing in model evaluation. Your technical skills include {', '.join(self.profile.technical_skills)}."
            },
            AgentSpecialization.SYSTEM_ARCHITECT: {
                "role": "Enterprise AI System Architect",
                "goal": "Design scalable and robust AI architectures for Lenovo's enterprise solutions",
                "backstory": f"You are a {self.profile.expertise_level} system architect with deep knowledge in {', '.join(self.profile.domain_knowledge)}. You excel at designing enterprise-grade AI systems."
            },
            AgentSpecialization.SECURITY_EXPERT: {
                "role": "AI Security Specialist",
                "goal": "Ensure AI systems meet Lenovo's security and compliance requirements",
                "backstory": f"You are a {self.profile.expertise_level} security expert specializing in AI systems. You have extensive knowledge in {', '.join(self.profile.technical_skills)}."
            },
            AgentSpecialization.PERFORMANCE_OPTIMIZER: {
                "role": "AI Performance Optimization Expert",
                "goal": "Optimize AI system performance for maximum efficiency and scalability",
                "backstory": f"You are a {self.profile.expertise_level} performance engineer with expertise in {', '.join(self.profile.technical_skills)}. You excel at optimizing AI systems for enterprise scale."
            },
            AgentSpecialization.COMPLIANCE_OFFICER: {
                "role": "AI Compliance and Governance Officer",
                "goal": "Ensure AI systems comply with regulatory requirements and ethical standards",
                "backstory": f"You are a {self.profile.expertise_level} compliance officer specializing in AI governance. You have deep knowledge in {', '.join(self.profile.domain_knowledge)}."
            },
            AgentSpecialization.CUSTOMER_SUCCESS: {
                "role": "AI Customer Success Manager",
                "goal": "Ensure AI solutions deliver value to Lenovo's customers",
                "backstory": f"You are a {self.profile.expertise_level} customer success manager with expertise in {', '.join(self.profile.soft_skills)}. You excel at understanding customer needs and ensuring AI solutions meet them."
            },
            AgentSpecialization.DEVOPS_ENGINEER: {
                "role": "AI DevOps Engineer",
                "goal": "Implement and maintain AI infrastructure and deployment pipelines",
                "backstory": f"You are a {self.profile.expertise_level} DevOps engineer specializing in AI infrastructure. Your technical skills include {', '.join(self.profile.technical_skills)}."
            },
            AgentSpecialization.RESEARCH_SCIENTIST: {
                "role": "AI Research Scientist",
                "goal": "Advance Lenovo's AI capabilities through cutting-edge research",
                "backstory": f"You are a {self.profile.expertise_level} research scientist with expertise in {', '.join(self.profile.domain_knowledge)}. You excel at pushing the boundaries of AI technology."
            },
            AgentSpecialization.PRODUCT_MANAGER: {
                "role": "AI Product Manager",
                "goal": "Define and deliver AI products that meet market needs",
                "backstory": f"You are a {self.profile.expertise_level} product manager with expertise in {', '.join(self.profile.soft_skills)}. You excel at translating market needs into successful AI products."
            }
        }
        
        template = identity_templates.get(self.specialization, identity_templates[AgentSpecialization.DATA_ANALYST])
        return template["role"], template["goal"], template["backstory"]
    
    async def execute_task(self, task_description: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a task using the specialized agent."""
        try:
            if not self.crewai_agent:
                raise Exception("CrewAI agent not initialized")
            
            # Record task start
            task_id = str(uuid.uuid4())
            task_start = datetime.now()
            
            # Create CrewAI task
            task = Task(
                description=task_description,
                agent=self.crewai_agent,
                context=context or {},
                expected_output="Detailed analysis and recommendations"
            )
            
            # Execute task
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: task.execute()
            )
            
            # Record task completion
            task_duration = (datetime.now() - task_start).total_seconds()
            self._record_task_completion(task_id, task_description, result, task_duration)
            
            return {
                "success": True,
                "task_id": task_id,
                "result": result,
                "agent_id": self.agent_id,
                "specialization": self.specialization.value,
                "execution_time": task_duration,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error executing task for agent {self.agent_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent_id": self.agent_id,
                "specialization": self.specialization.value,
                "timestamp": datetime.now().isoformat()
            }
    
    def _record_task_completion(self, task_id: str, description: str, result: Any, duration: float):
        """Record task completion for performance tracking."""
        task_record = {
            "task_id": task_id,
            "description": description,
            "result": result,
            "duration": duration,
            "completed_at": datetime.now(),
            "agent_id": self.agent_id,
            "specialization": self.specialization.value
        }
        
        self.task_history.append(task_record)
        
        # Update performance metrics
        if "task_count" not in self.performance_data:
            self.performance_data["task_count"] = 0
            self.performance_data["total_duration"] = 0.0
            self.performance_data["avg_duration"] = 0.0
        
        self.performance_data["task_count"] += 1
        self.performance_data["total_duration"] += duration
        self.performance_data["avg_duration"] = (
            self.performance_data["total_duration"] / self.performance_data["task_count"]
        )
    
    async def collaborate_with_agent(self, other_agent: 'LenovoEnterpriseAgent', 
                                   collaboration_task: str) -> Dict[str, Any]:
        """Initiate collaboration with another specialized agent."""
        try:
            session_id = str(uuid.uuid4())
            collaboration_start = datetime.now()
            
            # Create collaboration session
            session = {
                "session_id": session_id,
                "participants": [self.agent_id, other_agent.agent_id],
                "task": collaboration_task,
                "start_time": collaboration_start,
                "status": "active"
            }
            
            self.collaboration_sessions.append(session)
            
            # Create collaborative task for CrewAI
            if self.crewai_agent and other_agent.crewai_agent:
                crew = Crew(
                    agents=[self.crewai_agent, other_agent.crewai_agent],
                    tasks=[
                        Task(
                            description=collaboration_task,
                            agent=self.crewai_agent
                        )
                    ],
                    process=Process.sequential,
                    verbose=True
                )
                
                # Execute collaborative task
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: crew.kickoff()
                )
                
                # Record collaboration completion
                session["end_time"] = datetime.now()
                session["status"] = "completed"
                session["result"] = result
                
                return {
                    "success": True,
                    "session_id": session_id,
                    "result": result,
                    "participants": [self.agent_id, other_agent.agent_id],
                    "duration": (session["end_time"] - session["start_time"]).total_seconds(),
                    "timestamp": datetime.now().isoformat()
                }
            
            return {
                "success": False,
                "error": "Collaboration agents not available",
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"Error in agent collaboration: {e}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id if 'session_id' in locals() else None
            }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for the agent."""
        return {
            "agent_id": self.agent_id,
            "specialization": self.specialization.value,
            "profile": {
                "expertise_level": self.profile.expertise_level,
                "domain_knowledge": self.profile.domain_knowledge,
                "technical_skills": self.profile.technical_skills
            },
            "performance": self.performance_data,
            "task_history_count": len(self.task_history),
            "collaboration_sessions": len(self.collaboration_sessions),
            "recent_tasks": self.task_history[-5:] if self.task_history else []
        }


class LenovoCrewOrchestrator:
    """
    Advanced CrewAI orchestrator for Lenovo enterprise AI workflows.
    Manages complex multi-agent teams and sophisticated task orchestration.
    """
    
    def __init__(self):
        """Initialize the Lenovo Crew orchestrator."""
        self.enterprise_agents = {}
        self.active_crews = {}
        self.workflow_templates = {}
        self.performance_analytics = {}
        
        logger.info("Lenovo Crew Orchestrator initialized")
    
    async def register_enterprise_agent(self, agent: LenovoEnterpriseAgent) -> bool:
        """Register an enterprise agent with the orchestrator."""
        try:
            self.enterprise_agents[agent.agent_id] = agent
            logger.info(f"Enterprise agent registered: {agent.agent_id} ({agent.specialization.value})")
            return True
        except Exception as e:
            logger.error(f"Error registering enterprise agent: {e}")
            return False
    
    async def create_specialized_crew(self, crew_id: str, 
                                    agent_specializations: List[AgentSpecialization],
                                    crew_type: str = "collaborative") -> bool:
        """Create a specialized crew with specific agent types."""
        try:
            # Find agents with matching specializations
            selected_agents = []
            for spec in agent_specializations:
                matching_agents = [
                    agent for agent in self.enterprise_agents.values()
                    if agent.specialization == spec
                ]
                if matching_agents:
                    # Select the best performing agent
                    best_agent = max(matching_agents, 
                                   key=lambda a: a.performance_data.get("task_count", 0))
                    selected_agents.append(best_agent)
            
            if not selected_agents:
                logger.error(f"No agents found for specializations: {agent_specializations}")
                return False
            
            # Create CrewAI crew
            if CREWAI_AVAILABLE:
                crewai_agents = [agent.crewai_agent for agent in selected_agents 
                               if agent.crewai_agent]
                
                if not crewai_agents:
                    logger.error("No valid CrewAI agents available")
                    return False
                
                crew = Crew(
                    agents=crewai_agents,
                    process=Process.sequential if crew_type == "sequential" else Process.hierarchical,
                    verbose=True
                )
                
                self.active_crews[crew_id] = {
                    "crew": crew,
                    "agents": selected_agents,
                    "specializations": agent_specializations,
                    "crew_type": crew_type,
                    "created_at": datetime.now(),
                    "status": "active"
                }
                
                logger.info(f"Specialized crew created: {crew_id} with {len(selected_agents)} agents")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error creating specialized crew: {e}")
            return False
    
    async def execute_enterprise_workflow(self, workflow_id: str, 
                                        workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a complex enterprise workflow using multiple crews."""
        try:
            workflow_start = datetime.now()
            
            # Parse workflow configuration
            stages = workflow_config.get("stages", [])
            dependencies = workflow_config.get("dependencies", {})
            parallel_execution = workflow_config.get("parallel_execution", False)
            
            results = {}
            execution_order = self._determine_execution_order(stages, dependencies)
            
            for stage_group in execution_order:
                if parallel_execution:
                    # Execute stages in parallel
                    stage_results = await asyncio.gather(*[
                        self._execute_workflow_stage(stage, workflow_config)
                        for stage in stage_group
                    ])
                    results.update({stage["id"]: result for stage, result in zip(stage_group, stage_results)})
                else:
                    # Execute stages sequentially
                    for stage in stage_group:
                        result = await self._execute_workflow_stage(stage, workflow_config)
                        results[stage["id"]] = result
                        
                        # Check for stage failure
                        if not result.get("success", False):
                            logger.error(f"Stage {stage['id']} failed, stopping workflow")
                            break
            
            workflow_duration = (datetime.now() - workflow_start).total_seconds()
            
            # Record workflow execution
            workflow_record = {
                "workflow_id": workflow_id,
                "config": workflow_config,
                "results": results,
                "duration": workflow_duration,
                "completed_at": datetime.now(),
                "status": "completed" if all(r.get("success", False) for r in results.values()) else "failed"
            }
            
            if workflow_id not in self.performance_analytics:
                self.performance_analytics[workflow_id] = []
            self.performance_analytics[workflow_id].append(workflow_record)
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "results": results,
                "duration": workflow_duration,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error executing enterprise workflow: {e}")
            return {
                "success": False,
                "error": str(e),
                "workflow_id": workflow_id,
                "timestamp": datetime.now().isoformat()
            }
    
    def _determine_execution_order(self, stages: List[Dict], dependencies: Dict) -> List[List[Dict]]:
        """Determine the execution order based on dependencies."""
        # Simple topological sort for dependency resolution
        execution_order = []
        remaining_stages = stages.copy()
        completed_stages = set()
        
        while remaining_stages:
            ready_stages = []
            for stage in remaining_stages:
                stage_id = stage["id"]
                stage_deps = dependencies.get(stage_id, [])
                if all(dep in completed_stages for dep in stage_deps):
                    ready_stages.append(stage)
            
            if not ready_stages:
                logger.warning("Circular dependencies detected, executing remaining stages")
                ready_stages = remaining_stages
            
            execution_order.append(ready_stages)
            for stage in ready_stages:
                completed_stages.add(stage["id"])
                remaining_stages.remove(stage)
        
        return execution_order
    
    async def _execute_workflow_stage(self, stage: Dict, workflow_config: Dict) -> Dict[str, Any]:
        """Execute a single workflow stage."""
        try:
            stage_id = stage["id"]
            stage_type = stage.get("type", "crew_execution")
            crew_id = stage.get("crew_id")
            task_description = stage.get("task_description")
            
            if stage_type == "crew_execution" and crew_id in self.active_crews:
                crew_info = self.active_crews[crew_id]
                crew = crew_info["crew"]
                
                # Create task for crew
                task = Task(
                    description=task_description,
                    context=stage.get("context", {})
                )
                
                # Execute crew task
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: crew.kickoff(inputs=task)
                )
                
                return {
                    "success": True,
                    "stage_id": stage_id,
                    "result": result,
                    "crew_id": crew_id
                }
            
            elif stage_type == "agent_collaboration":
                # Handle agent-to-agent collaboration
                agent_ids = stage.get("agent_ids", [])
                agents = [self.enterprise_agents[aid] for aid in agent_ids if aid in self.enterprise_agents]
                
                if len(agents) >= 2:
                    collaboration_result = await agents[0].collaborate_with_agent(
                        agents[1], task_description
                    )
                    return {
                        "success": collaboration_result["success"],
                        "stage_id": stage_id,
                        "result": collaboration_result,
                        "type": "collaboration"
                    }
            
            return {
                "success": False,
                "stage_id": stage_id,
                "error": f"Unknown stage type: {stage_type}"
            }
            
        except Exception as e:
            logger.error(f"Error executing workflow stage: {e}")
            return {
                "success": False,
                "stage_id": stage.get("id", "unknown"),
                "error": str(e)
            }
    
    async def get_orchestrator_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics for the orchestrator."""
        try:
            total_agents = len(self.enterprise_agents)
            total_crews = len(self.active_crews)
            total_workflows = sum(len(records) for records in self.performance_analytics.values())
            
            # Agent performance summary
            agent_performance = {}
            for agent_id, agent in self.enterprise_agents.items():
                agent_performance[agent_id] = agent.get_performance_metrics()
            
            # Crew performance summary
            crew_performance = {}
            for crew_id, crew_info in self.active_crews.items():
                crew_performance[crew_id] = {
                    "crew_id": crew_id,
                    "specializations": [spec.value for spec in crew_info["specializations"]],
                    "agent_count": len(crew_info["agents"]),
                    "crew_type": crew_info["crew_type"],
                    "created_at": crew_info["created_at"].isoformat(),
                    "status": crew_info["status"]
                }
            
            # Workflow performance summary
            workflow_performance = {}
            for workflow_id, records in self.performance_analytics.items():
                if records:
                    successful_executions = len([r for r in records if r["status"] == "completed"])
                    avg_duration = sum(r["duration"] for r in records) / len(records)
                    
                    workflow_performance[workflow_id] = {
                        "total_executions": len(records),
                        "successful_executions": successful_executions,
                        "success_rate": (successful_executions / len(records)) * 100,
                        "avg_duration": avg_duration,
                        "last_execution": records[-1]["completed_at"].isoformat()
                    }
            
            return {
                "orchestrator_summary": {
                    "total_agents": total_agents,
                    "total_crews": total_crews,
                    "total_workflows": total_workflows,
                    "timestamp": datetime.now().isoformat()
                },
                "agent_performance": agent_performance,
                "crew_performance": crew_performance,
                "workflow_performance": workflow_performance
            }
            
        except Exception as e:
            logger.error(f"Error getting orchestrator analytics: {e}")
            return {"error": str(e)}


class LenovoTaskDecomposer:
    """
    Advanced task decomposition system for complex enterprise workflows.
    Breaks down complex tasks into manageable subtasks with proper dependencies.
    """
    
    def __init__(self):
        """Initialize the task decomposer."""
        self.decomposition_templates = {}
        self.complexity_analyzers = {}
        self.dependency_resolvers = {}
        
        logger.info("Lenovo Task Decomposer initialized")
    
    async def decompose_task(self, task_description: str, 
                           complexity: TaskComplexity = TaskComplexity.MODERATE,
                           target_agents: List[AgentSpecialization] = None) -> TaskDecomposition:
        """Decompose a complex task into manageable subtasks."""
        try:
            parent_task_id = str(uuid.uuid4())
            
            # Analyze task complexity
            analyzed_complexity = await self._analyze_task_complexity(task_description)
            
            # Generate subtasks based on complexity and target agents
            subtasks = await self._generate_subtasks(
                task_description, analyzed_complexity, target_agents
            )
            
            # Resolve dependencies between subtasks
            dependencies = await self._resolve_dependencies(subtasks)
            
            # Create task decomposition
            decomposition = TaskDecomposition(
                parent_task_id=parent_task_id,
                subtasks=subtasks,
                dependencies=dependencies,
                estimated_complexity=analyzed_complexity,
                required_agents=target_agents or [],
                parallel_execution=self._can_parallelize(subtasks, dependencies)
            )
            
            logger.info(f"Task decomposed: {parent_task_id} into {len(subtasks)} subtasks")
            return decomposition
            
        except Exception as e:
            logger.error(f"Error decomposing task: {e}")
            raise
    
    async def _analyze_task_complexity(self, task_description: str) -> TaskComplexity:
        """Analyze the complexity of a task based on various factors."""
        # Simple complexity analysis based on keywords and length
        complexity_indicators = {
            TaskComplexity.SIMPLE: ["analyze", "review", "summarize"],
            TaskComplexity.MODERATE: ["design", "implement", "evaluate", "compare"],
            TaskComplexity.COMPLEX: ["architect", "integrate", "optimize", "deploy"],
            TaskComplexity.ENTERPRISE: ["transform", "migrate", "scale", "govern"]
        }
        
        task_lower = task_description.lower()
        
        for complexity, indicators in complexity_indicators.items():
            if any(indicator in task_lower for indicator in indicators):
                return complexity
        
        # Default based on description length
        if len(task_description) < 100:
            return TaskComplexity.SIMPLE
        elif len(task_description) < 300:
            return TaskComplexity.MODERATE
        elif len(task_description) < 600:
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.ENTERPRISE
    
    async def _generate_subtasks(self, task_description: str, 
                               complexity: TaskComplexity,
                               target_agents: List[AgentSpecialization] = None) -> List[Dict[str, Any]]:
        """Generate subtasks based on task complexity and target agents."""
        subtasks = []
        
        if complexity == TaskComplexity.SIMPLE:
            subtasks = [{
                "id": str(uuid.uuid4()),
                "description": task_description,
                "assigned_agent": target_agents[0].value if target_agents else None,
                "estimated_duration": 30,
                "priority": "normal"
            }]
        
        elif complexity == TaskComplexity.MODERATE:
            subtasks = [
                {
                    "id": str(uuid.uuid4()),
                    "description": f"Analyze requirements for: {task_description}",
                    "assigned_agent": AgentSpecialization.DATA_ANALYST.value,
                    "estimated_duration": 60,
                    "priority": "high"
                },
                {
                    "id": str(uuid.uuid4()),
                    "description": f"Design solution for: {task_description}",
                    "assigned_agent": AgentSpecialization.SYSTEM_ARCHITECT.value,
                    "estimated_duration": 90,
                    "priority": "normal"
                }
            ]
        
        elif complexity == TaskComplexity.COMPLEX:
            subtasks = [
                {
                    "id": str(uuid.uuid4()),
                    "description": f"Research and analyze: {task_description}",
                    "assigned_agent": AgentSpecialization.RESEARCH_SCIENTIST.value,
                    "estimated_duration": 120,
                    "priority": "high"
                },
                {
                    "id": str(uuid.uuid4()),
                    "description": f"Design architecture for: {task_description}",
                    "assigned_agent": AgentSpecialization.SYSTEM_ARCHITECT.value,
                    "estimated_duration": 150,
                    "priority": "high"
                },
                {
                    "id": str(uuid.uuid4()),
                    "description": f"Implement security measures for: {task_description}",
                    "assigned_agent": AgentSpecialization.SECURITY_EXPERT.value,
                    "estimated_duration": 90,
                    "priority": "normal"
                },
                {
                    "id": str(uuid.uuid4()),
                    "description": f"Optimize performance for: {task_description}",
                    "assigned_agent": AgentSpecialization.PERFORMANCE_OPTIMIZER.value,
                    "estimated_duration": 120,
                    "priority": "normal"
                }
            ]
        
        else:  # ENTERPRISE
            subtasks = [
                {
                    "id": str(uuid.uuid4()),
                    "description": f"Market research and analysis for: {task_description}",
                    "assigned_agent": AgentSpecialization.RESEARCH_SCIENTIST.value,
                    "estimated_duration": 180,
                    "priority": "high"
                },
                {
                    "id": str(uuid.uuid4()),
                    "description": f"Product strategy for: {task_description}",
                    "assigned_agent": AgentSpecialization.PRODUCT_MANAGER.value,
                    "estimated_duration": 150,
                    "priority": "high"
                },
                {
                    "id": str(uuid.uuid4()),
                    "description": f"Enterprise architecture design for: {task_description}",
                    "assigned_agent": AgentSpecialization.SYSTEM_ARCHITECT.value,
                    "estimated_duration": 240,
                    "priority": "high"
                },
                {
                    "id": str(uuid.uuid4()),
                    "description": f"Security and compliance framework for: {task_description}",
                    "assigned_agent": AgentSpecialization.SECURITY_EXPERT.value,
                    "estimated_duration": 180,
                    "priority": "high"
                },
                {
                    "id": str(uuid.uuid4()),
                    "description": f"DevOps and deployment strategy for: {task_description}",
                    "assigned_agent": AgentSpecialization.DEVOPS_ENGINEER.value,
                    "estimated_duration": 120,
                    "priority": "normal"
                },
                {
                    "id": str(uuid.uuid4()),
                    "description": f"Customer success planning for: {task_description}",
                    "assigned_agent": AgentSpecialization.CUSTOMER_SUCCESS.value,
                    "estimated_duration": 90,
                    "priority": "normal"
                }
            ]
        
        return subtasks
    
    async def _resolve_dependencies(self, subtasks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Resolve dependencies between subtasks."""
        dependencies = {}
        
        # Simple dependency resolution based on agent types and priorities
        for i, subtask in enumerate(subtasks):
            subtask_id = subtask["id"]
            subtask_agent = subtask["assigned_agent"]
            subtask_priority = subtask["priority"]
            
            deps = []
            
            # Research tasks should come before design tasks
            if subtask_agent == AgentSpecialization.SYSTEM_ARCHITECT.value:
                research_tasks = [
                    st["id"] for st in subtasks 
                    if st["assigned_agent"] == AgentSpecialization.RESEARCH_SCIENTIST.value
                ]
                deps.extend(research_tasks)
            
            # Design tasks should come before implementation tasks
            if subtask_agent in [AgentSpecialization.DEVOPS_ENGINEER.value, 
                               AgentSpecialization.PERFORMANCE_OPTIMIZER.value]:
                design_tasks = [
                    st["id"] for st in subtasks 
                    if st["assigned_agent"] == AgentSpecialization.SYSTEM_ARCHITECT.value
                ]
                deps.extend(design_tasks)
            
            # Security tasks can run in parallel with design but before implementation
            if subtask_agent == AgentSpecialization.SECURITY_EXPERT.value:
                high_priority_tasks = [
                    st["id"] for st in subtasks 
                    if st["priority"] == "high" and st["assigned_agent"] != subtask_agent
                ]
                deps.extend(high_priority_tasks)
            
            dependencies[subtask_id] = deps
        
        return dependencies
    
    def _can_parallelize(self, subtasks: List[Dict[str, Any]], 
                        dependencies: Dict[str, List[str]]) -> bool:
        """Determine if subtasks can be executed in parallel."""
        # Check if there are independent subtasks
        independent_tasks = [
            task_id for task_id, deps in dependencies.items() 
            if not deps
        ]
        
        return len(independent_tasks) > 1


# Factory function for creating specialized Lenovo agents
def create_lenovo_enterprise_agent(
    specialization: AgentSpecialization,
    agent_id: str = None,
    expertise_level: str = "senior",
    custom_tools: List[BaseTool] = None
) -> LenovoEnterpriseAgent:
    """
    Factory function to create specialized Lenovo enterprise agents.
    
    Args:
        specialization: Agent specialization type
        agent_id: Optional custom agent ID
        expertise_level: Agent expertise level
        custom_tools: Optional custom tools for the agent
    
    Returns:
        Configured LenovoEnterpriseAgent instance
    """
    
    # Generate agent ID if not provided
    if not agent_id:
        agent_id = f"lenovo_{specialization.value}_{uuid.uuid4().hex[:8]}"
    
    # Create agent profile based on specialization
    profile = AgentProfile(
        specialization=specialization,
        expertise_level=expertise_level,
        domain_knowledge=_get_domain_knowledge(specialization),
        technical_skills=_get_technical_skills(specialization),
        soft_skills=_get_soft_skills(specialization)
    )
    
    return LenovoEnterpriseAgent(
        agent_id=agent_id,
        specialization=specialization,
        profile=profile,
        tools=custom_tools or []
    )


def _get_domain_knowledge(specialization: AgentSpecialization) -> List[str]:
    """Get domain knowledge for a specialization."""
    domain_knowledge_map = {
        AgentSpecialization.DATA_ANALYST: ["data analysis", "statistics", "business intelligence"],
        AgentSpecialization.MODEL_EVALUATOR: ["machine learning", "model evaluation", "AI metrics"],
        AgentSpecialization.SYSTEM_ARCHITECT: ["system design", "cloud architecture", "scalability"],
        AgentSpecialization.SECURITY_EXPERT: ["cybersecurity", "compliance", "risk management"],
        AgentSpecialization.PERFORMANCE_OPTIMIZER: ["performance tuning", "optimization", "monitoring"],
        AgentSpecialization.COMPLIANCE_OFFICER: ["regulatory compliance", "governance", "ethics"],
        AgentSpecialization.CUSTOMER_SUCCESS: ["customer relations", "product adoption", "support"],
        AgentSpecialization.DEVOPS_ENGINEER: ["CI/CD", "infrastructure", "automation"],
        AgentSpecialization.RESEARCH_SCIENTIST: ["AI research", "innovation", "technology trends"],
        AgentSpecialization.PRODUCT_MANAGER: ["product strategy", "market analysis", "roadmapping"]
    }
    return domain_knowledge_map.get(specialization, ["general business"])


def _get_technical_skills(specialization: AgentSpecialization) -> List[str]:
    """Get technical skills for a specialization."""
    technical_skills_map = {
        AgentSpecialization.DATA_ANALYST: ["Python", "SQL", "pandas", "visualization"],
        AgentSpecialization.MODEL_EVALUATOR: ["Python", "ML frameworks", "evaluation metrics"],
        AgentSpecialization.SYSTEM_ARCHITECT: ["cloud platforms", "microservices", "API design"],
        AgentSpecialization.SECURITY_EXPERT: ["security frameworks", "penetration testing", "encryption"],
        AgentSpecialization.PERFORMANCE_OPTIMIZER: ["profiling", "monitoring", "optimization tools"],
        AgentSpecialization.COMPLIANCE_OFFICER: ["regulatory frameworks", "audit tools", "documentation"],
        AgentSpecialization.CUSTOMER_SUCCESS: ["CRM systems", "analytics", "communication tools"],
        AgentSpecialization.DEVOPS_ENGINEER: ["Kubernetes", "Docker", "Terraform", "CI/CD"],
        AgentSpecialization.RESEARCH_SCIENTIST: ["research methodologies", "experimental design", "publication"],
        AgentSpecialization.PRODUCT_MANAGER: ["product management tools", "analytics", "strategy frameworks"]
    }
    return technical_skills_map.get(specialization, ["general technical skills"])


def _get_soft_skills(specialization: AgentSpecialization) -> List[str]:
    """Get soft skills for a specialization."""
    soft_skills_map = {
        AgentSpecialization.DATA_ANALYST: ["analytical thinking", "attention to detail", "communication"],
        AgentSpecialization.MODEL_EVALUATOR: ["critical thinking", "problem solving", "quality focus"],
        AgentSpecialization.SYSTEM_ARCHITECT: ["system thinking", "leadership", "innovation"],
        AgentSpecialization.SECURITY_EXPERT: ["risk assessment", "attention to detail", "compliance mindset"],
        AgentSpecialization.PERFORMANCE_OPTIMIZER: ["analytical thinking", "continuous improvement", "efficiency"],
        AgentSpecialization.COMPLIANCE_OFFICER: ["regulatory knowledge", "attention to detail", "ethical judgment"],
        AgentSpecialization.CUSTOMER_SUCCESS: ["empathy", "communication", "problem solving"],
        AgentSpecialization.DEVOPS_ENGINEER: ["automation mindset", "collaboration", "continuous learning"],
        AgentSpecialization.RESEARCH_SCIENTIST: ["curiosity", "analytical thinking", "innovation"],
        AgentSpecialization.PRODUCT_MANAGER: ["strategic thinking", "stakeholder management", "market awareness"]
    }
    return soft_skills_map.get(specialization, ["general soft skills"])
