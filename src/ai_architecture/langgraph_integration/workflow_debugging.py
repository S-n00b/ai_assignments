"""
Workflow Debugging

This module provides debugging capabilities for LangGraph workflows,
including step-by-step execution, breakpoints, and error analysis.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


class DebugLevel(Enum):
    """Debug levels for workflow execution."""
    NONE = "none"
    BASIC = "basic"
    DETAILED = "detailed"
    VERBOSE = "verbose"


class BreakpointType(Enum):
    """Types of breakpoints."""
    NODE_ENTRY = "node_entry"
    NODE_EXIT = "node_exit"
    CONDITION = "condition"
    ERROR = "error"
    CUSTOM = "custom"


@dataclass
class Breakpoint:
    """Breakpoint configuration."""
    breakpoint_id: str
    breakpoint_type: BreakpointType
    node_id: Optional[str] = None
    condition: Optional[str] = None
    enabled: bool = True
    action: str = "pause"  # pause, log, alert
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DebugStep:
    """Debug step information."""
    step_id: str
    node_id: str
    node_name: str
    node_type: str
    timestamp: float
    state_before: Dict[str, Any]
    state_after: Dict[str, Any]
    execution_time_ms: float
    status: str
    error_message: Optional[str] = None
    breakpoints_triggered: List[str] = None
    
    def __post_init__(self):
        if self.breakpoints_triggered is None:
            self.breakpoints_triggered = []


@dataclass
class DebugSession:
    """Debug session information."""
    session_id: str
    workflow_id: str
    start_time: float
    end_time: Optional[float] = None
    debug_level: DebugLevel = DebugLevel.BASIC
    breakpoints: List[Breakpoint] = None
    steps: List[DebugStep] = None
    status: str = "running"
    error_count: int = 0
    
    def __post_init__(self):
        if self.breakpoints is None:
            self.breakpoints = []
        if self.steps is None:
            self.steps = []


class WorkflowDebugger:
    """
    Workflow Debugger for LangGraph workflows.
    
    This class provides comprehensive debugging capabilities
    including step-by-step execution, breakpoints, and error analysis.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Workflow Debugger.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "config/workflow_debugging.json"
        self.active_sessions: Dict[str, DebugSession] = {}
        self.debug_callbacks: List[Callable[[DebugStep], None]] = []
        self.error_handlers: List[Callable[[DebugStep], None]] = []
        
        # Load configuration
        self._load_config()
        
        logger.info("Workflow Debugger initialized")
    
    def _load_config(self):
        """Load debugging configuration."""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    logger.info(f"Loaded debugging configuration from {config_file}")
        except Exception as e:
            logger.warning(f"Could not load debugging config: {e}")
    
    def _save_config(self):
        """Save debugging configuration."""
        try:
            config = {
                'active_sessions': len(self.active_sessions),
                'debug_callbacks': len(self.debug_callbacks),
                'error_handlers': len(self.error_handlers)
            }
            config_file = Path(self.config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save debugging config: {e}")
    
    def start_debug_session(self, workflow_id: str, 
                          debug_level: DebugLevel = DebugLevel.BASIC) -> str:
        """
        Start a new debug session.
        
        Args:
            workflow_id: Workflow ID to debug
            debug_level: Debug level
            
        Returns:
            Session ID
        """
        try:
            session_id = str(uuid.uuid4())
            
            session = DebugSession(
                session_id=session_id,
                workflow_id=workflow_id,
                start_time=time.time(),
                debug_level=debug_level
            )
            
            self.active_sessions[session_id] = session
            
            logger.info(f"Started debug session {session_id} for workflow {workflow_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to start debug session: {e}")
            raise
    
    def end_debug_session(self, session_id: str):
        """
        End a debug session.
        
        Args:
            session_id: Session ID
        """
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Debug session {session_id} not found")
            
            session = self.active_sessions[session_id]
            session.end_time = time.time()
            session.status = "completed"
            
            logger.info(f"Ended debug session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to end debug session: {e}")
    
    def add_breakpoint(self, session_id: str, breakpoint: Breakpoint) -> bool:
        """
        Add a breakpoint to a debug session.
        
        Args:
            session_id: Session ID
            breakpoint: Breakpoint configuration
            
        Returns:
            True if successful
        """
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Debug session {session_id} not found")
            
            session = self.active_sessions[session_id]
            session.breakpoints.append(breakpoint)
            
            logger.info(f"Added breakpoint {breakpoint.breakpoint_id} to session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add breakpoint: {e}")
            return False
    
    def remove_breakpoint(self, session_id: str, breakpoint_id: str) -> bool:
        """
        Remove a breakpoint from a debug session.
        
        Args:
            session_id: Session ID
            breakpoint_id: Breakpoint ID
            
        Returns:
            True if successful
        """
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Debug session {session_id} not found")
            
            session = self.active_sessions[session_id]
            session.breakpoints = [bp for bp in session.breakpoints if bp.breakpoint_id != breakpoint_id]
            
            logger.info(f"Removed breakpoint {breakpoint_id} from session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove breakpoint: {e}")
            return False
    
    def execute_with_debugging(self, session_id: str, 
                             workflow_config: Dict[str, Any],
                             initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute workflow with debugging enabled.
        
        Args:
            session_id: Debug session ID
            workflow_config: Workflow configuration
            initial_state: Initial state
            
        Returns:
            Execution results with debug information
        """
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Debug session {session_id} not found")
            
            session = self.active_sessions[session_id]
            session.status = "running"
            
            # Start debugging execution
            debug_result = self._debug_execution(session, workflow_config, initial_state)
            
            # End session
            self.end_debug_session(session_id)
            
            return debug_result
            
        except Exception as e:
            logger.error(f"Failed to execute with debugging: {e}")
            raise
    
    def _debug_execution(self, session: DebugSession, 
                        workflow_config: Dict[str, Any],
                        initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow with debugging."""
        try:
            current_state = initial_state.copy()
            execution_steps = []
            breakpoints_triggered = []
            
            # Get workflow nodes
            nodes = workflow_config.get('nodes', [])
            edges = workflow_config.get('edges', [])
            
            # Find start node
            start_node = next((node for node in nodes if node.get('node_type') == 'start'), None)
            if not start_node:
                raise ValueError("No start node found in workflow")
            
            # Execute workflow step by step
            current_node = start_node
            step_count = 0
            
            while current_node and current_node.get('node_type') != 'end':
                step_id = str(uuid.uuid4())
                step_start_time = time.time()
                
                # Create debug step
                debug_step = DebugStep(
                    step_id=step_id,
                    node_id=current_node.get('node_id', ''),
                    node_name=current_node.get('name', ''),
                    node_type=current_node.get('node_type', ''),
                    timestamp=step_start_time,
                    state_before=current_state.copy(),
                    state_after={},
                    execution_time_ms=0,
                    status="running"
                )
                
                try:
                    # Check for breakpoints
                    triggered_breakpoints = self._check_breakpoints(session, debug_step)
                    if triggered_breakpoints:
                        breakpoints_triggered.extend(triggered_breakpoints)
                        debug_step.breakpoints_triggered = triggered_breakpoints
                        
                        # Handle breakpoint actions
                        for bp_id in triggered_breakpoints:
                            self._handle_breakpoint(session, bp_id, debug_step)
                    
                    # Simulate node execution
                    node_result = self._simulate_node_execution(current_node, current_state)
                    current_state.update(node_result)
                    debug_step.state_after = current_state.copy()
                    debug_step.status = "completed"
                    
                except Exception as e:
                    debug_step.status = "error"
                    debug_step.error_message = str(e)
                    session.error_count += 1
                    
                    # Handle error
                    self._handle_error(debug_step)
                
                # Calculate execution time
                step_end_time = time.time()
                debug_step.execution_time_ms = (step_end_time - step_start_time) * 1000
                
                # Add to session steps
                session.steps.append(debug_step)
                execution_steps.append(debug_step)
                
                # Trigger debug callbacks
                for callback in self.debug_callbacks:
                    try:
                        callback(debug_step)
                    except Exception as e:
                        logger.error(f"Error in debug callback: {e}")
                
                # Find next node
                next_edge = next((edge for edge in edges 
                               if edge.get('source_node') == current_node.get('node_id')), None)
                if next_edge:
                    current_node = next((node for node in nodes 
                                      if node.get('node_id') == next_edge.get('target_node')), None)
                else:
                    break
                
                step_count += 1
            
            # Create debug result
            debug_result = {
                'session_id': session.session_id,
                'workflow_id': session.workflow_id,
                'execution_steps': [asdict(step) for step in execution_steps],
                'total_steps': len(execution_steps),
                'error_count': session.error_count,
                'breakpoints_triggered': breakpoints_triggered,
                'execution_time_ms': sum(step.execution_time_ms for step in execution_steps),
                'final_state': current_state,
                'status': session.status
            }
            
            logger.info(f"Debug execution completed for session {session.session_id}")
            return debug_result
            
        except Exception as e:
            logger.error(f"Debug execution failed: {e}")
            raise
    
    def _check_breakpoints(self, session: DebugSession, debug_step: DebugStep) -> List[str]:
        """Check if any breakpoints are triggered."""
        triggered = []
        
        for breakpoint in session.breakpoints:
            if not breakpoint.enabled:
                continue
            
            if breakpoint.breakpoint_type == BreakpointType.NODE_ENTRY:
                if breakpoint.node_id == debug_step.node_id:
                    triggered.append(breakpoint.breakpoint_id)
            
            elif breakpoint.breakpoint_type == BreakpointType.ERROR:
                if debug_step.status == "error":
                    triggered.append(breakpoint.breakpoint_id)
            
            elif breakpoint.breakpoint_type == BreakpointType.CONDITION:
                if breakpoint.condition and self._evaluate_condition(breakpoint.condition, debug_step):
                    triggered.append(breakpoint.breakpoint_id)
        
        return triggered
    
    def _evaluate_condition(self, condition: str, debug_step: DebugStep) -> bool:
        """Evaluate a breakpoint condition."""
        try:
            # Simple condition evaluation (in a real implementation, this would be more sophisticated)
            if "error" in condition.lower() and debug_step.status == "error":
                return True
            if "execution_time" in condition.lower():
                # Extract threshold from condition
                threshold = 1000  # Default threshold
                if ">" in condition:
                    threshold = int(condition.split(">")[1].strip())
                return debug_step.execution_time_ms > threshold
            return False
        except Exception:
            return False
    
    def _handle_breakpoint(self, session: DebugSession, breakpoint_id: str, debug_step: DebugStep):
        """Handle a triggered breakpoint."""
        breakpoint = next((bp for bp in session.breakpoints if bp.breakpoint_id == breakpoint_id), None)
        if not breakpoint:
            return
        
        if breakpoint.action == "pause":
            logger.info(f"Breakpoint {breakpoint_id} triggered: {debug_step.node_name}")
        elif breakpoint.action == "log":
            logger.info(f"Breakpoint {breakpoint_id} logged: {debug_step.node_name}")
        elif breakpoint.action == "alert":
            logger.warning(f"Breakpoint {breakpoint_id} alert: {debug_step.node_name}")
    
    def _handle_error(self, debug_step: DebugStep):
        """Handle an error in debug step."""
        for handler in self.error_handlers:
            try:
                handler(debug_step)
            except Exception as e:
                logger.error(f"Error in error handler: {e}")
    
    def _simulate_node_execution(self, node: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate node execution."""
        node_type = node.get('node_type', 'agent')
        
        if node_type == 'agent':
            return {
                'agent_output': f"Agent {node.get('name', 'Unknown')} processed the request",
                'status': 'processed'
            }
        elif node_type == 'tool':
            return {
                'tool_output': f"Tool {node.get('name', 'Unknown')} executed successfully",
                'status': 'completed'
            }
        elif node_type == 'conditional':
            return {
                'condition_result': True,
                'status': 'evaluated'
            }
        else:
            return {
                'output': f"Node {node.get('name', 'Unknown')} executed",
                'status': 'completed'
            }
    
    def get_debug_session(self, session_id: str) -> Optional[DebugSession]:
        """Get debug session by ID."""
        return self.active_sessions.get(session_id)
    
    def list_debug_sessions(self) -> List[DebugSession]:
        """List all debug sessions."""
        return list(self.active_sessions.values())
    
    def get_session_steps(self, session_id: str) -> List[DebugStep]:
        """Get debug steps for a session."""
        session = self.active_sessions.get(session_id)
        if session:
            return session.steps
        return []
    
    def add_debug_callback(self, callback: Callable[[DebugStep], None]):
        """Add a debug callback function."""
        self.debug_callbacks.append(callback)
        logger.info("Added debug callback")
    
    def add_error_handler(self, handler: Callable[[DebugStep], None]):
        """Add an error handler function."""
        self.error_handlers.append(handler)
        logger.info("Added error handler")
    
    def export_debug_session(self, session_id: str, export_path: str) -> bool:
        """
        Export debug session data.
        
        Args:
            session_id: Session ID
            export_path: Path to export file
            
        Returns:
            True if successful
        """
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return False
            
            # Export session data
            export_data = {
                'session': asdict(session),
                'steps': [asdict(step) for step in session.steps],
                'export_timestamp': time.time()
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported debug session {session_id} to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export debug session: {e}")
            return False
    
    def analyze_debug_session(self, session_id: str) -> Dict[str, Any]:
        """
        Analyze a debug session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Analysis results
        """
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return {}
            
            steps = session.steps
            if not steps:
                return {'error': 'No steps found in session'}
            
            # Calculate metrics
            total_execution_time = sum(step.execution_time_ms for step in steps)
            error_steps = [step for step in steps if step.status == "error"]
            successful_steps = [step for step in steps if step.status == "completed"]
            
            # Find slowest steps
            slowest_steps = sorted(steps, key=lambda x: x.execution_time_ms, reverse=True)[:3]
            
            # Find most error-prone nodes
            error_by_node = {}
            for step in error_steps:
                node_id = step.node_id
                error_by_node[node_id] = error_by_node.get(node_id, 0) + 1
            
            analysis = {
                'session_id': session_id,
                'workflow_id': session.workflow_id,
                'total_steps': len(steps),
                'error_count': len(error_steps),
                'success_count': len(successful_steps),
                'success_rate': len(successful_steps) / len(steps) if steps else 0,
                'total_execution_time_ms': total_execution_time,
                'average_step_time_ms': total_execution_time / len(steps) if steps else 0,
                'slowest_steps': [asdict(step) for step in slowest_steps],
                'error_by_node': error_by_node,
                'breakpoints_triggered': len([step for step in steps if step.breakpoints_triggered]),
                'debug_level': session.debug_level.value
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze debug session: {e}")
            return {'error': str(e)}


# Factory functions for common debugging scenarios
def create_basic_debugger() -> WorkflowDebugger:
    """Create a basic workflow debugger."""
    return WorkflowDebugger()


def create_verbose_debugger() -> WorkflowDebugger:
    """Create a verbose workflow debugger."""
    debugger = WorkflowDebugger()
    
    # Add verbose debug callback
    def verbose_callback(step: DebugStep):
        logger.info(f"Step {step.step_id}: {step.node_name} ({step.status}) - {step.execution_time_ms:.2f}ms")
    
    debugger.add_debug_callback(verbose_callback)
    return debugger


def create_error_focused_debugger() -> WorkflowDebugger:
    """Create an error-focused debugger."""
    debugger = WorkflowDebugger()
    
    # Add error handler
    def error_handler(step: DebugStep):
        logger.error(f"Error in {step.node_name}: {step.error_message}")
    
    debugger.add_error_handler(error_handler)
    return debugger


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create debugger
    debugger = create_verbose_debugger()
    
    # Start debug session
    session_id = debugger.start_debug_session("test_workflow", DebugLevel.DETAILED)
    
    # Add breakpoints
    breakpoint1 = Breakpoint(
        breakpoint_id="bp1",
        breakpoint_type=BreakpointType.NODE_ENTRY,
        node_id="analyzer",
        action="log"
    )
    debugger.add_breakpoint(session_id, breakpoint1)
    
    breakpoint2 = Breakpoint(
        breakpoint_id="bp2",
        breakpoint_type=BreakpointType.ERROR,
        action="alert"
    )
    debugger.add_breakpoint(session_id, breakpoint2)
    
    # Example workflow
    workflow_config = {
        "nodes": [
            {"node_id": "start", "node_type": "start", "name": "Start"},
            {"node_id": "analyzer", "node_type": "agent", "name": "Device Analyzer"},
            {"node_id": "solution", "node_type": "agent", "name": "Solution Provider"},
            {"node_id": "end", "node_type": "end", "name": "End"}
        ],
        "edges": [
            {"source_node": "start", "target_node": "analyzer"},
            {"source_node": "analyzer", "target_node": "solution"},
            {"source_node": "solution", "target_node": "end"}
        ]
    }
    
    # Execute with debugging
    initial_state = {
        "device_model": "ThinkPad X1 Carbon",
        "issue_description": "Laptop not booting"
    }
    
    try:
        result = debugger.execute_with_debugging(session_id, workflow_config, initial_state)
        print(f"Debug execution result: {result}")
        
        # Analyze session
        analysis = debugger.analyze_debug_session(session_id)
        print(f"Debug analysis: {analysis}")
        
    except Exception as e:
        print(f"Debug execution failed: {e}")
