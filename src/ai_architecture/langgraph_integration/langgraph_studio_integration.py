"""
LangGraph Studio Integration

This module provides integration with LangGraph Studio for interactive
workflow development, visualization, and debugging.
"""

import json
import logging
import subprocess
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from pathlib import Path
import requests

logger = logging.getLogger(__name__)


class StudioStatus(Enum):
    """LangGraph Studio status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"


@dataclass
class StudioConfig:
    """LangGraph Studio configuration."""
    host: str = "localhost"
    port: int = 8083
    workspace_path: str = "workspace"
    auto_start: bool = True
    debug_mode: bool = False
    log_level: str = "info"
    max_workflows: int = 10
    timeout_seconds: int = 30


@dataclass
class StudioWorkflow:
    """Studio workflow configuration."""
    workflow_id: str
    name: str
    description: str
    graph_definition: Dict[str, Any]
    state_schema: Dict[str, Any]
    created_at: str
    updated_at: str
    status: str = "draft"
    studio_url: Optional[str] = None


class LangGraphStudioIntegration:
    """
    LangGraph Studio Integration for interactive workflow development.
    
    This class provides comprehensive integration with LangGraph Studio
    including workflow management, visualization, and debugging.
    """
    
    def __init__(self, config: Optional[StudioConfig] = None):
        """
        Initialize the LangGraph Studio Integration.
        
        Args:
            config: Studio configuration
        """
        self.config = config or StudioConfig()
        self.studio_status = StudioStatus.STOPPED
        self.studio_process: Optional[subprocess.Popen] = None
        self.workflows: Dict[str, StudioWorkflow] = {}
        self.studio_url = f"http://{self.config.host}:{self.config.port}"
        
        # Check if studio is already running
        if self._check_studio_running():
            self.studio_status = StudioStatus.RUNNING
            logger.info("LangGraph Studio is already running")
        elif self.config.auto_start:
            self.start_studio()
        
        logger.info("LangGraph Studio Integration initialized")
    
    def _check_studio_running(self) -> bool:
        """Check if LangGraph Studio is running."""
        try:
            response = requests.get(f"{self.studio_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def start_studio(self) -> bool:
        """
        Start LangGraph Studio.
        
        Returns:
            True if started successfully
        """
        try:
            if self.studio_status == StudioStatus.RUNNING:
                logger.info("LangGraph Studio is already running")
                return True
            
            self.studio_status = StudioStatus.STARTING
            logger.info("Starting LangGraph Studio...")
            
            # Start studio process
            cmd = [
                "langgraph-studio",
                "--host", self.config.host,
                "--port", str(self.config.port),
                "--workspace", self.config.workspace_path
            ]
            
            if self.config.debug_mode:
                cmd.append("--debug")
            
            self.studio_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for studio to start
            max_wait = self.config.timeout_seconds
            wait_time = 0
            
            while wait_time < max_wait:
                if self._check_studio_running():
                    self.studio_status = StudioStatus.RUNNING
                    logger.info(f"LangGraph Studio started successfully at {self.studio_url}")
                    return True
                
                time.sleep(1)
                wait_time += 1
            
            # If we get here, studio didn't start
            self.studio_status = StudioStatus.ERROR
            logger.error("Failed to start LangGraph Studio")
            return False
            
        except Exception as e:
            self.studio_status = StudioStatus.ERROR
            logger.error(f"Failed to start LangGraph Studio: {e}")
            return False
    
    def stop_studio(self) -> bool:
        """
        Stop LangGraph Studio.
        
        Returns:
            True if stopped successfully
        """
        try:
            if self.studio_status == StudioStatus.STOPPED:
                logger.info("LangGraph Studio is already stopped")
                return True
            
            if self.studio_process:
                self.studio_process.terminate()
                self.studio_process.wait(timeout=10)
                self.studio_process = None
            
            self.studio_status = StudioStatus.STOPPED
            logger.info("LangGraph Studio stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop LangGraph Studio: {e}")
            return False
    
    def get_studio_status(self) -> Dict[str, Any]:
        """
        Get LangGraph Studio status.
        
        Returns:
            Status information
        """
        try:
            if self.studio_status == StudioStatus.RUNNING:
                # Get detailed status from studio
                response = requests.get(f"{self.studio_url}/api/status", timeout=5)
                if response.status_code == 200:
                    studio_info = response.json()
                    return {
                        'status': self.studio_status.value,
                        'url': self.studio_url,
                        'studio_info': studio_info,
                        'workflows_count': len(self.workflows)
                    }
            
            return {
                'status': self.studio_status.value,
                'url': self.studio_url,
                'studio_info': None,
                'workflows_count': len(self.workflows)
            }
            
        except Exception as e:
            logger.error(f"Failed to get studio status: {e}")
            return {
                'status': 'error',
                'url': self.studio_url,
                'error': str(e)
            }
    
    def create_workflow(self, name: str, description: str, 
                       graph_definition: Dict[str, Any],
                       state_schema: Dict[str, Any]) -> str:
        """
        Create a workflow in LangGraph Studio.
        
        Args:
            name: Workflow name
            description: Workflow description
            graph_definition: Graph definition
            state_schema: State schema
            
        Returns:
            Workflow ID
        """
        try:
            workflow_id = str(uuid.uuid4())
            
            # Create studio workflow
            studio_workflow = StudioWorkflow(
                workflow_id=workflow_id,
                name=name,
                description=description,
                graph_definition=graph_definition,
                state_schema=state_schema,
                created_at=str(uuid.uuid4().time_low),
                updated_at=str(uuid.uuid4().time_low),
                studio_url=f"{self.studio_url}/workflows/{workflow_id}"
            )
            
            # Store workflow
            self.workflows[workflow_id] = studio_workflow
            
            # Send to studio if running
            if self.studio_status == StudioStatus.RUNNING:
                self._send_workflow_to_studio(studio_workflow)
            
            logger.info(f"Created workflow: {name} (ID: {workflow_id})")
            return workflow_id
            
        except Exception as e:
            logger.error(f"Failed to create workflow: {e}")
            raise
    
    def _send_workflow_to_studio(self, workflow: StudioWorkflow) -> bool:
        """Send workflow to LangGraph Studio."""
        try:
            if self.studio_status != StudioStatus.RUNNING:
                return False
            
            # Prepare workflow data for studio
            workflow_data = {
                'id': workflow.workflow_id,
                'name': workflow.name,
                'description': workflow.description,
                'graph': workflow.graph_definition,
                'state_schema': workflow.state_schema,
                'status': workflow.status
            }
            
            # Send to studio API
            response = requests.post(
                f"{self.studio_url}/api/workflows",
                json=workflow_data,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Workflow {workflow.workflow_id} sent to studio")
                return True
            else:
                logger.error(f"Failed to send workflow to studio: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send workflow to studio: {e}")
            return False
    
    def get_workflow(self, workflow_id: str) -> Optional[StudioWorkflow]:
        """Get workflow by ID."""
        return self.workflows.get(workflow_id)
    
    def list_workflows(self) -> List[StudioWorkflow]:
        """List all workflows."""
        return list(self.workflows.values())
    
    def update_workflow(self, workflow_id: str, 
                       updates: Dict[str, Any]) -> bool:
        """
        Update a workflow.
        
        Args:
            workflow_id: Workflow ID
            updates: Updates to apply
            
        Returns:
            True if successful
        """
        try:
            if workflow_id not in self.workflows:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            workflow = self.workflows[workflow_id]
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(workflow, key):
                    setattr(workflow, key, value)
            
            workflow.updated_at = str(uuid.uuid4().time_low)
            
            # Send updates to studio if running
            if self.studio_status == StudioStatus.RUNNING:
                self._send_workflow_to_studio(workflow)
            
            logger.info(f"Updated workflow {workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update workflow: {e}")
            return False
    
    def delete_workflow(self, workflow_id: str) -> bool:
        """
        Delete a workflow.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            True if successful
        """
        try:
            if workflow_id not in self.workflows:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            # Remove from studio if running
            if self.studio_status == StudioStatus.RUNNING:
                try:
                    response = requests.delete(
                        f"{self.studio_url}/api/workflows/{workflow_id}",
                        timeout=10
                    )
                    if response.status_code != 200:
                        logger.warning(f"Failed to delete workflow from studio: {response.status_code}")
                except Exception as e:
                    logger.warning(f"Failed to delete workflow from studio: {e}")
            
            # Remove from local storage
            del self.workflows[workflow_id]
            
            logger.info(f"Deleted workflow {workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete workflow: {e}")
            return False
    
    def execute_workflow(self, workflow_id: str, 
                        initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a workflow in LangGraph Studio.
        
        Args:
            workflow_id: Workflow ID
            initial_state: Initial state
            
        Returns:
            Execution results
        """
        try:
            if workflow_id not in self.workflows:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            if self.studio_status != StudioStatus.RUNNING:
                raise RuntimeError("LangGraph Studio is not running")
            
            # Send execution request to studio
            execution_data = {
                'workflow_id': workflow_id,
                'initial_state': initial_state,
                'execution_id': str(uuid.uuid4())
            }
            
            response = requests.post(
                f"{self.studio_url}/api/execute",
                json=execution_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Workflow {workflow_id} executed successfully")
                return result
            else:
                logger.error(f"Failed to execute workflow: {response.status_code}")
                return {'error': f'Execution failed: {response.status_code}'}
                
        except Exception as e:
            logger.error(f"Failed to execute workflow: {e}")
            return {'error': str(e)}
    
    def get_workflow_visualization(self, workflow_id: str) -> Optional[str]:
        """
        Get workflow visualization URL.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            Visualization URL
        """
        try:
            if workflow_id not in self.workflows:
                return None
            
            if self.studio_status != StudioStatus.RUNNING:
                return None
            
            workflow = self.workflows[workflow_id]
            return workflow.studio_url
            
        except Exception as e:
            logger.error(f"Failed to get workflow visualization: {e}")
            return None
    
    def export_workflow(self, workflow_id: str, export_path: str) -> bool:
        """
        Export workflow to file.
        
        Args:
            workflow_id: Workflow ID
            export_path: Path to export file
            
        Returns:
            True if successful
        """
        try:
            if workflow_id not in self.workflows:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            workflow = self.workflows[workflow_id]
            
            # Export workflow data
            export_data = {
                'workflow': asdict(workflow),
                'export_timestamp': time.time(),
                'studio_url': self.studio_url
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported workflow {workflow_id} to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export workflow: {e}")
            return False
    
    def import_workflow(self, import_path: str) -> str:
        """
        Import workflow from file.
        
        Args:
            import_path: Path to import file
            
        Returns:
            Workflow ID
        """
        try:
            with open(import_path, 'r') as f:
                import_data = json.load(f)
            
            workflow_data = import_data['workflow']
            
            # Create workflow
            workflow = StudioWorkflow(
                workflow_id=workflow_data['workflow_id'],
                name=workflow_data['name'],
                description=workflow_data['description'],
                graph_definition=workflow_data['graph_definition'],
                state_schema=workflow_data['state_schema'],
                created_at=workflow_data['created_at'],
                updated_at=workflow_data['updated_at'],
                status=workflow_data.get('status', 'draft'),
                studio_url=workflow_data.get('studio_url')
            )
            
            # Store workflow
            self.workflows[workflow.workflow_id] = workflow
            
            # Send to studio if running
            if self.studio_status == StudioStatus.RUNNING:
                self._send_workflow_to_studio(workflow)
            
            logger.info(f"Imported workflow: {workflow.name} (ID: {workflow.workflow_id})")
            return workflow.workflow_id
            
        except Exception as e:
            logger.error(f"Failed to import workflow: {e}")
            raise
    
    def get_studio_health(self) -> Dict[str, Any]:
        """
        Get LangGraph Studio health status.
        
        Returns:
            Health information
        """
        try:
            if self.studio_status != StudioStatus.RUNNING:
                return {
                    'status': 'stopped',
                    'healthy': False,
                    'message': 'Studio is not running'
                }
            
            # Check studio health
            response = requests.get(f"{self.studio_url}/health", timeout=5)
            
            if response.status_code == 200:
                health_data = response.json()
                return {
                    'status': 'running',
                    'healthy': True,
                    'message': 'Studio is running',
                    'details': health_data
                }
            else:
                return {
                    'status': 'error',
                    'healthy': False,
                    'message': f'Studio returned status {response.status_code}'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'healthy': False,
                'message': f'Health check failed: {str(e)}'
            }
    
    def get_studio_metrics(self) -> Dict[str, Any]:
        """
        Get LangGraph Studio metrics.
        
        Returns:
            Metrics information
        """
        try:
            if self.studio_status != StudioStatus.RUNNING:
                return {'error': 'Studio is not running'}
            
            # Get metrics from studio
            response = requests.get(f"{self.studio_url}/api/metrics", timeout=5)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f'Failed to get metrics: {response.status_code}'}
                
        except Exception as e:
            return {'error': str(e)}


# Factory functions for common studio integration scenarios
def create_studio_integration() -> LangGraphStudioIntegration:
    """Create a standard studio integration."""
    config = StudioConfig()
    return LangGraphStudioIntegration(config)


def create_development_studio() -> LangGraphStudioIntegration:
    """Create a development-focused studio integration."""
    config = StudioConfig(
        debug_mode=True,
        log_level="debug",
        max_workflows=20
    )
    return LangGraphStudioIntegration(config)


def create_production_studio() -> LangGraphStudioIntegration:
    """Create a production-focused studio integration."""
    config = StudioConfig(
        auto_start=False,
        debug_mode=False,
        log_level="info",
        max_workflows=5
    )
    return LangGraphStudioIntegration(config)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create studio integration
    studio = create_studio_integration()
    
    # Check studio status
    status = studio.get_studio_status()
    print(f"Studio status: {status}")
    
    # Create a workflow
    graph_definition = {
        "nodes": [
            {"id": "start", "type": "start"},
            {"id": "analyzer", "type": "agent", "name": "Device Analyzer"},
            {"id": "solution", "type": "agent", "name": "Solution Provider"},
            {"id": "end", "type": "end"}
        ],
        "edges": [
            {"source": "start", "target": "analyzer"},
            {"source": "analyzer", "target": "solution"},
            {"source": "solution", "target": "end"}
        ]
    }
    
    state_schema = {
        "device_model": "string",
        "issue_description": "string",
        "diagnosis": "string",
        "solution": "string"
    }
    
    try:
        workflow_id = studio.create_workflow(
            name="Lenovo Device Support",
            description="Device support workflow",
            graph_definition=graph_definition,
            state_schema=state_schema
        )
        print(f"Created workflow: {workflow_id}")
        
        # Get workflow visualization URL
        viz_url = studio.get_workflow_visualization(workflow_id)
        print(f"Visualization URL: {viz_url}")
        
        # Execute workflow
        initial_state = {
            "device_model": "ThinkPad X1 Carbon",
            "issue_description": "Laptop not booting"
        }
        
        result = studio.execute_workflow(workflow_id, initial_state)
        print(f"Execution result: {result}")
        
    except Exception as e:
        print(f"Studio integration failed: {e}")
    
    # Get studio health
    health = studio.get_studio_health()
    print(f"Studio health: {health}")
