"""
LangGraph Studio Integration Module for Enterprise LLMOps Platform

This module provides integration with the actual LangGraph Studio from the LangChain repository,
enabling visualization, interaction, and debugging of agentic systems that implement the 
LangGraph Server API protocol.

Key Features:
- Integration with LangGraph Studio for agent visualization and debugging
- LangSmith integration for tracing, evaluation, and prompt engineering
- Graph mode and Chat mode support
- Real-time agent state monitoring via time travel
- Assistant and thread management
- Experiment management over datasets
- Long-term memory management

LangGraph Studio provides a specialized agent IDE that enables:
- Visualize your graph architecture
- Run and interact with your agent
- Manage assistants and threads
- Iterate on prompts
- Run experiments over a dataset
- Manage long term memory
- Debug agent state via time travel

Reference: https://github.com/langchain-ai/langgraph-studio
"""

import asyncio
import json
import uuid
import subprocess
import os
import signal
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Callable, TypedDict, Annotated
from dataclasses import dataclass, field
from enum import Enum
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StudioMode(Enum):
    """LangGraph Studio modes"""
    GRAPH = "graph"
    CHAT = "chat"


class StudioStatus(Enum):
    """Studio service status"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"


@dataclass
class StudioConfig:
    """LangGraph Studio configuration"""
    host: str = "localhost"
    port: int = 8080
    studio_port: int = 8083  # Changed from 8081 to avoid conflict with ChromaDB
    mode: StudioMode = StudioMode.GRAPH
    enable_langsmith: bool = True
    langsmith_api_key: Optional[str] = None
    langsmith_project: Optional[str] = None
    docker_compose_file: Optional[str] = None
    working_directory: str = "."
    log_level: str = "INFO"


@dataclass
class StudioSession:
    """Studio session information"""
    session_id: str
    assistant_id: Optional[str] = None
    thread_id: Optional[str] = None
    mode: StudioMode = StudioMode.GRAPH
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class LangGraphStudioManager:
    """Main LangGraph Studio integration manager"""
    
    def __init__(self, config: StudioConfig = None):
        self.config = config or StudioConfig()
        self.status = StudioStatus.STOPPED
        self.process: Optional[subprocess.Popen] = None
        self.sessions: Dict[str, StudioSession] = {}
        self.logger = logging.getLogger("langgraph_studio_manager")
        
        # API endpoints
        self.base_url = f"http://{self.config.host}:{self.config.studio_port}"
        self.api_url = f"http://{self.config.host}:{self.config.port}/api/langgraph/studios"
        
    async def start_studio(self) -> bool:
        """Start LangGraph Studio service"""
        try:
            if self.status == StudioStatus.RUNNING:
                self.logger.info("LangGraph Studio is already running")
                return True
                
            self.status = StudioStatus.STARTING
            self.logger.info("Starting LangGraph Studio...")
            
            # Check if LangGraph CLI is available
            try:
                result = subprocess.run(
                    ["langgraph", "--version"], 
                    capture_output=True, 
                    text=True, 
                    timeout=10
                )
                if result.returncode != 0:
                    raise RuntimeError("LangGraph CLI not found")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self.logger.error("LangGraph CLI not found. Please install with: pip install langgraph-cli")
                self.status = StudioStatus.ERROR
                return False
            
            # Start LangGraph Studio using the CLI
            # Note: This is a simplified approach. In production, you'd want to use
            # the proper LangGraph Studio setup with Docker Compose
            cmd = [
                "langgraph", "dev",
                "--host", self.config.host,
                "--port", str(self.config.studio_port),
                "--log-level", self.config.log_level.lower()
            ]
            
            if self.config.enable_langsmith and self.config.langsmith_api_key:
                cmd.extend(["--langsmith-api-key", self.config.langsmith_api_key])
                if self.config.langsmith_project:
                    cmd.extend(["--langsmith-project", self.config.langsmith_project])
            
            self.process = subprocess.Popen(
                cmd,
                cwd=self.config.working_directory,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for service to start
            await self._wait_for_service()
            
            if self.status == StudioStatus.RUNNING:
                self.logger.info(f"LangGraph Studio started successfully on {self.base_url}")
                return True
            else:
                self.logger.error("Failed to start LangGraph Studio")
                return False
                
        except Exception as e:
            self.logger.error(f"Error starting LangGraph Studio: {e}")
            self.status = StudioStatus.ERROR
            return False
    
    async def stop_studio(self) -> bool:
        """Stop LangGraph Studio service"""
        try:
            if self.status == StudioStatus.STOPPED:
                self.logger.info("LangGraph Studio is already stopped")
                return True
                
            self.logger.info("Stopping LangGraph Studio...")
            
            if self.process:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                self.process = None
            
            self.status = StudioStatus.STOPPED
            self.logger.info("LangGraph Studio stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping LangGraph Studio: {e}")
            return False
    
    async def _wait_for_service(self, timeout: int = 30) -> bool:
        """Wait for the service to become available"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}/health", timeout=5) as response:
                        if response.status == 200:
                            self.status = StudioStatus.RUNNING
                            return True
            except Exception:
                pass
            
            await asyncio.sleep(1)
        
        self.status = StudioStatus.ERROR
        return False
    
    async def get_studio_status(self) -> Dict[str, Any]:
        """Get current studio status"""
        return {
            "status": self.status.value,
            "base_url": self.base_url,
            "api_url": self.api_url,
            "sessions_count": len(self.sessions),
            "uptime": self._get_uptime() if self.status == StudioStatus.RUNNING else None
        }
    
    def _get_uptime(self) -> Optional[float]:
        """Get service uptime in seconds"""
        if self.process and self.process.poll() is None:
            return time.time() - self.process.start_time
        return None
    
    async def create_session(self, mode: StudioMode = StudioMode.GRAPH, metadata: Dict[str, Any] = None) -> str:
        """Create a new studio session"""
        session_id = str(uuid.uuid4())
        session = StudioSession(
            session_id=session_id,
            mode=mode,
            metadata=metadata or {}
        )
        
        self.sessions[session_id] = session
        self.logger.info(f"Created studio session: {session_id}")
        
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[StudioSession]:
        """Get session information"""
        return self.sessions.get(session_id)
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a studio session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.logger.info(f"Deleted studio session: {session_id}")
            return True
        return False
    
    async def get_assistants(self) -> List[Dict[str, Any]]:
        """Get available assistants from LangGraph Studio"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/assistants", timeout=10) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        self.logger.error(f"Failed to get assistants: {response.status}")
                        return []
        except Exception as e:
            self.logger.error(f"Error getting assistants: {e}")
            return []
    
    async def get_threads(self) -> List[Dict[str, Any]]:
        """Get available threads from LangGraph Studio"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/threads", timeout=10) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        self.logger.error(f"Failed to get threads: {response.status}")
                        return []
        except Exception as e:
            self.logger.error(f"Error getting threads: {e}")
            return []
    
    async def get_experiments(self) -> List[Dict[str, Any]]:
        """Get available experiments from LangGraph Studio"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/experiments", timeout=10) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        self.logger.error(f"Failed to get experiments: {response.status}")
                        return []
        except Exception as e:
            self.logger.error(f"Error getting experiments: {e}")
            return []
    
    async def get_datasets(self) -> List[Dict[str, Any]]:
        """Get available datasets from LangGraph Studio"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/datasets", timeout=10) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        self.logger.error(f"Failed to get datasets: {response.status}")
                        return []
        except Exception as e:
            self.logger.error(f"Error getting datasets: {e}")
            return []
    
    async def get_memory(self) -> List[Dict[str, Any]]:
        """Get long-term memory information from LangGraph Studio"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/memory", timeout=10) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        self.logger.error(f"Failed to get memory: {response.status}")
                        return []
        except Exception as e:
            self.logger.error(f"Error getting memory: {e}")
            return []
    
    async def get_studio_info(self) -> Dict[str, Any]:
        """Get comprehensive studio information"""
        return {
            "status": await self.get_studio_status(),
            "assistants": await self.get_assistants(),
            "threads": await self.get_threads(),
            "experiments": await self.get_experiments(),
            "datasets": await self.get_datasets(),
            "memory": await self.get_memory(),
            "sessions": list(self.sessions.keys())
        }


# Global LangGraph Studio manager instance
langgraph_studio_manager: Optional[LangGraphStudioManager] = None


def initialize_langgraph_studio_manager(config: StudioConfig = None) -> LangGraphStudioManager:
    """Initialize the global LangGraph Studio manager"""
    global langgraph_studio_manager
    langgraph_studio_manager = LangGraphStudioManager(config)
    return langgraph_studio_manager


def get_langgraph_studio_manager() -> LangGraphStudioManager:
    """Get the global LangGraph Studio manager"""
    if langgraph_studio_manager is None:
        raise RuntimeError("LangGraph Studio manager not initialized")
    return langgraph_studio_manager


# FastAPI integration functions
try:
    from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel
    
    class StudioConfigRequest(BaseModel):
        """Request model for studio configuration"""
        host: str = "localhost"
        port: int = 8080
        studio_port: int = 8081
        mode: str = "graph"
        enable_langsmith: bool = True
        langsmith_api_key: Optional[str] = None
        langsmith_project: Optional[str] = None
        log_level: str = "INFO"
    
    class StudioSessionRequest(BaseModel):
        """Request model for creating studio sessions"""
        mode: str = "graph"
        metadata: Optional[Dict[str, Any]] = None
    
    class StudioStatusResponse(BaseModel):
        """Response model for studio status"""
        status: str
        base_url: str
        api_url: str
        sessions_count: int
        uptime: Optional[float] = None
    
    class StudioInfoResponse(BaseModel):
        """Response model for comprehensive studio information"""
        status: Dict[str, Any]
        assistants: List[Dict[str, Any]]
        threads: List[Dict[str, Any]]
        experiments: List[Dict[str, Any]]
        datasets: List[Dict[str, Any]]
        memory: List[Dict[str, Any]]
        sessions: List[str]
    
    def create_langgraph_studio_endpoints(app: FastAPI):
        """Create LangGraph Studio API endpoints for FastAPI app"""
        
        @app.get("/api/langgraph/studios/status", response_model=StudioStatusResponse)
        async def get_studio_status():
            """Get LangGraph Studio status"""
            manager = get_langgraph_studio_manager()
            status = await manager.get_studio_status()
            return StudioStatusResponse(**status)
        
        @app.post("/api/langgraph/studios/start")
        async def start_studio():
            """Start LangGraph Studio service"""
            manager = get_langgraph_studio_manager()
            success = await manager.start_studio()
            if success:
                return {"message": "LangGraph Studio started successfully"}
            else:
                raise HTTPException(status_code=500, detail="Failed to start LangGraph Studio")
        
        @app.post("/api/langgraph/studios/stop")
        async def stop_studio():
            """Stop LangGraph Studio service"""
            manager = get_langgraph_studio_manager()
            success = await manager.stop_studio()
            if success:
                return {"message": "LangGraph Studio stopped successfully"}
            else:
                raise HTTPException(status_code=500, detail="Failed to stop LangGraph Studio")
        
        @app.get("/api/langgraph/studios/info", response_model=StudioInfoResponse)
        async def get_studio_info():
            """Get comprehensive LangGraph Studio information"""
            manager = get_langgraph_studio_manager()
            info = await manager.get_studio_info()
            return StudioInfoResponse(**info)
        
        @app.post("/api/langgraph/studios/sessions")
        async def create_studio_session(request: StudioSessionRequest):
            """Create a new studio session"""
            manager = get_langgraph_studio_manager()
            mode = StudioMode.GRAPH if request.mode == "graph" else StudioMode.CHAT
            session_id = await manager.create_session(mode, request.metadata)
            return {"session_id": session_id, "mode": request.mode}
        
        @app.get("/api/langgraph/studios/sessions/{session_id}")
        async def get_studio_session(session_id: str):
            """Get studio session information"""
            manager = get_langgraph_studio_manager()
            session = await manager.get_session(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            return {
                "session_id": session.session_id,
                "assistant_id": session.assistant_id,
                "thread_id": session.thread_id,
                "mode": session.mode.value,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "metadata": session.metadata
            }
        
        @app.delete("/api/langgraph/studios/sessions/{session_id}")
        async def delete_studio_session(session_id: str):
            """Delete a studio session"""
            manager = get_langgraph_studio_manager()
            success = await manager.delete_session(session_id)
            if success:
                return {"message": "Session deleted successfully"}
            else:
                raise HTTPException(status_code=404, detail="Session not found")
        
        @app.get("/api/langgraph/studios/assistants")
        async def get_assistants():
            """Get available assistants"""
            manager = get_langgraph_studio_manager()
            assistants = await manager.get_assistants()
            return {"assistants": assistants}
        
        @app.get("/api/langgraph/studios/threads")
        async def get_threads():
            """Get available threads"""
            manager = get_langgraph_studio_manager()
            threads = await manager.get_threads()
            return {"threads": threads}
        
        @app.get("/api/langgraph/studios/experiments")
        async def get_experiments():
            """Get available experiments"""
            manager = get_langgraph_studio_manager()
            experiments = await manager.get_experiments()
            return {"experiments": experiments}
        
        @app.get("/api/langgraph/studios/datasets")
        async def get_datasets():
            """Get available datasets"""
            manager = get_langgraph_studio_manager()
            datasets = await manager.get_datasets()
            return {"datasets": datasets}
        
        @app.get("/api/langgraph/studios/memory")
        async def get_memory():
            """Get long-term memory information"""
            manager = get_langgraph_studio_manager()
            memory = await manager.get_memory()
            return {"memory": memory}
        
        @app.get("/api/langgraph/studios/dashboard", response_class=HTMLResponse)
        async def get_langgraph_studio_dashboard():
            """Get LangGraph Studio dashboard UI"""
            return HTMLResponse("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>LangGraph Studio Dashboard</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
                    .container { max-width: 1400px; margin: 0 auto; }
                    .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
                    .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                    .status-indicator { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 8px; }
                    .status-running { background: #27ae60; }
                    .status-stopped { background: #e74c3c; }
                    .status-starting { background: #f39c12; }
                    .status-error { background: #e74c3c; }
                    .btn { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin: 5px; }
                    .btn:hover { background: #2980b9; }
                    .btn-success { background: #27ae60; }
                    .btn-success:hover { background: #229954; }
                    .btn-danger { background: #e74c3c; }
                    .btn-danger:hover { background: #c0392b; }
                    .iframe-container { width: 100%; height: 600px; border: 1px solid #ddd; border-radius: 4px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🎯 LangGraph Studio Dashboard</h1>
                        <p>Visualization, Interaction, and Debugging of Agentic Systems</p>
                    </div>
                    
                    <div class="grid">
                        <div class="card">
                            <h3>📊 Studio Status</h3>
                            <div id="studio-status">
                                <p><span class="status-indicator" id="status-indicator"></span><span id="status-text">Loading...</span></p>
                                <p><strong>Base URL:</strong> <span id="base-url">-</span></p>
                                <p><strong>API URL:</strong> <span id="api-url">-</span></p>
                                <p><strong>Sessions:</strong> <span id="sessions-count">-</span></p>
                                <p><strong>Uptime:</strong> <span id="uptime">-</span></p>
                            </div>
                            <button class="btn btn-success" onclick="startStudio()">Start Studio</button>
                            <button class="btn btn-danger" onclick="stopStudio()">Stop Studio</button>
                            <button class="btn" onclick="refreshStatus()">Refresh Status</button>
                        </div>
                        
                        <div class="card">
                            <h3>🎮 Studio Interface</h3>
                            <div id="studio-interface">
                                <p>LangGraph Studio provides two modes:</p>
                                <ul>
                                    <li><strong>Graph Mode:</strong> Full feature-set with detailed execution information</li>
                                    <li><strong>Chat Mode:</strong> Simplified UI for chat-specific agents</li>
                                </ul>
                                <button class="btn" onclick="openStudio()">Open LangGraph Studio</button>
                                <button class="btn" onclick="createSession('graph')">Create Graph Session</button>
                                <button class="btn" onclick="createSession('chat')">Create Chat Session</button>
                            </div>
                        </div>
                    </div>
                    
                    <div class="card" style="margin-top: 20px;">
                        <h3>🔗 Direct Studio Access</h3>
                        <div class="iframe-container">
                            <iframe id="studio-iframe" src="about:blank" title="LangGraph Studio"></iframe>
                        </div>
                    </div>
                </div>
                
                <script>
                    async function refreshStatus() {
                        try {
                            const response = await fetch('/api/langgraph/studios/status');
                            const status = await response.json();
                            
                            document.getElementById('status-text').textContent = status.status;
                            document.getElementById('base-url').textContent = status.base_url;
                            document.getElementById('api-url').textContent = status.api_url;
                            document.getElementById('sessions-count').textContent = status.sessions_count;
                            document.getElementById('uptime').textContent = status.uptime ? `${status.uptime.toFixed(1)}s` : '-';
                            
                            const indicator = document.getElementById('status-indicator');
                            indicator.className = 'status-indicator status-' + status.status;
                            
                        } catch (error) {
                            console.error('Error refreshing status:', error);
                        }
                    }
                    
                    async function startStudio() {
                        try {
                            const response = await fetch('/api/langgraph/studios/start', { method: 'POST' });
                            const result = await response.json();
                            alert(result.message);
                            refreshStatus();
                        } catch (error) {
                            console.error('Error starting studio:', error);
                            alert('Error starting studio');
                        }
                    }
                    
                    async function stopStudio() {
                        try {
                            const response = await fetch('/api/langgraph/studios/stop', { method: 'POST' });
                            const result = await response.json();
                            alert(result.message);
                            refreshStatus();
                        } catch (error) {
                            console.error('Error stopping studio:', error);
                            alert('Error stopping studio');
                        }
                    }
                    
                    async function createSession(mode) {
                        try {
                            const response = await fetch('/api/langgraph/studios/sessions', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ mode: mode })
                            });
                            const result = await response.json();
                            alert(`Session created: ${result.session_id} (${result.mode} mode)`);
                            refreshStatus();
                        } catch (error) {
                            console.error('Error creating session:', error);
                            alert('Error creating session');
                        }
                    }
                    
                    function openStudio() {
                        const iframe = document.getElementById('studio-iframe');
                        iframe.src = 'http://localhost:8081';
                    }
                    
                    // Load status on page load
                    refreshStatus();
                    
                    // Refresh status every 5 seconds
                    setInterval(refreshStatus, 5000);
                </script>
            </body>
            </html>
            """)

except ImportError:
    # FastAPI not available
    def create_langgraph_studio_endpoints(app):
        """Placeholder when FastAPI is not available"""
        pass
