#!/usr/bin/env python3
"""
Phase 6 Platform Startup Script

This script starts the Enhanced Unified Platform with all Phase 6 components
including AI Architect workspace, Model Evaluation workspace, Factory Roster
Dashboard, Real-time Monitoring, and Unified Data Flow Visualization.

Usage:
    python scripts/start_phase6_platform.py [--host HOST] [--port PORT] [--config CONFIG]
"""

import asyncio
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any
import yaml
import subprocess
import signal
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.enterprise_llmops.frontend.enhanced_unified_platform_app import app, platform


class Phase6PlatformStarter:
    """
    Phase 6 Platform Starter for Enhanced Unified Platform.
    
    This class provides comprehensive startup functionality for all Phase 6
    components with proper service orchestration and health monitoring.
    """
    
    def __init__(self, config_path: str = "config/enhanced_platform_config.yaml"):
        """Initialize the Phase 6 Platform Starter."""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.processes = {}
        self.startup_time = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load Phase 6 configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for Phase 6 Platform Starter."""
        logger = logging.getLogger("phase6_platform_starter")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def check_dependencies(self) -> bool:
        """Check if all required dependencies are available."""
        try:
            self.logger.info("Checking dependencies...")
            
            # Check Python packages
            required_packages = [
                "fastapi", "uvicorn", "aiohttp", "websockets",
                "mlflow", "chromadb", "neo4j", "duckdb",
                "langchain", "llamaindex", "smolagent", "langgraph"
            ]
            
            missing_packages = []
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    missing_packages.append(package)
            
            if missing_packages:
                self.logger.error(f"Missing required packages: {missing_packages}")
                return False
            
            # Check external services
            services = self.config.get("services", {})
            for service_name, service_config in services.items():
                if service_name != "duckdb":  # DuckDB is embedded
                    port = service_config.get("port")
                    if port and port != "embedded":
                        # Check if service is running
                        if not await self._check_service_health(service_config):
                            self.logger.warning(f"Service {service_name} may not be running on port {port}")
            
            self.logger.info("Dependencies check completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Dependencies check failed: {e}")
            return False
    
    async def _check_service_health(self, service_config: Dict[str, Any]) -> bool:
        """Check if a service is healthy."""
        try:
            import aiohttp
            
            url = service_config.get("url", "")
            health_endpoint = service_config.get("health_endpoint", "/health")
            full_url = f"{url}{health_endpoint}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(full_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    return response.status == 200
                    
        except Exception:
            return False
    
    async def start_services(self) -> bool:
        """Start all required services."""
        try:
            self.logger.info("Starting required services...")
            
            # Start ChromaDB if not running
            if not await self._check_service_health(self.config["services"]["chromadb"]):
                self.logger.info("Starting ChromaDB...")
                await self._start_chromadb()
            
            # Start MLflow if not running
            if not await self._check_service_health(self.config["services"]["mlflow_tracking"]):
                self.logger.info("Starting MLflow...")
                await self._start_mlflow()
            
            # Start Neo4j if not running
            if not await self._check_service_health(self.config["services"]["neo4j"]):
                self.logger.info("Starting Neo4j...")
                await self._start_neo4j()
            
            self.logger.info("All services started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start services: {e}")
            return False
    
    async def _start_chromadb(self):
        """Start ChromaDB service."""
        try:
            cmd = ["chroma", "run", "--host", "0.0.0.0", "--port", "8081", "--path", "chroma_data"]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.processes["chromadb"] = process
            self.logger.info("ChromaDB started")
        except Exception as e:
            self.logger.error(f"Failed to start ChromaDB: {e}")
    
    async def _start_mlflow(self):
        """Start MLflow service."""
        try:
            cmd = [
                "mlflow", "server",
                "--backend-store-uri", "sqlite:///mlflow.db",
                "--default-artifact-root", "./mlruns",
                "--host", "0.0.0.0",
                "--port", "5000"
            ]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.processes["mlflow"] = process
            self.logger.info("MLflow started")
        except Exception as e:
            self.logger.error(f"Failed to start MLflow: {e}")
    
    async def _start_neo4j(self):
        """Start Neo4j service."""
        try:
            # Note: This assumes Neo4j is installed and configured
            # In a real deployment, you would use Docker or systemd
            self.logger.info("Neo4j should be started manually or via Docker")
        except Exception as e:
            self.logger.error(f"Failed to start Neo4j: {e}")
    
    async def start_platform(self, host: str = "0.0.0.0", port: int = 8080) -> bool:
        """Start the Enhanced Unified Platform."""
        try:
            self.logger.info(f"Starting Enhanced Unified Platform on {host}:{port}")
            self.startup_time = time.time()
            
            # Initialize platform
            success = await platform.initialize()
            if not success:
                self.logger.error("Failed to initialize platform")
                return False
            
            # Start FastAPI application
            import uvicorn
            
            config = uvicorn.Config(
                app=app,
                host=host,
                port=port,
                log_level="info",
                reload=False
            )
            
            server = uvicorn.Server(config)
            await server.serve()
            
        except Exception as e:
            self.logger.error(f"Failed to start platform: {e}")
            return False
    
    async def stop_services(self):
        """Stop all started services."""
        try:
            self.logger.info("Stopping services...")
            
            for service_name, process in self.processes.items():
                if process and process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
                    self.logger.info(f"Stopped {service_name}")
            
            self.processes.clear()
            
        except Exception as e:
            self.logger.error(f"Failed to stop services: {e}")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down...")
            asyncio.create_task(self.stop_services())
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def run(self, host: str = "0.0.0.0", port: int = 8080):
        """Run the Phase 6 Platform."""
        try:
            self.logger.info("=" * 60)
            self.logger.info("Lenovo AI Architecture - Phase 6 Platform")
            self.logger.info("Enhanced Unified Platform with Clear Data Flow")
            self.logger.info("=" * 60)
            
            # Setup signal handlers
            self.setup_signal_handlers()
            
            # Check dependencies
            if not await self.check_dependencies():
                self.logger.error("Dependency check failed")
                return False
            
            # Start services
            if not await self.start_services():
                self.logger.error("Failed to start services")
                return False
            
            # Wait for services to be ready
            self.logger.info("Waiting for services to be ready...")
            await asyncio.sleep(5)
            
            # Start platform
            self.logger.info("Starting Enhanced Unified Platform...")
            await self.start_platform(host, port)
            
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            self.logger.error(f"Platform startup failed: {e}")
        finally:
            await self.stop_services()


def main():
    """Main entry point for Phase 6 Platform startup."""
    parser = argparse.ArgumentParser(description="Start Phase 6 Enhanced Unified Platform")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--config", default="config/enhanced_platform_config.yaml", help="Configuration file path")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run starter
    starter = Phase6PlatformStarter(args.config)
    
    try:
        asyncio.run(starter.run(args.host, args.port))
    except KeyboardInterrupt:
        print("\\nShutdown complete")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
