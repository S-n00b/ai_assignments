#!/usr/bin/env python3
"""
ChromaDB Server for Enterprise LLMOps (New Architecture)
This script starts ChromaDB using the new 1.x architecture with HTTP API server.
"""

import os
import sys
import logging
import asyncio
from pathlib import Path
import uvicorn
from chromadb.config import Settings
from chromadb.server import Server
from chromadb.api import API

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def start_chromadb_server():
    """Start ChromaDB server with new architecture."""
    try:
        logger.info("Starting ChromaDB server with new architecture...")
        
        # Create data directory
        data_dir = Path("chroma_data")
        data_dir.mkdir(exist_ok=True)
        
        # Configure ChromaDB settings
        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(data_dir),
            anonymized_telemetry=False,
            allow_reset=True,
            host="0.0.0.0",
            port=8081,
            chroma_api_impl="chromadb.api.fastapi.FastAPI",
            chroma_server_host="0.0.0.0",
            chroma_server_http_port=8081,
            chroma_server_grpc_port=8001,
        )
        
        logger.info(f"ChromaDB data directory: {data_dir}")
        logger.info(f"ChromaDB HTTP server: http://0.0.0.0:8081")
        logger.info(f"ChromaDB gRPC server: 0.0.0.0:8001")
        
        # Create and start the server
        server = Server(settings)
        server.start()
        
        logger.info("ChromaDB server started successfully!")
        logger.info("Available endpoints:")
        logger.info("  - Health check: http://localhost:8081/api/v1/heartbeat")
        logger.info("  - API docs: http://localhost:8081/docs")
        
        # Keep the server running
        try:
            server.wait()
        except KeyboardInterrupt:
            logger.info("Shutting down ChromaDB server...")
            server.stop()
            
    except Exception as e:
        logger.error(f"Failed to start ChromaDB server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_chromadb_server()
