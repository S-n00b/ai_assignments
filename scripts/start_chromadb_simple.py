#!/usr/bin/env python3
"""
Simple ChromaDB Server Startup Script
This script starts ChromaDB without OpenTelemetry dependencies
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def start_chromadb_server():
    """Start ChromaDB server with minimal configuration."""
    try:
        # Create chroma directory if it doesn't exist
        chroma_dir = Path("chroma_db")
        chroma_dir.mkdir(exist_ok=True)
        
        # Import and start ChromaDB
        import chromadb
        from chromadb.config import Settings
        
        logger.info("Starting ChromaDB server...")
        
        # Configure ChromaDB settings
        settings = Settings(
            chroma_api_impl="chromadb.api.fastapi.FastAPI",
            chroma_server_host="0.0.0.0",
            chroma_server_http_port=8081,
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(chroma_dir.absolute()),
            anonymized_telemetry=False,  # Disable telemetry
            allow_reset=True
        )
        
        # Start the server
        client = chromadb.Client(settings)
        
        logger.info(f"ChromaDB server started on http://0.0.0.0:8081")
        logger.info(f"Persist directory: {chroma_dir.absolute()}")
        logger.info("ChromaDB server is ready!")
        
        # Keep the server running
        import time
        while True:
            time.sleep(1)
            
    except ImportError as e:
        logger.error(f"ChromaDB not installed: {e}")
        logger.info("Please install ChromaDB: pip install chromadb")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start ChromaDB: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_chromadb_server()

