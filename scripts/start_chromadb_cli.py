#!/usr/bin/env python3
"""
ChromaDB CLI-based Server Startup
This script uses the ChromaDB CLI to start the server.
"""

import subprocess
import sys
import logging
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def start_chromadb_cli():
    """Start ChromaDB using CLI command."""
    try:
        logger.info("Starting ChromaDB using CLI...")
        
        # Create data directory
        data_dir = Path("chroma_data")
        data_dir.mkdir(exist_ok=True)
        
        # Try different CLI commands
        commands_to_try = [
            ["chroma", "run", "--host", "0.0.0.0", "--port", "8081", "--path", str(data_dir)],
            ["python", "-m", "chromadb", "run", "--host", "0.0.0.0", "--port", "8081", "--path", str(data_dir)],
            ["chroma", "server", "--host", "0.0.0.0", "--port", "8081", "--path", str(data_dir)],
        ]
        
        for cmd in commands_to_try:
            try:
                logger.info(f"Trying command: {' '.join(cmd)}")
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                logger.info("ChromaDB started successfully!")
                return
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                logger.warning(f"Command failed: {e}")
                continue
        
        # If all commands fail, try direct Python import
        logger.info("Trying direct Python import method...")
        import chromadb
        from chromadb.server import Server
        from chromadb.config import Settings
        
        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(data_dir),
            anonymized_telemetry=False,
            allow_reset=True,
            host="0.0.0.0",
            port=8081,
        )
        
        server = Server(settings)
        server.start()
        logger.info("ChromaDB server started via Python import!")
        
        # Keep running
        try:
            server.wait()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            server.stop()
            
    except Exception as e:
        logger.error(f"Failed to start ChromaDB: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_chromadb_cli()
