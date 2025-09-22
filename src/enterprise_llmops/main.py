"""
Enterprise LLMOps Platform - Main Entry Point

This is the main entry point for the Enterprise LLMOps platform,
providing a comprehensive solution for LLM operations with:

- Ollama integration for local model management
- MLflow for experiment tracking and model registry
- Optuna for automated hyperparameter optimization
- Kubernetes orchestration with monitoring
- Vector databases (Chroma, Weaviate, Pinecone)
- Monitoring stack (Prometheus, Grafana, LangFuse)
- LangGraph Studio and Neo4j integration
- Modern FastAPI frontend with real-time updates

Usage:
    python -m src.enterprise_llmops.main [options]

Options:
    --host HOST              Host to bind to (default: 0.0.0.0)
    --port PORT              Port to bind to (default: 8080)
    --workers WORKERS        Number of worker processes (default: 1)
    --config CONFIG          Path to configuration file
    --log-level LEVEL        Logging level (default: info)
    --enable-gpu             Enable GPU support
    --enable-monitoring      Enable monitoring stack
    --enable-automl          Enable AutoML features
"""

import asyncio
import logging
import argparse
import sys
from pathlib import Path
from typing import Optional
import uvicorn
from contextlib import asynccontextmanager

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.enterprise_llmops.frontend.fastapi_app import app
from src.enterprise_llmops.ollama_manager import OllamaManager
from src.enterprise_llmops.model_registry import EnterpriseModelRegistry
from src.enterprise_llmops.mlops.mlflow_manager import MLflowManager, ExperimentConfig
from src.enterprise_llmops.automl.optuna_optimizer import OptunaOptimizer, OptimizationConfig


def setup_logging(level: str = "info"):
    """Setup logging configuration."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logs_dir / "llmops.log")
        ]
    )
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(log_level)
    logging.getLogger("fastapi").setLevel(log_level)
    logging.getLogger("ollama_manager").setLevel(log_level)
    logging.getLogger("model_registry").setLevel(log_level)
    logging.getLogger("mlflow_manager").setLevel(log_level)
    logging.getLogger("optuna_optimizer").setLevel(log_level)


async def initialize_services(config: dict):
    """Initialize all enterprise services."""
    logger = logging.getLogger("main")
    
    try:
        logger.info("Initializing Enterprise LLMOps Platform...")
        
        # Initialize Ollama Manager (optional)
        if config.get("enable_ollama", True):
            try:
                logger.info("Initializing Ollama Manager...")
                ollama_manager = OllamaManager()
                await ollama_manager.initialize()
                logger.info("Ollama Manager initialized successfully")
            except Exception as e:
                logger.warning(f"Ollama Manager initialization failed: {e}")
                logger.info("Continuing without Ollama support...")
                config["enable_ollama"] = False
        
        # Initialize Model Registry
        if config.get("enable_model_registry", True):
            logger.info("Initializing Model Registry...")
            model_registry = EnterpriseModelRegistry()
            logger.info("Model Registry initialized successfully")
        
        # Initialize MLflow Manager
        if config.get("enable_mlflow", True):
            logger.info("Initializing MLflow Manager...")
            mlflow_config = ExperimentConfig(
                experiment_name="llmops_enterprise",
                tracking_uri=config.get("mlflow_tracking_uri", "http://localhost:5000"),
                description="Enterprise LLMOps Experiment Tracking"
            )
            mlflow_manager = MLflowManager(mlflow_config)
            logger.info("MLflow Manager initialized successfully")
        
        # Initialize Optuna Optimizer
        if config.get("enable_automl", True):
            logger.info("Initializing Optuna Optimizer...")
            optuna_config = OptimizationConfig(
                study_name="llm_optimization",
                direction="maximize",
                n_trials=config.get("optuna_n_trials", 100),
                pruning_enabled=config.get("optuna_pruning", True)
            )
            optuna_optimizer = OptunaOptimizer(optuna_config)
            logger.info("Optuna Optimizer initialized successfully")
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from file or use defaults."""
    default_config = {
        "host": "0.0.0.0",
        "port": 8080,
        "workers": 1,
        "log_level": "info",
        "enable_ollama": True,
        "enable_model_registry": True,
        "enable_mlflow": True,
        "enable_automl": True,
        "enable_monitoring": True,
        "enable_gpu": False,
        "mlflow_tracking_uri": "http://localhost:5000",
        "optuna_n_trials": 100,
        "optuna_pruning": True,
        "vector_databases": {
            "chroma": {"enabled": True, "url": "http://localhost:8081"},
            "weaviate": {"enabled": True, "url": "http://localhost:8083"},
            "pinecone": {"enabled": False, "api_key": None}
        },
        "monitoring": {
            "prometheus": {"enabled": True, "url": "http://localhost:9090"},
            "grafana": {"enabled": True, "url": "http://localhost:3000"},
            "langfuse": {"enabled": True, "url": "http://localhost:3000"}
        },
        "integrations": {
            "langgraph_studio": {"enabled": True, "url": "http://localhost:8080"},
            "neo4j": {"enabled": True, "url": "http://localhost:7474"}
        }
    }
    
    if config_path and Path(config_path).exists():
        import yaml
        with open(config_path, 'r') as f:
            file_config = yaml.safe_load(f)
            default_config.update(file_config)
    
    return default_config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enterprise LLMOps Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind to (default: 8080)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Logging level (default: info)"
    )
    
    parser.add_argument(
        "--enable-gpu",
        action="store_true",
        help="Enable GPU support"
    )
    
    parser.add_argument(
        "--enable-monitoring",
        action="store_true",
        help="Enable monitoring stack"
    )
    
    parser.add_argument(
        "--enable-automl",
        action="store_true",
        help="Enable AutoML features"
    )
    
    parser.add_argument(
        "--disable-ollama",
        action="store_true",
        help="Disable Ollama integration"
    )
    
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Start with minimal configuration (disable optional services)"
    )
    
    parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help="Disable MLflow integration"
    )
    
    parser.add_argument(
        "--disable-model-registry",
        action="store_true",
        help="Disable model registry"
    )
    
    parser.add_argument(
        "--enable-auth",
        action="store_true",
        help="Enable authentication (disabled by default for demo)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger("main")
    
    try:
        logger.info("Starting Enterprise LLMOps Platform...")
        logger.info(f"Arguments: {vars(args)}")
        
        # Load configuration
        config = load_config(args.config)
        
        # Override config with command line arguments
        config["host"] = args.host
        config["port"] = args.port
        config["workers"] = args.workers
        config["log_level"] = args.log_level
        config["enable_gpu"] = args.enable_gpu
        config["enable_monitoring"] = args.enable_monitoring
        config["enable_automl"] = args.enable_automl
        # Handle minimal configuration
        if args.minimal:
            config["enable_ollama"] = False
            config["enable_automl"] = False
            config["enable_monitoring"] = False
        
        config["enable_ollama"] = not args.disable_ollama and config["enable_ollama"]
        config["enable_mlflow"] = not args.disable_mlflow
        config["enable_model_registry"] = not args.disable_model_registry
        config["enable_auth"] = args.enable_auth
        
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        
        # Initialize services
        try:
            asyncio.run(initialize_services(config))
        except Exception as e:
            logger.error(f"Service initialization failed: {e}")
            logger.info("Starting with minimal configuration...")
        
        # Start the FastAPI application
        logger.info(f"Starting server on {config['host']}:{config['port']}")
        
        try:
            # Set the config in the FastAPI app
            from src.enterprise_llmops.frontend.fastapi_app import app
            app.state.config = config
            
            uvicorn.run(
                app,
                host=config["host"],
                port=config["port"],
                workers=config["workers"],
                log_level=config["log_level"],
                access_log=True,
                reload=False
            )
        except Exception as e:
            logger.error(f"Failed to start FastAPI server: {e}")
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
