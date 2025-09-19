"""
Optuna Hyperparameter Optimization for Enterprise LLMOps

This module provides comprehensive hyperparameter optimization using Optuna
for LLM models, including automated search strategies, pruning, and integration
with MLflow for experiment tracking.

Key Features:
- Automated hyperparameter search for LLM models
- Advanced pruning strategies (MedianPruner, SuccessiveHalving)
- Multi-objective optimization for performance vs. cost
- Integration with MLflow for experiment tracking
- Distributed optimization across multiple nodes
- Custom objective functions for LLM-specific metrics
"""

import optuna
import mlflow
import mlflow.optuna
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    study_name: str
    direction: Union[str, List[str]]  # "maximize", "minimize", or list for multi-objective
    n_trials: int = 100
    timeout: Optional[int] = None  # seconds
    pruning_enabled: bool = True
    pruning_strategy: str = "median"  # "median", "successive_halving", "hyperband"
    n_jobs: int = 1
    storage_url: Optional[str] = None  # for distributed optimization
    sampler: str = "tpe"  # "tpe", "random", "cmaes", "nsgaii"
    multi_objective: bool = False
    optimization_metrics: List[str] = None
    constraints: Dict[str, Any] = None


@dataclass
class LLMHyperparameterSpace:
    """Hyperparameter search space for LLM models."""
    # Model parameters
    model_name: str
    temperature: Tuple[float, float] = (0.1, 1.0)
    max_tokens: Tuple[int, int] = (512, 4096)
    top_p: Tuple[float, float] = (0.1, 1.0)
    top_k: Tuple[int, int] = (1, 100)
    frequency_penalty: Tuple[float, float] = (0.0, 2.0)
    presence_penalty: Tuple[float, float] = (0.0, 2.0)
    
    # Training parameters (for fine-tuning)
    learning_rate: Tuple[float, float] = (1e-6, 1e-3)
    batch_size: Tuple[int, int] = (1, 32)
    num_epochs: Tuple[int, int] = (1, 10)
    warmup_steps: Tuple[int, int] = (0, 1000)
    weight_decay: Tuple[float, float] = (0.0, 0.1)
    
    # Architecture parameters
    hidden_size: Tuple[int, int] = (512, 4096)
    num_attention_heads: Tuple[int, int] = (8, 32)
    num_layers: Tuple[int, int] = (6, 24)
    dropout_rate: Tuple[float, float] = (0.0, 0.5)
    
    # Custom parameters
    custom_params: Dict[str, Tuple[Any, Any]] = None


class OptunaOptimizer:
    """
    Optuna-based hyperparameter optimizer for LLM models.
    
    This class provides comprehensive hyperparameter optimization capabilities
    specifically designed for LLM models with enterprise-grade features.
    """
    
    def __init__(self, config: OptimizationConfig, mlflow_tracking_uri: str = None):
        """Initialize the Optuna optimizer."""
        self.config = config
        self.mlflow_tracking_uri = mlflow_tracking_uri or "http://localhost:5000"
        self.logger = self._setup_logging()
        
        # Initialize MLflow
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        # Initialize Optuna study
        self.study = None
        self.best_params = None
        self.optimization_history = []
        
        # Thread safety
        self._lock = threading.Lock()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for Optuna optimizer."""
        logger = logging.getLogger("optuna_optimizer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create Optuna sampler based on configuration."""
        if self.config.sampler == "tpe":
            return optuna.samplers.TPESampler(
                n_startup_trials=10,
                n_ei_candidates=24,
                seed=42
            )
        elif self.config.sampler == "random":
            return optuna.samplers.RandomSampler(seed=42)
        elif self.config.sampler == "cmaes":
            return optuna.samplers.CmaEsSampler(seed=42)
        elif self.config.sampler == "nsgaii" and self.config.multi_objective:
            return optuna.samplers.NSGAIISampler(
                population_size=50,
                seed=42
            )
        else:
            return optuna.samplers.TPESampler(seed=42)
    
    def _create_pruner(self) -> Optional[optuna.pruners.BasePruner]:
        """Create Optuna pruner based on configuration."""
        if not self.config.pruning_enabled:
            return None
        
        if self.config.pruning_strategy == "median":
            return optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        elif self.config.pruning_strategy == "successive_halving":
            return optuna.pruners.SuccessiveHalvingPruner(
                min_resource=1,
                reduction_factor=4,
                min_early_stopping_rate=0
            )
        elif self.config.pruning_strategy == "hyperband":
            return optuna.pruners.HyperbandPruner(
                min_resource=1,
                max_resource=100,
                reduction_factor=3
            )
        else:
            return optuna.pruners.MedianPruner()
    
    def _create_study(self):
        """Create Optuna study."""
        try:
            sampler = self._create_sampler()
            pruner = self._create_pruner()
            
            # Create or load study
            if self.config.storage_url:
                # Distributed optimization
                study = optuna.load_study(
                    study_name=self.config.study_name,
                    storage=self.config.storage_url,
                    sampler=sampler,
                    pruner=pruner
                )
            else:
                # Local optimization
                study = optuna.create_study(
                    study_name=self.config.study_name,
                    direction=self.config.direction,
                    sampler=sampler,
                    pruner=pruner
                )
            
            self.study = study
            self.logger.info(f"Created/loaded study: {self.config.study_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to create study: {e}")
            raise
    
    def _suggest_hyperparameters(
        self,
        trial: optuna.Trial,
        search_space: LLMHyperparameterSpace
    ) -> Dict[str, Any]:
        """Suggest hyperparameters for a trial."""
        params = {}
        
        # Model parameters
        params["temperature"] = trial.suggest_float(
            "temperature",
            search_space.temperature[0],
            search_space.temperature[1],
            log=True
        )
        
        params["max_tokens"] = trial.suggest_int(
            "max_tokens",
            search_space.max_tokens[0],
            search_space.max_tokens[1],
            step=64
        )
        
        params["top_p"] = trial.suggest_float(
            "top_p",
            search_space.top_p[0],
            search_space.top_p[1]
        )
        
        params["top_k"] = trial.suggest_int(
            "top_k",
            search_space.top_k[0],
            search_space.top_k[1]
        )
        
        params["frequency_penalty"] = trial.suggest_float(
            "frequency_penalty",
            search_space.frequency_penalty[0],
            search_space.frequency_penalty[1]
        )
        
        params["presence_penalty"] = trial.suggest_float(
            "presence_penalty",
            search_space.presence_penalty[0],
            search_space.presence_penalty[1]
        )
        
        # Training parameters
        params["learning_rate"] = trial.suggest_float(
            "learning_rate",
            search_space.learning_rate[0],
            search_space.learning_rate[1],
            log=True
        )
        
        params["batch_size"] = trial.suggest_int(
            "batch_size",
            search_space.batch_size[0],
            search_space.batch_size[1],
            step=1
        )
        
        params["num_epochs"] = trial.suggest_int(
            "num_epochs",
            search_space.num_epochs[0],
            search_space.num_epochs[1]
        )
        
        params["warmup_steps"] = trial.suggest_int(
            "warmup_steps",
            search_space.warmup_steps[0],
            search_space.warmup_steps[1]
        )
        
        params["weight_decay"] = trial.suggest_float(
            "weight_decay",
            search_space.weight_decay[0],
            search_space.weight_decay[1]
        )
        
        # Architecture parameters
        params["hidden_size"] = trial.suggest_int(
            "hidden_size",
            search_space.hidden_size[0],
            search_space.hidden_size[1],
            step=128
        )
        
        params["num_attention_heads"] = trial.suggest_int(
            "num_attention_heads",
            search_space.num_attention_heads[0],
            search_space.num_attention_heads[1]
        )
        
        params["num_layers"] = trial.suggest_int(
            "num_layers",
            search_space.num_layers[0],
            search_space.num_layers[1]
        )
        
        params["dropout_rate"] = trial.suggest_float(
            "dropout_rate",
            search_space.dropout_rate[0],
            search_space.dropout_rate[1]
        )
        
        # Custom parameters
        if search_space.custom_params:
            for param_name, (low, high) in search_space.custom_params.items():
                if isinstance(low, float) and isinstance(high, float):
                    params[param_name] = trial.suggest_float(
                        param_name, low, high
                    )
                elif isinstance(low, int) and isinstance(high, int):
                    params[param_name] = trial.suggest_int(
                        param_name, low, high
                    )
        
        return params
    
    def _create_objective_function(
        self,
        model_evaluator: Callable,
        search_space: LLMHyperparameterSpace,
        evaluation_data: Any,
        optimization_metrics: List[str]
    ) -> Callable:
        """Create objective function for optimization."""
        
        def objective(trial: optuna.Trial) -> Union[float, Tuple[float, ...]]:
            try:
                # Suggest hyperparameters
                params = self._suggest_hyperparameters(trial, search_space)
                
                # Start MLflow run
                with mlflow.start_run(nested=True):
                    # Log hyperparameters
                    mlflow.log_params(params)
                    
                    # Log trial information
                    mlflow.log_param("trial_number", trial.number)
                    mlflow.log_param("study_name", self.config.study_name)
                    
                    # Evaluate model
                    start_time = time.time()
                    evaluation_results = model_evaluator(params, evaluation_data)
                    evaluation_time = time.time() - start_time
                    
                    # Log evaluation results
                    mlflow.log_metrics(evaluation_results)
                    mlflow.log_metric("evaluation_time", evaluation_time)
                    
                    # Log optimization metadata
                    mlflow.log_param("optimization_metrics", optimization_metrics)
                    mlflow.log_param("search_space", asdict(search_space))
                    
                    # Handle multi-objective optimization
                    if self.config.multi_objective and len(optimization_metrics) > 1:
                        objective_values = tuple(
                            evaluation_results.get(metric, 0.0) 
                            for metric in optimization_metrics
                        )
                        
                        # Log individual objectives
                        for i, metric in enumerate(optimization_metrics):
                            mlflow.log_metric(f"objective_{i}_{metric}", objective_values[i])
                        
                        return objective_values
                    else:
                        # Single objective optimization
                        primary_metric = optimization_metrics[0]
                        objective_value = evaluation_results.get(primary_metric, 0.0)
                        
                        # Log secondary metrics
                        for metric in optimization_metrics[1:]:
                            mlflow.log_metric(f"secondary_{metric}", evaluation_results.get(metric, 0.0))
                        
                        return objective_value
                
            except Exception as e:
                self.logger.error(f"Trial {trial.number} failed: {e}")
                mlflow.log_param("trial_status", "failed")
                mlflow.log_param("error_message", str(e))
                
                # Return worst possible value for minimization, best for maximization
                if self.config.direction == "minimize":
                    return float('inf')
                else:
                    return float('-inf')
        
        return objective
    
    def optimize(
        self,
        model_evaluator: Callable,
        search_space: LLMHyperparameterSpace,
        evaluation_data: Any,
        optimization_metrics: List[str] = None
    ) -> Dict[str, Any]:
        """Run hyperparameter optimization."""
        try:
            if optimization_metrics is None:
                optimization_metrics = ["accuracy", "f1_score"]
            
            # Create study if not exists
            if self.study is None:
                self._create_study()
            
            # Create objective function
            objective = self._create_objective_function(
                model_evaluator,
                search_space,
                evaluation_data,
                optimization_metrics
            )
            
            # Start MLflow experiment
            mlflow.set_experiment(f"optuna_optimization_{self.config.study_name}")
            
            with mlflow.start_run(run_name=f"study_{self.config.study_name}"):
                # Log study configuration
                mlflow.log_params(asdict(self.config))
                
                # Run optimization
                self.logger.info(f"Starting optimization with {self.config.n_trials} trials")
                
                if self.config.timeout:
                    self.study.optimize(
                        objective,
                        n_trials=self.config.n_trials,
                        timeout=self.config.timeout,
                        n_jobs=self.config.n_jobs
                    )
                else:
                    self.study.optimize(
                        objective,
                        n_trials=self.config.n_trials,
                        n_jobs=self.config.n_jobs
                    )
                
                # Get best parameters
                if self.config.multi_objective:
                    best_trials = self.study.best_trials
                    best_params = [trial.params for trial in best_trials]
                    best_values = [trial.values for trial in best_trials]
                else:
                    best_params = self.study.best_params
                    best_values = self.study.best_value
                
                # Log optimization results
                mlflow.log_param("best_params", best_params)
                mlflow.log_param("best_values", best_values)
                mlflow.log_param("n_trials_completed", len(self.study.trials))
                
                # Store results
                self.best_params = best_params
                
                # Create optimization summary
                optimization_summary = {
                    "study_name": self.config.study_name,
                    "best_params": best_params,
                    "best_values": best_values,
                    "n_trials": len(self.study.trials),
                    "optimization_metrics": optimization_metrics,
                    "search_space": asdict(search_space),
                    "completed_at": datetime.now().isoformat()
                }
                
                # Log summary
                mlflow.log_text(
                    json.dumps(optimization_summary, indent=2),
                    "optimization_summary.json"
                )
                
                self.logger.info(f"Optimization completed. Best value: {best_values}")
                
                return optimization_summary
                
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        if not self.study:
            return []
        
        history = []
        for trial in self.study.trials:
            trial_data = {
                "number": trial.number,
                "value": trial.value,
                "values": trial.values,
                "params": trial.params,
                "state": trial.state.name,
                "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
                "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
                "duration": trial.duration.total_seconds() if trial.duration else None,
                "user_attrs": trial.user_attrs,
                "system_attrs": trial.system_attrs
            }
            history.append(trial_data)
        
        return history
    
    def suggest_next_trials(self, n_trials: int = 5) -> List[Dict[str, Any]]:
        """Suggest next trials for manual exploration."""
        if not self.study:
            return []
        
        suggestions = []
        for _ in range(n_trials):
            trial = self.study.ask()
            suggestion = {
                "number": trial.number,
                "params": trial.params,
                "suggested_at": datetime.now().isoformat()
            }
            suggestions.append(suggestion)
        
        return suggestions
    
    def get_parameter_importance(self) -> Dict[str, float]:
        """Get parameter importance scores."""
        if not self.study or len(self.study.trials) < 10:
            return {}
        
        try:
            importance = optuna.importance.get_param_importances(self.study)
            return importance
        except Exception as e:
            self.logger.warning(f"Could not calculate parameter importance: {e}")
            return {}
    
    def plot_optimization_history(self, save_path: str = None):
        """Plot optimization history."""
        if not self.study:
            return None
        
        try:
            import matplotlib.pyplot as plt
            
            fig = optuna.visualization.matplotlib.plot_optimization_history(self.study)
            if save_path:
                fig.savefig(save_path)
                self.logger.info(f"Optimization history plot saved to {save_path}")
            
            return fig
        except ImportError:
            self.logger.warning("Matplotlib not available for plotting")
            return None
        except Exception as e:
            self.logger.error(f"Failed to plot optimization history: {e}")
            return None
    
    def plot_parameter_relationships(self, save_path: str = None):
        """Plot parameter relationships."""
        if not self.study:
            return None
        
        try:
            import matplotlib.pyplot as plt
            
            fig = optuna.visualization.matplotlib.plot_parallel_coordinate(self.study)
            if save_path:
                fig.savefig(save_path)
                self.logger.info(f"Parameter relationships plot saved to {save_path}")
            
            return fig
        except ImportError:
            self.logger.warning("Matplotlib not available for plotting")
            return None
        except Exception as e:
            self.logger.error(f"Failed to plot parameter relationships: {e}")
            return None


class DistributedOptunaOptimizer:
    """
    Distributed Optuna optimizer for large-scale hyperparameter optimization.
    
    This class provides distributed optimization capabilities using multiple
    workers across different nodes for faster optimization.
    """
    
    def __init__(self, config: OptimizationConfig, storage_url: str):
        """Initialize distributed optimizer."""
        self.config = config
        self.storage_url = storage_url
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger("distributed_optuna")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def create_shared_study(self, study_name: str) -> optuna.Study:
        """Create a shared study for distributed optimization."""
        try:
            study = optuna.create_study(
                study_name=study_name,
                storage=self.storage_url,
                direction=self.config.direction,
                sampler=self._create_sampler(),
                pruner=self._create_pruner(),
                load_if_exists=True
            )
            
            self.logger.info(f"Created shared study: {study_name}")
            return study
            
        except Exception as e:
            self.logger.error(f"Failed to create shared study: {e}")
            raise
    
    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create sampler for distributed optimization."""
        if self.config.sampler == "tpe":
            return optuna.samplers.TPESampler(
                n_startup_trials=10,
                n_ei_candidates=24,
                seed=42
            )
        else:
            return optuna.samplers.RandomSampler(seed=42)
    
    def _create_pruner(self) -> Optional[optuna.pruners.BasePruner]:
        """Create pruner for distributed optimization."""
        if not self.config.pruning_enabled:
            return None
        
        return optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        )
    
    async def run_distributed_optimization(
        self,
        study_name: str,
        model_evaluator: Callable,
        search_space: LLMHyperparameterSpace,
        evaluation_data: Any,
        optimization_metrics: List[str],
        n_workers: int = 4
    ) -> Dict[str, Any]:
        """Run distributed optimization across multiple workers."""
        try:
            # Create shared study
            study = self.create_shared_study(study_name)
            
            # Create objective function
            optimizer = OptunaOptimizer(self.config, storage_url=self.storage_url)
            objective = optimizer._create_objective_function(
                model_evaluator,
                search_space,
                evaluation_data,
                optimization_metrics
            )
            
            # Run distributed optimization
            self.logger.info(f"Starting distributed optimization with {n_workers} workers")
            
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = []
                
                for worker_id in range(n_workers):
                    future = executor.submit(
                        study.optimize,
                        objective,
                        n_trials=self.config.n_trials // n_workers,
                        timeout=self.config.timeout
                    )
                    futures.append(future)
                
                # Wait for all workers to complete
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.error(f"Worker failed: {e}")
            
            # Get results
            if self.config.multi_objective:
                best_trials = study.best_trials
                best_params = [trial.params for trial in best_trials]
                best_values = [trial.values for trial in best_trials]
            else:
                best_params = study.best_params
                best_values = study.best_value
            
            optimization_summary = {
                "study_name": study_name,
                "best_params": best_params,
                "best_values": best_values,
                "n_trials": len(study.trials),
                "n_workers": n_workers,
                "optimization_metrics": optimization_metrics,
                "completed_at": datetime.now().isoformat()
            }
            
            self.logger.info(f"Distributed optimization completed. Best value: {best_values}")
            
            return optimization_summary
            
        except Exception as e:
            self.logger.error(f"Distributed optimization failed: {e}")
            raise
