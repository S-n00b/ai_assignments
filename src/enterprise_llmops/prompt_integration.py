"""
Prompt Integration Module for Enterprise LLMOps Platform

This module provides integration between the prompt registries and the
enterprise LLMOps platform, enabling seamless sync of AI tool prompts
and enhanced evaluation datasets.
"""

import asyncio
import logging
import json
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.model_evaluation.prompt_registries import PromptRegistryManager, PromptCategory, PromptEntry

logger = logging.getLogger(__name__)


class PromptIntegrationManager:
    """
    Manager for integrating prompt registries with the Enterprise LLMOps platform.
    
    This class handles the synchronization of AI tool prompts, generation of
    evaluation datasets, and integration with the model registry and MLflow.
    """
    
    def __init__(self, cache_dir: str = "cache/ai_tool_prompts"):
        """
        Initialize the prompt integration manager.
        
        Args:
            cache_dir: Directory containing cached AI tool prompts
        """
        self.cache_dir = Path(cache_dir)
        self.prompt_manager = PromptRegistryManager(enable_caching=True, cache_dir=str(cache_dir))
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Initialized PromptIntegrationManager")
    
    async def sync_ai_tool_prompts(self) -> Dict[str, Any]:
        """
        Sync AI tool prompts from cache with the platform.
        
        Returns:
            Dictionary containing sync results and statistics
        """
        try:
            self.logger.info("Starting AI tool prompts sync...")
            
            # Load all AI tool prompts
            ai_tool_prompts = await self.prompt_manager.load_ai_tool_system_prompts()
            
            # Get statistics
            stats = self._analyze_prompt_collection(ai_tool_prompts)
            
            # Save enhanced dataset
            enhanced_dataset = await self._create_enhanced_dataset(ai_tool_prompts)
            
            self.logger.info(f"Successfully synced {len(ai_tool_prompts)} AI tool prompts")
            
            return {
                "success": True,
                "total_prompts": len(ai_tool_prompts),
                "statistics": stats,
                "enhanced_dataset_size": len(enhanced_dataset),
                "categories": list(set(prompt.category.value for prompt in ai_tool_prompts))
            }
            
        except Exception as e:
            self.logger.error(f"Error syncing AI tool prompts: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "total_prompts": 0
            }
    
    def _analyze_prompt_collection(self, prompts: List[PromptEntry]) -> Dict[str, Any]:
        """Analyze a collection of prompts and return statistics."""
        if not prompts:
            return {"error": "No prompts to analyze"}
        
        # Group by category
        categories = {}
        for prompt in prompts:
            category = prompt.category.value
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
        
        # Group by source
        sources = {}
        for prompt in prompts:
            source = prompt.source
            if source not in sources:
                sources[source] = 0
            sources[source] += 1
        
        # Calculate quality metrics
        quality_scores = [prompt.quality_score for prompt in prompts]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        return {
            "total_prompts": len(prompts),
            "category_distribution": categories,
            "source_distribution": sources,
            "average_quality_score": avg_quality,
            "quality_range": {
                "min": min(quality_scores) if quality_scores else 0,
                "max": max(quality_scores) if quality_scores else 0
            }
        }
    
    async def _create_enhanced_dataset(self, ai_tool_prompts: List[PromptEntry]) -> pd.DataFrame:
        """Create an enhanced evaluation dataset from AI tool prompts."""
        try:
            # Convert AI tool prompts to DataFrame
            ai_tool_data = []
            for prompt in ai_tool_prompts:
                ai_tool_data.append({
                    'id': prompt.id,
                    'input': prompt.text,
                    'category': prompt.category.value,
                    'source': prompt.source,
                    'quality_score': prompt.quality_score,
                    'difficulty_level': prompt.difficulty_level,
                    'metadata': json.dumps(prompt.metadata)
                })
            
            ai_tool_df = pd.DataFrame(ai_tool_data)
            
            # Generate additional synthetic prompts for comprehensive coverage
            synthetic_dataset = self.prompt_manager.get_enhanced_evaluation_dataset(
                target_size=2000,
                categories=[
                    PromptCategory.CODE_GENERATION,
                    PromptCategory.CREATIVE_WRITING,
                    PromptCategory.TECHNICAL_DOCUMENTATION,
                    PromptCategory.REASONING,
                    PromptCategory.QUESTION_ANSWERING
                ],
                quality_threshold=0.6
            )
            
            # Combine datasets
            combined_dataset = pd.concat([ai_tool_df, synthetic_dataset], ignore_index=True)
            
            # Save the enhanced dataset
            output_path = project_root / "data" / "evaluation_datasets" / "enhanced_ai_tool_dataset.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            combined_dataset.to_csv(output_path, index=False)
            
            self.logger.info(f"Created enhanced dataset with {len(combined_dataset)} prompts")
            
            return combined_dataset
            
        except Exception as e:
            self.logger.error(f"Error creating enhanced dataset: {str(e)}")
            return pd.DataFrame()
    
    async def get_prompt_evaluation_metrics(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate evaluation metrics for a prompt dataset.
        
        Args:
            dataset: DataFrame containing prompts
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            metrics = self.prompt_manager.get_prompt_quality_metrics(dataset)
            
            # Add additional metrics specific to AI tool prompts
            if not dataset.empty:
                # Calculate category diversity
                category_diversity = len(dataset['category'].unique()) / len(PromptCategory)
                
                # Calculate source diversity
                source_diversity = len(dataset['source'].unique())
                
                # Calculate length distribution
                prompt_lengths = dataset['input'].str.len()
                length_stats = {
                    "mean_length": prompt_lengths.mean(),
                    "median_length": prompt_lengths.median(),
                    "std_length": prompt_lengths.std(),
                    "min_length": prompt_lengths.min(),
                    "max_length": prompt_lengths.max()
                }
                
                metrics.update({
                    "category_diversity": category_diversity,
                    "source_diversity": source_diversity,
                    "length_statistics": length_stats
                })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating evaluation metrics: {str(e)}")
            return {"error": str(e)}
    
    async def generate_model_specific_dataset(
        self, 
        model_capabilities: Dict[str, Any],
        target_size: int = 1000
    ) -> pd.DataFrame:
        """
        Generate a model-specific evaluation dataset based on capabilities.
        
        Args:
            model_capabilities: Dictionary describing model capabilities
            target_size: Target number of prompts
            
        Returns:
            DataFrame containing model-specific evaluation dataset
        """
        try:
            # Generate dynamic dataset based on model capabilities
            dynamic_dataset = await self.prompt_manager.get_dynamic_evaluation_dataset(
                model_capabilities=model_capabilities,
                evaluation_goals=["accuracy", "robustness", "creativity"],
                target_size=target_size
            )
            
            # Save model-specific dataset
            model_name = model_capabilities.get("name", "unknown_model")
            safe_name = "".join(c for c in model_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            output_path = project_root / "data" / "evaluation_datasets" / f"{safe_name}_evaluation_dataset.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            dynamic_dataset.to_csv(output_path, index=False)
            
            self.logger.info(f"Generated model-specific dataset for {model_name} with {len(dynamic_dataset)} prompts")
            
            return dynamic_dataset
            
        except Exception as e:
            self.logger.error(f"Error generating model-specific dataset: {str(e)}")
            return pd.DataFrame()
    
    async def load_cached_prompts_summary(self) -> Dict[str, Any]:
        """
        Load and summarize cached AI tool prompts.
        
        Returns:
            Dictionary containing summary of cached prompts
        """
        try:
            summary = {
                "cache_directory": str(self.cache_dir),
                "cache_exists": self.cache_dir.exists(),
                "tools": {},
                "total_prompts": 0
            }
            
            if not self.cache_dir.exists():
                return summary
            
            # Scan cache directory for JSON files
            json_files = list(self.cache_dir.glob("*.json"))
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    tool_name = data.get('tool_name', json_file.stem)
                    prompt_count = data.get('prompt_count', 0)
                    cached_at = data.get('cached_at', 0)
                    
                    summary["tools"][tool_name] = {
                        "prompt_count": prompt_count,
                        "cached_at": cached_at,
                        "file_size": json_file.stat().st_size
                    }
                    
                    summary["total_prompts"] += prompt_count
                    
                except Exception as e:
                    self.logger.warning(f"Error reading {json_file}: {str(e)}")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error loading cached prompts summary: {str(e)}")
            return {"error": str(e)}
    
    async def export_prompts_for_mlflow(
        self, 
        experiment_name: str,
        prompts: List[PromptEntry]
    ) -> Dict[str, Any]:
        """
        Export prompts for MLflow experiment tracking.
        
        Args:
            experiment_name: Name of the MLflow experiment
            prompts: List of prompts to export
            
        Returns:
            Dictionary containing export results
        """
        try:
            # Convert prompts to MLflow-compatible format
            mlflow_data = []
            for prompt in prompts:
                mlflow_data.append({
                    "prompt_id": prompt.id,
                    "prompt_text": prompt.text,
                    "category": prompt.category.value,
                    "source": prompt.source,
                    "quality_score": prompt.quality_score,
                    "difficulty_level": prompt.difficulty_level,
                    "metadata": json.dumps(prompt.metadata)
                })
            
            # Save to temporary file for MLflow
            temp_path = project_root / "temp" / f"{experiment_name}_prompts.json"
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(mlflow_data, f, indent=2, ensure_ascii=False)
            
            return {
                "success": True,
                "prompts_exported": len(prompts),
                "export_path": str(temp_path),
                "experiment_name": experiment_name
            }
            
        except Exception as e:
            self.logger.error(f"Error exporting prompts for MLflow: {str(e)}")
            return {"success": False, "error": str(e)}


# Example usage and testing
async def main():
    """Example usage of the PromptIntegrationManager."""
    try:
        # Initialize manager
        manager = PromptIntegrationManager()
        
        # Load cached prompts summary
        summary = await manager.load_cached_prompts_summary()
        print(f"Cached prompts summary: {json.dumps(summary, indent=2)}")
        
        # Sync AI tool prompts
        sync_result = await manager.sync_ai_tool_prompts()
        print(f"Sync result: {json.dumps(sync_result, indent=2)}")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
