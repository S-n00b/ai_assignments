#!/usr/bin/env python3
"""
Generate Enhanced Evaluation Dataset Script

This script generates an enhanced evaluation dataset using the prompt registries
and AI tool prompts for comprehensive model evaluation.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model_evaluation.prompt_registries import PromptRegistryManager, PromptCategory
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Generate enhanced evaluation dataset."""
    try:
        logger.info("Initializing PromptRegistryManager...")
        
        # Initialize prompt registry manager
        manager = PromptRegistryManager(
            enable_caching=True, 
            cache_dir=str(project_root / "cache" / "ai_tool_prompts")
        )
        
        # Load AI tool prompts
        logger.info("Loading AI tool prompts...")
        ai_tool_prompts = await manager.load_ai_tool_system_prompts()
        logger.info(f"Loaded {len(ai_tool_prompts)} AI tool prompts")
        
        # Generate enhanced evaluation dataset
        logger.info("Generating enhanced evaluation dataset...")
        dataset = manager.get_enhanced_evaluation_dataset(
            target_size=5000,
            categories=[
                PromptCategory.CODE_GENERATION,
                PromptCategory.CREATIVE_WRITING,
                PromptCategory.TECHNICAL_DOCUMENTATION,
                PromptCategory.REASONING,
                PromptCategory.QUESTION_ANSWERING
            ],
            quality_threshold=0.6
        )
        
        logger.info(f"Generated dataset with {len(dataset)} prompts")
        logger.info(f"Category distribution: {dataset['category'].value_counts().to_dict()}")
        
        # Save dataset
        output_path = project_root / "data" / "evaluation_datasets" / "enhanced_evaluation_dataset.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(output_path, index=False)
        logger.info(f"Dataset saved to: {output_path}")
        
        # Get quality metrics
        metrics = manager.get_prompt_quality_metrics(dataset)
        logger.info(f"Quality metrics: {metrics}")
        
        # Also create a summary report
        summary_path = project_root / "data" / "evaluation_datasets" / "dataset_summary.json"
        import json
        with open(summary_path, 'w') as f:
            json.dump({
                "dataset_size": len(dataset),
                "category_distribution": dataset['category'].value_counts().to_dict(),
                "quality_metrics": metrics,
                "ai_tool_prompts_loaded": len(ai_tool_prompts),
                "generation_timestamp": pd.Timestamp.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"Summary saved to: {summary_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating evaluation dataset: {str(e)}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
