"""
Device Support Embeddings

This module provides custom embedding training for Lenovo device support knowledge,
troubleshooting guides, and technical support scenarios.
"""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, InputExample, losses
from datasets import Dataset
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class DeviceSupportEmbeddings:
    """
    Device support embeddings trainer.
    
    Provides custom embedding training for Lenovo device support knowledge,
    troubleshooting guides, and technical support scenarios.
    """
    
    def __init__(self, 
                 base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 embedding_dim: int = 384):
        """
        Initialize device support embeddings trainer.
        
        Args:
            base_model_name: Base sentence transformer model
            embedding_dim: Dimension of embeddings
        """
        self.base_model_name = base_model_name
        self.embedding_dim = embedding_dim
        self.model = None
        
    def load_base_model(self) -> None:
        """Load the base sentence transformer model."""
        try:
            self.model = SentenceTransformer(self.base_model_name)
            logger.info(f"Loaded base model: {self.base_model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load base model: {e}")
            raise
    
    def prepare_device_support_data(self, data_path: str) -> List[InputExample]:
        """
        Prepare device support data for embedding training.
        
        Args:
            data_path: Path to device support data
            
        Returns:
            List of InputExample objects
        """
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            examples = []
            
            for item in data:
                # Create positive pairs (similar device support scenarios)
                if 'support_scenarios' in item:
                    for scenario in item['support_scenarios']:
                        if 'similar_issues' in scenario:
                            for similar_issue in scenario['similar_issues']:
                                examples.append(InputExample(
                                    texts=[scenario['description'], similar_issue],
                                    label=0.9
                                ))
                
                # Create troubleshooting pairs
                if 'troubleshooting_guides' in item:
                    for guide in item['troubleshooting_guides']:
                        if 'related_solutions' in guide:
                            for solution in guide['related_solutions']:
                                examples.append(InputExample(
                                    texts=[guide['problem'], solution],
                                    label=0.8
                                ))
                
                # Create device-specific pairs
                if 'device_issues' in item:
                    for issue in item['device_issues']:
                        if 'device_type' in issue and 'similar_devices' in issue:
                            for similar_device in issue['similar_devices']:
                                examples.append(InputExample(
                                    texts=[issue['description'], similar_device['description']],
                                    label=0.7
                                ))
            
            logger.info(f"Prepared {len(examples)} device support examples")
            return examples
            
        except Exception as e:
            logger.error(f"Failed to prepare device support data: {e}")
            raise
    
    def train_embeddings(self, 
                        training_data: List[InputExample],
                        output_dir: str,
                        num_epochs: int = 3,
                        batch_size: int = 16) -> str:
        """
        Train custom device support embeddings.
        
        Args:
            training_data: Training data
            output_dir: Output directory
            num_epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Path to trained model
        """
        try:
            # Create training dataset
            train_dataset = Dataset.from_list([
                {
                    'text1': ex.texts[0],
                    'text2': ex.texts[1],
                    'label': ex.label
                }
                for ex in training_data
            ])
            
            # Define loss function
            train_loss = losses.CosineSimilarityLoss(self.model)
            
            # Training arguments
            training_args = {
                'output_path': output_dir,
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'warmup_steps': 100,
                'evaluation_steps': 500,
                'save_steps': 500,
                'save_total_limit': 2,
                'load_best_model_at_end': True,
                'metric_for_best_model': 'eval_loss',
                'greater_is_better': False
            }
            
            # Train model
            self.model.fit(
                train_objectives=[(train_dataset, train_loss)],
                **training_args
            )
            
            # Save model
            self.model.save(output_dir)
            
            logger.info(f"Trained device support embeddings saved to {output_dir}")
            return output_dir
            
        except Exception as e:
            logger.error(f"Failed to train device support embeddings: {e}")
            raise
    
    def create_device_support_embeddings(self, 
                                        texts: List[str],
                                        batch_size: int = 32) -> np.ndarray:
        """
        Create embeddings for device support texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings
        """
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            logger.info(f"Created embeddings for {len(texts)} device support texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to create device support embeddings: {e}")
            raise
    
    def create_device_knowledge_base(self, 
                                    device_data: List[Dict],
                                    output_path: str) -> str:
        """
        Create device support knowledge base with embeddings.
        
        Args:
            device_data: Device support data
            output_path: Output path for knowledge base
            
        Returns:
            Path to knowledge base
        """
        try:
            knowledge_base = {
                'devices': [],
                'embeddings': [],
                'metadata': []
            }
            
            for device in device_data:
                # Extract device information
                device_name = device.get('name', '')
                device_type = device.get('type', '')
                support_info = device.get('support_info', '')
                common_issues = device.get('common_issues', [])
                
                # Create device description
                device_description = f"{device_name} {device_type} {support_info}"
                
                # Create embedding
                embedding = self.model.encode([device_description])[0]
                
                # Store in knowledge base
                knowledge_base['devices'].append({
                    'name': device_name,
                    'type': device_type,
                    'support_info': support_info,
                    'common_issues': common_issues,
                    'device_id': device.get('id', len(knowledge_base['devices']))
                })
                
                knowledge_base['embeddings'].append(embedding.tolist())
                knowledge_base['metadata'].append({
                    'device_name': device_name,
                    'device_type': device_type,
                    'device_id': device.get('id', len(knowledge_base['devices']) - 1)
                })
            
            # Save knowledge base
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Created device knowledge base with {len(device_data)} devices")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create device knowledge base: {e}")
            raise
    
    def search_device_support(self, 
                            query: str,
                            knowledge_base_path: str,
                            top_k: int = 5) -> List[Dict]:
        """
        Search device support knowledge base.
        
        Args:
            query: Search query
            knowledge_base_path: Path to knowledge base
            top_k: Number of top results
            
        Returns:
            List of relevant devices
        """
        try:
            # Load knowledge base
            with open(knowledge_base_path, 'r', encoding='utf-8') as f:
                knowledge_base = json.load(f)
            
            # Get query embedding
            query_embedding = self.model.encode([query])[0]
            
            # Calculate similarities
            similarities = []
            for i, device_embedding in enumerate(knowledge_base['embeddings']):
                similarity = np.dot(query_embedding, device_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(device_embedding)
                )
                similarities.append((i, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get top results
            results = []
            for i, (device_idx, similarity) in enumerate(similarities[:top_k]):
                device = knowledge_base['devices'][device_idx]
                metadata = knowledge_base['metadata'][device_idx]
                
                results.append({
                    'rank': i + 1,
                    'similarity': float(similarity),
                    'device_name': device['name'],
                    'device_type': device['type'],
                    'support_info': device['support_info'],
                    'common_issues': device['common_issues'],
                    'device_id': metadata['device_id']
                })
            
            logger.info(f"Found {len(results)} relevant devices for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search device support: {e}")
            raise
    
    def create_troubleshooting_embeddings(self, 
                                        troubleshooting_data: List[Dict],
                                        output_path: str) -> str:
        """
        Create troubleshooting embeddings.
        
        Args:
            troubleshooting_data: Troubleshooting data
            output_path: Output path for embeddings
            
        Returns:
            Path to embeddings
        """
        try:
            troubleshooting_embeddings = {
                'problems': [],
                'solutions': [],
                'embeddings': [],
                'metadata': []
            }
            
            for item in troubleshooting_data:
                problem = item.get('problem', '')
                solution = item.get('solution', '')
                category = item.get('category', 'general')
                
                # Create problem-solution pair
                problem_solution = f"{problem} {solution}"
                
                # Create embedding
                embedding = self.model.encode([problem_solution])[0]
                
                # Store in embeddings
                troubleshooting_embeddings['problems'].append(problem)
                troubleshooting_embeddings['solutions'].append(solution)
                troubleshooting_embeddings['embeddings'].append(embedding.tolist())
                troubleshooting_embeddings['metadata'].append({
                    'category': category,
                    'problem': problem,
                    'solution': solution
                })
            
            # Save embeddings
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(troubleshooting_embeddings, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Created troubleshooting embeddings for {len(troubleshooting_data)} items")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create troubleshooting embeddings: {e}")
            raise
    
    def find_similar_issues(self, 
                          issue_description: str,
                          troubleshooting_embeddings_path: str,
                          top_k: int = 3) -> List[Dict]:
        """
        Find similar issues in troubleshooting database.
        
        Args:
            issue_description: Description of the issue
            troubleshooting_embeddings_path: Path to troubleshooting embeddings
            top_k: Number of top results
            
        Returns:
            List of similar issues
        """
        try:
            # Load troubleshooting embeddings
            with open(troubleshooting_embeddings_path, 'r', encoding='utf-8') as f:
                troubleshooting_embeddings = json.load(f)
            
            # Get issue embedding
            issue_embedding = self.model.encode([issue_description])[0]
            
            # Calculate similarities
            similarities = []
            for i, embedding in enumerate(troubleshooting_embeddings['embeddings']):
                similarity = np.dot(issue_embedding, embedding) / (
                    np.linalg.norm(issue_embedding) * np.linalg.norm(embedding)
                )
                similarities.append((i, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get top results
            results = []
            for i, (idx, similarity) in enumerate(similarities[:top_k]):
                metadata = troubleshooting_embeddings['metadata'][idx]
                
                results.append({
                    'rank': i + 1,
                    'similarity': float(similarity),
                    'problem': troubleshooting_embeddings['problems'][idx],
                    'solution': troubleshooting_embeddings['solutions'][idx],
                    'category': metadata['category']
                })
            
            logger.info(f"Found {len(results)} similar issues for: {issue_description}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to find similar issues: {e}")
            raise
