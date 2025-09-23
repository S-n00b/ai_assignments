"""
Business Process Embeddings

This module provides custom embedding training for Lenovo business processes,
workflows, and organizational knowledge.
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

class BusinessProcessEmbeddings:
    """
    Business process embeddings trainer.
    
    Provides custom embedding training for Lenovo business processes,
    workflows, and organizational knowledge.
    """
    
    def __init__(self, 
                 base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 embedding_dim: int = 384):
        """
        Initialize business process embeddings trainer.
        
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
    
    def prepare_business_process_data(self, data_path: str) -> List[InputExample]:
        """
        Prepare business process data for embedding training.
        
        Args:
            data_path: Path to business process data
            
        Returns:
            List of InputExample objects
        """
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            examples = []
            
            for item in data:
                # Create positive pairs (similar business processes)
                if 'business_processes' in item:
                    for process in item['business_processes']:
                        if 'similar_processes' in process:
                            for similar_process in process['similar_processes']:
                                examples.append(InputExample(
                                    texts=[process['description'], similar_process],
                                    label=0.9
                                ))
                
                # Create workflow pairs
                if 'workflows' in item:
                    for workflow in item['workflows']:
                        if 'related_workflows' in workflow:
                            for related_workflow in workflow['related_workflows']:
                                examples.append(InputExample(
                                    texts=[workflow['description'], related_workflow],
                                    label=0.8
                                ))
                
                # Create organizational knowledge pairs
                if 'organizational_knowledge' in item:
                    for knowledge in item['organizational_knowledge']:
                        if 'related_knowledge' in knowledge:
                            for related_knowledge in knowledge['related_knowledge']:
                                examples.append(InputExample(
                                    texts=[knowledge['description'], related_knowledge],
                                    label=0.7
                                ))
            
            logger.info(f"Prepared {len(examples)} business process examples")
            return examples
            
        except Exception as e:
            logger.error(f"Failed to prepare business process data: {e}")
            raise
    
    def train_embeddings(self, 
                        training_data: List[InputExample],
                        output_dir: str,
                        num_epochs: int = 3,
                        batch_size: int = 16) -> str:
        """
        Train custom business process embeddings.
        
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
            
            logger.info(f"Trained business process embeddings saved to {output_dir}")
            return output_dir
            
        except Exception as e:
            logger.error(f"Failed to train business process embeddings: {e}")
            raise
    
    def create_business_process_embeddings(self, 
                                          texts: List[str],
                                          batch_size: int = 32) -> np.ndarray:
        """
        Create embeddings for business process texts.
        
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
            
            logger.info(f"Created embeddings for {len(texts)} business process texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to create business process embeddings: {e}")
            raise
    
    def create_business_knowledge_base(self, 
                                     business_data: List[Dict],
                                     output_path: str) -> str:
        """
        Create business process knowledge base with embeddings.
        
        Args:
            business_data: Business process data
            output_path: Output path for knowledge base
            
        Returns:
            Path to knowledge base
        """
        try:
            knowledge_base = {
                'processes': [],
                'embeddings': [],
                'metadata': []
            }
            
            for process in business_data:
                # Extract process information
                process_name = process.get('name', '')
                process_type = process.get('type', '')
                description = process.get('description', '')
                workflow_steps = process.get('workflow_steps', [])
                
                # Create process description
                process_description = f"{process_name} {process_type} {description}"
                
                # Create embedding
                embedding = self.model.encode([process_description])[0]
                
                # Store in knowledge base
                knowledge_base['processes'].append({
                    'name': process_name,
                    'type': process_type,
                    'description': description,
                    'workflow_steps': workflow_steps,
                    'process_id': process.get('id', len(knowledge_base['processes']))
                })
                
                knowledge_base['embeddings'].append(embedding.tolist())
                knowledge_base['metadata'].append({
                    'process_name': process_name,
                    'process_type': process_type,
                    'process_id': process.get('id', len(knowledge_base['processes']) - 1)
                })
            
            # Save knowledge base
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Created business knowledge base with {len(business_data)} processes")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create business knowledge base: {e}")
            raise
    
    def search_business_processes(self, 
                                query: str,
                                knowledge_base_path: str,
                                top_k: int = 5) -> List[Dict]:
        """
        Search business process knowledge base.
        
        Args:
            query: Search query
            knowledge_base_path: Path to knowledge base
            top_k: Number of top results
            
        Returns:
            List of relevant processes
        """
        try:
            # Load knowledge base
            with open(knowledge_base_path, 'r', encoding='utf-8') as f:
                knowledge_base = json.load(f)
            
            # Get query embedding
            query_embedding = self.model.encode([query])[0]
            
            # Calculate similarities
            similarities = []
            for i, process_embedding in enumerate(knowledge_base['embeddings']):
                similarity = np.dot(query_embedding, process_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(process_embedding)
                )
                similarities.append((i, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get top results
            results = []
            for i, (process_idx, similarity) in enumerate(similarities[:top_k]):
                process = knowledge_base['processes'][process_idx]
                metadata = knowledge_base['metadata'][process_idx]
                
                results.append({
                    'rank': i + 1,
                    'similarity': float(similarity),
                    'process_name': process['name'],
                    'process_type': process['type'],
                    'description': process['description'],
                    'workflow_steps': process['workflow_steps'],
                    'process_id': metadata['process_id']
                })
            
            logger.info(f"Found {len(results)} relevant processes for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search business processes: {e}")
            raise
    
    def create_workflow_embeddings(self, 
                                 workflow_data: List[Dict],
                                 output_path: str) -> str:
        """
        Create workflow embeddings.
        
        Args:
            workflow_data: Workflow data
            output_path: Output path for embeddings
            
        Returns:
            Path to embeddings
        """
        try:
            workflow_embeddings = {
                'workflows': [],
                'embeddings': [],
                'metadata': []
            }
            
            for item in workflow_data:
                workflow_name = item.get('name', '')
                description = item.get('description', '')
                steps = item.get('steps', [])
                
                # Create workflow description
                workflow_description = f"{workflow_name} {description} {' '.join(steps)}"
                
                # Create embedding
                embedding = self.model.encode([workflow_description])[0]
                
                # Store in embeddings
                workflow_embeddings['workflows'].append(workflow_description)
                workflow_embeddings['embeddings'].append(embedding.tolist())
                workflow_embeddings['metadata'].append({
                    'name': workflow_name,
                    'description': description,
                    'steps': steps
                })
            
            # Save embeddings
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(workflow_embeddings, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Created workflow embeddings for {len(workflow_data)} workflows")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create workflow embeddings: {e}")
            raise
    
    def find_similar_workflows(self, 
                             workflow_description: str,
                             workflow_embeddings_path: str,
                             top_k: int = 3) -> List[Dict]:
        """
        Find similar workflows.
        
        Args:
            workflow_description: Description of the workflow
            workflow_embeddings_path: Path to workflow embeddings
            top_k: Number of top results
            
        Returns:
            List of similar workflows
        """
        try:
            # Load workflow embeddings
            with open(workflow_embeddings_path, 'r', encoding='utf-8') as f:
                workflow_embeddings = json.load(f)
            
            # Get workflow embedding
            workflow_embedding = self.model.encode([workflow_description])[0]
            
            # Calculate similarities
            similarities = []
            for i, embedding in enumerate(workflow_embeddings['embeddings']):
                similarity = np.dot(workflow_embedding, embedding) / (
                    np.linalg.norm(workflow_embedding) * np.linalg.norm(embedding)
                )
                similarities.append((i, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get top results
            results = []
            for i, (idx, similarity) in enumerate(similarities[:top_k]):
                metadata = workflow_embeddings['metadata'][idx]
                
                results.append({
                    'rank': i + 1,
                    'similarity': float(similarity),
                    'workflow': workflow_embeddings['workflows'][idx],
                    'name': metadata['name'],
                    'description': metadata['description'],
                    'steps': metadata['steps']
                })
            
            logger.info(f"Found {len(results)} similar workflows for: {workflow_description}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to find similar workflows: {e}")
            raise
