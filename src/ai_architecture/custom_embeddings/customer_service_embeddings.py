"""
Customer Service Embeddings

This module provides custom embedding training for Lenovo customer service scenarios,
B2B client interactions, and customer support workflows.
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

class CustomerServiceEmbeddings:
    """
    Customer service embeddings trainer.
    
    Provides custom embedding training for Lenovo customer service scenarios,
    B2B client interactions, and customer support workflows.
    """
    
    def __init__(self, 
                 base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 embedding_dim: int = 384):
        """
        Initialize customer service embeddings trainer.
        
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
    
    def prepare_customer_service_data(self, data_path: str) -> List[InputExample]:
        """
        Prepare customer service data for embedding training.
        
        Args:
            data_path: Path to customer service data
            
        Returns:
            List of InputExample objects
        """
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            examples = []
            
            for item in data:
                # Create positive pairs (similar customer service scenarios)
                if 'service_scenarios' in item:
                    for scenario in item['service_scenarios']:
                        if 'similar_scenarios' in scenario:
                            for similar_scenario in scenario['similar_scenarios']:
                                examples.append(InputExample(
                                    texts=[scenario['description'], similar_scenario],
                                    label=0.9
                                ))
                
                # Create B2B client interaction pairs
                if 'b2b_interactions' in item:
                    for interaction in item['b2b_interactions']:
                        if 'related_interactions' in interaction:
                            for related_interaction in interaction['related_interactions']:
                                examples.append(InputExample(
                                    texts=[interaction['description'], related_interaction],
                                    label=0.8
                                ))
                
                # Create customer journey pairs
                if 'customer_journeys' in item:
                    for journey in item['customer_journeys']:
                        if 'similar_journeys' in journey:
                            for similar_journey in journey['similar_journeys']:
                                examples.append(InputExample(
                                    texts=[journey['description'], similar_journey],
                                    label=0.7
                                ))
            
            logger.info(f"Prepared {len(examples)} customer service examples")
            return examples
            
        except Exception as e:
            logger.error(f"Failed to prepare customer service data: {e}")
            raise
    
    def train_embeddings(self, 
                        training_data: List[InputExample],
                        output_dir: str,
                        num_epochs: int = 3,
                        batch_size: int = 16) -> str:
        """
        Train custom customer service embeddings.
        
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
            
            logger.info(f"Trained customer service embeddings saved to {output_dir}")
            return output_dir
            
        except Exception as e:
            logger.error(f"Failed to train customer service embeddings: {e}")
            raise
    
    def create_customer_service_embeddings(self, 
                                          texts: List[str],
                                          batch_size: int = 32) -> np.ndarray:
        """
        Create embeddings for customer service texts.
        
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
            
            logger.info(f"Created embeddings for {len(texts)} customer service texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to create customer service embeddings: {e}")
            raise
    
    def create_customer_knowledge_base(self, 
                                     customer_data: List[Dict],
                                     output_path: str) -> str:
        """
        Create customer service knowledge base with embeddings.
        
        Args:
            customer_data: Customer service data
            output_path: Output path for knowledge base
            
        Returns:
            Path to knowledge base
        """
        try:
            knowledge_base = {
                'customers': [],
                'embeddings': [],
                'metadata': []
            }
            
            for customer in customer_data:
                # Extract customer information
                customer_name = customer.get('name', '')
                customer_type = customer.get('type', '')
                service_info = customer.get('service_info', '')
                interaction_history = customer.get('interaction_history', [])
                
                # Create customer description
                customer_description = f"{customer_name} {customer_type} {service_info}"
                
                # Create embedding
                embedding = self.model.encode([customer_description])[0]
                
                # Store in knowledge base
                knowledge_base['customers'].append({
                    'name': customer_name,
                    'type': customer_type,
                    'service_info': service_info,
                    'interaction_history': interaction_history,
                    'customer_id': customer.get('id', len(knowledge_base['customers']))
                })
                
                knowledge_base['embeddings'].append(embedding.tolist())
                knowledge_base['metadata'].append({
                    'customer_name': customer_name,
                    'customer_type': customer_type,
                    'customer_id': customer.get('id', len(knowledge_base['customers']) - 1)
                })
            
            # Save knowledge base
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Created customer knowledge base with {len(customer_data)} customers")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create customer knowledge base: {e}")
            raise
    
    def search_customer_service(self, 
                               query: str,
                               knowledge_base_path: str,
                               top_k: int = 5) -> List[Dict]:
        """
        Search customer service knowledge base.
        
        Args:
            query: Search query
            knowledge_base_path: Path to knowledge base
            top_k: Number of top results
            
        Returns:
            List of relevant customers
        """
        try:
            # Load knowledge base
            with open(knowledge_base_path, 'r', encoding='utf-8') as f:
                knowledge_base = json.load(f)
            
            # Get query embedding
            query_embedding = self.model.encode([query])[0]
            
            # Calculate similarities
            similarities = []
            for i, customer_embedding in enumerate(knowledge_base['embeddings']):
                similarity = np.dot(query_embedding, customer_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(customer_embedding)
                )
                similarities.append((i, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get top results
            results = []
            for i, (customer_idx, similarity) in enumerate(similarities[:top_k]):
                customer = knowledge_base['customers'][customer_idx]
                metadata = knowledge_base['metadata'][customer_idx]
                
                results.append({
                    'rank': i + 1,
                    'similarity': float(similarity),
                    'customer_name': customer['name'],
                    'customer_type': customer['type'],
                    'service_info': customer['service_info'],
                    'interaction_history': customer['interaction_history'],
                    'customer_id': metadata['customer_id']
                })
            
            logger.info(f"Found {len(results)} relevant customers for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search customer service: {e}")
            raise
    
    def create_b2b_interaction_embeddings(self, 
                                        b2b_data: List[Dict],
                                        output_path: str) -> str:
        """
        Create B2B interaction embeddings.
        
        Args:
            b2b_data: B2B interaction data
            output_path: Output path for embeddings
            
        Returns:
            Path to embeddings
        """
        try:
            b2b_embeddings = {
                'interactions': [],
                'embeddings': [],
                'metadata': []
            }
            
            for item in b2b_data:
                interaction_type = item.get('type', '')
                description = item.get('description', '')
                client_info = item.get('client_info', '')
                
                # Create interaction description
                interaction_description = f"{interaction_type} {description} {client_info}"
                
                # Create embedding
                embedding = self.model.encode([interaction_description])[0]
                
                # Store in embeddings
                b2b_embeddings['interactions'].append(interaction_description)
                b2b_embeddings['embeddings'].append(embedding.tolist())
                b2b_embeddings['metadata'].append({
                    'type': interaction_type,
                    'description': description,
                    'client_info': client_info
                })
            
            # Save embeddings
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(b2b_embeddings, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Created B2B interaction embeddings for {len(b2b_data)} interactions")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create B2B interaction embeddings: {e}")
            raise
    
    def find_similar_interactions(self, 
                                interaction_description: str,
                                b2b_embeddings_path: str,
                                top_k: int = 3) -> List[Dict]:
        """
        Find similar B2B interactions.
        
        Args:
            interaction_description: Description of the interaction
            b2b_embeddings_path: Path to B2B embeddings
            top_k: Number of top results
            
        Returns:
            List of similar interactions
        """
        try:
            # Load B2B embeddings
            with open(b2b_embeddings_path, 'r', encoding='utf-8') as f:
                b2b_embeddings = json.load(f)
            
            # Get interaction embedding
            interaction_embedding = self.model.encode([interaction_description])[0]
            
            # Calculate similarities
            similarities = []
            for i, embedding in enumerate(b2b_embeddings['embeddings']):
                similarity = np.dot(interaction_embedding, embedding) / (
                    np.linalg.norm(interaction_embedding) * np.linalg.norm(embedding)
                )
                similarities.append((i, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get top results
            results = []
            for i, (idx, similarity) in enumerate(similarities[:top_k]):
                metadata = b2b_embeddings['metadata'][idx]
                
                results.append({
                    'rank': i + 1,
                    'similarity': float(similarity),
                    'interaction': b2b_embeddings['interactions'][idx],
                    'type': metadata['type'],
                    'description': metadata['description'],
                    'client_info': metadata['client_info']
                })
            
            logger.info(f"Found {len(results)} similar interactions for: {interaction_description}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to find similar interactions: {e}")
            raise
