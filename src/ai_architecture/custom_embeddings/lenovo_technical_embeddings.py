"""
Lenovo Technical Embeddings

This module provides custom embedding training for Lenovo technical documentation,
engineering specifications, and technical knowledge.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from sentence_transformers import SentenceTransformer, InputExample, losses
from datasets import Dataset
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class LenovoTechnicalEmbeddings:
    """
    Lenovo technical embeddings trainer.
    
    Provides custom embedding training for Lenovo technical documentation,
    engineering specifications, and technical knowledge.
    """
    
    def __init__(self, 
                 base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 embedding_dim: int = 384):
        """
        Initialize Lenovo technical embeddings trainer.
        
        Args:
            base_model_name: Base sentence transformer model
            embedding_dim: Dimension of embeddings
        """
        self.base_model_name = base_model_name
        self.embedding_dim = embedding_dim
        self.model = None
        self.tokenizer = None
        
    def load_base_model(self) -> None:
        """Load the base sentence transformer model."""
        try:
            self.model = SentenceTransformer(self.base_model_name)
            logger.info(f"Loaded base model: {self.base_model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load base model: {e}")
            raise
    
    def prepare_technical_data(self, data_path: str) -> List[InputExample]:
        """
        Prepare Lenovo technical data for embedding training.
        
        Args:
            data_path: Path to technical data
            
        Returns:
            List of InputExample objects
        """
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            examples = []
            
            for item in data:
                # Create positive pairs (similar technical content)
                if 'positive_pairs' in item:
                    for pair in item['positive_pairs']:
                        examples.append(InputExample(
                            texts=[pair['text1'], pair['text2']],
                            label=1.0
                        ))
                
                # Create negative pairs (dissimilar technical content)
                if 'negative_pairs' in item:
                    for pair in item['negative_pairs']:
                        examples.append(InputExample(
                            texts=[pair['text1'], pair['text2']],
                            label=0.0
                        ))
                
                # Create technical document pairs
                if 'technical_docs' in item:
                    for doc in item['technical_docs']:
                        # Create pairs with similar technical concepts
                        if 'similar_concepts' in doc:
                            for concept in doc['similar_concepts']:
                                examples.append(InputExample(
                                    texts=[doc['content'], concept],
                                    label=0.8
                                ))
            
            logger.info(f"Prepared {len(examples)} technical examples")
            return examples
            
        except Exception as e:
            logger.error(f"Failed to prepare technical data: {e}")
            raise
    
    def train_embeddings(self, 
                        training_data: List[InputExample],
                        output_dir: str,
                        num_epochs: int = 3,
                        batch_size: int = 16) -> str:
        """
        Train custom technical embeddings.
        
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
            
            logger.info(f"Trained technical embeddings saved to {output_dir}")
            return output_dir
            
        except Exception as e:
            logger.error(f"Failed to train technical embeddings: {e}")
            raise
    
    def create_technical_embeddings(self, 
                                   texts: List[str],
                                   batch_size: int = 32) -> np.ndarray:
        """
        Create embeddings for technical texts.
        
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
            
            logger.info(f"Created embeddings for {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to create technical embeddings: {e}")
            raise
    
    def evaluate_embeddings(self, 
                           test_data: List[Dict],
                           similarity_threshold: float = 0.7) -> Dict[str, float]:
        """
        Evaluate embedding quality.
        
        Args:
            test_data: Test data with ground truth
            similarity_threshold: Threshold for similarity
            
        Returns:
            Evaluation metrics
        """
        try:
            correct_predictions = 0
            total_predictions = 0
            
            for item in test_data:
                text1 = item['text1']
                text2 = item['text2']
                expected_similarity = item['similarity']
                
                # Get embeddings
                emb1 = self.model.encode([text1])
                emb2 = self.model.encode([text2])
                
                # Calculate cosine similarity
                similarity = np.dot(emb1[0], emb2[0]) / (
                    np.linalg.norm(emb1[0]) * np.linalg.norm(emb2[0])
                )
                
                # Check if prediction is correct
                predicted_similar = similarity >= similarity_threshold
                expected_similar = expected_similarity >= similarity_threshold
                
                if predicted_similar == expected_similar:
                    correct_predictions += 1
                
                total_predictions += 1
            
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            
            metrics = {
                'accuracy': accuracy,
                'correct_predictions': correct_predictions,
                'total_predictions': total_predictions,
                'similarity_threshold': similarity_threshold
            }
            
            logger.info(f"Embedding evaluation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to evaluate embeddings: {e}")
            raise
    
    def create_technical_knowledge_base(self, 
                                      technical_docs: List[Dict],
                                      output_path: str) -> str:
        """
        Create technical knowledge base with embeddings.
        
        Args:
            technical_docs: Technical documents
            output_path: Output path for knowledge base
            
        Returns:
            Path to knowledge base
        """
        try:
            knowledge_base = {
                'documents': [],
                'embeddings': [],
                'metadata': []
            }
            
            for doc in technical_docs:
                # Extract text content
                content = doc.get('content', '')
                title = doc.get('title', '')
                category = doc.get('category', 'technical')
                
                # Create embedding
                embedding = self.model.encode([content])[0]
                
                # Store in knowledge base
                knowledge_base['documents'].append({
                    'title': title,
                    'content': content,
                    'category': category,
                    'doc_id': doc.get('id', len(knowledge_base['documents']))
                })
                
                knowledge_base['embeddings'].append(embedding.tolist())
                knowledge_base['metadata'].append({
                    'category': category,
                    'title': title,
                    'doc_id': doc.get('id', len(knowledge_base['documents']) - 1)
                })
            
            # Save knowledge base
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Created technical knowledge base with {len(technical_docs)} documents")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create technical knowledge base: {e}")
            raise
    
    def search_technical_knowledge(self, 
                                 query: str,
                                 knowledge_base_path: str,
                                 top_k: int = 5) -> List[Dict]:
        """
        Search technical knowledge base.
        
        Args:
            query: Search query
            knowledge_base_path: Path to knowledge base
            top_k: Number of top results
            
        Returns:
            List of relevant documents
        """
        try:
            # Load knowledge base
            with open(knowledge_base_path, 'r', encoding='utf-8') as f:
                knowledge_base = json.load(f)
            
            # Get query embedding
            query_embedding = self.model.encode([query])[0]
            
            # Calculate similarities
            similarities = []
            for i, doc_embedding in enumerate(knowledge_base['embeddings']):
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                similarities.append((i, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get top results
            results = []
            for i, (doc_idx, similarity) in enumerate(similarities[:top_k]):
                doc = knowledge_base['documents'][doc_idx]
                metadata = knowledge_base['metadata'][doc_idx]
                
                results.append({
                    'rank': i + 1,
                    'similarity': float(similarity),
                    'title': doc['title'],
                    'content': doc['content'][:200] + '...' if len(doc['content']) > 200 else doc['content'],
                    'category': metadata['category'],
                    'doc_id': metadata['doc_id']
                })
            
            logger.info(f"Found {len(results)} relevant documents for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search technical knowledge: {e}")
            raise
    
    def get_embedding_statistics(self) -> Dict[str, Union[int, float]]:
        """
        Get embedding model statistics.
        
        Returns:
            Model statistics
        """
        try:
            if self.model is None:
                return {"error": "Model not loaded"}
            
            # Get model information
            model_info = {
                'model_name': self.base_model_name,
                'embedding_dimension': self.embedding_dim,
                'max_sequence_length': self.model.max_seq_length,
                'vocab_size': len(self.model.tokenizer.get_vocab()) if hasattr(self.model, 'tokenizer') else 0
            }
            
            logger.info(f"Embedding model statistics: {model_info}")
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to get embedding statistics: {e}")
            raise
