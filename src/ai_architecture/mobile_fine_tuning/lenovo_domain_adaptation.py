"""
Lenovo Domain Adaptation for Small Models

This module provides domain-specific fine-tuning for Lenovo use cases including
device support, technical documentation, and business processes.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class LenovoDomainAdapter:
    """
    Lenovo-specific domain adaptation for small models.
    
    Provides fine-tuning capabilities for Lenovo device support, technical documentation,
    and business process scenarios.
    """
    
    def __init__(self, model_name: str, base_model_path: Optional[str] = None):
        """
        Initialize Lenovo domain adapter.
        
        Args:
            model_name: Name of the base model to adapt
            base_model_path: Optional path to pre-trained model
        """
        self.model_name = model_name
        self.base_model_path = base_model_path
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self) -> None:
        """Load the base model and tokenizer."""
        try:
            model_path = self.base_model_path or self.model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info(f"Loaded model {self.model_name} on device {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def prepare_lenovo_dataset(self, data_path: str) -> Dataset:
        """
        Prepare Lenovo-specific training dataset.
        
        Args:
            data_path: Path to Lenovo training data
            
        Returns:
            Prepared dataset for fine-tuning
        """
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Format data for training
            formatted_data = []
            for item in data:
                # Create instruction-following format
                instruction = item.get('instruction', '')
                context = item.get('context', '')
                response = item.get('response', '')
                
                # Format as conversation
                formatted_text = f"<|system|>You are a Lenovo technical support assistant specializing in {item.get('category', 'device support')}.<|user|>{instruction}<|assistant|>{response}"
                
                formatted_data.append({
                    'text': formatted_text,
                    'category': item.get('category', 'general'),
                    'device_type': item.get('device_type', 'general')
                })
            
            # Create dataset
            dataset = Dataset.from_list(formatted_data)
            
            # Tokenize dataset
            def tokenize_function(examples):
                return self.tokenizer(
                    examples['text'],
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors="pt"
                )
            
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            logger.info(f"Prepared dataset with {len(tokenized_dataset)} examples")
            return tokenized_dataset
            
        except Exception as e:
            logger.error(f"Failed to prepare dataset: {e}")
            raise
    
    def create_training_arguments(self, output_dir: str, **kwargs) -> TrainingArguments:
        """
        Create training arguments optimized for mobile deployment.
        
        Args:
            output_dir: Directory to save model outputs
            **kwargs: Additional training arguments
            
        Returns:
            TrainingArguments object
        """
        default_args = {
            'output_dir': output_dir,
            'num_train_epochs': 3,
            'per_device_train_batch_size': 4,
            'per_device_eval_batch_size': 4,
            'warmup_steps': 100,
            'weight_decay': 0.01,
            'logging_dir': f"{output_dir}/logs",
            'logging_steps': 10,
            'save_steps': 500,
            'eval_steps': 500,
            'evaluation_strategy': "steps",
            'save_strategy': "steps",
            'load_best_model_at_end': True,
            'metric_for_best_model': "eval_loss",
            'greater_is_better': False,
            'report_to': "mlflow",
            'fp16': torch.cuda.is_available(),
            'dataloader_num_workers': 0,  # Reduce for mobile compatibility
            'remove_unused_columns': False,
            'gradient_accumulation_steps': 2,
            'learning_rate': 2e-5,
            'lr_scheduler_type': "cosine",
            'max_grad_norm': 1.0
        }
        
        # Update with provided arguments
        default_args.update(kwargs)
        
        return TrainingArguments(**default_args)
    
    def fine_tune_model(self, 
                       dataset: Dataset, 
                       output_dir: str,
                       lenovo_config: Optional[Dict] = None) -> str:
        """
        Fine-tune model for Lenovo domain.
        
        Args:
            dataset: Training dataset
            output_dir: Output directory for fine-tuned model
            lenovo_config: Lenovo-specific configuration
            
        Returns:
            Path to fine-tuned model
        """
        try:
            # Create training arguments
            training_args = self.create_training_arguments(
                output_dir=output_dir,
                lenovo_config=lenovo_config
            )
            
            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                eval_dataset=dataset,  # Use same dataset for eval in this example
                tokenizer=self.tokenizer,
                data_collator=self._data_collator
            )
            
            # Start training
            logger.info("Starting Lenovo domain fine-tuning...")
            trainer.train()
            
            # Save model
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            logger.info(f"Fine-tuned model saved to {output_dir}")
            return output_dir
            
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            raise
    
    def _data_collator(self, features):
        """Custom data collator for training."""
        batch = {}
        batch['input_ids'] = torch.stack([f['input_ids'] for f in features])
        batch['attention_mask'] = torch.stack([f['attention_mask'] for f in features])
        batch['labels'] = batch['input_ids'].clone()
        return batch
    
    def evaluate_model(self, test_dataset: Dataset) -> Dict[str, float]:
        """
        Evaluate fine-tuned model.
        
        Args:
            test_dataset: Test dataset for evaluation
            
        Returns:
            Evaluation metrics
        """
        try:
            trainer = Trainer(
                model=self.model,
                tokenizer=self.tokenizer,
                data_collator=self._data_collator
            )
            
            # Run evaluation
            eval_results = trainer.evaluate(test_dataset)
            
            logger.info(f"Evaluation results: {eval_results}")
            return eval_results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def create_lenovo_prompts(self) -> Dict[str, List[str]]:
        """
        Create Lenovo-specific prompt templates.
        
        Returns:
            Dictionary of prompt categories and templates
        """
        return {
            "device_support": [
                "How do I troubleshoot {device} connectivity issues?",
                "What are the specifications for {device}?",
                "How do I update firmware on {device}?",
                "What are common issues with {device} and how to resolve them?"
            ],
            "technical_documentation": [
                "Explain the technical architecture of {system}",
                "What are the API endpoints for {service}?",
                "How do I integrate {component} with {system}?",
                "What are the security considerations for {feature}?"
            ],
            "business_processes": [
                "What is the Lenovo {process} workflow?",
                "How do I escalate {issue} to the appropriate team?",
                "What are the approval requirements for {request}?",
                "How do I track the status of {process}?"
            ],
            "customer_service": [
                "How can I help with {customer_issue}?",
                "What are the available support options for {product}?",
                "How do I create a support ticket for {issue}?",
                "What is the expected resolution time for {problem}?"
            ]
        }
    
    def generate_lenovo_response(self, 
                               prompt: str, 
                               device_type: str = "general",
                               max_length: int = 256) -> str:
        """
        Generate Lenovo-specific response.
        
        Args:
            prompt: Input prompt
            device_type: Type of Lenovo device
            max_length: Maximum response length
            
        Returns:
            Generated response
        """
        try:
            # Format prompt with Lenovo context
            formatted_prompt = f"<|system|>You are a Lenovo technical support assistant specializing in {device_type} devices.<|user|>{prompt}<|assistant|>"
            
            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            if "<|assistant|>" in response:
                response = response.split("<|assistant|>")[-1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return f"Error generating response: {e}"
