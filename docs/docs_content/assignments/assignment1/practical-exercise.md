# Practical Exercise

## Overview

This practical exercise guides you through implementing a complete model evaluation pipeline using the framework components. You'll build a sentiment analysis system with comprehensive evaluation, profiling, and deployment capabilities.

## Exercise Objectives

By the end of this exercise, you will have:

1. **Created a sentiment classification model** using BERT
2. **Implemented comprehensive evaluation** including bias detection and robustness testing
3. **Performed detailed model profiling** to understand performance characteristics
4. **Deployed the model** using the Model Factory system
5. **Generated evaluation reports** with actionable insights

## Prerequisites

- Python 3.8+
- Basic understanding of machine learning concepts
- Familiarity with PyTorch and Transformers library
- Access to a GPU (recommended but not required)

## Setup

### 1. Environment Setup

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install additional dependencies
pip install transformers torch datasets evaluate
pip install matplotlib seaborn plotly
```

### 2. Dataset Preparation

```python
# Download and prepare dataset
from datasets import load_dataset

# Load IMDB dataset for sentiment analysis
dataset = load_dataset("imdb")
train_data = dataset["train"]
test_data = dataset["test"]

print(f"Training samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")
```

## Step 1: Model Creation

### 1.1 Create Model Configuration

```python
# config/model_config.py
from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_name: str = "bert-base-uncased"
    num_classes: int = 2
    max_length: int = 512
    dropout: float = 0.1
    learning_rate: float = 2e-5
    batch_size: int = 16
    epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
```

### 1.2 Implement Model Class

```python
# models/sentiment_model.py
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from config.model_config import ModelConfig

class SentimentClassifier(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, config.num_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_classes), labels.view(-1))

        return {"loss": loss, "logits": logits}
```

## Step 2: Training Pipeline

### 2.1 Data Preprocessing

```python
# data/preprocessing.py
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict

class SentimentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loaders(train_data, test_data, config):
    tokenizer = BertTokenizer.from_pretrained(config.model_name)

    train_dataset = SentimentDataset(
        train_data['text'], train_data['label'], tokenizer, config.max_length
    )
    test_dataset = SentimentDataset(
        test_data['text'], test_data['label'], tokenizer, config.max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, test_loader, tokenizer
```

### 2.2 Training Loop

```python
# training/trainer.py
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np

class ModelTrainer:
    def __init__(self, model, config, train_loader, test_loader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Setup optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        total_steps = len(train_loader) * config.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        progress_bar = tqdm(self.train_loader, desc="Training")
        for batch in progress_bar:
            self.optimizer.zero_grad()

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs['loss']
                logits = outputs['logits']

                total_loss += loss.item()
                predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        return {
            'loss': total_loss / len(self.test_loader),
            'predictions': np.array(predictions),
            'true_labels': np.array(true_labels)
        }

    def train(self):
        best_accuracy = 0

        for epoch in range(self.config.epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")

            # Training
            train_loss = self.train_epoch()

            # Evaluation
            eval_results = self.evaluate()
            accuracy = np.mean(eval_results['predictions'] == eval_results['true_labels'])

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Eval Loss: {eval_results['loss']:.4f}")
            print(f"Accuracy: {accuracy:.4f}")

            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(self.model.state_dict(), 'best_model.pt')

        return best_accuracy
```

## Step 3: Model Evaluation

### 3.1 Comprehensive Evaluation

```python
# evaluation/evaluator.py
from model_evaluation.pipeline import EvaluationPipeline
from model_evaluation.config import EvaluationConfig
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class SentimentEvaluator:
    def __init__(self, model, tokenizer, test_data):
        self.model = model
        self.tokenizer = tokenizer
        self.test_data = test_data

    def run_comprehensive_evaluation(self):
        # Initialize evaluation pipeline
        config = EvaluationConfig(
            model_path="best_model.pt",
            test_dataset=self.test_data,
            bias_analysis=True,
            robustness_testing=True,
            performance_profiling=True
        )

        pipeline = EvaluationPipeline(config)

        # Run evaluation
        results = pipeline.evaluate()

        # Generate detailed report
        pipeline.generate_report("sentiment_evaluation_report.html")

        return results

    def analyze_predictions(self, predictions, true_labels):
        # Classification report
        report = classification_report(true_labels, predictions, target_names=['Negative', 'Positive'])
        print("Classification Report:")
        print(report)

        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        plt.show()

        return report, cm
```

### 3.2 Bias Detection

```python
# evaluation/bias_analysis.py
from model_evaluation.bias_detection import BiasDetector
import pandas as pd

class SentimentBiasAnalyzer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.bias_detector = BiasDetector()

    def analyze_demographic_bias(self, test_data):
        # Create demographic groups (example: by text length)
        short_texts = [text for text in test_data['text'] if len(text.split()) < 20]
        long_texts = [text for text in test_data['text'] if len(text.split()) >= 50]

        # Analyze bias across groups
        bias_results = {}

        for group_name, texts in [("short", short_texts), ("long", long_texts)]:
            if len(texts) > 0:
                predictions = self.predict_batch(texts)
                bias_metrics = self.bias_detector.calculate_metrics(predictions, group_name)
                bias_results[group_name] = bias_metrics

        return bias_results

    def predict_batch(self, texts):
        predictions = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                prediction = torch.argmax(outputs['logits'], dim=-1).item()
                predictions.append(prediction)
        return predictions
```

## Step 4: Model Profiling

### 4.1 Performance Profiling

```python
# profiling/model_profiler.py
from model_evaluation.pipeline import ModelProfiler
import time
import psutil
import torch

class SentimentModelProfiler:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def profile_inference(self, sample_texts, num_runs=100):
        # Warmup
        for _ in range(10):
            self._single_inference(sample_texts[0])

        # Profile runs
        latencies = []
        memory_usage = []

        for _ in range(num_runs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            self._single_inference(sample_texts[0])

            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            latencies.append((end_time - start_time) * 1000)  # Convert to ms
            memory_usage.append(end_memory - start_memory)

        return {
            'latencies': latencies,
            'memory_usage': memory_usage,
            'avg_latency': np.mean(latencies),
            'p95_latency': np.percentile(latencies, 95),
            'max_memory': np.max(memory_usage)
        }

    def _single_inference(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction = torch.argmax(outputs['logits'], dim=-1)
        return prediction
```

## Step 5: Model Deployment

### 5.1 Model Factory Integration

```python
# deployment/deployer.py
from model_factory import ModelFactory
import torch

class SentimentModelDeployer:
    def __init__(self):
        self.factory = ModelFactory()

    def deploy_model(self, model_path, model_name="sentiment_classifier"):
        # Load trained model
        model = torch.load(model_path)

        # Register in factory
        model_id = self.factory.register_model(
            model=model,
            name=model_name,
            version="1.0.0",
            description="BERT-based sentiment classification model",
            tags=["nlp", "sentiment", "classification"]
        )

        # Deploy model
        deployment = self.factory.deploy_model(
            model_id=model_id,
            platform="local",
            config={
                "host": "0.0.0.0",
                "port": 8081,
                "workers": 2
            }
        )

        # Generate API
        api = self.factory.generate_api(
            model_id=model_id,
            api_type="rest",
            endpoints=["predict", "health", "metrics"]
        )

        # Setup monitoring
        self.factory.setup_monitoring(
            model_id=model_id,
            metrics=["latency", "throughput", "error_rate"]
        )

        return deployment, api
```

## Step 6: Complete Exercise Script

### 6.1 Main Exercise Script

```python
# main_exercise.py
import torch
from datasets import load_dataset
from config.model_config import ModelConfig
from models.sentiment_model import SentimentClassifier
from data.preprocessing import create_data_loaders
from training.trainer import ModelTrainer
from evaluation.evaluator import SentimentEvaluator
from profiling.model_profiler import SentimentModelProfiler
from deployment.deployer import SentimentModelDeployer

def main():
    print("=== Sentiment Analysis Model Evaluation Exercise ===\n")

    # Step 1: Setup
    print("Step 1: Setting up data and configuration...")
    dataset = load_dataset("imdb")
    config = ModelConfig()

    # Step 2: Prepare data
    print("Step 2: Preparing data...")
    train_loader, test_loader, tokenizer = create_data_loaders(
        dataset["train"], dataset["test"], config
    )

    # Step 3: Create and train model
    print("Step 3: Creating and training model...")
    model = SentimentClassifier(config)
    trainer = ModelTrainer(model, config, train_loader, test_loader)
    best_accuracy = trainer.train()
    print(f"Best accuracy achieved: {best_accuracy:.4f}")

    # Step 4: Comprehensive evaluation
    print("Step 4: Running comprehensive evaluation...")
    evaluator = SentimentEvaluator(model, tokenizer, dataset["test"])
    eval_results = evaluator.run_comprehensive_evaluation()

    # Step 5: Model profiling
    print("Step 5: Profiling model performance...")
    profiler = SentimentModelProfiler(model, tokenizer)
    profile_results = profiler.profile_inference(dataset["test"]["text"][:10])
    print(f"Average latency: {profile_results['avg_latency']:.2f}ms")
    print(f"P95 latency: {profile_results['p95_latency']:.2f}ms")

    # Step 6: Deploy model
    print("Step 6: Deploying model...")
    deployer = SentimentModelDeployer()
    deployment, api = deployer.deploy_model("best_model.pt")
    print(f"Model deployed at: {deployment.url}")

    print("\n=== Exercise completed successfully! ===")
    print("Check the generated reports:")
    print("- sentiment_evaluation_report.html")
    print("- confusion_matrix.png")
    print("- Model API available at deployment URL")

if __name__ == "__main__":
    main()
```

## Expected Results

After completing this exercise, you should have:

1. **Trained Model**: A BERT-based sentiment classifier with >90% accuracy
2. **Evaluation Report**: Comprehensive HTML report with performance metrics
3. **Profiling Data**: Detailed performance characteristics and resource usage
4. **Deployed API**: REST API endpoint for model inference
5. **Monitoring**: Real-time monitoring of model performance

## Key Learnings

This exercise demonstrates:

- **End-to-end ML pipeline** from data preparation to deployment
- **Comprehensive evaluation** including bias detection and robustness testing
- **Performance profiling** for optimization insights
- **Model deployment** using factory patterns
- **Monitoring and observability** for production systems

## Next Steps

1. **Experiment with different models** (RoBERTa, DistilBERT, etc.)
2. **Implement additional bias detection methods**
3. **Add more robustness tests** (adversarial examples, etc.)
4. **Scale deployment** to cloud platforms
5. **Implement A/B testing** for model comparison

## Troubleshooting

### Common Issues

- **CUDA out of memory**: Reduce batch size or use gradient checkpointing
- **Slow training**: Use mixed precision training or smaller model
- **Poor accuracy**: Increase training epochs or adjust learning rate
- **Deployment errors**: Check port availability and dependencies

### Performance Tips

- Use GPU acceleration when available
- Implement data loading optimization
- Use mixed precision training
- Cache tokenized data for faster training
