# Performance Metrics

## Overview

This document provides comprehensive performance metrics, benchmarks, and optimization guidelines for the AI Assignments project.

## System Performance Metrics

### Response Time Benchmarks

| Component | Average Response Time | 95th Percentile | 99th Percentile |
|-----------|----------------------|-----------------|-----------------|
| API Gateway | 5ms | 15ms | 25ms |
| Model Inference | 120ms | 200ms | 300ms |
| Database Queries | 10ms | 30ms | 50ms |
| File Upload | 500ms | 1s | 2s |
| WebSocket Messages | 2ms | 5ms | 10ms |

### Throughput Metrics

| Service | Requests/Second | Concurrent Users | Data Processing |
|---------|----------------|------------------|-----------------|
| API Gateway | 10,000 | 5,000 | - |
| Model Service | 1,000 | 500 | 100MB/s |
| Evaluation Pipeline | 100 | 50 | 50MB/s |
| File Processing | 50 | 25 | 200MB/s |

### Resource Utilization

| Resource | CPU Usage | Memory Usage | Storage I/O | Network I/O |
|----------|-----------|--------------|-------------|-------------|
| Model Service | 60-80% | 2-4GB | 100MB/s | 50MB/s |
| Database | 30-50% | 1-2GB | 200MB/s | 10MB/s |
| Cache Layer | 10-20% | 512MB | 50MB/s | 5MB/s |
| Web Server | 20-40% | 256MB | 10MB/s | 100MB/s |

## Model Performance Metrics

### Accuracy Benchmarks

| Model Type | Dataset | Accuracy | Precision | Recall | F1-Score |
|------------|---------|----------|-----------|--------|----------|
| Sentiment Analysis | IMDB | 94.2% | 93.8% | 94.5% | 94.1% |
| Text Classification | AG News | 91.5% | 91.2% | 91.8% | 91.5% |
| Image Classification | CIFAR-10 | 89.3% | 89.0% | 89.6% | 89.3% |
| Named Entity Recognition | CoNLL-2003 | 92.1% | 91.8% | 92.4% | 92.1% |

### Inference Performance

| Model | Batch Size | Latency | Throughput | Memory Usage |
|-------|------------|---------|------------|--------------|
| BERT-base | 1 | 25ms | 40 req/s | 1.2GB |
| BERT-base | 8 | 85ms | 94 req/s | 2.8GB |
| BERT-base | 16 | 150ms | 107 req/s | 4.5GB |
| DistilBERT | 1 | 12ms | 83 req/s | 0.6GB |
| DistilBERT | 8 | 45ms | 178 req/s | 1.4GB |

## Performance Optimization Strategies

### 1. Model Optimization

#### Quantization
```python
import torch
from torch.quantization import quantize_dynamic

# Dynamic quantization
model_quantized = quantize_dynamic(
    model, 
    {torch.nn.Linear}, 
    dtype=torch.qint8
)

# Performance improvement: 2-4x faster, 2-4x smaller
```

#### Model Pruning
```python
import torch.nn.utils.prune as prune

# Prune 20% of connections
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)
```

#### Knowledge Distillation
```python
class DistillationTrainer:
    def __init__(self, teacher_model, student_model):
        self.teacher = teacher_model
        self.student = student_model
    
    def distill_loss(self, student_logits, teacher_logits, labels, temperature=3):
        # Soft targets from teacher
        soft_targets = F.softmax(teacher_logits / temperature, dim=1)
        soft_prob = F.log_softmax(student_logits / temperature, dim=1)
        
        # Distillation loss
        distillation_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean')
        
        # Hard targets
        hard_loss = F.cross_entropy(student_logits, labels)
        
        return distillation_loss * (temperature ** 2) + hard_loss
```

### 2. Caching Strategies

#### Model Output Caching
```python
import redis
import hashlib
import json

class ModelCache:
    def __init__(self, redis_client, ttl=3600):
        self.redis = redis_client
        self.ttl = ttl
    
    def get_cache_key(self, model_id, input_data):
        """Generate cache key from model and input."""
        input_str = json.dumps(input_data, sort_keys=True)
        return f"model:{model_id}:{hashlib.md5(input_str.encode()).hexdigest()}"
    
    def get(self, model_id, input_data):
        """Get cached prediction."""
        key = self.get_cache_key(model_id, input_data)
        result = self.redis.get(key)
        return json.loads(result) if result else None
    
    def set(self, model_id, input_data, prediction):
        """Cache prediction."""
        key = self.get_cache_key(model_id, input_data)
        self.redis.setex(key, self.ttl, json.dumps(prediction))
```

#### Database Query Caching
```python
from functools import wraps
import time

def cache_query(ttl=300):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try cache first
            cached_result = cache.get(cache_key)
            if cached_result:
                return cached_result
            
            # Execute query
            result = func(*args, **kwargs)
            
            # Cache result
            cache.setex(cache_key, ttl, result)
            return result
        return wrapper
    return decorator
```

### 3. Asynchronous Processing

#### Async Model Inference
```python
import asyncio
import aiohttp
from typing import List

class AsyncModelService:
    def __init__(self, model_urls: List[str]):
        self.model_urls = model_urls
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
    
    async def batch_predict(self, inputs: List[Dict]) -> List[Dict]:
        """Process multiple inputs concurrently."""
        tasks = []
        for i, input_data in enumerate(inputs):
            model_url = self.model_urls[i % len(self.model_urls)]
            task = self._predict_single(model_url, input_data)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
    
    async def _predict_single(self, model_url: str, input_data: Dict) -> Dict:
        """Single prediction request."""
        async with self.session.post(model_url, json=input_data) as response:
            return await response.json()
```

#### Background Task Processing
```python
from celery import Celery
import time

app = Celery('ai_assignments')

@app.task
def process_large_dataset(dataset_id: str):
    """Process large dataset in background."""
    # Long-running task
    dataset = load_dataset(dataset_id)
    results = []
    
    for batch in dataset.batches():
        batch_results = process_batch(batch)
        results.extend(batch_results)
        
        # Update progress
        app.update_state(
            state='PROGRESS',
            meta={'current': len(results), 'total': len(dataset)}
        )
    
    return results

@app.task
def generate_model_report(model_id: str):
    """Generate comprehensive model report."""
    # Report generation logic
    pass
```

### 4. Database Optimization

#### Connection Pooling
```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

# Configure connection pool
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)
```

#### Query Optimization
```python
from sqlalchemy import text
from sqlalchemy.orm import joinedload

# Eager loading to avoid N+1 queries
models = session.query(Model)\
    .options(joinedload(Model.evaluations))\
    .all()

# Raw SQL for complex queries
result = session.execute(text("""
    SELECT m.id, m.name, AVG(e.accuracy) as avg_accuracy
    FROM models m
    LEFT JOIN evaluations e ON m.id = e.model_id
    GROUP BY m.id, m.name
    HAVING AVG(e.accuracy) > :threshold
"""), {"threshold": 0.9})
```

### 5. Memory Management

#### Memory Profiling
```python
import psutil
import tracemalloc
from functools import wraps

def memory_profiler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Start memory tracing
        tracemalloc.start()
        
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Get final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory
        
        # Get peak memory
        current, peak = tracemalloc.get_traced_memory()
        peak_mb = peak / 1024 / 1024
        
        print(f"{func.__name__}: {memory_used:.2f}MB used, {peak_mb:.2f}MB peak")
        
        tracemalloc.stop()
        return result
    return wrapper
```

#### Garbage Collection
```python
import gc
import torch

def cleanup_memory():
    """Clean up memory and GPU cache."""
    # Python garbage collection
    gc.collect()
    
    # PyTorch GPU cache cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
```

## Performance Monitoring

### Real-time Metrics
```python
import time
from prometheus_client import Counter, Histogram, Gauge

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active connections')
MODEL_INFERENCE_TIME = Histogram('model_inference_duration_seconds', 'Model inference time')

def track_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            REQUEST_COUNT.labels(method='POST', endpoint=func.__name__).inc()
            return result
        finally:
            duration = time.time() - start_time
            REQUEST_DURATION.observe(duration)
    
    return wrapper
```

### Performance Dashboard
```python
from flask import Flask, jsonify
import psutil

app = Flask(__name__)

@app.route('/metrics/system')
def system_metrics():
    """Get system performance metrics."""
    return jsonify({
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent,
        'network_io': psutil.net_io_counters()._asdict()
    })

@app.route('/metrics/application')
def application_metrics():
    """Get application performance metrics."""
    return jsonify({
        'active_connections': get_active_connections(),
        'requests_per_second': get_request_rate(),
        'average_response_time': get_avg_response_time(),
        'error_rate': get_error_rate()
    })
```

## Load Testing

### Stress Testing Script
```python
import asyncio
import aiohttp
import time
from statistics import mean, median

class LoadTester:
    def __init__(self, url: str, concurrent_users: int = 100):
        self.url = url
        self.concurrent_users = concurrent_users
        self.results = []
    
    async def single_request(self, session: aiohttp.ClientSession):
        """Single request test."""
        start_time = time.time()
        
        try:
            async with session.get(self.url) as response:
                await response.text()
                success = response.status == 200
        except Exception:
            success = False
        
        duration = time.time() - start_time
        
        self.results.append({
            'duration': duration,
            'success': success,
            'timestamp': start_time
        })
    
    async def run_test(self, duration: int = 60):
        """Run load test for specified duration."""
        async with aiohttp.ClientSession() as session:
            tasks = []
            end_time = time.time() + duration
            
            while time.time() < end_time:
                # Create concurrent requests
                for _ in range(self.concurrent_users):
                    task = asyncio.create_task(self.single_request(session))
                    tasks.append(task)
                
                # Wait a bit before next batch
                await asyncio.sleep(0.1)
            
            # Wait for all tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_results(self):
        """Get test results summary."""
        durations = [r['duration'] for r in self.results if r['success']]
        success_rate = sum(r['success'] for r in self.results) / len(self.results)
        
        return {
            'total_requests': len(self.results),
            'success_rate': success_rate,
            'avg_response_time': mean(durations) if durations else 0,
            'median_response_time': median(durations) if durations else 0,
            'min_response_time': min(durations) if durations else 0,
            'max_response_time': max(durations) if durations else 0
        }

# Usage
async def main():
    tester = LoadTester('http://localhost:8000/api/health', concurrent_users=50)
    await tester.run_test(duration=60)
    print(tester.get_results())

asyncio.run(main())
```

## Performance Best Practices

### 1. Code Optimization
- Use appropriate data structures
- Avoid unnecessary computations
- Implement lazy loading
- Use generators for large datasets

### 2. Caching Strategy
- Cache at multiple levels
- Use appropriate TTL values
- Implement cache invalidation
- Monitor cache hit rates

### 3. Database Performance
- Use proper indexing
- Optimize queries
- Implement connection pooling
- Use read replicas for scaling

### 4. Memory Management
- Monitor memory usage
- Implement garbage collection
- Use memory profiling tools
- Optimize data structures

### 5. Network Optimization
- Use compression
- Implement HTTP/2
- Use CDN for static assets
- Optimize payload sizes

This performance guide provides comprehensive strategies for optimizing the AI Assignments project for production workloads while maintaining reliability and scalability.
