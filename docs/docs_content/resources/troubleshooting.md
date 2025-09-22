# Troubleshooting Guide

## Overview

This comprehensive troubleshooting guide helps diagnose and resolve common issues in the AI Assignments project. It covers installation problems, runtime errors, performance issues, and deployment challenges.

## Common Installation Issues

### 1. Virtual Environment Issues

#### Problem: Virtual environment not activating

```bash
# Error: Execution policy prevents running scripts
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Problem: Python version conflicts

```bash
# Check Python version
python --version

# Create virtual environment with specific Python version
python3.9 -m venv venv
```

#### Problem: Package installation failures

```bash
# Clear pip cache
pip cache purge

# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# Install packages individually to identify issues
pip install torch
pip install transformers
pip install gradio
```

### 2. Dependency Conflicts

#### Problem: Conflicting package versions

```bash
# Check for conflicts
pip check

# Create requirements with exact versions
pip freeze > requirements-exact.txt

# Use conda for complex environments
conda create -n ai_assignments python=3.9
conda activate ai_assignments
conda install pytorch torchvision torchaudio -c pytorch
```

#### Problem: CUDA/PyTorch compatibility

```bash
# Check CUDA version
nvidia-smi

# Install compatible PyTorch version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Runtime Errors

### 1. Import Errors

#### Problem: Module not found

```python
# Check Python path
import sys
print(sys.path)

# Add project root to Python path
import os
sys.path.insert(0, os.path.abspath('.'))

# Use relative imports
from src.model_evaluation.pipeline import EvaluationPipeline
```

#### Problem: Circular imports

```python
# Solution: Use local imports
def some_function():
    from src.utils.config_utils import get_config
    config = get_config()
    return config
```

### 2. Model Loading Issues

#### Problem: Model file not found

```python
import os
from pathlib import Path

def load_model_safely(model_path: str):
    """Load model with proper error handling."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        model = torch.load(model_path, map_location='cpu')
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

# Usage
model_path = Path(__file__).parent / "models" / "model.pt"
model = load_model_safely(str(model_path))
```

#### Problem: CUDA out of memory

```python
import torch

def handle_cuda_memory():
    """Handle CUDA memory issues."""
    if torch.cuda.is_available():
        # Clear cache
        torch.cuda.empty_cache()

        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(0.8)

        # Use mixed precision
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()

        return scaler
    return None

# Usage in training loop
scaler = handle_cuda_memory()
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. Database Connection Issues

#### Problem: Database connection timeout

```python
import sqlite3
import psycopg2
from sqlalchemy import create_engine
import time

def test_database_connection(database_url: str, max_retries: int = 3):
    """Test database connection with retries."""
    for attempt in range(max_retries):
        try:
            engine = create_engine(database_url, pool_pre_ping=True)
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            print(f"Database connection successful (attempt {attempt + 1})")
            return engine
        except Exception as e:
            print(f"Database connection failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
```

#### Problem: Database locked

```sql
-- Check for locks (SQLite)
PRAGMA database_list;
PRAGMA lock_status;

-- Unlock database (if safe to do so)
PRAGMA wal_checkpoint(TRUNCATE);
```

## Performance Issues

### 1. Slow Model Inference

#### Problem: High latency

```python
import time
import torch
from torch.utils.data import DataLoader

def optimize_model_inference(model, dataloader):
    """Optimize model for inference."""
    model.eval()

    # Enable optimizations
    if torch.cuda.is_available():
        model = model.cuda()
        torch.backends.cudnn.benchmark = True

    # Use torch.jit for optimization
    model = torch.jit.optimize_for_inference(torch.jit.script(model))

    # Batch processing
    with torch.no_grad():
        for batch in dataloader:
            start_time = time.time()
            outputs = model(batch)
            inference_time = time.time() - start_time
            print(f"Batch inference time: {inference_time:.4f}s")
```

#### Problem: Memory issues during inference

```python
def inference_with_memory_management(model, inputs, batch_size=32):
    """Inference with memory management."""
    results = []

    # Process in batches
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i + batch_size]

        # Move to device
        if torch.cuda.is_available():
            batch = batch.cuda()

        # Inference
        with torch.no_grad():
            outputs = model(batch)
            results.append(outputs.cpu())

        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(results, dim=0)
```

### 2. Memory Leaks

#### Problem: Increasing memory usage

```python
import gc
import psutil
import torch

class MemoryMonitor:
    def __init__(self):
        self.initial_memory = self.get_memory_usage()

    def get_memory_usage(self):
        """Get current memory usage."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB

    def check_memory_leak(self, threshold_mb=100):
        """Check for memory leaks."""
        current_memory = self.get_memory_usage()
        memory_increase = current_memory - self.initial_memory

        if memory_increase > threshold_mb:
            print(f"Potential memory leak: {memory_increase:.2f}MB increase")
            return True
        return False

    def cleanup_memory(self):
        """Clean up memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Usage
monitor = MemoryMonitor()
# ... perform operations ...
if monitor.check_memory_leak():
    monitor.cleanup_memory()
```

## API and Web Service Issues

### 1. Gradio App Issues

#### Problem: Gradio interface not loading

```python
import gradio as gr

def create_gradio_interface():
    """Create Gradio interface with error handling."""
    try:
        interface = gr.Interface(
            fn=your_function,
            inputs="text",
            outputs="text",
            title="AI Assignments",
            description="Interactive AI interface"
        )
        return interface
    except Exception as e:
        print(f"Error creating Gradio interface: {e}")
        return None

# Start with error handling
try:
    interface = create_gradio_interface()
    if interface:
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True
        )
except Exception as e:
    print(f"Failed to start Gradio app: {e}")
```

#### Problem: CORS issues

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 2. API Rate Limiting

#### Problem: Too many requests

```python
from functools import wraps
import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, max_requests=100, window=60):
        self.max_requests = max_requests
        self.window = window
        self.requests = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        client_requests = self.requests[client_id]

        # Remove old requests
        client_requests[:] = [req_time for req_time in client_requests
                            if now - req_time < self.window]

        # Check limit
        if len(client_requests) >= self.max_requests:
            return False

        # Add current request
        client_requests.append(now)
        return True

# Usage
rate_limiter = RateLimiter(max_requests=100, window=60)

@app.post("/api/predict")
async def predict(request: dict, client_id: str = "default"):
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Process request
    return {"result": "success"}
```

## Deployment Issues

### 1. Docker Issues

#### Problem: Container won't start

```dockerfile
# Add debugging to Dockerfile
FROM python:3.9-slim

# Add debugging tools
RUN apt-get update && apt-get install -y \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Use non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Start with debugging
CMD ["python", "-m", "src.gradio_app.main", "--debug"]
```

#### Problem: Volume mounting issues

```bash
# Check volume permissions
docker run --rm -v $(pwd):/app alpine ls -la /app

# Fix permissions
sudo chown -R $USER:$USER .
chmod -R 755 .

# Use named volumes for data
docker volume create ai_assignments_data
docker run -v ai_assignments_data:/app/data your-image
```

### 2. Kubernetes Issues

#### Problem: Pods not starting

```yaml
# Add debugging to deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-assignments-debug
spec:
  template:
    spec:
      containers:
        - name: ai-assignments
          image: ai-assignments:latest
          command: ["/bin/bash"]
          args: ["-c", "while true; do sleep 30; done"] # Keep container alive
          env:
            - name: DEBUG
              value: "true"
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "500m"
```

#### Problem: Service not accessible

```bash
# Check service endpoints
kubectl get endpoints ai-assignments-service

# Check pod logs
kubectl logs -l app=ai-assignments

# Port forward for testing
kubectl port-forward service/ai-assignments-service 8080:80

# Check ingress
kubectl describe ingress ai-assignments-ingress
```

## Monitoring and Debugging

### 1. Logging Configuration

#### Comprehensive logging setup

```python
import logging
import logging.config
import sys
from pathlib import Path

def setup_logging(log_level="INFO", log_file=None):
    """Setup comprehensive logging configuration."""

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Console handler
    console_handler = {
        'class': 'logging.StreamHandler',
        'level': log_level,
        'formatter': 'detailed',
        'stream': sys.stdout
    }

    # File handler
    handlers = {'console': console_handler}

    if log_file:
        handlers['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': log_level,
            'formatter': 'detailed',
            'filename': log_file,
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        }

    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'simple': {
                'format': '%(levelname)s - %(message)s'
            },
            'detailed': {
                'format': log_format
            }
        },
        'handlers': handlers,
        'loggers': {
            '': {
                'handlers': list(handlers.keys()),
                'level': log_level,
                'propagate': False
            }
        }
    }

    logging.config.dictConfig(logging_config)

# Usage
setup_logging(log_level="DEBUG", log_file="logs/app.log")
logger = logging.getLogger(__name__)
```

### 2. Error Tracking

#### Exception handling and reporting

```python
import traceback
import logging
from functools import wraps

def error_handler(func):
    """Decorator for comprehensive error handling."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger = logging.getLogger(func.__module__)
            logger.error(f"Error in {func.__name__}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

            # Send to error tracking service (e.g., Sentry)
            # sentry_sdk.capture_exception()

            raise
    return wrapper

# Usage
@error_handler
def risky_function():
    # Function that might fail
    pass
```

### 3. Performance Monitoring

#### Real-time performance tracking

```python
import time
import psutil
import threading
from collections import deque

class PerformanceMonitor:
    def __init__(self, max_samples=100):
        self.max_samples = max_samples
        self.metrics = {
            'cpu': deque(maxlen=max_samples),
            'memory': deque(maxlen=max_samples),
            'response_time': deque(maxlen=max_samples)
        }
        self.running = False
        self.thread = None

    def start(self):
        """Start monitoring."""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self.thread:
            self.thread.join()

    def _monitor_loop(self):
        """Monitoring loop."""
        while self.running:
            # Collect system metrics
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent

            self.metrics['cpu'].append(cpu_percent)
            self.metrics['memory'].append(memory_percent)

            time.sleep(1)

    def record_response_time(self, response_time):
        """Record API response time."""
        self.metrics['response_time'].append(response_time)

    def get_stats(self):
        """Get performance statistics."""
        stats = {}
        for metric, values in self.metrics.items():
            if values:
                stats[metric] = {
                    'current': values[-1],
                    'average': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values)
                }
        return stats

# Usage
monitor = PerformanceMonitor()
monitor.start()

# In API endpoint
@app.get("/api/health")
async def health_check():
    start_time = time.time()
    # ... process request ...
    response_time = time.time() - start_time
    monitor.record_response_time(response_time)

    return {
        "status": "healthy",
        "performance": monitor.get_stats()
    }
```

## Recovery Procedures

### 1. Data Recovery

#### Database backup and restore

```python
import sqlite3
import shutil
from datetime import datetime

def backup_database(db_path: str, backup_dir: str = "backups"):
    """Create database backup."""
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"database_{timestamp}.db"

    shutil.copy2(db_path, backup_path)
    print(f"Database backed up to: {backup_path}")
    return backup_path

def restore_database(db_path: str, backup_path: str):
    """Restore database from backup."""
    shutil.copy2(backup_path, db_path)
    print(f"Database restored from: {backup_path}")

# Usage
backup_path = backup_database("ai_assignments.db")
# ... if issues occur ...
restore_database("ai_assignments.db", backup_path)
```

### 2. Service Recovery

#### Automatic restart on failure

```python
import subprocess
import time
import logging

class ServiceManager:
    def __init__(self, service_command, max_restarts=5):
        self.service_command = service_command
        self.max_restarts = max_restarts
        self.restart_count = 0
        self.process = None
        self.logger = logging.getLogger(__name__)

    def start_service(self):
        """Start the service."""
        try:
            self.process = subprocess.Popen(self.service_command)
            self.logger.info(f"Service started with PID: {self.process.pid}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start service: {e}")
            return False

    def monitor_service(self):
        """Monitor and restart service if needed."""
        while self.restart_count < self.max_restarts:
            if self.process is None or self.process.poll() is not None:
                self.logger.warning("Service stopped, attempting restart")

                if self.start_service():
                    self.restart_count += 1
                    time.sleep(5)  # Wait before monitoring again
                else:
                    self.logger.error("Failed to restart service")
                    break
            else:
                time.sleep(10)  # Check every 10 seconds

        self.logger.error("Max restart attempts reached")

# Usage
service_manager = ServiceManager(["python", "-m", "src.gradio_app.main"])
service_manager.monitor_service()
```

This troubleshooting guide provides comprehensive solutions for common issues encountered in the AI Assignments project, helping developers quickly diagnose and resolve problems.
