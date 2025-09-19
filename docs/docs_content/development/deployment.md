# Deployment Guide

## Overview

This guide covers deployment strategies, environments, and best practices for the AI Assignments project. It includes both local development deployments and production deployment strategies.

## Prerequisites

### System Requirements
- **Operating System**: Windows 10/11, macOS, or Linux
- **Python**: 3.8 or higher
- **Memory**: Minimum 8GB RAM (16GB recommended)
- **Storage**: At least 10GB free space
- **GPU**: Optional but recommended for model training

### Required Software
- Python 3.8+
- Git
- Docker (for containerized deployments)
- PowerShell (Windows) or Bash (Linux/macOS)

## Environment Setup

### 1. Local Development Environment

#### Windows (PowerShell)
```powershell
# Navigate to project directory
cd C:\Users\samne\PycharmProjects\ai_assignments

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r config\requirements.txt
pip install -r config\requirements-testing.txt

# Verify installation
python -c "import src; print('Installation successful')"
```

#### Linux/macOS
```bash
# Navigate to project directory
cd /path/to/ai_assignments

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r config/requirements.txt
pip install -r config/requirements-testing.txt

# Verify installation
python -c "import src; print('Installation successful')"
```

### 2. Environment Variables
Create a `.env` file in the project root:
```env
# Database Configuration
DATABASE_URL=sqlite:///./ai_assignments.db
REDIS_URL=redis://localhost:6379

# Model Configuration
MODEL_CACHE_DIR=./models/cache
DEFAULT_MODEL_PATH=./models/default

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret-here

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/application.log

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
```

## Deployment Strategies

### 1. Local Development Deployment

#### Start Development Server
```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Start Gradio application
python -m src.gradio_app.main

# Or start with custom configuration
python -m src.gradio_app.main --host 0.0.0.0 --port 7860 --share
```

#### Start with MCP Server
```powershell
python -m src.gradio_app.main --mcp-server --mcp-port 8001
```

#### Run Tests
```powershell
# Run all tests
python -m pytest tests\ -v --tb=short

# Run specific test categories
python -m pytest tests\unit\ -v
python -m pytest tests\integration\ -v
python -m pytest tests\e2e\ -v --timeout=600

# Run with coverage
python -m pytest tests\ -v --cov=src --cov-report=html
```

### 2. Docker Deployment

#### Build Docker Image
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY config/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["python", "-m", "src.gradio_app.main", "--host", "0.0.0.0", "--port", "8000"]
```

#### Build and Run
```bash
# Build image
docker build -t ai-assignments:latest .

# Run container
docker run -d \
  --name ai-assignments \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  --env-file .env \
  ai-assignments:latest

# Check logs
docker logs ai-assignments

# Stop container
docker stop ai-assignments
docker rm ai-assignments
```

#### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - DATABASE_URL=sqlite:///./data/ai_assignments.db
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    restart: unless-stopped

volumes:
  redis_data:
```

### 3. Kubernetes Deployment

#### Namespace
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ai-assignments
```

#### ConfigMap
```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ai-assignments-config
  namespace: ai-assignments
data:
  DATABASE_URL: "sqlite:///./data/ai_assignments.db"
  LOG_LEVEL: "INFO"
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
```

#### Secret
```yaml
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: ai-assignments-secret
  namespace: ai-assignments
type: Opaque
data:
  SECRET_KEY: <base64-encoded-secret-key>
  JWT_SECRET: <base64-encoded-jwt-secret>
```

#### Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-assignments
  namespace: ai-assignments
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-assignments
  template:
    metadata:
      labels:
        app: ai-assignments
    spec:
      containers:
      - name: ai-assignments
        image: ai-assignments:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            configMapKeyRef:
              name: ai-assignments-config
              key: DATABASE_URL
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: ai-assignments-secret
              key: SECRET_KEY
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
        - name: log-storage
          mountPath: /app/logs
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
      - name: log-storage
        persistentVolumeClaim:
          claimName: log-pvc
```

#### Service
```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ai-assignments-service
  namespace: ai-assignments
spec:
  selector:
    app: ai-assignments
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### 4. Cloud Deployment

#### AWS ECS Deployment
```json
{
  "family": "ai-assignments",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "ai-assignments",
      "image": "your-account.dkr.ecr.region.amazonaws.com/ai-assignments:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "DATABASE_URL",
          "value": "sqlite:///./data/ai_assignments.db"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/ai-assignments",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Google Cloud Run
```yaml
# cloud-run.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: ai-assignments
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-throttling: "false"
    spec:
      containerConcurrency: 80
      timeoutSeconds: 300
      containers:
      - image: gcr.io/project-id/ai-assignments:latest
        ports:
        - containerPort: 8000
        env:
        - name: PORT
          value: "8000"
        - name: DATABASE_URL
          value: "sqlite:///./data/ai_assignments.db"
        resources:
          limits:
            cpu: "2"
            memory: "2Gi"
```

## Configuration Management

### Environment-Specific Configs

#### Development
```python
# config/development.py
DEBUG = True
LOG_LEVEL = "DEBUG"
DATABASE_URL = "sqlite:///./dev.db"
REDIS_URL = "redis://localhost:6379"
```

#### Staging
```python
# config/staging.py
DEBUG = False
LOG_LEVEL = "INFO"
DATABASE_URL = "postgresql://user:pass@staging-db:5432/ai_assignments"
REDIS_URL = "redis://staging-redis:6379"
```

#### Production
```python
# config/production.py
DEBUG = False
LOG_LEVEL = "WARNING"
DATABASE_URL = "postgresql://user:pass@prod-db:5432/ai_assignments"
REDIS_URL = "redis://prod-redis:6379"
ENABLE_METRICS = True
```

## Monitoring and Logging

### Application Metrics
```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Number of active connections')

def start_metrics_server(port=9090):
    start_http_server(port)
```

### Logging Configuration
```python
# logging/logging_config.py
import logging
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'detailed',
            'class': 'logging.FileHandler',
            'filename': 'logs/application.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
```

## Health Checks

### Application Health Check
```python
# health/health_check.py
from fastapi import FastAPI
from typing import Dict
import psutil
import time

app = FastAPI()

@app.get("/health")
async def health_check() -> Dict:
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "uptime": time.time() - start_time
    }

@app.get("/ready")
async def readiness_check() -> Dict:
    """Readiness check endpoint"""
    checks = {
        "database": await check_database(),
        "redis": await check_redis(),
        "disk_space": check_disk_space(),
        "memory": check_memory()
    }
    
    all_healthy = all(checks.values())
    
    return {
        "status": "ready" if all_healthy else "not_ready",
        "checks": checks,
        "timestamp": time.time()
    }

def check_disk_space() -> bool:
    """Check available disk space"""
    disk_usage = psutil.disk_usage('/')
    free_percent = (disk_usage.free / disk_usage.total) * 100
    return free_percent > 10  # At least 10% free space

def check_memory() -> bool:
    """Check available memory"""
    memory = psutil.virtual_memory()
    return memory.percent < 90  # Less than 90% memory usage
```

## Security Considerations

### SSL/TLS Configuration
```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Environment Security
```bash
# Secure environment variables
export SECRET_KEY=$(openssl rand -hex 32)
export JWT_SECRET=$(openssl rand -hex 32)
export DATABASE_PASSWORD=$(openssl rand -base64 32)

# Use secrets management in production
# AWS Secrets Manager, Azure Key Vault, etc.
```

## Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Find process using port
netstat -tulpn | grep :8000
# or
lsof -i :8000

# Kill process
kill -9 <PID>
```

#### Memory Issues
```bash
# Check memory usage
free -h
# or
ps aux --sort=-%mem | head

# Increase swap if needed
sudo swapon -s
```

#### Database Connection Issues
```python
# Test database connection
import sqlite3
try:
    conn = sqlite3.connect('ai_assignments.db')
    print("Database connection successful")
    conn.close()
except Exception as e:
    print(f"Database connection failed: {e}")
```

### Log Analysis
```bash
# View application logs
tail -f logs/application.log

# Search for errors
grep -i error logs/application.log

# Monitor real-time logs
tail -f logs/application.log | grep -i "ERROR\|WARNING"
```

## Deployment Checklist

### Pre-Deployment
- [ ] All tests passing
- [ ] Environment variables configured
- [ ] Database migrations applied
- [ ] SSL certificates installed
- [ ] Monitoring configured
- [ ] Backup strategy in place

### Post-Deployment
- [ ] Health checks passing
- [ ] Application responding correctly
- [ ] Metrics collection working
- [ ] Logs being generated
- [ ] Performance monitoring active
- [ ] Alerting configured

### Rollback Plan
- [ ] Previous version tagged
- [ ] Database rollback procedures documented
- [ ] Configuration rollback procedures documented
- [ ] Rollback testing completed

This deployment guide provides comprehensive instructions for deploying the AI Assignments project across different environments and platforms, ensuring reliability, security, and maintainability.
