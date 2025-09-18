---
layout: post
title: "Deployment Guide"
date: 2025-09-18 10:00:00 -0400
categories: [Documentation, Deployment]
tags: [Deployment, Infrastructure, Kubernetes, Docker, CI/CD]
author: Lenovo AAITC Team
---

# Deployment Guide - Lenovo AAITC Solutions

## Overview

This guide provides comprehensive instructions for deploying the Lenovo AAITC Solutions framework in various environments, from local development to production-scale enterprise deployments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [CI/CD Pipeline Setup](#cicd-pipeline-setup)
7. [Monitoring and Observability](#monitoring-and-observability)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Python**: 3.8 or higher
- **Memory**: Minimum 8GB RAM (16GB recommended for production)
- **Storage**: Minimum 10GB free space
- **Network**: Internet connection for model API access

### Required Software

- **Python 3.8+**: [Download from python.org](https://www.python.org/downloads/)
- **Git**: [Download from git-scm.com](https://git-scm.com/downloads)
- **Docker**: [Download from docker.com](https://www.docker.com/products/docker-desktop)
- **Kubernetes**: For production deployments
- **Terraform**: For infrastructure as code

---

## Local Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ai_assignments.git
cd ai_assignments
```

### 2. Create Virtual Environment

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r config/requirements.txt
pip install -r config/requirements-testing.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Model API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# MCP Server Configuration
MCP_SERVER_PORT=8000
MCP_MAX_CONNECTIONS=100

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/aaitc.log

# Database Configuration (if using)
DATABASE_URL=sqlite:///aaitc.db
```

### 5. Launch the Application

```bash
python -m src.gradio_app.main
```

The application will be available at `http://localhost:7860`

---

## Docker Deployment

### 1. Build Docker Image

```bash
docker build -t lenovo-aaitc:latest .
```

### 2. Run Container

```bash
docker run -d \
  --name lenovo-aaitc \
  -p 7860:7860 \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your_key \
  -e ANTHROPIC_API_KEY=your_key \
  lenovo-aaitc:latest
```

### 3. Docker Compose (Recommended)

Create `docker-compose.yml`:

```yaml
version: "3.8"

services:
  aaitc-app:
    build: .
    ports:
      - "7860:7860"
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - LOG_LEVEL=INFO
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=aaitc
      - POSTGRES_USER=aaitc
      - POSTGRES_PASSWORD=secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  postgres_data:
```

Run with:

```bash
docker-compose up -d
```

---

## Kubernetes Deployment

### 1. Create Namespace

```bash
kubectl create namespace aaitc
```

### 2. Deploy Application

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lenovo-aaitc
  namespace: aaitc
spec:
  replicas: 3
  selector:
    matchLabels:
      app: lenovo-aaitc
  template:
    metadata:
      labels:
        app: lenovo-aaitc
    spec:
      containers:
        - name: aaitc-app
          image: lenovo-aaitc:latest
          ports:
            - containerPort: 7860
            - containerPort: 8000
          env:
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: aaitc-secrets
                  key: openai-api-key
            - name: ANTHROPIC_API_KEY
              valueFrom:
                secretKeyRef:
                  name: aaitc-secrets
                  key: anthropic-api-key
          resources:
            requests:
              memory: "2Gi"
              cpu: "1000m"
            limits:
              memory: "4Gi"
              cpu: "2000m"
          livenessProbe:
            httpGet:
              path: /health
              port: 7860
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: 7860
            initialDelaySeconds: 5
            periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: lenovo-aaitc-service
  namespace: aaitc
spec:
  selector:
    app: lenovo-aaitc
  ports:
    - name: gradio
      port: 7860
      targetPort: 7860
    - name: mcp
      port: 8000
      targetPort: 8000
  type: LoadBalancer
```

### 3. Deploy to Kubernetes

```bash
kubectl apply -f k8s-deployment.yaml
```

---

## Cloud Deployment

### AWS Deployment

#### 1. Using AWS EKS

```bash
# Create EKS cluster
eksctl create cluster --name aaitc-cluster --region us-west-2

# Deploy application
kubectl apply -f k8s-deployment.yaml
```

#### 2. Using AWS App Runner

```yaml
# apprunner.yaml
version: 1.0
runtime: python3
build:
  commands:
    build:
      - pip install -r config/requirements.txt
run:
  runtime-version: 3.8
  command: python -m src.gradio_app.main
  network:
    port: 7860
    env: PORT
  env:
    - name: OPENAI_API_KEY
      value: your_openai_key
    - name: ANTHROPIC_API_KEY
      value: your_anthropic_key
```

### Azure Deployment

#### 1. Using Azure Container Instances

```bash
az container create \
  --resource-group aaitc-rg \
  --name lenovo-aaitc \
  --image lenovo-aaitc:latest \
  --ports 7860 8000 \
  --environment-variables \
    OPENAI_API_KEY=your_key \
    ANTHROPIC_API_KEY=your_key
```

### Google Cloud Deployment

#### 1. Using Google Cloud Run

```bash
gcloud run deploy lenovo-aaitc \
  --image gcr.io/your-project/lenovo-aaitc:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=your_key,ANTHROPIC_API_KEY=your_key
```

---

## CI/CD Pipeline Setup

### GitHub Actions

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy Lenovo AAITC

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.8"
      - name: Install dependencies
        run: |
          pip install -r config/requirements.txt
          pip install -r config/requirements-testing.txt
      - name: Run tests
        run: |
          python -m pytest tests/ -v --cov=src

  build-and-deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker build -t lenovo-aaitc:${{ github.sha }} .
      - name: Deploy to production
        run: |
          # Add your deployment commands here
          echo "Deploying to production..."
```

---

## Monitoring and Observability

### 1. Prometheus Metrics

The application exposes metrics at `/metrics` endpoint:

```python
# Example metrics
aaitc_model_evaluations_total{model="gpt-5", status="success"} 150
aaitc_model_evaluations_duration_seconds{model="gpt-5"} 2.5
aaitc_active_connections 25
```

### 2. Grafana Dashboard

Import the provided Grafana dashboard configuration:

```json
{
  "dashboard": {
    "title": "Lenovo AAITC Monitoring",
    "panels": [
      {
        "title": "Model Evaluation Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(aaitc_model_evaluations_total[5m])"
          }
        ]
      }
    ]
  }
}
```

### 3. Logging Configuration

```yaml
# logging.yaml
version: 1
formatters:
  standard:
    format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: standard
    filename: logs/aaitc.log
    maxBytes: 10485760
    backupCount: 5
loggers:
  aaitc:
    level: DEBUG
    handlers: [console, file]
    propagate: no
```

---

## Troubleshooting

### Common Issues

#### 1. Port Already in Use

```bash
# Find process using port 7860
netstat -ano | findstr :7860

# Kill process (Windows)
taskkill /PID <PID> /F

# Kill process (Linux/macOS)
kill -9 <PID>
```

#### 2. API Key Issues

```bash
# Verify environment variables
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY

# Test API connectivity
python -c "import openai; print('OpenAI API accessible')"
```

#### 3. Memory Issues

```bash
# Monitor memory usage
docker stats

# Increase memory limits in docker-compose.yml
services:
  aaitc-app:
    deploy:
      resources:
        limits:
          memory: 8G
```

#### 4. Database Connection Issues

```bash
# Check database connectivity
python -c "import sqlite3; print('Database accessible')"

# Reset database
rm aaitc.db
python -m src.utils.init_db
```

### Performance Optimization

#### 1. Enable Caching

```python
# In your application
import redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)
```

#### 2. Optimize Model Loading

```python
# Lazy loading for models
class ModelManager:
    def __init__(self):
        self._models = {}

    def get_model(self, model_name):
        if model_name not in self._models:
            self._models[model_name] = load_model(model_name)
        return self._models[model_name]
```

---

## Security Considerations

### 1. API Key Management

- Store API keys in environment variables or secret management systems
- Never commit API keys to version control
- Rotate API keys regularly
- Use least-privilege access principles

### 2. Network Security

- Use HTTPS in production
- Implement rate limiting
- Configure firewall rules
- Use VPN for internal communications

### 3. Data Protection

- Encrypt sensitive data at rest
- Use secure communication protocols
- Implement access controls
- Regular security audits

---

## Support and Maintenance

### Regular Maintenance Tasks

1. **Weekly**: Review logs and performance metrics
2. **Monthly**: Update dependencies and security patches
3. **Quarterly**: Performance optimization and capacity planning
4. **Annually**: Security audit and disaster recovery testing

### Getting Help

- **Documentation**: Check this guide and API documentation
- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Join GitHub Discussions for questions
- **Email**: Contact aaitc-support@lenovo.com for enterprise support

---

**Deployment Guide - Lenovo AAITC Solutions**  
_Comprehensive deployment instructions for all environments_
