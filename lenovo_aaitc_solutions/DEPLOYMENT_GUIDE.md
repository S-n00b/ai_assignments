# Deployment Guide - Lenovo AAITC Solutions

## Overview

This guide provides comprehensive instructions for deploying the Lenovo AAITC Solutions framework in various environments, from development to production.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Development Deployment](#development-deployment)
3. [Production Deployment](#production-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Cloud Deployment](#cloud-deployment)
7. [Monitoring and Maintenance](#monitoring-and-maintenance)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

#### Minimum Requirements

- **CPU**: 4 cores, 2.4 GHz
- **RAM**: 8 GB
- **Storage**: 50 GB SSD
- **OS**: Ubuntu 20.04+, CentOS 8+, Windows 10+, macOS 10.15+

#### Recommended Requirements

- **CPU**: 8+ cores, 3.0+ GHz
- **RAM**: 32+ GB
- **Storage**: 200+ GB NVMe SSD
- **GPU**: NVIDIA RTX 3080+ or equivalent (for local model inference)

### Software Dependencies

#### Core Dependencies

- **Python**: 3.8+
- **Node.js**: 16+ (for frontend components)
- **Docker**: 20.10+ (for containerized deployment)
- **Kubernetes**: 1.20+ (for orchestrated deployment)

#### Python Dependencies

```bash
# Core packages
python>=3.8
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
gradio>=3.0.0

# AI/ML packages
torch>=1.12.0
transformers>=4.20.0
sentence-transformers>=2.2.0
langchain>=0.0.200
openai>=0.27.0
anthropic>=0.3.0

# Infrastructure packages
fastapi>=0.85.0
uvicorn>=0.18.0
pydantic>=1.10.0
sqlalchemy>=1.4.0
redis>=4.3.0
celery>=5.2.0

# Monitoring packages
prometheus-client>=0.14.0
grafana-api>=1.0.0
psutil>=5.9.0
```

### API Keys and Credentials

#### Required API Keys

```bash
# OpenAI API
export OPENAI_API_KEY="your_openai_api_key"

# Anthropic API
export ANTHROPIC_API_KEY="your_anthropic_api_key"

# Optional: Other model providers
export HUGGINGFACE_API_KEY="your_huggingface_api_key"
export COHERE_API_KEY="your_cohere_api_key"
```

#### Database Configuration

```bash
# PostgreSQL (recommended for production)
export DATABASE_URL="postgresql://user:password@localhost:5432/aaitc_db"

# Redis (for caching and task queue)
export REDIS_URL="redis://localhost:6379/0"

# Optional: Vector database
export PINECONE_API_KEY="your_pinecone_api_key"
export WEAVIATE_URL="http://localhost:8080"
```

---

## Development Deployment

### Local Development Setup

#### 1. Clone Repository

```bash
git clone <repository-url>
cd lenovo_aaitc_solutions
```

#### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

#### 4. Environment Configuration

```bash
# Copy environment template
cp .env.template .env

# Edit environment variables
nano .env
```

#### 5. Initialize Database

```bash
# Run database migrations
python -m alembic upgrade head

# Initialize default data
python scripts/init_database.py
```

#### 6. Start Development Server

```bash
# Start Gradio application
python -m gradio_app.main

# Or start with specific configuration
python -m gradio_app.main --host 0.0.0.0 --port 7860 --mcp-server
```

### Development Configuration

#### Environment Variables

```bash
# Development settings
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# API configuration
API_HOST=0.0.0.0
API_PORT=7860
MCP_SERVER_PORT=8000

# Database configuration
DATABASE_URL=sqlite:///./dev.db
REDIS_URL=redis://localhost:6379/1

# Model configuration
DEFAULT_MODEL_PROVIDER=openai
MODEL_CACHE_SIZE=1000
EVALUATION_TIMEOUT=300
```

#### Development Tools

```bash
# Install development tools
pip install black isort flake8 pytest pytest-cov

# Code formatting
black lenovo_aaitc_solutions/
isort lenovo_aaitc_solutions/

# Linting
flake8 lenovo_aaitc_solutions/

# Testing
pytest tests/ --cov=lenovo_aaitc_solutions/
```

---

## Production Deployment

### Production Architecture

#### Recommended Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   API Gateway   │    │   Web Server    │
│   (nginx/HAProxy)│────│   (Kong/Traefik)│────│   (Gradio/FastAPI)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │   Application   │
                       │   Services      │
                       └─────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Model         │    │   Database      │    │   Cache/Queue   │
│   Evaluation    │    │   (PostgreSQL)  │    │   (Redis)       │
│   Service       │    └─────────────────┘    └─────────────────┘
└─────────────────┘
        │
┌─────────────────┐
│   AI Architecture│
│   Service       │
└─────────────────┘
```

### Production Configuration

#### Environment Variables

```bash
# Production settings
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Security
SECRET_KEY=your_secret_key_here
JWT_SECRET=your_jwt_secret_here
ENCRYPTION_KEY=your_encryption_key_here

# API configuration
API_HOST=0.0.0.0
API_PORT=7860
MCP_SERVER_PORT=8000
MAX_WORKERS=4

# Database configuration
DATABASE_URL=postgresql://user:password@db_host:5432/aaitc_prod
REDIS_URL=redis://redis_host:6379/0

# Monitoring
PROMETHEUS_ENDPOINT=http://prometheus:9090
GRAFANA_ENDPOINT=http://grafana:3000
```

#### Production Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt
pip install gunicorn uvicorn[standard]

# Install monitoring tools
pip install prometheus-client grafana-api
```

### Deployment Steps

#### 1. System Preparation

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3.8 python3.8-venv python3.8-dev
sudo apt install -y postgresql postgresql-contrib redis-server
sudo apt install -y nginx certbot python3-certbot-nginx

# Create application user
sudo useradd -m -s /bin/bash aaitc
sudo usermod -aG sudo aaitc
```

#### 2. Application Deployment

```bash
# Switch to application user
sudo su - aaitc

# Clone and setup application
git clone <repository-url> /home/aaitc/lenovo_aaitc_solutions
cd /home/aaitc/lenovo_aaitc_solutions

# Create virtual environment
python3.8 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure application
cp .env.production .env
nano .env  # Edit configuration
```

#### 3. Database Setup

```bash
# Create database
sudo -u postgres createdb aaitc_prod
sudo -u postgres createuser aaitc_user
sudo -u postgres psql -c "ALTER USER aaitc_user PASSWORD 'secure_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE aaitc_prod TO aaitc_user;"

# Run migrations
python -m alembic upgrade head
```

#### 4. Service Configuration

```bash
# Create systemd service
sudo nano /etc/systemd/system/aaitc.service
```

```ini
[Unit]
Description=Lenovo AAITC Solutions
After=network.target postgresql.service redis.service

[Service]
Type=exec
User=aaitc
Group=aaitc
WorkingDirectory=/home/aaitc/lenovo_aaitc_solutions
Environment=PATH=/home/aaitc/lenovo_aaitc_solutions/venv/bin
ExecStart=/home/aaitc/lenovo_aaitc_solutions/venv/bin/python -m gradio_app.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### 5. Nginx Configuration

```bash
# Create nginx configuration
sudo nano /etc/nginx/sites-available/aaitc
```

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /mcp/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

#### 6. SSL Certificate

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/aaitc /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com
```

#### 7. Start Services

```bash
# Start application service
sudo systemctl enable aaitc
sudo systemctl start aaitc

# Check status
sudo systemctl status aaitc
```

---

## Docker Deployment

### Docker Configuration

#### Dockerfile

```dockerfile
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 aaitc && chown -R aaitc:aaitc /app
USER aaitc

# Expose ports
EXPOSE 7860 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start application
CMD ["python", "-m", "gradio_app.main", "--host", "0.0.0.0", "--port", "7860"]
```

#### Docker Compose

```yaml
version: "3.8"

services:
  app:
    build: .
    ports:
      - "7860:7860"
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://aaitc:password@db:5432/aaitc
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    restart: unless-stopped

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=aaitc
      - POSTGRES_USER=aaitc
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:6-alpine
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
  postgres_data:
  redis_data:
```

### Docker Deployment Commands

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f app

# Scale application
docker-compose up -d --scale app=3

# Update application
docker-compose pull
docker-compose up -d

# Stop services
docker-compose down
```

---

## Kubernetes Deployment

### Kubernetes Manifests

#### Namespace

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: aaitc
```

#### ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: aaitc-config
  namespace: aaitc
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  API_HOST: "0.0.0.0"
  API_PORT: "7860"
  MCP_SERVER_PORT: "8000"
```

#### Secret

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: aaitc-secrets
  namespace: aaitc
type: Opaque
data:
  OPENAI_API_KEY: <base64-encoded-key>
  ANTHROPIC_API_KEY: <base64-encoded-key>
  DATABASE_URL: <base64-encoded-url>
  REDIS_URL: <base64-encoded-url>
```

#### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aaitc-app
  namespace: aaitc
spec:
  replicas: 3
  selector:
    matchLabels:
      app: aaitc-app
  template:
    metadata:
      labels:
        app: aaitc-app
    spec:
      containers:
        - name: aaitc
          image: lenovo/aaitc:latest
          ports:
            - containerPort: 7860
            - containerPort: 8000
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: aaitc-secrets
                  key: DATABASE_URL
            - name: REDIS_URL
              valueFrom:
                secretKeyRef:
                  name: aaitc-secrets
                  key: REDIS_URL
          envFrom:
            - configMapRef:
                name: aaitc-config
            - secretRef:
                name: aaitc-secrets
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
```

#### Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: aaitc-service
  namespace: aaitc
spec:
  selector:
    app: aaitc-app
  ports:
    - name: http
      port: 80
      targetPort: 7860
    - name: mcp
      port: 8000
      targetPort: 8000
  type: ClusterIP
```

#### Ingress

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: aaitc-ingress
  namespace: aaitc
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
    - hosts:
        - your-domain.com
      secretName: aaitc-tls
  rules:
    - host: your-domain.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: aaitc-service
                port:
                  number: 80
```

### Kubernetes Deployment Commands

```bash
# Apply manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n aaitc
kubectl get services -n aaitc
kubectl get ingress -n aaitc

# View logs
kubectl logs -f deployment/aaitc-app -n aaitc

# Scale deployment
kubectl scale deployment aaitc-app --replicas=5 -n aaitc

# Update deployment
kubectl set image deployment/aaitc-app aaitc=lenovo/aaitc:v1.1 -n aaitc
```

---

## Cloud Deployment

### AWS Deployment

#### ECS with Fargate

```yaml
# task-definition.json
{
  "family": "aaitc-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions":
    [
      {
        "name": "aaitc",
        "image": "your-account.dkr.ecr.region.amazonaws.com/aaitc:latest",
        "portMappings":
          [
            { "containerPort": 7860, "protocol": "tcp" },
            { "containerPort": 8000, "protocol": "tcp" },
          ],
        "environment": [{ "name": "ENVIRONMENT", "value": "production" }],
        "secrets":
          [
            {
              "name": "OPENAI_API_KEY",
              "valueFrom": "arn:aws:secretsmanager:region:account:secret:aaitc/openai-api-key",
            },
          ],
        "logConfiguration":
          {
            "logDriver": "awslogs",
            "options":
              {
                "awslogs-group": "/ecs/aaitc",
                "awslogs-region": "us-west-2",
                "awslogs-stream-prefix": "ecs",
              },
          },
      },
    ],
}
```

### Google Cloud Deployment

#### Cloud Run

```yaml
# cloud-run.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: aaitc-service
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-throttling: "false"
    spec:
      containerConcurrency: 100
      timeoutSeconds: 300
      containers:
        - image: gcr.io/your-project/aaitc:latest
          ports:
            - containerPort: 7860
          env:
            - name: ENVIRONMENT
              value: "production"
          resources:
            limits:
              cpu: "2000m"
              memory: "4Gi"
```

### Azure Deployment

#### Container Instances

```yaml
# azure-container-instance.yaml
apiVersion: 2018-10-01
location: eastus
name: aaitc-container
properties:
  containers:
    - name: aaitc
      properties:
        image: your-registry.azurecr.io/aaitc:latest
        ports:
          - port: 7860
            protocol: TCP
          - port: 8000
            protocol: TCP
        environmentVariables:
          - name: ENVIRONMENT
            value: production
        resources:
          requests:
            cpu: 2
            memoryInGb: 4
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
      - protocol: TCP
        port: 7860
      - protocol: TCP
        port: 8000
```

---

## Monitoring and Maintenance

### Monitoring Setup

#### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "aaitc"
    static_configs:
      - targets: ["aaitc-app:7860"]
    metrics_path: /metrics
    scrape_interval: 30s
```

#### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Lenovo AAITC Solutions",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

### Health Checks

#### Application Health Endpoints

```python
# Health check endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/ready")
async def readiness_check():
    # Check database connection
    # Check Redis connection
    # Check external APIs
    return {"status": "ready", "checks": {"db": "ok", "redis": "ok"}}
```

### Backup and Recovery

#### Database Backup

```bash
#!/bin/bash
# backup-database.sh

BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/aaitc_backup_$DATE.sql"

# Create backup
pg_dump $DATABASE_URL > $BACKUP_FILE

# Compress backup
gzip $BACKUP_FILE

# Upload to S3 (optional)
aws s3 cp $BACKUP_FILE.gz s3://your-backup-bucket/

# Cleanup old backups (keep last 7 days)
find $BACKUP_DIR -name "aaitc_backup_*.sql.gz" -mtime +7 -delete
```

#### Application Backup

```bash
#!/bin/bash
# backup-application.sh

BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
APP_DIR="/home/aaitc/lenovo_aaitc_solutions"

# Create application backup
tar -czf $BACKUP_DIR/aaitc_app_$DATE.tar.gz -C $APP_DIR .

# Upload to S3
aws s3 cp $BACKUP_DIR/aaitc_app_$DATE.tar.gz s3://your-backup-bucket/
```

### Log Management

#### Log Rotation

```bash
# /etc/logrotate.d/aaitc
/home/aaitc/lenovo_aaitc_solutions/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 aaitc aaitc
    postrotate
        systemctl reload aaitc
    endscript
}
```

---

## Troubleshooting

### Common Issues

#### 1. Application Won't Start

```bash
# Check logs
journalctl -u aaitc -f

# Check configuration
python -c "from gradio_app.main import app; print('Config OK')"

# Check dependencies
pip check
```

#### 2. Database Connection Issues

```bash
# Test database connection
python -c "import psycopg2; psycopg2.connect('$DATABASE_URL')"

# Check database status
sudo systemctl status postgresql

# Check database logs
sudo tail -f /var/log/postgresql/postgresql-13-main.log
```

#### 3. Memory Issues

```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head

# Monitor memory in real-time
htop

# Check for memory leaks
python -c "import tracemalloc; tracemalloc.start(); # your code here"
```

#### 4. Performance Issues

```bash
# Check CPU usage
top
htop

# Check disk I/O
iotop

# Check network connections
netstat -tulpn | grep :7860
```

### Debug Mode

#### Enable Debug Logging

```bash
# Set debug environment
export DEBUG=true
export LOG_LEVEL=DEBUG

# Restart application
sudo systemctl restart aaitc
```

#### Performance Profiling

```python
# Add to application code
import cProfile
import pstats

def profile_function(func):
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()

        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(10)

        return result
    return wrapper
```

### Support and Resources

#### Documentation

- [API Documentation](API_DOCUMENTATION.md)
- [User Guide](USER_GUIDE.md)
- [Configuration Reference](CONFIG_REFERENCE.md)

#### Community Support

- GitHub Issues: [Repository Issues](https://github.com/lenovo/aaitc-solutions/issues)
- Discussions: [GitHub Discussions](https://github.com/lenovo/aaitc-solutions/discussions)
- Email: aaitc-support@lenovo.com

#### Professional Support

- Enterprise Support: enterprise-support@lenovo.com
- Consulting Services: consulting@lenovo.com
- Training Programs: training@lenovo.com

---

This deployment guide provides comprehensive instructions for deploying the Lenovo AAITC Solutions framework in various environments. For additional support or specific deployment scenarios, please contact the Lenovo AAITC team.
