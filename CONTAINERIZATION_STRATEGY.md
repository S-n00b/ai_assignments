# 🚀 Containerization & GitHub Integration Strategy

## **Executive Summary**

This strategy transforms the current multi-service AI platform into a clean, containerized solution running on GitHub Codespaces with a GitHub Pages frontend. We'll use Podman pods to orchestrate services and implement an API gateway for secure communication between the static frontend and backend services.

## **Phase 1: Codebase Cleanup & Restructuring** 🧹

### 1.1 Repository Structure Reorganization
```
ai_assignments/
├── .devcontainer/              # GitHub Codespaces configuration
│   ├── devcontainer.json       # Main dev container config
│   ├── Containerfile.base      # Base container image
│   └── scripts/                # Setup scripts
├── containers/                 # Service container definitions
│   ├── fastapi/
│   │   ├── Containerfile
│   │   └── config/
│   ├── gradio/
│   │   ├── Containerfile
│   │   └── config/
│   ├── mlflow/
│   │   ├── Containerfile
│   │   └── config/
│   ├── chromadb/
│   │   ├── Containerfile
│   │   └── config/
│   ├── neo4j/
│   │   ├── Containerfile
│   │   └── config/
│   └── gateway/                # API Gateway service
│       ├── Containerfile
│       └── nginx.conf
├── frontend/                   # GitHub Pages static site
│   ├── index.html             # enhanced_unified_platform.html
│   ├── js/
│   │   ├── api-client.js      # API client library
│   │   └── service-config.js  # Service configuration
│   └── css/
├── scripts/                    # Utility scripts
│   ├── cleanup.sh             # Cleanup script
│   ├── build-containers.sh    # Container build script
│   └── start-services.sh      # Service startup script
└── docs/                      # Documentation
```

### 1.2 Cleanup Actions

```bash
#!/bin/bash
# cleanup.sh - Comprehensive cleanup script

# Remove unused dependencies
pip-autoremove
npm prune

# Clean up Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Remove old logs
find logs/ -type f -mtime +7 -delete

# Clean up unused Docker/Podman images
podman image prune -a -f

# Remove duplicate code files
fdupes -r -d -N src/

# Format all Python files
black src/ --line-length 100
isort src/

# Lint JavaScript/TypeScript
eslint frontend/ --fix

# Update dependencies
pip-compile requirements.in -o requirements.txt
```

## **Phase 2: Create Podman Container Configurations** 🐳

### 2.1 Base Container Configuration

```dockerfile
# containers/base/Containerfile
FROM python:3.11-slim

# Install common dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
WORKDIR /app
COPY requirements-base.txt .
RUN pip install --no-cache-dir -r requirements-base.txt
```

### 2.2 FastAPI Service Container

```dockerfile
# containers/fastapi/Containerfile
FROM localhost/ai-base:latest

WORKDIR /app
COPY src/enterprise_llmops /app/enterprise_llmops
COPY src/model_evaluation /app/model_evaluation

# Install specific dependencies
COPY requirements-fastapi.txt .
RUN pip install --no-cache-dir -r requirements-fastapi.txt

# Configure environment
ENV PYTHONPATH=/app
ENV PORT=8080

EXPOSE 8080
CMD ["uvicorn", "enterprise_llmops.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### 2.3 Podman Pod Configuration

```yaml
# podman-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: ai-platform-pod
spec:
  containers:
  - name: gateway
    image: localhost/ai-gateway:latest
    ports:
    - containerPort: 80
      hostPort: 8000
  - name: fastapi
    image: localhost/ai-fastapi:latest
    ports:
    - containerPort: 8080
  - name: gradio
    image: localhost/ai-gradio:latest
    ports:
    - containerPort: 7860
  - name: mlflow
    image: localhost/ai-mlflow:latest
    ports:
    - containerPort: 5000
  - name: chromadb
    image: localhost/ai-chromadb:latest
    ports:
    - containerPort: 8081
  - name: neo4j
    image: localhost/ai-neo4j:latest
    ports:
    - containerPort: 7687
      containerPort: 7474
```

## **Phase 3: Setup GitHub Codespaces DevContainer** 💻

### 3.1 DevContainer Configuration

```json
{
  "name": "AI Platform Development",
  "build": {
    "dockerfile": "Containerfile.base",
    "context": ".."
  },
  "features": {
    "ghcr.io/devcontainers/features/docker-in-docker:2": {
      "version": "latest",
      "dockerDashComposeVersion": "v2"
    },
    "ghcr.io/devcontainers/features/python:1": {
      "version": "3.11"
    },
    "ghcr.io/devcontainers/features/node:1": {
      "version": "18"
    }
  },
  "forwardPorts": [8000, 8080, 7860, 5000, 8081, 7687, 7474],
  "portsAttributes": {
    "8000": {
      "label": "API Gateway",
      "onAutoForward": "notify"
    },
    "8080": {
      "label": "FastAPI",
      "onAutoForward": "silent"
    },
    "7860": {
      "label": "Gradio",
      "onAutoForward": "silent"
    }
  },
  "postCreateCommand": ".devcontainer/scripts/setup.sh",
  "postStartCommand": ".devcontainer/scripts/start-services.sh",
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "dbaeumer.vscode-eslint",
        "esbenp.prettier-vscode"
      ]
    }
  }
}
```

### 3.2 Setup Script

```bash
#!/bin/bash
# .devcontainer/scripts/setup.sh

# Install Podman
sudo apt-get update
sudo apt-get install -y podman podman-compose

# Build all containers
cd /workspace
./scripts/build-containers.sh

# Initialize databases
podman run --rm -v $(pwd)/data:/data localhost/ai-init:latest

# Set up environment variables
cp .env.example .env.codespaces
echo "CODESPACE_NAME=${CODESPACE_NAME}" >> .env.codespaces
echo "GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN=${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN}" >> .env.codespaces
```

## **Phase 4: Configure API Gateway & Service Discovery** 🌐

### 4.1 NGINX API Gateway Configuration

```nginx
# containers/gateway/nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream fastapi {
        server localhost:8080;
    }
    
    upstream gradio {
        server localhost:7860;
    }
    
    upstream mlflow {
        server localhost:5000;
    }
    
    upstream chromadb {
        server localhost:8081;
    }

    server {
        listen 80;
        
        # CORS headers for GitHub Pages
        add_header 'Access-Control-Allow-Origin' 'https://s-n00b.github.io' always;
        add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE, OPTIONS' always;
        add_header 'Access-Control-Allow-Headers' 'Authorization, Content-Type' always;
        add_header 'Access-Control-Allow-Credentials' 'true' always;
        
        # Handle preflight requests
        if ($request_method = 'OPTIONS') {
            return 204;
        }
        
        # API routes
        location /api/ {
            proxy_pass http://fastapi/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
        
        location /gradio/ {
            proxy_pass http://gradio/;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
        
        location /mlflow/ {
            proxy_pass http://mlflow/;
        }
        
        location /chromadb/ {
            proxy_pass http://chromadb/;
        }
    }
}
```

### 4.2 Service Discovery Configuration

```javascript
// frontend/js/service-config.js
const ServiceConfig = {
    getBaseUrl() {
        // Detect if running on GitHub Pages or Codespaces
        if (window.location.hostname === 's-n00b.github.io') {
            // GitHub Pages - use Codespaces URL from environment
            return `https://${CODESPACE_NAME}-8000.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN}`;
        } else if (window.location.hostname.includes('github.dev')) {
            // GitHub Codespaces preview
            return window.location.origin.replace('-5500', '-8000'); // Replace preview port with gateway port
        } else {
            // Local development
            return 'http://localhost:8000';
        }
    },
    
    services: {
        fastapi: '/api',
        gradio: '/gradio',
        mlflow: '/mlflow',
        chromadb: '/chromadb'
    }
};
```

## **Phase 5: GitHub Pages Integration Strategy** 📄

### 5.1 Enhanced Unified Platform Updates

```javascript
// frontend/js/api-client.js
class AIServiceClient {
    constructor() {
        this.baseUrl = ServiceConfig.getBaseUrl();
        this.headers = {
            'Content-Type': 'application/json'
        };
    }
    
    async checkHealth() {
        try {
            const response = await fetch(`${this.baseUrl}/api/health`);
            return await response.json();
        } catch (error) {
            console.error('Service health check failed:', error);
            return { status: 'offline', error: error.message };
        }
    }
    
    async callModel(modelName, prompt) {
        const response = await fetch(`${this.baseUrl}/api/models/${modelName}/generate`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({ prompt })
        });
        return await response.json();
    }
    
    async startFineTuning(config) {
        const response = await fetch(`${this.baseUrl}/api/fine-tuning/start`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify(config)
        });
        return await response.json();
    }
}
```

### 5.2 GitHub Actions Deployment

```yaml
# .github/workflows/deploy-frontend.yml
name: Deploy to GitHub Pages

on:
  push:
    branches: [main]
    paths:
      - 'frontend/**'
      - '.github/workflows/deploy-frontend.yml'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
      
      - name: Build frontend
        run: |
          cd frontend
          npm install
          npm run build
          
      - name: Configure for GitHub Pages
        run: |
          echo "window.CODESPACE_NAME = '${{ secrets.CODESPACE_NAME }}';" > frontend/config.js
          echo "window.GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN = '${{ secrets.CODESPACES_DOMAIN }}';" >> frontend/config.js
          
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./frontend
```

## **Phase 6: CI/CD Pipeline Setup** 🔄

### 6.1 Container Build Pipeline

```yaml
# .github/workflows/build-containers.yml
name: Build and Push Containers

on:
  push:
    branches: [main]
    paths:
      - 'containers/**'
      - 'src/**'

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        service: [fastapi, gradio, mlflow, chromadb, neo4j, gateway]
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Login to GitHub Container Registry
        uses: docker/login-action@v2
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Build and push container
        run: |
          cd containers/${{ matrix.service }}
          podman build -t ghcr.io/${{ github.repository }}/${{ matrix.service }}:latest .
          podman push ghcr.io/${{ github.repository }}/${{ matrix.service }}:latest
```

### 6.2 Automated Testing

```yaml
# .github/workflows/test-services.yml
name: Test Services

on:
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
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements-test.txt
          
      - name: Run tests
        run: |
          pytest tests/ -v --cov=src --cov-report=xml
          
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## **Implementation Timeline**

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| Phase 1 | 2 days | Clean repository structure, removed redundancies |
| Phase 2 | 3 days | All services containerized with Podman |
| Phase 3 | 2 days | GitHub Codespaces fully configured |
| Phase 4 | 2 days | API Gateway operational with CORS |
| Phase 5 | 2 days | GitHub Pages integrated with services |
| Phase 6 | 1 day | CI/CD pipelines operational |

## **Security Considerations**

1. **Authentication**: Implement JWT-based authentication in the API Gateway
2. **Rate Limiting**: Configure rate limits to prevent abuse
3. **HTTPS**: Ensure all communications use HTTPS
4. **Secrets Management**: Use GitHub Secrets for sensitive configuration
5. **Network Isolation**: Services communicate only through the gateway

## **Success Metrics**

- ✅ All services running in a single Podman pod
- ✅ GitHub Codespaces setup time < 5 minutes
- ✅ Frontend loads from GitHub Pages successfully
- ✅ All API calls work with proper CORS headers
- ✅ Automated builds and deployments functional
- ✅ Documentation complete and accurate

---

**Last Updated**: January 2025  
**Version**: 1.0  
**Status**: Implementation Ready