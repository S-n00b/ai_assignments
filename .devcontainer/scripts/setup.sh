#!/bin/bash
set -euo pipefail

echo "🚀 Setting up AI Platform Development Environment..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Navigate to workspace
cd /workspace

# Create necessary directories
print_step "Creating directory structure..."
mkdir -p data/{chromadb,neo4j,mlflow,duckdb}
mkdir -p logs/{services,applications}
mkdir -p containers/{fastapi,gradio,mlflow,chromadb,neo4j,gateway}
mkdir -p frontend/{js,css,assets}
mkdir -p scripts
print_success "Directory structure created"

# Set up environment variables
print_step "Setting up environment variables..."
if [ ! -f .env ]; then
    cat > .env << EOF
# Codespaces Environment
CODESPACE_NAME=${CODESPACE_NAME:-localhost}
GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN=${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:-}

# Service Ports
FASTAPI_PORT=8080
GRADIO_PORT=7860
MLFLOW_PORT=5000
CHROMADB_PORT=8081
NEO4J_BOLT_PORT=7687
NEO4J_HTTP_PORT=7474
MKDOCS_PORT=8082
GATEWAY_PORT=8000

# Database Configuration
NEO4J_AUTH=neo4j/password123
MLFLOW_BACKEND_STORE_URI=sqlite:///data/mlflow/mlflow.db
MLFLOW_DEFAULT_ARTIFACT_ROOT=/workspace/data/mlflow/artifacts

# API Keys (to be configured)
GITHUB_TOKEN=${GITHUB_TOKEN:-}
OPENAI_API_KEY=${OPENAI_API_KEY:-}
EOF
    print_success "Environment file created"
else
    print_warning "Environment file already exists"
fi

# Install Python dependencies
print_step "Installing Python dependencies..."
if [ -f config/requirements.txt ]; then
    pip install -r config/requirements.txt
    print_success "Python dependencies installed"
else
    print_warning "requirements.txt not found, skipping Python dependencies"
fi

# Create container build script
print_step "Creating container build script..."
cat > scripts/build-containers.sh << 'EOF'
#!/bin/bash
set -euo pipefail

echo "🐳 Building container images..."

# Build base image
if [ -f containers/base/Containerfile ]; then
    echo "Building base image..."
    podman build -t localhost/ai-base:latest -f containers/base/Containerfile .
fi

# Build service images
for service in fastapi gradio mlflow chromadb neo4j gateway; do
    if [ -f containers/${service}/Containerfile ]; then
        echo "Building ${service} image..."
        podman build -t localhost/ai-${service}:latest -f containers/${service}/Containerfile .
    fi
done

echo "✅ All containers built successfully!"
EOF
chmod +x scripts/build-containers.sh
print_success "Container build script created"

# Create pod configuration
print_step "Creating Podman pod configuration..."
cat > podman-pod.yaml << EOF
apiVersion: v1
kind: Pod
metadata:
  name: ai-platform-pod
  labels:
    app: ai-platform
spec:
  containers:
  - name: gateway
    image: nginx:alpine
    ports:
    - containerPort: 80
      hostPort: 8000
    volumeMounts:
    - name: nginx-config
      mountPath: /etc/nginx/nginx.conf
      subPath: nginx.conf
  - name: fastapi
    image: localhost/ai-fastapi:latest
    ports:
    - containerPort: 8080
    env:
    - name: PYTHONPATH
      value: /app
    volumeMounts:
    - name: app-data
      mountPath: /app/data
  - name: gradio
    image: localhost/ai-gradio:latest
    ports:
    - containerPort: 7860
    volumeMounts:
    - name: app-data
      mountPath: /app/data
  - name: mlflow
    image: mlflow/mlflow:latest
    ports:
    - containerPort: 5000
    command: ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
    volumeMounts:
    - name: mlflow-data
      mountPath: /mlflow
  volumes:
  - name: nginx-config
    hostPath:
      path: /workspace/containers/gateway
  - name: app-data
    hostPath:
      path: /workspace/data
  - name: mlflow-data
    hostPath:
      path: /workspace/data/mlflow
EOF
print_success "Podman pod configuration created"

# Create a simple NGINX configuration for the gateway
print_step "Creating NGINX gateway configuration..."
mkdir -p containers/gateway
cat > containers/gateway/nginx.conf << 'EOF'
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
        add_header 'Access-Control-Allow-Origin' '*' always;
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
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /gradio/ {
            proxy_pass http://gradio/;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
        }
        
        location /mlflow/ {
            proxy_pass http://mlflow/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
        
        location /chromadb/ {
            proxy_pass http://chromadb/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
        
        # Health check endpoint
        location /health {
            return 200 '{"status": "healthy"}';
            add_header Content-Type application/json;
        }
    }
}
EOF
print_success "NGINX configuration created"

# Set up Git hooks
print_step "Setting up Git hooks..."
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Run linting before commit
black --check src/
isort --check-only src/
flake8 src/
EOF
chmod +x .git/hooks/pre-commit
print_success "Git hooks configured"

# Create VS Code workspace settings
print_step "Creating VS Code workspace settings..."
mkdir -p .vscode
cat > .vscode/settings.json << EOF
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "files.autoSave": "afterDelay",
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/node_modules": true,
        "**/.pytest_cache": true
    }
}
EOF
print_success "VS Code settings created"

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "📝 Next steps:"
echo "  1. Build containers: ./scripts/build-containers.sh"
echo "  2. Start services: .devcontainer/scripts/start-services.sh"
echo "  3. Access services:"
echo "     - API Gateway: http://localhost:8000"
echo "     - FastAPI Docs: http://localhost:8000/api/docs"
echo "     - Gradio App: http://localhost:8000/gradio"
echo "     - MLflow UI: http://localhost:8000/mlflow"
echo ""