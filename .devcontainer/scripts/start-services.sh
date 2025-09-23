#!/bin/bash
set -euo pipefail

echo "🚀 Starting AI Platform Services..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Load environment variables
if [ -f /workspace/.env ]; then
    export $(cat /workspace/.env | grep -v '^#' | xargs)
fi

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to wait for a service to be ready
wait_for_service() {
    local service_name=$1
    local port=$2
    local max_attempts=30
    local attempt=0
    
    echo -n "Waiting for $service_name to be ready"
    while [ $attempt -lt $max_attempts ]; do
        if check_port $port; then
            echo ""
            print_success "$service_name is ready on port $port"
            return 0
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    echo ""
    print_error "$service_name failed to start on port $port"
    return 1
}

# Stop any existing pod
print_step "Stopping existing services..."
podman pod stop ai-platform-pod 2>/dev/null || true
podman pod rm ai-platform-pod 2>/dev/null || true
print_success "Existing services stopped"

# Create and start the pod
print_step "Creating AI platform pod..."
podman pod create --name ai-platform-pod \
    -p 8000:80 \
    -p 8080:8080 \
    -p 7860:7860 \
    -p 5000:5000 \
    -p 8081:8081 \
    -p 7687:7687 \
    -p 7474:7474 \
    -p 8082:8082
print_success "Pod created"

# Start NGINX Gateway
print_step "Starting NGINX Gateway..."
podman run -d --pod ai-platform-pod \
    --name gateway \
    -v /workspace/containers/gateway/nginx.conf:/etc/nginx/nginx.conf:ro \
    nginx:alpine
wait_for_service "NGINX Gateway" 8000

# Start MLflow
print_step "Starting MLflow..."
podman run -d --pod ai-platform-pod \
    --name mlflow \
    -v /workspace/data/mlflow:/mlflow \
    -e MLFLOW_BACKEND_STORE_URI=sqlite:////mlflow/mlflow.db \
    -e MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts \
    ghcr.io/mlflow/mlflow:latest \
    mlflow server --host 0.0.0.0 --port 5000
wait_for_service "MLflow" 5000

# Start ChromaDB
print_step "Starting ChromaDB..."
podman run -d --pod ai-platform-pod \
    --name chromadb \
    -v /workspace/data/chromadb:/chroma/chroma \
    ghcr.io/chroma-core/chroma:latest
wait_for_service "ChromaDB" 8081

# Start Neo4j
print_step "Starting Neo4j..."
podman run -d --pod ai-platform-pod \
    --name neo4j \
    -v /workspace/data/neo4j:/data \
    -e NEO4J_AUTH=neo4j/password123 \
    -e NEO4J_PLUGINS='["apoc", "graph-data-science"]' \
    neo4j:5-community
wait_for_service "Neo4j" 7474

# Start FastAPI (if image exists)
if podman image exists localhost/ai-fastapi:latest; then
    print_step "Starting FastAPI Platform..."
    podman run -d --pod ai-platform-pod \
        --name fastapi \
        -v /workspace:/app \
        -e PYTHONPATH=/app \
        -e ENVIRONMENT=codespaces \
        localhost/ai-fastapi:latest
    wait_for_service "FastAPI" 8080
else
    print_warning "FastAPI image not found. Build it with: ./scripts/build-containers.sh"
    # Start a simple FastAPI server as fallback
    print_step "Starting FastAPI in development mode..."
    cd /workspace
    nohup python -m uvicorn src.enterprise_llmops.main:app --host 0.0.0.0 --port 8080 --reload > logs/fastapi.log 2>&1 &
    wait_for_service "FastAPI" 8080
fi

# Start Gradio (if image exists)
if podman image exists localhost/ai-gradio:latest; then
    print_step "Starting Gradio App..."
    podman run -d --pod ai-platform-pod \
        --name gradio \
        -v /workspace:/app \
        -e PYTHONPATH=/app \
        localhost/ai-gradio:latest
    wait_for_service "Gradio" 7860
else
    print_warning "Gradio image not found. Build it with: ./scripts/build-containers.sh"
    # Start Gradio in development mode
    print_step "Starting Gradio in development mode..."
    cd /workspace
    nohup python -m src.gradio_app.main --host 0.0.0.0 --port 7860 > logs/gradio.log 2>&1 &
    wait_for_service "Gradio" 7860
fi

# Start MkDocs
print_step "Starting MkDocs..."
cd /workspace/docs
nohup mkdocs serve --dev-addr 0.0.0.0:8082 > /workspace/logs/mkdocs.log 2>&1 &
wait_for_service "MkDocs" 8082

# Display service status
echo ""
echo "📊 Service Status:"
echo "=================="
podman pod ps
echo ""
podman ps --filter "pod=ai-platform-pod" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Display access URLs
echo ""
echo "🌐 Access URLs:"
echo "==============="
if [ -n "${CODESPACE_NAME:-}" ]; then
    echo "API Gateway: https://${CODESPACE_NAME}-8000.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN}"
    echo "FastAPI Docs: https://${CODESPACE_NAME}-8000.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN}/api/docs"
    echo "Gradio App: https://${CODESPACE_NAME}-8000.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN}/gradio"
    echo "MLflow UI: https://${CODESPACE_NAME}-8000.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN}/mlflow"
    echo "Neo4j Browser: https://${CODESPACE_NAME}-7474.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN}"
    echo "MkDocs: https://${CODESPACE_NAME}-8082.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN}"
else
    echo "API Gateway: http://localhost:8000"
    echo "FastAPI Docs: http://localhost:8000/api/docs"
    echo "Gradio App: http://localhost:8000/gradio"
    echo "MLflow UI: http://localhost:8000/mlflow"
    echo "Neo4j Browser: http://localhost:7474"
    echo "MkDocs: http://localhost:8082"
fi

echo ""
echo "✅ All services started successfully!"
echo ""
echo "📝 Tips:"
echo "  - View logs: podman logs <service-name>"
echo "  - Stop services: podman pod stop ai-platform-pod"
echo "  - Restart a service: podman restart <service-name>"
echo ""