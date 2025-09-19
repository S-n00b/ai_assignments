#!/bin/bash

# Enterprise LLMOps Platform Deployment Script
# This script deploys the complete enterprise LLMOps platform with all components

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="llmops-enterprise"
CLUSTER_NAME="llmops-cluster"
REGION="us-west-2"
PROJECT_ID="llmops-enterprise"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi
    
    # Check if helm is installed
    if ! command -v helm &> /dev/null; then
        log_error "helm is not installed. Please install helm first."
        exit 1
    fi
    
    # Check if terraform is installed
    if ! command -v terraform &> /dev/null; then
        log_error "terraform is not installed. Please install terraform first."
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed. Please install docker first."
        exit 1
    fi
    
    # Check if ollama is installed
    if ! command -v ollama &> /dev/null; then
        log_warning "ollama is not installed. Installing ollama..."
        curl -fsSL https://ollama.ai/install.sh | sh
    fi
    
    log_success "Prerequisites check completed"
}

setup_kubernetes_cluster() {
    log_info "Setting up Kubernetes cluster..."
    
    # Create namespace
    kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply namespace and resource quota
    kubectl apply -f infrastructure/kubernetes/namespace.yaml
    
    # Apply storage classes
    kubectl apply -f infrastructure/kubernetes/storage-classes.yaml
    
    log_success "Kubernetes cluster setup completed"
}

deploy_infrastructure() {
    log_info "Deploying infrastructure with Terraform..."
    
    cd infrastructure/terraform
    
    # Initialize Terraform
    terraform init
    
    # Plan deployment
    terraform plan -var="cluster_name=${CLUSTER_NAME}" -var="environment=production"
    
    # Apply infrastructure
    terraform apply -auto-approve -var="cluster_name=${CLUSTER_NAME}" -var="environment=production"
    
    cd ../..
    
    log_success "Infrastructure deployment completed"
}

deploy_monitoring_stack() {
    log_info "Deploying monitoring stack..."
    
    # Deploy Prometheus and Grafana
    kubectl apply -f infrastructure/kubernetes/monitoring.yaml
    
    # Wait for deployments to be ready
    kubectl wait --for=condition=available --timeout=300s deployment/prometheus -n ${NAMESPACE}
    kubectl wait --for=condition=available --timeout=300s deployment/grafana -n ${NAMESPACE}
    kubectl wait --for=condition=available --timeout=300s deployment/langfuse-server -n ${NAMESPACE}
    
    log_success "Monitoring stack deployment completed"
}

deploy_vector_databases() {
    log_info "Deploying vector databases..."
    
    # Deploy Chroma and Weaviate
    kubectl apply -f infrastructure/kubernetes/vector-databases.yaml
    
    # Wait for deployments to be ready
    kubectl wait --for=condition=available --timeout=300s deployment/chroma-server -n ${NAMESPACE}
    kubectl wait --for=condition=available --timeout=300s deployment/weaviate-server -n ${NAMESPACE}
    
    log_success "Vector databases deployment completed"
}

deploy_mlflow_stack() {
    log_info "Deploying MLflow stack..."
    
    # Deploy MLflow tracking server and UI
    kubectl apply -f infrastructure/kubernetes/mlflow-deployment.yaml
    
    # Wait for deployments to be ready
    kubectl wait --for=condition=available --timeout=300s deployment/mlflow-tracking -n ${NAMESPACE}
    kubectl wait --for=condition=available --timeout=300s deployment/postgres-mlflow -n ${NAMESPACE}
    kubectl wait --for=condition=available --timeout=300s deployment/minio -n ${NAMESPACE}
    
    log_success "MLflow stack deployment completed"
}

deploy_ollama() {
    log_info "Deploying Ollama..."
    
    # Deploy Ollama server
    kubectl apply -f infrastructure/kubernetes/ollama-deployment.yaml
    
    # Wait for deployment to be ready
    kubectl wait --for=condition=available --timeout=300s deployment/ollama-server -n ${NAMESPACE}
    
    # Pull default models
    log_info "Pulling default Ollama models..."
    
    # Get Ollama service endpoint
    OLLAMA_ENDPOINT=$(kubectl get service ollama-service -n ${NAMESPACE} -o jsonpath='{.spec.clusterIP}'):11434
    
    # Wait for Ollama to be ready
    sleep 30
    
    # Pull models
    kubectl run ollama-client --image=ollama/ollama:latest --rm -i --restart=Never -- \
        ollama pull llama3.1:8b
    
    kubectl run ollama-client --image=ollama/ollama:latest --rm -i --restart=Never -- \
        ollama pull codellama:7b
    
    kubectl run ollama-client --image=ollama/ollama:latest --rm -i --restart=Never -- \
        ollama pull mistral:7b
    
    log_success "Ollama deployment completed"
}

deploy_main_application() {
    log_info "Building and deploying main application..."
    
    # Build Docker image
    docker build -t llmops-enterprise:latest -f infrastructure/docker/Dockerfile .
    
    # Create deployment manifest
    cat > infrastructure/kubernetes/main-app-deployment.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llmops-frontend
  namespace: ${NAMESPACE}
  labels:
    app: llmops-frontend
    component: frontend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llmops-frontend
  template:
    metadata:
      labels:
        app: llmops-frontend
    spec:
      containers:
      - name: llmops-frontend
        image: llmops-enterprise:latest
        ports:
        - containerPort: 8080
          name: http
        env:
        - name: OLLAMA_HOST
          value: "ollama-service"
        - name: OLLAMA_PORT
          value: "11434"
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow-tracking-service:5000"
        - name: CHROMA_HOST
          value: "chroma-service"
        - name: CHROMA_PORT
          value: "8000"
        - name: WEAVIATE_HOST
          value: "weaviate-service"
        - name: WEAVIATE_PORT
          value: "8080"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: llmops-frontend-service
  namespace: ${NAMESPACE}
  labels:
    app: llmops-frontend
spec:
  type: ClusterIP
  ports:
  - port: 8080
    targetPort: 8080
    name: http
  selector:
    app: llmops-frontend
EOF
    
    # Apply deployment
    kubectl apply -f infrastructure/kubernetes/main-app-deployment.yaml
    
    # Wait for deployment to be ready
    kubectl wait --for=condition=available --timeout=300s deployment/llmops-frontend -n ${NAMESPACE}
    
    log_success "Main application deployment completed"
}

setup_ingress() {
    log_info "Setting up ingress..."
    
    # Apply ingress configuration
    kubectl apply -f infrastructure/kubernetes/ingress.yaml
    
    # Get ingress IP
    INGRESS_IP=$(kubectl get ingress llmops-ingress -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    if [ -z "$INGRESS_IP" ]; then
        log_warning "Ingress IP not available yet. Please check later."
    else
        log_success "Ingress setup completed. Access the application at: http://${INGRESS_IP}"
    fi
}

verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check all deployments
    kubectl get deployments -n ${NAMESPACE}
    
    # Check all services
    kubectl get services -n ${NAMESPACE}
    
    # Check all pods
    kubectl get pods -n ${NAMESPACE}
    
    # Test endpoints
    log_info "Testing endpoints..."
    
    # Test main application
    kubectl port-forward service/llmops-frontend-service 8080:8080 -n ${NAMESPACE} &
    PORT_FORWARD_PID=$!
    sleep 10
    
    if curl -f http://localhost:8080/health > /dev/null 2>&1; then
        log_success "Main application is healthy"
    else
        log_error "Main application health check failed"
    fi
    
    # Test Ollama
    kubectl port-forward service/ollama-service 11434:11434 -n ${NAMESPACE} &
    sleep 5
    
    if curl -f http://localhost:11434/api/tags > /dev/null 2>&1; then
        log_success "Ollama is healthy"
    else
        log_error "Ollama health check failed"
    fi
    
    # Cleanup port forwards
    kill $PORT_FORWARD_PID 2>/dev/null || true
    
    log_success "Deployment verification completed"
}

show_access_info() {
    log_info "Deployment completed successfully!"
    echo ""
    echo "=== Access Information ==="
    echo ""
    echo "Main Application:"
    echo "  URL: http://localhost:8080 (via port-forward)"
    echo "  Health: kubectl port-forward service/llmops-frontend-service 8080:8080 -n ${NAMESPACE}"
    echo ""
    echo "Ollama:"
    echo "  URL: http://localhost:11434 (via port-forward)"
    echo "  Health: kubectl port-forward service/ollama-service 11434:11434 -n ${NAMESPACE}"
    echo ""
    echo "MLflow:"
    echo "  URL: http://localhost:5000 (via port-forward)"
    echo "  Health: kubectl port-forward service/mlflow-tracking-service 5000:5000 -n ${NAMESPACE}"
    echo ""
    echo "Grafana:"
    echo "  URL: http://localhost:3000 (via port-forward)"
    echo "  Health: kubectl port-forward service/grafana-service 3000:3000 -n ${NAMESPACE}"
    echo ""
    echo "Prometheus:"
    echo "  URL: http://localhost:9090 (via port-forward)"
    echo "  Health: kubectl port-forward service/prometheus-service 9090:9090 -n ${NAMESPACE}"
    echo ""
    echo "LangFuse:"
    echo "  URL: http://localhost:3000 (via port-forward)"
    echo "  Health: kubectl port-forward service/langfuse-service 3000:3000 -n ${NAMESPACE}"
    echo ""
    echo "=== Useful Commands ==="
    echo ""
    echo "View logs:"
    echo "  kubectl logs -f deployment/llmops-frontend -n ${NAMESPACE}"
    echo ""
    echo "Scale deployment:"
    echo "  kubectl scale deployment llmops-frontend --replicas=5 -n ${NAMESPACE}"
    echo ""
    echo "Port forward all services:"
    echo "  kubectl port-forward service/llmops-frontend-service 8080:8080 -n ${NAMESPACE} &"
    echo "  kubectl port-forward service/ollama-service 11434:11434 -n ${NAMESPACE} &"
    echo "  kubectl port-forward service/mlflow-tracking-service 5000:5000 -n ${NAMESPACE} &"
    echo "  kubectl port-forward service/grafana-service 3000:3000 -n ${NAMESPACE} &"
    echo "  kubectl port-forward service/prometheus-service 9090:9090 -n ${NAMESPACE} &"
    echo ""
    echo "Delete deployment:"
    echo "  kubectl delete namespace ${NAMESPACE}"
    echo ""
}

# Main deployment function
main() {
    log_info "Starting Enterprise LLMOps Platform deployment..."
    
    # Parse command line arguments
    SKIP_TERRAFORM=false
    SKIP_MONITORING=false
    SKIP_VECTOR_DB=false
    SKIP_MLFLOW=false
    SKIP_OLLAMA=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-terraform)
                SKIP_TERRAFORM=true
                shift
                ;;
            --skip-monitoring)
                SKIP_MONITORING=true
                shift
                ;;
            --skip-vector-db)
                SKIP_VECTOR_DB=true
                shift
                ;;
            --skip-mlflow)
                SKIP_MLFLOW=true
                shift
                ;;
            --skip-ollama)
                SKIP_OLLAMA=true
                shift
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo ""
                echo "Options:"
                echo "  --skip-terraform     Skip Terraform infrastructure deployment"
                echo "  --skip-monitoring    Skip monitoring stack deployment"
                echo "  --skip-vector-db     Skip vector databases deployment"
                echo "  --skip-mlflow        Skip MLflow stack deployment"
                echo "  --skip-ollama        Skip Ollama deployment"
                echo "  --help               Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Run deployment steps
    check_prerequisites
    setup_kubernetes_cluster
    
    if [ "$SKIP_TERRAFORM" = false ]; then
        deploy_infrastructure
    fi
    
    if [ "$SKIP_MONITORING" = false ]; then
        deploy_monitoring_stack
    fi
    
    if [ "$SKIP_VECTOR_DB" = false ]; then
        deploy_vector_databases
    fi
    
    if [ "$SKIP_MLFLOW" = false ]; then
        deploy_mlflow_stack
    fi
    
    if [ "$SKIP_OLLAMA" = false ]; then
        deploy_ollama
    fi
    
    deploy_main_application
    setup_ingress
    verify_deployment
    show_access_info
    
    log_success "Enterprise LLMOps Platform deployment completed successfully!"
}

# Run main function
main "$@"
