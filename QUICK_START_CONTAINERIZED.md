# 🚀 Quick Start: Containerized AI Platform

## **Overview**

This guide will help you run the AI platform using GitHub Codespaces with Podman containers and connect it to a GitHub Pages frontend.

## **Prerequisites**

- GitHub account with Codespaces access
- Repository forked/cloned to your GitHub account
- Basic familiarity with terminal commands

## **Step 1: Launch GitHub Codespace**

1. Go to your repository on GitHub
2. Click the green "Code" button
3. Select "Codespaces" tab
4. Click "Create codespace on main"

The Codespace will automatically:
- Install all dependencies
- Set up Podman
- Configure the development environment

## **Step 2: Run Initial Setup**

Once your Codespace is ready, run:

```bash
# Make scripts executable (if needed)
chmod +x scripts/cleanup.sh .devcontainer/scripts/*.sh

# Clean up the codebase
./scripts/cleanup.sh

# The setup script runs automatically, but you can run it manually:
.devcontainer/scripts/setup.sh
```

## **Step 3: Build Containers (Optional)**

If you want to use containerized services:

```bash
# Build all container images
./scripts/build-containers.sh
```

## **Step 4: Start Services**

```bash
# Start all services
.devcontainer/scripts/start-services.sh
```

This will start:
- **API Gateway** (Port 8000) - NGINX reverse proxy with CORS
- **FastAPI Platform** (Port 8080) - Main backend API
- **Gradio App** (Port 7860) - Model evaluation interface
- **MLflow** (Port 5000) - Experiment tracking
- **ChromaDB** (Port 8081) - Vector database
- **Neo4j** (Port 7687/7474) - Graph database
- **MkDocs** (Port 8082) - Documentation

## **Step 5: Access Services**

In GitHub Codespaces, all ports are automatically forwarded. You'll see notifications for each service.

### Service URLs in Codespaces:
- **API Gateway**: `https://<codespace-name>-8000.app.github.dev`
- **FastAPI Docs**: `https://<codespace-name>-8000.app.github.dev/api/docs`
- **Gradio App**: `https://<codespace-name>-8000.app.github.dev/gradio`
- **MLflow UI**: `https://<codespace-name>-8000.app.github.dev/mlflow`

## **Step 6: Deploy Frontend to GitHub Pages**

1. Enable GitHub Pages in repository settings:
   - Go to Settings → Pages
   - Source: Deploy from a branch
   - Branch: `gh-pages` (will be created by workflow)

2. Update secrets for Codespace connection:
   - Go to Settings → Secrets and variables → Actions
   - Add `CODESPACE_NAME`: Your Codespace name
   - Add `CODESPACES_DOMAIN`: `app.github.dev`

3. The GitHub Actions workflow will automatically deploy when you push to main

## **Step 7: Connect Frontend to Backend**

Once deployed, your GitHub Pages site will be available at:
`https://<username>.github.io/<repository-name>/`

The frontend will automatically detect and connect to your Codespace services if configured correctly.

## **Quick Commands Reference**

```bash
# View service status
podman pod ps
podman ps --filter "pod=ai-platform-pod"

# View logs
podman logs <service-name>

# Stop all services
podman pod stop ai-platform-pod

# Restart a specific service
podman restart <service-name>

# Access service shells
podman exec -it fastapi /bin/bash
podman exec -it mlflow /bin/bash

# Clean up everything
podman pod rm -f ai-platform-pod
podman system prune -a
```

## **Development Workflow**

### For AI Architect Tasks:
1. Access the unified platform at the API Gateway URL
2. Use the AI Architect workspace for:
   - Model fine-tuning
   - QLoRA adapter creation
   - Custom embedding training
   - RAG workflow setup

### For Model Evaluation:
1. Use the Model Evaluation workspace for:
   - Testing raw models
   - Testing custom models
   - Evaluating workflows
   - Viewing results

### For Production Deployment:
1. Use the Factory Roster dashboard for:
   - Managing production models
   - Monitoring deployments
   - Viewing metrics

## **Troubleshooting**

### Services not starting:
```bash
# Check pod status
podman pod ps -a

# Check individual containers
podman ps -a

# View logs
podman logs <container-name>
```

### Port conflicts:
```bash
# Kill processes using ports
lsof -ti:8000 | xargs kill -9
lsof -ti:8080 | xargs kill -9
```

### Frontend can't connect to backend:
1. Verify CORS is enabled in NGINX config
2. Check browser console for errors
3. Ensure Codespace is running and ports are forwarded
4. Update GitHub Secrets with correct Codespace details

## **Architecture Overview**

```
GitHub Pages (Static Frontend)
    ↓
[CORS-enabled API Gateway :8000]
    ├── FastAPI Platform :8080
    ├── Gradio App :7860
    ├── MLflow Tracking :5000
    ├── ChromaDB :8081
    └── Neo4j :7687/7474
```

## **Next Steps**

1. Explore the unified platform interface
2. Run model evaluations
3. Train custom models
4. Deploy to production
5. Monitor performance

---

**Need Help?** Check the `/docs` folder or run `mkdocs serve` to view full documentation.