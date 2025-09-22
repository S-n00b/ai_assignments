# GitHub Pages Migration Guide

## Replace Jekyll with Unified Platform

### 🎯 Overview

This guide will help you completely remove the old Jekyll documentation site and replace it with your unified platform, making it accessible via GitHub Pages.

### 🚨 Current Problem

The current GitHub Pages site at https://s-n00b.github.io/ai_assignments/ is showing a broken Jekyll site with Chirpy theme. We need to:

1. **Remove all Jekyll files and configurations**
2. **Deploy the unified platform as the main site**
3. **Integrate MkDocs documentation within the platform**
4. **Set up proper GitHub Actions for deployment**

### ✅ Solution Architecture

#### **New GitHub Pages Structure**

```
https://s-n00b.github.io/ai_assignments/
├── index.html (Unified Platform - Main Entry Point)
├── about/ (Assignment Overview)
├── site/ (MkDocs Documentation)
└── assets/ (Static Assets)
```

#### **Deployment Strategy**

1. **Main Entry Point**: `unified_platform.html` becomes `index.html`
2. **Documentation Hub**: MkDocs builds to `site/` directory
3. **Static Assets**: All assets served from root
4. **GitHub Actions**: Automated deployment on push to main

### 🚀 Step-by-Step Migration

#### **Step 1: Clean Up Old Jekyll Files**

```powershell
# Run the cleanup script
.\scripts\deploy-github-pages.ps1 -Clean
```

This will remove:

- `_config.yml`
- `_layouts/`
- `_includes/`
- `_posts/`
- `_sass/`
- `_site/`
- `Gemfile`
- `Gemfile.lock`
- `.jekyll-cache/`
- `.jekyll-metadata`

#### **Step 2: Build the New Platform**

```powershell
# Build the unified platform
.\scripts\deploy-github-pages.ps1 -Build
```

This will:

- Build MkDocs documentation to `site/`
- Copy `unified_platform.html` to `index.html`
- Create `about/index.html` for assignment overview
- Set up proper directory structure

#### **Step 3: Deploy to GitHub Pages**

```powershell
# Deploy everything
.\scripts\deploy-github-pages.ps1 -Deploy
```

This will:

- Commit all changes to git
- Push to main branch
- Trigger GitHub Actions deployment
- Deploy to GitHub Pages

### 🔧 GitHub Actions Configuration

The new workflow (`.github/workflows/deploy-unified-platform.yml`) will:

1. **Build MkDocs Documentation**

   - Install dependencies
   - Build documentation to `site/`
   - Create unified platform structure

2. **Create Main Entry Point**

   - Copy `unified_platform.html` to `index.html`
   - Create `about/index.html` for assignment overview
   - Set up proper directory structure

3. **Deploy to GitHub Pages**
   - Upload all files as artifact
   - Deploy to GitHub Pages environment

### 📁 New File Structure

After migration, your repository will have:

```
ai_assignments/
├── index.html (Unified Platform - Main Entry)
├── about/
│   └── index.html (Assignment Overview)
├── site/ (MkDocs Documentation)
├── src/ (Source Code)
├── docs/ (MkDocs Source)
└── .github/workflows/
    └── deploy-unified-platform.yml
```

### 🌐 Access Points

After deployment, users can access:

- **Main Platform**: https://s-n00b.github.io/ai_assignments/
- **Assignment Overview**: https://s-n00b.github.io/ai_assignments/about/
- **Documentation**: https://s-n00b.github.io/ai_assignments/site/
- **Unified Platform**: https://s-n00b.github.io/ai_assignments/ (main entry)

### 🔄 GitHub Codespaces Alternative

Since you mentioned GitHub Codespaces, here's how to set it up:

#### **Option 1: GitHub Codespaces (Recommended for Demos)**

1. **Create `.devcontainer/devcontainer.json`**:

```json
{
  "name": "Lenovo AI Architecture",
  "image": "mcr.microsoft.com/devcontainers/python:3.11",
  "features": {
    "ghcr.io/devcontainers/features/node:1": {}
  },
  "postCreateCommand": "pip install -r config/requirements.txt && pip install -r docs/requirements-docs.txt",
  "portsAttributes": {
    "8080": { "label": "FastAPI Platform" },
    "7860": { "label": "Gradio App" },
    "5000": { "label": "MLflow" },
    "8000": { "label": "ChromaDB" },
    "8082": { "label": "MkDocs" }
  },
  "forwardPorts": [8080, 7860, 5000, 8000, 8082],
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.pylint",
        "ms-toolsai.jupyter"
      ]
    }
  }
}
```

2. **Create `codespace-setup.sh`**:

```bash
#!/bin/bash
# Setup script for GitHub Codespaces

echo "Setting up Lenovo AI Architecture environment..."

# Install dependencies
pip install -r config/requirements.txt
pip install -r docs/requirements-docs.txt

# Start services in background
python -m src.enterprise_llmops.main --host 0.0.0.0 --port 8080 &
python -m src.gradio_app.main --host 0.0.0.0 --port 7860 &
cd docs && mkdocs serve --dev-addr 0.0.0.0:8082 &

echo "Services started:"
echo "- FastAPI Platform: http://localhost:8080"
echo "- Gradio App: http://localhost:7860"
echo "- MkDocs: http://localhost:8082"
echo "- GitHub Pages: https://s-n00b.github.io/ai_assignments/"
```

#### **Option 2: GitHub Pages with Static Demo**

The unified platform will work as a static demo on GitHub Pages, with:

- **Interactive UI**: Full unified platform interface
- **Static Content**: Assignment overview and documentation
- **Local Development**: Instructions for running full services

### 🛠️ Troubleshooting

#### **Common Issues**

1. **Jekyll Still Showing**

   - Check if `_config.yml` exists and remove it
   - Ensure GitHub Pages is set to "GitHub Actions" source
   - Clear browser cache

2. **Unified Platform Not Loading**

   - Verify `index.html` exists in root directory
   - Check GitHub Actions deployment logs
   - Ensure all assets are properly copied

3. **MkDocs Not Accessible**
   - Verify `site/` directory exists
   - Check MkDocs build process in GitHub Actions
   - Ensure proper file permissions

#### **Debug Commands**

```powershell
# Check current status
git status
git log --oneline -5

# Verify file structure
ls -la
ls -la site/
ls -la about/

# Test local build
.\scripts\deploy-github-pages.ps1 -Build
```

### 📊 Expected Results

After successful migration:

1. **GitHub Pages Site**: https://s-n00b.github.io/ai_assignments/

   - Shows unified platform interface
   - All navigation working
   - Documentation accessible

2. **GitHub Actions**:

   - Automatic deployment on push
   - Build logs show successful deployment
   - No Jekyll-related errors

3. **Repository Structure**:
   - Clean, organized file structure
   - No Jekyll files remaining
   - Proper GitHub Actions workflow

### 🎉 Next Steps

After migration:

1. **Test the Site**: Visit https://s-n00b.github.io/ai_assignments/
2. **Verify Navigation**: Check all links work
3. **Test Documentation**: Ensure MkDocs is accessible
4. **Update README**: Document the new structure
5. **Share with Stakeholders**: Provide the GitHub Pages URL

### 📞 Support

If you encounter issues:

1. **Check GitHub Actions**: Look for deployment errors
2. **Verify File Structure**: Ensure all files are in correct locations
3. **Test Locally**: Use the build script to test locally
4. **Check Permissions**: Ensure GitHub Pages has proper permissions

---

**Last Updated**: 2025-01-27  
**Version**: 1.0.0  
**Status**: Ready for Migration  
**Integration**: Full GitHub Pages Deployment
