# GitHub Pages Migration Guide - Unified Platform

## 🎯 Overview

This guide provides step-by-step instructions to migrate from the old Jekyll site to the new unified platform on GitHub Pages. The migration replaces the static Jekyll site with your interactive unified platform while maintaining all MkDocs documentation.

## 🚀 Quick Migration (3 Commands)

```powershell
# 1. Clean up old Jekyll files
.\scripts\deploy-github-pages.ps1 -Clean

# 2. Build the unified platform
.\scripts\deploy-github-pages.ps1 -Build

# 3. Deploy to GitHub Pages
.\scripts\deploy-github-pages.ps1 -Deploy
```

## 📁 What You'll Get

After migration, your GitHub Pages site will have:

- **Main Site**: https://s-n00b.github.io/ai_assignments/ (Unified Platform)
- **Assignment Overview**: https://s-n00b.github.io/ai_assignments/about/
- **Documentation**: https://s-n00b.github.io/ai_assignments/site/
- **No More Jekyll**: Completely removed and replaced

## 🔧 How It Works

### Unified Platform as Main Entry
- Your `unified_platform.html` becomes the main `index.html`
- Interactive demo environment accessible via GitHub Pages
- All functionality preserved in static form

### MkDocs Integration
- Documentation builds to `site/` directory
- Accessible within the platform at `/site/`
- Full MkDocs functionality maintained

### Assignment Overview
- Static page showing both assignments
- Professional presentation for stakeholders
- Links to live demos and documentation

### GitHub Actions Automation
- Automated deployment on every push to main
- No manual intervention required
- Builds and deploys both platform and documentation

## 📋 Detailed Migration Steps

### Step 1: Clean Up Old Jekyll Files

```powershell
# Run the cleanup script
.\scripts\deploy-github-pages.ps1 -Clean
```

This removes:
- `_config.yml`
- `_layouts/`, `_includes/`, `_posts/`, `_sass/`
- `_site/`, `Gemfile`, `Gemfile.lock`
- `.jekyll-cache/`, `.jekyll-metadata`
- Any old Jekyll-generated `index.html`

### Step 2: Build the Unified Platform

```powershell
# Build everything locally
.\scripts\deploy-github-pages.ps1 -Build
```

This creates:
- `index.html` (unified platform)
- `about/index.html` (assignment overview)
- `site/` (MkDocs documentation)

### Step 3: Deploy to GitHub Pages

```powershell
# Deploy to GitHub Pages
.\scripts\deploy-github-pages.ps1 -Deploy
```

This:
- Adds all changes to git
- Commits with timestamp
- Pushes to main branch
- Triggers GitHub Actions deployment

## 🔄 GitHub Actions Workflow

The migration uses the existing `.github/workflows/deploy-unified-platform.yml` workflow:

### Workflow Features
- **Trigger**: Push to main/master branch
- **Permissions**: Full GitHub Pages deployment access
- **Concurrency**: Prevents multiple deployments
- **Build Process**: 
  1. Setup Python 3.11
  2. Install MkDocs dependencies
  3. Build documentation to `site/`
  4. Create unified platform structure
  5. Upload artifact for deployment

### Workflow Steps
1. **Checkout**: Full repository checkout
2. **Setup Python**: Python 3.11 with caching
3. **Install Dependencies**: MkDocs and plugins
4. **Build MkDocs**: Documentation to `site/`
5. **Create Platform**: Unified platform as `index.html`
6. **Create About Page**: Assignment overview
7. **Upload Artifact**: Ready for deployment
8. **Deploy**: Automatic GitHub Pages deployment

## 🌐 Service Integration

### Port Configuration
| Service | Port | URL | Description |
|---|---|-----|----|
| **GitHub Pages** | 443 | https://s-n00b.github.io/ai_assignments/ | Main unified platform |
| **About Page** | 443 | https://s-n00b.github.io/ai_assignments/about/ | Assignment overview |
| **Documentation** | 443 | https://s-n00b.github.io/ai_assignments/site/ | MkDocs documentation |

### Local Development URLs
| Service | Port | URL | Description |
|---|---|-----|----|
| **Enterprise Platform** | 8080 | http://localhost:8080 | Full interactive platform |
| **Gradio App** | 7860 | http://localhost:7860 | Model evaluation interface |
| **MkDocs** | 8082 | http://localhost:8082 | Local documentation |
| **MLflow** | 5000 | http://localhost:5000 | Experiment tracking |
| **ChromaDB** | 8081 | http://localhost:8081 | Vector database |

## 🛠️ Troubleshooting

### Common Issues

#### 1. Jekyll Files Still Present
```powershell
# Check for remaining Jekyll files
Get-ChildItem -Recurse -Name | Where-Object { $_ -match "jekyll|_config|_layouts|_includes|_posts|_sass|_site|Gemfile" }

# Remove manually if needed
Remove-Item -Recurse -Force "_config.yml", "_layouts", "_includes", "_posts", "_sass", "_site", "Gemfile", "Gemfile.lock", ".jekyll-cache", ".jekyll-metadata" -ErrorAction SilentlyContinue
```

#### 2. GitHub Actions Not Triggering
```powershell
# Check workflow file exists
Test-Path ".github/workflows/deploy-unified-platform.yml"

# Verify it's not disabled
Get-Content ".github/workflows/deploy-unified-platform.yml" | Select-String "on:"
```

#### 3. Build Failures
```powershell
# Check MkDocs build locally
cd docs
mkdocs build --site-dir ../site

# Check Python dependencies
pip install -r docs/requirements-docs.txt
```

#### 4. Deployment Issues
```powershell
# Check git status
git status

# Check if changes are committed
git log --oneline -5

# Force push if needed (be careful!)
git push origin main --force
```

### Debug Commands

```powershell
# Check GitHub Actions status
# Go to: https://github.com/s-n00b/ai_assignments/actions

# Check Pages deployment
# Go to: https://github.com/s-n00b/ai_assignments/settings/pages

# Test local build
.\scripts\deploy-github-pages.ps1 -Build
# Then open index.html in browser

# Check file structure
Get-ChildItem -Name | Sort-Object
```

## 📊 Migration Checklist

### Pre-Migration
- [ ] Backup current repository
- [ ] Check GitHub Pages settings
- [ ] Verify workflow permissions
- [ ] Test local build process

### Migration Steps
- [ ] Run cleanup script
- [ ] Build unified platform
- [ ] Deploy to GitHub Pages
- [ ] Verify deployment

### Post-Migration
- [ ] Test main site functionality
- [ ] Verify about page loads
- [ ] Check documentation access
- [ ] Test all links and navigation

## 🚀 Advanced Configuration

### Custom Domain (Optional)
If you want to use a custom domain:

1. Add `CNAME` file to repository root:
```
your-domain.com
```

2. Update GitHub Pages settings:
- Go to repository Settings > Pages
- Set custom domain
- Enable HTTPS

### Environment Variables
The workflow uses these default settings:
- Python version: 3.11
- MkDocs theme: Material
- Build directory: `site/`
- Output directory: repository root

### Customization
To customize the deployment:

1. **Modify workflow**: Edit `.github/workflows/deploy-unified-platform.yml`
2. **Update script**: Edit `scripts/deploy-github-pages.ps1`
3. **Change content**: Modify `src/enterprise_llmops/frontend/unified_platform.html`

## 🔄 Rollback Procedure

If you need to rollback to the previous setup:

```powershell
# 1. Disable the new workflow
Rename-Item ".github/workflows/deploy-unified-platform.yml" ".github/workflows/deploy-unified-platform.yml.disabled"

# 2. Re-enable old workflow (if it exists)
Rename-Item ".github/workflows/mkdocs-deploy.yml.disabled" ".github/workflows/mkdocs-deploy.yml"

# 3. Remove new files
Remove-Item "index.html", "about" -Recurse -Force -ErrorAction SilentlyContinue

# 4. Commit and push
git add .
git commit -m "Rollback to previous GitHub Pages setup"
git push origin main
```

## 📞 Support

### Getting Help
- Check GitHub Actions logs: https://github.com/s-n00b/ai_assignments/actions
- Review Pages settings: https://github.com/s-n00b/ai_assignments/settings/pages
- Test locally first with `.\scripts\deploy-github-pages.ps1 -Build`

### Verification Commands
```powershell
# Test local build
.\scripts\deploy-github-pages.ps1 -Build

# Check file structure
Get-ChildItem -Name | Sort-Object

# Verify git status
git status

# Check workflow file
Get-Content ".github/workflows/deploy-unified-platform.yml" | Select-String "on:"
```

---

**Last Updated**: 2025-01-27  
**Version**: 1.0.0  
**Status**: Production Ready  
**Integration**: Full GitHub Pages Integration