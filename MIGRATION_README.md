# 🚀 GitHub Pages Migration - Complete Guide

## 🎯 Quick Start (TL;DR)

```powershell
# 1. Clean up old Jekyll files
.\scripts\deploy-github-pages.ps1 -Clean

# 2. Build the new unified platform
.\scripts\deploy-github-pages.ps1 -Build

# 3. Deploy to GitHub Pages
.\scripts\deploy-github-pages.ps1 -Deploy
```

## 📋 What This Migration Does

### ✅ **Removes**

- ❌ Old Jekyll site with Chirpy theme
- ❌ Broken documentation structure
- ❌ Jekyll configuration files
- ❌ Outdated GitHub Actions workflows

### ✅ **Adds**

- ✅ Unified platform as main entry point
- ✅ MkDocs documentation integrated
- ✅ Assignment overview page
- ✅ Modern GitHub Actions deployment
- ✅ Clean, organized file structure

## 🔧 Before You Start

### **Prerequisites**

- ✅ Virtual environment activated
- ✅ Git repository with GitHub remote
- ✅ GitHub Pages enabled in repository settings
- ✅ GitHub Actions permissions configured

### **Check Current Status**

```powershell
# Check if you're in the right directory
pwd
# Should show: C:\Users\samne\PycharmProjects\ai_assignments

# Check git status
git status

# Check if virtual environment is activated
python --version
# Should show Python 3.11.x
```

## 🚀 Step-by-Step Migration

### **Step 1: Clean Up Old Files**

```powershell
# This removes all Jekyll-related files
.\scripts\deploy-github-pages.ps1 -Clean
```

**What it removes:**

- `_config.yml` (Jekyll config)
- `_layouts/` (Jekyll layouts)
- `_includes/` (Jekyll includes)
- `_posts/` (Jekyll posts)
- `_sass/` (Jekyll styles)
- `_site/` (Jekyll build output)
- `Gemfile` (Ruby dependencies)
- `Gemfile.lock` (Ruby lock file)
- `.jekyll-cache/` (Jekyll cache)
- `.jekyll-metadata` (Jekyll metadata)

### **Step 2: Build New Platform**

```powershell
# This builds the unified platform and documentation
.\scripts\deploy-github-pages.ps1 -Build
```

**What it creates:**

- `index.html` (Unified platform - main entry)
- `about/index.html` (Assignment overview)
- `site/` (MkDocs documentation)
- Proper directory structure

### **Step 3: Deploy to GitHub Pages**

```powershell
# This deploys everything to GitHub Pages
.\scripts\deploy-github-pages.ps1 -Deploy
```

**What it does:**

- Commits all changes to git
- Pushes to main branch
- Triggers GitHub Actions deployment
- Deploys to GitHub Pages

## 🌐 After Migration

### **Your New Site Structure**

```
https://s-n00b.github.io/ai_assignments/
├── / (Unified Platform - Main Entry)
├── /about/ (Assignment Overview)
├── /site/ (MkDocs Documentation)
└── /assets/ (Static Assets)
```

### **Access Points**

- **Main Platform**: https://s-n00b.github.io/ai_assignments/
- **Assignment Overview**: https://s-n00b.github.io/ai_assignments/about/
- **Documentation**: https://s-n00b.github.io/ai_assignments/site/
- **GitHub Repository**: https://github.com/s-n00b/ai_assignments

## 🔍 Verification Steps

### **1. Check GitHub Actions**

1. Go to your repository on GitHub
2. Click "Actions" tab
3. Look for "Deploy Unified Platform to GitHub Pages"
4. Verify it completed successfully

### **2. Check GitHub Pages Settings**

1. Go to repository Settings
2. Click "Pages" in left sidebar
3. Verify source is set to "GitHub Actions"
4. Check that deployment is successful

### **3. Test the Site**

1. Visit https://s-n00b.github.io/ai_assignments/
2. Verify unified platform loads
3. Test navigation between sections
4. Check documentation is accessible

## 🛠️ Troubleshooting

### **Common Issues**

#### **Issue: Jekyll Still Showing**

```powershell
# Check for remaining Jekyll files
ls -la | findstr "_"
ls -la | findstr "Gemfile"

# If found, remove manually
Remove-Item -Recurse -Force _config.yml
Remove-Item -Recurse -Force _layouts
# ... etc
```

#### **Issue: Unified Platform Not Loading**

```powershell
# Check if index.html exists
Test-Path index.html

# Check if it's the right file
Get-Content index.html | Select-String "unified_platform"
```

#### **Issue: GitHub Actions Failing**

1. Check repository permissions
2. Verify GitHub Pages is enabled
3. Check if environments exist
4. Review workflow logs

### **Debug Commands**

```powershell
# Check file structure
ls -la
ls -la site/
ls -la about/

# Check git status
git status
git log --oneline -5

# Test local build
.\scripts\deploy-github-pages.ps1 -Build
```

## 🎉 Success Indicators

### **✅ Migration Successful When:**

- [ ] GitHub Actions deployment completes successfully
- [ ] Site loads at https://s-n00b.github.io/ai_assignments/
- [ ] Unified platform interface is visible
- [ ] Navigation works between sections
- [ ] Documentation is accessible
- [ ] No Jekyll-related errors in browser console

### **✅ File Structure Correct When:**

- [ ] `index.html` exists in root
- [ ] `about/index.html` exists
- [ ] `site/` directory contains MkDocs output
- [ ] No Jekyll files remain
- [ ] GitHub Actions workflow is active

## 🚀 Next Steps

### **After Successful Migration:**

1. **Test Everything**

   - Visit all pages
   - Test navigation
   - Verify documentation

2. **Update Documentation**

   - Update README.md with new structure
   - Update SERVER_COMMANDS.md
   - Update any internal links

3. **Share with Stakeholders**

   - Provide GitHub Pages URL
   - Explain the new structure
   - Demonstrate the unified platform

4. **Monitor Deployment**
   - Check GitHub Actions regularly
   - Monitor site performance
   - Update as needed

## 📞 Support

### **If You Need Help:**

1. **Check the logs**:

   - GitHub Actions logs
   - Browser console errors
   - PowerShell error messages

2. **Verify prerequisites**:

   - Virtual environment activated
   - Git repository configured
   - GitHub Pages enabled

3. **Test locally first**:
   - Use the build script
   - Check file structure
   - Verify all files exist

### **Emergency Rollback**

If something goes wrong:

```powershell
# Reset to previous commit
git reset --hard HEAD~1
git push origin main --force

# Or restore from backup
git checkout main
git pull origin main
```

---

**Last Updated**: 2025-01-27  
**Version**: 1.0.0  
**Status**: Ready for Migration  
**Integration**: Full GitHub Pages Deployment
