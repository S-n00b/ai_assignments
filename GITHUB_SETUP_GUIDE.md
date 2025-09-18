# GitHub Setup Guide - Fix Workflow Failures

## 🚨 Current Issues

Based on the GitHub Actions failures, here are the immediate steps needed to fix the workflows:

## ✅ **IMMEDIATE FIXES APPLIED**

### 1. **Created Missing requirements.txt**

- Added root-level `requirements.txt` file that workflows were looking for
- Contains essential dependencies for CI/CD

### 2. **Fixed Jekyll Pages Workflow**

- Updated to use `docs/` directory as working directory
- Fixed artifact path to `docs/_site`

### 3. **Fixed API Documentation Workflow**

- Updated to use correct requirements file path
- Fixed dependency installation

## ⏳ **MANUAL SETUP REQUIRED**

### 1. **Create GitHub Environments** (CRITICAL)

**You MUST create these environments in your GitHub repository settings:**

1. Go to: `https://github.com/s-n00b/ai_assignments/settings/environments`
2. Click **"New environment"**
3. Create these environments:

#### Environment 1: `github-pages`

- **Name**: `github-pages`
- **URL**: `https://s-n00b.github.io/ai_assignments`
- **Protection rules**: None (for now)

#### Environment 2: `production`

- **Name**: `production`
- **URL**: `https://s-n00b.github.io/ai_assignments`
- **Protection rules**: None (for now)

**Why this is needed**: The workflows reference these environments, and they don't exist yet, causing the failures.

### 2. **Enable GitHub Discussions** (For Giscus)

1. Go to: `https://github.com/s-n00b/ai_assignments/settings`
2. Scroll to **"Features"** section
3. Check **"Discussions"** checkbox
4. Click **"Set up discussions"**

### 3. **Configure GitHub Pages**

1. Go to: `https://github.com/s-n00b/ai_assignments/settings/pages`
2. **Source**: Select **"GitHub Actions"**
3. Save the settings

### 4. **Configure Giscus Comments**

1. Visit: [https://giscus.app](https://giscus.app)
2. Enter repository: `s-n00b/ai_assignments`
3. Select category: `General`
4. Copy the configuration values
5. Update `docs/_config.yml`:
   ```yaml
   giscus:
     repo: s-n00b/ai_assignments
     repo_id: [YOUR_REPO_ID_HERE]
     category: General
     category_id: [YOUR_CATEGORY_ID_HERE]
   ```

## 🚀 **DEPLOYMENT STEPS**

### 1. **Commit and Push Changes**

```bash
git add .
git commit -m "Fix GitHub Actions workflows and add missing requirements.txt"
git push origin main
```

### 2. **Verify Workflows**

After pushing:

1. Go to: `https://github.com/s-n00b/ai_assignments/actions`
2. Check that all 4 workflows now pass
3. Verify the site deploys at: `https://s-n00b.github.io/ai_assignments`

## 📋 **Workflow Status After Fixes**

| Workflow              | Status  | Issue                                 | Fix Applied            |
| --------------------- | ------- | ------------------------------------- | ---------------------- |
| **Jekyll Pages**      | ❌ → ✅ | Missing requirements.txt, wrong paths | ✅ Fixed               |
| **API Documentation** | ❌ → ✅ | Wrong requirements path               | ✅ Fixed               |
| **CI/CD Pipeline**    | ❌ → ✅ | Missing production environment        | ⏳ Manual setup needed |
| **Test Suite**        | ❌ → ✅ | Context access issue                  | ✅ Fixed               |

## 🔧 **About Giscus**

**Giscus is NOT a Python package** - it's a JavaScript-based comment system that integrates with GitHub Discussions. It works by:

1. **Client-side integration**: The Giscus script runs in the browser
2. **GitHub Discussions backend**: Uses GitHub's native discussion system
3. **No server required**: Works with static sites like Jekyll/GitHub Pages

**Setup process**:

1. Enable GitHub Discussions (manual step above)
2. Configure Giscus at [giscus.app](https://giscus.app) (manual step above)
3. Update `_config.yml` with the generated values (manual step above)
4. The Jekyll theme will automatically include the Giscus script

## 🎯 **Next Steps After Manual Setup**

1. **Complete the 4 manual setup steps above**
2. **Push the current changes**
3. **Verify all workflows pass**
4. **Test the deployed site**
5. **Configure analytics and monitoring**

## 📞 **Troubleshooting**

If workflows still fail after manual setup:

1. **Check environment names**: Must be exactly `github-pages` and `production`
2. **Verify GitHub Pages settings**: Source must be "GitHub Actions"
3. **Check Discussions**: Must be enabled in repository settings
4. **Review workflow logs**: Click on failed runs to see specific errors

---

**Status**: ✅ Code fixes applied, ⏳ Manual setup required
**Priority**: HIGH - Manual setup needed to fix workflow failures
