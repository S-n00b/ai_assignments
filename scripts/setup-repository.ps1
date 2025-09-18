#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Repository Setup Script for Lenovo AAITC Solutions

.DESCRIPTION
    This script helps set up the repository after updating references from samne to s-n00b.
    It provides instructions and automates some setup tasks.

.PARAMETER RepositoryUrl
    The GitHub repository URL (default: https://github.com/s-n00b/ai_assignments)

.PARAMETER SiteUrl
    The GitHub Pages site URL (default: https://s-n00b.github.io/ai_assignments)

.EXAMPLE
    .\setup-repository.ps1
    .\setup-repository.ps1 -RepositoryUrl "https://github.com/s-n00b/ai_assignments" -SiteUrl "https://s-n00b.github.io/ai_assignments"
#>

param(
    [string]$RepositoryUrl = "https://github.com/s-n00b/ai_assignments",
    [string]$SiteUrl = "https://s-n00b.github.io/ai_assignments"
)

# Color functions for better output
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    else {
        $input | Write-Output
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Write-Success { Write-ColorOutput Green $args }
function Write-Warning { Write-ColorOutput Yellow $args }
function Write-Error { Write-ColorOutput Red $args }
function Write-Info { Write-ColorOutput Cyan $args }
function Write-Header { Write-ColorOutput Magenta $args }

# Main setup function
function Start-RepositorySetup {
    Write-Header "🚀 Lenovo AAITC Solutions Repository Setup"
    Write-Info "Repository: $RepositoryUrl"
    Write-Info "Site URL: $SiteUrl"
    Write-Output ""

    # Step 1: Repository Reference Updates
    Write-Header "✅ Step 1: Repository Reference Updates"
    Write-Success "All repository references have been updated from 'samne' to 's-n00b'"
    Write-Info "Updated files:"
    Write-Info "  - docs/_config.yml"
    Write-Info "  - docs/_data/seo.yml"
    Write-Info "  - docs/_data/sitemap.yml"
    Write-Info "  - docs/sphinx/conf.py"
    Write-Info "  - docs/test_site_functionality.py"
    Write-Info "  - docs/robots.txt"
    Write-Info "  - docs/_posts/setup-guide/2025-09-18-interactive-features-setup.md"
    Write-Info "  - docs/INTERACTIVE_FEATURES_SETUP.md"
    Write-Output ""

    # Step 2: GitHub Environments Setup
    Write-Header "🔧 Step 2: GitHub Environments Setup"
    Write-Warning "You need to create GitHub environments in your repository settings:"
    Write-Output ""
    Write-Info "1. Go to: $RepositoryUrl/settings/environments"
    Write-Info "2. Create the following environments:"
    Write-Info "   - github-pages (for Jekyll deployment)"
    Write-Info "   - production (for CI/CD deployment)"
    Write-Output ""
    Write-Warning "Environment Configuration:"
    Write-Info "   github-pages:"
    Write-Info "     - URL: $SiteUrl"
    Write-Info "     - Protection rules: None (or as needed)"
    Write-Info "   production:"
    Write-Info "     - URL: $SiteUrl"
    Write-Info "     - Protection rules: Require reviewers (recommended)"
    Write-Output ""

    # Step 3: GitHub Discussions Setup
    Write-Header "💬 Step 3: GitHub Discussions Setup"
    Write-Warning "Enable GitHub Discussions for Giscus comments:"
    Write-Output ""
    Write-Info "1. Go to: $RepositoryUrl/settings"
    Write-Info "2. Scroll to 'Features' section"
    Write-Info "3. Check 'Discussions' checkbox"
    Write-Info "4. Click 'Set up discussions'"
    Write-Output ""

    # Step 4: Giscus Configuration
    Write-Header "🔧 Step 4: Giscus Configuration"
    Write-Warning "Configure Giscus for comments:"
    Write-Output ""
    Write-Info "1. Visit: https://giscus.app"
    Write-Info "2. Enter repository: s-n00b/ai_assignments"
    Write-Info "3. Select category: General"
    Write-Info "4. Copy the configuration values"
    Write-Info "5. Update docs/_config.yml with the values:"
    Write-Output ""
    Write-Warning "Update these values in docs/_config.yml:"
    Write-Info "   giscus:"
    Write-Info "     repo: s-n00b/ai_assignments"
    Write-Info "     repo_id: [YOUR_REPO_ID_HERE]"
    Write-Info "     category: General"
    Write-Info "     category_id: [YOUR_CATEGORY_ID_HERE]"
    Write-Output ""

    # Step 5: GitHub Pages Setup
    Write-Header "📄 Step 5: GitHub Pages Setup"
    Write-Warning "Configure GitHub Pages:"
    Write-Output ""
    Write-Info "1. Go to: $RepositoryUrl/settings/pages"
    Write-Info "2. Source: GitHub Actions"
    Write-Info "3. The workflow will automatically deploy from the docs/ folder"
    Write-Output ""

    # Step 6: Commit and Push
    Write-Header "📤 Step 6: Commit and Push Changes"
    Write-Warning "Commit and push your changes:"
    Write-Output ""
    Write-Info "git add ."
    Write-Info "git commit -m \"Update repository references from samne to s-n00b\""
    Write-Info "git push origin main"
    Write-Output ""

    # Step 7: Verification
    Write-Header "✅ Step 7: Verification"
    Write-Warning "After pushing, verify the setup:"
    Write-Output ""
    Write-Info "1. Check GitHub Actions: $RepositoryUrl/actions"
    Write-Info "2. Verify site deployment: $SiteUrl"
    Write-Info "3. Test search functionality"
    Write-Info "4. Test comments system (after Giscus setup)"
    Write-Output ""

    # Step 8: Additional Configuration
    Write-Header "⚙️ Step 8: Additional Configuration (Optional)"
    Write-Warning "Consider setting up:"
    Write-Output ""
    Write-Info "1. Google Analytics (replace G-XXXXXXXXXX in _config.yml)"
    Write-Info "2. Custom domain (if needed)"
    Write-Info "3. SEO verification codes"
    Write-Info "4. Social media integration"
    Write-Output ""

    # Summary
    Write-Header "📋 Setup Summary"
    Write-Success "Repository reference updates: ✅ Completed"
    Write-Warning "GitHub environments: ⏳ Manual setup required"
    Write-Warning "GitHub Discussions: ⏳ Manual setup required"
    Write-Warning "Giscus configuration: ⏳ Manual setup required"
    Write-Warning "GitHub Pages: ⏳ Manual setup required"
    Write-Warning "Commit and push: ⏳ Manual action required"
    Write-Output ""

    Write-Info "🎉 Setup script completed! Follow the manual steps above to complete the configuration."
}

# Run the setup
Start-RepositorySetup
