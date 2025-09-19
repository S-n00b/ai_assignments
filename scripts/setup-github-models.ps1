# GitHub Models Setup Script for PowerShell
# This script helps set up GitHub Models API authentication

Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "🚀 GitHub Models API Setup" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host ""

# Check if GitHub CLI is installed
Write-Host "Checking GitHub CLI installation..." -ForegroundColor Yellow
try {
    $ghVersion = gh --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ GitHub CLI is installed" -ForegroundColor Green
        $hasGitHubCLI = $true
    } else {
        Write-Host "❌ GitHub CLI is not installed" -ForegroundColor Red
        $hasGitHubCLI = $false
    }
} catch {
    Write-Host "❌ GitHub CLI is not installed" -ForegroundColor Red
    $hasGitHubCLI = $false
}

# Check current authentication
Write-Host "`nChecking current authentication..." -ForegroundColor Yellow
$currentToken = $env:GITHUB_TOKEN

if ($currentToken) {
    if ($currentToken -eq "demo_token") {
        Write-Host "⚠️  Demo token detected - limited functionality" -ForegroundColor Yellow
    } else {
        Write-Host "✅ GITHUB_TOKEN environment variable is set" -ForegroundColor Green
        Write-Host "🎉 GitHub Models authentication is ready!" -ForegroundColor Green
        exit 0
    }
} else {
    Write-Host "❌ No GITHUB_TOKEN environment variable found" -ForegroundColor Red
}

# Setup options
Write-Host "`n🔧 GitHub Models Authentication Setup" -ForegroundColor Cyan
Write-Host "=" * 40 -ForegroundColor Cyan

if ($hasGitHubCLI) {
    Write-Host "`nOption 1: GitHub CLI Authentication (Recommended)" -ForegroundColor Green
    Write-Host "-" * 40 -ForegroundColor Gray
    $useCLI = Read-Host "Set up GitHub CLI authentication? (y/n)"
    
    if ($useCLI -match "^[yY]") {
        Write-Host "`n📋 Setting up GitHub CLI authentication..." -ForegroundColor Yellow
        Write-Host "This will open a browser for authentication." -ForegroundColor Yellow
        
        try {
            gh auth login
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✅ GitHub CLI authentication successful!" -ForegroundColor Green
                Write-Host "🎉 GitHub Models authentication is ready!" -ForegroundColor Green
                exit 0
            } else {
                Write-Host "❌ GitHub CLI authentication failed" -ForegroundColor Red
            }
        } catch {
            Write-Host "❌ Error during GitHub CLI setup: $_" -ForegroundColor Red
        }
    }
}

# Manual PAT setup
Write-Host "`nOption 2: Personal Access Token Setup" -ForegroundColor Green
Write-Host "-" * 40 -ForegroundColor Gray
Write-Host "📋 Manual PAT Setup Instructions:" -ForegroundColor Yellow
Write-Host "1. Go to: https://github.com/settings/tokens" -ForegroundColor White
Write-Host "2. Click 'Generate new token (classic)'" -ForegroundColor White
Write-Host "3. Give it a name: 'Lenovo AAITC Models API'" -ForegroundColor White
Write-Host "4. Select scopes:" -ForegroundColor White
Write-Host "   ✅ models (required for GitHub Models API)" -ForegroundColor Green
Write-Host "   ✅ repo (optional, for repository access)" -ForegroundColor Green
Write-Host "5. Set expiration: 90 days (recommended)" -ForegroundColor White
Write-Host "6. Click 'Generate token'" -ForegroundColor White
Write-Host "7. Copy the token (it won't be shown again!)" -ForegroundColor White
Write-Host ""

# Set environment variable
Write-Host "8. Set environment variable:" -ForegroundColor White
Write-Host "   For current session:" -ForegroundColor Yellow
Write-Host "   `$env:GITHUB_TOKEN = 'your_token_here'" -ForegroundColor Cyan
Write-Host "   For permanent setup:" -ForegroundColor Yellow
Write-Host "   [Environment]::SetEnvironmentVariable('GITHUB_TOKEN', 'your_token_here', 'User')" -ForegroundColor Cyan
Write-Host ""

# Create .env file
$envFile = ".env"
if (-not (Test-Path $envFile)) {
    Write-Host "📝 Creating .env file template..." -ForegroundColor Yellow
    @"
# GitHub Models API Configuration
# Replace 'your_token_here' with your actual GitHub token
GITHUB_TOKEN=your_token_here
"@ | Out-File -FilePath $envFile -Encoding UTF8
    Write-Host "✅ Created $envFile" -ForegroundColor Green
    Write-Host "   Edit this file and add your GitHub token" -ForegroundColor Yellow
} else {
    Write-Host "✅ $envFile already exists" -ForegroundColor Green
}

Write-Host "`n📋 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Follow the PAT setup instructions above" -ForegroundColor White
Write-Host "2. Set your GITHUB_TOKEN environment variable" -ForegroundColor White
Write-Host "3. Run this script again to verify setup" -ForegroundColor White
Write-Host "4. Start using GitHub Models in your application!" -ForegroundColor White

Write-Host "`n💡 Quick Test:" -ForegroundColor Cyan
Write-Host "   python -c `"from src.github_models_backend import GitHubModelsClient; print('Setup complete!')`"" -ForegroundColor Yellow

Write-Host "`n🎯 For immediate testing with demo mode:" -ForegroundColor Cyan
Write-Host "   `$env:GITHUB_TOKEN = 'demo_token'" -ForegroundColor Yellow
