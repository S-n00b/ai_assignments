# Deploy MkDocs to GitHub Pages
# This script helps with local testing and deployment preparation

param(
    [switch]$Build,
    [switch]$Serve,
    [switch]$Deploy,
    [string]$Port = "8082"
)

# Activate virtual environment
& C:\Users\samne\PycharmProjects\ai_assignments\venv\Scripts\Activate.ps1

# Change to docs directory
Set-Location docs

if ($Build) {
    Write-Host "Building MkDocs documentation..." -ForegroundColor Green
    mkdocs build
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Build completed successfully!" -ForegroundColor Green
    } else {
        Write-Host "Build failed!" -ForegroundColor Red
        exit 1
    }
}

if ($Serve) {
    Write-Host "Serving MkDocs documentation on port $Port..." -ForegroundColor Green
    mkdocs serve --dev-addr 0.0.0.0:$Port
}

if ($Deploy) {
    Write-Host "Preparing for deployment..." -ForegroundColor Green
    Write-Host "1. Ensure all changes are committed to git" -ForegroundColor Yellow
    Write-Host "2. Push changes to main branch" -ForegroundColor Yellow
    Write-Host "3. GitHub Actions will automatically deploy to GitHub Pages" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "GitHub Pages URL: https://s-n00b.github.io/ai_assignments/" -ForegroundColor Cyan
}

if (-not $Build -and -not $Serve -and -not $Deploy) {
    Write-Host "MkDocs Deployment Script" -ForegroundColor Cyan
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\scripts\deploy-mkdocs.ps1 -Build    # Build the documentation" -ForegroundColor White
    Write-Host "  .\scripts\deploy-mkdocs.ps1 -Serve    # Serve locally on port 8082" -ForegroundColor White
    Write-Host "  .\scripts\deploy-mkdocs.ps1 -Deploy   # Show deployment instructions" -ForegroundColor White
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\scripts\deploy-mkdocs.ps1 -Build -Serve  # Build and serve" -ForegroundColor White
    Write-Host "  .\scripts\deploy-mkdocs.ps1 -Serve -Port 8083  # Serve on custom port" -ForegroundColor White
}
