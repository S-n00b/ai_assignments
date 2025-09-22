# Deploy GitHub Pages - Unified Platform
# This script helps deploy the unified platform to GitHub Pages

param(
    [switch]$Build,
    [switch]$Deploy,
    [switch]$Clean,
    [switch]$Help
)

if ($Help) {
    Write-Host "GitHub Pages Deployment Script" -ForegroundColor Green
    Write-Host ""
    Write-Host "Usage: .\scripts\deploy-github-pages.ps1 [options]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Build    Build the unified platform locally"
    Write-Host "  -Deploy   Deploy to GitHub Pages"
    Write-Host "  -Clean    Clean up old Jekyll files"
    Write-Host "  -Help     Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\scripts\deploy-github-pages.ps1 -Build"
    Write-Host "  .\scripts\deploy-github-pages.ps1 -Deploy"
    Write-Host "  .\scripts\deploy-github-pages.ps1 -Clean -Build -Deploy"
    exit 0
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& C:\Users\samne\PycharmProjects\ai_assignments\venv\Scripts\Activate.ps1

if ($Clean) {
    Write-Host "Cleaning up old Jekyll files..." -ForegroundColor Yellow
    
    # Remove any Jekyll-related files
    $jekyllFiles = @(
        "_config.yml",
        "_layouts",
        "_includes", 
        "_posts",
        "_sass",
        "_site",
        "Gemfile",
        "Gemfile.lock",
        ".jekyll-cache",
        ".jekyll-metadata"
    )
    
    foreach ($file in $jekyllFiles) {
        if (Test-Path $file) {
            Write-Host "Removing: $file" -ForegroundColor Red
            Remove-Item -Recurse -Force $file -ErrorAction SilentlyContinue
        }
    }
    
    # Remove any old index.html that might be Jekyll-generated
    if (Test-Path "index.html") {
        $content = Get-Content "index.html" -Raw
        if ($content -match "jekyll|chirpy|minima") {
            Write-Host "Removing old Jekyll index.html" -ForegroundColor Red
            Remove-Item "index.html" -Force
        }
    }
    
    Write-Host "Cleanup completed!" -ForegroundColor Green
}

if ($Build) {
    Write-Host "Building unified platform..." -ForegroundColor Yellow
    
    # Build MkDocs documentation
    Write-Host "Building MkDocs documentation..." -ForegroundColor Cyan
    Set-Location docs
    mkdocs build --site-dir ../site
    Set-Location ..
    
    # Create the main index.html for GitHub Pages
    Write-Host "Creating unified platform index.html..." -ForegroundColor Cyan
    Copy-Item "src/enterprise_llmops/frontend/unified_platform.html" "index.html"
    
    # Create about directory and page
    Write-Host "Creating about page..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Force -Path "about" | Out-Null
    
    $aboutContent = @"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lenovo AI Architecture - Assignment Portfolio</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&display=swap" rel="stylesheet">
    <style>
        :root {
            --lenovo-red: #E2231A;
            --lenovo-dark-red: #C01E17;
            --lenovo-black: #000000;
            --lenovo-gray: #666666;
            --lenovo-light-gray: #F5F5F5;
            --lenovo-white: #FFFFFF;
            --lenovo-blue: #0066CC;
            --lenovo-dark: #1A1A1A;
            --lenovo-card: #2A2A2A;
            --lenovo-border: #404040;
        }
        
        html, body {
            margin: 0;
            padding: 0;
            height: 100%;
            font-family: 'Inter', sans-serif;
            background-color: var(--lenovo-dark);
            color: var(--lenovo-white);
        }
        
        .lenovo-gradient {
            background: linear-gradient(135deg, var(--lenovo-red), var(--lenovo-dark-red));
        }
        
        .lenovo-text-gradient {
            background: linear-gradient(90deg, var(--lenovo-red), var(--lenovo-blue));
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    </style>
</head>
<body>
    <div class="min-h-screen bg-lenovo-dark flex items-center justify-center">
        <div class="max-w-4xl mx-auto p-8 text-center">
            <div class="lenovo-text-gradient text-6xl font-bold mb-8">Lenovo AI Architecture</div>
            <div class="text-2xl text-gray-300 mb-12">Advanced AI Model Evaluation & Architecture Framework</div>
            
            <div class="grid md:grid-cols-2 gap-8 mb-12">
                <div class="bg-lenovo-card p-6 rounded-lg border border-lenovo-border">
                    <h3 class="text-xl font-semibold mb-4 text-lenovo-red">Assignment 1: Model Evaluation Engineer</h3>
                    <p class="text-gray-300 mb-4">Comprehensive model evaluation framework with Gradio interface for testing and comparing AI models.</p>
                    <ul class="text-sm text-gray-400 space-y-2">
                        <li>• Model Evaluation Pipeline</li>
                        <li>• Model Profiling & Characterization</li>
                        <li>• Practical Evaluation Exercise</li>
                        <li>• Model Factory Framework</li>
                    </ul>
                </div>
                
                <div class="bg-lenovo-card p-6 rounded-lg border border-lenovo-border">
                    <h3 class="text-xl font-semibold mb-4 text-lenovo-red">Assignment 2: Enterprise LLMOps Platform</h3>
                    <p class="text-gray-300 mb-4">Full-stack enterprise platform with FastAPI backend, model registry, and MLOps capabilities.</p>
                    <ul class="text-sm text-gray-400 space-y-2">
                        <li>• Enterprise FastAPI Platform</li>
                        <li>• Model Registry & Lifecycle</li>
                        <li>• RAG System with ChromaDB</li>
                        <li>• Agent System with LangGraph</li>
                    </ul>
                </div>
            </div>
            
            <div class="bg-lenovo-card p-8 rounded-lg border border-lenovo-border mb-8">
                <h3 class="text-2xl font-semibold mb-6 text-lenovo-red">Live Demo Access</h3>
                <div class="grid md:grid-cols-3 gap-6 text-center">
                    <div>
                        <div class="text-4xl mb-2">🚀</div>
                        <div class="font-semibold mb-2">Unified Platform</div>
                        <div class="text-sm text-gray-400">Interactive demo environment</div>
                    </div>
                    <div>
                        <div class="text-4xl mb-2">📊</div>
                        <div class="font-semibold mb-2">Model Evaluation</div>
                        <div class="text-sm text-gray-400">Gradio testing interface</div>
                    </div>
                    <div>
                        <div class="text-4xl mb-2">📚</div>
                        <div class="font-semibold mb-2">Documentation</div>
                        <div class="text-sm text-gray-400">Comprehensive guides</div>
                    </div>
                </div>
            </div>
            
            <div class="text-center">
                <div class="text-lg text-gray-300 mb-4">Ready to explore the Lenovo AI Architecture framework?</div>
                <div class="text-sm text-gray-500">
                    This is a static preview. For the full interactive experience, 
                    <br>run the services locally using the provided scripts.
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"@
    
    Set-Content -Path "about/index.html" -Value $aboutContent
    
    Write-Host "Build completed successfully!" -ForegroundColor Green
    Write-Host "Files created:" -ForegroundColor Cyan
    Write-Host "  - index.html (unified platform)" -ForegroundColor White
    Write-Host "  - about/index.html (assignment overview)" -ForegroundColor White
    Write-Host "  - site/ (MkDocs documentation)" -ForegroundColor White
}

if ($Deploy) {
    Write-Host "Deploying to GitHub Pages..." -ForegroundColor Yellow
    
    # Check if we're in a git repository
    if (-not (Test-Path ".git")) {
        Write-Host "Error: Not in a git repository!" -ForegroundColor Red
        exit 1
    }
    
    # Add all changes
    Write-Host "Adding changes to git..." -ForegroundColor Cyan
    git add .
    
    # Commit changes
    $commitMessage = "Deploy unified platform to GitHub Pages - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "Committing changes..." -ForegroundColor Cyan
    git commit -m $commitMessage
    
    # Push to main branch
    Write-Host "Pushing to main branch..." -ForegroundColor Cyan
    git push origin main
    
    Write-Host "Deployment initiated!" -ForegroundColor Green
    Write-Host "Check GitHub Actions for deployment status." -ForegroundColor Yellow
    Write-Host "Site will be available at: https://s-n00b.github.io/ai_assignments/" -ForegroundColor Cyan
}

Write-Host "GitHub Pages deployment script completed!" -ForegroundColor Green
