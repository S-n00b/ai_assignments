# Ollama Setup Script for Enterprise LLMOps
# This script sets up Ollama properly for the AI Assignments project

Write-Host "Starting setup of Ollama for Enterprise LLMOps..." -ForegroundColor Green

# Check if Ollama is already installed
$ollamaInstalled = Get-Command ollama -ErrorAction SilentlyContinue

if ($ollamaInstalled) {
    Write-Host "Ollama is already installed" -ForegroundColor Green
    $version = ollama --version
    Write-Host "Version: $version" -ForegroundColor Cyan
}
else {
    Write-Host "Ollama is not installed. Installing..." -ForegroundColor Yellow
    
    # Download and install Ollama for Windows
    $ollamaUrl = "https://ollama.com/download/windows"
    Write-Host "Please download and install Ollama from: $ollamaUrl" -ForegroundColor Cyan
    Write-Host "After installation, restart your terminal and run this script again." -ForegroundColor Yellow
    
    # Open the download page
    Start-Process $ollamaUrl
    
    Read-Host "Press Enter after installing Ollama to continue..."
}

# Check Docker installation
$dockerInstalled = Get-Command docker -ErrorAction SilentlyContinue

if ($dockerInstalled) {
    Write-Host "Docker is installed" -ForegroundColor Green
    try {
        $dockerVersion = docker --version
        Write-Host "Version: $dockerVersion" -ForegroundColor Cyan
    }
    catch {
        Write-Host "Docker is installed but not running" -ForegroundColor Yellow
        Write-Host "Please start Docker Desktop and try again." -ForegroundColor Yellow
    }
}
else {
    Write-Host "Docker is not installed" -ForegroundColor Yellow
    Write-Host "Docker is optional but recommended for advanced Ollama features." -ForegroundColor Cyan
}

# Create necessary directories
Write-Host "Creating directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
New-Item -ItemType Directory -Force -Path "ollama_models" | Out-Null

# Test Ollama service
Write-Host "Testing Ollama service..." -ForegroundColor Yellow
try {
    $ollamaStatus = ollama list 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Ollama service is running" -ForegroundColor Green
        Write-Host "Available models:" -ForegroundColor Cyan
        Write-Host $ollamaStatus -ForegroundColor White
    }
    else {
        Write-Host "Ollama service is not running. Starting..." -ForegroundColor Yellow
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
        Start-Sleep -Seconds 5
        
        # Test again
        $ollamaStatus = ollama list 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Ollama service started successfully" -ForegroundColor Green
        }
        else {
            Write-Host "Failed to start Ollama service" -ForegroundColor Red
        }
    }
}
catch {
    Write-Host "Error testing Ollama service: $_" -ForegroundColor Red
}

# Download recommended models
Write-Host "Downloading recommended models..." -ForegroundColor Yellow

$recommendedModels = @(
    "llama3.3:latest",
    "codellama:latest", 
    "mistral:latest",
    "phi3:latest"
)

foreach ($model in $recommendedModels) {
    Write-Host "Downloading $model..." -ForegroundColor Cyan
    try {
        ollama pull $model
        if ($LASTEXITCODE -eq 0) {
            Write-Host "$model downloaded successfully" -ForegroundColor Green
        }
        else {
            Write-Host "Failed to download $model" -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "Error downloading $model : $_" -ForegroundColor Red
    }
}

# Test model inference
Write-Host "Testing model inference..." -ForegroundColor Yellow
try {
    $testResponse = ollama run llama3.3:latest "Hello, how are you?" --verbose 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Model inference test successful" -ForegroundColor Green
        Write-Host "Sample response: $($testResponse | Select-String -Pattern 'Hello' | Select-Object -First 1)" -ForegroundColor White
    }
    else {
        Write-Host "Model inference test failed" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "Error testing model inference: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "Ollama setup completed!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Start the Enterprise FastAPI app: python -m src.enterprise_llmops.main --host 0.0.0.0 --port 8080" -ForegroundColor White
Write-Host "2. Test the complete integration: .\scripts\service-integration.ps1 -FullIntegration" -ForegroundColor White
Write-Host "3. Access the applications:" -ForegroundColor White
Write-Host "   - Gradio App (Assignment 1): http://localhost:7860" -ForegroundColor White
Write-Host "   - Enterprise FastAPI (Assignment 2): http://localhost:8080" -ForegroundColor White
Write-Host "   - MLflow Tracking: http://localhost:5000" -ForegroundColor White
Write-Host "   - ChromaDB Vector Store: http://localhost:8081" -ForegroundColor White
