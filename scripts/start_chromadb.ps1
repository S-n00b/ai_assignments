# ChromaDB Startup Script for Enterprise LLMOps
# This script starts ChromaDB using the new 1.x architecture

Write-Host "Starting ChromaDB server..." -ForegroundColor Green

# Check if virtual environment is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "Virtual environment not activated. Activating..." -ForegroundColor Yellow
    & ".\venv\Scripts\Activate.ps1"
}

# Create data directory
$dataDir = "chroma_data"
if (-not (Test-Path $dataDir)) {
    New-Item -ItemType Directory -Path $dataDir | Out-Null
    Write-Host "Created ChromaDB data directory: $dataDir" -ForegroundColor Cyan
}

# Try different methods to start ChromaDB
Write-Host "Attempting to start ChromaDB..." -ForegroundColor Yellow

# Method 1: Try chroma CLI command
try {
    Write-Host "Trying chroma CLI command..." -ForegroundColor Cyan
    $chromaCommand = "chroma", "run", "--host", "0.0.0.0", "--port", "8081", "--path", $dataDir
    Start-Process -FilePath "chroma" -ArgumentList $chromaCommand[1..($chromaCommand.Length - 1)] -NoNewWindow
    Write-Host "ChromaDB started via CLI!" -ForegroundColor Green
    exit 0
}
catch {
    Write-Host "Chroma CLI not available, trying Python method..." -ForegroundColor Yellow
}

# Method 2: Try Python script
try {
    Write-Host "Starting ChromaDB via Python script..." -ForegroundColor Cyan
    python start_chromadb_new.py
}
catch {
    Write-Host "Python method failed, trying CLI script..." -ForegroundColor Yellow
    python start_chromadb_cli.py
}

Write-Host "ChromaDB startup completed!" -ForegroundColor Green
Write-Host "ChromaDB should be available at: http://localhost:8081" -ForegroundColor Cyan
Write-Host "Health check: http://localhost:8081/api/v1/heartbeat" -ForegroundColor Cyan
