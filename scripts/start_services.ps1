# Enterprise LLMOps Service Startup Script
# This script starts all required services with proper configuration

Write-Host "🚀 Starting Enterprise LLMOps Services..." -ForegroundColor Green

# Create necessary directories
Write-Host "📁 Creating directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
New-Item -ItemType Directory -Force -Path "chroma_db" | Out-Null

# Start ChromaDB with configuration
Write-Host "🗄️ Starting ChromaDB..." -ForegroundColor Yellow
Start-Process -FilePath "python" -ArgumentList "-c", "import chromadb; chromadb.Client()" -WindowStyle Hidden
Start-Sleep -Seconds 2

# Alternative ChromaDB startup
Write-Host "🗄️ Starting ChromaDB server..." -ForegroundColor Yellow
Start-Process -FilePath "python" -ArgumentList "-m", "chromadb.cli", "--host", "0.0.0.0", "--port", "8081" -WindowStyle Hidden

# Start Enterprise FastAPI
Write-Host "🚀 Starting Enterprise FastAPI..." -ForegroundColor Yellow
Start-Process -FilePath "python" -ArgumentList "-m", "src.enterprise_llmops.main", "--host", "0.0.0.0", "--port", "8080" -WindowStyle Hidden

Write-Host "✅ Services started!" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Service URLs:" -ForegroundColor Cyan
Write-Host "  • Gradio App (Assignment 1): http://localhost:7860" -ForegroundColor White
Write-Host "  • MLflow Tracking: http://localhost:5000" -ForegroundColor White
Write-Host "  • ChromaDB Vector Store: http://localhost:8081" -ForegroundColor White
Write-Host "  • Enterprise FastAPI (Assignment 2): http://localhost:8080" -ForegroundColor White
Write-Host ""
Write-Host "🔧 Testing commands:" -ForegroundColor Cyan
Write-Host "  curl http://localhost:8080/health" -ForegroundColor White
Write-Host "  curl http://localhost:8081/api/v1/heartbeat" -ForegroundColor White
Write-Host "  curl http://localhost:5000/health" -ForegroundColor White

