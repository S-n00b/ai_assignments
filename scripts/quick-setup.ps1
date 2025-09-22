# =============================================================================
# AI Assignments - Quick 5-Terminal Setup
# =============================================================================
# Quick setup script for 5 terminals with essential services
# Usage: .\scripts\quick-setup.ps1
# =============================================================================

$ProjectRoot = "C:\Users\samne\PycharmProjects\ai_assignments"
$VenvPath = "$ProjectRoot\venv\Scripts\Activate.ps1"

Write-Host "🚀 Setting up 5-terminal development environment..." -ForegroundColor Green

# Change to project directory
Set-Location $ProjectRoot

# Terminal 1: ChromaDB Vector Store
$chromaCommand = @"
cd '$ProjectRoot'
& '$VenvPath'
Write-Host '🗄️ ChromaDB Vector Store' -ForegroundColor Green
Write-Host 'Port: 8081' -ForegroundColor Cyan
python -m chromadb.server --host 0.0.0.0 --port 8081
"@
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", $chromaCommand

# Wait for ChromaDB to start
Start-Sleep -Seconds 3

# Terminal 2: MLflow Tracking Server
$mlflowCommand = @"
cd '$ProjectRoot'
& '$VenvPath'
Write-Host '📊 MLflow Tracking Server' -ForegroundColor Green
Write-Host 'Port: 5000' -ForegroundColor Cyan
python -m mlflow.server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db
"@
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", $mlflowCommand

# Wait for MLflow to start
Start-Sleep -Seconds 5

# Terminal 3: Gradio Model Evaluation App
$gradioCommand = @"
cd '$ProjectRoot'
& '$VenvPath'
Write-Host '🎯 Gradio Model Evaluation App' -ForegroundColor Green
Write-Host 'Port: 7860' -ForegroundColor Cyan
python -m src.gradio_app.main --host 0.0.0.0 --port 7860
"@
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", $gradioCommand

# Wait for Gradio to start
Start-Sleep -Seconds 8

# Terminal 4: Enterprise FastAPI Platform
$fastapiCommand = @"
cd '$ProjectRoot'
& '$VenvPath'
Write-Host '🏢 Enterprise FastAPI Platform' -ForegroundColor Green
Write-Host 'Port: 8080' -ForegroundColor Cyan
python -m src.enterprise_llmops.main --host 0.0.0.0 --port 8080
"@
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", $fastapiCommand

# Terminal 5: Development & Testing Terminal
$devCommand = @"
cd '$ProjectRoot'
& '$VenvPath'
Write-Host '🛠️ Development & Testing Terminal' -ForegroundColor Green
Write-Host 'Available commands:' -ForegroundColor Cyan
Write-Host '  python -m pytest tests\ -v' -ForegroundColor White
Write-Host '  black src\ tests\' -ForegroundColor White
Write-Host '  isort src\ tests\' -ForegroundColor White
Write-Host '  python -m src.enterprise_llmops.simple_app' -ForegroundColor White
Write-Host ''
Write-Host 'Service URLs:' -ForegroundColor Yellow
Write-Host '  • FastAPI: http://localhost:8080' -ForegroundColor White
Write-Host '  • Gradio: http://localhost:7860' -ForegroundColor White
Write-Host '  • MLflow: http://localhost:5000' -ForegroundColor White
Write-Host '  • ChromaDB: http://localhost:8081' -ForegroundColor White
Write-Host ''
"@
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", $devCommand

Write-Host ""
Write-Host "✅ All 5 terminals started successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "🌐 Service URLs:" -ForegroundColor Cyan
Write-Host "  • Enterprise FastAPI: http://localhost:8080" -ForegroundColor White
Write-Host "  • API Documentation: http://localhost:8080/docs" -ForegroundColor White
Write-Host "  • Gradio App: http://localhost:7860" -ForegroundColor White
Write-Host "  • MLflow UI: http://localhost:5000" -ForegroundColor White
Write-Host "  • ChromaDB: http://localhost:8081" -ForegroundColor White
Write-Host ""
Write-Host "🔧 Testing commands:" -ForegroundColor Cyan
Write-Host "  curl http://localhost:8080/health" -ForegroundColor White
Write-Host "  curl http://localhost:7860" -ForegroundColor White
Write-Host "  curl http://localhost:5000/health" -ForegroundColor White
Write-Host "  curl http://localhost:8081/api/v1/heartbeat" -ForegroundColor White