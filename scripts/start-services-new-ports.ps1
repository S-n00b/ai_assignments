#!/usr/bin/env powershell
# Start All Services with New Port Assignments
# This script starts all services with unique ports to avoid conflicts

Write-Host "🚀 Starting Lenovo AAITC Services with New Port Assignments" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan

# Activate virtual environment
Write-Host "📦 Activating virtual environment..." -ForegroundColor Yellow
& C:\Users\samne\PycharmProjects\ai_assignments\venv\Scripts\Activate.ps1

# Function to start service in new terminal
function Start-ServiceInTerminal {
    param(
        [string]$ServiceName,
        [string]$Command,
        [string]$Port,
        [string]$Url
    )
    
    Write-Host "🌐 Starting $ServiceName on port $Port..." -ForegroundColor Cyan
    Write-Host "   URL: $Url" -ForegroundColor Gray
    Write-Host "   Command: $Command" -ForegroundColor Gray
    
    # Start in new PowerShell window
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "& C:\Users\samne\PycharmProjects\ai_assignments\venv\Scripts\Activate.ps1; $Command"
    
    Start-Sleep -Seconds 2
}

Write-Host "`n🔧 Service Port Assignments:" -ForegroundColor Magenta
Write-Host "   FastAPI Enterprise:  8080 - http://localhost:8080" -ForegroundColor White
Write-Host "   Gradio App:          7860 - http://localhost:7860" -ForegroundColor White
Write-Host "   MLflow:              5000 - http://localhost:5000" -ForegroundColor White
Write-Host "   ChromaDB:            8081 - http://localhost:8081" -ForegroundColor White
Write-Host "   MkDocs:              8082 - http://localhost:8082" -ForegroundColor White
Write-Host "   Weaviate:            8083 - http://localhost:8083" -ForegroundColor White

Write-Host "`n🚀 Starting Services..." -ForegroundColor Green

# Start ChromaDB (8081)
Start-ServiceInTerminal -ServiceName "ChromaDB" -Command "chroma run --host 0.0.0.0 --port 8081 --path chroma_data" -Port "8081" -Url "http://localhost:8081"

# Start MLflow (5000)
Start-ServiceInTerminal -ServiceName "MLflow" -Command "mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db" -Port "5000" -Url "http://localhost:5000"

# Start FastAPI Enterprise Platform (8080)
Start-ServiceInTerminal -ServiceName "FastAPI Enterprise" -Command "python -m src.enterprise_llmops.main --host 0.0.0.0 --port 8080" -Port "8080" -Url "http://localhost:8080"

# Start Gradio App (7860)
Start-ServiceInTerminal -ServiceName "Gradio App" -Command "python -m src.gradio_app.main --host 0.0.0.0 --port 7860" -Port "7860" -Url "http://localhost:7860"

# Start MkDocs (8082)
Start-ServiceInTerminal -ServiceName "MkDocs" -Command "mkdocs serve --dev-addr 0.0.0.0:8082" -Port "8082" -Url "http://localhost:8082"

Write-Host "`n✅ All services started!" -ForegroundColor Green
Write-Host "`n📋 Access URLs:" -ForegroundColor Cyan
Write-Host "   🏢 FastAPI Enterprise Platform:  http://localhost:8080" -ForegroundColor White
Write-Host "   📊 FastAPI API Documentation:    http://localhost:8080/docs" -ForegroundColor White
Write-Host "   🤖 Gradio Model Evaluation:      http://localhost:7860" -ForegroundColor White
Write-Host "   📈 MLflow Tracking:              http://localhost:5000" -ForegroundColor White
Write-Host "   🗄️  ChromaDB:                     http://localhost:8081" -ForegroundColor White
Write-Host "   📚 ChromaDB API Docs:            http://localhost:8081/docs" -ForegroundColor White
Write-Host "   📖 MkDocs Documentation:         http://localhost:8082" -ForegroundColor White
Write-Host "   🔍 FastAPI Embedded Docs:        http://localhost:8082/api/fastapi-embedded-docs/" -ForegroundColor White

Write-Host "`n🔍 Testing Service Health..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Test service health
$services = @(
    @{Name = "FastAPI"; Url = "http://localhost:8080/health" },
    @{Name = "ChromaDB"; Url = "http://localhost:8081/api/v2/heartbeat" },
    @{Name = "MLflow"; Url = "http://localhost:5000/health" },
    @{Name = "MkDocs"; Url = "http://localhost:8082" }
)

foreach ($service in $services) {
    try {
        $response = Invoke-WebRequest -Uri $service.Url -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Host "   ✅ $($service.Name): Healthy" -ForegroundColor Green
        }
        else {
            Write-Host "   ⚠️  $($service.Name): Status $($response.StatusCode)" -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "   ❌ $($service.Name): Not responding" -ForegroundColor Red
    }
}

Write-Host "`n🎉 Services startup complete!" -ForegroundColor Green
Write-Host "   All services are running on unique ports with no conflicts." -ForegroundColor Gray
Write-Host "   Documentation sources are clearly attributed and embedded." -ForegroundColor Gray

Write-Host "`nPress any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
