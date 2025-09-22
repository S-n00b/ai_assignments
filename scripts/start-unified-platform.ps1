# Start Unified Platform with Correct Ports
# This script starts all services with the correct port configuration for the unified platform

Write-Host "🚀 Starting Lenovo AI Architecture Unified Platform..." -ForegroundColor Green

# Activate virtual environment
Write-Host "📦 Activating virtual environment..." -ForegroundColor Yellow
& C:\Users\samne\PycharmProjects\ai_assignments\venv\Scripts\Activate.ps1

# Function to check if port is available
function Test-Port {
    param([int]$Port)
    try {
        $connection = New-Object System.Net.Sockets.TcpClient
        $connection.Connect("localhost", $Port)
        $connection.Close()
        return $true
    }
    catch {
        return $false
    }
}

# Function to start service in background
function Start-ServiceInBackground {
    param(
        [string]$Name,
        [string]$Command,
        [int]$Port,
        [string]$Description
    )
    
    Write-Host "🔧 Starting $Name on port $Port..." -ForegroundColor Cyan
    
    if (Test-Port $Port) {
        Write-Host "⚠️  Port $Port is already in use. Skipping $Name." -ForegroundColor Yellow
        return
    }
    
    try {
        Start-Process powershell -ArgumentList "-Command", "& C:\Users\samne\PycharmProjects\ai_assignments\venv\Scripts\Activate.ps1; $Command" -WindowStyle Minimized
        Write-Host "✅ $Name started successfully" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Failed to start $Name - $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Start services with correct ports
Write-Host "`n🌐 Starting External Services..." -ForegroundColor Magenta

# Start MLflow on port 5000
Start-ServiceInBackground -Name "MLflow" -Command "mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db" -Port 5000 -Description "MLflow Tracking UI"

# Start ChromaDB on port 8081
Start-ServiceInBackground -Name "ChromaDB" -Command "chroma run --host 0.0.0.0 --port 8081 --path chroma_data" -Port 8081 -Description "ChromaDB Vector Store"

# Start MkDocs on port 8082
Start-ServiceInBackground -Name "MkDocs" -Command "mkdocs serve --dev-addr 0.0.0.0:8082" -Port 8082 -Description "MkDocs Documentation Hub"

# Start LangGraph Studio on port 8083
Start-ServiceInBackground -Name "LangGraph Studio" -Command "langgraph-studio --host 0.0.0.0 --port 8083" -Port 8083 -Description "LangGraph Studio Agent Visualization"

# Start Gradio on port 7860
Start-ServiceInBackground -Name "Gradio" -Command "python -m src.gradio_app.main --host 0.0.0.0 --port 7860" -Port 7860 -Description "Model Evaluation Interface"

# Wait a moment for services to start
Write-Host "`n⏳ Waiting for services to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Start the main FastAPI application
Write-Host "`n🚀 Starting FastAPI Enterprise Platform on port 8080..." -ForegroundColor Green

if (Test-Port 8080) {
    Write-Host "⚠️  Port 8080 is already in use. Please stop the existing service first." -ForegroundColor Yellow
    Write-Host "You can access the unified platform at: http://localhost:8080" -ForegroundColor Cyan
}
else {
    try {
        # Start FastAPI with unified platform
        python -m src.enterprise_llmops.main --host 0.0.0.0 --port 8080
    }
    catch {
        Write-Host "❌ Failed to start FastAPI: $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host "`n🎉 Unified Platform Services Started!" -ForegroundColor Green
Write-Host "`n📋 Service URLs:" -ForegroundColor Cyan
Write-Host "   🏠 Unified Platform: http://localhost:8080" -ForegroundColor White
Write-Host "   📖 About & Pitch: http://localhost:8080/about" -ForegroundColor White
Write-Host "   📚 API Docs: http://localhost:8080/docs" -ForegroundColor White
Write-Host "   🧪 Model Evaluation: http://localhost:7860" -ForegroundColor White
Write-Host "   📈 MLflow Tracking: http://localhost:5000" -ForegroundColor White
Write-Host "   🗄️ ChromaDB Vector Store: http://localhost:8081" -ForegroundColor White
Write-Host "   📚 MkDocs Documentation: http://localhost:8082" -ForegroundColor White
Write-Host "   🎯 LangGraph Studio: http://localhost:8083" -ForegroundColor White

Write-Host "`n💡 Tips:" -ForegroundColor Yellow
Write-Host "   • The unified platform embeds all services as iframes" -ForegroundColor Gray
Write-Host "   • No need to open multiple tabs - everything is in one interface" -ForegroundColor Gray
Write-Host "   • Use the sidebar navigation to switch between services" -ForegroundColor Gray
Write-Host "   • Services are automatically checked for availability" -ForegroundColor Gray
