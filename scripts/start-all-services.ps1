# =============================================================================
# Start All Services Script
# =============================================================================
# This script helps you start all services in the correct order
# Usage: .\scripts\start-all-services.ps1
# =============================================================================

Write-Host "🚀 Starting All AI Assignments Services..." -ForegroundColor Green
Write-Host "===========================================" -ForegroundColor Green

# Activate virtual environment
Write-Host ""
Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
& C:\Users\samne\PycharmProjects\ai_assignments\venv\Scripts\Activate.ps1

Write-Host ""
Write-Host "📋 Service Startup Instructions:" -ForegroundColor Cyan
Write-Host "You need to open 4 separate PowerShell terminals and run the following commands:" -ForegroundColor White

Write-Host ""
Write-Host "Terminal 1 - ChromaDB Vector Store:" -ForegroundColor Green
Write-Host "cd C:\Users\samne\PycharmProjects\ai_assignments" -ForegroundColor Yellow
Write-Host "& C:\Users\samne\PycharmProjects\ai_assignments\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "chroma run --host 0.0.0.0 --port 8081 --path chroma_data" -ForegroundColor Yellow

Write-Host ""
Write-Host "Terminal 2 - MLflow Experiment Tracking:" -ForegroundColor Green
Write-Host "cd C:\Users\samne\PycharmProjects\ai_assignments" -ForegroundColor Yellow
Write-Host "& C:\Users\samne\PycharmProjects\ai_assignments\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000" -ForegroundColor Yellow

Write-Host ""
Write-Host "Terminal 3 - Enterprise LLMOps Platform:" -ForegroundColor Green
Write-Host "cd C:\Users\samne\PycharmProjects\ai_assignments" -ForegroundColor Yellow
Write-Host "& C:\Users\samne\PycharmProjects\ai_assignments\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "python -m src.enterprise_llmops.main --host 0.0.0.0 --port 8080" -ForegroundColor Yellow

Write-Host ""
Write-Host "Terminal 4 - Gradio Model Evaluation App:" -ForegroundColor Green
Write-Host "cd C:\Users\samne\PycharmProjects\ai_assignments" -ForegroundColor Yellow
Write-Host "& C:\Users\samne\PycharmProjects\ai_assignments\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "python -m src.gradio_app.main --host 0.0.0.0 --port 7860" -ForegroundColor Yellow

Write-Host ""
Write-Host "⏱️ Wait for all services to start, then run:" -ForegroundColor Cyan
Write-Host ".\scripts\basic-sync.ps1" -ForegroundColor Yellow

Write-Host ""
Write-Host "🌐 Service URLs:" -ForegroundColor Cyan
Write-Host "• Enterprise Platform: http://localhost:8080" -ForegroundColor White
Write-Host "• Gradio App: http://localhost:7860" -ForegroundColor White
Write-Host "• MLflow UI: http://localhost:5000" -ForegroundColor White
Write-Host "• ChromaDB: http://localhost:8081" -ForegroundColor White

Write-Host ""
Write-Host "🔐 Authentication Token:" -ForegroundColor Cyan
Write-Host "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTc1ODQ4NDA0Mn0.R3acFFWMofqsmuklauIeYuwf74D6lS0-vgdGp-Me6mE" -ForegroundColor Yellow

Write-Host ""
Write-Host "✅ All services started! Check the URLs above to verify." -ForegroundColor Green
