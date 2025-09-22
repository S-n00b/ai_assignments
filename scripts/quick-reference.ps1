# =============================================================================
# Quick Reference Commands for AI Assignments Project
# =============================================================================
# This script provides quick access to the most commonly used commands
# Usage: .\scripts\quick-reference.ps1
# =============================================================================

Write-Host "🚀 AI Assignments Project - Quick Reference" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green

Write-Host ""
Write-Host "📋 Core Server Commands:" -ForegroundColor Cyan
Write-Host "1. Activate Virtual Environment:" -ForegroundColor White
Write-Host "   & C:\Users\samne\PycharmProjects\ai_assignments\venv\Scripts\Activate.ps1" -ForegroundColor Yellow

Write-Host ""
Write-Host "2. Start Services (4 terminals):" -ForegroundColor White
Write-Host "   Terminal 1: chroma run --host 0.0.0.0 --port 8081 --path chroma_data" -ForegroundColor Yellow
Write-Host "   Terminal 2: mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000" -ForegroundColor Yellow
Write-Host "   Terminal 3: python -m src.enterprise_llmops.main --host 0.0.0.0 --port 8080" -ForegroundColor Yellow
Write-Host "   Terminal 4: python -m src.gradio_app.main --host 0.0.0.0 --port 7860" -ForegroundColor Yellow

Write-Host ""
Write-Host "🌐 Service URLs:" -ForegroundColor Cyan
Write-Host "   • Enterprise Platform: http://localhost:8080" -ForegroundColor White
Write-Host "   • Gradio App: http://localhost:7860" -ForegroundColor White
Write-Host "   • MLflow UI: http://localhost:5000" -ForegroundColor White
Write-Host "   • ChromaDB: http://localhost:8081" -ForegroundColor White

Write-Host ""
Write-Host "🔐 Authentication:" -ForegroundColor Cyan
Write-Host "   Current Token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTc1ODQ4NDA0Mn0.R3acFFWMofqsmuklauIeYuwf74D6lS0-vgdGp-Me6mE" -ForegroundColor Yellow
Write-Host "   Get New Token: .\scripts\get-auth-token.ps1" -ForegroundColor Yellow

Write-Host ""
Write-Host "🔄 Sync Commands:" -ForegroundColor Cyan
Write-Host "   • Basic Sync: .\scripts\basic-sync.ps1" -ForegroundColor White
Write-Host "   • Generate Dataset: python scripts\generate_evaluation_dataset.py" -ForegroundColor White
Write-Host "   • Check Prompts: ls cache\ai_tool_prompts\" -ForegroundColor White

Write-Host ""
Write-Host "🧪 Testing Commands:" -ForegroundColor Cyan
Write-Host "   • Run Tests: pytest tests/ -v" -ForegroundColor White
Write-Host "   • Format Code: black src/ tests/" -ForegroundColor White
Write-Host "   • Lint Code: flake8 src/ tests/" -ForegroundColor White

Write-Host ""
Write-Host "📚 Documentation:" -ForegroundColor Cyan
Write-Host "   • Build Docs: cd docs/ && mkdocs build" -ForegroundColor White
Write-Host "   • Serve Docs: cd docs/ && mkdocs serve" -ForegroundColor White

Write-Host ""
Write-Host "🔍 Health Checks:" -ForegroundColor Cyan
Write-Host "   • Enterprise: curl http://localhost:8080/health" -ForegroundColor White
Write-Host "   • MLflow: curl http://localhost:5000/health" -ForegroundColor White
Write-Host "   • ChromaDB: curl http://localhost:8081/api/v1/heartbeat" -ForegroundColor White

Write-Host ""
Write-Host "📊 API Endpoints:" -ForegroundColor Cyan
Write-Host "   • Models: http://localhost:8080/api/models" -ForegroundColor White
Write-Host "   • Experiments: http://localhost:8080/api/experiments" -ForegroundColor White
Write-Host "   • Ollama Models: http://localhost:8080/api/ollama/models" -ForegroundColor White
Write-Host "   • Prompt Cache: http://localhost:8080/api/prompts/cache/summary" -ForegroundColor White

Write-Host ""
Write-Host "🛠️ Troubleshooting:" -ForegroundColor Cyan
Write-Host "   • Check Ports: netstat -an | findstr \":8080\"" -ForegroundColor White
Write-Host "   • Check Processes: tasklist /fi \"imagename eq python.exe\"" -ForegroundColor White
Write-Host "   • View Logs: type logs\llmops.log" -ForegroundColor White

Write-Host ""
Write-Host "🎯 Quick Start Workflow:" -ForegroundColor Cyan
Write-Host "   1. Activate venv" -ForegroundColor White
Write-Host "   2. Start all 4 services" -ForegroundColor White
Write-Host "   3. Run: .\scripts\basic-sync.ps1" -ForegroundColor White
Write-Host "   4. Access: http://localhost:8080/docs" -ForegroundColor White

Write-Host ""
Write-Host "📁 Key Files:" -ForegroundColor Cyan
Write-Host "   • SERVER_COMMANDS.md - Complete command reference" -ForegroundColor White
Write-Host "   • scripts/basic-sync.ps1 - Model sync script" -ForegroundColor White
Write-Host "   • data/evaluation_datasets/ - Generated datasets" -ForegroundColor White

Write-Host ""
Write-Host "✨ For complete documentation, see: SERVER_COMMANDS.md" -ForegroundColor Green
