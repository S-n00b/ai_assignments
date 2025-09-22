# Test Lenovo Integration Script
# Tests the integrated Lenovo AI Architecture pitch page and service styling

Write-Host "🚀 Testing Lenovo AI Architecture Integration" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green

# Check if virtual environment is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "⚠️  Virtual environment not activated. Activating..." -ForegroundColor Yellow
    & C:\Users\samne\PycharmProjects\ai_assignments\venv\Scripts\Activate.ps1
}

# Test 1: Check if Lenovo pitch page exists
Write-Host "`n📄 Test 1: Checking Lenovo pitch page..." -ForegroundColor Cyan
if (Test-Path "lenovo_ai_architecture_pitch.html") {
    Write-Host "✅ Lenovo pitch page found" -ForegroundColor Green
}
else {
    Write-Host "❌ Lenovo pitch page not found" -ForegroundColor Red
    exit 1
}

# Test 2: Check if unified theme CSS exists
Write-Host "`n🎨 Test 2: Checking unified theme CSS..." -ForegroundColor Cyan
if (Test-Path "src/enterprise_llmops/frontend/lenovo-theme.css") {
    Write-Host "✅ Unified theme CSS found" -ForegroundColor Green
}
else {
    Write-Host "❌ Unified theme CSS not found" -ForegroundColor Red
    exit 1
}

# Test 3: Check if MkDocs Lenovo theme exists
Write-Host "`n📚 Test 3: Checking MkDocs Lenovo theme..." -ForegroundColor Cyan
if (Test-Path "docs/assets/css/lenovo-material-theme.css") {
    Write-Host "✅ MkDocs Lenovo theme found" -ForegroundColor Green
}
else {
    Write-Host "❌ MkDocs Lenovo theme not found" -ForegroundColor Red
    exit 1
}

# Test 4: Check FastAPI app modifications
Write-Host "`n🔧 Test 4: Checking FastAPI app modifications..." -ForegroundColor Cyan
$fastapiContent = Get-Content "src/enterprise_llmops/frontend/fastapi_app.py" -Raw
if ($fastapiContent -match "lenovo_ai_architecture_pitch.html") {
    Write-Host "✅ FastAPI app modified for Lenovo integration" -ForegroundColor Green
}
else {
    Write-Host "❌ FastAPI app not modified for Lenovo integration" -ForegroundColor Red
    exit 1
}

# Test 5: Check Gradio app modifications
Write-Host "`n🤖 Test 5: Checking Gradio app modifications..." -ForegroundColor Cyan
$gradioContent = Get-Content "src/gradio_app/main.py" -Raw
if ($gradioContent -match "lenovo-theme.css") {
    Write-Host "✅ Gradio app modified for Lenovo theme" -ForegroundColor Green
}
else {
    Write-Host "❌ Gradio app not modified for Lenovo theme" -ForegroundColor Red
    exit 1
}

# Test 6: Check MkDocs configuration
Write-Host "`n📖 Test 6: Checking MkDocs configuration..." -ForegroundColor Cyan
$mkdocsContent = Get-Content "docs/mkdocs.yml" -Raw
if ($mkdocsContent -match "lenovo-material-theme.css") {
    Write-Host "✅ MkDocs configured for Lenovo theme" -ForegroundColor Green
}
else {
    Write-Host "❌ MkDocs not configured for Lenovo theme" -ForegroundColor Red
    exit 1
}

# Test 7: Check if services can be started (basic check)
Write-Host "`n🌐 Test 7: Checking service startup capabilities..." -ForegroundColor Cyan

# Check if required Python modules exist
$requiredModules = @(
    "src.enterprise_llmops.frontend.fastapi_app",
    "src.gradio_app.main"
)

foreach ($module in $requiredModules) {
    $modulePath = $module -replace "\.", "/" -replace "src/", "src/"
    if (Test-Path "$modulePath.py") {
        Write-Host "✅ Module $module found" -ForegroundColor Green
    }
    else {
        Write-Host "❌ Module $module not found" -ForegroundColor Red
    }
}

# Test 8: Check port availability (basic check)
Write-Host "`n🔌 Test 8: Checking port availability..." -ForegroundColor Cyan
$ports = @(8080, 7860, 8082, 5000, 8000)
foreach ($port in $ports) {
    $connection = Test-NetConnection -ComputerName localhost -Port $port -InformationLevel Quiet -WarningAction SilentlyContinue
    if (-not $connection) {
        Write-Host "✅ Port $port is available" -ForegroundColor Green
    }
    else {
        Write-Host "⚠️  Port $port is in use (may be expected if services are running)" -ForegroundColor Yellow
    }
}

# Summary
Write-Host "`n🎉 Integration Test Summary" -ForegroundColor Green
Write-Host "=========================" -ForegroundColor Green
Write-Host "✅ Lenovo pitch page integration: Complete" -ForegroundColor Green
Write-Host "✅ Unified theme CSS: Complete" -ForegroundColor Green
Write-Host "✅ FastAPI integration: Complete" -ForegroundColor Green
Write-Host "✅ Gradio theme integration: Complete" -ForegroundColor Green
Write-Host "✅ MkDocs theme integration: Complete" -ForegroundColor Green

Write-Host "`n🚀 Ready to start services with Lenovo integration!" -ForegroundColor Green
Write-Host "`nTo start the integrated services:" -ForegroundColor Cyan
Write-Host "1. Start ChromaDB: chroma run --host 0.0.0.0 --port 8000 --path chroma_data" -ForegroundColor White
Write-Host "2. Start MLflow: mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000" -ForegroundColor White
Write-Host "3. Start FastAPI: python -m src.enterprise_llmops.main --host 0.0.0.0 --port 8080" -ForegroundColor White
Write-Host "4. Start Gradio: python -m src.gradio_app.main --host 0.0.0.0 --port 7860" -ForegroundColor White
Write-Host "5. Start MkDocs: cd docs && mkdocs serve --dev-addr 0.0.0.0:8082" -ForegroundColor White

Write-Host "`n🌐 Access URLs:" -ForegroundColor Cyan
Write-Host "• Lenovo Pitch (Main Landing): http://localhost:8080" -ForegroundColor White
Write-Host "• FastAPI Docs: http://localhost:8080/docs" -ForegroundColor White
Write-Host "• Gradio App: http://localhost:7860" -ForegroundColor White
Write-Host "• MkDocs: http://localhost:8082" -ForegroundColor White
Write-Host "• MLflow: http://localhost:5000" -ForegroundColor White

Write-Host "`n✨ All services now use unified Lenovo branding and styling!" -ForegroundColor Green
