# =============================================================================
# Basic Model Sync Script
# =============================================================================

$ProjectRoot = "C:\Users\samne\PycharmProjects\ai_assignments"
$apiUrl = "http://localhost:8080"

Write-Host "🔄 Starting Basic Model Sync..." -ForegroundColor Green

# Change to project directory and activate venv
Set-Location $ProjectRoot
& "$ProjectRoot\venv\Scripts\Activate.ps1"

# Get authentication token
Write-Host "🔐 Getting authentication token..." -ForegroundColor Yellow
$loginData = @{
    username = "admin"
    password = "admin"
} | ConvertTo-Json

try {
    $authResponse = Invoke-RestMethod -Uri "$apiUrl/api/auth/login" -Method Post -Body $loginData -ContentType "application/json"
    $token = $authResponse.access_token
    Write-Host "✅ Authentication successful" -ForegroundColor Green
}
catch {
    Write-Host "❌ Authentication failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

$headers = @{
    "Authorization" = "Bearer $token"
}

# Step 1: Get and register Ollama models
Write-Host ""
Write-Host "📋 Step 1: Syncing Ollama Models..." -ForegroundColor Cyan
try {
    $ollamaModels = Invoke-RestMethod -Uri "$apiUrl/api/ollama/models" -Method Get -Headers $headers
    Write-Host "✅ Found $($ollamaModels.models.Count) Ollama models" -ForegroundColor Green
    
    foreach ($model in $ollamaModels.models) {
        Write-Host "  • $($model.name) ($($model.size))" -ForegroundColor White
    }
}
catch {
    Write-Host "❌ Failed to get Ollama models: $($_.Exception.Message)" -ForegroundColor Red
}

# Step 2: Get registered models
Write-Host ""
Write-Host "📋 Step 2: Getting Registered Models..." -ForegroundColor Cyan
try {
    $registeredModels = Invoke-RestMethod -Uri "$apiUrl/api/models" -Method Get -Headers $headers
    Write-Host "✅ Found $($registeredModels.models.Count) registered models" -ForegroundColor Green
    
    foreach ($model in $registeredModels.models) {
        Write-Host "  • $($model.name) - $($model.status) ($($model.model_type))" -ForegroundColor White
    }
}
catch {
    Write-Host "❌ Failed to get registered models: $($_.Exception.Message)" -ForegroundColor Red
}

# Step 3: Generate evaluation dataset
Write-Host ""
Write-Host "📋 Step 3: Generating Evaluation Dataset..." -ForegroundColor Cyan
try {
    python "$ProjectRoot\scripts\generate_evaluation_dataset.py"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Enhanced evaluation dataset generated successfully" -ForegroundColor Green
    } else {
        Write-Host "❌ Python script failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    }
}
catch {
    Write-Host "❌ Failed to generate evaluation dataset: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "🎯 Basic Sync Complete!" -ForegroundColor Green
