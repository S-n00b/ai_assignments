# =============================================================================
# Sync Ollama Models with Project Registry
# =============================================================================
# This script syncs your Ollama models from the default location to the project
# Usage: .\scripts\sync-ollama-models.ps1
# =============================================================================

$ProjectRoot = "C:\Users\samne\PycharmProjects\ai_assignments"
$OllamaModelsPath = "C:\Users\samne\.ollama\models"
$ProjectModelsPath = "$ProjectRoot\data\model_registry\models"
$apiUrl = "http://localhost:8080"

Write-Host "Syncing Ollama models with project registry..." -ForegroundColor Green

# Change to project directory
Set-Location $ProjectRoot

# Activate virtual environment
& "$ProjectRoot\venv\Scripts\Activate.ps1"

# Create project models directory if it doesn't exist
if (-not (Test-Path $ProjectModelsPath)) {
    New-Item -ItemType Directory -Force -Path $ProjectModelsPath | Out-Null
    Write-Host "Created project models directory: $ProjectModelsPath" -ForegroundColor Green
}

# Get authentication token
Write-Host "Getting authentication token..." -ForegroundColor Yellow
$loginData = @{
    username = "admin"
    password = "admin"
} | ConvertTo-Json

try {
    $authResponse = Invoke-RestMethod -Uri "$apiUrl/api/auth/login" -Method Post -Body $loginData -ContentType "application/json"
    $token = $authResponse.access_token
    Write-Host "Authentication successful" -ForegroundColor Green
}
catch {
    Write-Host "Authentication failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Please ensure the FastAPI app is running on port 8080" -ForegroundColor Yellow
    exit 1
}

# Get list of Ollama models
Write-Host "Getting list of Ollama models..." -ForegroundColor Yellow
try {
    $headers = @{
        "Authorization" = "Bearer $token"
    }
    $ollamaModels = Invoke-RestMethod -Uri "$apiUrl/api/ollama/models" -Method Get -Headers $headers
    Write-Host "Found $($ollamaModels.models.Count) Ollama models" -ForegroundColor Green
    
    # Display models
    Write-Host ""
    Write-Host "Available Ollama Models:" -ForegroundColor Cyan
    foreach ($model in $ollamaModels.models) {
        Write-Host "  • $($model.name) ($($model.size))" -ForegroundColor White
    }
    
    # Register models in the project registry
    Write-Host ""
    Write-Host "Registering models in project registry..." -ForegroundColor Yellow
    
    foreach ($model in $ollamaModels.models) {
        try {
            $modelData = @{
                name        = $model.name
                model_type  = "llm"
                description = "Ollama model: $($model.name)"
                version     = "1.0.0"
                tags        = @("ollama", "local")
                created_by  = "system"
                model_path  = "$OllamaModelsPath\$($model.name)"
                status      = "active"
            } | ConvertTo-Json
            
            $registerResponse = Invoke-RestMethod -Uri "$apiUrl/api/models/register" -Method Post -Body $modelData -ContentType "application/json" -Headers $headers
            Write-Host "  Registered: $($model.name)" -ForegroundColor Green
        }
        catch {
            Write-Host "  Failed to register $($model.name): $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
    
    # Get updated model registry
    Write-Host ""
    Write-Host "Updated Model Registry:" -ForegroundColor Cyan
    try {
        $registeredModels = Invoke-RestMethod -Uri "$apiUrl/api/models" -Method Get -Headers $headers
        foreach ($model in $registeredModels.models) {
            Write-Host "  • $($model.name) - $($model.status)" -ForegroundColor White
        }
    }
    catch {
        Write-Host "  Could not retrieve registered models: $($_.Exception.Message)" -ForegroundColor Yellow
    }
    
}
catch {
    Write-Host "Failed to get Ollama models: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Please ensure Ollama is running and accessible" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "1. Use the token above to authorize API access" -ForegroundColor White
Write-Host "2. Access http://localhost:8080/docs and click 'Authorize'" -ForegroundColor White
Write-Host "3. Enter the Bearer token to unlock protected endpoints" -ForegroundColor White
Write-Host "4. Test endpoints like /api/models and /api/experiments" -ForegroundColor White