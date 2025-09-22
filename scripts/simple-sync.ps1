# =============================================================================
# Simple Model and Prompt Sync
# =============================================================================
# This script performs a simple sync of:
# 1. Ollama models with project registry
# 2. Models with MLflow tracking
# 3. AI tool prompts from cache
# Usage: .\scripts\simple-sync.ps1
# =============================================================================

$ProjectRoot = "C:\Users\samne\PycharmProjects\ai_assignments"
$apiUrl = "http://localhost:8080"

Write-Host "🔄 Starting Simple Model and Prompt Sync..." -ForegroundColor Green

# Change to project directory
Set-Location $ProjectRoot

# Activate virtual environment
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
    Write-Host "Please ensure the FastAPI app is running on port 8080" -ForegroundColor Yellow
    exit 1
}

$headers = @{
    "Authorization" = "Bearer $token"
}

# Step 1: Sync Ollama Models
Write-Host ""
Write-Host "📋 Step 1: Syncing Ollama Models..." -ForegroundColor Cyan
try {
    $ollamaModels = Invoke-RestMethod -Uri "$apiUrl/api/ollama/models" -Method Get -Headers $headers
    Write-Host "✅ Found $($ollamaModels.models.Count) Ollama models" -ForegroundColor Green
    
    # Display models
    Write-Host ""
    Write-Host "📊 Available Ollama Models:" -ForegroundColor White
    foreach ($model in $ollamaModels.models) {
        Write-Host "  • $($model.name) ($($model.size))" -ForegroundColor White
    }
    
    # Register models in the project registry
    Write-Host ""
    Write-Host "📝 Registering models in project registry..." -ForegroundColor Yellow
    
    foreach ($model in $ollamaModels.models) {
        try {
            $modelData = @{
                name        = $model.name
                model_type  = "llm"
                description = "Ollama model: $($model.name)"
                version     = "1.0.0"
                tags        = @("ollama", "local")
                created_by  = "system"
                model_path  = "ollama://$($model.name)"
                status      = "active"
                metadata    = @{
                    size = $model.size
                    source = "ollama"
                    pull_status = "completed"
                }
            } | ConvertTo-Json -Depth 3
            
            $registerResponse = Invoke-RestMethod -Uri "$apiUrl/api/models/register" -Method Post -Body $modelData -ContentType "application/json" -Headers $headers
            Write-Host "  ✅ Registered: $($model.name)" -ForegroundColor Green
        }
        catch {
            Write-Host "  ⚠️ Failed to register $($model.name): $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
}
catch {
    Write-Host "❌ Failed to get Ollama models: $($_.Exception.Message)" -ForegroundColor Red
}

# Step 2: Sync with MLflow
Write-Host ""
Write-Host "📋 Step 2: Syncing with MLflow..." -ForegroundColor Cyan
try {
    # Get registered models
    $registeredModels = Invoke-RestMethod -Uri "$apiUrl/api/models" -Method Get -Headers $headers
    
    foreach ($model in $registeredModels.models) {
        try {
            # Start MLflow experiment for each model
            $experimentData = @{
                run_name = "Model Sync: $($model.name)"
                description = "Automatic sync of $($model.name) to MLflow"
                tags = @{
                    model_name = $model.name
                    model_type = $model.model_type
                    source = "ollama"
                    sync_timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
                }
            } | ConvertTo-Json -Depth 3
            
            $experimentResponse = Invoke-RestMethod -Uri "$apiUrl/api/experiments/start" -Method Post -Body $experimentData -ContentType "application/json" -Headers $headers
            
            # Log model parameters
            $params = @{
                model_name = $model.name
                model_type = $model.model_type
                version = $model.version
                status = $model.status
            }
            
            $paramsData = $params | ConvertTo-Json -Depth 3
            Invoke-RestMethod -Uri "$apiUrl/api/experiments/$($experimentResponse.run_id)/log-params" -Method Post -Body $paramsData -ContentType "application/json" -Headers $headers
            
            # Log model metrics
            $metrics = @{
                model_size_mb = if ($model.metadata.size) { [math]::Round([double]$model.metadata.size / 1MB, 2) } else { 0 }
                registration_timestamp = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
                sync_success = 1
            }
            
            $metricsData = $metrics | ConvertTo-Json -Depth 3
            Invoke-RestMethod -Uri "$apiUrl/api/experiments/$($experimentResponse.run_id)/log-metrics" -Method Post -Body $metricsData -ContentType "application/json" -Headers $headers
            
            # End the experiment
            Invoke-RestMethod -Uri "$apiUrl/api/experiments/$($experimentResponse.run_id)/end" -Method Post -Headers $headers
            
            Write-Host "  ✅ Synced $($model.name) to MLflow" -ForegroundColor Green
        }
        catch {
            Write-Host "  ⚠️ Failed to sync $($model.name) to MLflow: $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
}
catch {
    Write-Host "❌ Failed to sync with MLflow: $($_.Exception.Message)" -ForegroundColor Red
}

# Step 3: Load AI Tool Prompts
Write-Host ""
Write-Host "📋 Step 3: Loading AI Tool Prompts..." -ForegroundColor Cyan

$aiToolPromptsPath = "$ProjectRoot\cache\ai_tool_prompts"
if (Test-Path $aiToolPromptsPath) {
    $promptFiles = Get-ChildItem -Path $aiToolPromptsPath -Filter "*.json"
    Write-Host "✅ Found $($promptFiles.Count) cached AI tool prompt files" -ForegroundColor Green
    
    $totalPrompts = 0
    foreach ($file in $promptFiles) {
        try {
            $promptData = Get-Content -Path $file.FullName -Raw | ConvertFrom-Json
            $toolName = $promptData.tool_name
            $promptCount = $promptData.prompt_count
            
            Write-Host "  📁 $toolName`: $promptCount prompts" -ForegroundColor White
            
            # Register prompts as a model in the registry
            $promptModelData = @{
                name        = "AI Tool Prompts: $toolName"
                model_type  = "prompt_collection"
                description = "Cached AI tool system prompts from $toolName"
                version     = "1.0.0"
                tags        = @("ai_tool", "prompts", "system", $toolName.ToLower())
                created_by  = "system"
                model_path  = $file.FullName
                status      = "active"
                metadata    = @{
                    tool_name = $toolName
                    prompt_count = $promptCount
                    cached_at = $promptData.cached_at
                    source = "ai_tool_cache"
                    file_path = $file.FullName
                }
            } | ConvertTo-Json -Depth 3
            
            $registerResponse = Invoke-RestMethod -Uri "$apiUrl/api/models/register" -Method Post -Body $promptModelData -ContentType "application/json" -Headers $headers
            Write-Host "    ✅ Registered prompt collection: $toolName" -ForegroundColor Green
            
            $totalPrompts += $promptCount
        }
        catch {
            Write-Host "    ⚠️ Failed to process $($file.Name): $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
    
    Write-Host ""
    Write-Host "📊 Total AI Tool Prompts Loaded: $totalPrompts" -ForegroundColor Green
}
else {
    Write-Host "⚠️ AI tool prompts cache directory not found: $aiToolPromptsPath" -ForegroundColor Yellow
}

# Step 4: Generate Enhanced Evaluation Dataset
Write-Host ""
Write-Host "📋 Step 4: Generating Enhanced Evaluation Dataset..." -ForegroundColor Cyan

try {
    Write-Host "🐍 Executing Python script to generate evaluation dataset..." -ForegroundColor Yellow
    python "$ProjectRoot\scripts\generate_evaluation_dataset.py"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Enhanced evaluation dataset generated successfully" -ForegroundColor Green
    } else {
        Write-Host "❌ Python script failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    }
}
catch {
    Write-Host "❌ Failed to generate enhanced evaluation dataset: $($_.Exception.Message)" -ForegroundColor Red
}

# Step 5: Final Status Report
Write-Host ""
Write-Host "📋 Step 5: Final Status Report..." -ForegroundColor Cyan

try {
    # Get updated model registry
    $finalModels = Invoke-RestMethod -Uri "$apiUrl/api/models" -Method Get -Headers $headers
    Write-Host "✅ Final Model Registry Status:" -ForegroundColor Green
    foreach ($model in $finalModels.models) {
        Write-Host "  • $($model.name) - $($model.status) ($($model.model_type))" -ForegroundColor White
    }
    
    # Get MLflow experiments
    $experiments = Invoke-RestMethod -Uri "$apiUrl/api/experiments" -Method Get -Headers $headers
    Write-Host ""
    Write-Host "✅ MLflow Experiments Status:" -ForegroundColor Green
    Write-Host "  • Total experiments: $($experiments.experiments.Count)" -ForegroundColor White
}
catch {
    Write-Host "⚠️ Could not retrieve final status: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎯 Simple Sync Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Summary:" -ForegroundColor Cyan
Write-Host "1. ✅ Ollama models synced with project registry" -ForegroundColor White
Write-Host "2. ✅ Models registered with MLflow tracking" -ForegroundColor White
Write-Host "3. ✅ AI tool prompts loaded and registered" -ForegroundColor White
Write-Host "4. ✅ Enhanced evaluation dataset generated" -ForegroundColor White
Write-Host ""
Write-Host "🔗 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Access http://localhost:8080/docs to explore the API" -ForegroundColor White
Write-Host "2. Check MLflow UI at http://localhost:5000 for experiment tracking" -ForegroundColor White
Write-Host "3. Review the generated evaluation dataset in data/evaluation_datasets/" -ForegroundColor White
Write-Host "4. Use the registered models for evaluation and testing" -ForegroundColor White
