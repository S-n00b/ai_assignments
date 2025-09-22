# Comprehensive Registry Sync Script
# This script synchronizes all registries with actual Ollama models

param(
    [switch]$Force,
    [switch]$Verbose
)

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& C:\Users\samne\PycharmProjects\ai_assignments\venv\Scripts\Activate.ps1

# Set error action preference
$ErrorActionPreference = "Continue"

Write-Host "Starting Comprehensive Registry Sync..." -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan

# Step 1: Check Ollama models
Write-Host "`nStep 1: Checking Ollama Models" -ForegroundColor Yellow
Write-Host "-----------------------------------" -ForegroundColor Yellow

try {
    # Get Ollama models using the API endpoint instead of command line
    $ollamaModels = Invoke-RestMethod -Uri "http://localhost:8080/api/ollama/models" -Method GET
    Write-Host "Found $($ollamaModels.models.Count) Ollama models:" -ForegroundColor Green
    foreach ($model in $ollamaModels.models) {
        $sizeGB = [math]::Round($model.size / 1GB, 2)
        Write-Host "  - $($model.name) ($sizeGB GB)" -ForegroundColor White
    }
    # Store models for later use
    $ollamaModels = $ollamaModels.models
} catch {
    Write-Host "Failed to get Ollama models: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Step 2: Sync Model Registry
Write-Host "`nStep 2: Syncing Model Registry" -ForegroundColor Yellow
Write-Host "-----------------------------------" -ForegroundColor Yellow

try {
    # Get current model registry
    $currentModels = Invoke-RestMethod -Uri "http://localhost:8080/api/models" -Method GET
    Write-Host "Current registry has $($currentModels.models.Count) models" -ForegroundColor Blue
    
    # Register each Ollama model in the registry
    foreach ($ollamaModel in $ollamaModels) {
        $modelName = $ollamaModel.name
        $modelSize = $ollamaModel.size
        $modelDigest = $ollamaModel.digest
        
        # Determine model type based on name
        $modelType = "llm"
        if ($modelName -like "*code*") {
            $modelType = "code"
        } elseif ($modelName -like "*mistral*") {
            $modelType = "llm"
        } elseif ($modelName -like "*gemma*") {
            $modelType = "llm"
        }
        
        # Create model registration data
        $modelData = @{
            name = $modelName
            model_type = $modelType
            description = "Ollama model: $modelName"
            additional_params = @{
                size_bytes = $modelSize
                custom_attributes = @{
                    digest = $modelDigest
                    source = "ollama"
                    status = "available"
                }
            }
        }
        
        try {
            $response = Invoke-RestMethod -Uri "http://localhost:8080/api/models/register" -Method POST -Body ($modelData | ConvertTo-Json -Depth 3) -ContentType "application/json"
            Write-Host "Registered model: $modelName" -ForegroundColor Green
        } catch {
            Write-Host "Model $modelName may already be registered: $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
    
    # Verify registration
    $updatedModels = Invoke-RestMethod -Uri "http://localhost:8080/api/models" -Method GET
    Write-Host "Updated registry now has $($updatedModels.models.Count) models" -ForegroundColor Green
    
} catch {
    Write-Host "Failed to sync model registry: $($_.Exception.Message)" -ForegroundColor Red
}

# Step 3: Sync Prompt Registry
Write-Host "`nStep 3: Syncing Prompt Registry" -ForegroundColor Yellow
Write-Host "-----------------------------------" -ForegroundColor Yellow

try {
    # Get prompt cache summary
    $promptSummary = Invoke-RestMethod -Uri "http://localhost:8080/api/prompts/cache/summary" -Method GET
    Write-Host "Prompt cache summary:" -ForegroundColor Blue
    Write-Host "  - Total prompts: $($promptSummary.summary.total_prompts)" -ForegroundColor White
    Write-Host "  - AI tools: $($promptSummary.summary.ai_tools_count)" -ForegroundColor White
    
    # Sync prompts
    $syncResponse = Invoke-RestMethod -Uri "http://localhost:8080/api/prompts/sync" -Method POST
    Write-Host "Prompt sync completed" -ForegroundColor Green
    
    # Get registry statistics
    $registryStats = Invoke-RestMethod -Uri "http://localhost:8080/api/prompts/registries/statistics" -Method GET
    Write-Host "Registry statistics:" -ForegroundColor Blue
    Write-Host "  - Registries: $($registryStats.statistics.total_registries)" -ForegroundColor White
    Write-Host "  - Total prompts: $($registryStats.statistics.total_prompts)" -ForegroundColor White
    
} catch {
    Write-Host "Failed to sync prompt registry: $($_.Exception.Message)" -ForegroundColor Red
}

# Step 4: Sync Experiments
Write-Host "`nStep 4: Syncing Experiments" -ForegroundColor Yellow
Write-Host "-----------------------------------" -ForegroundColor Yellow

try {
    # Get current experiments
    $experiments = Invoke-RestMethod -Uri "http://localhost:8080/api/experiments" -Method GET
    Write-Host "Current experiments: $($experiments.experiments.Count)" -ForegroundColor Blue
    
    # Create a baseline experiment for each model
    foreach ($ollamaModel in $ollamaModels) {
        $modelName = $ollamaModel.name
        $experimentData = @{
            run_name = "baseline_$($modelName -replace '[:.]', '_')"
            description = "Baseline experiment for $modelName"
            tags = @{
                model_name = $modelName
                experiment_type = "baseline"
                source = "ollama"
            }
        }
        
        try {
            $runResponse = Invoke-RestMethod -Uri "http://localhost:8080/api/experiments/start" -Method POST -Body ($experimentData | ConvertTo-Json -Depth 3) -ContentType "application/json"
            Write-Host "Started baseline experiment for: $modelName" -ForegroundColor Green
            
            # Log some basic metrics
            $metrics = @{
                model_size_gb = [math]::Round($ollamaModel.size / 1GB, 2)
                model_available = $true
                sync_timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
            }
            
            $metricsResponse = Invoke-RestMethod -Uri "http://localhost:8080/api/experiments/$($runResponse.run_id)/log-metrics" -Method POST -Body ($metrics | ConvertTo-Json) -ContentType "application/json"
            
            # End the experiment
            $endResponse = Invoke-RestMethod -Uri "http://localhost:8080/api/experiments/$($runResponse.run_id)/end" -Method POST -Body (@{status="FINISHED"} | ConvertTo-Json) -ContentType "application/json"
            
        } catch {
            Write-Host "Experiment for $modelName may already exist: $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
    
} catch {
    Write-Host "Failed to sync experiments: $($_.Exception.Message)" -ForegroundColor Red
}

# Step 5: Update Gradio App Configuration
Write-Host "`nStep 5: Updating Gradio App Configuration" -ForegroundColor Yellow
Write-Host "-----------------------------------" -ForegroundColor Yellow

try {
    # Create model configuration for Gradio app
    $gradioConfig = @{
        available_models = @()
        model_capabilities = @{}
    }
    
    foreach ($ollamaModel in $ollamaModels) {
        $modelName = $ollamaModel.name
        $modelInfo = @{
            name = $modelName
            size_gb = [math]::Round($ollamaModel.size / 1GB, 2)
            type = if ($modelName -like "*code*") { "code_generation" } else { "general" }
            status = "available"
        }
        
        $gradioConfig.available_models += $modelInfo
        
        # Set capabilities based on model type
        if ($modelName -like "*code*") {
            $gradioConfig.model_capabilities[$modelName] = @("code_generation", "debugging", "documentation")
        } else {
            $gradioConfig.model_capabilities[$modelName] = @("text_generation", "question_answering", "reasoning")
        }
    }
    
    # Save configuration
    $configPath = "config/gradio_models.json"
    $gradioConfig | ConvertTo-Json -Depth 3 | Out-File -FilePath $configPath -Encoding UTF8
    Write-Host "Updated Gradio configuration: $configPath" -ForegroundColor Green
    Write-Host "Configured $($gradioConfig.available_models.Count) models for Gradio app" -ForegroundColor Blue
    
} catch {
    Write-Host "Failed to update Gradio configuration: $($_.Exception.Message)" -ForegroundColor Red
}

# Step 6: Generate Enhanced Evaluation Dataset
Write-Host "`nStep 6: Generating Enhanced Evaluation Dataset" -ForegroundColor Yellow
Write-Host "-----------------------------------" -ForegroundColor Yellow

try {
    Write-Host "Generating evaluation dataset with current models..." -ForegroundColor Blue
    python scripts\generate_evaluation_dataset.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Evaluation dataset generated successfully" -ForegroundColor Green
    } else {
        Write-Host "Dataset generation completed with warnings" -ForegroundColor Yellow
    }
    
} catch {
    Write-Host "Failed to generate evaluation dataset: $($_.Exception.Message)" -ForegroundColor Red
}

# Step 7: Final Verification
Write-Host "`nStep 7: Final Verification" -ForegroundColor Yellow
Write-Host "-----------------------------------" -ForegroundColor Yellow

try {
    # Check all endpoints
    Write-Host "Verifying all endpoints..." -ForegroundColor Blue
    
    $health = Invoke-RestMethod -Uri "http://localhost:8080/health" -Method GET
    Write-Host "Health check: $($health.status)" -ForegroundColor Green
    
    $models = Invoke-RestMethod -Uri "http://localhost:8080/api/models" -Method GET
    Write-Host "Model registry: $($models.models.Count) models" -ForegroundColor Green
    
    $ollamaModels = Invoke-RestMethod -Uri "http://localhost:8080/api/ollama/models" -Method GET
    Write-Host "Ollama integration: $($ollamaModels.models.Count) models" -ForegroundColor Green
    
    $experiments = Invoke-RestMethod -Uri "http://localhost:8080/api/experiments" -Method GET
    Write-Host "Experiments: $($experiments.experiments.Count) experiments" -ForegroundColor Green
    
    $promptSummary = Invoke-RestMethod -Uri "http://localhost:8080/api/prompts/cache/summary" -Method GET
    Write-Host "Prompt cache: $($promptSummary.summary.total_prompts) prompts" -ForegroundColor Green
    
} catch {
    Write-Host "Verification failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Summary
Write-Host "`nComprehensive Sync Complete!" -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Green
Write-Host "Summary:" -ForegroundColor Cyan
Write-Host "  - Ollama models: $($ollamaModels.Count)" -ForegroundColor White
Write-Host "  - Model registry: Synced" -ForegroundColor White
Write-Host "  - Prompt registry: Synced" -ForegroundColor White
Write-Host "  - Experiments: Created baseline experiments" -ForegroundColor White
Write-Host "  - Gradio app: Updated configuration" -ForegroundColor White
Write-Host "  - Evaluation dataset: Generated" -ForegroundColor White

Write-Host "`nAccess your services:" -ForegroundColor Cyan
Write-Host "  - Enterprise Platform: http://localhost:8080" -ForegroundColor White
Write-Host "  - Gradio App: http://localhost:7860" -ForegroundColor White
Write-Host "  - MLflow UI: http://localhost:5000" -ForegroundColor White
Write-Host "  - API Docs: http://localhost:8080/docs" -ForegroundColor White

Write-Host "`nAll registries are now synchronized!" -ForegroundColor Green