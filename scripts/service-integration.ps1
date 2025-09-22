# Enterprise LLMOps Service Integration Script
# This script connects and validates all enterprise services

param(
    [switch]$CheckOnly,
    [switch]$StartServices,
    [switch]$FullIntegration,
    [string]$ConfigPath = "config/enterprise-config.yaml"
)

# Import required modules
Import-Module -Name ".\venv\Lib\site-packages" -ErrorAction SilentlyContinue

# Color output functions
function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Color
}

function Write-Success { param([string]$Message) Write-ColorOutput "[OK] $Message" "Green" }
function Write-Error { param([string]$Message) Write-ColorOutput "[ERROR] $Message" "Red" }
function Write-Warning { param([string]$Message) Write-ColorOutput "[WARN] $Message" "Yellow" }
function Write-Info { param([string]$Message) Write-ColorOutput "i $Message" "Cyan" }

# Service configuration
$Services = @{
    Ollama = @{
        Url = "http://localhost:11434"
        HealthEndpoint = "/api/tags"
        StartCommand = "ollama serve"
        Required = $true
    }
    MLflow = @{
        Url = "http://localhost:5000"
        HealthEndpoint = "/health"
        StartCommand = "mlflow server --host 0.0.0.0 --port 5000"
        Required = $true
    }
    Chroma = @{
        Url = "http://localhost:8081"
        HealthEndpoint = "/api/v1/heartbeat"
        StartCommand = "chroma run --host 0.0.0.0 --port 8081"
        Required = $true
    }
    Weaviate = @{
        Url = "http://localhost:8080"
        HealthEndpoint = "/v1/meta"
        StartCommand = "docker run -p 8080:8080 semitechnologies/weaviate:1.21.6"
        Required = $false
    }
    Prometheus = @{
        Url = "http://localhost:9090"
        HealthEndpoint = "/-/healthy"
        StartCommand = "prometheus --config.file=prometheus.yml"
        Required = $false
    }
    Grafana = @{
        Url = "http://localhost:3000"
        HealthEndpoint = "/api/health"
        StartCommand = "grafana-server --config grafana.ini"
        Required = $false
    }
}

# Function to check service health
function Test-ServiceHealth {
    param([string]$ServiceName, [hashtable]$Config)
    
    try {
        $response = Invoke-WebRequest -Uri "$($Config.Url)$($Config.HealthEndpoint)" -TimeoutSec 5 -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            Write-Success "$ServiceName is running (Status: $($response.StatusCode))"
            return $true
        } else {
            Write-Warning "$ServiceName responded with status: $($response.StatusCode)"
            return $false
        }
    }
    catch {
        Write-Error "$ServiceName is not running or not accessible"
        return $false
    }
}

# Function to start a service
function Start-Service {
    param([string]$ServiceName, [hashtable]$Config)
    
    Write-Info "Starting $ServiceName..."
    
    try {
        if ($ServiceName -eq "Ollama") {
            # Check if Ollama is installed
            $ollamaVersion = Get-Command ollama -ErrorAction SilentlyContinue
            if (-not $ollamaVersion) {
                Write-Error "Ollama is not installed. Please install Ollama first."
                return $false
            }
            
            # Start Ollama in background
            Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
            Start-Sleep -Seconds 3
        }
        elseif ($ServiceName -eq "MLflow") {
            # Start MLflow server
            $mlflowCommand = "mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db"
            Start-Process -FilePath "python" -ArgumentList "-m", "mlflow.server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:///mlflow.db" -WindowStyle Hidden
            Start-Sleep -Seconds 5
        }
        elseif ($ServiceName -eq "Chroma") {
            # Start Chroma server
            Start-Process -FilePath "python" -ArgumentList "-m", "chromadb.server", "--host", "0.0.0.0", "--port", "8081" -WindowStyle Hidden
            Start-Sleep -Seconds 3
        }
        elseif ($ServiceName -eq "Weaviate") {
            # Start Weaviate using Docker
            $dockerCheck = Get-Command docker -ErrorAction SilentlyContinue
            if ($dockerCheck) {
                Start-Process -FilePath "docker" -ArgumentList "run", "-d", "-p", "8080:8080", "--name", "weaviate", "semitechnologies/weaviate:1.21.6" -WindowStyle Hidden
                Start-Sleep -Seconds 10
            } else {
                Write-Warning "Docker not found. Skipping Weaviate startup."
                return $false
            }
        }
        
        # Wait and test the service
        Start-Sleep -Seconds 2
        return Test-ServiceHealth $ServiceName $Config
    }
    catch {
        Write-Error "Failed to start $ServiceName`: $_"
        return $false
    }
}

# Function to check all services
function Test-AllServices {
    Write-Info "Checking service status..."
    Write-Host "`n" + "="*60
    Write-Info "ENTERPRISE LLMOPS SERVICE STATUS CHECK"
    Write-Host "="*60
    
    $runningServices = @()
    $failedServices = @()
    
    foreach ($service in $Services.Keys) {
        $config = $Services[$service]
        Write-Host "`nChecking $service..."
        
        if (Test-ServiceHealth $service $config) {
            $runningServices += $service
        } else {
            $failedServices += $service
            if ($config.Required) {
                Write-Warning "$service is required but not running"
            }
        }
    }
    
    Write-Host "`n" + "="*60
    Write-Info "SERVICE STATUS SUMMARY"
    Write-Host "="*60
    Write-Success "Running Services: $($runningServices -join ', ')"
    if ($failedServices.Count -gt 0) {
        Write-Warning "Failed Services: $($failedServices -join ', ')"
    }
    
    return @{
        Running = $runningServices
        Failed = $failedServices
    }
}

# Function to start all required services
function Start-RequiredServices {
    Write-Info "Starting required services..."
    
    $status = Test-AllServices
    $requiredFailed = $status.Failed | Where-Object { $Services[$_].Required }
    
    if ($requiredFailed.Count -eq 0) {
        Write-Success "All required services are already running!"
        return $true
    }
    
    foreach ($service in $requiredFailed) {
        if ($Services[$service].Required) {
            Write-Info "Attempting to start required service: $service"
            Start-Service $service $Services[$service]
        }
    }
    
    # Re-check after starting services
    Start-Sleep -Seconds 3
    $newStatus = Test-AllServices
    $stillFailed = $newStatus.Failed | Where-Object { $Services[$_].Required }
    
    if ($stillFailed.Count -eq 0) {
        Write-Success "All required services are now running!"
        return $true
    } else {
        Write-Error "Some required services failed to start: $($stillFailed -join ', ')"
        return $false
    }
}

# Function to run full integration test
function Test-FullIntegration {
    Write-Info "Running full integration test..."
    Write-Host "`n" + "="*60
    Write-Info "ENTERPRISE LLMOPS INTEGRATION TEST"
    Write-Host "="*60
    
    # Test service connectivity
    $serviceStatus = Test-AllServices
    if ($serviceStatus.Failed.Count -gt 0) {
        Write-Error "Integration test failed: Some services are not running"
        return $false
    }
    
    # Test Ollama model listing
    Write-Info "Testing Ollama model integration..."
    try {
        $ollamaModels = Invoke-RestMethod -Uri "$($Services.Ollama.Url)/api/tags" -TimeoutSec 10
        Write-Success "Ollama models available: $($ollamaModels.models.Count)"
        if ($ollamaModels.models.Count -eq 0) {
            Write-Warning "No models found in Ollama. Consider pulling some models."
        }
    }
    catch {
        Write-Error "Failed to connect to Ollama: $_"
    }
    
    # Test MLflow connectivity
    Write-Info "Testing MLflow integration..."
    try {
        $mlflowHealth = Invoke-WebRequest -Uri "$($Services.MLflow.Url)/health" -TimeoutSec 10
        Write-Success "MLflow is accessible"
    }
    catch {
        Write-Error "Failed to connect to MLflow: $_"
    }
    
    # Test Chroma connectivity
    Write-Info "Testing Chroma integration..."
    try {
        $chromaHealth = Invoke-WebRequest -Uri "$($Services.Chroma.Url)$($Services.Chroma.HealthEndpoint)" -TimeoutSec 10
        Write-Success "Chroma is accessible"
    }
    catch {
        Write-Error "Failed to connect to Chroma: $_"
    }
    
    # Test Gradio app
    Write-Info "Testing Gradio application..."
    try {
        $gradioHealth = Invoke-WebRequest -Uri "http://localhost:7860" -TimeoutSec 10
        Write-Success "Gradio app is accessible"
    }
    catch {
        Write-Warning "Gradio app not running (expected if not started)"
    }
    
    # Test Enterprise FastAPI
    Write-Info "Testing Enterprise FastAPI..."
    try {
        $fastapiHealth = Invoke-WebRequest -Uri "http://localhost:8080/docs" -TimeoutSec 10
        Write-Success "Enterprise FastAPI is accessible"
    }
    catch {
        Write-Warning "Enterprise FastAPI not running (expected if not started)"
    }
    
    Write-Host "`n" + "="*60
    Write-Success "INTEGRATION TEST COMPLETED"
    Write-Host "="*60
    
    return $true
}

# Main execution
Write-ColorOutput "`n[START] ENTERPRISE LLMOPS SERVICE INTEGRATION SCRIPT" "Magenta"
Write-Host "="*60

if ($CheckOnly) {
    Test-AllServices
}
elseif ($StartServices) {
    Start-RequiredServices
}
elseif ($FullIntegration) {
    if (Start-RequiredServices) {
        Test-FullIntegration
    }
}
else {
    # Default: Check services and start if needed
    $status = Test-AllServices
    $requiredFailed = $status.Failed | Where-Object { $Services[$_].Required }
    
    if ($requiredFailed.Count -gt 0) {
        Write-Info "Some required services are not running. Attempting to start them..."
        Start-RequiredServices
    } else {
        Write-Success "All required services are running!"
    }
    
    # Offer to run full integration test
    $runIntegration = Read-Host "`nWould you like to run a full integration test? (y/N)"
    if ($runIntegration -eq "y" -or $runIntegration -eq "Y") {
        Test-FullIntegration
    }
}

Write-Info "`nService integration script completed."
Write-Host "`nNext steps:"
Write-Info "1. Start Gradio app: python -m src.gradio_app.main --host 0.0.0.0 --port 7860"
Write-Info "2. Start Enterprise FastAPI: python -m src.enterprise_llmops.main --host 0.0.0.0 --port 8080"
Write-Info "3. Access documentation: http://localhost:8082"
