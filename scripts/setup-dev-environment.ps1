# =============================================================================
# AI Assignments - 5-Terminal Development Environment Setup
# =============================================================================
# This script sets up a complete development environment with 5 terminals
# running different components of the FastAPI app in the correct order
# with proper wait policies and service dependencies.
#
# Usage: .\scripts\setup-dev-environment.ps1
# =============================================================================

param(
    [switch]$SkipDependencies,
    [switch]$SkipServices,
    [switch]$TerminalsOnly,
    [string]$ConfigPath = "config/enterprise-config.yaml"
)

# Color output functions
function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Color
}

function Write-Success { param([string]$Message) Write-ColorOutput "✅ $Message" "Green" }
function Write-Error { param([string]$Message) Write-ColorOutput "❌ $Message" "Red" }
function Write-Warning { param([string]$Message) Write-ColorOutput "⚠️ $Message" "Yellow" }
function Write-Info { param([string]$Message) Write-ColorOutput "ℹ️ $Message" "Cyan" }
function Write-Header { param([string]$Message) Write-ColorOutput "🚀 $Message" "Magenta" }

# Global variables
$ProjectRoot = "C:\Users\samne\PycharmProjects\ai_assignments"
$VenvPath = "$ProjectRoot\venv\Scripts\Activate.ps1"
$TerminalProcesses = @{}

# Service configuration with startup order and dependencies
$Services = @{
    ChromaDB = @{
        Name = "ChromaDB Vector Store"
        Port = 8081
        HealthEndpoint = "/api/v1/heartbeat"
        StartCommand = "python -m chromadb.server --host 0.0.0.0 --port 8081"
        WaitTime = 5
        Required = $true
        Priority = 1
    }
    MLflow = @{
        Name = "MLflow Tracking Server"
        Port = 5000
        HealthEndpoint = "/health"
        StartCommand = "python -m mlflow.server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db"
        WaitTime = 8
        Required = $true
        Priority = 2
    }
    Ollama = @{
        Name = "Ollama LLM Server"
        Port = 11434
        HealthEndpoint = "/api/tags"
        StartCommand = "ollama serve"
        WaitTime = 3
        Required = $false
        Priority = 3
    }
    GradioApp = @{
        Name = "Gradio Model Evaluation App"
        Port = 7860
        HealthEndpoint = "/"
        StartCommand = "python -m src.gradio_app.main --host 0.0.0.0 --port 7860"
        WaitTime = 10
        Required = $true
        Priority = 4
        Dependencies = @("ChromaDB", "MLflow")
    }
    EnterpriseFastAPI = @{
        Name = "Enterprise FastAPI Platform"
        Port = 8080
        HealthEndpoint = "/health"
        StartCommand = "python -m src.enterprise_llmops.main --host 0.0.0.0 --port 8080"
        WaitTime = 12
        Required = $true
        Priority = 5
        Dependencies = @("ChromaDB", "MLflow", "GradioApp")
    }
}

# Function to check if a port is available
function Test-PortAvailable {
    param([int]$Port)
    try {
        $connection = New-Object System.Net.Sockets.TcpClient
        $connection.Connect("localhost", $Port)
        $connection.Close()
        return $false  # Port is in use
    }
    catch {
        return $true   # Port is available
    }
}

# Function to wait for service to be ready
function Wait-ForService {
    param([string]$ServiceName, [hashtable]$Config, [int]$MaxWaitTime = 60)
    
    $url = "http://localhost:$($Config.Port)$($Config.HealthEndpoint)"
    $elapsed = 0
    
    Write-Info "Waiting for $ServiceName to be ready..."
    
    while ($elapsed -lt $MaxWaitTime) {
        try {
            $response = Invoke-WebRequest -Uri $url -TimeoutSec 5 -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                Write-Success "$ServiceName is ready!"
                return $true
            }
        }
        catch {
            # Service not ready yet
        }
        
        Start-Sleep -Seconds 2
        $elapsed += 2
        Write-Host "." -NoNewline
    }
    
    Write-Warning "$ServiceName did not become ready within $MaxWaitTime seconds"
    return $false
}

# Function to start a service in a new terminal
function Start-ServiceInTerminal {
    param([string]$ServiceName, [hashtable]$Config)
    
    Write-Info "Starting $($Config.Name) in new terminal..."
    
    # Check if port is already in use
    if (-not (Test-PortAvailable $Config.Port)) {
        Write-Warning "Port $($Config.Port) is already in use. Service may already be running."
        return $true
    }
    
    # Create PowerShell command to run in new terminal
    $psCommand = @"
cd '$ProjectRoot'
& '$VenvPath'
Write-Host '🚀 Starting $($Config.Name)...' -ForegroundColor Green
Write-Host 'Port: $($Config.Port)' -ForegroundColor Cyan
Write-Host 'Health Check: http://localhost:$($Config.Port)$($Config.HealthEndpoint)' -ForegroundColor Cyan
Write-Host 'Press Ctrl+C to stop this service' -ForegroundColor Yellow
Write-Host ''
$($Config.StartCommand)
"@
    
    # Start new PowerShell window
    $process = Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", $psCommand -PassThru
    $TerminalProcesses[$ServiceName] = $process
    
    Write-Success "Started $($Config.Name) in terminal (PID: $($process.Id))"
    
    # Wait for service to be ready
    if ($Config.Required) {
        Start-Sleep -Seconds $Config.WaitTime
        return Wait-ForService $ServiceName $Config
    }
    
    return $true
}

# Function to check dependencies
function Test-ServiceDependencies {
    param([string]$ServiceName, [hashtable]$Config)
    
    if (-not $Config.Dependencies) {
        return $true
    }
    
    foreach ($dependency in $Config.Dependencies) {
        if (-not $TerminalProcesses.ContainsKey($dependency)) {
            Write-Error "Dependency $dependency is not running for $ServiceName"
            return $false
        }
    }
    
    return $true
}

# Function to install dependencies
function Install-Dependencies {
    Write-Header "Installing Dependencies"
    
    # Change to project directory
    Set-Location $ProjectRoot
    
    # Activate virtual environment
    Write-Info "Activating virtual environment..."
    & $VenvPath
    
    # Install main dependencies
    Write-Info "Installing main dependencies..."
    pip install -r config\requirements.txt
    
    # Install testing dependencies
    Write-Info "Installing testing dependencies..."
    pip install -r config\requirements-testing.txt
    
    # Install documentation dependencies
    Write-Info "Installing documentation dependencies..."
    pip install -r docs\requirements-docs.txt
    
    Write-Success "Dependencies installed successfully!"
}

# Function to create necessary directories
function Initialize-Directories {
    Write-Header "Creating Required Directories"
    
    $directories = @("logs", "chroma_db", "data", "data/model_registry")
    
    foreach ($dir in $directories) {
        $fullPath = Join-Path $ProjectRoot $dir
        if (-not (Test-Path $fullPath)) {
            New-Item -ItemType Directory -Force -Path $fullPath | Out-Null
            Write-Success "Created directory: $dir"
        } else {
            Write-Info "Directory already exists: $dir"
        }
    }
}

# Function to start all services in order
function Start-AllServices {
    Write-Header "Starting Services in Priority Order"
    
    # Sort services by priority
    $sortedServices = $Services.GetEnumerator() | Sort-Object { $_.Value.Priority }
    
    foreach ($service in $sortedServices) {
        $serviceName = $service.Key
        $config = $service.Value
        
        Write-Info "Processing service: $serviceName (Priority: $($config.Priority))"
        
        # Check dependencies
        if (-not (Test-ServiceDependencies $serviceName $config)) {
            Write-Error "Cannot start $serviceName due to missing dependencies"
            continue
        }
        
        # Start service
        $success = Start-ServiceInTerminal $serviceName $config
        if (-not $success -and $config.Required) {
            Write-Error "Failed to start required service: $serviceName"
            return $false
        }
        
        # Wait between services
        if ($serviceName -ne ($sortedServices | Select-Object -Last 1).Key) {
            Start-Sleep -Seconds 2
        }
    }
    
    return $true
}

# Function to display service status
function Show-ServiceStatus {
    Write-Header "Service Status Summary"
    
    Write-Host ""
    Write-Host "🌐 Service URLs:" -ForegroundColor Cyan
    Write-Host "  • Enterprise FastAPI: http://localhost:8080" -ForegroundColor White
    Write-Host "  • API Documentation: http://localhost:8080/docs" -ForegroundColor White
    Write-Host "  • Gradio App: http://localhost:7860" -ForegroundColor White
    Write-Host "  • MLflow UI: http://localhost:5000" -ForegroundColor White
    Write-Host "  • ChromaDB: http://localhost:8081" -ForegroundColor White
    Write-Host "  • Ollama API: http://localhost:11434" -ForegroundColor White
    
    Write-Host ""
    Write-Host "🔧 Testing Commands:" -ForegroundColor Cyan
    Write-Host "  curl http://localhost:8080/health" -ForegroundColor White
    Write-Host "  curl http://localhost:7860" -ForegroundColor White
    Write-Host "  curl http://localhost:5000/health" -ForegroundColor White
    Write-Host "  curl http://localhost:8081/api/v1/heartbeat" -ForegroundColor White
    
    Write-Host ""
    Write-Host "📊 Active Terminals:" -ForegroundColor Cyan
    foreach ($service in $TerminalProcesses.GetEnumerator()) {
        $process = Get-Process -Id $service.Value.Id -ErrorAction SilentlyContinue
        if ($process) {
            Write-Host "  ✅ $($service.Key): PID $($service.Value.Id)" -ForegroundColor Green
        } else {
            Write-Host "  ❌ $($service.Key): Process not found" -ForegroundColor Red
        }
    }
}

# Function to create additional development terminals
function Start-DevelopmentTerminals {
    Write-Header "Starting Additional Development Terminals"
    
    # Terminal 6: Documentation Server
    $docsCommand = @"
cd '$ProjectRoot\docs'
& '$VenvPath'
Write-Host '📚 Starting MkDocs Documentation Server...' -ForegroundColor Green
Write-Host 'URL: http://localhost:8001' -ForegroundColor Cyan
Write-Host 'Press Ctrl+C to stop' -ForegroundColor Yellow
Write-Host ''
mkdocs serve --dev-addr 0.0.0.0:8001
"@
    
    $docsProcess = Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", $docsCommand -PassThru
    $TerminalProcesses["MkDocs"] = $docsProcess
    Write-Success "Started MkDocs server in terminal (PID: $($docsProcess.Id))"
    
    # Terminal 7: Test Runner
    $testCommand = @"
cd '$ProjectRoot'
& '$VenvPath'
Write-Host '🧪 Test Runner Terminal' -ForegroundColor Green
Write-Host 'Available commands:' -ForegroundColor Cyan
Write-Host '  python -m pytest tests\unit\ -v' -ForegroundColor White
Write-Host '  python -m pytest tests\integration\ -v' -ForegroundColor White
Write-Host '  python -m pytest tests\e2e\ -v' -ForegroundColor White
Write-Host '  python -m pytest tests\ -v --cov=src' -ForegroundColor White
Write-Host ''
"@
    
    $testProcess = Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", $testCommand -PassThru
    $TerminalProcesses["TestRunner"] = $testProcess
    Write-Success "Started Test Runner terminal (PID: $($testProcess.Id))"
}

# Function to cleanup on exit
function Stop-AllServices {
    Write-Header "Stopping All Services"
    
    foreach ($service in $TerminalProcesses.GetEnumerator()) {
        try {
            $process = Get-Process -Id $service.Value.Id -ErrorAction SilentlyContinue
            if ($process) {
                $process.Kill()
                Write-Success "Stopped $($service.Key) (PID: $($service.Value.Id))"
            }
        }
        catch {
            Write-Warning "Could not stop $($service.Key): $_"
        }
    }
    
    $TerminalProcesses.Clear()
}

# Main execution
try {
    Write-Header "AI Assignments - 5-Terminal Development Environment Setup"
    Write-Host "=" * 80
    
    # Change to project directory
    Set-Location $ProjectRoot
    
    # Install dependencies if not skipped
    if (-not $SkipDependencies) {
        Install-Dependencies
    }
    
    # Initialize directories
    Initialize-Directories
    
    # Start services if not skipped
    if (-not $SkipServices) {
        $success = Start-AllServices
        if (-not $success) {
            Write-Error "Failed to start all required services"
            exit 1
        }
    }
    
    # Start additional development terminals
    Start-DevelopmentTerminals
    
    # Show final status
    Show-ServiceStatus
    
    Write-Host ""
    Write-Success "Development environment setup complete!"
    Write-Info "All services are running in separate terminals."
    Write-Info "Use Ctrl+C in each terminal to stop individual services."
    
    # Keep script running to monitor services
    Write-Host ""
    Write-Info "Press 'q' to quit and stop all services, or 's' to show status again..."
    
    do {
        $key = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        if ($key.Character -eq 'q') {
            break
        } elseif ($key.Character -eq 's') {
            Show-ServiceStatus
        }
    } while ($true)
    
}
catch {
    Write-Error "Setup failed: $_"
    exit 1
}
finally {
    # Cleanup on exit
    Stop-AllServices
    Write-Info "Development environment stopped."
}
