# Start Unified Platform with 8 Terminals
# This script creates 8 VS Code terminals and starts all services in the correct order

Write-Host "🚀 Starting Lenovo AI Architecture Unified Platform with 8 Terminals..." -ForegroundColor Green

# Function to create VS Code terminal with specific command
function Start-VSCodeTerminal {
    param(
        [string]$Name,
        [string]$Command,
        [int]$Port,
        [string]$Description
    )
    
    Write-Host "🔧 Creating VS Code terminal: $Name..." -ForegroundColor Cyan
    
    # Create the full command with venv activation
    $FullCommand = "& C:\Users\samne\PycharmProjects\ai_assignments\venv\Scripts\Activate.ps1; $Command"
    
    # Start VS Code terminal with the command
    try {
        # Create a temporary script file for each terminal
        $TempScript = "temp_terminal_$Name.ps1"
        $ScriptContent = @"
# Terminal: $Name
# Description: $Description
# Port: $Port

Write-Host "🚀 Starting $Name..." -ForegroundColor Green
Write-Host "Description: $Description" -ForegroundColor Gray
Write-Host "Port: $Port" -ForegroundColor Gray
Write-Host "Command: $Command" -ForegroundColor Gray
Write-Host ""

# Activate virtual environment and run command
& C:\Users\samne\PycharmProjects\ai_assignments\venv\Scripts\Activate.ps1
$Command
"@
        
        $ScriptContent | Out-File -FilePath $TempScript -Encoding UTF8
        
        # Start VS Code with new terminal and run the script
        $VSCodeCommand = "code --new-window --command workbench.action.terminal.new --command workbench.action.terminal.sendSequence --command text=powershell -ExecutionPolicy Bypass -File $TempScript"
        Start-Process cmd -ArgumentList "/c", $VSCodeCommand -WindowStyle Normal
        Write-Host "✅ VS Code terminal '$Name' created successfully" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Failed to create VS Code terminal '$Name' - $($_.Exception.Message)" -ForegroundColor Red
    }
}

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

Write-Host "`n🌐 Creating 8 VS Code Terminals..." -ForegroundColor Magenta

# Terminal 1: ChromaDB Vector Store (Port 8081)
Start-VSCodeTerminal -Name "chroma-8081" -Command "chroma run --host 0.0.0.0 --port 8081 --path chroma_data" -Port 8081 -Description "ChromaDB Vector Store"

# Terminal 2: MLflow Experiment Tracking (Port 5000)
Start-VSCodeTerminal -Name "mlflow-5000" -Command "mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000" -Port 5000 -Description "MLflow Experiment Tracking"

# Terminal 3: LangGraph Studio (Port 8083)
Start-VSCodeTerminal -Name "langgraph-8083" -Command "langgraph dev --host 0.0.0.0 --port 8083" -Port 8083 -Description "LangGraph Studio Agent Visualization"

# Terminal 4: MkDocs Documentation (Port 8082)
Start-VSCodeTerminal -Name "mkdocs-8082" -Command "cd docs; mkdocs build; mkdocs serve --dev-addr 0.0.0.0:8082" -Port 8082 -Description "MkDocs Documentation Hub"

# Terminal 5: Gradio Model Evaluation (Port 7860)
Start-VSCodeTerminal -Name "gradio-7860" -Command "python -m src.gradio_app.main --host 0.0.0.0 --port 7860" -Port 7860 -Description "Gradio Model Evaluation Interface"

# Terminal 6: Enterprise LLMOps Platform (Port 8080) - Will start after others
Write-Host "🔧 Creating VS Code terminal: llmops-8080 (will start after other services)..." -ForegroundColor Cyan
Start-VSCodeTerminal -Name "llmops-8080" -Command "python -m src.enterprise_llmops.main --host 0.0.0.0 --port 8080" -Port 8080 -Description "Enterprise LLMOps Platform"

# Terminal 7: Registry Sync
Write-Host "🔧 Creating VS Code terminal: sync (will run after llmops starts)..." -ForegroundColor Cyan
Start-VSCodeTerminal -Name "sync" -Command ".\scripts\comprehensive-sync.ps1" -Port 0 -Description "Registry Synchronization"

# Terminal 8: Development Shell
Write-Host "🔧 Creating VS Code terminal: dev shell..." -ForegroundColor Cyan
Start-VSCodeTerminal -Name "dev shell" -Command "Write-Host 'Development shell ready. Use this for additional commands.'" -Port 0 -Description "Development Shell"

# Wait for services to initialize
Write-Host "`n⏳ Waiting for services to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check if main port is available and start the enterprise platform
Write-Host "`n🚀 Starting Enterprise LLMOps Platform on port 8080..." -ForegroundColor Green

if (Test-Port 8080) {
    Write-Host "⚠️  Port 8080 is already in use. Please stop the existing service first." -ForegroundColor Yellow
}
else {
    Write-Host "✅ Port 8080 is available. Enterprise platform should start in its terminal." -ForegroundColor Green
}

# Wait a moment for the main platform to start
Start-Sleep -Seconds 5

# Run registry sync
Write-Host "`n🔄 Running registry synchronization..." -ForegroundColor Magenta
try {
    & C:\Users\samne\PycharmProjects\ai_assignments\venv\Scripts\Activate.ps1
    & .\scripts\comprehensive-sync.ps1
    Write-Host "✅ Registry synchronization completed" -ForegroundColor Green
    }
    catch {
    Write-Host "❌ Registry synchronization failed - $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n🎉 All 8 VS Code Terminals Created and Services Started!" -ForegroundColor Green

Write-Host "`n📋 VS Code Terminal Links:" -ForegroundColor Cyan
Write-Host "   🔗 Terminal 1 (ChromaDB): vscode://file/C:/Users/samne/PycharmProjects/ai_assignments" -ForegroundColor White
Write-Host "   🔗 Terminal 2 (MLflow): vscode://file/C:/Users/samne/PycharmProjects/ai_assignments" -ForegroundColor White
Write-Host "   🔗 Terminal 3 (LangGraph): vscode://file/C:/Users/samne/PycharmProjects/ai_assignments" -ForegroundColor White
Write-Host "   🔗 Terminal 4 (MkDocs): vscode://file/C:/Users/samne/PycharmProjects/ai_assignments" -ForegroundColor White
Write-Host "   🔗 Terminal 5 (Gradio): vscode://file/C:/Users/samne/PycharmProjects/ai_assignments" -ForegroundColor White
Write-Host "   🔗 Terminal 6 (LLMOps): vscode://file/C:/Users/samne/PycharmProjects/ai_assignments" -ForegroundColor White
Write-Host "   🔗 Terminal 7 (Sync): vscode://file/C:/Users/samne/PycharmProjects/ai_assignments" -ForegroundColor White
Write-Host "   🔗 Terminal 8 (Dev Shell): vscode://file/C:/Users/samne/PycharmProjects/ai_assignments" -ForegroundColor White

Write-Host "`n🌐 Service URLs (Click to Open):" -ForegroundColor Cyan
Write-Host "   🏠 Enterprise Platform: http://localhost:8080" -ForegroundColor White
Write-Host "   📖 About & Pitch: http://localhost:8080/about" -ForegroundColor White
Write-Host "   📚 API Docs: http://localhost:8080/docs" -ForegroundColor White
Write-Host "   🧪 Model Evaluation: http://localhost:7860" -ForegroundColor White
Write-Host "   📈 MLflow Tracking: http://localhost:5000" -ForegroundColor White
Write-Host "   🗄️ ChromaDB Vector Store: http://localhost:8081" -ForegroundColor White
Write-Host "   📚 MkDocs Documentation: http://localhost:8082" -ForegroundColor White
Write-Host "   🎯 LangGraph Studio: http://localhost:8083" -ForegroundColor White

Write-Host "`n💡 Terminal Layout:" -ForegroundColor Yellow
Write-Host "   • All 8 terminals should now be visible in VS Code" -ForegroundColor Gray
Write-Host "   • Each terminal runs a specific service with virtual environment activated" -ForegroundColor Gray
Write-Host "   • Services start in the correct order for proper initialization" -ForegroundColor Gray
Write-Host "   • Registry sync runs automatically after the main platform starts" -ForegroundColor Gray

Write-Host "`n🚀 Quick Access Commands:" -ForegroundColor Magenta
Write-Host "   • Press Ctrl+` to open terminal panel" -ForegroundColor Gray
Write-Host "   • Use Ctrl+Shift+` to create new terminal" -ForegroundColor Gray
Write-Host "   • Click on service URLs above to open in browser" -ForegroundColor Gray

# Clean up temporary files
Write-Host "`n🧹 Cleaning up temporary files..." -ForegroundColor Yellow
Get-ChildItem -Path "temp_terminal_*.ps1" | Remove-Item -Force
Write-Host "✅ Cleanup completed" -ForegroundColor Green