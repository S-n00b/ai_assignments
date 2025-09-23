# Comprehensive Test Execution Script for Windows
# This script provides easy test execution for the GitHub Pages frontend,
# platform architecture layers, and Phase 7 demonstration flow.

param(
    [Parameter(Position = 0)]
    [ValidateSet("unit", "integration", "e2e", "phase7", "github-pages", "all")]
    [string]$TestType = "all",
    
    [switch]$Verbose,
    [switch]$Coverage,
    [switch]$Parallel,
    [switch]$LiveServices,
    [switch]$ProductionUrls,
    [switch]$SkipHealthCheck,
    [switch]$Help
)

# Help information
if ($Help) {
    Write-Host @"
Comprehensive Test Execution Script

Usage: .\scripts\run-tests.ps1 [TestType] [Options]

Test Types:
  unit          Run unit tests only
  integration   Run integration tests only
  e2e          Run end-to-end tests only
  phase7       Run Phase 7 demonstration tests only
  github-pages Run GitHub Pages frontend tests only
  all          Run complete test suite (default)

Options:
  -Verbose         Enable verbose output
  -Coverage        Generate coverage report
  -Parallel        Run tests in parallel
  -LiveServices    Use live services for testing
  -ProductionUrls  Use production URLs for testing
  -SkipHealthCheck Skip service health validation
  -Help           Show this help message

Examples:
  .\scripts\run-tests.ps1 unit -Verbose
  .\scripts\run-tests.ps1 all -Coverage
  .\scripts\run-tests.ps1 phase7 -LiveServices
  .\scripts\run-tests.ps1 github-pages -ProductionUrls

"@
    exit 0
}

# Set error action preference
$ErrorActionPreference = "Stop"

# Get script directory and project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Write-Host "🎯 Comprehensive Test Execution Script" -ForegroundColor Cyan
Write-Host "Project Root: $ProjectRoot" -ForegroundColor Gray
Write-Host "Test Type: $TestType" -ForegroundColor Gray
Write-Host ""

# Function to check service health
function Test-ServiceHealth {
    param(
        [string]$ServiceName,
        [int]$Port,
        [string]$HealthPath = "/"
    )
    
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$Port$HealthPath" -TimeoutSec 5 -UseBasicParsing
        return $response.StatusCode -eq 200
    }
    catch {
        return $false
    }
}

# Function to run PowerShell command
function Invoke-TestCommand {
    param(
        [string]$Command,
        [string]$WorkingDirectory = $ProjectRoot
    )
    
    Write-Host "🚀 Running: $Command" -ForegroundColor Yellow
    try {
        $result = Invoke-Expression $Command
        return $true
    }
    catch {
        Write-Host "❌ Command failed: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to validate services
function Test-Services {
    Write-Host "🔍 Checking service health..." -ForegroundColor Cyan
    
    $services = @{
        "FastAPI"  = @{ Port = 8080; Path = "/health" }
        "Gradio"   = @{ Port = 7860; Path = "/" }
        "MLflow"   = @{ Port = 5000; Path = "/" }
        "ChromaDB" = @{ Port = 8000; Path = "/" }
    }
    
    $allHealthy = $true
    
    foreach ($service in $services.GetEnumerator()) {
        $isHealthy = Test-ServiceHealth -ServiceName $service.Key -Port $service.Value.Port -HealthPath $service.Value.Path
        $statusIcon = if ($isHealthy) { "✅" } else { "❌" }
        $statusText = if ($isHealthy) { "Healthy" } else { "Not Running" }
        
        Write-Host "  $statusIcon $($service.Key): $statusText" -ForegroundColor $(if ($isHealthy) { "Green" } else { "Red" })
        
        if (-not $isHealthy) {
            $allHealthy = $false
        }
    }
    
    return $allHealthy
}

# Function to run unit tests
function Test-UnitTests {
    Write-Host "`n📋 Running Unit Tests..." -ForegroundColor Cyan
    
    $command = "python -m pytest tests/unit/"
    if ($Verbose) { $command += " -v" }
    if ($Coverage) { $command += " --cov=src --cov-report=html --cov-report=term" }
    
    return Invoke-TestCommand -Command $command
}

# Function to run integration tests
function Test-IntegrationTests {
    Write-Host "`n🔗 Running Integration Tests..." -ForegroundColor Cyan
    
    $command = "python -m pytest tests/integration/"
    if ($Verbose) { $command += " -v" }
    if ($LiveServices) { $command += " --live-services" }
    
    return Invoke-TestCommand -Command $command
}

# Function to run E2E tests
function Test-E2ETests {
    Write-Host "`n🌐 Running End-to-End Tests..." -ForegroundColor Cyan
    
    $command = "python -m pytest tests/e2e/"
    if ($Verbose) { $command += " -v" }
    if ($ProductionUrls) { $command += " --production-urls" }
    
    return Invoke-TestCommand -Command $command
}

# Function to run Phase 7 tests
function Test-Phase7Tests {
    Write-Host "`n🎯 Running Phase 7 Demonstration Tests..." -ForegroundColor Cyan
    
    $command = "python -m pytest tests/e2e/test_phase7_complete_demonstration.py"
    if ($Verbose) { $command += " -v" }
    
    return Invoke-TestCommand -Command $command
}

# Function to run GitHub Pages tests
function Test-GitHubPagesTests {
    Write-Host "`n📄 Running GitHub Pages Tests..." -ForegroundColor Cyan
    
    if ($ProductionUrls) {
        $command = "python -m pytest tests/e2e/test_github_pages_frontend_integration.py --production-urls"
    }
    else {
        $command = "python -m pytest tests/unit/test_github_pages_integration.py"
    }
    
    if ($Verbose) { $command += " -v" }
    
    return Invoke-TestCommand -Command $command
}

# Function to generate report
function Show-TestReport {
    param([hashtable]$Results)
    
    Write-Host "`n📊 Test Execution Report" -ForegroundColor Cyan
    Write-Host "=" * 50 -ForegroundColor Gray
    
    $totalTests = $Results.Count
    $passedTests = ($Results.Values | Where-Object { $_ -eq $true }).Count
    
    foreach ($testCategory in $Results.GetEnumerator()) {
        $statusIcon = if ($testCategory.Value) { "✅" } else { "❌" }
        $statusText = if ($testCategory.Value) { "PASSED" } else { "FAILED" }
        $categoryName = $testCategory.Key -replace '_', ' '
        
        Write-Host "$statusIcon $categoryName`: $statusText" -ForegroundColor $(if ($testCategory.Value) { "Green" } else { "Red" })
    }
    
    Write-Host "`nOverall: $passedTests/$totalTests test categories passed" -ForegroundColor Gray
    
    if ($passedTests -eq $totalTests) {
        Write-Host "🎉 All tests passed successfully!" -ForegroundColor Green
    }
    else {
        Write-Host "⚠️  Some tests failed. Check the output above for details." -ForegroundColor Yellow
    }
}

# Main execution
try {
    # Change to project directory
    Set-Location $ProjectRoot
    
    # Activate virtual environment
    $venvScript = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"
    if (Test-Path $venvScript) {
        Write-Host "✅ Virtual environment found" -ForegroundColor Green
        & $venvScript
    }
    else {
        Write-Host "❌ Virtual environment not found. Please create it first." -ForegroundColor Red
        exit 1
    }
    
    # Validate services unless skipped
    if (-not $SkipHealthCheck) {
        $servicesHealthy = Test-Services
        if (-not $servicesHealthy) {
            Write-Host "`n⚠️  Some services are not running. Some tests may fail." -ForegroundColor Yellow
            $response = Read-Host "Continue anyway? (y/N)"
            if ($response -ne "y" -and $response -ne "Y") {
                exit 1
            }
        }
    }
    
    # Run tests based on selection
    $results = @{}
    
    switch ($TestType) {
        "unit" {
            $results["unit"] = Test-UnitTests
        }
        "integration" {
            $results["integration"] = Test-IntegrationTests
        }
        "e2e" {
            $results["e2e"] = Test-E2ETests
        }
        "phase7" {
            $results["phase7"] = Test-Phase7Tests
        }
        "github-pages" {
            $results["github_pages"] = Test-GitHubPagesTests
        }
        "all" {
            Write-Host "`n🎯 Running Comprehensive Test Suite..." -ForegroundColor Cyan
            $results["unit"] = Test-UnitTests
            $results["integration"] = Test-IntegrationTests
            $results["e2e"] = Test-E2ETests
            $results["phase7"] = Test-Phase7Tests
            $results["github_pages"] = Test-GitHubPagesTests
        }
    }
    
    # Generate report
    Show-TestReport -Results $results
    
    # Exit with appropriate code
    $allPassed = $results.Values | ForEach-Object { $_ } | Where-Object { $_ -eq $true } | Measure-Object | Select-Object -ExpandProperty Count
    $totalCount = $results.Count
    
    if ($allPassed -eq $totalCount) {
        exit 0
    }
    else {
        exit 1
    }
}
catch {
    Write-Host "❌ An error occurred: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
finally {
    # Return to original directory
    Set-Location $PSScriptRoot
}
