# =============================================================================
# Get Authentication Token for Enterprise LLMOps Platform
# =============================================================================
# This script gets an authentication token for the FastAPI app
# Usage: .\scripts\get-auth-token.ps1
# =============================================================================

$apiUrl = "http://localhost:8080"

Write-Host "Getting authentication token for Enterprise LLMOps Platform..." -ForegroundColor Green

# Login credentials (from the code)
$loginData = @{
    username = "admin"
    password = "admin"
} | ConvertTo-Json

try {
    Write-Host "Attempting to login with admin credentials..." -ForegroundColor Yellow
    
    $response = Invoke-RestMethod -Uri "$apiUrl/api/auth/login" -Method Post -Body $loginData -ContentType "application/json"
    
    if ($response.access_token) {
        Write-Host "Login successful!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Your Bearer Token:" -ForegroundColor Cyan
        Write-Host $response.access_token -ForegroundColor White
        Write-Host ""
        Write-Host "How to use this token:" -ForegroundColor Cyan
        Write-Host "1. Go to http://localhost:8080/docs" -ForegroundColor White
        Write-Host "2. Click the 'Authorize' button" -ForegroundColor White
        Write-Host "3. Enter the token above in the 'Value' field" -ForegroundColor White
        Write-Host "4. Click 'Authorize'" -ForegroundColor White
        Write-Host ""
        Write-Host "Test the token:" -ForegroundColor Cyan
        Write-Host "curl -H 'Authorization: Bearer $($response.access_token)' http://localhost:8080/api/models" -ForegroundColor White
        Write-Host "curl -H 'Authorization: Bearer $($response.access_token)' http://localhost:8080/api/experiments" -ForegroundColor White
    }
    else {
        Write-Host "Login failed - no token received" -ForegroundColor Red
    }
}
catch {
    Write-Host "Login failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "Alternative: Use the simple app without authentication:" -ForegroundColor Yellow
    Write-Host "python -m src.enterprise_llmops.simple_app" -ForegroundColor White
}
