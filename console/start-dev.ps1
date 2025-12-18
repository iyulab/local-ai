# LMSupply Console - Development Server Startup Script
# Runs backend and frontend in separate Windows Terminal tabs with hot-reload

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$hostPath = Join-Path $scriptPath "host"
$uiPath = Join-Path $scriptPath "ui"

Write-Host "Starting LMSupply Console Development Servers..." -ForegroundColor Cyan
Write-Host ""
Write-Host "  Backend:  http://localhost:5000" -ForegroundColor Green
Write-Host "  Frontend: http://localhost:3000" -ForegroundColor Green
Write-Host ""

# Start Windows Terminal with two tabs using cmd
wt -w 0 `
    new-tab --title "Backend" -d $hostPath cmd /k "dotnet watch run" `; `
    new-tab --title "Frontend" -d $uiPath cmd /k "npm run dev"
