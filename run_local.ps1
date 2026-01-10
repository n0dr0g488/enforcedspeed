# EnforcedSpeed local runner (Windows PowerShell)
# Sets DATABASE_URL for local Docker Postgres only if not already set.

Write-Host "Starting EnforcedSpeed (local)..." -ForegroundColor Cyan

if (-not $env:DATABASE_URL -or $env:DATABASE_URL.Trim() -eq "") {
    $env:DATABASE_URL = "postgresql+psycopg://enforcedspeed:enforcedspeed_password@localhost:5433/enforcedspeed_local"
    Write-Host "DATABASE_URL set to local Postgres (localhost:5433)" -ForegroundColor Yellow
} else {
    Write-Host "DATABASE_URL already set (leaving as-is)" -ForegroundColor Green
}

# Optional: unblock files (safe)
try {
    Get-ChildItem -Recurse | Unblock-File -ErrorAction SilentlyContinue | Out-Null
} catch {}

# Ensure venv exists
if (!(Test-Path ".\venv")) {
    Write-Host "Virtual environment not found. Creating venv..." -ForegroundColor Yellow
    python -m venv venv
}

# Install/update dependencies into venv
Write-Host "Installing requirements..." -ForegroundColor Cyan
.\venv\Scripts\python.exe -m pip install -r requirements.txt

# Run app
Write-Host "Launching app..." -ForegroundColor Green
.\venv\Scripts\python.exe app.py
