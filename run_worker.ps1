# EnforcedSpeed RQ worker runner (Windows PowerShell)
# - Pins DATABASE_URL + REDIS_URL for local Docker Postgres/Redis (unless already set)
# - Clears stale R2_* env vars so .env is authoritative for R2 locally
# - Creates venv if missing, installs deps, launches worker

Write-Host "Starting EnforcedSpeed RQ worker (local)..." -ForegroundColor Cyan

# Ensure we are running from the folder that contains this script (ZIP root)
Set-Location -Path $PSScriptRoot

function Try-GetFileHash($path) {
  if (Get-Command Get-FileHash -ErrorAction SilentlyContinue) {
    return (Get-FileHash $path -Algorithm SHA256).Hash
  }
  return ""
}


# ---- Local DB + Redis (pin for local dev unless already set) ----
if (-not $env:DATABASE_URL -or $env:DATABASE_URL.Trim() -eq "") {
    $env:DATABASE_URL = "postgresql+pg8000://enforcedspeed:enforcedspeed_password@localhost:5433/enforcedspeed_local"
    Write-Host "DATABASE_URL set to local Postgres (localhost:5433)" -ForegroundColor Yellow
} else {
    Write-Host "DATABASE_URL already set (leaving as-is)" -ForegroundColor Green
}

if (-not $env:REDIS_URL -or $env:REDIS_URL.Trim() -eq "") {
    $env:REDIS_URL = "redis://localhost:6379/0"
    Write-Host "REDIS_URL set to local Redis (localhost:6379)" -ForegroundColor Yellow
} else {
    Write-Host "REDIS_URL already set (leaving as-is)" -ForegroundColor Green
}

# ---- IMPORTANT: Clear stale R2 vars so .env controls R2 config ----
$stale = @(
  "R2_ENDPOINT",
  "R2_ACCESS_KEY_ID",
  "R2_SECRET_ACCESS_KEY",
  "R2_QUARANTINE_BUCKET",
  "R2_PREFIX"
)
foreach ($k in $stale) {
    Remove-Item "Env:$k" -ErrorAction SilentlyContinue
}
Write-Host "Cleared any stale R2_* env vars (so .env is authoritative for R2)." -ForegroundColor DarkGray

# Optional: unblock files (safe)
try {
    Get-ChildItem -Recurse | Unblock-File -ErrorAction SilentlyContinue | Out-Null
} catch {}

# ---- Ensure venv exists ----
if (!(Test-Path ".\\venv")) {
    Write-Host "Virtual environment not found. Creating venv..." -ForegroundColor Yellow
    python -m venv venv
}

# If the venv exists but pip/python inside it are missing or broken (common after WinError 32
# interruptions), force a reinstall on this run.
$venvPy  = Join-Path $PSScriptRoot "venv\\Scripts\\python.exe"
$venvPip = Join-Path $PSScriptRoot "venv\\Scripts\\pip.exe"
$venvHealthy = (Test-Path $venvPy) -and (Test-Path $venvPip)

# ---- Install/update dependencies (only when requirements.txt changes) ----
$reqFile = Join-Path $PSScriptRoot "requirements.txt"
$hashFile = Join-Path $PSScriptRoot "venv\\.requirements.sha256"
$currentHash = (Try-GetFileHash $reqFile)
$previousHash = ""
if (Test-Path $hashFile) { $previousHash = (Get-Content $hashFile -ErrorAction SilentlyContinue) }

if (-not $venvHealthy) {
    Write-Host "venv exists but pip/python are missing. Forcing dependency install..." -ForegroundColor Yellow
    $previousHash = ""
}

if ($currentHash -ne $previousHash) {
    Write-Host "Installing requirements (requirements.txt changed)..." -ForegroundColor Cyan
    try {
        .\\venv\\Scripts\\python.exe -m ensurepip --upgrade | Out-Null
        .\\venv\\Scripts\\python.exe -m pip install -r requirements.txt
        # Write requirements hash (best-effort; do not fail worker on Windows file/AV quirks)
        try {
            if (Test-Path $hashFile) {
                attrib -R $hashFile 2>$null
                Remove-Item -Force $hashFile -ErrorAction SilentlyContinue
            }
            [System.IO.File]::WriteAllText($hashFile, $currentHash)
        } catch {
            Write-Host "Warning: Could not write requirements hash file ($hashFile). Continuing..." -ForegroundColor Yellow
        }
    } catch {
        Write-Host "Dependency install failed. If you see WinError 32, make sure no other EnforcedSpeed app/worker window is still running and then re-run es-up." -ForegroundColor Red
        throw
    }
} else {
    Write-Host "Requirements unchanged. Skipping pip install." -ForegroundColor DarkGray
}

# ---- Launch worker ----
Write-Host "Launching worker..." -ForegroundColor Green
.\\venv\\Scripts\\python.exe worker.py
