# EnforcedSpeed RQ worker runner (Windows PowerShell)
# - Pins DATABASE_URL + REDIS_URL for local Docker Postgres/Redis (unless already set)
# - Clears stale R2_* env vars so .env is authoritative for R2 locally
# - Creates venv if missing, installs deps, launches worker
#
# v184 hardening:
# - Never trust requirements hash if key imports are missing (redis/rq)
# - Treat pip failures as real failures (check exit codes)
# - Simple install lock to avoid parallel pip installs when es-up starts app + worker together

Write-Host "Starting EnforcedSpeed RQ worker (local)" -ForegroundColor Cyan

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

# ---- Ensure venv exists + is healthy ----
$venvDir = Join-Path $PSScriptRoot "venv"
$venvPy  = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
$venvPip = Join-Path $PSScriptRoot "venv\Scripts\pip.exe"
$venvHealthy = (Test-Path $venvPy) -and (Test-Path $venvPip)

$forceInstall = $false
$lockFile = Join-Path $PSScriptRoot ".install.lock"
function Acquire-InstallLock() {
    # Clear stale lock (e.g., crashed install)
    if (Test-Path $lockFile) {
        try {
            $age = (Get-Date) - (Get-Item $lockFile).LastWriteTime
            if ($age.TotalMinutes -gt 10) {
                Remove-Item -Force $lockFile -ErrorAction SilentlyContinue
            }
        } catch {}
    }
    while (Test-Path $lockFile) { Start-Sleep -Seconds 1 }
    New-Item -ItemType File -Path $lockFile -Force | Out-Null
}
function Release-InstallLock() {
    Remove-Item -Force $lockFile -ErrorAction SilentlyContinue
}


if (-not $venvHealthy) {
    Acquire-InstallLock
    try {
        # Re-check venv health after acquiring lock (es-up can start app + worker simultaneously)
        $venvHealthy = (Test-Path $venvPy) -and (Test-Path $venvPip)
        if (-not $venvHealthy) {
        if (Test-Path $venvDir) {
            Write-Host "Found broken venv folder (missing pip/python). Removing and recreating..." -ForegroundColor Yellow
            try {
                Remove-Item -Recurse -Force $venvDir -ErrorAction Stop
            } catch {
                Write-Host "Could not remove venv. Close any other EnforcedSpeed app/worker windows and try again." -ForegroundColor Red
                throw
            }
        }

        Write-Host "Creating venv..." -ForegroundColor Yellow
        try {
            python -m venv $venvDir
        } catch {
            Write-Host "venv creation failed. Ensure 'python' is available on PATH, then re-run." -ForegroundColor Red
            throw
        }

        $venvHealthy = (Test-Path $venvPy) -and (Test-Path $venvPip)
        $forceInstall = $true
        }
    } finally {
        Release-InstallLock
    }
}
function Test-PyImport([string]$module) {
    & $venvPy -c "import $module" 2>$null
    return ($LASTEXITCODE -eq 0)
}

# ---- Install/update dependencies ----
$reqFile = Join-Path $PSScriptRoot "requirements.txt"
$hashFile = Join-Path $PSScriptRoot "venv\.requirements.sha256"
$currentHash = (Try-GetFileHash $reqFile)
$previousHash = ""
if (Test-Path $hashFile) { $previousHash = (Get-Content $hashFile -ErrorAction SilentlyContinue) }

# Decide if we need to install deps. Do NOT rely solely on the hash.
$needInstall = $false
if ($forceInstall) { $needInstall = $true }

# If we cannot compute a hash, assume we need to install.
if (-not $currentHash -or $currentHash.Trim() -eq "") { $needInstall = $true }
elseif ($currentHash -ne $previousHash) { $needInstall = $true }

# If key imports are missing, force reinstall even if hash matches
if ($venvHealthy) {
    $missing = @()
    foreach ($m in @("redis","rq")) {
        if (-not (Test-PyImport $m)) { $missing += $m }
    }
    if ($missing.Count -gt 0) {
        Write-Host ("Detected missing python packages in venv: " + ($missing -join ", ") + " - forcing pip install...") -ForegroundColor Yellow
        $needInstall = $true
    }
}

if ($needInstall) {
    Write-Host "Installing requirements..." -ForegroundColor Cyan
    Acquire-InstallLock
    try {
        & $venvPy -m ensurepip --upgrade | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "ensurepip failed (exit code $LASTEXITCODE)" }

        & $venvPy -m pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) { throw "pip install failed (exit code $LASTEXITCODE)" }

        if ($currentHash -and $currentHash.Trim() -ne "") {
            try {
                if (Test-Path $hashFile) {
                    attrib -R $hashFile 2>$null
                    Remove-Item -Force $hashFile -ErrorAction SilentlyContinue
                }
                [System.IO.File]::WriteAllText($hashFile, $currentHash)
            } catch {
                Write-Host "Warning: Could not write requirements hash file ($hashFile). Continuing..." -ForegroundColor Yellow
            }
        }
    } finally {
        Release-InstallLock
    }
} else {
    Write-Host "Requirements unchanged. Skipping pip install." -ForegroundColor DarkGray
}

# ---- Launch worker ----# ---- Launch worker ----
Write-Host "Launching worker..." -ForegroundColor Green
& $venvPy worker.py
