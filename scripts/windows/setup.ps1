<#
.SYNOPSIS
  One-time setup on native Windows: Python environment, NVIDIA libraries, models, database tables.

.DESCRIPTION
  Needs: Python 3.11 (python.org installer, "py" launcher), an NVIDIA driver (GPU), PostgreSQL 16
  and a Redis-compatible server on localhost (see README "Run on Windows").
  Run from any folder:   powershell -ExecutionPolicy Bypass -File scripts\windows\setup.ps1
  Options:  -Config <yaml>   profile to prepare (default config\prod-gpu.yaml)
            -CpuOnly         skip the NVIDIA libraries (CPU decoding)
            -CreateDb        create the stt user and database with psql (asks for the postgres password)
#>
param(
  [string]$Config = "config\prod-gpu.yaml",
  [switch]$CpuOnly,
  [switch]$CreateDb
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$env:PYTHONPATH = Join-Path $Root "src"
$env:UV_PROJECT_ENVIRONMENT = ".venv"

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Test-Port([int]$Port) {
  $c = New-Object System.Net.Sockets.TcpClient
  try { $c.Connect("127.0.0.1", $Port); return $true } catch { return $false } finally { $c.Close() }
}
function Invoke-Checked {  # run a native command and stop on a non-zero exit code
  param([string]$Exe, [string[]]$Arguments)
  & $Exe @Arguments
  if ($LASTEXITCODE -ne 0) { throw "failed ($LASTEXITCODE): $Exe $($Arguments -join ' ')" }
}

Step "Python 3.11 environment (.venv)"
if (-not (Test-Path $Py)) {
  if (Get-Command py -ErrorAction SilentlyContinue) { Invoke-Checked py @("-3.11", "-m", "venv", ".venv") }
  else { Invoke-Checked python @("-m", "venv", ".venv") }
}
$ver = & $Py -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ($ver -ne "3.11") { throw ".venv uses Python $ver; this project pins Python 3.11 (install it from python.org)" }
Invoke-Checked $Py @("-m", "pip", "install", "-q", "--upgrade", "pip", "uv")

Step "Locked dependencies (uv.lock)"
Invoke-Checked (Join-Path $Root ".venv\Scripts\uv.exe") @("sync", "--frozen", "--no-editable", "--extra", "bench", "--python", $Py)

if (-not $CpuOnly) {
  Step "NVIDIA cuBLAS 12 + cuDNN 9 (pip, pinned)"
  Invoke-Checked $Py @("-m", "pip", "install", "-q", "-r", "scripts\windows\requirements-cuda.txt")
  $gpus = & $Py -c "from live_stt.winsupport import add_cuda_dll_dirs; add_cuda_dll_dirs(); import ctranslate2; print(ctranslate2.get_cuda_device_count())"
  if ([int]$gpus -lt 1) { Write-Warning "CTranslate2 sees no CUDA GPU: update the NVIDIA driver, or use -CpuOnly" }
  else { Write-Host "CUDA GPUs visible to CTranslate2: $gpus" -ForegroundColor Green }
}

Step "Models (pinned revisions, SHA-256 verified)"
Invoke-Checked $Py @("-m", "live_stt.cli", "download-models", "-c", $Config)
Invoke-Checked $Py @("-m", "live_stt.cli", "download-models", "-c", "config\default.yaml")

if ($CreateDb) {
  Step "PostgreSQL user and database"
  $psql = Get-Command psql -ErrorAction SilentlyContinue
  if (-not $psql) { $psql = Get-ChildItem "$env:ProgramFiles\PostgreSQL\*\bin\psql.exe" -ErrorAction SilentlyContinue | Select-Object -Last 1 }
  if (-not $psql) { throw "psql not found: install PostgreSQL 16 (EDB installer) or create the database by hand" }
  $exe = if ($psql.Source) { $psql.Source } else { $psql.FullName }
  & $exe -U postgres -h localhost -c "CREATE USER stt WITH PASSWORD 'stt'" 2>$null
  & $exe -U postgres -h localhost -c "CREATE DATABASE stt OWNER stt" 2>$null
}

Step "Database tables"
if (-not (Test-Port 5432)) { throw "PostgreSQL is not answering on localhost:5432 (start the postgresql-x64-16 service)" }
if (-not (Test-Port 6379)) { Write-Warning "Redis is not answering on localhost:6379: start Memurai / Redis before scripts\windows\live.ps1 start" }
Invoke-Checked $Py @("-m", "live_stt.cli", "migrate", "-c", $Config)

Write-Host "`nSetup complete. Next:" -ForegroundColor Green
Write-Host "  scripts\windows\live.ps1 start"
Write-Host "  scripts\windows\stream.ps1 udp://@239.1.1.1:5000"
