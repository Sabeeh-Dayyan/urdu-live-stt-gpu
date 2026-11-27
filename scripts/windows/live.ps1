<#
.SYNOPSIS
  Run the whole system in the background on native Windows (Windows equivalent of scripts/live.sh).

.EXAMPLE
  scripts\windows\live.ps1 start                 # default profile config\prod-gpu.yaml
  scripts\windows\live.ps1 start config\my.yaml
  scripts\windows\live.ps1 status | stop | restart | logs

  State: run\live.pid, run\live.config   Logs: var\log\live.log (+ live.err.log)
  PostgreSQL (5432) and Redis/Memurai (6379) must already be running as Windows services.
#>
param(
  [Parameter(Position = 0)][ValidateSet("start", "stop", "restart", "status", "logs")][string]$Action = "status",
  [Parameter(Position = 1)][string]$Config = ""
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$PidFile = Join-Path $Root "run\live.pid"
$CfgFile = Join-Path $Root "run\live.config"
$Log = Join-Path $Root "var\log\live.log"
$ErrLog = Join-Path $Root "var\log\live.err.log"
$Api = "http://127.0.0.1:8000"   # not "localhost": Windows tries IPv6 first and a refused ::1 costs ~2 s
New-Item -ItemType Directory -Force (Join-Path $Root "run"), (Join-Path $Root "var\log") | Out-Null
$env:PYTHONPATH = Join-Path $Root "src"

function Get-LivePid {
  if (-not (Test-Path $PidFile)) { return $null }
  $id = [int](Get-Content $PidFile -Raw)
  if (Get-Process -Id $id -ErrorAction SilentlyContinue) { return $id }
  return $null
}
function Test-Port([int]$Port) {
  $c = New-Object System.Net.Sockets.TcpClient
  try { $c.Connect("127.0.0.1", $Port); return $true } catch { return $false } finally { $c.Close() }
}
function Get-Health {
  # /health answers 503 with the same JSON when Redis or the database is unhealthy; show that too.
  try { return Invoke-RestMethod -Uri "$Api/health" -TimeoutSec 10 }
  catch {
    if ($_.ErrorDetails -and $_.ErrorDetails.Message) {
      try { return ($_.ErrorDetails.Message | ConvertFrom-Json) } catch { }
    }
    return $null
  }
}

function Show-Status {
  $id = Get-LivePid
  if (-not $id) { Write-Host "not running"; return }
  Write-Host "running: pid $id, config $(Get-Content $CfgFile -Raw -ErrorAction SilentlyContinue)".Trim()
  $h = Get-Health
  if (-not $h) { Write-Host "  api: not answering yet"; return }
  Write-Host ("  status {0}  redis {1}  db {2}" -f $h.status, $h.redis, $h.db)
  $kids = $h.components.supervisor.children
  if ($kids) {
    foreach ($p in ($kids.PSObject.Properties | Sort-Object Name)) {
      Write-Host ("  {0,-14} alive={1} restarts={2}" -f $p.Name, $p.Value.alive, $p.Value.restarts)
    }
  }
  if ($h.status -ne "ok") { Write-Host "  DEGRADED: $($h.redis_error) (check the PostgreSQL / Redis services)" -ForegroundColor Yellow }
  Write-Host "  UI       http://localhost:8000"
  Write-Host "  Grafana  http://localhost:3000/d/live-stt  (if installed)"
  Write-Host "  logs     $Log"
}

function Start-Live([string]$Cfg) {
  if (-not $Cfg) { $Cfg = "config\prod-gpu.yaml" }
  if (Get-LivePid) { Write-Host "already running"; Show-Status; return }
  if (-not (Test-Path $Py)) { throw "no .venv: run scripts\windows\setup.ps1 first" }
  if (-not (Test-Path $Cfg)) { throw "no such config: $Cfg" }
  if (-not (Test-Port 5432)) { throw "PostgreSQL is not answering on localhost:5432 (services.msc -> postgresql-x64-16)" }
  if (-not (Test-Port 6379)) { throw "Redis is not answering on localhost:6379 (start Memurai, or Redis in WSL/Docker)" }
  & $Py -m live_stt.cli show-config -c $Cfg | Out-Null
  if ($LASTEXITCODE -ne 0) { throw "invalid config: $Cfg" }
  foreach ($f in @($Log, $ErrLog)) { if (Test-Path $f) { Move-Item -Force $f "$f.1" } }
  $p = Start-Process -FilePath $Py -ArgumentList @("-m", "live_stt.cli", "supervise", "-c", $Cfg) `
        -WorkingDirectory $Root -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $Log -RedirectStandardError $ErrLog
  Set-Content -Path $PidFile -Value $p.Id -NoNewline
  Set-Content -Path $CfgFile -Value $Cfg -NoNewline
  for ($i = 0; $i -lt 60; $i++) {
    if (Get-Health) { break }
    if ($p.HasExited) {
      Write-Host "supervisor exited - last log lines:" -ForegroundColor Red
      Get-Content $Log, $ErrLog -Tail 20 -ErrorAction SilentlyContinue
      exit 1
    }
    Start-Sleep -Seconds 1
  }
  Show-Status
}

function Stop-Live {
  $id = Get-LivePid
  if (-not $id) { Write-Host "not running"; Remove-Item $PidFile -ErrorAction SilentlyContinue; return }
  # The supervisor starts its workers with multiprocessing (spawn): stop the whole process tree.
  # Windows has no SIGTERM, so this is a hard stop; unfinished ASR work stays in Redis (consumer
  # groups) and is redelivered on the next start.
  & taskkill.exe /PID $id /T /F | Out-Null
  Remove-Item $PidFile -ErrorAction SilentlyContinue
  Write-Host "stopped"
}

switch ($Action) {
  "start"   { Start-Live $Config }
  "stop"    { Stop-Live }
  "restart" {
    $prev = Get-Content $CfgFile -Raw -ErrorAction SilentlyContinue
    Stop-Live; Start-Sleep -Seconds 2
    if ($Config) { Start-Live $Config } else { Start-Live $prev }
  }
  "status"  { Show-Status }
  "logs"    { Get-Content $Log -Tail 100 -Wait }
}
