<#
.SYNOPSIS
  Point the live channel at a stream link and start transcribing it (Windows equivalent of
  scripts/set_stream.sh / `make stream`). Uses the running API; no restart needed.

.EXAMPLE
  scripts\windows\stream.ps1 udp://@239.1.1.1:5000       # multicast (or just 239.1.1.1:5000)
  scripts\windows\stream.ps1 udp://0.0.0.0:5000          # unicast sent to this PC
  scripts\windows\stream.ps1 239.1.1.1:5000 -DryRun      # only print the normalised link
  $env:STT_API_TOKEN = "..."; scripts\windows\stream.ps1 ... -Api http://server:8000
#>
param(
  [Parameter(Mandatory = $true, Position = 0)][string]$Url,
  [Parameter(Position = 1)][string]$Channel = "ch1",
  [string]$Api = $(if ($env:API) { $env:API } else { "http://127.0.0.1:8000" }),
  [switch]$DryRun
)
$ErrorActionPreference = "Stop"

function ConvertTo-StreamUrl([string]$Link) {
  # Same rules as ui/common.js normalizeLink and scripts/set_stream.sh: a bare host:port becomes
  # udp://, and a multicast group (224-239.x.x.x) gets "@" so the receiver joins the group.
  $Link = $Link.Trim()
  if ($Link -match '^[a-zA-Z][a-zA-Z0-9+.-]*://') { return $Link }
  $hostPart = $Link.TrimStart('@')
  $first = 0
  if ([int]::TryParse($hostPart.Split('.')[0], [ref]$first) -and $first -ge 224 -and $first -le 239) {
    return "udp://@$hostPart"
  }
  return "udp://$Link"
}

$link = ConvertTo-StreamUrl $Url
if ($DryRun) { Write-Output $link; exit 0 }

$headers = @{}
if ($env:STT_API_TOKEN) { $headers["Authorization"] = "Bearer $($env:STT_API_TOKEN)" }

function Send-Json([string]$Method, [string]$Path, [hashtable]$Body) {
  $bytes = [System.Text.Encoding]::UTF8.GetBytes(($Body | ConvertTo-Json -Depth 5 -Compress))
  Invoke-RestMethod -Method $Method -Uri "$Api$Path" -Headers $headers `
    -ContentType "application/json; charset=utf-8" -Body $bytes -TimeoutSec 15 | Out-Null
}
function Get-StatusCode($ErrorRecord) {
  $r = $ErrorRecord.Exception.Response
  if ($r) { return [int]$r.StatusCode }
  return 0
}

try {
  Send-Json "Patch" "/channels/$Channel" @{ url = $link; enabled = $true }
} catch {
  $code = Get-StatusCode $_
  if ($code -ne 404) { Write-Error "FAILED ($code): $($_.Exception.Message)"; exit 1 }
  try {
    Send-Json "Post" "/channels" @{
      id = $Channel; name = "Channel 1"; url = $link; language = "ur"; enabled = $true
      audio_track = @{ index = 0; language = "urd" }
    }
  } catch { Write-Error "FAILED ($(Get-StatusCode $_)): $($_.Exception.Message)"; exit 1 }
}
Write-Host "OK: $Channel now reads $link" -ForegroundColor Green
Write-Host "Open $Api  (video and Urdu text start about 20 s after the stream arrives)"
if ($link -like "udp://*") {
  $port = ($link -split ':')[-1] -replace '\D.*$', ''
  Write-Host "If nothing arrives: allow the UDP port through Windows Firewall (admin PowerShell):"
  Write-Host "  New-NetFirewallRule -DisplayName 'live-stt UDP $port' -Direction Inbound -Protocol UDP -LocalPort $port -Action Allow"
}
