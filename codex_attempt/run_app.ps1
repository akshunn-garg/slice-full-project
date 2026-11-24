param([switch]$Foreground)
$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$url = 'http://127.0.0.1:5001'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw "Python environment not found: $pythonPath" }
if ($Foreground) {
    Push-Location -LiteralPath $PSScriptRoot
    try { & $pythonPath -B -u app.py --port 5001 --no-browser }
    finally { Pop-Location }
    exit $LASTEXITCODE
}
try {
    $health = Invoke-RestMethod -Uri "$url/health" -TimeoutSec 2
    if ($health.app -eq 'codex_attempt' -and $health.ready) { Write-Output "Already running: $url"; exit 0 }
    throw 'Port 5001 is occupied by another application.'
} catch { if ($_.Exception.Message -like 'Port 5001*') { throw } }
$logDirectory = Join-Path $PSScriptRoot 'private'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$process = Start-Process -FilePath $pythonPath -ArgumentList @('-B','-u','app.py','--port','5001','--no-browser') `
    -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logDirectory 'app_stdout.log') `
    -RedirectStandardError (Join-Path $logDirectory 'app_stderr.log')
$process.Id | Set-Content -LiteralPath (Join-Path $logDirectory 'app_launcher.pid')
Write-Output "Starting codex_attempt in background (launcher PID $($process.Id))."
for ($attempt = 0; $attempt -lt 20; $attempt++) {
    try {
        $health = Invoke-RestMethod -Uri "$url/health" -TimeoutSec 1
        if ($health.app -eq 'codex_attempt' -and $health.ready) { Write-Output "Ready: $url"; exit 0 }
    } catch {}
    $process.Refresh()
    if ($process.HasExited) {
        Get-Content -LiteralPath (Join-Path $logDirectory 'app_stderr.log') -Tail 20
        throw 'The app exited during startup.'
    }
    Start-Sleep -Milliseconds 500
}
Write-Output "Startup is continuing. Logs: $logDirectory\app_stdout.log and app_stderr.log"
