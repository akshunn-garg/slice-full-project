param([switch]$Foreground)

$ErrorActionPreference = 'Stop'
$appDirectory = $PSScriptRoot
$pythonPath = [IO.Path]::GetFullPath((Join-Path $appDirectory '..\.venv\Scripts\python.exe'))
$privateDirectory = Join-Path $appDirectory 'private'
$url = 'http://127.0.0.1:5000'

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Project Python environment not found: $pythonPath"
}
if ($Foreground) {
    Push-Location -LiteralPath $appDirectory
    try { & $pythonPath -u app.py --no-browser }
    finally { Pop-Location }
    exit $LASTEXITCODE
}

# Starting again should reuse the already-running local SLiCE app.
try {
    $existing = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2
    if ($existing.Content -match 'SLiCE') {
        Write-Output "SLiCE is already running: $url"
        exit 0
    }
    throw "Port 5000 is occupied by another web application."
} catch {
    if ($_.Exception.Message -like 'Port 5000*') { throw }
}

New-Item -ItemType Directory -Path $privateDirectory -Force | Out-Null
$stdoutPath = Join-Path $privateDirectory 'app_stdout.log'
$stderrPath = Join-Path $privateDirectory 'app_stderr.log'
$process = Start-Process -FilePath $pythonPath -ArgumentList @('-u', 'app.py', '--no-browser') `
    -WorkingDirectory $appDirectory -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
$process.Id | Set-Content -LiteralPath (Join-Path $privateDirectory 'app_launcher.pid')
Write-Output "Starting SLiCE in the background (launcher PID $($process.Id))."
Write-Output "Logs: $stdoutPath and $stderrPath"

for ($attempt = 0; $attempt -lt 20; $attempt++) {
    try {
        $response = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 1
        if ($response.StatusCode -eq 200 -and $response.Content -match 'SLiCE') {
            Write-Output "Ready: $url"
            Write-Output 'The server runs in the background; this terminal remains available.'
            exit 0
        }
    } catch {}
    $process.Refresh()
    if ($process.HasExited) {
        if (Test-Path -LiteralPath $stderrPath) { Get-Content -LiteralPath $stderrPath -Tail 20 }
        throw 'SLiCE exited during startup. See the app logs above.'
    }
    Start-Sleep -Milliseconds 500
}
Write-Output 'Startup is still in progress; check the logs above for progress or errors.'
Write-Output "Open $url when the log reports Ready."
