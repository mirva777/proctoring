param(
    [string]$Results = ".\windows_smoke_results",
    [string]$Snapshots = ".\test_data",
    [string]$ReviewDb = "",
    [string]$HostName = "127.0.0.1",
    [int]$Port = 5001,
    [switch]$OpenBrowser
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = Join-Path $Root ".venv-win\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Windows venv not found. Run: .\scripts\setup_windows.ps1"
}

if (-not (Test-Path $Results)) {
    throw "Results folder not found at $Results. Run: .\scripts\run_windows_analysis.ps1"
}

$ArgsList = @(
    "review_dashboard.py",
    "--results", $Results,
    "--snapshots", $Snapshots,
    "--host", $HostName,
    "--port", "$Port"
)

if ($ReviewDb) {
    $ArgsList += @("--review-db", $ReviewDb)
}

if ($OpenBrowser) {
    Start-Job -ScriptBlock {
        param($Url)
        Start-Sleep -Seconds 2
        Start-Process $Url
    } -ArgumentList "http://${HostName}:$Port" | Out-Null
}

& $Python @ArgsList
