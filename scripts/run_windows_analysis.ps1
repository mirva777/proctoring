param(
    [string]$Metadata = ".\test_data\metadata.csv",
    [string]$Output = ".\windows_smoke_results",
    [string]$Config = ".\config_test.yaml",
    [string]$ReferenceFaces = ".\reference_faces",
    [ValidateSet("auto", "cpu", "cuda", "mps")]
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = Join-Path $Root ".venv-win\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Windows venv not found. Run: .\scripts\setup_windows.ps1"
}

if (-not (Test-Path $Metadata)) {
    Write-Host "Metadata not found at $Metadata. Generating local synthetic test data..."
    & $Python .\generate_test_data.py
}

$ArgsList = @(
    "analyze_exam_snapshots.py",
    "--metadata", $Metadata,
    "--output", $Output,
    "--config", $Config,
    "--device", $Device
)

if ($ReferenceFaces -and (Test-Path $ReferenceFaces)) {
    $ArgsList += @("--reference-faces", $ReferenceFaces)
}

& $Python @ArgsList
