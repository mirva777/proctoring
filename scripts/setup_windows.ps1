param(
    [string]$PythonVersion = "3.11.9",
    [string]$TorchCuda = "cu130",
    [string]$TorchVersion = "2.11.0",
    [string]$TorchVisionVersion = "0.26.0",
    [switch]$SkipPythonInstall,
    [switch]$CpuTorch
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$PythonHome = Join-Path $env:LOCALAPPDATA "Programs\Python\Python311"
$SystemPython = Join-Path $PythonHome "python.exe"

if (-not $SkipPythonInstall -and -not (Test-Path $SystemPython)) {
    $Installer = Join-Path $env:TEMP "python-$PythonVersion-amd64.exe"
    $Url = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-amd64.exe"

    Write-Host "Downloading Python $PythonVersion..."
    Invoke-WebRequest -Uri $Url -OutFile $Installer

    Write-Host "Installing Python $PythonVersion per-user..."
    $InstallArgs = @(
        "/quiet",
        "InstallAllUsers=0",
        "InstallLauncherAllUsers=0",
        "Include_launcher=0",
        "PrependPath=1",
        "Include_pip=1",
        "Include_test=0",
        "Include_tcltk=0",
        "Include_doc=0",
        "Include_tools=0",
        "Shortcuts=0",
        "TargetDir=$PythonHome"
    )
    $Process = Start-Process -FilePath $Installer -ArgumentList $InstallArgs -Wait -PassThru
    if ($Process.ExitCode -ne 0) {
        throw "Python installer failed with exit code $($Process.ExitCode)"
    }
}

if (-not (Test-Path $SystemPython)) {
    throw "Could not find Python at $SystemPython"
}

if (-not (Test-Path ".\.venv-win\Scripts\python.exe")) {
    Write-Host "Creating .venv-win..."
    & $SystemPython -m venv .venv-win
}

$VenvPython = ".\.venv-win\Scripts\python.exe"

Write-Host "Upgrading packaging tools..."
& $VenvPython -m pip install --upgrade pip setuptools wheel

Write-Host "Installing Windows runtime requirements..."
& $VenvPython -m pip install --prefer-binary -r .\requirements-windows.txt

if (-not $CpuTorch) {
    Write-Host "Installing PyTorch CUDA wheel ($TorchCuda)..."
    & $VenvPython -m pip install --force-reinstall --index-url "https://download.pytorch.org/whl/$TorchCuda" "torch==$TorchVersion+$TorchCuda" "torchvision==$TorchVisionVersion+$TorchCuda"
}

Write-Host "Checking runtime..."
& .\scripts\check_windows_runtime.ps1

Write-Host ""
Write-Host "Windows setup complete."
Write-Host "Run analysis:  .\scripts\run_windows_analysis.ps1"
Write-Host "Run dashboard: .\scripts\run_windows_dashboard.ps1 -OpenBrowser"
