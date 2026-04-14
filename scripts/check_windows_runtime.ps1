param(
    [string]$Venv = ".\.venv-win"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Windows venv not found at $Venv. Run: .\scripts\setup_windows.ps1"
}

@'
import sys
print("python", sys.version)

import cv2
import mediapipe
import numpy
import yaml
from importlib.metadata import version

print("numpy", numpy.__version__)
print("opencv", cv2.__version__)
print("mediapipe", mediapipe.__version__)
print("flask", version("flask"))

import torch

print("torch", torch.__version__)
print("torch cuda version", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device", torch.cuda.get_device_name(0))
    x = torch.ones((256, 256), device="cuda")
    print("cuda smoke sum", float((x @ x).sum().cpu()))

from ultralytics import YOLO
print("ultralytics import ok")
'@ | & $Python -
