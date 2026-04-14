# Windows Run Guide

This repo now runs natively on Windows. WSL is not required.

## One-time setup

Run this from PowerShell in the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
```

What it does:

- Installs Python 3.11.9 for the current Windows user if needed.
- Creates `.venv-win`.
- Installs the Windows runtime packages.
- Installs CUDA PyTorch `2.11.0+cu130` for the NVIDIA GPU.
- Runs a runtime smoke test.

## Check the Runtime

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check_windows_runtime.ps1
```

Expected important lines:

```text
torch 2.11.0+cu130
cuda available True
cuda device NVIDIA GeForce RTX 5070 Ti
ultralytics import ok
```

## Run Analysis

Synthetic/local test data:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_windows_analysis.ps1
```

Real Moodle export already on this machine:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_windows_analysis.ps1 `
  -Metadata .\server_dumps\offline_export_20260406\metadata.csv `
  -Output .\windows_real_results `
  -Config .\config_test.yaml `
  -ReferenceFaces .\real_moodle_export\reference_faces `
  -Device auto
```

YOLO runs on CUDA when `-Device auto` detects the NVIDIA GPU. MediaPipe may log that its GPU delegate is disabled and then fall back to CPU; that is expected for the pip wheel and does not stop the app.

## Run Dashboard

For the smoke-test results:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_windows_dashboard.ps1 -OpenBrowser
```

For the existing real Moodle export results:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_windows_dashboard.ps1 `
  -Results .\server_dumps\offline_export_20260406_results `
  -Snapshots .\server_dumps\offline_export_20260406 `
  -ReviewDb .\review_data\offline_export_20260406_results\review_labels.sqlite3 `
  -Port 5001 `
  -OpenBrowser
```

Open this URL if the browser does not open automatically:

```text
http://127.0.0.1:5001
```
