$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Creating local Python 3.12 environment..."
    py -3.12 -m venv .venv
    & $python -m pip install --upgrade pip
    & $python -m pip install -r requirements.txt
}

& $python -c "import cv2, fastapi, numpy, onnxruntime, PIL, rapidocr_onnxruntime, uvicorn" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing missing local dependencies..."
    & $python -m pip install -r requirements.txt
}

$listeners = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
foreach ($listener in $listeners) {
    $listenerPid = $listener.OwningProcess
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $listenerPid"
    if ($process.CommandLine -like "*$PSScriptRoot*" -and
        ($process.CommandLine -like "*main.py*" -or $process.CommandLine -like "*uvicorn*")) {
        Write-Host "Stopping stale Number Plate AI server (PID $listenerPid)..."
        Stop-Process -Id $listenerPid -Force
        Start-Sleep -Milliseconds 500
    } else {
        throw "Port 8000 is used by another application (PID $listenerPid). Stop it and run this script again."
    }
}

Write-Host "Starting Number Plate AI at http://127.0.0.1:8000"
Write-Host "Keep this window open. Press Ctrl+C to stop."
& $python main.py
