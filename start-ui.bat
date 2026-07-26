@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. Run init.bat first.
    exit /b 1
)
".venv\Scripts\python.exe" -m blurface.webui %*
