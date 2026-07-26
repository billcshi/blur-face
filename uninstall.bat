@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Remove the .venv directory manually.
    exit /b 1
)

python scripts\uninstall.py %*
exit /b %errorlevel%
