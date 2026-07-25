@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10 or newer.
    exit /b 1
)

echo [1/3] Creating isolated environment...
python -m venv .venv
if errorlevel 1 exit /b 1
set "VENV_PYTHON=%CD%\.venv\Scripts\python.exe"

echo [2/3] Installing tested dependencies...
"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
"%VENV_PYTHON%" -m pip install -r requirements.lock
if errorlevel 1 exit /b 1
"%VENV_PYTHON%" -m pip install --no-deps .
if errorlevel 1 exit /b 1

echo [3/3] Downloading and verifying models...
"%VENV_PYTHON%" scripts\download_models.py
if errorlevel 1 exit /b 1

echo.
echo Setup complete.
echo Run: .venv\Scripts\blur-face.exe input.mov -o output.mp4
exit /b 0
