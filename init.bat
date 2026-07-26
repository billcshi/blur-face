@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10 or newer.
    exit /b 1
)

echo [1/4] Creating isolated environment...
python -m venv .venv
if errorlevel 1 exit /b 1
set "VENV_PYTHON=%CD%\.venv\Scripts\python.exe"

echo [2/4] Selecting PyTorch compute backend...
"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 exit /b 1

if /I "%BLUR_FACE_CPU_ONLY%"=="1" goto CPU_BACKEND
where nvidia-smi >nul 2>&1
if errorlevel 1 goto CPU_BACKEND

rem Keep an already working CUDA PyTorch installation on repeated setup runs.
"%VENV_PYTHON%" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if not errorlevel 1 goto CUDA_VERIFY

echo [GPU] NVIDIA driver detected; installing the tested CUDA 12.6 PyTorch wheel...
"%VENV_PYTHON%" -m pip install --upgrade --force-reinstall -r requirements.cuda126.lock
if errorlevel 1 goto ERR_CUDA

:CUDA_VERIFY
"%VENV_PYTHON%" -c "import torch, sys; print('[GPU] torch', torch.__version__, '| runtime CUDA', torch.version.cuda, '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'UNAVAILABLE'); sys.exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 goto ERR_CUDA
goto RUNTIME_DEPS

:CPU_BACKEND
echo [CPU] NVIDIA driver not selected; installing the standard PyTorch dependency.

:RUNTIME_DEPS
echo [3/4] Installing tested runtime dependencies...
"%VENV_PYTHON%" -m pip install -r requirements.lock
if errorlevel 1 exit /b 1
"%VENV_PYTHON%" -m pip install --no-deps --editable .
if errorlevel 1 exit /b 1

echo [4/4] Downloading and verifying models...
"%VENV_PYTHON%" scripts\download_models.py
if errorlevel 1 exit /b 1

echo.
echo Setup complete.
echo Run: .venv\Scripts\blur-face.exe input.mov -o output.mp4
exit /b 0

:ERR_CUDA
echo [ERROR] NVIDIA was detected, but the CUDA PyTorch verification failed.
echo Update the NVIDIA driver or use the official PyTorch selector.
echo To intentionally install CPU-only, run: set BLUR_FACE_CPU_ONLY=1 ^&^& init.bat
exit /b 1
