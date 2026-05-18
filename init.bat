@echo off
REM ==============================================
REM  blur-face - init script (Windows)
REM  Double-click to run
REM ==============================================

cd /d "%~dp0"

echo ============================================
echo   blur-face - Setup
echo ============================================

REM --- Check Python ---
where python >nul 2>&1
if %errorlevel% neq 0 goto ERR_PYTHON
echo [OK] Python found

REM --- Install deps ---
echo [1/3] Installing Python packages...
pip install ultralytics opencv-python imageio-ffmpeg numpy
if %errorlevel% neq 0 goto ERR_PIP

REM --- CUDA ---
echo [2/3] Checking CUDA...
python -c "import torch; print('[OK] GPU:', torch.cuda.get_device_name(0)) if torch.cuda.is_available() else print('[WARN] CUDA not available - will use CPU')"

REM --- Download models ---
echo [3/3] Downloading models...

if not exist models mkdir models

if exist models\yolo26n-face.pt goto SKIP_NANO
echo   Downloading yolo26n-face.pt ^(5.6 MB^)...
powershell -Command "Invoke-WebRequest -Uri 'https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolo26n-face.pt' -OutFile 'models\yolo26n-face.pt' -UseBasicParsing"
if exist models\yolo26n-face.pt (echo     [OK]) else echo     [SKIP]
goto AFTER_NANO
:SKIP_NANO
echo   [OK] yolo26n-face.pt exists
:AFTER_NANO

if exist models\yolov11m-face.pt goto SKIP_MEDIUM
echo   Downloading yolov11m-face.pt ^(38.6 MB^)...
powershell -Command "Invoke-WebRequest -Uri 'https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11m-face.pt' -OutFile 'models\yolov11m-face.pt' -UseBasicParsing"
if exist models\yolov11m-face.pt (echo     [OK]) else echo     [SKIP]
goto DONE
:SKIP_MEDIUM
echo   [OK] yolov11m-face.pt exists

:DONE
echo ============================================
echo   Setup complete.
echo   Usage: python blur-face.py input.mov --debug
echo ============================================
pause
exit /b 0

:ERR_PYTHON
echo [ERROR] Python not found. Install from https://python.org
pause
exit /b 1

:ERR_PIP
echo [ERROR] Failed to install packages.
pause
exit /b 1
