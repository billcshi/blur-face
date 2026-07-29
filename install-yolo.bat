@echo off
setlocal
cd /d "%~dp0"

set "VENV_PYTHON=%CD%\.venv\Scripts\python.exe"
if not exist "%VENV_PYTHON%" (
    echo [ERROR] Project environment not found. Run init.bat first.
    exit /b 1
)

if /I "%BLUR_FACE_YOLO_LICENSE_ACCEPTED%"=="1" goto LICENSE_ACCEPTED
echo.
echo Optional YOLO license notice:
echo - Ultralytics is AGPL-3.0, or requires an Ultralytics Enterprise license.
echo - akanametov/yolo-face 1.0.0 states GPL-3.0 or Enterprise terms for
echo   yolov11m-face.pt; its training recipe uses WIDER FACE, whose data
echo   terms also apply.
echo - Neither component is part of blur-face's default MIT/Apache runtime.
echo See THIRD_PARTY_NOTICES.md before continuing.
choice /C YN /N /M "Accept these upstream terms and install optional YOLO? [Y/N] "
if errorlevel 2 (
    echo [NOTICE] Optional YOLO installation cancelled.
    exit /b 0
)

:LICENSE_ACCEPTED
set "SETUP_START=%TIME%"
if /I "%BLUR_FACE_CPU_ONLY%"=="1" goto YOLO_CPU
where nvidia-smi >nul 2>&1
if errorlevel 1 goto YOLO_CPU
echo [1/3] Installing the tested CUDA 12.6 PyTorch runtime...
"%VENV_PYTHON%" -m pip install --upgrade --force-reinstall -r requirements.cuda126.lock
if errorlevel 1 exit /b 1
goto YOLO_DEPS

:YOLO_CPU
echo [1/3] Installing the tested CPU PyTorch runtime...
"%VENV_PYTHON%" -m pip install --upgrade --force-reinstall -r requirements.cpu.lock
if errorlevel 1 exit /b 1

:YOLO_DEPS
echo [2/3] Installing the optional AGPL-3.0 Ultralytics backend...
"%VENV_PYTHON%" -m pip install -r requirements.yolo.lock
if errorlevel 1 exit /b 1

echo [3/3] Downloading and SHA-256 verifying the optional face weight...
"%VENV_PYTHON%" scripts\download_models.py --yolo
if errorlevel 1 exit /b 1

"%VENV_PYTHON%" -c "import torch, ultralytics; print('[OK] Optional YOLO is ready | ultralytics', ultralytics.__version__, '| torch', torch.__version__, '| device', 'cuda' if torch.cuda.is_available() else 'cpu')"
if errorlevel 1 exit /b 1
echo Choose YOLO in the UI, or use: --detector yolo --model yolov11m-face.pt
exit /b 0
