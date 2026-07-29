@echo off
setlocal
cd /d "%~dp0"

set "VENV_PYTHON=%CD%\.venv\Scripts\python.exe"
if not exist "%VENV_PYTHON%" (
    echo [ERROR] Project environment not found. Run init.bat first.
    exit /b 1
)
for /f %%T in ('"%VENV_PYTHON%" -c "import time; print(time.time())"') do set "SETUP_START=%%T"

for /f %%T in ('"%VENV_PYTHON%" -c "import time; print(time.time())"') do set "STAGE_START=%%T"
if /I "%BLUR_FACE_CPU_ONLY%"=="1" goto SAM_CPU
where nvidia-smi >nul 2>&1
if errorlevel 1 goto SAM_CPU
echo [1/2] Installing the tested CUDA 12.6 PyTorch runtime...
"%VENV_PYTHON%" -m pip install --upgrade --force-reinstall -r requirements.cuda126.lock
if errorlevel 1 exit /b 1
goto SAM_DEPS

:SAM_CPU
echo [1/2] Installing the tested CPU PyTorch runtime...
"%VENV_PYTHON%" -m pip install --upgrade --force-reinstall -r requirements.cpu.lock
if errorlevel 1 exit /b 1

:SAM_DEPS
"%VENV_PYTHON%" -c "import time; print('[TIME] PyTorch: {:.1f}s'.format(time.time()-float('%STAGE_START%')))"
for /f %%T in ('"%VENV_PYTHON%" -c "import time; print(time.time())"') do set "STAGE_START=%%T"
echo [2/2] Installing optional SAM 2.1 support...
"%VENV_PYTHON%" -m pip install -r requirements.sam2.lock
if errorlevel 1 exit /b 1

"%VENV_PYTHON%" -c "import torch; from transformers import Sam2Model, Sam2Processor, Sam2VideoModel, Sam2VideoProcessor; print('[OK] SAM 2.1 support is ready | torch', torch.__version__, '| device', 'cuda' if torch.cuda.is_available() else 'cpu')"
if errorlevel 1 exit /b 1
"%VENV_PYTHON%" -c "import time; print('[TIME] SAM dependencies: {:.1f}s'.format(time.time()-float('%STAGE_START%')))"
"%VENV_PYTHON%" -c "import time; print('SAM setup complete in {:.1f}s. The checkpoint downloads on first use unless offline mode is enabled.'.format(time.time()-float('%SETUP_START%')))"
echo Run start-ui.bat and choose SAM 2.1 under Mask engine.
