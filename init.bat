@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>&1
if errorlevel 1 (
    where python >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python not found. Install Python 3.10 or newer.
        exit /b 1
    )
    set "BASE_PYTHON=python"
) else (
    set "BASE_PYTHON=py -3"
)
if /I "%BASE_PYTHON%"=="python" if defined VIRTUAL_ENV (
    echo [ERROR] Deactivate the existing virtual environment before running init.bat.
    exit /b 1
)
for /f %%T in ('%BASE_PYTHON% -c "import time; print(time.time())"') do set "SETUP_START=%%T"

for /f %%T in ('%BASE_PYTHON% -c "import time; print(time.time())"') do set "STAGE_START=%%T"
echo [1/4] Removing the previous isolated environment and generated metadata...
%BASE_PYTHON% scripts\uninstall.py
if errorlevel 1 exit /b 1
%BASE_PYTHON% -c "import time; print('[TIME] Cleanup: {:.1f}s'.format(time.time()-float('%STAGE_START%')))"

for /f %%T in ('%BASE_PYTHON% -c "import time; print(time.time())"') do set "STAGE_START=%%T"
echo [2/4] Creating isolated environment...
%BASE_PYTHON% -m venv .venv
if errorlevel 1 exit /b 1
set "VENV_PYTHON=%CD%\.venv\Scripts\python.exe"
"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
%BASE_PYTHON% -c "import time; print('[TIME] Environment: {:.1f}s'.format(time.time()-float('%STAGE_START%')))"

for /f %%T in ('%BASE_PYTHON% -c "import time; print(time.time())"') do set "STAGE_START=%%T"
echo [3/4] Installing tested base runtime dependencies...
"%VENV_PYTHON%" -m pip install -r requirements.lock
if errorlevel 1 exit /b 1
"%VENV_PYTHON%" -m pip install --no-deps --editable .
if errorlevel 1 exit /b 1
%BASE_PYTHON% -c "import time; print('[TIME] Runtime dependencies: {:.1f}s'.format(time.time()-float('%STAGE_START%')))"

for /f %%T in ('%BASE_PYTHON% -c "import time; print(time.time())"') do set "STAGE_START=%%T"
echo [4/4] Downloading and verifying the MIT-licensed OpenCV YuNet model...
"%VENV_PYTHON%" scripts\download_models.py
if errorlevel 1 exit /b 1
%BASE_PYTHON% -c "import time; print('[TIME] YuNet model: {:.1f}s'.format(time.time()-float('%STAGE_START%')))"

where ffmpeg >nul 2>&1
if not errorlevel 1 goto FFMPEG_READY
echo.
echo [WARN] FFmpeg is not on PATH and is required for video output.
echo Gyan.FFmpeg is an external third-party package of about 250 MiB.
echo Its full build is GPL-licensed and is not distributed by blur-face.
if /I "%CI%"=="true" goto FFMPEG_SKIPPED
if /I "%BLUR_FACE_SKIP_FFMPEG_INSTALL%"=="1" goto FFMPEG_SKIPPED
where winget >nul 2>&1
if errorlevel 1 goto FFMPEG_NO_WINGET
choice /C YN /N /M "Install Gyan.FFmpeg with WinGet now? [Y/N] "
if errorlevel 2 goto FFMPEG_SKIPPED
winget install --id Gyan.FFmpeg --exact --accept-package-agreements --accept-source-agreements
if errorlevel 1 goto FFMPEG_INSTALL_FAILED
set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%"
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo [NOTICE] FFmpeg installed. Open a new terminal before running blur-face.
) else (
    echo [OK] FFmpeg installed and available in this terminal.
)
goto FFMPEG_DONE

:FFMPEG_NO_WINGET
echo [WARN] WinGet is unavailable. Install a system FFmpeg build manually.
echo        Then open a new terminal and run: ffmpeg -version
goto FFMPEG_DONE

:FFMPEG_INSTALL_FAILED
echo [WARN] WinGet could not install FFmpeg. Install it manually or use --ffmpeg.
goto FFMPEG_DONE

:FFMPEG_SKIPPED
echo [NOTICE] FFmpeg installation skipped. Video processing will require FFmpeg later.
goto FFMPEG_DONE

:FFMPEG_READY
echo [OK] System FFmpeg found on PATH.

:FFMPEG_DONE
echo.
echo Optional detector:
echo YuNet is already installed and remains the MIT-licensed default.
echo YOLO may improve difficult-face recall, but Ultralytics is AGPL-3.0
echo unless you have an Enterprise license. The akanametov/yolo-face 1.0.0
echo weight is GPL-3.0 or requires an applicable Enterprise license; its
echo training recipe uses WIDER FACE, whose dataset terms also apply.
echo See THIRD_PARTY_NOTICES.md for exact upstream links.
if /I "%CI%"=="true" goto YOLO_SKIPPED
if /I "%BLUR_FACE_SKIP_YOLO_INSTALL%"=="1" goto YOLO_SKIPPED
if /I "%BLUR_FACE_INSTALL_YOLO%"=="1" goto YOLO_INSTALL
choice /C YN /N /M "Install the optional YOLO backend and face weight? [Y/N] "
if errorlevel 2 goto YOLO_SKIPPED

:YOLO_INSTALL
set "BLUR_FACE_YOLO_LICENSE_ACCEPTED=1"
call install-yolo.bat
if errorlevel 1 exit /b 1
goto YOLO_DONE

:YOLO_SKIPPED
echo [NOTICE] Optional YOLO skipped; YuNet remains selected.

:YOLO_DONE

echo.
echo Optional mask engine:
echo SAM 2.1 installs PyTorch and Transformers and may use several GiB of disk.
echo The selected SAM checkpoint downloads separately on first use unless it is
echo already cached or you select a local model directory.
if /I "%CI%"=="true" goto SAM_SKIPPED
if /I "%BLUR_FACE_SKIP_SAM_INSTALL%"=="1" goto SAM_SKIPPED
if /I "%BLUR_FACE_INSTALL_SAM%"=="1" goto SAM_INSTALL
choice /C YN /N /M "Install optional SAM 2.1 support now? [Y/N] "
if errorlevel 2 goto SAM_SKIPPED

:SAM_INSTALL
call install-sam2.bat
if errorlevel 1 exit /b 1
goto SAM_DONE

:SAM_SKIPPED
echo [NOTICE] Optional SAM 2.1 skipped; geometric masking remains ready.

:SAM_DONE

echo.
%BASE_PYTHON% -c "import time; print('Setup complete in {:.1f}s.'.format(time.time()-float('%SETUP_START%')))"
echo Geometric mode with YuNet is ready without PyTorch.
echo Run install-sam2.bat for SAM or install-yolo.bat for YOLO if skipped above.
echo Dependency and model licenses: THIRD_PARTY_NOTICES.md
echo Run: .venv\Scripts\blur-face.exe input.mov -o output.mp4
exit /b 0
