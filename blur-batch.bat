@echo off
setlocal
cd /d "%~dp0"

set "BLUR_FACE=%~dp0.venv\Scripts\blur-face.exe"
set "OUTPUT_DIR=%~dp0blurred"
if not exist "%BLUR_FACE%" (
    echo [ERROR] blur-face is not installed. Run init.bat first.
    exit /b 1
)
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

if not "%~1"=="" goto ONE

set "FAILED=0"
for %%f in (*.mp4 *.mov *.mkv *.avi) do (
    echo [%%f]
    "%BLUR_FACE%" "%%f" -o "%OUTPUT_DIR%\%%~nxf.mp4"
    if errorlevel 1 set "FAILED=1"
)
if "%FAILED%"=="1" (
    echo [ERROR] One or more videos failed.
    exit /b 1
)
echo Done. Output: %OUTPUT_DIR%
exit /b 0

:ONE
set "INPUT=%~1"
set "OUTPUT_NAME=%~nx1.mp4"
set "ARGS="
shift
:ONE_ARGS
if "%~1"=="" goto RUN_ONE
set "ARGS=%ARGS% "%~1""
shift
goto ONE_ARGS

:RUN_ONE
"%BLUR_FACE%" "%INPUT%" -o "%OUTPUT_DIR%\%OUTPUT_NAME%" %ARGS%
if errorlevel 1 exit /b 1
echo Done. Output: %OUTPUT_DIR%\%OUTPUT_NAME%
