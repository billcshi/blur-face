@echo off
REM ===========================================
REM blur-face batch processor
REM Put this .bat in the same folder as your videos
REM Drag a video onto it, or double-click to process all videos
REM ===========================================
setlocal enabledelayedexpansion

set SCRIPT=%~dp0blur-face.py
set OUTPUT_DIR=%~dp0blurred

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

if "%~1"=="" (
    echo Processing all videos in current folder...
    for %%f in (*.mp4 *.mov *.mkv *.avi *.MP4 *.MOV *.MKV *.AVI) do (
        echo [%%f]
        python "%SCRIPT%" "%%f" -o "%OUTPUT_DIR%\%%f" %*
        echo.
    )
) else (
    echo Processing: %~nx1
    python "%SCRIPT%" "%~1" -o "%OUTPUT_DIR%\%~nx1" %2 %3 %4 %5 %6 %7 %8 %9
)

echo.
echo Done! Output in: %OUTPUT_DIR%
pause
