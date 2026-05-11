@echo off
chcp 65001 >nul 2>&1

REM Set ffmpeg path
set "PATH=%~dp0ffmpeg\ffmpeg-8.1.1-full_build\bin;%PATH%"

REM Set Python encoding
set PYTHONIOENCODING=utf-8

REM Launch WebUI
cd /d "%~dp0scripts"
echo ========================================
echo   MiniMind-O WebUI starting...
echo   Open http://localhost:8888
echo ========================================
python web_demo_omni.py --port 8888

pause
