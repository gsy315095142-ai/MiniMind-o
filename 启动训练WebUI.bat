@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo.
echo   ==========================================
echo     MiniMind-O  Training WebUI
echo     http://localhost:7861
echo   ==========================================
echo.

python webui/train_server.py --port 7861

pause
