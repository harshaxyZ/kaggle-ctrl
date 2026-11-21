@echo off
title kaggle ctrl
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  echo Python not found. Install Python 3.11 or newer.
  pause
  exit /b 1
)

echo Starting kaggle ctrl...
echo Close this window to stop the dashboard.
echo.

python run.py

echo.
echo Dashboard stopped.
pause
