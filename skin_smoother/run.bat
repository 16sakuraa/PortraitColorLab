@echo off
setlocal
cd /d "%~dp0"
title Skin Smoother

rem Prefer the Python launcher "py" (usually on PATH), then fall back to python.
set "PYEXE="
where py >nul 2>nul && set "PYEXE=py"
if not defined PYEXE (
  where python >nul 2>nul && set "PYEXE=python"
)

if not defined PYEXE (
  echo.
  echo Could not find Python. Install Python 3 from python.org and tick
  echo "Add python.exe to PATH" during setup, then run this file again.
  echo.
  pause
  exit /b 1
)

echo Using interpreter: %PYEXE%
%PYEXE% -c "import cv2, numpy, PIL" 2>nul
if errorlevel 1 (
  echo Installing required packages, one moment...
  %PYEXE% -m pip install -r requirements.txt
)

%PYEXE% serve.py
pause
