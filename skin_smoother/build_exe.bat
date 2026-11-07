@echo off
setlocal
cd /d "%~dp0"
title Build Skin Smoother EXE

set "PYEXE="
where py >nul 2>nul && set "PYEXE=py"
if not defined PYEXE (
  where python >nul 2>nul && set "PYEXE=python"
)
if not defined PYEXE (
  echo Python not found. Install Python 3 and try again.
  pause
  exit /b 1
)

echo Installing build tools and dependencies...
%PYEXE% -m pip install --quiet pyinstaller -r requirements.txt

echo Building SkinSmoother.exe (this takes a minute or two)...
%PYEXE% -m PyInstaller --noconfirm SkinSmoother.spec

echo.
echo Done. Your program is here:  dist\SkinSmoother.exe
echo Send that single file to anyone; they just double-click it.
pause
