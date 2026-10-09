@echo off
REM Runs the guided tool-calling demo (press Enter between scenes)
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (echo Run setup.bat first.& pause & exit /b 1)
cls
.venv\Scripts\python.exe demo.py %*
pause
