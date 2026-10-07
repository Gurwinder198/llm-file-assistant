@echo off
REM One-time setup: finds Python, creates .venv, installs requirements, creates .env
cd /d "%~dp0"
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (where python >nul 2>nul && set "PY=python")
if not defined PY (
  echo Python was not found. Install it from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" during install, or run:
  echo     winget install Python.Python.3.12
  echo Then open a NEW terminal and run setup.bat again.
  pause
  exit /b 1
)
echo Using: %PY%
%PY% --version
if not exist .venv (%PY% -m venv .venv || goto :fail)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt || goto :fail
if not exist .env (copy .env.example .env >nul & echo Created .env - open it and paste your OPENROUTER_API_KEY.)
echo.
echo Setup complete. Run the assistant with:
echo     .venv\Scripts\python.exe llm_file_assistant.py -v "Find resumes mentioning Python experience"
pause
exit /b 0
:fail
echo Setup failed - see the error above.
pause
exit /b 1
