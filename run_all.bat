@echo off
setlocal
cd /d "%~dp0"
echo ========================================================
echo MediPredict AI - Tests and React + FastAPI launcher
echo ========================================================
if not exist ".venv\Scripts\python.exe" (
  echo Creating Python virtual environment...
  python -m venv .venv
  if errorlevel 1 (pause & exit /b 1)
)
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (pause & exit /b 1)
.venv\Scripts\python.exe -m pytest -q
if errorlevel 1 (echo Tests failed. Review the output above. & pause & exit /b 1)
call run_web.bat
