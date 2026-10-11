@echo off
setlocal
cd /d "%~dp0"
echo ========================================================
echo MediPredict AI - React frontend + FastAPI backend
echo ========================================================

if not exist ".venv\Scripts\python.exe" (
  echo Creating Python virtual environment...
  python -m venv .venv
  if errorlevel 1 (echo Could not create virtual environment. & pause & exit /b 1)
)

echo Checking Python dependencies...
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (echo Python dependency installation failed. & pause & exit /b 1)

if not exist "frontend\node_modules" (
  echo Installing React frontend dependencies...
  pushd frontend
  call npm install --no-audit --no-fund
  if errorlevel 1 (popd & echo Frontend dependency installation failed. & pause & exit /b 1)
  popd
)

start "MediPredict FastAPI" cmd /k "cd /d "%~dp0" && .venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload"
timeout /t 3 /nobreak >nul
start "MediPredict React" cmd /k "cd /d "%~dp0frontend" && npm run dev -- --host 127.0.0.1"
timeout /t 2 /nobreak >nul
start "" "http://127.0.0.1:5173"
echo.
echo Frontend: http://127.0.0.1:5173
echo API docs: http://127.0.0.1:8000/docs
echo Close the React and FastAPI terminal windows to stop the app.
