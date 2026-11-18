@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: .venv\Scripts\python.exe was not found.
  echo Please ensure the virtual environment is present in the project directory.
  pause
  exit /b 1
)
echo ======================================================================
echo Starting CampusMate AI — Your NSBM ^& SLIIT Academic Assistant
echo Loading knowledge base and starting HTTP server on http://127.0.0.1:8000
echo ======================================================================
echo.
".venv\Scripts\python.exe" "src\interfaces\campusmate_server.py"
pause
