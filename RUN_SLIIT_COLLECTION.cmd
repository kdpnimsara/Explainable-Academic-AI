@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: .venv\Scripts\python.exe was not found.
  echo Extract this package into the Explainable-Academic-AI project root.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" "scripts\collect_sliit_sources.py"
set "RESULT=%ERRORLEVEL%"
echo.
if "%RESULT%"=="0" (
  echo Incremental SLIIT collection finished. Existing files were preserved.
) else (
  echo Collection finished with errors or no manifest sources. Review output.
)
pause
exit /b %RESULT%
