@echo off
setlocal
cd /d "%~dp0"

echo Checking the Upbit background worker status...
py -3 .\src\main.py worker-status

echo.
pause
