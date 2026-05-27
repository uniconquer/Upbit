@echo off
setlocal
cd /d "%~dp0"

echo Stopping the Upbit background worker...
py -3 .\src\main.py worker-stop

echo.
pause
