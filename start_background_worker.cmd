@echo off
setlocal
cd /d "%~dp0"

echo Starting the Upbit background worker...
py -3 .\src\main.py worker-start

echo.
pause
