@echo off
setlocal
cd /d "%~dp0"

echo Starting Telegram control.
echo Keep this window open while using Telegram commands like /status.
py -3 .\src\main.py telegram-control

echo.
echo Telegram control stopped.
pause
