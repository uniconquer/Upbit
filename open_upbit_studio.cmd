@echo off
setlocal
cd /d "%~dp0"

echo Starting Upbit Studio...
echo Browser URL: http://localhost:8501
py -3 -m streamlit run .\src\app_streamlit.py

echo.
echo Upbit Studio stopped.
pause
