@echo off
REM ML Trading Predictor - Startup Script for Windows

REM Change to the directory containing this script
cd /d "%~dp0"

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH. Please install Python and try again.
    pause
    exit /b 1
)

REM Check if we need to install dependencies
if not exist venv\ (
    echo Setting up virtual environment for first run...
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

REM Start the application
python main.py
pause 