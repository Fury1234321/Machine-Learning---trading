@echo off
:: ML 2.0 Trading Predictor - Startup Script for Windows

:: Set window title
title ML 2.0 Trading Predictor

:: ASCII Art Title
echo ============================================================
echo   __  __ _      ____    ___    _____               _
echo  ^|  \/  ^| ^|    ^|___ \  / _ \  ^|_   _^|             ^| ^|
echo  ^| \  / ^| ^|      __) ^|^| ^| ^| ^|   ^| ^|  _ __ __ _  __^| ^| ___ _ __
echo  ^| ^|\/^| ^| ^|     ^|__ ^< ^| ^| ^| ^|   ^| ^| ^| '__/ _` ^|/ _` ^|/ _ \ '__^|
echo  ^| ^|  ^| ^| ^|____ ___) ^|^| ^|_^| ^|  _^| ^|_^| ^| ^| (_^| ^| (_^| ^|  __/ ^|
echo  ^|_^|  ^|_^|______^|____/  \___/  ^|_____^|_^|  \__,_^|\__,_^|\___^|_^|
echo                                                             
echo ============================================================
echo              Machine Learning Trading Predictor             
echo ============================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python and try again.
    pause
    exit /b 1
)

:: Check if virtual environment exists
if not exist venv\ (
    echo Creating virtual environment...
    python -m venv venv
    
    if %ERRORLEVEL% NEQ 0 (
        echo Error: Failed to create virtual environment.
        echo Please make sure Python venv module is available.
        pause
        exit /b 1
    )
    
    echo Installing dependencies...
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
    
    if %ERRORLEVEL% NEQ 0 (
        echo Error: Failed to install dependencies.
        echo Please check the requirements.txt file.
        pause
        exit /b 1
    )
) else (
    :: Activate virtual environment
    call venv\Scripts\activate.bat
)

:: Start ML 2.0 Trading Predictor
echo Starting ML 2.0 Trading Predictor...
python main.py

:: Deactivate virtual environment on exit
call venv\Scripts\deactivate.bat

pause 