@echo off
title YT Idea Evaluator Pro v4
echo.
echo ========================================
echo    YT IDEA EVALUATOR PRO v4
echo    Teraz z ocena TEMATOW!
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python nie jest zainstalowany lub nie jest w PATH
    echo Pobierz Python z: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Create data directories
if not exist "channel_data" mkdir channel_data
if not exist "app_data" mkdir app_data

REM Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Instaluje wymagane pakiety...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Blad instalacji pakietow
        pause
        exit /b 1
    )
)

echo.
echo [OK] Uruchamiam aplikacje...
echo [INFO] Otworz przegladarke na: http://localhost:8501
echo [INFO] Aby zamknac: Ctrl+C
echo.

REM Run Streamlit
python -m streamlit run app.py --server.headless true

pause
