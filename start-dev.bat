@echo off
setlocal enabledelayedexpansion
set ROOT=%~dp0

echo ========================================================
echo        Jiaoshi AI One-Click Start Script
echo ========================================================

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [Error] Python is not installed or not in PATH.
    echo Please install Python 3.10+ and try again.
    pause
    exit /b 1
)

echo [Info] Root Directory: %ROOT%

:: --- Backend Setup ---
echo.
echo [Backend] Checking environment...
if not exist "%ROOT%Backend\.venv" (
    echo [Backend] Creating virtual environment...
    cd /d "%ROOT%Backend"
    python -m venv .venv
    if !errorlevel! neq 0 (
        echo [Error] Failed to create Backend venv.
        pause
        exit /b 1
    )
    echo [Backend] Installing dependencies...
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo [Error] Failed to install Backend dependencies.
        pause
        exit /b 1
    )
) else (
    echo [Backend] Environment exists.
)

:: --- Frontend Setup ---
echo.
echo [Frontend] Checking environment...
if not exist "%ROOT%Frontend\venv" (
    echo [Frontend] Creating virtual environment...
    cd /d "%ROOT%Frontend"
    python -m venv venv
    if !errorlevel! neq 0 (
        echo [Error] Failed to create Frontend venv.
        pause
        exit /b 1
    )
    echo [Frontend] Installing dependencies...
    venv\Scripts\python.exe -m pip install --upgrade pip
    venv\Scripts\pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo [Error] Failed to install Frontend dependencies.
        pause
        exit /b 1
    )
) else (
    echo [Frontend] Environment exists.
)

:: --- Start Services ---
echo.
echo [Start] Launching services...

:: Start Backend
pushd "%ROOT%Backend"
start "Backend Service" "%ROOT%Backend\.venv\Scripts\python.exe" app.py
popd

:: Start Frontend
pushd "%ROOT%Frontend"
start "Frontend Service" "%ROOT%Frontend\venv\Scripts\python.exe" manage.py runserver 8001
popd

:: Open Browser
timeout /t 5 >nul
start "" "http://127.0.0.1:8001/"

echo.
echo [Info] Services started. Please do not close the console windows.
echo Press any key to exit this launcher (services will keep running).
pause
endlocal
