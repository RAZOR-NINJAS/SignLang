@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe
    echo Run install.ps1 to set up SignLang first.
    pause
    exit /b 1
)

if "%~1"=="" (
    rem Default double-click launch: start ISL live mode
    "%PYTHON_EXE%" "%~dp0main.py"
    if errorlevel 1 (
        echo.
        echo SignLang stopped with an error code %errorlevel%.
        pause
    )
) else (
    "%PYTHON_EXE%" "%~dp0main.py" %*
)
