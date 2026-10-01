@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Похоже, установка ещё не выполнена.
    echo Сначала запустите install.bat, потом уже этот файл.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
python main.py
pause
