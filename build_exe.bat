@echo off
setlocal
cd /d "%~dp0"

REM Auto-build of dist\TelegramDigest\TelegramDigest.exe from the current sources.
REM Needs the project venv (install.bat creates it).
REM config.yaml, .env, session, data and models are NOT packed: the exe looks
REM upward from dist\TelegramDigest for the folder holding .env (project root).
REM   build_exe.bat           normal build
REM   build_exe.bat --clean   wipe build and dist first

set PY=.venv\Scripts\python.exe
if not exist "%PY%" (
    echo [ERROR] No .venv - run install.bat first
    exit /b 1
)

if /I "%~1"=="--clean" (
    echo Cleaning build and dist...
    if exist build rmdir /s /q build
    if exist dist rmdir /s /q dist
)

"%PY%" -m PyInstaller --version >nul 2>nul
if errorlevel 1 (
    echo Installing PyInstaller...
    "%PY%" -m pip install pyinstaller
    if errorlevel 1 exit /b 1
)

echo Building TelegramDigest.exe ...
"%PY%" -m PyInstaller --noconfirm --onedir --name TelegramDigest ^
    --collect-data customtkinter ^
    --collect-all llama_cpp ^
    --collect-submodules sentence_transformers ^
    --hidden-import auto_topics --hidden-import image_topics --hidden-import llm ^
    --hidden-import pipeline --hidden-import embed_filter --hidden-import storage ^
    --hidden-import plyer.platforms.win.notification ^
    gui_main.py
if errorlevel 1 (
    echo.
    echo [ERROR] Build failed.
    exit /b 1
)

echo.
echo Done: dist\TelegramDigest\TelegramDigest.exe
exit /b 0
