@echo off
chcp 65001 >nul
cd /d "%~dp0"

if exist "dist\TelegramDigest\TelegramDigest.exe" (
    echo Запускаю TelegramDigest.exe...
    echo Если это первый запуск -- Telegram пришлёт код подтверждения,
    echo его нужно ввести прямо в этом окне.
    echo.
    "dist\TelegramDigest\TelegramDigest.exe"
    echo.
    echo Приложение закрылось ^(код выхода: %errorlevel%^).
    if not "%errorlevel%"=="0" (
        echo.
        echo Похоже, что-то пошло не так -- см. текст выше.
        echo Если ошибка про соединение с Telegram -- см. раздел "Прокси" в README.md.
    )
    pause
    exit /b 0
)

if not exist ".venv\Scripts\python.exe" (
    echo Похоже, установка ещё не выполнена.
    echo Сначала запустите install.bat, потом уже этот файл.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
echo Если это первый запуск -- Telegram пришлёт код подтверждения,
echo его нужно ввести прямо в этом окне.
echo.
python gui_main.py
echo.
echo Программа завершилась ^(код выхода: %errorlevel%^).
if not "%errorlevel%"=="0" (
    echo Если ошибка про соединение с Telegram -- см. раздел "Прокси" в README.md.
)
pause
