@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ===============================================
echo  Telegram Digest - Установка и сборка .exe
echo ===============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ОШИБКА] Python не найден вообще.
    echo Установите Python 3.11 или 3.12 с https://python.org
    echo При установке ОБЯЗАТЕЛЬНО отметьте галочку "Add python.exe to PATH".
    echo.
    pause
    exit /b 1
)

REM PyTorch usually lags behind brand-new Python releases by many months
REM (no wheels for the newest one or two versions yet). Rather than fail
REM on whatever "python" happens to resolve to in PATH, look for a known
REM PyTorch-compatible version via the "py" launcher first, and only fall
REM back to plain "python" (with a warning) if none is found.
set PYCMD=

where py >nul 2>nul
if not errorlevel 1 (
    for %%V in (3.12 3.11 3.10) do (
        if "!PYCMD!"=="" (
            py -%%V --version >nul 2>nul
            if not errorlevel 1 (
                set PYCMD=py -%%V
                echo Найден подходящий для PyTorch Python: %%V
            )
        )
    )
)

if "!PYCMD!"=="" (
    for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
    echo.
    echo [ПРЕДУПРЕЖДЕНИЕ] Не нашёл Python 3.10/3.11/3.12 через "py"-лаунчер.
    echo Буду использовать системный python ^(версия !PYVER!^).
    echo Если версия 3.13 или новее -- установка PyTorch, скорее всего,
    echo не сработает ни одним способом ^(ниже будет предложено продолжить
    echo только с фильтром по ключевым словам, без ИИ-модели^).
    echo.
    echo Чтобы получить полноценный ИИ-фильтр: поставьте Python 3.11 или 3.12
    echo с https://python.org ^(отметьте "Add python.exe to PATH"^), НЕ удаляя
    echo текущий Python, затем запустите install.bat заново -- он сам найдёт
    echo подходящую версию через "py"-лаунчер.
    echo.
    pause
    set PYCMD=python
)

set RECREATE_VENV=0
if exist ".venv\Scripts\python.exe" (
    for /f "tokens=2 delims=." %%v in ('".venv\Scripts\python.exe" --version 2^>^&1') do set VENV_MINOR=%%v
    echo.
    echo Уже есть виртуальное окружение .venv ^(Python 3.!VENV_MINOR!^).
    if !VENV_MINOR! GEQ 13 (
        echo Эта версия слишком новая для PyTorch -- пересоздаю .venv с !PYCMD!.
        set RECREATE_VENV=1
    )
) else (
    set RECREATE_VENV=1
)

if "!RECREATE_VENV!"=="1" (
    if exist ".venv" rmdir /s /q ".venv"
    echo Создаю виртуальное окружение .venv ^(!PYCMD!^)...
    !PYCMD! -m venv .venv
    if errorlevel 1 (
        echo [ОШИБКА] Не удалось создать виртуальное окружение.
        pause
        exit /b 1
    )
) else (
    echo Виртуальное окружение подходит, пропускаю пересоздание.
)

call ".venv\Scripts\activate.bat"

echo.
python -m pip install --upgrade pip >nul

echo Определяю версию Python в виртуальном окружении...
for /f "tokens=2 delims= " %%v in ('python --version') do set PYVER=%%v
echo Версия: %PYVER%

where nvidia-smi >nul 2>nul
if errorlevel 1 (
    set HAS_GPU=0
    echo Видеокарта NVIDIA не обнаружена -- буду ставить CPU-версию torch.
) else (
    set HAS_GPU=1
    echo Видеокарта NVIDIA обнаружена -- сначала попробую CUDA-версию torch ^(быстрее^).
)

set TORCH_OK=0

if "%HAS_GPU%"=="1" (
    echo.
    echo Пробую torch с CUDA 12.1...
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    if not errorlevel 1 set TORCH_OK=1

    if "!TORCH_OK!"=="0" (
        echo Не подошло. Пробую torch с CUDA 11.8...
        pip install torch --index-url https://download.pytorch.org/whl/cu118
        if not errorlevel 1 set TORCH_OK=1
    )
)

if "!TORCH_OK!"=="0" (
    echo.
    echo Пробую CPU-версию torch с официального индекса PyTorch...
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    if not errorlevel 1 set TORCH_OK=1
)

if "!TORCH_OK!"=="0" (
    echo.
    echo Пробую обычную установку torch с PyPI ^(последняя попытка^)...
    pip install torch
    if not errorlevel 1 set TORCH_OK=1
)

if "!TORCH_OK!"=="0" (
    echo.
    echo [ОШИБКА] Не удалось установить torch ни одним способом.
    echo Скорее всего, для вашей версии Python ^(%PYVER%^) ещё нет готовых
    echo сборок PyTorch -- обычно это самые новые версии Python ^(3.13+^).
    echo Решение: поставьте Python 3.11 или 3.12 с https://python.org
    echo ^(можно параллельно с текущей версией^) и запустите install.bat заново.
    echo.
    echo Также приложение можно использовать БЕЗ torch: тогда фильтр постов
    echo автоматически работает по ключевым словам вместо ИИ-модели.
    echo Для этого пропустите установку torch и продолжите install.bat --
    echo остальные шаги отработают нормально.
    echo.
    choice /C YN /M "Продолжить установку без torch (только ключевые слова)"
    if errorlevel 2 (
        pause
        exit /b 1
    )
) else (
    echo.
    echo torch установлен успешно.
)

echo.
echo Устанавливаю остальные зависимости из requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ОШИБКА] Не удалось установить зависимости.
    pause
    exit /b 1
)

echo.
if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo Создан файл .env
) else (
    echo Файл .env уже существует, не трогаю.
)

if not exist "config.yaml" (
    copy "config.example.yaml" "config.yaml" >nul
    echo Создан файл config.yaml
) else (
    echo Файл config.yaml уже существует, не трогаю.
)

echo.
echo ===============================================
echo  Сборка TelegramDigest.exe
echo ===============================================
echo Это может занять несколько минут и потребует несколько ГБ места
echo ^(torch внутри exe весит много^). Если сборка не удастся -- не страшно,
echo приложение всё равно можно запускать через run_gui.bat.
echo.

REM NOTE: deliberately NOT using --windowed here. Telethon asks for your
REM Telegram login code (and 2FA password, if set) via a plain console
REM prompt on first run -- with --windowed there is no console at all and
REM nowhere to type it. A console window will appear alongside the app
REM window; type the code there the first time, then you can ignore it.
pip install pyinstaller
pyinstaller --noconfirm --onedir --name "TelegramDigest" --collect-data customtkinter gui_main.py

if errorlevel 1 (
    echo.
    echo [ПРЕДУПРЕЖДЕНИЕ] Сборка .exe не удалась. Ничего страшного --
    echo запускайте приложение через run_gui.bat вместо готового .exe.
) else (
    if exist "dist\TelegramDigest\TelegramDigest.exe" (
        echo.
        echo Готово: dist\TelegramDigest\TelegramDigest.exe
        echo Запускайте через run_gui.bat или "Telegram Digest.vbs".
    )
)

echo.
echo ===============================================
echo  Установка завершена!
echo ===============================================
echo.
echo Сейчас откроются два файла для настройки:
echo   1) .env          -- впишите TG_API_ID и TG_API_HASH
echo      ^(получить на https://my.telegram.org, раздел API development tools^)
echo   2) config.yaml    -- впишите свои интересы (interest_examples)
echo.
echo После того как впишете данные -- сохраните файл (Ctrl+S) и закройте блокнот.
echo.
pause

notepad ".env"
notepad "config.yaml"

echo.
echo ===============================================
echo  Готово! Дальше запускайте TelegramDigest.exe
echo  (либо "Telegram Digest.vbs", либо run_gui.bat, если .exe не собрался).
echo ===============================================
pause
