@echo off
cd /d "%~dp0"
echo Running pre-flight automated test suite...
.venv\Scripts\python.exe suite_test.py > test_run.tmp 2>&1
if %errorlevel% neq 0 (
    type test_run.tmp
    if exist test_run.tmp del test_run.tmp
    echo.
    echo ❌ Automated test suite failed! Bot startup aborted.
    pause
    exit /b %errorlevel%
)
if exist test_run.tmp del test_run.tmp
title Shallot-CUI Bot
echo [✓] Test suite passed cleanly!

echo Checking for daily codebase changes...
.venv\Scripts\python.exe auto_changelog.py
echo.
echo Starting Shallot-CUI Bot...
.venv\Scripts\python.exe bot.py
pause

