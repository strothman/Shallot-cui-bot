@echo off
cd /d "%~dp0\.."
title Generate Character Dataset from Reference Photos
echo =================================================================
echo   Character Dataset Generator from 1-12 Reference Photos
echo   Generates 30 high-res synthetic samples with Florence-2 captions
echo   Includes automatic resume if paused or interrupted!
echo =================================================================
echo.

set PYTHON_BIN=python
if exist ".venv\Scripts\python.exe" set PYTHON_BIN=.venv\Scripts\python.exe

if not exist "inputs\reference_character" (
    mkdir "inputs\reference_character"
    echo [!] Created folder: inputs\reference_character\
    echo [!] Please place 1 to 12 clear reference photos into that folder.
    echo.
    pause
    exit /b
)

set HAS_IMAGES=0
for %%e in (png jpg jpeg webp PNG JPG JPEG WEBP) do (
    if exist "inputs\reference_character\*.%%e" set HAS_IMAGES=1
)

if "%HAS_IMAGES%"=="0" (
    echo [!] No image files found in "inputs\reference_character\"!
    echo [!] Please place 1 to 12 photos into: inputs\reference_character\
    echo     Supported formats: .jpg, .png, .jpeg, .webp
    echo.
    pause
    exit /b
)

set TRIGGER=mychar
set /p USER_TRIGGER="Enter character trigger name [press Enter for 'mychar']: "
if defined USER_TRIGGER (
    for /f "tokens=* delims= " %%a in ("%USER_TRIGGER%") do set TRIGGER=%%a
)
if "%TRIGGER%"=="" set TRIGGER=mychar

set COUNT=30
set /p USER_COUNT="Enter total target image count [press Enter for 30]: "
if defined USER_COUNT (
    for /f "tokens=* delims= " %%a in ("%USER_COUNT%") do set COUNT=%%a
)
if "%COUNT%"=="" set COUNT=30

echo.
echo Launching generation for character trigger: "%TRIGGER%" (Target: %COUNT% samples)...
echo.

"%PYTHON_BIN%" tools\create_character_dataset_from_photos.py --input_dir inputs/reference_character --trigger "%TRIGGER%" --count %COUNT%

echo.
echo =================================================================
echo   Job Complete (or Paused). You can re-run this file at any time
echo   to automatically resume where you left off.
echo =================================================================
pause
