@echo off
cd /d "%~dp0\.."
title Generate Valerie Krea 2 Dataset
echo ===================================================
echo   Generating Valerie Synthetic Dataset for Krea 2
echo   Engine: SDXL (hyphoriaIlluNAI + jen_epoch_5 + Semi-Realism 0.90)
echo   Target: 30 Diverse Modeling Samples with Florence-2
echo ===================================================
echo.

set PYTHON_BIN=python
if exist ".venv\Scripts\python.exe" set PYTHON_BIN=.venv\Scripts\python.exe

"%PYTHON_BIN%" tools\build_character_dataset.py --character valerie --count 30 --engine sdxl --overwrite
pause
