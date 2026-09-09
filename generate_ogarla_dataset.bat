@echo off
title Generate Ogarla Krea 2 Dataset
echo ===================================================
echo   Generating Ogarla Synthetic Dataset for Krea 2
echo ===================================================
echo.
python tools\build_character_dataset.py --character ogarla --count 30
pause
