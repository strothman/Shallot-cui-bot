@echo off
cd /d "%~dp0"
title "ComfyUI Server Error & Diagnostics Report"
.venv\Scripts\python check_comfy_errors.py
echo.
pause
