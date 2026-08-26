@echo off
cd /d "%~dp0"
echo Starting ComfyUI CLI Monitor...
.venv\Scripts\python monitor.py
pause
