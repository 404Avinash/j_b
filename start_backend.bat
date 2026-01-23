@echo off
echo ========================================
echo   JATAYU - Starting Backend Server
echo ========================================
echo.
cd /d %~dp0
python -m uvicorn backend.server:app --reload --host 0.0.0.0 --port 8000
