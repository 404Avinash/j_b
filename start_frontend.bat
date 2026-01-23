@echo off
echo ========================================
echo   JATAYU - Starting Frontend
echo ========================================
echo.
cd /d %~dp0\frontend
npm install
npm run dev
