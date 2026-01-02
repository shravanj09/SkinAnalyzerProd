@echo off
REM ==============================================================================
REM Docker Desktop Preventative Maintenance Script
REM ==============================================================================
REM Run this weekly to prevent Docker WSL corruption and keep Docker healthy
REM ==============================================================================

echo.
echo ========================================
echo Docker Desktop Maintenance
echo ========================================
echo This script will:
echo  - Clean up unused Docker resources
echo  - Optimize WSL
echo  - Prevent corruption issues
echo.
echo Press Ctrl+C to cancel, or
pause

echo.
echo [1/6] Checking Docker status...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)
echo      Docker is running.

echo [2/6] Cleaning up unused containers...
docker container prune -f
echo      Done.

echo [3/6] Cleaning up unused images...
docker image prune -a -f --filter "until=168h"
echo      Done (kept images from last 7 days).

echo [4/6] Cleaning up unused volumes...
docker volume prune -f
echo      Done.

echo [5/6] Cleaning up build cache...
docker builder prune -f --filter "until=168h"
echo      Done (kept cache from last 7 days).

echo [6/6] Optimizing WSL...
wsl --shutdown
timeout /t 5 /nobreak >nul
echo      WSL optimized and will restart when needed.

echo.
echo ========================================
echo Maintenance Complete!
echo ========================================
echo Docker is now optimized. Restart Docker Desktop
echo if you encounter any issues.
echo.
echo TIP: Run this script weekly to keep Docker healthy!
echo.
pause
