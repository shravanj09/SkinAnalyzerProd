@echo off
REM ===================================================================
REM Quick Fix Script - Restart API Gateway with ML-Custom Disabled
REM ===================================================================
REM This script applies the quick fix for the "No response from server" error
REM by disabling the slow ml-custom service and increasing timeouts
REM ===================================================================

echo.
echo ===================================================================
echo  QUICK FIX: Disable ML-Custom and Restart Services
echo ===================================================================
echo.
echo This will:
echo  - Keep ml-custom disabled (ENABLE_ML_CUSTOM=false in .env)
echo  - Restart API Gateway to pick up the change
echo  - Analysis will complete in 20-40 seconds instead of 90+
echo  - You'll get 165+ features instead of 192
echo.

set /p confirm="Continue? (Y/N): "
if /i not "%confirm%"=="Y" (
    echo Cancelled.
    pause
    exit /b
)

echo.
echo [1/3] Verifying .env configuration...
findstr /C:"ENABLE_ML_CUSTOM=false" .env > nul
if errorlevel 1 (
    echo ERROR: .env does not have ENABLE_ML_CUSTOM=false
    echo Please ensure this line exists in .env:
    echo   ENABLE_ML_CUSTOM=false
    pause
    exit /b 1
)
echo   ✓ ML-Custom is disabled in .env

echo.
echo [2/3] Restarting API Gateway...

REM Try docker-compose first
docker-compose restart api-gateway 2>nul
if errorlevel 1 (
    echo   Docker API error detected. Please manually restart Docker Desktop:
    echo   1. Right-click Docker Desktop in system tray
    echo   2. Select "Restart"
    echo   3. Wait 1-2 minutes
    echo   4. Run this script again
    pause
    exit /b 1
)

echo   ✓ API Gateway restarted

echo.
echo [3/3] Waiting for services to be ready...
timeout /t 15 /nobreak > nul

echo.
echo Testing API Gateway...
curl -s http://localhost:8000/health
echo.

echo.
echo ===================================================================
echo  Quick Fix Applied!
echo ===================================================================
echo.
echo Expected behavior:
echo  ✓ Analysis completes in 20-40 seconds
echo  ✓ No "No response from server" error
echo  ✓ 165+ features extracted (ml-custom's 27 features excluded)
echo.
echo Test the application at: http://localhost:3000
echo.
echo If still having issues, run: restart-services.bat
echo.
pause
