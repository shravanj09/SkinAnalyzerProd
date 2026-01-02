@echo off
REM ===================================================================
REM Verification and Testing Script
REM ===================================================================
REM This script checks if all services are running and accessible
REM ===================================================================

echo.
echo ===================================================================
echo  Service Verification and Testing
echo ===================================================================
echo.

echo [1/5] Checking if Docker is running...
docker --version > nul 2>&1
if errorlevel 1 (
    echo ✗ Docker is not running!
    echo.
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)
echo ✓ Docker is running

echo.
echo [2/5] Checking container status...
echo.
docker-compose ps
echo.

echo [3/5] Testing API Gateway (port 8000)...
curl -s http://localhost:8000/health > temp_health.json 2>&1
if exist temp_health.json (
    type temp_health.json
    del temp_health.json
    echo ✓ API Gateway is accessible
) else (
    echo ✗ API Gateway not responding
)

echo.
echo.
echo [4/5] Testing Frontend (port 3000)...
curl -s -m 5 http://localhost:3000 > temp_frontend.html 2>&1
if exist temp_frontend.html (
    findstr /C:"DOCTYPE" temp_frontend.html > nul
    if errorlevel 1 (
        echo ✗ Frontend returned unexpected response
        echo Content:
        type temp_frontend.html
    ) else (
        echo ✓ Frontend is accessible
    )
    del temp_frontend.html
) else (
    echo ✗ Frontend not responding
)

echo.
echo.
echo [5/5] Checking environment configuration...
echo.
echo Checking .env file for ML-Custom status:
findstr /C:"ENABLE_ML_CUSTOM" .env
echo.

echo Checking frontend timeout:
findstr /C:"timeout:" frontend\src\services\api.js
echo.

echo.
echo ===================================================================
echo  Quick Health Check Summary
echo ===================================================================
echo.
echo Service URLs:
echo  Frontend:     http://localhost:3000
echo  API Gateway:  http://localhost:8000
echo  API Docs:     http://localhost:8000/docs
echo  Health Check: http://localhost:8000/health
echo.
echo ML Services:
echo  MediaPipe:    http://localhost:8001/health
echo  OpenCV:       http://localhost:8003/health
echo  ML-Custom:    http://localhost:8025/health
echo.
echo Configuration:
findstr /C:"ENABLE_ML_CUSTOM" .env
findstr /C:"timeout:" frontend\src\services\api.js | findstr "300000"
echo.
echo ===================================================================
echo.

echo Open http://localhost:3000 in your browser to test the application.
echo.
pause
