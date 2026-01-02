@echo off
REM ===================================================================
REM Restart Script for Skin Analyzer Services
REM ===================================================================
REM This script restarts all Docker containers to pick up configuration changes
REM Run this after modifying .env or service configuration files
REM ===================================================================

echo.
echo ===================================================================
echo  Restarting Skin Analyzer Services
echo ===================================================================
echo.

echo [1/5] Stopping all containers...
docker-compose down

echo.
echo [2/5] Rebuilding modified services...
docker-compose build frontend api-gateway

echo.
echo [3/5] Starting all services...
docker-compose up -d

echo.
echo [4/5] Waiting for services to be ready (30 seconds)...
timeout /t 30 /nobreak > nul

echo.
echo [5/5] Checking service health...
echo.

echo API Gateway:
curl -s http://localhost:8000/health

echo.
echo.
echo Frontend:
curl -s http://localhost:3000 | findstr "DOCTYPE"

echo.
echo.
echo ===================================================================
echo Service Status:
echo ===================================================================
docker-compose ps

echo.
echo ===================================================================
echo  Restart Complete!
echo ===================================================================
echo.
echo Access the application at: http://localhost:3000
echo API documentation at: http://localhost:8000/docs
echo.
echo If you see errors, check logs with:
echo   docker logs api-gateway
echo.
pause
