@echo off
REM Windows startup script for AI Facial Analysis Platform

echo ===================================
echo AI Facial Analysis Platform
echo Starting all services...
echo ===================================
echo.

REM Check if Docker is running
docker ps >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running!
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

echo [OK] Docker is running
echo.

REM Check if .env file exists
if not exist ".env" (
    echo ERROR: .env file not found!
    echo Please copy .env.example to .env and configure it.
    pause
    exit /b 1
)

echo [OK] Environment file found
echo.

REM Start services
echo Starting all services with Docker Compose...
echo This may take 2-5 minutes on first run...
echo.

docker-compose up -d --build

if errorlevel 1 (
    echo.
    echo ERROR: Failed to start services!
    echo Check the logs above for details.
    pause
    exit /b 1
)

echo.
echo ===================================
echo Services started successfully!
echo ===================================
echo.
echo Waiting for services to be ready...
timeout /t 10 /nobreak >nul

echo.
echo Checking service health...
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo WARNING: API Gateway not responding yet
    echo Services may still be initializing...
    echo Please wait 1-2 minutes and check http://localhost:8000/health
) else (
    echo [OK] API Gateway is healthy
)

echo.
echo ===================================
echo Access Points:
echo ===================================
echo Frontend:    http://localhost:3000
echo API Gateway: http://localhost:8000
echo API Docs:    http://localhost:8000/docs
echo MinIO UI:    http://localhost:9001
echo ===================================
echo.
echo Opening frontend in browser...
timeout /t 2 /nobreak >nul
start http://localhost:3000

echo.
echo To view logs: docker-compose logs -f
echo To stop:      docker-compose down
echo.
pause
