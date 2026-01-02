@echo off
REM ===================================================================
REM COMPLETE FIX: Rebuild with ML-Custom Enabled + 5-Minute Timeout
REM ===================================================================
REM This script applies ALL fixes:
REM - Frontend timeout: 5 minutes (300 seconds)
REM - ML-Custom: Re-enabled (all 192 features)
REM - Rebuild: Frontend + API Gateway containers
REM ===================================================================

echo.
echo ===================================================================
echo  COMPLETE FIX - Rebuild with ML-Custom Enabled
echo ===================================================================
echo.
echo This will:
echo  [1] Stop and remove old containers
echo  [2] Rebuild frontend with 5-minute timeout (300s)
echo  [3] Rebuild API Gateway with ML-Custom enabled
echo  [4] Start all services
echo  [5] Wait for services to be ready
echo  [6] Test the application
echo.
echo Expected results:
echo  - Analysis time: ~95 seconds (all 192 features)
echo  - Frontend timeout: 300 seconds (plenty of buffer)
echo  - No "No response from server" errors
echo.

set /p confirm="Continue? This will restart all services (Y/N): "
if /i not "%confirm%"=="Y" (
    echo Cancelled.
    pause
    exit /b
)

echo.
echo ===================================================================
echo Step 1/6: Stopping containers...
echo ===================================================================
docker-compose down
if errorlevel 1 (
    echo.
    echo ERROR: Docker command failed!
    echo.
    echo Please restart Docker Desktop:
    echo  1. Right-click Docker Desktop in system tray
    echo  2. Select "Restart"
    echo  3. Wait 2 minutes
    echo  4. Run this script again
    echo.
    pause
    exit /b 1
)
echo ✓ Containers stopped

echo.
echo ===================================================================
echo Step 2/6: Rebuilding frontend (with 5-minute timeout)...
echo ===================================================================
docker-compose build frontend
if errorlevel 1 (
    echo ERROR: Frontend build failed!
    echo Check the error messages above.
    pause
    exit /b 1
)
echo ✓ Frontend rebuilt

echo.
echo ===================================================================
echo Step 3/6: Rebuilding API Gateway (with ML-Custom enabled)...
echo ===================================================================
docker-compose build api-gateway
if errorlevel 1 (
    echo ERROR: API Gateway build failed!
    echo Check the error messages above.
    pause
    exit /b 1
)
echo ✓ API Gateway rebuilt

echo.
echo ===================================================================
echo Step 4/6: Starting all services...
echo ===================================================================
docker-compose up -d
if errorlevel 1 (
    echo ERROR: Failed to start services!
    pause
    exit /b 1
)
echo ✓ Services started

echo.
echo ===================================================================
echo Step 5/6: Waiting for services to initialize (60 seconds)...
echo ===================================================================
echo This allows:
echo  - ML models to load (30s)
echo  - Face embedder to initialize (30s)
echo  - Health checks to pass
echo.
timeout /t 60 /nobreak

echo.
echo ===================================================================
echo Step 6/6: Testing services...
echo ===================================================================
echo.

echo Testing API Gateway:
curl -s http://localhost:8000/health
if errorlevel 1 (
    echo WARNING: API Gateway may not be ready yet
) else (
    echo ✓ API Gateway is responding
)

echo.
echo.
echo Testing Frontend:
curl -s http://localhost:3000 | findstr "DOCTYPE" > nul
if errorlevel 1 (
    echo WARNING: Frontend may not be ready yet
) else (
    echo ✓ Frontend is responding
)

echo.
echo.
echo ===================================================================
echo Checking service status:
echo ===================================================================
docker-compose ps

echo.
echo.
echo ===================================================================
echo  REBUILD COMPLETE!
echo ===================================================================
echo.
echo Configuration:
echo  ✓ Frontend timeout: 300 seconds (5 minutes)
echo  ✓ ML-Custom: ENABLED (all 192 features)
echo  ✓ Expected analysis time: ~95 seconds
echo.
echo Access the application:
echo  Frontend: http://localhost:3000
echo  API Docs: http://localhost:8000/docs
echo  API Health: http://localhost:8000/health
echo.
echo Testing instructions:
echo  1. Open http://localhost:3000 in your browser
echo  2. Allow camera access or upload a face image
echo  3. Click "Analyze"
echo  4. Wait ~95 seconds for complete analysis
echo  5. Should receive ALL 192 features including ML-Custom
echo.
echo If you see "No response from server":
echo  - Check that ML-Custom is running: docker logs ml-custom-service
echo  - Verify timeout in browser console (should allow 5 minutes)
echo  - Check API Gateway logs: docker logs api-gateway --tail 50
echo.
pause
