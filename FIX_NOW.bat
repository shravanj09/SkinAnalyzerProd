@echo off
REM ===================================================================
REM ACTUAL FIX - Rebuild Containers with New Timeout Settings
REM ===================================================================
REM This rebuilds the containers to pick up the 300-second timeout
REM ===================================================================

echo.
echo ===================================================================
echo  FIXING ML-CUSTOM TIMEOUT ISSUE
echo ===================================================================
echo.
echo Current Problem:
echo  X ml-custom times out at 90 seconds
echo  X Only getting 88/215 features
echo  X Containers running OLD code from 5+ hours ago
echo.
echo This fix will:
echo  1. Stop old containers
echo  2. Rebuild API Gateway (with 300s timeout)
echo  3. Rebuild Frontend (with 300s timeout)
echo  4. Start with new containers
echo  5. Test ml-custom
echo.
echo Expected results after fix:
echo  - ml-custom will have 300 second timeout (not 90s)
echo  - Analysis will complete in ~95 seconds
echo  - You'll get ~192 features (not 88)
echo  - ml-custom will show checkmark (not X)
echo.

set /p confirm="Rebuild containers now? (Y/N): "
if /i not "%confirm%"=="Y" (
    echo Cancelled.
    pause
    exit /b
)

echo.
echo ===================================================================
echo Step 1/5: Stopping old containers...
echo ===================================================================
docker-compose down
if errorlevel 1 (
    echo.
    echo ERROR: Docker command failed!
    echo.
    echo Please ensure Docker Desktop is running:
    echo  1. Check system tray for Docker whale icon
    echo  2. If not running, open Docker Desktop
    echo  3. Wait 2 minutes
    echo  4. Run this script again
    echo.
    pause
    exit /b 1
)
echo Done.

echo.
echo ===================================================================
echo Step 2/5: Rebuilding API Gateway (this takes 3-5 minutes)...
echo ===================================================================
echo This picks up the new 300-second timeout for ml-custom
echo.
docker-compose build api-gateway
if errorlevel 1 (
    echo ERROR: API Gateway build failed!
    pause
    exit /b 1
)
echo Done.

echo.
echo ===================================================================
echo Step 3/5: Rebuilding Frontend (this takes 3-5 minutes)...
echo ===================================================================
echo This picks up the new 300-second timeout
echo.
docker-compose build frontend
if errorlevel 1 (
    echo ERROR: Frontend build failed!
    pause
    exit /b 1
)
echo Done.

echo.
echo ===================================================================
echo Step 4/5: Starting all services with NEW containers...
echo ===================================================================
docker-compose up -d
if errorlevel 1 (
    echo ERROR: Failed to start containers!
    pause
    exit /b 1
)
echo Done.

echo.
echo ===================================================================
echo Step 5/5: Waiting for services to initialize (90 seconds)...
echo ===================================================================
echo Please wait while:
echo  - ML models load (30-45s)
echo  - Face embedder initializes (30-45s)
echo  - Health checks pass
echo.
timeout /t 90 /nobreak

echo.
echo ===================================================================
echo Verifying the fix...
echo ===================================================================
echo.

echo Checking API Gateway:
curl -s http://localhost:8000/health
echo.

echo.
echo Checking container creation time:
docker ps --filter "name=api-gateway" --format "Created: {{.CreatedAt}}"
echo.

echo.
echo Checking timeout in new container:
docker exec api-gateway cat app/services/orchestrator.py | findstr "ml-custom" -A 2
echo.

echo.
echo ===================================================================
echo Container Status:
echo ===================================================================
docker-compose ps
echo.

echo.
echo ===================================================================
echo  FIX COMPLETE!
echo ===================================================================
echo.
echo The containers have been rebuilt with:
echo  - API Gateway timeout: 300 seconds (was 90s)
echo  - Frontend timeout: 300 seconds (was 120s)
echo  - ML-Custom: ENABLED with plenty of buffer time
echo.
echo IMPORTANT - How to test:
echo  1. Open: http://localhost:3000
echo  2. Upload a clear face photo
echo  3. Click "Analyze"
echo  4. Wait patiently for ~95 seconds (don't refresh!)
echo  5. You should see:
echo     - ml-custom: checkmark (not X)
echo     - ~192 features (not 88)
echo     - All categories "COMPLETE" (not PARTIAL)
echo.
echo If ml-custom still shows X:
echo  - Check logs: docker logs ml-custom-service --tail 50
echo  - Check gateway: docker logs api-gateway --tail 50
echo  - ML-custom needs 90s to process, timeout is now 300s
echo.
echo If you see timeout errors still:
echo  - The frontend might be cached in browser
echo  - Try hard refresh: Ctrl+Shift+R
echo  - Or open in incognito/private window
echo.
pause
