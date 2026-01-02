@echo off
echo ========================================
echo Completing Derm-Foundation Integration
echo ========================================
echo.

echo Step 1: Building api-gateway with new configuration...
docker-compose build --no-cache api-gateway
echo.

echo Step 2: Starting all services...
docker-compose up -d
echo.

echo Step 3: Waiting for services to initialize (60 seconds)...
timeout /t 60 /nobreak
echo.

echo Step 4: Checking service status...
docker-compose ps
echo.

echo Step 5: Checking derm-foundation logs...
docker logs derm-foundation-service --tail 30
echo.

echo Step 6: Checking api-gateway logs...
docker logs api-gateway --tail 30 | findstr "derm-foundation"
echo.

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next Steps:
echo 1. Open http://localhost:3000
echo 2. Upload a test image
echo 3. Expected: 156+ features (up from 113)
echo.
echo For detailed info, see:
echo   DERM_FOUNDATION_INTEGRATION_COMPLETE.md
echo.
pause
