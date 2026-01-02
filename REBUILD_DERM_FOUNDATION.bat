@echo off
echo ========================================
echo Rebuilding Derm-Foundation Integration
echo ========================================
echo.

echo Step 1: Stopping containers...
docker-compose stop api-gateway derm-foundation
echo.

echo Step 2: Rebuilding derm-foundation with enhanced implementation...
docker-compose build --no-cache derm-foundation
echo.

echo Step 3: Rebuilding api-gateway with new configuration...
docker-compose build --no-cache api-gateway
echo.

echo Step 4: Starting containers...
docker-compose up -d api-gateway derm-foundation
echo.

echo Step 5: Waiting for services to initialize (30 seconds)...
timeout /t 30 /nobreak
echo.

echo Step 6: Checking container status...
docker-compose ps
echo.

echo ========================================
echo Rebuild Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Wait 1 minute for all services to fully start
echo 2. Open http://localhost:3000
echo 3. Upload an image to test
echo 4. Expected: 156+ features (113 + 43 from Derm-Foundation)
echo.
echo To check logs:
echo   docker logs derm-foundation-service --tail 50
echo   docker logs api-gateway --tail 50
echo.
pause
