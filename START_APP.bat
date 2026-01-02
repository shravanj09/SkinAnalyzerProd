@echo off
echo ========================================
echo  Starting Facial Analysis Application
echo ========================================
echo.

echo [1/4] Stopping Docker Desktop...
taskkill /F /IM "Docker Desktop.exe" >nul 2>&1
timeout /t 5 /nobreak >nul

echo [2/4] Starting Docker Desktop...
start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
echo Waiting for Docker to start (60 seconds)...
timeout /t 60 /nobreak

echo [3/4] Starting all services...
cd /d D:\code\SkinAnalyzerProd
docker compose down >nul 2>&1
docker compose up -d

echo [4/4] Waiting for services to be ready (30 seconds)...
timeout /t 30 /nobreak

echo.
echo ========================================
echo  Application Starting!
echo ========================================
echo.
echo  Frontend: http://localhost:3000
echo  API:      http://localhost:8000/health
echo.
echo Opening browser...
start http://localhost:3000

echo.
echo Press any key to check service status...
pause >nul

docker compose ps

echo.
echo Done! If services show "Up", the app is ready.
echo.
pause
