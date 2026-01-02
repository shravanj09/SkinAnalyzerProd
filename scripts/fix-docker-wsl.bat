@echo off
REM ==============================================================================
REM Docker Desktop WSL Corruption Fix Script
REM ==============================================================================
REM This script fixes the "Cannot create a file when that file already exists"
REM error when starting Docker Desktop
REM ==============================================================================

echo.
echo ========================================
echo Docker Desktop WSL Corruption Fix
echo ========================================
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: This script must be run as Administrator
    echo Right-click and select "Run as administrator"
    pause
    exit /b 1
)

echo [1/5] Stopping Docker Desktop...
taskkill /F /IM "Docker Desktop.exe" /T >nul 2>&1
taskkill /F /IM "com.docker.backend.exe" /T >nul 2>&1
taskkill /F /IM "com.docker.proxy.exe" /T >nul 2>&1
timeout /t 3 /nobreak >nul
echo      Done.

echo [2/5] Unregistering old WSL distributions...
wsl --unregister docker-desktop >nul 2>&1
wsl --unregister docker-desktop-data >nul 2>&1
echo      Done.

echo [3/5] Cleaning up WSL directories...
if exist "%LOCALAPPDATA%\Docker\wsl\main" (
    echo      Found corrupted 'main' directory, removing...
    rmdir /S /Q "%LOCALAPPDATA%\Docker\wsl\main" 2>nul
    if exist "%LOCALAPPDATA%\Docker\wsl\main" (
        move "%LOCALAPPDATA%\Docker\wsl\main" "%LOCALAPPDATA%\Docker\wsl\main.old.%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
    )
)
if exist "%LOCALAPPDATA%\Docker\wsl\disk" (
    echo      Found 'disk' directory, cleaning...
    rmdir /S /Q "%LOCALAPPDATA%\Docker\wsl\disk\data" 2>nul
)
echo      Done.

echo [4/5] Cleaning Docker cache (optional, keeps volumes)...
if exist "%LOCALAPPDATA%\Docker\wsl\distro\data" (
    rmdir /S /Q "%LOCALAPPDATA%\Docker\wsl\distro\data" 2>nul
)
echo      Done.

echo [5/5] Starting Docker Desktop...
start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
echo      Launched. Please wait for Docker Desktop to initialize...

echo.
echo ========================================
echo Fix Complete!
echo ========================================
echo Docker Desktop is starting. Wait for the Docker icon in your
echo system tray to stop spinning before using Docker.
echo.
pause
