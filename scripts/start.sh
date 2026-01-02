#!/bin/bash
# Linux/Mac startup script for AI Facial Analysis Platform

echo "==================================="
echo "AI Facial Analysis Platform"
echo "Starting all services..."
echo "==================================="
echo

# Check if Docker is running
if ! docker ps >/dev/null 2>&1; then
    echo "ERROR: Docker is not running!"
    echo "Please start Docker and try again."
    exit 1
fi

echo "[OK] Docker is running"
echo

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found!"
    echo "Please copy .env.example to .env and configure it."
    exit 1
fi

echo "[OK] Environment file found"
echo

# Start services
echo "Starting all services with Docker Compose..."
echo "This may take 2-5 minutes on first run..."
echo

docker-compose up -d --build

if [ $? -ne 0 ]; then
    echo
    echo "ERROR: Failed to start services!"
    echo "Check the logs above for details."
    exit 1
fi

echo
echo "==================================="
echo "Services started successfully!"
echo "==================================="
echo

echo "Waiting for services to be ready..."
sleep 10

echo
echo "Checking service health..."
if curl -sf http://localhost:8000/health >/dev/null; then
    echo "[OK] API Gateway is healthy"
else
    echo "WARNING: API Gateway not responding yet"
    echo "Services may still be initializing..."
    echo "Please wait 1-2 minutes and check http://localhost:8000/health"
fi

echo
echo "==================================="
echo "Access Points:"
echo "==================================="
echo "Frontend:    http://localhost:3000"
echo "API Gateway: http://localhost:8000"
echo "API Docs:    http://localhost:8000/docs"
echo "MinIO UI:    http://localhost:9001"
echo "==================================="
echo

echo "To view logs: docker-compose logs -f"
echo "To stop:      docker-compose down"
echo

# Try to open browser (works on most systems)
if command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:3000
elif command -v open &> /dev/null; then
    open http://localhost:3000
fi
