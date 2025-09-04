#!/bin/bash
set -e

echo "Starting Dashboard with Debug Configuration"
echo "=========================================="

# Create logs directory
mkdir -p logs

# Stop any existing containers
echo "Stopping existing containers..."
docker compose down 2>/dev/null || true

# Start services with debug configuration
echo "Starting services with debug logging..."
docker compose -f docker-compose.yml -f docker-compose.debug.yml up -d

# Wait for services to start
echo "Waiting for services to start..."
sleep 10

# Check service status
echo -e "\nService Status:"
echo "==============="
docker compose ps

# Check container logs
echo -e "\nAnnotation App Logs (last 20 lines):"
echo "===================================="
docker compose logs --tail=20 annotation-app

echo -e "\nNginx Logs (last 20 lines):"
echo "============================"
docker compose logs --tail=20 nginx

# Test connectivity
echo -e "\nTesting connectivity..."
echo "======================="
python3 test_annotation.py localhost 2>/dev/null || echo "Test script failed - install requests: pip3 install requests"

echo -e "\nDebug complete! Services are running at:"
echo "- Dashboard: http://localhost:8080"
echo "- Annotation: http://localhost:8080/annotation/"
echo "- Direct Flask: http://localhost:5000"
echo ""
echo "To view live logs: docker compose logs -f"
echo "To stop services: docker compose down"