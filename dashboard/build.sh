#!/bin/bash
# Build script for CRPlayer Dashboard with Annotation Tool

set -e

echo "Building CRPlayer Dashboard..."

# Ensure we're in the dashboard directory
cd "$(dirname "$0")"

# Check if research models.py exists
if [ ! -f "../research/screen-page-classification/models.py" ]; then
    echo "Warning: Research models.py not found. Autolabel feature may not work."
    echo "Expected at: ../research/screen-page-classification/models.py"
fi

# Check if model checkpoint exists
if [ ! -f "../data/models/SingleLabelClassification/model.pth" ]; then
    echo "Warning: Model checkpoint not found. Autolabel will not work until a model is trained."
    echo "Expected at: ../data/models/SingleLabelClassification/model.pth"
fi

# Build and start the containers
echo "Building Docker containers..."
docker-compose build --no-cache

echo "Starting services..."
docker-compose up -d

# Wait for services to be healthy
echo "Waiting for services to be healthy..."
sleep 5

# Check health
docker-compose ps

echo ""
echo "Dashboard is available at: http://localhost:8080"
echo "Annotation tool is available at: http://localhost:8080/annotation/"
echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop: docker-compose down"
